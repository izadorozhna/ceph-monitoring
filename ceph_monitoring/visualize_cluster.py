import io
import sys
import json
import time
import pstats
import shutil
import bisect
import logging
import os.path
import tempfile
import cProfile
import argparse
import subprocess
import collections
import logging.config
from typing import Dict, Optional, List

from cephlib.storage import make_storage, make_attr_storage

from . import html
from .hw_info import b2ssize, b2ssizei
from .cluster import CephCluster, NO_VALUE, CephOSD


H = html.rtag
logger = logging.getLogger('cephlib.report')


class TypedStorage:
    def __init__(self, storage):
        self.txt = make_attr_storage(storage.sstorage, 'txt')
        self.json = make_attr_storage(storage.sstorage, 'json')
        self.xml = make_attr_storage(storage.sstorage, 'xml')
        self.raw = storage

    def substorage(self, path):
        return self.__class__(self.raw.substorage(path))


HTML_UNKNOWN = H.font('???', color="orange")


def html_ok(text):
    return H.font(text, color="green")


def html_fail(text):
    return H.font(text, color="red")


def_color_map = [
    (0.0, (0.500, 0.000, 1.000)),
    (0.1, (0.304, 0.303, 0.988)),
    (0.2, (0.100, 0.588, 0.951)),
    (0.3, (0.096, 0.805, 0.892)),
    (0.4, (0.300, 0.951, 0.809)),
    (0.5, (0.504, 1.000, 0.705)),
    (0.6, (0.700, 0.951, 0.588)),
    (0.7, (0.904, 0.805, 0.451)),
    (0.8, (1.000, 0.588, 0.309)),
    (0.9, (1.000, 0.303, 0.153)),
    (1.0, (1.000, 0.000, 0.000))
]


def val_to_color(val, color_map=def_color_map):
    idx = [i[0] for i in color_map]
    assert idx == sorted(idx)
    pos = bisect.bisect_left(idx, val)
    if pos <= 0:
        ncolor = color_map[0][1]
    elif pos >= len(idx):
        ncolor = color_map[-1][1]
    else:
        color1 = color_map[pos - 1][1]
        color2 = color_map[pos][1]

        dx1 = (val - idx[pos - 1]) / (idx[pos] - idx[pos - 1])
        dx2 = (idx[pos] - val) / (idx[pos] - idx[pos - 1])

        ncolor = [(v1 * dx2 + v2 * dx1)
                  for v1, v2 in zip(color1, color2)]

    ncolor = [(channel * 255 + 255) / 2 for channel in ncolor]
    return "#%02X%02X%02X" % tuple(map(int, ncolor))


class Report:
    def __init__(self, cluster_name, output_file):
        self.cluster_name = cluster_name
        self.output_file = output_file
        self.style = []
        self.style_links = []
        self.script_links = []
        self.scripts = []
        self.onload = []
        self.divs = []

    def next_line(self):
        self.add_block(None, None, None)

    def add_block(self, weight, header, block_obj, menu_item=None):
        if menu_item is None:
            menu_item = header

        self.divs.append((weight, header, menu_item,
                         str(block_obj) if block_obj is not None else None))

    def add_hidden(self, block_obj):
        self.add_block(None, None, block_obj)

    def save_to(self, output_dir, pretty_html=False):
        pt = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_files_dir = os.path.join(pt, "html_js_css")

        self.style_links.append("http://getbootstrap.com/examples/dashboard/dashboard.css")

        css = """
        table.zebra-table tr:nth-child(even) {background-color: #E0E0FF;}
        table th {background: #ededed;}
        .right{
            position:relative;
            margin:0;
            padding:0;
            float:right;
        }
        .left{
            position:relative   ;
            margin:0;
            padding:0;
            float:left;
        }
        th {text-align: center;}
        td {text-align: right;}
        tr td:first-child {text-align: left;}
        """

        self.style.append(css)

        links = []
        for link in self.style_links + self.script_links:
            fname = link.rsplit('/', 1)[-1]
            src_path = os.path.join(static_files_dir, fname)
            dst_path = os.path.join(output_dir, fname)
            if os.path.exists(src_path):
                if not os.path.exists(dst_path):
                    shutil.copyfile(src_path, dst_path)
                link = fname
            links.append(link)

        css_links = links[:len(self.style_links)]
        js_links = links[len(self.style_links):]

        doc = html.Doc()
        with doc.html:
            with doc.head:
                doc.title("Ceph cluster report: " + self.cluster_name)

                for url in css_links:
                    doc.link(href=url, rel="stylesheet", type="text/css")

                doc.style("\n".join(self.style), type="text/css")

                for url in js_links:
                    doc.script(type="text/javascript", src=url)

                for script in self.scripts:
                    doc.script(script, type="text/javascript")

            with doc.body(onload=";".join(self.onload)):
                with doc.div(_class="container-fluid"):
                    with doc.div(_class="row row-offcanvas row-offcanvas-left"):
                        with doc.div(_class="col-md-2 sidebar-offcanvas", id="sidebar", role="navigation"):
                            with doc.div(data_spy="affix", data_offset_top="45", data_offset_bottom="90"):
                                with doc.ul(_class="nav", id="sidebar-nav"):
                                    for sid, (_, _, menu, data) in enumerate(self.divs):
                                        if menu is None or data is None or data == "":
                                            continue

                                        if menu.endswith(":"):
                                            menu = menu[:-1]
                                        doc.li.a(menu, href="#section" + str(sid))

                        with doc.div(_class="col-md-10", data_spy="scroll", data_target="#sidebar-nav"):
                            doc("\n")
                            doc._enter("div", _class="row")
                            for sid, (w, header, _, block) in enumerate(self.divs):
                                if block is None:
                                    doc._exit()
                                    doc("\n")
                                    doc._enter("div", _class="row")
                                elif w is None:
                                    doc(block)
                                else:
                                    with doc.div(_class="col-md-" + str(w)):
                                        doc.H3.center(header, id="section" + str(sid))
                                        doc.br

                                        if block != "":
                                            doc.center(block)
                            doc._exit()

        index = "<!doctype html>" + str(doc)
        index_path = os.path.join(output_dir, self.output_file)

        try:
            if pretty_html:
                import BeautifulSoup
                index = BeautifulSoup.BeautifulSoup(index).prettify()
        except:
            pass

        open(index_path, "w").write(index)


def show_summary(report, cluster):
    t = html.HTMLTable(["Setting", "Value"])
    t.add_cells("Collected at", cluster.report_collected_at_local)
    t.add_cells("Collected at GMT", cluster.report_collected_at_gmt)
    t.add_cells("Status", cluster.overall_status)
    t.add_cells("PG count", cluster.num_pgs)
    t.add_cells("Pool count", len(cluster.pools))
    t.add_cells("Used", b2ssize(cluster.bytes_used, False))
    t.add_cells("Avail", b2ssize(cluster.bytes_avail, False))
    t.add_cells("Data", b2ssize(cluster.data_bytes, False))

    avail_perc = cluster.bytes_avail * 100 / cluster.bytes_total
    t.add_cells("Free %", int(avail_perc))

    osd_count = len(cluster.osds)
    t.add_cells("Mon count", len(cluster.mons))

    report.add_block(3, "Status:", t)

    stor_set = set(osd.storage_type for osd in cluster.osds)
    has_bs = 'bluestore' in stor_set
    has_fs = 'filestore' in stor_set

    if cluster.settings is None:
        t = H.font("No live OSD found!", color="red")
    else:
        t = html.HTMLTable(["Setting", "Value"])
        t.add_cells("Count", osd_count)
        t.add_cells("Average PG per OSD", cluster.num_pgs // osd_count)
        t.add_cells("Cluster net", cluster.cluster_net)
        t.add_cells("Public net", cluster.public_net)
        t.add_cells("Near full ratio", "%0.3f" % (float(cluster.settings.mon_osd_nearfull_ratio,)))
        t.add_cells("Full ratio", "%0.3f" % (float(cluster.settings.mon_osd_full_ratio,)))
        t.add_cells("Backfill full ratio", getattr(cluster.settings, "osd_backfill_full_ratio", "?"))
        t.add_cells("Filesafe full ratio", "%0.3f" % (float(cluster.settings.osd_failsafe_full_ratio,)))
        t.add_cells("Storage types", list(stor_set)[0] if len(stor_set) == 1 else html_fail(",".join(stor_set)))

        if has_fs:
            t.add_cells("Journal aio", cluster.settings.journal_aio)
            t.add_cells("Journal dio", cluster.settings.journal_dio)
            t.add_cells("Filestorage sync", str(int(float(cluster.settings.filestore_max_sync_interval))) + 's')
        t.add_cells("Monmap version", str(cluster.monmap_stat['epoch']))

        mon_tm = time.mktime(time.strptime(cluster.monmap_stat['modified'], "%Y-%m-%d %H:%M:%S.%f"))
        collect_tm = time.mktime(time.strptime(cluster.report_collected_at_local, "%Y-%m-%d %H:%M:%S"))
        diff = int(collect_tm - mon_tm)
        h = diff // 3600
        m = (diff % 3600) % 60
        s = diff % 60
        t.add_cells("Monmap modified in", '{0}:{1:<02d}:{2:<02d}'.format(h, m, s))
    report.add_block(3, "OSD:", t)

    t = html.HTMLTable(["Setting", "Value"])
    t.add_cells("Client IO Bps", b2ssize(cluster.write_bytes_sec, False))
    t.add_cells("Client IO IOPS", b2ssize(cluster.op_per_sec, False))
    report.add_block(2, "Activity:", t)

    colors = {
        "HEALTH_WARN": "orange",
        "HEALTH_ERR": "red"
    }

    if len(cluster.health_summary) != 0:
        t = html.Doc()

        if cluster.is_luminous:
            for check in cluster.health_summary.values():
                color = colors.get(check["severity"], "black")
                t.font(check['summary']['message'].capitalize(), color=color)
                t.br
        else:
            for msg in cluster.health_summary:
                color = colors.get(msg['severity'], "black")
                t.font(msg['summary'].capitalize(), color=color)
                t.br

        report.add_block(2, "Status messages:", t)


def show_mons_info(report, cluster):
    table = html.HTMLTable(headers=["Name", "Health", "Role", "Disk free<br>B (%)"])

    for mon in cluster.mons:
        if mon.health is None:
            health = "Unknown"
        else:
            health = html_ok("HEALTH_OK") if mon.health == "HEALTH_OK" else html_fail(mon.health)

        perc = ("Unknown" if cluster.is_luminous else
                "{0} ({1})".format(b2ssize(mon.kb_avail * 1024, False), mon.avail_percent))
        table.add_row(map(str, [mon.name, health, "Unknown" if mon.role is None else mon.role, perc]))

    report.add_block(3, "Monitors info:", table)


def show_pg_state(report, cluster):
    statuses = collections.defaultdict(lambda: 0)

    for pg_group in cluster.pgmap_stat['pgs_by_state']:
        for state_name in pg_group['state_name'].split('+'):
            statuses[state_name] += pg_group["count"]

    table = html.HTMLTable(headers=["Status", "Count", "%"])
    table.add_row(["any", str(cluster.num_pgs), "100.00"])
    for status, count in sorted(statuses.items()):
        table.add_row([status, str(count), "%.2f" % (100.0 * count / cluster.num_pgs)])

    report.add_block(2, "PG's status:", table)


def show_osd_state(report, cluster):
    statuses = collections.defaultdict(lambda: [])

    for osd in cluster.osds:
        statuses[osd.status].append("{0.host.name}:{0.id}".format(osd))

    table = html.HTMLTable(headers=["Status", "Count", "ID's"])

    for status, osds in sorted(statuses.items()):
        table.add_row([status, len(osds), "" if status == "up" else
            ("<br>".join(osds) if len(osds) < 5 else "<br>".join(osds[:4]) + "<br>and {} more".format(len(osds) - 4))])

    report.add_block(2, "OSD's state:", table)


def show_pools_info(report, cluster):
    table = html.HTMLTable(headers=["Pool",
                                     "Id",
                                     "size",
                                     "min_size",
                                     "obj",
                                     "data",
                                     "free",
                                     "read",
                                     "write",
                                     "ruleset",
                                     "PG(PGP)",
                                     "Bytes/PG",
                                     "Objs/PR",
                                     "PG per OSD<br>Dev %"])

    for _, pool in sorted(cluster.pools.items()):
        table.add_cell(pool.name)
        table.add_cell(str(pool.id))
        table.add_cell(str(pool.size))
        table.add_cell(str(pool.min_size))
        table.add_cell(b2ssize(int(pool.num_objects), base=1000), sorttable_customkey=str(int(pool.num_objects)))
        table.add_cell(b2ssize(int(pool.size_bytes), False), sorttable_customkey=str(int(pool.size_bytes)))
        table.add_cell('---')
        table.add_cell(b2ssize(int(pool.read_bytes), False), sorttable_customkey=str(int(pool.read_bytes)))
        table.add_cell(b2ssize(int(pool.write_bytes), False), sorttable_customkey=str(int(pool.write_bytes)))
        table.add_cell(str(pool.crush_rule))

        if pool.pgp != pool.pg:
            table.add_cell("{0}({1})".format(pool.pg, H.font(str(pool.pgp), color="red")),
                                             sorttable_customkey=str(pool.pg))
        else:
            table.add_cell(str(pool.pg), sorttable_customkey=str(pool.pg))

        bytes_per_pg = int(pool.size_bytes) // int(pool.pg)
        bytes_per_pg_s = b2ssize(bytes_per_pg)
        table.add_cell(bytes_per_pg_s, sorttable_customkey=str(bytes_per_pg))

        obj_per_pg = int(pool.num_objects) // int(pool.pg)
        obj_per_pg_s = b2ssize(obj_per_pg, base=1000)
        table.add_cell(obj_per_pg_s, sorttable_customkey=str(obj_per_pg))

        if cluster.sum_per_osd is None:
            table.add_cell('-')
        else:
            row = [osd_pg.get(pool.name, 0) for osd_pg in cluster.osd_pool_pg_2d.values()]
            avg = float(sum(row)) / len(row)
            if len(row) < 2:
                dev = None
            else:
                dev = (sum((i - avg) ** 2.0 for i in row) / (len(row) - 1)) ** 0.5

            if avg < 1E-5:
                table.add_cell('--')
            else:
                if dev is None:
                    dev_perc = "--"
                else:
                    dev_perc = str(int(dev * 100. / avg))

                table.add_cell("{0:.1f} ~ {1}%".format(avg, dev_perc), sorttable_customkey=str(int(avg * 1000)))

        table.next_row()

    report.add_block(8, "Pool's stats:", table)


def show_osd_info(report, cluster):
    # osds_info = [get_osd_info(cluster, osd) for osd in cluster.osds]  # type: List[OSDInfo]

    w_headers = ["weight<br>" + root.name for root in cluster.crush.roots]

    stor_set = set(osd.storage_type for osd in cluster.osds)
    has_bs = 'bluestore' in stor_set
    has_fs = 'filestore' in stor_set
    assert has_fs or has_bs

    headers=["OSD", "node", "status", "version [hash]", "daemon<br>run", "open<br>files", "ip<br>conn", "threads"] + \
             w_headers + ["reweight", "PG"]

    if has_bs and has_fs:
        headers.append("Type")

    headers.extend(["Storage<br>dev type", "Storage<br>total", "Storage<br>free %"])

    if has_fs and has_bs:
        headers.extend(["Journal<br>or wal<br>colocated", "Journal<br>or wal<br>dev type",
                        "Journal<br>on file", "Journal<br>or wal<br>size"])
    elif has_fs and not has_bs:
        headers.extend(["Journal<br>colocated", "Journal<br>dev type",
                        "Journal<br>on file", "Journal<br>size"])
    else:
        headers.extend(["wal<br>colocated", "wal<br>dev type", "wal<br>size"])

    if has_bs:
        headers.extend(["DB<br>colocated", "DB<br>dev type", "DB<br>size"])

    table = html.HTMLTable(headers)

    for osd in sorted(cluster.osds, key=lambda x: x.id):
        if osd.daemon_runs is None:
            daemon_msg = HTML_UNKNOWN
        elif osd.daemon_runs:
            daemon_msg = html_ok('yes')
        else:
            daemon_msg = html_fail('no')

        table.add_cell(str(osd.id))
        table.add_cell(osd.host.name)
        table.add_cell(html_ok("up") if osd.status == 'up' else html_fail("down"))
        if osd.version is None:
            table.add_cell("Unknown")
        else:
            ver, hash = osd.version
            table.add_cell("{0}  [{1}]".format(ver, hash[:8]))
        table.add_cell(daemon_msg)

        if osd.procinfo:
            table.add_cell(str(osd.procinfo["fd_count"]))
            socks = sum(int(tp.get('inuse', 0)) for tp in osd.procinfo.get('ipv4', {}).values())
            socks += sum(int(tp.get('inuse', 0)) for tp in osd.procinfo.get('ipv6', {}).values())
            table.add_cell(str(socks))
            table.add_cell(str(osd.procinfo["th_count"]))
        else:
            table.add_cell("-")
            table.add_cell("-")
            table.add_cell("-")

        disks = cluster.hosts[osd.host.name].disks
        storage_devs = cluster.hosts[osd.host.name].storage_devs
        total_b = int(storage_devs[osd.data_partition].size)

        for root in cluster.crush.roots:
            nodes = cluster.crush.find_nodes([('root', root.name), ('osd', "osd.{0}".format(osd.id))])
            if not nodes:
                table.add_cell("-")
            elif len(nodes) > 1:
                table.add_cell("+")
            else:
                expected_weight = float(total_b) / (1024 ** 4)
                weight = "{0:.2f}".format(nodes[0].weight)
                weight_s = html_fail(weight) if abs(1 - (nodes[0].weight / expected_weight)) > .2 else weight
                table.add_cell(weight_s)

        rew = "{0:.2f}".format(osd.reweight)
        table.add_cell(html_fail(rew) if abs(osd.reweight - 1.0) > .01 else rew)

        if osd.pg_count is None:
            table.add_cell(HTML_UNKNOWN, sorttable_customkey=0)
        else:
            table.add_cell(str(osd.pg_count))

        if has_fs and has_bs:
            table.add_cell(osd.storage_type)

        data_drive_type = disks[osd.data_dev].tp
        data_drive_type = (html_ok if data_drive_type in ('ssd', 'nvme') else lambda x: x)(data_drive_type)
        table.add_cell(data_drive_type)

        avail_perc = osd.free_perc

        if avail_perc < 20:
            color = "red"
        elif avail_perc < 40:
            color = "yellow"
        else:
            color = "green"
        avail_perc_str = H.font(avail_perc, color=color)

        table.add_cell(b2ssize(total_b, False), sorttable_customkey=total_b)
        table.add_cell(avail_perc_str, sorttable_customkey=avail_perc)

        if osd.storage_type == 'filestore':
            j_on_same_drive = html_ok("no") if osd.journal_dev != osd.data_dev else html_fail("yes")
            j_on_file = html_ok("no") if osd.journal_partition != osd.data_partition else html_fail("yes")
            j_drive_type = disks[osd.journal_dev].tp
            j_drive_type = (html_ok if j_drive_type in ('ssd', 'nvme') else html_fail)(j_drive_type)

            table.add_cell(j_on_same_drive)
            table.add_cell(j_drive_type)
            table.add_cell(j_on_file)

            if j_on_same_drive:
                j_size_s = '-'
            else:
                j_size = storage_devs[osd.journal_partition].size
                j_size_s = b2ssize(j_size)
                osd_sync = float(osd.config['filestore_max_sync_interval'])
                j_size_s = j_size_s if j_size >= osd_sync * 100 * (1024 ** 2) else html_fail(j_size_s)
            table.add_cell(j_size_s)

            if has_bs:
                table.add_cell('-')
                table.add_cell('-')
                table.add_cell('-')
        else:
            assert osd.storage_type == 'bluestore'
            wal_on_same_drive = html_ok("no") if osd.wal_dev != osd.data_dev else html_fail("yes")
            wal_drive_type = disks[osd.wal_dev].tp
            wal_drive_type = (html_ok if wal_drive_type in ('ssd', 'nvme') else html_fail)(wal_drive_type)

            if osd.wal_partition not in (osd.db_partition, osd.data_partition):
                wal_size = int(storage_devs[osd.wal_partition].size)
                wal_size_s = b2ssize(wal_size) + 'B'
                wal_size = html_fail(wal_size_s) if wal_size < 512 * 1024 * 1024 else wal_size_s
            else:
                wal_size = '-'

            table.add_cell(wal_on_same_drive)
            table.add_cell(wal_drive_type)
            if has_fs:
                table.add_cell('-')
            table.add_cell(wal_size)

            db_on_same_drive = html_ok("no") if osd.db_dev != osd.data_dev else html_fail("yes")
            db_drive_type = disks[osd.db_dev].tp
            db_drive_type = (html_ok if db_drive_type in ('ssd', 'nvme') else html_fail)(db_drive_type)
            if osd.db_partition != osd.data_partition:
                db_size = int(storage_devs[osd.db_partition].size)
                data_size = int(storage_devs[osd.data_partition].size)
                db_sz_s = b2ssize(db_size) + "B"
                db_size = html_fail(db_sz_s) if db_size < data_size * 0.0095 else db_sz_s
            else:
                db_size = '-'

            table.add_cell(db_on_same_drive)
            table.add_cell(db_drive_type)
            table.add_cell(db_size)
        table.next_row()

    report.add_block(10, "OSD's info:", table)


def show_osd_perf_info(report, cluster):
    table = html.HTMLTable(headers=["OSD",
                                    "node",
                                    "journal<br>lat, ms",
                                    "apply<br>lat, ms<br>avg/osd perf",
                                    "commit<br>lat, ms<br>avg/osd perf",
                                    "D dev",
                                    "D read<br>Bps/OPS",
                                    "D write<br>Bps/OPS",
                                    "D lat<br>ms",
                                    "D IO<br>time %",
                                    "J dev",
                                    "J write<br>Bps/OPS",
                                    "J lat<br>ms",
                                    "J IO<br>time %"])

    for osd in cluster.osds:
        perf_info = []

        if osd.data_stor_stats is None:
            perf_info.extend([('-', 0)] * 5)
        else:
            load = osd.data_stor_stats.load

            perf_info.extend([
                (os.path.basename(osd.data_stor_stats.root_dev), None),
                (b2ssizei(load.read_bytes, False) + " / " + b2ssizei(load.read_iops, False), load.read_bytes),
                (b2ssizei(load.write_bytes, False) + " / " + b2ssizei(load.write_iops, False), load.write_bytes),
                (int(load.lat * 1000), load.lat) if load.lat is not None else ('-', 0),
                (int(load.io_time * 100), load.io_time)
            ])

        if osd.j_stor_stats is None:
            perf_info.extend([('-', 0)] * 3)
        else:
            if osd.j_stor_stats.root_dev != osd.data_stor_stats.root_dev:
                load = osd.j_stor_stats.load
                perf_info.extend([
                    (os.path.basename(osd.j_stor_stats.root_dev), None),
                    (b2ssizei(load.write_bytes, False) + " / " + b2ssizei(load.write_iops, False), load.write_bytes),
                    (int(load.lat * 1000), load.lat) if load.lat is not None else ('-', 0),
                    (int(load.io_time * 100), load.io_time)
                ])
            else:
                perf_info.extend([("<--", None), ('-', 0), ('-', 0), ('-', 0)])

        table.add_cell(str(osd.id))
        table.add_cell(osd.host.name)

        def lat2s(val):
            if val is None or val < 1e-6:
                return HTML_UNKNOWN
            elif val < 1e-3:
                return "{0:.3f}".format(val * 1000)
            else:
                return "{0}".format(int(val * 1000))

        lats = (("journal_latency", None), ("apply_latency", "apply_latency_s"),
                ("commitcycle_latency", "commitcycle_latency_s"))

        if osd.osd_perf is not None:
            for name, name_s in lats:
                s_val = None if name_s is None else str(osd.osd_perf[name_s])

                if name not in osd.osd_perf:
                    v_val = HTML_UNKNOWN
                elif isinstance(osd.osd_perf[name], int):
                    v_val = str(osd.osd_perf[name])
                else:
                    # scan for latest available value
                    for val in osd.osd_perf[name][::-1]:
                        if val != NO_VALUE:
                            v_val = lat2s(val)
                            break
                    else:
                        v_val = HTML_UNKNOWN

                if s_val:
                    table.add_cell("{} / {}".format(v_val, s_val))
                else:
                    table.add_cell(v_val)
        else:
            for name, name_s in lats:
                if name_s:
                    table.add_cell("{0} / {0}".format(HTML_UNKNOWN))
                else:
                    table.add_cell(HTML_UNKNOWN)

        for val, sorted_val in perf_info:
            if sorted_val is None:
                table.add_cell(str(val))
            else:
                table.add_cell(str(val), sorttable_customkey=str(sorted_val))
        table.next_row()

    report.add_block(10, "OSD's current load:", table)


def show_host_network_load_in_color(report, cluster):
    send_net_io = collections.defaultdict(list)
    recv_net_io = collections.defaultdict(list)

    max_net_count = 0
    for host in cluster.hosts.values():
        nets = [('cluster', host.ceph_cluster_adapter), ('public', host.ceph_public_adapter)]
        nets += sorted((net.name, net) for net in host.net_adapters.values()
                       if net.is_phy and net not in (host.ceph_cluster_adapter, host.ceph_cluster_adapter))

        max_net_count = max(max_net_count, len(nets))
        for name, net in nets:
            if net is None or net.load is None:
                if name in {'cluster', 'public'}:
                    send_net_io[host.name].append((name, None, None, None))
                    recv_net_io[host.name].append((name, None, None, None))
            else:
                send_net_io[host.name].append((name, net.load.send_bytes_avg, net.load.send_packets_avg, net.speed))
                recv_net_io[host.name].append((name, net.load.recv_bytes_avg, net.load.recv_packets_avg, net.speed))

    std_nets = {'public', 'cluster'}
    ceph_net_count = len(std_nets)
    report.add_block(12, "Network load", "")

    for io, name in [(send_net_io, "Send (KiBps/Pps)"), (recv_net_io, "Receive (KiBps/Pps)")]:

        table = html.HTMLTable(["host", "cluster", "public"] + ["hw adapter"] * (max_net_count - ceph_net_count),
                               zebra=False)

        for host_name, net_loads in sorted(io.items()):
            table.add_cell(host_name)
            for net_name, load_bps, load_pps, speed in net_loads:
                if load_bps is None:
                    table.add_cell('-', sorttable_customkey="")
                else:
                    load_bps = int(load_bps) // 1024
                    load_pps = int(load_pps)
                    color = "#FFFFFF" if speed is None else val_to_color(float(load_bps) / speed)
                    load_text = "{0}/{1}".format(b2ssize(load_bps, False), b2ssize(load_pps, base=1000))

                    if net_name in std_nets:
                        text = load_text
                    else:
                        text = H.div(net_name, _class="left") + H.div(load_text, _class="right")

                    table.add_cell(text, bgcolor=color, sorttable_customkey=str(load_bps))

            for i in range(max_net_count - len(net_loads)):
                table.add_cell("-", sorttable_customkey='0')

            table.next_row()

        report.add_block(6, name, table, "Network - " + name)


def show_host_io_load_in_color(report, cluster):
    rbts = collections.defaultdict(dict)
    wbts = collections.defaultdict(dict)
    bts = collections.defaultdict(dict)

    wiops = collections.defaultdict(dict)
    riops = collections.defaultdict(dict)
    iops = collections.defaultdict(dict)

    queue_depth = collections.defaultdict(dict)
    lat = collections.defaultdict(dict)
    io_time = collections.defaultdict(dict)
    w_io_time = collections.defaultdict(dict)

    for osd in cluster.osds:
        for dev_stat in (osd.data_stor_stats, osd.j_stor_stats):
            if dev_stat is not None:
                hname = osd.host.name
                load_stats = dev_stat.load
                dev = dev_stat.root_dev

                wbts[hname][dev] = load_stats.write_bytes
                rbts[hname][dev] = load_stats.read_bytes
                bts[hname][dev] = load_stats.write_bytes + load_stats.read_bytes

                wiops[hname][dev] = load_stats.write_iops
                riops[hname][dev] = load_stats.read_iops
                iops[hname][dev] = load_stats.write_iops + load_stats.read_iops

                queue_depth[hname][dev] = load_stats.w_io_time
                lat[hname][dev] = None if load_stats.lat is None else load_stats.lat * 1000
                io_time[hname][dev] = int(load_stats.io_time * 100)
                w_io_time[hname][dev] = load_stats.w_io_time

    if len(wbts) == 0:
        report.add_block(1, "No current IO load awailable", "")
        return

    loads = [
        (iops, 1000, 'IOPS', 50),
        (riops, 1000, 'Read IOPS', 30),
        (wiops, 1000, 'Write IOPS', 30),

        (bts, 1024, 'Bps', 100 * 1024 ** 2),
        (rbts, 1024, 'Read Bps', 100 * 1024 ** 2),
        (wbts, 1024, 'Write Bps', 100 * 1024 ** 2),

        (lat, None, 'Latency, ms', 20),
        (queue_depth, None, 'Average QD', 3),
        (io_time, None, 'Active time %', 100),
        # (w_io_time, None, 'W. time share', 5.0),
    ]

    report.add_block(12, "Current disk IO load", "")

    for pos, (target, base, tp, min_max_val) in enumerate(loads, 1):
        if target is lat:
            max_val = 300.0
        else:
            max_val = max(map(max, [data.values() for data in target.values()]))
            max_val = max(min_max_val, max_val)

        max_len = max(map(len, target.values()))

        table = html.HTMLTable(['host'] + ['load'] * max_len, zebra=False)
        for host_name, data in sorted(target.items()):
            table.add_cell(host_name)
            for dev, val in sorted(data.items()):
                color = "#FFFFFF" if val is None or max_val == 0 else val_to_color(float(val) / max_val)

                if target is lat:
                    s_val = '-' if val is None else str(int(val))
                elif target is queue_depth:
                    s_val = "%.1f" % (val,)
                elif base is None:
                    s_val = "%.1f" % (val,)
                else:
                    s_val = b2ssize(val, False, base=base)

                cell_data = H.div(dev, _class="left") + H.div(s_val, _class="right")
                table.add_cell(cell_data, bgcolor=color, sorttable_customkey=str(val))

            for i in range(max_len - len(data)):
                table.add_cell("-", sorttable_customkey='0')
            table.next_row()

        weight = max(2, min(12, max_len + 1 ))
        report.add_block(weight, tp, table, "Disk IO - " + tp)


def show_hosts_info(report, cluster):
    header_row = ["Name",
                  "Services",
                  "CPU's",
                  "RAM<br>total",
                  "RAM<br>free",
                  "Swap<br>used",
                  "Load avg<br>5m",
                  "Conn<br>tcp/udp"]
    table = html.HTMLTable(headers=header_row)
    for host in sorted(cluster.hosts.values(), key=lambda x: x.name):
        services = ["osd-{0}".format(osd_id) for osd_id in host.osd_ids]
        if host.mon_name:
            services.append("mon({0.mon_name})".format(host))

        table.add_cell(host.name)
        srv_strs = []
        steps = len(services) // 3 + (1 if len(services) % 3 else 0)
        for idx in range(steps):
            srv_strs.append(",".join(services[idx * 3: idx * 3 + 3]))
        table.add_cell("<br>".join(srv_strs))

        if host.hw_info is None:
            map(table.add_cell, ['-'] * (len(header_row) - 2))
            table.next_row()
            continue

        if host.hw_info.cores == []:
            table.add_cell("Error", sorttable_customkey='0')
        else:
            table.add_cell(sum(count for _, count in host.hw_info.cores))

        table.add_cell(b2ssize(host.mem_total), sorttable_customkey=str(host.mem_total))
        table.add_cell(b2ssize(host.mem_free), sorttable_customkey=str(host.mem_free))

        table.add_cell(b2ssize(host.swap_total - host.swap_free),
                       sorttable_customkey=str(host.swap_total - host.swap_free))
        table.add_cell(host.load_5m)
        table.add_cell("{0}/{1}".format(host.open_tcp_sock, host.open_udp_sock))
        table.next_row()

    report.add_block(6, "Host's info:", table)


def show_osd_pool_PG_distribution(report, cluster):
    if cluster.sum_per_osd is None:
        report.add_block(6, "PG copy per OSD: No pg dump data. Probably too many PG", "")
        return

    pools = sorted(cluster.sum_per_pool)
    table = html.HTMLTable(headers=["OSD/pool"] + list(pools) + ['sum'])

    for osd_id, row in sorted(cluster.osd_pool_pg_2d.items()):
        data = [osd_id] + [row.get(pool_name, 0) for pool_name in pools] + [cluster.sum_per_osd[osd_id]]
        table.add_row(map(str, data))

    table.add_cell("Total cluster PG", sorttable_customkey=str(max(osd.id for osd in cluster.osds) + 1))

    list(map(table.add_cell, (cluster.sum_per_pool[pool_name] for pool_name in pools)))
    table.add_cell(str(sum(cluster.sum_per_pool.values())))

    report.add_block(8, "PG copy per OSD:", table)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-l", "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default=None, help="Console log level")
    p.add_argument("-n", "--no-plots", help="Don't draw any plots, generate tables only", action="store_true")
    p.add_argument("-N", "--name", help="Report name", default="Nemo")
    p.add_argument("-o", '--out', help="report output folder", required=True)
    p.add_argument("-s", "--simple", help="Generate simple report", action="store_true")
    p.add_argument("-w", '--overwrite', action='store_true',  help="Overwrite result folder data")
    p.add_argument("-p", "--pretty-html", help="Prettify index.html", action="store_true")
    p.add_argument("data_folder", help="Folder with data, or .tar.gz archive")
    p.add_argument("--profile", help="Profile report creation", action="store_true")

    return p.parse_args(argv[1:])


def setup_logging(opts):
    lconf_path = os.path.join(os.path.dirname(__file__), 'logging.json')
    log_config = json.load(open(lconf_path))

    del log_config["handlers"]['log_file']

    if opts.log_level is not None:
        log_config["handlers"]["console"]["level"] = opts.log_level

    for settings in log_config["loggers"].values():
        settings['handlers'].remove('log_file')

    logging.config.dictConfig(log_config)


def main(argv):
    opts = parse_args(argv)
    setup_logging(opts)
    remove_folder = False

    logger.info("Generating report from %r to %r", opts.data_folder, opts.out)

    if opts.profile:
        prof = cProfile.Profile()
        prof.enable()

    if os.path.isfile(opts.data_folder):
        arch_name = opts.data_folder
        folder = tempfile.mkdtemp()
        logger.info("Unpacking %r to temporary folder %r", opts.data_folder, folder)
        remove_folder = True
        subprocess.call("tar -zxvf {0} -C {1} >/dev/null 2>&1".format(arch_name, folder), shell=True)
    elif not os.path.isdir(opts.data_folder):
        logger.error("Path argument (%r) should be a folder with data or path to archive", opts.data_folder)
        return 1
    else:
        folder = opts.data_folder

    index_path = os.path.join(opts.out, 'index.html')
    if os.path.exists(index_path):
        if not opts.overwrite:
            logger.error("%r already exists. Pass -w/--overwrite to overwrite files", index_path)
            return 1
    elif not os.path.exists(opts.out):
        os.makedirs(opts.out)

    try:
        storage = make_storage(folder, existing=True)
        storage = TypedStorage(storage)
        cluster = CephCluster(storage)
        logger.info("Loading cluster info into RAM")
        cluster.load()
        logger.info("Done")

        report = Report(opts.name, "index.html")
        report.style.append('body {font: 10pt sans;}')
        report.style_links.append("https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css")

        if opts.simple:
            dct = html.HTMLTable.def_table_attrs
            dct['class'] = dct['class'].replace("sortable", "").replace("zebra-table", "")
        else:
            report.script_links.append("http://www.kryogenix.org/code/browser/sorttable/sorttable.js")

        show_summary(report, cluster)
        report.next_line()

        show_hosts_info(report, cluster)
        show_mons_info(report, cluster)
        show_osd_state(report, cluster)
        report.next_line()

        show_osd_info(report, cluster)

        if cluster.has_performance_data:
            show_osd_perf_info(report, cluster)

        report.next_line()

        show_pools_info(report, cluster)
        show_pg_state(report, cluster)
        report.next_line()

        show_osd_pool_PG_distribution(report, cluster)
        report.next_line()

        if cluster.has_performance_data:
            show_host_io_load_in_color(report, cluster)
            report.next_line()

        if cluster.has_performance_data:
            show_host_network_load_in_color(report, cluster)
            report.next_line()

        if not opts.no_plots:
            from .plot_data import (show_osd_used_space_histo,
                                    show_osd_load,
                                    show_osd_lat_heatmaps,
                                    show_osd_ops_boxplot,
                                    show_osd_pg_histo)

            show_osd_used_space_histo(report, cluster)
            show_osd_pg_histo(report, cluster)
            report.next_line()

            if cluster.has_performance_data:
                show_osd_load(report, cluster)
                report.next_line()

            if cluster.has_performance_data:
                show_osd_lat_heatmaps(report, cluster)
                report.next_line()

            if cluster.has_performance_data:
                show_osd_ops_boxplot(report, cluster)
                report.next_line()

        report.save_to(opts.out, opts.pretty_html)
        logger.info("Report successfully stored to %r", index_path)
    finally:
        if remove_folder:
            shutil.rmtree(folder)

    if opts.profile:
        prof.disable()  # type: ignore
        s = io.StringIO()
        ps = pstats.Stats(prof, stream=s).sort_stats('time', 'calls')
        ps.print_stats(40)
        print(s.getvalue())


if __name__ == "__main__":
    exit(main(sys.argv))
