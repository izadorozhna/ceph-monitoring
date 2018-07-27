import io
import sys
import json
import time
import pstats
import shutil
import bisect
import inspect
import logging
import tempfile
import cProfile
import argparse
import subprocess
import collections
import logging.config
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Set, Sequence, Optional

import dataclasses

from cephlib.storage import make_storage, TypedStorage
from cephlib.units import b2ssize, b2ssize_10
from cephlib.common import flatten

from . import html
from .cluster import (NO_VALUE, CephInfo, Cluster, OSDStoreType, CephMonitor, CephOSD, Host,
                      FileStoreInfo, BlueStoreInfo, load_all, DiskLoad, OSDStatus, CephVersion)

from .plot_data import (per_info_required, show_osd_used_space_histo,
                        show_osd_load, show_osd_lat_heatmaps, show_osd_ops_boxplot, show_osd_pg_histo)

H = html.rtag
logger = logging.getLogger('cephlib.report')


HTML_UNKNOWN = H.font('???', color="orange")


def html_ok(text: str) -> html.TagProxy:
    return H.font(text, color="green")


def html_fail(text: str) -> html.TagProxy:
    return H.font(text, color="red")


CMap = List[Tuple[float, Tuple[float, float, float]]]


# replace with hvs mapping
def_color_map: CMap = [
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


def val_to_color(val: float, color_map: CMap = def_color_map) -> str:
    idx = [i[0] for i in color_map]
    assert idx == sorted(idx)
    pos = bisect.bisect_left(idx, val)
    if pos <= 0:
        ncolor: Sequence[float] = color_map[0][1]
    elif pos >= len(idx):
        ncolor = color_map[-1][1]
    else:
        color1 = color_map[pos - 1][1]
        color2 = color_map[pos][1]

        dx1 = (val - idx[pos - 1]) / (idx[pos] - idx[pos - 1])
        dx2 = (idx[pos] - val) / (idx[pos] - idx[pos - 1])

        ncolor = [(v1 * dx2 + v2 * dx1)
                  for v1, v2 in zip(color1, color2)]

    ncolor = [int((channel * 255 + 255) / 2) for channel in ncolor]
    return f"#{ncolor[0]:02X}{ncolor[1]:02X}{ncolor[2]:02X}"


class Report:
    def __init__(self, cluster_name: str, output_file_name: str) -> None:
        self.cluster_name = cluster_name
        self.output_file_name = output_file_name
        self.style: List[str] = []
        self.style_links: List[str] = []
        self.script_links: List[str] = []
        self.scripts: List[str] = []
        self.onload: List[str] = []
        self.divs: List[Tuple[Optional[str], Optional[str], str]] = []

    def add_block(self, header: Optional[str], block_obj: Any, menu_item: str = None):
        if menu_item is None:
            menu_item = header
        self.divs.append((header, menu_item, str(block_obj)))

    def save_to(self, output_dir: Path, pretty_html: bool = False):

        self.style_links.append("bootstrap.min.css")
        self.style_links.append("report.css")
        self.script_links.append("report.js")

        links: List[str] = []
        static_files_dir = Path(__file__).absolute().parent.parent / "html_js_css"

        for link in self.style_links + self.script_links:
            fname = link.rsplit('/', 1)[-1]
            src_path = static_files_dir / fname
            dst_path = output_dir / fname
            if src_path.exists():
                if not dst_path.is_dir():
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
                with doc.div(_class="menu-ceph"):
                    with doc.ul():
                        for sid, div_cnt in enumerate(self.divs):
                            if div_cnt is not None:
                                _, menu, _ = div_cnt
                                if menu:
                                    if menu.endswith(":"):
                                        menu = menu[:-1]
                                    doc.li.a(menu, onclick=f"clicked('{sid}')")

                for sid, div_cnt in enumerate(self.divs):
                    doc("\n")
                    if div_cnt is not None:
                        header, menu_item, block = div_cnt
                        if menu_item:
                            doc._enter("div", _class="main-ceph" if sid == 0 else "data-ceph", id=f"{sid}")

                        if header is None:
                            doc(block)
                        else:
                            doc.H3.center(header)
                            doc.br

                            if block != "":
                                doc.center(block)

                        if menu_item:
                            doc._exit()

        index = f"<!doctype html>{doc}"
        index_path = output_dir / self.output_file_name

        try:
            if pretty_html:
                from bs4 import BeautifulSoup
                index = BeautifulSoup.BeautifulSoup(index).prettify()
        except:
            pass

        index_path.open("w").write(index)


def get_all_versions(services: List[Union[CephOSD, CephMonitor]]) -> Dict[CephVersion, int]:
    all_version: Dict[CephVersion, int] = collections.Counter()
    for srv in services:
        all_version[srv.version] += 1
    return all_version


def show_cluster_summary(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    t = html.HTMLTable(["Setting", "Value"], id="table-summary", extra_cls="table_c")
    t.add_cells("Collected at", cluster.report_collected_at_local)
    t.add_cells("Collected at GMT", cluster.report_collected_at_gmt)
    t.add_cells("Status/overrall", f"{ceph.status.status.name.upper()} / {ceph.status.overall_status.name.upper()}")
    t.add_cells("PG count", str(ceph.status.num_pgs))
    t.add_cells("Pool count", str(len(ceph.pools)))
    t.add_cells("Used", b2ssize(ceph.status.bytes_used))
    t.add_cells("Avail", b2ssize(ceph.status.bytes_avail))
    t.add_cells("Data", b2ssize(ceph.status.data_bytes))

    avail_perc = ceph.status.bytes_avail * 100 / ceph.status.bytes_total
    t.add_cells("Free %", str(int(avail_perc)))
    t.add_cells("Mon count", str(len(ceph.mons)))
    t.add_cells("OSD count", str(len(ceph.osds)))

    mon_vers = get_all_versions(ceph.mons)
    t.add_cells("Mon version", ", ".join(map(str, sorted(mon_vers))))

    osd_vers = get_all_versions(ceph.osds)
    t.add_cells("OSD version", ", ".join(map(str, sorted(osd_vers))))

    t.add_cells("Monmap version", str(ceph.status.monmap_stat['epoch']))
    mon_tm = time.mktime(time.strptime(ceph.status.monmap_stat['modified'], "%Y-%m-%d %H:%M:%S.%f"))
    collect_tm = time.mktime(time.strptime(cluster.report_collected_at_local, "%Y-%m-%d %H:%M:%S"))
    diff = int(collect_tm - mon_tm)

    h = diff // 3600
    m = (diff % 3600) % 60
    s = diff % 60
    if h < 24:
        t.add_cells("Monmap modified in", f'{h}:{m:<02d}:{s:<02d}')
    else:
        d = h // 24
        h %= 24
        t.add_cells("Monmap modified in", f'{d}d {h}:{m:<02d}:{s:<02d}')

    return "Status:", t


def show_osd_summary(ceph: CephInfo) -> Tuple[str, Any]:
    osd_count = len(ceph.osds)
    stor_set = {osd.storage_type for osd in ceph.osds}
    stor_set_names = {stor.name for stor in stor_set}
    has_bs = OSDStoreType.bluestore in stor_set
    has_fs = OSDStoreType.filestore in stor_set

    if ceph.settings is None:
        t = H.font("No live OSD found!", color="red")
    else:
        t = html.HTMLTable(["Setting", "Value"], id="table-osd-summary", extra_cls="table_c")
        t.add_cells("Count", str(osd_count))
        t.add_cells("Average PG per OSD", str(ceph.status.num_pgs // osd_count))
        t.add_cells("Cluster net", str(ceph.cluster_net))
        t.add_cells("Public net", str(ceph.public_net))
        t.add_cells("Near full ratio", "%0.3f" % (float(ceph.settings.mon_osd_nearfull_ratio,)))
        t.add_cells("Full ratio", "%0.3f" % (float(ceph.settings.mon_osd_full_ratio,)))
        t.add_cells("Backfill full ratio", getattr(ceph.settings, "osd_backfill_full_ratio", "?"))
        t.add_cells("Filesafe full ratio", "%0.3f" % (float(ceph.settings.osd_failsafe_full_ratio,)))
        t.add_cells("Storage types", list(stor_set_names)[0]
                                     if len(stor_set_names) == 1
                                     else html_fail(",".join(map(str, stor_set_names))))

        if has_fs:
            t.add_cells("Journal aio", ceph.settings.journal_aio)
            t.add_cells("Journal dio", ceph.settings.journal_dio)
            t.add_cells("Filestorage sync", str(int(float(ceph.settings.filestore_max_sync_interval))) + 's')
        elif has_bs:
            # TODO: add BS info
            pass

    return "OSD:", t


def show_io_status(ceph: CephInfo) -> Tuple[str, Any]:
    t = html.HTMLTable(["IO type", "Value"], id="table-io-summary", extra_cls="table_c")
    t.add_cells("Client IO Write MiBps", b2ssize(ceph.status.write_bytes_sec // 2 ** 20))
    t.add_cells("Client IO Write OPS", b2ssize(ceph.status.write_op_per_sec))
    t.add_cells("Client IO Read MiBps", b2ssize(ceph.status.read_bytes_sec // 2 ** 20))
    t.add_cells("Client IO Read OPS", b2ssize(ceph.status.read_op_per_sec))

    t.add_cells("Recovery IO MiBps", "Not parsed yet")
    t.add_cells("Recovery OPS", "Not parsed yet")

    return "IO Activity:", t


def show_health_messages(ceph: CephInfo) -> Optional[Tuple[str, Any]]:
    colors = {
        "HEALTH_WARN": "orange",
        "HEALTH_ERR": "red"
    }

    if len(ceph.status.health_summary) != 0:
        t = html.Doc()

        if ceph.is_luminous:
            for check in ceph.status.health_summary.values():
                color = colors.get(check["severity"], "black")
                t.font(check['summary']['message'].capitalize(), color=color)
                t.br
        else:
            for msg in ceph.status.health_summary:
                color = colors.get(msg['severity'], "black")
                t.font(msg['summary'].capitalize(), color=color)
                t.br

        return "Status messages:", t


def show_mons_info(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    table = html.HTMLTable(headers=["Name", "Health", "Role", "Disk free<br>B (%)"],
                           id="table-mon-info", extra_cls="table_c")

    for mon in sorted(ceph.mons, key=lambda x: x.name):
        role = "Unknown"
        health = html_fail("HEALTH_FAIL")
        if mon.status is None:
            for mon_info in ceph.status.monmap_stat['mons']:
                if mon_info['name'] == mon.name:
                    if mon_info.get('rank') in [0, 1, 2, 3, 4]:
                        health = html_ok("HEALTH_OK")
                        role = "leader" if mon_info.get('rank') == 0 else "follower"
                        break
        else:
            health = html_ok("HEALTH_OK") if mon.status == "HEALTH_OK" else html_fail(mon.status)
            role = mon.role

        if mon.kb_avail is None:
            mon_db_root = "/var/lib/ceph/mon"
            root = ""
            blocks = 0
            available = 0
            for df_info in cluster.hosts[mon.host].df_info.values():
                if mon_db_root.startswith(df_info.mountpoint) and len(root) < len(df_info.mountpoint):
                    root = df_info.mountpoint
                    blocks = df_info.size
                    available = df_info.free

            if root == "":
                perc = "Unknown"
            else:
                perc = f"{b2ssize(available * 1024)} ({available * 100 // blocks})"
        else:
            perc = f"{b2ssize(mon.kb_avail * 1024)} ({mon.avail_percent})"

        table.add_row(map(str, [mon.name, health, role, perc]))

    return "Monitors info:", table


def show_pg_state(ceph: CephInfo) -> Tuple[str, Any]:
    statuses: Dict[str, int] = collections.Counter()

    for pg_group in ceph.status.pgmap_stat['pgs_by_state']:
        for state_name in pg_group['state_name'].split('+'):
            statuses[state_name] += pg_group["count"]

    table = html.HTMLTable(headers=["Status", "Count", "%"], id="table-pgs", extra_cls="table_lr")
    table.add_row(["any", str(ceph.status.num_pgs), "100.00"])
    for status, count in sorted(statuses.items()):
        table.add_row([status, str(count), f"{100.0 * count / ceph.status.num_pgs:.2f}"])

    return "PG's status:", table


def show_osd_state(ceph: CephInfo) -> Tuple[str, Any]:
    statuses: Dict[OSDStatus, List[str]] = collections.defaultdict(list)

    for osd in ceph.osds:
        statuses[osd.status].append(f"{osd.host.name}:{osd.id}")

    table = html.HTMLTable(headers=["Status", "Count", "ID's"],
                           id="table-osds-state", extra_cls="table_c")

    for status, osds in sorted(statuses.items()):
        table.add_row([html_ok(status.name) if status == OSDStatus.up else html_fail(status.name),
                       len(osds),
                    ("<br>".join(osds)
                     if len(osds) < 5
                     else "<br>".join(osds[:4]) + f"<br>and {len(osds) - 4} more")])

    return "OSD's state:", table


def show_pools_info(ceph: CephInfo) -> Tuple[str, Any]:
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
                                     "PG per OSD<br>Dev %"],
                            id="table-pools", extra_cls="table_lr")

    for _, pool in sorted(ceph.pools.items()):
        table.add_cell(pool.name)
        table.add_cell(str(pool.id))
        table.add_cell(str(pool.size))
        table.add_cell(str(pool.min_size))
        table.add_cell(b2ssize_10(int(pool.df.num_objects)), sorttable_customkey=str(int(pool.df.num_objects)))
        table.add_cell(b2ssize(int(pool.df.size_bytes)), sorttable_customkey=str(int(pool.df.size_bytes)))
        table.add_cell('---')
        table.add_cell(b2ssize(int(pool.df.read_bytes)), sorttable_customkey=str(int(pool.df.read_bytes)))
        table.add_cell(b2ssize(int(pool.df.write_bytes)), sorttable_customkey=str(int(pool.df.write_bytes)))
        table.add_cell(str(pool.crush_rule))

        if pool.pgp != pool.pg:
            table.add_cell(f"{pool.pg}({H.font(str(pool.pgp), color='red')})",
                           sorttable_customkey=str(pool.pg))
        else:
            table.add_cell(str(pool.pg), sorttable_customkey=str(pool.pg))

        bytes_per_pg = int(pool.df.size_bytes) // int(pool.pg)
        bytes_per_pg_s = b2ssize(bytes_per_pg)
        table.add_cell(bytes_per_pg_s, sorttable_customkey=str(bytes_per_pg))

        obj_per_pg = int(pool.df.num_objects) // int(pool.pg)
        obj_per_pg_s = b2ssize_10(obj_per_pg)
        table.add_cell(obj_per_pg_s, sorttable_customkey=str(obj_per_pg))

        if ceph.sum_per_osd is None:
            table.add_cell('-')
        else:
            row = [osd_pg.get(pool.name, 0) for osd_pg in ceph.osd_pool_pg_2d.values()]
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

                table.add_cell(f"{avg:.1f} ~ {dev_perc}%", sorttable_customkey=str(int(avg * 1000)))

        table.next_row()

    return "Pool's stats:", table


def show_osd_info(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    # osds_info = [get_osd_info(cluster, osd) for osd in cluster.osds]  # type: List[OSDInfo]

    w_headers = ["weight<br>" + root.name for root in ceph.crush.roots]

    stor_set = set(osd.storage_type for osd in ceph.osds)
    has_bs = OSDStoreType.bluestore in stor_set
    has_fs = OSDStoreType.filestore in stor_set
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

    table = html.HTMLTable(headers, id="table-osd-info", extra_cls="table_lr")

    all_vers = get_all_versions(ceph.osds)
    primary_ver = max(all_vers)

    for osd in sorted(ceph.osds, key=lambda x: x.id):
        if osd.daemon_runs is None:
            daemon_msg = HTML_UNKNOWN
        elif osd.daemon_runs:
            daemon_msg = html_ok('yes')
        else:
            daemon_msg = html_fail('no')

        table.add_cell(str(osd.id))
        table.add_cell(osd.host.name)
        table.add_cell(html_ok("up") if osd.status == OSDStatus.up else html_fail("down"))

        if osd.version is None:
            table.add_cell("Unknown")
        else:
            if len(all_vers) != 1:
                color = html_ok if osd.version == primary_ver else html_fail
            else:
                color = lambda x: x
            table.add_cell(color(str(osd.version)))
        table.add_cell(daemon_msg)

        if osd.run_info is not None:
            pinfo = osd.run_info.procinfo
            table.add_cell(str(pinfo["fd_count"]))
            socks = sum(int(tp.get('inuse', 0)) for tp in pinfo.get('ipv4', {}).values())
            socks += sum(int(tp.get('inuse', 0)) for tp in pinfo.get('ipv6', {}).values())
            table.add_cell(str(socks))
            table.add_cell(str(pinfo["th_count"]))
        else:
            table.add_cell("-")
            table.add_cell("-")
            table.add_cell("-")

        disks = cluster.hosts[osd.host.name].disks
        storage_devs = cluster.hosts[osd.host.name].storage_devs
        assert osd.storage_info is not None
        total_b = int(storage_devs[osd.storage_info.data_partition].size)

        for root in ceph.crush.roots:
            nodes = ceph.crush.find_nodes([('root', root.name), ('osd', f"osd.{osd.id}")])
            if not nodes:
                table.add_cell("-")
            elif len(nodes) > 1:
                table.add_cell("+")
            else:
                expected_weight = float(total_b) / (1024 ** 4)
                weight = f"{nodes[0].weight:.2f}"
                weight_s = html_fail(weight) if abs(1 - (nodes[0].weight / expected_weight)) > .2 else weight
                table.add_cell(weight_s)

        rew = f"{osd.reweight:.2f}"
        table.add_cell(html_fail(rew) if abs(osd.reweight - 1.0) > .01 else rew)

        if osd.pg_count is None:
            table.add_cell(HTML_UNKNOWN, sorttable_customkey="0")
        else:
            table.add_cell(str(osd.pg_count))

        if has_fs and has_bs:
            table.add_cell(str(osd.storage_type))

        data_drive_type = disks[osd.storage_info.data_dev].tp
        data_drive_type = (html_ok if data_drive_type in ('ssd', 'nvme') else lambda x: x)(data_drive_type)
        table.add_cell(data_drive_type)

        avail_perc = osd.free_perc
        color = "red" if avail_perc < 20 else ( "yellow" if avail_perc < 40 else "green")
        avail_perc_str = H.font(avail_perc, color=color)

        table.add_cell(b2ssize(total_b), sorttable_customkey=str(total_b))
        table.add_cell(avail_perc_str, sorttable_customkey=str(avail_perc))

        sinfo = osd.storage_info
        if isinstance(sinfo, FileStoreInfo):
            j_on_same_drive = html_ok("no") if sinfo.journal_dev != sinfo.data_dev else html_fail("yes")
            j_on_file = html_ok("no") if sinfo.journal_partition != sinfo.data_partition else html_fail("yes")
            j_drive_type = disks[sinfo.journal_dev].tp
            j_drive_type = (html_ok if j_drive_type in ('ssd', 'nvme') else html_fail)(j_drive_type)

            table.add_cell(j_on_same_drive)
            table.add_cell(j_drive_type)
            table.add_cell(j_on_file)

            if j_on_same_drive:
                j_size_s = '-'
            else:
                j_size = storage_devs[sinfo.journal_partition].size
                j_size_s = b2ssize(j_size)
                assert osd.run_info is not None
                osd_sync = float(osd.run_info.config['filestore_max_sync_interval'])
                j_size_s = j_size_s if j_size >= osd_sync * 100 * (1024 ** 2) else html_fail(j_size_s)
            table.add_cell(j_size_s)

            if has_bs:
                table.add_cell('-')
                table.add_cell('-')
                table.add_cell('-')
        else:
            assert isinstance(sinfo, BlueStoreInfo)
            wal_on_same_drive = html_ok("no") if sinfo.wal_dev != sinfo.data_dev else html_fail("yes")
            wal_drive_type = disks[sinfo.wal_dev].tp
            wal_drive_type = (html_ok if wal_drive_type in ('ssd', 'nvme') else html_fail)(wal_drive_type)

            if sinfo.wal_partition not in (sinfo.db_partition, sinfo.data_partition):
                wal_size = int(storage_devs[sinfo.wal_partition].size)
                wal_size_s = b2ssize(wal_size) + 'B'
                wal_size_s = html_fail(wal_size_s) if wal_size < 512 * 1024 * 1024 else wal_size_s
            else:
                wal_size_s = '-'

            table.add_cell(wal_on_same_drive)
            table.add_cell(wal_drive_type)
            if has_fs:
                table.add_cell('-')
            table.add_cell(wal_size_s)

            db_on_same_drive = html_ok("no") if sinfo.db_dev != sinfo.data_dev else html_fail("yes")
            db_drive_type = disks[sinfo.db_dev].tp
            db_drive_type = (html_ok if db_drive_type in ('ssd', 'nvme') else html_fail)(db_drive_type)
            if sinfo.db_partition != sinfo.data_partition:
                db_size = int(storage_devs[sinfo.db_partition].size)
                data_size = int(storage_devs[sinfo.data_partition].size)
                db_size_s = b2ssize(db_size) + "B"
                db_size_s = html_fail(db_size_s) if db_size < data_size * 0.0095 else db_size_s
            else:
                db_size_s = '-'

            table.add_cell(db_on_same_drive)
            table.add_cell(db_drive_type)
            table.add_cell(db_size_s)
        table.next_row()

    return "OSD's info:", table


@per_info_required
def show_osd_perf_info(ceph: CephInfo) -> Tuple[str, Any]:
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

    for osd in ceph.osds:
        perf_info: List[Tuple[Union[int, str], Union[float, None, int]]] = []

        assert osd.storage_info is not None

        if osd.storage_info.data_stor_stats is None:
            perf_info.extend([('-', 0)] * 5)
        else:
            load = osd.storage_info.data_stor_stats
            perf_info.extend([  # type: ignore
                (str(Path(osd.storage_info.data_dev).parent), None),
                (f"{b2ssize(load.read_bytes)} / {b2ssize(load.read_iops)}", load.read_bytes),
                (f"{b2ssize(load.write_bytes)} / {b2ssize(load.write_iops)}", load.write_bytes),
                (int(load.lat * 1000), load.lat) if load.lat is not None else ('-', 0),
                (int(load.io_time * 100), load.io_time)
            ])

        sinfo = osd.storage_info
        if isinstance(sinfo, FileStoreInfo):
            assert sinfo.j_stor_stats is not None
            assert sinfo.data_stor_stats is not None
            if sinfo.journal_dev != sinfo.data_dev:
                load = sinfo.j_stor_stats
                perf_info.extend([  # type: ignore
                    (str(Path(sinfo.journal_dev).parent), None),
                    (f"{b2ssize(load.write_bytes)}  / {b2ssize(load.write_iops)}", load.write_bytes),
                    (int(load.lat * 1000), load.lat) if load.lat is not None else ('-', 0),
                    (int(load.io_time * 100), load.io_time)
                ])
            else:
                perf_info.extend([("<--", None), ('-', 0), ('-', 0), ('-', 0)])
        else:
            perf_info.extend([('-', 0)] * 3)

        table.add_cell(str(osd.id))
        table.add_cell(osd.host.name)

        def lat2s(val):
            if val is None or val < 1e-6:
                return HTML_UNKNOWN
            elif val < 1e-3:
                return f"{val * 1000:.3f}"
            else:
                return f"{int(val * 1000)}"

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
                    for val in osd.osd_perf[name][::-1]:  # type: ignore
                        if val != NO_VALUE:
                            v_val = lat2s(val)
                            break
                    else:
                        v_val = HTML_UNKNOWN

                if s_val:
                    table.add_cell(f"{v_val} / {s_val}")
                else:
                    table.add_cell(v_val)
        else:
            for name, name_s in lats:
                if name_s:
                    table.add_cell(f"{HTML_UNKNOWN} / {HTML_UNKNOWN}")
                else:
                    table.add_cell(HTML_UNKNOWN)

        for val, sorted_val in perf_info:
            if sorted_val is None:
                table.add_cell(str(val))
            else:
                table.add_cell(str(val), sorttable_customkey=str(sorted_val))
        table.next_row()

    return "OSD's current load:", table


@per_info_required
def show_host_network_load_in_color(report: Report, cluster: Cluster, ceph: CephInfo):
    send_net_io: Dict[str, List[Tuple]] = collections.defaultdict(list)
    recv_net_io: Dict[str, List[Tuple]] = collections.defaultdict(list)

    max_net_count = 0
    for host in cluster.hosts.values():
        cl_if = host.find_interface(ceph.cluster_net)
        pb_if = host.find_interface(ceph.public_net)
        nets = [('cluster', cl_if), ('public', pb_if)]
        nets += sorted((net.dev, net) for net in host.net_adapters.values()  # type: ignore
                       if net is not None and net.is_phy and net != pb_if and net != cl_if )

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
    report.add_block("Network load", "")

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
                    load_text = f"{b2ssize(load_bps)}/{b2ssize_10(load_pps)}"

                    if net_name in std_nets:
                        text = load_text
                    else:
                        text = H.div(net_name, _class="left") + H.div(load_text, _class="right")

                    table.add_cell(text, bgcolor=color, sorttable_customkey=str(load_bps))

            for i in range(max_net_count - len(net_loads)):
                table.add_cell("-", sorttable_customkey='0')

            table.next_row()

        report.add_block(name, table, "Network - " + name)


@dataclasses.dataclass
class CephHDDInfo:
    hostname: str
    name: str

    wbts: float
    rbts: float
    bts: float
    wiops: float
    riops: float
    iops: float
    queue_depth: float
    lat: Optional[float]
    io_time: float
    w_io_time: float

    roles: Set[str]

    @classmethod
    def from_disk_info(cls, hostname: str, name: str, load: DiskLoad) -> 'CephHDDInfo':
        return cls(hostname=hostname,
                   name=name,
                   wbts=load.write_bytes,
                   rbts=load.read_bytes,
                   bts=load.write_bytes + load.read_bytes,
                   wiops=load.write_iops,
                   riops=load.read_iops,
                   iops=load.write_iops + load.read_iops,
                   queue_depth=load.w_io_time,
                   lat=None if load.lat is None else load.lat * 1000,
                   io_time=int(load.io_time * 100),
                   w_io_time=load.w_io_time,
                   roles=set())


@per_info_required
def show_host_io_load_in_color(report: Report, ceph: CephInfo):
    devs: Dict[str, Dict[str, CephHDDInfo]] = collections.defaultdict(dict)

    for osd in ceph.osds:
        sinfo = osd.storage_info
        assert sinfo is not None
        assert sinfo.data_stor_stats is not None
        osd_devs: List[Tuple[str, DiskLoad]] = [(sinfo.data_dev, sinfo.data_stor_stats)]
        roles = {"data"}

        if isinstance(sinfo, FileStoreInfo):
            assert sinfo.j_stor_stats is not None
            osd_devs.append((sinfo.journal_dev, sinfo.j_stor_stats))
            roles.add("journal")
        else:
            assert isinstance(sinfo, BlueStoreInfo)
            assert sinfo.db_stor_stats is not None
            assert sinfo.wal_stor_stats is not None

            osd_devs.append((sinfo.db_dev, sinfo.db_stor_stats))
            roles.add("db")
            osd_devs.append((sinfo.wal_dev, sinfo.wal_stor_stats))
            roles.add("wal")

        hset = devs[osd.host.name]
        for dev, stat in osd_devs:
            if dev not in hset:
                hset[dev] = CephHDDInfo.from_disk_info(osd.host.name, dev, stat)
            hset[dev].roles.update(roles)

    if len(devs) == 0:
        report.add_block("No current IO load available", "")
        return

    loads = [
        ('iops', b2ssize_10, 'IOPS', 50),
        ('riops', b2ssize_10, 'Read IOPS', 30),
        ('wiops', b2ssize_10, 'Write IOPS', 30),

        ('bts', b2ssize, 'Bps', 100 * 1024 ** 2),
        ('rbts', b2ssize, 'Read Bps', 100 * 1024 ** 2),
        ('wbts', b2ssize, 'Write Bps', 100 * 1024 ** 2),

        ('lat', None, 'Latency, ms', 20),
        ('queue_depth', None, 'Average QD', 3),
        ('io_time', None, 'Active time %', 100),
    ]

    report.add_block("Current disk IO load", "")

    for pos, (target_attr, tostrfunc, tp, min_max_val) in enumerate(loads, 1):
        if target_attr == 'lat':
            max_val = 300.0
        else:
            max_val = max(map(max, [getattr(dev, target_attr)  # type: ignore
                                    for all_host_devs in devs.values()
                                    for dev in all_host_devs.values()]))
            max_val = max(min_max_val, max_val)

        max_len = max(map(len, devs.values()))

        table = html.HTMLTable(['host'] + ['load'] * max_len, zebra=False)
        for host_name, devices in sorted(devs.items()):
            # TODO: mark devices with data/journal/wal/db tags
            table.add_cell(host_name)
            for dev_name, dev_info in sorted(devices.items()):
                val = getattr(dev_info, target_attr)
                color = "#FFFFFF" if val is None or max_val == 0 else val_to_color(float(val) / max_val)

                if target_attr == 'lat':
                    s_val = '-' if val is None else str(int(val))
                elif target_attr == 'queue_depth':
                    s_val = "%.1f" % (val,)
                elif tostrfunc is None:
                    s_val = "%.1f" % (val,)
                else:
                    s_val = tostrfunc(val)

                cell_data = H.div(dev_name, _class="left") + H.div(s_val, _class="right")
                table.add_cell(cell_data, bgcolor=color, sorttable_customkey=str(val))

            for i in range(max_len - len(devices)):
                table.add_cell("-", sorttable_customkey='0')
            table.next_row()

        report.add_block(tp, table, "Disk IO - " + tp)


def show_hosts_info(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    header_row = ["Name",
                  "Services",
                  "CPU's",
                  "RAM<br>total",
                  "RAM<br>free",
                  "Swap<br>used",
                  "Load avg<br>5m",
                  "Conn<br>tcp/udp",
                  "IP's"]
    table = html.HTMLTable(headers=header_row, id="table-hosts-info", extra_cls="table_c")
    for host in sorted(cluster.hosts.values(), key=lambda x: x.name):
        services = [f"osd-{osd.id}" for osd in ceph.osds if osd.host is host]
        for mon in ceph.mons:
            if mon.host is host:
                services.append(f"mon({host.name})")

        table.add_cell(f"""<a onclick="clicked('host-{host.name}')">{host.name}</a>""")
        srv_strs = []
        steps = len(services) // 3 + (1 if len(services) % 3 else 0)
        for idx in range(steps):
            srv_strs.append(",".join(services[idx * 3: idx * 3 + 3]))
        table.add_cell("<br>".join(srv_strs))

        if host.hw_info is None:
            map(table.add_cell, ['-'] * (len(header_row) - 2))
            table.next_row()
            continue

        if not host.hw_info.cpu_info:
            table.add_cell("Error", sorttable_customkey='0')
        else:
            table.add_cell(str(sum(inf.cores for inf in host.hw_info.cpu_info)))

        table.add_cell(b2ssize(host.mem_total), sorttable_customkey=str(host.mem_total))
        table.add_cell(b2ssize(host.mem_free), sorttable_customkey=str(host.mem_free))

        table.add_cell(b2ssize(host.swap_total - host.swap_free),
                       sorttable_customkey=str(host.swap_total - host.swap_free))
        table.add_cell(str(host.load_5m))
        table.add_cell(f"{host.open_tcp_sock}/{host.open_udp_sock}")

        all_ip = flatten(adapter.ips for adapter in host.net_adapters.values())
        table.add_cell("<br>".join(str(addr.ip) for addr in all_ip if not addr.ip.is_loopback))
        table.next_row()

    return "Host's info:", table


def show_osd_pool_PG_distribution(ceph: CephInfo) -> Tuple[str, Any]:
    if ceph.sum_per_osd is None:
        return "PG copy per OSD: No pg dump data. Probably too many PG", ""

    pools_order_t = sorted((-count, name) for name, count in ceph.sum_per_pool.items())
    # pools = sorted(cluster.sum_per_pool)
    pools = [name for _, name in pools_order_t]

    name_headers = []
    for name in pools:
        if len(name) > 10:
            parts = name.split(".")
            best_idx = 0
            best_delta = len(name) + 1
            for idx in range(len(parts)):
                sz1 = len(".".join(parts[:idx]))
                sz2 = len(".".join(parts[idx:]))
                if abs(sz2 - sz1) < best_delta:
                    best_idx = idx
                    best_delta = abs(sz2 - sz1)
            if best_idx != 0 and best_idx != len(parts):
                name = ".".join(parts[:best_idx]) + '.' + '<br>' + ".".join(parts[best_idx:])
        name_headers.append(name)

    table = html.HTMLTable(headers=["OSD/pool"] + name_headers + ['sum'],
                           id="table-pg-per-osd", extra_cls="table_c")

    for osd_id, row in sorted(ceph.osd_pool_pg_2d.items()):
        data = [osd_id] + [row.get(pool_name, 0) for pool_name in pools] + [ceph.sum_per_osd[osd_id]]
        table.add_row(map(str, data))

    table.add_cell("Total cluster PG", sorttable_customkey=str(max(osd.id for osd in ceph.osds) + 1))

    list(map(table.add_cell, (ceph.sum_per_pool[pool_name] for pool_name in pools)))
    table.add_cell(str(sum(ceph.sum_per_pool.values())))

    return "PG copy per OSD:", table


def host_info(host: Host) -> str:
    doc = html.Doc()
    with doc.div(id=f"host-{host.name}", _class="ceph-data"):
        doc.center.H3(host.name)
        doc.br
        doc.center.H4("Interfaces")
        doc.br
        table = html.HTMLTable(None, extra_cls="table_c hostinfo-net")
        for name, adapter in sorted(host.net_adapters.items()):
            table.add_row(
                [name,
                 f"{adapter.dev}<br>Is phy: {adapter.is_phy}<br>Duplex: {adapter.duplex}<br>MTU: {adapter.mtu}<br>" +
                 f"Speed: {adapter.speed}<br>" + "".join(f"{ip.ip} / {ip.net.prefixlen}" for ip in adapter.ips)])
        doc.center(str(table))
        doc.br
        doc.center.H4("HW disks")
        doc.br
        table = html.HTMLTable(["Name", "Type", "Size", "Rota", "Scheduler", "Model info", "RQ-size", "Phy Sec",
                                "Min IO"],
                                extra_cls="table_c hostinfo-disks")

        mib_and_mb = lambda x: f"{b2ssize(x)}B / {b2ssize_10(x)}B"

        for _, disk in sorted(host.disks.items()):
            table.add_row([disk.name, disk.tp, mib_and_mb(disk.size),
                           str(disk.rota), str(disk.scheduler),
                           "Model: {0.model}<br>Vendor: {0.vendor}<br>Serial:{0.serial}".format(disk.hw_model),
                           str(disk.rq_size), str(disk.phy_sec), str(disk.min_io)])

        doc.center(str(table))
        doc.br
        doc.center.H4("Mountable")
        doc.br
        table = html.HTMLTable(["Name", "Size", "Mountpoint", "Fs", "Free space", "Label"],
                               extra_cls="table_c hostinfo-mountable",
                               table_attrs={'class': 'table-bordered zebra-table'})

        def run_over_children(obj, level: int):
            table.add_row([f'<div class="disk-children-{level}">{obj.name}</div>',
                           mib_and_mb(obj.size),
                           obj.mountpoint if obj.mountpoint else '',
                           obj.fs if obj.fs else '',
                           mib_and_mb(obj.free_space) if obj.free_space is not None else '',
                           "" if obj.label is None else obj.label])
            for _, ch in sorted(obj.children.items()):
                run_over_children(ch, level + 1)

        for _, disk in sorted(host.disks.items()):
            run_over_children(disk, 0)

        doc.center(str(table))

    return str(doc)


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
    log_config = json.load((Path(__file__).parent / 'logging.json').open())

    del log_config["handlers"]['log_file']

    if opts.log_level is not None:
        log_config["handlers"]["console"]["level"] = opts.log_level

    for settings in log_config["loggers"].values():
        settings['handlers'].remove('log_file')

    logging.config.dictConfig(log_config)


def main(argv: List[str]):
    opts = parse_args(argv)
    setup_logging(opts)
    remove_folder = False

    logger.info("Generating report from %r to %r", opts.data_folder, opts.out)

    if opts.profile:
        prof = cProfile.Profile()
        prof.enable()

    if Path(opts.data_folder).is_file():
        arch_name = opts.data_folder
        folder = tempfile.mkdtemp()
        logger.info("Unpacking %r to temporary folder %r", opts.data_folder, folder)
        remove_folder = True
        subprocess.call(f"tar -zxvf {arch_name} -C {folder} >/dev/null 2>&1", shell=True)
    elif not Path(opts.data_folder).is_dir():
        logger.error("Path argument (%r) should be a folder with data or path to archive", opts.data_folder)
        return 1
    else:
        folder = opts.data_folder

    out_p = Path(opts.out)
    index_path = out_p / 'index.html'
    if index_path.exists():
        if not opts.overwrite:
            logger.error("%r already exists. Pass -w/--overwrite to overwrite files", index_path)
            return 1
    elif not out_p.exists():
        out_p.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading cluster info into RAM")
        cluster, ceph = load_all(TypedStorage(make_storage(folder, existing=True)))
        logger.info("Done")

        report = Report(opts.name, "index.html")
        report.style.append('body {font: 10pt sans;}')
        report.style_links.append("https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css")

        if opts.simple:
            dct = html.HTMLTable.def_table_attrs
            dct['class'] = dct['class'].replace("sortable", "").replace("zebra-table", "")
        else:
            report.script_links.append("http://www.kryogenix.org/code/browser/sorttable/sorttable.js")

        cluster_reporters = [
            show_cluster_summary,
            show_osd_summary,
            show_io_status,
            show_health_messages,
            show_hosts_info,
            show_mons_info,
            show_osd_state,
            show_osd_info,
            show_osd_perf_info,
            show_pools_info,
            show_pg_state,
            show_osd_pool_PG_distribution,
            show_host_io_load_in_color,
            show_host_network_load_in_color,
            show_osd_used_space_histo,
            show_osd_pg_histo,
            show_osd_load,
            show_osd_lat_heatmaps,
            show_osd_ops_boxplot
        ]

        params = {"ceph": ceph, "cluster": cluster, "report": report}

        for reporter in cluster_reporters:
            if getattr(reporter, "perf_info_required", False) and not cluster.has_performance_data:
                continue
            sig = inspect.signature(reporter)
            curr_params = {name: params[name] for name in sig.parameters}
            new_block = reporter(**curr_params)

            if new_block is not None:
                assert 'report' not in sig.parameters
                report.add_block(*new_block)

        for _, host in sorted(cluster.hosts.items()):
            report.add_block(None, host_info(host))

        report.save_to(Path(opts.out), opts.pretty_html)
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
