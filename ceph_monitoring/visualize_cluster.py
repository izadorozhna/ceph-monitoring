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
import urllib.request
import logging.config
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Set, Sequence, Optional, Iterable, TypeVar

import dataclasses
import numpy

from cephlib.storage import make_storage, TypedStorage
from cephlib.units import b2ssize, b2ssize_10
from cephlib.common import flatten

from . import html
from .cluster_classes import (CephInfo, Cluster, OSDStoreType, Host, FileStoreInfo, BlueStoreInfo, DiskLoad, OSDStatus,
                              OSDSInfo, DiskType)
from .cluster import NO_VALUE, load_all, get_all_versions, get_nodes_pg_info

from .plot_data import (per_info_required, plot, show_osd_used_space_histo,
                        show_osd_load, show_osd_lat_heatmaps, show_osd_ops_boxplot, get_histo_img, get_kde_img)

H = html.rtag
logger = logging.getLogger('cephlib.report')


HTML_UNKNOWN = H.font('???', color="orange")


def html_ok(text: str) -> html.TagProxy:
    return H.font(text, color="green")


def html_fail(text: str) -> html.TagProxy:
    return H.font(text, color="red")


CMap = Sequence[Tuple[float, Tuple[float, float, float]]]


# replace with hvs mapping
def_color_map: CMap = (
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
)


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


T = TypeVar('T')


def partition(items: Iterable[T], size: int) -> Iterable[List[T]]:
    curr = []
    for idx, val in enumerate(items):
        curr.append(val)
        if (idx + 1) % size == 0:
            yield curr
            curr = []

    if curr:
        yield curr


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

    def save_to(self, output_dir: Path, pretty_html: bool = False,
                embed: bool = False):

        self.style_links.append("bootstrap.min.css")
        self.style_links.append("report.css")
        self.script_links.append("report.js")
        self.script_links.append("sorttable_utf.js")

        links: List[str] = []
        static_files_dir = Path(__file__).absolute().parent.parent / "html_js_css"

        def get_path(link: str) -> Tuple[bool, str]:
            if link.startswith("http://") or link.startswith("https://"):
                return False, link
            fname = link.rsplit('/', 1)[-1]
            return True, static_files_dir / fname

        for link in self.style_links + self.script_links:
            local, fname = get_path(link)
            data = None

            if local:
                if embed:
                    data = open(fname, 'rb').read().decode("utf8")
                else:
                    shutil.copyfile(fname, output_dir / Path(fname).name)
            else:
                try:
                    data = urllib.request.urlopen(fname, timeout=10)
                except (TimeoutError, urllib.request.URLError):
                    logger.warning(f"Can't retrive {fname}")

            if data is not None:
                if link in self.style_links:
                    self.style.append(data)
                else:
                    self.scripts.append(data)
            else:
                links.append(link)

        css_links = links[:len(self.style_links)]
        js_links = links[len(self.style_links):]

        doc = html.Doc()
        with doc.html:
            with doc.head:
                doc.title("Ceph cluster report: " + self.cluster_name)

                for url in css_links:
                    doc.link(href=url, rel="stylesheet", type="text/css")

                if self.style:
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


def show_cluster_summary(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    t = html.HTMLTable("table-summary", ["Setting", "Value"], sortable=False)
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


# TODO: need to move this into checks.py module
def show_issues_table(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    t = html.HTMLTable("table-issues", ["Check", "Result", "Comment"], sortable=False)

    failed = html_fail("Failed")
    passed = html_ok("Passed")

    # ---------------------------------  No scrub errors ---------------------------------------------------------------

    if ceph.osds_info:
        total_scrub_err = 0
        for osd_info in ceph.osds_info.osds.values():
            total_scrub_err += osd_info.pg_stats.scrub_errors + \
                               osd_info.pg_stats.deep_scrub_errors + \
                               osd_info.pg_stats.shallow_scrub_errors

        t.add_cells("No scrub errors",
                    passed if total_scrub_err == 0 else failed,
                    "" if total_scrub_err == 0 else f"Error count {total_scrub_err}")
    else:
        t.add_cells("No scrub errors", failed, "No data available")

    # ---------------------------------  SSD/NVME journals -------------------------------------------------------------

    count = 0
    for osd in ceph.osds:
        if isinstance(osd.storage_info, FileStoreInfo):
            if not osd.host.disks[osd.storage_info.journal_dev].tp.is_fast():
                count += 1
        else:
            if not osd.host.disks[osd.storage_info.wal_dev].tp.is_fast():
                count += 1
            elif not osd.host.disks[osd.storage_info.db_dev].tp.is_fast():
                count += 1

    t.add_cells("SSD/NVME used for journals/wal/dev",
                passed if count == 0 else failed,
                "" if 0 == count else f"{count} OSD's failed")

    # ---------------------------------  separated ceph nets -----------------------------------------------------------

    t.add_cells("Separated client/cluster ceph networks",
                passed if ceph.cluster_net != ceph.public_net else failed,
                "" if ceph.cluster_net != ceph.public_net else f"Same net {ceph.public_net}<br>is used for both")

    # ---------------------------------  journal per drive count -------------------------------------------------------

    host_dev_j_count = collections.Counter()
    for osd in ceph.osds:
        if isinstance(osd.storage_info, FileStoreInfo):
            host_dev_j_count[f"{osd.host.name}/{osd.storage_info.journal_dev}"] += 1
        else:
            host_dev_j_count[f"{osd.host.name}/{osd.storage_info.wal_dev}"] += 1
            if osd.storage_info.db_dev != osd.storage_info.wal_dev:
                host_dev_j_count[f"{osd.host.name}/{osd.storage_info.db_dev}"] += 1

    fcount = 0
    for host in cluster.hosts.values():
        for disk in host.disks.values():
            count = host_dev_j_count[f"{host.name}/{disk.name}"]
            if (count > 1 and not disk.tp.is_fast()) or (count > 4 and disk.tp != DiskType.nvme) or (count > 6):
                fcount += 1

    t.add_cells("Max 6 journals per nvme, 4 per SSD, 1 per HDD",
                passed if fcount == 0 else failed,
                "" if fcount == 0 else f"Failed for {fcount} drives")

    # ---------------------------------  PG counts for OSD's -----------------------------------------------------------

    wrong_pg_count = 0

    for osd in ceph.osds:
        if osd.pg_count < 100 or osd.pg_count > 400:
            wrong_pg_count += 1

    t.add_cells("400 >= OSD PG count >= 100",
                passed if wrong_pg_count == 0 else failed,
                "" if wrong_pg_count == 0 else f"Failed for {wrong_pg_count} drives")

    # ---------------------------------  Pool size ---------------------------------------------------------------------

    wrong_size = 0
    for name, pool in ceph.pools.items():
        expected_size = (2, 1) if name == 'gnocci' else (3, 2)
        if (pool.size, pool.min_size) != expected_size:
            wrong_size += 1

    t.add_cells("3/2 for all major pools, 2/1 for gnocci",
                passed if wrong_size == 0 else failed,
                "" if wrong_size == 0 else f"Failed for {wrong_size} pools")

    # ---------------------------------  PG counts for pools -----------------------------------------------------------

    # ---------------------------------  Net performance ---------------------------------------------------------------

    # ---------------------------------  Scrubbing at night ------------------------------------------------------------

    # ---------------------------------  OSD threads and tcp connections -----------------------------------------------

    # ---------------------------------  Same pg and pgp for all pools -------------------------------------------------

    pg_ne_pgp = 0
    for name, pool in ceph.pools.items():
        if pool.pg != pool.pgp:
            pg_ne_pgp += 1

    t.add_cells("PG == PGP for all pools",
                passed if pg_ne_pgp == 0 else failed,
                "" if pg_ne_pgp == 0 else f"Failed for {pg_ne_pgp} pools")
    # ---------------------------------  OSD nodes load ----------------------------------------------------------------

    # ---------------------------------  OSD ram consumption -----------------------------------------------------------

    # ---------------------------------  Second level nodes for all crush roots either count > 3 or same size ----------

    # ---------------------------------  OSD of the same type data evenly distributed ----------------------------------

    data_per_osd = collections.defaultdict(list)
    for osd in ceph.osds:
        total_space = osd.used_space + osd.free_space
        data_per_osd[total_space].append(osd.free_space / total_space)

    # for key, vals in data_per_osd.items():
    #     print(f"{key} => {min(vals):.3f}, {max(vals):.3f}")

    for sz, free in data_per_osd.items():
        if abs(max(free) - min(free)) > 0.2:
            t.add_cells("(Max used OSD - min used OSD) < 0.2 for same OSD sizes", failed, "")
            break
    else:
        t.add_cells("(Max used OSD - min used OSD) < 0.2 for same OSD sizes", passed, "")

    # ---------------------------------  OSD of the same type load evenly distributed ----------------------------------

    # ---------------------------------  Journal & wal & db sizes ------------------------------------------------------

    bad_size_count = 0
    for osd in ceph.osds:
        osd_info = ceph.osds_info.osds[osd.id]
        if osd_info.j_info:
            sync = float(osd.run_info.config['filestore_max_sync_interval']) if osd.run_info else 5
            min_j_size = osd_info.data_drive_type.bandwith_mbps() * 2 * sync

            if osd_info.j_info.size / 2 ** 20 < min_j_size:
                bad_size_count += 1
        else:
            assert osd_info.wal_db_info
            assert isinstance(osd.storage_info, BlueStoreInfo)
            data_part_size = osd.host.storage_devs[osd.storage_info.data_partition].size
            wal_part_size = osd.host.storage_devs[osd.storage_info.wal_partition].size

            min_wal_size =  data_part_size * 0.005
            min_db_size = data_part_size * 0.05

            if osd_info.wal_db_info.wal_size is not None:
                if osd_info.wal_db_info.wal_size < min_wal_size:
                    bad_size_count += 1
            else:
                if osd.storage_info.wal_partition == osd.storage_info.db_partition:
                    if wal_part_size < min_wal_size + min_db_size:
                        bad_size_count += 1
                elif wal_part_size < min_wal_size:
                    bad_size_count += 1
                elif osd.host.storage_devs[osd.storage_info.db_partition].size < min_db_size:
                    bad_size_count += 1


    t.add_cells("Journal and DB has enough space",
                passed if bad_size_count == 0 else failed,
                "" if fcount == 0 else f"Failed for {bad_size_count} journals/wal/db")

    return "Issues:", t


def show_osd_summary(ceph: CephInfo) -> Tuple[str, Any]:
    osd_count = len(ceph.osds)
    stor_set = {osd.storage_type for osd in ceph.osds}
    stor_set_names = {stor.name for stor in stor_set}
    has_bs = OSDStoreType.bluestore in stor_set
    has_fs = OSDStoreType.filestore in stor_set

    if ceph.settings is None:
        t = H.font("No live OSD found!", color="red")
    else:
        t = html.HTMLTable("table-osd-summary", ["Setting", "Value"], sortable=False)
        t.add_cells("Count", str(osd_count))
        pgcount = sum(pool.size * pool.pg for pool in ceph.pools.values())
        t.add_cells("Average PG copy per OSD", str(pgcount // osd_count))
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
    t = html.HTMLTable("table-io-summary", ["IO type", "Value"], sortable=False)
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
    return None


def show_mons_info(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    table = html.HTMLTable("table-mon-info", ["Name", "Health", "Role", "Disk free<br>B (%)"])

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
                sort_by = "0"
            else:
                perc = f"{b2ssize(available * 1024)} ({available * 100 // blocks})"
                sort_by = str(available)
        else:
            perc = f"{b2ssize(mon.kb_avail * 1024)} ({mon.avail_percent})"
            sort_by = str(mon.kb_avail)

        table.add_cell(mon.name)
        table.add_cell(health)
        table.add_cell(role)
        table.add_cell(perc, sorttable_customkey=sort_by)
        table.next_row()

    return "Monitors info:", table


def show_pg_state(ceph: CephInfo) -> Tuple[str, Any]:
    statuses: Dict[str, int] = collections.Counter()

    for pg_group in ceph.status.pgmap_stat['pgs_by_state']:
        for state_name in pg_group['state_name'].split('+'):
            statuses[state_name] += pg_group["count"]

    table = html.HTMLTable("table-pgs", ["Status", "Count", "%"])
    table.add_row(["any", str(ceph.status.num_pgs), "100.00"])
    for status, count in sorted(statuses.items()):
        table.add_row([status, str(count), f"{100.0 * count / ceph.status.num_pgs:.2f}"])

    return "PG's status:", table


def show_osd_state(ceph: CephInfo) -> Tuple[str, Any]:
    statuses: Dict[OSDStatus, List[str]] = collections.defaultdict(list)

    for osd in ceph.osds:
        statuses[osd.status].append(str(osd.id))

    table = html.HTMLTable("table-osds-state", ["Status", "Count", "ID's"])

    for status, osds in sorted(statuses.items()):
        table.add_row([html_ok(status.name) if status == OSDStatus.up else html_fail(status.name),
                       len(osds),
                      "<br>".join(", ".join(grp) for grp in partition(osds, 20))])

    return "OSD's state:", table


def show_pools_info(ceph: CephInfo) -> Tuple[str, Any]:
    table = html.HTMLTable("table-pools",
                           ["Pool",
                            "Id",
                            "Size/Min_size",
                            "Obj",
                            "Bytes",
                            "Ruleset",
                            "PG(PGP)",
                            "PG<br>recommended",
                            "Bytes/PG",
                            "Objs/PG",
                            "PG per OSD<br>Dev %"],
                            align=html.TableAlign.left_right)

    table2 = html.HTMLTable("table-pools-io", ["Pool", "Read B", "Write B", "Read ops", "Write ops", "Rd/PG", "Wr/PG"],
                            align=html.TableAlign.left_right)

    # TODO: separate info and load (read/write/iops)
    # add iops per PG
    # add recommended PG count

    for _, pool in sorted(ceph.pools.items()):
        table.add_cell(pool.name)
        table.add_cell(str(pool.id))
        vl = f"{pool.size} / {pool.min_size}"
        if pool.size != 3 or pool.min_size != 2:
            vl = html_fail(vl)
        table.add_cell(vl, sorttable_customkey=str(pool.size))
        table.add_cell_b2ssize_10(pool.df.num_objects)
        table.add_cell_b2ssize(pool.df.size_bytes)
        table.add_cell(str(pool.crush_rule))

        if pool.pgp != pool.pg:
            table.add_cell(f"{pool.pg}({H.font(str(pool.pgp), color='red')})",
                           sorttable_customkey=str(pool.pg))
        else:
            table.add_cell(str(pool.pg))

        table.add_cell("-")

        table.add_cell_b2ssize(pool.df.size_bytes // pool.pg)
        table.add_cell_b2ssize_10(pool.df.num_objects // pool.pg)

        if ceph.sum_per_osd is None:
            table.add_cell('-')
        else:
            osds = list(ceph.crush.iter_osds_for_rule(pool.crush_rule))
            osds_pgs = []

            for osd in osds:
                osds_pgs.append(ceph.osd_pool_pg_2d[osd.id].get(pool.name, 0))

            avg = float(sum(osds_pgs)) / len(osds_pgs)
            if len(osds_pgs) < 2:
                dev = None
            else:
                dev = (sum((i - avg) ** 2.0 for i in osds_pgs) / (len(osds_pgs) - 1)) ** 0.5

            if avg < 1E-5:
                table.add_cell('--')
            else:
                if dev is None:
                    dev_perc = "--"
                else:
                    dev_perc = str(int(dev * 100. / avg))

                table.add_cell(f"{avg:.1f} ~ {dev_perc}%", sorttable_customkey=str(int(avg * 1000)))

        table.next_row()

        table2.add_cell(pool.name)
        table2.add_cell_b2ssize(pool.df.read_bytes)
        table2.add_cell_b2ssize(pool.df.write_bytes)
        table2.add_cell_b2ssize_10(pool.df.read_ops)
        table2.add_cell_b2ssize_10(pool.df.write_ops)
        table2.add_cell_b2ssize_10(pool.df.read_ops // pool.pg)
        table2.add_cell_b2ssize_10(pool.df.write_ops // pool.pg)
        table2.next_row()

    return "Pool's stats:", f"<center>{table}<br><br><br><H4>Load info</H4><br>{table2}</center>"


def get_osd_load_table(cluster: Cluster, ceph: CephInfo, osds_info: OSDSInfo) -> str:
    table = html.HTMLTable("table-osd-load",
                           ["ID", "node", "PG's", "open<br>files", "ip<br>conn", "thr", "RSS", "VMM", "CPU<br>Used, s",
                            "Total<br>data", "Read<br>ops", "Read", "Write<br>ops", "Write"] +
                           (["+Bps", "Read<br>IOPS", "Read<br>Bps", "Write<br>IOPS", "Write<br>Bps"]
                             if ceph.pgs_second else []),
                           align=html.TableAlign.left_right)

    for osd in sorted(ceph.osds, key=lambda x: x.id):
        osd_info = osds_info.osds[osd.id]

        table.add_cell(str(osd.id))
        table.add_cell(osd.host.name)

        #   PG COUNT
        if osd.pg_count is None:
            table.add_cell(HTML_UNKNOWN, sorttable_customkey="0")
        else:
            table.add_cell(str(osd.pg_count))

        #  RUN INFO - FD COUNT, TCP CONN, THREADS
        if osd_info.run_info is not None:
            table.add_cell(str(osd_info.run_info.fd_count))
            table.add_cell(str(osd_info.run_info.opened_socks))
            table.add_cell(str(osd_info.run_info.th_count))
            table.add_cell_b2ssize(osd_info.run_info.vm_rss)
            table.add_cell_b2ssize(osd_info.run_info.vm_size)
            table.add_cell_b2ssize_10(osd_info.run_info.cpu_usage)
        else:
            for i in range(6):
                table.add_cell("")

        #  USER DATA SIZE & TOTAL IO
        if osd_info.pg_stats:
            table.add_cell_b2ssize(osd_info.pg_stats.bytes)
            table.add_cell_b2ssize_10(osd_info.pg_stats.reads)
            table.add_cell_b2ssize(osd_info.pg_stats.read_b)
            table.add_cell_b2ssize_10(osd_info.pg_stats.writes)
            table.add_cell_b2ssize(osd_info.pg_stats.write_b)
        else:
            for i in range(5):
                table.add_cell("")

        # USER DATA SIZE DELTA
        if osd_info.d_stats:
            table.add_cell_b2ssize(osd_info.d_stats.d_used_space)
            table.add_cell_b2ssize_10(osd_info.d_stats.d_reads)
            table.add_cell_b2ssize(osd_info.d_stats.d_read_b)
            table.add_cell_b2ssize_10(osd_info.d_stats.d_writes)
            table.add_cell_b2ssize(osd_info.d_stats.d_write_b)
        else:
            for i in range(5):
                table.add_cell("")

        table.next_row()

    return str(table)


def show_osd_info(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    osds_info = ceph.osds_info

    headers = ["ID", "node", "status", "version [hash]", "daemon<br>run", ] + \
              ["weight<br>" + root.name for root in ceph.crush.roots] + \
              ["reweight", "PG", "Scrub<br>ERR"] + \
              (["Type"] if osds_info.has_bs and osds_info.has_fs else []) + \
              ["Storage<br>dev type", "Storage<br>total", "Storage<br>free %"]

    if osds_info.has_bs and osds_info.has_fs:
        headers.extend(["Journal<br>or wal<br>colocated", "Journal<br>or wal<br>dev type",
                        "Journal<br>on file", "Journal<br>or wal<br>size"])
    elif osds_info.has_fs and not osds_info.has_bs:
        headers.extend(["Journal<br>colocated", "Journal<br>dev type",
                        "Journal<br>on file", "Journal<br>size"])
    else:
        headers.extend(["wal<br>colocated", "wal<br>dev type", "wal<br>size"])

    if osds_info.has_bs:
        headers.extend(["DB<br>colocated", "DB<br>dev type", "DB<br>size"])

    table = html.HTMLTable("table-osd-info", headers, align=html.TableAlign.left_right)

    for osd in sorted(ceph.osds, key=lambda x: x.id):
        osd_info = osds_info.osds[osd.id]

        if osd.daemon_runs is None:
            daemon_msg = HTML_UNKNOWN
        else:
            daemon_msg = html_ok('yes') if osd.daemon_runs else html_fail('no')

        table.add_cell(str(osd.id))
        table.add_cell(osd.host.name)
        table.add_cell(html_ok("up") if osd.status == OSDStatus.up else html_fail("down"))

        if osd.version is None:
            table.add_cell("Unknown")
        else:
            if len(osds_info.all_versions) != 1:
                color = html_ok if osd.version == osds_info.largest_ver else html_fail
            else:
                color = lambda x: x
            table.add_cell(color(str(osd.version)))
        table.add_cell(daemon_msg)

        for root in ceph.crush.roots:
            if root.name not in osd_info.crush_trees_weights:
                table.add_cell("-")
            else:
                wok, weight, expected_weight = osd_info.crush_trees_weights[root.name]

                if not wok:
                    table.add_cell("+")
                else:
                    weight_s = f"{weight:.2f}"
                    table.add_cell(html_fail(weight_s) if abs(1 - weight / expected_weight) > .2 else weight_s)

        #   REWEIGTHS
        rew = f"{osd.reweight:.2f}"
        table.add_cell(html_fail(rew) if abs(osd.reweight - 1.0) > .01 else rew)

        #   PG COUNT
        if osd.pg_count is None:
            table.add_cell(HTML_UNKNOWN, sorttable_customkey="0")
        else:
            table.add_cell(str(osd.pg_count))

        if osd_info.pg_stats:
            table.add_cell(str(osd_info.pg_stats.scrub_errors +
                               osd_info.pg_stats.deep_scrub_errors +
                               osd_info.pg_stats.shallow_scrub_errors))
        else:
            table.add_cell("")

        # STORAGE TYPE
        if osds_info.has_fs and osds_info.has_bs:
            table.add_cell(str(osd.storage_type))

        # STORAGE DEV TYPE
        data_drive_color_fn = html_ok if osd_info.data_drive_type.is_fast() else lambda x: x
        table.add_cell(data_drive_color_fn(osd_info.data_drive_type.name))

        #  USED & FREE SPACE
        color = "red" if osd.free_perc < 20 else ( "yellow" if osd.free_perc < 40 else "green")
        avail_perc_str = H.font(osd.free_perc, color=color)
        table.add_cell_b2ssize(osd_info.total_space)
        table.add_cell(avail_perc_str, sorttable_customkey=str(osd.free_perc))

        # JOURNAL/WAL/DB info
        if osd_info.j_info:
            table.add_cell(str(osd_info.j_info.collocation))
            table.add_cell(osd_info.j_info.drive_type.name)
            table.add_cell(str(osd_info.j_info.on_file))

            if osd_info.j_info.size is None:
                j_size_s = '-'
            else:
                j_size_s = b2ssize(osd_info.j_info.size)
                assert osd.run_info is not None
                osd_sync = float(osd.run_info.config['filestore_max_sync_interval'])
                j_size_s = j_size_s if osd_info.j_info.size >= osd_sync * 100 * (1024 ** 2) else html_fail(j_size_s)
            table.add_cell(j_size_s)

            if osds_info.has_bs:
                table.add_cell('-')
                table.add_cell('-')
                table.add_cell('-')
        else:
            assert osd_info.wal_db_info is not None
            wd_info = osd_info.wal_db_info
            wal_on_same_drive = html_ok("no") if wd_info.wal_collocation else html_fail("yes")

            table.add_cell(wal_on_same_drive)
            wal_drive_type_color_fn = html_ok if wd_info.wal_drive_type.is_fast() else html_fail
            table.add_cell(wal_drive_type_color_fn(wd_info.wal_drive_type.name))

            if osds_info.has_fs:
                table.add_cell('-')

            if wd_info.wal_size is not None:
                wal_size_s = b2ssize(wd_info.wal_size) + 'B'
                table.add_cell(html_fail(wal_size_s) if wd_info.wal_size < 512 * 1024 * 1024 else wal_size_s,
                               sorttable_customkey=str(wd_info.wal_size))
            else:
                table.add_cell('-')

            db_colocation = html_ok("no") if wd_info.db_collocation else html_fail("yes")
            db_drive_type_color_fn = html_ok if wd_info.wal_drive_type.is_fast() else html_fail
            if wd_info.db_size is not None:
                db_size_s = b2ssize(wd_info.db_size) + "B"
                size_sort_by = str(wd_info.db_size)
                db_size_s = html_fail(db_size_s) if wd_info.db_size < osd_info.data_part_size * 0.0095 else db_size_s
            else:
                db_size_s = '-'
                size_sort_by = ''

            table.add_cell(db_colocation)
            table.add_cell(db_drive_type_color_fn(wd_info.wal_drive_type.name))
            table.add_cell(db_size_s, sorttable_customkey=size_sort_by)

        table.next_row()

    load_table = get_osd_load_table(cluster, ceph, osds_info)
    return "OSD's info:", f"<center>{table}<br><br><br><H4>Run info:</H4><br>{load_table}</center>"


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

        table = html.HTMLTable(headers=["host", "cluster", "public"] +
                                       ["hw adapter"] * (max_net_count - ceph_net_count),
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

        table = html.HTMLTable(headers=['host'] + ['load'] * max_len, zebra=False)
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


def host_link(hostname: str) -> str:
    return f"""<a class="link" onclick="clicked('host-{hostname}')">{hostname}</a>"""


def hosts_pg_info(cluster: Cluster, ceph: CephInfo) -> str:
    if ceph.osds_info is None:
        return ""

    header_row = ["Name",
                  "PGs",
                  "User data",
                  "Reads",
                  "Read B",
                  "Writes",
                  "Write B",
                  "User data<br>store rate",
                  "Reads<br>ps",
                  "Read Bps",
                  "Writes<br>ps",
                  "Write Bps",
                  ]
    table = html.HTMLTable("table-hosts-pg-info", header_row, align=html.TableAlign.center_right)
    hosts_pgs_info = get_nodes_pg_info(ceph)
    for host in sorted(cluster.hosts.values(), key=lambda x: x.name):
        if host.name in hosts_pgs_info:
            table.add_cell(host_link(host.name))
            pgs_info = hosts_pgs_info[host.name]
            table.add_cell(str(len(pgs_info.pgs)))
            table.add_cell_b2ssize(pgs_info.io_stat.bytes)
            table.add_cell_b2ssize_10(pgs_info.io_stat.reads)
            table.add_cell_b2ssize(pgs_info.io_stat.read_b)
            table.add_cell_b2ssize_10(pgs_info.io_stat.writes)
            table.add_cell_b2ssize(pgs_info.io_stat.write_b)

            if pgs_info.dio_stat:
                table.add_cell_b2ssize(pgs_info.dio_stat.d_used_space)
                table.add_cell_b2ssize_10(pgs_info.dio_stat.d_reads)
                table.add_cell_b2ssize(pgs_info.dio_stat.d_read_b)
                table.add_cell_b2ssize_10(pgs_info.dio_stat.d_writes)
                table.add_cell_b2ssize(pgs_info.dio_stat.d_write_b)

            table.next_row()

    return f"<br><br><center><H4>Hosts PG's info:</H4><br><br>{table}</center>"


def show_hosts_info(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    headers = ["Name",
               "Services",
               "CPU's",
               "RAM<br>total",
               "RAM<br>free",
               "Swap<br>used",
               "Load avg<br>5m",
               "Conn<br>tcp/udp",
               "IP's"]

    if ceph.osds_info:
        headers.append("Scrub err")
        hosts_pgs_info = get_nodes_pg_info(ceph)
    else:
        hosts_pgs_info = None

    table = html.HTMLTable("table-hosts-info", headers, align=html.TableAlign.center_right)
    for host in sorted(cluster.hosts.values(), key=lambda x: x.name):

        srv_strs = []
        for mon in ceph.mons:
            if mon.host == host.name:
                srv_strs.append(f'''<font color="#8080FF">Mon</font>: {host.name}''')

        table.add_cell(host_link(host.name))
        osd_services = [str(osd.id) for osd in ceph.osds if osd.host is host]
        if osd_services:
            srv_strs.append('''<font color="#c77405">OSD's</font>: ''' + ", ".join(osd_services[:3]))
            for ids in partition(osd_services[3:], 4):
                srv_strs.append(", ".join(ids))

        table.add_cell("<br>".join(srv_strs))

        if host.hw_info is None:
            for i in ['-'] * (len(headers) - 2):
                table.add_cell(i)
            table.next_row()
            continue

        if not host.hw_info.cpu_info:
            table.add_cell("Error", sorttable_customkey='0')
        else:
            table.add_cell(str(sum(inf.cores for inf in host.hw_info.cpu_info)))

        table.add_cell_b2ssize(host.mem_total)
        table.add_cell_b2ssize(host.mem_free)

        table.add_cell_b2ssize(host.swap_total - host.swap_free)
        table.add_cell(str(host.load_5m))
        table.add_cell(f"{host.open_tcp_sock}/{host.open_udp_sock}",
                       sorttable_customkey=str(host.open_tcp_sock + host.open_udp_sock))

        all_ip = flatten(adapter.ips for adapter in host.net_adapters.values())
        table.add_cell("<br>".join(str(addr.ip) for addr in all_ip if not addr.ip.is_loopback))
        if hosts_pgs_info:
            pgs_info = hosts_pgs_info[host.name]
            table.add_cell(str(pgs_info.io_stat.scrub_errors + pgs_info.io_stat.deep_scrub_errors
                               + pgs_info.io_stat.shallow_scrub_errors))
        table.next_row()

    return "Host's info:", str(table) + hosts_pg_info(cluster, ceph)


def show_osd_pool_pg_distribution(ceph: CephInfo) -> Tuple[str, Any]:
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

    table = html.HTMLTable("table-pg-per-osd", ["OSD/pool"] + name_headers + ['sum'])

    for osd_id, row in sorted(ceph.osd_pool_pg_2d.items()):
        data = [osd_id] + [row.get(pool_name, 0) for pool_name in pools] + [ceph.sum_per_osd[osd_id]]
        table.add_row(map(str, data))

    table.add_cell("Total cluster PG", sorttable_customkey=str(max(osd.id for osd in ceph.osds) + 1))

    list(map(table.add_cell, (ceph.sum_per_pool[pool_name] for pool_name in pools)))
    table.add_cell(str(sum(ceph.sum_per_pool.values())))

    return "PG copy per OSD:", table


def host_info(host: Host, ceph: CephInfo) -> str:
    cluster_networks = [(ceph.public_net, 'ceph-public'), (ceph.cluster_net, 'ceph-cluster')]

    doc = html.Doc()
    with doc.div(id=f"host-{host.name}", _class="data-ceph"):
        doc.center.H3(host.name)
        doc.br
        doc.center.H4("Interfaces")
        doc.br

        table = html.HTMLTable(headers=["Name", "Type", "Duplex", "MTU", "Speed", "IP's", "Roles"],
                               extra_cls=["hostinfo-net"],
                               sortable=False,
                               align=html.TableAlign.left_center)

        def add_adapter_line(adapter, name):
            tp = "phy" if adapter.is_phy else ('bond' if adapter.dev in host.bonds else 'virt')
            roles = [role for net, role in cluster_networks if any((ip.ip in net) for ip in adapter.ips)]
            table.add_row(
                [name,
                 tp,
                 "" if adapter.duplex is None else str(adapter.duplex),
                 "" if adapter.mtu is None else str(adapter.mtu),
                 "" if adapter.speed is None else str(adapter.speed),
                 "<br>".join(f"{ip.ip} / {ip.net.prefixlen}" for ip in adapter.ips),
                 "<br>".join(roles)])

        all_ifaces = set(host.net_adapters)

        # first show bonds
        for _, bond in sorted(host.bonds.items()):
            assert bond.name in all_ifaces
            all_ifaces.remove(bond.name)
            adapter = host.net_adapters[bond.name]
            add_adapter_line(adapter, adapter.dev)

            for src in bond.sources:
                assert src in all_ifaces
                all_ifaces.remove(src)
                add_adapter_line(host.net_adapters[src],
                                 f'<div class="disk-children-1">+{src}</dev>')

            for adapter_name in list(all_ifaces):
                if adapter_name.startswith(bond.name + '.'):
                    all_ifaces.remove(adapter_name)
                    add_adapter_line(host.net_adapters[adapter_name],
                                     f'<div class="disk-children-2">->{adapter_name}</dev>')

            table.add_row(["-------"] * len(table.headers))

        for name in sorted(all_ifaces):
            if name != 'lo':
                add_adapter_line(host.net_adapters[name], name)

        doc.center(str(table))
        doc.br
        doc.center.H4("HW disks")
        doc.br

        mib_and_mb = lambda x: f"{b2ssize(x)}B / {b2ssize_10(x)}B"

        stor_roles = collections.defaultdict(lambda: collections.defaultdict(list))

        for osd in ceph.osds:
            if osd.host is host:
                stor_roles[osd.storage_info.data_partition]['data'].append(osd.id)
                stor_roles[osd.storage_info.data_dev]['data'].append(osd.id)
                if osd.storage_type == OSDStoreType.filestore:
                    assert isinstance(osd.storage_info, FileStoreInfo)
                    stor_roles[osd.storage_info.journal_partition]['journal'].append(osd.id)
                    stor_roles[osd.storage_info.journal_dev]['journal'].append(osd.id)
                else:
                    assert isinstance(osd.storage_info, BlueStoreInfo)
                    stor_roles[osd.storage_info.db_partition]['db'].append(osd.id)
                    stor_roles[osd.storage_info.db_dev]['db'].append(osd.id)
                    stor_roles[osd.storage_info.wal_partition]['wal'].append(osd.id)
                    stor_roles[osd.storage_info.wal_dev]['wal'].append(osd.id)

        order = ['data', 'journal', 'db', 'wal']

        def html_roles(name: str) -> str:
            return "<br>".join(name + ": " + ", ".join(map(str, sorted(ids)))
                               for name, ids in sorted(stor_roles[name].items(), key=lambda x: order.index(x[0])))

        table = html.HTMLTable(headers=["Name", "Type", "Size", "Roles", "Rota", "Scheduler",
                                        "Model info", "RQ-size", "Phy Sec", "Min IO"],
                                extra_cls=["hostinfo-disks"])

        for _, disk in sorted(host.disks.items()):

            if disk.hw_model.model and disk.hw_model.vendor:
                model = f"Model: {disk.hw_model.vendor.strip()} / {disk.hw_model.model}"
            elif disk.hw_model.model:
                model = f"Model: {disk.hw_model.model}"
            elif disk.hw_model.vendor:
                model = f"Vendor: {disk.hw_model.vendor.strip()}"
            else:
                model = ""

            serial = "" if disk.hw_model.serial is None else ("Serial: " + disk.hw_model.serial)

            if model and serial:
                info = f"{model}<br>{serial}"
            elif model or serial:
                info = model if model else serial
            else:
                info = ""

            table.add_row([disk.name, disk.tp.name, mib_and_mb(disk.size),
                           html_roles(disk.path),
                           str(disk.rota),
                           "" if disk.scheduler is None else str(disk.scheduler),
                           info,
                           str(disk.rq_size),
                           str(disk.phy_sec),
                           str(disk.min_io)])

        doc.center(str(table))
        doc.br
        doc.center.H4("Mountable")
        doc.br
        table = html.HTMLTable(headers=["Name", "Size", "Type", "Mountpoint", "Ceph role", "Fs", "Free space", "Label"],
                               extra_cls=["hostinfo-mountable"], sortable=False)

        def run_over_children(obj, level: int):
            table.add_row([f'<div class="disk-children-{level}">{obj.name}</div>',
                           mib_and_mb(obj.size),
                           obj.tp.name if level == 0 else '',
                           obj.mountpoint if obj.mountpoint else '',
                           "" if level == 0 else html_roles(obj.path),
                           obj.fs if obj.fs else '',
                           mib_and_mb(obj.free_space) if obj.free_space is not None else '',
                           "" if obj.label is None else obj.label])
            for _, ch in sorted(obj.children.items()):
                run_over_children(ch, level + 1)

        for _, disk in sorted(host.disks.items()):
            run_over_children(disk, 0)

        doc.center(str(table))

    return str(doc)


@plot
def show_osd_pg_histo(ceph: CephInfo) -> Optional[Tuple[str, Any]]:
    vals = [osd.pg_count for osd in ceph.osds if osd.pg_count is not None]
    if vals:
        return "PG per OSD", get_histo_img(numpy.array(vals))
    return None


@plot
def show_pg_size_kde(ceph: CephInfo) -> Optional[Tuple[str, Any]]:
    if ceph.pgs:
        vals = [pg.stat_sum.num_bytes for pg in ceph.pgs.pgs.values()]
        # return "PG sizes", get_kde_img(numpy.array(vals))
        return "PG sizes", get_histo_img(numpy.array(vals))
    return None


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-l", "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default=None, help="Console log level")
    p.add_argument("-n", "--no-plots", help="Don't draw any plots, generate tables only", action="store_true")
    p.add_argument("-N", "--name", help="Report name", default="Nemo")
    p.add_argument("-o", '--out', help="report output folder", required=True)
    p.add_argument("-w", '--overwrite', action='store_true',  help="Overwrite result folder data")
    p.add_argument("-p", "--pretty-html", help="Prettify index.html", action="store_true")
    p.add_argument("data_folder", help="Folder with data, or .tar.gz archive")
    p.add_argument("--profile", help="Profile report creation", action="store_true")
    p.add_argument("-e", "--embed", action='store_true', help="Embed js/css files into report to make it stand-alone")
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

        cluster_reporters = [
            show_cluster_summary,
            show_issues_table,
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
            show_osd_pool_pg_distribution,
            show_host_io_load_in_color,
            show_host_network_load_in_color,
            show_osd_used_space_histo,
            show_osd_pg_histo,
            show_osd_load,
            show_osd_lat_heatmaps,
            show_osd_ops_boxplot,
            show_pg_size_kde
        ]

        params = {"ceph": ceph, "cluster": cluster, "report": report}

        for reporter in cluster_reporters:
            if getattr(reporter, "perf_info_required", False) and not cluster.has_performance_data:
                continue
            if getattr(reporter, 'plot', False) and opts.no_plots:
                continue
            sig = inspect.signature(reporter)
            curr_params = {name: params[name] for name in sig.parameters}
            new_block = reporter(**curr_params)

            if new_block is not None:
                assert 'report' not in sig.parameters
                report.add_block(*new_block)

        for _, host in sorted(cluster.hosts.items()):
            report.add_block(None, host_info(host, ceph))

        report.save_to(Path(opts.out), opts.pretty_html, embed=opts.embed)
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
