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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union, Sequence, Optional, Iterable, TypeVar, Callable, NamedTuple

import yaml
import numpy

from cephlib.storage import make_storage, TypedStorage
from cephlib.units import b2ssize, b2ssize_10
from cephlib.common import flatten

from . import html
from .html import html_ok, html_fail, partition, partition_by_len

from .cluster_classes import (CephInfo, Cluster, Host, FileStoreInfo, BlueStoreInfo, OSDStatus, LogicBlockDev,
                              CephOSD, CephMonitor, CephVersion, DiskType, CephDevInfo, DevPath, OSDPGStats,
                              NodePGStats, Disk)
from .cluster import NO_VALUE, load_all, get_nodes_pg_info
from .checks import run_all_checks, expected_wr_speed
from .plot_data import (perf_info_required, plot, show_osd_used_space_histo,
                        show_osd_load, show_osd_lat_heatmaps, show_osd_ops_boxplot, get_histo_img, get_kde_img)

logger = logging.getLogger('cephlib.report')


H = html.rtag
HTML_UNKNOWN = H.font('???', color="orange")


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


Link = NamedTuple('Link', [('link', str), ('id', str)])


def host_link(hostname: str) -> Link:
    return Link(f"""<a class="link link-host" onclick="clicked('host-{hostname}')">{hostname}</a>""",
                f'host-{hostname}')


def osd_link(osd_id: int) -> Link:
    return Link(f"""<a class="link link-osd" onclick="clicked('osd-{osd_id}')">{osd_id}</a>""",
                f'osd-{osd_id}')


def mon_link(name: str) -> Link:
    return Link(f"""<a class="link link-mon" onclick="clicked('mon-{name}')">{name}</a>""",
                f'mon-{name}')


def err_link(reporter_id: str, mess: str = None) -> Link:
    return Link(f"""<a class="link link-err" onclick="clicked('err-{reporter_id}')">{mess}</a>""",
                f'err-{reporter_id}')


def rule_link(rule_name: str, rule_id: int) -> Link:
    return Link(f"""<a class="link link-rule" onclick="clicked('ruleset_info')">{rule_name}({rule_id})</a>""",
                'ruleset_info')


def pool_link(pool_name: str) -> Link:
    return Link(f"""<a class="link link-pool" onclick="clicked('pools_info')">{pool_name}</a>""",
                'pools_info')


class Report:
    def __init__(self, cluster_name: str, output_file_name: str) -> None:
        self.cluster_name = cluster_name
        self.output_file_name = output_file_name
        self.style: List[str] = []
        self.style_links: List[str] = []
        self.script_links: List[str] = []
        self.scripts: List[str] = []
        self.divs: List[Tuple[str, Optional[str], Optional[str], str]] = []

    def add_block(self, name: str, header: Optional[str], block_obj: Any, menu_item: str = None):
        if menu_item is None:
            menu_item = header
        self.divs.append((name, header, menu_item, str(block_obj)))

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
                    logger.warning(f"Can't retrieve {fname}")

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

            with doc.body(onload="onHashChanged()"):
                with doc.div(_class="menu-ceph"):
                    with doc.ul():
                        for div in self.divs:
                            if div is not None:
                                name, _, menu, _ = div
                                if menu:
                                    if menu.endswith(":"):
                                        menu = menu[:-1]
                                    doc.li.a(menu, onclick=f"clicked('{name}')")

                for div in self.divs:
                    doc("\n")
                    if div is not None:
                        name, header, menu_item, block = div
                        with doc.div(_class="data-ceph", id=name):
                            if header is None:
                                doc(block)
                            else:
                                doc.H3.center(header)
                                doc.br

                                if block != "":
                                    doc.center(block)

        index = f"<!doctype html>{doc}"
        index_path = output_dir / self.output_file_name

        try:
            if pretty_html:
                from bs4 import BeautifulSoup
                index = BeautifulSoup.BeautifulSoup(index).prettify()
        except:
            pass

        index_path.open("w").write(index)


def seconds_to_str(seconds: Union[int, float]) -> str:
    seconds = int(seconds)

    s = seconds % 60
    m = (seconds // 60) % 60
    h = (seconds // 3600) % 24
    d = seconds // (3600 * 24)

    if s != 0 and h != 0:
        if d == 0:
            return f"{h}:{m:<02d}:{s:<02d}"
        return f"{d} days {h}:{m:<02d}:{s:<02d}"

    data = []
    if d != 0:
        data.append(f"{d} days")
    if h != 0:
        data.append(f"{h}h")
    if m != 0:
        data.append(f"{m}m")
    if s != 0:
        data.append(f"{s}s")

    return " ".join(data)


def get_all_versions(services: Iterable[Union[CephOSD, CephMonitor]]) -> Dict[CephVersion, int]:
    all_version: Dict[CephVersion, int] = collections.Counter()
    for srv in services:
        all_version[srv.version] += 1
    return all_version


def tab(name: str) -> Callable[[Callable], Callable]:
    def closure(func: Callable) -> Callable:
        func.report_name = name
        return func
    return closure


@tab("Status")
def show_cluster_summary(cluster: Cluster, ceph: CephInfo) -> html.HTMLTable:
    class SummaryTable(html.Table):
        setting = html.ident()
        value = html.to_str()

    t = SummaryTable()
    t.add_row("Collected at", cluster.report_collected_at_local)
    t.add_row("Collected at GMT", cluster.report_collected_at_gmt)
    t.add_row("Status", ceph.status.status.name.upper())
    t.add_row("PG count", ceph.status.num_pgs)
    t.add_row("Pool count", len(ceph.pools))
    t.add_row("Used", b2ssize(ceph.status.bytes_used))
    t.add_row("Avail", b2ssize(ceph.status.bytes_avail))
    t.add_row("Data", b2ssize(ceph.status.data_bytes))

    avail_perc = ceph.status.bytes_avail * 100 / ceph.status.bytes_total
    t.add_row("Free %", int(avail_perc))
    t.add_row("Mon count", len(ceph.mons))
    t.add_row("OSD count", len(ceph.osds))

    mon_vers = get_all_versions(ceph.mons.values())
    t.add_row("Mon version", ", ".join(map(str, sorted(mon_vers))))

    osd_vers = get_all_versions(ceph.osds.values())
    t.add_row("OSD version", ", ".join(map(str, sorted(osd_vers))))

    if ceph.has_fs and ceph.has_bs:
        stor_types = "filestore & bluestore"
    elif ceph.has_fs:
        stor_types = "filestore"
    elif ceph.has_bs:
        stor_types = "bluestore"
    else:
        assert False

    t.add_row("Storage types",  stor_types)

    t.add_row("Monmap version", ceph.status.monmap_stat['epoch'])
    mon_tm = time.mktime(time.strptime(ceph.status.monmap_stat['modified'], "%Y-%m-%d %H:%M:%S.%f"))
    collect_tm = time.mktime(time.strptime(cluster.report_collected_at_local, "%Y-%m-%d %H:%M:%S"))
    t.add_row("Monmap modified in", seconds_to_str(int(collect_tm - mon_tm)))

    # pgmap:
        # "degraded_objects": 16
        # "degraded_total": 38761686
        # "degraded_ratio": 0.000000
        # "misplaced_objects": 4507
        # "misplaced_total": 38761686
        # "misplaced_ratio": 0.000116
        # "recovering_objects_per_sec": 14
        # "recovering_bytes_per_sec": 47604609
        # "recovering_keys_per_sec": 0
        # "num_objects_recovered": 156
        # "num_bytes_recovered": 500393209
        # "num_keys_recovered": 0,


    # colors = {
    #     #     "HEALTH_WARN": "orange",
    #     #     "HEALTH_ERR": "red"
    #     # }
    #     #
    #     # if len(ceph.status.health_summary) != 0:
    #     #     mess = html.Doc()
    #     #
    #     #     mess.br
    #     #     mess.br
    #     #     mess.br
    #     #     mess.br
    #     #     mess.br
    #     #
    #     #     mess.center.h4("Status messages:")
    #     #     if ceph.is_luminous:
    #     #         for check in ceph.status.health_summary.values():
    #     #             color = colors.get(check["severity"], "black")
    #     #             mess.font(check['summary']['message'].capitalize(), color=color)
    #     #             mess.br
    #     #     else:
    #     #         for msg in ceph.status.health_summary:
    #     #             color = colors.get(msg['severity'], "black")
    #     #             mess.font(msg['summary'].capitalize(), color=color)
    #     #             mess.br
    #     #
    #     #     mess_s = str(mess)
    #     # else:
    #     #     mess_s = ""

    return t.html(id="table-summary", sortable=False)
    # return "Status:", str(t) + mess_s


def show_issues_table(cluster: Cluster, ceph: CephInfo, report: Report):
    config = yaml.load((Path(__file__).parent / 'check_conf.yaml').open())
    check_results = run_all_checks(config, cluster, ceph)

    t = html.HTMLTable("table-issues", ["Check", "Result", "Comment"], sortable=False)

    failed = html_fail("Failed")
    passed = html_ok("Passed")

    err_per_test = collections.defaultdict(list)

    for result in check_results:
        t.add_cells(result.check_description, passed if result.passed else failed,
                    err_link(result.reporter_id, result.message).link)
        for err in result.fails:
            err_per_test[err.reporter_id].append(err.message)

    report.add_block('issues', "Issues:", str(t))

    for reporter_id, errs in err_per_test.items():
        report.add_block(err_link(reporter_id).id, None, "<br>".join(errs))


@tab("Current IO Activity")
def show_io_status(ceph: CephInfo) -> html.HTMLTable:
    t = html.HTMLTable("table-io-summary", ["IO type", "Value"], sortable=False)
    t.add_cells("Client IO Write MiBps", b2ssize(ceph.status.write_bytes_sec // 2 ** 20))
    t.add_cells("Client IO Write OPS", b2ssize(ceph.status.write_op_per_sec))
    t.add_cells("Client IO Read MiBps", b2ssize(ceph.status.read_bytes_sec // 2 ** 20))
    t.add_cells("Client IO Read OPS", b2ssize(ceph.status.read_op_per_sec))
    if "recovering_bytes_per_sec" in ceph.status.pgmap_stat:
        t.add_cells("Recovery IO MiBps", b2ssize(ceph.status.pgmap_stat["recovering_bytes_per_sec"]))
    if "recovering_objects_per_sec" in ceph.status.pgmap_stat:
        t.add_cells("Recovery obj per second", b2ssize_10(ceph.status.pgmap_stat["recovering_objects_per_sec"]))
    return t


@tab("Monitors info")
def show_mons_info(ceph: CephInfo) -> html.HTMLTable:
    table = html.HTMLTable("table-mon-info", ["Name", "Health", "Role", "Disk free<br>B (%)"])

    for _, mon in sorted(ceph.mons.items()):
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
            perc = "Unknown"
            sort_by = "0"
        else:
            perc = f"{b2ssize(mon.kb_avail * 1024)} ({mon.avail_percent})"
            sort_by = str(mon.kb_avail)

        table.add_cell(mon_link(mon.name).link)
        table.add_cell(health)
        table.add_cell(role)
        table.add_cell(perc, sorttable_customkey=sort_by)
        table.next_row()

    return table


@tab("PG's status")
def show_pg_state(ceph: CephInfo) -> html.HTMLTable:
    statuses: Dict[str, int] = collections.Counter()

    for pg_group in ceph.status.pgmap_stat['pgs_by_state']:
        for state_name in pg_group['state_name'].split('+'):
            statuses[state_name] += pg_group["count"]

    table = html.HTMLTable("table-pgs", ["Status", "Count", "%"])
    table.add_row(["any", str(ceph.status.num_pgs), "100.00"])
    for status, count in sorted(statuses.items()):
        table.add_row([status, str(count), f"{100.0 * count / ceph.status.num_pgs:.2f}"])

    return table


@tab("OSD's state")
def show_osd_state(ceph: CephInfo) -> html.HTMLTable:
    statuses: Dict[OSDStatus, List[str]] = collections.defaultdict(list)

    for osd in ceph.osds.values():
        statuses[osd.status].append(str(osd.id))

    table = html.HTMLTable("table-osds-state", ["Status", "Count", "ID's"])

    for status, osds in sorted(statuses.items()):
        table.add_row([html_ok(status.name) if status == OSDStatus.up else html_fail(status.name),
                       len(osds),
                      "<br>".join(", ".join(grp) for grp in partition_by_len(osds, 120, 1))])

    return table


@tab("Settings")
def show_primary_settings(ceph: CephInfo) -> html.HTMLTable:
    table = html.HTMLTable("table-settings", ["Name", "Value"])

    table.add_cell("<b>Common</b>", colspan="2")
    table.next_row()

    table.add_cells("Cluster net", str(ceph.cluster_net))
    table.add_cells("Public net", str(ceph.public_net))
    table.add_cells("Near full ratio", "%0.3f" % (float(ceph.settings.mon_osd_nearfull_ratio,)))

    if 'mon_osd_backfillfull_ratio' in ceph.settings:
        bfratio = f"{float(ceph.settings['mon_osd_backfillfull_ratio']):.3f}"
    elif 'osd_backfill_full_ratio' in ceph.settings:
        bfratio = f"{float(ceph.settings['osd_backfill_full_ratio']):.3f}"
    else:
        bfratio = '?'

    table.add_cells("Backfill full ratio", bfratio)
    table.add_cells("Full ratio", "%0.3f" % (float(ceph.settings.mon_osd_full_ratio,)))
    table.add_cells("Filesafe full ratio", "%0.3f" % (float(ceph.settings.osd_failsafe_full_ratio,)))

    def show_opt(name: str, tr_func: Callable[[str], str] = None):
        name_under = name.replace(" ", "_")
        if name_under in ceph.settings:
            vl = ceph.settings[name_under]
            if tr_func is not None:
                vl = tr_func(vl)
            table.add_cells(name.capitalize(), vl)

    table.add_cell("<b>Fail detection</b>", colspan="2")
    table.next_row()

    show_opt("mon osd down out interval", lambda x: seconds_to_str(int(x)))
    show_opt("mon osd adjust down out interval")
    show_opt("mon osd down out subtree limit")
    show_opt("mon osd report timeout", lambda x: seconds_to_str(int(x)))
    show_opt("mon osd min down reporters")
    show_opt("mon osd reporter subtree level")
    show_opt("osd heartbeat grace", lambda x: seconds_to_str(int(x)))

    table.add_cell("<b>Other</b>", colspan="2")
    table.next_row()

    show_opt("osd max object size", lambda x: b2ssize(int(x)) + "B")
    show_opt("osd mount options xfs")

    table.add_cell("<b>Scrub</b>", colspan="2")
    table.next_row()

    show_opt("osd max scrubs")
    show_opt("osd scrub begin hour")
    show_opt("osd scrub end hour")
    show_opt("osd scrub during recovery")
    show_opt("osd scrub thread timeout", lambda x: seconds_to_str(int(x)))
    show_opt("osd scrub min interval", lambda x: seconds_to_str(int(float(x))))
    show_opt("osd scrub chunk max", lambda x: b2ssize(int(x)))
    show_opt("osd scrub sleep", lambda x: seconds_to_str(int(float(x))))
    show_opt("osd deep scrub interval", lambda x: seconds_to_str(int(float(x))))
    show_opt("osd deep scrub stride", lambda x: b2ssize(int(x)) + "B")

    table.add_cell("<b>OSD io</b>", colspan="2")
    table.next_row()

    show_opt("osd op queue")
    show_opt("osd client op priority")
    show_opt("osd recovery op priority")
    show_opt("osd scrub priority")
    show_opt("osd op thread timeout", lambda x: seconds_to_str(float(x)))
    show_opt("osd op complaint time", lambda x: seconds_to_str(float(x)))
    show_opt("osd disk threads")
    show_opt("osd disk thread ioprio class")
    show_opt("osd disk thread ioprio priority")
    show_opt("osd op history size")
    show_opt("osd op history duration")
    show_opt("osd recovery max chunk", lambda x: b2ssize(int(x)) + "B")
    show_opt("osd max backfills")
    show_opt("osd backfill scan min")
    show_opt("osd backfill scan max")
    show_opt("osd map cache size")
    show_opt("osd map message max")
    show_opt("osd recovery max active")
    show_opt("osd recovery thread timeout")

    if ceph.has_bs:
        table.add_cell("<b>Bluestore</b>", colspan="2")
        table.next_row()

        show_opt("bluestore cache size hdd", lambda x: b2ssize(int(x)) + "B")
        show_opt("bluestore cache size ssd", lambda x: b2ssize(int(x)) + "B")
        show_opt("bluestore cache meta ratio", lambda x: f"{float(x):.3f}")
        show_opt("bluestore cache kv ratio", lambda x: f"{float(x):.3f}")
        show_opt("bluestore cache kv max", lambda x: b2ssize(int(x)) + "B")
        show_opt("bluestore csum type")

    if ceph.has_fs:
        table.add_cell("<b>Filestore</b>", colspan="2")
        table.next_row()
        table.add_cells("Journal aio", ceph.settings.journal_aio)
        table.add_cells("Journal dio", ceph.settings.journal_dio)
        table.add_cells("Filestorage sync", str(int(float(ceph.settings.filestore_max_sync_interval))) + 's')

    return table


@tab("Crush rulesets")
def show_ruleset_info(ceph: CephInfo) -> html.HTMLTable:

    class RulesetsTable(html.Table):
        rule = html.ident()
        id = html.exact_count()
        pools = html.idents_list()
        osd_class = html.ident()
        num_osd = html.exact_count("# OSD")
        size = html.bytes_sz()
        free_size = html.to_str()
        data = html.bytes_sz()
        objs = html.count()
        pg = html.exact_count("PG")
        pg_per_osd = html.exact_count("PG/OSD")
        disk_sizes = html.ident()
        disk_types = html.idents_list(delim=', ')
        disk_models = html.idents_list()

    pools = {}
    for pool in ceph.pools.values():
        pools.setdefault(pool.crush_rule, []).append(pool)

    table = RulesetsTable()

    for rule in ceph.crush.rules.values():
        row = table.next_row()
        row.rule = rule.name
        row.id = rule.id
        row.pools = [pool_link(pool.name).link for pool in pools.get(rule.id, [])]

        if ceph.is_luminous:
            row.osd_class = '*' if rule.class_name is None else rule.class_name

        osds = [ceph.osds[osd_node.id] for osd_node in ceph.crush.iter_osds_for_rule(rule.id)]
        row.num_osd = len(osds)
        total_sz = sum(osd.free_space + osd.used_space for osd in osds)
        row.size = total_sz
        total_free = sum(osd.free_space for osd in osds)
        row.free_size = f"{b2ssize(total_free)} ({total_free * 100 // total_sz}%)", total_free
        row.data = sum(pool.df.size_bytes for pool in pools.get(rule.id, []))
        row.objs = sum(pool.df.num_objects for pool in pools.get(rule.id, []))
        total_pg = sum(pool.pg for pool in pools.get(rule.id, []))
        row.pg = total_pg
        row.pg_per_osd = total_pg // len(osds)

        disks_types = set()
        disks_sizes = set()
        disks_info = set()

        for osd in osds:
            dsk = osd.storage_info.data.dev_info
            disks_types.add(dsk.tp.name)
            disks_sizes.add(dsk.logic_dev.size)
            disks_info.add(f"{dsk.hw_model.vendor}::{dsk.hw_model.model}")

        row.disk_sizes = ", ".join(map(b2ssize, sorted(disks_sizes)))
        row.disk_types = sorted(disks_types)
        row.disk_models = sorted(disks_info)

    return table.html(id="table-rules")


@tab("Average pools load")
def show_pools_average_load(ceph: CephInfo) -> html.HTMLTable:
    class PoolLoadAvg(html.Table):
        pool = html.ident()
        read_b = html.bytes_sz("Read B")
        write_b = html.bytes_sz("Write B")
        read_ops = html.count()
        write_ops = html.count()
        read_per_pg = html.count("Rd/PG")
        write_per_pg = html.count("Wr/PG")

    table = PoolLoadAvg()

    for pool in ceph.pools.values():
        row = table.next_row()
        row.pool = pool.name
        row.read_b = pool.df.read_bytes
        row.write_b = pool.df.write_bytes
        row.read_ops = pool.df.read_ops
        row.write = pool.df.write_ops
        row.read_per_pg = pool.df.read_ops // pool.pg
        row.write_per_pg = pool.df.write_ops // pool.pg

    return table.html(id="table-pools-io", align=html.TableAlign.left_right)


@tab("Pool's stats")
def show_pools_info(ceph: CephInfo) -> html.HTMLTable:
    class PoolsTable(html.Table):
        pool = html.ident()
        id = html.count()
        size = html.ident()
        obj = html.ident()
        bytes = html.ident()
        ruleset = html.ident()
        apps = html.idents_list(delim=", ")
        pg = html.ident()
        osds = html.exact_count("OSD's")
        space = html.bytes_sz("Total<br>OSD Space")
        pg_recommended = html.exact_count("PG<br>recc.")
        byte_per_pg = html.bytes_sz("Bytes/PG")
        obj_per_pg = html.count("Objs/PG")
        pg_count_deviation = html.ident("PG/OSD<br>Deviation")

    table = PoolsTable()

    total_objs = sum(pool.df.num_objects for pool in ceph.pools.values())

    osds_for_rule: Dict[int, int] = {}
    total_size_for_rule: Dict[int, int] = {}

    for _, pool in sorted(ceph.pools.items()):
        row = table.next_row()
        row.pool = pool_link(pool.name).link, pool.name
        row.id = pool.id

        vl = f"{pool.size} / {pool.min_size}"
        if pool.name == 'gnocchi':
            if pool.size != 2 or pool.min_size != 1:
                vl = html_fail(vl)
        else:
            if pool.size != 3 or pool.min_size != 2:
                vl = html_fail(vl)
        row.size = vl, pool.size

        obj_perc = pool.df.num_objects * 100 // total_objs
        row.obj = f"{b2ssize_10(pool.df.num_objects)} ({obj_perc}%)", pool.df.num_objects

        bytes_perc = pool.df.size_bytes * 100 // ceph.status.data_bytes
        row.bytes = f"{b2ssize(pool.df.size_bytes)} ({bytes_perc}%)", pool.df.size_bytes

        rule_name = ceph.crush.rules[pool.crush_rule].name
        row.ruleset = rule_link(rule_name, pool.crush_rule).link, pool.crush_rule

        if ceph.is_luminous:
            row.apps = pool.apps

        pg_perc = (pool.pg * 100) // ceph.status.num_pgs

        if pool.pgp != pool.pg:
            pg_message = f"{pool.pg}, {H.font(str(pool.pgp), color='red')}"
        else:
            pg_message = str(pool.pg)

        row.pg = pg_message + f" ({pg_perc}%)", pool.pg

        if pool.crush_rule not in osds_for_rule:
            rule = ceph.crush.rules[pool.crush_rule]
            all_osd = list(ceph.crush.iter_osds_for_rule(rule.id))
            osds_for_rule[pool.crush_rule] = len(all_osd)
            total_size_for_rule[pool.crush_rule] = sum(ceph.osds[osd.id].total_space for osd in all_osd)

        row.osds = osds_for_rule[pool.crush_rule]
        row.space = total_size_for_rule[pool.crush_rule]
        row.byte_per_pg = pool.df.size_bytes // pool.pg
        row.obj_per_pg = pool.df.num_objects // pool.pg

        if ceph.sum_per_osd is not None:
            osds = list(ceph.crush.iter_osds_for_rule(pool.crush_rule))
            osds_pgs = []

            for osd in osds:
                osds_pgs.append(ceph.osd_pool_pg_2d[osd.id].get(pool.name, 0))

            avg = float(sum(osds_pgs)) / len(osds_pgs)
            if len(osds_pgs) < 2:
                dev = None
            else:
                dev = (sum((i - avg) ** 2.0 for i in osds_pgs) / (len(osds_pgs) - 1)) ** 0.5

            if avg >= 1E-5:
                if dev is None:
                    dev_perc = "--"
                else:
                    dev_perc = str(int(dev * 100. / avg))

                row.pg_count_deviation = f"{avg:.1f} ~ {dev_perc}%", int(avg * 1000)

    return table.html(id="table-pools", align=html.TableAlign.left_right)


@tab("OSD load")
def get_osd_load_table(ceph: CephInfo) -> html.HTMLTable:
    class OSDLoadTable(html.Table):
        id = html.ident()
        node = html.ident()
        pgs = html.exact_count("PG's")
        open_files = html.exact_count("open<br>files")
        ip_conn = html.exact_count("ip<br>conn")
        threads = html.exact_count("thr")
        rss = html.exact_count("RSS")
        vmm = html.exact_count("VMM")
        cpu_used = html.seconds("CPU<br>Used, s")
        data = html.bytes_sz("Total<br>data")
        read_ops = html.count("Read<br>ops")
        read = html.bytes_sz("Read")
        write_ops = html.count("Write<br>ops")
        write = html.bytes_sz("Write")
        bytes_delta = html.bytes_sz("+Bps")
        read_iops = html.count("Read<br>IOPS")
        read_bps = html.bytes_sz("Read<br>Bps")
        write_iops = html.count("Write<br>IOPS")
        write_bps = html.bytes_sz("Write<br>Bps")

    table = OSDLoadTable()

    for osd in ceph.sorted_osds:
        row = table.next_row()
        row.id = osd_link(osd.id).link, osd.id
        row.node = host_link(osd.host.name).link, osd.host.name
        row.pgs = osd.pg_count

        #  RUN INFO - FD COUNT, TCP CONN, THREADS
        if osd.run_info:
            row.open_files = osd.run_info.fd_count
            row.ip_conn = osd.run_info.opened_socks
            row.threads = osd.run_info.th_count
            row.rss = osd.run_info.vm_rss
            row.vms = osd.run_info.vm_size
            row.cpu_used = osd.run_info.cpu_usage

        row.data = osd.pg_stats.bytes
        row.read_ops = osd.pg_stats.reads
        row.read = osd.pg_stats.read_b
        row.write_ops = osd.pg_stats.writes
        row.write = osd.pg_stats.write_b

        if osd.d_pg_stats:
            row.bytes_delta = osd.d_pg_stats.bytes
            row.read_iops = osd.d_pg_stats.reads
            row.read_bps = osd.d_pg_stats.read_b
            row.write_iops = osd.d_pg_stats.writes
            row.write_bps = osd.d_pg_stats.write_b

    return table.html(id="table-osd-load", align=html.TableAlign.left_right)


@dataclass
class OSDInfo:
    id: Union[int, List[int]]
    status: bool
    version: CephVersion
    daemon_runs: bool
    dev_class: str
    reweight: float
    storage_type: str
    storage_total: int
    storage_dev_type: DiskType
    journal_or_wal_collocated: bool
    journal_or_wal_type: DiskType
    journal_or_wal_size: int
    journal_on_file: bool
    db_collocated: Optional[bool]
    db_type: Optional[DiskType]
    db_size: Optional[int]


def group_osds(ceph: CephInfo) -> Iterable[OSDInfo]:
    infos = []
    pgs_counts = {}

    for osd in ceph.sorted_osds:
        data_dev_path = osd.storage_info.data.path
        jinfo = osd.storage_info.journal if isinstance(osd.storage_info, FileStoreInfo) else osd.storage_info.wal

        if isinstance(osd.storage_info, BlueStoreInfo):
            info = osd.storage_info.db.dev_info
            db_collocated = info.dev_path != data_dev_path
            db_type = info.tp.name
            db_size = osd.storage_info.db.partition_info.size
        else:
            db_collocated = None
            db_type = None
            db_size = None

        osd_info = OSDInfo(
            id=osd.id,
            status=osd.status == OSDStatus.up,
            version=osd.version,
            daemon_runs=osd.daemon_runs,
            dev_class=osd.class_name if osd.class_name else "",
            reweight=osd.reweight,
            storage_type="bluestore" if isinstance(osd.storage_info, BlueStoreInfo) else "filestore",
            storage_total=osd.total_space,
            storage_dev_type=osd.storage_info.data.dev_info.tp,
            journal_or_wal_collocated=jinfo.dev_info.dev_path == data_dev_path,
            journal_or_wal_type=jinfo.dev_info.tp,
            journal_or_wal_size=jinfo.partition_info.size,
            journal_on_file=(osd.storage_info.journal.partition_name == osd.storage_info.data.partition_name
                             if isinstance(osd.storage_info, FileStoreInfo) else False),
            db_collocated=db_collocated,
            db_type=db_type,
            db_size=db_size)
        infos.append(osd_info.__dict__)

    return (OSDInfo(**data) for data in group_by(infos, mutable_keys='id'))


@tab("OSD info")
def show_osd_info(ceph: CephInfo) -> html.HTMLTable:
    class OSDInfoTable(html.Table):
        # id = html.ident()
        count = html.exact_count()
        ids = html.idents_list(chars_per_line=50)
        node = html.ident()
        status = html.ok_or_fail()
        version = html.ident("version [hash]")
        daemon_runs = html.yes_or_no("daemon<br>run")
        dev_class = html.ident("Storage<br>class")

        weights = html.extra_columns(html.ident(),
                                     **{f"w_{rule.name}": f"Weight for<br>{rule.name}"
                                     for rule in ceph.crush.rules.values()})

        reweight = html.ident()
        pg = html.ident("PG")
        scrub_err = html.exact_count("Scrub<br>ERR")
        storage_type = html.ident("Type")
        storage_dev = html.ident("Storage<br>dev type")
        storage_total = html.bytes_sz("Storage<br>total")
        journal_or_wal_collocated = html.yes_or_no("Journal<br>or wal<br>colocated", true_fine=False)
        journal_or_wal_type = html.ident("Journal<br>or wal<br>dev type")
        journal_or_wal_size = html.ident("Journal<br>or wal<br>size")
        journal_on_file = html.yes_or_no("Journal<br>on file", true_fine=False)
        db_collocated = html.yes_or_no("DB<br>colocated", true_fine=False)
        db_type = html.ident("DB<br>dev type")
        db_size = html.ident("DB<br>size")

    all_versions = [osd.version for osd in ceph.osds.values()]
    all_versions += [mon.version for mon in ceph.mons.values()]
    all_versions_set = set(all_versions)
    largest_ver = max(all_versions_set)

    fast_drives = {DiskType.nvme, DiskType.sata_ssd, DiskType.sas_ssd}

    table = OSDInfoTable()
    for osd_info in sorted(group_osds(ceph), key=lambda x: x.dev_class):
        osds = [ceph.osds[osd_id] for osd_id in osd_info.id]
        osd = osds[0]

        row = table.next_row()

        row.count = len(osd_info.id)
        row.ids = [(host_link(id).link, str(id)) for id in osd_info.id]
        row.status = osd_info.status

        pgs = [ceph.osds[osd_id].pg_count for osd_id in osd_info.id]
        if len(set(pgs)) == 1:
            row.pg = str(pgs[0]), pgs[0]
        elif len(set(pgs)) < 7:
            row.pg = "<br>".join(f"{val:.2f}: {pgs.count(val)} osd" for val in sorted(set(pgs)))
        else:
            p25, p50, p75, p90, p95 = map(int, numpy.percentile(pgs, [25, 50, 75, 90, 95]))
            row.pg = (f"min = {min(pgs)}<br>25% < {p25}<br>mediana = {p50}<br>75% < {p75}" +
                      f"<br>90% < {p90}<br>95% < {p95}<br>max = {max(pgs)}", p50)

        for rule in ceph.crush.rules.values():
            weights = []
            for gosd in osds:
                wok, weight, expected_weight = gosd.crush_rules_weights.get(rule.name, (False, None, None))
                if wok:
                    weights.append(weight)

            if weights:
                if len(set(weights)) == 1:
                    row.weights[f"w_{rule.name}"] = f"{weights[0]:.2f}"
                elif len(set(weights)) < 7:
                    row.weights[f"w_{rule.name}"] = \
                        "<br>".join(f"{val:.2f}: {weights.count(val)} osd" for val in sorted(set(weights)))
                else:
                    p25, p50, p75, p90, p95 = numpy.percentile(weights, [25, 50, 75, 90, 95])
                    row.weights[f"w_{rule.name}"] = (f"min = {min(weights):.2f}<br>25% < {p25:.2f}" +
                                                     f"<br>mediana = {p50:.2f}<br>75% < {p75:.2f}" +
                                                     f"<br>90% < {p90:.2f}<br>95% < {p95:.2f}" +
                                                     f"<br>max = {max(weights):.2f}", p50)

        if osd.version is None:
            row.version = html_fail("Unknown"), ""
        else:
            if len(all_versions_set) != 1:
                color = html_ok if osd.version == largest_ver else html_fail
            else:
                color = lambda x: x
            row.version = color(str(osd.version)), osd.version

        row.daemon_runs = osd.daemon_runs
        row.dev_class = osd.class_name

        # for rule, weight in zip(ceph.crush.rules.values(), osd_info.weights):
        #     ok, _, expected_weight = osd.crush_rules_weights.get(rule.name, (False, None, None))
        #     if ok:
        #         weight_s = f"{weight:.2f}"
        #         row.weights[f"w_{rule.name}"] = \
        #             (html_fail(weight_s) if abs(1 - weight / expected_weight) > .2 else weight_s), weight

        rew = f"{osd.reweight:.2f}"
        row.reweight = html_fail(rew) if abs(osd.reweight - 1.0) > .01 else rew
        row.storage_type = osd_info.storage_type
        data_drive_color_fn = html_ok if osd.storage_info.data.dev_info.tp in fast_drives else lambda x: x
        row.storage_dev = data_drive_color_fn(osd.storage_info.data.dev_info.tp.name)

        # color = "red" if osd.free_perc < 20 else ( "yellow" if osd.free_perc < 40 else "green")
        # avail_perc_str = H.font(osd.free_perc, color=color)

        row.storage_total = osd.total_space

        data_dev_path = osd.storage_info.data.path

        # JOURNAL/WAL/DB info
        if isinstance(osd.storage_info, FileStoreInfo):
            jinfo = osd.storage_info.journal
            if osd.run_info:
                osd_sync = float(osd.config['filestore_max_sync_interval'])
                min_size = osd_sync * expected_wr_speed[osd.storage_info.data.dev_info.tp] * (1024 ** 2)
            else:
                min_size = 0
        else:
            jinfo = osd.storage_info.wal
            min_size = 512 * 1024 * 1024

        row.journal_or_wal_collocated = jinfo.dev_info.dev_path == data_dev_path
        color = html_ok if jinfo.dev_info.tp in fast_drives else html_fail
        row.journal_or_wal_type = color(jinfo.dev_info.tp.name), jinfo.dev_info.tp.name
        size_s = b2ssize(osd_info.journal_or_wal_size)
        row.journal_or_wal_size = (size_s if osd_info.journal_or_wal_size >= min_size else html_fail(size_s)), \
            osd_info.journal_or_wal_size

        if isinstance(osd.storage_info, FileStoreInfo):
            row.journal_on_file = osd.storage_info.journal.partition_name == osd.storage_info.data.partition_name

        if isinstance(osd.storage_info, BlueStoreInfo):
            if osd.storage_info.db.dev_info.size is not None:
                min_db_size = osd.storage_info.data.dev_info.size * 0.05
            else:
                min_db_size = 0

            info = osd.storage_info.db.dev_info
            row.db_collocated = info.dev_path == data_dev_path
            color = html_ok if info.tp in fast_drives else html_fail
            row.db_type = color(info.tp.name), info.tp.name
            size_s = b2ssize(osd_info.db_size)
            row.db_size = size_s if osd_info.db_size >= min_db_size else html_fail(size_s), osd_info.db_size

    return table.html(id="table-osd-info", align=html.TableAlign.left_right)

    # for osd in ceph.sorted_osds:
    #     row = table.next_row()
    #     row.id = osd_link(osd.id).link, osd.id
    #     row.node = host_link(osd.host.name).link, osd.host.name
    #     row.status = osd.status == OSDStatus.up
    #
    #     if osd.version is None:
    #         row.version = html_fail("Unknown"), ""
    #     else:
    #         if len(all_versions_set) != 1:
    #             color = html_ok if osd.version == largest_ver else html_fail
    #         else:
    #             color = lambda x: x
    #         row.version = color(str(osd.version)), osd.version
    #
    #     row.daemon_runs = osd.daemon_runs
    #
    #     for root in ceph.crush.roots:
    #         if root.name in osd.crush_trees_weights:
    #             wok, weight, expected_weight = osd.crush_trees_weights[root.name]
    #
    #             if wok:
    #                 weight_s = f"{weight:.2f}"
    #                 row.weights[f"w_{root.name}"] = \
    #                     (html_fail(weight_s) if abs(1 - weight / expected_weight) > .2 else weight_s), weight
    #
    #     rew = f"{osd.reweight:.2f}"
    #     row.reweight = html_fail(rew) if abs(osd.reweight - 1.0) > .01 else rew
    #     row.pg = osd.pg_count
    #
    #     if osd.pg_stats:
    #         row.scrub_err = osd.pg_stats.scrub_errors + osd.pg_stats.deep_scrub_errors + \
    #             osd.pg_stats.shallow_scrub_errors
    #
    #     row.storage_type = "bluestore" if isinstance(osd.storage_info, BlueStoreInfo) else "filestore"
    #     data_drive_color_fn = html_ok if osd.storage_info.data.dev_info.tp in fast_drives else lambda x: x
    #     row.storage_dev = data_drive_color_fn(osd.storage_info.data.dev_info.tp.name)
    #     color = "red" if osd.free_perc < 20 else ( "yellow" if osd.free_perc < 40 else "green")
    #     avail_perc_str = H.font(osd.free_perc, color=color)
    #     row.storage_total = osd.total_space
    #     row.storage_free = avail_perc_str, osd.free_perc
    #
    #     data_dev_path = osd.storage_info.data.path
    #
    #     # JOURNAL/WAL/DB info
    #     if isinstance(osd.storage_info, FileStoreInfo):
    #         jinfo = osd.storage_info.journal
    #         if osd.run_info:
    #             osd_sync = float(osd.config['filestore_max_sync_interval'])
    #             min_size = osd_sync * expected_wr_speed[osd.storage_info.data.dev_info.tp] * (1024 ** 2)
    #         else:
    #             min_size = 0
    #     else:
    #         jinfo = osd.storage_info.wal
    #         min_size = 512 * 1024 * 1024
    #
    #     row.journal_or_wal_collocated = jinfo.dev_info.dev_path != data_dev_path
    #     color = html_ok if jinfo.dev_info.tp in fast_drives else html_fail
    #     row.journal_or_wal_type = color(jinfo.dev_info.tp.name), jinfo.dev_info.tp.name
    #     size_s = b2ssize(jinfo.dev_info.size)
    #     row.journal_or_wal_size = (size_s if jinfo.dev_info.size >= min_size else html_fail(size_s)), \
    #         jinfo.dev_info.size
    #
    #     if isinstance(osd.storage_info, FileStoreInfo):
    #         row.journal_on_file = osd.storage_info.journal.partition_name == osd.storage_info.data.partition_name
    #
    #     if isinstance(osd.storage_info, BlueStoreInfo):
    #         if osd.storage_info.db.dev_info.size is not None:
    #             min_db_size = osd.storage_info.data.dev_info.size * 0.0095
    #         else:
    #             min_db_size = 0
    #
    #         info = osd.storage_info.db.dev_info
    #         row.db_collocated = info.dev_path != data_dev_path
    #         color = html_ok if info.tp in fast_drives else html_fail
    #         row.db_type = color(info.tp.name), info.tp.name
    #         size_s = b2ssize(info.size)
    #         row.db_size = size_s if info.size >= min_db_size else html_fail(size_s), info.size

    # load_table = get_osd_load_table(ceph)
    # return "OSD's info:", f"<center>{table}<br><br><br><H4>Run info:</H4><br>{load_table}</center>"


@perf_info_required
def show_osd_perf_info(ceph: CephInfo) -> Tuple[str, Any]:
    headers = ["OSD", "node", "apply<br>lat, ms<br>avg/osd perf", "commit<br>lat"]

    if ceph.has_fs:
        headers.append("journal<br>lat, ms")

    headers.extend(["ms<br>avg/osd perf", "D dev", "D read<br>Bps/OPS", "D write<br>Bps/OPS",
                    "D lat<br>ms", "D IO<br>time %"])

    headers.extend(["J/WAL dev", "J/WAL write<br>Bps/OPS", "J/WAL lat<br>ms", "J/WAL IO<br>time %"])

    if ceph.has_bs:
        headers.extend(["DB dev", "DB write<br>Bps/OPS", "DB lat<br>ms", "DB IO<br>time %"])

    table = html.HTMLTable(headers=headers)

    def lat2s(val):
        if val is None or val < 1e-6:
            return HTML_UNKNOWN
        elif val < 1e-3:
            return f"{val * 1000:.3f}"
        else:
            return f"{int(val * 1000)}"

    def add_dev_info(dev: Union[Disk, LogicBlockDev]):
        table.add_cell(dev.name)
        table.add_cell(f"{b2ssize(dev.usage.read_bytes)} / {b2ssize(dev.usage.read_iops)}",
                       sorttable_customkey=str(dev.usage.read_bytes))
        table.add_cell(f"{b2ssize(dev.usage.write_bytes)} / {b2ssize(dev.usage.write_iops)}",
                       sorttable_customkey=str(dev.usage.write_bytes))

        if dev.usage.lat is None:
            table.add_cell('-', sorttable_customkey="0")
        else:
            table.add_cell(lat2s(dev.usage.lat), sorttable_customkey=str(dev.usage.lat))

        table.add_cell(str(int(dev.usage.io_time * 100)), sorttable_customkey=str(dev.usage.io_time))

    for osd in ceph.sorted_osds:
        assert osd.storage_info is not None

        table.add_cell(osd_link(osd.id).link)
        table.add_cell(host_link(osd.host.name).link)

        lats = [("apply_latency", "apply_latency_s"), ("commitcycle_latency", "commitcycle_latency_s")]
        if ceph.has_fs:
            lats.append(("journal_latency", None))

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

        add_dev_info(osd.storage_info.data.dev_info)
        if isinstance(osd.storage_info, FileStoreInfo):
            add_dev_info(osd.storage_info.journal.dev_info)
        else:
            add_dev_info(osd.storage_info.wal.dev_info)
            add_dev_info(osd.storage_info.db.dev_info)

        table.next_row()

    return "OSD's current load:", table


@perf_info_required
def show_host_network_load_in_color(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
    send_net_io: Dict[str, List[Tuple]] = collections.defaultdict(list)
    recv_net_io: Dict[str, List[Tuple]] = collections.defaultdict(list)

    max_net_count = 0
    for host in cluster.hosts.values():
        cl_if = host.find_interface(ceph.cluster_net)
        pb_if = host.find_interface(ceph.public_net)
        nets = [('cluster', cl_if), ('public', pb_if)]
        nets += sorted((net.dev, net) for net in host.net_adapters.values()  # type: ignore
                       if net is not None and net.is_phy and net != pb_if and net != cl_if )

        uptime = host.uptime
        max_net_count = max(max_net_count, len(nets))
        for name, net in nets:
            if net is None or net.usage is None:
                if name in {'cluster', 'public'}:
                    send_net_io[host.name].append((name, None, None, None))
                    recv_net_io[host.name].append((name, None, None, None))
            else:
                send_net_io[host.name].append((name, net.usage.send_bytes / uptime,
                                              net.usage.send_packets / uptime,
                                              net.speed))
                recv_net_io[host.name].append((name, net.usage.recv_bytes / uptime,
                                              net.usage.recv_packets / uptime,
                                              net.speed))

    std_nets = {'public', 'cluster'}
    ceph_net_count = len(std_nets)
    tables = []

    for io, name in [(send_net_io, "Send (KiBps/Pps)"), (recv_net_io, "Receive (KiBps/Pps)")]:

        table = html.HTMLTable(headers=["host", "cluster", "public"] +
                                       ["hw adapter"] * (max_net_count - ceph_net_count),
                               zebra=False)

        for host_name, net_loads in sorted(io.items()):
            table.add_cell(host_link(host_name).link)
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

        tables.append(table)

    return "Net load", "<br>".join(map(str, tables))


def make_storage_devs_load_table(hosts: Iterable[Host],
                                 attr: str,
                                 conversion: Callable[[Union[float, int]], str],
                                 uptime: bool,
                                 max_val: Optional[float],
                                 min_max_val: float) -> html.HTMLTable:

        vals = []
        uptimes = []
        for host in hosts:
            vals.append(sorted(host.disks.values(), key=lambda x: x.name))
            if uptime:
                uptimes.append(host.uptime)

        max_len = max(map(len, vals))

        if max_val is None:
            max_val = min_max_val
            for host_devs in vals:
                max_val = max(max_val, max(getattr(dev, attr) for dev in host_devs))

        table = html.HTMLTable(headers=['host'] + ['load'] * max_len, zebra=False, extra_cls=["io_load_in_color"])
        for idx, host_devs in enumerate(vals):
            table.add_cell(host_link(host_devs[0].logic_dev.hostname).link)
            for dev_info in host_devs:

                if uptime:
                    val = getattr(dev_info.logic_dev.usage, attr) / uptimes[idx]
                else:
                    val = getattr(dev_info.logic_dev.d_usage, attr)

                color = "#FFFFFF" if val is None or max_val == 0 else val_to_color(float(val) / max_val)

                s_val = 0 if val < 1 else conversion(val)

                cell_data = H.div(dev_info.name + " ", _class="left") + H.div(s_val, _class="right")
                table.add_cell(cell_data, bgcolor=color, sorttable_customkey=str(val))

            for i in range(max_len - len(host_devs)):
                table.add_cell("-", sorttable_customkey='0')
            table.next_row()
        return table


@perf_info_required
def show_host_io_load_in_color(cluster: Cluster, uptime: bool) -> Tuple[str, Any]:
    hosts = [host for _, host in sorted(cluster.hosts.items())]

    loads = [
        ('iops', b2ssize_10, 'IOPS', 50),
        ('read_iops', b2ssize_10, 'Read IOPS', 30),
        ('write_iops', b2ssize_10, 'Write IOPS', 30),

        ('total_bytes', lambda x: str(int(x / 2 ** 20)), 'MiBps', 100 * 1024 ** 2),
        ('read_bytes', lambda x: str(int(x / 2 ** 20)), 'Read MiBps', 100 * 1024 ** 2),
        ('write_bytes', lambda x: str(int(x / 2 ** 20)), 'Write MiBps', 100 * 1024 ** 2),

        ('lat', lambda x: '-' if x is None else str(int(x)), 'Latency, ms', 20),
        ('queue_depth', lambda x: f"{x:.1f}", 'Average QD', 3),
        ('io_time', lambda x: f"{x:.1f}", 'Active time %', 100),
    ]

    blocks = []
    for pos, (attr, conversion, tp, min_max_val) in enumerate(loads, 1):
        max_val = 300.0 if attr == 'lat' else None
        table = make_storage_devs_load_table(hosts, attr, conversion, uptime, max_val, min_max_val)
        blocks.append(f"<br><H4>{tp}</H4><br><br>{table}")

    return "IO load" + (" uptime" if uptime else ""), "<br>".join(blocks)


def hosts_pg_info(cluster: Cluster, hosts_pgs_info: Dict[str, NodePGStats]) -> str:
    header_row = ["Name",
                  "PGs",
                  "User data",
                  "Reads",
                  "Read B",
                  "Writes",
                  "Write B"]

    if cluster.has_second_report:
        header_row.extend(["User data<br>store rate", "Reads<br>ps", "Read Bps", "Writes<br>ps", "Write Bps"])

    table = html.HTMLTable("table-hosts-pg-info-long" if cluster.has_second_report else "table-hosts-pg-info",
                           header_row, align=html.TableAlign.center_right)

    def add_pg_stats(stats: OSDPGStats):
        table.add_cell_b2ssize(stats.bytes)
        table.add_cell_b2ssize_10(stats.reads)
        table.add_cell_b2ssize(stats.read_b)
        table.add_cell_b2ssize_10(stats.writes)
        table.add_cell_b2ssize(stats.write_b)

    for host in sorted(cluster.hosts.values(), key=lambda x: x.name):
        if host.name in hosts_pgs_info:
            table.add_cell(host_link(host.name).link)
            pgs_info = hosts_pgs_info[host.name]
            table.add_cell(str(len(pgs_info.pgs)))

            add_pg_stats(pgs_info.pg_stats)
            if cluster.has_second_report:
                assert pgs_info.d_pg_stats
                add_pg_stats(pgs_info.d_pg_stats)
            table.next_row()

    return f"<br><br><center><H4>Hosts PG's info:</H4><br><br>{table}</center>"


def group_by(items: List[Dict[str, Any]], mutable_keys=Union[str, Tuple[str, ...]]) -> Iterable[Dict[str, Any]]:
    if isinstance(mutable_keys, str):
        mutable_keys = (mutable_keys,)

    group_by_items = None
    grouped = {}
    for dct in items:
        dct = dct.copy()
        mutable = tuple(dct.pop(mutable_key) for mutable_key in mutable_keys)

        if group_by_items is None:
            group_by_items = list(dct.keys())

        group_idx = tuple(dct.pop(key_name) for key_name in group_by_items)
        assert len(dct) == 0
        grouped.setdefault(group_idx, []).append(mutable)

    assert group_by_items is not None

    for keys, vals in grouped.items():
        dct_itm = dict(zip(group_by_items, keys))
        dct_itm.update(dict(zip(mutable_keys, zip(*vals))))
        yield dct_itm


@tab("Hosts config")
def show_hosts_info(cluster: Cluster, ceph: CephInfo) -> str:
    hosts_pgs_info = get_nodes_pg_info(ceph)
    mon_hosts = {mon.host.name for mon in ceph.mons.values()}

    host2osds = {}
    for osd in ceph.osds.values():
        host2osds.setdefault(osd.host.name, []).append(str(osd.id))

    rule2host2osds: Dict[Dict[str, List[CephOSD]]] = {}
    for rule in ceph.crush.rules.values():
        curr = rule2host2osds[rule.name] = {}
        for osd_node in ceph.crush.iter_osds_for_rule(rule.id):
            osd = ceph.osds[osd_node.id]
            curr.setdefault(osd.host.name, []).append(osd)

    root_names = sorted(rule2host2osds)

    hosts_configs = []
    for host in cluster.sorted_hosts:
        by_speed = collections.Counter()
        for adapter in host.net_adapters.values():
            if adapter.is_phy:
                by_speed[adapter.speed if adapter.speed else 0] += 1

        nets = "<br>".join(f"{b2ssize_10(speed * 8) if speed else '???'} * {count}"
                           for speed, count in sorted(by_speed.items()))

        hosts_configs.append({
            "name": host.name,
            "osds_count": tuple(len(rule2host2osds[root_name].get(host.name, [])) for root_name in root_names),
            'has_mon': host.name in mon_hosts,
            'cores': sum(inf.cores for inf in host.hw_info.cpu_info),
            'ram': int(host.mem_total / 2 ** 30 + 0.5),
            'net': nets
        })

    class HostsConfigTable(html.Table):
        count = html.exact_count()
        names = html.idents_list(chars_per_line=60)
        osds_count = html.extra_columns(html.exact_count(),
                                        **{root_name: f"osd count for<br>{root_name}" for root_name in root_names})
        has_mon = html.ident()
        cores = html.exact_count("CPU<br>Cores+HT")
        ram = html.bytes_sz("RAM<br>total")
        storage_devices = html.ident("Storage<br>devices")
        network = html.ident("Network<br>devices")

    class HostRunInfo(html.Table):
        name = html.ident()
        services = html.idents_list()
        ram_free = html.bytes_sz("RAM<br>free")
        swap = html.bytes_sz("Swap<br>used")
        load = html.float_vl("Load avg<br>5m")
        ip_conn = html.ident("Conn<br>tcp/udp")
        ips = html.idents_list("IP's")
        scrub_err = html.exact_count("Scrub<br>errors")

    configs = HostsConfigTable()

    for grouped_item in group_by(hosts_configs, mutable_keys="name"):
        row = configs.next_row()
        row.count = len(grouped_item['name'])
        row.names = [(host_link(name).link, name) for name in grouped_item['name']]
        for name, vl in zip(root_names, grouped_item["osds_count"]):
            row.osds_count[name] = vl
        row.has_mon = 'yes' if grouped_item["has_mon"] else 'no'
        row.cores = grouped_item["cores"]
        row.ram = grouped_item["ram"]
        row.network = grouped_item["net"]

    run_info = HostRunInfo()
    for host in cluster.sorted_hosts:

        # TODO: add storage devices
        # TODO: add net info

        row2 = run_info.next_row()
        row2.name = host_link(host.name).link, host.name

        srv_strs = []
        if host.name in mon_hosts:
            srv_strs.append(f'<font color="#8080FF">Mon</font>: ' + mon_link(host.name).link)

        if host.name in host2osds:
            osds = host2osds[host.name]
            srv_strs.append('''<font color="#c77405">OSD's</font>: ''' +
                            ", ".join(osd_link(osd_id).link for osd_id in osds[:3]))
            for ids in partition(osds[3:], 4):
                srv_strs.append(", ".join(osd_link(osd_id).link for osd_id in ids))

        row2.services = srv_strs

        row2.ram_free = host.mem_free
        row2.swap = host.swap_total - host.swap_free
        row2.load = host.load_5m
        row2.ip_conn = f"{host.open_tcp_sock}/{host.open_udp_sock}", host.open_tcp_sock + host.open_udp_sock

        all_ip = flatten(adapter.ips for adapter in host.net_adapters.values())
        row2.ips = [str(addr.ip) for addr in all_ip if not addr.ip.is_loopback]

        if hosts_pgs_info and host.name in hosts_pgs_info:
            pgs_info = hosts_pgs_info[host.name]
            row2.scrub_err  = pgs_info.pg_stats.scrub_errors + pgs_info.pg_stats.deep_scrub_errors + \
                pgs_info.pg_stats.shallow_scrub_errors

    return str(configs.html(id="table-hosts-info", align=html.TableAlign.center_right)) + "<br>" * 5 + \
           str(run_info.html(id="table-hosts-run-info", align=html.TableAlign.center_right))

    # TODO: +hosts_pg_info(cluster, hosts_pgs_info)


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
        data = [osd_link(osd_id).link] + [row.get(pool_name, 0) for pool_name in pools] + [ceph.sum_per_osd[osd_id]]
        table.add_row(map(str, data))

    table.add_cell("Total cluster PG", sorttable_customkey=str(max(ceph.osds) + 1))

    list(map(table.add_cell, (ceph.sum_per_pool[pool_name] for pool_name in pools)))
    table.add_cell(str(sum(ceph.sum_per_pool.values())))

    return "PG copy per OSD:", table


def host_info(host: Host, ceph: CephInfo) -> str:
    cluster_networks = [(ceph.public_net, 'ceph-public'), (ceph.cluster_net, 'ceph-cluster')]

    doc = html.Doc()
    doc.center.H3(host.name)
    doc.br
    doc.center.H4("Interfaces:")
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
    doc.center.H4("HW disks:")
    doc.br

    mib_and_mb = lambda x: f"{b2ssize(x)}B / {b2ssize_10(x)}B"

    stor_roles = collections.defaultdict(lambda: collections.defaultdict(set))
    stor_classes = collections.defaultdict(set)

    for osd in ceph.osds.values():
        if osd.host is host:
            stor_roles[osd.storage_info.data.partition_name]['data'].add(osd.id)
            stor_roles[osd.storage_info.data.name]['data'].add(osd.id)

            if osd.class_name:
                stor_classes[osd.storage_info.data.partition_name].add(osd.class_name)
                stor_classes[osd.storage_info.data.name].add(osd.class_name)

            if isinstance(osd.storage_info, FileStoreInfo):
                stor_roles[osd.storage_info.journal.partition_name]['journal'].add(osd.id)
                stor_roles[osd.storage_info.journal.name]['journal'].add(osd.id)

                if osd.class_name:
                    stor_classes[osd.storage_info.journal.name].add(osd.class_name)
                    stor_classes[osd.storage_info.journal.partition_name].add(osd.class_name)
            else:
                assert isinstance(osd.storage_info, BlueStoreInfo)
                stor_roles[osd.storage_info.db.partition_name]['db'].add(osd.id)
                stor_roles[osd.storage_info.db.name]['db'].add(osd.id)
                stor_roles[osd.storage_info.wal.partition_name]['wal'].add(osd.id)
                stor_roles[osd.storage_info.wal.name]['wal'].add(osd.id)

                if osd.class_name:
                    stor_classes[osd.storage_info.db.partition_name].add(osd.class_name)
                    stor_classes[osd.storage_info.db.name].add(osd.class_name)
                    stor_classes[osd.storage_info.wal.partition_name].add(osd.class_name)
                    stor_classes[osd.storage_info.wal.name].add(osd.class_name)

    order = ['data', 'journal', 'db', 'wal']

    def html_roles(name: str) -> str:
        return "<br>".join(name + ": " + ", ".join(map(str, sorted(ids)))
                           for name, ids in sorted(stor_roles[name].items(), key=lambda x: order.index(x[0])))

    table = html.HTMLTable(headers=["Name", "Type", "Size"] +
                                   (["Roles"] if ceph.is_luminous else []) +
                                   ["Classes", "Scheduler", "Model info", "RQ-size", "Phy Sec", "Min IO"],
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
                       html_roles(disk.name)] +
                       (["<br>".join(stor_classes.get(disk.name, []))] if ceph.is_luminous else []) +
                       ["" if disk.scheduler is None else str(disk.scheduler),
                       info,
                       str(disk.rq_size),
                       str(disk.phy_sec),
                       str(disk.min_io)])

    doc.center(str(table))
    doc.br
    doc.center.H4("Mountable:")
    doc.br
    table = html.HTMLTable(headers=["Name", "Size", "Type", "Mountpoint", "Ceph role", "Fs", "Free space", "Label"],
                           extra_cls=["hostinfo-mountable"], sortable=False)

    def run_over_children(obj: Union[Disk, LogicBlockDev], level: int):
        table.add_row([f'<div class="disk-children-{level}">{obj.name}</div>',
                       mib_and_mb(obj.size),
                       obj.tp.name if level == 0 else '',
                       obj.mountpoint if obj.mountpoint else '',
                       "" if level == 0 else html_roles(obj.name),
                       obj.fs if obj.fs else '',
                       mib_and_mb(obj.free_space) if obj.free_space is not None else '',
                       "" if obj.label is None else obj.label])

        for _, ch in sorted(obj.children.items(), key=lambda x: x[1].partition_num):
            run_over_children(ch, level + 1)

    for _, disk in sorted(host.disks.items()):
        run_over_children(disk, 0)

    doc.center(str(table))

    return str(doc)


@plot
def show_osd_pg_histo(ceph: CephInfo) -> Optional[Tuple[str, Any]]:
    vals = [osd.pg_count for osd in ceph.osds.values() if osd.pg_count is not None]
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
    p.add_argument("--profile", help="Profile report creation", action="store_true")
    p.add_argument("-e", "--embed", action='store_true', help="Embed js/css files into report to make it stand-alone")
    p.add_argument("path", help="Folder with data, or .tar.gz archive")
    p.add_argument("old_path", nargs='?', help="Older folder with data, or .tar.gz archive to calculate load")
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

    logger.info("Generating report from %r to %r", opts.path, opts.out)

    if opts.profile:
        prof = cProfile.Profile()
        prof.enable()

    if Path(opts.path).is_file():
        arch_name = opts.path
        folder = tempfile.mkdtemp()
        logger.info("Unpacking %r to temporary folder %r", opts.path, folder)
        remove_folder = True
        subprocess.call(f"tar -zxvf {arch_name} -C {folder} >/dev/null 2>&1", shell=True)
    elif not Path(opts.path).is_dir():
        logger.error("Path argument (%r) should be a folder with data or path to archive", opts.path)
        return 1
    else:
        folder = opts.path

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
            show_primary_settings,
            show_pools_info,
            show_ruleset_info,
            show_io_status,
            show_hosts_info,
            show_mons_info,
            show_osd_state,
            show_osd_info,
            show_osd_perf_info,
            show_pg_state,
            show_osd_pool_pg_distribution,
            show_host_io_load_in_color,
            show_host_network_load_in_color,
            # show_osd_used_space_histo,
            # show_osd_pg_histo,
            # show_osd_load,
            show_osd_lat_heatmaps,
            # show_osd_ops_boxplot,
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

            if hasattr(reporter, 'report_name'):
                assert 'report' not in sig.parameters
                report.add_block(reporter.__name__.replace("show_", ""), reporter.report_name, new_block)
            elif new_block is not None:
                assert 'report' not in sig.parameters
                report.add_block(reporter.__name__.replace("show_", ""), *new_block)

        for _, host in sorted(cluster.hosts.items()):
            report.add_block(host_link(host.name).id, None, host_info(host, ceph))

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
