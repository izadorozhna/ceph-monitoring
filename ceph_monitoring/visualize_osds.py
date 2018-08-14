import collections
from typing import Dict, List, Optional, Sequence, Union, Tuple, Any

import numpy
from dataclasses import dataclass

from cephlib.units import b2ssize

from . import html
from .cluster_classes import CephInfo, OSDStatus, DiskType, CephVersion, BlueStoreInfo, FileStoreInfo, LogicBlockDev, \
                             Disk
from .visualize_utils import tab, partition_by_len, to_html_histo, plot, perf_info_required
from .obj_links import osd_link, host_link
from .checks import expected_wr_speed
from .plot_data import get_histo_img
from .table import Table, count, bytes_sz, ident, seconds, exact_count, ok_or_fail, idents_list, \
                   yes_or_no, extra_columns
from .groupby import group_by

@tab("OSD's state")
def show_osd_state(ceph: CephInfo) -> html.HTMLTable:
    statuses: Dict[OSDStatus, List[str]] = collections.defaultdict(list)

    for osd in ceph.osds.values():
        statuses[osd.status].append(str(osd.id))

    table = html.HTMLTable("table-osds-state", ["Status", "Count", "ID's"])

    for status, osds in sorted(statuses.items()):
        table.add_row([html.ok(status.name) if status == OSDStatus.up else html.fail(status.name),
                       len(osds),
                      "<br>".join(", ".join(grp) for grp in partition_by_len(osds, 120, 1))])

    return table


@tab("OSD load")
def show_osd_load_table(ceph: CephInfo) -> html.HTMLTable:
    class OSDLoadTable(Table):
        id = ident()
        node = ident()
        pgs = exact_count("PG's")
        open_files = exact_count("open<br>files")
        ip_conn = exact_count("ip<br>conn")
        threads = exact_count("thr")
        rss = exact_count("RSS")
        vmm = exact_count("VMM")
        cpu_used = seconds("CPU<br>Used, s")
        data = bytes_sz("Total<br>data")
        read_ops = count("Read<br>ops")
        read = bytes_sz("Read")
        write_ops = count("Write<br>ops")
        write = bytes_sz("Write")
        bytes_delta = bytes_sz("+Bps")
        read_iops = count("Read<br>IOPS")
        read_bps = bytes_sz("Read<br>Bps")
        write_iops = count("Write<br>IOPS")
        write_bps = bytes_sz("Write<br>Bps")

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
    id: int
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


def group_osds(ceph: CephInfo) -> List[List[OSDInfo]]:
    objs = []

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
        objs.append(osd_info)

    return [[objs[idx] for idx in group] for group in group_by((obj.__dict__ for obj in objs), mutable_keys='id')]


@tab("OSD info")
def show_osd_info(ceph: CephInfo) -> html.HTMLTable:
    class OSDInfoTable(Table):
        # id = html.ident()
        count = exact_count()
        ids = idents_list(chars_per_line=50)
        node = ident()
        status = ok_or_fail()
        version = ident("version [hash]")
        daemon_runs = yes_or_no("daemon<br>run")
        dev_class = ident("Storage<br>class")

        weights = extra_columns(ident(), **{f"w_{rule.name}": f"Weight for<br>{rule.name}"
                                            for rule in ceph.crush.rules.values()})

        reweight = ident()
        pg = ident("PG")
        scrub_err = exact_count("Scrub<br>ERR")
        storage_type = ident("Type")
        storage_dev = ident("Storage<br>dev type")
        storage_total = bytes_sz("Storage<br>total")
        journal_or_wal_collocated = yes_or_no("Journal<br>or wal<br>colocated", true_fine=False)
        journal_or_wal_type = ident("Journal<br>or wal<br>dev type")
        journal_or_wal_size = ident("Journal<br>or wal<br>size")
        journal_on_file = yes_or_no("Journal<br>on file", true_fine=False)
        db_collocated = yes_or_no("DB<br>colocated", true_fine=False)
        db_type = ident("DB<br>dev type")
        db_size = ident("DB<br>size")

    all_versions = [osd.version for osd in ceph.osds.values()]
    all_versions += [mon.version for mon in ceph.mons.values()]
    all_versions_set = set(all_versions)
    largest_ver = max(all_versions_set)

    fast_drives = {DiskType.nvme, DiskType.sata_ssd, DiskType.sas_ssd}

    table = OSDInfoTable()

    for osd_infos in sorted(group_osds(ceph), key=lambda x: x[0].dev_class):
        osds = [ceph.osds[osd_info.id] for osd_info in osd_infos]
        osd = osds[0]
        osd_info = osd_infos[0]

        row = table.next_row()

        row.count = len(osd_infos)
        row.ids = [(host_link(str(osd_info.id)).link, str(osd_info.id)) for osd_info in osd_infos]
        row.status = osd_info.status

        pgs = [ceph.osds[osd_info.id].pg_count for osd_info in osd_infos]
        row.pg = to_html_histo(pgs)

        for rule in ceph.crush.rules.values():
            weights = []
            for gosd in osds:
                wok, weight, expected_weight = gosd.crush_rules_weights.get(rule.name, (False, None, None))
                if wok:
                    weights.append(weight)

            if weights:
                row.weights[f"w_{rule.name}"] = to_html_histo(weights, show_int=False)

        if osd.version is None:
            row.version = html.fail("Unknown"), ""
        else:
            if len(all_versions_set) != 1:
                color = html.ok if osd.version == largest_ver else html.fail
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
        row.reweight = html.fail(rew) if abs(osd.reweight - 1.0) > .01 else rew
        row.storage_type = osd_info.storage_type
        data_drive_color_fn = html.ok if osd.storage_info.data.dev_info.tp in fast_drives else lambda x: x
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
        color = html.ok if jinfo.dev_info.tp in fast_drives else html.fail
        row.journal_or_wal_type = color(jinfo.dev_info.tp.name), jinfo.dev_info.tp.name
        size_s = b2ssize(osd_info.journal_or_wal_size)
        row.journal_or_wal_size = (size_s if osd_info.journal_or_wal_size >= min_size else html.fail(size_s)), \
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
            color = html.ok if info.tp in fast_drives else html.fail
            row.db_type = color(info.tp.name), info.tp.name
            size_s = b2ssize(osd_info.db_size)
            row.db_size = size_s if osd_info.db_size >= min_db_size else html.fail(size_s), osd_info.db_size

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


@plot
def show_osd_pg_histo(ceph: CephInfo) -> Optional[Tuple[str, Any]]:
    vals = [osd.pg_count for osd in ceph.osds.values() if osd.pg_count is not None]
    if vals:
        return "PG per OSD", get_histo_img(numpy.array(vals))
    return None