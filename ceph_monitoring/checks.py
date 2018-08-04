import collections
from enum import Enum
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass

from .cluster_classes import Cluster, CephInfo, FileStoreInfo, BlueStoreInfo
from cephlib.units import b2ssize

# Check
# mount options
# network speed
# mtu
# ceph settings
# rgw settings
# osd/mon/cluster log errors
# rgw/monitor collocation
# pools pg/load/data
# total cluster pg
# services cpu/ram consumption
# BS cache settings
# kernel settings
# thread count/net connection count
# deep scrub time
# pg io bottleneck
# radosgw performance metrics!
# cpu/disk/net per user request


class Severity(Enum):
    ok = 0
    notice = 1
    warning = 2
    error = 3
    critical = 4


@dataclass
class CheckResult:
    reporter: str
    severity: Severity
    message: str
    service: str


class CheckReport:
    def __init__(self) -> None:
        self.results: List[Tuple[str, Severity, bool, Optional[str]]] = []
        self.curr_msg: str = ""
        self.curr_reporter: str = ""
        self.curr_sev: Severity = Severity.ok
        self.extra: Dict[str, List[CheckResult]] = {}

    def set_curr(self, reporter: str, msg: str, sev: Severity):
        self.curr_msg = msg
        self.curr_reporter = reporter
        self.curr_sev = sev

    def add_result(self, passed: bool, comment: str = None, sev: Severity = None):
        self.results.append((self.curr_msg, sev if sev is not None else self.curr_sev, passed, comment))

    def add_extra_message(self, sev: Severity, message: str, *affected_services: str):
        for affected_service in affected_services:
            self.extra.setdefault(self.curr_reporter, []).append(
                CheckResult(self.curr_reporter, sev, message, affected_service))

        if not affected_services:
            self.extra.setdefault(self.curr_reporter, []).append(
                CheckResult(self.curr_reporter, sev, message, 'cluster'))


CheckConfig = Dict[str, Any]
CheckerFunc = Callable[[CheckConfig, Cluster, CephInfo, CheckReport], None]

ALL_CHECKS = []

def checker(severity: Severity, message: str) -> Callable[[CheckerFunc], CheckerFunc]:
    def closure(func: CheckerFunc) -> CheckerFunc:
        func.severity = severity
        func.message = message
        ALL_CHECKS.append(func)
        return func
    return closure


def osd_name(osd_id: int) -> str:
    return f'osd.{osd_id}'


@checker(Severity.warning, "Scrub errors count")
def no_scrub_errors(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    if ceph.osds_info:
        total_scrub_err = 0
        err_per_osd: Dict[int, int] = {}
        for osd_id, osd_info in ceph.osds_info.osds.items():
            err = osd_info.pg_stats.scrub_errors + osd_info.pg_stats.deep_scrub_errors + \
                  osd_info.pg_stats.shallow_scrub_errors
            err_per_osd[osd_id] = err

        for osd_id, err_count in sorted(err_per_osd.items()):
            if err_count > 0:
                report.add_extra_message(Severity.warning, f"OSD {osd_id} has {err_count} scrub errors",
                                         osd_name(osd_id))

        passed = total_scrub_err <= config.get('max_scrub_errors', 0)
        report.add_result(passed, "" if passed else f"Error count {total_scrub_err}")
    else:
        report.add_result(False, "No data available")


@checker(Severity.warning, "SSD/NVME used for journals/wal/db")
def ssd_nvme_journals(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    count = 0
    for osd in ceph.osds:
        if isinstance(osd.storage_info, FileStoreInfo):
            j_classes = config.get('classes_for_j', ["sata_ssd", "sas-ssd", "nvme"])
            j_disk_type = osd.host.disks[osd.storage_info.journal_dev].tp.name
            if j_disk_type not in j_classes:
                report.add_extra_message(Severity.warning,
                    f"OSD {osd.id} uses wrong disk type {j_disk_type} for journal. One of {j_classes} required",
                    osd_name(osd.id))
                count += 1
        else:
            assert isinstance(osd.storage_info, BlueStoreInfo)

            wal_classes = config.get('classes_for_wal', ["sata_ssd", "sas-ssd", "nvme"])
            wal_disk_type = osd.host.disks[osd.storage_info.wal_dev].tp.name
            if wal_disk_type not in wal_classes:
                report.add_extra_message(Severity.warning,
                    f"OSD {osd.id} uses wrong disk type {wal_disk_type} for wal. One of {wal_classes} required",
                    osd_name(osd.id))

            db_classes = config.get('classes_for_db', ["sata_ssd", "sas-ssd", "nvme"])
            db_disk_type = osd.host.disks[osd.storage_info.db_dev].tp.name
            if db_disk_type not in db_classes:
                report.add_extra_message(Severity.warning,
                    f"OSD {osd.id} uses wrong disk type {db_disk_type} for db. One of {db_classes} required",
                    osd_name(osd.id))

            if db_disk_type not in db_classes or wal_disk_type not in wal_classes:
                count += 1

    report.add_result(count == 0, "" if count == 0 else f"{count} OSD's uses slow disks for journal/wal/db")


@checker(Severity.warning, "Separated client/cluster ceph networks")
def separated_ceph_nets(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    if config.get('separated_nets', True):
        if ceph.cluster_net == ceph.public_net:
            msg = f"Same net {ceph.public_net} is used for both ceph cluster and public"
            report.add_extra_message(Severity.warning, msg)
            report.add_result(False, msg)
        else:
            report.add_result(True, "")


@checker(Severity.warning, "Max journal per device")
def max_journal_per_device(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    host_dev_j_count = collections.Counter()

    duuid = lambda node, dev_name: f"{node.name}::{dev_name}"

    for osd in ceph.osds:
        if isinstance(osd.storage_info, FileStoreInfo):
            host_dev_j_count[duuid(osd.host, osd.storage_info.journal_dev)] += 1
        else:
            host_dev_j_count[duuid(osd.host, osd.storage_info.wal_dev)] += 1
            if osd.storage_info.db_dev != osd.storage_info.wal_dev:
                host_dev_j_count[duuid(osd.host, osd.storage_info.db_dev)] += 1

    fcount = 0

    defaults = {
        "virtio": 1,
        "unknown": 1,
        "sata_hdd": 1,
        "sas_hdd": 1,
        "sata_ssd": 6,
        "sas_ssd": 6,
        "nvme": 8
    }

    for host in cluster.hosts.values():
        for disk in host.disks.values():
            count = host_dev_j_count[duuid(host, disk.name)]
            max_journals = config.get('max_journals_per_class', {}).get(disk.tp.name, defaults[disk.tp.name])

            if count > max_journals:
                fcount += 1
                report.add_extra_message(Severity.warning,
                    f"Device {disk.name} on node {host.name} has {count} journals, max {max_journals} recommended",
                    host.name)

    report.add_result(fcount == 0, "" if fcount == 0 else f"{fcount} devices have too may journals")


@checker(Severity.warning, "PG for OSD mast be in configured range")
def max_journal_per_device(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    min_pg = config.get("min_pg_per_osd", 100)
    max_pg = config.get("max_pg_per_osd", 400)

    wrong_pg_count = 0
    for osd in ceph.osds:
        if osd.pg_count < min_pg or osd.pg_count > max_pg:
            wrong_pg_count += 1
            report.add_extra_message(Severity.warning,
                f"OSD {osd.id} has wrong PG count {osd.pg_count}. Recommended to be in range [{min_pg}, {max_pg}]",
                osd_name(osd.id))

    report.add_result(wrong_pg_count == 0, "" if wrong_pg_count == 0 else f"{wrong_pg_count} osd's has wrong PG count")


@checker(Severity.warning, "Pool size/min_size")
def pool_size_minsize(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    pool_size = tuple(map(int, config.get('pool_size', "3/2").split("/")))
    gnocci_pool_size = tuple(map(int, config.get('gnocci_pool_size', "2/1").split("/")))

    wrong_size = 0
    for name, pool in ceph.pools.items():
        expected_size = gnocci_pool_size if name == 'gnocci' else pool_size
        if (pool.size, pool.min_size) != expected_size:
            wrong_size += 1
            report.add_extra_message(Severity.warning,
                f"Pool {pool.name} size={pool.size}, min_size={pool.min_size}, " +
                f"must be {expected_size[0]}/{expected_size[1]} respectivelly")

    report.add_result(wrong_size == 0, "" if wrong_size == 0 else f"{wrong_size} pools has wrong sizes")


@checker(Severity.warning, "Pool pg must be equals to pgp")
def pg_eq_pgp(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    if not config.get('pg_eq_pgp', True):
        return

    pg_ne_pgp = 0
    for name, pool in ceph.pools.items():
        if pool.pg != pool.pgp:
            pg_ne_pgp += 1
            report.add_extra_message(Severity.warning, f"Pool {pool.name} pg({pool.pg}) != pgp(pool.pgp)")

    report.add_result(pg_ne_pgp == 0, "" if pg_ne_pgp == 0 else f"{pg_ne_pgp} pools has pg != pgp")


@checker(Severity.warning, "Small pools should not have lots PG")
def pg_in_empty_pools(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    sett = config.get("almost_empty_pool", {})
    min_cluster_data = sett.get("min_total_GiB", 100) * 1024 ** 3
    if ceph.status.data_bytes < min_cluster_data:
        return

    data_part = sett.get("data_part", 0.0001)
    max_bytes = ceph.status.data_bytes * data_part
    max_pg = sett.get("max_pg", 128)
    max_pg_per_osd = sett.get("max_pg_per_osd", 1)
    max_pg_for_empty_pool = max([max_pg, len(ceph.osds) * max_pg_per_osd])
    max_objs = sum(pool.df.num_object_copies for pool in ceph.pools.values()) * data_part

    failed_pools = []
    too_many_pg = False
    for name, pool in sorted(ceph.pools.items()):
        if pool.pg > max_pg_for_empty_pool:
            if pool.df.size_bytes * pool.size < max_bytes and pool.df.num_object_copies < max_objs:
                too_many_pg = True
                report.add_extra_message(Severity.warning, f"Pool {pool.name} is very small " +
                    f"(total_data={pool.df.size_bytes * pool.size}, total_obj_copies={pool.df.num_object_copies}, " +
                    f"less then {data_part} from total) but has to many PG:" +
                    f" {pool.pg} > {max_pg_for_empty_pool}")
                failed_pools.append(name)

    report.add_result(not too_many_pg, f"Pools {', '.join(failed_pools)} has too many pg" if too_many_pg else "")


@checker(Severity.warning, "Large pools should have PG corresponding to their size")
def large_pools_pg(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    sett = config.get("large_pools", {})
    min_cluster_data = sett.get("min_total_GiB", 100) * 1024 ** 3

    if ceph.status.data_bytes < min_cluster_data:
        return

    min_large_pool_data = sett.get("large_pool_data_part", 0.05) * ceph.status.data_bytes
    min_size_diff = sett.get("min_size_diff", 1.5)

    pgs_and_data = []
    for name, pool in sorted(ceph.pools.items()):
        if pool.df.size_bytes * pool.size > min_large_pool_data:
            pgs_and_data.append((pool.name, pool.df.size_bytes, pool.pg))

    smaller_pools = []
    pg_count_messed = False
    if len(pgs_and_data) >= 2:
        pg_seq = sorted(pgs_and_data)
        for (sm_name, sm_data, sm_pg), (lg_name, lg_data, lg_pg) in zip(pg_seq[1:], pg_seq[:-1]):
            if sm_data * min_size_diff < lg_data and sm_pg > lg_pg:
                pg_count_messed = True
                report.add_extra_message(Severity.warning,
                    f"Pool {sm_name} has subsantially less data then {lg_name} ({min_size_diff:.1f})" +
                    f" but more PG {sm_pg} > {lg_pg}")
                smaller_pools.append(sm_name)
    report.add_result(not pg_count_messed,
                      f"Smaller pools({', '.join(smaller_pools)}) has more PG" if pg_count_messed else "")


@checker(Severity.warning, "All crush roots on replication level has >= 3 nodes, and same size in case if 3")
def replication_level_same_size(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    if not config.get('replication_level_same_size', True):
        return

    passed = True
    for root in ceph.crush.roots:
        childs = root.childs
        if len(childs) == 3:
            if abs(childs[0].weight - childs[1].weight) > 1 or abs(childs[2].weight - childs[1].weight) > 1:
                report.add_extra_message(Severity.warning, f"root {root.name} has 3 top childs with different weights")
                passed = False
        if len(childs) < 3:
            report.add_extra_message(Severity.warning, f"root {root.name} has less then 3 top childs")
            passed = False
    report.add_result(passed, "" if passed else "Incorrect tol level childs count of weights")


@checker(Severity.warning, "All OSD's under the same root has same storage types and almost the same size")
def all_osds_in_root_the_same(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):

    sett = config.get("osd_under_same_root", {})
    max_size_diff = sett.get("max_size_diff", 0.01)
    all_eq_types = sett.get("all_eq_types", True)

    failed_size = False

    if all_eq_types:
        failed_type = False

    for root in ceph.crush.roots:
        osds = list(root.iter_nodes('osd'))

        sizes = collections.Counter()

        if all_eq_types:
            types = collections.Counter()

        for osd_node in osds:
            osd = ceph.osds[osd_node.id]
            sizes[osd.host.storage_devs[osd.storage_info.data_partition].size] += 1
            if all_eq_types:
                types[osd.host.disks[osd.storage_info.data_dev].tp] += 1

        most_often_size = sorted((count, sz) for sz, count in sizes.items())[0][1]

        if all_eq_types:
            most_often_type = sorted((count, tp) for tp, count in types.items())[0][1]

        for osd_node in osds:
            osd = ceph.osds[osd_node.id]

            sz = osd.host.storage_devs[osd.storage_info.data_partition].size

            if abs(abs(sz - most_often_size) / sz - 1) > max_size_diff:
                report.add_extra_message(Severity.warning,
                    f"OSD {osd.id} has size {b2ssize(sz)} which is too different " +
                    f"from most often one {b2ssize(most_often_size)} for root {root.name}")
                failed_size = True

            if all_eq_types:
                tp = osd.host.disks[osd.storage_info.data_dev].tp
                if tp != most_often_type:
                    report.add_extra_message(Severity.warning,
                        f"OSD {osd.id} has disk type {tp.name} which is different " +
                        f"from most often one {most_often_type.name} for root {root.name}")
                    failed_type = True

    if all_eq_types:
        report.add_result(not failed_type, "Not all OSD's for same root has same drive types" if failed_type else "")

    report.add_result(not failed_size, "Not all OSD's for same root has same drive size" if failed_size else "")


@checker(Severity.warning, "OSD's with same size have close usage")
def osd_of_same_size_has_close_data_size(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    data_per_osd = collections.defaultdict(list)
    for osd in ceph.osds:
        total_space = osd.used_space + osd.free_space
        data_per_osd[total_space].append(osd.free_space / total_space)

    failed = False
    max_used_space_diff = config.get("max_used_space_diff", 0.2)
    for sz, free in data_per_osd.items():
        if abs(max(free) - min(free)) > max_used_space_diff:
            report.add_extra_message(Severity.warning,
                f"Min usage for osd's of size {b2ssize(sz)} is {min(free):.2f}, max is {min(free):.2f}, " +
                f"which is greter then allowed {max_used_space_diff}")
            failed = True

    report.add_result(not failed, "Too large dispersion in usage of osd's with same sizes" if failed else "")


@checker(Severity.warning, "OSD of the same type load evenly distributed")
def osd_of_same_size_has_close_load(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    io_per_osd = collections.defaultdict(lambda: collections.defaultdict(list))
    for osd_id, osd_info in ceph.osds_info.osds.items():
        for attr in ("reads", "writes", "read_b", "write_b"):
            io_per_osd[attr][osd_info.total_space].append((getattr(osd_info.pg_stats, attr), osd_id))

    sett = config.get("max_load_diff", {})
    min_ops = sett.get("min_mops", 1) * 1000000
    min_bytes = sett.get("min_GiB", 100) * 1024 ** 3
    max_diff = sett.get("max_diff", 0.2)
    failed = False
    for name, dct in io_per_osd.items():
        for sz, ops in dct.items():
            max_osd_ops, osd_max_id = max(ops)
            min_osd_ops, osd_min_id = min(ops)
            if name in ('reads', 'writes') and max_osd_ops < min_ops:
                continue
            if name in ('read_b', 'write_b') and min_osd_ops < min_bytes:
                continue

            diff = abs((max_osd_ops - min_osd_ops) / max_osd_ops)

            if diff > max_diff:
                report.add_extra_message(Severity.warning,
                    f"OSD's of size {b2ssize(sz)} has too large deviation in {name} load - " +
                    f"{diff:.2f} between most({osd_max_id}) and least({osd_min_id}) loaded OSD's, " +
                    f"which is greter then max allowed {max_diff}")
                failed = True

    report.add_result(not failed, "Too large dispersion in load of osd's with same sizes" if failed else "")


@checker(Severity.warning, "OSD of the same type load evenly distributed")
def osd_of_same_size_has_close_load(config: CheckConfig, cluster: Cluster, ceph: CephInfo, report: CheckReport):
    bad_size_count = 0

    wal_part_min = config.get("wal_part_min", 0.005)
    db_part_min = config.get("db_part_min", 0.05)

    for osd in ceph.osds:
        osd_info = ceph.osds_info.osds[osd.id]
        if osd_info.j_info:
            sync = float(osd.run_info.config['filestore_max_sync_interval']) if osd.run_info else 5
            min_j_size = osd_info.data_drive_type.bandwith_mbps() * 2 * sync

            if osd_info.j_info.size / 2 ** 20 < min_j_size:
                report.add_extra_message(Severity.warning,
                    f"OSD {osd.id} has too small journal - " +
                    f"{b2ssize(osd_info.j_info.size)}, min required is {min_j_size}")
                bad_size_count += 1
        else:
            assert osd_info.wal_db_info
            assert isinstance(osd.storage_info, BlueStoreInfo)
            data_part_size = osd.host.storage_devs[osd.storage_info.data_partition].size
            wal_part_size = osd.host.storage_devs[osd.storage_info.wal_partition].size

            min_wal_size =  data_part_size * wal_part_min
            min_db_size = data_part_size * db_part_min

            if osd_info.wal_db_info.wal_size is not None:
                if osd_info.wal_db_info.wal_size < min_wal_size:
                    bad_size_count += 1
                    report.add_extra_message(Severity.warning,
                        f"OSD {osd.id} has too small wal - {b2ssize(osd_info.j_info.size)}, " +
                        f"min required is {min_wal_size}")
            else:
                this_failed = False

                if osd.storage_info.wal_partition == osd.storage_info.db_partition:
                    if wal_part_size < min_wal_size + min_db_size:
                        this_failed = True
                        report.add_extra_message(Severity.warning,
                            f"OSD {osd.id} has too small wal+db partition - {b2ssize(wal_part_size)}, " +
                            f"min required is {min_wal_size + min_db_size}")
                else:
                    if wal_part_size < min_wal_size:
                        this_failed = True
                        report.add_extra_message(Severity.warning,
                            f"OSD {osd.id} has too small wal partition - {b2ssize(wal_part_size)}, " +
                            f"min required is {min_wal_size}")

                    db_part_size = osd.host.storage_devs[osd.storage_info.db_partition].size
                    if db_part_size < min_db_size:
                        this_failed = True
                        report.add_extra_message(Severity.warning,
                            f"OSD {osd.id} has too small db partition - {b2ssize(db_part_size)}, " +
                            f"min required is {min_db_size}")

                if this_failed:
                    bad_size_count += 1

    report.add_result(bad_size_count == 0, f"Journal/DB/wal has not enough space on {bad_size_count} OSD's"
                                           if bad_size_count != 0 else "")


def run_all_checks(config: CheckConfig, cluster: Cluster, ceph: CephInfo) \
        -> Tuple[List[Tuple[str, Severity, bool, Optional[str]]], Dict[str, List[str]]]:

    report = CheckReport()

    for check in ALL_CHECKS:
        report.set_curr(check.__name__, check.message, check.severity)
        check(config, cluster, ceph, report)

    services_parent_mapping = {osd_name(osd.id): osd.host.name for osd in ceph.osds}
    service_errors = collections.defaultdict(list)

    for name, results in report.extra.items():
        service_errors[name].extend(results)
        for parent in services_parent_mapping.get(name, []):
            service_errors[parent].extend(results)

    return report.results, service_errors

#
#
# def show_issues_table(cluster: Cluster, ceph: CephInfo) -> Tuple[str, Any]:
#     t = html.HTMLTable("table-issues", ["Check", "Result", "Comment"], sortable=False)
#
#     failed = html_fail("Failed")
#     passed = html_ok("Passed")
#
#     # ---------------------------------  No scrub errors ---------------------------------------------------------------
#
#     if ceph.osds_info:
#         total_scrub_err = 0
#         for osd_info in ceph.osds_info.osds.values():
#             total_scrub_err += osd_info.pg_stats.scrub_errors + \
#                                osd_info.pg_stats.deep_scrub_errors + \
#                                osd_info.pg_stats.shallow_scrub_errors
#
#         t.add_cells("No scrub errors",
#                     passed if total_scrub_err == 0 else failed,
#                     "" if total_scrub_err == 0 else f"Error count {total_scrub_err}")
#     else:
#         t.add_cells("No scrub errors", failed, "No data available")
#
#     # ---------------------------------  SSD/NVME journals -------------------------------------------------------------
#
#     count = 0
#     for osd in ceph.osds:
#         if isinstance(osd.storage_info, FileStoreInfo):
#             if not osd.host.disks[osd.storage_info.journal_dev].tp.is_fast():
#                 count += 1
#         else:
#             if not osd.host.disks[osd.storage_info.wal_dev].tp.is_fast():
#                 count += 1
#             elif not osd.host.disks[osd.storage_info.db_dev].tp.is_fast():
#                 count += 1
#
#     t.add_cells("SSD/NVME used for journals/wal/db",
#                 passed if count == 0 else failed,
#                 "" if 0 == count else f"{count} OSD's failed")
#
#     # ---------------------------------  separated ceph nets -----------------------------------------------------------
#
#     t.add_cells("Separated client/cluster ceph networks",
#                 passed if ceph.cluster_net != ceph.public_net else failed,
#                 "" if ceph.cluster_net != ceph.public_net else f"Same net {ceph.public_net}<br>is used for both")
#
#     # ---------------------------------  journal per drive count -------------------------------------------------------
#
#     host_dev_j_count = collections.Counter()
#     for osd in ceph.osds:
#         if isinstance(osd.storage_info, FileStoreInfo):
#             host_dev_j_count[f"{osd.host.name}/{osd.storage_info.journal_dev}"] += 1
#         else:
#             host_dev_j_count[f"{osd.host.name}/{osd.storage_info.wal_dev}"] += 1
#             if osd.storage_info.db_dev != osd.storage_info.wal_dev:
#                 host_dev_j_count[f"{osd.host.name}/{osd.storage_info.db_dev}"] += 1
#
#     fcount = 0
#     for host in cluster.hosts.values():
#         for disk in host.disks.values():
#             count = host_dev_j_count[f"{host.name}/{disk.name}"]
#             if (count > 1 and not disk.tp.is_fast()) or (count > 4 and disk.tp != DiskType.nvme) or (count > 6):
#                 fcount += 1
#
#     t.add_cells("Max 6 journals per nvme, 4 per SSD, 1 per HDD",
#                 passed if fcount == 0 else failed,
#                 "" if fcount == 0 else f"Failed for {fcount} drives")
#
#     # ---------------------------------  PG counts for OSD's -----------------------------------------------------------
#
#     wrong_pg_count = 0
#
#     for osd in ceph.osds:
#         if osd.pg_count < 100 or osd.pg_count > 400:
#             wrong_pg_count += 1
#
#     t.add_cells("400 >= OSD PG count >= 100",
#                 passed if wrong_pg_count == 0 else failed,
#                 "" if wrong_pg_count == 0 else f"Failed for {wrong_pg_count} drives")
#
#     # ---------------------------------  Pool size ---------------------------------------------------------------------
#
#     wrong_size = 0
#     for name, pool in ceph.pools.items():
#         expected_size = (2, 1) if name == 'gnocci' else (3, 2)
#         if (pool.size, pool.min_size) != expected_size:
#             wrong_size += 1
#
#     t.add_cells("3/2 for all pools, except 2/1 for gnocci",
#                 passed if wrong_size == 0 else failed,
#                 "" if wrong_size == 0 else f"Failed for {wrong_size} pools")
#
#     # ---------------------------------  Net performance ---------------------------------------------------------------
#
#     # ---------------------------------  Scrubbing at night ------------------------------------------------------------
#
#     # ---------------------------------  OSD threads and tcp connections -----------------------------------------------
#
#     # ---------------------------------  Same pg and pgp for all pools -------------------------------------------------
#
#     pg_ne_pgp = 0
#     for name, pool in ceph.pools.items():
#         if pool.pg != pool.pgp:
#             pg_ne_pgp += 1
#
#     t.add_cells("PG == PGP for all pools",
#                 passed if pg_ne_pgp == 0 else failed,
#                 "" if pg_ne_pgp == 0 else f"Failed for {pg_ne_pgp} pools")
#
#     # ---------------------------------  Find pools with virtual no data and many PG -----------------------------------
#     # ---------------------------------  PG counts for pools -----------------------------------------------------------
#
#     total_data = sum(pool.df.size_bytes for pool in ceph.pools.values())
#     total_objs = sum(pool.df.num_objects for pool in ceph.pools.values())
#     max_pg_for_empty_pool = max([128, len(ceph.osds)])
#     too_many_pg = False
#
#     pgs_and_data = []
#
#     for pool in ceph.pools.values():
#         if pool.pg > max_pg_for_empty_pool:
#             if pool.df.size_bytes < total_data / 10000 and pool.df.num_objects < total_objs / 10000:
#                 too_many_pg = True
#             if pool.df.size_bytes > total_data / 20 and pool.df.size_bytes > 100 * 2 ** 30:
#                 pgs_and_data.append((pool.df.size_bytes, pool.pg))
#
#     t.add_cells("Almost empty pool with too many PG's", failed if too_many_pg else passed, "")
#
#     pg_seq = [pg for _, pg in sorted(pgs_and_data)]
#     pg_count_messed = False
#     if len(pg_seq) >= 2:
#         for (sm_data, sm_pg), (lg_data, lg_pg) in zip(pgs_and_data[1:], pgs_and_data[:-1]):
#             if sm_data * 1.5 < lg_data and sm_pg > lg_pg:
#                 pg_count_messed = True
#     t.add_cells("Large pools has correct pg order", failed if pg_count_messed else passed, "")
#     # ---------------------------------  OSD nodes load ----------------------------------------------------------------
#
#     # ---------------------------------  OSD ram consumption -----------------------------------------------------------
#
#     # ---------------------------------  Second level nodes for all crush roots either count > 3 or same size ----------
#
#     for root in ceph.crush.roots:
#         childs = root.childs
#         if len(childs) == 3:
#             if abs(childs[0].weight - childs[1].weight) > 1 or abs(childs[2].weight - childs[1].weight) > 1:
#                 t.add_cells(">=3 top level childs, if ==3 - all childs have the same weights", failed, "")
#                 break
#         if len(childs) < 3:
#             t.add_cells(">=3 top level childs", failed, "")
#             break
#     else:
#         t.add_cells("3 top level childs should have same weights", passed, "")
#
#     # ---------------------------------  Each root has the same OSD's --------------------------------------------------
#
#     different_sizes = False
#     different_types = False
#
#     for root in ceph.crush.roots:
#         osds = list(root.iter_nodes('osd'))
#
#         osd = ceph.osds[osds[0].id]
#         devs = osd.host.disks
#         parts = osd.host.storage_devs
#         osd0_size = parts[osd.storage_info.data_partition].size
#         osd0_data_tp = devs[osd.storage_info.data_dev].tp
#
#         for osd_node in osds[1:]:
#             osd = ceph.osds[osd_node.id]
#             parts = osd.host.storage_devs
#             if abs(osd0_size - parts[osd.storage_info.data_partition].size) / osd0_size > 0.01:
#                 different_sizes = True
#             if osd.host.disks[osd.storage_info.data_dev].tp != osd0_data_tp:
#                 different_types = True
#
#     t.add_cells("All osds for same root have almost same size",
#                 failed if different_sizes else passed, "")
#
#     t.add_cells("All osds for same root have have same storage type",
#                 failed if different_types else passed, "")
#
#     # ---------------------------------  OSD of the same type data evenly distributed ----------------------------------
#
#     data_per_osd = collections.defaultdict(list)
#     for osd in ceph.osds:
#         total_space = osd.used_space + osd.free_space
#         data_per_osd[total_space].append(osd.free_space / total_space)
#
#     # for key, vals in data_per_osd.items():
#     #     print(f"{key} => {min(vals):.3f}, {max(vals):.3f}")
#
#     for sz, free in data_per_osd.items():
#         if abs(max(free) - min(free)) > 0.2:
#             t.add_cells("(max used OSD - min used OSD) < 0.2 for same OSD sizes", failed, "")
#             break
#     else:
#         t.add_cells("(max used OSD - min used OSD) < 0.2 for same OSD sizes", passed, "")
#
#     # ---------------------------------  OSD of the same type load evenly distributed ----------------------------------
#     io_per_osd = collections.defaultdict(lambda: collections.defaultdict(list))
#     for osd_info in ceph.osds_info.osds.values():
#         for attr in ("reads", "writes", "read_b", "write_b"):
#             io_per_osd[attr][osd_info.total_space].append(getattr(osd_info.pg_stats, attr))
#
#     for name, dct in io_per_osd.items():
#         for sz, free in dct.items():
#             mx = max(free)
#             if name in ('reads', 'writes') and mx < 1000000:
#                 continue
#             if name in ('read_b', 'write_b') and mx < 100000000000:
#                 continue
#             if abs((mx - min(free)) / mx) > 0.2:
#                 t.add_cells(f"(Max {name} OSD - min {name} OSD) < 0.2 for same OSD sizes", failed, "")
#                 break
#         else:
#             t.add_cells(f"(Max {name} OSD - min {name} OSD) < 0.2 for same OSD sizes", passed, "")
#     # ---------------------------------  Journal & wal & db sizes ------------------------------------------------------
#
#     bad_size_count = 0
#     for osd in ceph.osds:
#         osd_info = ceph.osds_info.osds[osd.id]
#         if osd_info.j_info:
#             sync = float(osd.run_info.config['filestore_max_sync_interval']) if osd.run_info else 5
#             min_j_size = osd_info.data_drive_type.bandwith_mbps() * 2 * sync
#
#             if osd_info.j_info.size / 2 ** 20 < min_j_size:
#                 bad_size_count += 1
#         else:
#             assert osd_info.wal_db_info
#             assert isinstance(osd.storage_info, BlueStoreInfo)
#             data_part_size = osd.host.storage_devs[osd.storage_info.data_partition].size
#             wal_part_size = osd.host.storage_devs[osd.storage_info.wal_partition].size
#
#             min_wal_size =  data_part_size * 0.005
#             min_db_size = data_part_size * 0.05
#
#             if osd_info.wal_db_info.wal_size is not None:
#                 if osd_info.wal_db_info.wal_size < min_wal_size:
#                     bad_size_count += 1
#             else:
#                 if osd.storage_info.wal_partition == osd.storage_info.db_partition:
#                     if wal_part_size < min_wal_size + min_db_size:
#                         bad_size_count += 1
#                 elif wal_part_size < min_wal_size:
#                     bad_size_count += 1
#                 elif osd.host.storage_devs[osd.storage_info.db_partition].size < min_db_size:
#                     bad_size_count += 1
#
#     t.add_cells("Journal and DB has enough space",
#                 passed if bad_size_count == 0 else failed,
#                 "" if fcount == 0 else f"Failed for {bad_size_count} journals/wal/db")
#
#     return "Issues:", t
