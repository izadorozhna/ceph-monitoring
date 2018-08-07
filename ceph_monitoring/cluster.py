import re
import copy
import json
import logging
import weakref
import datetime
import collections
from enum import Enum
from ipaddress import IPv4Network, IPv4Address
from typing import Iterable, Iterator, Dict, Any, List, Union, Tuple, Optional

import numpy
import dataclasses

from cephlib.crush import load_crushmap, Crush
from cephlib.units import unit_conversion_coef_f, ssize2b
from cephlib.storage import AttredStorage, TypedStorage
from cephlib.common import AttredDict

from .hw_info import parse_hw_info
from .cluster_classes import (IPANetDevInfo, NetStats, Disk, DiskType, HWModel, LogicBlockDev, BlockDevType, DevPath,
                              PGDump, PG, PGStatSum, PGId, PGState, NetworkBond, NetworkAdapter, NetAdapterAddr,
                              Pool, PoolDF, CephVersion, CephMonitor, MonRole, CephStatus, CephStatusCode,
                              Host, Cluster, CephInfo, CephOSD, OSDStatus, CephVersions, FileStoreInfo, BlueStoreInfo,
                              OSDProcessInfo, CephDevInfo, OSDPGStats, NodePGStats)
from .collect_info import parse_proc_file


logger = logging.getLogger("cephlib.parse")
NO_VALUE = -1

netstat_fields = "recv_bytes recv_packets rerrs rdrop rfifo rframe rcompressed" + \
                 " rmulticast send_bytes send_packets serrs sdrop sfifo scolls" + \
                 " scarrier scompressed"


# -----  parse functions -----------------------------------------------------------------------------------------------


def parse_ipa(data: str) -> Dict[str, IPANetDevInfo]:
    """
    parse 'ip a' output
    """
    res = {}
    for line in data.split("\n"):
        if line.strip() != '' and not line[0].isspace():
            rr = re.match(r"\d+:\s+(?P<name>[^:]*):.*?mtu\s+(?P<mtu>\d+)", line)
            if rr:
                name = rr.group('name')
                res[name] = IPANetDevInfo(name, int(rr.group('mtu')))
    return res


def parse_netdev(netdev: str) -> Dict[str, NetStats]:
    info: Dict[str, NetStats] = {}
    for line in netdev.strip().split("\n")[2:]:
        adapter, data = line.split(":")
        adapter = adapter.strip()
        assert adapter not in info
        info[adapter] = NetStats(**dict(zip(netstat_fields, map(int, data.split()))))
    return info


def parse_meminfo(meminfo: str) -> Dict[str, int]:
    info = {}
    for line in meminfo.split("\n"):
        line = line.strip()
        if line == '':
            continue
        name, data = line.split(":", 1)
        data = data.strip()
        if " " in data:
            data = data.replace(" ", "")
            assert data[-1] == 'B'
            val = ssize2b(data[:-1])
        else:
            val = int(data)
        info[name] = val
    return info


def parse_txt_ceph_config(data: str) -> Dict[str, str]:
    config = {}
    for line in data.strip().split("\n"):
        name, val = line.split("=", 1)
        config[name.strip()] = val.strip()
    return config


def is_routable(ip: IPv4Address, net: IPv4Network) -> bool:
    return not (ip.is_loopback or net.prefixlen == 32)


def parse_lsblkjs(data: List[Dict[str, Any]], hostname: str) -> Iterable[Disk]:
    for disk_js in data:
        name = disk_js['name']

        if re.match(r"sr\d+|loop\d+", name):
            continue

        if disk_js['type'] != 'disk':
            logger.warning("lsblk for node %r return device %r, which is %r not 'disk'. Ignoring it",
                           hostname, name, disk_js['type'])
            continue

        tp = disk_js["tran"]
        if tp is None:
            if disk_js['subsystems'] == 'block:scsi:pci':
                tp = 'sas'
            elif disk_js['subsystems'] == 'block:nvme:pci':
                tp = 'nvme'
            elif disk_js['subsystems'] == 'block:virtio:pci':
                tp = 'virtio'

        stor_tp = {
            ('sata', '1'): DiskType.sata_hdd,
            ('sata', '0'): DiskType.sata_ssd,
            ('nvme', '0'): DiskType.nvme,
            ('nvme', '1'): DiskType.unknown,
            ('sas',  '1'): DiskType.sas_hdd,
            ('sas',  '0'): DiskType.sas_ssd,
            ('virtio', '0'): DiskType.virtio,
            ('virtio', '1'): DiskType.virtio,
        }.get((tp, disk_js["rota"]))

        if stor_tp is None:
            logger.warning("Can't detect disk type for %r in node %r. tran=%r, rota=%r, subsystem=%r." +
                           " Treating it as " + DiskType.sata_hdd.name,
                           name, hostname, disk_js['tran'], disk_js['rota'], disk_js['subsystems'])
            stor_tp = DiskType.sata_hdd

        lbd = LogicBlockDev(name=name,
                            dev_path=DevPath('/dev/' + name),
                            size=int(disk_js['size']),
                            mountpoint=disk_js['mountpoint'],
                            fs=disk_js["fstype"],
                            tp=BlockDevType.hwdisk)

        dsk = Disk(logic_dev=lbd,
                   rota=disk_js['rota'] == '1',
                   scheduler=disk_js['sched'],
                   tp=stor_tp,
                   extra=disk_js,
                   hw_model=HWModel(vendor=disk_js['vendor'], model=disk_js['model'], serial=disk_js['serial']),
                   rq_size=int(disk_js["rq-size"]),
                   phy_sec=int(disk_js["phy-sec"]),
                   min_io=int(disk_js["min-io"]))

        def fill_mountable(root: Dict[str, Any], parent: Union[Disk, LogicBlockDev]):
            if 'children' in root:
                for ch_js in root['children']:
                    assert '/' not in ch_js['name']
                    part = LogicBlockDev(name=ch_js['name'],
                                         tp=BlockDevType.unknown,
                                         dev_path=DevPath('/dev/' + ch_js['name']),
                                         size=int(ch_js['size']),
                                         mountpoint=ch_js['mountpoint'],
                                         fs=ch_js["fstype"],
                                         parent=weakref.ref(dsk),
                                         label=ch_js.get("partlabel"))
                    parent.children[part.dev_path] = part
                    fill_mountable(ch_js, part)

        fill_mountable(disk_js, dsk)

        yield dsk


@dataclasses.dataclass
class DFInfo:
    path: DevPath
    name: str
    size: int
    free: int
    mountpoint: str


def parse_df(data: str) -> Iterator[DFInfo]:
    lines = data.strip().split("\n")
    assert lines[0].split() == ["Filesystem", "1K-blocks", "Used", "Available", "Use%", "Mounted", "on"]
    for ln in lines[1:]:
        name, size, _, free, _, mountpoint = ln.split()
        yield DFInfo(path=name, name=name.split("/")[-1], size=int(size), free=int(free), mountpoint=mountpoint)


def parse_pg_dump(data: Dict[str, Any]) -> PGDump:
    pgs: List[PG] = []
    for pg_info in data['pg_stats']:
        dt = {}
        for field in dataclasses.fields(PG):
            name = field.name
            tp = field.type
            vl_any = pg_info[name]
            if tp is datetime.datetime:
                vl_s, mks = vl_any.split(".")
                vl = datetime.datetime.strptime(vl_s, '%Y-%m-%d %H:%M:%S').replace(microsecond=int(mks))
            elif name in ("reported_epoch", "reported_seq"):
                vl = int(vl_any)
            elif tp in (int, str, bool):
                assert isinstance(vl_any, tp), f"Wrong type for field {name}. Expected {tp} got {type(vl_any)}"
                vl = vl_any
            elif name == 'stat_sum':
                vl = PGStatSum(**pg_info['stat_sum'])
            elif name in ('acting', 'blocked_by', 'up'):
                assert isinstance(vl_any, list)
                assert all(isinstance(vl_elem, int) for vl_elem in vl_any)
                vl = vl_any
            elif name == 'state':
                statuses = vl_any.split("+")
                vl = set(getattr(PGState, status) for status in statuses)
            elif name == 'pgid':
                pool_s, pg_s = vl_any.split(".")
                vl = PGId(pool=int(pool_s), num=int(pg_s, 16), id=vl_any)
            else:
                raise ValueError(f"Unknown field tp {tp} for field {name}")
            dt[name] = vl
        pgs.append(PG(**dt))

    datetm, mks = data['stamp'].split(".")
    collected_at = datetime.datetime.strptime(datetm, '%Y-%m-%d %H:%M:%S').replace(microsecond=int(mks))
    return PGDump(collected_at=collected_at,
                  pgs={pg.pgid.id: pg for pg in pgs},
                  version=data['version'])


# --- load functions ---------------------------------------------------------------------------------------------------

def parse_adapter_info(interface: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[bool]]:
    speed = None
    duplex = None
    speed_s = None

    if 'ethtool' in interface:
        for line in interface['ethtool'].split("\n"):
            if 'Speed:' in line:
                speed = line.split(":")[1].strip()
            if 'Duplex:' in line:
                duplex = line.split(":")[1].strip() == 'Full'

    if 'iwconfig' in interface:
        if 'Bit Rate=' in interface['iwconfig']:
            br1 = interface['iwconfig'].split('Bit Rate=')[1]
            if 'Tx-Power=' in br1:
                speed = br1.split('Tx-Power=')[0]

    if speed is not None:
        mults = {
            'Kb/s': 125,
            'Mb/s': 125000,
            'Gb/s': 125000000,
        }
        for name, mult in mults.items():
            if name in speed:  # type: ignore
                speed = int(float(speed.replace(name, '')) * mult)    # type: ignore
                break

        if isinstance(speed, int):
            speed = speed
        else:
            speed_s = speed

    return speed, speed_s, duplex


def load_bonds(stor_node: AttredStorage, jstor_node: AttredStorage) -> Dict[str, NetworkBond]:
    if 'bonds.json' not in jstor_node:
        return {}

    res = {}
    slave_rr = re.compile(r"Slave Interface: (?P<name>[^\s]+)")
    for name, file in jstor_node.bonds.items():
        slaves = [slave_info.group('name') for slave_info in slave_rr.finditer(stor_node[file])]
        res[name] = NetworkBond(name, slaves)
    return res


def load_interfaces(dtime: Optional[float],
                    hostname: str,
                    perf_monitoring: Any,
                    jstor_node: AttredStorage,
                    stor_node: AttredStorage) -> Dict[str, NetworkAdapter]:

    # net_stats = parse_netdev(stor_node.netdev)
    ifs_info = parse_ipa(stor_node.ipa)
    net_adapters = {}

    for name, vl in list(ifs_info.items()):
        if '@' in name:
            bond, _ = name.split('@')
            if bond not in ifs_info:
                ifs_info[bond] = vl

    for name, adapter_dct in jstor_node.interfaces.items():
        speed, speed_s, duplex = parse_adapter_info(adapter_dct)
        adapter = NetworkAdapter(dev=adapter_dct['dev'],
                                 is_phy=adapter_dct['is_phy'],
                                 mtu=None,
                                 speed=speed,
                                 speed_s=speed_s,
                                 duplex=duplex)

        try:
            adapter.mtu = ifs_info[adapter.dev].mtu
        except KeyError:
            logger.warning("Can't get mtu for interface %s on node %s", adapter.dev, hostname)

        net_adapters[adapter.dev] = adapter

    # find ip addresses
    ip_rr_s = r"\d+:\s+(?P<adapter>.*?)\s+inet\s+(?P<ip>\d+\.\d+\.\d+\.\d+)/(?P<size>\d+)"
    for line in stor_node.ipa4.split("\n"):
        match = re.match(ip_rr_s, line)
        if match is not None:
            ip_addr = IPv4Address(match.group('ip'))
            net_addr = IPv4Network("{}/{}".format(ip_addr, int(match.group('size'))), strict=False)
            dev_name = match.group('adapter')
            try:
                adapter = net_adapters[dev_name]
                adapter.ips.append(NetAdapterAddr(ip_addr, net_addr, is_routable=is_routable(ip_addr, net_addr)))
            except KeyError:
                logger.warning("Can't find adapter %r for ip %r", dev_name, ip_addr)

    return net_adapters


def load_hdds(hostname: str,
              jstor_node: AttredStorage,
              stor_node: AttredStorage) -> Tuple[Dict[str, Disk], Dict[str, LogicBlockDev]]:

    disks: Dict[str, Disk] = {}
    storage_devs: Dict[str, LogicBlockDev] = {}
    df_info = {info.name: info for info in parse_df(stor_node.df)}

    def walk_all(dsk):
        if dsk.name in df_info:
            dsk.free_space = df_info[dsk.name].free
        for ch in dsk.children.values():
            walk_all(ch)

    for dsk in parse_lsblkjs(jstor_node.lsblkjs['blockdevices'], hostname):
        disks[dsk.name] = dsk
        walk_all(dsk)
        storage_devs[dsk.name] = dsk.logic_dev
        storage_devs.update(dsk.children)

    return disks, storage_devs


def fill_io_devices_usage_stats(dtime: float, perf_monitoring: Any, disks: Dict[str, Disk]):
    logger.erro("io usage stats!")
#     if 'block-io' not in perf_monitoring or dtime < 1.0:
#         return
#
#     for dev, data in perf_monitoring['block-io'].items():
#         if dev in disks:
#             MS2S = 0.001
#             read_iops = sum(data['reads_completed']) / dtime
#             write_iops = sum(data['writes_completed']) / dtime
#             iops = read_iops + write_iops
#             w_io_time = MS2S * sum(data['weighted_io_time']) / dtime
#             load = DiskLoad(read_bytes=sum(data['sectors_read']) / dtime,
#                             write_bytes=sum(data['sectors_written']) / dtime,
#                             read_iops=read_iops,
#                             write_iops=write_iops,
#                             io_time=MS2S * sum(data['io_time']) / dtime,
#                             w_io_time=w_io_time,
#                             iops=iops,
#                             queue_depth=w_io_time,
#                             lat=w_io_time / iops if iops > 1E-5 else None,
#                             read_bytes_v=data['sectors_read'],
#                             write_bytes_v=data['sectors_written'],
#                             read_iops_v=data['reads_completed'],
#                             write_iops_v=data['writes_completed'],
#                             io_time_v=MS2S * data['io_time'],
#                             w_io_time_v=MS2S * data['weighted_io_time'],
#                             iops_v=data['reads_completed'] + data['writes_completed'],
#                             queue_depth_v=data['weighted_io_time'])
#             disks[dev].load = load
#         else:
#             logger.warning("No load info for %s", dev)


def load_pools(storage: TypedStorage, ver: CephVersions) -> Dict[int, Pool]:
    df_info: Dict[int, PoolDF] = {}

    for df_dict in storage.json.master.rados_df['pools']:
        if 'categories' in df_dict:
            assert len(df_dict['categories']) == 1
            df_dict = df_dict['categories'][0]

        del df_dict['name']
        id = df_dict.pop('id')
        df_info[id] = PoolDF(**df_dict)

    pools: Dict[int, Pool] = {}
    for info in storage.json.master.osd_dump['pools']:
        pool_id = int(info['pool'])
        pools[pool_id] = Pool(id=pool_id,
                              name=info['pool_name'],
                              size=int(info['size']),
                              min_size=int(info['min_size']),
                              pg=int(info['pg_num']),
                              pgp=int(info['pg_placement_num']),
                              crush_rule=int(info['crush_rule'] if ver >= CephVersions.luminous
                                             else info['crush_ruleset']),
                              extra=info,
                              df=df_info[pool_id])
    return pools


def load_perf_monitoring(storage: TypedStorage) \
        -> Tuple[Optional[Dict[str, Dict]], Optional[Dict[int, List[Dict]]], Optional[Dict[int, str]]]:

    if 'perf_monitoring' not in storage.txt:
        return None, None, None

    all_data: Dict[str, Dict] = {}
    osd_perf_dump: Dict[int, List[Dict]] = {}
    osd_historic_ops_paths: Dict[int, str] = {}
    osd_rr = re.compile(r"osd(\d+)$")
    for is_file, host_id in storage.txt.perf_monitoring:
        if is_file:
            logger.warning("Unexpected file %r in perf_monitoring folder", host_id)

        host_data = all_data[host_id] = {}
        for is_file, fname in storage.txt.perf_monitoring[host_id]:
            if is_file and fname == 'collected_at.csv':
                path = "perf_monitoring/{0}/collected_at.csv".format(host_id, fname)
                (_, _, _, units), _, data = storage.raw.get_array(path)
                host_data['collected_at'] = numpy.array(data)[::2] * unit_conversion_coef_f(units, 's')
                continue

            if is_file and fname.count('.') == 3:
                sensor, dev_or_osdid, metric, ext = fname.split(".")
                if ext == 'csv':
                    path = "perf_monitoring/{0}/{1}".format(host_id, fname)
                    _, _, data = storage.raw.get_array(path)

                    try:
                        int(dev_or_osdid)
                    except ValueError:
                        pass
                    else:
                        raise ValueError("Incorrect device name {}".format(dev_or_osdid))

                    name = '/dev/' + dev_or_osdid
                    host_data.setdefault(sensor, {}).setdefault(name, {})[metric] = numpy.array(data)
                    continue
                elif ext == 'json' and sensor == 'ceph' and metric == 'perf_dump':
                    osd_id = osd_rr.match(dev_or_osdid)
                    assert osd_id, "{0!r} don't match osdXXX name".format(dev_or_osdid)
                    assert osd_id.group(1) not in osd_perf_dump, "Two set of perf_dump data for osd {0}"\
                        .format(osd_id.group(1))
                    path = "perf_monitoring/{0}/{1}".format(host_id, fname)
                    osd_perf_dump[int(osd_id.group(1))] = json.loads(storage.raw.get_raw(path).decode("utf8"))
                    continue
                elif ext == 'bin' and sensor == 'ceph' and metric == 'historic':
                    osd_id = osd_rr.match(dev_or_osdid)
                    assert osd_id, "{0!r} don't match osdXXX name".format(dev_or_osdid)
                    assert osd_id.group(1) not in osd_perf_dump, \
                        "Two set of osd_historic_ops_paths data for osd {0}".format(osd_id.group(1))
                    osd_historic_ops_paths[int(osd_id.group(1))] = "perf_monitoring/{0}/{1}".format(host_id, fname)
                    continue

            logger.warning("Unexpected %s %r in %r host performance_data folder",
                           'file' if is_file else 'folder', fname, host_id)

    return all_data, osd_perf_dump, osd_historic_ops_paths


def load_PG_distribution(pools: Dict[str, Pool], pg_dump: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, int]],
                                                                                   Dict[str, int],
                                                                                   Dict[int, int]]:

    pool_id2name = {pool.id: pool.name for pool in pools.values()}
    osd_pool_pg_2d: Dict[int, Dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
    sum_per_pool: Dict[str, int] = collections.Counter()
    sum_per_osd: Dict[int, int] = collections.Counter()

    if pg_dump is None:
        for pg in pg_dump['pg_stats']:
            pool_id = int(pg['pgid'].split('.', 1)[0])
            for osd_id in pg['acting']:
                pool_name = pool_id2name[pool_id]
                osd_pool_pg_2d[osd_id][pool_name] += 1
                sum_per_pool[pool_name] += 1
                sum_per_osd[osd_id] += 1

    return {osd_id: dict(per_pool.items()) for osd_id, per_pool in osd_pool_pg_2d.items()}, \
            dict(sum_per_pool.items()), dict(sum_per_osd.items())


version_rr = re.compile(r'ceph version\s+(?P<version>[^ ]*)\s+\((?P<hash>[^)]*?)\)')


def parse_ceph_versions(data: str) -> Dict[str, CephVersion]:
    vers: Dict[str, CephVersion] = {}
    for line in data.split("\n"):
        line = line.strip()
        if line:
            name, data_js = line.split(":", 1)
            rr = version_rr.match(json.loads(data_js)["version"])
            if not rr:
                logger.error("Can't parse version %r from %r", json.loads(data_js)["version"], line)
            major, minor, bugfix = map(int, rr.group("version").split("."))
            vers[name.strip()] = CephVersion(major, minor, bugfix, rr.group("hash"))
    return vers


def load_monitors(storage: TypedStorage, ver: CephVersions, hosts: Dict[str, Host]) -> Dict[str, CephMonitor]:
    if ver >= CephVersions.luminous:
        mons = storage.json.master.status['monmap']['mons']
        # mon_status = self.storage.json.master.mon_status
    else:
        srv_health = storage.json.master.status['health']['health']['health_services']
        assert len(srv_health) == 1
        mons = srv_health[0]['mons']

    vers = parse_ceph_versions(storage.txt.master.mon_versions)
    result: Dict[str, CephMonitor] = {}
    for srv in mons:
        health = None if ver >= CephVersions.luminous else srv["health"]
        mon = CephMonitor(srv["name"], health, srv["name"], role=MonRole.unknown, version=vers["mon." + srv["name"]])

        if ver < CephVersions.luminous:
            mon.kb_avail = srv["kb_avail"]
            mon.avail_percent = srv["avail_percent"]
        else:
            mon_db_root = "/var/lib/ceph/mon"
            root = ""
            for info in hosts[mon.host].logic_block_devs.values():
                if info.mountpoint is not None:
                    if mon_db_root.startswith(info.mountpoint) and len(root) < len(info.mountpoint):
                        mon.kb_avail = info.free
                        root = info.mountpoint

        result[mon.name] = mon
    return result


def load_ceph_status(storage: TypedStorage, ver: CephVersions) -> CephStatus:
    mstorage = storage.json.master
    pgmap = mstorage.status['pgmap']
    health = mstorage.status['health']
    overall_status = health.get('overall_status', health['status'])
    return CephStatus(overall_status=CephStatusCode.from_str(overall_status),
                      status=CephStatusCode.from_str(health['status']),
                      health_summary=health['checks' if ver >= CephVersions.luminous else 'summary'],
                      num_pgs=pgmap['num_pgs'],
                      bytes_used=pgmap["bytes_used"],
                      bytes_total=pgmap["bytes_total"],
                      bytes_avail=pgmap["bytes_avail"],
                      data_bytes=pgmap["data_bytes"],
                      pgmap_stat=pgmap,
                      monmap_stat=mstorage.status['monmap'],

                      write_bytes_sec=pgmap.get("write_bytes_sec", 0),
                      write_op_per_sec=pgmap.get("write_op_per_sec", 0),
                      read_bytes_sec=pgmap.get("read_bytes_sec", 0),
                      read_op_per_sec=pgmap.get("read_op_per_sec", 0))


def avg_counters(counts: List[int], values: List[float]) -> numpy.ndarray:
    counts_a = numpy.array(counts, dtype=numpy.float32)
    values_a = numpy.array(values, dtype=numpy.float32)

    with numpy.errstate(divide='ignore', invalid='ignore'):  # type: ignore
        avg_vals = (values_a[1:] - values_a[:-1]) / (counts_a[1:] - counts_a[:-1])

    avg_vals[avg_vals == numpy.inf] = NO_VALUE
    avg_vals[numpy.isnan(avg_vals)] = NO_VALUE  # type: ignore

    return avg_vals  # type: ignore


class OSDStoreType(Enum):
    filestore = 0
    bluestore = 1
    unknown = 2


def load_osd_perf_data(osd_id: int,
                       storage_type: OSDStoreType,
                       osd_perf_dump: Dict[int, List[Dict]]) -> Dict[str, numpy.ndarray]:
    osd_perf_dump = osd_perf_dump[osd_id]
    osd_perf: Dict[str, numpy.ndarray] = {}

    if storage_type == OSDStoreType.filestore:
        fstor = [obj["filestore"] for obj in osd_perf_dump]
        for field in ("apply_latency", "commitcycle_latency", "journal_latency"):
            count = [obj[field]["avgcount"] for obj in fstor]
            values = [obj[field]["sum"] for obj in fstor]
            osd_perf[field] = avg_counters(count, values)

        arr = numpy.array([obj['journal_wr_bytes']["avgcount"] for obj in fstor], dtype=numpy.float32)
        osd_perf["journal_ops"] = arr[1:] - arr[:-1]  # type: ignore
        arr = numpy.array([obj['journal_wr_bytes']["sum"] for obj in fstor], dtype=numpy.float32)
        osd_perf["journal_bytes"] = arr[1:] - arr[:-1]  # type: ignore
    else:
        bstor = [obj["bluestore"] for obj in osd_perf_dump]
        for field in ("commit_lat",):
            count = [obj[field]["avgcount"] for obj in bstor]
            values = [obj[field]["sum"] for obj in bstor]
            osd_perf[field] = avg_counters(count, values)

    return osd_perf


class Loader:
    def __init__(self, storage: TypedStorage) -> None:
        self.storage = storage
        self.ip2host: Dict[str, Host] = {}
        self.hosts: Dict[str, Host] = {}
        self.perf_data: Any = None
        self.osd_perf_dump: Any = None
        self.osd_historic_ops_paths: Any = None

    def load_perf_monitoring(self):
        self.perf_data, self.osd_perf_dump, self.osd_historic_ops_paths = load_perf_monitoring(self.storage)

    def load_cluster(self) -> Cluster:
        coll_time = self.storage.txt['master/collected_at'].strip()
        report_collected_at_local, report_collected_at_gmt, _ = coll_time.split("\n")
        return Cluster(hosts=self.hosts,
                       ip2host=self.ip2host,
                       report_collected_at_local=report_collected_at_local,
                       report_collected_at_gmt=report_collected_at_gmt,
                       perf_data=self.perf_data)

    def load_ceph(self) -> CephInfo:
        settings = AttredDict(**parse_txt_ceph_config(self.storage.txt.master.default_config))

        mon_vers = parse_ceph_versions(self.storage.txt.master.mon_versions)
        ceph_master_version = max(mon_vers.values()).version
        try:
            pg_dump = self.storage.json.master.pg_dump
        except AttributeError:
            pg_dump = None

        pools = load_pools(self.storage, ceph_master_version)
        name2pool = {pool.name: pool for pool in pools.values()}
        osd_pool_pg_2d, sum_per_pool, sum_per_osd = load_PG_distribution(name2pool, pg_dump)
        crush = load_crushmap(content=self.storage.txt.master.crushmap)
        pgs = parse_pg_dump(pg_dump) if pg_dump else None

        return CephInfo(osds=self.load_osds(sum_per_osd, self.osd_perf_dump, self.osd_historic_ops_paths, crush, pgs),
                        mons=load_monitors(self.storage, ceph_master_version),
                        status=load_ceph_status(self.storage, ceph_master_version),
                        version=max(mon_vers.values()),
                        pools=name2pool,
                        osd_pool_pg_2d=osd_pool_pg_2d,
                        sum_per_pool=sum_per_pool,
                        sum_per_osd=sum_per_osd,
                        cluster_net=IPv4Network(settings['cluster_network']),
                        public_net=IPv4Network(settings['public_network']),
                        settings=settings,
                        pgs=pgs,
                        pgs_second=None,
                        mgrs=None,
                        radosgw=None,
                        crush=crush)

    def load_hosts(self) -> Tuple[Dict[str, Host], Dict[str, Host]]:
        tcp_sock_re = re.compile('(?im)^tcp6?\\b')
        udp_sock_re = re.compile('(?im)^udp6?\\b')

        for is_file, hostname in self.storage.txt.hosts:
            assert not is_file

            stor_node = self.storage.txt.hosts[hostname]
            jstor_node = self.storage.json.hosts[hostname]

            lshw_xml = stor_node.get('lshw', ext='xml')
            hw_info = None
            if lshw_xml is not None:
                try:
                    hw_info = parse_hw_info(lshw_xml)
                except:
                    pass

            loadavg = stor_node.get('loadavg')
            disks, logic_block_devs = load_hdds(hostname, jstor_node, stor_node)

            if self.perf_data is not None:
                perf_monitoring = self.perf_data.get(hostname)
                dtime = perf_monitoring['collected_at'][-1] - perf_monitoring['collected_at'][0]
                fill_io_devices_usage_stats(dtime, perf_monitoring, disks)
            else:
                perf_monitoring = None
                dtime = None

            net_adapters = load_interfaces(dtime, hostname, perf_monitoring, jstor_node, stor_node)
            bonds = load_bonds(stor_node, jstor_node)

            info = parse_meminfo(stor_node.meminfo)
            host = Host(name=hostname,
                        stor_id=hostname,
                        net_adapters=net_adapters,
                        disks=disks,
                        bonds=bonds,
                        logic_block_devs=logic_block_devs,
                        uptime=float(stor_node.uptime.split()[0]),
                        perf_monitoring=perf_monitoring,
                        open_tcp_sock=len(tcp_sock_re.findall(stor_node.netstat)),
                        open_udp_sock=len(udp_sock_re.findall(stor_node.netstat)),
                        mem_total=info['MemTotal'],
                        mem_free=info['MemFree'],
                        swap_total=info['SwapTotal'],
                        swap_free=info['SwapFree'],
                        load_5m=None if loadavg is None else float(loadavg.strip().split()[1]),
                        hw_info=hw_info)

            self.hosts[host.name] = host

            for adapter in net_adapters.values():
                for ip in adapter.ips:
                    if ip.is_routable:
                        ip_s = str(ip.ip)
                        if ip_s in self.ip2host:
                            logger.error("Ip %s belong to both %s and %s. Skipping new host",
                                         ip_s, hostname, self.ip2host[ip_s].name)
                            continue
                        self.ip2host[ip_s] = host

        return self.hosts, self.ip2host

    def load_osd_procinfo(self, osd_id: int) -> OSDProcessInfo:
        cmdln = [i.decode("utf8")
                 for i in self.storage.raw.get_raw('osd/{0}/cmdline.bin'.format(osd_id)).split(b'\x00')]

        pinfo = self.storage.json.osd[str(osd_id)].procinfo
        opened_socks = sum(int(tp.get('inuse', 0)) for tp in pinfo.get('ipv4', {}).values())
        opened_socks += sum(int(tp.get('inuse', 0)) for tp in pinfo.get('ipv6', {}).values())

        sched = pinfo['sched']
        if not sched:
            data = "\n".join(pinfo['sched_raw'].strip().split("\n")[2:])
            sched = parse_proc_file(data, ignore_err=True)

        return OSDProcessInfo(cmdline=cmdln,
                              procinfo=pinfo,
                              fd_count=pinfo["fd_count"],
                              opened_socks=opened_socks,
                              th_count=pinfo["th_count"],
                              cpu_usage=float(sched["se.sum_exec_runtime"]),
                              vm_rss=mem2bytes(pinfo["mem"]["VmRSS"]),
                              vm_size=mem2bytes(pinfo["mem"]["VmSize"]))

    def load_osd_devices(self, osd_id: int, host: Host) -> Union[FileStoreInfo, BlueStoreInfo]:
        osd_stor_node = self.storage.json.osd[str(osd_id)]
        osd_disks_info = osd_stor_node.devs_cfg

        def get_ceph_dev_info(devpath: str, partpath: str) -> CephDevInfo:
            dev_name = devpath.split("/")[-1]
            partition_name = partpath.split("/")[-1]
            return CephDevInfo(hostname=host.name,
                               path=DevPath(devpath),
                               name=dev_name,
                               partition_path=DevPath(partpath),
                               partition_name=partition_name,
                               dev_info=host.logic_block_devs[dev_name],
                               partition_info=host.logic_block_devs[partition_name],
                               d_usage=None)

        data_dev = get_ceph_dev_info(osd_disks_info['r_data'], osd_disks_info['data'])

        if osd_disks_info['type'] == 'filestore':
            j_dev = get_ceph_dev_info(osd_disks_info['r_journal'], osd_disks_info['journal'])
            return FileStoreInfo(data_dev, j_dev)
        else:
            assert osd_disks_info['type']  == 'bluestore'
            db_dev = get_ceph_dev_info(osd_disks_info['r_db'], osd_disks_info['db'])
            wal_dev = get_ceph_dev_info(osd_disks_info['r_wal'], osd_disks_info['wal'])
            return BlueStoreInfo(data_dev, db_dev, wal_dev)

    def load_osds(self,
                  sum_per_osd: Optional[Dict[int, int]],
                  osd_perf_dump: Optional[Dict[int, List[Dict]]],
                  osd_historic_ops_paths: Optional[Dict[int, str]],
                  crush: Crush,
                  pgs: PGDump = None) -> Dict[int, CephOSD]:
        try:
            fc = self.storage.txt.master.osd_versions
        except:
            fc = self.storage.raw.get_raw('master/osd_versions.err').decode("utf8")

        osd_versions: Dict[int, CephVersion] = {}

        for name, ver in parse_ceph_versions(fc).items():
            nm, id = name.split(".")
            assert nm == 'osd'
            osd_versions[int(id)] = ver

        osd_rw_dict = dict((node['id'], node['reweight'])
                           for node in self.storage.json.master.osd_tree['nodes']
                           if node['id'] >= 0)

        osd_perf_scalar = {}
        for node in self.storage.json.master.osd_perf['osd_perf_infos']:
            osd_perf_scalar[node['id']] = {"apply_latency_s": node["perf_stats"]["apply_latency_ms"],
                                           "commitcycle_latency_s": node["perf_stats"]["commit_latency_ms"]}

        osd_df_map = {node['id']: node for node in self.storage.json.master.osd_df['nodes']}
        osds: Dict[int, CephOSD] = {}

        osd2pg: Optional[Dict[int, List[PG]]] = None

        if pgs:
            osd2pg = collections.defaultdict(list)
            for pg in pgs.pgs.values():
                for osd_id in pg.acting:
                    osd2pg[osd_id].append(pg)

        for osd_data in self.storage.json.master.osd_dump['osds']:
            osd_id = osd_data['osd']
            osd_df_data = osd_df_map[osd_id]
            used_space = osd_df_data['kb_used'] * 1024
            free_space = osd_df_data['kb_avail']  * 1024
            cluster_ip = osd_data['cluster_addr'].split(":", 1)[0]

            config = AttredDict(**parse_txt_ceph_config(self.storage.txt.osd[str(osd_id)].config))

            try:
                host = self.ip2host[cluster_ip]
            except KeyError:
                logger.exception("Can't found host for osd %s, as no host own %r ip addr", osd_id, cluster_ip)
                raise

            status = OSDStatus.up if osd_data['up'] else OSDStatus.down
            storage_info = self.load_osd_devices(osd_id, host)

            #  CRUSH RULES/WEIGHTS
            crush_info: Dict[str, Tuple[bool, float, float]] = {}
            for root in crush.roots:
                nodes = crush.find_nodes([('root', root.name), ('osd', f"osd.{osd_id}")])
                if nodes:
                    if len(nodes) == 1:
                        expected_weight = storage_info.data.dev_info.size / (1024 ** 4)
                        crush_info[root.name] = (True, nodes[0].weight, expected_weight)
                    else:
                        crush_info[root.name] = (False, -1.0, -1.0)

            pg_stats: Optional[OSDPGStats] = None
            osd_pgs: Optional[List[PG]] = None

            if osd2pg:
                osd_pgs = [pg for pg in osd2pg[osd_id]]
                bytes = sum(pg.stat_sum.num_bytes for pg in osd_pgs)
                reads = sum(pg.stat_sum.num_read for pg in osd_pgs)
                read_b = sum(pg.stat_sum.num_read_kb * 1024 for pg in osd_pgs)
                writes = sum(pg.stat_sum.num_write for pg in osd_pgs)
                write_b = sum(pg.stat_sum.num_write_kb * 1024 for pg in osd_pgs)
                scrub_err = sum(pg.stat_sum.num_scrub_errors for pg in osd_pgs)
                deep_scrub_err = sum(pg.stat_sum.num_deep_scrub_errors for pg in osd_pgs)
                shallow_scrub_err = sum(pg.stat_sum.num_shallow_scrub_errors for pg in osd_pgs)
                pg_stats = OSDPGStats(bytes, reads, read_b, writes, write_b, scrub_err, deep_scrub_err,
                                      shallow_scrub_err)

            historic_ops_storage_path = osd_historic_ops_paths[osd_id] if osd_historic_ops_paths is not None else None
            osds[osd_id] = CephOSD(id=osd_id,
                                   config=config,
                                   cluster_ip=cluster_ip,
                                   public_ip=osd_data['public_addr'].split(":", 1)[0],
                                   reweight=osd_rw_dict[osd_id],
                                   version=osd_versions[osd_id],
                                   pg_count=None if sum_per_osd is None else sum_per_osd[osd_id],
                                   used_space=used_space,
                                   free_space=free_space,
                                   free_perc=int((free_space * 100.0) / (free_space + used_space) + 0.5),
                                   host=host,
                                   storage_info=storage_info,
                                   run_info=self.load_osd_procinfo(osd_id) if status != OSDStatus.down else None,
                                   crush_trees_weights=crush_info,
                                   status=status,
                                   total_space=storage_info.data.dev_info.size,
                                   pgs=osd_pgs,
                                   pg_stats=pg_stats,
                                   d_pg_stats=None,
                                   historic_ops_storage_path=historic_ops_storage_path,
                                   osd_perf=osd_perf_dump[osd_id] if osd_perf_dump is not None else None)

        return osds


def load_all(storage: TypedStorage) -> Tuple[Cluster, CephInfo]:
    l = Loader(storage)
    l.load_perf_monitoring()
    l.load_hosts()
    ceph = l.load_ceph()
    cluster = l.load_cluster()
    return cluster, ceph


def mem2bytes(vl: str) -> int:
    vl_sz, units = vl.split()
    assert units == 'kB'
    return int(vl_sz) * 1024


def get_nodes_pg_info(ceph: CephInfo) -> Dict[str, NodePGStats]:
    nodes_pg_info: Dict[str, NodePGStats] = {}
    assert ceph.osds_info is not None
    osds = ceph.osds_info
    for osd_id, osd_info in osds.osds.items():
        hostname = ceph.osds[osd_id].host.name
        if hostname in nodes_pg_info:
            info = nodes_pg_info[hostname]

            info.io_stat.shallow_scrub_errors += osd_info.pg_stats.shallow_scrub_errors
            info.io_stat.scrub_errors += osd_info.pg_stats.scrub_errors
            info.io_stat.deep_scrub_errors += osd_info.pg_stats.deep_scrub_errors
            info.io_stat.write_b += osd_info.pg_stats.write_b
            info.io_stat.writes += osd_info.pg_stats.writes
            info.io_stat.read_b += osd_info.pg_stats.read_b
            info.io_stat.reads += osd_info.pg_stats.reads
            info.io_stat.bytes += osd_info.pg_stats.bytes

            info.dio_stat.d_write_b += osd_info.d_stats.d_write_b
            info.dio_stat.d_writes += osd_info.d_stats.d_writes
            info.dio_stat.d_read_b += osd_info.d_stats.d_read_b
            info.dio_stat.d_reads += osd_info.d_stats.d_reads
            info.dio_stat.d_used_space += osd_info.d_stats.d_used_space

            info.pgs.extend(osd_info.pgs)
        else:
            nodes_pg_info[hostname] = NodePGStats(
                name=hostname,
                io_stat=copy.copy(osd_info.pg_stats),
                dio_stat=None if osd_info.d_stats is None else copy.copy(osd_info.d_stats),
                pgs=osd_info.pgs[:])
    return nodes_pg_info


# def get_osds_info(cluster: Cluster, ceph: CephInfo) -> OSDSInfo:
#
#     stor_set = set(osd.storage_type for osd in ceph.osds)
#     has_bs = OSDStoreType.bluestore in stor_set
#     has_fs = OSDStoreType.filestore in stor_set
#     assert has_fs or has_bs
#
#     all_vers = get_all_versions(ceph.osds)
#
#     osd2pg_map = collections.defaultdict(list)
#     if ceph.pgs:
#         for pg in ceph.pgs.pgs.values():
#             for osd_id in pg.acting:
#                 osd2pg_map[osd_id].append(pg)
#
#     osds: Dict[int, OSDInfo] = {}
#     for osd in ceph.osds:
#         disks = cluster.hosts[osd.host.name].disks
#         storage_devs = cluster.hosts[osd.host.name].storage_devs
#         assert osd.storage_info is not None
#
#         total_b = int(storage_devs[osd.storage_info.data_partition].size)
#
#
#         #  RUN INFO - FD COUNT, TCP CONN, THREADS
#         if osd.run_info is not None:
#             pinfo = osd.run_info.procinfo
#             opened_socks = sum(int(tp.get('inuse', 0)) for tp in pinfo.get('ipv4', {}).values())
#             opened_socks += sum(int(tp.get('inuse', 0)) for tp in pinfo.get('ipv6', {}).values())
#
#             sched = pinfo['sched']
#             if not sched:
#                 data = "\n".join(pinfo['sched_raw'].strip().split("\n")[2:])
#                 sched = parse_proc_file(data, ignore_err=True)
#
#             run_info = OSDProcessInfo(fd_count=pinfo["fd_count"],
#                                       opened_socks=opened_socks,
#                                       th_count=pinfo["th_count"],
#                                       cpu_usage=float(sched["se.sum_exec_runtime"]),
#                                       vm_rss=mem2bytes(pinfo["mem"]["VmRSS"]),
#                                       vm_size=mem2bytes(pinfo["mem"]["VmSize"]))
#         else:
#             run_info = None
#
#         #  USER DATA SIZE & TOTAL IO
#         if ceph.pgs:
#             pgs = osd2pg_map[osd.id]
#             bytes = sum(pg.stat_sum.num_bytes for pg in pgs)
#             reads = sum(pg.stat_sum.num_read for pg in pgs)
#             read_b = sum(pg.stat_sum.num_read_kb * 1024 for pg in pgs)
#             writes = sum(pg.stat_sum.num_write for pg in pgs)
#             write_b = sum(pg.stat_sum.num_write_kb * 1024 for pg in pgs)
#             scrub_err = sum(pg.stat_sum.num_scrub_errors for pg in pgs)
#             deep_scrub_err = sum(pg.stat_sum.num_deep_scrub_errors for pg in pgs)
#             shallow_scrub_err = sum(pg.stat_sum.num_shallow_scrub_errors for pg in pgs)
#
#             pg_stats = OSDPGStats(bytes, reads, read_b, writes, write_b, scrub_err, deep_scrub_err, shallow_scrub_err)
#
#             # USER DATA SIZE DELTA
#             if ceph.pgs_second:
#                 smap = ceph.pgs_second.pgs
#                 d_bytes = sum(smap[pg.pgid.id].stat_sum.num_bytes for pg in pgs) - bytes
#                 d_reads = sum(smap[pg.pgid.id].stat_sum.num_read for pg in pgs) - reads
#                 d_read_b = sum(smap[pg.pgid.id].stat_sum.num_read_kb * 1024 for pg in pgs) - read_b
#                 d_writes = sum(smap[pg.pgid.id].stat_sum.num_write for pg in pgs) - writes
#                 d_write_b = sum(smap[pg.pgid.id].stat_sum.num_write_kb * 1024 for pg in pgs) - write_b
#
#                 dtime = (ceph.pgs_second.collected_at - ceph.pgs.collected_at).total_seconds()
#
#                 d_stats = OSDDStats(d_bytes // dtime,
#                                     d_reads // dtime,
#                                     d_read_b // dtime,
#                                     d_writes // dtime,
#                                     d_write_b // dtime)
#             else:
#                 d_stats = None
#         else:
#             d_stats = None
#             pg_stats = None
#
#         # JOURNAL/WAL/DB info
#         sinfo = osd.storage_info
#         if isinstance(sinfo, FileStoreInfo):
#             collocation = sinfo.journal_dev != sinfo.data_dev
#             on_file = sinfo.journal_partition == sinfo.data_partition
#             drive_type = disks[sinfo.journal_dev].tp
#             size = None if collocation and on_file else storage_devs[sinfo.journal_partition].size
#             fs_info = JInfo(collocation, on_file, drive_type, size)
#             bs_info = None
#         else:
#             assert isinstance(sinfo, BlueStoreInfo)
#             wal_collocation = sinfo.wal_dev != sinfo.data_dev
#             wal_drive_type = disks[sinfo.wal_dev].tp
#
#             if sinfo.wal_partition not in (sinfo.db_partition, sinfo.data_partition):
#                 wal_size = int(storage_devs[sinfo.wal_partition].size)
#             else:
#                 wal_size = None
#
#             db_collocation = sinfo.db_dev != sinfo.data_dev
#             db_drive_type = disks[sinfo.db_dev].tp
#             if sinfo.db_partition != sinfo.data_partition:
#                 db_size = int(storage_devs[sinfo.db_partition].size)
#             else:
#                 db_size = None
#
#             fs_info = None
#             bs_info = WALDBInfo(wal_collocation=wal_collocation,
#                                 wal_drive_type=wal_drive_type,
#                                 wal_size=wal_size,
#                                 db_collocation=db_collocation,
#                                 db_drive_type=db_drive_type,
#                                 db_size=db_size)
#
#         osds[osd.id] = OSDInfo(pgs=osd2pg_map[osd.id] if ceph.pgs else None,
#                                total_space=total_b,
#                                used_space=0,
#                                free_space=0,
#                                total_user_data=0,
#                                crush_trees_weights=crush_info,
#                                data_drive_type=disks[osd.storage_info.data_dev].tp,
#                                data_part_size=storage_devs[osd.storage_info.data_partition].size,
#                                j_info=fs_info,
#                                wal_db_info=bs_info,
#                                run_info=run_info,
#                                pg_stats=pg_stats,
#                                d_stats=d_stats)
#
#     return OSDSInfo(has_bs=has_bs, has_fs=has_fs, all_versions=all_vers, largest_ver=max(all_vers),
#                     osds=osds)

