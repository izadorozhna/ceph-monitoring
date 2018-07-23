import re
import json
import logging
import weakref
import collections
from ipaddress import IPv4Network, IPv4Address
from typing import List, Dict, Any, Iterable, Tuple, Union, Set, Callable, Optional, Iterator
from enum import Enum

import numpy
from dataclasses import dataclass, field

from cephlib.crush import load_crushmap, Crush
from cephlib.common import AttredDict
from cephlib.units import unit_conversion_coef_f, ssize2b
from cephlib.storage import AttredStorage, TypedStorage

from .hw_info import parse_hw_info, get_dev_file_name, HWInfo


logger = logging.getLogger("cephlib.parse")
NO_VALUE = -1


class OSDStoreType(Enum):
    filestore = 0
    bluestore = 1
    unknown = 2


class MonRole(Enum):
    master = 0
    follower = 1
    unknown = 2


@dataclass
class PoolDF:
    size_bytes: int
    size_kb: int
    num_objects: int
    num_object_clones: int
    num_object_copies: int
    num_objects_missing_on_primary: int
    num_objects_unfound: int
    num_objects_degraded: int
    read_ops: int
    read_bytes: int
    write_ops: int
    write_bytes: int


@dataclass
class Pool:
    id: int
    name: str
    size: int
    min_size: int
    pg: int
    pgp: int
    crush_rule: int
    extra: Dict[str, Any]
    df: PoolDF


@dataclass
class NetLoad:
    send_bytes: numpy.ndarray[int]
    recv_bytes: numpy.ndarray[int]
    send_packets: numpy.ndarray[int]
    recv_packets: numpy.ndarray[int]

    send_bytes_avg: float = 0.0
    recv_bytes_avg: float = 0.0
    send_packets_avg: float = 0.0
    recv_packets_avg: float = 0.0

    def __post_init__(self):
        self.send_bytes_avg = self.send_bytes.mean()
        self.recv_bytes_avg = self.recv_bytes.mean()
        self.send_packets_avg = self.send_packets.mean()
        self.recv_packets_avg = self.recv_packets.mean()


@dataclass
class NetAdapterAddr:
    ip: IPv4Address
    net: IPv4Network


@dataclass
class NetworkAdapter:
    dev: str
    is_phy: bool
    speed: str
    duplex: bool
    mtu: int
    ips: List[NetAdapterAddr] = field(default_factory=list)

    load: Optional[NetLoad] = None


@dataclass
class DiskLoad:
    read_bytes: float
    write_bytes: float
    read_iops: float
    write_iops: float
    io_time: float
    w_io_time: float
    iops: float
    queue_depth: float
    lat: Optional[float]

    read_bytes_v: numpy.ndarray
    write_bytes_v: numpy.ndarray
    read_iops_v: numpy.ndarray
    write_iops_v: numpy.ndarray
    io_time_v: numpy.ndarray
    w_io_time_v: numpy.ndarray
    iops_v: numpy.ndarray
    queue_depth_v: numpy.ndarray


@dataclass
class Mountable:
    name: str
    size: int
    mountpoint: Optional[str] = None
    fs: Optional[str] = None
    free_space: Optional[int] = None
    parent: Optional[Callable[[], Optional['Disk']]] = None


@dataclass
class CephVersion:
    major: int
    minor: int
    bugfix: int
    commit_hash: str


@dataclass
class Disk:
    tp: str
    extra: Dict[str, Any]
    name: str
    size: int
    load: Optional[DiskLoad] = None
    partitions: Dict[str, Mountable] = field(default_factory=dict)
    mountpoint: Optional[str] = None
    fs: Optional[str] = None
    free_space: Optional[int] = None
    parent: Optional[Callable[[], Optional['Disk']]] = None


@dataclass
class Host:
    name: str
    stor_id: str
    net_adapters: Dict[str, NetworkAdapter]
    disks: Dict[str, Disk]
    storage_devs: Dict[str, Mountable]
    hw_info: Optional[HWInfo]

    uptime: float
    perf_monitoring: Any
    open_tcp_sock: int
    open_udp_sock: int
    mem_total: int
    mem_free: int
    swap_total: int
    swap_free: int
    load_5m: Optional[float]

    def find_interface(self, net: IPv4Network) -> Optional[NetworkAdapter]:
        for adapter in self.net_adapters.values():
            for ip in adapter.ips:
                if ip.ip in net:
                    return adapter
        return None


@dataclass
class CephMonitor:
    name: str
    status: Optional[str]
    host: Host
    role: MonRole

    version: Optional[CephVersion] = None
    kb_avail: Optional[int] = None
    avail_percent: Optional[int] = None


@dataclass
class BlueStoreInfo:
    data_dev: str
    data_partition: str
    db_dev: str
    db_partition: str
    wal_dev: str
    wal_partition: str

    data_stor_stats: Optional[DiskLoad] = None
    wal_stor_stats: Optional[DiskLoad] = None
    db_stor_stats: Optional[DiskLoad] = None


@dataclass
class FileStoreInfo:
    data_dev: str
    data_partition: str
    journal_dev: str
    journal_partition: str

    data_stor_stats: Optional[DiskLoad] = None
    j_stor_stats: Optional[DiskLoad] = None


class OSDStatus(Enum):
    up = 0
    down = 1
    out = 2


@dataclass
class OSDRunInfo:
    procinfo: Any
    cmdline: str
    config: Dict[str, str]


@dataclass
class CephOSD:
    id: int
    host: Host
    version: CephVersion
    status: OSDStatus

    cluster_ip: str
    public_ip: str
    pg_count: Optional[int]
    reweight: float
    used_space: int
    free_space: int
    free_perc: int

    storage_type: OSDStoreType = OSDStoreType.unknown
    storage_info: Union[BlueStoreInfo, FileStoreInfo, None] = None

    run_info: Optional[OSDRunInfo] = None
    osd_perf: Optional[Dict[str, Union[numpy.ndarray, float]]] = None
    historic_ops_storage_path: Optional[str] = None

    @property
    def daemon_runs(self) -> bool:
        return self.run_info is not None

    def __str__(self) -> str:
        return "OSD(id={0.id}, host={0.host.name}, free_perc={0.free_perc})".format(self)


netstat_fields = "recv_bytes recv_packets rerrs rdrop rfifo rframe rcompressed" + \
                 " rmulticast send_bytes send_packets serrs sdrop sfifo scolls" + \
                 " scarrier scompressed"

@dataclass
class NetStats:
    recv_bytes: int
    recv_packets: int
    rerrs: int
    rdrop: int
    rfifo: int
    rframe: int
    rcompressed: int
    rmulticast: int
    send_bytes: int
    send_packets: int
    serrs: int
    sdrop: int
    sfifo: int
    scolls: int
    scarrier: int
    scompressed: int


@dataclass
class DFInfo:
    name: str
    size: int
    free: int
    mountpoint: str


@dataclass
class IPANetDevInfo:
    name: str
    mtu: int


class CephStatusCode(Enum):
    ok = 0
    warn = 1
    err = 2


@dataclass
class Cluster:
    hosts: Dict[str, Host]
    ip2host: Dict[str, Host]
    report_collected_at_local: str
    report_collected_at_gmt: str
    perf_data: Any = None

    @property
    def has_performance_data(self) -> bool:
        return self.perf_data is not None


@dataclass
class CephStatus:
    overall_status: CephStatusCode
    health_summary: Any
    num_pgs: int

    bytes_used: int
    bytes_total: int
    bytes_avail: int

    data_bytes: int
    write_bytes_sec: int
    op_per_sec: int
    pgmap_stat: Any
    monmap_stat: Any


@dataclass
class CephInfo:
    osds: List[CephOSD]
    mons: List[CephMonitor]
    pools: Dict[str, Pool]
    osd_map: Dict[int, CephOSD]

    # pg distribution
    osd_pool_pg_2d: Dict[int, Dict[str, int]]
    sum_per_pool: Dict[str, int]
    sum_per_osd: Dict[int, int]

    is_luminous: bool

    crush: Crush
    cluster_net: IPv4Network
    public_net: IPv4Network
    settings: AttredDict

    status: CephStatus


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


def parse_lsblkjs(data: List[Dict[str, Any]], hostname: str) -> Iterable[Disk]:
    for disk_js in data:
        name = disk_js['name']

        if re.match(r"sr\d+|loop\d+", name):
            continue

        if disk_js['type'] != 'disk':
            logger.warning("lsblk for node %r return device %r, which is %r not 'disk'. Ignoring it",
                           hostname, name, disk_js['type'])
            continue

        stor_tp = {
            ('sata', '1'): 'hdd',
            ('sata', '0'): 'ssd',
            ('nvme', '0'): 'nvme'
        }.get((disk_js["tran"], disk_js["rota"]))

        if stor_tp is None:
            logger.warning("Can't detect disk type for %r in node %r. tran=%r and rota=%r. Treating it as hdd",
                           name, hostname, disk_js['tran'], disk_js['rota'])
            stor_tp = 'hdd'

        dsk = Disk(name='/dev/' + name,
                   size=disk_js['size'],
                   tp=stor_tp,
                   extra=disk_js,
                   mountpoint=disk_js['mountpoint'],
                   fs=disk_js["fstype"])

        for prt_js in disk_js['children']:
            assert '/' not in prt_js['name']
            part = Mountable(name='/dev/' + prt_js['name'],
                             size=prt_js['size'],
                             mountpoint=prt_js['mountpoint'],
                             fs=prt_js["fstype"],
                             parent=weakref.ref(dsk))
            dsk.partitions[part.name] = part
        yield dsk


def parse_df(data: str) -> Iterator[DFInfo]:
    lines = data.strip().split("\n")
    assert lines[0].split() == ["Filesystem", "1K-blocks", "Used", "Available", "Use%", "Mounted", "on"]
    for ln in lines[1:]:
        name, size, _, free, _, mountpoint = ln.split()
        yield DFInfo(name, int(size), int(free), mountpoint)


# --- load functions ---------------------------------------------------------------------------------------------------

def load_interfaces(dtime: float,
                    hostname: str,
                    perf_monitoring: Any,
                    jstor_node: AttredStorage,
                    stor_node: AttredStorage) -> Dict[str, NetworkAdapter]:
    # net_stats = parse_netdev(stor_node.netdev)
    ifs_info = parse_ipa(stor_node.ipa)
    net_adapters = {}

    for name, adapter_dct in jstor_node.interfaces.items():
        if dtime > 1.0 and perf_monitoring:
            params: Dict[str, numpy.ndarray[int]] = {}
            load_node = perf_monitoring.get('net-io', {}).get(adapter_dct['dev'], {})
            for metric in 'send_bytes send_packets recv_packets recv_bytes'.split():
                params[metric] = numpy.array(load_node[metric])
            load: Optional[NetLoad] = NetLoad(**params)  # type: ignore
        else:
            load = None

        adapter = NetworkAdapter(load=load, **adapter_dct)

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
                adapter.ips.append(NetAdapterAddr(ip_addr, net_addr))
            except KeyError:
                logger.warning("Can't find adapter %r for ip %r", dev_name, ip_addr)

    return net_adapters


def load_hdds(hostname: str,
              jstor_node: AttredStorage,
              stor_node: AttredStorage) -> Tuple[Dict[str, Disk], Dict[str, Mountable]]:

    disks: Dict[str, Disk] = {}
    storage_devs: Dict[str, Mountable] = {}
    df_info = {info.name: info for info in parse_df(stor_node.df)}

    for dsk in parse_lsblkjs(jstor_node.lsblkjs['blockdevices'], hostname):
        disks[dsk.name] = dsk
        for part in [dsk] + list(dsk.partitions.values()):  # type: ignore
            if part.name in df_info:
                part.free_space = df_info[part.name].free
        storage_devs[dsk.name] = dsk  # type: ignore
        storage_devs.update(dsk.partitions)
    return disks, storage_devs


def fill_io_devices_usage_stats(dtime: float, perf_monitoring: Any, disks: Dict[str, Disk]):
    if 'block-io' not in perf_monitoring or dtime < 1.0:
        return

    for dev, data in perf_monitoring['block-io'].items():
        if dev in disks:
            MS2S = 0.001
            read_iops = sum(data['reads_completed']) / dtime
            write_iops = sum(data['writes_completed']) / dtime
            iops = read_iops + write_iops
            w_io_time = MS2S * sum(data['weighted_io_time']) / dtime
            load = DiskLoad(read_bytes=sum(data['sectors_read']) / dtime,
                            write_bytes=sum(data['sectors_written']) / dtime,
                            read_iops=read_iops,
                            write_iops=write_iops,
                            io_time=MS2S * sum(data['io_time']) / dtime,
                            w_io_time=w_io_time,
                            iops=iops,
                            queue_depth=w_io_time,
                            lat=w_io_time / iops if iops > 1E-5 else None,
                            read_bytes_v=data['sectors_read'],
                            write_bytes_v=data['sectors_written'],
                            read_iops_v=data['reads_completed'],
                            write_iops_v=data['writes_completed'],
                            io_time_v=MS2S * data['io_time'],
                            w_io_time_v=MS2S * data['weighted_io_time'],
                            iops_v=data['reads_completed'] + data['writes_completed'],
                            queue_depth_v=data['weighted_io_time'])
            disks[dev].load = load
        else:
            logger.warning("No load info for %s", dev)


def load_pools(storage: TypedStorage, is_luminous: bool) -> Dict[int, Pool]:
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
                              crush_rule=int(info['crush_rule'] if is_luminous
                                                  else info['crush_ruleset']),
                              extra=info,
                              df=df_info[pool_id])
    return pools


def load_perf_monitoring(storage: TypedStorage) \
        -> Tuple[Optional[Dict[str, Dict]], Optional[Dict[int, Dict]], Optional[Dict[int, str]]]:

    if 'perf_monitoring' not in storage.txt:
        return None, None, None

    all_data: Dict[str, Dict] = {}
    osd_perf_dump: Dict[int, Any] = {}
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


def load_PG_distribution(storage: TypedStorage) -> Tuple[Dict[int, Dict[str, int]], Dict[str, int], Dict[int, int]]:
    try:
        pg_dump = storage.json.master.pg_dump
    except AttributeError:
        pg_dump = None

    pool_id2name: Dict[int, str] = {dt['poolnum']: dt['poolname'] for dt in storage.json.master.osd_lspools}

    pg_missing_reported = False
    osd_pool_pg_2d: Dict[int, Dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
    sum_per_pool: Dict[str, int] = collections.Counter()
    sum_per_osd: Dict[int, int] = collections.Counter()

    if pg_dump is None:
        for is_file, node in storage.txt.osd:
            if not is_file and node.isdigit():
                osd_id = int(node)
                if 'pgs' in node:
                    for pg in node.pgs.split():
                        pool_id_s, _ = pg.split(".")
                        pool_name = pool_id2name[int(pool_id_s)]
                        osd_pool_pg_2d[osd_id][pool_name] += 1
                        sum_per_pool[pool_name] += 1
                        sum_per_osd[osd_id] += 1
                elif not pg_missing_reported:
                    logger.warning("Can't load PG's for osd %s - data absent." +
                                   " This might happened if you are using bluestore and you" +
                                   " cluster has too many pg. You can collect data one more time and set " +
                                   "--max-pg-dump-count XXX to value, which greater or equal to you pg count."
                                   "All subsequent messages of missing pg for osd will be ommited", osd_id)
                    pg_missing_reported = True
    else:
        for pg in pg_dump['pg_stats']:
            pool_id = int(pg['pgid'].split('.', 1)[0])
            for osd_id in pg['acting']:
                pool_name = pool_id2name[pool_id]
                osd_pool_pg_2d[osd_id][pool_name] += 1
                sum_per_pool[pool_name] += 1
                sum_per_osd[osd_id] += 1

    return {osd_id: dict(per_pool.items()) for osd_id, per_pool in osd_pool_pg_2d.items()}, \
            dict(sum_per_pool.items()), dict(sum_per_osd.items())


def load_monitors(storage: TypedStorage, is_luminous: bool) -> List[CephMonitor]:
    if is_luminous:
        mons = storage.json.master.status['monmap']['mons']
        # mon_status = self.storage.json.master.mon_status
    else:
        srv_health = storage.json.master.status['health']['health']['health_services']
        assert len(srv_health) == 1
        mons = srv_health[0]['mons']

    for srv in mons:
        health = None if is_luminous else srv["health"]
        mon = CephMonitor(srv["name"], health, srv["name"], MonRole.unknown)

        if not is_luminous:
            mon.kb_avail = srv["kb_avail"]
            mon.avail_percent = srv["avail_percent"]
        mons.append(mon)
    return mons


def load_ceph_status(storage: TypedStorage, is_luminous: bool) -> CephStatus:
    mstorage = storage.json.master
    pgmap = mstorage.status['pgmap']
    health = mstorage.status['health']
    return CephStatus(overall_status=health['overall_status'],
                      health_summary=health['checks' if is_luminous else 'summary'],
                      num_pgs=pgmap['num_pgs'],
                      bytes_used=pgmap["bytes_used"],
                      bytes_total=pgmap["bytes_total"],
                      bytes_avail=pgmap["bytes_avail"],
                      data_bytes=pgmap["data_bytes"],
                      write_bytes_sec=pgmap.get("write_bytes_sec", 0),
                      op_per_sec=pgmap.get("op_per_sec", 0),
                      pgmap_stat=pgmap,
                      monmap_stat=mstorage.status['monmap'])


version_rr = re.compile(r'osd.(?P<osd_id>\d+)\s*:\s*' +
                        r'\{"version":"ceph version\s+(?P<version>[^ ]*)\s+\((?P<hash>[^)]*?)\).*?"\}\s*$')


def avg_counters(counts: List[int], values: List[float]) -> numpy.ndarray:
    counts_a = numpy.array(counts, dtype=numpy.float32)
    values_a = numpy.array(values, dtype=numpy.float32)

    with numpy.errstate(divide='ignore', invalid='ignore'):  # type: ignore
        avg_vals = (values_a[1:] - values_a[:-1]) / (counts_a[1:] - counts_a[:-1])

    avg_vals[avg_vals == numpy.inf] = NO_VALUE
    avg_vals[numpy.isnan(avg_vals)] = NO_VALUE  # type: ignore

    return avg_vals  # type: ignore


def load_osd_perf_data(osd_id: int,
                       storage_type: OSDStoreType,
                       osd_perf_dump: Dict[int, Dict]) -> Dict[str, numpy.ndarray]:
    osd_perf_dump = osd_perf_dump[osd_id]
    osd_perf: Dict[str,numpy.ndarray] = {}

    if storage_type == OSDStoreType.filestore:
        fstor = [obj["filestore"] for obj in osd_perf_dump.values()]
        for field in ("apply_latency", "commitcycle_latency", "journal_latency"):
            count = [obj[field]["avgcount"] for obj in fstor]
            values = [obj[field]["sum"] for obj in fstor]
            osd_perf[field] = avg_counters(count, values)

        arr = numpy.array([obj['journal_wr_bytes']["avgcount"] for obj in fstor], dtype=numpy.float32)
        osd_perf["journal_ops"] = arr[1:] - arr[:-1]  # type: ignore
        arr = numpy.array([obj['journal_wr_bytes']["sum"] for obj in fstor], dtype=numpy.float32)
        osd_perf["journal_bytes"] = arr[1:] - arr[:-1]  # type: ignore
    else:
        bstor = [obj["bluestore"] for obj in osd_perf_dump.values()]
        for field in ("commit_lat",):
            count = [obj[field]["avgcount"] for obj in bstor]
            values = [obj[field]["sum"] for obj in bstor]
            osd_perf[field] = avg_counters(count, values)

    return osd_perf


class Loader:
    def __init__(self, storage: TypedStorage) -> None:
        self.storage = storage
        self.ip2host: Dict[str, Host] = {}
        self.perf_data: Any = None

    def load_cluster(self) -> Cluster:
        coll_time = self.storage.txt['master/collected_at'].strip()
        report_collected_at_local, report_collected_at_gmt, _ = coll_time.split("\n")
        hosts, self.ip2host = self.load_hosts()
        return Cluster(hosts=hosts,
                       ip2host=self.ip2host,
                       report_collected_at_local=report_collected_at_local,
                       report_collected_at_gmt=report_collected_at_gmt,
                       perf_data=self.perf_data)

    def load_ceph(self) -> CephInfo:
        settings = AttredDict(**parse_txt_ceph_config(self.storage.txt.master.default_config))
        cluster_net = IPv4Network(settings['cluster_network'])
        public_net = IPv4Network(settings['public_network'])

        try:
            info = self.storage.json.master.mon_status['feature_map']['mon']['group']
            is_luminous = info['release'] == 'luminous'
        except KeyError:
            is_luminous = False

        crush = load_crushmap(content=self.storage.txt.master.crushmap)
        perf_data, osd_perf_dump, osd_historic_ops_paths = load_perf_monitoring(self.storage)

        status = load_ceph_status(self.storage, is_luminous)

        osd_pool_pg_2d, sum_per_pool, sum_per_osd = load_PG_distribution(self.storage)

        osd_map = self.load_osds(sum_per_osd, osd_perf_dump, osd_historic_ops_paths)

        pools = load_pools(self.storage, is_luminous)
        mons = load_monitors(self.storage, is_luminous)

        return CephInfo(osds=sorted(osd_map.values(), key=lambda x: x.id),
                        mons=mons,
                        status=status,
                        pools={pool.name: pool for pool in pools.values()},
                        osd_map=osd_map,
                        osd_pool_pg_2d=osd_pool_pg_2d,
                        sum_per_pool=sum_per_pool,
                        sum_per_osd=sum_per_osd,
                        is_luminous=is_luminous,
                        crush=crush,
                        cluster_net=cluster_net,
                        public_net=public_net,
                        settings=settings)

    def load_hosts(self) -> Tuple[Dict[str, Host], Dict[str, Host]]:
        hosts: Dict[str, Host] = {}
        tcp_sock_re = re.compile('(?im)^tcp6?\\b')
        udp_sock_re = re.compile('(?im)^udp6?\\b')
        ip2host: Dict[str, Host] = {}

        for is_file, host_ip_name in self.storage.txt.hosts:
            assert not is_file
            _, hostname = host_ip_name.split("-", 1)

            stor_node = self.storage.txt.hosts[host_ip_name]
            jstor_node = self.storage.json.hosts[host_ip_name]

            lshw_xml = stor_node.get('lshw', ext='xml')
            hw_info = None
            if lshw_xml is not None:
                try:
                    hw_info = parse_hw_info(lshw_xml)
                except:
                    pass

            loadavg = stor_node.get('loadavg')
            disks, storage_devs = load_hdds(hostname, jstor_node, stor_node)

            if self.perf_data is not None:
                perf_monitoring = self.perf_data.get(host_ip_name)
                dtime = perf_monitoring['collected_at'][-1] - perf_monitoring['collected_at'][0]
                fill_io_devices_usage_stats(dtime, perf_monitoring, disks)
            else:
                perf_monitoring = None
                dtime = None

            net_adapters = load_interfaces(dtime, hostname, perf_monitoring, jstor_node, stor_node)

            info = parse_meminfo(stor_node.meminfo)
            host = Host(name=hostname,
                        stor_id=host_ip_name,
                        net_adapters=net_adapters,
                        disks=disks,
                        storage_devs=storage_devs,
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

            hosts[host.name] = host

            for adapter in net_adapters.values():
                for ip in adapter.ips:
                    ip_s = str(ip.ip)
                    if not ip_s.startswith('127.'):
                        if ip_s in ip2host:
                            logger.error("Ip %s beelong to both %s and %s. Skipping new host",
                                         ip_s, hostname, ip2host[ip_s].name)
                            continue
                        ip2host[ip_s] = host

        return hosts, ip2host

    def load_osds(self,
                  sum_per_osd: Optional[Dict[int, int]],
                  osd_perf_dump: Optional[Dict[int, Dict]],
                  osd_historic_ops_paths: Optional[Dict[int, str]]) -> Dict[int, CephOSD]:

        osd_rw_dict = dict((node['id'], node['reweight'])
                           for node in self.storage.json.master.osd_tree['nodes']
                           if node['id'] >= 0)

        try:
            fc = self.storage.txt.master.osd_versions
        except:
            fc = self.storage.raw.get_raw('master/osd_versions.err').decode("utf8")

        osd_versions: Dict[int, CephVersion] = {}
        for line in fc.split("\n"):
            line = line.strip()
            if line:
                rr = version_rr.match(line.strip())
                if not rr:
                    raise ValueError(f"Can't parse OSD version {line.strip()}")
                major, minor, bugfix = map(int, rr.group('version').split('.'))
                osd_versions[int(rr.group('osd_id'))] = CephVersion(major, minor, bugfix, rr.group('hash'))

        osd_perf_scalar = {}
        for node in self.storage.json.master.osd_perf['osd_perf_infos']:
            osd_perf_scalar[node['id']] = {"apply_latency_s": node["perf_stats"]["apply_latency_ms"],
                                           "commitcycle_latency_s": node["perf_stats"]["commit_latency_ms"]}

        osd_df_map = {node['id']: node for node in self.storage.json.master.osd_df['nodes']}
        osd_map: Dict[int, CephOSD] = {}

        for osd_data in self.storage.json.master.osd_dump['osds']:
            osd_id = osd_data['osd']
            osd_df_data = osd_df_map[osd_id]
            used_space = osd_df_data['kb_used'] * 1024
            free_space = osd_df_data['kb_avail']  * 1024
            cluster_ip = osd_data['cluster_addr'].split(":", 1)[0]

            try:
                host = self.ip2host[cluster_ip]
            except KeyError:
                logger.exception("Can't found host for osd %s, as no host own %r ip addr", osd_id, cluster_ip)
                raise

            osd = CephOSD(
                    id=osd_id,
                    cluster_ip=cluster_ip,
                    public_ip=osd_data['public_addr'].split(":", 1)[0],
                    reweight=osd_rw_dict[osd_id],
                    status=OSDStatus.up if osd_data['up'] else OSDStatus.down,
                    version=osd_versions[osd_id],
                    pg_count=None if sum_per_osd is None else sum_per_osd[osd_id],
                    used_space=used_space,
                    free_space=free_space,
                    free_perc=int((free_space * 100.0) / (free_space + used_space) + 0.5),
                    host=host)

            osd_map[osd.id] = osd

            if osd.status != OSDStatus.down:
                config = AttredDict(**parse_txt_ceph_config(self.storage.txt.osd[str(osd_id)].config))

                osd_stor_node = self.storage.json.osd[str(osd.id)]
                osd_disks_info = osd_stor_node.devs_cfg

                if osd_disks_info['type'] == 'filestore':
                    osd.storage_type = OSDStoreType.filestore
                elif osd_disks_info['type'] == 'bluestore':
                    osd.storage_type = OSDStoreType.bluestore
                else:
                    raise ValueError(f"Unknown storage type {osd_disks_info['type']} for osd {osd_id}")

                osd.run_info = OSDRunInfo(
                    config=config,
                    procinfo=self.storage.json.osd[str(osd.id)].procinfo,
                    cmdline=self.storage.raw.get_raw('osd/{0}/cmdline'.format(osd.id)).split("\x00")
                )

                if osd.storage_type == OSDStoreType.filestore:
                    stor_info: Union[FileStoreInfo, BlueStoreInfo] = FileStoreInfo(
                        data_dev = osd_disks_info['r_data'],
                        data_partition = osd_disks_info['data'],
                        journal_dev=osd_disks_info['r_journal'],
                        journal_partition=osd_disks_info['journal'],
                    )

                    if self.perf_data is not None:
                        assert isinstance(stor_info, FileStoreInfo)  # make mypy happy
                        j_dev = get_dev_file_name(stor_info.journal_dev)
                        stor_info.j_stor_stats = osd.host.disks[j_dev].load
                else:
                    assert osd.storage_type == OSDStoreType.bluestore

                    stor_info = BlueStoreInfo(
                        data_dev=osd_disks_info['r_data'],
                        data_partition=osd_disks_info['data'],
                        db_dev=osd_disks_info['r_db'],
                        db_partition=osd_disks_info['db'],
                        wal_dev=osd_disks_info['r_wal'],
                        wal_partition=osd_disks_info['wal']
                    )

                    if self.perf_data is not None:
                        assert isinstance(stor_info, BlueStoreInfo)  # make mypy happy
                        w_dev = get_dev_file_name(stor_info.wal_dev)
                        stor_info.wal_stor_stats = osd.host.disks[w_dev].load
                        db_dev = get_dev_file_name(stor_info.db_dev)
                        stor_info.db_stor_stats = osd.host.disks[db_dev].load

                if osd_perf_dump is not None and osd.id in osd_perf_dump:
                    osd.osd_perf = load_osd_perf_data(osd.id, osd.storage_type, osd_perf_dump)  # type: ignore
                    osd.osd_perf.update(osd_perf_scalar[osd_id])  # type: ignore
                    stor_info.data_stor_stats = osd.host.disks[get_dev_file_name(stor_info.data_dev)].load

                    if osd_historic_ops_paths is not None:
                        osd.historic_ops_storage_path = osd_historic_ops_paths.get(osd.id)

                osd.storage_info = stor_info

        return osd_map


def load_all(storage: TypedStorage) -> Tuple[Cluster, CephInfo]:
    l = Loader(storage)
    return l.load_cluster(), l.load_ceph()
