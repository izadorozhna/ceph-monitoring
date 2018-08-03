from enum import Enum
import datetime
from ipaddr import IPv4Address, IPv4Network
from typing import Optional, List, Set, Dict, Any, Callable, Union, Tuple

import numpy
from dataclasses import dataclass, field

from cephlib.crush import Crush
from cephlib.units import b2ssize
from cephlib.common import AttredDict


@dataclass
class LSHWNetInfo:
    name: str
    speed: Optional[str]
    duplex: bool


@dataclass
class DiskInfo:
    size: int
    mount_point: Optional[str]
    device: Optional[str]


@dataclass
class CPUInfo:
    model: str
    cores: int


@dataclass
class HWInfo:
    hostname: Optional[str]
    cpu_info: List[CPUInfo]
    disks_info: Dict[str, DiskInfo]
    ram_size: int
    sys_name: Optional[str]
    mb: Optional[str]
    raw: str
    disks_raw_info: Dict[str, str]
    net_info: Dict[str, LSHWNetInfo]
    storage_controllers: List[str]

    def get_HDD_count(self):
        # SATA HDD COUNT, SAS 10k HDD COUNT, SAS SSD count, PCI-E SSD count
        return []

    def get_summary(self):
        cores = sum(count for _, count in self.cpu_info)
        disks = sum(info.size for info in self.disks_info.values())

        return {'cores': cores,
                'ram': self.ram_size,
                'storage': disks,
                'disk_count': len(self.disks_info)}

    def __str__(self):
        res = []

        summ = self.get_summary()
        summary = "Simmary: {cores} cores, {ram}B RAM, {disk}B storage"
        res.append(summary.format(cores=summ['cores'],
                                  ram=b2ssize(summ['ram']),
                                  disk=b2ssize(summ['storage'])))
        res.append(str(self.sys_name))
        if self.mb is not None:
            res.append("Motherboard: " + self.mb)

        if self.ram_size == 0:
            res.append("RAM: Failed to get RAM size")
        else:
            res.append(f"RAM {b2ssize(self.ram_size)}B")

        if self.cpu_info:
            res.append("CPU cores: Failed to get CPU info")
        else:
            res.append("CPU cores:")
            for name, count in self.cpu_info:
                res.append(f"    {count} * {name}" if count > 1 else f"    {name}")

        if self.storage_controllers:
            res.append("Disk controllers:")
            for descr in self.storage_controllers:
                res.append(f"    {descr}")

        if self.disks_info != {}:
            res.append("Storage devices:")
            for dev, info in sorted(self.disks_info.items()):
                res.append(f"    {dev} {b2ssize(info.size)}B {info.model}")
        else:
            res.append("Storage devices's: Failed to get info")

        if self.disks_raw_info != {}:
            res.append("Disks devices:")
            for dev, descr in sorted(self.disks_raw_info.items()):
                res.append(f"    {dev} {descr}")
        else:
            res.append("Disks devices's: Failed to get info")

        if self.net_info != {}:
            res.append("Net adapters:")
            for name, adapter in self.net_info.items():
                res.append(f"    {name} {adapter.is_phy} duplex={adapter.speed}")
        else:
            res.append("Net adapters: Failed to get net info")

        return str(self.hostname) + ":\n" + "\n".join("    " + i for i in res)


@dataclass
class HDDsResourceUsage:
    # 2D arrays for entire cluster load
    wio: numpy.ndarray
    wbytes: numpy.ndarray
    rio: numpy.ndarray
    rbytes: numpy.ndarray
    qd: numpy.ndarray


@dataclass
class CephDisksResourceUsage:
    # 2D arrays for entire cluster load
    data: HDDsResourceUsage
    journal: Optional[HDDsResourceUsage]
    wal: Optional[HDDsResourceUsage]
    db: Optional[HDDsResourceUsage]


@dataclass
class NetLoad:
    send_bytes: numpy.ndarray
    recv_bytes: numpy.ndarray
    send_packets: numpy.ndarray
    recv_packets: numpy.ndarray

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
class NetAdapterAddr:
    ip: IPv4Address
    net: IPv4Network
    is_routable: bool


@dataclass
class NetworkAdapter:
    dev: str
    is_phy: bool
    duplex: Optional[bool] = None
    mtu: Optional[int] = None
    speed: Optional[str] = None
    ips: List[NetAdapterAddr] = field(default_factory=list)
    speed_s: Optional[str] = None
    load: Optional[NetLoad] = None


@dataclass
class NetworkBond:
    name: str
    sources: List[str]


@dataclass
class DFInfo:
    path: str
    name: str
    size: int
    free: int
    mountpoint: str


@dataclass
class Mountable:
    name: str
    path: str
    size: int
    mountpoint: Optional[str] = None
    fs: Optional[str] = None
    free_space: Optional[int] = None
    parent: Optional[Callable[[], Optional['Disk']]] = None
    children: Dict[str, 'Mountable'] = field(default_factory=dict)
    label: Optional[str] = None


@dataclass(order=True, unsafe_hash=True)
class CephVersion:
    major: int
    minor: int
    bugfix: int
    commit_hash: str

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.bugfix}  [{self.commit_hash[:8]}]"


@dataclass
class HWModel:
    model: str
    serial: str
    vendor: str


class DiskType(Enum):
    sata_hdd = 0
    sas_hdd = 1
    nvme = 2
    sata_ssd = 3
    sas_ssd = 4
    virtio = 5
    unknown = 6

    def is_fast(self) -> bool:
        res = self.value in (self.nvme.value, self.sata_ssd.value, self.sas_ssd.value)
        return res

    def bandwith_mbps(self) -> int:
        if self.is_fast():
            return 200
        elif self.value == self.sas_hdd.value:
            return 100
        else:
            return 50


@dataclass
class Disk:
    tp: DiskType
    extra: Dict[str, Any]
    name: str
    path: str
    size: int
    rota: bool
    scheduler: Optional[str]
    hw_model: HWModel
    rq_size: int
    phy_sec: int
    min_io: int
    load: Optional[DiskLoad] = None
    children: Dict[str, Mountable] = field(default_factory=dict)
    mountpoint: Optional[str] = None
    fs: Optional[str] = None
    free_space: Optional[int] = None
    parent: Optional[Callable[[], Optional['Disk']]] = None
    label = None


@dataclass
class Host:
    name: str
    stor_id: str
    net_adapters: Dict[str, NetworkAdapter]
    disks: Dict[str, Disk]
    storage_devs: Dict[str, Mountable]
    hw_info: Optional[HWInfo]
    df_info: Dict[str, DFInfo]
    bonds: Dict[str, NetworkBond]

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
    version: CephVersion

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
    procinfo: Dict[str, Any]
    cmdline: List[str]
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
class IPANetDevInfo:
    name: str
    mtu: int


class CephStatusCode(Enum):
    ok = 0
    warn = 1
    err = 2

    @classmethod
    def from_str(cls, val: str) -> 'CephStatusCode':
        val_l = val.lower()
        if val_l in ('error', 'err', 'health_err'):
            return cls.err
        if val_l in ('warning', 'warn', 'health_warn'):
            return cls.warn
        assert val_l in ('ok', 'health_ok'), val_l
        return cls.ok


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
    status: CephStatusCode
    overall_status: CephStatusCode
    health_summary: Any
    num_pgs: int

    bytes_used: int
    bytes_total: int
    bytes_avail: int

    data_bytes: int
    pgmap_stat: Any
    monmap_stat: Dict[str, Any]

    write_bytes_sec: int
    read_bytes_sec: int
    write_op_per_sec: int
    read_op_per_sec: int


class PGState(Enum):
    active = 0
    clean = 1
    peering = 2


@dataclass
class PGStatSum:
    num_bytes: int
    num_bytes_hit_set_archive: int
    num_bytes_recovered: int
    num_deep_scrub_errors: int
    num_evict: int
    num_evict_kb: int
    num_evict_mode_full: int
    num_evict_mode_some: int
    num_flush: int
    num_flush_kb: int
    num_flush_mode_high: int
    num_flush_mode_low: int
    num_keys_recovered: int
    num_legacy_snapsets: int
    num_object_clones: int
    num_object_copies: int
    num_objects: int
    num_objects_degraded: int
    num_objects_dirty: int
    num_objects_hit_set_archive: int
    num_objects_misplaced: int
    num_objects_missing: int
    num_objects_missing_on_primary: int
    num_objects_omap: int
    num_objects_pinned: int
    num_objects_recovered: int
    num_objects_unfound: int
    num_promote: int
    num_read: int
    num_read_kb: int
    num_scrub_errors: int
    num_shallow_scrub_errors: int
    num_whiteouts: int
    num_write: int
    num_write_kb: int


@dataclass
class PGId:
    pool: int
    num: int
    id: str


@dataclass
class PG:
    acting: List[int]
    acting_primary: int
    blocked_by: List[int]
    created: int
    dirty_stats_invalid: bool
    hitset_bytes_stats_invalid: bool
    hitset_stats_invalid: bool
    last_active: datetime.datetime
    last_became_active: datetime.datetime
    last_became_peered: datetime.datetime
    last_change: datetime.datetime
    last_clean: datetime.datetime
    last_clean_scrub_stamp: datetime.datetime
    last_deep_scrub: str
    last_deep_scrub_stamp: datetime.datetime
    last_epoch_clean: int
    last_fresh: datetime.datetime
    last_fullsized: datetime.datetime
    last_peered: datetime.datetime
    last_scrub: str
    last_scrub_stamp: datetime.datetime
    last_undegraded: datetime.datetime
    last_unstale: datetime.datetime
    log_size: int
    log_start: str
    mapping_epoch: int
    omap_stats_invalid: bool
    ondisk_log_size: int
    ondisk_log_start: str
    parent: str
    parent_split_bits: int
    pgid: PGId
    pin_stats_invalid: bool
    reported_epoch: int
    reported_seq: int
    state: Set[PGState]
    stats_invalid: bool
    up: List[int]
    up_primary: int
    version: str
    stat_sum: PGStatSum


@dataclass
class PGDump:
    version: int
    collected_at: datetime.datetime
    pgs: Dict[str, PG]


@dataclass
class JInfo:
    collocation: bool
    on_file: bool
    drive_type: DiskType
    size: Optional[int]


@dataclass
class WALDBInfo:
    wal_collocation: bool
    wal_drive_type: DiskType
    wal_size: Optional[int]

    db_collocation: bool
    db_drive_type: DiskType
    db_size: Optional[int]


@dataclass
class OSDProcessInfo:
    opened_socks: int
    fd_count: int
    th_count: int
    cpu_usage: float
    vm_rss: int
    vm_size: int


@dataclass
class OSDPGStats:
    bytes: int

    reads: int
    read_b: int
    writes: int
    write_b: int

    scrub_errors: int
    deep_scrub_errors: int
    shallow_scrub_errors: int


@dataclass
class OSDDStats:
    d_used_space: int
    d_reads: int
    d_read_b: int
    d_writes: int
    d_write_b: int


@dataclass
class NodePGStats:
    name: str
    pgs: List[PG]
    io_stat: OSDPGStats
    dio_stat: Optional[OSDDStats]


@dataclass
class OSDInfo:
    pgs: Optional[List[PG]]
    total_space: int
    used_space: int
    free_space: int
    total_user_data: int
    crush_trees_weights: Dict[str, Tuple[bool, float, float]]
    data_drive_type: DiskType
    data_part_size: int

    # FS/BS info
    j_info: Optional[JInfo]
    wal_db_info: Optional[WALDBInfo]

    run_info: Optional[OSDProcessInfo]

    # load from the very beginning for all owned PG's & other pg stats
    pg_stats: Optional[OSDPGStats]

    # load during report collection for all owned PG's
    d_stats: Optional[OSDDStats]


@dataclass
class OSDSInfo:
    has_fs: bool
    has_bs: bool
    osds: Dict[int, OSDInfo]
    all_versions: Dict[CephVersion, int]
    largest_ver: CephVersion


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

    pgs: Optional[PGDump]
    pgs_second: Optional[PGDump]
    osds_info: Optional[OSDSInfo] = None
