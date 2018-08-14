import re
import datetime
from enum import Enum
from ipaddr import IPv4Address, IPv4Network
from typing import Optional, List, Set, Dict, Any, Callable, Union, Tuple, NewType, Iterable

import numpy
from dataclasses import dataclass, field

from cephlib.crush import Crush
from cephlib.units import b2ssize
from cephlib.common import AttredDict


# -----------  HW DEVICES CLASSES  -------------------------------------------------------------------------------------

# ---------------  LSHW ------------------------------------------------------------------------------------------------


@dataclass
class LSHWNetInfo:
    name: str
    speed: Optional[str]
    duplex: bool


@dataclass
class LSHWDiskInfo:
    size: int
    mount_point: Optional[str]
    device: Optional[str]


@dataclass
class LSHWCPUInfo:
    model: str
    cores: int


@dataclass
class LSHWInfo:
    hostname: Optional[str]
    cpu_info: List[LSHWCPUInfo]
    disks_info: Dict[str, LSHWDiskInfo]
    ram_size: int
    sys_name: Optional[str]
    mb: Optional[str]
    raw: str
    disks_raw_info: Dict[str, str]
    net_info: Dict[str, LSHWNetInfo]
    storage_controllers: List[str]
    summary: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.summary = {'cores': sum(info.cores for info in self.cpu_info),
                        'ram': self.ram_size,
                        'storage': sum(info.size for info in self.disks_info.values()),
                        'disk_count': len(self.disks_info)}

    def __str__(self):
        res = ["Simmary: {cores} cores, {ram}B RAM, {storage}B storage".format(**self.summary), str(self.sys_name)]
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
            for info in self.cpu_info:
                res.append(f"    {info.count} * {info.name}" if info.cores > 1 else f"    {info.name}")

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


# --------------------------------  COMMON -----------------------------------------------------------------------------


@dataclass
class HWModel:
    model: str
    serial: str
    vendor: str


# ---------------------------  NETWORK ---------------------------------------------------------------------------------

@dataclass
class NetAdapterAddr:
    ip: IPv4Address
    net: IPv4Network
    is_routable: bool


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
class NetworkAdapter:
    dev: str
    is_phy: bool
    usage: NetStats
    duplex: Optional[bool] = None
    mtu: Optional[int] = None
    speed: Optional[str] = None
    ips: List[NetAdapterAddr] = field(default_factory=list)
    speed_s: Optional[str] = None
    d_usage: Optional[NetStats] = None


@dataclass
class NetworkBond:
    name: str
    sources: List[str]


@dataclass
class IPANetDevInfo:
    name: str
    mtu: int


# ----------------------------------------  DISKS ----------------------------------------------------------------------


class BlockDevType(Enum):
    hwdisk = 0
    partition = 1
    lvm_lv = 2
    unknown = 3


class DiskType(Enum):
    sata_hdd = 0
    sas_hdd = 1
    nvme = 2
    sata_ssd = 3
    sas_ssd = 4
    virtio = 5
    unknown = 6


@dataclass
class BlockUsage:
    read_bytes: float
    write_bytes: float
    total_bytes: float
    read_iops: float
    write_iops: float
    io_time: float
    w_io_time: float
    iops: float
    queue_depth: float
    lat: Optional[float]


DevPath = NewType('DevPath', str)


@dataclass
class LogicBlockDev:
    tp: BlockDevType
    name: str
    hostname: str
    dev_path: DevPath
    size: int
    usage: BlockUsage
    mountpoint: Optional[str] = None
    fs: Optional[str] = None
    free_space: Optional[int] = None
    parent: Optional[Callable[[], Optional['Disk']]] = None
    children: Dict[str, 'LogicBlockDev'] = field(default_factory=dict)
    label: Optional[str] = None
    d_usage: Optional[BlockUsage] = None
    partition_num: int = 0

    def __post_init__(self):
        assert '/dev/' + self.name == self.dev_path
        assert '/' not in self.name
        for key in self.children:
            assert '/' not in key

        for pattern in ["sd[a-z]+(\d+)", "hd[a-z]+(\d+)", "vd[a-z]+(\d+)", "nvme\d+n\d+p(\d+)"]:
            rr = re.match(pattern, self.name)
            if rr:
                self.partition_num = int(rr.group(1))


@dataclass
class Disk:
    tp: DiskType
    logic_dev: LogicBlockDev
    extra: Dict[str, Any]
    scheduler: Optional[str]
    hw_model: HWModel
    rota: bool
    rq_size: int
    phy_sec: int
    min_io: int
    parent: Optional[Callable[[], Optional['Disk']]] = None

    # have to simulate LogicBlockDev inheritance, as mypy/pycharm works poorly with dataclass inheritance
    @property
    def name(self) -> str:
        return self.logic_dev.name

    @property
    def dev_path(self) -> DevPath:
        return self.logic_dev.dev_path

    @property
    def children(self) -> Dict[str, 'LogicBlockDev']:
        return self.logic_dev.children

    @property
    def size(self) -> int:
        return self.logic_dev.size

    @property
    def mountpoint(self) -> Optional[str]:
        return self.logic_dev.mountpoint

    @property
    def partition_num(self) -> int:
        return self.logic_dev.partition_num

    def __getattr__(self, name):
        return getattr(self.logic_dev, name)
# ----------------   CLUSTER  ------------------------------------------------------------------------------------------


@dataclass
class Host:
    name: str
    stor_id: str
    net_adapters: Dict[str, NetworkAdapter]
    disks: Dict[str, Disk]
    logic_block_devs: Dict[str, LogicBlockDev]
    hw_info: Optional[LSHWInfo]
    bonds: Dict[str, NetworkBond]

    uptime: float
    open_tcp_sock: int
    open_udp_sock: int
    mem_total: int
    mem_free: int
    swap_total: int
    swap_free: int
    load_5m: Optional[float]

    perf_monitoring: Any

    def find_interface(self, net: IPv4Network) -> Optional[NetworkAdapter]:
        for adapter in self.net_adapters.values():
            for ip in adapter.ips:
                if ip.ip in net:
                    return adapter
        return None


@dataclass
class Cluster:
    hosts: Dict[str, Host]
    ip2host: Dict[str, Host]
    report_collected_at_local: str
    report_collected_at_gmt: str
    has_second_report: bool
    perf_data: Any = None

    @property
    def has_performance_data(self) -> bool:
        return self.perf_data is not None

    @property
    def sorted_hosts(self) -> Iterable[Host]:
        return [host for _, host in sorted(self.hosts.items())]


# ----------------  CEPH -----------------------------------------------------------------------------------------------


class CephVersions(Enum):
    jewel = 10
    kraken = 11
    luminous = 12
    mimic = 13

    def __lt__(self, other: 'CephVersions') -> bool:
        return self.value < other.value

    def __gt__(self, other: 'CephVersions') -> bool:
        return self.value > other.value

    def __ge__(self, other: 'CephVersions') -> bool:
        return self.value >= other.value


@dataclass(order=True, unsafe_hash=True)
class CephVersion:
    major: int
    minor: int
    bugfix: int
    commit_hash: str

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.bugfix}  [{self.commit_hash[:8]}]"

    @property
    def version(self) -> CephVersions:
        return CephVersions(self.major)


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
    apps: List[str]


@dataclass
class CephMonitor:
    name: str
    status: Optional[str]
    host: Host
    role: MonRole
    version: CephVersion

    kb_avail: Optional[int] = None
    avail_percent: Optional[int] = None


class OSDStatus(Enum):
    up = 0
    down = 1
    out = 2


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
class CephStatus:
    status: CephStatusCode
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
class CephDevInfo:
    hostname: str
    dev_info: Disk
    partition_info: LogicBlockDev

    @property
    def name(self) -> str:
        return self.dev_info.name

    @property
    def path(self) -> DevPath:
        return self.dev_info.dev_path

    @property
    def partition_name(self) -> str:
        return self.partition_info.name

    @property
    def partition_path(self) -> DevPath:
        return self.partition_info.dev_path


@dataclass
class BlueStoreInfo:
    data: CephDevInfo
    db: CephDevInfo
    wal: CephDevInfo


@dataclass
class FileStoreInfo:
    data: CephDevInfo
    journal: CephDevInfo


@dataclass
class OSDProcessInfo:
    procinfo: Dict[str, Any]
    cmdline: List[str]
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
class NodePGStats:
    name: str
    pgs: List[PG]
    pg_stats: OSDPGStats
    d_pg_stats: Optional[OSDPGStats]


@dataclass
class CephOSD:
    id: int
    host: Host
    version: CephVersion
    status: OSDStatus
    config: Dict[str, str]

    cluster_ip: str
    public_ip: str
    pg_count: Optional[int]
    reweight: float
    class_name: Optional[str]

    free_perc: int
    used_space: int
    free_space: int
    total_space: int

    pgs: Optional[List[PG]]
    crush_rules_weights: Dict[str, Tuple[bool, float, float]]

    run_info: Optional[OSDProcessInfo]

    # load from the very beginning for all owned PG's & other pg stats
    pg_stats: Optional[OSDPGStats]

    # load during report collection for all owned PG's
    d_pg_stats: Optional[OSDPGStats]

    storage_info: Union[BlueStoreInfo, FileStoreInfo, None]

    osd_perf: Optional[Dict[str, Union[numpy.ndarray, float]]]
    historic_ops_storage_path: Optional[str]

    @property
    def daemon_runs(self) -> bool:
        return self.run_info is not None

    def __str__(self) -> str:
        return "OSD(id={0.id}, host={0.host.name}, free_perc={0.free_perc})".format(self)


class OSDStorageTypes(Enum):
    fs = 0
    bs = 1
    fs_and_bs = 2


class CephMGR:
    pass


class RadosGW:
    pass


@dataclass
class CephInfo:
    osds: Dict[int, CephOSD]
    mons: Dict[str, CephMonitor]
    mgrs: Optional[Dict[str, CephMGR]]
    radosgw: Optional[Dict[str, RadosGW]]
    pools: Dict[str, Pool]

    version: CephVersion  # largest monitor version
    status: CephStatus

    # pg distribution
    osd_pool_pg_2d: Dict[int, Dict[str, int]]
    sum_per_pool: Dict[str, int]
    sum_per_osd: Dict[int, int]

    crush: Crush
    cluster_net: IPv4Network
    public_net: IPv4Network
    settings: AttredDict

    pgs: Optional[PGDump]
    has_second_report: bool
    pgs_second: Optional[PGDump]

    has_fs: bool = field(init=False)
    has_bs: bool = field(init=False)

    def __post_init__(self):
        self.has_fs = any(isinstance(osd.storage_info, FileStoreInfo) for osd in self.osds.values())
        self.has_bs = any(isinstance(osd.storage_info, BlueStoreInfo) for osd in self.osds.values())
        assert self.has_fs or self.has_bs

    @property
    def is_luminous(self) -> bool:
        return self.version.major >= 12

    @property
    def fs_types(self) -> OSDStorageTypes:
        if self.has_bs and self.has_fs:
            return OSDStorageTypes.fs_and_bs
        elif self.has_bs:
            return OSDStorageTypes.bs
        else:
            return OSDStorageTypes.fs

    @property
    def sorted_osds(self) -> List[CephOSD]:
        return [osd for _, osd in sorted(self.osds.items())]
