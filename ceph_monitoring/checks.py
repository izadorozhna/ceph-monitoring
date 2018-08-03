from enum import Enum
from typing import List, Set
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from .cluster_classes import Cluster, CephInfo, DiskType


# Check
# disk types
# partition sizes
# mount options
# network speed
# mtu
# ceph settings
# rgw settings
# osd/mon/cluster log errors
# rgw/monitor collocation
# osd usage balance
# pools pg/load/data
# total cluster pg
# services cpu/ram consumption
# BS cache settings
# crush replication level sizes
# pools size/min_size
# kernel settings
# thread count/net connection count
# deep scrub time
# pg io bottleneck
# radosgw performance metrics!
# cpu/disk/net per user request


class CheckCode(Enum):
    journal_type = 0
    db_type = 1
    wal_type = 2
    journal_size = 3
    wal_size = 4
    db_size = 5


class Severity(Enum):
    ok = 0
    notice = 1
    warning = 2
    error = 3
    critical = 4


@dataclass
class CheckResult(metaclass=ABCMeta):
    code: CheckCode
    severity: Severity

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class OSDCheckResult(CheckResult):
    osd_id: int


@dataclass
class NodeCheckResult(CheckResult):
    hostname: str


@dataclass
class MonCheckResult(CheckResult):
    mon_name: str


class OSDDevice(Enum):
    data = 0
    journal = 1
    wal = 2
    db = 3


@dataclass
class WrongDiskType(OSDCheckResult):
    target: OSDDevice
    dev_name: str
    required_types: Set[str]
    found_type: str

    def __str__(self) -> str:
        return f"Dev {self.dev_name} used by osd {self.osd_id} as {self.target.name} has type {self.found_type}" + \
               f", but one of {', '.join(self.required_types)} recommended"


def check_osds_disk_types_and_sizes(ceph: CephInfo) -> List[CheckResult]:
    if ceph.osds_info is None:
        return []

    wrong_j_devices = []
    wrong_db_devices = []
    wrong_wal_devices = []
    wrong_j_size = []
    wrong_db_size = []
    wrong_wal_size = []

    for osd in ceph.osds:
        osd_info = ceph.osds_info.osds[osd.id]
        if osd_info.j_info:
            if osd_info.j_info.drive_type.is_fast():
                wrong_j_devices.append(osd.id)

            if osd.run_info:
                sync = int(osd.run_info.config['filestore_max_sync_interval'])
            else:
                sync = 5

            min_j_size = osd_info.data_drive_type.bandwith_mbps() * 2 * sync

            if osd_info.j_info.size / 2 ** 20 < min_j_size:
                wrong_j_size.append((osd.id, (osd_info.j_info.size, 2 ** 20 < min_j_size)))
        else:
            assert osd_info.wal_db_info
            if not osd_info.wal_db_info.wal_drive_type.is_fast():
                wrong_wal_devices.append(osd.id)

            min_db_size = osd_info.data_part_size * 0.02
            min_db_size = 0

            if not osd_info.wal_db_info.db_drive_type.is_fast():
                wrong_db_devices.append(osd.id)



def check_osds_networking():
    pass


def check_osds_weights():
    pass


def check_osds_evens_load():
    pass


def check_osds_ram_settings():
    pass


def check_osds_scrubbing_settings():
    pass


