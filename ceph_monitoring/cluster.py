import re
import copy
import json
import logging
import weakref
import collections
from ipaddress import IPv4Network, IPv4Address
from typing import Iterable, Iterator

from cephlib.crush import load_crushmap
from cephlib.units import unit_conversion_coef_f, ssize2b
from cephlib.storage import AttredStorage, TypedStorage

from .hw_info import parse_hw_info, get_dev_file_name
from .cluster_classes import *
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

        stor_tp = {
            ('sata', '1'): 'hdd',
            ('sata', '0'): 'ssd',
            ('nvme', '0'): 'nvme',
            ('sas',  '1'): 'sas hdd',
            ('sas',  '0'): 'ssd',
        }.get((tp, disk_js["rota"]))

        if stor_tp is None:
            logger.warning("Can't detect disk type for %r in node %r. tran=%r, rota=%r, subsystem=%r." +
                           "Treating it as hdd",
                           name, hostname, disk_js['tran'], disk_js['rota'], disk_js['subsystems'])
            stor_tp = 'hdd'

        dsk = Disk(name=name,
                   path='/dev/' + name,
                   size=int(disk_js['size']),
                   rota=disk_js['rota'] == '1',
                   scheduler=disk_js['sched'],
                   tp=stor_tp,
                   extra=disk_js,
                   mountpoint=disk_js['mountpoint'],
                   fs=disk_js["fstype"],
                   hw_model=HWModel(vendor=disk_js['vendor'], model=disk_js['model'], serial=disk_js['serial']),
                   rq_size=int(disk_js["rq-size"]),
                   phy_sec=int(disk_js["phy-sec"]),
                   min_io=int(disk_js["min-io"]))

        def fill_mountable(root: Dict[str, Any], parent: Any):
            if 'children' in root:
                for ch_js in root['children']:
                    assert '/' not in ch_js['name']
                    part = Mountable(name=ch_js['name'],
                                     path='/dev/' + ch_js['name'],
                                     size=int(ch_js['size']),
                                     mountpoint=ch_js['mountpoint'],
                                     fs=ch_js["fstype"],
                                     parent=weakref.ref(dsk),
                                     label=ch_js.get("partlabel"))
                    parent.children[part.path] = part
                    fill_mountable(ch_js, part)

        fill_mountable(disk_js, dsk)

        yield dsk


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
        for name, tp in PG.__annotations__.items():
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

# if code == 0:
#     # TODO: move this to report code, only collect here

#
# # TODO: move this to report code, only collect here
# if code == 0 and 'Bit Rate=' in out:
#     br1 = out.split('Bit Rate=')[1]
#     if 'Tx-Power=' in br1:
#         speed = br1.split('Tx-Power=')[0]
#
# # TODO: move this to report code, only collect here
# if speed is not None:
#     mults = {
#         'Kb/s': 125,
#         'Mb/s': 125000,
#         'Gb/s': 125000000,
#     }
#     for name, mult in mults.items():
#         if name in speed:  # type: ignore
#             speed = int(float(speed.replace(name, '')) * mult)    # type: ignore
#             break
#     else:
#         if speed == 'Unknown!':
#             unknown_speed.append(dev)
#         else:
#             logger.warning("Node %s - can't transform %s interface speed %r to Bps", self.node, dev, speed)
#
#     if isinstance(speed, int):
#         interface['speed'] = speed
#     else:
#         interface['speed_s'] = speed


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
        if dtime is not None and dtime > 1.0 and perf_monitoring:
            params: Dict[str, numpy.ndarray[int]] = {}
            load_node = perf_monitoring.get('net-io', {}).get(adapter_dct['dev'], {})
            for metric in 'send_bytes send_packets recv_packets recv_bytes'.split():
                params[metric] = numpy.array(load_node[metric])
            load: Optional[NetLoad] = NetLoad(**params)  # type: ignore
        else:
            load = None

        speed, speed_s, duplex = parse_adapter_info(adapter_dct)
        adapter = NetworkAdapter(dev=adapter_dct['dev'],
                                 is_phy=adapter_dct['is_phy'],
                                 load=load,
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
                adapter.ips.append(NetAdapterAddr(ip_addr, net_addr))
            except KeyError:
                logger.warning("Can't find adapter %r for ip %r", dev_name, ip_addr)

    return net_adapters


def load_hdds(hostname: str,
              jstor_node: AttredStorage,
              stor_node: AttredStorage) -> Tuple[Dict[str, Disk], Dict[str, Mountable], Dict[str, DFInfo]]:

    disks: Dict[str, Disk] = {}
    storage_devs: Dict[str, Mountable] = {}
    df_info = {info.name: info for info in parse_df(stor_node.df)}

    def walk_all(dsk):
        if dsk.name in df_info:
            dsk.free_space = df_info[dsk.name].free
        for ch in dsk.children.values():
            walk_all(ch)

    for dsk in parse_lsblkjs(jstor_node.lsblkjs['blockdevices'], hostname):
        disks[dsk.path] = dsk
        walk_all(dsk)
        storage_devs[dsk.path] = dsk  # type: ignore
        storage_devs.update(dsk.children)
    return disks, storage_devs, df_info


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


def load_PG_distribution(storage: TypedStorage, pg_dump: Optional[Dict[str, Any]]) \
        -> Tuple[Dict[int, Dict[str, int]], Dict[str, int], Dict[int, int]]:

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


def load_monitors(storage: TypedStorage, is_luminous: bool) -> List[CephMonitor]:
    if is_luminous:
        mons = storage.json.master.status['monmap']['mons']
        # mon_status = self.storage.json.master.mon_status
    else:
        srv_health = storage.json.master.status['health']['health']['health_services']
        assert len(srv_health) == 1
        mons = srv_health[0]['mons']

    vers = parse_ceph_versions(storage.txt.master.mon_versions)
    result: List[CephMonitor] = []
    for srv in mons:
        health = None if is_luminous else srv["health"]
        mon = CephMonitor(srv["name"], health, srv["name"], role=MonRole.unknown, version=vers["mon." + srv["name"]])

        if not is_luminous:
            mon.kb_avail = srv["kb_avail"]
            mon.avail_percent = srv["avail_percent"]
        result.append(mon)
    return result


def load_ceph_status(storage: TypedStorage, is_luminous: bool) -> CephStatus:
    mstorage = storage.json.master
    pgmap = mstorage.status['pgmap']
    health = mstorage.status['health']
    overall_status = health.get('overall_status', health['status'])
    return CephStatus(overall_status=CephStatusCode.from_str(overall_status),
                      status=CephStatusCode.from_str(health['status']),
                      health_summary=health['checks' if is_luminous else 'summary'],
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


def load_osd_perf_data(osd_id: int,
                       storage_type: OSDStoreType,
                       osd_perf_dump: Dict[int, List[Dict]]) -> Dict[str, numpy.ndarray]:
    osd_perf_dump = osd_perf_dump[osd_id]
    osd_perf: Dict[str,numpy.ndarray] = {}

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

        try:
            pg_dump = self.storage.json.master.pg_dump
        except AttributeError:
            pg_dump = None

        osd_pool_pg_2d, sum_per_pool, sum_per_osd = load_PG_distribution(self.storage, pg_dump)

        osd_map = self.load_osds(sum_per_osd, osd_perf_dump, osd_historic_ops_paths)

        pools = load_pools(self.storage, is_luminous)
        mons = load_monitors(self.storage, is_luminous)

        if pg_dump:
            pgs = parse_pg_dump(pg_dump)
            pgs_second = parse_pg_dump(self.storage.json.master.pg_dump_second)
        else:
            pgs = None
            pgs_second = None

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
                        settings=settings,
                        pgs=pgs,
                        pgs_second=pgs_second)

    def load_hosts(self) -> Tuple[Dict[str, Host], Dict[str, Host]]:
        hosts: Dict[str, Host] = {}
        tcp_sock_re = re.compile('(?im)^tcp6?\\b')
        udp_sock_re = re.compile('(?im)^udp6?\\b')
        ip2host: Dict[str, Host] = {}

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
            disks, storage_devs, df_info = load_hdds(hostname, jstor_node, stor_node)

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
                        hw_info=hw_info,
                        df_info=df_info)

            hosts[host.name] = host

            for adapter in net_adapters.values():
                for ip in adapter.ips:
                    ip_s = str(ip.ip)
                    if not ip_s.startswith('127.'):
                        if ip_s in ip2host:
                            logger.error("Ip %s belong to both %s and %s. Skipping new host",
                                         ip_s, hostname, ip2host[ip_s].name)
                            continue
                        ip2host[ip_s] = host

        return hosts, ip2host

    def load_osds(self,
                  sum_per_osd: Optional[Dict[int, int]],
                  osd_perf_dump: Optional[Dict[int, List[Dict]]],
                  osd_historic_ops_paths: Optional[Dict[int, str]]) -> Dict[int, CephOSD]:

        osd_rw_dict = dict((node['id'], node['reweight'])
                           for node in self.storage.json.master.osd_tree['nodes']
                           if node['id'] >= 0)

        try:
            fc = self.storage.txt.master.osd_versions
        except:
            fc = self.storage.raw.get_raw('master/osd_versions.err').decode("utf8")

        osd_versions: Dict[int, CephVersion] = {}
        for name, ver in parse_ceph_versions(fc).items():
            nm, id = name.split(".")
            assert nm == 'osd'
            osd_versions[int(id)] = ver

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

                cmdln = [i.decode("utf8")
                         for i in self.storage.raw.get_raw('osd/{0}/cmdline.bin'.format(osd.id)).split(b'\x00')]

                osd.run_info = OSDRunInfo(
                    config=config,
                    procinfo=self.storage.json.osd[str(osd.id)].procinfo,
                    cmdline=cmdln
                )

                if osd.storage_type == OSDStoreType.filestore:
                    stor_info: Union[FileStoreInfo, BlueStoreInfo] = FileStoreInfo(
                        data_dev=osd_disks_info['r_data'],
                        data_partition=osd_disks_info['data'],
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
                    stor_info.data_stor_stats = osd.host.disks[stor_info.data_dev].load

                    if osd_historic_ops_paths is not None:
                        osd.historic_ops_storage_path = osd_historic_ops_paths.get(osd.id)

                osd.storage_info = stor_info

        return osd_map


def load_all(storage: TypedStorage) -> Tuple[Cluster, CephInfo]:
    l = Loader(storage)
    cluster = l.load_cluster()
    ceph = l.load_ceph()
    ceph.osds_info = get_osds_info(cluster, ceph)
    return cluster, ceph


def get_all_versions(services: List[Union[CephOSD, CephMonitor]]) -> Dict[CephVersion, int]:
    all_version: Dict[CephVersion, int] = collections.Counter()
    for srv in services:
        all_version[srv.version] += 1
    return all_version


def mem2bytes(vl: str) -> int:
    vl_sz, units = vl.split()
    assert units == 'kB'
    return int(vl_sz) * 1024


def get_osds_info(cluster: Cluster, ceph: CephInfo) -> OSDSInfo:

    stor_set = set(osd.storage_type for osd in ceph.osds)
    has_bs = OSDStoreType.bluestore in stor_set
    has_fs = OSDStoreType.filestore in stor_set
    assert has_fs or has_bs

    all_vers = get_all_versions(ceph.osds)

    osd2pg_map = collections.defaultdict(list)
    if ceph.pgs:
        for pg in ceph.pgs.pgs.values():
            for osd_id in pg.acting:
                osd2pg_map[osd_id].append(pg)

    osds: Dict[int, OSDInfo] = {}
    for osd in ceph.osds:
        disks = cluster.hosts[osd.host.name].disks
        storage_devs = cluster.hosts[osd.host.name].storage_devs
        assert osd.storage_info is not None

        total_b = int(storage_devs[osd.storage_info.data_partition].size)

        #  CRUSH RULES/WEIGHTS
        crush_info: Dict[str, Tuple[bool, float, float]] = {}
        for root in ceph.crush.roots:
            nodes = ceph.crush.find_nodes([('root', root.name), ('osd', f"osd.{osd.id}")])
            if nodes:
                if len(nodes) == 1:
                    expected_weight = float(total_b) / (1024 ** 4)
                    crush_info[root.name] = [True, nodes[0].weight, expected_weight]
                else:
                    crush_info[root.name] = [False, -1.0, -1.0]

        #  RUN INFO - FD COUNT, TCP CONN, THREADS
        if osd.run_info is not None:
            pinfo = osd.run_info.procinfo
            opened_socks = sum(int(tp.get('inuse', 0)) for tp in pinfo.get('ipv4', {}).values())
            opened_socks += sum(int(tp.get('inuse', 0)) for tp in pinfo.get('ipv6', {}).values())

            sched = pinfo['sched']
            if not sched:
                data = "\n".join(pinfo['sched_raw'].strip().split("\n")[2:])
                sched = parse_proc_file(data, ignore_err=True)

            run_info = OSDProcessInfo(fd_count=pinfo["fd_count"],
                                      opened_socks=opened_socks,
                                      th_count=pinfo["th_count"],
                                      cpu_usage=float(sched["se.sum_exec_runtime"]),
                                      vm_rss=mem2bytes(pinfo["mem"]["VmRSS"]),
                                      vm_size=mem2bytes(pinfo["mem"]["VmSize"]))
        else:
            run_info = None

        #  USER DATA SIZE & TOTAL IO
        if ceph.pgs:
            pgs = osd2pg_map[osd.id]
            bytes = sum(pg.stat_sum.num_bytes for pg in pgs)
            reads = sum(pg.stat_sum.num_read for pg in pgs)
            read_b = sum(pg.stat_sum.num_read_kb * 1024 for pg in pgs)
            writes = sum(pg.stat_sum.num_write for pg in pgs)
            write_b = sum(pg.stat_sum.num_write_kb * 1024 for pg in pgs)
            scrub_err = sum(pg.stat_sum.num_scrub_errors for pg in pgs)
            deep_scrub_err = sum(pg.stat_sum.num_deep_scrub_errors for pg in pgs)
            shallow_scrub_err = sum(pg.stat_sum.num_shallow_scrub_errors for pg in pgs)

            pg_stats = OSDPGStats(bytes, reads, read_b, writes, write_b, scrub_err, deep_scrub_err, shallow_scrub_err)

            # USER DATA SIZE DELTA
            if ceph.pgs_second:
                smap = ceph.pgs_second.pgs
                d_bytes = sum(smap[pg.pgid.id].stat_sum.num_bytes for pg in pgs) - bytes
                d_reads = sum(smap[pg.pgid.id].stat_sum.num_read for pg in pgs) - reads
                d_read_b = sum(smap[pg.pgid.id].stat_sum.num_read_kb * 1024 for pg in pgs) - read_b
                d_writes = sum(smap[pg.pgid.id].stat_sum.num_write for pg in pgs) - writes
                d_write_b = sum(smap[pg.pgid.id].stat_sum.num_write_kb * 1024 for pg in pgs) - write_b

                dtime = (ceph.pgs_second.collected_at - ceph.pgs.collected_at).total_seconds()

                d_stats = OSDDStats(d_bytes // dtime,
                                    d_reads // dtime,
                                    d_read_b // dtime,
                                    d_writes // dtime,
                                    d_write_b // dtime)
            else:
                d_stats = None
        else:
            d_stats = None
            pg_stats = None

        # JOURNAL/WAL/DB info
        sinfo = osd.storage_info
        if isinstance(sinfo, FileStoreInfo):
            collocation = sinfo.journal_dev != sinfo.data_dev
            on_file = sinfo.journal_partition == sinfo.data_partition
            drive_type = disks[sinfo.journal_dev].tp
            size = None if collocation and on_file else storage_devs[sinfo.journal_partition].size
            fs_info = JInfo(collocation, on_file, drive_type, size)
            bs_info = None
        else:
            assert isinstance(sinfo, BlueStoreInfo)
            wal_collocation = sinfo.wal_dev != sinfo.data_dev
            wal_drive_type = disks[sinfo.wal_dev].tp

            if sinfo.wal_partition not in (sinfo.db_partition, sinfo.data_partition):
                wal_size = int(storage_devs[sinfo.wal_partition].size)
            else:
                wal_size = None

            db_collocation = sinfo.db_dev != sinfo.data_dev
            db_drive_type = disks[sinfo.db_dev].tp
            if sinfo.db_partition != sinfo.data_partition:
                db_size = int(storage_devs[sinfo.db_partition].size)
            else:
                db_size = None

            fs_info = None
            bs_info = WALDBInfo(wal_collocation=wal_collocation,
                                wal_drive_type=wal_drive_type,
                                wal_size=wal_size,
                                db_collocation=db_collocation,
                                db_drive_type=db_drive_type,
                                db_size=db_size)

        osds[osd.id] = OSDInfo(pgs=osd2pg_map[osd.id] if ceph.pgs else None,
                               total_space=total_b,
                               used_space=0,
                               free_space=0,
                               total_user_data=0,
                               crush_trees_weights=crush_info,
                               data_drive_type=disks[osd.storage_info.data_dev].tp,
                               data_part_size=storage_devs[osd.storage_info.data_partition].size,
                               j_info=fs_info,
                               wal_db_info=bs_info,
                               run_info=run_info,
                               pg_stats=pg_stats,
                               d_stats=d_stats)

    return OSDSInfo(has_bs=has_bs, has_fs=has_fs, all_versions=all_vers, largest_ver=max(all_vers),
                    osds=osds)


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
