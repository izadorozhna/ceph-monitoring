import re
import abc
import sys
import time
import json
import zlib
import array
import socket
import shutil
import logging
import os.path
import tempfile
import argparse
import datetime
import traceback
import contextlib
import subprocess
import logging.config
from concurrent.futures import ThreadPoolExecutor, Executor
from typing import Dict, Any, List, Tuple, Set, Callable, Optional, Union, Type

try:
    import logging_tree
except ImportError:
    logging_tree = None

from agent.agent import ConnectionClosed, SimpleRPCClient
from cephlib.storage import make_storage, IStorage
from cephlib.common import run_locally, get_sshable_hosts, tmpnam, setup_logging
from cephlib.rpc import init_node, rpc_run
from cephlib.discover import get_osds_nodes, get_mons_nodes
from cephlib.sensors_rpc_plugin import unpack_rpc_updates


logger = logging.getLogger('collect')


class Node:
    def __init__(self, ip: str, hostname: str = None) -> None:
        self.ip = ip
        self.hostname = hostname
        self.rpc = None
        self.daemon_pid = None   # type: int
        self.osds = []    # type: List[int]
        self.mon = None
        self.roles = set()  # type: Set[str]
        self.ssh_ok = False

    def dct(self) -> Dict[str, Any]:
        return {'ip': self.ip, 'name': self.hostname, 'roles': list(self.roles), 'ssh_ok': self.ssh_ok}

    @property
    def name(self) -> str:
        return self.ip if self.hostname is None else "{0.hostname}({0.ip})".format(self)

    @property
    def fs_name(self) -> str:
        if self.hostname is not None:
            return "{0.ip}-{0.hostname}".format(self)
        else:
            return self.ip


# CMD executors --------------------------------------------------------------------------------------------------------


def run_rpc_with_code(rpc: SimpleRPCClient, cmd: str, *args, **kwargs) -> Tuple[int, str]:
    try:
        code, out = 0, rpc_run(rpc, cmd, *args, **kwargs)
    except subprocess.CalledProcessError as exc:
        code, out = exc.returncode, (exc.output + exc.stderr)

    return code, out.decode('utf8')


def run_with_code(cmd: str) -> Tuple[int, str]:
    try:
        code, out = 0, run_locally(cmd)
    except subprocess.CalledProcessError as exc:
        code, out = exc.returncode, (exc.output + exc.stderr)

    return code, out.decode('utf8')


def get_host_interfaces(node: SimpleRPCClient) -> List[Tuple[bool, str]]:
    res = []  # type: List[Tuple[bool, str]]
    if node is None:
        content = run_locally("ls -l /sys/class/net").decode('utf8')
    else:
        content = rpc_run(node.rpc, "ls -l /sys/class/net", node_name=node.name).decode('utf8')

    for line in content.strip().split("\n")[1:]:
        if not line.startswith('l'):
            continue

        params = line.split()
        if len(params) < 11:
            continue

        res.append(('/devices/virtual/' not in params[10], params[8]))
    return res


def get_device_for_file(node: SimpleRPCClient, fname: str) -> Tuple[str, str]:
    dev = node.rpc.fs.get_dev_for_file(fname)
    dev = dev.decode('utf8')
    assert dev.startswith('/dev'), "{0!r} is not starts with /dev".format(dev)
    root_dev = dev = dev.strip()
    rr = re.match('^(/dev/[shv]d.*?)\\d+', root_dev)
    if rr:
        root_dev = rr.group(1)
    return root_dev, dev


class Collector:
    """Base class for data collectors. Can collect data for only one node."""
    name = None  # type: str
    collect_roles = []  # type: List[str]

    def __init__(self,
                 allowed_path: Callable[[str], bool],
                 storage: IStorage,
                 opts: Any,
                 node: SimpleRPCClient,
                 pretty_json: bool = False) -> None:
        self.allowed_path = allowed_path
        self.storage = storage
        self.opts = opts
        self.node = node
        self.pretty_json = pretty_json

    @contextlib.contextmanager
    def chdir(self, path: str):
        """Chdir for point in storage tree, where current results are stored"""
        saved = self.storage
        self.storage = self.storage.sub_storage(path)
        try:
            yield
        finally:
            self.storage = saved

    def save(self, path: str, frmt: str, code: int, data: Union[str, bytes, array.array],
             check: bool = True) -> Optional[str]:
        """Save results into storage"""
        if check and not self.allowed_path(path):
            logger.debug("Skipped saving %s bytes of data to path %r", len(data), path)
            return None

        if code == 0 and frmt == 'json':
            data = json.dumps(json.loads(data), indent=4, sort_keys=True)

        rpath = path + "." + ("err" if code != 0 else frmt)

        # TODO: fix storage
        if isinstance(data, str):
            self.storage.put_raw(data.encode('utf8'), rpath)
        elif isinstance(data, bytes):
            self.storage.put_raw(data, rpath)
        elif isinstance(data, array.array):
            self.storage.put_array(rpath, data, [])
        else:
            raise TypeError("Can't save value of type {0!r} (to {1!r})".format(type(data), rpath))

        return data if code == 0 else None

    def save_file(self, path: str, file_path: str, frmt: str = 'txt', check: bool = True) -> Optional[bytes]:
        """Download file from node and save it into storage"""
        if check and not self.allowed_path(path):
            logger.debug("Skipped file %r as result path %r is disabled", file_path, path)
            return None

        # make pylinter happy
        loc = None
        try:
            if self.node is None:
                loc = 'local'
                content = open(file_path, 'rb').read()
            else:
                loc = self.node.name
                content = zlib.decompress(self.node.rpc.fs.get_file(file_path))
            code = 0
        except (IOError, RuntimeError) as exc:
            logger.warning("Can't get file %r from node %s. %s", file_path, loc, exc)
            content = str(exc)
            code = 1

        self.save(path, frmt, code, content, check=False)
        return content if code == 0 else None

    def save_output(self, path: str, cmd: str, frmt: str = 'txt', check: bool = True) -> Optional[Tuple[int, str]]:
        """Run command on node and store result into storage"""
        if check and not self.allowed_path(path):
            logger.debug("Skipped cmd %r as result path %r is disabled", cmd, path)
            return None

        if self.node is None:
            code, out = run_with_code(cmd)
            loc = "locally"
        else:
            code, out = run_rpc_with_code(self.node.rpc, cmd, node_name=self.node.name)
            loc = "on node {0}".format(self.node.name)

        if code != 0:
            logger.warning("Cmd %s failed %s with code %s", cmd, loc, code)

        self.save(path, frmt, code, out, check=False)
        return code, out

    @abc.abstractmethod
    def collect(self, collect_roles_restriction: List[str]) -> None:
        """Do collect data,
        collect_roles_restriction is a list of allowed roles to be collected"""
        pass


def split_sockstat_file(fc: str) -> Optional[Dict[str, Dict[str, str]]]:
    res = {}  # type: Dict[str, Dict[str, str]]
    for ln in fc.split("\n"):
        if ln.strip():
            if ':' not in ln:
                return None
            name, params = ln.split(":", 1)
            params_l = params.split()
            if len(params_l) % 2 != 0:
                return None
            res[name] = dict(zip(params_l[:-1:2], params_l[1::2]))
    return res


def get_devices_tree(lsblkdct):
    def fall_down(node, root, res_dict):
        res_dict['/dev/' + node['name']] = root
        for ch_node in node.get('children', []):
            fall_down(ch_node, root, res_dict)

    res = {}
    for node in lsblkdct['blockdevices']:
        fall_down(node, '/dev/' + node['name'], res)
        res['/dev/' + node['name']] = '/dev/' + node['name']
    return res


class CephDataCollector(Collector):
    name = 'ceph'
    collect_roles = ['osd', 'mon', 'ceph-master']
    master_collected = False
    cluster_name = 'ceph'

    def __init__(self, *args, **kwargs) -> None:
        Collector.__init__(self, *args, **kwargs)
        opt = self.opts.ceph_extra + " " if self.opts.ceph_extra else ""
        self.ceph_cmd = "ceph {0}--format json ".format(opt)
        self.rados_cmd = "rados {0}--format json ".format(opt)

    def collect(self, collect_roles_restriction: List[str]) -> None:
        if 'mon' in collect_roles_restriction:
            self.collect_monitor()

        if 'osd' in collect_roles_restriction:
            self.collect_osd()

        if 'ceph-master' in collect_roles_restriction:
            # TODO: there a race condition in next two lines, but no reason to care about it
            assert not self.master_collected, "ceph-master role have to be collected only once"
            self.__class__.master_collected = True  # type: ignore
            self.collect_master()

    def collect_master(self) -> None:
        assert self.node is None, "Master data can only be collected from local node"

        with self.chdir("master"):
            curr_data = "{0}\n{1}\n{2}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                time.time())

            self.save("collected_at", 'txt', 0, curr_data)
            _, out = self.save_output("status", self.ceph_cmd + "status", 'json')  # type: ignore
            assert out, "{0!r} failed".format(self.ceph_cmd + "status")
            status = json.loads(out)

            ceph_health = {'status': status['health']['overall_status']}  # type: Dict[str, Union[str, int]]
            avail = status['pgmap']['bytes_avail']
            total = status['pgmap']['bytes_total']
            ceph_health['free_pc'] = int(avail * 100 / total + 0.5)

            active_clean = sum(pg_sum['count']
                               for pg_sum in status['pgmap']['pgs_by_state']
                               if pg_sum['state_name'] == "active+clean")
            total_pg = status['pgmap']['num_pgs']
            ceph_health["ac_perc"] = int(active_clean * 100 / total_pg + 0.5)
            ceph_health["blocked"] = "unknown"

            ceph_health_js = json.dumps(ceph_health)
            self.save("ceph_health_dict", "js", 0, ceph_health_js, check=False)

            cmds = ['osd tree',
                    'df',
                    'auth list',
                    'osd dump',
                    'health',
                    'mon_status',
                    'osd lspools',
                    'osd perf',
                    'health detail']

            num_pgs = status['pgmap']['num_pgs']
            if num_pgs > self.opts.max_pg_dump_count:
                logger.warning(
                    ("pg dump skipped, as num_pg ({0}) > max_pg_dump_count ({1})." +
                     " Use --max-pg-dump-count NUM option to change the limit").format(
                         num_pgs, self.opts.max_pg_dump_count))
            else:
                cmds.append('pg dump')

            for cmd in cmds:
                self.save_output(cmd.replace(" ", "_"), self.ceph_cmd + cmd, 'json')

            self.save_output("default_config", self.ceph_cmd + "--show-config", 'txt')
            self.save_output("osd_versions", self.ceph_cmd + "tell 'osd.*' version", "txt")
            self.save_output("mon_versions", self.ceph_cmd + "tell 'mon.*' version", "txt")
            self.save_output("rados_df", self.rados_cmd + "df", 'json')

            with tempfile.NamedTemporaryFile() as fd:
                code, data = run_with_code(self.ceph_cmd + "osd getcrushmap -o " + fd.name)
                if code == 0:
                    data = open(fd.name, "rb").read()

                self.save('crushmap', 'bin', code, data)
                with tempfile.NamedTemporaryFile() as fd_txt:
                    code, data = run_with_code("crushtool -d {0} -o {1}".format(fd.name, fd_txt.name))
                    if code == 0:
                        data = open(fd_txt.name, "rb").read()
                    self.save('crushmap', 'txt', code, data)

            with tempfile.NamedTemporaryFile() as fd:
                code, res = run_with_code(self.ceph_cmd + "osd getmap -o " + fd.name)
                if code == 0:
                    data = open(fd.name, "rb").read()
                self.save('osdmap', 'bin', code, data)
                self.save_output('osdmap', "osdmaptool --print {0}".format(fd.name), "txt")

    def collect_dev_info(self, root_dev: str, device: str) -> Optional[str]:
        dev_str = rpc_run(self.node.rpc, "df " + device, node_name=self.node.name).decode('utf8')
        dev_data = dev_str.strip().split("\n")[1].split()

        used = int(dev_data[2]) * 1024
        avail = int(dev_data[3]) * 1024

        rot_info_path = "/sys/block/{0}/queue/rotational".format(os.path.basename(root_dev))
        is_ssd = self.node.rpc.fs.get_file(rot_info_path, compress=False).decode('utf8').strip() == '0'

        if run_rpc_with_code(self.node.rpc, "which hdparm", node_name=self.node.name, log=False)[0]:
            logger.warning("hdparm is not installed on %s", self.node.name)
        else:
            self.save_output('hdparm', "sudo hdparm -I " + root_dev)

        if run_rpc_with_code(self.node.rpc, "which smartctl", node_name=self.node.name, log=False)[0]:
            logger.warning("smartctl is not installed on %s", self.node.name)
        else:
            self.save_output('smartctl', "sudo smartctl -a " + root_dev)

        val = {'dev': device, 'root_dev': root_dev, 'used': used, 'avail': avail, 'is_ssd': is_ssd}
        self.save('stats', 'json', 0, json.dumps(val))

        return root_dev

    def collect_osd(self) -> None:
        # check OSD process status
        out = rpc_run(self.node.rpc, "ps aux | grep ceph-osd", node_name=self.node.name).decode("utf8")
        osd_re = re.compile(r"ceph-osd[\t ]+.*(-i|--id)[\t ]+(\d+)")
        running_osds = set(int(rr.group(2)) for rr in osd_re.finditer(out))

        ids_from_ceph = set(osd.id for osd in self.node.osds)
        # expected_osds = running_osds.intersection(ids_from_ceph)
        # down_osds = ids_from_ceph - running_osds
        unexpected_osds = running_osds.difference(ids_from_ceph)

        for osd_id in unexpected_osds:
            logger.warning("Unexpected osd-{0} in node {1}.".format(osd_id, self.node.name))

        cephdisk_js = rpc_run(self.node.rpc, "ceph-disk list --format=json", node_name=self.node.name).decode('utf8')
        cephdisk_ls = rpc_run(self.node.rpc, "ceph-disk list", node_name=self.node.name).decode('utf8')
        lsblk_js = rpc_run(self.node.rpc, "lsblk -a --json", node_name=self.node.name).decode('utf8')
        cephdisk_dct = json.loads(cephdisk_js)
        dev_tree = get_devices_tree(json.loads(lsblk_js))

        with self.chdir('hosts/' + self.node.fs_name):
            self.save("cephdisk", 'txt', 0, cephdisk_ls)
            self.save("cephdisk", 'json', 0, cephdisk_js)

        devs_for_osd = {}  # type: Dict[int, Tuple[str, str, Optional[str]]]
        for dev_info in cephdisk_dct:
            for part_info in dev_info.get('partitions', []):
                if part_info.get('type') == 'data':
                    devs_for_osd[int(part_info['whoami'])] = (part_info['path'], part_info['journal_dev'], None)

        for osd in self.node.osds:
            with self.chdir('osd/{0}'.format(osd.id)):
                cmd = "tail -n {0} /var/log/ceph/ceph-osd.{1}.log".format(self.opts.ceph_log_max_lines, osd.id)
                self.save_output("log", cmd)

                if osd.id in running_osds:
                    self.save("config", "txt", 0, osd.config)
                    cmd = "ls -1 '{0}'".format(os.path.join(osd.storage, 'current'))
                    code, res = run_rpc_with_code(self.node.rpc, cmd)

                    if code == 0:
                        pgs = [name.split("_")[0] for name in res.split() if "_head" in name]
                        res = "\n".join(pgs)

                    self.save("pgs", "txt", code, res)

                    data_dev, j_dev, _ = devs_for_osd[osd.id]
                    r_data_dev = dev_tree[data_dev]
                    r_j_dev = dev_tree[j_dev]

                    osd_dev_conf = {'data': data_dev,
                                    'r_data': r_data_dev,
                                    'journal': j_dev,
                                    'r_journal': r_j_dev}

                    self.save('devs_cfg', 'json', 0, json.dumps(osd_dev_conf))

                    with self.chdir("data"):
                        self.collect_dev_info(r_data_dev, data_dev)

                    if r_data_dev != r_j_dev:
                        with self.chdir("journal"):
                            self.collect_dev_info(r_j_dev, j_dev)
                else:
                    logger.warning("osd-{0} in node {1} is down.".format(osd.id, self.node.name) +
                                   " No config available, will use default data and journal path")

        pids = self.node.rpc.sensors.find_pids_for_cmd('ceph-osd')
        logger.debug("Found next pids for OSD's on node %s: %r", self.node.name, pids)
        if pids:
            for pid in pids:
                cmdline = self.node.rpc.fs.get_file('/proc/{0}/cmdline'.format(pid), compress=False).decode('utf8')
                opts = cmdline.split("\x00")
                for op, val in zip(opts[:-1], opts[1:]):
                    if op in ('-i', '--id'):
                        osd_id = val
                        break
                else:
                    logger.warning("Can't get osd id for cmd line %r", cmdline.replace("\x00", ' '))
                    continue

                osd_proc_info = {}  # type: Dict[str, Any]

                with self.chdir('osd/{0}'.format(osd_id)):
                    self.save("cmdline", "bin", 0, cmdline)
                    osd_proc_info['fd_count'] = len(self.node.rpc.fs.listdir("/proc/{0}/fd".format(pid)))

                    fpath = "/proc/{0}/net/sockstat".format(pid)
                    ssv4 = split_sockstat_file(self.node.rpc.fs.get_file(fpath, compress=False).decode('utf8'))
                    if not ssv4:
                        logger.warning("Broken file {0!r} on node {1}".format(fpath, self.node.name))
                    else:
                        osd_proc_info['ipv4'] = ssv4

                    fpath = "/proc/{0}/net/sockstat6".format(pid)
                    ssv6 = split_sockstat_file(self.node.rpc.fs.get_file(fpath, compress=False).decode('utf8'))
                    if not ssv6:
                        logger.warning("Broken file {0!r} on node {1}".format(fpath, self.node.name))
                    else:
                        osd_proc_info['ipv6'] = ssv6

                    proc_stat = self.node.rpc.fs.get_file("/proc/{0}/status".format(pid), compress=False).decode('utf8')
                    self.save("proc_status", "txt", 0, proc_stat)
                    osd_proc_info['th_count'] = int(proc_stat.split('Threads:')[1].split()[0])

                    self.save("procinfo", "json", 0, json.dumps(osd_proc_info))

    def collect_monitor(self) -> None:
        with self.chdir("mon/{0}".format(self.node.mon)):
            self.save_output("mon_daemons", "ps aux | grep ceph-mon")
            tail_ln = "tail -n {0} ".format(self.opts.ceph_log_max_lines)
            self.save_output("mon_log", tail_ln + "/var/log/ceph/ceph-mon.{0}.log".format(self.node.mon))
            self.save_output("ceph_log", tail_ln + " /var/log/ceph/ceph.log")
            self.save_output("ceph_audit", tail_ln + " /var/log/ceph/ceph.audit.log")
            self.save_output("config", self.ceph_cmd + "daemon mon.{0} config show".format(self.node.mon),
                             frmt='json')


class NodeCollector(Collector):
    name = 'node'
    collect_roles = ['node']

    node_commands = [
        ("lshw", "xml", "lshw -xml"),
        ("lsblk", "txt", "lsblk -a"),
        ("lsblkjs", "js", "lsblk -a --json"),
        ("uname", "txt", "uname -a"),
        ("dmidecode", "txt", "dmidecode"),
        ("mount", "txt", "mount"),
        ("ipa", "txt", "ip -o -4 a"),
        ("netstat", "txt", "netstat -nap"),
        ("sysctl", "txt", "sysctl -a"),
        ("df", 'txt', 'df'),
    ]

    node_files = [
        "/proc/diskstats", "/proc/meminfo", "/proc/loadavg", "/proc/cpuinfo", "/proc/uptime", "/var/log/dmesg",
        "/proc/vmstat"
    ]

    node_renamed_files = [("netdev", "/proc/net/dev"),
                          ("dev_netstat", "/proc/net/netstat"),
                          ("ceph_conf", "/etc/ceph/ceph.conf")]

    def collect(self, collect_roles_restriction: List[str]) -> None:
        if 'node' not in collect_roles_restriction:
            return

        with self.chdir('hosts/' + self.node.fs_name):
            for path_offset, frmt, cmd in self.node_commands:
                try:
                    self.save_output(path_offset, cmd, frmt=frmt)
                except Exception as exc:
                    logger.warning("Failed to run %r on node %s: %s", cmd, self.node.name, exc)

            for fpath in self.node_files:
                try:
                    self.save_file(os.path.basename(fpath), fpath)
                except Exception as exc:
                    logger.warning("Failed to download file %r from node %s: %s", fpath, self.node.name, exc)

            for name, fpath in self.node_renamed_files:
                try:
                    self.save_file(name, fpath)
                except Exception as exc:
                    logger.warning("Failed to download file %r from node %s: %s", fpath, self.node.name, exc)

            self.collect_interfaces_info()

            try:
                if self.node.rpc.fs.file_exists("/etc/debian_version"):
                    self.save_output("packages_deb", "dpkg -l", frmt="txt")
                else:
                    self.save_output("packages_rpm", "yum list installed", frmt="txt")
            except Exception as exc:
                logger.warning("Failed to download packages information from node %s: %s", self.node.name, exc)

    def collect_interfaces_info(self) -> None:
        interfaces = {}
        for is_phy, dev in get_host_interfaces(self.node):
            interface = {'dev': dev, 'is_phy': is_phy}
            interfaces[dev] = interface

            if not is_phy:
                continue

            speed = None
            code, out = run_rpc_with_code(self.node.rpc, "ethtool " + dev, node_name=self.node.name)
            if code == 0:
                # TODO: move this to report code, only collect here
                for line in out.split("\n"):
                    if 'Speed:' in line:
                        speed = line.split(":")[1].strip()
                    if 'Duplex:' in line:
                        interface['duplex'] = line.split(":")[1].strip() == 'Full'

            code, out = run_rpc_with_code(self.node.rpc, "iwconfig " + dev, node_name=self.node.name)
            # TODO: move this to report code, only collect here
            if code == 0 and 'Bit Rate=' in out:
                br1 = out.split('Bit Rate=')[1]
                if 'Tx-Power=' in br1:
                    speed = br1.split('Tx-Power=')[0]

            # TODO: move this to report code, only collect here
            if speed is not None:
                mults = {
                    'Kb/s': 125,
                    'Mb/s': 125000,
                    'Gb/s': 125000000,
                }
                for name, mult in mults.items():
                    if name in speed:
                        speed = int(float(speed.replace(name, '')) * mult)
                        break
                else:
                    logger.warning("Node %s - can't transform %s interface speed %r to Bps",
                                   self.node.name, dev, speed)

                if isinstance(speed, int):
                    interface['speed'] = speed
                else:
                    interface['speed_s'] = speed

        self.save('interfaces', 'json', 0, json.dumps(interfaces))


class LoadCollector(Collector):
    name = 'load'
    collect_roles = ['load']

    def collect(self, collect_roles_restriction: List[str]) -> None:
        raise RuntimeError("collect should not be called for {0} class instance".format(self.__class__.__name__))

    def start_performance_monitoring(self) -> None:
        cfg = {
            "block-io": {},
            "net-io": {},
            "vm-io": {},
            "ceph": {"sources": ['historic', 'historic_js', 'perf_dump'], "osds": "all"}
        }

        # self.results = {}

        #     "perprocess-cpu": {"allowed": ".*"},
        #     "perprocess-ram": {"allowed": ".*"},
        #     "system-cpu": {"allowed": ".*"},
        #     "system-ram": {"allowed": ".*"},
        #     "ceph": {"osds": []},

        self.node.rpc.sensors.start(cfg)

    def collect_performance_data(self) -> None:
        for sensor_path, data, is_unpacked, _ in unpack_rpc_updates(self.node.rpc.sensors.get_updates()):
            if is_unpacked:
                ext = 'arr' if isinstance(data, array.array) else 'json'
                self.save("perf_monitoring/{0}/{1}".format(self.node.fs_name, sensor_path), ext, 0, data)
            else:
                sensor, dev, metric = sensor_path.split(".")
                if metric == 'historic':
                    self.storage.put_raw(data, "perf_monitoring/{0}/{1}.bin".format(self.node.fs_name, sensor_path))
                elif metric == 'perf_dump':
                    self.storage.put_raw(data, "perf_monitoring/{0}/{1}.json".format(self.node.fs_name, sensor_path))
                else:
                    assert metric == 'historic_js', sensor_path
                    self.storage.put_raw(data, "perf_monitoring/{0}/{1}.json".format(self.node.fs_name, sensor_path))


def discover_nodes(opts: Any) -> List[Node]:
    nodes = {}

    for mon_id, (ip, name) in get_mons_nodes(lambda x: run_locally(x).decode('utf8'), opts.ceph_extra).items():
        node = Node(ip)
        node.mon = name
        node.roles.add('mon')
        nodes[ip] = node

    for ip, osds in get_osds_nodes(lambda x: run_locally(x).decode('utf8'), opts.ceph_extra).items():
        node = nodes.setdefault(ip, Node(ip))
        node.osds = osds
        node.roles.add('osd')

    return list(nodes.values())


ALL_COLLECTORS = [CephDataCollector, NodeCollector, LoadCollector]  # type: List[Type[Collector]]
ALL_COLLECTORS_MAP = dict((collector.name, collector)
                          for collector in ALL_COLLECTORS)  # type: Dict[str, Type[Collector]]


def get_allowed_paths_checker(disabled_patterns: Optional[List[str]]) -> Callable[[str], bool]:
    if not disabled_patterns:
        return lambda path: True
    disabled = [re.compile(pattern) for pattern in disabled_patterns]
    return lambda path: not any(pattern.search(path) for pattern in disabled)


def set_node_name(node: SimpleRPCClient) -> None:
    if node.hostname is None:
        node.hostname = rpc_run(node.rpc, "hostname", node_name=node.name).decode('utf8').strip()
    logger.debug("%s -> %s", node.ip, node.hostname)


def collect(storage: IStorage, opts: Any, executor: Executor) -> None:
    if run_with_code('which ceph')[0] != 0:
        logger.error("No 'ceph' command available. Run this script from node, which has ceph access")
        return

    # This variable is updated from main function
    ssh_opts = "-o LogLevel=quiet -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    ssh_opts += "-o ConnectTimeout={0} -o ConnectionAttempts=2".format(opts.ssh_conn_timeout)

    if opts.ssh_user:
        ssh_opts += " -l {0} ".format(opts.ssh_user)

    if opts.ssh_opts:
        ssh_opts += opts.ssh_opts + " "

    allowed_path_checker = get_allowed_paths_checker(opts.disable)
    allowed_collectors = opts.collectors.split(',')

    nodes = []
    explicit_nodes = open(opts.inventory).read().split() if opts.inventory else []
    for node in explicit_nodes + opts.node:
        if not node:
            pass

        if not re.match(r"\d+\.\d+\.\d+\.\d+", node):
            ip = socket.gethostbyname(node)
            hostname = node
        else:
            ip = node
            hostname = None

        nodes.append(Node(ip, hostname))

    if not opts.no_detect:
        detect_nodes = discover_nodes(opts)
        all_nodes_dct = dict((node.ip, node) for node in detect_nodes)
    else:
        all_nodes_dct = {}

    for node in nodes:
        if node.ip not in all_nodes_dct:
            all_nodes_dct[node.ip] = node

    ips, nodes = zip(*all_nodes_dct.items())

    logger.info("Found %s hosts in total", len(nodes))

    good_hosts = set(get_sshable_hosts(ips, ssh_opts))
    bad_hosts = set(ips) - good_hosts

    if len(bad_hosts) != 0:
        logger.warning("Next hosts aren't available over ssh and would be skipped: %s", ", ".join(bad_hosts))

    for node in nodes:
        node.ssh_ok = node.ip not in bad_hosts
        descr = ("" if node.ssh_ok else "OFFLINE ") + node.name
        if node.mon:
            descr += " mon=" + node.mon
        if node.osds:
            descr += " osds=[" + ",".join(str(osd.id) for osd in node.osds) + "]"
        logger.debug("Node: %s", descr)

    if opts.detect_only:
        return

    storage.put_raw(json.dumps([node.dct() for node in nodes]).encode('utf8'), "hosts.json")

    logger.info("Initializing RPC connections")
    nodes = [node for node in nodes if node.ip in good_hosts]
    nodes_map = dict((node.ip, node) for node in nodes if node.ip in good_hosts)

    try:
        rpc_nodes = []

        def init_node_with_code(node):
            try:
                return True, init_node(node, ssh_opts=ssh_opts)
            except Exception:
                return False, (None, None)

        rpc_res = executor.map(init_node_with_code, good_hosts)  # type: ignore
        for (ok, (rpc_conn, _)), ip in zip(rpc_res, good_hosts):
            if ok:
                nodes_map[ip].rpc = rpc_conn
                rpc_nodes.append(nodes_map[ip])
            else:
                logger.error("Failed to setup RPC connection with %s - %r", ip, rpc_conn)

        nodes = rpc_nodes
        logger.info("RPC connected")
        logger.info("Getting node names")

        for future in [executor.submit(set_node_name, node) for node in nodes]:
            future.result()

        load_collectors = []

        collect_load_data_at = 0  # make pylint happy
        if LoadCollector.name in allowed_collectors:
            futures = []
            for node in nodes:
                collector = LoadCollector(allowed_path_checker, storage, opts, node,
                                          pretty_json=not opts.no_pretty_json)
                futures.append(executor.submit(collector.start_performance_monitoring))
                load_collectors.append(collector)
            logger.info("Start performance collectors")

            for future in futures:
                future.result()

            # time when to stop load collection
            collect_load_data_at = time.time() + opts.load_collect_seconds

        futures = []
        if CephDataCollector.name in allowed_collectors:
            for node in nodes:
                func = CephDataCollector(allowed_path_checker, storage, opts, node,
                                         pretty_json=not opts.no_pretty_json).collect
                roles = ["mon"] if node.mon else []
                if node.osds:
                    roles.append("osd")
                futures.append(executor.submit(func, roles))
            func = CephDataCollector(allowed_path_checker, storage, opts, None,
                                     pretty_json=not opts.no_pretty_json).collect
            futures.append(executor.submit(func, ('ceph-master',)))

        if futures:
            for future in futures:
                future.result()

        futures = []
        for collector_name in allowed_collectors:
            if collector_name not in (LoadCollector.name, CephDataCollector.name):
                for node in nodes:
                    collector_cls = ALL_COLLECTORS_MAP[collector_name]
                    func = collector_cls(allowed_path_checker, storage, opts, node,
                                         pretty_json=not opts.no_pretty_json).collect
                    futures.append(executor.submit(func, ('node',)))

        if futures:
            for future in futures:
                future.result()

        if LoadCollector.name in allowed_collectors:
            stime = collect_load_data_at - time.time()
            if stime > 0:
                logger.info("Waiting for %s seconds for performance collectors", int(stime + 0.5))
                time.sleep(stime)

            logger.info("Collecting performance info")
            for future in [executor.submit(collector.collect_performance_data) for collector in load_collectors]:
                future.result()

    finally:
        logger.info("Collecting logs and teardown RPC servers")
        for node in nodes:
            if node.rpc is not None:
                try:
                    storage.put_raw(node.rpc.server.get_logs().encode('utf8'), "rpc_logs/{0}.txt".format(node.fs_name))
                except Exception:
                    pass
                try:
                    node.rpc.server.stop()
                except ConnectionClosed:
                    node.rpc = None


def parse_args(argv: List[str]) -> Any:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--collectors", default="ceph,node,load",
                   help="Comma separated list of collectors. Select from : " +
                   ",".join(coll.name for coll in ALL_COLLECTORS))
    p.add_argument("-C", "--ceph-extra", default="", help="Extra opts to pass to 'ceph' command")
    p.add_argument("-d", "--disable", default=[], nargs='*', help="Disable collect pattern")
    p.add_argument("-D", "--detect-only", action="store_true", help="Don't collect any data, only detect cluster nodes")
    p.add_argument("--no-detect", action="store_true", help="Don't detect nodes. Inventory must be provided")
    p.add_argument("-g", "--save-to-git", metavar="DIR", help="Commit into git repo, cloned to folder")
    p.add_argument("-G", "--git-push", metavar="DIR", help="Commit into git repo, cloned to folder and run 'git push'")
    p.add_argument("-i", "--inventory", metavar="FILE", help="File with node names/addrs")
    p.add_argument("-I", "--node", nargs='+', default=[], help="pass additional nodes from cli")
    p.add_argument("-j", "--no-pretty-json", action="store_true", help="Don't prettify json data")
    p.add_argument("-l", "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default=None, help="Console log level")
    p.add_argument("-L", "--log-config", help="json file with logging config")
    p.add_argument("-m", "--max-pg-dump-count", default=2 ** 15, type=int,
                   help="maximum PG count to by dumped with 'pg dump' cmd")
    p.add_argument("-M", "--ceph-log-max-lines", default=1000, type=int, help="Max lines from osd/mon log")
    p.add_argument("-n", "--dont-remove-unpacked", action="store_true", help="Keep unpacked data")
    p.add_argument("-N", "--dont-pack-result", action="store_true", help="Don't create archive")
    p.add_argument("-o", "--result", default=None, help="Result file")
    p.add_argument("-O", "--output-folder", default=None, help="Result folder")
    p.add_argument("-p", "--pool-size", default=32, type=int, help="RPC/local worker pool size")
    p.add_argument("-P", "--ssh-pool-size", default=4, type=int, help="SSH worker pool size")
    p.add_argument("-s", "--load-collect-seconds", default=60, type=int, metavar="SEC",
                   help="Collect performance stats for SEC seconds")
    p.add_argument("-S", "--ssh-opts", default=None, help="SSH cli options")
    p.add_argument("-t", "--ssh-conn-timeout", default=60, type=int, help="SSH connection timeout")
    p.add_argument("-u", "--ssh-user", default=None, help="SSH user. Current user by default")
    p.add_argument("-w", "--wipe", action='store_true', help="Wipe results directory before store data")

    if logging_tree:
        p.add_argument("-T", "--show-log-tree", action="store_true", help="Show logger tree.")

    return p.parse_args(argv[1:])


def setup_logging2(opts: Any, out_folder: str, tmp_log_file: bool = False) -> Optional[str]:
    default_lconf_path = os.path.join(os.path.dirname(__file__), 'logging.json')
    log_config_fname = default_lconf_path if opts.log_config is None else opts.log_config

    if opts.detect_only:
        log_file = '/dev/null'
    else:
        if tmp_log_file:
            fd, log_file = tempfile.mkstemp()
            os.close(fd)
        else:
            log_file = os.path.join(out_folder, "log.txt")

    setup_logging(log_config_fname, log_file=log_file, log_level=opts.log_level)

    return log_file if log_file != '/dev/null' else None


def git_prepare(path: str) -> bool:
    logger.info("Checking and cleaning git dir %r", path)
    code, res = run_with_code("cd {0} ; git status --porcelain".format(path))

    if code != 0:
        sys.stderr.write("Folder {0} doesn't looks like under git control\n".format(path))
        return False

    if len(res.strip()) != 0:
        sys.stderr.write("Uncommited or untracked files in {0}. ".format(path) +
                         "Cleanup directory before proceed\n")
        return False

    for name in os.listdir(path):
        if not name.startswith('.'):
            objpath = os.path.join(path, name)
            if os.path.isdir(objpath):
                shutil.rmtree(objpath)
            else:
                os.unlink(objpath)

    return True


def git_commit(path: str, message: str, push: bool = False):
    cmd = "cd {0} ; git add -A ; git commit -m '{1}'".format(path, message)
    if push:
        cmd += " ; git push"
    run_locally(cmd)


def main(argv: List[str]) -> int:
    out_folder = None
    log_file = None

    try:
        opts = parse_args(argv)

        if opts.git_push and opts.save_to_git:
            sys.stderr.write("At most one of -g/--save-to-git and -G/--git-push option can be provided\n")
            return 1

        git_dir = opts.git_push if opts.git_push else opts.save_to_git

        if git_dir and opts.output_folder:
            sys.stderr.write("--output-folder can't be used with -g/-G option\n")
            return 1

        if not opts.detect_only:
            if git_dir:
                if not git_prepare(git_dir):
                    return 1
                out_folder = git_dir
            elif opts.output_folder:
                out_folder = opts.output_folder
                if os.path.exists(out_folder):
                    if opts.wipe:
                        shutil.rmtree(out_folder)
                        os.mkdir(out_folder)
                else:
                    os.mkdir(out_folder)
            else:
                out_folder = tempfile.mkdtemp()

        log_file = setup_logging2(opts, out_folder, git_dir is not None)

        if logging_tree and opts.show_log_tree:
            logging_tree.printout()
            return 0

    except Exception:
        if log_file:
            with open(log_file, 'wt') as fd:
                fd.write(traceback.format_exc())
        raise

    try:
        if out_folder:
            if opts.output_folder:
                logger.info("Store data into %r", out_folder)
            else:
                logger.info("Temporary folder %r", out_folder)
            storage = make_storage(out_folder, existing=False, serializer='raw')
        else:
            storage = None

        with ThreadPoolExecutor(opts.pool_size) as executor:
            collect(storage, opts, executor)

        if out_folder:
            if git_dir:
                [h_weak_ref().flush() for h_weak_ref in logging._handlerList]  # type: ignore
                target_log_file = os.path.join(git_dir, "log.txt")
                shutil.copy(log_file, target_log_file)
            else:
                target_log_file = None

            if not opts.dont_pack_result:
                if opts.result is None:
                    out_file = tmpnam(remove_after=False) + ".tar.gz"
                else:
                    out_file = opts.result

                code, res = run_with_code("cd {0} ; tar -zcvf {1} *".format(out_folder, out_file))
                if code != 0:
                    logger.error("Fail to archive results. Please found raw data at %r", out_folder)
                else:
                    logger.info("Result saved into %r", out_file)

                    if opts.log_level in ('WARNING', 'ERROR', "CRITICAL"):
                        print("Result saved into %r" % (out_file,))

            if git_dir:
                logger.info("Comiting into git")

                if git_dir:
                    [h_weak_ref().flush() for h_weak_ref in logging._handlerList]  # type: ignore
                    shutil.copy(log_file, target_log_file)
                    os.unlink(log_file)

                status_templ = "{0:%H:%M %d %b %y}, {1[status]}, {1[free_pc]}% free, {1[blocked]} req blocked, " + \
                               "{1[ac_perc]}% PG active+clean"
                ceph_health = json.loads(storage.get_raw("ceph_health_dict").decode("utf8"))
                message = status_templ.format(datetime.datetime.now(), ceph_health)
                git_commit(git_dir, message, opts.git_push is not None)
            elif not opts.dont_remove_unpacked:
                if opts.dont_pack_result:
                    logger.warning("Unpacked tree is kept as --dont-pack-result option is set, so no archive created")
                else:
                    shutil.rmtree(out_folder)
                    out_folder = None

            if out_folder:
                logger.info("Temporary folder %r", out_folder)
                if opts.log_level in ('WARNING', 'ERROR', "CRITICAL"):
                    print("Temporary folder %r" % (out_folder,))
    except Exception:
        logger.exception("During make_storage/collect")
        raise

    return 0


if __name__ == "__main__":
    exit(main(sys.argv))
