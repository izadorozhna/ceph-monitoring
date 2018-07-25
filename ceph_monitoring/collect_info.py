import re
import abc
import sys
import time
import json
import zlib
import array
import socket
import random
import shutil
import logging
import os.path
import tempfile
import argparse
import datetime
import ipaddress
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
from cephlib.storage import make_storage, IStorageNNP
from cephlib.common import run_locally, get_sshable_hosts, tmpnam, setup_logging
from cephlib.rpc import init_node, rpc_run
from cephlib.discover import get_osds_nodes, get_mons_nodes
from cephlib.sensors_rpc_plugin import unpack_rpc_updates


logger = logging.getLogger('collect')


class Node:
    def __init__(self, ip: str, hostname: str = None, roles: Set[str] = None) -> None:
        self.ip = ip
        self.hostname = hostname
        self.rpc = None
        self.daemon_pid = None   # type: int
        self.osds = []    # type: List[int]
        self.mon = None
        self.roles = roles if roles else set() # type: Set[str]
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
        code, out = exc.returncode, (exc.output + (b"" if exc.stderr is None else exc.stderr))

    return code, out.decode('utf8')


def run_with_code(cmd: str) -> Tuple[int, str]:
    try:
        code, out = 0, run_locally(cmd)
    except subprocess.CalledProcessError as exc:
        code, out = exc.returncode, (exc.output + (b"" if exc.stderr is None else exc.stderr))

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
                 storage: IStorageNNP,
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

    def run(self, cmd):
        if self.node is None:
            return run_with_code(cmd)
        else:
            return run_rpc_with_code(self.node.rpc, cmd, node_name=self.node.name)

    def save(self, path: str, frmt: str, code: int, data: Union[str, bytes, array.array],
             check: bool = True, extra: List[str] = None) -> Optional[str]:
        """Save results into storage"""
        if check and not self.allowed_path(path):
            logger.debug("Skipped saving %s bytes of data to path %r", len(data), path)
            return None

        if code == 0 and frmt == 'json':
            data = json.dumps(json.loads(data), indent=4, sort_keys=True)

        rpath = path + "." + ("err" if code != 0 else frmt)

        # TODO: fix storage
        if isinstance(data, str):
            assert extra is None
            self.storage.put_raw(data.encode('utf8'), rpath)
        elif isinstance(data, bytes):
            assert extra is None
            self.storage.put_raw(data, rpath)
        elif isinstance(data, array.array):
            self.storage.put_array(rpath, data, extra if extra else [])
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
            code, out = self.run(cmd)
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
        self.radosgw_admin_cmd = "radosgw-admin {0} ".format(opt)
        self.ceph_cmd = "ceph {0}--format json ".format(opt)
        self.ceph_cmd_txt = "ceph {0} ".format(opt)
        self.rados_cmd = "rados {0}--format json ".format(opt)
        self.rados_cmd_txt = "rados {0} ".format(opt)

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
        # assert self.node is None, "Master data can only be collected from local node"

        with self.chdir("master"):
            curr_data = "{0}\n{1}\n{2}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                time.time())

            code, out = self.run(self.ceph_cmd + "osd versions")
            is_luminous = (code == 0)
            if is_luminous:
                self.save("osd_versions2", 'txt', 0, out)
                self.save_output("mon_versions2", self.ceph_cmd + "mon versions", "txt")
                self.save_output("mgr_versions2", self.ceph_cmd + "mgr versions", "txt")

            self.save_output("osd_versions", self.ceph_cmd + "tell 'osd.*' version", "txt")
            self.save_output("mon_versions", self.ceph_cmd + "tell 'mon.*' version", "txt")

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
                    'osd df',
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
            self.save_output("rados_df", self.rados_cmd + "df", 'json')
            self.save_output("rados_df", self.rados_cmd_txt + "df", 'txt')
            self.save_output("ceph_s", self.ceph_cmd_txt + "-s", 'txt')
            self.save_output("ceph_osd_dump", self.ceph_cmd_txt + "osd dump", 'txt')
            self.save_output("time_sync_status", self.ceph_cmd_txt + 'time-sync-status', 'json')
            self.save_output("report", self.ceph_cmd_txt + 'report', 'json')
            self.save_output("osd_utilization", self.ceph_cmd_txt + 'osd utilization', 'txt')
            self.save_output("osd_blocked_by", self.ceph_cmd_txt + "osd blocked_by", "txt")
            self.save_output("node_ls", self.ceph_cmd_txt + "node ls", "json")
            self.save_output("features", self.ceph_cmd_txt + "features", 'json')
            self.save_output("realm_list", self.radosgw_admin_cmd + "realm list", "txt")
            self.save_output("zonegroup_list", self.radosgw_admin_cmd + "zonegroup list", "txt")
            self.save_output("zone_list", self.radosgw_admin_cmd + "zone list", "txt")

            if self.node is None:
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
            else:
                temp_fl = "%08X" % random.randint(0, 2 << 64)
                cr_fname = "/tmp/ceph_collect." + temp_fl + ".cr"
                code, _ = self.run(self.ceph_cmd + "osd getcrushmap -o " + cr_fname)
                if code != 0:
                    logger.error("Fail to get crushmap")
                else:
                    self.save_file('crushmap', cr_fname, 'bin')
                    code, data = self.run("crushtool -d {0} -o {0}.txt".format(cr_fname))
                    if code != 0:
                        logger.error("Fail to decompile crushmap")
                    else:
                        self.save_file('crushmap', cr_fname + ".txt", 'txt')

                osd_fname = "/tmp/ceph_collect." + temp_fl + ".osd"
                code, _ = self.run(self.ceph_cmd + "osd getmap -o " + osd_fname)
                if code != 0:
                    logger.error("Fail to get osdmap")
                else:
                    self.save_file('osdmap', osd_fname, 'bin')
                    self.save_output('osdmap', "osdmaptool --print " + osd_fname, "txt")

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

        devs_for_osd = {}  # type: Dict[int, Dict[str, str]]
        for dev_info in cephdisk_dct:
            for part_info in dev_info.get('partitions', []):
                if "cluster" in part_info and part_info.get('type') == 'data':
                    osd_id = int(part_info['whoami'])
                    devs_for_osd[osd_id] = {attr: part_info[attr]
                                            for attr in ("block_dev", "journal_dev", "path",
                                                         "block.db_dev", "block.wal_dev")
                                            if attr in part_info}

        for osd in self.node.osds:
            with self.chdir('osd/{0}'.format(osd.id)):
                cmd = "tail -n {0} /var/log/ceph/ceph-osd.{1}.log".format(self.opts.ceph_log_max_lines, osd.id)
                self.save_output("log", cmd)

                # TODO: much of this can be done even id osd is down for filestore
                if osd.id in running_osds:
                    self.save("config", "txt", 0, osd.config)
                    dir_path = re.search("osd_data = (?P<dir_path>.*?)\n", osd.config).group("dir_path")
                    stor_type = self.node.rpc.fs.get_file(os.path.join(dir_path, 'type'), compress=False)
                    stor_type = stor_type.decode('utf8').strip()
                    assert stor_type in ('filestore', 'bluestore')
                    devs_for_osd[osd.id]['store_type'] = stor_type

                    if stor_type == 'filestore':
                        cmd = "ls -1 '{0}'".format(os.path.join(osd.storage, 'current'))
                        code, res = run_rpc_with_code(self.node.rpc, cmd)

                        if code == 0:
                            pgs = [name.split("_")[0] for name in res.split() if "_head" in name]
                            res = "\n".join(pgs)

                        self.save("pgs", "txt", code, res)

                    if stor_type == 'filestore':
                        data_dev = devs_for_osd[osd.id]["path"]
                        j_dev = devs_for_osd[osd.id]["journal_dev"]
                        osd_dev_conf = {'data': data_dev,
                                        'journal': j_dev,
                                        'r_data': dev_tree[data_dev],
                                        'r_journal': dev_tree[j_dev],
                                        'type': stor_type}
                    else:
                        assert stor_type == 'bluestore'
                        data_dev = devs_for_osd[osd.id]["block_dev"]
                        db_dev = devs_for_osd[osd.id].get('block.db_dev', data_dev)
                        wal_dev = devs_for_osd[osd.id].get('block.wal_dev', db_dev)
                        osd_dev_conf = {'data': data_dev,
                                        'wal': wal_dev,
                                        'db': db_dev,
                                        'r_data': dev_tree[data_dev],
                                        'r_wal': dev_tree[wal_dev],
                                        'r_db': dev_tree[db_dev],
                                        'type': stor_type}

                    self.save('devs_cfg', 'json', 0, json.dumps(osd_dev_conf))
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
        ("df", 'txt', 'df'),
        ("dmidecode", "txt", "dmidecode"),
        ("dmesg", "txt", "dmesg"),
        ("ipa4", "txt", "ip -o -4 a"),
        ("ipa", "txt", "ip a"),
        ("ifconfig", "txt", "ifconfig"),
        ("ifconfig_short", "txt", "ifconfig -s"),
        ("journalctl", 'txt', 'journalctl -b'),
        ("lshw", "xml", "lshw -xml"),
        ("lsblk", "txt", "lsblk -O"),
        ("lsblk_short", "txt", "lsblk"),
        ("mount", "txt", "mount"),
        ("netstat", "txt", "netstat -nap"),
        ("netstat_stat", "txt", "netstat -s"),
        ("sysctl", "txt", "sysctl -a"),
        ("uname", "txt", "uname -a"),
        ("bonds", "txt", "cat /proc/net/bonding/*")
    ]

    node_files = [
        "/proc/diskstats", "/proc/meminfo", "/proc/loadavg", "/proc/cpuinfo", "/proc/uptime", "/proc/vmstat"
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
            self.collect_block_devs()

            try:
                if self.node.rpc.fs.file_exists("/etc/debian_version"):
                    self.save_output("packages_deb", "dpkg -l", frmt="txt")
                else:
                    self.save_output("packages_rpm", "yum list installed", frmt="txt")
            except Exception as exc:
                logger.warning("Failed to download packages information from node %s: %s", self.node.name, exc)

    def collect_block_devs(self) -> None:
        bdevs_info = self.node.rpc.sensors.get_block_devs_info()
        bdevs_info = {name.decode('utf8'): data for name, data in bdevs_info.items()}
        for name_prefix in ['loop']:
            for name in bdevs_info:
                if name.startswith(name_prefix):
                    del bdevs_info[name]

        hdparm_exists, smartctl_exists, nvme_exists = self.node.rpc.fs.binarys_exists(['hdparm', 'smartctl', 'nvme'])   

        missing = []
        if not hdparm_exists:
            missing.append('hdparm')

        if not smartctl_exists:
            missing.append('smartctl-tools')

        if not nvme_exists:
            missing.append('nvme-tools')
        else:
            code, out = self.save_output('nvme_list', 'nvme list -o json', frmt='json')
            if code == 0:
                try:
                    for dev in json.loads(out)['Devices']:
                        name = os.path.basename(dev['DevicePath'])
                        self.save_output('block_devs/{}/nvme_smart_log'.format(name),
                                         'nvme smart-log {} -o json'.format(dev['DevicePath']), frmt='json')
                except:
                    logging.warning("Failed to process nvme list output")

        if missing:
            logger.warning("%s is not installed on %s", ",".join(missing), self.node.name)

        code, out_js = self.save_output("lsblkjs", "lsblk -O -b -J", frmt="json")
        if code == 0:
            for dev_node in json.loads(out_js)['blockdevices']:
                name = dev_node['name']
                with self.chdir('block_devs/' + name):
                    if hdparm_exists:
                        self.save_output('hdparm', "sudo hdparm -I /dev/" + name)

                    if smartctl_exists:
                        self.save_output('smartctl', "sudo smartctl -a /dev/" + name)

    def collect_interfaces_info(self) -> None:
        interfaces = {}
        unknown_speed = []
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
                    if speed == 'Unknown!':
                        unknown_speed.append(dev)
                    else:
                        logger.warning("Node %s - can't transform %s interface speed %r to Bps",
                                       self.node.name, dev, speed)

                if isinstance(speed, int):
                    interface['speed'] = speed
                else:
                    interface['speed_s'] = speed

        self.save('interfaces', 'json', 0, json.dumps(interfaces))
        if unknown_speed:
            logger.warning("Can't detect speed for interface(s): %s", ",".join(unknown_speed))


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
            "ceph": {"sources": ['historic', 'perf_dump'], "osds": "all"}
        }

        # self.results = {}

        #     "perprocess-cpu": {"allowed": ".*"},
        #     "perprocess-ram": {"allowed": ".*"},
        #     "system-cpu": {"allowed": ".*"},
        #     "system-ram": {"allowed": ".*"},
        #     "ceph": {"osds": []},

        self.node.rpc.sensors.start(cfg)

    def collect_performance_data(self) -> None:
        for sensor_path, data, is_unpacked, units in unpack_rpc_updates(self.node.rpc.sensors.get_updates()):
            if '.' in sensor_path:
                sensor, dev, metric = sensor_path.split(".")
            else:
                sensor, dev, metric = sensor_path, "", ""

            if is_unpacked:
                ext = 'csv' if isinstance(data, array.array) else 'json'
                extra = [sensor, dev, metric, units]
                self.save("perf_monitoring/{0}/{1}".format(self.node.fs_name, sensor_path), ext, 0, data,
                          extra=extra)
            else:
                if metric == 'historic':
                    self.storage.put_raw(data, "perf_monitoring/{0}/{1}.bin".format(self.node.fs_name, sensor_path))
                elif metric == 'perf_dump':
                    self.storage.put_raw(data, "perf_monitoring/{0}/{1}.json".format(self.node.fs_name, sensor_path))


ALL_COLLECTORS = [CephDataCollector, NodeCollector, LoadCollector]  # type: List[Type[Collector]]


def discover_nodes(opts: Any, ip_mapping: Dict[str, str] = None, master_node: Node = None) -> List[Node]:
    nodes = {}
    ip_mapping = {} if not ip_mapping else ip_mapping

    if master_node:
        exec_func = lambda x: rpc_run(master_node.rpc, x, node_name=master_node.name).decode('utf8')
    else:
        exec_func = lambda x: run_locally(x).decode('utf8')

    mons = get_mons_nodes(exec_func, opts.ceph_extra)
    for mon_id, (ip, name) in mons.items():
        ip = ip_mapping.get(ip, ip)
        node = Node(ip)
        node.mon = name
        node.roles.add('mon')
        nodes[ip] = node

    osd_nodes = get_osds_nodes(exec_func, opts.ceph_extra)
    for ip, osds in osd_nodes.items():
        ip = ip_mapping.get(ip, ip)
        node = nodes.setdefault(ip, Node(ip))
        node.osds = osds
        node.roles.add('osd')

    return list(nodes.values())


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


def collect(storage: IStorageNNP, opts: Any, executor: Executor) -> None:
    # This variable is updated from main function
    ssh_opts = "-o LogLevel=quiet -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    ssh_opts += "-o ConnectTimeout={0} -o ConnectionAttempts=2".format(opts.ssh_conn_timeout)

    if opts.ssh_user:
        ssh_opts += " -l {0} ".format(opts.ssh_user)

    if opts.ssh_opts:
        ssh_opts += opts.ssh_opts + " "

    allowed_path_checker = get_allowed_paths_checker(opts.disable)
    allowed_collectors = opts.collectors.split(',')

    for collector in allowed_collectors:
        if collector not in ALL_COLLECTORS_MAP:
            logger.error(f"Can't found collector {collector}. Only {','.join(ALL_COLLECTORS_MAP)} are available")
            return

    ip_mapping = {}
    if opts.ip_mapping:
        with open(opts.ip_mapping) as fd:
            for line in fd:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ip, ssh_ip = line.split()
                ip_mapping[ip] = ssh_ip

    if opts.ceph_master:
        logger.info(f"Connecting to ceph-master: {opts.ceph_master}")
        try:
            rpc_conn, _ = init_node(opts.ceph_master, ssh_opts=ssh_opts, with_sudo=opts.sudo)
            master_node = Node(ip=socket.gethostbyname(opts.ceph_master),
                               hostname=opts.ceph_master)
            master_node.rpc = rpc_conn
            set_node_name(master_node)
        except:
            logger.error(f"Can't connect to ceph-master: {opts.ceph_master}")
            raise
    else:
        master_conn = None

    code = run_with_code('which ceph')[0] if master_node is None \
            else run_rpc_with_code(master_node.rpc, 'which ceph', node_name=master_node.name)[0]

    if code != 0:
        logger.error("No 'ceph' command available. Run this script from node, which has ceph access")
        return

    if not opts.no_detect and not opts.ceph_master_only:
        detect_nodes = discover_nodes(opts, ip_mapping, master_node)
        all_nodes_dct = dict((node.ip, node) for node in detect_nodes)
    else:
        all_nodes_dct = {}

    for node_name_or_ip in opts.node:
        try:
            ipaddress.ip_address(node_name_or_ip)
        except ValueError:
            ip = socket.gethostbyname(node_name_or_ip)
            hostname = node_name_or_ip
        else:
            ip = node_name_or_ip
            hostname = None
            if not opts.dont_rev_resolve:
                try:
                    hostname = socket.gethostbyaddr(node_name_or_ip)[0]
                except:
                    pass
        all_nodes_dct.setdefault(ip, Node(ip, hostname))

    if all_nodes_dct:
        ips, nodes = zip(*all_nodes_dct.items())
        logger.info("Found %s hosts in total", len(nodes))
    else:
        ips = []
        nodes = []

    nodes_per_type = {}
    no_role = []
    for node in nodes:
        for role in node.roles:
            nodes_per_type.setdefault(role, []).append(node.name)
        if not node.roles:
            no_role.append(node.name)

    for role, role_nodes in sorted(nodes_per_type.items()):
        logger.info("Found %s nodes of role %s: %s", len(role_nodes), role, ",".join(sorted(role_nodes)))
    if no_role:
        logger.info("Found %s nodes without roles: %s", len(no_role), ",".join(sorted(no_role)))

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
                return True, init_node(node, ssh_opts=ssh_opts, with_sudo=opts.sudo)
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
            if not opts.ceph_master_only:
                for node in nodes:
                    func = CephDataCollector(allowed_path_checker, storage, opts, node,
                                             pretty_json=not opts.no_pretty_json).collect
                    roles = ["mon"] if node.mon else []
                    if node.osds:
                        roles.append("osd")
                    futures.append(executor.submit(func, roles))
            func = CephDataCollector(allowed_path_checker, storage, opts, master_node,
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
    except Exception as exc:
        logger.error("Exception happened(see full tb below): %s", exc)
        raise
    finally:
        logger.info("Collecting logs and teardown RPC servers")
        for node in nodes + [master_node]:
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
    p.add_argument("--ceph-master", default=None, help="Run all ceph cluster commands from NODE")
    p.add_argument("--ceph-master-only", action="store_true", help="Run only ceph master data collection, " +
                   "no osd data collections")
    p.add_argument("-C", "--ceph-extra", default="", help="Extra opts to pass to 'ceph' command")
    p.add_argument("-d", "--disable", default=[], nargs='*', help="Disable collect pattern")
    p.add_argument("-D", "--detect-only", action="store_true", help="Don't collect any data, only detect cluster nodes")
    p.add_argument("--dont-rev-resolve", action="store_true", default=False, help="Disable node name detection by ip")
    p.add_argument("--no-detect", action="store_true", help="Don't detect nodes. Inventory must be provided")
    p.add_argument("-g", "--save-to-git", metavar="DIR", help="Commit into git repo, cloned to folder")
    p.add_argument("-G", "--git-push", metavar="DIR", help="Commit into git repo, cloned to folder and run 'git push'")
    p.add_argument("-I", "--node", nargs='+', default=[],
                   help="Pass additional nodes from cli. ip_or_name[,ip_or_name...]")
    p.add_argument("--ip-mapping", metavar='FILE',
                   help="File, which maps node ip to sshable ip of same node. Use if sshd don't listen on ceph " +
                        "public ip or in other cases. Format: node_ceph_public_ip node_sshd_ip\\n... ")
    p.add_argument("-j", "--no-pretty-json", action="store_true", help="Don't prettify json data")
    p.add_argument("-l", "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default=None, help="Console log level")
    p.add_argument("-L", "--log-config", help="json file with logging config")
    p.add_argument("-m", "--max-pg-dump-count", default=2 ** 15, type=int,
                   help="maximum PG count to by dumped with 'pg dump' cmd")
    p.add_argument("-M", "--ceph-log-max-lines", default=10000, type=int, help="Max lines from osd/mon log")
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
    p.add_argument("--sudo", action="store_true", help="Run agent with sudo on remote node")
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
