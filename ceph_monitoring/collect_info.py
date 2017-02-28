import re
import sys
import time
import json
import Queue
import shutil
import logging
import os.path
import tempfile
import argparse
import warnings
import datetime
import threading
import functools
import contextlib
import collections


from agent.agent import ConnectionClosed
from cephlib.storage import make_storage
from cephlib.common import setup_loggers, check_output, check_output_ssh, pmap, run
from cephlib.rpc import init_node, rpc_run, rpc_run_ch
from cephlib.discover import get_osds_nodes, get_mons_nodes


logger = logging.getLogger('collect')


class CollectSettings(object):
    def __init__(self):
        self.disabled = []

    def disable(self, pattern):
        self.disabled.append(re.compile(pattern))

    def allowed(self, path):
        for pattern in self.disabled:
            if pattern.search(path):
                return False
        return True


class Node(object):
    def __init__(self, ip, name):
        self.ip = ip
        self.name = name
        self.rpc = None
        self.daemon_pid = None
        self.osds = []
        self.mon = None


# CMD executors --------------------------------------------------------------------------------------------------------


def rpc_get_host_interfaces():
    res = []
    for line in open("/sys/class/net").read().strip().split("\n")[1:]:
        if not line.startswith('l'):
            continue

        params = line.split()

        if len(params) < 11:
            # logger.warning("Strange line in '/sys/class/net' node %s: %r", socket.gethostname(), line)
            continue

        res.append(('devices/pci' in params[10], params[8]))
    return res


def get_device_for_file(node, fname):
    dev = node.rpc.sensors.get_dev_for_file(fname)

    assert dev.startswith('/dev')

    logger.info("dev = %s", dev)
    root_dev = dev = dev.strip()
    while root_dev[-1].isdigit():
        root_dev = root_dev[:-1]
    return root_dev, dev


class Collector(object):
    name = None
    run_alone = False

    def __init__(self, collect_settings, storage, opts):
        self.collect_settings = collect_settings
        self.storage = storage
        self.opts = opts
        self.node = None

    @contextlib.contextmanager
    def chdir(self, path):
        saved = self.storage
        self.storage = self.storage.substorage(path)
        try:
            yield
        finally:
            self.storage = saved

    def save(self, path, format, code, data, check=True):
        if check:
            if not self.collect_settings.allowed(path):
                return

        self.storage.put(path + "." + ("err" if code != 0 else format), data)
        return data if code == 0 else None

    def save_file(self, path, file_path, format='txt', check=True):
        "Download and save file from node"
        if check:
            if not self.collect_settings.allowed(path):
                return

        try:
            if self.node is None:
                loc = 'local'
                content = open(file_path, 'rb').read()
            else:
                loc = self.node.name
                content = self.node.rpc.fs.get_file(file_path)
            code = 0
        except (IOError, RuntimeError) as exc:
            logger.warning("Can't get file %r from node %s. %s", file_path, loc, exc)
            content = str(exc)
            code = 1

        self.save(path, format, code, content)
        return content if code == 0 else None

    def save_output(self, path, cmd, format='txt', check=True):
        if check:
            if not self.collect_settings.allowed(path):
                return

        if self.node is None:
            res = check_output(cmd)
            loc = "locally"
        else:
            res = rpc_run(self.node, cmd)
            loc = "on node {0}".format(self.node.name)

        if res.code != 0:
            logger.warning("Cmd {0} failed {1} with code {2}".format(cmd, loc, res.code))

        self.save(path, format, res.code, res.out)
        return res


CEPH_DEFAULT_CMD = "ceph -c {0.conf} -k {0.key} --format json "


class CephDataCollector(Collector):

    name = 'ceph'
    run_alone = False

    def __init__(self, *args, **kwargs):
        Collector.__init__(self, *args, **kwargs)
        self.ceph_cmd = CEPH_DEFAULT_CMD.format(self.opts)

    def collect_master(self):
        with self.chdir("master"):
            curr_data = "{0}\n{1}\n{2}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                time.time())

            self.save("collected_at", 'txt', 0, curr_data)
            res = self.save_output("status", self.ceph_cmd + "status", 'json')
            assert res, "{!r} failed".format(self.ceph_cmd + "status")

            cmds = ['osd tree',
                    'df',
                    'auth list',
                    'osd dump',
                    'health',
                    'mon_status',
                    'osd lspools',
                    'osd perf',
                    'health detail',
                    "tell 'osd.*' version"]

            status = json.loads(res.out)
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

            self.save_output("rados_df",
                             "rados df -c {0.conf} -k {0.key} --format json".format(self.opts),
                             'json')

            with tempfile.NamedTemporaryFile() as fd:
                res = check_output(self.ceph_cmd + "osd getcrushmap -o " + fd.name)
                data = res.out if res.code != 0 else open(fd.name, "rb").read()
                self.save('crushmap', 'bin', res.code, data)

            with tempfile.NamedTemporaryFile() as fd:
                res = check_output(self.ceph_cmd + "osd getmap -o " + fd.name)
                data = res.out if res.code != 0 else open(fd.name, "rb").read()
                self.save('osdmap', 'bin', res.code, data)

    def emit_device_info(self, device_file):
        dev_str = rpc_run_ch(self.node, "df " + device_file)
        dev_data = dev_str.strip().split("\n")[1].split()

        used = int(dev_data[2]) * 1024
        avail = int(dev_data[3]) * 1024

        try:
            root_dev, dev = get_device_for_file(self.node, device_file)
        except AssertionError:
            logger.warning("Can't detect device for %r", device_file)
            return None

        rot_info_path = "/sys/block/{0}/queue/rotational".format(os.path.basename(root_dev))
        is_ssd = self.node.rpc.fs.get_file(rot_info_path).strip() == '0'

        self.save_output('hdparm', "sudo hdparm -I " + root_dev)
        self.save_output('smartctl', "sudo smartctl -a " + root_dev)
        val = {'dev': dev, 'root_dev': root_dev, 'used': used, 'avail': avail, 'is_ssd': is_ssd}
        self.save('stats', 'json', 0, json.dumps(val))

        return root_dev

    def collect_osd(self, osd_list):
        # check OSD process status
        out = rpc_run(self.node, "ps aux | grep ceph-osd")
        osd_re = re.compile(r"ceph-osd[\t ]+.*(-i|--id)[\t ]+(\d+)")
        running_osds = set(int(rr.group(2)) for rr in osd_re.finditer(out))

        ids_from_ceph = set(osd.id for osd in osd_list)
        expected_osds = running_osds.intersection(ids_from_ceph)
        down_osds = ids_from_ceph - running_osds
        unexpected_osds = running_osds.difference(ids_from_ceph)

        for osd_id in unexpected_osds:
            logger.warning("Unexpected osd-{0} in node {1}.".format(osd_id, self.node.name))

        for osd in osd_list:
            with self.chdir('osd/{0}'.format(osd.id)):
                cmd = "tail -n {0} /var/log/ceph/ceph-osd.{1}.log".format(self.opts.ceph_log_max_lines, osd.id)
                self.save_output("log", cmd)

                if osd.id in running_osds:
                    self.save_output("config", osd.config, 'json')
                    self.save_output("storage_ls", "ls -1 " + os.path.join(osd.storage, 'current'))

                    with self.chdir("data"):
                        self.emit_device_info(osd.storage)

                    with self.chdir("journal"):
                        self.emit_device_info(osd.journal)
                else:
                    logger.warning("osd-{0} in node {1} is down.".format(osd.id, self.node.name) +
                                   " No config available, will use default data and journal path")

    def collect_monitor(self):
        with self.chdir("mon/{0}".format(self.node.name)):
            self.save_output("mon_daemons", "ps aux | grep ceph-mon")
            tail_ln = "tail -n {}".format(self.opts.ceph_log_max_lines)
            self.save_output("mon_log", tail_ln + "/var/log/ceph/ceph-mon.{}.log".format(self.node.params['mon_name']))
            self.save_output("ceph_log", tail_ln + " /var/log/ceph/ceph.log")
            self.save_output("ceph_audit", tail_ln + " /var/log/ceph/ceph.audit.log")


class VirshCollector(Collector):
    def collect_node(self, path, node):
        # rpc_run("virsh list")
        pass


class NodeCollector(Collector):

    name = 'node'
    run_alone = False

    node_commands = [
        ("lshw", "xml", "lshw -xml"),
        ("lsblk", "txt", "lsblk -a"),
        ("uname", "txt", "uname -a"),
        ("dmidecode", "txt", "dmidecode"),
        ("mount", "txt", "mount"),
        ("ipa", "txt", "ip -o -4 a"),
        ("netstat", "txt", "netstat -nap"),
        ("sysctl", "txt", "sysctl -a")
    ]

    node_files = [
        "/proc/diskstats", "/proc/meminfo", "/proc/loadavg", "/proc/cpuinfo", "/proc/uptime", "/var/log/dmesg",
        "/proc/vmstat"
    ]

    node_renamed_files = [("netdev", "/proc/net/dev"), ("ceph_conf", "/etc/ceph/ceph.conf")]

    def collect_node(self):
        with self.chdir('hosts/' + self.node.name):
            for path_offset, frmt, cmd in self.node_commands:
                self.save_output(path_offset, cmd, format=frmt)

            for fpath in self.node_files:
                self.save_file(os.path.basename(fpath), fpath)

            for name, fpath in self.node_renamed_files:
                self.save_output(name, fpath)

    def collect_interfaces_info(self):
        interfaces = {}
        for is_phy, dev in self.get_host_interfaces(self.node):
            interface = {'dev': dev, 'is_phy': is_phy}
            interfaces[dev] = interface

            if not is_phy:
                continue

            speed = None
            res = rpc_run(self.node, "ethtool " + dev)
            if res.code == 0:
                # TODO: move this to report code, only collect here
                for line in res.out.split("\n"):
                    if 'Speed:' in line:
                        speed = line.split(":")[1].strip()
                    if 'Duplex:' in line:
                        interface['duplex'] = line.split(":")[1].strip() == 'Full'

            res = rpc_run(self.node, "iwconfig " + dev)
            # TODO: move this to report code, only collect here
            if res.code == 0 and 'Bit Rate=' in res.out:
                br1 = res.out.split('Bit Rate=')[1]
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
                        speed = int(speed.replace(name, '')) * mult
                        break
                else:
                    logger.warning("Node %s - can't transform %s interface speed %r to Bps",
                                   self.node.name, dev, speed)

                if isinstance(speed, int):
                    interface['speed'] = speed
                else:
                    interface['speed_s'] = speed

        self.save('interfaces', 'json', 0, json.dumps(interfaces))


class CephLoadCollector(Collector):
    name = 'load'

    def start_performance_monitoring(self, node):
        node.rpc.ceph_mon.start_ceph_perf_collectors("cpu", "net", "blk")

    def collect_performance_data(self, path, node):
        dt = list(node.rpc.ceph_mon.get_perf_data())
        for tp, data in dt:
            self.emit("{0}/perf_monitoring/{1}/{2}".format(path, node.name, tp), "txt", 0, data)


def discover_ceph(opts):
    extra_args = "-c {0.conf} -k {0.key}".format(opts)

    for mon_id, (ip, name) in get_mons_nodes(run, extra_args).items():
        yield Node(ip, None, ['monitor'], mon_name=name)

    for ip, osds in get_osds_nodes(run, extra_args).items():
        for osd in osds:
            yield Node(ip, None, ['osd'], osd_id=osd.id, osd_jdev=osd.journal, osd_ddev=osd.storage)


def discover_nodes(opts):
    nodes = {}
    for node in discover_ceph(opts):
        if node.name in nodes:
            nodes[node.name].merge(node)
        else:
            nodes[node.name] = node

    return list(nodes.values())


def run_all(opts, run_q):
    def pool_thread():
        val = run_q.get()
        while val is not None:
            try:
                func, path, node = val
                func(path, node)
            except Exception:
                logger.exception("In worker thread")
            val = run_q.get()

    running_threads = []
    for i in range(opts.pool_size):
        th = threading.Thread(target=pool_thread)
        th.daemon = True
        th.start()
        running_threads.append(th)
        run_q.put(None)

    while True:
        time.sleep(0.01)
        if all(not th.is_alive() for th in running_threads):
            break


Collectors = collections.namedtuple("Collectors", ["ceph", "node", "load", "vm"])


ALL_COLLECTORS = [
    CephDataCollector,
    NodeCollector,
    CephLoadCollector,
    VirshCollector,
]


# need one separated collector per node
def get_collectors(allowed_collectors, opts, collector_settings, storage):
    collectors = []

    if CephDataCollector.name in allowed_collectors:
        ceph_collector = CephDataCollector(collector_settings, storage, opts)
        collectors.append(ceph_collector)
    else:
        ceph_collector = None

    if NodeCollector.name in allowed_collectors:
        node_collector = NodeCollector(collector_settings, storage, opts)
        collectors.append(node_collector)
    else:
        node_collector = None

    ceph_load_collector = None
    if CephLoadCollector.name in allowed_collectors:
        if CephDataCollector.name not in allowed_collectors:
            logger.error("Can't collect performance info without ceph info collected")
            exit(1)
        else:
            ceph_load_collector = CephLoadCollector(collector_settings, storage, opts)

    return Collectors(ceph_collector, node_collector, ceph_load_collector)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--conf",
                   default="/etc/ceph/ceph.conf",
                   help="Ceph cluster config file")

    p.add_argument("-k", "--key",
                   default="/etc/ceph/ceph.client.admin.keyring",
                   help="Ceph cluster key file")

    p.add_argument("-l", "--log-level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default="INFO",
                   help="Colsole log level")

    p.add_argument("-p", "--pool-size",
                   default=16, type=int,
                   help="Worker pool size")

    p.add_argument("-t", "--ssh-conn-timeout",
                   default=60, type=int,
                   help="SSH connection timeout")

    p.add_argument("-s", "--load-collect-seconds",
                   default=60, type=int, metavar="SEC",
                   help="Collect performance stats for SEC seconds")

    p.add_argument("--ceph-log-max-lines", default=1000,
                   type=int, help="Max lines from osd/mon log")

    p.add_argument("--collectors", default="ceph,node,load",
                   help="Comma separated list of collectors" +
                   "select from : " +
                   ",".join(coll.name for coll in ALL_COLLECTORS))

    p.add_argument("--max-pg-dump-count", default=2 ** 15,
                   type=int,
                   help="maximum PG count to by dumped with 'pg dump' cmd")

    p.add_argument("-o", "--result", default=None, help="Result file")

    p.add_argument("-n", "--dont-remove-unpacked", default=False,
                   action="store_true",
                   help="Keep unpacked data")

    p.add_argument("-d", "--disable", default=[],
                   nargs='*', help="Disable collect pattern")

    p.add_argument("-j", "--no-pretty-json", default=False,
                   action="store_true",
                   help="Don't prettify json data")

    return p.parse_args(argv[1:])


def main(argv):
    opts = parse_args(argv)

    run_q = Queue.Queue()

    out_folder = tempfile.mkdtemp()

    setup_loggers([logger],
                  getattr(logging, opts.log_level),
                  os.path.join(out_folder, "log.txt"))
    global logger_ready
    logger_ready = True

    if check_output('which ceph').code != 0:
        logger.error("No 'ceph' command available. Run this script from node, which has ceph access")
        return

    # This variable is updated from main function
    SSH_OPTS = "-o LogLevel=quiet -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    SSH_OPTS += "-o ConnectTimeout={0} -o ConnectionAttempts=2".format(opts.ssh_conn_timeout)

    collector_settings = CollectSettings()
    map(collector_settings.disable, opts.disable)
    allowed_collectors = opts.collectors.split(',')
    storage = make_storage(out_folder, existing=False, serializer='raw')
    collectors = get_collectors(allowed_collectors, opts, collector_settings, storage)

    nodes = discover_nodes(opts)
    nodes = [nodes[0]]

    per_role = collections.defaultdict(list)
    for node in nodes:
        for role in node.roles:
            per_role[role].append(node)

    for role, role_nodes in per_role.items():
        logger.info("Found %s hosts with role %r", len(role_nodes), role)

    logger.info("Found %s hosts in total", len(nodes))

    good_hosts = set(get_sshable_hosts(nodes, SSH_OPTS))
    bad_hosts = set(node.name for node in nodes) - good_hosts
    res_q.put((True, "bad_hosts", 'json', json.dumps(list(bad_hosts))))

    if len(bad_hosts) != 0:
        logger.warning("Next hosts aren't awailable over ssh and would be skipped: %s",
                       ",".join(bad_hosts))

    logger.info("Initializing RPC connections")
    nodes = [node for node in nodes if node.name in good_hosts]

    try:
        pmap(functools.partial(init_node, ssh_opts=SSH_OPTS), nodes)
        # collect data at the beginning
        if collectors.load is not None:
            for node in nodes:
                collectors.load.start_performance_monitoring(node)

        for node in nodes:
            for collector in collectors:
                for role in node.roles + ['node']:
                    handler = 'collect_' + role
                    if hasattr(collector, handler):
                        coll_func = getattr(collector, handler)
                        run_q.put((coll_func, "", node))

        save_results_thread = threading.Thread(target=save_results_th_func,
                                               args=(opts, res_q, out_folder))
        save_results_thread.daemon = True
        save_results_thread.start()

        monitor_till = time.time() + opts.load_collect_seconds
        try:
            run_all(opts, run_q)

            # collect data at the end
            if collectors.load is not None:
                if monitor_till >= time.time():
                    logger.info("Will wait for {0} seconds for usage data collection".format(int(dt)))
                    while monitor_till >= time.time():
                        time.sleep(0.1)

                logger.info("Collect load information")
                for node in nodes:
                    collectors.load.collect_performance_data("", node)

        except Exception:
            logger.exception("When collecting data:")
        finally:
            res_q.put(None)
            # wait till all data collected
            save_results_thread.join()

        if opts.result is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out_file = os.tempnam() + ".tar.gz"
        else:
            out_file = opts.result

        res = check_output("cd {0} ; tar -zcvf {1} *".format(out_folder, out_file))
        if res.code != 0:
            logger.error("Fail to archive results. Please found raw data at %r", out_folder)
        else:
            logger.info("Result saved into %r", out_file)
            if opts.log_level in ('WARNING', 'ERROR', "CRITICAL"):
                print "Result saved into %r" % (out_file,)

            if not opts.dont_remove_unpacked:
                shutil.rmtree(out_folder)
            else:
                logger.info("Temporary folder %r", out_folder)
                if opts.log_level in ('WARNING', 'ERROR', "CRITICAL"):
                    print "Temporary folder %r" % (out_folder,)
    finally:
        logger.info("Teardown RPC servers")
        for node in nodes:
            if node.rpc is not None:
                try:
                    node.rpc.server.stop()
                except ConnectionClosed:
                    node.rpc = None


if __name__ == "__main__":
    try:
        exit(main(sys.argv))
    except Exception:
        if logger_ready:
            logger.exception("During main")
        else:
            raise
