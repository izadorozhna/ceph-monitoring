import collections
from typing import Dict, List, Union, Iterable, Callable, Optional, Tuple, Any

from cephlib.units import b2ssize, b2ssize_10
from cephlib.common import flatten

from . import html
from .cluster_classes import Cluster, CephInfo, CephOSD, Host, FileStoreInfo, BlueStoreInfo, Disk, LogicBlockDev,\
                             OSDPGStats, NodePGStats
from .visualize_utils import tab, partition, perf_info_required, val_to_color
from .obj_links import host_link, osd_link, mon_link
from .groupby import group_by
from .cluster import get_nodes_pg_info
from .table import Table, bytes_sz, ident, idents_list, exact_count, extra_columns, float_vl


class HostRunInfo(Table):
    name = ident()
    services = idents_list()
    ram_free = bytes_sz("RAM<br>free")
    swap = bytes_sz("Swap<br>used")
    load = float_vl("Load avg<br>5m")
    ip_conn = ident("Conn<br>tcp/udp")
    ips = idents_list("IP's")
    scrub_err = exact_count("Scrub<br>errors")


@tab("Hosts config")
def show_hosts_info(cluster: Cluster, ceph: CephInfo) -> str:
    hosts_pgs_info = get_nodes_pg_info(ceph)
    mon_hosts = {mon.host.name for mon in ceph.mons.values()}

    host2osds = {}
    for osd in ceph.osds.values():
        host2osds.setdefault(osd.host.name, []).append(str(osd.id))

    rule2host2osds: Dict[Dict[str, List[CephOSD]]] = {}
    for rule in ceph.crush.rules.values():
        curr = rule2host2osds[rule.name] = {}
        for osd_node in ceph.crush.iter_osds_for_rule(rule.id):
            osd = ceph.osds[osd_node.id]
            curr.setdefault(osd.host.name, []).append(osd)

    root_names = sorted(rule2host2osds)

    hosts_configs = []
    for host in cluster.sorted_hosts:
        by_speed = collections.Counter()
        for adapter in host.net_adapters.values():
            if adapter.is_phy:
                by_speed[adapter.speed if adapter.speed else 0] += 1

        nets = "<br>".join(f"{b2ssize_10(speed * 8) if speed else '???'} * {count}"
                           for speed, count in sorted(by_speed.items()))

        hosts_configs.append({
            "name": host.name,
            "osds_count": tuple(len(rule2host2osds[root_name].get(host.name, [])) for root_name in root_names),
            'has_mon': host.name in mon_hosts,
            'cores': sum(inf.cores for inf in host.hw_info.cpu_info),
            'ram': int(host.mem_total / 2 ** 30 + 0.5),
            'net': nets
        })


    class HostsConfigTable(Table):
        count = exact_count()
        names = idents_list(chars_per_line=60)
        osds_count = extra_columns(exact_count(),
                                   **{root_name: f"osd count for<br>{root_name}" for root_name in root_names})
        has_mon = ident()
        cores = exact_count("CPU<br>Cores+HT")
        ram = bytes_sz("RAM<br>total")
        storage_devices = ident("Storage<br>devices")
        network = ident("Network<br>devices")

    configs = HostsConfigTable()

    for grouped_idx in group_by(hosts_configs, mutable_keys="name"):
        items = [hosts_configs[idx] for idx in grouped_idx]
        first_item = items[0]
        row = configs.next_row()
        row.count = len(items)
        row.names = [(host_link(itm['name']).link, itm['name']) for itm in items]

        for name, vl in zip(root_names, first_item["osds_count"]):
            row.osds_count[name] = vl

        row.has_mon = 'yes' if first_item["has_mon"] else 'no'
        row.cores = first_item["cores"]
        row.ram = first_item["ram"]
        row.network = first_item["net"]

    run_info = HostRunInfo()
    for host in cluster.sorted_hosts:

        # TODO: add storage devices
        # TODO: add net info

        row2 = run_info.next_row()
        row2.name = host_link(host.name).link, host.name

        srv_strs = []
        if host.name in mon_hosts:
            srv_strs.append(f'<font color="#8080FF">Mon</font>: ' + mon_link(host.name).link)

        if host.name in host2osds:
            osds = host2osds[host.name]
            srv_strs.append('''<font color="#c77405">OSD's</font>: ''' +
                            ", ".join(osd_link(osd_id).link for osd_id in osds[:3]))
            for ids in partition(osds[3:], 4):
                srv_strs.append(", ".join(osd_link(osd_id).link for osd_id in ids))

        row2.services = srv_strs

        row2.ram_free = host.mem_free
        row2.swap = host.swap_total - host.swap_free
        row2.load = host.load_5m
        row2.ip_conn = f"{host.open_tcp_sock}/{host.open_udp_sock}", host.open_tcp_sock + host.open_udp_sock

        all_ip = flatten(adapter.ips for adapter in host.net_adapters.values())
        row2.ips = [str(addr.ip) for addr in all_ip if not addr.ip.is_loopback]

        if hosts_pgs_info and host.name in hosts_pgs_info:
            pgs_info = hosts_pgs_info[host.name]
            row2.scrub_err  = pgs_info.pg_stats.scrub_errors + pgs_info.pg_stats.deep_scrub_errors + \
                pgs_info.pg_stats.shallow_scrub_errors

    return str(configs.html(id="table-hosts-info", align=html.TableAlign.center_right)) + "<br>" * 5 + \
           str(run_info.html(id="table-hosts-run-info", align=html.TableAlign.center_right))

    # TODO: +hosts_pg_info(cluster, hosts_pgs_info)


def host_info(host: Host, ceph: CephInfo) -> str:
    cluster_networks = [(ceph.public_net, 'ceph-public'), (ceph.cluster_net, 'ceph-cluster')]

    doc = html.Doc()
    doc.center.H3(host.name)
    doc.br
    doc.center.H4("Interfaces:")
    doc.br

    table = html.HTMLTable(headers=["Name", "Type", "Duplex", "MTU", "Speed", "IP's", "Roles"],
                           extra_cls=["hostinfo-net"],
                           sortable=False,
                           align=html.TableAlign.left_center)

    def add_adapter_line(adapter, name):
        tp = "phy" if adapter.is_phy else ('bond' if adapter.dev in host.bonds else 'virt')
        roles = [role for net, role in cluster_networks if any((ip.ip in net) for ip in adapter.ips)]
        table.add_row(
            [name,
             tp,
             "" if adapter.duplex is None else str(adapter.duplex),
             "" if adapter.mtu is None else str(adapter.mtu),
             "" if adapter.speed is None else str(adapter.speed),
             "<br>".join(f"{ip.ip} / {ip.net.prefixlen}" for ip in adapter.ips),
             "<br>".join(roles)])

    all_ifaces = set(host.net_adapters)

    # first show bonds
    for _, bond in sorted(host.bonds.items()):
        assert bond.name in all_ifaces
        all_ifaces.remove(bond.name)
        adapter = host.net_adapters[bond.name]
        add_adapter_line(adapter, adapter.dev)

        for src in bond.sources:
            assert src in all_ifaces
            all_ifaces.remove(src)
            add_adapter_line(host.net_adapters[src],
                             f'<div class="disk-children-1">+{src}</dev>')

        for adapter_name in list(all_ifaces):
            if adapter_name.startswith(bond.name + '.'):
                all_ifaces.remove(adapter_name)
                add_adapter_line(host.net_adapters[adapter_name],
                                 f'<div class="disk-children-2">->{adapter_name}</dev>')

        table.add_row(["-------"] * len(table.headers))

    for name in sorted(all_ifaces):
        if name != 'lo':
            add_adapter_line(host.net_adapters[name], name)

    doc.center(str(table))
    doc.br
    doc.center.H4("HW disks:")
    doc.br

    mib_and_mb = lambda x: f"{b2ssize(x)}B / {b2ssize_10(x)}B"

    stor_roles = collections.defaultdict(lambda: collections.defaultdict(set))
    stor_classes = collections.defaultdict(set)

    for osd in ceph.osds.values():
        if osd.host is host:
            stor_roles[osd.storage_info.data.partition_name]['data'].add(osd.id)
            stor_roles[osd.storage_info.data.name]['data'].add(osd.id)

            if osd.class_name:
                stor_classes[osd.storage_info.data.partition_name].add(osd.class_name)
                stor_classes[osd.storage_info.data.name].add(osd.class_name)

            if isinstance(osd.storage_info, FileStoreInfo):
                stor_roles[osd.storage_info.journal.partition_name]['journal'].add(osd.id)
                stor_roles[osd.storage_info.journal.name]['journal'].add(osd.id)

                if osd.class_name:
                    stor_classes[osd.storage_info.journal.name].add(osd.class_name)
                    stor_classes[osd.storage_info.journal.partition_name].add(osd.class_name)
            else:
                assert isinstance(osd.storage_info, BlueStoreInfo)
                stor_roles[osd.storage_info.db.partition_name]['db'].add(osd.id)
                stor_roles[osd.storage_info.db.name]['db'].add(osd.id)
                stor_roles[osd.storage_info.wal.partition_name]['wal'].add(osd.id)
                stor_roles[osd.storage_info.wal.name]['wal'].add(osd.id)

                if osd.class_name:
                    stor_classes[osd.storage_info.db.partition_name].add(osd.class_name)
                    stor_classes[osd.storage_info.db.name].add(osd.class_name)
                    stor_classes[osd.storage_info.wal.partition_name].add(osd.class_name)
                    stor_classes[osd.storage_info.wal.name].add(osd.class_name)

    order = ['data', 'journal', 'db', 'wal']

    def html_roles(name: str) -> str:
        return "<br>".join(name + ": " + ", ".join(map(str, sorted(ids)))
                           for name, ids in sorted(stor_roles[name].items(), key=lambda x: order.index(x[0])))

    table = html.HTMLTable(headers=["Name", "Type", "Size"] +
                                   (["Roles"] if ceph.is_luminous else []) +
                                   ["Classes", "Scheduler", "Model info", "RQ-size", "Phy Sec", "Min IO"],
                            extra_cls=["hostinfo-disks"])

    for _, disk in sorted(host.disks.items()):

        if disk.hw_model.model and disk.hw_model.vendor:
            model = f"Model: {disk.hw_model.vendor.strip()} / {disk.hw_model.model}"
        elif disk.hw_model.model:
            model = f"Model: {disk.hw_model.model}"
        elif disk.hw_model.vendor:
            model = f"Vendor: {disk.hw_model.vendor.strip()}"
        else:
            model = ""

        serial = "" if disk.hw_model.serial is None else ("Serial: " + disk.hw_model.serial)

        if model and serial:
            info = f"{model}<br>{serial}"
        elif model or serial:
            info = model if model else serial
        else:
            info = ""

        table.add_row([disk.name, disk.tp.name, mib_and_mb(disk.size),
                       html_roles(disk.name)] +
                       (["<br>".join(stor_classes.get(disk.name, []))] if ceph.is_luminous else []) +
                       ["" if disk.scheduler is None else str(disk.scheduler),
                       info,
                       str(disk.rq_size),
                       str(disk.phy_sec),
                       str(disk.min_io)])

    doc.center(str(table))
    doc.br
    doc.center.H4("Mountable:")
    doc.br
    table = html.HTMLTable(headers=["Name", "Size", "Type", "Mountpoint", "Ceph role", "Fs", "Free space", "Label"],
                           extra_cls=["hostinfo-mountable"], sortable=False)

    def run_over_children(obj: Union[Disk, LogicBlockDev], level: int):
        table.add_row([f'<div class="disk-children-{level}">{obj.name}</div>',
                       mib_and_mb(obj.size),
                       obj.tp.name if level == 0 else '',
                       obj.mountpoint if obj.mountpoint else '',
                       "" if level == 0 else html_roles(obj.name),
                       obj.fs if obj.fs else '',
                       mib_and_mb(obj.free_space) if obj.free_space is not None else '',
                       "" if obj.label is None else obj.label])

        for _, ch in sorted(obj.children.items(), key=lambda x: x[1].partition_num):
            run_over_children(ch, level + 1)

    for _, disk in sorted(host.disks.items()):
        run_over_children(disk, 0)

    doc.center(str(table))

    return str(doc)


@tab("Hosts PG's info")
def hosts_pg_info(cluster: Cluster, hosts_pgs_info: Dict[str, NodePGStats]) -> html.HTMLTable:
    header_row = ["Name",
                  "PGs",
                  "User data",
                  "Reads",
                  "Read B",
                  "Writes",
                  "Write B"]

    if cluster.has_second_report:
        header_row.extend(["User data<br>store rate", "Reads<br>ps", "Read Bps", "Writes<br>ps", "Write Bps"])

    table = html.HTMLTable("table-hosts-pg-info-long" if cluster.has_second_report else "table-hosts-pg-info",
                           header_row, align=html.TableAlign.center_right)

    def add_pg_stats(stats: OSDPGStats):
        table.add_cell_b2ssize(stats.bytes)
        table.add_cell_b2ssize_10(stats.reads)
        table.add_cell_b2ssize(stats.read_b)
        table.add_cell_b2ssize_10(stats.writes)
        table.add_cell_b2ssize(stats.write_b)

    for host in sorted(cluster.hosts.values(), key=lambda x: x.name):
        if host.name in hosts_pgs_info:
            table.add_cell(host_link(host.name).link)
            pgs_info = hosts_pgs_info[host.name]
            table.add_cell(str(len(pgs_info.pgs)))

            add_pg_stats(pgs_info.pg_stats)
            if cluster.has_second_report:
                assert pgs_info.d_pg_stats
                add_pg_stats(pgs_info.d_pg_stats)
            table.next_row()

    return table


@perf_info_required
@tab("Storage devs load2")
def show_host_io_load_in_color(cluster: Cluster, uptime: bool) -> str:
    hosts = [host for _, host in sorted(cluster.hosts.items())]

    loads = [
        ('iops', b2ssize_10, 'IOPS', 50),
        ('read_iops', b2ssize_10, 'Read IOPS', 30),
        ('write_iops', b2ssize_10, 'Write IOPS', 30),

        ('total_bytes', lambda x: str(int(x / 2 ** 20)), 'MiBps', 100 * 1024 ** 2),
        ('read_bytes', lambda x: str(int(x / 2 ** 20)), 'Read MiBps', 100 * 1024 ** 2),
        ('write_bytes', lambda x: str(int(x / 2 ** 20)), 'Write MiBps', 100 * 1024 ** 2),

        ('lat', lambda x: '-' if x is None else str(int(x)), 'Latency, ms', 20),
        ('queue_depth', lambda x: f"{x:.1f}", 'Average QD', 3),
        ('io_time', lambda x: f"{x:.1f}", 'Active time %', 100),
    ]

    blocks = []
    for pos, (attr, conversion, tp, min_max_val) in enumerate(loads, 1):
        max_val = 300.0 if attr == 'lat' else None
        table = make_storage_devs_load_table(hosts, attr, conversion, uptime, max_val, min_max_val)
        blocks.append(f"<br><H4>{tp}</H4><br><br>{table}")

    return "<br>".join(blocks)


@tab("Net load")
@perf_info_required
def show_host_network_load_in_color(cluster: Cluster, ceph: CephInfo) -> str:
    send_net_io: Dict[str, List[Tuple]] = collections.defaultdict(list)
    recv_net_io: Dict[str, List[Tuple]] = collections.defaultdict(list)

    max_net_count = 0
    for host in cluster.hosts.values():
        cl_if = host.find_interface(ceph.cluster_net)
        pb_if = host.find_interface(ceph.public_net)
        nets = [('cluster', cl_if), ('public', pb_if)]
        nets += sorted((net.dev, net) for net in host.net_adapters.values()  # type: ignore
                       if net is not None and net.is_phy and net != pb_if and net != cl_if )

        uptime = host.uptime
        max_net_count = max(max_net_count, len(nets))
        for name, net in nets:
            if net is None or net.usage is None:
                if name in {'cluster', 'public'}:
                    send_net_io[host.name].append((name, None, None, None))
                    recv_net_io[host.name].append((name, None, None, None))
            else:
                send_net_io[host.name].append((name, net.usage.send_bytes / uptime,
                                              net.usage.send_packets / uptime,
                                              net.speed))
                recv_net_io[host.name].append((name, net.usage.recv_bytes / uptime,
                                              net.usage.recv_packets / uptime,
                                              net.speed))

    std_nets = {'public', 'cluster'}
    ceph_net_count = len(std_nets)
    tables = []

    for io, name in [(send_net_io, "Send (KiBps/Pps)"), (recv_net_io, "Receive (KiBps/Pps)")]:

        table = html.HTMLTable(headers=["host", "cluster", "public"] +
                                       ["hw adapter"] * (max_net_count - ceph_net_count),
                               zebra=False)

        for host_name, net_loads in sorted(io.items()):
            table.add_cell(host_link(host_name).link)
            for net_name, load_bps, load_pps, speed in net_loads:
                if load_bps is None:
                    table.add_cell('-', sorttable_customkey="")
                else:
                    load_bps = int(load_bps) // 1024
                    load_pps = int(load_pps)
                    color = "#FFFFFF" if speed is None else val_to_color(float(load_bps) / speed)
                    load_text = f"{b2ssize(load_bps)}/{b2ssize_10(load_pps)}"

                    if net_name in std_nets:
                        text = load_text
                    else:
                        text = html.rtag.div(net_name, _class="left") + H.div(load_text, _class="right")

                    table.add_cell(text, bgcolor=color, sorttable_customkey=str(load_bps))

            for i in range(max_net_count - len(net_loads)):
                table.add_cell("-", sorttable_customkey='0')

            table.next_row()

        tables.append(table)

    return "<br>".join(map(str, tables))


@tab("Storage devs load")
def make_storage_devs_load_table(hosts: Iterable[Host],
                                 attr: str,
                                 conversion: Callable[[Union[float, int]], str],
                                 uptime: bool,
                                 max_val: Optional[float],
                                 min_max_val: float) -> html.HTMLTable:

        vals = []
        uptimes = []
        for host in hosts:
            vals.append(sorted(host.disks.values(), key=lambda x: x.name))
            if uptime:
                uptimes.append(host.uptime)

        max_len = max(map(len, vals))

        if max_val is None:
            max_val = min_max_val
            for host_devs in vals:
                max_val = max(max_val, max(getattr(dev, attr) for dev in host_devs))

        table = html.HTMLTable(headers=['host'] + ['load'] * max_len, zebra=False, extra_cls=["io_load_in_color"])
        for idx, host_devs in enumerate(vals):
            table.add_cell(host_link(host_devs[0].logic_dev.hostname).link)
            for dev_info in host_devs:

                if uptime:
                    val = getattr(dev_info.logic_dev.usage, attr) / uptimes[idx]
                else:
                    val = getattr(dev_info.logic_dev.d_usage, attr)

                color = "#FFFFFF" if val is None or max_val == 0 else val_to_color(float(val) / max_val)

                s_val = 0 if val < 1 else conversion(val)

                cell_data = html.rtag.div(dev_info.name + " ", _class="left") + html.rtag.div(s_val, _class="right")
                table.add_cell(cell_data, bgcolor=color, sorttable_customkey=str(val))

            for i in range(max_len - len(host_devs)):
                table.add_cell("-", sorttable_customkey='0')
            table.next_row()
        return table

