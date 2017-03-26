def show_hosts_resource_usage(report, cluster):
    nets_info = {}

    for host in cluster.hosts.values():
        ceph_adapters = {net.name
                         for net in (host.ceph_cluster_adapter, host.ceph_public_adapter)
                         if net is not None}

        nets = [net for net in host.net_adapters.values()
                if net.is_phy and net.name not in ceph_adapters]

        nets_info[host.name] = sorted(nets, key=lambda x: x.name)

    if len(nets_info) == 0:
        max_nets = 0
    else:
        max_nets = max(map(len, nets_info.values()))

    header_row = ["Hostname",
                  "Cluster net<br>dev, ip",
                  "Cluster net<br>send/recv",
                  "Public net<br>dev, ip",
                  "Public net<br>send/recv"]

    header_row += ["Net"] * max_nets
    row_len = len(header_row)

    table = html.HTMLTable(headers=header_row)
    for host in sorted(cluster.hosts.values(), key=lambda x: x.name):
        perf_info = [host.name]

        for net in (host.ceph_cluster_adapter, host.ceph_public_adapter):
            if net is None:
                perf_info.extend(["-"] * 3)
            else:
                dev_ip = "{0}<br>{1}".format(net.name, ",".join(str(ip) for ip in net.ips))

                if host.hw_info is None or net.name not in host.hw_info.net_info:
                    perf_info.append(dev_ip)
                else:
                    speed, dtype, _ = host.hw_info.net_info[net.name]
                    settings = "{0}, {1}".format(speed, dtype)
                    perf_info.append("{0}<br>{1}".format(dev_ip, settings))

                if net.load is not None:
                    perf_info.append("{0} / {1} Bps<br>{2} / {3} Pps".format(
                        b2ssize(net.load.send_bytes_avg, False),
                        b2ssize(net.load.recv_bytes_avg, False),
                        b2ssize(net.load.send_packets_avg, False),
                        b2ssize(net.load.recv_packets_avg, False),
                    ))
                else:
                    perf_info.append('-')

        for net in nets_info[host.name]:
            if net.speed is not None:
                cell_data = H.div(net.name, _class="left") + H.div(b2ssize(net.speed), _class="right")
                perf_info.append(cell_data)
            else:
                perf_info.append(net.name)

        perf_info += ['-'] * (row_len - len(perf_info))
        table.add_row(map(str, perf_info))

    report.add_block(12, "Host's resource usage:", table)




def draw_resource_usage(report, cluster):
    script = ceph_report_template.body_script

    writes_per_dev, reads_per_dev, order = get_io_resource_usage(cluster)

    for dct in (writes_per_dev, reads_per_dev):
        for key in list(dct):
            dct[key] = ','.join(map(str, dct[key]))

    if len(writes_per_dev) != 0 or len(reads_per_dev) != 0:
        report.style.append(ceph_report_template.css)
        report.style.append(
            ".usage {width: 700px; height: 600px; border: 1px solid lightgray;}"
        )

        report.script_links.extend(ceph_report_template.scripts)

    if len(writes_per_dev) != 0:
        wdata_param = [
            '{0!r}: [{1}]'.format(str(dname), writes_per_dev[dname])
            for dname in order
        ]
        wall_devs = map(repr, order)

        div_id = 'io_w_usage'

        report.add_block(12, "Disk writes:", H.div(id=div_id))

        report.add_hidden(
            script
            .replace('__id__', div_id)
            .replace('__devs__', ", ".join(wall_devs))
            .replace('__data__', ",".join(wdata_param))
        )

    if len(reads_per_dev) != 0:
        rdata_param = [
            '{0!r}: [{1}]'.format(str(dname), reads_per_dev[dname])
            for dname in order
        ]
        rall_devs = map(repr, order)

        div_id = 'io_r_usage'

        report.add_block(12, "Disk reads:", H.div(id=div_id))

        report.add_hidden(
            script
            .replace('__id__', div_id)
            .replace('__devs__', ", ".join(rall_devs))
            .replace('__data__', ",".join(rdata_param))
        )


rickshaw_js_links = [
    "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.6/d3.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/rickshaw/1.5.1/rickshaw.min.js"
]

rickshaw_code = """
<script>
var __graph_var_name__ = new Rickshaw.Graph( {
    element: document.querySelector("#__div_id__"),
    width: __width__,
    height: __height__,
    renderer: 'line',
    series: [
        __series__
    ]
});

__graph_var_name__.render();
</script>
"""


def draw_resource_usage_rsw(report, cluster):
    report.script_links.extend(rickshaw_js_links)
    writes_per_dev, reads_per_dev, order = get_io_resource_usage(cluster)

    if len(writes_per_dev) != 0:
        series = {}
        for dname, vals in writes_per_dev.items():
            series[dname] = [{'x': pos, 'y': val}
                             for pos, val in enumerate(vals)]

        plots_per_div = 2
        for dev in range(0, len(order), plots_per_div):
            div_id = "rsw_{0}".format(dev)
            var_name = "rsw_var_{0}".format(dev)

            data = []
            for name in order[dev: dev + plots_per_div]:
                data.append({
                    'color': "#c05020",
                    'data': series[name],
                    'name': name
                })

            report.add_block(12, "New load for " + ",".join(order[dev: dev + plots_per_div]),
                             H.div(id=div_id, _class="usage"))

            report.add_hidden(
                rickshaw_code
                .replace('__div_id__', div_id)
                .replace('__graph_var_name__', var_name)
                .replace('__width__', '600')
                .replace('__height__', '80')
                .replace('__color__', 'green')
                .replace('__series__', ",\n".join(map(json.dumps, data))) + "\n"
            )


        # perf_path = os.path.join(opts.out, "performance.html")
        # load_report = Report(opts.name, "performance.html")
        # # draw_resource_usage(load_report, cluster)
        # draw_resource_usage_rsw(load_report, cluster)
        # load_report.save_to(opts.out)
        # print "Peformance report successfully stored in", perf_path



visjs_script = """
  var network__id__ = null;
  function draw__id__() {
    if (network__id__ !== null) network__id__.destroy();
    var data = {nodes: [__nodes__], edges: [__eges__]};
    var options = {
      // layout: { hierarchical: {sortMethod: 'directed'}},
        edges: {smooth: true, arrows: {to : true }},
        nodes: {shape: 'dot'}
    };
    network__id__ = new vis.Network(document.getElementById('mynetwork__id__'), data, options);
  }
"""

visjs_css = """
.graph {width: 500px;height: 500px;border: 1px solid lightgray;}
"""


def tree_to_visjs(report, cluster):
    report.style.append(visjs_css)
    report.style_links.append(
        "https://cdnjs.cloudflare.com/ajax/libs/vis/4.7.0/vis.min.css")
    report.script_links.append(
        "https://cdnjs.cloudflare.com/ajax/libs/vis/4.7.0/vis.min.js"
    )

    # nodes = jstorage.master.osd_tree["nodes"]

    max_w = max(float(osd.crush_weight) for osd in cluster.osds)
    min_w = min(float(osd.crush_weight) for osd in cluster.osds)

    def get_color_w(node):
        if max_w - min_w < 1E-2 or node['type'] != 'osd':
            return "#ffffff"
        w = (float(node['crush_weight']) - min_w) / (max_w - min_w)
        return val_to_color(w)

    try:
        min_pg = min(cluster.sum_per_osd.values())
        max_pg = max(cluster.sum_per_osd.values())
        min_pg = min(max_pg / 2, min_pg)
    except AttributeError:
        min_pg = max_pg = None

    def get_color_pg_count(node):
        if cluster.sum_per_osd is None or min_pg is None or \
             (max_pg - min_pg) / float(max_pg) < 1E-2 or node['type'] != 'osd':

            return "#ffffff"

        w = (float(cluster.sum_per_osd[node['id']]) - min_pg) / (max_pg - min_pg)
        return val_to_color(w)

    def get_graph(color_func):
        nodes_list = []
        eges_list = []
        nodes_list = [
            "{{id:{0}, label:'{1}', color:'{2}'}}".format(
                node['id'],
                str(node['name']),
                color_func(node)
            )
            for node in cluster.osd_tree.values()
        ]

        for node in cluster.osd_tree.values():
            for child_id in node.get('children', []):
                eges_list.append(
                    "{{from: {0}, to: {1} }}".format(node['id'], child_id)
                )

        return ",".join(nodes_list), ",".join(eges_list)

    gnodes, geges = get_graph(get_color_w)
    report.scripts.append(
        visjs_script.replace('__nodes__', gnodes)
                    .replace('__eges__', geges)
                    .replace('__id__', '0')
    )
    report.add_block(6, 'Crush weight:', H.div(_class="graph", id="mynetwork0"))
    report.onload.append("draw0()")

    if cluster.sum_per_osd is not None:
        gnodes, geges = get_graph(get_color_pg_count)
        report.scripts.append(
            visjs_script.replace('__nodes__', gnodes)
                        .replace('__eges__', geges)
                        .replace('__id__', '1')
        )
        report.add_block(6, 'PG\'s count:', H.div('', _class="graph", id="mynetwork1"))
        report.onload.append("draw1()")

def get_io_resource_usage(cluster):
    writes_per_dev = {}
    reads_per_dev = {}
    order = []

    for osd in sorted(cluster.osds, key=lambda x: x.id):
        perf_m = cluster.hosts[osd.host].perf_monitoring
        if perf_m is None or 'io' not in perf_m:
            continue

        if osd.data_stor_stats is not None and osd.j_stor_stats is not None and \
           osd.j_stor_stats.root_dev == osd.data_stor_stats.root_dev:
            dev_list = [('data/jornal', osd.data_stor_stats)]
        else:
            dev_list = [('data', osd.data_stor_stats), ('journal', osd.j_stor_stats)]

        for tp, dev_stat in dev_list:
            if dev_stat is None:
                continue

            dev = os.path.basename(dev_stat.root_dev)
            if dev not in perf_m['io']:
                continue

            prev_val = perf_m['io'][dev].values[0]
            writes = []
            reads = []
            for val in perf_m['io'][dev].values[1:]:
                writes.append(val.writes_completed - prev_val.writes_completed)
                reads.append(val.reads_completed - prev_val.reads_completed)
                prev_val = val

            dev_uuid = "osd-{0}.{1}".format(str(osd.id), tp)
            writes_per_dev[dev_uuid] = writes
            reads_per_dev[dev_uuid] = reads
            order.append(dev_uuid)
    return writes_per_dev, reads_per_dev, order


#
#
# def show_agg_osd_lat_histo(report, cluster, max_xbins=25, cut_percentile=(0.01, 0.99)):
#     for field, header in (('journal_latency', "Journal latency"),
#                           ('apply_latency',  "Apply latency"),
#                           ('commitcycle_latency', "Commit cycle latency")):
#         lats = numpy.concatenate([osd.osd_perf[field] for osd in cluster.osds])
#         lats = lats[lats != NO_VALUE]
#
#         mmin, mmax = numpy.percentile(lats, (cut_percentile[0] * 100, cut_percentile[1] * 100))
#         numpy.clip(lats, mmin, mmax, lats)
#
#         bins = auto_edges(lats, bins=max_xbins, round_base=None)
#
#         seaborn.plt.clf()
#         _, ax = seaborn.plt.subplots(figsize=(6, 4))
#         plot_histo(lats, bins=bins, ax=ax)
#         seaborn.plt.tight_layout()
#         report.add_block(4, header + " histo", get_img(seaborn.plt))
