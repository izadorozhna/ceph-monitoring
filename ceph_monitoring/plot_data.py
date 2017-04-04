import warnings

import numpy
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

from .cluster import NO_VALUE
from .perf_parser import (load_ops_from_fd, ALL_STAGES_IN_ORDER, ALL_STAGES, STAGES_PRINTABLE_NAMES, get_img)
from .resource_usage import get_resource_usage
from cephlib.plot import plot_histo, hmap_from_2d, plot_hmap_with_y_histo


# ----------------------------------------------------------------------------------------------------------------------


def show_osd_used_space_histo(report, cluster):
    vals = [(100 - osd.free_perc) for osd in cluster.osds if osd.free_perc is not None]

    seaborn.plt.clf()
    _, ax = seaborn.plt.subplots(figsize=(6, 4))
    plot_histo(vals, left=0, right=100, ax=ax)
    seaborn.plt.tight_layout()
    report.add_block(6, "OSD used space (GiB)", get_img(seaborn.plt))


def show_osd_pg_histo(report, cluster):
    vals = [osd.pg_count for osd in cluster.osds if osd.pg_count is not None]

    seaborn.plt.clf()
    _, ax = seaborn.plt.subplots(figsize=(6, 4))
    plot_histo(vals, left=min(vals) * 0.9, right=max(vals) * 1.1, ax=ax)
    seaborn.plt.tight_layout()
    report.add_block(6, "OSD PG", get_img(seaborn.plt))


def show_osd_load(report, cluster, max_xbins=25):
    usage = get_resource_usage(cluster)

    seaborn.plt.clf()
    hm_vals, ranges = hmap_from_2d(usage.ceph_data_dev_wbytes, max_xbins=max_xbins)
    hm_vals /= 1024. ** 2
    plot_hmap_with_y_histo(hm_vals, ranges)
    seaborn.plt.tight_layout()
    img = get_img(seaborn.plt)
    if usage.colocated_journals:
        report.add_block(6, "OSD data/journal write (MiBps)", img)
    else:
        report.add_block(6, "OSD data write (MiBps)", img)

    seaborn.plt.clf()
    data, chunk_ranges = hmap_from_2d(usage.ceph_data_dev_wio, max_xbins=max_xbins)
    plot_hmap_with_y_histo(data, chunk_ranges)
    seaborn.plt.tight_layout()
    img = get_img(seaborn.plt)
    if usage.colocated_journals:
        report.add_block(6, "OSD data/journal write IOPS", img)
    else:
        report.add_block(6, "OSD data write IOPS", img)

    seaborn.plt.clf()
    data, chunk_ranges = hmap_from_2d(usage.ceph_data_dev_qd, max_xbins=max_xbins)
    plot_hmap_with_y_histo(data, chunk_ranges)
    seaborn.plt.tight_layout()
    img = get_img(seaborn.plt)
    if usage.colocated_journals:
        report.add_block(6, "OSD data/journal QD", img)
    else:
        report.add_block(6, "OSD data write QD", img)

    if not usage.colocated_journals:
        seaborn.plt.clf()
        hm_vals, ranges = hmap_from_2d(usage.ceph_j_dev_wbytes, max_xbins=max_xbins)
        hm_vals /= 1024. ** 2
        plot_hmap_with_y_histo(hm_vals, ranges)
        seaborn.plt.tight_layout()
        report.add_block(6, "OSD journal write (MiBps)", get_img(seaborn.plt))

        seaborn.plt.clf()
        data, chunk_ranges = hmap_from_2d(usage.ceph_j_dev_wio, max_xbins=max_xbins)
        plot_hmap_with_y_histo(data, chunk_ranges)
        seaborn.plt.tight_layout()
        report.add_block(6, "OSD journal write IOPS", get_img(seaborn.plt))

        seaborn.plt.clf()
        data, chunk_ranges = hmap_from_2d(usage.ceph_j_dev_qd, max_xbins=max_xbins)
        plot_hmap_with_y_histo(data, chunk_ranges)
        seaborn.plt.tight_layout()
        report.add_block(6, "OSD journal QD", get_img(seaborn.plt))


def show_osd_lat_heatmaps(report, cluster, max_xbins=25):
    for field, header in (('journal_latency', "Journal latency, ms"),
                          ('apply_latency',  "Apply latency, ms"),
                          ('commitcycle_latency', "Commit cycle latency, ms")):

        min_perf_len = min(osd.osd_perf[field].size for osd in cluster.osds)
        lats = numpy.concatenate([osd.osd_perf[field][:min_perf_len] for osd in cluster.osds])
        lats = lats.reshape((len(cluster.osds), min_perf_len))

        seaborn.plt.clf()
        hmap, ranges = hmap_from_2d(lats, max_xbins=max_xbins, noval=NO_VALUE)
        hmap *= 1000
        plot_hmap_with_y_histo(hmap, ranges)
        seaborn.plt.tight_layout()
        report.add_block(6, header, get_img(seaborn.plt))


def show_osd_ops_boxplot(report, cluster):
    ops = []
    for osd in cluster.osds:
        fd = cluster.storage.raw.get_fd(osd.historic_ops_storage_path)
        ops.extend(load_ops_from_fd(fd))
    ops.sort(key=lambda x: x.initiated_at)

    values = []
    for name in ALL_STAGES_IN_ORDER:
        stage_idx = ALL_STAGES.index(name)
        # convert us -> ms
        op_timings = numpy.array([op.stages[stage_idx] for op in ops]) / 1000.
        values.append(op_timings)

    # cut upper 2%
    for arr in values:
        mmax = numpy.percentile(arr, 98)
        numpy.clip(arr, 0, mmax, arr)

    seaborn.plt.clf()
    _, ax = seaborn.plt.subplots(figsize=(12, 9))
    seaborn.boxplot(data=values, ax=ax)
    names = [STAGES_PRINTABLE_NAMES[name] for name in ALL_STAGES_IN_ORDER]
    ax.set_xticklabels(names, rotation=35)
    report.add_block(12, "Ceph OPS time", get_img(seaborn.plt))


def plot_crush(report, cluster):
    import networkx
    import matplotlib.pyplot as plt

    G = networkx.DiGraph()

    G.add_node("ROOT")
    for i in xrange(5):
        G.add_node("Child_%i" % i)
        G.add_node("Grandchild_%i" % i)
        G.add_node("Greatgrandchild_%i" % i)

        G.add_edge("ROOT", "Child_%i" % i)
        G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
        G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)

    # write dot file to use with graphviz
    # run "dot -Tpng test.dot >test.png"
    # nx.write_dot(G,'test.dot')

    # same layout using matplotlib with no labels
    # plt.title('draw_networkx')
    plt.clf()
    pos = networkx.fruchterman_reingold_layout(G)
    networkx.draw(G, pos, with_labels=False, arrows=False)
    # plt.savefig('nx_test.png')
    report.add_block(4, "nodes", get_img(plt))
