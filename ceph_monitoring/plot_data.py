import logging
import warnings
from io import BytesIO
from typing import Callable, Any

import numpy
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

from matplotlib import pyplot

from .cluster import NO_VALUE
from .perf_parser import (load_jsops_from_fd, OSD_OP_PRIMARY_STEPS, OSD_OP_PRIMARY_STEPS_PARENT,
                          STAGES_PRINTABLE_NAMES, OSD_OP_PRIMARY_FSTEPS)
from .resource_usage import get_resource_usage

from cephlib.plot import plot_histo, hmap_from_2d, plot_hmap_with_histo


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger('cephlib.report')

# ----------------------------------------------------------------------------------------------------------------------

def get_img(plt, format='svg'):
    bio = BytesIO()
    if format in ('png', 'jpg'):
        plt.savefig(bio, format=format)
        return bio.getvalue()
    elif format == 'svg':
        plt.savefig(bio, format='svg')
        img_start = b"<!-- Created with matplotlib (http://matplotlib.org/) -->"
        return bio.getvalue().split(img_start, 1)[1].decode("utf8")


def plot_img(func: Callable, *args, **kwargs) -> Any:
    fs = kwargs.pop('figsize', (6, 4))
    tl = kwargs.pop('tight_layout', True)

    if kwargs.pop('ax', False):
        fig, ax = pyplot.subplots(figsize=fs, tight_layout=tl)
        func(ax, *args, **kwargs)
    else:
        fig = pyplot.figure(figsize=fs, tight_layout=tl)
        func(fig, *args, **kwargs)

    return get_img(fig)


# def get_heatmap_img(heatmap, bins_ranges, clear=True, figsize=None):
#     labels = ["{:.2e}".format((beg + end) / 2) for beg, end in bins_ranges]
#     if figsize:
#         _, ax = seaborn.plt.subplots(figsize=figsize)
#     else:
#         ax = None
#     ax = seaborn.heatmap(heatmap[:,::-1].T, xticklabels=False, cmap="Blues", ax=ax)
#     ax.set_yticklabels(labels, rotation='horizontal')
#     res = get_img(seaborn.plt)
#     if clear:
#         seaborn.plt.clf()
#     return res
#
#
def show_osd_used_space_histo(report, cluster, min_osd=3):
    vals = [(100 - osd.free_perc) for osd in cluster.osds if osd.free_perc is not None]
    if len(vals) >= min_osd:
        img = plot_img(plot_histo, numpy.array(vals), left=0, right=100, ax=True)
        report.add_block(6, "OSD used space (GiB)", img)


def show_osd_pg_histo(report, cluster):
    vals = numpy.array([osd.pg_count for osd in cluster.osds if osd.pg_count is not None])
    img = plot_img(plot_histo, vals, left=min(vals) * 0.9, right=max(vals) * 1.1, ax=True)
    report.add_block(6, "OSD PG", img)


def show_osd_load(report, cluster, max_xbins=25):
    logger.info("Plot osd load")
    usage = get_resource_usage(cluster)

    hm_vals, ranges = hmap_from_2d(usage.ceph_data_dev_wbytes, max_xbins=max_xbins)
    hm_vals /= 1024. ** 2
    img = plot_img(plot_hmap_with_histo, hm_vals, ranges)

    if usage.colocated_journals:
        report.add_block(6, "OSD data/journal write (MiBps)", img)
    else:
        report.add_block(6, "OSD data write (MiBps)", img)

    seaborn.plt.clf()
    data, chunk_ranges = hmap_from_2d(usage.ceph_data_dev_wio, max_xbins=max_xbins)
    img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
    msg = "OSD data/journal write IOPS" if usage.colocated_journals else "OSD data write IOPS"
    report.add_block(6, msg, img)

    data, chunk_ranges = hmap_from_2d(usage.ceph_data_dev_qd, max_xbins=max_xbins)
    img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
    msg = "OSD data/journal QD" if usage.colocated_journals else "OSD data write QD"
    report.add_block(6, msg, img)

    if not usage.colocated_journals:
        hm_vals, ranges = hmap_from_2d(usage.ceph_j_dev_wbytes, max_xbins=max_xbins)
        hm_vals /= 1024. ** 2
        img = plot_img(plot_hmap_with_histo, hm_vals, ranges)
        report.add_block(6, "OSD journal write (MiBps)", img)

        data, chunk_ranges = hmap_from_2d(usage.ceph_j_dev_wio, max_xbins=max_xbins)
        img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
        report.add_block(6, "OSD journal write IOPS", img)

        seaborn.plt.clf()
        data, chunk_ranges = hmap_from_2d(usage.ceph_j_dev_qd, max_xbins=max_xbins)
        img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
        report.add_block(6, "OSD journal QD", img)


def show_osd_lat_heatmaps(report, cluster, max_xbins=25):
    logger.info("Plot osd latency heatmaps")
    for field, header in (('journal_latency', "Journal latency, ms"),
                          ('apply_latency',  "Apply latency, ms"),
                          ('commitcycle_latency', "Commit cycle latency, ms")):

        if field not in cluster.osds[0].osd_perf:
            continue

        import IPython
        IPython.embed()

        min_perf_len = min(osd.osd_perf[field].size for osd in cluster.osds)

        lats = numpy.concatenate([osd.osd_perf[field][:min_perf_len] for osd in cluster.osds])
        lats = lats.reshape((len(cluster.osds), min_perf_len))

        print(field)
        hmap, ranges = hmap_from_2d(lats, max_xbins=max_xbins, noval=NO_VALUE)

        if len(hmap) != 0:
            img = plot_img(plot_hmap_with_histo, hmap, ranges)
            report.add_block(6, header, img)


def show_osd_ops_boxplot(report, cluster):
    # logger.warning("show_osd_ops_boxplot skipped!")

    steps = OSD_OP_PRIMARY_STEPS
    fsteps = OSD_OP_PRIMARY_FSTEPS
    ops = []
    logger.info("Loading ceph historic ops data")
    for osd in cluster.osds:
        fd = cluster.storage.raw.get_fd(osd.historicjs_ops_storage_path, mode='rt')
        ops.extend(load_jsops_from_fd(fd, steps))
    logger.info("Plotting historic boxplots")

    stage_times_map = {name: numpy.array([op[name] for op in ops]) / 1000. for name in steps}
    stage_times_map = {name: stage_times_map[name] - stage_times_map[parent]
                       for name, parent in OSD_OP_PRIMARY_STEPS_PARENT.items()}

    stage_times_map["sub_op_commit_rec"] = numpy.concatenate([stage_times_map["sub_op_commit_rec_0"],
                                                              stage_times_map["sub_op_commit_rec_1"]])
    del stage_times_map["sub_op_commit_rec_0"]
    del stage_times_map["sub_op_commit_rec_1"]

    # cut upper 2%
    for arr in stage_times_map.values():
        numpy.clip(arr, 0, numpy.percentile(arr, 98), arr)

    stage_names, stage_times = zip(*sorted(stage_times_map.items(), key=lambda x: fsteps.index(x[0])))
    pyplot.clf()
    _, ax = pyplot.subplots(figsize=(12, 9))

    ax.boxplot(stage_times, 0, '')
    names = [STAGES_PRINTABLE_NAMES[name] for name in stage_names]
    ax.set_xticklabels(names, rotation=35)
    report.add_block(12, "Ceph OPS time", get_img(seaborn.plt))
    logger.info("Done")


def plot_crush(report, cluster):
    logger.info("Plot crushmap")

    import networkx
    import matplotlib.pyplot as plt

    G = networkx.DiGraph()

    G.add_node("ROOT")
    for i in range(5):
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
