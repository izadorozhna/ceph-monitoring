import logging
import warnings
from io import BytesIO
from typing import Callable, Any, AnyStr
from typing_extensions import Protocol

import numpy
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

import networkx
from matplotlib import pyplot

from cephlib.plot import plot_histo, hmap_from_2d, plot_hmap_with_histo

from .cluster import NO_VALUE, CephInfo
from .osd_ops import calc_stages_time, iter_ceph_ops, ALL_STAGES
from .perf_parser import STAGES_PRINTABLE_NAMES
from .resource_usage import get_hdd_resource_usage


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger('cephlib.report')

# ----------------------------------------------------------------------------------------------------------------------


class ReportProto(Protocol):
    def add_block(self, width: int, name: str, img: AnyStr):
        ...


def get_img(plt: Any, format: str = 'svg') -> AnyStr:
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


def show_osd_used_space_histo(report: ReportProto, cluster: CephInfo, min_osd: int = 3):
    vals = [(100 - osd.free_perc) for osd in cluster.osds if osd.free_perc is not None]
    if len(vals) >= min_osd:
        img = plot_img(plot_histo, numpy.array(vals), left=0, right=100, ax=True)
        report.add_block(6, "OSD used space (GiB)", img)


def show_osd_pg_histo(report: ReportProto, cluster: CephInfo):
    vals = numpy.array([osd.pg_count for osd in cluster.osds if osd.pg_count is not None])
    img = plot_img(plot_histo, vals, left=min(vals) * 0.9, right=max(vals) * 1.1, ax=True)
    report.add_block(6, "OSD PG", img)


def show_osd_load(report: ReportProto, cluster: CephInfo, max_xbins: int = 25):
    logger.info("Plot osd load")
    usage = get_hdd_resource_usage(cluster)

    hm_vals, ranges = hmap_from_2d(usage.ceph_data_dev_wbytes, max_xbins=max_xbins)
    hm_vals /= 1024. ** 2
    img = plot_img(plot_hmap_with_histo, hm_vals, ranges)

    if usage.colocated_journals:
        report.add_block(6, "OSD data/journal write (MiBps)", img)
    else:
        report.add_block(6, "OSD data write (MiBps)", img)

    pyplot.clf()
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

        pyplot.clf()
        data, chunk_ranges = hmap_from_2d(usage.ceph_j_dev_qd, max_xbins=max_xbins)
        img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
        report.add_block(6, "OSD journal QD", img)


def show_osd_lat_heatmaps(report: ReportProto, cluster: CephInfo, max_xbins: int = 25):
    logger.info("Plot osd latency heatmaps")
    for field, header in (('journal_latency', "Journal latency, ms"),
                          ('apply_latency',  "Apply latency, ms"),
                          ('commitcycle_latency', "Commit cycle latency, ms")):

        if field not in cluster.osds[0].osd_perf:
            continue

        min_perf_len = min(osd.osd_perf[field].size for osd in cluster.osds)

        lats = numpy.concatenate([osd.osd_perf[field][:min_perf_len] for osd in cluster.osds])
        lats = lats.reshape((len(cluster.osds), min_perf_len))

        print(field)
        hmap, ranges = hmap_from_2d(lats, max_xbins=max_xbins, noval=NO_VALUE)

        if len(hmap) != 0:
            img = plot_img(plot_hmap_with_histo, hmap, ranges)
            report.add_block(6, header, img)


def show_osd_ops_boxplot(report: ReportProto, cluster: CephInfo):
    logger.info("Loading ceph historic ops data")
    ops = []

    for osd in cluster.osds:
        fd = cluster.storage.raw.get_fd(osd.historic_ops_storage_path, mode='rb')
        ops.extend(iter_ceph_ops(fd))

    stimes = {}
    for op in ops:
        for name, vl in calc_stages_time(op).items():
            stimes.setdefault(name, []).append(vl)

    logger.info("Plotting historic boxplots")

    stage_times_map = {name: numpy.array(values) / 1000. for name, values in stimes.items()}

    # cut upper 2%
    for arr in stage_times_map.values():
        numpy.clip(arr, 0, numpy.percentile(arr, 98), arr)

    stage_names, stage_times = zip(*sorted(stage_times_map.items(), key=lambda x: ALL_STAGES.index(x[0])))
    pyplot.clf()
    plt, ax = pyplot.subplots(figsize=(12, 9))

    seaborn.boxplot(data=stage_times, ax=ax)
    ax.set_yscale("log")
    ax.set_xticklabels([STAGES_PRINTABLE_NAMES[name] for name in stage_names], size=14, rotation=35)
    report.add_block(12, "Ceph OPS time", get_img(plt))
    logger.info("Done")


def plot_crush(report: ReportProto, cluster: CephInfo):
    logger.info("Plot crushmap")

    G = networkx.DiGraph()

    G.add_node("ROOT")
    for i in range(5):
        G.add_node("Child_%i" % i)
        G.add_node("Grandchild_%i" % i)
        G.add_node("Greatgrandchild_%i" % i)

        G.add_edge("ROOT", "Child_%i" % i)
        G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
        G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)

    pyplot.clf()
    pos = networkx.fruchterman_reingold_layout(G)
    networkx.draw(G, pos, with_labels=False, arrows=False)
    report.add_block(4, "nodes", get_img(pyplot))
