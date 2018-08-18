import logging
import warnings
import subprocess
from io import BytesIO
from typing import Callable, Any, AnyStr, Optional, Tuple, List

import distutils.spawn
from typing_extensions import Protocol

import numpy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

from cephlib.plot import plot_histo, hmap_from_2d, plot_hmap_with_histo
from cephlib.crush import Node


from .cluster import Cluster
from .ceph_loader import NO_VALUE, CephInfo
from .osd_ops import calc_stages_time, iter_ceph_ops, ALL_STAGES
from .perf_parser import STAGES_PRINTABLE_NAMES
from .visualize_utils import perf_info_required, plot
from .report import Report

# from .resource_usage import get_hdd_resource_usage


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger('cephlib.report')

# ----------------------------------------------------------------------------------------------------------------------



class ReportProto(Protocol):
    def add_block(self, tag: str, name: str, img: AnyStr):
        ...


def get_img(plt: Any, format: str = 'svg') -> AnyStr:
    "returns svg image"
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


@plot
def show_osd_used_space_histo(ceph: CephInfo) -> Optional[Tuple[str, Any]]:
    min_osd: int = 3
    vals = [(100 - osd.free_perc) for osd in ceph.osds if osd.free_perc is not None]
    if len(vals) >= min_osd:
        img = plot_img(plot_histo, numpy.array(vals), left=0, right=100, ax=True)
        return "OSD used space (GiB)", img


def get_histo_img(vals: numpy.ndarray) -> str:
    return plot_img(plot_histo, vals, left=min(vals) * 0.9, right=max(vals) * 1.1, ax=True)


def get_kde_img(vals: numpy.ndarray) -> str:
    fig, ax = pyplot.subplots(figsize=(6, 4), tight_layout=True)
    seaborn.kdeplot(vals, ax=ax, clip=(vals.min(), vals.max()))
    return get_img(fig)


@plot
@perf_info_required
def show_osd_load(report: ReportProto, cluster: Cluster, ceph: CephInfo):
    max_xbins: int = 25
    logger.info("Plot osd load")
    usage = get_hdd_resource_usage(cluster.perf_data, ceph.osds)

    hm_vals, ranges = hmap_from_2d(usage.data.wbytes, max_xbins=max_xbins)
    hm_vals /= 1024. ** 2
    img = plot_img(plot_hmap_with_histo, hm_vals, ranges)

    logger.warning("Show only data disk load for now")

    report.add_block("OSD_write_MiBps", "OSD data write (MiBps)", img)

    pyplot.clf()
    data, chunk_ranges = hmap_from_2d(usage.data.wio, max_xbins=max_xbins)
    img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
    report.add_block("OSD_write_IOPS", "OSD data write IOPS", img)

    data, chunk_ranges = hmap_from_2d(usage.data.qd, max_xbins=max_xbins)
    img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
    report.add_block("OSD_write_QD", "OSD data write QD", img)

    # if not usage.colocated_journals:
    #     hm_vals, ranges = hmap_from_2d(usage.ceph_j_dev_wbytes, max_xbins=max_xbins)
    #     hm_vals /= 1024. ** 2
    #     img = plot_img(plot_hmap_with_histo, hm_vals, ranges)
    #     report.add_block(6, "OSD journal write (MiBps)", img)
    #
    #     data, chunk_ranges = hmap_from_2d(usage.ceph_j_dev_wio, max_xbins=max_xbins)
    #     img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
    #     report.add_block(6, "OSD journal write IOPS", img)
    #
    #     pyplot.clf()
    #     data, chunk_ranges = hmap_from_2d(usage.ceph_j_dev_qd, max_xbins=max_xbins)
    #     img = plot_img(plot_hmap_with_histo, data, chunk_ranges)
    #     report.add_block(6, "OSD journal QD", img)


@plot
@perf_info_required
def show_osd_lat_heatmaps(report: ReportProto, ceph: CephInfo):
    max_xbins: int = 25
    logger.info("Plot osd latency heatmaps")

    for field, header in (('journal_latency', "Journal latency, ms"),
                          ('apply_latency',  "Apply latency, ms"),
                          ('commitcycle_latency', "Commit cycle latency, ms")):

        if field not in ceph.osds[0].osd_perf:
            continue

        min_perf_len = min(osd.osd_perf[field].size for osd in ceph.osds)

        lats = numpy.concatenate([osd.osd_perf[field][:min_perf_len] for osd in ceph.osds])
        lats = lats.reshape((len(ceph.osds), min_perf_len))

        print(field)
        hmap, ranges = hmap_from_2d(lats, max_xbins=max_xbins, noval=NO_VALUE)

        if len(hmap) != 0:
            img = plot_img(plot_hmap_with_histo, hmap, ranges)
            report.add_block("OSD_" + field, header, img)


@plot
@perf_info_required
def show_osd_ops_boxplot(ceph: CephInfo) -> Tuple[str, Any]:
    logger.info("Loading ceph historic ops data")
    ops = []

    for osd in ceph.osds:
        fd = ceph.storage.raw.get_fd(osd.historic_ops_storage_path, mode='rb')
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
    return "Ceph OPS time", get_img(plt)

# def show_crush(ceph: CephInfo) -> Tuple[str, Any]:
#     logger.info("Plot crushmap")
#
#     G = networkx.DiGraph()
#
#     G.add_node("ROOT")
#     for i in range(5):
#         G.add_node("Child_%i" % i)
#         G.add_node("Grandchild_%i" % i)
#         G.add_node("Greatgrandchild_%i" % i)
#
#         G.add_edge("ROOT", "Child_%i" % i)
#         G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
#         G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)
#
#     pyplot.clf()
#     pos = networkx.fruchterman_reingold_layout(G)
#     networkx.draw(G, pos, with_labels=False, arrows=False)
#     return "nodes", get_img(pyplot)


@plot
def plot_crush_rules(ceph: CephInfo, report: Report):
    neato_path = distutils.spawn.find_executable('neato')

    colormap = numpy.array([(0.8584083044982699, 0.9134486735870818, 0.9645674740484429),
                            (0.7309496347558632, 0.8394771241830065, 0.9213225682429834),
                            (0.5356862745098039, 0.746082276047674, 0.8642522106881968),
                            (0.32628988850442137, 0.6186236063052672, 0.802798923490965),
                            (0.16696655132641292, 0.48069204152249134, 0.7291503267973857),
                            (0.044059976931949255, 0.3338869665513264, 0.6244521337946944)])

    colormap_i = (colormap * 256).astype(numpy.uint8)

    if not neato_path:
        logger.warning("Neato tool not found, probably graphviz package is not installed. " +
                       "Rules graphs would not be generated")
    else:
        for rule in ceph.crush.rules.values():
            root_node = ceph.crush.copy_tree_for_rule(rule)
            if not root_node:
                continue

            max_node_w = max(node.weight for node in root_node.iter_nodes('host'))

            def get_color(w: float) -> Tuple[int, int, int]:
                if abs(w - max_node_w) < 1e-5:
                    return tuple(colormap_i[-1])
                idx = w / max_node_w * (len(colormap) - 1)
                lidx = int(idx)
                c1 = 1.0 - (idx - lidx)
                return (c1 * colormap[lidx] + (1.0 - c1) * colormap[lidx + 1]).astype(numpy.uint8)

            # make dot
            def make_dot(node: Node) -> List[str]:
                if node.type == 'host':
                    r, g, b = get_color(node.weight)
                    bgcolor = f"#{r:02X}{g:02X}{b:02X}A0"
                else:
                    bgcolor = "#FFFFFFFF"

                def to_dot_name(name: str) -> str:
                    return name.replace("-", "_").replace('.', '_')

                yield f'{to_dot_name(node.name)} [label="{node.name}\\n{node.weight:.2f}"' + \
                      f' fillcolor="{bgcolor}" style=filled];'

                for ch in node.childs:
                    if ch.type != 'osd':
                        yield from make_dot(ch)
                        yield f"{to_dot_name(node.name)} -> {to_dot_name(ch.name)};"

            dot = f"digraph {rule.name} {{\n    overlap = scale;\n    "
            dot += "\n    ".join(make_dot(root_node)) + "\n}"
            svg = subprocess.check_output("neato -Tsvg", shell=True, input=dot.encode('utf8')).decode("utf8")
            svg = svg[svg.index("<svg "):]
            report.add_block(f"crush_svg_{rule.name}", None, svg, f"Tree for '{rule.name}'")