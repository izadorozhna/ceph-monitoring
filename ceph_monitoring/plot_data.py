import warnings

import numpy
from matplotlib import gridspec
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

from .cluster import NO_VALUE
from .perf_parser import (auto_edges, load_ops_from_fd, ALL_STAGES_IN_ORDER, ALL_STAGES,
                          STAGES_PRINTABLE_NAMES, get_img)
from .resource_usage import get_resource_usage


def float2str3dig(val):
    if val < 0:
        return '-' + float2str3dig(-val)

    if val < 0.1:
        return "{0:.2e}".format(val)

    if val < 1:
        return "{0:.3f}".format(val)

    if val < 1000 and (isinstance(val, int) or val >= 100):
        return str(int(val))

    if val < 10:
        return "{0:1.2f}".format(val)

    if val < 100:
        return "{0:2.1f}".format(val)

    if val < 10000:
        return str(int(val) // 10 * 10)

    if val < 100000:
        return str(int(val) // 100 * 100)

    if val < 1000000:
        return str(int(val) // 1000 * 1000)

    return "{0:.2e}".format(val)


def hmap_from_2d(data, max_xbins=25, noval=None):
    """

    :param data: 2D array of input data, [num_measurments * num_objects], each row contains single measurement for
                 every object
    :param max_xbins: maximum time ranges to split to
    :param noval: Optional, if not None - faked value, which mean "no measurement done, or measurement invalid",
                  removed from results array
    :return: pair if 1D array of all valid values and list of pairs of indexes in this
             array, which belong to one time slot
    """

    # calculate how many 'data' rows fit into single output interval
    num_points = data.shape[1]
    step = int(round(float(num_points) / max_xbins + 0.5))

    # drop last columns, as in other case it's hard to make heatmap looks correctly
    idxs = range(0, num_points, step)

    # list of chunks of data array, which belong to same result slot, with noval removed
    result_chunks = []
    for bg, en in zip(idxs[:-1], idxs[1:]):
        block_data = data[:, bg:en]
        filtered = block_data.reshape(block_data.size)
        if noval is not None:
            filtered = filtered[filtered != noval]
        result_chunks.append(filtered)

    # generate begin:end indexes of chunks in chunks concatenated array
    chunk_lens = numpy.cumsum(map(len, result_chunks))
    bin_ranges = zip(chunk_lens[:-1], chunk_lens[1:])

    return numpy.concatenate(result_chunks), bin_ranges


def process_heatmap_data(values, bin_ranges, cut_percentile=(0.02, 0.98), ybins=20, log_edges=True):
    """
    Transform 1D input array of values into 2D array of histograms.
    All data from 'values' which belong to same region, provided by 'bin_ranges' array
    goes to one histogram

    :param values: 1D array of values - this array gets modified if cut_percentile provided
    :param bin_ranges: List of pairs begin:end indexes in 'values' array for items, belong to one section
    :param cut_percentile: Options, pair of two floats - clip items from 'values', which not fit into provided
                           percentiles (100% == 1.0)
    :param ybins: ybin count for histogram
    :param log_edges: use logarithmic scale for histogram bins edges
    :return: 2D array of heatmap
    """

    nvalues = [values[idx1:idx2] for idx1, idx2 in bin_ranges]

    if cut_percentile:
        mmin, mmax = numpy.percentile(values, (cut_percentile[0] * 100, cut_percentile[1] * 100))
        numpy.clip(values, mmin, mmax, values)
    else:
        mmin = values.min()
        mmax = values.max()

    if log_edges:
        bins = auto_edges(values, bins=ybins, round_base=None)
    else:
        bins = numpy.linspace(mmin, mmax, ybins + 1)

    return numpy.array([numpy.histogram(src_line, bins)[0] for src_line in nvalues]), bins


def plot_heatmap(hmap_vals, chunk_ranges, ax):
    heatmap, bins = process_heatmap_data(hmap_vals, chunk_ranges)
    labels = map(float2str3dig, (bins[:-1] + bins[1:]) / 2)
    ax = seaborn.heatmap(heatmap[:,::-1].T, xticklabels=False, cmap="Blues", ax=ax)
    ax.set_yticklabels(labels, rotation='horizontal')
    return ax, bins


def plot_histo(vals, bins=None, kde=False, left=None, right=None, ax=None):
    ax = seaborn.distplot(vals, bins=bins, ax=ax, kde=kde)
    ax.set_yticklabels([])

    if left is not None or right is not None:
        ax.set_xlim(left=left, right=right)

    return ax


def plot_hmap_with_y_histo(data2d, chunk_ranges, figsize=(9, 4), boxes=3, kde=False):
    fig = seaborn.plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, boxes)
    ax = fig.add_subplot(gs[0, :boxes - 1])

    _, bins = plot_heatmap(data2d, chunk_ranges, ax=ax)

    ax2 = fig.add_subplot(gs[0, boxes - 1])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_ylim(top=bins[-1], bottom=bins[0])
    seaborn.distplot(data2d, bins=bins, ax=ax2, kde=kde, vertical=True)
    return ax, ax2


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
    report.add_block(6, "OSD data write (MiBps)", get_img(seaborn.plt))

    seaborn.plt.clf()
    data, chunk_ranges = hmap_from_2d(usage.ceph_data_dev_wio, max_xbins=max_xbins)
    plot_hmap_with_y_histo(data, chunk_ranges)
    seaborn.plt.tight_layout()
    report.add_block(6, "OSD data write IOPS", get_img(seaborn.plt))

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
    report.add_block(12, "Cepg OPS time", get_img(seaborn.plt))
