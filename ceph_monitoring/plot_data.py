import warnings

import numpy
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

from .cluster import NO_VALUE
from .perf_parser import (process_heatmap_data, get_heatmap_img, auto_edges, group_times, load_ops_from_fd,
                          ALL_STAGES_IN_ORDER, ALL_STAGES, STAGES_PRINTABLE_NAMES, get_img)
from .resource_usage import load_resource_usage


def float2str3dig(val):
    if val < 0:
        return '-' + float2str3dig(-val)

    if val < 0.1:
        return "{0:.2e}".format(val)

    if val < 1:
        return "{0:.3f}".format(val)

    if val < 10:
        if val - int(val) < 1E-2:
            return str(int(val))
        return "{0:1.2f}".format(val)

    if val < 100:
        if val - int(val) < 1E-1:
            return str(int(val))
        return "{0:2.1f}".format(val)

    if val < 1000:
        return str(int(val))

    if val < 10000:
        return str(int(val) // 10 * 10)

    if val < 100000:
        return str(int(val) // 100 * 100)

    if val < 1000000:
        return str(int(val) // 1000 * 1000)

    return "{0:.2e}".format(val)


def hmap_from_2d(data, max_xbins=25, noval=None):
    num_points = len(data[0])
    step = int(round(float(num_points) / max_xbins + 0.5))
    idxs = range(0, num_points, step) + [num_points]
    csize = 0
    bin_ranges = []
    hmap_vals = numpy.empty((data.size,), dtype=data.dtype)
    for bg, en in zip(idxs[:-1], idxs[1:]):
        block_data = data[:, bg:en]
        filtered = block_data.reshape(block_data.size)
        if noval is not None:
            filtered = filtered[filtered != noval]
        hmap_vals[csize:csize + filtered.size] = filtered
        bin_ranges.append((csize, csize + filtered.size))
        csize += filtered.size
    return hmap_vals[:csize], bin_ranges


def plot_heatmap(hmap_vals, bins_ranges, figsize=(6, 4), tight=True):
    heatmap = process_heatmap_data(hmap_vals, bins_ranges)
    labels = [float2str3dig((beg + end) / 2) for beg, end in bins_ranges]

    if figsize:
        _, ax = seaborn.plt.subplots(figsize=figsize)
    else:
        ax = None

    ax = seaborn.heatmap(heatmap[:,::-1].T, xticklabels=False, cmap="Blues", ax=ax)
    ax.set_yticklabels(labels, rotation='horizontal')
    if tight:
        seaborn.plt.tight_layout()


def plot_histo(vals, bins=None, figsize=(6, 4), kde=False, left=None, right=None, tight=True):

    if figsize:
        _, ax = seaborn.plt.subplots(figsize=figsize)
    else:
        ax = None

    ax = seaborn.distplot(vals, bins=bins, ax=ax, kde=kde)

    ax.set_yticklabels([])
    if left is not None or right is not None:
        ax.set_xlim(left=left, right=right)

    if tight:
        seaborn.plt.tight_layout()


def show_osd_used_space_histo(report, cluster):
    vals = [100 - osd.free_perc
            for osd in cluster.osds
            if osd.data_stor_stats is not None and osd.data_stor_stats.get('used') is not None]

    seaborn.plt.clf()
    plot_histo(vals, left=0, right=100)
    report.add_block(6, "OSD used space (GiB)", get_img(seaborn.plt))


def show_osd_pg_histo(report, cluster):
    vals = [osd.pg_count for osd in cluster.osds if osd.pg_count is not None]

    mn = min(vals) * 0.9
    mx = max(vals) * 1.1

    seaborn.plt.clf()
    plot_histo(vals, left=mn, right=mx)
    report.add_block(6, "OSD PG", get_img(seaborn.plt))


def show_agg_datadisk_iops_heatmaps(report, cluster, max_xbins=25):
    usage = load_resource_usage(cluster)
    hmap_vals, bins_ranges = hmap_from_2d(usage.ceph_data_dev_wio, max_xbins=max_xbins)
    seaborn.plt.clf()
    plot_heatmap(hmap_vals, bins_ranges)
    report.add_block(4, "OSD data IOPS", get_img(seaborn.plt))


def show_agg_datadisk_bw_heatmaps(report, cluster, max_xbins=25):
    usage = load_resource_usage(cluster)
    hmap_vals, bins_ranges = hmap_from_2d(usage.ceph_data_dev_wbytes, max_xbins=max_xbins)
    seaborn.plt.clf()
    plot_heatmap(hmap_vals, bins_ranges)
    report.add_block(4, "OSD data BW", get_img(seaborn.plt))


def show_osd_lat_heatmaps(report, cluster, max_xbins=25):
    for field, header in (('journal_latency', "Journal latency:"),
                          ('apply_latency',  "Apply latency"),
                          ('commitcycle_latency', "Commit cycle latency")):

        max_perf_len = max(len(osd.osd_perf[field]) for osd in cluster.osds)
        lats = numpy.empty((len(cluster.osds), max_perf_len))

        for lat_line, osd in zip(lats, cluster.osds):
            arr = osd.osd_perf[field]
            lat_line[:arr.size] = arr
            lat_line[arr.size:] = NO_VALUE

        hmap_vals, bins_ranges = hmap_from_2d(lats, max_xbins=max_xbins, noval=NO_VALUE)
        seaborn.plt.clf()
        plot_heatmap(hmap_vals, bins_ranges)
        report.add_block(4, header, get_img(seaborn.plt))


def show_agg_osd_lat_histo(report, cluster, max_xbins=25, cut_percentile=(0.01, 0.99)):
    for field, header in (('journal_latency', "Journal latency:"),
                          ('apply_latency',  "Apply latency"),
                          ('commitcycle_latency', "Commit cycle latency")):
        max_perf_len = max(len(osd.osd_perf[field]) for osd in cluster.osds)
        lats = numpy.empty((len(cluster.osds) * max_perf_len))
        curr_count = 0
        for osd in cluster.osds:
            vl = osd.osd_perf[field]
            vl = vl[vl != NO_VALUE]
            lats[curr_count: curr_count + vl.size] = vl
            curr_count += vl.size

        lats = lats[:curr_count]

        mmin, mmax = numpy.percentile(lats, (cut_percentile[0] * 100, cut_percentile[1] * 100))
        lats[lats > mmax] = mmax
        lats[lats < mmin] = mmin

        bins = auto_edges(lats, bins=max_xbins, round_base=None)
        seaborn.plt.clf()
        plot_histo(lats, bins=bins)
        report.add_block(4, header + " histo", get_img(seaborn.plt))


def show_agg_osd_load_histo(report, cluster):
    for field, header in (('journal_ops', "Journal iops"), ('journal_bytes',  "Journal write bps")):
        max_perf_len = max(len(osd.osd_perf[field]) for osd in cluster.osds)
        load = numpy.empty((len(cluster.osds) * max_perf_len))

        curr_count = 0
        for osd in cluster.osds:
            vl = osd.osd_perf[field]
            load[curr_count: curr_count + vl.size] = vl
            curr_count += vl.size
        load = load[:curr_count]

        seaborn.plt.clf()
        plot_histo(load)
        report.add_block(4, header, get_img(seaborn.plt))


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

    # cut upper 5%
    for arr in values:
        arr.sort()
        if len(arr) >= 100:
            upper_idx = int(len(arr) * 0.95)
            arr[upper_idx:] = arr[upper_idx]

    seaborn.plt.clf()
    _, ax = seaborn.plt.subplots(figsize=(12, 9))
    seaborn.boxplot(data=values, ax=ax)
    names = [STAGES_PRINTABLE_NAMES[name] for name in ALL_STAGES_IN_ORDER]
    ax.set_xticklabels(names, rotation=35)
    report.add_block(12, "Cepg OPS time", get_img(seaborn.plt))
