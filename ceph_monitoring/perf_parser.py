from __future__ import print_function
import os
import sys
import glob
import json
import math
import itertools
from StringIO import StringIO as BIO

import numpy

import seaborn
seaborn.set()

import matplotlib.pyplot as plt


from cephlib.sensors_rpc_plugin import CephSensor, pack_ceph_op, CEPH_OP_DESCR_IDX
from cephlib.sensors_rpc_plugin import ALL_STAGES


all_pairs = [(x, y)
             for x, y in itertools.product(ALL_STAGES, ALL_STAGES)
             if x != y]


STAGES_INFO = [
    ("queued_for_pg", "queued"),
    ("reached_pg", "reached pg"),
    ("started", "started"),
    ("commit_queued_for_journal_write", "commit queued\nfor journal"),
    ("waiting_for_subop", "waiting for subops"),
    ("write_thread_in_journal_buffer", "in journal buffer"),
    ("op_commit", "commit"),
    ("op_applied", "applied"),
    ("sub_op_applied", "subop applied"),
    ("journaled_completion_queued", "journaled completion\nqueued"),
    ("subop_commit_rec", "subop commit\nrec"),
    ("subop_apply_rec", "subop apply\nrec"),
    ("commit_sent", "commit send"),
    ("done", "done"),
]


ALL_STAGES_IN_ORDER = [json_name for json_name, _ in STAGES_INFO]
STAGES_PRINTABLE_NAMES = dict(STAGES_INFO)
assert set(STAGES_PRINTABLE_NAMES) == set(ALL_STAGES)


def group_ops(ops):
    assert ops
    res = {stage: [] for stage in ALL_STAGES_IN_ORDER}
    for op in ops:
        for name, tm in zip(ALL_STAGES, op.stages):
            if tm:
                res[name].append(tm)
        res["subop_commit_rec"].extend(op.extra1)
        res["subop_apply_rec"].extend(op.extra2)

    return res


# def validate(op, active, seen_so_far):
#     new = set()
#     seen_so_far2 = seen_so_far.copy()
#     curr = {"subop_commit_rec": op.extra1,
#             "subop_apply_rec": op.extra2}
#
#     for name, tm in zip(stage_json_names, op.stages):
#         if tm:
#             curr[name] = [tm]
#             seen_so_far2.add(name)
#
#     if 'started' not in curr or 'waiting_for_subop' not in curr or curr['started'] > curr['waiting_for_subop']:
#         print(op, curr)
#
#     for gt, ls in active:
#         for gt_val in curr.get(gt, [-1) < curr.get(ls, -1):
#             continue
#         new.add((gt, ls))

    # return new, seen_so_far2


def auto_edges(vals, log_base=2, bins=20, round_base=10):
    lower = numpy.min(vals)
    upper = numpy.max(vals)

    if lower == upper:
        return numpy.array([lower * 0.9, lower * 1.1])

    if lower < 1:
        return numpy.linspace(lower, upper, bins + 1)

    if round_base:
        # to upper and lower poser of 10
        lower_lg = math.floor(math.log(lower) / math.log(round_base)) * (math.log(round_base) / math.log(log_base))
        upper_lg = int(math.log(upper) / math.log(round_base) + 0.99) * (math.log(round_base) / math.log(log_base))
    else:
        lower_lg = math.log(lower) / math.log(log_base)
        upper_lg = math.log(upper) / math.log(log_base)

    return numpy.logspace(lower_lg, upper_lg, bins + 1, base=log_base)


def txthist(values, bin_edges=None, hist_width=70):
    if not bin_edges:
        bin_edges = auto_edges(values)
    hist, _ = numpy.histogram(values, bin_edges)
    hist = hist.astype(numpy.float32) / numpy.amax(hist)
    bins_labels = bin_edges[:-1]
    res = ""
    for vl, rel_count in zip(bins_labels, hist):
        print(vl, rel_count)
        res += " {:1.3e} |  ".format(vl) + '#' * int(hist_width * rel_count) + "\n"
    return res


def group_times(times, x_bins=25):
    """ops must be sorted by initiated_at field"""
    min_time = times[0]
    bin_size = (times[-1] - min_time) / float(x_bins)
    bin_idxs = ((times - min_time) / bin_size).astype(numpy.uint32)
    bin_idxs[bin_idxs >= x_bins] = x_bins - 1

    # now init_at array holds bin idx
    idxs = numpy.arange(x_bins)
    bin_starts = numpy.searchsorted(bin_idxs, idxs)
    bin_ranges = list(zip(bin_starts[:-1], bin_starts[1:]))
    bin_ranges.append((bin_ranges[-1][1], len(times)))

    return bin_ranges



def get_img(plt, format='svg'):
    bio = BIO()
    if format in ('png', 'jpg'):
        plt.savefig(bio, format=format)
        return bio.getvalue()
    elif format == 'svg':
        plt.savefig(bio, format='svg')
        img_start = "<!-- Created with matplotlib (http://matplotlib.org/) -->"
        return bio.getvalue().split(img_start, 1)[1]


def get_heatmap_img(heatmap, bins_ranges, clear=True, figsize=None):
    labels = ["{:.2e}".format((beg + end) / 2) for beg, end in bins_ranges]
    if figsize:
        _, ax = seaborn.plt.subplots(figsize=figsize)
    else:
        ax = None
    ax = seaborn.heatmap(heatmap[:,::-1].T, xticklabels=False, cmap="Blues", ax=ax)
    ax.set_yticklabels(labels, rotation='horizontal')
    res = get_img(seaborn.plt)
    if clear:
        seaborn.plt.clf()
    return res


def test_parse_pack_unpack(js_file):
    for block in open(js_file).read().split(CephSensor.items_sep):
        for js_op in json.loads(block)['Ops']:
            op = CephSensor.parse_op(js_op)
            data = pack_ceph_op(op)
            op2 = CephSensor.unpack_historic(data)[0]
            assert op2.stages == op.stages, "{0.stages!r} != {1.stages!r}\n{2}".format(op, op2, op)
            assert op2.extra1 == op.extra1, "{0.extra1!r} != {1.extra1!r}\n{2}".format(op, op2, op)
            assert op2.extra2 == op.extra2, "{0.extra2!r} != {1.extra2!r}\n{2}".format(op, op2, op)


def load_ops(files, tp='osd_op'):
    ops = []
    for name in files:
        with open(name, 'rb') as fd:
            for op in load_ops_from_fd(fd, tp=tp):
                ops.append(op)
    return ops


def load_ops_from_fd(fd, size=None, tp='osd_op'):
    ops = []
    op_idx = CEPH_OP_DESCR_IDX[tp]

    if size is None:
        cpos = fd.tell()
        fd.seek(0, os.SEEK_END)
        size = fd.tell() - cpos
        fd.seek(cpos)

    for op in CephSensor.unpack_historic_fd(fd, size):
        if op.descr_idx == op_idx:
            ops.append(op)
    return ops


def load_stage(ops, stage_name, no_gaps=True):
    step_idx = ALL_STAGES.index(stage_name)
    values = numpy.array([op.stages[step_idx] for op in ops], dtype=numpy.int32)
    assert not no_gaps or values.min() > 0, "Some ops has no {0!r} stage".format(stage_name)
    return values


def main(argv):
    # load init_at and done

    templ = os.path.join(argv[1], 'perf_monitoring', '*', 'ceph.osd*.perf_dump.json')
    ops = []
    for name in glob.glob(templ):
        agg_ops = []
        for obj in json.load(open(name)):
            dt = obj["filestore"]["apply_latency"]
            agg_ops.append((dt['avgcount'], dt['sum']))

        for idx, cs in enumerate(zip(agg_ops[:-1], agg_ops[1:])):
            (c1, s1), (c2, s2) = cs
            if c2 != c1:
                ops.append((idx, (s2 - s1) / (c2 - c1)))

    ops.sort()
    values = numpy.array([op[1] for op in ops])
    times = numpy.array([op[0] for op in ops])

    vls, bin_ranges = group_values(times, values)
    heatmap = process_heatmap_data(vls, bin_ranges)
    open("/tmp/hmap.svg", "wb").write(get_heatmap_img(heatmap, bin_ranges))
    return

    # templ = os.path.join(argv[1], 'perf_monitoring', '*', 'ceph.osd*.historic.bin')
    # files = glob.glob(templ)
    # ops = load_ops(files)
    # ops.sort(key=lambda x: x.initiated_at)
    # values = load_stage(ops, 'done')
    # times = numpy.array([op.initiated_at // 1000000 for op in ops], dtype=numpy.uint32)
    # seen_so_far = set()
    # curr = all_pairs
    # for op in ops:
    #     if op.descr_idx == CEPH_OP_DESCR_IDX["osd_op"]:
    #         curr, seen_so_far = validate(op, curr, seen_so_far)
    #
    # #
    # lt_than = {}
    # for gt, lt in sorted(curr):
    #     lt_than[lt] = lt_than.get(lt, 0) + 1
    #
    # sorted_curr = sorted(curr, key=lambda pair: [-lt_than.get(pair[1], 0), pair[1]])
    #
    # print("All ops :\n    " + "\n    ".join(sorted(seen_so_far)))
    # for gt, lt in sorted_curr:
    #     if gt in seen_so_far and lt in seen_so_far:
    #         print(lt.replace("\n", ' '), "<=", gt.replace("\n", ' '))
    # # return 1

    grouped_res = group_ops(ops)

    # convert us -> ms
    values = [numpy.array(grouped_res[name]) / 1000. for name in all_stages_in_order]

    # cut upper 5%
    for arr in values:
        arr.sort()
        if len(arr) >= 100:
            upper_idx = int(len(arr) * 0.95)
            arr[upper_idx:] = arr[upper_idx]

    _, ax = plt.subplots(figsize=(14, 14))
    seaborn.boxplot(data=values, ax=ax)
    # seaborn.violinplot(data=values, ax=ax)
    ax.set_xticklabels([user_names[name] for name in all_stages_in_order], rotation=35, size='large')
    plt.show()
    return 0


if __name__ == "__main__":
    # test_parse_pack_unpack(sys.argv[1])
    # compare(sys.argv[1], sys.argv[2])
    exit(main(sys.argv))
