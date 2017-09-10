import sys
import glob
import json
import os.path
import itertools
import collections
from pprint import pprint
from typing import Set, Tuple, Dict, List, Any, Iterator


def main():
    num_ops = collections.defaultdict(int)
    num_evt_per_tp = collections.defaultdict(lambda: collections.defaultdict(int))

    names = glob.glob(os.path.join(sys.argv[1], "perf_monitoring/*ceph-osd*/ceph.osd*.historic_js.json"))
    all_loaded = {name: json.load(open(name)) for name in names}
    show_all_ops_types(all_loaded)
    # import IPython
    # IPython.embed()
    exit(1)
        # for dct in curr:
        #     for op in dct['Ops']:
        #         tp = op['description'].split("(", 1)[0]
        #         num_ops[tp] += 1
        #         evt_mp = num_evt_per_tp[tp]
        #         idx = 1 if tp in ('osd_repop', 'osd_repop_reply') else 2
        #         for evt in op['type_data'][idx]:
        #             evt_mp[evt['event']] += 1

    # pprint.pprint(num_ops)
    # pprint.pprint(num_evt_per_tp)

    find_deps2(all_loaded, "osd_op")


def find_closest_pars(pairs: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    # find pairs, which never breaks by other event (possible direct deps)
    close_pairs = set()  # type: Set[Tuple[str, str]]

    # need all events, that appears on first and second positions, as only such events can be in the middle of any other
    e1s, e2s = zip(*pairs)
    all_events = set(e1s).union(e2s)

    for e1, e2 in pairs:
        # for each pair iterate over all other event and check that pairs with this new env inside
        # are missing in allowed_pairs
        for e3 in all_events:
            if (e1, e3) in pairs and (e3, e2) in pairs:
                break
        else:
            close_pairs.add((e1, e2))

    return close_pairs


def detect_order(pairs: Set[Tuple[str, str]]) -> Iterator[Tuple[str, List[str]]]:
    next_evts = {e1 for e1, _ in pairs}.difference({e2 for _, e2 in pairs})
    processed = set()
    while next_evts:
        new_next = set()
        for evt in next_evts:
            if evt not in processed:
                next_to_evt = [evt2 for evt1, evt2 in pairs if evt1 == evt]
                yield evt, next_to_evt
                new_next.update(next_to_evt)
                processed.add(evt)
        next_evts = new_next


def get_events_for_op(op_events: List[Dict[str, str]]) -> List[str]:
    name_map = {"waiting for rw locks": "waiting_for_rw_locks"}
    names = []  # type: List[str]
    for step in op_events:
        ename = step['event']
        if ename.startswith("waiting for subops from"):
            id_first, id_second = ename[len("waiting for subops from "):].split(",")
            name_map = {
                "sub_op_applied_rec from {}".format(id_first): "sub_op_applied_rec_first",
                "sub_op_applied_rec from {}".format(id_second): "sub_op_applied_rec_second",
                "sub_op_commit_rec from {}".format(id_first): "sub_op_commit_rec_first",
                "sub_op_commit_rec from {}".format(id_second): "sub_op_commit_rec_second"
            }
            ename = "waiting_for_subops"
        else:
            ename = name_map.get(ename, ename)
        names.append(ename)

    return names


def get_uniq_evt_sets_counts(evts: List[Tuple[str, ...]]) -> Dict[Tuple[str], int]:
    uniq = collections.defaultdict(int)  # type: Dict[Tuple[str], int]
    for names in evts:
        uniq[tuple(sorted(names))] += 1
    return uniq


def show_all_ops_types(all_loaded: Dict[str, Any]):
    res = collections.defaultdict(int)
    for dt in all_loaded.values():
        for dct in dt:
            try:
                lst = dct['Ops']
            except:
                lst = dct

            for op in lst:
                op_tp = op['description'].split("(", 1)[0]
                stps = op['description'].split()[-2]
                evts = tuple(sorted(get_events_for_op(op['type_data'][-1])))
                res[(op_tp, stps, evts)] += 1
    pprint(dict(res.items()))


def find_deps2(all_loaded: Dict[str, Any], op_type: str):
    ops = []
    evts = []
    for dt in all_loaded.values():
        for dct in dt:
            for op in dct['Ops']:
                if op['description'].split("(", 1)[0] == op_type:
                    ops.append(op)
                    evts.append(get_events_for_op(op['type_data'][-1]))

    uniq = get_uniq_evt_sets_counts(evts)
    uniq = list(sorted(uniq.items(), key=lambda x: -x[1]))

    for idx, (evt_set, _) in enumerate(uniq[:1]):
        all_evts = set()  # type: Set[str]
        may_be_dependent = set()  # type: Set[Tuple[str, str]]
        evts_without_dup = [names for names in evts if len(set(names)) == len(names)]
        primary_evts = [evt for evt in evts_without_dup if set(evt) == set(evt_set)]

        if not primary_evts:
            continue

        for names in primary_evts:
            all_evts.update(names)
            may_be_dependent.update(itertools.combinations(names, 2))

        always_ordered = set()
        for e1, e2 in sorted(may_be_dependent):
            if (e2, e1) not in may_be_dependent:
                always_ordered.add((e1, e2))

        try:
            closest = find_closest_pars(always_ordered)
        except:
            print(may_be_dependent, always_ordered)
            raise

        for e1, e2 in closest:
            if (e2, e1) in closest:
                print((e1, e2))

        print("digraph G{} {{".format(idx))
        for evt, deps in detect_order(closest):
            for dep in deps:
                print("    {} -> {};".format(evt, dep))
            # sd = ",".join(deps) if deps else "STOP"
            # print("{} => {{{}}}".format(evt, sd))
        print("}")


if __name__ == "__main__":
    main()