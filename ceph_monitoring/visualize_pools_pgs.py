import collections
from typing import Dict, Optional

import numpy

from cephlib.units import b2ssize_10, b2ssize

from . import html
from .visualize_utils import tab, plot
from .cluster_classes import CephInfo
from .obj_links import pool_link, rule_link
from .plot_data import get_histo_img
from .table import Table, count, bytes_sz, ident, idents_list, exact_count


@tab("Pool's lifetime load")
def show_pools_lifetime_load(ceph: CephInfo) -> html.HTMLTable:
    class PoolLoadAvgTable(Table):
        pool = ident()
        read_b = bytes_sz("Read B")
        write_b = bytes_sz("Write B")
        read_ops = count()
        write_ops = count()
        read_per_pg = count("Rd/PG")
        write_per_pg = count("Wr/PG")

    table = PoolLoadAvgTable()

    for pool in ceph.pools.values():
        row = table.next_row()
        row.pool = pool.name
        row.read_b = pool.df.read_bytes
        row.write_b = pool.df.write_bytes
        row.read_ops = pool.df.read_ops
        row.write_ops = pool.df.write_ops
        row.read_per_pg = pool.df.read_ops // pool.pg
        row.write_per_pg = pool.df.write_ops // pool.pg

    return table.html(id="table-pools-io", align=html.TableAlign.left_right)


@tab("Pool's curr load")
def show_pools_curr_load(ceph: CephInfo) -> Optional[html.HTMLTable]:
    if all(pool.d_df is None for pool in ceph.pools.values()):
        return None

    class PoolLoadCurrTable(Table):
        pool = ident()
        new_data = bytes_sz("New data")
        read_b = bytes_sz("Read Bps")
        write_b = bytes_sz("Write Bps")
        read_ops = count("Read IOPS")
        write_ops = count("Write IOPS")
        read_per_pg = count("Rd/PG")
        write_per_pg = count("Wr/PG")

    table = PoolLoadCurrTable()

    for pool in ceph.pools.values():
        if pool.d_df:
            row = table.next_row()
            row.pool = pool.name
            row.new_data = pool.d_df.size_bytes
            row.read_b = pool.d_df.read_bytes
            row.write_b = pool.d_df.write_bytes
            row.read_ops = pool.d_df.read_ops
            row.write_ops = pool.d_df.write_ops
            row.read_per_pg = pool.d_df.read_ops // pool.pg
            row.write_per_pg = pool.d_df.write_ops // pool.pg

    return table.html(id="table-pools-io", align=html.TableAlign.left_right)


@tab("Pool's stats")
def show_pools_info(ceph: CephInfo) -> html.HTMLTable:
    class PoolsTable(Table):
        pool = ident()
        id = count()
        size = ident()
        obj = ident()
        bytes = ident()
        ruleset = ident()
        apps = idents_list(delim=", ")
        pg = ident()
        osds = exact_count("OSD's")
        space = bytes_sz("Total<br>OSD Space")
        avg_obj_size = bytes_sz("Obj size")
        pg_recommended = exact_count("PG<br>recc.")
        byte_per_pg = bytes_sz("Bytes/PG")
        obj_per_pg = count("Objs/PG")
        pg_count_deviation = ident("PG/OSD<br>Deviation")

    table = PoolsTable()

    total_objs = sum(pool.df.num_objects for pool in ceph.pools.values())

    osds_for_rule: Dict[int, int] = {}
    total_size_for_rule: Dict[int, int] = {}
    rules_names = [rule.name for rule in ceph.crush.rules.values()]

    for _, pool in sorted(ceph.pools.items()):
        row = table.next_row()
        row.pool = pool_link(pool.name).link, pool.name
        row.id = pool.id

        vl = f"{pool.size} / {pool.min_size}"
        if pool.name == 'gnocchi':
            if pool.size != 2 or pool.min_size != 1:
                vl = html.fail(vl)
        else:
            if pool.size != 3 or pool.min_size != 2:
                vl = html.fail(vl)
        row.size = vl, pool.size

        obj_perc = pool.df.num_objects * 100 // total_objs
        row.obj = f"{b2ssize_10(pool.df.num_objects)} ({obj_perc}%)", pool.df.num_objects

        bytes_perc = pool.df.size_bytes * 100 // ceph.status.data_bytes
        row.bytes = f"{b2ssize(pool.df.size_bytes)} ({bytes_perc}%)", pool.df.size_bytes

        rule_name = ceph.crush.rules[pool.crush_rule].name
        cls_name = f"rule{rules_names.index(rule_name)}"
        row.ruleset = f'<div class="{cls_name}">{rule_link(rule_name, pool.crush_rule).link}</div>', pool.crush_rule

        if ceph.is_luminous:
            row.apps = pool.apps

        pg_perc = (pool.pg * 100) // ceph.status.num_pgs

        if pool.pgp != pool.pg:
            pg_message = f"{pool.pg}, {html.H.font(str(pool.pgp), color='red')}"
        else:
            pg_message = str(pool.pg)

        row.pg = pg_message + f" ({pg_perc}%)", pool.pg

        if pool.crush_rule not in osds_for_rule:
            rule = ceph.crush.rules[pool.crush_rule]
            all_osd = list(ceph.crush.iter_osds_for_rule(rule.id))
            osds_for_rule[pool.crush_rule] = len(all_osd)
            total_size_for_rule[pool.crush_rule] = sum(ceph.osds[osd.id].total_space for osd in all_osd)

        row.osds = osds_for_rule[pool.crush_rule]
        row.space = total_size_for_rule[pool.crush_rule]
        row.byte_per_pg = pool.df.size_bytes // pool.pg
        row.avg_obj_size = 0 if pool.df.num_objects == 0 else pool.df.size_bytes // pool.df.num_objects
        row.obj_per_pg = pool.df.num_objects // pool.pg

        if ceph.sum_per_osd is not None:
            osds = list(ceph.crush.iter_osds_for_rule(pool.crush_rule))
            osds_pgs = []

            for osd in osds:
                osds_pgs.append(ceph.osd_pool_pg_2d[osd.id].get(pool.name, 0))

            avg = float(sum(osds_pgs)) / len(osds_pgs)
            if len(osds_pgs) < 2:
                dev = None
            else:
                dev = (sum((i - avg) ** 2.0 for i in osds_pgs) / (len(osds_pgs) - 1)) ** 0.5

            if avg >= 1E-5:
                if dev is None:
                    dev_perc = "--"
                else:
                    dev_perc = str(int(dev * 100. / avg))

                row.pg_count_deviation = f"{avg:.1f} ~ {dev_perc}%", int(avg * 1000)

    return table.html(id="table-pools", align=html.TableAlign.left_right)


@tab("PG's status")
def show_pg_state(ceph: CephInfo) -> html.HTMLTable:
    statuses: Dict[str, int] = collections.Counter()

    for pg_group in ceph.status.pgmap_stat['pgs_by_state']:
        for state_name in pg_group['state_name'].split('+'):
            statuses[state_name] += pg_group["count"]

    table = html.HTMLTable("table-pgs", ["Status", "Count", "%"])
    table.add_row(["any", str(ceph.status.num_pgs), "100.00"])
    for status, count in sorted(statuses.items()):
        table.add_row([status, str(count), f"{100.0 * count / ceph.status.num_pgs:.2f}"])

    return table


@plot
@tab("PG's sizes histo")
def show_pg_size_kde(ceph: CephInfo) -> Optional[str]:
    if ceph.pgs:
        vals = [pg.stat_sum.num_bytes / 2 ** 30 for pg in ceph.pgs.pgs.values()]
        return get_histo_img(numpy.array(vals), xlabel="PG size GiB", y_ticks=True)
    return None