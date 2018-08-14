import io
import sys
import json
import pstats
import shutil
import inspect
import logging
import tempfile
import cProfile
import argparse
import subprocess
import logging.config
from pathlib import Path
from typing import List

from cephlib.storage import make_storage, TypedStorage

from .cluster import load_all
from .plot_data import show_osd_lat_heatmaps
from .obj_links import host_link
from .report import Report

from .visualize_cluster import show_cluster_summary, show_issues_table, show_primary_settings, show_ruleset_info, \
                               show_io_status, show_mons_info
from .visualize_pools_pgs import show_pools_info, show_pg_state, show_pg_size_kde
from .visualize_hosts import show_hosts_info, show_host_io_load_in_color, show_host_network_load_in_color, host_info
from .visualize_osds import show_osd_state, show_osd_info, show_osd_perf_info, show_osd_pool_pg_distribution


logger = logging.getLogger('report')


def setup_logging(opts):
    log_config = json.load((Path(__file__).parent / 'logging.json').open())

    del log_config["handlers"]['log_file']

    if opts.log_level is not None:
        log_config["handlers"]["console"]["level"] = opts.log_level

    for settings in log_config["loggers"].values():
        settings['handlers'].remove('log_file')

    logging.config.dictConfig(log_config)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-l", "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   default=None, help="Console log level")
    p.add_argument("-n", "--no-plots", help="Don't draw any plots, generate tables only", action="store_true")
    p.add_argument("-N", "--name", help="Report name", default="Nemo")
    p.add_argument("-o", '--out', help="report output folder", required=True)
    p.add_argument("-w", '--overwrite', action='store_true',  help="Overwrite result folder data")
    p.add_argument("-p", "--pretty-html", help="Prettify index.html", action="store_true")
    p.add_argument("--profile", help="Profile report creation", action="store_true")
    p.add_argument("-e", "--embed", action='store_true', help="Embed js/css files into report to make it stand-alone")
    p.add_argument("path", help="Folder with data, or .tar.gz archive")
    p.add_argument("old_path", nargs='?', help="Older folder with data, or .tar.gz archive to calculate load")
    return p.parse_args(argv[1:])


def main(argv: List[str]):
    opts = parse_args(argv)
    setup_logging(opts)
    remove_folder = False

    logger.info("Generating report from %r to %r", opts.path, opts.out)

    if opts.profile:
        prof = cProfile.Profile()
        prof.enable()

    if Path(opts.path).is_file():
        arch_name = opts.path
        folder = tempfile.mkdtemp()
        logger.info("Unpacking %r to temporary folder %r", opts.path, folder)
        remove_folder = True
        subprocess.call(f"tar -zxvf {arch_name} -C {folder} >/dev/null 2>&1", shell=True)
    elif not Path(opts.path).is_dir():
        logger.error("Path argument (%r) should be a folder with data or path to archive", opts.path)
        return 1
    else:
        folder = opts.path

    out_p = Path(opts.out)
    index_path = out_p / 'index.html'
    if index_path.exists():
        if not opts.overwrite:
            logger.error("%r already exists. Pass -w/--overwrite to overwrite files", index_path)
            return 1
    elif not out_p.exists():
        out_p.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading cluster info into RAM")
        cluster, ceph = load_all(TypedStorage(make_storage(folder, existing=True)))
        logger.info("Done")

        report = Report(opts.name, "index.html")

        cluster_reporters = [
            show_cluster_summary,
            show_issues_table,
            show_primary_settings,
            show_pools_info,
            show_ruleset_info,
            show_io_status,
            show_hosts_info,
            show_mons_info,
            show_osd_state,
            show_osd_info,
            show_osd_perf_info,
            show_pg_state,
            show_osd_pool_pg_distribution,
            show_host_io_load_in_color,
            show_host_network_load_in_color,
            # show_osd_used_space_histo,
            # show_osd_pg_histo,
            # show_osd_load,
            show_osd_lat_heatmaps,
            # show_osd_ops_boxplot,
            show_pg_size_kde
        ]

        params = {"ceph": ceph, "cluster": cluster, "report": report}

        for reporter in cluster_reporters:
            if getattr(reporter, "perf_info_required", False) and not cluster.has_performance_data:
                continue
            if getattr(reporter, 'plot', False) and opts.no_plots:
                continue

            sig = inspect.signature(reporter)
            curr_params = {name: params[name] for name in sig.parameters}
            new_block = reporter(**curr_params)

            if hasattr(reporter, 'report_name'):
                assert 'report' not in sig.parameters
                report.add_block(reporter.__name__.replace("show_", ""), reporter.report_name, new_block)
            elif new_block is not None:
                assert 'report' not in sig.parameters
                report.add_block(reporter.__name__.replace("show_", ""), *new_block)

        for _, host in sorted(cluster.hosts.items()):
            report.add_block(host_link(host.name).id, None, host_info(host, ceph))

        report.save_to(Path(opts.out), opts.pretty_html, embed=opts.embed)
        logger.info("Report successfully stored to %r", index_path)
    finally:
        if remove_folder:
            shutil.rmtree(folder)

    if opts.profile:
        prof.disable()  # type: ignore
        s = io.StringIO()
        ps = pstats.Stats(prof, stream=s).sort_stats('time', 'calls')
        ps.print_stats(40)
        print(s.getvalue())


if __name__ == "__main__":
    exit(main(sys.argv))
