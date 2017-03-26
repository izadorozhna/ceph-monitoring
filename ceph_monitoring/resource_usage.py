from __future__ import print_function

import logging

import numpy


logger = logging.getLogger('cephlib.report')


class ResourceUsage(object):
    def __init__(self):
        self.ceph_data_dev_wio = None    # 2D array
        self.ceph_data_dev_wbytes = None # 2D array
        self.ceph_data_dev_rio = None    # 2D array
        self.ceph_data_dev_rbytes = None # 2D array
        self.ceph_data_dev_qd = None
        self.colocated_journals = False

        self.ceph_j_dev_wbytes = None    # 2D array
        self.ceph_j_dev_wio = None       # 2D array
        self.ceph_j_dev_qd = None


def get_resource_usage(cluster):
    if cluster.usage:
        return cluster.usage

    logger.info("Loading cluster usage into RAM")

    usage = ResourceUsage()
    US2S = 1000 * 1000
    collected = {host: host_data['collected_at'] for host, host_data in cluster.perf_data.items()}

    # select range, which present on all nodes
    lower = max(arr.min() // US2S for arr in collected.values())
    upper = min(arr.max() // US2S for arr in collected.values())

    # node name to node storageid
    name2stor = {name.split('-', 1)[1]: name for name in cluster.perf_data}

    times = numpy.arange(lower, upper + 1)

    # find indexes for each node to select [lower, upper] range
    node_bounds = {
        host: host_data['collected_at'].searchsorted((lower * US2S, (upper + 1) * US2S))
        for host, host_data in cluster.perf_data.items()
    }

    # usually each OSD has dedicated data device (but not always)
    # but several OSD often share the same device for journal

    all_data_disks = {(osd.host.name, osd.j_stor_stats.src_dev) for osd in cluster.osds}
    all_j_disks = {(osd.host.name, osd.data_stor_stats.src_dev) for osd in cluster.osds}

    if all_j_disks.issubset(all_data_disks):
        usage.colocated_journals = True

    for is_journal in (True, False):
        all_disks = all_j_disks if is_journal else all_data_disks
        shape = len(all_disks), len(times)
        wio = numpy.empty(shape)
        wbytes = numpy.empty(shape)
        qd = numpy.empty(shape)

        if not is_journal:
            rio = numpy.empty(shape)
            rbytes = numpy.empty(shape)

        for idx, (hostname, dev) in enumerate(all_disks):
            storname = name2stor[hostname]
            stor = cluster.perf_data[storname]['block-io'][dev]
            idx1, idx2 = node_bounds[storname]
            wio[idx] = stor['writes_completed'][idx1:idx2]
            wbytes[idx] = stor['sectors_written'][idx1:idx2]
            qd[idx] = stor['io_queue'][idx1:idx2]

            if not is_journal:
                rio[idx] = stor['reads_completed'][idx1:idx2]
                rbytes[idx] = stor['sectors_read'][idx1:idx2]

        if is_journal:
            usage.ceph_j_dev_wio = wio
            usage.ceph_j_dev_wbytes = wbytes
            usage.ceph_j_dev_qd = wio
        else:
            usage.ceph_data_dev_wio = wio
            usage.ceph_data_dev_wbytes = wbytes
            usage.ceph_data_dev_rio = rio
            usage.ceph_data_dev_rbytes = rbytes
            usage.ceph_data_dev_qd = qd

    logger.info("Done")
    cluster.usage = usage
    return usage