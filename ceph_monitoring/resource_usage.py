from __future__ import print_function

import logging

import numpy

from .hw_info import get_dev_file_name


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
    collected = {host: host_data['collected_at'] for host, host_data in cluster.perf_data.items()}

    # node name to node storageid
    name2stor = {name.split('-', 1)[1]: name for name in cluster.perf_data}

    # select range, which present on all nodes
    lower = int(max(arr.min() for arr in collected.values()))
    upper = int(min(arr.max() for arr in collected.values()) + 0.999)

    node_bounds = {host: host_data['collected_at'].searchsorted((lower, upper + 1))
                   for host, host_data in cluster.perf_data.items()}

    min_sz = min((i2 - i1) for i1, i2 in node_bounds.values())
    times = numpy.arange(lower, lower + min_sz)

    # usually each OSD has dedicated data device (but not always)
    # but several OSD often share the same device for journal

    all_data_disks = {(osd.host.name, osd.j_stor_stats.root_dev) for osd in cluster.osds}
    all_j_disks = {(osd.host.name, osd.data_stor_stats.root_dev) for osd in cluster.osds}

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
            stor = cluster.perf_data[storname]['block-io'][get_dev_file_name(dev)]
            idx1, _ = node_bounds[storname]
            idx2 = idx1 + min_sz

            # print(idx, wio[idx], idx1, idx2, stor['writes_completed'][idx1:idx2])
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