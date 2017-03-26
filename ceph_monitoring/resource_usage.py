from __future__ import print_function


import numpy


class ResourceUsage(object):
    def __init__(self):
        self.ceph_data_dev_wio = None  # 2D array
        self.ceph_data_dev_wbytes = None


def load_resource_usage(cluster):
    # self.perf_data = {host: {sensor: {dev: {metric: array}}}}
    usage = ResourceUsage()
    US2S = 1000 * 1000
    collected = {host: host_data['collected_at'] for host, host_data in cluster.perf_data.items()}
    lower = max(arr.min() // US2S for arr in collected.values())
    upper = min(arr.max() // US2S for arr in collected.values())

    name2stor = {name.split('-', 1)[1]: name for name in cluster.perf_data}

    times = numpy.arange(lower, upper + 1)
    node_bounds = {
        host: host_data['collected_at'].searchsorted((lower * US2S, (upper + 1) * US2S))
        for host, host_data in cluster.perf_data.items()
    }

    usage.ceph_data_dev_wio = numpy.empty((len(cluster.osds), len(times)))
    for osd_idx, osd in enumerate(cluster.osds):
        storname = name2stor[osd.host.name]
        vl = cluster.perf_data[storname]['block-io'][osd.data_stor_stats.src_dev]['writes_completed']
        idx1, idx2 = node_bounds[storname]
        usage.ceph_data_dev_wio[osd_idx] = vl[idx1:idx2]

    usage.ceph_data_dev_wbytes = numpy.empty((len(cluster.osds), len(times)))
    for osd_idx, osd in enumerate(cluster.osds):
        storname = name2stor[osd.host.name]
        vl = cluster.perf_data[storname]['block-io'][osd.data_stor_stats.src_dev]['sectors_written']
        idx1, idx2 = node_bounds[storname]
        usage.ceph_data_dev_wbytes[osd_idx] = vl[idx1:idx2]

    return usage