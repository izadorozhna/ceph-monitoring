.PHONY: mypy pylint pylint_e prepare archive

ALL_FILES=$(shell find ceph_monitoring/ -type f -name '*.py')
# ALL_FILES=ceph_monitoring/collect_info.py
# ALL_FILES=ceph_monitoring/visualize_cluster.py ceph_monitoring/resource_usage.py ceph_monitoring/osd_ops.py ceph_monitoring/hw_info.py ceph_monitoring/cluster.py
STUBS="/home/koder/workspace/typeshed"

mypy:
		MYPYPATH=${STUBS} python -m mypy --ignore-missing-imports --follow-imports=skip ${ALL_FILES}

PYLINT_FMT=--msg-template={path}:{line}: [{msg_id}({symbol}), {obj}] {msg}

pylint:
		python -m pylint '${PYLINT_FMT}' --rcfile=pylint.rc ${ALL_FILES}

pylint_e:
		python3 -m pylint -E '${PYLINT_FMT}' --rcfile=pylint.rc ${ALL_FILES}

archive:
		bash scripts/makearch.sh binary/ceph_report.sh
