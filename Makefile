.PHONY: mypy pylint pylint_e

ALL_FILES=$(shell find ceph_monitoring/ -type f -name '*.py')
# ALL_FILES=ceph_monitoring/collect_info.py
STUBS="stubs:../venvs/wally/lib/python3.5/site-packages/"

mypy:
		MYPYPATH=${STUBS} python -m mypy --ignore-missing-imports --follow-imports=skip ${ALL_FILES}

PYLINT_FMT=--msg-template={path}:{line}: [{msg_id}({symbol}), {obj}] {msg}

pylint:
		python -m pylint '${PYLINT_FMT}' --rcfile=pylint.rc ${ALL_FILES}

pylint_e:
		python3 -m pylint -E '${PYLINT_FMT}' --rcfile=pylint.rc ${ALL_FILES}
