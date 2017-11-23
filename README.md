Ceph cluster data collect tool.
-------------------------------

How to collect data:
--------------------
On ceph cluster:

* ssh to any node, which has ceph access (controller/compute/osd)
* Passwordless ssh access need to be setup to all required ceph nodes
* Hammer (0.94) and more resent versions of ceph are supported
* Run 'curl -o ceph_report_v2.sh 'https://raw.githubusercontent.com/Mirantis/ceph-monitoring/v2.0/binary/ceph_report_v2.sh'
* See 'bash ceph_report_v2.sh --help' for usage and update command in next step, if nessesary
* Run 'bash ceph_report_v2.sh -c node,ceph,load --log-level DEBUG -O OUTPUT_FOLDER -w -s LOAD_COLLECT_SECONDS'
* The last log message should looks like:
18:21:12 - INFO - Result saved into '/tmp/tmpv7tvwtlr.tar.gz'
You can ignore warnings, if they appears in log. The file name in last log line
is the name of results archive.

How to visualize result:
------------------------

* Download result archive to current node
* Install ceph-monitoring and all deps:
    
    - python3.5+ is required
    - mkdir ceph-monitoring ; cd ceph-monitoring
    - git clone -b 'v2.0' https://github.com/Mirantis/ceph-monitoring.git
    - git clone https://github.com/koder-ua/cephlib.git
    - python -m pip install -r cephlib/requirements.txt
    - python -m pip install -r ceph-monitoring/requirements.txt

* export PYTHONPATH="$PYTHONPATH:cephlib:ceph-monitoring"
* read python -m ceph-monitoring.visualize_cluster --help and update next command line if nessesary
* python -m ceph-monitoring.visualize_cluster -l DEBUG -o REPORT_OUTPUT_FOLDER RESULT_ARCHIVE_PATH
* Open REPORT_OUTPUT_FOLDER/index.html in browser
