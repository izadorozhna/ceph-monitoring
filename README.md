Ceph cluster data collect tool.
-------------------------------

How to collect data:
--------------------
On ceph cluster:

* ssh to any node, which has ceph access (controller/compute/osd)
* Passwordless ssh access need to be setup to all required ceph nodes. If you key are encrypted you need to decrypt it,
  put decrypted key as default ~/.ssh/id_rsa file, run tool and then remove decrypted key. Usually you can
  decrypt ssh key with 'openssl rsa –in enc.key -out dec.key'.
* SSH daemons must listen on ceph network (consult with ceph osd map),
  if they not - you need to manually provide inventory file, see --help for details
* root or user with passwordless sudo need to used, if you are not using root for ssh. In case if not root used you need
  to add '-u USER_NAME --sudo' options to call.
* Only Hammer(0.94) and more resent versions of ceph are supported
* Run 'curl -o ceph_report_v2.sh 'https://raw.githubusercontent.com/Mirantis/ceph-monitoring/v2.0/binary/ceph_report_v2.sh'
* See 'bash ceph_report_v2.sh --help' for usage and update command in next step, if nessesary
* Run 'bash ceph_report_v2.sh -c node,ceph --log-level DEBUG -O OUTPUT_FOLDER -w'
* Follow the logs, in case of error it should give you a hint what's broken
* The last log message should looks like:
  18:21:12 - INFO - Result saved into '/tmp/tmpv7tvwtlr.tar.gz'
  You can ignore warnings, if they appears in log. The file name in last log line
  is the name of results archive.

Known problems
--------------
* -S/--ssh-opts are not working properly
* load/performance log is not visualized, so you need to avoid them

How to visualize result:
------------------------

* python3.5+ is required
* Run next commands:
    - git clone -b v2.0 https://github.com/Mirantis/ceph-monitoring.git
    - cd ceph-monitoring
    - make prepare
    - source venv3/bin/activate
* read python -m ceph-monitoring.visualize_cluster --help and update next command line if nessesary
* python -m ceph-monitoring.visualize_cluster -l DEBUG -w -o REPORT_OUTPUT_FOLDER RESULT_ARCHIVE_PATH
* Open REPORT_OUTPUT_FOLDER/index.html in browser
