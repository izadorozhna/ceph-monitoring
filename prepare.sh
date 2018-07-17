#!/usr/bin/env bash
if [ ! -d venv3 ]; then
    set -xe
    git submodule init
    git submodule update --recursive --remote
    python3 -m venv venv3
    set +xe
    echo "source venv3/bin/activate"
    source venv3/bin/activate
    set -xe
    pip install wheel
    pip install -r requirements.txt
    pip install -r modules/cephlib/requirements.txt
    ln -s ${PWD}/modules/agent/agent ${PWD}/venv3/lib/python3.?/site-packages
    ln -s ${PWD}/modules/cephlib/cephlib ${PWD}/venv3/lib/python3.?/site-packages
    ln -s ${PWD}/modules/xmlbuilder3/xmlbuilder3 ${PWD}/venv3/lib/python3.?/site-packages
    set +xe
fi
