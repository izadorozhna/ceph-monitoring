#!/usr/bin/env bash
if [ ! -d venv3 ]; then
    set -xe
    python3.6 -m venv venv3
    set +xe
    echo "source venv3/bin/activate"
    source venv3/bin/activate
    set -xe
    pip install wheel
    pip install -r requirements.txt

    mkdir modules
    pushd modules
    git clone https://github.com/koder-ua/cephlib.git
    git clone https://github.com/koder-ua/agent.git
    git clone https://github.com/koder-ua/xmlbuilder3.git
    popd

    pip install -r modules/cephlib/requirements.txt

    ln -s ${PWD}/modules/agent/agent ${PWD}/venv3/lib/python3.?/site-packages
    ln -s ${PWD}/modules/cephlib/cephlib ${PWD}/venv3/lib/python3.?/site-packages
    ln -s ${PWD}/modules/xmlbuilder3/xmlbuilder3 ${PWD}/venv3/lib/python3.?/site-packages
    set +xe
fi
