#!/usr/bin/env bash
set -ex
set -o pipefail

output_file=$(realpath "$1")
cdir=$(pwd)
tdir=$(mktemp -d)
archfile=$(mktemp)

mkdir -p "$tdir"
cd "$tdir"

function gh_dep() {
    modpath=$1
    modname=$(basename $modpath)
    wget --no-check-certificate https://github.com/$modpath/archive/master.tar.gz
    tar --strip-components=1 -xzf master.tar.gz ${modname}-master/${modname}
    rm master.tar.gz
}

URL=https://pypi.python.org/packages/44/88/d09c6a7fe1af4a02f16d2f1766212bec752aadb04e5699a9706a10a1a37d/typing-3.6.2-py3-none-any.whl

gh_dep koder-ua/cephlib
gh_dep koder-ua/agent
wget $URL

cp "${cdir}/ceph_monitoring/collect_info.py" .
cp "${cdir}/ceph_monitoring/logging.json" .
tar -zcf "$archfile" ./*
cat "${cdir}/scripts/unpack.sh" "$archfile" > "$output_file"
chmod +x "$output_file"

echo "Executable archive stored into $output_file"
