#!/usr/bin/env bash
set -ex
set -o pipefail

output_file="$1"
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

gh_dep koder-ua/cephlib
gh_dep koder-ua/agent

cp "${cdir}/ceph_monitoring/collect_info.py" .
tar -zcf "$archfile" ./*
cat "${cdir}/scripts/unpack.sh" "$archfile" > "$output_file"
chmod +x "$output_file"

echo "Executable archive stored into $output_file"
