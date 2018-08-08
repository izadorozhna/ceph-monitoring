#!/usr/bin/env bash
set -ex
set -o pipefail

output_file=$(realpath "$1")
cdir=$(pwd)
tdir=$(mktemp -d)
archfile=$(mktemp)

mkdir -p "$tdir"
pushd "$tdir"

function gh_dep() {
    modpath=$1
    modname=$(basename $modpath)
    wget --no-check-certificate https://github.com/$modpath/archive/master.tar.gz
    tar --strip-components=1 -xzf master.tar.gz ${modname}-master/${modname}
    rm master.tar.gz
}

for name in cephlib agent xmlbuilder3 ; do
    #gh_dep "koder-ua/$name"
    cp -r "${cdir}/../$name/$name" .
done

#URL=https://pypi.python.org/packages/44/88/d09c6a7fe1af4a02f16d2f1766212bec752aadb04e5699a9706a10a1a37d/typing-3.6.2-py3-none-any.whl
URL="https://files.pythonhosted.org/packages/ec/cc/28444132a25c113149cec54618abc909596f0b272a74c55bab9593f8876c/typing-3.6.4.tar.gz"
wget $URL

cp "${cdir}/ceph_monitoring/collect_info.py" .
cp "${cdir}/ceph_monitoring/logging.json" .
tar -zcf "$archfile" ./*
cat "${cdir}/scripts/unpack.sh" "$archfile" > "$output_file"
chmod +x "$output_file"

popd

rm -rf "$tdir"
rm "$archfile"

echo "Executable archive stored into $output_file"
