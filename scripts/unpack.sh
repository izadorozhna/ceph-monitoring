#!/usr/bin/env bash
set -e
set -o pipefail

tmpdir=$(mktemp -d)
archive=$(awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0; }' "$0")
cdir=$(pwd)

tail "-n+$archive" "$0" | tar -zx -C "$tmpdir"
cd "$tmpdir"

set +e
python2.6 collect_info.py $@
code=$?

cd "$cdir"
rm -rf "$tmpdir"
exit $code

__ARCHIVE_BELOW__
