#!/usr/bin/env bash
set -e
set -o pipefail

tmpdir=$(mktemp -d)
archive=$(awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0; }' "$0")
cdir=$(pwd)

tail "-n+$archive" "$0" | tar -zx -C "$tmpdir"
cd "$tmpdir"

set +e
if which python3 ; then
    env PYTHONPATH=typing-3.6.2-py3-none-any.whl python3 collect_info.py $@
else
    echo "You need python3.4+ to run this tool. 'python3' alias should be created"
    exit 1
fi

# if which python2.7 ; then
#     python2.7 collect_info.py $@
# else
#     if which python2.6 ; then
#         python2.6 collect_info.py $@
#     else
#         echo "Nither python2.6 nor python2.7 found"
#         exit 1
#     fi
# fi

code=$?

cd "$cdir"
rm -rf "$tmpdir"
exit $code

__ARCHIVE_BELOW__
