#!/usr/bin/env bash
set -euo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
DIR=${SELF%/*/*}

cd -- "$DIR"

function show_help()
{
  cat << HELP
USAGE

  ${0##*/} [-h]

OPTIONS

  -h
    Show this help message and exit.

HELP
  exit "$1"
}

while getopts "h" opt
do
  case "$opt" in
    h) show_help 0 ;;
    *) show_help 1 ;;
  esac
done

if [[ ! -e venv ]]; then
  python3 -m venv venv
  venv/bin/pip install -U pip
fi
source venv/bin/activate

pip install -e '.[sql,tutorials,doc]' # install tanat with all doc-build dependencies
rm -rf public   # clean stale output files
sphinx-apidoc -o doc/source/reference/api -f -H "API Documentation" src/tanat

# sphinx-gallery overwrites auto_examples/index.rst and auto_tutorials/index.rst
# with its own generated versions.  We seed them with our hand-written indexes
# so the source-read hook in conf.py can replace the content reliably.
mkdir -p doc/source/user-guide/auto_examples
mkdir -p doc/source/user-guide/auto_tutorials
cp doc/source/user-guide/examples/index.rst  doc/source/user-guide/auto_examples/index.rst
cp doc/source/user-guide/tutorials/index.rst doc/source/user-guide/auto_tutorials/index.rst

sphinx-build -b html doc/source public

# -- generates files for LLMs --
python scripts/generate_llms_txt.py --output-dir public
