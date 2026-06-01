#!/usr/bin/env bash
set -euo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
DIR=${SELF%/*/*}

cd -- "$DIR"

function show_help()
{
  cat << HELP
USAGE

  ${0##*/} [-h] [-n]

OPTIONS

  -n
    Skip virtual-environment creation and activation (useful in CI where
    the Python environment is already set up by the runner).

  -h
    Show this help message and exit.

HELP
  exit "$1"
}

NO_VENV=0

while getopts "hn" opt
do
  case "$opt" in
    h) show_help 0 ;;
    n) NO_VENV=1 ;;
    *) show_help 1 ;;
  esac
done

if [[ $NO_VENV -eq 0 ]]; then
  if [[ ! -e venv ]]; then
    python3 -m venv venv
    venv/bin/pip install -U pip
  fi
  source venv/bin/activate
fi

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
