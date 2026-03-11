#!/bin/bash
cd "${0%/*}" || exit 1

export CC=gcc14-g++
export CXX=gcc14-g++

#ulimit -v 524288  # limit to 512 MB virtual memory
taskset -c 0 nice -n 19 \
   python3 -m pip install -e . --verbose --global-option=build_ext --global-option="--parallel=1"
