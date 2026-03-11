#!/bin/bash
cd "${0%/*}" || exit 1

python3 bench_matrix.py
