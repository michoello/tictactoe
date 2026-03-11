#!/bin/bash
cd "${0%/*}" || exit 1

g++ -std=c++17 -O2 cpp/invert.cpp cpp/bench_matrix.cpp -o bench_matrix
time ./bench_matrix
