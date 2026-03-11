#!/bin/bash
cd "${0%/*}" || exit 1

valgrind --leak-check=full --show-leak-kinds=all ./test_matrix
