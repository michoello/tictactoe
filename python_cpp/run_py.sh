#!/bin/bash
cd "${0%/*}" || exit 1

python3 examples/invert_cli.py 1 2 3 4 5
