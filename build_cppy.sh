rm -rf build/ *.egg-info

export CC=gcc14-g++
export CXX=gcc14-g++
python3 -m pip install -e .
