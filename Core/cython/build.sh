#!/usr/bin/env bash
filename="${1%.*}"
echo "Building C file..."
cython probas.pyx ${filename}.pyx &&
echo "Compiling..." &&
gcc -w -fopenmp -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python3.6 -o ${filename}.so ${filename}.c &&
echo "Done."
