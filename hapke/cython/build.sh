echo "Building C file from hapke.pyx ..."
cython hapke.pyx &&
echo "Compiling..." &&
gcc -w -fopenmp -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.6 -o hapke.so hapke.c &&
echo "Done."
