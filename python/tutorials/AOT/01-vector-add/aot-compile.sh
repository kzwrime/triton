#!/bin/bash

# Enable echo, i.e., print executed commands
set -x

# Exit on error
set -e

# Create the kernels directory if it does not exist
if [ ! -d "./kernels" ]; then
    mkdir ./kernels
else
    find ./kernels/ -type f -print0 | xargs -0 rm -f
fi

python3 generate_kernel.py

python3 -m triton.tools.link_v2 "./kernels/*.h" -o add_kernel

# ==================== Makefile Version ====================

# make clean
# make

# ===========================================================

# ==================== CMake Version =======================

if [ ! -d "./build" ]; then
    mkdir -p ./build
else
    rm build/* -rf
fi

cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug
make
cd ..

# ===========================================================

nvcc add_verify.cu build/libadd_kernel_fp32.a -lcuda -o add_verify.out
