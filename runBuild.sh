#!/bin/bash

rm -rf ./build 
mkdir -p ./build

cd ./build

#CXX_FLAGS="-v -I/usr/local/cuda-12.6/targets/x86_64-linux/include -I/usr/local/cuda-12.6/targets/x86_64-linux/include/cuda/std/__cuda -O3"
CXX_FLAGS="-I/usr/local/cuda-12.6/targets/x86_64-linux/include -L/usr/local/cuda-12.6/targets/x86_64-linux/lib -O3"

cmake -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CUDA_HOST_COMPILER=clang++ \
      -DBUILD_ALL=ON \
      -DBUILD_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
      -DCMAKE_C_FLAGS="${CXX_FLAGS} -v" \
      -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -v" \
      -DCMAKE_CUDA_FLAGS="${CXX_FLAGS} -v" \
      -DCMAKE_EXE_LINKER_FLAGS="-v" \
      -DOMP_INCLUDE_DIR="/usr/lib/gcc/x86_64-linux-gnu/13/include" \
      -DCUDA_THRUST_INCLUDE_DIR="/usr/local/cuda-12.6/targets/x86_64-linux/include/thrust" \
      -DBOOST_INCLUDE_DIR="/usr/include/boost" \
      -S../ -B./

make -j14 VERBOSE=1 all

cd ..

      #-DCMAKE_CUDA_COMPILER=clang++ \
