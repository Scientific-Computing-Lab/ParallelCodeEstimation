#!/bin/bash

rm -rf ./build 
mkdir -p ./build

cd ./build

CXX_FLAGS="-v -I/usr/local/cuda-12.6/targets/x86_64-linux/include -I/usr/include/c++/13/bits -I/usr/local/cuda-12.6/targets/x86_64-linux/include/cuda/std/__cuda -O3"

cmake -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CUDA_HOST_COMPILER=clang++ \
      -DBUILD_ALL=ON \
      -DBUILD_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
      -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -std=c++14 --offload-arch=sm_86" \
      -DCMAKE_C_FLAGS="${CXX_FLAGS}" \
      -DCUDA_EXTRA_FLAGS="${CXX_FLAGS} -arch=sm_86" \
      -DCUDA_NVCC_FLAGS="-ccbin clang++ -Xcompiler --generate-code=arch=compute_68,code=[compute_68,sm_68]" \
      -S../ -B./

make -j1 VERBOSE=1 all

cd ..

