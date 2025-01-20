#!/bin/bash

rm -rf ./build 
mkdir -p ./build

cd ./build

CXX_FLAGS="-O3"

cmake -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CUDA_HOST_COMPILER=clang++ \
      -DBUILD_ALL=ON \
      -DBUILD_OMP=ON \
      -DBUILD_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
      -DCMAKE_C_FLAGS="${CXX_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
      -DCMAKE_CUDA_FLAGS="${CXX_FLAGS}" \
      -S../ -B./

make -j14 all

cd ..

      #-DCUDA_RUNTIME_HEADER_DIR="/usr/local/cuda-12.6/targets/x86_64-linux/include" \
      #-DOMP_INCLUDE_DIR="/usr/lib/llvm-18/lib/clang/18/include" \
      #-DOMP_LINKER_FLAGS="-L/usr/lib/llvm-18/lib" \
      #-DOMP_LINKER_FLAGS="-L/usr/lib/llvm-18/lib;-lgomp" \
      #-DBUILD_CUDA=ON \

#make -j1 VERBOSE=1 all


      #-DCMAKE_EXE_LINKER_FLAGS="-v" \
      #-DOMP_INCLUDE_DIR="/usr/lib/gcc/x86_64-linux-gnu/13/include" \
      #-DCUDA_THRUST_INCLUDE_DIR="/usr/local/cuda-12.6/targets/x86_64-linux/include/thrust" \
      #-DBOOST_INCLUDE_DIR="/usr/include/boost" \
      #-DCMAKE_CUDA_COMPILER=clang++ \
