#!/bin/bash

rm -rf ./build 
mkdir -p ./build

cd ./build

#CXX_FLAGS="-v -I/usr/local/cuda-12.6/targets/x86_64-linux/include -I/usr/local/cuda-12.6/targets/x86_64-linux/include/cuda/std/__cuda -O3"
CXX_FLAGS="-I/usr/local/cuda-12.6/targets/x86_64-linux/include -L/usr/local/cuda-12.6/targets/x86_64-linux/lib -O3"

# now sure why clang isn't automatically linking openmp, so we include some extra flags

cmake -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CUDA_HOST_COMPILER=clang++ \
      -DBUILD_ALL=ON \
      -DBUILD_OMP=ON \
      -DBUILD_CUDA=OFF \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
      -DCMAKE_C_FLAGS="${CXX_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
      -DCMAKE_CUDA_FLAGS="${CXX_FLAGS}" \
      -DOMP_INCLUDE_DIR="/usr/lib/llvm-18/lib/clang/18/include" \
      -DOMP_LINKER_FLAGS="-L/usr/lib/llvm-18/lib;-lgomp" \
      -S../ -B./

      #-DBUILD_CUDA=ON \

#make -j1 VERBOSE=1 streamUM-cuda
#make -j14 VERBOSE=1 all
#make -j14 all
#make -j1 VERBOSE=1 all
#make -j4 VERBOSE=1 heartwall-cuda

cd ..

      #-DCMAKE_EXE_LINKER_FLAGS="-v" \
      #-DOMP_INCLUDE_DIR="/usr/lib/gcc/x86_64-linux-gnu/13/include" \
      #-DCUDA_THRUST_INCLUDE_DIR="/usr/local/cuda-12.6/targets/x86_64-linux/include/thrust" \
      #-DBOOST_INCLUDE_DIR="/usr/include/boost" \
      #-DCMAKE_CUDA_COMPILER=clang++ \
