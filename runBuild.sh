#!/bin/bash

rm -rf ./build 
mkdir -p ./build

cd ./build

# for some reason on lassen, clang is struggling to properly order the include
# directories at build time, so we need to forcibly set the correct directories
LASSEN_NICSLU_FLAGS=""
# if you're having issues building, it's most likely due to include issues
# Add `-H` to the build command to see what include files are being added
# you can use `make target-name 2>&1 | grep -ni "math.h"` to find the instances
# of the math header being included and decide if clang is including the correct one
LASSEN_OMP_FLAGS="-fopenmp-offload-mandatory -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include/openmp_wrappers -isystem /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11 -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include/cuda_wrappers -nobuiltininc"

#EXTRA_FLAGS="-O3 -v -H" 
EXTRA_FLAGS="-O3 -v -H" 

# We have modified all the flags in the build system to be clang-specific
# We originally had this working with `nvcc` for the CUDA codes, but switched
# to LLVM because it's popular and keeps the build pipeline simpler. 
# It'll also allow us to build SYCL in the future.

cmake -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CUDA_HOST_COMPILER=clang++ \
      -DCMAKE_CUDA_COMPILER=clang++ \
      -DBUILD_ALL=ON \
      -DBUILD_OMP=OFF \
      -DBUILD_CUDA=ON \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
      -DCMAKE_C_FLAGS="${EXTRA_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${EXTRA_FLAGS}" \
      -DCMAKE_CUDA_FLAGS="${EXTRA_FLAGS}" \
      -DCMAKE_BUILD_TYPE=Release \
      -S../ -B./

#make -j20 all

#cd ..

      #-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
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
