#!/bin/bash

# fast delete ./build dir due to IO limitations on shared remote login nodes
mkdir -p ./build
mkdir -p ./emptydir
rsync -a --delete ./emptydir ./build
rm -rf ./emptydir
mkdir -p ./build
#rm -rf ./build 

cd ./build

# if you're having issues building, it's most likely due to include order issues.
# Add `-H` to the build flags to see what include files are being added and 
# in what order the include directories get searched at build time.
# For example, you can use `make target-name 2>&1 | grep -ni "math.h"` to find the instances
# of the math header being included and decide if clang is including the correct one

# You may need to adjust the order of included directories like we do for Lassen
# below. Adding the `-nobuiltininc` gets rid of a lot of auto system directories
# that clang tries to smartly add. Then you can go in an add the directories yourself
# as we do below.  

# for some reason on lassen, clang is struggling to properly order the include
# directories at build time, so we need to forcibly set the correct directories
LASSEN_OMP_FLAGS="-isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include/openmp_wrappers -isystem /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11 -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include/cuda_wrappers -nobuiltininc"

LASSEN_CUDA_FLAGS="-isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include/cuda_wrappers -isystem /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11 -isystem /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11/ppc64le-redhat-linux -isystem /usr/tce/packages/cuda/cuda-12.2.2/nvidia/targets/ppc64le-linux/include -isystem /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11/backward -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include -isystem /usr/tce/packages/cuda/cuda-12.2.2/include -nobuiltininc"

EXTRA_OMP_FLAGS="${LASSEN_OMP_FLAGS}"
EXTRA_CUDA_FLAGS="${LASSEN_CUDA_FLAGS}"

#EXTRA_BUILD_FLAGS="-O3 -v -H" 
#EXTRA_LINK_FLAGS="-v"

EXTRA_BUILD_FLAGS="-O3" 
EXTRA_LINK_FLAGS=""

# We have modified all the flags in the build system to be clang-specific
# We originally had this working with `nvcc` for the CUDA codes, but switched
# to LLVM because it's popular and keeps the build pipeline simpler. 
# It'll also allow us to build SYCL in the future.

# The CUDAToolkit_ROOT can be left as it, Cmake will auto-detect the correct one

cmake -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CUDA_HOST_COMPILER=clang++ \
      -DCMAKE_CUDA_COMPILER=clang++ \
      -DBUILD_ALL=ON \
      -DBUILD_OMP=ON \
      -DBUILD_CUDA=ON \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
      -DCMAKE_C_FLAGS="${EXTRA_BUILD_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${EXTRA_BUILD_FLAGS}" \
      -DCMAKE_CUDA_FLAGS="${EXTRA_BUILD_FLAGS}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUSTOM_OMP_FLAGS="${EXTRA_OMP_FLAGS}" \
      -DCUSTOM_CUDA_FLAGS="${EXTRA_CUDA_FLAGS}" \
      -DCUSTOM_OMP_LINK_FLAGS="${EXTRA_LINK_FLAGS}" \
      -DCUSTOM_CUDA_LINK_FLAGS="${EXTRA_LINK_FLAGS}" \
      -S../ -B./

#make -j20 all
