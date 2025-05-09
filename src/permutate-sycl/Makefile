#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = gpu_permutation_testing.cpp \
         ../permutate-cuda/main.cpp \
         ../permutate-cuda/permutation_testing.cpp \
         ../permutate-cuda/utils.cpp \
         ../permutate-cuda/statistical_test.cpp \
         ../permutate-cuda/bzip/blocksort.cpp \
         ../permutate-cuda/bzip/bzlib.cpp \
         ../permutate-cuda/bzip/compress.cpp \
         ../permutate-cuda/bzip/crctable.cpp \
         ../permutate-cuda/bzip/decompress.cpp \
         ../permutate-cuda/bzip/huffman.cpp \
         ../permutate-cuda/bzip/randtable.cpp

obj = main.o permutation_testing.o utils.o statistical_test.o \
      blocksort.o bzlib.o compress.o crctable.o decompress.o \
      huffman.o randtable.o gpu_permutation_testing.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../permutate-cuda -fsycl

# Linker Flags
LDFLAGS = 

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH) \
            --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

gpu_permutation_testing.o: gpu_permutation_testing.cpp \
                           ../permutate-cuda/header.h \
                           kernel_functions.hpp device_functions.hpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: ../permutate-cuda/%.cpp ../permutate-cuda/header.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: ../permutate-cuda/bzip/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../permutate-cuda/test_data/truerand_1bit.bin
