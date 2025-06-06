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

source = main.cpp kernels.cpp util.cpp parse.cpp sequential.cpp

obj = main.o kernels.o util.o parse.o sequential.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../include -fsycl -I../hbc-cuda

# Linker Flags
LDFLAGS = 

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH) \
            -Xcuda-ptxas --maxrregcount=45 \
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

main.o: main.cpp ../hbc-cuda/parse.h ../hbc-cuda/sequential.h util.h
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.cpp util.h
	$(CC) $(CFLAGS) -c $< -o $@

util.o: ../hbc-cuda/util.cpp util.h 
	$(CC) $(CFLAGS) -c $< -o $@

sequential.o: ../hbc-cuda/sequential.cpp ../hbc-cuda/sequential.h
	$(CC) $(CFLAGS) -c $< -o $@

parse.o: ../hbc-cuda/parse.cpp ../hbc-cuda/parse.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) output.txt

run: $(program)
	$(LAUNCHER) ./$(program) -i ../hbc-cuda/graphs/breast_coexpress.txt -v --printscores=output.txt
