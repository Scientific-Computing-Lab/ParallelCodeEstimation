#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp kernel.cpp host_task.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall -I../tqs-cuda

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h ../tqs-cuda/support/task.h \
                   ../tqs-cuda/support/setup.h \
                   ../tqs-cuda/support/verify.h
	$(CC) $(CFLAGS) -c $< -o $@

kernel.o: kernel.cpp kernel.h
	$(CC) $(CFLAGS) -c $< -o $@

host_task.o: host_task.cpp kernel.h ../tqs-cuda/support/task.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -f ../tqs-cuda/input/patternsNP100NB512FB25.txt
