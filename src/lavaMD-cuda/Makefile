#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
OUTPUT    = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu ./util/num/num.cu ./util/timer/timer.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

ifeq ($(OUTPUT),yes)
  CFLAGS += -DOUTPUT
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

main.o:	main.cu main.h kernel.h

./util/num/num.o: ./util/num/num.cu ./util/num/num.h 

./util/timer/timer.o: ./util/timer/timer.cu ./util/timer/timer.h 

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -boxes1d 30
