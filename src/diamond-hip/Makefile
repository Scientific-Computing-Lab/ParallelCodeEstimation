#===============================================================================
# User Options
#===============================================================================

#intel
CC          = hipcc 
OPTIMIZE    = yes
DEBUG       = no
DMND_PATH   = ../diamond-sycl
LAUNCHER    =

#===============================================================================
# Program name & source code list
#===============================================================================

program = diamond

c_src = \
  $(DMND_PATH)/src/blast/sm_blosum45.c \
  $(DMND_PATH)/src/blast/sm_blosum50.c \
  $(DMND_PATH)/src/blast/sm_blosum62.c \
  $(DMND_PATH)/src/blast/sm_blosum80.c \
  $(DMND_PATH)/src/blast/sm_blosum90.c \
  $(DMND_PATH)/src/blast/sm_pam30.c \
  $(DMND_PATH)/src/blast/sm_pam70.c \
  $(DMND_PATH)/src/blast/sm_pam250.c

cpp_src = \
  $(DMND_PATH)/src/run/main.cpp \
  $(DMND_PATH)/src/basic/config.cpp \
  $(DMND_PATH)/src/util/tinythread.cpp \
  $(DMND_PATH)/src/util/compressed_stream.cpp \
  $(DMND_PATH)/src/basic/score_matrix.cpp \
  $(DMND_PATH)/src/blast/blast_filter.cpp \
  $(DMND_PATH)/src/blast/blast_seg.cpp \
  $(DMND_PATH)/src/data/queries.cpp \
  $(DMND_PATH)/src/data/reference.cpp \
  $(DMND_PATH)/src/data/seed_histogram.cpp \
  $(DMND_PATH)/src/output/daa_record.cpp \
  $(DMND_PATH)/src/search/search.cpp \
  $(DMND_PATH)/src/util/command_line_parser.cpp \
  $(DMND_PATH)/src/util/seq_file_format.cpp \
  $(DMND_PATH)/src/util/util.cpp  \
  $(DMND_PATH)/src/util/Timer.cpp \
  $(DMND_PATH)/src/basic/basic.cpp \
  $(DMND_PATH)/src/dp/floating_sw.cpp \
  $(DMND_PATH)/src/basic/hssp.cpp \
  $(DMND_PATH)/src/dp/ungapped_align.cpp \
  $(DMND_PATH)/src/run/tools.cpp \
  $(DMND_PATH)/src/dp/greedy_align.cpp \
  $(DMND_PATH)/src/run/benchmark.cpp \
  $(DMND_PATH)/src/search/stage2.cpp \
  $(DMND_PATH)/src/output/output_format.cpp \
  $(DMND_PATH)/src/output/join_blocks.cpp \
  $(DMND_PATH)/src/data/frequent_seeds.cpp \
  $(DMND_PATH)/src/align/query_mapper.cpp \
  $(DMND_PATH)/src/align/align_target.cpp \
  $(DMND_PATH)/src/output/blast_tab_format.cpp \
  $(DMND_PATH)/src/dp/padded_banded_sw.cpp \
  $(DMND_PATH)/src/dp/needleman_wunsch.cpp \
  $(DMND_PATH)/src/output/blast_pairwise_format.cpp \
  $(DMND_PATH)/src/extra/roc.cpp \
  $(DMND_PATH)/src/dp/comp_based_stats.cpp \
  $(DMND_PATH)/src/extra/model_sim.cpp \
  $(DMND_PATH)/src/run/double_indexed.cpp \
  $(DMND_PATH)/src/search/collision.cpp \
  $(DMND_PATH)/src/output/sam_format.cpp \
  $(DMND_PATH)/src/align/align.cpp \
  $(DMND_PATH)/src/search/setup.cpp \
  $(DMND_PATH)/src/extra/opt.cpp \
  $(DMND_PATH)/src/dp/diag_scores.cpp \
  $(DMND_PATH)/src/data/taxonomy.cpp \
  $(DMND_PATH)/src/lib/tantan/tantan.cpp \
  $(DMND_PATH)/src/lib/tantan/tantale.cpp \
  $(DMND_PATH)/src/dp/swipe.cpp \
  $(DMND_PATH)/src/dp/banded_sw.cpp \
  $(DMND_PATH)/src/data/sorted_list.cpp \
  $(DMND_PATH)/src/data/seed_set.cpp \
  $(DMND_PATH)/src/util/binary_file.cpp \
  $(DMND_PATH)/src/output/taxon_format.cpp \
  $(DMND_PATH)/src/output/view.cpp \
  $(DMND_PATH)/src/output/output_sink.cpp \
  $(DMND_PATH)/src/output/target_culling.cpp

hip_src = ./masking.cu

obj = $(c_src:.c=.o) $(cpp_src:.cpp=.o) $(hip_src:.cu=.o) 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall -x hip

# Linker Flags
LDFLAGS = -lz -lpthread

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) 
	$(CC) $(obj) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.m8

test: $(program)
	./$(program) blastx -q $(DMND_PATH)/short.fastq.gz -d $(DMND_PATH)/ARDB.dmnd -o short_cur.m8 --tmpdir /dev/shm -p1

run: $(program)
	$(LAUNCHER) ./$(program) blastx -q $(DMND_PATH)/long.fastq.gz -d $(DMND_PATH)/ARDB.dmnd -o long_cur.m8 --tmpdir /dev/shm -p1

