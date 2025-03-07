

cudaBB: cuda_BB_kernel.cu
	/usr/bin/clang++ -v -O3 -DNDEBUG -std=gnu++14 --cuda-gpu-arch=sm_86 --cuda-path=/usr/local/cuda-12.6 -x cuda -o cudaBB cuda_BB_kernel.cu -L"/usr/local/cuda-12.6/lib64" -lcudart 

cudaCB: cuda_CB_kernel.cu
	/usr/bin/clang++ -v -O3 -DNDEBUG -std=gnu++14 --cuda-gpu-arch=sm_86 --cuda-path=/usr/local/cuda-12.6 -x cuda -o cudaCB cuda_CB_kernel.cu -L"/usr/local/cuda-12.6/lib64" -lcudart 

ompBB: omp_BB_kernel.cpp
	/usr/bin/clang++ -x c++ -O3 -DNDEBUG -std=gnu++14 -fopenmp -fopenmp-offload-mandatory  -fopenmp-targets=nvptx64-nvidia-cuda -DOMP_TARGET -DOPENMP_TARGET -DUSE_OPENMP_TARGET -L/usr/lib/llvm-18/lib -o ompBB omp_BB_kernel.cpp 

ompCB: omp_CB_kernel.cpp
	/usr/bin/clang++ -x c++ -O3 -DNDEBUG -std=gnu++14 -fopenmp -fopenmp-offload-mandatory  -fopenmp-targets=nvptx64-nvidia-cuda -DOMP_TARGET -DOPENMP_TARGET -DUSE_OPENMP_TARGET -L/usr/lib/llvm-18/lib -o ompCB omp_CB_kernel.cpp 

#/usr/bin/clang++ -x c++ -O3 -DNDEBUG -std=gnu++14 -fopenmp -fopenmp-offload-mandatory  -fopenmp-targets=nvptx64-nvidia-cuda -DOMP_TARGET -DOPENMP_TARGET -DUSE_OPENMP_TARGET -L/usr/lib/llvm-18/lib -Wl,-rpath=/usr/lib/llvm-18/lib -o ompBB omp_BB_kernel.cpp 


test:
	ncu -f -o cudaBB-report --set=roofline -c 2 ./cudaBB
	ncu -f -o cudaCB-report --set=roofline -c 2 ./cudaCB
	LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH ncu -f -o ompBB-report --set=roofline -c 2 ./ompBB
	LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH ncu -f -o ompCB-report --set=roofline -c 2 ./ompCB

check:
	@echo "-------------------- cudaBB --------------------"
	cuobjdump --list-text ./cudaBB
	@echo "\n\n-------------------- cudaCB --------------------"
	cuobjdump --list-text ./cudaCB
	@echo "\n\n-------------------- ompBB --------------------"
	objdump -t --section=omp_offloading_entries ./ompBB
	@echo "\n\n-------------------- ompCB --------------------"
	objdump -t --section=omp_offloading_entries ./ompCB

all: cudaBB cudaCB ompBB ompCB

clean:
	rm -f cudaBB cudaCB
	rm -f ompBB ompCB

