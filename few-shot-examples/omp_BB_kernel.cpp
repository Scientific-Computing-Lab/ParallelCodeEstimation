#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 32764
#define ITERS 1024
#define THREADS_PER_BLOCK 256
#define BLOCKS ((N * N) / (ITERS * THREADS_PER_BLOCK))
int main() {
    const int matrix_size = N * N;
    const size_t size = matrix_size * sizeof(float);
    float *h_data;
    // Allocate host memory
    h_data = (float*) malloc(size);
    if (!h_data) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }
    // Initialize host array with sample data
    for (int i = 0; i < matrix_size; i++) {
        h_data[i] = 1.0f;
    }
    // Offload computation to target device
    #pragma omp target teams distribute parallel for num_teams(BLOCKS) thread_limit(THREADS_PER_BLOCK) map(tofrom: h_data[0:matrix_size])
    for (unsigned tid = 0; tid < (BLOCKS * THREADS_PER_BLOCK); tid++) {
        unsigned stride = (tid * 7919) % 1023 + 1;
        unsigned start = tid * 12345;
        for (int i = 0; i < ITERS; i++) {
            unsigned idx = (start + i * stride) % matrix_size;
            float val = h_data[idx];
            h_data[idx] = val * 2.0f + 3.0f;
        }
    }
    // Cleanup
    free(h_data);
    return 0;
}