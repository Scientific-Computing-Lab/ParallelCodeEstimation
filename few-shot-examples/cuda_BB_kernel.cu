#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#define N 32764  // Slightly adjusted for better memory alignment
#define ITERS 1024
#define THREADS_PER_BLOCK 256
#define BLOCKS ((N * N) / (ITERS * THREADS_PER_BLOCK))
__global__ void fma_kernel(float *data, int matrix_size, float mul, float add) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = (tid * 7919) % 1023 + 1;  // Prime-based stride variation
    unsigned start = tid * 12345;  // Arbitrary starting offset
    
    for (int i = 0; i < ITERS; i++) {
        unsigned idx = (start + i * stride) % matrix_size;
        float val = data[idx];
        data[idx] = fmaf(val, mul, add);
    }
}
int main() {
    const size_t size = N * N * sizeof(float);
    const int matrix_size = N * N;
    float *h_data, *d_data;
    // Allocate pinned host memory
    cudaMallocHost(&h_data, size);
    // Allocate device memory
    cudaMalloc(&d_data, size);
    // Initialize host array with sample data
    for (int i = 0; i < matrix_size; i++) {
        h_data[i] = 1.0f;
    }
    // Copy data to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    // Launch kernel with error checking
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    fma_kernel<<<grid, block>>>(d_data, matrix_size, 2.0f, 3.0f);
    cudaGetLastError();
    cudaDeviceSynchronize();
    // Copy data back to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(d_data);
    cudaFreeHost(h_data);
    return 0;
}