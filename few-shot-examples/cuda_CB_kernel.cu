#include <stdio.h>
#include <cuda_runtime.h>

__global__ void fma_intensive_kernel(float* output, float a, float b, int iterations) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val1 = 1.0f;
    float val2 = 1.0f;

    for (int i = 0; i < iterations; ++i) {
        val1 = fmaf(val1, a, b);
        val2 = fmaf(val2, a, b);
    }

    // Combine results to prevent optimization
    output[idx] = val1 + val2; 
}

int main() {
    // Configuration parameters
    const int num_elements = 1 << 22;    // 1M elements
    const int threads_per_block = 512;   // Optimal for most GPUs
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    const int sm_aligned_blocks = (blocks / 68) * 68; // we don't care that all the data isn't being compute correctly
    const float a = 0.999f;              // FMA parameter
    const float b = 0.001f;              // FMA parameter
    const int iterations = 100000;        // Sufficient work to saturate SMs

    // Allocate device memory
    float* d_output;
    cudaMalloc(&d_output, num_elements * sizeof(float));

    // Launch kernel with optimal grid configuration
    fma_intensive_kernel<<<sm_aligned_blocks, threads_per_block>>>(d_output, a, b, iterations);

    // Cleanup
    cudaFree(d_output);

    return 0;
}
