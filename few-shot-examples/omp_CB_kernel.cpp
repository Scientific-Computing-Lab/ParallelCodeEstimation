#include <stdio.h>
#include <omp.h>
#include <math.h>
int main() {
    // Configuration parameters
    const int num_elements = 1 << 22;     // 4M elements
    const int threads_per_block = 512;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    const int sm_aligned_blocks = (blocks / 68) * 68;
    const int iterations = 100000;
    const int elements_per_launch = sm_aligned_blocks * threads_per_block;
    // By making these non-const, the compiler doesn't optimize-out the val1 val2 update loop
    float a = 0.999f;
    float b = 0.001f;
    // Allocate device memory -- we're not going to do any tofrom copying
    float* d_output = (float*) omp_target_alloc(num_elements * sizeof(float), omp_get_default_device());
    #pragma omp target teams distribute parallel for is_device_ptr(d_output) firstprivate(a,b,iterations) num_teams(sm_aligned_blocks) thread_limit(threads_per_block)
    for (int idx = 0; idx < elements_per_launch; ++idx) {
        float val1 = 1.0f;
        float val2 = 1.0f;
        for (int j = 0; j < iterations; ++j) {
            val1 = fmaf(val1, a, b);
            val2 = fmaf(val2, a, b);
        }
        d_output[idx] = val1 + val2;
    }
    // Cleanup
    omp_target_free(d_output, omp_get_default_device());
    return 0;
}