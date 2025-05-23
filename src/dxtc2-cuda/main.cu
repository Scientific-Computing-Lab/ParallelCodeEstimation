/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Utilities and system includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <float.h>  // for FLT_MAX
#include <chrono>

#include "helper_math.h"
#include "dds.h"
#include "permutations.h"
#include "shrUtils.h"

#define ERROR_THRESHOLD 0.02f

#define NUM_THREADS 64  // Number of threads per block.

// Use power method to find the first eigenvector.
// https://en.wikipedia.org/wiki/Power_iteration
inline __device__ __host__ float3 firstEigenVector(float matrix[6]) {
  // 8 iterations seems to be more than enough.

  float3 v = make_float3(1.0f, 1.0f, 1.0f);

  for (int i = 0; i < 8; i++) {
    float x = v.x * matrix[0] + v.y * matrix[1] + v.z * matrix[2];
    float y = v.x * matrix[1] + v.y * matrix[3] + v.z * matrix[4];
    float z = v.x * matrix[2] + v.y * matrix[4] + v.z * matrix[5];
    float m = max(max(x, y), z);
    float iv = 1.0f / m;
    v = make_float3(x * iv, y * iv, z * iv);
  }

  return v;
}

inline __device__ void colorSums(const float3 *colors, float3 *sums) {
  const int idx = threadIdx.x;

  sums[idx] = colors[idx];
  sums[idx] += sums[idx ^ 8];
  sums[idx] += sums[idx ^ 4];
  sums[idx] += sums[idx ^ 2];
  sums[idx] += sums[idx ^ 1];
}

inline __device__ float3 bestFitLine(const float3 *colors, float3 color_sum)
{
  // Compute covariance matrix of the given colors.
  const int idx = threadIdx.x;

  float3 diff = colors[idx] - color_sum * (1.0f / 16.0f);

  // @@ Eliminate two-way bank conflicts here.
  // @@ It seems that doing that and unrolling the reduction doesn't help...
  __shared__ float covariance[16 * 6];

  covariance[6 * idx + 0] = diff.x * diff.x;  // 0, 6, 12, 2, 8, 14, 4, 10, 0
  covariance[6 * idx + 1] = diff.x * diff.y;
  covariance[6 * idx + 2] = diff.x * diff.z;
  covariance[6 * idx + 3] = diff.y * diff.y;
  covariance[6 * idx + 4] = diff.y * diff.z;
  covariance[6 * idx + 5] = diff.z * diff.z;

  for (int d = 8; d > 0; d >>= 1) {
    if (idx < d) {
      covariance[6 * idx + 0] += covariance[6 * (idx + d) + 0];
      covariance[6 * idx + 1] += covariance[6 * (idx + d) + 1];
      covariance[6 * idx + 2] += covariance[6 * (idx + d) + 2];
      covariance[6 * idx + 3] += covariance[6 * (idx + d) + 3];
      covariance[6 * idx + 4] += covariance[6 * (idx + d) + 4];
      covariance[6 * idx + 5] += covariance[6 * (idx + d) + 5];
    }
  }

  // Compute first eigen vector.
  return firstEigenVector(covariance);
}

template <class T>
__device__ inline void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

////////////////////////////////////////////////////////////////////////////////
// Sort colors
////////////////////////////////////////////////////////////////////////////////
__device__ void sortColors(const float *values, int *ranks) {
  const int tid = threadIdx.x;

  int rank = 0;

#pragma unroll

  for (int i = 0; i < 16; i++) {
    rank += (values[i] < values[tid]);
  }

  ranks[tid] = rank;

  // Resolve elements with the same index.
  for (int i = 0; i < 15; i++) {
    if (tid > i && ranks[tid] == ranks[i]) {
      ++ranks[tid];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Load color block to shared mem
////////////////////////////////////////////////////////////////////////////////
__device__ void loadColorBlock(const uint *image, float3 colors[16],
                               float3 sums[16], int xrefs[16], int blockOffset)
{
  const int bid = blockIdx.x + blockOffset;
  const int idx = threadIdx.x;

  __shared__ float dps[16];

  float3 tmp;

  if (idx < 16) {
    // Read color and copy to shared mem.
    uint c = image[(bid)*16 + idx];

    colors[idx].x = ((c >> 0) & 0xFF) * (1.0f / 255.0f);
    colors[idx].y = ((c >> 8) & 0xFF) * (1.0f / 255.0f);
    colors[idx].z = ((c >> 16) & 0xFF) * (1.0f / 255.0f);

    // Sort colors along the best fit line.
    colorSums(colors, sums);

    float3 axis = bestFitLine(colors, sums[0]);

    dps[idx] = colors[idx].x * axis.x + 
               colors[idx].y * axis.y +
               colors[idx].z * axis.z;

    sortColors(dps, xrefs);

    tmp = colors[idx];

    colors[xrefs[idx]] = tmp;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Round color to RGB565 and expand
////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 roundAndExpand(float3 v, ushort *w) {
  v.x = rintf(__saturatef(v.x) * 31.0f);
  v.y = rintf(__saturatef(v.y) * 63.0f);
  v.z = rintf(__saturatef(v.z) * 31.0f);

  *w = ((ushort)v.x << 11) | ((ushort)v.y << 5) | (ushort)v.z;
  v.x *= 0.03227752766457f;  // approximate integer bit expansion.
  v.y *= 0.01583151765563f;
  v.z *= 0.03227752766457f;
  return v;
}

__constant__ float alphaTable4[4] = {9.0f, 0.0f, 6.0f, 3.0f};
__constant__ float alphaTable3[4] = {4.0f, 0.0f, 2.0f, 2.0f};
__constant__ const int prods4[4] = {0x090000, 0x000900, 0x040102, 0x010402};
__constant__ const int prods3[4] = {0x040000, 0x000400, 0x040101, 0x010401};

#define USE_TABLES 1

////////////////////////////////////////////////////////////////////////////////
// Evaluate permutations
////////////////////////////////////////////////////////////////////////////////
static __device__ float evalPermutation4(const float3 *colors, uint permutation,
                                         ushort *start, ushort *end,
                                         float3 color_sum) {
// Compute endpoints using least squares.
#if USE_TABLES
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  int akku = 0;

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++) {
    const uint bits = permutation >> (2 * i);

    alphax_sum += alphaTable4[bits & 3] * colors[i];
    akku += prods4[bits & 3];
  }

  float alpha2_sum = float(akku >> 16);
  float beta2_sum = float((akku >> 8) & 0xff);
  float alphabeta_sum = float((akku >> 0) & 0xff);
  float3 betax_sum = (9.0f * color_sum) - alphax_sum;
#else
  float alpha2_sum = 0.0f;
  float beta2_sum = 0.0f;
  float alphabeta_sum = 0.0f;
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++) {
    const uint bits = permutation >> (2 * i);

    float beta = (bits & 1);

    if (bits & 2) {
      beta = (1 + beta) * (1.0f / 3.0f);
    }

    float alpha = 1.0f - beta;

    alpha2_sum += alpha * alpha;
    beta2_sum += beta * beta;
    alphabeta_sum += alpha * beta;
    alphax_sum += alpha * colors[i];
  }

  float3 betax_sum = color_sum - alphax_sum;
#endif

  // alpha2, beta2, alphabeta and factor could be precomputed for each
  // permutation, but it's faster to recompute them.
  const float factor =
      1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

  float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
  float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

  // Round a, b to the closest 5-6-5 color and expand...
  a = roundAndExpand(a, start);
  b = roundAndExpand(b, end);

  // compute the error
  float3 e = a * a * alpha2_sum + b * b * beta2_sum +
             2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

  return (0.111111111111f) * (e.x + e.y + e.z);
}

static __device__ float evalPermutation3(const float3 *colors, uint permutation,
                                         ushort *start, ushort *end,
                                         float3 color_sum) {
// Compute endpoints using least squares.
#if USE_TABLES
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  int akku = 0;

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++) {
    const uint bits = permutation >> (2 * i);

    alphax_sum += alphaTable3[bits & 3] * colors[i];
    akku += prods3[bits & 3];
  }

  float alpha2_sum = float(akku >> 16);
  float beta2_sum = float((akku >> 8) & 0xff);
  float alphabeta_sum = float((akku >> 0) & 0xff);
  float3 betax_sum = (4.0f * color_sum) - alphax_sum;
#else
  float alpha2_sum = 0.0f;
  float beta2_sum = 0.0f;
  float alphabeta_sum = 0.0f;
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++) {
    const uint bits = permutation >> (2 * i);

    float beta = (bits & 1);

    if (bits & 2) {
      beta = 0.5f;
    }

    float alpha = 1.0f - beta;

    alpha2_sum += alpha * alpha;
    beta2_sum += beta * beta;
    alphabeta_sum += alpha * beta;
    alphax_sum += alpha * colors[i];
  }

  float3 betax_sum = color_sum - alphax_sum;
#endif

  const float factor =
      1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

  float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
  float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

  // Round a, b to the closest 5-6-5 color and expand...
  a = roundAndExpand(a, start);
  b = roundAndExpand(b, end);

  // compute the error
  float3 e = a * a * alpha2_sum + b * b * beta2_sum +
             2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

  return (0.25f) * (e.x + e.y + e.z);
}

__device__ void evalAllPermutations(const float3 *colors,
                                    const uint *permutations, ushort &bestStart,
                                    ushort &bestEnd, uint &bestPermutation,
                                    float *errors, float3 color_sum,
                                    cg::thread_block cta) {
  const int idx = threadIdx.x;

  float bestError = FLT_MAX;

  __shared__ uint s_permutations[160];

  for (int i = 0; i < 16; i++) {
    int pidx = idx + NUM_THREADS * i;

    if (pidx >= 992) {
      break;
    }

    ushort start, end;
    uint permutation = permutations[pidx];

    if (pidx < 160) {
      s_permutations[pidx] = permutation;
    }

    float error =
        evalPermutation4(colors, permutation, &start, &end, color_sum);

    if (error < bestError) {
      bestError = error;
      bestPermutation = permutation;
      bestStart = start;
      bestEnd = end;
    }
  }

  if (bestStart < bestEnd) {
    swap(bestEnd, bestStart);
    bestPermutation ^= 0x55555555;  // Flip indices.
  }

  cg::sync(cta);  // Sync here to ensure s_permutations is valid going forward

  for (int i = 0; i < 3; i++) {
    int pidx = idx + NUM_THREADS * i;

    if (pidx >= 160) {
      break;
    }

    ushort start, end;
    uint permutation = s_permutations[pidx];
    float error =
        evalPermutation3(colors, permutation, &start, &end, color_sum);

    if (error < bestError) {
      bestError = error;
      bestPermutation = permutation;
      bestStart = start;
      bestEnd = end;

      if (bestStart > bestEnd) {
        swap(bestEnd, bestStart);
        bestPermutation ^=
            (~bestPermutation >> 1) & 0x55555555;  // Flip indices.
      }
    }
  }

  errors[idx] = bestError;
}

////////////////////////////////////////////////////////////////////////////////
// Find index with minimum error
////////////////////////////////////////////////////////////////////////////////
__device__ int findMinError(float *errors, cg::thread_block cta) {
  const int idx = threadIdx.x;
  __shared__ int indices[NUM_THREADS];
  indices[idx] = idx;

  cg::sync(cta);

  for (int d = NUM_THREADS / 2; d > 0; d >>= 1) {
    float err0 = errors[idx];
    float err1 = (idx + d) < NUM_THREADS ? errors[idx + d] : FLT_MAX;
    int index1 = (idx + d) < NUM_THREADS ? indices[idx + d] : 0;

    cg::sync(cta);

    if (err1 < err0) {
      errors[idx] = err1;
      indices[idx] = index1;
    }

    cg::sync(cta);
  }

  return indices[0];
}

////////////////////////////////////////////////////////////////////////////////
// Save DXT block
////////////////////////////////////////////////////////////////////////////////
__device__ void saveBlockDXT1(ushort start, ushort end, uint permutation,
                              int xrefs[16], uint2 *result, int blockOffset) {
  const int bid = blockIdx.x + blockOffset;

  if (start == end) {
    permutation = 0;
  }

  // Reorder permutation.
  uint indices = 0;

  for (int i = 0; i < 16; i++) {
    int ref = xrefs[i];
    indices |= ((permutation >> (2 * ref)) & 3) << (2 * i);
  }

  // Write endpoints.
  result[bid].x = (end << 16) | start;

  // Write palette indices.
  result[bid].y = indices;
}

////////////////////////////////////////////////////////////////////////////////
// Compress color block
////////////////////////////////////////////////////////////////////////////////
__global__ void compress(const uint *permutations, const uint *image,
                         uint2 *result, int blockOffset) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  const int idx = threadIdx.x;

  __shared__ float3 colors[16];
  __shared__ float3 sums[16];
  __shared__ int xrefs[16];

  loadColorBlock(image, colors, sums, xrefs, blockOffset);

  cg::sync(cta);

  ushort bestStart, bestEnd;
  uint bestPermutation;

  __shared__ float errors[NUM_THREADS];

  evalAllPermutations(colors, permutations, bestStart, bestEnd, bestPermutation,
                      errors, sums[0], cta);

  // Use a parallel reduction to find minimum error.
  const int minIdx = findMinError(errors, cta);

  cg::sync(cta);

  // Only write the result of the winner thread.
  if (idx == minIdx) {
    saveBlockDXT1(bestStart, bestEnd, bestPermutation, xrefs, result,
                  blockOffset);
  }
}

// Helper structs and functions to validate the output of the compressor.
// We cannot simply do a bitwise compare, because different compilers produce
// different
// results for different targets due to floating point arithmetic.

union Color32 {
  struct {
    unsigned char b, g, r, a;
  };
  unsigned int u;
};

union Color16 {
  struct {
    unsigned short b : 5;
    unsigned short g : 6;
    unsigned short r : 5;
  };
  unsigned short u;
};

struct BlockDXT1 {
  Color16 col0;
  Color16 col1;
  union {
    unsigned char row[4];
    unsigned int indices;
  };

  void decompress(Color32 colors[16]) const;
};

void BlockDXT1::decompress(Color32 *colors) const {
  Color32 palette[4];

  // Does bit expansion before interpolation.
  palette[0].b = (col0.b << 3) | (col0.b >> 2);
  palette[0].g = (col0.g << 2) | (col0.g >> 4);
  palette[0].r = (col0.r << 3) | (col0.r >> 2);
  palette[0].a = 0xFF;

  palette[1].r = (col1.r << 3) | (col1.r >> 2);
  palette[1].g = (col1.g << 2) | (col1.g >> 4);
  palette[1].b = (col1.b << 3) | (col1.b >> 2);
  palette[1].a = 0xFF;

  if (col0.u > col1.u) {
    // Four-color block: derive the other two colors.
    palette[2].r = (2 * palette[0].r + palette[1].r) / 3;
    palette[2].g = (2 * palette[0].g + palette[1].g) / 3;
    palette[2].b = (2 * palette[0].b + palette[1].b) / 3;
    palette[2].a = 0xFF;

    palette[3].r = (2 * palette[1].r + palette[0].r) / 3;
    palette[3].g = (2 * palette[1].g + palette[0].g) / 3;
    palette[3].b = (2 * palette[1].b + palette[0].b) / 3;
    palette[3].a = 0xFF;
  } else {
    // Three-color block: derive the other color.
    palette[2].r = (palette[0].r + palette[1].r) / 2;
    palette[2].g = (palette[0].g + palette[1].g) / 2;
    palette[2].b = (palette[0].b + palette[1].b) / 2;
    palette[2].a = 0xFF;

    palette[3].r = 0x00;
    palette[3].g = 0x00;
    palette[3].b = 0x00;
    palette[3].a = 0x00;
  }

  for (int i = 0; i < 16; i++) {
    colors[i] = palette[(indices >> (2 * i)) & 0x3];
  }
}

static int compareColors(const Color32 *b0, const Color32 *b1) {
  int sum = 0;

  for (int i = 0; i < 16; i++) {
    int r = (b0[i].r - b1[i].r);
    int g = (b0[i].g - b1[i].g);
    int b = (b0[i].b - b1[i].b);
    sum += r * r + g * g + b * b;
  }

  return sum;
}

static int compareBlock(const BlockDXT1 *b0, const BlockDXT1 *b1) {
  Color32 colors0[16];
  Color32 colors1[16];

  if (memcmp(b0, b1, sizeof(BlockDXT1)) == 0) {
    return 0;
  } else {
    b0->decompress(colors0);
    b1->decompress(colors1);

    return compareColors(colors0, colors1);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  // Load input image.
  unsigned char *data = NULL;
  uint W, H;

  const char *image_path = argv[1];

  const char *reference_image_path = argv[2];

  const int numIterations = atoi(argv[3]);

  if (!shrLoadPPM4ub(image_path, (unsigned char **)&data, &W, &H)) {

    printf("Error, unable to open source image file <%s>\n", image_path);

    exit(EXIT_FAILURE);
  }

  uint w = W, h = H;

  printf("Image Loaded '%s', %d x %d pixels\n\n", image_path, w, h);

  // Allocate input image.
  const uint memSize = w * h * 4;
  assert(0 != memSize);
  uint *block_image = (uint *)malloc(memSize);

  // Convert linear image to block linear.
  for (uint by = 0; by < h / 4; by++) {
    for (uint bx = 0; bx < w / 4; bx++) {
      for (int i = 0; i < 16; i++) {
        const int x = i & 3;
        const int y = i / 4;
        block_image[(by * w / 4 + bx) * 16 + i] =
            ((uint *)data)[(by * 4 + y) * 4 * (W / 4) + bx * 4 + x];
      }
    }
  }

  // copy into global mem
  uint *d_data = NULL;
  cudaMalloc((void **)&d_data, memSize);

  // Result
  uint *d_result = NULL;
  const uint compressedSize = (w / 4) * (h / 4) * 8;
  cudaMalloc((void **)&d_result, compressedSize);
  uint *h_result = (uint *)malloc(compressedSize);

  // Compute permutations.
  uint permutations[1024];
  computePermutations(permutations);

  // Copy permutations host to devie.
  uint *d_permutations = NULL;
  cudaMalloc((void **)&d_permutations, 1024 * sizeof(uint));
  cudaMemcpy(d_permutations, permutations, 1024 * sizeof(uint),
                             cudaMemcpyHostToDevice);

  // Copy image from host to device
  cudaMemcpy(d_data, block_image, memSize, cudaMemcpyHostToDevice);

  // Determine launch configuration and run timed computation numIterations
  // times
  uint blocks = ((w + 3) / 4) *
                ((h + 3) / 4);  // rounds up by 1 block in each dim if %4 != 0

  // Restrict the numbers of blocks to launch on low end GPUs to avoid kernel
  // timeout
  //int blocksPerLaunch = min(blocks, 768 * deviceProp.multiProcessorCount);
  int blocksPerLaunch = MIN(blocks, 768 * 24);

  printf("Running DXT Compression on %u x %u image...\n", w, h);
  printf("\n%u Blocks, %u Threads per Block, %u Threads in Grid...\n\n", blocks,
         NUM_THREADS, blocks * NUM_THREADS);

  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; ++i) {
    for (int j = 0; j < (int)blocks; j += blocksPerLaunch) {
      compress<<<MIN(blocksPerLaunch, blocks - j), NUM_THREADS>>>(
          d_permutations, d_data, (uint2 *)d_result, j);
    }
  }

  // sync to host
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / numIterations);

  // copy result data from device to host
  cudaMemcpy(h_result, d_result, compressedSize, cudaMemcpyDeviceToHost);

  // Write out result data to DDS file
  char output_filename[1024];
  strcpy(output_filename, image_path);
  strcpy(output_filename + strlen(image_path) - 3, "dds");
  FILE *fp = fopen(output_filename, "wb");

  if (fp == 0) {
    printf("Error, unable to open output image <%s>\n", output_filename);
    exit(EXIT_FAILURE);
  }

  DDSHeader header;
  header.fourcc = FOURCC_DDS;
  header.size = 124;
  header.flags = (DDSD_WIDTH | DDSD_HEIGHT | DDSD_CAPS | DDSD_PIXELFORMAT |
                  DDSD_LINEARSIZE);
  header.height = h;
  header.width = w;
  header.pitch = compressedSize;
  header.depth = 0;
  header.mipmapcount = 0;
  memset(header.reserved, 0, sizeof(header.reserved));
  header.pf.size = 32;
  header.pf.flags = DDPF_FOURCC;
  header.pf.fourcc = FOURCC_DXT1;
  header.pf.bitcount = 0;
  header.pf.rmask = 0;
  header.pf.gmask = 0;
  header.pf.bmask = 0;
  header.pf.amask = 0;
  header.caps.caps1 = DDSCAPS_TEXTURE;
  header.caps.caps2 = 0;
  header.caps.caps3 = 0;
  header.caps.caps4 = 0;
  header.notused = 0;
  fwrite(&header, sizeof(DDSHeader), 1, fp);
  fwrite(h_result, compressedSize, 1, fp);
  fclose(fp);

  fp = fopen(reference_image_path, "rb");

  if (fp == 0) {
    printf("Error, unable to open reference image\n");

    exit(EXIT_FAILURE);
  }

  fseek(fp, sizeof(DDSHeader), SEEK_SET);
  uint referenceSize = (W / 4) * (H / 4) * 8;
  uint *reference = (uint *)malloc(referenceSize);
  fread(reference, referenceSize, 1, fp);
  fclose(fp);

  printf("\nChecking accuracy...\n");
  float rms = 0;

  for (uint y = 0; y < h; y += 4) {
    for (uint x = 0; x < w; x += 4) {
      uint referenceBlockIdx = ((y / 4) * (W / 4) + (x / 4));
      uint resultBlockIdx = ((y / 4) * (w / 4) + (x / 4));

      int cmp = compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx,
                             ((BlockDXT1 *)reference) + referenceBlockIdx);

      //if (cmp != 0.0f) {
      //  printf("Deviation at (%4d,%4d):\t%f rms\n", x / 4, y / 4,
      //         float(cmp) / 16 / 3);
      //}

      rms += cmp;
    }
  }

  rms /= w * h * 3;

  // Free allocated resources
  cudaFree(d_permutations);
  cudaFree(d_data);
  cudaFree(d_result);
  free(data);
  free(block_image);
  free(h_result);
  free(reference);

  printf("RMS(reference, result) = %f\n\n", rms);
  printf(rms <= ERROR_THRESHOLD ? "Test passed\n" : "Test failed!\n");
  /* Return zero if test passed, one otherwise */
  return rms > ERROR_THRESHOLD;
}
