/***********************************************************************************
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License

  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 

 ***********************************************************************************/

#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <omp.h>
#include "constants.h"

#include "kernels.cpp"

int main()
{
  // a set of variables defined in constants.h
  double  VariablesIn[VarINNUM];

  A           = 0.;    // black hole spin
  INCLINATION = acos(0.25)/PI*180.;     // inclination angle in unit of degree                    
  SIZE        = IMAGE_SIZE; // pixels 
  printf("task1: image size = %d  x  %d  pixels\n",IMAGE_SIZE,IMAGE_SIZE);

  // number of grids; the coordinate of each grid is given by (GridIdxX,GridIdY)
  int ImaDimX, ImaDimY; 

  // number of blocks; the coordinate of each block is given by (blockIdx.x ,blockIdx.y )
  int GridDimX, GridDimY;

  // number of threads; the coordinate of each thread is given by (threadIdx.x,threadIdx.y)
  int BlockDimX, BlockDimY;

  // save output results in files
  double* Results;
  FILE *fp;
  Results = new double[IMAGE_SIZE * IMAGE_SIZE * 3];

  BlockDimX = 100;
  BlockDimY = 1;
  GridDimX  = 1;
  GridDimY  = 50;

  //compute number of grids, to cover the whole image plane
  ImaDimX = (int)ceil((double)IMAGE_SIZE / (BlockDimX * GridDimX));
  ImaDimY = (int)ceil((double)IMAGE_SIZE / (BlockDimY * GridDimY));

#pragma omp target data map(alloc: VariablesIn[0:VarINNUM], \
                                   Results[0: IMAGE_SIZE * IMAGE_SIZE * 3])
{
  #pragma omp target update to (VariablesIn[0:VarINNUM])

  auto start = std::chrono::steady_clock::now();

  for(int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++){
    for(int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++){                      
      task1(Results, VariablesIn, GridIdxX, GridIdxY);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (task1) %f (s)\n", time * 1e-9f);

  #pragma omp target update from (Results[0:IMAGE_SIZE* IMAGE_SIZE * 3])

  //save result to output
  fp=fopen("Output_task1.txt","w");  
  if (fp != NULL) {
    fprintf(fp,"###output data:(alpha,  beta,  redshift)\n");

    for(int j = 0; j < IMAGE_SIZE; j++)
      for(int i = 0; i < IMAGE_SIZE; i++)
      {
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 0]);
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 1]);
        fprintf(fp, "%f\n", (float)Results[3 * (IMAGE_SIZE * j + i) + 2]);
      }
    fclose(fp);
  }

  A           = 0.;    // black hole spin
  INCLINATION = 45.;   // inclination angle in unit of degree                    
  SIZE        = IMAGE_SIZE; // pixels 
  freq_obs    = 340e9; // observed frequency
  printf("task2: image size = %d  x  %d  pixels\n",IMAGE_SIZE,IMAGE_SIZE);

  #pragma omp target update to (VariablesIn[0:VarINNUM])

  start = std::chrono::steady_clock::now();

  for(int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++){
    for(int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++){                      
      task2(Results, VariablesIn, GridIdxX, GridIdxY);
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (task2) %f (s)\n", time * 1e-9f);

  #pragma omp target update from (Results[0:IMAGE_SIZE * IMAGE_SIZE * 3])
}

  fp=fopen("Output_task2.txt","w");  
  if (fp != NULL) {
    fprintf(fp,"###output data:(alpha,  beta, Luminosity (erg/sec))\n");

    for(int j = 0; j < IMAGE_SIZE; j++)
      for(int i = 0; i < IMAGE_SIZE; i++)
      {
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 0]);
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 1]);
        fprintf(fp, "%f\n", (float)Results[3 * (IMAGE_SIZE * j + i) + 2]);
      }
    fclose(fp);
  }

  delete [] Results;
  return 0;
}
