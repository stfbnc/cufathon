#ifndef UTILS
#define UTILS

#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"


/*int kb = 1024;
int mb = kb * kb;
cudaDeviceProp props;
*/
namespace gpuUtils
{
    std::string getGpuName(int gpu);
    std::string getComputeCapability(int gpu);
    size_t getGlobalMem(int gpu);
    size_t getConstMem(int gpu);
    size_t getSharedMemPerBlock(int gpu);
    int getRegsPerBlock(int gpu);
    int getWarpSize(int gpu);
    int getMaxThreadsPerBlock(int gpu);
    void getMaxBlockDimension(int gpu, int bDim[3]);
    void getMaxGridDimension(int gpu, int gDim[3]);
    void getGpuInfo(int gpu);
}

#endif

