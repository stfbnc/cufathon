#include "utils.h"


namespace gpuUtils
{
    std::string getGpuName(int gpu)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        return props.name;
    }

    std::string getComputeCapability(int gpu)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        return std::to_string(props.major) + std::string(".") + std::to_string(props.minor);
    }

    size_t getGlobalMem(int gpu)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        return props.totalGlobalMem / (1024 * 1024);
    }

    size_t getConstMem(int gpu)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        return props.totalConstMem / 1024;
    }

    size_t getSharedMemPerBlock(int gpu)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        return props.sharedMemPerBlock / 1024;
    }

    int getRegsPerBlock(int gpu)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        return props.regsPerBlock;
    }

    int getWarpSize(int gpu)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        return props.warpSize;
    }

    int getMaxThreadsPerBlock(int gpu)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        return props.maxThreadsPerBlock;
    }

    void getMaxBlockDimension(int gpu, int bDim[3])
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        bDim[0] = props.maxThreadsDim[0];
        bDim[1] = props.maxThreadsDim[1];
        bDim[2] = props.maxThreadsDim[2];
    }

    void getMaxGridDimension(int gpu, int gDim[3])
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        gDim[0] = props.maxGridSize[0];
        gDim[1] = props.maxGridSize[1];
        gDim[2] = props.maxGridSize[2];
    }

    void getGpuInfo(int gpu)
    {
        int kb = 1024;
        int mb = kb * kb;
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu);
        std::cout << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::cout << "  Global memory:        " << props.totalGlobalMem / mb << " mb" << std::endl;
        std::cout << "  Shared memory:        " << props.sharedMemPerBlock / kb << " kb" << std::endl;
        std::cout << "  Constant memory:      " << props.totalConstMem / kb << " kb" << std::endl;
        std::cout << "  Block registers:      " << props.regsPerBlock << std::endl;
        std::cout << "  Warp size:            " << props.warpSize << std::endl;
        std::cout << "  Threads per block:    " << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: (" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid dimensions:  (" << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << ")" << std::endl << std::endl;
    }
}

