#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define TILE_W 16
#define TILE_H 16

#define MAX_KERNEL_RADIUS 15
#define MAX_KERNEL_SIZE   (2 * MAX_KERNEL_RADIUS + 1)

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));       \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

void cpuConvolution(const float *input, float *output,
                    const float *kernel2d,
                    int width, int height, int kernelRadius);

void launchNaiveConvolution(const float *d_in, float *d_out,
                            const float *d_kernel,
                            int width, int height, int kernelRadius);

void launchTiledConvolution(const float *d_in, float *d_out,
                            const float *d_kernel,
                            int width, int height, int kernelRadius);

void launchSeparableConvolution(const float *d_in, float *d_out,
                                float *d_temp,
                                const float *h_kernel1d,
                                int width, int height, int kernelRadius);

void generateGaussianKernel(float *kernel2d, float *kernel1d,
                             int radius, float sigma);

float maxAbsDiff(const float *a, const float *b, int n);
