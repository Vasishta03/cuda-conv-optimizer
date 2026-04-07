#include "../include/convolution.h"

__constant__ float d_kernel1d[MAX_KERNEL_SIZE];

__global__ void sepHorizontalKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int kernelRadius)
{
    extern __shared__ float smem[];

    int halo   = kernelRadius;
    int smW    = TILE_W + 2 * halo;
    int tx     = threadIdx.x;
    int ty     = threadIdx.y;
    int outRow = blockIdx.y * TILE_H + ty;
    int outCol = blockIdx.x * TILE_W + tx;
    int baseCol = (int)(blockIdx.x * TILE_W) - halo;

    if (outRow < height) {
        for (int dx = tx; dx < smW; dx += TILE_W) {
            int c = min(max(baseCol + dx, 0), width - 1);
            smem[ty * smW + dx] = input[outRow * width + c];
        }
    }
    __syncthreads();

    if (outRow < height && outCol < width) {
        int kSize = 2 * halo + 1;
        float sum = 0.0f;
        for (int k = 0; k < kSize; ++k)
            sum += smem[ty * smW + tx + k] * d_kernel1d[k];
        output[outRow * width + outCol] = sum;
    }
}

__global__ void sepVerticalKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int kernelRadius)
{
    extern __shared__ float smem[];

    int halo    = kernelRadius;
    int smH     = TILE_H + 2 * halo;
    int tx      = threadIdx.x;
    int ty      = threadIdx.y;
    int outRow  = blockIdx.y * TILE_H + ty;
    int outCol  = blockIdx.x * TILE_W + tx;
    int baseRow = (int)(blockIdx.y * TILE_H) - halo;

    if (outCol < width) {
        for (int dy = ty; dy < smH; dy += TILE_H) {
            int r = min(max(baseRow + dy, 0), height - 1);
            smem[dy * TILE_W + tx] = input[r * width + outCol];
        }
    }
    __syncthreads();

    if (outRow < height && outCol < width) {
        int kSize = 2 * halo + 1;
        float sum = 0.0f;
        for (int k = 0; k < kSize; ++k)
            sum += smem[(ty + k) * TILE_W + tx] * d_kernel1d[k];
        output[outRow * width + outCol] = sum;
    }
}

void launchSeparableConvolution(
    const float* d_in, float* d_out, float* d_temp,
    const float* h_kernel1d,
    int width, int height, int kernelRadius)
{
    int kSize = 2 * kernelRadius + 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel1d, h_kernel1d,
                                  (size_t)kSize * sizeof(float)));

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width + TILE_W - 1) / TILE_W,
              (height + TILE_H - 1) / TILE_H);

    size_t hSmem = (size_t)TILE_H * (TILE_W + 2 * kernelRadius) * sizeof(float);
    sepHorizontalKernel<<<grid, block, hSmem>>>(d_in, d_temp, width, height, kernelRadius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t vSmem = (size_t)(TILE_H + 2 * kernelRadius) * TILE_W * sizeof(float);
    sepVerticalKernel<<<grid, block, vSmem>>>(d_temp, d_out, width, height, kernelRadius);
    CUDA_CHECK(cudaGetLastError());
}
