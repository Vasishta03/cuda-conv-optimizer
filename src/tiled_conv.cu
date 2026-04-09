#include "../include/convolution.h"

__constant__ float d_kernelTiled[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void tiledConvKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int kernelRadius)
{
    extern __shared__ float smem[];

    int halo = kernelRadius;
    int smW  = TILE_W + 2 * halo;
    int smH  = TILE_H + 2 * halo;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int baseRow = (int)(blockIdx.y * TILE_H) - halo;
    int baseCol = (int)(blockIdx.x * TILE_W) - halo;

    for (int dy = ty; dy < smH; dy += TILE_H) {
        for (int dx = tx; dx < smW; dx += TILE_W) {
            int r = min(max(baseRow + dy, 0), height - 1);
            int c = min(max(baseCol + dx, 0), width - 1);
            smem[dy * smW + dx] = input[r * width + c];
        }
    }
    __syncthreads();

    int outCol = blockIdx.x * TILE_W + tx;
    int outRow = blockIdx.y * TILE_H + ty;

    if (outCol < width && outRow < height) {
        int kSize = 2 * halo + 1;
        float sum = 0.0f;
        for (int kr = 0; kr < kSize; ++kr)
            for (int kc = 0; kc < kSize; ++kc)
                sum += smem[(ty + kr) * smW + (tx + kc)] * d_kernelTiled[kr * kSize + kc];
        output[outRow * width + outCol] = sum;
    }
}

void launchTiledConvolution(
    const float* d_in, float* d_out,
    const float* d_kernel,
    int width, int height, int kernelRadius)
{
    int kSize = 2 * kernelRadius + 1;
    int smW   = TILE_W + 2 * kernelRadius;
    int smH   = TILE_H + 2 * kernelRadius;
    size_t smBytes = (size_t)smW * smH * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_kernelTiled, d_kernel,
                                  (size_t)kSize * kSize * sizeof(float)));

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width + TILE_W - 1) / TILE_W,
              (height + TILE_H - 1) / TILE_H);

    tiledConvKernel<<<grid, block, smBytes>>>(d_in, d_out, width, height, kernelRadius);
    CUDA_CHECK(cudaGetLastError());
}
