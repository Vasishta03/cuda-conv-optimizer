#include "../include/convolution.h"

__constant__ float d_kernelNaive[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void naiveConvKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int kernelRadius)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int kSize = 2 * kernelRadius + 1;
    float sum = 0.0f;

    for (int kr = 0; kr < kSize; ++kr) {
        for (int kc = 0; kc < kSize; ++kc) {
            int r = min(max(row + kr - kernelRadius, 0), height - 1);
            int c = min(max(col + kc - kernelRadius, 0), width - 1);
            sum += input[r * width + c] * d_kernelNaive[kr * kSize + kc];
        }
    }
    output[row * width + col] = sum;
}

void launchNaiveConvolution(
    const float* d_in, float* d_out,
    const float* d_kernel,
    int width, int height, int kernelRadius)
{
    int kSize = 2 * kernelRadius + 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernelNaive, d_kernel,
                                  (size_t)kSize * kSize * sizeof(float)));

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width + TILE_W - 1) / TILE_W,
              (height + TILE_H - 1) / TILE_H);

    naiveConvKernel<<<grid, block>>>(d_in, d_out, width, height, kernelRadius);
    CUDA_CHECK(cudaGetLastError());
}
