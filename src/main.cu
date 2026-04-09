#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../include/convolution.h"

#define WARMUP_RUNS 3
#define BENCH_RUNS  10
#define TOLERANCE   1e-3f

static void generateTestImage(float *img, int width, int height)
{
    int r, c;
    for (r = 0; r < height; r++) {
        for (c = 0; c < width; c++) {
            float v = sinf(r * 0.07f) * cosf(c * 0.05f)
                    + cosf((r + c) * 0.03f)
                    + 0.5f * sinf(r * 0.2f + c * 0.13f);
            img[r * width + c] = v * 0.5f + 0.5f;
        }
    }
}

void cpuConvolution(const float *input, float *output,
                    const float *kernel2d,
                    int width, int height, int kernelRadius)
{
    int kSize = 2 * kernelRadius + 1;
    int row, col, kr, kc;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float sum = 0.0f;
            for (kr = 0; kr < kSize; kr++) {
                for (kc = 0; kc < kSize; kc++) {
                    int r = row + kr - kernelRadius;
                    int c = col + kc - kernelRadius;
                    if (r < 0) r = 0;
                    if (r >= height) r = height - 1;
                    if (c < 0) c = 0;
                    if (c >= width) c = width - 1;
                    sum += input[r * width + c] * kernel2d[kr * kSize + kc];
                }
            }
            output[row * width + col] = sum;
        }
    }
}

void generateGaussianKernel(float *kernel2d, float *kernel1d, int radius, float sigma)
{
    int kSize = 2 * radius + 1;
    float sum = 0.0f;
    int i, j;
    for (i = 0; i < kSize; i++) {
        float x = (float)(i - radius);
        kernel1d[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel1d[i];
    }
    for (i = 0; i < kSize; i++) kernel1d[i] /= sum;
    for (i = 0; i < kSize; i++)
        for (j = 0; j < kSize; j++)
            kernel2d[i * kSize + j] = kernel1d[i] * kernel1d[j];
}

float maxAbsDiff(const float *a, const float *b, int n)
{
    float mx = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static double getTimeMs(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static float timeNaive(const float *d_in, float *d_out, const float *d_kernel,
                        int width, int height, int radius)
{
    int i;
    float ms = 0.0f;
    cudaEvent_t t0, t1;
    for (i = 0; i < WARMUP_RUNS; i++) {
        launchNaiveConvolution(d_in, d_out, d_kernel, width, height, radius);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (i = 0; i < BENCH_RUNS; i++)
        launchNaiveConvolution(d_in, d_out, d_kernel, width, height, radius);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / (float)BENCH_RUNS;
}

static float timeTiled(const float *d_in, float *d_out, const float *d_kernel,
                        int width, int height, int radius)
{
    int i;
    float ms = 0.0f;
    cudaEvent_t t0, t1;
    for (i = 0; i < WARMUP_RUNS; i++) {
        launchTiledConvolution(d_in, d_out, d_kernel, width, height, radius);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (i = 0; i < BENCH_RUNS; i++)
        launchTiledConvolution(d_in, d_out, d_kernel, width, height, radius);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / (float)BENCH_RUNS;
}

static float timeSeparable(const float *d_in, float *d_out, float *d_temp,
                             const float *kernel1d,
                             int width, int height, int radius)
{
    int i;
    float ms = 0.0f;
    cudaEvent_t t0, t1;
    for (i = 0; i < WARMUP_RUNS; i++) {
        launchSeparableConvolution(d_in, d_out, d_temp, kernel1d, width, height, radius);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (i = 0; i < BENCH_RUNS; i++)
        launchSeparableConvolution(d_in, d_out, d_temp, kernel1d, width, height, radius);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / (float)BENCH_RUNS;
}

static void printGpuInfo(void)
{
    int dev = 0;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("GPU : %s\n", p.name);
    printf("SMs : %d   Peak BW: %.1f GB/s   Shared mem: %zu KB\n\n",
           p.multiProcessorCount,
           2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6,
           p.sharedMemPerBlock / 1024);
}

static float ai2D(int radius)
{
    int k = 2 * radius + 1;
    return (float)(2 * k * k) / (2.0f * sizeof(float));
}

static float aiSep(int radius)
{
    int k = 2 * radius + 1;
    return (float)(4 * k) / (2.0f * sizeof(float));
}

static void runBenchmark(int width, int height)
{
    int radii[4] = {1, 3, 7, 15};
    int ri;
    size_t nPix  = (size_t)width * height;
    size_t bytes = nPix * sizeof(float);

    printf("--- %d x %d (%zu pixels) ---\n", width, height, nPix);
    printf("%-16s  r  %6s  %8s  %8s  %10s  %7s\n",
           "Method", "Err", "ms", "GB/s", "GFLOP/s", "Spdup");

    float *h_in  = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    generateTestImage(h_in, width, height);

    float *d_in, *d_out, *d_temp, *d_kernel2d;
    CUDA_CHECK(cudaMalloc(&d_in,       bytes));
    CUDA_CHECK(cudaMalloc(&d_out,      bytes));
    CUDA_CHECK(cudaMalloc(&d_temp,     bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel2d, MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    for (ri = 0; ri < 4; ri++) {
        int radius = radii[ri];
        int kSize  = 2 * radius + 1;
        float sigma = (float)radius / 2.0f;
        double t0, t1;
        float cpuMs, cpuGF, naiveMs, tiledMs, sepMs;
        float naiveErr, tiledErr, sepErr;
        double flops2d, flopsSep;

        float kernel2d[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
        float kernel1d[MAX_KERNEL_SIZE];
        generateGaussianKernel(kernel2d, kernel1d, radius, sigma);
        CUDA_CHECK(cudaMemcpy(d_kernel2d, kernel2d,
                              kSize * kSize * sizeof(float), cudaMemcpyHostToDevice));

        t0 = getTimeMs();
        cpuConvolution(h_in, h_ref, kernel2d, width, height, radius);
        t1 = getTimeMs();
        cpuMs = (float)(t1 - t0);
        cpuGF = (2.0f * nPix * kSize * kSize) / (cpuMs * 1e6f);
        printf("%-16s  %2d  %6s  %8.3f  %8.4f  %10.2f  %7.2fx\n",
               "CPU Reference", radius, "-", cpuMs, 0.0f, cpuGF, 1.0f);

        flops2d  = 2.0 * nPix * kSize * kSize;
        flopsSep = 2.0 * nPix * kSize * 2.0;

        naiveMs = timeNaive(d_in, d_out, d_kernel2d, width, height, radius);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        naiveErr = maxAbsDiff(h_ref, h_out, (int)nPix);
        printf("%-16s  %2d  %6.1e  %8.3f  %8.3f  %10.2f  %7.2fx  %s\n",
               "Naive GPU", radius, naiveErr, naiveMs,
               (2.0f * bytes) / (naiveMs * 1e6f),
               (float)(flops2d / (naiveMs * 1e6)),
               cpuMs / naiveMs,
               naiveErr < TOLERANCE ? "OK" : "FAIL");

        tiledMs = timeTiled(d_in, d_out, d_kernel2d, width, height, radius);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        tiledErr = maxAbsDiff(h_ref, h_out, (int)nPix);
        printf("%-16s  %2d  %6.1e  %8.3f  %8.3f  %10.2f  %7.2fx  %s\n",
               "Tiled GPU", radius, tiledErr, tiledMs,
               (2.0f * bytes) / (tiledMs * 1e6f),
               (float)(flops2d / (tiledMs * 1e6)),
               cpuMs / tiledMs,
               tiledErr < TOLERANCE ? "OK" : "FAIL");

        sepMs = timeSeparable(d_in, d_out, d_temp, kernel1d, width, height, radius);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        sepErr = maxAbsDiff(h_ref, h_out, (int)nPix);
        printf("%-16s  %2d  %6.1e  %8.3f  %8.3f  %10.2f  %7.2fx  %s\n",
               "Separable GPU", radius, sepErr, sepMs,
               (2.0f * bytes) / (sepMs * 1e6f),
               (float)(flopsSep / (sepMs * 1e6)),
               cpuMs / sepMs,
               sepErr < TOLERANCE ? "OK" : "FAIL");

        printf("  AI(2D)=%.1f  AI(Sep)=%.1f  Tiled %.2fx over Naive\n\n",
               ai2D(radius), aiSep(radius), naiveMs / tiledMs);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_kernel2d));
    free(h_in); free(h_ref); free(h_out);
}

int main(int argc, char *argv[])
{
    int resolutions[4][2] = { {256,256}, {512,512}, {1024,1024}, {2048,2048} };
    int i;

    printf("CUDA Image Convolution Benchmark\n");
    printf("Shared Memory Tiling and Separable Kernels\n\n");
    printGpuInfo();

    if (argc == 3) {
        int W = atoi(argv[1]);
        int H = atoi(argv[2]);
        if (W > 0 && H > 0) { runBenchmark(W, H); return 0; }
    }

    for (i = 0; i < 4; i++) runBenchmark(resolutions[i][0], resolutions[i][1]);

    return 0;
}
