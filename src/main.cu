#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include "../include/convolution.h"

static const int   WARMUP_RUNS = 3;
static const int   BENCH_RUNS  = 10;
static const float TOLERANCE   = 1e-3f;

static void generateTestImage(float* img, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float v = sinf(r * 0.07f) * cosf(c * 0.05f)
                    + cosf((r + c) * 0.03f)
                    + 0.5f * sinf(r * 0.2f + c * 0.13f);
            img[r * width + c] = v * 0.5f + 0.5f;
        }
    }
}

void cpuConvolution(const float* input, float* output,
                    const float* kernel2d,
                    int width, int height, int kernelRadius)
{
    int kSize = 2 * kernelRadius + 1;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f;
            for (int kr = 0; kr < kSize; ++kr) {
                for (int kc = 0; kc < kSize; ++kc) {
                    int r = min(max(row + kr - kernelRadius, 0), height - 1);
                    int c = min(max(col + kc - kernelRadius, 0), width - 1);
                    sum += input[r * width + c] * kernel2d[kr * kSize + kc];
                }
            }
            output[row * width + col] = sum;
        }
    }
}

void generateGaussianKernel(float* kernel2d, float* kernel1d, int radius, float sigma)
{
    int kSize = 2 * radius + 1;
    float sum = 0.0f;
    for (int i = 0; i < kSize; ++i) {
        float x = (float)(i - radius);
        kernel1d[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel1d[i];
    }
    for (int i = 0; i < kSize; ++i) kernel1d[i] /= sum;

    for (int i = 0; i < kSize; ++i)
        for (int j = 0; j < kSize; ++j)
            kernel2d[i * kSize + j] = kernel1d[i] * kernel1d[j];
}

float maxAbsDiff(const float* a, const float* b, int n)
{
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) mx = fmaxf(mx, fabsf(a[i] - b[i]));
    return mx;
}

static float gpuTimeKernel(std::function<void()> fn,
                            int warmup = WARMUP_RUNS, int runs = BENCH_RUNS)
{
    for (int i = 0; i < warmup; ++i) { fn(); CUDA_CHECK(cudaDeviceSynchronize()); }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < runs; ++i) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / (float)runs;
}

static void printGpuInfo()
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

static float ai2D(int radius) {
    int k = 2 * radius + 1;
    return (float)(2 * k * k) / (2.0f * sizeof(float));
}

static float aiSep(int radius) {
    int k = 2 * radius + 1;
    return (float)(4 * k) / (2.0f * sizeof(float));
}

static void runBenchmark(int width, int height)
{
    size_t nPix  = (size_t)width * height;
    size_t bytes = nPix * sizeof(float);
    int radii[]  = {1, 3, 7, 15};

    printf("--- %d x %d (%zu pixels) ---\n", width, height, nPix);
    printf("%-16s  r  %6s  %8s  %8s  %10s  %7s\n",
           "Method", "Err", "ms", "GB/s", "GFLOP/s", "Spdup");

    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    generateTestImage(h_in, width, height);

    float *d_in, *d_out, *d_temp, *d_kernel2d;
    CUDA_CHECK(cudaMalloc(&d_in,      bytes));
    CUDA_CHECK(cudaMalloc(&d_out,     bytes));
    CUDA_CHECK(cudaMalloc(&d_temp,    bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel2d, MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    for (int radius : radii) {
        int   kSize = 2 * radius + 1;
        float sigma = (float)radius / 2.0f;

        float kernel2d[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
        float kernel1d[MAX_KERNEL_SIZE];
        generateGaussianKernel(kernel2d, kernel1d, radius, sigma);
        CUDA_CHECK(cudaMemcpy(d_kernel2d, kernel2d,
                              kSize * kSize * sizeof(float), cudaMemcpyHostToDevice));

        auto t0 = std::chrono::high_resolution_clock::now();
        cpuConvolution(h_in, h_ref, kernel2d, width, height, radius);
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpuMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float cpuGF = (2.0f * nPix * kSize * kSize) / (cpuMs * 1e6f);
        printf("%-16s  %2d  %6s  %8.3f  %8.4f  %10.2f  %7.2fx\n",
               "CPU Reference", radius, "-", cpuMs, 0.0f, cpuGF, 1.0f);

        double flops2d  = 2.0 * nPix * kSize * kSize;
        double flopsSep = 2.0 * nPix * kSize * 2.0;

        float naiveMs = gpuTimeKernel([&](){
            launchNaiveConvolution(d_in, d_out, d_kernel2d, width, height, radius);
        });
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        float naiveErr = maxAbsDiff(h_ref, h_out, (int)nPix);
        printf("%-16s  %2d  %6.1e  %8.3f  %8.3f  %10.2f  %7.2fx  %s\n",
               "Naive GPU", radius, naiveErr, naiveMs,
               (2.0f * bytes) / (naiveMs * 1e6f),
               (float)(flops2d / (naiveMs * 1e6)),
               cpuMs / naiveMs,
               naiveErr < TOLERANCE ? "OK" : "FAIL");

        float tiledMs = gpuTimeKernel([&](){
            launchTiledConvolution(d_in, d_out, d_kernel2d, width, height, radius);
        });
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        float tiledErr = maxAbsDiff(h_ref, h_out, (int)nPix);
        printf("%-16s  %2d  %6.1e  %8.3f  %8.3f  %10.2f  %7.2fx  %s\n",
               "Tiled GPU", radius, tiledErr, tiledMs,
               (2.0f * bytes) / (tiledMs * 1e6f),
               (float)(flops2d / (tiledMs * 1e6)),
               cpuMs / tiledMs,
               tiledErr < TOLERANCE ? "OK" : "FAIL");

        float sepMs = gpuTimeKernel([&](){
            launchSeparableConvolution(d_in, d_out, d_temp, kernel1d, width, height, radius);
        });
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        float sepErr = maxAbsDiff(h_ref, h_out, (int)nPix);
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

int main(int argc, char* argv[])
{
    printf("CUDA Image Convolution Benchmark\n");
    printf("Shared Memory Tiling and Separable Kernels\n\n");
    printGpuInfo();

    if (argc == 3) {
        int W = atoi(argv[1]);
        int H = atoi(argv[2]);
        if (W > 0 && H > 0) { runBenchmark(W, H); return 0; }
    }

    int resolutions[][2] = { {256,256}, {512,512}, {1024,1024}, {2048,2048} };
    for (auto& r : resolutions) runBenchmark(r[0], r[1]);

    return 0;
}
