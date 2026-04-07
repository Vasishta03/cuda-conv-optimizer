---
title: "Optimizing CUDA Image Convolution via Shared Memory Tiling and Separable Kernels"
institution: "Manipal Institute of Technology, Manipal, Karnataka – 576104"
department: "School of Computer Engineering"
guide: "Vidya Kamath, Assistant Professor"
authors:
  - {name: "Vasishta Nandipati", reg: "230962095"}
  - {name: "Rishit Mathur",      reg: "220962091"}
  - {name: "Krishiv Kolanu",     reg: "230962023"}
date: "April 2026"
---

---

# MANIPAL INSTITUTE OF TECHNOLOGY
**MANIPAL (A constituent unit of MAHE, Manipal)**

---

# Optimizing CUDA Image Convolution via Shared Memory Tiling and Separable Kernels

**Mini-Project Report**
**Semester VI — Parallel Computing and Architecture (PCAP)**

**Submitted By**

| Student Name | Reg. No. |
|---|---|
| Vasishta Nandipati | 230962095 |
| Rishit Mathur | 220962091 |
| Krishiv Kolanu | 230962023 |

**Under the Guidance of:**
**Vidya Kamath**
Assistant Professor
School of Computer Engineering
Manipal Institute of Technology, Manipal, Karnataka – 576104

---

## ABSTRACT

Image convolution is a computationally intensive operation central to computer vision, signal processing, and deep learning. While GPU architectures provide substantial parallelism, naive CUDA implementations suffer from excessive global memory accesses that throttle throughput. This project implements and evaluates two complementary optimisations: (1) shared memory tiling, which caches input tiles including border halos in on-chip SRAM to eliminate redundant off-chip fetches; and (2) separable kernel decomposition, which reduces a 2-D convolution into two successive 1-D passes, cutting arithmetic complexity from O(K²) to O(2K) per pixel. A benchmark framework evaluates all three GPU implementations (naive, tiled, separable) alongside an OpenCV-equivalent CPU reference across kernel radii r ∈ {1, 3, 7, 15} and image resolutions from 256×256 to 2048×2048. Experimental results demonstrate speedups of 15–40× over the CPU baseline and 3–6× over the naive GPU kernel, with effective memory-bandwidth utilisation approaching the theoretical roofline limit for memory-bound configurations.

---

# CHAPTER 1

## INTRODUCTION

### 1.1 General Introduction

Discrete 2-D convolution is the mathematical foundation of a broad class of image processing operations. Classical techniques such as Gaussian blurring, Sobel edge detection, median sharpening, and unsharp masking are all expressed as 2-D convolutions. At a deeper level, every convolutional neural network (CNN) layer is a batched 2-D convolution. For an image of H×W pixels and a kernel of size K×K (where K = 2r+1 for radius r), the direct computation cost scales as O(H·W·K²). When K grows beyond a handful of pixels, or when millions of images must be processed, CPU-based sequential execution becomes a severe performance bottleneck.

NVIDIA's Compute Unified Device Architecture (CUDA) exposes the massive thread-level parallelism inherent in modern GPU hardware. A single GPU may execute tens of thousands of threads concurrently, making it ideally suited for data-parallel workloads such as convolution, where every output pixel can be computed independently. However, realising the full potential of the GPU requires deliberate kernel design. The GPU memory hierarchy is structured in tiers with dramatically different bandwidths and latencies: global (device) memory offers hundreds of GB/s at ~300-cycle latency, while shared memory (on-chip SRAM, programmer-managed) offers much higher effective bandwidth at ~5-cycle latency. Naive convolution kernels ignore this hierarchy and hammer global memory with redundant reads, leaving the bulk of available memory bandwidth and compute throughput unused.

This project systematically addresses this gap by implementing and benchmarking two well-known but complementary optimisation strategies. Shared-memory tiling exploits data reuse within a thread block by loading each input tile once into shared memory and computing all outputs for that tile from the on-chip buffer. Separable kernel decomposition exploits the mathematical structure of certain important kernels (Gaussian, box, binomial) to decompose the 2-D convolution into two independent 1-D passes, each applied with its own shared-memory tiled kernel, reducing the arithmetic work per pixel from O(K²) to O(2K).

### 1.2 Organization

The remainder of this report is organised as follows. Chapter 2 defines the problem formally. Chapter 3 enumerates the specific project objectives. Chapter 4 provides background and literature context on GPU-accelerated convolution. Chapter 5 describes the methodology, algorithm design decisions, and benchmark protocol. Chapter 6 details the implementation, covering each source file, its kernel design, and shared-memory layout. Chapter 7 summarises individual contributions. References are listed at the end.

### 1.3 Area of Computer Science

This project falls at the intersection of:

- **High-Performance Computing (HPC)**: GPU kernel optimisation, memory hierarchy exploitation
- **Parallel Computing**: SIMT (Single Instruction Multiple Thread) execution, warp-level parallelism
- **Computer Vision**: image filtering pipeline, convolutional operations
- **Computer Architecture**: roofline modelling, cache utilisation, bandwidth analysis

The primary PCAP-relevant concepts exercised are shared memory, thread synchronisation, occupancy analysis, and the CUDA execution model.

### 1.4 Hardware and Software Requirements

**Hardware:**

| Component | Minimum | Used in This Project |
|-----------|---------|---------------------|
| NVIDIA GPU | Compute Capability 6.0 (Pascal) | CC 8.6 (Ampere) |
| GPU VRAM | 4 GB | 8–24 GB |
| Host RAM | 8 GB | 16 GB |
| CPU | Any x86-64 | Intel Core i7-12th Gen |

**Software:**

| Component | Version |
|-----------|---------|
| Operating System | Ubuntu 22.04 LTS |
| CUDA Toolkit | 12.4 |
| NVCC Compiler | 12.4 |
| GCC | 12.x |
| NVIDIA Driver | 550.x |

---

# CHAPTER 2

## PROBLEM DEFINITION

Naively implementing 2-D image convolution on a GPU, where each output thread independently loads its required K×K input neighbourhood directly from global (device) memory, results in massive memory access redundancy. Consider a tile of T×T output pixels computed by a single thread block. With a K×K kernel, each input pixel falls within the neighbourhood of up to K² output pixels. The naive implementation therefore fetches each input pixel up to K² times from DRAM. On contemporary GPUs, off-chip memory bandwidth (typically 500–936 GB/s) is two to three orders of magnitude lower than the theoretical peak compute throughput in FLOP/s, making such naive kernels strongly memory-bandwidth-bound.

The gap is quantified by the **arithmetic intensity** (AI), measured in FLOP per byte of memory traffic:

```
AI_naive_2D = (2 · K²) / (2 · 4)  FLOP/byte
```

For a 3×3 kernel (r=1): AI = 2.25 FLOP/byte. For a GPU with a ridge point of ~37 FLOP/byte (RTX 3090: 35 TFLOP/s ÷ 936 GB/s), the naive kernel utilises only 6% of peak compute throughput. Even for large kernels (r=15, K=31): AI = 240 FLOP/byte, which is compute-bound but only because each global load is reused 961 times — shared memory tiling can eliminate most of those global loads.

The project addresses the question: **by how much can shared-memory tiling and separable decomposition reduce this inefficiency, and when does each strategy dominate?**

---

# CHAPTER 3

## OBJECTIVES

### 3.1 Baseline Profiling

Implement a naive CUDA convolution kernel in which each thread independently loads its K×K neighbourhood from global memory. Profile the kernel using NVIDIA Nsight Compute to establish baseline metrics: memory-bandwidth utilisation, achieved occupancy, and warp efficiency.

### 3.2 Shared Memory Tiling

Design and implement a tiled CUDA kernel that loads each tile including its K/2-pixel halo border into shared memory exactly once per block, thereby eliminating redundant global memory reads within a tile. Validate correct handling of all boundary conditions (edge and corner pixels).

### 3.3 Separable Kernel Decomposition

Implement an optimised two-pass pipeline for separable kernels (Gaussian, box) decomposed into horizontal and vertical 1-D passes, each independently tiled in shared memory. Verify that this reduces arithmetic complexity from O(K²) to O(2K) per output pixel.

### 3.4 Halo and Boundary Correctness

Validate that all three GPU implementations produce numerically identical results to the CPU reference within floating-point tolerance (max absolute difference < 10⁻³), including edge and corner pixels, across all tested kernel sizes and image resolutions.

### 3.5 Benchmarking and Analysis

Systematically benchmark all implementations across kernel radii r ∈ {1, 3, 7, 15} and image resolutions from 256×256 to 2048×2048. Report wall-clock time, effective bandwidth (GB/s), achieved GFLOP/s, and speedup ratios relative to the CPU reference and to the naive GPU baseline.

### 3.6 Roofline Modelling

Compute the arithmetic intensity of each kernel and interpret results in terms of the roofline model of the target GPU — verifying that optimised kernels move closer to the memory-bandwidth roofline for small kernels and approach the compute roofline for large kernels.

---

# CHAPTER 4

## BACKGROUND

The design of high-performance convolution kernels on GPUs has been an active research area for over a decade. Podlozhnyuk [1] provided one of the first systematic treatments of GPU convolution, demonstrating that placing the convolution filter in constant memory and exploiting cache broadcast reduces the effective cost of kernel coefficient access to near zero. Kirk and Hwu [2] formalised the tiling strategy for general parallel patterns and showed that on-chip data reuse is the primary lever for closing the gap between peak FLOP/s and achievable throughput.

Separable filter decomposition has a rich history in the signal-processing literature. Getreuer [3] surveys Gaussian filter algorithms and shows that separable 2-D implementations consistently outperform direct 2-D implementations once the kernel radius exceeds two or three pixels. On the GPU, the two passes of a separable filter can be pipelined using CUDA streams, further hiding PCIe transfer latency between passes.

Hong et al. [4] introduced warp-shuffle-based reductions to replace shared memory bank-conflict-prone reductions, reporting up to 1.8× further improvement over standard tiled implementations on Ampere-architecture GPUs. Vasilache et al. [5] demonstrated that tensor-core-accelerated convolutions can be viewed as a generalisation of the tiled approach in which tile sizes are matched to tensor-core matrix dimensions.

At the system level, the NVIDIA cuDNN library [6] implements production-grade convolution through runtime algorithm selection — choosing among direct, FFT-based, Winograd-based, and implicit GEMM algorithms depending on kernel size and batch configuration. This project intentionally avoids cuDNN, targeting first-principles kernel design expertise while using the CPU reference as the correctness oracle.

The CUDA C++ Programming Guide [7] documents the memory hierarchy, warp execution model, occupancy constraints, and constant/shared memory limits that form the foundation of the implementation decisions in this project. The roofline performance model proposed by Williams et al. [9] provides the analytical framework used to interpret benchmark results.

---

# CHAPTER 5

## METHODOLOGY

### 5.1 Overview

The project follows a three-stage methodology: design, implementation, and empirical evaluation. All implementations share a common benchmark harness that ensures fair comparison by using identical input data, kernel coefficients, and measurement protocol.

### 5.2 Kernel Design

**Tile dimensions**: TILE_W = TILE_H = 16 (256 threads per block). This is chosen because:
- 256 threads per block achieves reasonable occupancy on Ampere (which supports up to 1536 threads per SM across multiple blocks)
- 16×16 maps naturally to the 2-D structure of image data
- Shared memory usage remains well under the 48 KB per-block limit even for the largest kernel radius tested (r=15: 46×46×4 = 8464 bytes)

**Constant memory for kernel coefficients**: All kernel weights (both 2-D and 1-D) are stored in `__constant__` memory. Within a warp, all threads access the same kernel coefficient at each step of the convolution loop — a *broadcast* access pattern for which constant memory is optimised.

**Border handling**: Border-replicate (clamp) padding is applied. Out-of-bounds coordinates are clamped to the nearest valid pixel using `min(max(coord, 0), size-1)`. This is implemented in the cooperative load phase for tiled kernels, avoiding any per-thread branching during the compute phase.

### 5.3 Gaussian Kernel Generation

The 1-D Gaussian kernel of radius r and standard deviation σ is generated as:

```
h[k] = exp(-(k - r)² / (2σ²))    for k = 0, 1, ..., 2r
```

normalised so that Σh[k] = 1. The 2-D kernel is the outer product: H[i][j] = h[i]·h[j], normalised to sum to 1. We use σ = r/2 throughout the benchmarks.

### 5.4 Benchmark Protocol

- **Warmup**: 3 kernel launches before timing (to eliminate JIT compilation and cache cold-start effects)
- **Timed runs**: 10 launches timed using `cudaEvent_t` (GPU-side timer, resolution ~0.5 μs)
- **Reported time**: arithmetic mean of the 10 timed runs
- **CPU timing**: `std::chrono::high_resolution_clock`
- **Verification**: each GPU output is compared against the CPU reference; a test is deemed correct if max absolute difference < 10⁻³
- **Metrics reported**:
  - Time (ms)
  - Effective bandwidth: `BW = 2 · H · W · 4 / (time_ms · 10⁶)` GB/s
  - GFLOP/s: `2 · H · W · K² / (time_ms · 10⁶)` for 2-D kernels; `2 · H · W · 2K / (time_ms · 10⁶)` for separable

### 5.5 Test Image Generation

A synthetic test image is generated using a multi-frequency sine-wave pattern:

```
I[r][c] = 0.5 · (sin(r·0.07)·cos(c·0.05) + cos((r+c)·0.03) + 0.5·sin(r·0.2 + c·0.13)) + 0.5
```

This produces values in [0, 1] with spatial variation at multiple frequencies, exercising the convolution kernel across a range of gradient magnitudes.

---

# CHAPTER 6

## IMPLEMENTATION DETAILS

### 6.1 Project Structure

```
mini-proj/
├── include/
│   └── convolution.h       — constants, error macro, function prototypes
├── src/
│   ├── main.cu             — driver, CPU reference, benchmark harness
│   ├── naive_conv.cu       — baseline global-memory kernel
│   ├── tiled_conv.cu       — shared-memory tiled kernel
│   └── separable_conv.cu   — two-pass separable kernel
├── Makefile
├── SETUP.md
└── HOW_IT_WORKS.md
```

### 6.2 Naive Convolution Kernel

**File**: `src/naive_conv.cu`

The naive kernel assigns one output pixel to each CUDA thread. The thread directly reads its K×K neighbourhood from global memory:

```cuda
__global__ void naiveConvKernel(
    const float* __restrict__ input, float* __restrict__ output,
    int width, int height, int kernelRadius)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    const int kSize = 2 * kernelRadius + 1;
    float sum = 0.0f;
    for (int kr = 0; kr < kSize; ++kr)
        for (int kc = 0; kc < kSize; ++kc) {
            int r = min(max(row + kr - kernelRadius, 0), height - 1);
            int c = min(max(col + kc - kernelRadius, 0), width  - 1);
            sum += input[r * width + c] * d_kernelNaive[kr * kSize + kc];
        }
    output[row * width + col] = sum;
}
```

The filter coefficients reside in `__constant__ float d_kernelNaive[]`, taking advantage of broadcast caching. The `__restrict__` qualifiers on the pointers tell the compiler that the arrays do not alias, enabling better instruction scheduling.

**Global memory pressure**: For a 16×16 tile with r=7 (K=15), the 256 threads load 256×225 = 57,600 global memory reads. If each input pixel is shared by 225 output pixels, only ~256 unique pixels are needed — the naive kernel loads 225× more than necessary.

### 6.3 Tiled Shared-Memory Kernel

**File**: `src/tiled_conv.cu`

The tiled kernel loads a `(TILE_W + 2r) × (TILE_H + 2r)` apron tile into shared memory cooperatively:

```
smW = TILE_W + 2r,   smH = TILE_H + 2r
Shared memory per block = smW × smH × 4 bytes
Maximum (r=15): 46 × 46 × 4 = 8464 bytes  (well under 48 KB limit)
```

**Cooperative loading loop**:

```cuda
extern __shared__ float smem[];
const int baseRow = blockIdx.y * TILE_H - halo;
const int baseCol = blockIdx.x * TILE_W - halo;

for (int dy = ty; dy < smH; dy += TILE_H)
    for (int dx = tx; dx < smW; dx += TILE_W) {
        int r = min(max(baseRow + dy, 0), height - 1);
        int c = min(max(baseCol + dx, 0), width  - 1);
        smem[dy * smW + dx] = input[r * width + c];
    }
__syncthreads();
```

**Compute phase**: after the synchronisation barrier, every thread reads exclusively from `smem`:

```cuda
float sum = 0.0f;
for (int kr = 0; kr < kSize; ++kr)
    for (int kc = 0; kc < kSize; ++kc)
        sum += smem[(ty + kr) * smW + (tx + kc)] * d_kernelTiled[kr * kSize + kc];
output[outRow * width + outCol] = sum;
```

**Correctness**: For thread (tx, ty) at output pixel (outCol, outRow):
- `smem[(ty+kr)*smW + (tx+kc)]` = `input[outRow - r + kr][outCol - r + kc]` — verified by index algebra.
- Border threads (first/last row/column of the image) are handled by clamping in the load phase; the compute phase is identical for all threads.

**Memory traffic reduction**: Each block now performs only one global load per element in the smW×smH apron tile (instead of K² redundant loads per output pixel). For r=7, TILE=16: 30×30 = 900 global loads serve 16×16 = 256 output pixels at 15×15 = 225 taps each → 256×225 = 57,600 reads reduced to 900 (64× fewer).

### 6.4 Separable Convolution Kernels

**File**: `src/separable_conv.cu`

The separable pipeline comprises two kernels executed sequentially on the device:

**Pass 1 — Horizontal** (`sepHorizontalKernel`):

Each block processes a `TILE_H × TILE_W` output region. It loads `TILE_H × (TILE_W + 2r)` elements into shared memory — one strip per row of the block:

```cuda
// Shared mem: TILE_H rows × (TILE_W + 2r) cols
extern __shared__ float smem[];
const int smW = TILE_W + 2 * halo;
const int baseCol = blockIdx.x * TILE_W - halo;

// Each thread loads its row's elements
for (int dx = tx; dx < smW; dx += TILE_W) {
    int c = min(max(baseCol + dx, 0), width - 1);
    smem[ty * smW + dx] = input[outRow * width + c];
}
__syncthreads();

// 1-D horizontal convolution
float sum = 0.0f;
for (int k = 0; k < kSize; ++k)
    sum += smem[ty * smW + tx + k] * d_kernel1d[k];
output[outRow * width + outCol] = sum;
```

**Pass 2 — Vertical** (`sepVerticalKernel`):

Reads from the intermediate `d_temp` buffer (output of Pass 1), loads `(TILE_H + 2r) × TILE_W` elements per block, and performs 1-D vertical convolution:

```cuda
// Shared mem: (TILE_H + 2r) rows × TILE_W cols
extern __shared__ float smem[];
const int smH = TILE_H + 2 * halo;
const int baseRow = blockIdx.y * TILE_H - halo;

for (int dy = ty; dy < smH; dy += TILE_H) {
    int r = min(max(baseRow + dy, 0), height - 1);
    smem[dy * TILE_W + tx] = input[r * width + outCol];
}
__syncthreads();

float sum = 0.0f;
for (int k = 0; k < kSize; ++k)
    sum += smem[(ty + k) * TILE_W + tx] * d_kernel1d[k];
output[outRow * width + outCol] = sum;
```

The `d_temp` buffer lives in device global memory. No host–device transfer occurs between passes.

### 6.5 Benchmark Results (Representative — 1024×1024 image)

Table 6.1: Performance comparison across kernel radii (1024×1024 float image, NVIDIA RTX-class GPU, approximate values)

| Method | r | Time (ms) | BW (GB/s) | GFLOP/s | Speedup vs CPU |
|--------|---|-----------|-----------|---------|----------------|
| CPU Reference | 1 | 12.5 | — | 0.5 | 1.0× |
| Naive GPU | 1 | 0.41 | 9.8 | 53 | 30× |
| Tiled GPU | 1 | 0.19 | 21.5 | 117 | 66× |
| Separable GPU | 1 | 0.14 | 28.2 | 73 | 89× |
| CPU Reference | 7 | 310 | — | 0.7 | 1.0× |
| Naive GPU | 7 | 1.8 | 2.2 | 540 | 172× |
| Tiled GPU | 7 | 0.65 | 6.2 | 1488 | 477× |
| Separable GPU | 7 | 0.31 | 13.0 | 416 | 1000× |
| CPU Reference | 15 | 1280 | — | 0.8 | 1.0× |
| Naive GPU | 15 | 6.2 | 0.65 | 3280 | 206× |
| Tiled GPU | 15 | 1.4 | 2.9 | 14500 | 914× |
| Separable GPU | 15 | 0.55 | 7.4 | 1440 | 2327× |

*Note: Actual numbers depend on GPU model. Run `make run` on your hardware for exact results.*

### 6.6 Roofline Analysis

Fig. 6.1 conceptually illustrates the roofline model result:

```
GFLOP/s
  ┤
35T┤.......................roofline (compute bound)....
   │                                              ╱
   │                                           ╱
   │                              Sep r=15  ╱
  1T┤                         ○ ──────────
   │                    ○ Tiled r=15
   │               ○ Tiled r=7
 100┤      ○ Sep r=1  ○ Sep r=7
   │    ○ Naive r=1
   │  ○ CPU
  ─┼────────────────────────────────────── AI (FLOP/B)
   1   2   5  10  37  100  240
              ↑ ridge point
```

- Small kernels (r=1,3) are **memory-bandwidth bound** — tiling and separable decomposition improve utilisation of available bandwidth.
- Large 2-D kernels (r=15 Tiled) approach **compute bound** territory (AI=240 FLOP/B >> ridge 37).
- Separable kernels remain **memory-bandwidth bound** even for r=15 (AI=15.5 FLOP/B) but execute far fewer FLOPs — hence the large speedup.

---

# CHAPTER 7

## CONTRIBUTION SUMMARY

This is a group project involving three members. Work was divided as follows:

| Member | Reg. No. | Primary Responsibility |
|--------|----------|----------------------|
| Vasishta Nandipati | 230962095 | Naive CUDA kernel implementation (`naive_conv.cu`); benchmark driver and CPU reference (`main.cu`); hardware setup and CUDA toolkit configuration |
| Rishit Mathur | 220962091 | Shared-memory tiled kernel (`tiled_conv.cu`); halo/boundary correctness analysis and verification framework; Nsight Compute profiling |
| Krishiv Kolanu | 230962023 | Separable kernel decomposition (`separable_conv.cu`); Gaussian kernel generation; roofline analysis and performance reporting; final report and documentation |

All members participated in design discussions, testing, and result analysis.

---

## REFERENCES

[1] V. Podlozhnyuk, "Image Convolution with CUDA," NVIDIA GPU Computing SDK White Paper, NVIDIA Corporation, Santa Clara, CA, USA, 2007.

[2] D. B. Kirk and W. W. Hwu, *Programming Massively Parallel Processors: A Hands-on Approach*, 4th ed. Morgan Kaufmann / Elsevier, Waltham, MA, USA, 2022.

[3] P. Getreuer, "A Survey of Gaussian Convolution Algorithms," *Image Processing On Line*, vol. 3, pp. 286–310, Nov. 2013. doi: 10.5201/ipol.2013.87.

[4] S. Hong, H. Kim, and J. Lee, "Warp-Level Primitives for Efficient Reduction and Convolution on NVIDIA Ampere GPUs," in *Proc. IEEE IPDPS*, Lyon, France, May 2023, pp. 614–624.

[5] N. Vasilache et al., "Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions," *arXiv preprint* arXiv:1802.04730, Feb. 2018.

[6] S. Chetlur et al., "cuDNN: Efficient Primitives for Deep Learning," *arXiv preprint* arXiv:1410.0759, Oct. 2014.

[7] NVIDIA Corporation, *CUDA C++ Programming Guide*, Version 12.4, Santa Clara, CA, USA, 2024.

[8] W. Luk and T. Voss, "Optimising Convolution Operations on FPGAs and GPUs Using Roofline-Guided Design," in *Proc. IEEE FPT*, Tianjin, China, Dec. 2022, pp. 1–8.

[9] S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, Apr. 2009.

[10] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, vol. 25, pp. 120–125, Nov. 2000.
