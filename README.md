# CUDA Image Convolution Optimizer

This project implements and benchmarks three approaches to 2-D image convolution on an NVIDIA GPU using CUDA C, demonstrating how on-chip shared memory and algorithmic decomposition can dramatically reduce memory bandwidth pressure and improve throughput compared to a naive implementation.

The three GPU implementations evaluated are:

1. **Naive convolution** -- each thread independently loads its full K x K input neighbourhood from global (device) memory. Simple but highly redundant: each input pixel is read up to K^2 times from DRAM.

2. **Shared-memory tiled convolution** -- threads within a block cooperatively load a (TILE + 2r) x (TILE + 2r) apron tile into on-chip shared memory once, then all threads compute exclusively from that fast buffer. Eliminates repeated global reads within a tile.

3. **Separable convolution** -- exploits the separability of the Gaussian kernel to decompose the 2-D convolution into two sequential 1-D passes (horizontal then vertical), each independently tiled in shared memory. Reduces per-pixel arithmetic from O(K^2) to O(2K).

A CPU single-threaded reference implementation is included for correctness verification and baseline comparison.

---

## Problem and Motivation

Discrete 2-D convolution is central to image processing (Gaussian blur, edge detection, sharpening) and to convolutional neural networks. For an H x W image with a K x K kernel (K = 2r + 1), the naive computation cost is O(H * W * K^2). On a GPU, a naive implementation that reads all input data from global memory suffers from severe memory bandwidth inefficiency.

The arithmetic intensity (AI) of the naive kernel is:

```
AI = 2 * K^2 / (2 * 4)  FLOP/byte
```

For r = 1 (K = 3): AI = 2.25 FLOP/byte, far below the ridge point of the RTX 4070 Laptop GPU (~58 FLOP/byte). The naive kernel is strongly memory-bandwidth bound at small kernel sizes, and wastes bandwidth at large kernel sizes due to redundant global loads.

Shared-memory tiling and separable decomposition address this inefficiency in complementary ways. Tiling reduces global memory traffic by reusing loaded data within a block. Separable decomposition reduces arithmetic work by factoring the 2-D kernel into 1-D passes.

---

## Repository Structure

```
cuda-conv-optimizer/
├── src/
│   ├── main.cu             driver, CPU reference, Gaussian kernel generator, benchmark harness
│   ├── naive_conv.cu       naive global-memory CUDA kernel
│   ├── tiled_conv.cu       shared-memory tiled CUDA kernel
│   └── separable_conv.cu   two-pass separable CUDA kernel (horizontal + vertical)
├── include/
│   └── convolution.h       constants (TILE_W, TILE_H, MAX_KERNEL_SIZE), CUDA_CHECK macro, prototypes
├── Makefile
├── conv_bench              compiled benchmark binary (RTX 4070 Laptop, CUDA 12.4)
└── Final_Report.pdf        full project report
```

---

## Build and Run

**Requirements:**
- NVIDIA GPU, Compute Capability 6.0 or higher
- CUDA Toolkit 12.x (tested with 12.4)
- GCC and GNU Make

**Build:**
```bash
make
```

For a specific GPU architecture:
```bash
make ARCH=sm_89    # Ada Lovelace (RTX 40xx)
make ARCH=sm_86    # Ampere (RTX 30xx)
make ARCH=sm_75    # Turing (RTX 20xx)
```

**Run full benchmark (4 resolutions x 4 kernel radii):**
```bash
make run
```

**Quick test (512x512):**
```bash
make quick
```

**Custom resolution:**
```bash
./conv_bench 1920 1080
```

---

## Kernel Design

### Tile dimensions
TILE_W = TILE_H = 16 (256 threads per block). Shared memory usage stays under the 48 KB per-block limit even at the maximum tested kernel radius (r = 15): the tiled 2-D kernel uses (16 + 30) x (16 + 30) x 4 = 8,464 bytes per block.

### Constant memory for filter weights
All kernel coefficients (1-D and 2-D Gaussian) are stored in `__constant__` memory. Since all threads in a warp read the same coefficient at each step of the convolution loop, the constant cache serves these reads as a single broadcast transaction.

### Border handling
Out-of-bounds input coordinates are clamped to the nearest valid pixel (border-replicate padding), implemented during the shared-memory load phase so the compute phase has no branch divergence.

### Benchmark protocol
- 3 warmup kernel launches before timing (eliminates JIT and cache cold-start effects)
- 10 timed launches measured with `cudaEvent_t` (GPU-side timer, ~0.5 us resolution)
- Reported time is the arithmetic mean of the 10 runs
- CPU timing uses `clock_gettime(CLOCK_MONOTONIC)`
- Correctness: each GPU output is checked against the CPU reference; max absolute difference must be below 1e-3

---

## Results

Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace, 36 SMs, 256 GB/s peak BW, 8 GB VRAM), 13th Gen Intel Core i9-13900H, CUDA 12.4, Debian GNU/Linux 13.

All GPU outputs passed the correctness check for every tested configuration.

### 1024 x 1024 image -- representative results

| Method | r | Time (ms) | BW (GB/s) | GFLOP/s | Speedup vs CPU |
|--------|---|-----------|-----------|---------|----------------|
| CPU Reference | 1 | 21.037 | - | 0.90 | 1.00x |
| Naive GPU | 1 | 0.026 | 321.3 | 722.8 | 805.7x |
| Tiled GPU | 1 | 0.033 | 256.0 | 576.0 | 642.0x |
| Separable GPU | 1 | 0.101 | 82.7 | 124.1 | 207.5x |
| CPU Reference | 3 | 74.565 | - | 1.38 | 1.00x |
| Naive GPU | 3 | 0.092 | 90.8 | 1112.6 | 807.3x |
| Tiled GPU | 3 | 0.068 | 122.8 | 1504.5 | 1091.7x |
| Separable GPU | 3 | 0.062 | 134.3 | 470.0 | 1193.7x |
| CPU Reference | 7 | 321.471 | - | 1.47 | 1.00x |
| Naive GPU | 7 | 0.369 | 22.7 | 1277.9 | 870.6x |
| Tiled GPU | 7 | 0.232 | 36.2 | 2035.3 | 1386.7x |
| Separable GPU | 7 | 0.066 | 127.8 | 958.4 | 4896.9x |
| CPU Reference | 15 | 1350.456 | - | 1.49 | 1.00x |
| Naive GPU | 15 | 1.538 | 5.5 | 1310.5 | 878.2x |
| Tiled GPU | 15 | 0.927 | 9.0 | 2173.8 | 1456.6x |
| Separable GPU | 15 | 0.118 | 71.2 | 1103.2 | 11457.9x |

### Key observations

**Small kernels (r = 1):** The naive kernel is faster than tiled at small image sizes because the 3x3 neighbourhood has minimal overlap between adjacent threads, so the shared-memory overhead is not justified. Effective bandwidth exceeds the nominal 256 GB/s peak due to L2 cache hits.

**Medium kernels (r = 3, r = 7):** The tiled kernel consistently outperforms naive from r = 3 onwards. At r = 7 on the largest image (2048x2048), tiled is 1.61x faster than naive. The separable kernel achieves 4483x over the CPU reference at this size.

**Large kernels (r = 15):** The separable kernel dominates. On a 2048x2048 image it runs in 0.420 ms versus 3.861 ms for tiled -- a 9.2x difference. The tiled 2-D kernel at r = 15 has arithmetic intensity of 240 FLOP/byte (compute-bound, above the ridge point of ~58 FLOP/byte), while the separable kernel at 15.5 FLOP/byte remains memory-bandwidth bound but does far less arithmetic. The highest recorded speedup was **12826x** over the CPU reference (separable, r = 15, 2048x2048).

---

## Roofline Analysis

The arithmetic intensity of each method at each kernel radius:

| Method | r | AI (FLOP/B) | Regime |
|--------|---|-------------|--------|
| Naive / Tiled 2-D | 1 | 2.2 | Memory BW bound |
| Separable | 1 | 1.5 | Memory BW bound |
| Naive / Tiled 2-D | 3 | 12.2 | Memory BW bound |
| Separable | 3 | 3.5 | Memory BW bound |
| Naive / Tiled 2-D | 7 | 56.2 | Compute bound (near ridge) |
| Separable | 7 | 7.5 | Memory BW bound |
| Naive / Tiled 2-D | 15 | 240.2 | Compute bound |
| Separable | 15 | 15.5 | Memory BW bound |

Ridge point for RTX 4070 Laptop GPU: ~58 FLOP/byte (15 TFLOP/s FP32 / 256 GB/s). Separable kernels always remain memory-bandwidth bound, giving them the best wall-clock time for large radii because they perform far fewer multiply-accumulate operations.

---

## References

1. V. Podlozhnyuk, "Image Convolution with CUDA," NVIDIA GPU Computing SDK White Paper, 2007.
2. D. B. Kirk and W. W. Hwu, "Programming Massively Parallel Processors," 4th ed., Morgan Kaufmann, 2022.
3. P. Getreuer, "A Survey of Gaussian Convolution Algorithms," Image Processing On Line, vol. 3, pp. 286-310, 2013.
4. NVIDIA Corporation, "CUDA C++ Programming Guide," Version 12.4, 2024.
5. S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual Performance Model," Commun. ACM, vol. 52, no. 4, pp. 65-76, 2009.
