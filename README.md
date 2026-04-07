# CUDA Image Convolution Optimizer

Benchmarking shared-memory tiling and separable kernel decomposition against a naive GPU implementation for 2-D image convolution — tested on an NVIDIA RTX 4070 Laptop GPU.

---

## Overview

Image convolution is at the core of Gaussian blur, edge detection, and CNN layers. A naive CUDA kernel reads the same input pixel from global memory up to K² times per output pixel, making it heavily memory-bandwidth bound. This project implements and benchmarks two optimisations:

- **Shared-memory tiling** — loads each tile (including a halo border) into on-chip SRAM once per block, eliminating redundant global memory reads
- **Separable kernel decomposition** — splits a 2-D convolution into two 1-D passes (horizontal then vertical), reducing per-pixel arithmetic from O(K²) to O(2K)

---

## Results (RTX 4070 Laptop GPU, 1024×1024 image)

| Method | r | Time (ms) | GB/s | GFLOP/s | Speedup vs CPU |
|---|---|---|---|---|---|
| CPU Reference | 1 | 10.706 | — | 1.76 | 1.00× |
| Naive GPU | 1 | 0.025 | 341.3 | 768.0 | 435× |
| Tiled GPU | 1 | 0.035 | 237.4 | 534.3 | 303× |
| Separable GPU | 1 | 0.042 | 198.8 | 298.3 | 253× |
| CPU Reference | 7 | 184.220 | — | 2.56 | 1.00× |
| Naive GPU | 7 | 0.367 | 22.9 | 1285.4 | 501× |
| Tiled GPU | 7 | 0.228 | 36.8 | 2068.2 | 807× |
| Separable GPU | 7 | 0.063 | 133.6 | 1002.3 | **2934×** |
| CPU Reference | 15 | 779.465 | — | 2.59 | 1.00× |
| Naive GPU | 15 | 1.495 | 5.6 | 1348.1 | 521× |
| Tiled GPU | 15 | 0.927 | 9.1 | 2174.3 | 840× |
| Separable GPU | 15 | 0.092 | 90.9 | 1409.3 | **8448×** |

> All GPU results verified correct (max absolute error < 10⁻³ vs CPU reference).

---

## Sample Output

```
CUDA Image Convolution Benchmark
Shared Memory Tiling and Separable Kernels

GPU : NVIDIA GeForce RTX 4070 Laptop GPU
SMs : 36   Peak BW: 256.0 GB/s   Shared mem: 48 KB

--- 1024 x 1024 (1048576 pixels) ---
Method            r     Err        ms      GB/s     GFLOP/s    Spdup
CPU Reference      1       -    10.706    0.0000        1.76     1.00x
Naive GPU          1  3.6e-07     0.025   341.333      768.00   435.62x  OK
Tiled GPU          1  3.6e-07     0.035   237.449      534.26   303.04x  OK
Separable GPU      1  3.6e-07     0.042   198.835      298.25   253.76x  OK
  AI(2D)=2.2  AI(Sep)=1.5  Tiled 0.70x over Naive

CPU Reference     15       -   779.465    0.0000        2.59     1.00x
Naive GPU         15  3.6e-07     1.495     5.611     1348.13   521.40x  OK
Tiled GPU         15  3.6e-07     0.927     9.050     2174.25   840.91x  OK
Separable GPU     15  3.2e-06     0.092    90.921     1409.28  8448.35x  OK
  AI(2D)=240.2  AI(Sep)=15.5  Tiled 1.61x over Naive
```

---

## Project Structure

```
cuda-conv-optimizer/
├── include/
│   └── convolution.h       # constants, error macro, function prototypes
├── src/
│   ├── main.cu             # benchmark driver, CPU reference, timing
│   ├── naive_conv.cu       # baseline: each thread reads from global memory
│   ├── tiled_conv.cu       # optimisation 1: shared-memory tiling
│   └── separable_conv.cu   # optimisation 2: two-pass separable pipeline
├── report/
│   ├── Final_Report.docx   # full project report
│   └── Final_Report.pdf
└── Makefile
```

---

## Requirements

- NVIDIA GPU (Compute Capability ≥ 6.0)
- CUDA Toolkit 11.6+
- GCC / G++ 9+
- GNU Make

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

---

## Build

```bash
# Auto-detect GPU architecture (CUDA >= 11.6)
make

# Or specify manually
make ARCH=sm_89   # RTX 40xx (Ada Lovelace)
make ARCH=sm_86   # RTX 30xx (Ampere)
make ARCH=sm_75   # RTX 20xx (Turing)
make ARCH=sm_70   # V100 (Volta)
```

---

## Run

```bash
# Full benchmark — 4 resolutions × 4 kernel radii
make run

# Quick 512×512 test
make quick

# Custom resolution
./conv_bench 1920 1080
```

---

## Key Findings

- For **small kernels (r=1)**, the naive GPU is faster than tiled — the shared-memory load overhead outweighs the benefit at only 9 unique data reuses per pixel
- For **r ≥ 3**, tiled consistently beats naive (1.3–1.7×)
- The **separable kernel dominates** for large radii — at r=15, it is ~10× faster than tiled and 8448× faster than the CPU on a 1024×1024 image
- The tiled 2-D kernel at r=15 is **compute-bound** (AI = 240 FLOP/byte > ridge point), while the separable kernel remains **bandwidth-bound** (AI = 15.5 FLOP/byte) but performs far fewer operations

---

## Tech Stack

- CUDA C++ (C++17)
- NVIDIA constant memory, shared memory, `cudaEvent_t` timing
- GNU Make

---

## Authors

Vasishta Nandipati · Rishit Mathur · Krishiv Kolanu  
*School of Computer Engineering, MIT Manipal*  
*Guide: Vidya Kamath, Assistant Professor*
