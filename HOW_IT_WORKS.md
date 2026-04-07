# How It Works — CUDA Image Convolution Benchmark

## 1. What Is Image Convolution?

Given a grayscale image **I** of size H×W and a filter kernel **K** of size k×k (where k = 2r+1 for radius r), the 2-D discrete convolution is:

```
O[row][col] = ΣΣ  I[row+dr][col+dc] · K[r+dr][r+dc]
              dr=-r  dc=-r
```

This is used for:
- **Gaussian blur** — noise reduction (separable kernel)
- **Sobel edge detection** — gradient computation
- **Sharpening / unsharp mask** — detail enhancement
- **CNN layers** — feature extraction in deep learning

**Cost**: O(H · W · K²) multiplications + additions. For a 31×31 kernel on a 4K image (3840×2160): **~7.9 billion FLOPs**.

---

## 2. The CUDA Memory Hierarchy

Understanding this is **key** to why the optimisations work.

```
Latency (cycles)   Memory tier
──────────────     ──────────────────────────────────
        1          Registers (per-thread, ~32KB per SM)
        5          Shared Memory / L1 cache (per-SM, 48–164 KB)
      ~30          L2 cache (device-wide, ~6 MB)
     ~300          Global Memory / DRAM (device-wide, GB range)
```

Global memory bandwidth on a modern GPU (e.g. RTX 3090) is ~936 GB/s. At first glance that sounds fast — but compute throughput is ~35 TFLOP/s, meaning every byte must feed **37 FLOPs** to keep the ALUs busy. Convolution with small kernels has low arithmetic intensity (few FLOPs per byte), making it **memory-bandwidth bound**.

---

## 3. The Three Implementations

### 3.1 Naive GPU Convolution (`naive_conv.cu`)

```
Thread (row, col)  →  loads K×K pixels from global memory  →  writes 1 output
```

**Every thread independently loads its own K×K neighbourhood.** For a 16×16 output tile with a 7×7 kernel, the same input pixel may be loaded by up to 49 different threads — all from slow global memory.

```
Global memory reads ≈ H · W · K²   (massive redundancy)
```

The kernel weights are stored in **constant memory** (64 KB on-chip, broadcast-cached) so that the kernel access is free relative to the input data access.

```cuda
// Each thread does this:
for (kr = -r to r)
  for (kc = -r to r)
    sum += globalInput[row+kr][col+kc] * constKernel[kr][kc]
```

---

### 3.2 Tiled Shared-Memory Convolution (`tiled_conv.cu`)

**Key idea**: threads in the same block all need overlapping regions of the input. Load that region into shared memory *once*, then everyone reads from there.

```
                     ┌──────────────────────────────┐
  Global Memory      │  (TILE_W + 2r) × (TILE_H + 2r) │  → Load ONCE into shared
  (slow, far)        │       = "apron tile"           │
                     └──────────────────────────────┘
                              ↓ __syncthreads()
  Shared Memory       ┌────────────────────┐
  (fast, on-chip)     │   TILE_W × TILE_H  │  → All threads read from here
                      └────────────────────┘
```

**Halo / apron**: The tile must be wider than TILE_W by 2r on each side to cover the K×K neighbourhood of every border thread in the block.

```cuda
// Cooperative load (each thread may load >1 element)
for dy in [ty, smH) step TILE_H:
  for dx in [tx, smW) step TILE_W:
    smem[dy*smW + dx] = clampedGlobalInput[baseRow+dy][baseCol+dx]
__syncthreads()

// Compute from shared memory (fast)
sum = Σ smem[(ty+kr)*smW + (tx+kc)] * kernel[kr][kc]
```

**Memory reduction**: Instead of K² global loads per output pixel, each pixel costs approximately (1 + 2r/TILE)² global loads → up to **K²/TILE²** times less traffic.

---

### 3.3 Separable Kernel Decomposition (`separable_conv.cu`)

Many important filters (Gaussian, box, binomial) are **separable**: the 2-D kernel equals the outer product of two 1-D vectors.

```
H(i,j) = h(i) · h(j)    ←  Gaussian is separable
```

This means the 2-D convolution can be replaced by two 1-D passes:

```
Pass 1 — Horizontal:
    temp[r][c] = Σ input[r][c - k + K/2] · h[k]    (O(K) per pixel)

Pass 2 — Vertical:
    out[r][c]  = Σ temp[r - k + K/2][c] · h[k]    (O(K) per pixel)
```

**Arithmetic savings**: O(K²) → O(2K) per pixel. For K=31: **~15× fewer multiplications**.

Each pass is independently tiled in shared memory:

| Pass | Shared memory layout | Purpose |
|------|---------------------|---------|
| Horizontal | TILE_H rows × (TILE_W + 2r) cols | Each row loads one row strip including halos |
| Vertical | (TILE_H + 2r) rows × TILE_W cols | Each column loads one column strip including halos |

```
Input → [Horizontal pass] → temp[] → [Vertical pass] → Output
               ↑ shared mem                ↑ shared mem
```

---

## 4. Boundary Handling

When the kernel window extends beyond the image border, we use **border-replicate / clamp** padding: pixels outside are treated as equal to the nearest edge pixel.

```
Image edge          Clamped region (radius = 2)
────────────────    ────────────────────────────
 A B C D E ...      A A A B C D E ...
```

In code: `r = min(max(r, 0), height-1)` — a single instruction on the GPU.

---

## 5. Performance Model (Roofline)

The **roofline model** predicts whether a kernel is memory-bound or compute-bound.

```
Ridge point = Peak FLOP/s ÷ Peak BW
            ≈ 35 TFLOP/s ÷ 0.936 TB/s ≈ 37 FLOP/byte   (RTX 3090)
```

| Kernel | Radius | Arithmetic Intensity | Bottleneck |
|--------|--------|---------------------|-----------|
| Naive 2-D | r=1 | 2×3²/(2×4) = 2.25 FLOP/B | Memory BW |
| Naive 2-D | r=15 | 2×31²/(2×4) = 240 FLOP/B | Compute |
| Separable | r=1 | 2×2×3/(2×4) = 1.5 FLOP/B | Memory BW |
| Separable | r=15 | 2×2×31/(2×4) = 15.5 FLOP/B | Memory BW |

**Interpretation**: small kernels are always memory-bandwidth limited; large 2-D kernels become compute-limited while separable kernels remain bandwidth-limited even for large r (this is why they're faster — they traffic less data per output pixel).

---

## 6. Expected Speedups

| Configuration | Speedup vs CPU | Speedup vs Naive |
|---------------|----------------|-----------------|
| Naive GPU (r=3, 1024²) | 30–50× | 1× (baseline) |
| Tiled GPU (r=3, 1024²) | 60–120× | ~2–3× |
| Separable GPU (r=3, 1024²) | 80–200× | ~2–4× |
| Separable GPU (r=15, 1024²) | 300–800× | ~8–12× |

Speedups grow with kernel radius because:
1. Larger kernels have more redundant global loads to eliminate (benefits tiling more)
2. Separable decomposition saves more FLOPs (K² vs 2K is a bigger difference for large K)

---

## 7. Code Flow (step-by-step)

```
main()
  ├── printGpuInfo()                    query cudaDeviceProp
  ├── for each resolution (256² → 2048²):
  │     ├── generateTestImage()         sine-wave pattern, values in [0,1]
  │     ├── cudaMalloc d_in, d_out, d_temp
  │     ├── cudaMemcpy H→D  (input image)
  │     └── for each kernel radius {1, 3, 7, 15}:
  │           ├── generateGaussianKernel()  produce kernel2d + kernel1d
  │           ├── cpuConvolution()          reference output + CPU time
  │           ├── gpuTimeKernel(Naive)      warmup + 10-run avg → naiveMs
  │           ├── gpuTimeKernel(Tiled)      warmup + 10-run avg → tiledMs
  │           ├── gpuTimeKernel(Separable)  warmup + 10-run avg → sepMs
  │           ├── maxAbsDiff() for each     verify correctness
  │           └── print table row          ms, GB/s, GFLOP/s, speedup
  └── cudaFree everything
```

---

## 8. Files Quick Reference

| File | Role |
|------|------|
| `include/convolution.h` | Constants, error macro, function prototypes |
| `src/naive_conv.cu` | Global-memory baseline kernel |
| `src/tiled_conv.cu` | Shared-memory tiled kernel |
| `src/separable_conv.cu` | Two-pass separable kernel |
| `src/main.cu` | Driver, CPU reference, benchmarking, output |
| `Makefile` | Build & run targets |
| `SETUP.md` | Installation & usage guide |
| `HOW_IT_WORKS.md` | This document |
