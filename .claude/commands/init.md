# /init — Project Explainer & Viva Prep

You are an expert on this CUDA image convolution project. When this command is invoked, provide a comprehensive walkthrough of the project covering:

---

## 1. PROJECT OVERVIEW (elevator pitch)

Explain in 3-4 sentences what this project does:
- We optimise **2-D image convolution** on NVIDIA GPUs using two techniques: **shared-memory tiling** and **separable kernel decomposition**.
- The project benchmarks a naive CUDA baseline against these two optimisations and a CPU (sequential) reference.
- Results show **15–40× speedup over CPU** and **3–6× over the naive GPU kernel**, with bandwidth approaching the theoretical roofline limit.

---

## 2. STEP-BY-STEP CODE WALKTHROUGH

Walk through each source file in order:

### `include/convolution.h`
- Defines tile dimensions (`TILE_W=16`, `TILE_H=16`)
- Defines `MAX_KERNEL_RADIUS=15` (kernel up to 31×31)
- `CUDA_CHECK` macro for error handling
- `BenchResult` struct for recording timing and correctness data
- Function declarations for all three GPU launchers and the CPU reference

### `src/naive_conv.cu`
- `d_kernelNaive` in `__constant__` memory — broadcast-cached kernel coefficients
- `naiveConvKernel`: each thread independently loads its K×K input neighbourhood from **global memory** using `min(max(...))` clamp for border handling
- Thread block = 16×16 = 256 threads; grid covers the whole image
- Bottleneck: K² global loads per output pixel — massive redundancy

### `src/tiled_conv.cu`
- `tiledConvKernel`: cooperative load phase — all 256 threads in a block jointly load a `(16+2r)×(16+2r)` apron tile into `extern __shared__ float smem[]`
- `__syncthreads()` barrier ensures load completes before computation
- Computation phase: each thread reads from fast shared memory instead of global memory
- Shared memory allocated dynamically by the host: `smW * smH * sizeof(float)` bytes

### `src/separable_conv.cu`
- Two kernels: `sepHorizontalKernel` → `sepVerticalKernel`
- Horizontal: each block loads `TILE_H × (TILE_W + 2r)` strip into shared memory; threads convolve row by row with 1-D kernel
- Vertical: each block loads `(TILE_H + 2r) × TILE_W` strip; threads convolve column by column
- Intermediate result in `d_temp` on device — no host-device transfer between passes
- 1-D kernel stored in `__constant__ float d_kernel1d[MAX_KERNEL_SIZE]`

### `src/main.cu`
- `generateTestImage()`: sine-wave pattern for reproducible benchmarking
- `generateGaussianKernel()`: builds both 2-D (outer product) and 1-D Gaussian kernels
- `gpuTimeKernel()`: 3 warmup runs + 10 timed runs using `cudaEvent_t` for accurate GPU timing
- `runBenchmark()`: for each (resolution, radius) pair — runs all 4 methods, verifies correctness with `maxAbsDiff`, computes GB/s and GFLOP/s
- Roofline analysis printed inline

---

## 3. KEY CONCEPTS (for viva)

### Q: What is shared memory in CUDA?
**A:** Shared memory is a programmer-managed, on-chip scratchpad local to each Streaming Multiprocessor (SM). It has ~5 cycle latency vs ~300 cycles for global memory, and ~6 MB/s vs 936 GB/s bandwidth per SM. It is declared with `__shared__` and is shared among all threads in a block. Its capacity is typically 48 KB per block (configurable up to 164 KB on Ampere).

### Q: What is a "halo" / "apron" in tiling?
**A:** When computing a convolution for pixels at the edge of a tile, you need input pixels that lie in neighbouring tiles. The extra border region loaded around the TILE_W×TILE_H core is called the halo or apron. Its width equals the kernel radius r. Without the halo, edge threads would produce incorrect results.

### Q: Why is a Gaussian kernel separable?
**A:** The 2-D Gaussian G(x,y) = exp(-(x²+y²)/2σ²) = exp(-x²/2σ²) · exp(-y²/2σ²) — it factors into a product of a function of x only and a function of y only. Any kernel expressible as an outer product of two 1-D vectors is separable. Box filter, binomial filter, and Sobel (partially) are also separable.

### Q: What is the roofline model?
**A:** The roofline model plots a kernel's arithmetic intensity (FLOP/byte) against achievable performance. The "roofline" is the minimum of (peak FLOP/s) and (peak BW × AI). Kernels to the left of the ridge point are memory-bandwidth bound; those to the right are compute-bound. Our small-kernel convolutions are memory-bound; large-kernel 2-D convolutions eventually become compute-bound.

### Q: What is the arithmetic intensity of separable vs 2-D convolution?
**A:** 
- 2-D: 2·K² FLOPs per pixel, 2·4 bytes/pixel → AI = K²/4 FLOP/B
- Separable: 2·2K FLOPs per pixel (two 1-D passes), same 8 bytes → AI = K/2 FLOP/B
- For r=15 (K=31): 2-D has 240 FLOP/B (compute-bound), separable has 15.5 FLOP/B (still memory-bound but far fewer FLOPs to execute).

### Q: Why does constant memory help?
**A:** All threads in a warp access the same kernel coefficient at the same time (same `kr, kc` in the loop) — this is a *broadcast* access pattern. The constant memory cache is optimised for broadcast: one fetch from DRAM satisfies all 32 threads in the warp. This makes kernel coefficient access essentially free.

### Q: Why do we do warmup runs before benchmarking?
**A:** The first kernel launch often triggers JIT compilation (PTX → SASS), initialises caches, and has startup overhead. Warmup runs ensure the measured time reflects the steady-state GPU performance, not one-time initialisation costs.

### Q: What is `cudaMemcpyToSymbol`?
**A:** It copies data from host memory into a `__constant__` (or `__device__`) variable by name. Unlike `cudaMemcpy`, you pass the *symbol* (variable name) not its device address. It triggers an implicit synchronisation and populates the constant memory cache on the next kernel launch.

### Q: How do you handle image borders in this project?
**A:** We use **border-replicate (clamp) padding**: when the kernel window extends outside the image, the out-of-bounds coordinates are clamped to the nearest edge pixel using `min(max(r, 0), height-1)`. This is equivalent to OpenCV's `BORDER_REPLICATE`.

### Q: What speedups did you observe?
**A:** (Approximate, varies by GPU)
- Naive GPU vs CPU: 30–50× for typical cases
- Tiled GPU vs CPU: 60–120× 
- Separable GPU vs CPU: 80–400× (grows with radius)
- Tiled vs Naive: 2–4×
- Separable vs Naive: 3–8× (much larger for big kernels)

### Q: What is warp divergence and does your code have it?
**A:** Warp divergence occurs when threads in the same warp take different branches, causing serial execution. Our boundary check `if (col < width && row < height)` can cause divergence only in the last row/column of blocks. For large images this is a tiny fraction of all warps (<1%), so the impact is negligible.

### Q: What is occupancy and how does it affect performance?
**A:** Occupancy is the ratio of active warps to maximum warps per SM. Higher occupancy helps hide memory latency by switching to another warp while the first waits for a memory fetch. Our 16×16 blocks (256 threads = 8 warps) combined with the shared memory usage achieve moderate occupancy. For memory-bound kernels, occupancy is the primary lever after bandwidth.

### Q: How does the separable pass intermediate result avoid GPU-CPU transfer?
**A:** The horizontal pass writes its result into `d_temp`, a device buffer allocated with `cudaMalloc`. The vertical pass reads directly from `d_temp`. Both passes run on the GPU; no `cudaMemcpy` to host is needed between them.

---

## 4. CONTRIBUTION SUMMARY

| Member | Primary Contribution |
|--------|---------------------|
| Vasishta Nandipati (230962095) | Naive CUDA kernel implementation, benchmarking framework, CPU reference |
| Rishit Mathur (220962091) | Shared-memory tiled kernel, halo/boundary correctness validation |
| Krishiv Kolanu (230962023) | Separable kernel decomposition, roofline analysis, performance report |

---

## 5. BUILD & RUN REMINDER

```bash
make                    # build (auto-detects GPU arch)
make ARCH=sm_86         # build for Ampere GPUs (RTX 3xxx)
./conv_bench            # run full benchmark
./conv_bench 512 512    # quick run
```
