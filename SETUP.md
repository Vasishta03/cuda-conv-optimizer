# Setup Guide — CUDA Image Convolution Benchmark

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| CUDA Toolkit | 11.6 | 12.x |
| GPU | Compute Capability 6.0 (Pascal) | CC 8.6+ (Ampere / Ada) |
| GCC / G++ | 9.x | 12.x |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

### Check your GPU compute capability

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

Common GPUs and their compute capabilities:

| GPU | Architecture | CC | ARCH flag |
|-----|-------------|-----|-----------|
| RTX 4090 / 4080 | Ada Lovelace | 8.9 | sm_89 |
| RTX 3090 / 3080 / A100 | Ampere | 8.6 | sm_86 |
| RTX 2080 / T4 | Turing | 7.5 | sm_75 |
| V100 | Volta | 7.0 | sm_70 |
| GTX 1080 / P100 | Pascal | 6.1 | sm_61 |

---

## 1. Install CUDA Toolkit (if not already installed)

```bash
# Ubuntu — install CUDA 12.x
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Add to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
nvcc --version    # should print CUDA 12.x
```

---

## 2. Clone / enter the project directory

```bash
cd /path/to/mini-proj
```

Project structure:
```
mini-proj/
├── include/
│   └── convolution.h       ← shared constants & function declarations
├── src/
│   ├── main.cu             ← benchmark driver
│   ├── naive_conv.cu       ← Baseline: global-memory kernel
│   ├── tiled_conv.cu       ← Optimisation 1: shared-memory tiling
│   └── separable_conv.cu   ← Optimisation 2: separable 1-D passes
├── Makefile
├── SETUP.md                ← this file
└── HOW_IT_WORKS.md
```

---

## 3. Build

### Option A — auto-detect GPU architecture (CUDA ≥ 11.6)
```bash
make
```

### Option B — specify your GPU architecture explicitly
```bash
make ARCH=sm_86    # Ampere (RTX 3xxx)
make ARCH=sm_89    # Ada (RTX 4xxx)
make ARCH=sm_75    # Turing (RTX 2xxx / T4)
```

Successful build output:
```
[NVCC] Building conv_bench (arch=native) ...
[OK]  Build successful → ./conv_bench
```

---

## 4. Run

### Full benchmark (4 resolutions × 4 kernel radii)
```bash
make run
# or directly:
./conv_bench
```

### Quick single-resolution test
```bash
make quick
# or:
./conv_bench 512 512
```

### Custom resolution
```bash
./conv_bench 1920 1080
./conv_bench 3840 2160
```

---

## 5. Expected output (example — RTX 3090)

```
  ╔══════════════════════════════════════════════════════════╗
  ║  CUDA Image Convolution Benchmark                        ║
  ║  Shared-Memory Tiling & Separable Kernels                ║
  ╚══════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│ GPU : NVIDIA GeForce RTX 3090                               │
│ SMs : 82    Peak BW: 936.2 GB/s   Shared mem: 48    KB     │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
  Resolution : 1024 × 1024   (1048576 pixels)
═══════════════════════════════════════════════════════════════
Method            r    Err        ms      GB/s    GFLOP/s    Spdup
────────────────  ─  ──────  ────────  ────────  ──────────  ───────
CPU Reference     1  ─       12.453    0.0000       0.54      1.00x
Naive GPU         1  3.8e-7   0.412    9.786       53.29    30.23x  ✓
Tiled GPU         1  3.8e-7   0.187   21.545      117.39    66.59x  ✓
Separable GPU     1  4.1e-7   0.143   28.155       72.99    87.08x  ✓
  [Roofline] r=1   AI_2D=2.3 FLOP/B  AI_Sep=1.5 FLOP/B  Tiled 2.20x over Naive
...
```

---

## 6. Profiling with NVIDIA Nsight Compute

```bash
ncu --set full -o profile_naive   ./conv_bench 1024 1024
ncu --set full -o profile_tiled   ./conv_bench 1024 1024
ncu-ui profile_naive.ncu-rep      # open in GUI
```

Key metrics to check:
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` → global memory loads
- `sm__shared_load_transactions.sum` → shared memory load transactions  
- `sm__warps_active.avg.pct_of_peak_sustained_active` → occupancy

---

## 7. Troubleshooting

| Error | Fix |
|-------|-----|
| `nvcc: not found` | Add `/usr/local/cuda/bin` to PATH |
| `no kernel image available` | Set ARCH to match your GPU (see table above) |
| `out of memory` | Reduce image size or use `make quick` |
| `CUDA error: invalid device function` | Recompile with correct `-arch=smXX` |
| `make: nvcc: Command not found` | `sudo apt install cuda-toolkit-12-4` |

---

## 8. Dependencies (all included / standard)

- CUDA Runtime (`libcudart`) — ships with CUDA Toolkit
- Standard C++ library (`libstdc++`) — included with GCC
- No external image library required — synthetic test images are generated at runtime

> **Optional**: To compare against OpenCV's `cv::filter2D`, install OpenCV 4.x with `sudo apt install libopencv-dev` and recompile with `make OPENCV_ENABLE=1` (if you add that flag to the Makefile).
