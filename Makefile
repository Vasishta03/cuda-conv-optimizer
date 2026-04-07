# ─────────────────────────────────────────────────────────────────────────────
#  Makefile — CUDA Image Convolution Benchmark
#
#  Usage:
#    make            →  build with auto-detected GPU architecture
#    make ARCH=sm_86 →  build for Ampere (RTX 30xx / A-series)
#    make ARCH=sm_89 →  build for Ada Lovelace (RTX 40xx)
#    make ARCH=sm_75 →  build for Turing (RTX 20xx / T4)
#    make ARCH=sm_70 →  build for Volta (V100)
#    make ARCH=sm_61 →  build for Pascal (GTX 10xx / P-series)
#    make run        →  build + run full benchmark
#    make quick      →  build + run quick 512×512 test
#    make clean      →  remove binaries
# ─────────────────────────────────────────────────────────────────────────────

NVCC      := nvcc
INCDIR    := include
SRCDIR    := src
TARGET    := conv_bench

# Detect GPU architecture automatically (requires nvcc ≥ 11.6)
# Override with: make ARCH=sm_XX
ARCH ?= native

NVCCFLAGS := -O3 -std=c++17 --use_fast_math -lineinfo
ifeq ($(ARCH),native)
    NVCCFLAGS += -arch=native
else
    NVCCFLAGS += -arch=$(ARCH)
endif

SRCS := $(SRCDIR)/main.cu          \
        $(SRCDIR)/naive_conv.cu     \
        $(SRCDIR)/tiled_conv.cu     \
        $(SRCDIR)/separable_conv.cu

# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all run quick clean info

all: $(TARGET)

$(TARGET): $(SRCS) $(INCDIR)/convolution.h
	@echo "[NVCC] Building $@ (arch=$(ARCH)) ..."
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) -o $@ $(SRCS)
	@echo "[OK]  Build successful → ./$(TARGET)"

run: $(TARGET)
	@echo "[RUN] Full benchmark matrix (4 resolutions × 4 kernel radii) ..."
	./$(TARGET)

quick: $(TARGET)
	@echo "[RUN] Quick 512×512 benchmark ..."
	./$(TARGET) 512 512

info:
	@nvcc --version
	@nvidia-smi --query-gpu=name,driver_version,memory.total \
	            --format=csv,noheader

clean:
	rm -f $(TARGET)
	@echo "[OK]  Cleaned."
