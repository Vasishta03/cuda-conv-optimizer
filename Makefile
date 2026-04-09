NVCC      := nvcc
INCDIR    := include
SRCDIR    := src
TARGET    := conv_bench

ARCH ?= native

NVCCFLAGS := -O3 --use_fast_math -lineinfo
ifeq ($(ARCH),native)
    NVCCFLAGS += -arch=native
else
    NVCCFLAGS += -arch=$(ARCH)
endif

SRCS := $(SRCDIR)/main.cu          \
        $(SRCDIR)/naive_conv.cu     \
        $(SRCDIR)/tiled_conv.cu     \
        $(SRCDIR)/separable_conv.cu

.PHONY: all run quick clean info

all: $(TARGET)

$(TARGET): $(SRCS) $(INCDIR)/convolution.h
	@echo "[NVCC] Building $@ (arch=$(ARCH)) ..."
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) -o $@ $(SRCS)
	@echo "[OK]  Build successful -> ./$(TARGET)"

run: $(TARGET)
	@echo "[RUN] Full benchmark matrix (4 resolutions x 4 kernel radii) ..."
	./$(TARGET)

quick: $(TARGET)
	@echo "[RUN] Quick 512x512 benchmark ..."
	./$(TARGET) 512 512

info:
	@nvcc --version
	@nvidia-smi --query-gpu=name,driver_version,memory.total \
	            --format=csv,noheader

clean:
	rm -f $(TARGET)
	@echo "[OK]  Cleaned."
