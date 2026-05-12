# Compiler and flags
CC = gcc
NVCC = nvcc

MAX_SPHERES    ?= 512

CONFIG_DEFINES = -DMAX_SPHERES=$(MAX_SPHERES)

BUILD_DIR = renders/obj

CFLAGS = -fopenmp -I ./include -I ./renders $(CONFIG_DEFINES)
NVCCFLAGS = -Xcompiler -fopenmp -Drestrict=__restrict__ -I ./include -I ./renders $(CONFIG_DEFINES)
LDFLAGS = -lm

# Source files
CUDA_SRCS = renders/cuda_renderer.cu
RENDER_C_SRCS = renders/main.c \
                renders/openmp_renderer.c
SOURCE_SRCS = sources/allocator.c \
              sources/camera.c \
              sources/color.c \
              sources/hitRecord.c \
              sources/material.c \
              sources/outfile.c \
              sources/ray.c \
              sources/sphere.c \
              sources/texture.c \
              sources/util.c

RENDER_C_OBJS = $(patsubst renders/%.c,$(BUILD_DIR)/%.o,$(RENDER_C_SRCS))
SOURCE_OBJS = $(patsubst sources/%.c,$(BUILD_DIR)/%.o,$(SOURCE_SRCS))
CUDA_OBJS = $(patsubst renders/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SRCS))
OBJS = $(RENDER_C_OBJS) $(SOURCE_OBJS) $(CUDA_OBJS)

# Output binary
TARGET = renders/raytracer

# Build rule
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) | renders
	$(NVCC) -Xcompiler -fopenmp $(OBJS) -o $@ $(LDFLAGS)

renders $(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/cuda_renderer.o: renders/cuda_renderer.cu renders/cuda_helper.cuh renders/metrics.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: renders/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: sources/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o $(BUILD_DIR)/openmp_renderer.o: renders/openmp_helper.h renders/metrics.h

# Clean rule
clean:
	rm -f $(TARGET) raytracer $(OBJS) *.o sources/*.o renders/*.o
	rm -rf $(BUILD_DIR)
