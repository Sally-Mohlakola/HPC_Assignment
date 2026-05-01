# Compiler and flags
CC = gcc
NVCC = nvcc
CFLAGS = -fopenmp -I ./include
NVCCFLAGS = -Xcompiler -fopenmp -Drestrict=__restrict__ -I ./include
LDFLAGS = -lm

# Source files
CUDA_SRCS = cuda_renderer.cu
C_SRCS = main.c \
         openmp_renderer.c \
         sources/allocator.c \
         sources/camera.c \
         sources/color.c \
         sources/hitRecord.c \
         sources/material.c \
         sources/outfile.c \
         sources/ray.c \
         sources/sphere.c \
         sources/texture.c \
         sources/util.c

OBJS = $(C_SRCS:.c=.o) $(CUDA_SRCS:.cu=.o)

# Output binary
TARGET = raytracer

# Build rule
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -Xcompiler -fopenmp $(OBJS) -o $(TARGET) $(LDFLAGS)

cuda_renderer.o: cuda_renderer.cu cuda_helper.cuh
	$(NVCC) $(NVCCFLAGS) -c cuda_renderer.cu -o cuda_renderer.o

openmp_renderer.o: openmp_renderer.c openmp_helper.h
	$(CC) $(CFLAGS) -c openmp_renderer.c -o openmp_renderer.o

sources/%.o: sources/%.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGET) $(OBJS)
