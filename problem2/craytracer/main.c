#define HYPATIA_IMPLEMENTATION
#include <assert.h>
#include <omp.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <time.h>

#include "allocator.h"
#include "camera.h"
#include "color.h"
#include "hitRecord.h"
#include "hypatiaINC.h"
#include "material.h"
#include "outfile.h"
#include "openmp_helper.h"
#include "ray.h"
#include "sphere.h"
#include "texture.h"
#include "types.h"
#include "util.h"

void printProgressBar(int i, int max) {
    int p = (int)(100 * (CFLOAT)i / max);

    printf("|");
    for (int j = 0; j < p; j++) {
        printf("=");
    }

    for (int j = p; j < 100; j++) {
        printf("*");
    }

    if (p == 100) {
        printf("|\n");
    } else {
        printf("|\r");
    }
}

CFLOAT lcg(int *n) {

    static int seed;
    const int m = 2147483647;
    const int a = 1103515245;
    const int c = 12345;

    if (n != NULL) {
        seed = *n;
    }

    seed = (seed * a + c) % m;
    *n = seed;

    return fabs((CFLOAT)seed / m);
}

#include "cuda_helper.h"

__global__ void cuRaytracerBase(
    RGBColorU8 *dImage,
    Camera dCamera,
    const DeviceSphere *dSpheres,
    const DeviceMaterial *dMaterials,
    int numSpheres,
    int width,
    int height,
    int samplesPerPixel,
    int maxDepth,
    unsigned int seed
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;

    if (l >= width * height) {
        return;
    }

    int j = (height - 1) - l / width;
    int i = l % width;

    unsigned int rngState = seed ^ l;

    CFLOAT pcR, pcG, pcB;
    pcR = pcG = pcB = 0.0;

    for (int k = 0; k < samplesPerPixel; k++) {
        CFLOAT u =
            ((CFLOAT)i + cuRand(&rngState)) / (width - 1);
        CFLOAT v =
            ((CFLOAT)j + cuRand(&rngState)) / (height - 1);
        Ray r = cuCamGetRay(&dCamera, u, v, &rngState);

        RGBColorF temp = cuRayC(
            r,
            d_spheres,
            d_materials,
            d_textures,
            d_imageTextures,
            numSpheres,
            maxDepth,
            &rngState
        );

        pcR += temp.r;
        pcG += temp.g;
        pcB += temp.b;
    }

    dImage[i + width * (height - 1 - j)] =
        cuWriteColor(pcR, pcG, pcB, samplesPerPixel);

    localSteps += 1;

    __syncthreads();
    if (localSteps % stepSize == stepSize - 1) {
        stepsCompleted += 1;

        if (stepsCompleted % 100 == 1) {


            printf("Progress %lu of %u (%0.2lf%%)\n", stepsCompleted,
                    totalSteps,
                    100.0 * (CFLOAT)stepsCompleted / totalSteps);
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("FATAL ERROR: Output file name not provided.\n");
        printf("EXITING ...\n");
        return 0;
    }

    srand(time(NULL));

    const CFLOAT aspect_ratio = 16.0 / 9.0;
    const int WIDTH = 640;
    const int HEIGHT = 640;
    const int SAMPLES_PER_PIXEL = 100;
    const int MAX_DEPTH = 50;
    RGBColorU8 *image =
        (RGBColorU8 *)malloc(sizeof(RGBColorU8) * HEIGHT * WIDTH);
    RGBColorU8 *hImage =
        (RGBColorU8 *)malloc(sizeof(RGBColorU8) * HEIGHT * WIDTH);

    CFLOAT start = omp_get_wtime();

    uint32_t stepSize = 500;
    uint32_t totalSteps = (WIDTH * HEIGHT) / stepSize + 1;
    size_t stepsCompleted = 0;

    size_t localSteps = 0;

    int seed = 100;

    // set camera parameters
    vec3 lookFrom = {.x = 13.0, .y = 2.0, .z = 3.0};
    vec3 lookAt = {.x = 0.0, .y = 0.0, .z = 0.0};
    vec3 up = {.x = 0.0, .y = 1.0, .z = 0.0};

    CFLOAT distToFocus = 10.0;
    CFLOAT aperture = 0.1;

    Camera hC;
    cam_setLookAtCamera(&hC, lookFrom, lookAt, up, 20, aspect_ratio,
                        aperture, distToFocus);

    Ray r;
    RGBColorF temp;

    // memory allocation for the objects in the world
    DynamicStackAlloc *dsa = alloc_createDynamicStackAllocD(1024, 100);
    DynamicStackAlloc *dsa0 = alloc_createDynamicStackAllocD(1024, 10);
    ObjectLL *world = obj_createObjectLL(dsa0, dsa);

    Image img[4];

    tex_loadImage(&img[0], "./test_textures/kitchen_probe.jpg");
    tex_loadImage(&img[1], "./test_textures/campus_probe.jpg");
    tex_loadImage(&img[2], "./test_textures/building_probe.jpg");
    tex_loadImage(&img[3], "./test_textures/kitchen_probe.jpg");

    randomSpheres2(world, dsa, 4, img, &seed);
    printf("Scene initialized\n");

    LinearAllocFC *lafc =
        alloc_createLinearAllocFC(MAX_DEPTH * world->numObjects,
                                    sizeof(HitRecord), alignof(HitRecord));

    world->hrAlloc = lafc;

// OPENMP PARALLELIZATION   

#pragma omp for
        for (int l = 0; l < WIDTH * HEIGHT; l++) {
            int j = (HEIGHT - 1) - l / WIDTH;
            int i = l % WIDTH;
            pcR = pcG = pcB = 0.0;

            for (int k = 0; k < SAMPLES_PER_PIXEL; k++) {
                CFLOAT u =
                    ((CFLOAT)i + util_randomFloat(0.0, 1.0)) / (WIDTH - 1);
                CFLOAT v =
                    ((CFLOAT)j + util_randomFloat(0.0, 1.0)) / (HEIGHT - 1);
                r = cam_getRay(&hC, u, v);

                temp = ray_c(r, world, MAX_DEPTH);

                pcR += temp.r;
                pcG += temp.g;
                pcB += temp.b;

                alloc_linearAllocFCFreeAll(lafc);
            }

            hImage[i + WIDTH * (HEIGHT - 1 - j)] =
                writeColor(pcR, pcG, pcB, SAMPLES_PER_PIXEL);

            localSteps += 1;

            if (localSteps % stepSize == stepSize - 1) {
#pragma omp atomic
                stepsCompleted += 1;

                if (stepsCompleted % 100 == 1) {
#pragma omp critical
                    printf("Progress %lu of %u (%0.2lf%%)\n", stepsCompleted,
                           totalSteps,
                           100.0 * (CFLOAT)stepsCompleted / totalSteps);
                }
            }
        }

        alloc_freeLinearAllocFC(lafc);
        alloc_freeDynamicStackAllocD(dsa);
        alloc_freeDynamicStackAllocD(dsa0);
    }

    CFLOAT end = omp_get_wtime();

    printf("Execution time: %lf\n", end - start);

// CUDA PARALLELIZATION
    RGBColorU8 *dImage;
    cudaMalloc(&dImage, sizeof(RGBColorU8) * HEIGHT * WIDTH);

    DeviceSphere *h_spheres =
        malloc(sizeof(DeviceSphere) * MAX_SPHERES);

    DeviceMaterial *h_materials =
        malloc(sizeof(DeviceMaterial) * MAX_MATERIALS);

    int numSpheres = 0;
    int numMaterials = 0;

    cuRandomSpheres2(
    h_spheres,
    h_materials,
    &numSpheres,
    &numMaterials,
    MAX_SPHERES,
    MAX_MATERIALS,
    &seed
    );

    

    cudaMalloc(&d_spheres, sizeof(DeviceSphere) * numSpheres);
    cudaMalloc(&d_materials, sizeof(DeviceMaterial) * numMaterials);

    cudaMemcpy(d_spheres, h_spheres,
            sizeof(DeviceSphere) * numSpheres,
            cudaMemcpyHostToDevice);

    cudaMemcpy(d_materials, h_materials,
            sizeof(DeviceMaterial) * numMaterials,
            cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numPixels = WIDTH * HEIGHT;
    int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

    cuRaytracerBase<<<blocks, threadsPerBlock>>>(
        dImage,
        hC,
        d_spheres,
        d_materials,
        numSpheres,
        WIDTH,
        HEIGHT,
        SAMPLES_PER_PIXEL,
        MAX_DEPTH,
        100u
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    
    cudaMemcpy(hImage, dImage,
           sizeof(RGBColorU8) * WIDTH * HEIGHT,
           cudaMemcpyDeviceToHost);

    writeToPPM(argv[1], WIDTH, HEIGHT, image);
    writeToPPM(argv[1], WIDTH, HEIGHT, hImage);

    free(image);
    free(hImage);
    cudaFree(dImage);
    cudaFree(dC);

    return 0;
}
