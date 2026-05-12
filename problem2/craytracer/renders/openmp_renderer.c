#include <omp.h>
#include <stdio.h>

#include "openmp_helper.h"
#include "outfile.h"

void renderOpenMP(
    RGBColorU8 *hImage,
    int WIDTH,
    int HEIGHT,
    int SAMPLES_PER_PIXEL,
    int MAX_DEPTH,
    Camera hC,
    Image img[4],
    int seed
) {
    DynamicStackAlloc *dsa = alloc_createDynamicStackAllocD(1024, 100);
    DynamicStackAlloc *dsa0 = alloc_createDynamicStackAllocD(1024, 10);
    ObjectLL *world = obj_createObjectLL(dsa0, dsa);

    randomSpheres2(world, dsa, 4, img, &seed);

    LinearAllocFC *lafc =
        alloc_createLinearAllocFC(MAX_DEPTH * world->numObjects,
                                  sizeof(HitRecord), alignof(HitRecord));

    world->hrAlloc = lafc;

    CFLOAT pcR, pcG, pcB;
    Ray r;
    RGBColorF temp;

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
    }

    writeToPPM("output/openmp.jpg", WIDTH, HEIGHT, hImage);

    alloc_freeLinearAllocFC(lafc);
    alloc_freeDynamicStackAllocD(dsa);
    alloc_freeDynamicStackAllocD(dsa0);
}
