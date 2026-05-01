#ifndef OPENMP_HELPER_H
#define OPENMP_HELPER_H

#include <stdbool.h>

#include "allocator.h"
#include "camera.h"
#include "color.h"
#include "hitRecord.h"
#include "hypatiaINC.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "texture.h"
#include "types.h"
#include "util.h"

CFLOAT lcg(int *n);
RGBColorU8 writeColor(CFLOAT r, CFLOAT g, CFLOAT b, int sample_per_pixel);
RGBColorF ray_c(Ray r, const ObjectLL *world, int depth);
void randomSpheres(ObjectLL *world, DynamicStackAlloc *dsa);
void randomSpheres2(ObjectLL *world, DynamicStackAlloc *dsa, int n,
                    Image imgs[n], int *seed);

void renderOpenMP(
    RGBColorU8 *hImage,
    int WIDTH,
    int HEIGHT,
    int SAMPLES_PER_PIXEL,
    int MAX_DEPTH,
    Camera hC,
    Image img[4],
    int seed,
    unsigned int stepSize
);

#endif
