#ifndef OPENMP_HELPER_H
#define OPENMP_HELPER_H

#include "allocator.h"
#include "color.h"
#include "ray.h"
#include "sphere.h"
#include "texture.h"
#include "types.h"

RGBColorF ray_c(Ray r, const ObjectLL *world, int depth);
void randomSpheres(ObjectLL *world, DynamicStackAlloc *dsa);
void randomSpheres2(ObjectLL *world, DynamicStackAlloc *dsa, int n,
                    Image imgs[n], int *seed);

#endif
