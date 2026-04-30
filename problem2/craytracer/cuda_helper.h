#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

#include "camera.h"
#include "color.h"
#include "hitRecord.h"
#include "hypatiaINC.h"
#include "ray.h"
#include "types.h"
#include "util.h"

#define MAT_LAMBERTIAN 0
#define MAT_METAL      1
#define MAT_DIELECTRIC 2

#define MAX_SPHERES   512
#define MAX_MATERIALS 512

typedef struct {
    vec3 center;
    CFLOAT radius;
    int materialIndex;
} DeviceSphere;

typedef struct {
    int type;
    RGBColorF albedo;
    CFLOAT fuzz;
    CFLOAT ir;
} DeviceMaterial;

CFLOAT lcg(int *n);
RGBColorU8 writeColor(CFLOAT r, CFLOAT g, CFLOAT b, int sample_per_pixel);

__device__ CFLOAT cuRand(unsigned int *rngState);
__device__ vec3 cuRandomUnitVector(unsigned int *rngState);
__device__ bool cuNearZero(vec3 v);
__device__ bool cuHitWorld(
    const DeviceSphere *spheres,
    const DeviceMaterial *materials,
    int numSpheres,
    Ray r,
    CFLOAT tMin,
    CFLOAT tMax,
    HitRecord *rec
);
__device__ bool cuHitSphere(
    const DeviceSphere *sphere,
    Ray r,
    CFLOAT tMin,
    CFLOAT tMax,
    HitRecord *rec
);

void cuAddSphere(
    DeviceSphere *h_spheres,
    DeviceMaterial *h_materials,
    int *numSpheres,
    int *numMaterials,
    int maxSpheres,
    int maxMaterials,
    vec3 center,
    CFLOAT radius,
    DeviceMaterial material
) {
    if (*numSpheres >= maxSpheres || *numMaterials >= maxMaterials) {
        fprintf(stderr, "Scene arrays are full\n");
        exit(1);
    }

    int matIndex = *numMaterials;

    h_materials[matIndex] = material;
    (*numMaterials)++;

    h_spheres[*numSpheres].center = center;
    h_spheres[*numSpheres].radius = radius;
    h_spheres[*numSpheres].materialIndex = matIndex;
    (*numSpheres)++;
}

__device__ vec3 cuReflect(vec3 v, vec3 n) {
    vec3 result = n;
    CFLOAT scale = 2.0 * vector3_dot(v, n);
    vector3_multiplyScalar(&result, scale);
    vector3_subtract(&v, &result);
    return v;
}

__device__ CFLOAT cuReflectance(CFLOAT cosine, CFLOAT refIdx) {
    CFLOAT r0 = (1.0 - refIdx) / (1.0 + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

__device__ vec3 cuRefract(vec3 uv, vec3 n, CFLOAT etaiOverEtat) {
    vec3 negUv = uv;
    vector3_multiplyScalar(&negUv, -1.0);

    CFLOAT cosTheta = fmin(vector3_dot(negUv, n), 1.0);

    vec3 rOutPerp = uv;

    vec3 temp = n;
    vector3_multiplyScalar(&temp, cosTheta);
    vector3_add(&rOutPerp, &temp);
    vector3_multiplyScalar(&rOutPerp, etaiOverEtat);

    CFLOAT lenSq =
        rOutPerp.x * rOutPerp.x +
        rOutPerp.y * rOutPerp.y +
        rOutPerp.z * rOutPerp.z;

    vec3 rOutParallel = n;
    vector3_multiplyScalar(&rOutParallel, -sqrt(fabs(1.0 - lenSq)));

    vector3_add(&rOutPerp, &rOutParallel);

    return rOutPerp;
}

__device__ bool cuScatterLambertian(
    Ray rayIn,
    HitRecord rec,
    DeviceMaterial mat,
    RGBColorF *attenuation,
    Ray *scattered,
    unsigned int *rngState
) {
    vec3 scatterDirection = rec.normal;
    vec3 randomVec = cuRandomUnitVector(rngState);
    vector3_add(&scatterDirection, &randomVec);

    if (cuNearZero(scatterDirection)) {
        scatterDirection = rec.normal;
    }

    scattered->origin = rec.p;
    scattered->direction = scatterDirection;

    *attenuation = mat.albedo;

    return true;
}

__device__ bool cuScatterMetal(
    Ray rayIn,
    HitRecord rec,
    DeviceMaterial mat,
    RGBColorF *attenuation,
    Ray *scattered,
    unsigned int *rngState
) {
    vec3 unitDirection = rayIn.direction;
    vector3_normalize(&unitDirection);

    vec3 reflected = cuReflect(unitDirection, rec.normal);

    vec3 fuzzVec = cuRandomUnitVector(rngState);
    vector3_multiplyScalar(&fuzzVec, mat.fuzz);
    vector3_add(&reflected, &fuzzVec);

    scattered->origin = rec.p;
    scattered->direction = reflected;

    *attenuation = mat.albedo;

    return vector3_dot(scattered->direction, rec.normal) > 0.0;
}

__device__ bool cuScatterDielectric(
    Ray rayIn,
    HitRecord rec,
    DeviceMaterial mat,
    RGBColorF *attenuation,
    Ray *scattered,
    unsigned int *rngState
) {
    *attenuation = (RGBColorF){1.0, 1.0, 1.0};

    CFLOAT refractionRatio;

    if (rec.frontFace) {
        refractionRatio = 1.0 / mat.ir;
    } else {
        refractionRatio = mat.ir;
    }

    vec3 unitDirection = rayIn.direction;
    vector3_normalize(&unitDirection);

    vec3 negUnitDirection = unitDirection;
    vector3_multiplyScalar(&negUnitDirection, -1.0);

    CFLOAT cosTheta = fmin(vector3_dot(negUnitDirection, rec.normal), 1.0);
    CFLOAT sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    vec3 direction;

    if (cannotRefract ||
        cuReflectance(cosTheta, refractionRatio) > cuRand(rngState)) {
        direction = cuReflect(unitDirection, rec.normal);
    } else {
        direction = cuRefract(unitDirection, rec.normal, refractionRatio);
    }

    scattered->origin = rec.p;
    scattered->direction = direction;

    return true;
}

__device__ bool cuScatter(
    Ray rayIn,
    HitRecord rec,
    DeviceMaterial mat,
    RGBColorF *attenuation,
    Ray *scattered,
    unsigned int *rngState
) {
    if (mat.type == MAT_LAMBERTIAN) {
        return cuScatterLambertian(rayIn, rec, mat, attenuation, scattered, rngState);
    }

    if (mat.type == MAT_METAL) {
        return cuScatterMetal(rayIn, rec, mat, attenuation, scattered, rngState);
    }

    if (mat.type == MAT_DIELECTRIC) {
        return cuScatterDielectric(rayIn, rec, mat, attenuation, scattered, rngState);
    }

    return false;
}

__device__ RGBColorF cuRayC(
    Ray r,
    const DeviceSphere *spheres,
    const DeviceMaterial *materials,
    int numSpheres,
    int maxDepth,
    unsigned int *rngState
) {
    RGBColorF finalColor = {1.0, 1.0, 1.0};

    for (int depth = 0; depth < maxDepth; depth++) {
        HitRecord rec;

        if (cuHitWorld(spheres, materials, numSpheres, r, 0.0001, FLT_MAX, &rec)) {
            Ray scattered;
            RGBColorF attenuation;

            if (cuScatter(r, rec, materials[rec.materialIndex],
                          &attenuation, &scattered, rngState)) {
                finalColor = colorf_multiply(finalColor, attenuation);
                r = scattered;
            } else {
                return (RGBColorF){0.0, 0.0, 0.0};
            }
        } else {
            vec3 unitDirection = r.direction;
            vector3_normalize(&unitDirection);

            CFLOAT t = 0.5 * (unitDirection.y + 1.0);

            RGBColorF background = {
                .r = (1.0 - t) * 1.0 + t * 0.5,
                .g = (1.0 - t) * 1.0 + t * 0.7,
                .b = (1.0 - t) * 1.0 + t * 1.0
            };

            return colorf_multiply(finalColor, background);
        }
    }

    return (RGBColorF){0.0, 0.0, 0.0};
}

void cuRandomSpheres2(DeviceSphere *h_spheres,
    DeviceMaterial *h_materials,
    int *numSpheres,
    int *numMaterials,
    int maxSpheres,
    int maxMaterials,
    int *seed) {

    RGBColorF albedo = {
    .r = lcg(seed) / 2 + 0.5,
    .g = lcg(seed) / 2 + 0.5,
    .b = lcg(seed) / 2 + 0.5
    };

    CFLOAT fuzz = lcg(seed) / 2 + 0.5;

    DeviceMaterial mat;
    mat.type = MAT_METAL;
    mat.albedo = albedo;
    mat.fuzz = fuzz;
    mat.ir = 1.0;

    DeviceMaterial groundMat;

    groundMat.type = MAT_LAMBERTIAN;
    groundMat.albedo = (RGBColorF){.r = 0.4, .g = 0.4, .b = 0.4};
    groundMat.fuzz = 0.0;
    groundMat.ir = 1.0;

    cuAddSphere(
    h_spheres,
    h_materials,
    numSpheres,
    numMaterials,
    maxSpheres,
    maxMaterials,
    (vec3){.x = 0, .y = -1000, .z = 0},
    1000,
    groundMat
    );

    for (int a = -2; a < 9; a++) {
        for (int b = -9; b < 9; b++) {
            CFLOAT chooseMat = lcg(seed);
            vec3 center = {
                .x = a + 0.9 * lcg(seed), .y = 0.2, .z = b + 0.9 * lcg(seed)};

            if (chooseMat < 0.8) {
                RGBColorF albedo = {
                    .r = lcg(seed) * lcg(seed),
                    .g = lcg(seed) * lcg(seed),
                    .b = lcg(seed) * lcg(seed),
                };

                DeviceMaterial mat;

                mat.type = MAT_LAMBERTIAN;
                mat.albedo = albedo;
                mat.fuzz = 0.0;
                mat.ir = 1.0;

                cuAddSphere(
                    h_spheres,
                    h_materials,
                    numSpheres,
                    numMaterials,
                    maxSpheres,
                    maxMaterials,
                    center,
                    0.2,
                    mat
                );

            } else if (chooseMat < 0.95) {
                DeviceMaterial mat;

                mat.type = MAT_METAL;
                mat.albedo = albedo;
                mat.fuzz = fuzz;
                mat.ir = 1.0;

                cuAddSphere(
                    h_spheres,
                    h_materials,
                    numSpheres,
                    numMaterials,
                    maxSpheres,
                    maxMaterials,
                    center,
                    0.2,
                    mat
                );

            } else {
                DeviceMaterial mat;

                mat.type = MAT_DIELECTRIC;
                mat.albedo = (RGBColorF){.r = 1.0, .g = 1.0, .b = 1.0};
                mat.fuzz = 0.0;
                mat.ir = 1.5;

                cuAddSphere(
                    h_spheres,
                    h_materials,
                    numSpheres,
                    numMaterials,
                    maxSpheres,
                    maxMaterials,
                    center,
                    0.2,
                    mat
                );
            }
        }
    }

    mat.type = MAT_LAMBERTIAN;
    mat.albedo = (RGBColorF){.r = 0.4, .g = 0.2, .b = 0.1};
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    cuAddSphere(
        h_spheres,
        h_materials,
        numSpheres,
        numMaterials,
        maxSpheres,
        maxMaterials,
        (vec3){.x = -4, .y = 1, .z = 0},
        1.0,
        mat
    );

    mat.type = MAT_LAMBERTIAN;
    mat.albedo = (RGBColorF){.r = 0.2, .g = 0.4, .b = 0.8};
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    cuAddSphere(
        h_spheres,
        h_materials,
        numSpheres,
        numMaterials,
        maxSpheres,
        maxMaterials,
        (vec3){.x = -4, .y = 1, .z = -2.2},
        1.0,
        mat
    );

    mat.type = MAT_LAMBERTIAN;
    mat.albedo = (RGBColorF){.r = 0.8, .g = 0.4, .b = 0.2};
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    cuAddSphere(
        h_spheres,
        h_materials,
        numSpheres,
        numMaterials,
        maxSpheres,
        maxMaterials,
        (vec3){.x = -4, .y = 1, .z = 2.2},
        1.0,
        mat
    );

    mat.type = MAT_LAMBERTIAN;
    mat.albedo = (RGBColorF){.r = 0.4, .g = 0.8, .b = 0.3};
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    cuAddSphere(
        h_spheres,
        h_materials,
        numSpheres,
        numMaterials,
        maxSpheres,
        maxMaterials,
        (vec3){.x = -4, .y = 1, .z = -4.2},
        1.0,
        mat
    );
}

#endif
