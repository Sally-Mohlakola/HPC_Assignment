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
#include "ray.h"
#include "sphere.h"
#include "texture.h"
#include "types.h"
#include "util.h"

RGBColorU8 writeColor(CFLOAT r, CFLOAT g, CFLOAT b, int sample_per_pixel) {
    CFLOAT scale = 1.0 / sample_per_pixel;

    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    return COLOR_U8CREATE(r, g, b);
}
RGBColorF ray_c(Ray r, const ObjectLL *world, int depth) {

    if (depth <= 0) {
        return (RGBColorF){0};
    }

    HitRecord rec;
    rec.valid = false;
    // checks if the ray hits an object
    bool checkHit = obj_objLLHit(world, r, 0.00001, FLT_MAX, &rec);

    if (checkHit) {
        // the scattered ray
        Ray scattered = {0};
        // attenuation factor due to scattering and absorption
        RGBColorF attenuation = {0};

        // calculate the scattered ray and the attenuation factor based on the
        // material
        if (mat_scatter(&r, &rec, &attenuation, &scattered)) {
            // calcuate the colour of the scattere ray
            RGBColorF color = ray_c(scattered, world, depth - 1);
            // multiply the colour by the attenuation factor
            color = colorf_multiply(color, attenuation);

            return color;
        }

        return (RGBColorF){0};
    }

    // if the ray doesn't hit and object, return the background colour

    vec3 ud = r.direction;
    vector3_normalize(&ud);
    CFLOAT t = 0.5 * (ud.y + 1.0);
    vec3 inter4;
    vector3_setf3(&inter4, 1.0 - t, 1.0 - t, 1.0 - t);
    vec3 inter3;
    vector3_setf3(&inter3, 0.5 * t, 0.7 * t, 1.0 * t);
    vector3_add(&inter3, &inter4);

    return (RGBColorF){.r = inter3.x, .g = inter3.y, .b = inter3.z};
}

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

#define randomFloat() util_randomFloat(0.0, 1.0)

void randomSpheres(ObjectLL *world, DynamicStackAlloc *dsa) {

    LambertianMat *materialGround = alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));
    SolidColor *sc1 = alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                      alignof(SolidColor));

    SolidColor *sc = alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                     alignof(SolidColor));

    Checker *c =
        alloc_dynamicStackAllocAllocate(dsa, sizeof(Checker), alignof(Checker));

    sc1->color = (RGBColorF){.r = 0.2, .b = 0.3, .g = 0.1};
    sc->color = (RGBColorF){.r = 0.9, .b = 0.9, .g = 0.9};

    c->even.tex = sc1;
    c->even.texType = SOLID_COLOR;
    c->odd.tex = sc;
    c->odd.texType = SOLID_COLOR;

    materialGround->lambTexture.tex = c;
    materialGround->lambTexture.texType = CHECKER;

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = 0, .y = -1000, .z = 0},
                                .radius = 1000,
                                .sphMat = MAT_CREATE_LAMB_IP(materialGround)});

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            CFLOAT chooseMat = randomFloat();
            vec3 center = {.x = a + 0.9 * randomFloat(),
                           .y = 0.2,
                           .z = b + 0.9 * randomFloat()};

            CFLOAT length = sqrtf((center.x - 4) * (center.x - 4) +
                                  (center.y - 0.2) * (center.y - 0.2) +
                                  (center.z - 0) * (center.z - 0));

            if (length > 0.9) {
                if (chooseMat < 0.8) {
                    // diffuse
                    RGBColorF albedo = {
                        .r = randomFloat() * randomFloat(),
                        .g = randomFloat() * randomFloat(),
                        .b = randomFloat() * randomFloat(),
                    };

                    LambertianMat *lambMat = alloc_dynamicStackAllocAllocate(
                        dsa, sizeof(LambertianMat), alignof(LambertianMat));

                    SolidColor *sc = alloc_dynamicStackAllocAllocate(
                        dsa, sizeof(SolidColor), alignof(SolidColor));

                    sc->color = albedo;

                    lambMat->lambTexture.tex = sc;
                    lambMat->lambTexture.texType = SOLID_COLOR;

                    obj_objLLAddSphere(
                        world, (Sphere){.center = center,
                                        .radius = 0.2,
                                        .sphMat = MAT_CREATE_LAMB_IP(lambMat)});

                } else if (chooseMat < 0.95) {
                    // metal
                    RGBColorF albedo = {.r = util_randomFloat(0.5, 1.0),
                                        .g = util_randomFloat(0.5, 1.0),
                                        .b = util_randomFloat(0.5, 1.0)};
                    CFLOAT fuzz = util_randomFloat(0.5, 1.0);

                    MetalMat *metalMat = alloc_dynamicStackAllocAllocate(
                        dsa, sizeof(MetalMat), alignof(MetalMat));

                    metalMat->albedo = albedo;
                    metalMat->fuzz = fuzz;

                    obj_objLLAddSphere(
                        world,
                        (Sphere){.center = center,
                                 .radius = 0.2,
                                 .sphMat = MAT_CREATE_METAL_IP(metalMat)});

                } else {
                    DielectricMat *dMat = alloc_dynamicStackAllocAllocate(
                        dsa, sizeof(DielectricMat), alignof(DielectricMat));
                    dMat->ir = 1.5;
                    obj_objLLAddSphere(
                        world,
                        (Sphere){.center = center,
                                 .radius = 0.2,
                                 .sphMat = MAT_CREATE_DIELECTRIC_IP(dMat)});
                }
            }
        }
    }

    DielectricMat *material1 = alloc_dynamicStackAllocAllocate(
        dsa, sizeof(DielectricMat), alignof(DielectricMat));
    material1->ir = 1.5;

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = 0, .y = 1, .z = 0},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_DIELECTRIC_IP(material1)});

    LambertianMat *material2 = alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));

    sc = alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                         alignof(SolidColor));

    sc->color = (RGBColorF){.r = 0.4, .g = 0.2, .b = 0.1};
    material2->lambTexture.tex = sc;
    material2->lambTexture.texType = SOLID_COLOR;
    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = 0},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});

    MetalMat *material3 = alloc_dynamicStackAllocAllocate(dsa, sizeof(MetalMat),
                                                          alignof(MetalMat));
    material3->albedo.r = 0.7;
    material3->albedo.g = 0.6;
    material3->albedo.b = 0.5;
    material3->fuzz = 0.0;

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = 4, .y = 1, .z = 0},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_METAL_IP(material3)});
}

CFLOAT lcg(int *n) {c

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

void randomSpheres2(ObjectLL *world, DynamicStackAlloc *dsa, int n,
                    Image imgs[n], int *seed) {

    LambertianMat *materialGround = alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));
    SolidColor *sc1 = alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                      alignof(SolidColor));

    SolidColor *sc = alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                     alignof(SolidColor));

    Checker *c =
        alloc_dynamicStackAllocAllocate(dsa, sizeof(Checker), alignof(Checker));

    sc1->color = (RGBColorF){.r = 0.0, .b = 0.0, .g = 0.0};
    sc->color = (RGBColorF){.r = 0.4, .b = 0.4, .g = 0.4};

    c->even.tex = sc1;
    c->even.texType = SOLID_COLOR;
    c->odd.tex = sc;
    c->odd.texType = SOLID_COLOR;

    materialGround->lambTexture.tex = c;
    materialGround->lambTexture.texType = CHECKER;
}

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = 0, .y = -1000, .z = 0},
                                .radius = 1000,
                                .sphMat = MAT_CREATE_LAMB_IP(materialGround)});
    

    for (int a = -2; a < 9; a++) {
        for (int b = -9; b < 9; b++) {
            CFLOAT chooseMat = lcg(seed);
            vec3 center = {
                .x = a + 0.9 * lcg(seed), .y = 0.2, .z = b + 0.9 * lcg(seed)};

            if (chooseMat < 0.8) {
                // diffuse
                RGBColorF albedo = {
                    .r = lcg(seed) * lcg(seed),
                    .g = lcg(seed) * lcg(seed),
                    .b = lcg(seed) * lcg(seed),

                };

                LambertianMat *lambMat = alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(LambertianMat), alignof(LambertianMat));

                SolidColor *sc = alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(SolidColor), alignof(SolidColor));

                sc->color = albedo;

                lambMat->lambTexture.tex = sc;
                lambMat->lambTexture.texType = SOLID_COLOR;

                obj_objLLAddSphere(
                    world, (Sphere){.center = center,
                                    .radius = 0.2,
                                    .sphMat = MAT_CREATE_LAMB_IP(lambMat)});

            } else if (chooseMat < 0.95) {
                // metal
                RGBColorF albedo = {.r = lcg(seed) / 2 + 0.5,
                                    .g = lcg(seed) / 2 + 0.5,
                                    .b = lcg(seed) / 2 + 0.5};
                CFLOAT fuzz = lcg(seed) / 2 + 0.5;

                MetalMat *metalMat = alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(MetalMat), alignof(MetalMat));

                metalMat->albedo = albedo;
                metalMat->fuzz = fuzz;

                obj_objLLAddSphere(
                    world, (Sphere){.center = center,
                                    .radius = 0.2,
                                    .sphMat = MAT_CREATE_METAL_IP(metalMat)});

            } else {
                DielectricMat *dMat = alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(DielectricMat), alignof(DielectricMat));
                dMat->ir = 1.5;
                obj_objLLAddSphere(
                    world, (Sphere){.center = center,
                                    .radius = 0.2,
                                    .sphMat = MAT_CREATE_DIELECTRIC_IP(dMat)});
            }
        }
    }

    LambertianMat *material2 = alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[0];
    material2->lambTexture.texType = IMAGE;
    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = 0},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});

    material2 = alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[1];
    material2->lambTexture.texType = IMAGE;
    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = -2.2},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});

    material2 = alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[2];
    material2->lambTexture.texType = IMAGE;
    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = +2.2},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});

    material2 = alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[3];
    material2->lambTexture.texType = IMAGE;
    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = -4.2},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});
}
#undef randomFloat

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

void addCudaSphere(
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

void randomSpheres2_raytracer(DeviceSphere *h_spheres,
    DeviceMaterial *h_materials,
    int *numSpheres,
    int *numMaterials,
    int maxSpheres,
    int maxMaterials,
    int *seed) {

    DeviceMaterial groundMat;

    groundMat.type = MAT_LAMBERTIAN;
    groundMat.albedo = (RGBColorF){.r = 0.4, .g = 0.4, .b = 0.4};
    groundMat.fuzz = 0.0;
    groundMat.ir = 1.0;

    addCudaSphere(
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
                // diffuse
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

                addCudaSphere(
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
                // metal
                DeviceMaterial mat;

                mat.type = MAT_METAL;
                mat.albedo = albedo;
                mat.fuzz = fuzz;
                mat.ir = 1.0;

                addCudaSphere(
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

                addCudaSphere(
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

    DeviceMaterial mat;

    mat.type = MAT_LAMBERTIAN;
    mat.albedo = (RGBColorF){.r = 0.4, .g = 0.2, .b = 0.1};
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    addCudaSphere(
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

    addCudaSphere(
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

    addCudaSphere(
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

    addCudaSphere(
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

__global__ void cuRaytracer_base(int HEIGHT, int WIDTH) {
    for (int l = 0; l < WIDTH * HEIGHT; l++) {
        int j = (HEIGHT - 1) - l / WIDTH;
        int i = l % WIDTH;
        CFLOAT pcR, pcG, pcB;
        pcR = pcG = pcB = 0.0;

        for (int k = 0; k < SAMPLES_PER_PIXEL; k++) {
            CFLOAT u =
                ((CFLOAT)i + util_randomFloat(0.0, 1.0)) / (WIDTH - 1);
            CFLOAT v =
                ((CFLOAT)j + util_randomFloat(0.0, 1.0)) / (HEIGHT - 1);
            r = cam_getRay(&dC, u, v);

            temp = ray_c(r, world, MAX_DEPTH);

            pcR += temp.r;
            pcG += temp.g;
            pcB += temp.b;

            alloc_linearAllocFCFreeAll(lafc);
        }

        dImage[i + WIDTH * (HEIGHT - 1 - j)] =
            writeColor(pcR, pcG, pcB, SAMPLES_PER_PIXEL);

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

 # OPENMP PARALLELIZATION   

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

# CUDA PARALLELIZATION
    cudaMalloc(&dImage, sizeof(RGBColorF) * HEIGHT * WIDTH);
    cudaMemcpy(dImage, hImage, sizeof(RGBColorF) * HEIGHT * WIDTH, cudaMemcpyHostToDevice);

    
    cudaMalloc(&dC, sizeof(Camera));
    cudaMemcpy(dC, &hC, sizeof(Camera), cudaMemcpyHostToDevice);

    DeviceSphere *h_spheres =
        malloc(sizeof(DeviceSphere) * MAX_SPHERES);

    DeviceMaterial *h_materials =
        malloc(sizeof(DeviceMaterial) * MAX_MATERIALS);

    int numSpheres = 0;
    int numMaterials = 0;

    cudaMalloc(&d_spheres, sizeof(DeviceSphere) * world->numObjects);
    cudaMalloc(&d_materials, sizeof(DeviceMaterial) * world->numObjects);   
    cudaMemcpy(d_spheres, h_spheres, sizeof(DeviceSphere) * world->numObjects, cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, h_materials, sizeof(DeviceMaterial) * world->numObjects, cudaMemcpyHostToDevice);

    randomSpheres2_cuda(
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

    writeToPPM(argv[1], WIDTH, HEIGHT, hImage);
    writeToPPM(argv[1], WIDTH, HEIGHT, dImage);

    free(image);
    free(hImage);
    cudaFree(dImage);
    cudaFree(dC);

    return 0;
}
