#include <float.h>
#include <stdalign.h>
#include <stdbool.h>
#include <tgmath.h>

#include "allocator.h"
#include "color.h"
#include "hitRecord.h"
#include "hypatiaINC.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "texture.h"
#include "types.h"
#include "util.h"
#include "openmp_helper.h"

CFLOAT lcg(int *n);

RGBColorF ray_c(Ray r, const ObjectLL *world, int depth) {

    if (depth <= 0) {
        return (RGBColorF){0};
    }

    HitRecord rec;
    rec.valid = false;
    bool checkHit = obj_objLLHit(world, r, 0.00001, FLT_MAX, &rec);

    if (checkHit) {
        Ray scattered = {0};
        RGBColorF attenuation = {0};

        if (mat_scatter(&r, &rec, &attenuation, &scattered)) {
            RGBColorF color = ray_c(scattered, world, depth - 1);
            color = colorf_multiply(color, attenuation);

            return color;
        }

        return (RGBColorF){0};
    }

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
