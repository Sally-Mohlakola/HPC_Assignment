#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

#include <cuda_runtime.h>

#include "camera.h"
#include "color.h"
#include "hypatiaINC.h"
#include "ray.h"
#include "texture.h"
#include "types.h"
#include "util.h"

#define MAX_SPHERES   512
#define MAX_MATERIALS 512

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    vec3 center;
    CFLOAT radius;
    int materialIndex;
} DeviceSphere;

enum TextureType {
    TEX_SOLID = 0,
    TEX_CHECKER = 1,
    TEX_IMAGE = 2
};

enum DeviceMaterialType {
    MAT_LAMBERTIAN = 0,
    MAT_METAL = 1,
    MAT_DIELECTRIC = 2
};

typedef struct {
    int type;

    RGBColorF solidColor;

    RGBColorF checkerEven;
    RGBColorF checkerOdd;
    float checkerScale;

    int imageIndex;
} DeviceTexture;

typedef struct {
    cudaTextureObject_t texObj;
    int width;
    int height;
} DeviceImageTexture;

typedef struct {
    int type;

    int albedoTextureIndex;

    RGBColorF albedo;
    float fuzz;
    float ir;
} DeviceMaterial;

typedef struct {
    vec3 point;
    vec3 normal;

    CFLOAT distanceFromOrigin;

    int materialIndex;

    CFLOAT u;
    CFLOAT v;

    bool frontFace;
    bool valid;
} DeviceHitRecord;

#ifdef __cplusplus
extern "C" {
#endif
CFLOAT lcg(int *n);
#ifdef __cplusplus
}
#endif

static __host__ __device__ vec3 cuVec3(CFLOAT x, CFLOAT y, CFLOAT z)
{
    vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

__device__ CFLOAT cuVector3Dot(vec3 a, vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ vec3 *cuVector3Normalize(vec3 *v)
{
    CFLOAT length = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);

    if (length == 0.0) {
        return v;
    }

    v->x /= length;
    v->y /= length;
    v->z /= length;

    return v;
}

__device__ vec3 *cuVector3Add(vec3 *self, const vec3 *other)
{
    self->x += other->x;
    self->y += other->y;
    self->z += other->z;

    return self;
}

__device__ vec3 *cuVector3Subtract(vec3 *self, const vec3 *other)
{
    self->x -= other->x;
    self->y -= other->y;
    self->z -= other->z;

    return self;
}

__device__ vec3 *cuVector3MultiplyScalar(vec3 *self, CFLOAT scalar)
{
    self->x *= scalar;
    self->y *= scalar;
    self->z *= scalar;

    return self;
}

__device__ vec3 *cuVector3DivideScalar(vec3 *self, CFLOAT scalar)
{
    self->x /= scalar;
    self->y /= scalar;
    self->z /= scalar;

    return self;
}

__device__ RGBColorF cuColorfMultiply(RGBColorF x, RGBColorF y)
{
    return (RGBColorF){
        .r = x.r * y.r,
        .g = x.g * y.g,
        .b = x.b * y.b
    };
}

__device__ Ray cuRayCreate(vec3 origin, vec3 direction)
{
    cuVector3Normalize(&direction);

    return (Ray){
        .origin = origin,
        .direction = direction
    };
}

__device__ vec3 cuRayAt(Ray r, CFLOAT t)
{
    vec3 point = r.direction;
    cuVector3MultiplyScalar(&point, t);
    cuVector3Add(&point, &r.origin);

    return point;
}

__device__ RGBColorU8 cuWriteColor(CFLOAT r, CFLOAT g, CFLOAT b, int samplesPerPixel)
{
    CFLOAT scale = 1.0 / samplesPerPixel;

    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    return (RGBColorU8){
        .r = (uint8_t)(fmin(r * 256.0, 255.0)),
        .g = (uint8_t)(fmin(g * 256.0, 255.0)),
        .b = (uint8_t)(fmin(b * 256.0, 255.0))
    };
}

__device__ CFLOAT cuRand(unsigned int *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return (CFLOAT)(*state & 0x00FFFFFF) / (CFLOAT)0x01000000;
}

__device__ vec3 cuRandomUnitDisk(unsigned int *rngState)
{
    while (true) {
        vec3 p = cuVec3(
            2.0 * cuRand(rngState) - 1.0,
            2.0 * cuRand(rngState) - 1.0,
            0.0
        );

        if (cuVector3Dot(p, p) < 1.0) {
            return p;
        }
    }
}

__device__ RGBColorF cuSampleTexture(
    const DeviceTexture *textures,
    const DeviceImageTexture *imageTextures,
    int textureIndex,
    CFLOAT u,
    CFLOAT v,
    vec3 point
) {
    if (textureIndex < 0) {
        return (RGBColorF){1.0, 0.0, 1.0};
    }

    DeviceTexture tex = textures[textureIndex];

    if (tex.type == TEX_SOLID) {
        return tex.solidColor;
    }

    if (tex.type == TEX_CHECKER) {
        CFLOAT s = tex.checkerScale;

        CFLOAT value =
            sin(s * point.x) *
            sin(s * point.y) *
            sin(s * point.z);

        if (value < 0.0) {
            return tex.checkerOdd;
        }

        return tex.checkerEven;
    }

    if (tex.type == TEX_IMAGE) {
        DeviceImageTexture img = imageTextures[tex.imageIndex];

        u = fmin(fmax(u, 0.0), 1.0);
        v = fmin(fmax(v, 0.0), 1.0);

        v = 1.0 - v;

        float4 c = tex2D<float4>(img.texObj, (float)u, (float)v);

        return (RGBColorF){
            .r = c.x,
            .g = c.y,
            .b = c.z
        };
    }

    return (RGBColorF){1.0, 0.0, 1.0};
}

__device__ Ray cuCamGetRay(const Camera *cam, CFLOAT u, CFLOAT v, unsigned int *rngState)
{
    vec3 randOnDisk = cuRandomUnitDisk(rngState);
    cuVector3MultiplyScalar(&randOnDisk, cam->lensRadius);

    CFLOAT x = randOnDisk.x;
    CFLOAT y = randOnDisk.y;

    vec3 offset = cuVec3(
        x * cam->u.x + y * cam->v.x,
        x * cam->u.y + y * cam->v.y,
        x * cam->u.z + y * cam->v.z
    );

    vec3 origin = cam->origin;
    cuVector3Add(&origin, &offset);

    vec3 horizontal = cam->horizontal;
    cuVector3MultiplyScalar(&horizontal, u);

    vec3 vertical = cam->vertical;
    cuVector3MultiplyScalar(&vertical, v);

    vec3 direction = cam->lowerLeftCorner;
    cuVector3Add(&direction, &horizontal);
    cuVector3Add(&direction, &vertical);
    cuVector3Subtract(&direction, &cam->origin);
    cuVector3Subtract(&direction, &offset);

    return cuRayCreate(origin, direction);
}

__device__ vec3 cuRandomInUnitSphere(unsigned int *rngState)
{
    while (true) {
        vec3 p = cuVec3(
            2.0 * cuRand(rngState) - 1.0,
            2.0 * cuRand(rngState) - 1.0,
            2.0 * cuRand(rngState) - 1.0
        );

        if (cuVector3Dot(p, p) < 1.0) {
            return p;
        }
    }
}

__device__ vec3 cuRandomUnitVector(unsigned int *rngState)
{
    vec3 p = cuRandomInUnitSphere(rngState);
    cuVector3Normalize(&p);
    return p;
}

__device__ bool cuNearZero(vec3 v)
{
    const CFLOAT s = 1e-8;
    return fabs(v.x) < s && fabs(v.y) < s && fabs(v.z) < s; 
}

__device__ void cuGetSphereUV(vec3 outwardNormal, CFLOAT *u, CFLOAT *v)
{
    CFLOAT theta = acos(-outwardNormal.y);
    CFLOAT phi = atan2(-outwardNormal.z, outwardNormal.x) + M_PI;

    *u = phi / (2.0 * M_PI);
    *v = theta / M_PI;
}

__device__ void cuSetFaceNormal(DeviceHitRecord *rec, Ray r, vec3 outwardNormal)
{
    rec->frontFace = cuVector3Dot(r.direction, outwardNormal) < 0.0;

    if (rec->frontFace) {
        rec->normal = outwardNormal;
    } else {
        rec->normal = outwardNormal;
        cuVector3MultiplyScalar(&rec->normal, -1.0);
    }
}

__device__ bool cuHitSphere(
    const DeviceSphere *sphere,
    Ray r,
    CFLOAT tMin,
    CFLOAT tMax,
    DeviceHitRecord *rec
)
{
    vec3 oc = r.origin;
    cuVector3Subtract(&oc, &sphere->center);

    CFLOAT a = cuVector3Dot(r.direction, r.direction);
    CFLOAT halfB = cuVector3Dot(oc, r.direction);
    CFLOAT c = cuVector3Dot(oc, oc) - sphere->radius * sphere->radius;

    CFLOAT discriminant = halfB * halfB - a * c;

    if (discriminant < 0.0) {
        return false;
    }

    CFLOAT sqrtd = sqrt(discriminant);

    CFLOAT root = (-halfB - sqrtd) / a;

    if (root < tMin || root > tMax) {
        root = (-halfB + sqrtd) / a;

        if (root < tMin || root > tMax) {
            return false;
        }
    }

    rec->distanceFromOrigin = root;
    rec->point = cuRayAt(r, root);

    vec3 outwardNormal = rec->point;
    cuVector3Subtract(&outwardNormal, &sphere->center);
    cuVector3DivideScalar(&outwardNormal, sphere->radius);

    cuSetFaceNormal(rec, r, outwardNormal);
    cuGetSphereUV(outwardNormal, &rec->u, &rec->v);

    rec->materialIndex = sphere->materialIndex;
    rec->valid = true;

    return true;
}

__device__ bool cuHitWorld(
    const DeviceSphere *spheres,
    const DeviceMaterial *materials,
    int numSpheres,
    Ray r,
    CFLOAT tMin,
    CFLOAT tMax,
    DeviceHitRecord *rec
)
{
    DeviceHitRecord tempRec;
    bool hitAnything = false;
    CFLOAT closestSoFar = tMax;

    for (int s = 0; s < numSpheres; s++) {
        if (cuHitSphere(&spheres[s], r, tMin, closestSoFar, &tempRec)) {
            hitAnything = true;
            closestSoFar = tempRec.distanceFromOrigin;
            *rec = tempRec;
        }
    }

    return hitAnything;
}

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
    CFLOAT scale = 2.0 * cuVector3Dot(v, n);
    cuVector3MultiplyScalar(&result, scale);
    cuVector3Subtract(&v, &result);
    return v;
}

__device__ CFLOAT cuReflectance(CFLOAT cosine, CFLOAT refIdx) {
    CFLOAT r0 = (1.0 - refIdx) / (1.0 + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

__device__ vec3 cuRefract(vec3 uv, vec3 n, CFLOAT etaiOverEtat) {
    vec3 negUv = uv;
    cuVector3MultiplyScalar(&negUv, -1.0);

    CFLOAT cosTheta = fmin(cuVector3Dot(negUv, n), 1.0);

    vec3 rOutPerp = uv;

    vec3 temp = n;
    cuVector3MultiplyScalar(&temp, cosTheta);
    cuVector3Add(&rOutPerp, &temp);
    cuVector3MultiplyScalar(&rOutPerp, etaiOverEtat);

    CFLOAT lenSq =
        rOutPerp.x * rOutPerp.x +
        rOutPerp.y * rOutPerp.y +
        rOutPerp.z * rOutPerp.z;

    vec3 rOutParallel = n;
    cuVector3MultiplyScalar(&rOutParallel, -sqrt(fabs(1.0 - lenSq)));

    cuVector3Add(&rOutPerp, &rOutParallel);

    return rOutPerp;
}

__device__ bool cuScatterLambertian(
    Ray rayIn,
    DeviceHitRecord rec,
    DeviceMaterial mat,
    const DeviceTexture *textures,
    const DeviceImageTexture *imageTextures,
    RGBColorF *attenuation,
    Ray *scattered,
    unsigned int *rngState
) {
    vec3 scatterDirection = rec.normal;
    vec3 randomVec = cuRandomUnitVector(rngState);
    cuVector3Add(&scatterDirection, &randomVec);

    if (cuNearZero(scatterDirection)) {
        scatterDirection = rec.normal;
    }

    scattered->origin = rec.point;
    scattered->direction = scatterDirection;

    *attenuation = cuSampleTexture(
        textures,
        imageTextures,
        mat.albedoTextureIndex,
        rec.u,
        rec.v,
        rec.point
    );

    return true;
}

__device__ bool cuScatterMetal(
    Ray rayIn,
    DeviceHitRecord rec,
    DeviceMaterial mat,
    RGBColorF *attenuation,
    Ray *scattered,
    unsigned int *rngState
) {
    vec3 unitDirection = rayIn.direction;
    cuVector3Normalize(&unitDirection);

    vec3 reflected = cuReflect(unitDirection, rec.normal);

    vec3 fuzzVec = cuRandomUnitVector(rngState);
    cuVector3MultiplyScalar(&fuzzVec, mat.fuzz);
    cuVector3Add(&reflected, &fuzzVec);

    scattered->origin = rec.point;
    scattered->direction = reflected;

    *attenuation = mat.albedo;

    return cuVector3Dot(scattered->direction, rec.normal) > 0.0;
}

__device__ bool cuScatterDielectric(
    Ray rayIn,
    DeviceHitRecord rec,
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
    cuVector3Normalize(&unitDirection);

    vec3 negUnitDirection = unitDirection;
    cuVector3MultiplyScalar(&negUnitDirection, -1.0);

    CFLOAT cosTheta = fmin(cuVector3Dot(negUnitDirection, rec.normal), 1.0);
    CFLOAT sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    bool cannotRefract = refractionRatio * sinTheta > 1.0;

    vec3 direction;

    if (cannotRefract ||
        cuReflectance(cosTheta, refractionRatio) > cuRand(rngState)) {
        direction = cuReflect(unitDirection, rec.normal);
    } else {
        direction = cuRefract(unitDirection, rec.normal, refractionRatio);
    }

    scattered->origin = rec.point;
    scattered->direction = direction;

    return true;
}

int cuAddSolidTexture(
    DeviceTexture *h_textures,
    int *numTextures,
    int maxTextures,
    RGBColorF color
) {
    if (*numTextures >= maxTextures) {
        fprintf(stderr, "Texture array is full\n");
        exit(1);
    }

    int texIndex = *numTextures;

    h_textures[texIndex] = (DeviceTexture){
        .type = TEX_SOLID,
        .solidColor = color,
        .checkerEven = {0},
        .checkerOdd = {0},
        .checkerScale = 0.0,
        .imageIndex = -1
    };

    (*numTextures)++;

    return texIndex;
}

__device__ bool cuScatter(
    Ray rayIn,
    DeviceHitRecord rec,
    DeviceMaterial mat,
    const DeviceTexture *textures,
    const DeviceImageTexture *imageTextures,
    RGBColorF *attenuation,
    Ray *scattered,
    unsigned int *rngState
) {
    if (mat.type == MAT_LAMBERTIAN) {
        return cuScatterLambertian(
            rayIn,
            rec,
            mat,
            textures,
            imageTextures,
            attenuation,
            scattered,
            rngState
        );
    }

    if (mat.type == MAT_METAL) {
        return cuScatterMetal(rayIn, rec, mat, attenuation, scattered, rngState);
    }

    if (mat.type == MAT_DIELECTRIC) {
        return cuScatterDielectric(rayIn, rec, mat, attenuation, scattered, rngState);
    }

    return false;
}

int cuAddCheckerTexture(
    DeviceTexture *h_textures,
    int *numTextures,
    int maxTextures,
    RGBColorF even,
    RGBColorF odd,
    CFLOAT scale
) {
    if (*numTextures >= maxTextures) {
        fprintf(stderr, "Texture array is full\n");
        exit(1);
    }

    int texIndex = *numTextures;

    h_textures[texIndex] = (DeviceTexture){
        .type = TEX_CHECKER,
        .solidColor = {0},
        .checkerEven = even,
        .checkerOdd = odd,
        .checkerScale = (float)scale,
        .imageIndex = -1
    };

    (*numTextures)++;

    return texIndex;
}

int cuAddImageTexture(
    DeviceTexture *h_textures,
    int *numTextures,
    int maxTextures,
    int imageIndex
) {
    if (*numTextures >= maxTextures) {
        fprintf(stderr, "Texture array is full\n");
        exit(1);
    }

    int texIndex = *numTextures;

    h_textures[texIndex] = (DeviceTexture){
        .type = TEX_IMAGE,
        .solidColor = {0},
        .checkerEven = {0},
        .checkerOdd = {0},
        .checkerScale = 0.0,
        .imageIndex = imageIndex
    };

    (*numTextures)++;

    return texIndex;
}

__device__ RGBColorF cuRayC(
    Ray r,
    const DeviceSphere *spheres,
    const DeviceMaterial *materials,
    const DeviceTexture *textures,
    const DeviceImageTexture *imageTextures,
    int numSpheres,
    int maxDepth,
    unsigned int *rngState
) {
    RGBColorF finalColor = {1.0, 1.0, 1.0};

    for (int depth = 0; depth < maxDepth; depth++) {
        DeviceHitRecord rec;

        if (cuHitWorld(spheres, materials, numSpheres, r, 0.0001, FLT_MAX, &rec)) {
            Ray scattered;
            RGBColorF attenuation;

            if (cuScatter(
                    r,
                    rec,
                    materials[rec.materialIndex],
                    textures,
                    imageTextures,
                    &attenuation,
                    &scattered,
                    rngState
                )) {
                finalColor = cuColorfMultiply(finalColor, attenuation);
                r = scattered;
            } else {
                return (RGBColorF){0.0, 0.0, 0.0};
            }
        } else {
            vec3 unitDirection = r.direction;
            cuVector3Normalize(&unitDirection);

            CFLOAT t = 0.5 * (unitDirection.y + 1.0);

            RGBColorF background = {
                .r = (1.0 - t) * 1.0 + t * 0.5,
                .g = (1.0 - t) * 1.0 + t * 0.7,
                .b = (1.0 - t) * 1.0 + t * 1.0
            };

            return cuColorfMultiply(finalColor, background);
        }
    }

    return (RGBColorF){0.0, 0.0, 0.0};
}

void cuCreateImageTextureObject(
    const Image *image,
    DeviceImageTexture *outImageTexture,
    cudaArray_t *outArray
) {
    cudaArray_t cuArray;
    int width = image->width;
    int height = image->height;

    if (image->data == NULL || width <= 0 || height <= 0) {
        fprintf(stderr, "Cannot create CUDA texture from empty image\n");
        exit(1);
    }

    uchar4 *hostPixels = (uchar4 *)malloc(sizeof(uchar4) * width * height);

    if (hostPixels == NULL) {
        fprintf(stderr, "Failed to allocate CUDA texture staging pixels\n");
        exit(1);
    }

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            uint8_t *pixel =
                image->data +
                j * image->bytesPerScanLine +
                i * image->compsPerPixel;

            hostPixels[j * width + i] = make_uchar4(
                pixel[0],
                pixel[1],
                pixel[2],
                image->compsPerPixel > 3 ? pixel[3] : 255
            );
        }
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    cudaMallocArray(&cuArray, &channelDesc, width, height);

    cudaMemcpy2DToArray(
        cuArray,
        0,
        0,
        hostPixels,
        width * sizeof(uchar4),
        width * sizeof(uchar4),
        height,
        cudaMemcpyHostToDevice
    );

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;

    cudaCreateTextureObject(
        &texObj,
        &resDesc,
        &texDesc,
        NULL
    );

    outImageTexture->texObj = texObj;
    outImageTexture->width = width;
    outImageTexture->height = height;

    *outArray = cuArray;
    free(hostPixels);
}

void cuRandomSpheres2(
    DeviceSphere *h_spheres,
    DeviceMaterial *h_materials,
    DeviceTexture *h_textures,
    const int imageTextureIndices[4],
    int *numSpheres,
    int *numMaterials,
    int *numTextures,
    int maxSpheres,
    int maxMaterials,
    int maxTextures,
    int *seed
    ) {
    DeviceMaterial mat;

    int groundTex = cuAddCheckerTexture(
        h_textures,
        numTextures,
        maxTextures,
        (RGBColorF){.r = 0.0, .g = 0.0, .b = 0.0},
        (RGBColorF){.r = 0.4, .g = 0.4, .b = 0.4},
        10.0
    );

    DeviceMaterial groundMat;

    groundMat.type = MAT_LAMBERTIAN;
    groundMat.albedoTextureIndex = groundTex;
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
    cuVec3(0, -1000, 0),
    1000,
    groundMat
    );

    for (int a = -2; a < 9; a++) {
        for (int b = -9; b < 9; b++) {
            CFLOAT chooseMat = lcg(seed);
            vec3 center = cuVec3(
                a + 0.9 * lcg(seed),
                0.2,
                b + 0.9 * lcg(seed)
            );

            if (chooseMat < 0.8) {
                RGBColorF albedo = {
                    .r = lcg(seed) * lcg(seed),
                    .g = lcg(seed) * lcg(seed),
                    .b = lcg(seed) * lcg(seed),
                };

                int texIndex = cuAddSolidTexture(
                    h_textures,
                    numTextures,
                    maxTextures,
                    albedo
                );

                DeviceMaterial mat;

                mat.type = MAT_LAMBERTIAN;
                mat.albedo = albedo;
                mat.albedoTextureIndex = texIndex;
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
                RGBColorF albedo = {
                    .r = lcg(seed) / 2 + 0.5,
                    .g = lcg(seed) / 2 + 0.5,
                    .b = lcg(seed) / 2 + 0.5
                };
                CFLOAT fuzz = lcg(seed) / 2 + 0.5;
                DeviceMaterial mat;

                mat.type = MAT_METAL;
                mat.albedo = albedo;
                mat.albedoTextureIndex = -1;
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
                mat.albedoTextureIndex = -1;
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
    mat.albedo = (RGBColorF){.r = 1.0, .g = 1.0, .b = 1.0};
    mat.albedoTextureIndex = imageTextureIndices[0];
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    cuAddSphere(
        h_spheres,
        h_materials,
        numSpheres,
        numMaterials,
        maxSpheres,
        maxMaterials,
        cuVec3(-4, 1, 0),
        1.0,
        mat
    );

    mat.type = MAT_LAMBERTIAN;
    mat.albedo = (RGBColorF){.r = 1.0, .g = 1.0, .b = 1.0};
    mat.albedoTextureIndex = imageTextureIndices[1];
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    cuAddSphere(
        h_spheres,
        h_materials,
        numSpheres,
        numMaterials,
        maxSpheres,
        maxMaterials,
        cuVec3(-4, 1, -2.2),
        1.0,
        mat
    );

    mat.type = MAT_LAMBERTIAN;
    mat.albedo = (RGBColorF){.r = 1.0, .g = 1.0, .b = 1.0};
    mat.albedoTextureIndex = imageTextureIndices[2];
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    cuAddSphere(
        h_spheres,
        h_materials,
        numSpheres,
        numMaterials,
        maxSpheres,
        maxMaterials,
        cuVec3(-4, 1, 2.2),
        1.0,
        mat
    );

    mat.type = MAT_LAMBERTIAN;
    mat.albedo = (RGBColorF){.r = 1.0, .g = 1.0, .b = 1.0};
    mat.albedoTextureIndex = imageTextureIndices[3];
    mat.fuzz = 0.0;
    mat.ir = 1.0;

    cuAddSphere(
        h_spheres,
        h_materials,
        numSpheres,
        numMaterials,
        maxSpheres,
        maxMaterials,
        cuVec3(-4, 1, -4.2),
        1.0,
        mat
    );
}

#endif
