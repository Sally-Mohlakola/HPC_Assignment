#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "camera.h"
#include "color.h"
#include "outfile.h"
#include "texture.h"
}

#include "cuda_helper.cuh"

#define MAX_TEXTURES 512
#define MAX_IMAGE_TEXTURES 16

__constant__ DeviceSphere dcSpheres[MAX_SPHERES];

__global__ void cuRaytracerBase(
    RGBColorU8 *dImage,
    Camera dCamera,
    const DeviceSphere *dSpheres,
    const DeviceMaterial *dMaterials,
    const DeviceTexture *dTextures,
    const DeviceImageTexture *dImageTextures,
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

    CFLOAT pcR = 0.0;
    CFLOAT pcG = 0.0;
    CFLOAT pcB = 0.0;

    for (int k = 0; k < samplesPerPixel; k++) {
        CFLOAT u =
            ((CFLOAT)i + cuRand(&rngState)) / (width - 1);
        CFLOAT v =
            ((CFLOAT)j + cuRand(&rngState)) / (height - 1);
        Ray r = cuCamGetRay(&dCamera, u, v, &rngState);

        RGBColorF temp = cuRayC(
            r,
            dSpheres,
            dMaterials,
            dTextures,
            dImageTextures,
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
}

__global__ void cuRaytracerConstant(
    RGBColorU8 *dImage,
    Camera dCamera,
    const DeviceMaterial *dMaterials,
    const DeviceTexture *dTextures,
    const DeviceImageTexture *dImageTextures,
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

    CFLOAT pcR = 0.0;
    CFLOAT pcG = 0.0;
    CFLOAT pcB = 0.0;

    for (int k = 0; k < samplesPerPixel; k++) {
        CFLOAT u =
            ((CFLOAT)i + cuRand(&rngState)) / (width - 1);
        CFLOAT v =
            ((CFLOAT)j + cuRand(&rngState)) / (height - 1);
        Ray r = cuCamGetRay(&dCamera, u, v, &rngState);

        RGBColorF temp = cuRayC(
            r,
            dcSpheres,
            dMaterials,
            dTextures,
            dImageTextures,
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
}

__global__ void cuRaytracerShared(
    RGBColorU8 *dImage,
    Camera dCamera,
    const DeviceMaterial *dMaterials,
    const DeviceTexture *dTextures,
    const DeviceImageTexture *dImageTextures,
    int numSpheres,
    int width,
    int height,
    int samplesPerPixel,
    int maxDepth,
    unsigned int seed
) {
    __shared__ DeviceSphere dsSpheres[MAX_SPHERES];

    for (int s = threadIdx.x; s < numSpheres; s += blockDim.x) {
        dsSpheres[s] = dcSpheres[s];
    }

    __syncthreads();

    int l = blockIdx.x * blockDim.x + threadIdx.x;

    if (l >= width * height) {
        return;
    }

    int j = (height - 1) - l / width;
    int i = l % width;

    unsigned int rngState = seed ^ l;

    CFLOAT pcR = 0.0;
    CFLOAT pcG = 0.0;
    CFLOAT pcB = 0.0;

    for (int k = 0; k < samplesPerPixel; k++) {
        CFLOAT u =
            ((CFLOAT)i + cuRand(&rngState)) / (width - 1);
        CFLOAT v =
            ((CFLOAT)j + cuRand(&rngState)) / (height - 1);
        Ray r = cuCamGetRay(&dCamera, u, v, &rngState);

        RGBColorF temp = cuRayC(
            r,
            dsSpheres,
            dMaterials,
            dTextures,
            dImageTextures,
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
}

extern "C" void renderCuda(
    RGBColorU8 *image,
    int width,
    int height,
    int samplesPerPixel,
    int maxDepth,
    Camera camera,
    Image images[4],
    unsigned int seed
) {
    RGBColorU8 *dImage;
    cudaMalloc(&dImage, sizeof(RGBColorU8) * height * width);

    cudaEvent_t cudaStart;
    cudaEvent_t cudaStop;
    float cudaElapsedMs = 0.0f;

    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);

    DeviceSphere *h_spheres =
        (DeviceSphere *)malloc(sizeof(DeviceSphere) * MAX_SPHERES);
    DeviceMaterial *h_materials =
        (DeviceMaterial *)malloc(sizeof(DeviceMaterial) * MAX_MATERIALS);
    DeviceTexture *h_textures =
        (DeviceTexture *)malloc(sizeof(DeviceTexture) * MAX_TEXTURES);
    DeviceImageTexture *h_imageTextures =
        (DeviceImageTexture *)malloc(sizeof(DeviceImageTexture) * MAX_IMAGE_TEXTURES);

    int numSpheres = 0;
    int numMaterials = 0;
    int numTextures = 0;
    int numImageTextures = 4;
    int imageTextureIndices[4];
    cudaArray_t h_imageArrays[MAX_IMAGE_TEXTURES] = {0};

    cudaEventRecord(cudaStart);

    for (int t = 0; t < numImageTextures; t++) {
        cuCreateImageTextureObject(&images[t], &h_imageTextures[t], &h_imageArrays[t]);
        imageTextureIndices[t] = cuAddImageTexture(
            h_textures,
            &numTextures,
            MAX_TEXTURES,
            t
        );
    }

    cudaEventRecord(cudaStop);
    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&cudaElapsedMs, cudaStart, cudaStop);
    printf("CUDA texture memory loading time: %f ms\n", cudaElapsedMs);

    int sceneSeed = (int)seed;

    cuRandomSpheres2(
        h_spheres,
        h_materials,
        h_textures,
        imageTextureIndices,
        &numSpheres,
        &numMaterials,
        &numTextures,
        MAX_SPHERES,
        MAX_MATERIALS,
        MAX_TEXTURES,
        &sceneSeed
    );

    DeviceSphere *d_spheres;
    DeviceMaterial *d_materials;
    DeviceTexture *d_textures;
    DeviceImageTexture *d_imageTextures;

    cudaMalloc(&d_spheres, sizeof(DeviceSphere) * numSpheres);
    cudaMalloc(&d_materials, sizeof(DeviceMaterial) * numMaterials);
    cudaMalloc(&d_textures, sizeof(DeviceTexture) * numTextures);
    cudaMalloc(&d_imageTextures, sizeof(DeviceImageTexture) * numImageTextures);

    cudaEventRecord(cudaStart);

    cudaMemcpy(d_spheres, h_spheres,
               sizeof(DeviceSphere) * numSpheres, cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, h_materials,
               sizeof(DeviceMaterial) * numMaterials, cudaMemcpyHostToDevice);
    cudaMemcpy(d_textures, h_textures,
               sizeof(DeviceTexture) * numTextures, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imageTextures, h_imageTextures,
               sizeof(DeviceImageTexture) * numImageTextures, cudaMemcpyHostToDevice);

    cudaEventRecord(cudaStop);
    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&cudaElapsedMs, cudaStart, cudaStop);
    printf("CUDA global memory loading time: %f ms\n", cudaElapsedMs);

    cudaEventRecord(cudaStart);

    cudaMemcpyToSymbol(dcSpheres, h_spheres, sizeof(DeviceSphere) * numSpheres);

    cudaEventRecord(cudaStop);
    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&cudaElapsedMs, cudaStart, cudaStop);
    printf("CUDA constant memory loading time: %f ms\n", cudaElapsedMs);
    
    int threadsPerBlock = 256;
    int numPixels = width * height;
    int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA global memory running\n");
    fflush(stdout);

    cudaEventRecord(cudaStart);

    cuRaytracerBase<<<blocks, threadsPerBlock>>>(
        dImage,
        camera,
        d_spheres,
        d_materials,
        d_textures,
        d_imageTextures,
        numSpheres,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        seed
    );

    cudaEventRecord(cudaStop);
    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&cudaElapsedMs, cudaStart, cudaStop);
    printf("CUDA global memory execution time: %f ms\n", cudaElapsedMs);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(image, dImage,
               sizeof(RGBColorU8) * width * height,
               cudaMemcpyDeviceToHost);

    writeToPPM("output/cuda_global.jpg", width, height, image);

    printf("CUDA constant memory running\n");
    fflush(stdout);

    cudaEventRecord(cudaStart);

    cuRaytracerConstant<<<blocks, threadsPerBlock>>>(
        dImage,
        camera,
        d_materials,
        d_textures,
        d_imageTextures,
        numSpheres,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        seed
    );

    cudaEventRecord(cudaStop);
    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&cudaElapsedMs, cudaStart, cudaStop);
    printf("CUDA constant memory execution time: %f ms\n", cudaElapsedMs);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(image, dImage,
                sizeof(RGBColorU8) * width * height,
                cudaMemcpyDeviceToHost);

    writeToPPM("output/cuda_constant.jpg", width, height, image);

     printf("CUDA shared memory running\n");
    fflush(stdout);

    cudaEventRecord(cudaStart);

    cuRaytracerShared<<<blocks, threadsPerBlock>>>(
        dImage,
        camera,
        d_materials,
        d_textures,
        d_imageTextures,
        numSpheres,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        seed
    );

    cudaEventRecord(cudaStop);
    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&cudaElapsedMs, cudaStart, cudaStop);
    printf("CUDA shared memory execution time: %f ms\n", cudaElapsedMs);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(image, dImage,
                sizeof(RGBColorU8) * width * height,
                cudaMemcpyDeviceToHost);

    writeToPPM("output/cuda_shared.jpg", width, height, image);

    for (int t = 0; t < numImageTextures; t++) {
        cudaDestroyTextureObject(h_imageTextures[t].texObj);
        cudaFreeArray(h_imageArrays[t]);
    }

    free(h_spheres);
    free(h_materials);
    free(h_textures);
    free(h_imageTextures);

    cudaFree(dImage);
    cudaFree(d_spheres);
    cudaFree(d_materials);
    cudaFree(d_textures);
    cudaFree(d_imageTextures);

    cudaEventDestroy(cudaStart);
    cudaEventDestroy(cudaStop);
}
