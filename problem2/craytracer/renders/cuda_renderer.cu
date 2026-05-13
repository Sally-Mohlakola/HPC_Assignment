#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "camera.h"
#include "color.h"
#include "outfile.h"
#include "texture.h"
}

#include "cuda_helper.cuh"
#include "metrics.h"

#define MAX_TEXTURES 512
#define MAX_IMAGE_TEXTURES 16

__constant__ DeviceSphere dcSpheres[MAX_SPHERES];

static float cudaElapsedSeconds(cudaEvent_t start, cudaEvent_t stop)
{
    float milliseconds = 0.0f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds / 1000.0f;
}

static void printKernelStats(
    const char *name,
    int numPixels,
    double openmpSeconds,
    float kernelSeconds,
    float memSeconds,
    const char *outputPath,
    const MetricsRunConfig *metricsConfig
) {
    MetricsTiming timing = metrics_make_timing(
        numPixels,
        openmpSeconds,
        (double)kernelSeconds,
        (double)memSeconds
    );

    printf("\n============= %s =============\n", name);
    printf("Kernel time:                  %8.3f s\n", kernelSeconds);
    printf("Memory transfer time:         %8.3f s\n", memSeconds);
    printf("Total (without mem):          %8.3f s\n", timing.totalWithoutMemSeconds);
    printf("Total (with mem):             %8.3f s\n", timing.totalWithMemSeconds);
    printf("Throughput:                   %8.2f Mpixels/sec\n", timing.throughputMpixelsSec);
    printf("Speedup vs OpenMP (kernel):   %8.2fx\n", timing.speedupVsOpenmpKernel);
    printf("Speedup vs OpenMP (with mem): %8.2fx\n", timing.speedupVsOpenmpWithMem);
    printf("Wrote '%s'\n", outputPath);
    fflush(stdout);

    if (metricsConfig != NULL) {
        metrics_append_summary_row(metricsConfig, name, &timing);
    }
}

// =====================================================================
// GLOBAL
// =====================================================================

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

// =====================================================================

// =====================================================================
// CONSTANT
// =====================================================================
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
// =====================================================================

// =====================================================================
// SHARED
// =====================================================================
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
// =====================================================================

// =====================================================================
// 1D TEXTURE
// =====================================================================
__global__ void cuRaytracer1DTexture(
    RGBColorU8 *dImage,
    Camera dCamera,
    const DeviceSphere *dSpheres,
    const DeviceMaterial *dMaterials,
    const DeviceTexture *dTextures,
    const DeviceImage1DTexture *dImageTextures,
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

        RGBColorF temp = cuRayC1D(
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
// =====================================================================
// 2D TEXTURE
// =====================================================================
__global__ void cuRaytracer2DTexture(
    RGBColorU8 *dImage,
    Camera dCamera,
    const DeviceSphere *dSpheres,
    const DeviceMaterial *dMaterials,
    const DeviceTexture *dTextures,
    const DeviceImage2DTexture *dImageTextures,
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

        RGBColorF temp = cuRayC2D(
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
// =====================================================================

// =====================================================================
// 2D TEXTURE + CONSTANT
// =====================================================================
__global__ void cuRaytracer2DTextureConstant(
    RGBColorU8 *dImage,
    Camera dCamera,
    const DeviceMaterial *dMaterials,
    const DeviceTexture *dTextures,
    const DeviceImage2DTexture *dImageTextures,
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

        RGBColorF temp = cuRayC2D(
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
// =====================================================================

// =====================================================================
// 2D TEXTURE + FRESNEL (REALISTIC)
// =====================================================================
__global__ void cuRaytracerRealistic(
    RGBColorU8 *dImage,
    Camera dCamera,
    const DeviceSphere *dSpheres,
    const DeviceMaterial *dMaterials,
    const DeviceTexture *dTextures,
    const DeviceImage2DTexture *dImageTextures,
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

        RGBColorF temp = cuRayC2DRealistic(
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
        cuWriteColorRealistic(pcR, pcG, pcB, samplesPerPixel);
}
// =====================================================================

extern "C" void renderCuda(
    RGBColorU8 *image,
    int width,
    int height,
    int samplesPerPixel,
    int maxDepth,
    int blockSize,
    int numSpheresRequest,
    Camera camera,
    Image images[4],
    unsigned int seed,
    double openmpSeconds,
    const MetricsRunConfig *metricsConfig
) {

    RGBColorU8 *dImage;
    cudaMalloc(&dImage, sizeof(RGBColorU8) * height * width);

    cudaEvent_t cudaStart;
    cudaEvent_t cudaStop;

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
    DeviceImage1DTexture *h_image1DTextures =
        (DeviceImage1DTexture *)malloc(sizeof(DeviceImage1DTexture) * MAX_IMAGE_TEXTURES);
    DeviceImage2DTexture *h_image2DTextures =
    (DeviceImage2DTexture *)malloc(sizeof(DeviceImage2DTexture) * MAX_IMAGE_TEXTURES);


    int numSpheres = 0;
    int numMaterials = 0;
    int numTextures = 0;
    int numImageTextures = 4;
    int imageTextureIndices[4];
    cudaArray_t h_imageArrays[MAX_IMAGE_TEXTURES] = {0};

    DeviceImage1DTexture *d_image1DTextures;
    cudaMalloc(&d_image1DTextures, sizeof(DeviceImage1DTexture) * numImageTextures);

    DeviceImage2DTexture *d_image2DTextures;
    cudaMalloc(&d_image2DTextures, sizeof(DeviceImage2DTexture) * numImageTextures);

    DeviceImageTexture *d_imageTextures;
    cudaMalloc(&d_imageTextures, sizeof(DeviceImageTexture) * numImageTextures);

    // Base image textures (raw pixel arrays for GLOBAL/CONSTANT/SHARED kernels)
    cudaEventRecord(cudaStart);
    for (int t = 0; t < numImageTextures; t++) {
        cuCreateImageTextureObject(&images[t], &h_imageTextures[t]);
    }
    cudaMemcpy(d_imageTextures, h_imageTextures,
               sizeof(DeviceImageTexture) * numImageTextures,
               cudaMemcpyHostToDevice);
    cudaEventRecord(cudaStop);
    float t_imageTexBase = cudaElapsedSeconds(cudaStart, cudaStop);

    // 1D texture objects (for 1D TEXTURE kernel)
    cudaEventRecord(cudaStart);
    for (int t = 0; t < numImageTextures; t++) {
        cuCreate1DImageTextureObject(&images[t], &h_image1DTextures[t]);
    }
    cudaMemcpy(d_image1DTextures, h_image1DTextures,
               sizeof(DeviceImage1DTexture) * numImageTextures,
               cudaMemcpyHostToDevice);
    cudaEventRecord(cudaStop);
    float t_image1DTex = cudaElapsedSeconds(cudaStart, cudaStop);

    // 2D texture objects (for 2D TEXTURE/REALISTIC kernels)
    cudaEventRecord(cudaStart);
    for (int t = 0; t < numImageTextures; t++) {
        cuCreate2DImageTextureObject(&images[t], &h_image2DTextures[t], &h_imageArrays[t]);
    }
    cudaMemcpy(d_image2DTextures, h_image2DTextures,
               sizeof(DeviceImage2DTexture) * numImageTextures,
               cudaMemcpyHostToDevice);
    cudaEventRecord(cudaStop);
    float t_image2DTex = cudaElapsedSeconds(cudaStart, cudaStop);

    for (int t = 0; t < numImageTextures; t++) {
        imageTextureIndices[t] = cuAddImageTexture(
            h_textures,
            &numTextures,
            MAX_TEXTURES,
            t
        );
    }

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
        numSpheresRequest,
        &sceneSeed
    );

    printf("\n");
    printf("Spheres generated: %d (requested %d)\n",
           numSpheres,
           numSpheresRequest > 0 ? numSpheresRequest : numSpheres);

    MetricsRunConfig cudaMetricsConfig;
    if (metricsConfig != NULL) {
        cudaMetricsConfig = *metricsConfig;
        cudaMetricsConfig.numSpheres = numSpheres;
        cudaMetricsConfig.openmpSeconds = openmpSeconds;
    }
    const MetricsRunConfig *cudaMetricsConfigPtr =
        (metricsConfig != NULL) ? &cudaMetricsConfig : NULL;

    DeviceSphere *d_spheres;
    DeviceMaterial *d_materials;
    DeviceTexture *d_textures;

    cudaMalloc(&d_spheres, sizeof(DeviceSphere) * numSpheres);
    cudaMalloc(&d_materials, sizeof(DeviceMaterial) * numMaterials);
    cudaMalloc(&d_textures, sizeof(DeviceTexture) * numTextures);

    cudaEventRecord(cudaStart);
    cudaMemcpy(d_spheres, h_spheres,
               sizeof(DeviceSphere) * numSpheres, cudaMemcpyHostToDevice);
    cudaEventRecord(cudaStop);
    float t_spheresHtoD = cudaElapsedSeconds(cudaStart, cudaStop);

    cudaEventRecord(cudaStart);
    cudaMemcpy(d_materials, h_materials,
               sizeof(DeviceMaterial) * numMaterials, cudaMemcpyHostToDevice);
    cudaEventRecord(cudaStop);
    float t_materialsHtoD = cudaElapsedSeconds(cudaStart, cudaStop);

    cudaEventRecord(cudaStart);
    cudaMemcpy(d_textures, h_textures,
               sizeof(DeviceTexture) * numTextures, cudaMemcpyHostToDevice);
    cudaEventRecord(cudaStop);
    float t_texturesHtoD = cudaElapsedSeconds(cudaStart, cudaStop);

    cudaEventRecord(cudaStart);
    cudaMemcpyToSymbol(dcSpheres, h_spheres, sizeof(DeviceSphere) * numSpheres);
    cudaEventRecord(cudaStop);
    float t_spheresConstHtoD = cudaElapsedSeconds(cudaStart, cudaStop);

    int threadsPerBlock = (blockSize > 0) ? blockSize : 256;
    int numPixels = width * height;
    int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

    float kernelSeconds = 0.0f;
    float dtohSeconds = 0.0f;
    size_t imageBytes = sizeof(RGBColorU8) * width * height;

// =====================================================================
// GLOBAL MEMORY KERNEL LAUNCH
// =====================================================================
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
    kernelSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(cudaStart);
    cudaMemcpy(image, dImage, imageBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop);
    dtohSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    writeToPPM("output/cuda_global.jpg", width, height, image);

    printKernelStats(
        "GLOBAL",
        numPixels,
        openmpSeconds,
        kernelSeconds,
        t_spheresHtoD + t_materialsHtoD + t_texturesHtoD + t_imageTexBase + dtohSeconds,
        "output/cuda_global.jpg",
        cudaMetricsConfigPtr
    );

// =====================================================================

// =====================================================================
// CONSTANT MEMORY KERNEL LAUNCH
// =====================================================================
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
    kernelSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(cudaStart);
    cudaMemcpy(image, dImage, imageBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop);
    dtohSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    writeToPPM("output/cuda_constant.jpg", width, height, image);

    printKernelStats(
        "CONSTANT",
        numPixels,
        openmpSeconds,
        kernelSeconds,
        t_spheresConstHtoD + t_materialsHtoD + t_texturesHtoD + t_imageTexBase + dtohSeconds,
        "output/cuda_constant.jpg",
        cudaMetricsConfigPtr
    );

// =====================================================================

// =====================================================================
// SHARED MEMORY KERNEL LAUNCH
// =====================================================================
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
    kernelSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(cudaStart);
    cudaMemcpy(image, dImage, imageBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop);
    dtohSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    writeToPPM("output/cuda_shared.jpg", width, height, image);

    printKernelStats(
        "SHARED",
        numPixels,
        openmpSeconds,
        kernelSeconds,
        t_spheresConstHtoD + t_materialsHtoD + t_texturesHtoD + t_imageTexBase + dtohSeconds,
        "output/cuda_shared.jpg",
        cudaMetricsConfigPtr
    );
// =====================================================================

// =====================================================================
// 1D TEXTURE MEMORY KERNEL LAUNCH
// =====================================================================
    cudaEventRecord(cudaStart);
    cuRaytracer1DTexture<<<blocks, threadsPerBlock>>>(
        dImage,
        camera,
        d_spheres,
        d_materials,
        d_textures,
        d_image1DTextures,
        numSpheres,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        seed
    );
    cudaEventRecord(cudaStop);
    kernelSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(cudaStart);
    cudaMemcpy(image, dImage, imageBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop);
    dtohSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    writeToPPM("output/cuda_1d_texture.jpg", width, height, image);

    printKernelStats(
        "1D TEXTURE",
        numPixels,
        openmpSeconds,
        kernelSeconds,
        t_spheresHtoD + t_materialsHtoD + t_texturesHtoD + t_image1DTex + dtohSeconds,
        "output/cuda_1d_texture.jpg",
        cudaMetricsConfigPtr
    );

// =====================================================================

// =====================================================================
// 2D TEXTURE MEMORY KERNEL LAUNCH
// =====================================================================
    cudaEventRecord(cudaStart);
    cuRaytracer2DTexture<<<blocks, threadsPerBlock>>>(
        dImage,
        camera,
        d_spheres,
        d_materials,
        d_textures,
        d_image2DTextures,
        numSpheres,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        seed
    );
    cudaEventRecord(cudaStop);
    kernelSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(cudaStart);
    cudaMemcpy(image, dImage, imageBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop);
    dtohSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    writeToPPM("output/cuda_2d_texture.jpg", width, height, image);

    printKernelStats(
        "2D TEXTURE",
        numPixels,
        openmpSeconds,
        kernelSeconds,
        t_spheresHtoD + t_materialsHtoD + t_texturesHtoD + t_image2DTex + dtohSeconds,
        "output/cuda_2d_texture.jpg",
        cudaMetricsConfigPtr
    );

// =====================================================================
// 2D TEXTURE MEMORY + CONSTANT MEMORY KERNEL LAUNCH
// =====================================================================
    cudaEventRecord(cudaStart);
    cuRaytracer2DTextureConstant<<<blocks, threadsPerBlock>>>(
        dImage,
        camera,
        d_materials,
        d_textures,
        d_image2DTextures,
        numSpheres,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        seed
    );
    cudaEventRecord(cudaStop);
    kernelSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(cudaStart);
    cudaMemcpy(image, dImage, imageBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop);
    dtohSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    writeToPPM("output/cuda_2d_texture_constant.jpg", width, height, image);

    printKernelStats(
        "2D TEXTURE + CONSTANT",
        numPixels,
        openmpSeconds,
        kernelSeconds,
        t_spheresConstHtoD + t_materialsHtoD + t_texturesHtoD + t_image2DTex + dtohSeconds,
        "output/cuda_2d_texture_constant.jpg",
        cudaMetricsConfigPtr
    );

// =====================================================================

// =====================================================================
// REALISTIC (2D TEXTURE + FRESNEL METAL + APERTURE + TONE MAPPING) KERNEL LAUNCH
// =====================================================================
    Camera realisticCamera = camera;
    realisticCamera.lensRadius = 0.03f / 2.0f;

    cudaEventRecord(cudaStart);
    cuRaytracerRealistic<<<blocks, threadsPerBlock>>>(
        dImage,
        realisticCamera,
        d_spheres,
        d_materials,
        d_textures,
        d_image2DTextures,
        numSpheres,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        seed
    );
    cudaEventRecord(cudaStop);
    kernelSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(cudaStart);
    cudaMemcpy(image, dImage, imageBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop);
    dtohSeconds = cudaElapsedSeconds(cudaStart, cudaStop);

    writeToPPM("output/cuda_realistic.jpg", width, height, image);

    printKernelStats(
        "REALISTIC",
        numPixels,
        openmpSeconds,
        kernelSeconds,
        t_spheresHtoD + t_materialsHtoD + t_texturesHtoD + t_image2DTex + dtohSeconds,
        "output/cuda_realistic.jpg",
        cudaMetricsConfigPtr
    );

// =====================================================================

    for (int t = 0; t < numImageTextures; t++) {
        cudaFree((void *)h_imageTextures[t].pixels);
    }

    for (int t = 0; t < numImageTextures; t++) {
        cudaDestroyTextureObject(h_image1DTextures[t].texObj);
        cudaFree((void *)h_image1DTextures[t].pixels);
    }

    for (int t = 0; t < numImageTextures; t++) {
        cudaDestroyTextureObject(h_image2DTextures[t].texObj);
        cudaFreeArray(h_imageArrays[t]);
    }

    free(h_spheres);
    free(h_materials);
    free(h_textures);
    free(h_imageTextures);
    free(h_image1DTextures);

    cudaFree(dImage);
    cudaFree(d_spheres);
    cudaFree(d_materials);
    cudaFree(d_textures);
    cudaFree(d_imageTextures);
    cudaFree(d_image1DTextures);

    cudaEventDestroy(cudaStart);
    cudaEventDestroy(cudaStop);
    free(h_image2DTextures);
    cudaFree(d_image2DTextures);


}
