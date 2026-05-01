#define HYPATIA_IMPLEMENTATION
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <time.h>

#include "camera.h"
#include "color.h"
#include "hypatiaINC.h"
#include "openmp_helper.h"
#include "outfile.h"
#include "texture.h"
#include "types.h"

void renderCuda(
    RGBColorU8 *image,
    int width,
    int height,
    int samplesPerPixel,
    int maxDepth,
    Camera camera,
    Image images[4],
    unsigned int seed,
    unsigned int progressStep
);

static vec3 makeVec3(CFLOAT x, CFLOAT y, CFLOAT z)
{
    vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

CFLOAT lcg(int *n)
{
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

int main(int argc, char *argv[])
{

    srand(time(NULL));

    const CFLOAT aspectRatio = 16.0 / 9.0;
    const int width = 640;
    const int height = 640;
    const int samplesPerPixel = 100;
    const int maxDepth = 50;
    const unsigned int progressStep = 500;
    const int sceneSeed = 100;

    RGBColorU8 *openmpImage =
        (RGBColorU8 *)malloc(sizeof(RGBColorU8) * height * width);
    RGBColorU8 *cudaImage =
        (RGBColorU8 *)malloc(sizeof(RGBColorU8) * height * width);

    if (openmpImage == NULL || cudaImage == NULL) {
        printf("FATAL ERROR: Image allocation failed.\n");
        free(openmpImage);
        free(cudaImage);
        return 1;
    }

    vec3 lookFrom = makeVec3(13.0, 2.0, 3.0);
    vec3 lookAt = makeVec3(0.0, 0.0, 0.0);
    vec3 up = makeVec3(0.0, 1.0, 0.0);

    CFLOAT distToFocus = 10.0;
    CFLOAT aperture = 0.1;

    Camera camera;
    cam_setLookAtCamera(&camera, lookFrom, lookAt, up, 20, aspectRatio,
                        aperture, distToFocus);

    Image images[4];

    tex_loadImage(&images[0], "./test_textures/kitchen_probe.jpg");
    tex_loadImage(&images[1], "./test_textures/campus_probe.jpg");
    tex_loadImage(&images[2], "./test_textures/building_probe.jpg");
    tex_loadImage(&images[3], "./test_textures/kitchen_probe.jpg");

    printf("Scene initialized\n");

    CFLOAT openmpStart = omp_get_wtime();
    renderOpenMP(
        openmpImage,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        camera,
        images,
        sceneSeed,
        progressStep
    );
    CFLOAT openmpEnd = omp_get_wtime();
    printf("OpenMP execution time: %lf\n", openmpEnd - openmpStart);

    CFLOAT cudaStart = omp_get_wtime();
    renderCuda(
        cudaImage,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        camera,
        images,
        (unsigned int)sceneSeed,
        progressStep
    );
    CFLOAT cudaEnd = omp_get_wtime();
    printf("CUDA execution time: %lf\n", cudaEnd - cudaStart);

    writeToPPM("openmp.jpg", width, height, openmpImage);
    writeToPPM("cuda.jpg", width, height, cudaImage);

    free(openmpImage);
    free(cudaImage);

    return 0;
}
