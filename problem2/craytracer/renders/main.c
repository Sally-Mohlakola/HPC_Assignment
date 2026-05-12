#define HYPATIA_IMPLEMENTATION
#include <omp.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
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
    int blockSize,
    int numSpheresRequest,
    Camera camera,
    Image images[4],
    unsigned int seed,
    double openmpMs
);

static vec3 makeVec3(CFLOAT x, CFLOAT y, CFLOAT z)
{
    vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

static int ensureOutputDirectory(void)
{
    struct stat st;

    if (mkdir("output", 0755) == 0) {
        return 0;
    }

    if (errno == EEXIST && stat("output", &st) == 0 && S_ISDIR(st.st_mode)) {
        return 0;
    }

    perror("Failed to create output directory");
    return 1;
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

    if (ensureOutputDirectory() != 0) {
        return 1;
    }

    const CFLOAT aspectRatio = 16.0 / 9.0;
    const int width = 640;
    const int height = 640;
    const int samplesPerPixel = 100;
    int maxDepth = 50;
    int blockSize = 256;
    int numSpheresRequest = 0;
    const int sceneSeed = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v > 0) blockSize = v;
        } else if (strcmp(argv[i], "--max-depth") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v > 0) maxDepth = v;
        } else if (strcmp(argv[i], "--num-spheres") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v > 0) numSpheresRequest = v;
        }
    }
    if (numSpheresRequest > 0) {
        printf("Config: blockSize=%d MAX_SPHERES=%d maxDepth=%d numSpheres=%d\n",
               blockSize, MAX_SPHERES, maxDepth, numSpheresRequest);
    } else {
        printf("Config: blockSize=%d MAX_SPHERES=%d maxDepth=%d numSpheres=full\n",
               blockSize, MAX_SPHERES, maxDepth);
    }

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
    fflush(stdout);

    double openmpStart = omp_get_wtime();
    renderOpenMP(
        openmpImage,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        camera,
        images,
        sceneSeed
    );
    double openmpEnd = omp_get_wtime();
    double openmpMs = (openmpEnd - openmpStart) * 1000.0;

    printf("\n============= OPENMP =============\n");
    printf("Time:                         %8.3f ms\n", openmpMs);
    printf("Wrote 'output/openmp.jpg'\n");
    fflush(stdout);

    renderCuda(
        cudaImage,
        width,
        height,
        samplesPerPixel,
        maxDepth,
        blockSize,
        numSpheresRequest,
        camera,
        images,
        (unsigned int)sceneSeed,
        openmpMs
    );

    free(openmpImage);
    free(cudaImage);

    return 0;
}
