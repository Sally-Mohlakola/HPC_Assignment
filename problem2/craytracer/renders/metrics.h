#ifndef METRICS_H
#define METRICS_H

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#define METRICS_DIR "metrics"
#define METRICS_SUMMARY_PATH "metrics/summary.csv"

typedef struct {
    char dateTime[32];
    int maxSpheres;
    int rayDepth;
    int numSpheres;
    int blockSize;
    int numPixels;
    double openmpMs;
} MetricsRunConfig;

typedef struct {
    double kernelMs;
    double memoryTransferMs;
    double totalWithoutMemMs;
    double totalWithMemMs;
    double throughputMpixelsSec;
    double speedupVsOpenmpKernel;
    double speedupVsOpenmpWithMem;
} MetricsTiming;

static inline void metrics_current_datetime(char *buffer, size_t bufferSize)
{
    time_t now = time(NULL);
    struct tm localTime;
    localtime_r(&now, &localTime);
    strftime(buffer, bufferSize, "%Y-%m-%d_%H-%M-%S", &localTime);
}

static inline int metrics_ensure_directory(const char *path)
{
    struct stat st;

    if (mkdir(path, 0755) == 0) {
        return 0;
    }

    if (errno == EEXIST && stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        return 0;
    }

    fprintf(stderr, "Failed to create metrics directory '%s': %s\n",
            path,
            strerror(errno));
    return 1;
}

static inline int metrics_summary_needs_header(const char *path)
{
    FILE *file = fopen(path, "r");

    if (file == NULL) {
        return 1;
    }

    int firstChar = fgetc(file);
    fclose(file);

    return firstChar == EOF;
}

static inline MetricsTiming metrics_make_timing(
    int numPixels,
    double openmpMs,
    double kernelMs,
    double memoryTransferMs)
{
    MetricsTiming timing;

    timing.kernelMs = kernelMs;
    timing.memoryTransferMs = memoryTransferMs;
    timing.totalWithoutMemMs = kernelMs;
    timing.totalWithMemMs = kernelMs + memoryTransferMs;
    timing.throughputMpixelsSec = (kernelMs > 0.0)
        ? ((double)numPixels / kernelMs / 1000.0)
        : 0.0;
    timing.speedupVsOpenmpKernel = (kernelMs > 0.0)
        ? openmpMs / kernelMs
        : 0.0;
    timing.speedupVsOpenmpWithMem = (timing.totalWithMemMs > 0.0)
        ? openmpMs / timing.totalWithMemMs
        : 0.0;

    return timing;
}

static inline int metrics_append_summary_row(
    const MetricsRunConfig *config,
    const char *implementation,
    const MetricsTiming *timing)
{
    if (metrics_ensure_directory(METRICS_DIR) != 0) {
        return 1;
    }

    int writeHeader = metrics_summary_needs_header(METRICS_SUMMARY_PATH);
    FILE *file = fopen(METRICS_SUMMARY_PATH, "a");

    if (file == NULL) {
        fprintf(stderr, "Failed to open '%s': %s\n",
                METRICS_SUMMARY_PATH,
                strerror(errno));
        return 1;
    }

    if (writeHeader) {
        fprintf(file,
                "date_time,max_spheres,ray_depth,num_spheres,block_size,"
                "implementation,openmp_time_ms,kernel_time_ms,"
                "memory_transfer_time_ms,total_without_mem_ms,total_with_mem_ms,"
                "throughput_mpixels_sec,speedup_vs_openmp_kernel,"
                "speedup_vs_openmp_with_mem\n");
    }

    fprintf(file,
            "%s,%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f\n",
            config->dateTime,
            config->maxSpheres,
            config->rayDepth,
            config->numSpheres,
            config->blockSize,
            implementation,
            config->openmpMs,
            timing->kernelMs,
            timing->memoryTransferMs,
            timing->totalWithoutMemMs,
            timing->totalWithMemMs,
            timing->throughputMpixelsSec,
            timing->speedupVsOpenmpKernel,
            timing->speedupVsOpenmpWithMem);

    fclose(file);
    return 0;
}

#endif
