#ifndef METRICS_H
#define METRICS_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#define METRICS_DIR_DEFAULT "metrics"
#define METRICS_SUMMARY_FILE "summary.csv"
#define METRICS_PATH_BUFFER_SIZE 4096

typedef struct {
    char dateTime[32];
    int maxSpheres;
    int rayDepth;
    int numSpheres;
    int blockSize;
    int numPixels;
    double openmpSeconds;
} MetricsRunConfig;

typedef struct {
    double kernelSeconds;
    double memoryTransferSeconds;
    double totalWithoutMemSeconds;
    double totalWithMemSeconds;
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

static inline const char *metrics_get_directory(void)
{
    const char *dir = getenv("CRAYTRACER_METRICS_DIR");
    return (dir != NULL && dir[0] != '\0') ? dir : METRICS_DIR_DEFAULT;
}

static inline const char *metrics_get_sweep_id(void)
{
    const char *sweepId = getenv("CRAYTRACER_SWEEP_ID");
    return (sweepId != NULL && sweepId[0] != '\0') ? sweepId : "0";
}

static inline int metrics_summary_path(char *buffer, size_t bufferSize)
{
    int written = snprintf(buffer,
                           bufferSize,
                           "%s/%s",
                           metrics_get_directory(),
                           METRICS_SUMMARY_FILE);

    if (written < 0 || (size_t)written >= bufferSize) {
        fprintf(stderr, "Metrics summary path is too long\n");
        return 1;
    }

    return 0;
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
    double openmpSeconds,
    double kernelSeconds,
    double memoryTransferSeconds)
{
    MetricsTiming timing;

    timing.kernelSeconds = kernelSeconds;
    timing.memoryTransferSeconds = memoryTransferSeconds;
    timing.totalWithoutMemSeconds = kernelSeconds;
    timing.totalWithMemSeconds = kernelSeconds + memoryTransferSeconds;
    timing.throughputMpixelsSec = (kernelSeconds > 0.0)
        ? ((double)numPixels / kernelSeconds / 1000000.0)
        : 0.0;
    timing.speedupVsOpenmpKernel = (kernelSeconds > 0.0)
        ? openmpSeconds / kernelSeconds
        : 0.0;
    timing.speedupVsOpenmpWithMem = (timing.totalWithMemSeconds > 0.0)
        ? openmpSeconds / timing.totalWithMemSeconds
        : 0.0;

    return timing;
}

static inline int metrics_append_summary_row(
    const MetricsRunConfig *config,
    const char *implementation,
    const MetricsTiming *timing)
{
    const char *metricsDir = metrics_get_directory();
    char summaryPath[METRICS_PATH_BUFFER_SIZE];

    if (metrics_ensure_directory(metricsDir) != 0) {
        return 1;
    }

    if (metrics_summary_path(summaryPath, sizeof(summaryPath)) != 0) {
        return 1;
    }

    int writeHeader = metrics_summary_needs_header(summaryPath);
    FILE *file = fopen(summaryPath, "a");

    if (file == NULL) {
        fprintf(stderr, "Failed to open '%s': %s\n",
                summaryPath,
                strerror(errno));
        return 1;
    }

    if (writeHeader) {
        fprintf(file,
                "sweep_id,date_time,max_spheres,ray_depth,num_spheres,block_size,"
                "implementation,openmp_time_s,kernel_time_s,"
                "memory_transfer_time_s,total_without_mem_s,total_with_mem_s,"
                "throughput_mpixels_sec,speedup_vs_openmp_kernel,"
                "speedup_vs_openmp_with_mem\n");
    }

    fprintf(file,
            "%s,%s,%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f\n",
            metrics_get_sweep_id(),
            config->dateTime,
            config->maxSpheres,
            config->rayDepth,
            config->numSpheres,
            config->blockSize,
            implementation,
            config->openmpSeconds,
            timing->kernelSeconds,
            timing->memoryTransferSeconds,
            timing->totalWithoutMemSeconds,
            timing->totalWithMemSeconds,
            timing->throughputMpixelsSec,
            timing->speedupVsOpenmpKernel,
            timing->speedupVsOpenmpWithMem);

    fclose(file);
    return 0;
}

#endif
