#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <cerrno>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace csv_utils {

using namespace std;

inline string current_datetime_string() {
    time_t now = time(nullptr);
    tm local_time{};
    localtime_r(&now, &local_time);

    char buffer[32];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &local_time);
    return string(buffer);
}

inline bool ensure_directory(const string &path) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }

    if (mkdir(path.c_str(), 0755) == 0) {
        return true;
    }

    cerr << "Could not create directory: " << path
         << " (errno " << errno << ")\n";
    return false;
}

// Resolve this run's unique identifier. The harness sets `env_var` to a value
// it has guaranteed unique across the whole sweep, so concurrent runs never
// collide on an output directory. When the binary is launched directly with no
// harness, fall back to a timestamp plus PID -- still collision-free even if a
// few runs are started by hand at the same second.
inline string resolve_run_id(const string &env_var) {
    const char *v = getenv(env_var.c_str());
    if (v && *v) {
        return string(v);
    }
    return current_datetime_string() + "_pid" + to_string(getpid());
}

inline ofstream create_metrics_csv(
    const string &path,
    const string &header)
{
    ofstream csv(path);
    csv << header << "\n";
    return csv;
}

inline ofstream create_model_metrics_csv(const string &path) {
    return create_metrics_csv(path, "epoch,train_loss,train_acc,test_acc");
}

inline ofstream create_worker_metrics_csv(const string &path) {
    return create_metrics_csv(path, "round,loss,train_acc");
}

inline void write_model_metrics(
    ostream &csv,
    int epoch,
    float train_loss,
    float train_accuracy,
    float test_accuracy)
{
    csv << epoch << ","
        << train_loss << ","
        << train_accuracy << ","
        << test_accuracy << "\n";
}

inline void write_worker_metrics(
    ostream &csv,
    int round,
    float loss,
    float train_accuracy)
{
    csv << round << ","
        << loss << ","
        << train_accuracy << "\n";
}

// Each run writes its own self-contained summary row into its private result
// directory (run_summary.csv). Nothing is ever appended to a shared file, so
// concurrent runs cannot race or interleave. The aggregate phase later gathers
// every run_summary.csv into a single summary.csv, single-threaded and safely.
inline void write_centralised_run_summary(
    const string &path,
    const string &sweep_id,
    const string &run_id,
    const string &run_started,
    int epochs_to_80,
    float final_test_accuracy,
    double run_time_seconds)
{
    ofstream csv(path, ios::trunc);
    csv << "sweep_id,run_id,date_time_started,epochs_to_80,test_acc,"
           "run_time_seconds\n";
    csv << sweep_id << ","
        << run_id << ","
        << run_started << ","
        << (epochs_to_80 < 0 ? "not_reached" : to_string(epochs_to_80)) << ","
        << final_test_accuracy << ","
        << run_time_seconds << "\n";
}

inline void write_federated_run_summary(
    const string &path,
    const string &sweep_id,
    const string &run_id,
    const string &run_started,
    int num_processes,
    int num_nodes,
    const string &data_distribution,
    int epochs_to_80,
    double run_time_seconds)
{
    ofstream csv(path, ios::trunc);
    csv << "sweep_id,run_id,date_time_started,num_processes,num_nodes,"
           "data_distribution,epochs_to_80,run_time_seconds\n";
    csv << sweep_id << ","
        << run_id << ","
        << run_started << ","
        << num_processes << ","
        << num_nodes << ","
        << data_distribution << ","
        << (epochs_to_80 < 0 ? "not_reached" : to_string(epochs_to_80)) << ","
        << run_time_seconds << "\n";
}

} // namespace csv_utils

#endif
