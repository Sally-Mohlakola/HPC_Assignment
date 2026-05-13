#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <cerrno>
#include <ctime>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

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

inline bool csv_needs_header(const string &path) {
    ifstream existing(path);
    return !existing.good() ||
           existing.peek() == ifstream::traits_type::eof();
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

inline void append_centralised_summary(
    const string &summary_path,
    const string &run_started,
    int epochs_to_80,
    float final_test_accuracy,
    double run_time_seconds)
{
    bool write_header = csv_needs_header(summary_path);
    ofstream summary_csv(summary_path, ios::app);
    if (write_header) {
        summary_csv << "date_time_started,epochs_to_80,test_acc,run_time_seconds\n";
    }

    summary_csv << run_started << ","
                << (epochs_to_80 < 0 ? "not_reached" : to_string(epochs_to_80))
                << ","
                << final_test_accuracy
                << ","
                << run_time_seconds
                << "\n";
}

inline void append_federated_summary(
    const string &summary_path,
    const string &run_started,
    int num_processes,
    int num_nodes,
    const string &data_distribution,
    int epochs_to_80,
    double run_time_seconds)
{
    const string NEW_HEADER =
        "date_time_started,num_processes,num_nodes,data_distribution,epochs_to_80,run_time_seconds";
    const string OLD_HEADER =
        "date_time_started,num_processes,data_distribution,epochs_to_80,run_time_seconds";

    // Self-heal: if the file was written by the pre-num_nodes binary, migrate
    // it in-place so this run can append cleanly. Backfills num_nodes=1 on
    // every existing row (those runs were all single-node by construction).
    {
        ifstream existing(summary_path);
        if (existing.good() && existing.peek() != ifstream::traits_type::eof()) {
            string first_line;
            getline(existing, first_line);
            if (first_line == OLD_HEADER) {
                vector<string> migrated;
                string line;
                while (getline(existing, line)) {
                    size_t c1 = line.find(',');
                    size_t c2 = (c1 == string::npos) ? string::npos
                                                    : line.find(',', c1 + 1);
                    if (c2 == string::npos) {
                        migrated.push_back(line);
                    } else {
                        migrated.push_back(line.substr(0, c2) + ",1" +
                                           line.substr(c2));
                    }
                }
                existing.close();
                ofstream rewrite(summary_path, ios::trunc);
                rewrite << NEW_HEADER << "\n";
                for (const auto &r : migrated) rewrite << r << "\n";
            }
        }
    }

    bool write_header = csv_needs_header(summary_path);
    ofstream summary_csv(summary_path, ios::app);
    if (write_header) {
        summary_csv << NEW_HEADER << "\n";
    }

    summary_csv << run_started << ","
                << num_processes << ","
                << num_nodes << ","
                << data_distribution << ","
                << (epochs_to_80 < 0 ? "not_reached" : to_string(epochs_to_80))
                << ","
                << run_time_seconds
                << "\n";
}

} // namespace csv_utils

#endif
