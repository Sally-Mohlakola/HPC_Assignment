#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>
#include "csv_utils.h"
#include "federated_model.h"

using namespace std;
using namespace csv_utils;
using namespace std::chrono;

#define LEARNING_RATE 0.01f

//Big to little endian converter (to interpret MNIST headers in a reversed byte order)
static uint32_t idx_to_integer(ifstream &file) {
    unsigned char header[4];
    file.read(reinterpret_cast<char *>(header), 4);
    return (uint32_t(header[0])<< 24)|(uint32_t(header[1])<< 16)|
           (uint32_t(header[2])<< 8)|uint32_t(header[3]);
}

//=================== MNIST data loaders ======================

static vector<int> get_labels(const string &path) {
    vector<int> empty = {};
    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Could not open file: " << path << "\n";
        return empty;
    }
    uint32_t format =idx_to_integer(file);
    if (format != 0x00000801) {
        cerr << "Format mismatch from labels\n";
        return empty;
    }

    uint32_t num_images = idx_to_integer(file);
    vector<int> labels(num_images);
    for (uint32_t i = 0; i < num_images; i++) {
        unsigned char l;
        file.read(reinterpret_cast<char *>(&l), 1);
        labels[i] = l;
    }
    return labels;
}

static vector<vector<float>> get_images(const string &path) {
    vector<vector<float>> empty ={};
    float pnorm = 255.0f;

    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Could not open file: " << path << "\n";
        return empty;
    }
    uint32_t format = idx_to_integer(file);
    if (format != 0x00000803) {
        cerr << "Format mismatch from images\n";
        return empty;
    }
    uint32_t num_images = idx_to_integer(file);
    uint32_t rows = idx_to_integer(file);
    uint32_t cols = idx_to_integer(file);
    uint32_t dim = rows*cols;

    vector<vector<float>> images(num_images,vector<float>(dim));

    for (uint32_t i = 0; i < num_images; i++) {
        vector<unsigned char> pixels(dim);
        file.read(reinterpret_cast<char *>(pixels.data()), dim);

        for (uint32_t j =0; j <dim; j++) {
            images[i][j] = pixels[j]/ pnorm;
        }
    }
    return images;
}

//============================== end MNIST loaders ==============================

static void mean_std_norm(vector<Image> &images) {
    const float MEAN = 0.1307f;
    const float STD = 0.3081f;
    for (auto &img : images) {
        for (auto &px :img.image) {
            px = (px-MEAN) / STD;
        }
    }
}

//==================================================================================================


int main(){
    auto run_timer_start = steady_clock::now();

    int max_epochs = 50;
    int min_epochs = 30;
    int patience = 5;
    float min_delta = 0.001f;
    float best_accuracy = -1.0f;
    int epochs_without_improvement = 0;
    int epochs_to_80 = -1;
    float final_test_accuracy = 0.0f;

    const string output_root = "../centralised_metrics";
    const string run_started = current_datetime_string();
    ensure_directory(output_root);
    const string output_dir = output_root + "/" + run_started;
    ensure_directory(output_dir);

    vector<vector<float>> test_images = get_images("../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    vector<int> test_labels = get_labels("../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    auto train_images =get_images("../data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    auto train_labels = get_labels("../data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    auto train_pair = make_pairs(train_labels, train_images);
    mean_std_norm(train_pair);


    vector<Image> test_pair =make_pairs(test_labels, test_images);
    mean_std_norm(test_pair);

    Softmax model(LEARNING_RATE);

    ofstream metrics = create_model_metrics_csv(
        output_dir + "/centralised_metrics.csv");

    mt19937 rnd(42);
    int batch_size=128;

    for (int epoch=1; epoch<=max_epochs;epoch++){

        shuffle(train_pair.begin(), train_pair.end(),rnd);

        float epoch_loss=0;
        int total_correct=0;
        int total_seen=0;
        int total_data_size = train_pair.size();
        int num_batches=0;

        for (int i=0; i<total_data_size; i+=batch_size){

            int iend = min(i + batch_size, total_data_size);
            vector<Image> batch(train_pair.begin()+ i, train_pair.begin() + iend);

            int batch_correct = 0;
            float loss = model.train_batch(batch, batch_correct);

            epoch_loss += loss;
            total_correct += batch_correct;
            total_seen += (int)batch.size();
            num_batches++;

        }

        float avg_loss = epoch_loss / num_batches;
        float train_accuracy = 100.0f * total_correct /total_seen;
        float test_accuracy = test_pair.empty() ? 0.0f :model.correctness(test_pair) * 100.0f;
        final_test_accuracy = test_accuracy;

        if (epochs_to_80 < 0 && test_accuracy >= 80.0f) {
            epochs_to_80 = epoch;
        }

        if (test_accuracy > best_accuracy + min_delta){
            best_accuracy = test_accuracy;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement++;
        }

        cout << "Epoch " << epoch << " | Loss: "<<avg_loss << " | Train Accuracy: " << train_accuracy  << "%"
        << " | Test Accuracy: "  << test_accuracy   << "%\n";

        write_model_metrics(
            metrics,
            epoch,
            avg_loss,
            train_accuracy,
            test_accuracy);

        if (epoch >= min_epochs && epochs_without_improvement >= patience){
            cout << "[Centralised] Early stopping at epoch " << epoch
            << " after " << patience << " epochs without test accuracy improvement.\n";
            break;
        }
    }

    metrics.close();
    double run_time_seconds =
        duration<double>(steady_clock::now() - run_timer_start).count();

    string summary_path = output_root + "/summary.csv";
    append_centralised_summary(
        summary_path,
        run_started,
        epochs_to_80,
        final_test_accuracy,
        run_time_seconds);

    cout << "\n[Centralised] Done. Centralised data saved to " << output_dir << "\n";

    return 0;

    }
