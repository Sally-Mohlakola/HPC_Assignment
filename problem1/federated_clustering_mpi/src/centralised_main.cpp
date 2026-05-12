#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <filesystem>
#include <numeric>
#include "federated_model.h"


#define LEARNING_RATE 0.01f
#define LOCAL_EPOCHS 50
#define NUM_MODEL_VARIABLES 7850

//Big to little endian converter (to interpret MNIST headers in a reversed byte order)
static uint32_t idx_to_integer(std::ifstream &file) {
    unsigned char header[4];
    file.read(reinterpret_cast<char *>(header), 4);
    return (uint32_t(header[0])<< 24)|(uint32_t(header[1])<< 16)|
           (uint32_t(header[2])<< 8)|uint32_t(header[3]);
}

//=================== MNIST data loaders ======================

static std::vector<int> get_labels(const std::string &path) {
    std::vector<int> empty = {};
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Could not open file: " << path << "\n";
        return empty;
    }
    uint32_t format =idx_to_integer(file);
    if (format != 0x00000801) {
        std::cerr << "Format mismatch from labels\n";
        return empty;
    }

    uint32_t num_images = idx_to_integer(file);
    std::vector<int> labels(num_images);
    for (uint32_t i = 0; i < num_images; i++) {
        unsigned char l;
        file.read(reinterpret_cast<char *>(&l), 1);
        labels[i] = l;
    }
    return labels;
}

static std::vector<std::vector<float>> get_images(const std::string &path) {
    std::vector<std::vector<float>> empty ={};
    float pnorm = 255.0f;

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Could not open file: " << path << "\n";
        return empty;
    }
    uint32_t format = idx_to_integer(file);
    if (format != 0x00000803) {
        std::cerr << "Format mismatch from images\n";
        return empty;
    }
    uint32_t num_images = idx_to_integer(file);
    uint32_t rows = idx_to_integer(file);
    uint32_t cols = idx_to_integer(file);
    uint32_t dim = rows*cols;

    std::vector<std::vector<float>> images(num_images,std::vector<float>(dim));

    for (uint32_t i = 0; i < num_images; i++) {
        std::vector<unsigned char> pixels(dim);
        file.read(reinterpret_cast<char *>(pixels.data()), dim);

        for (uint32_t j =0; j <dim; j++) {
            images[i][j] = pixels[j]/ pnorm;
        }
    }
    return images;
}

//============================== end MNIST loaders ==============================

static void mean_std_norm(std::vector<Image> &images) {
    const float MEAN = 0.1307f;
    const float STD = 0.3081f;
    for (auto &img : images) {
        for (auto &px :img.image) {
            px = (px-MEAN) / STD;
        }
    }
}

//=============================================================================

// Serialise a Softmax model's weights + bias into a flat vector for MPI transfer.
static std::vector<float> serialise(const Softmax &model) {
    std::vector<float> flat;
    flat.reserve(NUM_MODEL_VARIABLES);
    for (int i = 0; i < model.num_classes; i++)
        for (int j = 0; j < model.num_features; j++)
            flat.push_back(model.weights[i][j]);
    for (int i = 0; i < model.num_classes; i++)
        flat.push_back(model.bias[i]);
    return flat;
}

// Deserialise a flat buffer back into a Softmax model's weights and bias.
static void deserialise(Softmax &model, const std::vector<float> &flat) {
    int idx = 0;
    for (int i = 0; i < model.num_classes; i++)
        for (int j = 0; j < model.num_features; j++)
            model.weights[i][j] = flat[idx++];

    for (int i = 0; i < model.num_classes; i++)
        model.bias[i] = flat[idx++];
}

//==================================================================================================


int main(){

    std::vector<std::vector<float>> test_images = get_images("../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    std::vector<int> test_labels = get_labels("../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    auto train_images =get_images("../data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    auto train_labels = get_labels("../data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    auto train_pair = pair(train_labels, train_images);


    std::vector<Image> test_pair =pair(test_labels, test_images);
    mean_std_norm(test_pair);

    Softmax model(LEARNING_RATE);
    std::vector<float> central_model= serialise(model);

    std::ofstream srv_csv("../figures/global_run.csv");
    srv_csv << "round,test_acc\n";

    std::ofstream metrics("..figures/metrics.csv");
    metrics << "epoch,train_loss,train_acc,test_acc\n";

    std::mt19937 rnd(42);
    int batch_size=128;

    for (int epoch=1; epoch<=LOCAL_EPOCHS;epoch++){

        std::shuffle(train_pair.begin(), train_pair.end(),rnd);

        float epoch_loss=0;
        int total_correct=0;
        int total_seen=0;
        int total_data_size = train_pair.size();
        int num_batches=0;

        for (int i=0; i<total_data_size; i+=batch_size){

            int iend = std::min(i + batch_size, total_data_size);
            std::vector<Image> batch(train_pair.begin()+ i, train_pair.begin() + iend);

            int batch_correct = 0;
            float loss = model.train_batch(batch, batch_correct);

            epoch_loss += loss;
            total_correct += batch_correct;
            total_seen += (int)batch.size();
            num_batches++;

        }

        float avg_loss = epoch_loss / num_batches;
        float train_accuracy = 100.0f * total_correct /total_seen;
        float test_accuracy = test_pair.empty() ? 0.0f :model.correctness(test_pair);

        std::cout << "Epoch " << epoch << " | Loss: "<<avg_loss << " | Train Accuracy: " << train_accuracy  << "%"
        << " | Test Accuracy: "  << test_accuracy   << "%\n";

        metrics << epoch << "," << avg_loss << ","<< train_accuracy << "," << test_accuracy << "\n";
    }

    metrics.close();

    return 0;

    }


