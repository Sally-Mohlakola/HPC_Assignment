#include <mpi.h>
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

#define COMM_ROUNDS 50
#define CLASS_COUNT 10
#define DIMENSION 784
#define LEARNING_RATE 0.01f
#define NUM_MODEL_VARIABLES 7850   // weights*x + bias = class_count*dim +class_count
#define BATCH_SIZE 32  

namespace filesys = std::filesystem;

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
// ===================================================================

static void mean_std_norm(std::vector<Image> &images) {
    const float MEAN = 0.1307f;
    const float STD = 0.3081f;
    for (auto &img : images) {
        for (auto &px :img.image) {
            px = (px-MEAN) / STD;
        }
    }
}

static std::vector<Image> sharding(std::vector<Image> &dataset,int data_holder_num, int num_data_holders) {
    std::stable_sort(dataset.begin(), dataset.end(),
    [](const Image &a, const Image &b) { return a.label < b.label; });

    int data_size = (int)dataset.size();
    int shard_size = data_size /num_data_holders;

    int ibegin =(data_holder_num - 1)*shard_size;
    int iend;

    if (data_holder_num == num_data_holders)
        iend = data_size;
        
    else
        iend= ibegin + shard_size;

    return std::vector<Image>(dataset.begin()+ ibegin,dataset.begin()+ iend);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    std::cout << "Rank " << rank << " of " << comm_size << " processes\n";

    if (comm_size < 3) {
        if (rank == 0)
            std::cerr << "Launch at least 3 processes for federated learning\n";
        MPI_Finalize();
        return 0;
    }

    int data_holder_num = comm_size - 1;

    //========================= SERVER ============================
    if (rank == 0) {

        std::vector<std::vector<float>> test_images = get_images("../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
        std::vector<int> test_labels = get_labels("../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

        std::vector<Image> test_pair =pair(test_labels, test_images);
        mean_std_norm(test_pair);

        Softmax model(LEARNING_RATE);
        std::vector<float> central_model= serialise(model);

        std::ofstream srv_csv("../figures/global_run.csv");
        srv_csv << "round,test_acc\n";

        for (int round = 1; round <= COMM_ROUNDS; round++) {
            MPI_Bcast(central_model.data(), NUM_MODEL_VARIABLES,MPI_FLOAT, 0, MPI_COMM_WORLD);

            std::vector<float> recv_buf(NUM_MODEL_VARIABLES);
            std::vector<float> recv_weights(NUM_MODEL_VARIABLES, 0.0f);
            std::vector<int> shard_sizes(data_holder_num + 1, 0);
            int global_size = 0;

            for (int id = 1; id<= data_holder_num;id++) {
                int local_size;
                MPI_Recv(&local_size, 1, MPI_INT, id,1,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_buf.data(), NUM_MODEL_VARIABLES, MPI_FLOAT, id, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                global_size += local_size;
                shard_sizes[id] = local_size;

                for (int i = 0; i < NUM_MODEL_VARIABLES; i++)
                    recv_weights[i] += (float)local_size * recv_buf[i];
            }

            for (int i = 0; i < NUM_MODEL_VARIABLES; i++) {
                central_model[i] = recv_weights[i]/ (float)global_size;
            }


            deserialise(model, central_model);
            float test_accuracy = model.correctness(test_pair) * 100.f;

            std::cout << "[Central Round] " << round << " [Global Test Accuracy]: " << test_accuracy << "%\n";
            srv_csv << round << "," << test_accuracy << "\n";
        }

        srv_csv.close();

        MPI_Bcast(central_model.data(), NUM_MODEL_VARIABLES,MPI_FLOAT, 0, MPI_COMM_WORLD);

        std::cout << "\n[Central] Done. Figures saved to figures/\n";

    // ========================= WORKERS ==========================
    }

    else {

        auto train_images =get_images("../data/train-images-idx3-ubyte/train-images.idx3-ubyte");
        auto train_labels = get_labels("../data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
        auto train_pair = pair(train_labels, train_images);

        const int data_holder_id = rank;

        auto shard = sharding(train_pair, data_holder_id, data_holder_num);
        mean_std_norm(shard);

        std::cout << "[Distributed Worker " << data_holder_id << "] Shard size: " << shard.size() << "\n";

        Softmax model(LEARNING_RATE);

        std::string wcsv_path = "../figures/worker_" + std::to_string(data_holder_id) + "_metrics.csv";
        std::ofstream wcsv(wcsv_path);
        wcsv << "round,loss,train_acc\n";

        std::mt19937 rng(42 +data_holder_id);

        for (int round = 1; round<= COMM_ROUNDS; round++) {

            float round_loss = 0.f;
            int round_correct = 0;
            int round_seen = 0;
            int num_batches = 0;

            std::vector<float> global_flat(NUM_MODEL_VARIABLES);
            MPI_Bcast(global_flat.data(), NUM_MODEL_VARIABLES, MPI_FLOAT, 0, MPI_COMM_WORLD);
            deserialise(model, global_flat);

            //adaptive epoch decrease
            int LOCAL_EPOCHS = (round < 5) ? 5 : 3;

            for (int epochs_per_round = 0; epochs_per_round < LOCAL_EPOCHS; epochs_per_round++) {

                std::shuffle(shard.begin(), shard.end(), rng);

                for (int i = 0; i < (int)shard.size(); i += BATCH_SIZE) {
                    int end = std::min(i + BATCH_SIZE, (int)shard.size());
                    std::vector<Image> batch(shard.begin() + i, shard.begin() + end);
                    int bc = 0;
                    float loss = model.train_batch(batch, bc);
                    round_loss += loss;
                    round_correct += bc;
                    round_seen += (int)batch.size();
                    num_batches++;
                }
            }

            float avg_loss = (num_batches > 0) ? round_loss / num_batches: 0.f;
            float train_accuracy = 100.f * round_correct /round_seen;

            std::cout << "[Worker " << data_holder_id << "] Round " << round << " | Loss: " << avg_loss
            << " | Train Acc: " << train_accuracy << "%\n";
            wcsv << round << "," << avg_loss << "," << train_accuracy << "\n";

            int sz = static_cast<int>(shard.size());
            MPI_Send(&sz, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            auto updated_flat = serialise(model);
            MPI_Send(updated_flat.data(), NUM_MODEL_VARIABLES,MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        std::vector<float> updated_model(NUM_MODEL_VARIABLES);
        MPI_Bcast(updated_model.data(), NUM_MODEL_VARIABLES,MPI_FLOAT, 0, MPI_COMM_WORLD);

        wcsv.close();
    }

    MPI_Finalize();
    return 0;
}