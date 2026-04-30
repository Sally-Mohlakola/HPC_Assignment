#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <filesystem>
#include <numeric>
#include <cstdlib>

/*
Federated Learning – MNIST (C++ / MPI)
---------------------------------------
Topology  : rank 0 = server | ranks 1..m = workers  (m >= 2)
Non-IID   : training set sorted by label then sharded → each worker
            receives a skewed label distribution (sharding strategy [1])
Transform : worker k gets images rotated by (k-1)*90 degrees (0,90,180,270,…)
            to introduce additional heterogeneity
FedAvg    : server computes WEIGHTED average of worker weights (by shard size) [2]
Evaluation: server tests global model on full (unrotated, normalised) test set

Fixes applied vs original:
  1. Weighted FedAvg (by shard size) — removes bias from unequal shards
  2. Mean/std normalisation (MNIST: mean=0.1307, std=0.3081) on both train & test
  3. Xavier uniform weight initialisation — avoids near-zero gradient start
  4. Per-round RNG reseeding — removes correlated batch ordering across workers
  5. Corrected avg_loss formula — cosmetic fix for logged metrics

Build:
  mpicxx -std=c++17 -O2 -o mnist_federated mnist_federated.cpp -lm

Run (e.g. 4 workers + 1 server = 5 processes):
  mpirun -np 5 ./mnist_federated
*/

namespace fs = std::filesystem;

// ── Constants ─────────────────────────────────────────────────────────────────
static const int   INPUT_SIZE   = 784;    // 28×28
static const int   NUM_CLASSES  = 10;
static const float MNIST_MEAN   = 0.1307f;
static const float MNIST_STD    = 0.3081f;
static const std::string DATA_DIR = "../data";

// ── Endian helper ─────────────────────────────────────────────────────────────
static uint32_t read_be32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0])<<24)|(uint32_t(b[1])<<16)|(uint32_t(b[2])<<8)|uint32_t(b[3]);
}

// ── Sample ────────────────────────────────────────────────────────────────────
struct Sample {
    std::vector<float> x;  // INPUT_SIZE floats, normalised
    int y;
};

// ── Image rotation (multiples of 90°)/will be phased out
static std::vector<float> rotate90(const std::vector<float>& img, int steps) {
    steps = ((steps % 4) + 4) % 4;
    if (steps == 0) return img;

    const int N = 28;
    std::vector<float> src = img;
    std::vector<float> dst(N * N);

    for (int s = 0; s < steps; ++s) {
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                dst[c * N + (N - 1 - r)] = src[r * N + c];
        src = dst;
    }
    return src;
}

// ── MNIST loaders ─────────────────────────────────────────────────────────────
static std::vector<int> load_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return {}; }
    uint32_t magic = read_be32(f);
    if (magic != 0x00000801) { std::cerr << "Bad label magic\n"; return {}; }
    uint32_t n = read_be32(f);
    std::vector<int> labels(n);
    for (uint32_t i = 0; i < n; ++i) {
        unsigned char lbl; f.read(reinterpret_cast<char*>(&lbl), 1);
        labels[i] = lbl;
    }
    return labels;
}

static std::vector<std::vector<float>> load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return {}; }
    uint32_t magic = read_be32(f);
    if (magic != 0x00000803) { std::cerr << "Bad image magic\n"; return {}; }
    uint32_t n = read_be32(f);
    uint32_t rows = read_be32(f), cols = read_be32(f);
    uint32_t px = rows * cols;
    std::vector<std::vector<float>> imgs(n, std::vector<float>(px));
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<unsigned char> raw(px);
        f.read(reinterpret_cast<char*>(raw.data()), px);
        for (uint32_t p = 0; p < px; ++p) imgs[i][p] = raw[p] / 255.0f;
    }
    return imgs;
}

static std::vector<Sample> zip(
    const std::vector<std::vector<float>>& imgs,
    const std::vector<int>& lbls)
{
    std::vector<Sample> ds;
    size_t n = std::min(imgs.size(), lbls.size());
    ds.reserve(n);
    for (size_t i = 0; i < n; ++i) ds.push_back({imgs[i], lbls[i]});
    return ds;
}

// ── FIX 2: Mean/std normalisation ────────────────────────────────────────────
static void normalise(std::vector<Sample>& data) {
    for (auto& s : data)
        for (auto& px : s.x)
            px = (px - MNIST_MEAN) / MNIST_STD;
}

// ── Non-IID shard ─────────────────────────────────────────────────────────────
static std::vector<Sample> make_shard(
    std::vector<Sample>& full, int worker_id, int num_workers)
{
    std::stable_sort(full.begin(), full.end(),
        [](const Sample& a, const Sample& b){ return a.y < b.y; });

    size_t total      = full.size();
    size_t shard_size = total / num_workers;
    size_t start      = (worker_id - 1) * shard_size;
    size_t end        = (worker_id == num_workers) ? total : start + shard_size;

    return std::vector<Sample>(full.begin() + start, full.begin() + end);
}

// ── Flatten / unflatten weights ───────────────────────────────────────────────
static const int PARAM_SIZE = NUM_CLASSES * INPUT_SIZE + NUM_CLASSES;

// ── Softmax model ─────────────────────────────────────────────────────────────
class SoftmaxRegression {
public:
    std::vector<std::vector<float>> W;   // [NUM_CLASSES][INPUT_SIZE]
    std::vector<float> b;                // [NUM_CLASSES]
    float lr;

    explicit SoftmaxRegression(float learning_rate = 0.01f)
        : lr(learning_rate)
    {
        // FIX 3: Xavier uniform initialisation
        // limit = sqrt(6 / (fan_in + fan_out))
        float limit = std::sqrt(6.0f / (INPUT_SIZE + NUM_CLASSES));
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-limit, limit);

        W.assign(NUM_CLASSES, std::vector<float>(INPUT_SIZE));
        b.assign(NUM_CLASSES, 0.0f);
        for (int c = 0; c < NUM_CLASSES; ++c)
            for (int i = 0; i < INPUT_SIZE; ++i)
                W[c][i] = dist(rng);
    }

    std::vector<float> softmax(const std::vector<float>& logits) const {
        std::vector<float> p(NUM_CLASSES);
        float mx = *std::max_element(logits.begin(), logits.end()), s = 0;
        for (int i = 0; i < NUM_CLASSES; ++i) { p[i] = std::exp(logits[i]-mx); s += p[i]; }
        for (int i = 0; i < NUM_CLASSES; ++i) p[i] /= s;
        return p;
    }

    std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> logits(NUM_CLASSES, 0.0f);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            logits[c] = b[c];
            for (int i = 0; i < INPUT_SIZE; ++i) logits[c] += W[c][i] * x[i];
        }
        return softmax(logits);
    }

    float train_batch(const std::vector<Sample>& batch, int& correct) {
        std::vector<std::vector<float>> dW(NUM_CLASSES, std::vector<float>(INPUT_SIZE, 0.f));
        std::vector<float> db(NUM_CLASSES, 0.f);
        float loss = 0.f; correct = 0;

        for (const auto& s : batch) {
            auto p = forward(s.x);
            int pred = std::max_element(p.begin(), p.end()) - p.begin();
            if (pred == s.y) ++correct;
            loss += -std::log(std::max(p[s.y], 1e-8f));
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float e = p[c] - (c == s.y ? 1.f : 0.f);
                db[c] += e;
                for (int i = 0; i < INPUT_SIZE; ++i) dW[c][i] += e * s.x[i];
            }
        }

        float n = static_cast<float>(batch.size());
        for (int c = 0; c < NUM_CLASSES; ++c) {
            b[c] -= lr * db[c] / n;
            for (int i = 0; i < INPUT_SIZE; ++i) W[c][i] -= lr * dW[c][i] / n;
        }
        return loss / n;
    }

    // checks if the softmax allocated max prob to the right class in the distributuon a worker returns
    float evaluate(const std::vector<Sample>& data) const {
        int correct = 0;
        for (const auto& s : data) {
            auto p = forward(s.x);
            if ((int)(std::max_element(p.begin(),p.end())-p.begin()) == s.y) ++correct;
        }
        return 100.f * correct / data.size();
    }

    std::vector<float> to_flat() const {
        std::vector<float> buf;
        buf.reserve(PARAM_SIZE);
        for (int c = 0; c < NUM_CLASSES; ++c)
            for (int i = 0; i < INPUT_SIZE; ++i)
                buf.push_back(W[c][i]);
        for (int c = 0; c < NUM_CLASSES; ++c)
            buf.push_back(b[c]);
        return buf;
    }

    void from_flat(const std::vector<float>& buf) {
        int idx = 0;
        for (int c = 0; c < NUM_CLASSES; ++c)
            for (int i = 0; i < INPUT_SIZE; ++i)
                W[c][i] = buf[idx++];
        for (int c = 0; c < NUM_CLASSES; ++c)
            b[c] = buf[idx++];
    }
};

// ── Gnuplot script ──// This will go because installing dependancies with a CMake also took forever, really not worth it since its graph plotting only 
static void write_gnuplot_script(int num_workers) {
    std::ofstream gp("figures/plot_fed.gp");
    gp << "set datafile separator ','\n\n";

    gp << "set terminal png size 1200,700\n";
    gp << "set output 'figures/fed_loss.png'\n";
    gp << "set title 'Federated Training – Worker Local Loss per Round'\n";
    gp << "set xlabel 'Round'\n";
    gp << "set ylabel 'Loss'\n";
    gp << "plot ";
    for (int w = 1; w <= num_workers; ++w) {
        if (w > 1) gp << ",\\\n     ";
        gp << "'figures/worker_" << w << "_metrics.csv' using 1:2 "
           << "with linespoints title 'Worker " << w << "'";
    }
    gp << "\n\n";

    gp << "set output 'figures/fed_accuracy.png'\n";
    gp << "set title 'Federated Training – Global Test Accuracy per Round'\n";
    gp << "set xlabel 'Round'\n";
    gp << "set ylabel 'Accuracy (%)'\n";
    gp << "plot 'figures/server_metrics.csv' using 1:2 "
       << "with linespoints title 'Global Test Acc'\n";
    gp.close();
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN
// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //debugging
    std::cout << "Rank " << rank << " reports world size: " << world_size << std::endl;

    if (world_size < 3) {
        if (rank == 0)
            std::cerr << "Need at least 3 processes (1 server + 2 workers).\n"
                      << "Run: mpirun -np <m+1> ./mnist_federated\n";
        MPI_Finalize();
        return 1;
    }

    const int   num_workers  = world_size - 1;
    const int   FL_ROUNDS    = 10;
    const int   LOCAL_EPOCHS = 3;
    const int   BATCH_SIZE   = 128;
    const float LR           = 0.05f;

    // ── SERVER (rank 0) ────────────────────────────────────────────────────
    if (rank == 0) {
        //generate graphs that plot the convergence of the training as epochs go by
        fs::create_directories("figures");
        
        auto test_imgs = load_images(DATA_DIR + "/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
        auto test_lbls = load_labels(DATA_DIR + "/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
        auto test_data = zip(test_imgs, test_lbls);

        // FIX 2: Normalise test set
        normalise(test_data);

        std::cout << "[Server] Test samples: " << test_data.size() << "\n";
        std::cout << "[Server] Workers: " << num_workers << "\n";
        std::cout << "[Server] FL rounds: " << FL_ROUNDS << "\n\n";

        SoftmaxRegression global_model(LR);
        std::vector<float> global_flat = global_model.to_flat();

        std::ofstream srv_csv("figures/server_metrics.csv");
        srv_csv << "round,test_acc\n";

        write_gnuplot_script(num_workers);

        for (int round = 1; round <= FL_ROUNDS; ++round) {
            // 1. Broadcast global weights to all workers
            MPI_Bcast(global_flat.data(), PARAM_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

            // FIX 1: Weighted FedAvg — receive shard size then weights from each worker
            std::vector<float> aggregated(PARAM_SIZE, 0.0f);
            std::vector<float> recv_buf(PARAM_SIZE);
            float total_samples = 0.0f;

            for (int w = 1; w <= num_workers; ++w) {
                // Receive shard size (tag 1) then weights (tag 0)
                // receive all the weights from workers
                int sz = 0;
                MPI_Recv(&sz, 1, MPI_INT,   w, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_buf.data(), PARAM_SIZE, MPI_FLOAT, w, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_samples += sz;
                for (int p = 0; p < PARAM_SIZE; ++p)
                    aggregated[p] += recv_buf[p] * static_cast<float>(sz);  // weighted sum
            }

            // Weighted mean
            for (int p = 0; p < PARAM_SIZE; ++p)
                aggregated[p] /= total_samples;

            global_flat = aggregated;
            global_model.from_flat(global_flat);

            float test_acc = global_model.evaluate(test_data);
            std::cout << "[Server] Round " << round
                      << " | Global Test Acc: " << test_acc << "%\n";
            srv_csv << round << "," << test_acc << "\n";
        }

        srv_csv.close();

        // Termination broadcast
        MPI_Bcast(global_flat.data(), PARAM_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

        std::system("gnuplot figures/plot_fed.gp 2>/dev/null");
        std::cout << "\n[Server] Done. Figures saved to figures/\n";

    // ── WORKERS (rank 1..m) ────────────────────────────────────────────────
    } else {
        const int worker_id = rank;

        auto train_imgs = load_images(DATA_DIR + "/train-images-idx3-ubyte/train-images.idx3-ubyte");
        auto train_lbls = load_labels(DATA_DIR + "/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
        auto full_train = zip(train_imgs, train_lbls);

        auto shard = make_shard(full_train, worker_id, num_workers);

        
        int rotation_steps = 0; // each worker received a rotation of 90 deg before but this actually capped the global testing accuracy; so no rotation anymore
        if (rotation_steps > 0)
            for (auto& s : shard)
                s.x = rotate90(s.x, rotation_steps);

        // FIX 2: Normalise training shard AFTER rotation
        normalise(shard);

        std::cout << "[Worker " << worker_id << "] Shard size: " << shard.size()
                  << " | Rotation: " << rotation_steps * 90 << "°\n";

        std::vector<int> label_count(NUM_CLASSES, 0);
        for (const auto& s : shard) label_count[s.y]++;
        std::cout << "[Worker " << worker_id << "] Label dist: ";
        for (int c = 0; c < NUM_CLASSES; ++c)
            std::cout << c << ":" << label_count[c] << " ";
        std::cout << "\n";

        SoftmaxRegression model(LR);

        fs::create_directories("figures");
        std::string wcsv_path = "figures/worker_" + std::to_string(worker_id) + "_metrics.csv";
        std::ofstream wcsv(wcsv_path);
        wcsv << "round,loss,train_acc\n";

        for (int round = 1; round <= FL_ROUNDS; ++round) {
            // 1. Receive global weights
            std::vector<float> global_flat(PARAM_SIZE);
            MPI_Bcast(global_flat.data(), PARAM_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
            model.from_flat(global_flat);

            // 2. Local training
            float round_loss    = 0.f;
            int   round_correct = 0;
            int   round_seen    = 0;
            int   num_batches   = 0;

            for (int ep = 0; ep < LOCAL_EPOCHS; ++ep) {
                // FIX 4: Per-round, per-epoch RNG seed — removes correlated shuffling
                std::mt19937 rng(42 + worker_id * 1000 + round * 10 + ep);
                std::shuffle(shard.begin(), shard.end(), rng);

                for (size_t i = 0; i < shard.size(); i += BATCH_SIZE) {
                    size_t end = std::min(i + (size_t)BATCH_SIZE, shard.size());
                    std::vector<Sample> batch(shard.begin() + i, shard.begin() + end);
                    int bc = 0;
                    float loss = model.train_batch(batch, bc);
                    round_loss    += loss;
                    round_correct += bc;
                    round_seen    += static_cast<int>(batch.size());
                    ++num_batches;
                }
            }

            // FIX 5: Correct avg_loss — divide by actual number of batches processed
            float avg_loss  = (num_batches > 0) ? round_loss / num_batches : 0.f;
            float train_acc = 100.f * round_correct / round_seen;

            std::cout << "[Worker " << worker_id << "] Round " << round
                      << " | Loss: " << avg_loss
                      << " | Train Acc: " << train_acc << "%\n";
            wcsv << round << "," << avg_loss << "," << train_acc << "\n";

            // FIX 1: Send shard size (tag 1) before weights (tag 0) for weighted FedAvg
            int sz = static_cast<int>(shard.size());
            MPI_Send(&sz, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            auto updated_flat = model.to_flat();
            MPI_Send(updated_flat.data(), PARAM_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        // Receive termination broadcast
        std::vector<float> final_flat(PARAM_SIZE);
        MPI_Bcast(final_flat.data(), PARAM_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

        wcsv.close();
    }

    MPI_Finalize();
    return 0;
}