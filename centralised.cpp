#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <filesystem>
#include <cstdlib>

/*
Centralized MNIST baseline (C++)
---------------------------------
- Reads MNIST binary files:
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte
- Uses mini-batch SGD
- Model: Softmax Logistic Regression
- Saves epoch metrics to:
    figures/metrics.csv
- Auto-generates plots using gnuplot (if installed):
    figures/loss.png
    figures/accuracy.png

MNIST IDX binary format:
  Images: magic(4) + num_items(4) + rows(4) + cols(4) + pixels (row-major, uint8)
  Labels: magic(4) + num_items(4) + labels (uint8)
  All multi-byte integers are big-endian.

Classes: 0-9 (digits)
*/

namespace fs = std::filesystem;

const int INPUT_SIZE  = 784;   // 28x28
const int NUM_CLASSES = 10;

// ── Endian helper ────────────────────────────────────────────────────────────
static uint32_t read_be32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) <<  8) |  uint32_t(b[3]);
}

// ── Data structures ──────────────────────────────────────────────────────────
struct Sample {
    std::vector<float> x;
    int y;
};

// ── Model ────────────────────────────────────────────────────────────────────
class SoftmaxRegression {
private:
    std::vector<std::vector<float>> W;
    std::vector<float> b;
    float lr;

public:
    explicit SoftmaxRegression(float learning_rate = 0.01f)
        : lr(learning_rate)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

        W.assign(NUM_CLASSES, std::vector<float>(INPUT_SIZE));
        b.assign(NUM_CLASSES, 0.0f);

        for (int c = 0; c < NUM_CLASSES; ++c)
            for (int i = 0; i < INPUT_SIZE; ++i)
                W[c][i] = dist(rng);
    }

    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> probs(NUM_CLASSES);
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (int i = 0; i < NUM_CLASSES; ++i) {
            probs[i] = std::exp(logits[i] - max_val);
            sum += probs[i];
        }
        for (int i = 0; i < NUM_CLASSES; ++i)
            probs[i] /= sum;
        return probs;
    }

    std::vector<float> forward(const std::vector<float>& x) {
        std::vector<float> logits(NUM_CLASSES, 0.0f);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            logits[c] = b[c];
            for (int i = 0; i < INPUT_SIZE; ++i)
                logits[c] += W[c][i] * x[i];
        }
        return softmax(logits);
    }

    // Returns mean cross-entropy loss; increments correct count
    float train_batch(const std::vector<Sample>& batch, int& correct) {
        std::vector<std::vector<float>> dW(NUM_CLASSES, std::vector<float>(INPUT_SIZE, 0.0f));
        std::vector<float> db(NUM_CLASSES, 0.0f);

        float loss = 0.0f;
        correct = 0;

        for (const auto& s : batch) {
            auto probs = forward(s.x);

            int pred = static_cast<int>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());
            if (pred == s.y) ++correct;

            loss += -std::log(std::max(probs[s.y], 1e-8f));

            for (int c = 0; c < NUM_CLASSES; ++c) {
                float err = probs[c] - (c == s.y ? 1.0f : 0.0f);
                db[c] += err;
                for (int i = 0; i < INPUT_SIZE; ++i)
                    dW[c][i] += err * s.x[i];
            }
        }

        float n = static_cast<float>(batch.size());
        for (int c = 0; c < NUM_CLASSES; ++c) {
            b[c] -= lr * db[c] / n;
            for (int i = 0; i < INPUT_SIZE; ++i)
                W[c][i] -= lr * dW[c][i] / n;
        }

        return loss / n;
    }

    // Evaluate accuracy on a held-out set (no weight update)
    float evaluate(const std::vector<Sample>& data) {
        int correct = 0;
        for (const auto& s : data) {
            auto probs = forward(s.x);
            int pred = static_cast<int>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());
            if (pred == s.y) ++correct;
        }
        return 100.0f * correct / static_cast<float>(data.size());
    }
};

// ── MNIST loaders ─────────────────────────────────────────────────────────────

std::vector<int> load_mnist_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open labels: " << path << "\n";
        return {};
    }

    uint32_t magic = read_be32(f);
    if (magic != 0x00000801) {
        std::cerr << "Bad magic in labels file: " << magic << "\n";
        return {};
    }

    uint32_t n = read_be32(f);
    std::vector<int> labels(n);
    for (uint32_t i = 0; i < n; ++i) {
        unsigned char lbl;
        f.read(reinterpret_cast<char*>(&lbl), 1);
        labels[i] = static_cast<int>(lbl);
    }
    std::cout << "Loaded " << n << " labels from " << path << "\n";
    return labels;
}

std::vector<std::vector<float>> load_mnist_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open images: " << path << "\n";
        return {};
    }

    uint32_t magic = read_be32(f);
    if (magic != 0x00000803) {
        std::cerr << "Bad magic in images file: " << magic << "\n";
        return {};
    }

    uint32_t n    = read_be32(f);
    uint32_t rows = read_be32(f);
    uint32_t cols = read_be32(f);
    uint32_t pixels = rows * cols;

    std::vector<std::vector<float>> images(n, std::vector<float>(pixels));
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<unsigned char> raw(pixels);
        f.read(reinterpret_cast<char*>(raw.data()), pixels);
        for (uint32_t p = 0; p < pixels; ++p)
            images[i][p] = raw[p] / 255.0f;
    }
    std::cout << "Loaded " << n << " images (" << rows << "x" << cols
              << ") from " << path << "\n";
    return images;
}

std::vector<Sample> zip_mnist(
    const std::vector<std::vector<float>>& images,
    const std::vector<int>& labels)
{
    std::vector<Sample> dataset;
    size_t n = std::min(images.size(), labels.size());
    dataset.reserve(n);
    for (size_t i = 0; i < n; ++i)
        dataset.push_back({ images[i], labels[i] });
    return dataset;
}

// ── Gnuplot script ────────────────────────────────────────────────────────────
void write_gnuplot_script() {
    std::ofstream gp("figures/plot.gp");
    gp << "set datafile separator ','\n\n";

    gp << "set terminal png size 1000,700\n";
    gp << "set output 'figures/loss.png'\n";
    gp << "set title 'Training Loss per Epoch'\n";
    gp << "set xlabel 'Epoch'\n";
    gp << "set ylabel 'Loss'\n";
    gp << "plot 'figures/metrics.csv' using 1:2 with linespoints title 'Train Loss'\n\n";

    gp << "set output 'figures/accuracy.png'\n";
    gp << "set title 'Accuracy per Epoch'\n";
    gp << "set xlabel 'Epoch'\n";
    gp << "set ylabel 'Accuracy (%)'\n";
    gp << "plot 'figures/metrics.csv' using 1:3 with linespoints title 'Train Acc', \\\n";
    gp << "     'figures/metrics.csv' using 1:4 with linespoints title 'Test Acc'\n";
    gp.close();
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    // ── Adjust these paths to match your layout ──────────────────────────────
    const std::string data_dir   = "../data";   // folder containing the four IDX files
    const int   epochs        = 50;
    const int   batch_size    = 128;
    const float learning_rate = 0.05f;
    // ─────────────────────────────────────────────────────────────────────────

    fs::create_directories("figures");

    // Load training set
    std::cout << "Loading MNIST training data...\n";
    auto train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte/train-images.idx3-ubyte");
    auto train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    auto train_data   = zip_mnist(train_images, train_labels);

    // Load test set
    std::cout << "Loading MNIST test data...\n";
    auto test_images = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    auto test_labels = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    auto test_data   = zip_mnist(test_images, test_labels);

    if (train_data.empty()) {
        std::cerr << "Training data loading failed.\n";
        return 1;
    }

    std::cout << "Train samples: " << train_data.size()
              << "  |  Test samples: " << test_data.size() << "\n\n";

    SoftmaxRegression model(learning_rate);

    std::ofstream metrics("figures/metrics.csv");
    metrics << "epoch,train_loss,train_acc,test_acc\n";

    std::mt19937 rng(42);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(train_data.begin(), train_data.end(), rng);

        float epoch_loss   = 0.0f;
        int   total_correct = 0;
        int   total_seen    = 0;
        int   num_batches   = 0;

        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            size_t end = std::min(i + static_cast<size_t>(batch_size), train_data.size());
            std::vector<Sample> batch(train_data.begin() + i, train_data.begin() + end);

            int batch_correct = 0;
            float loss = model.train_batch(batch, batch_correct);

            epoch_loss    += loss;
            total_correct += batch_correct;
            total_seen    += static_cast<int>(batch.size());
            ++num_batches;
        }

        float avg_loss   = epoch_loss / num_batches;
        float train_acc  = 100.0f * total_correct / total_seen;
        float test_acc   = test_data.empty() ? 0.0f : model.evaluate(test_data);

        std::cout
            << "Epoch " << epoch
            << " | Loss: "      << avg_loss
            << " | Train Acc: " << train_acc  << "%"
            << " | Test Acc: "  << test_acc   << "%\n";

        metrics << epoch << "," << avg_loss << ","
                << train_acc << "," << test_acc << "\n";
    }

    metrics.close();

    write_gnuplot_script();
    std::system("gnuplot figures/plot.gp");

    std::cout << "\nSaved to figures/:\n";
    std::cout << "  metrics.csv\n";
    std::cout << "  loss.png\n";
    std::cout << "  accuracy.png\n";

    return 0;
}