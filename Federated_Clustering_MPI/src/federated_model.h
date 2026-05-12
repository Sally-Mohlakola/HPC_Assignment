#include <vector>
#include <random>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

struct Image {
    vector<float> image; // image is an array of normalised pixel values [0,1]
    int label; // the range of numbers have labels {0,1,2,3,4,5,6,7,8,9}
};

// Image-label data pairs (zip dataset here since data is read from different files)
static vector<Image> make_pairs(const vector<int> &label,const vector<vector<float>> &image) {
    int size = min(image.size(), label.size());
    vector<Image> pairs;
    for (int i = 0; i < size; i++) {
        pairs.emplace_back(Image{image[i], label[i]});
    }
    return pairs;
}

class Softmax {
public:
    vector<vector<float>> weights;
    vector<float> bias;

    float learning_rate =0.1f;//tweak hyperparameters for experimentation
    int num_classes= 10;
    int num_features = 28*28;

    Softmax() {
        initialise_weights();
    }

    Softmax(float lr): learning_rate(lr) { 
        initialise_weights();
    }

    void initialise_weights() {
    
        random_device rad;
        mt19937 random_gen(rad());
        normal_distribution<float> dist(-0.01f, 0.01f);

        weights.assign(num_classes,vector<float>(num_features));
        bias.assign(num_classes,0.0f);

        for (int i= 0; i < num_classes; i++) {
            for (int j = 0; j<num_features; j++) {
                weights[i][j]= dist(random_gen);
            }
        }

        cout << "Softmax regression: " << num_classes<< " classes; " << num_features << " features\n";
    }

    vector<float> softmax_prediction(const vector<float> &features) {
        vector<float> probability(num_classes);
        float max_el = *max_element(features.begin(),features.end());
        float sum_probability= 0;

        for (int n = 0; n< num_classes; n++) {
            probability[n] = exp(features[n]- max_el);
            sum_probability += probability[n];
        }
        for (int n= 0; n <num_classes; n++) {
            probability[n]/= sum_probability; // normalise each probability score
        }
        return probability; // return the probability distribution [0.n, ..., ...,...]
    }

    vector<float> forward(const vector<float> &input) {
        vector<float> weighted_sum(num_classes, 0.0f);
        for (int n = 0; n< num_classes; n++) {
            weighted_sum[n] = bias[n];
            for (int i = 0; i < num_features; i++) {
                weighted_sum[n]+= weights[n][i]* input[i];
            }
        }
        return softmax_prediction(weighted_sum);
    }

    bool class_match(const vector<float> &dist, int label) {
        int pred_label = (int)(max_element(dist.begin(), dist.end())- dist.begin());
        return pred_label== label;
    }

    float correctness(const vector<Image> &batch_features, int &true_labels) {
        true_labels = 0;
        int batch_size = batch_features.size();

        for (int i = 0; i <batch_size; i++) {
            vector<float> output = forward(batch_features[i].image);
            if (class_match(output, batch_features[i].label)) {
                true_labels++;
            }
        }
        return static_cast<float>(true_labels)/ static_cast<float>(batch_size); //ration of correct scores
    }

 
    // Overload correctness method to call  in main function
    float correctness(const vector<Image> &batch_features) {
        int temp= 0;
        return correctness(batch_features,temp);
    }

    float train_batch(const vector<Image> &batch_features, int &true_labels) {
        vector<vector<float>> update_weights(num_classes, vector<float>(num_features, 0.f));
        vector<float> update_bias(num_classes, 0.f);

        true_labels = 0;
        float loss_function= 0.f;
        int batch_size= batch_features.size();
        float float_batch_size= static_cast<float>(batch_size);

        for (int i =0; i <batch_size; i++) {
            vector<float> output =forward(batch_features[i].image);

            int prediction =(int)(max_element(output.begin(), output.end())-output.begin());

            if (prediction== batch_features[i].label) {
                true_labels++;
            }

            loss_function += -log(max(output[batch_features[i].label], 1e-8f));

            for (int n = 0; n < num_classes; n++) {
                int true_label;

                if (n ==batch_features[i].label) {
                    true_label = 1;
                }
                else{
                    true_label = 0;
                }

                float error = output[n] -static_cast<float>(true_label);
                update_bias[n]+= error;

                for (int f= 0; f < num_features; f++) {
                    update_weights[n][f]+= error*batch_features[i].image[f];
                }
            }
        } // end batch training

        for (int i= 0; i <num_classes;i++) {
            bias[i] -= learning_rate * update_bias[i]/float_batch_size;
            for (int j = 0; j < num_features; j++) {
                weights[i][j] -=learning_rate*update_weights[i][j]/ float_batch_size;
            }
        }// end update weights and bias at gradient descent step

        return loss_function /float_batch_size;
    }



};

