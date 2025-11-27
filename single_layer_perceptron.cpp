#include <iostream>
#include <vector>
#include <random>
#include <numeric> // for std::accumulate 
#include <algorithm> // for std::max

using namespace std;


random_device rd;
mt19937 generator(rd());

normal_distribution<double> distribution(0.0, 0.1); 

template <typename T>
class SingleLayerPerceptron {

private:
    size_t dim_;
    T bias_; 
    T learning_rate_;
    vector<T> weights_; 

public:
    
    SingleLayerPerceptron(size_t dim, T learning_rate)
    : dim_(dim), learning_rate_(learning_rate), weights_(dim, 0.0), bias_(0.0) {
        
        // Initializing weights with random noise
        for(size_t i = 0; i < dim_; i++){
            weights_[i] = distribution(generator);
        }
    }

    int activation(T z) const {
        return (z >= 0) ? 1 : -1; // Using classes +1 e -1.
    }

    // --- PREDICT (Forward Pass) ---
    T forward_pass(const vector<T>& input) const {
        // Z = W . X + B
        T activation_signal = bias_;

        for(size_t i = 0; i < dim_; i++){
            activation_signal += input[i] * weights_[i];
        }

        return activation_signal;
    }

    int predict(const vector<T>& input) const {
        return activation(forward_pass(input));
    }

    void train(const vector<vector<T>>& training_set, const vector<T>& answers, size_t epochs) {
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            
            bool errors_occurred = false;

            for(size_t i = 0; i < training_set.size(); ++i) { 
                const vector<T>& current_input = training_set[i];
                T target_output = answers[i];

                // FORWARD PASS
                T net_input = forward_pass(current_input);
                int predicted_output = activation(net_input);

               
                T error = target_output - predicted_output;

               // BACKWARD PASS 
                if (error != 0) {
                    errors_occurred = true;
                    
                   // weights adjustment
                    for(size_t j = 0; j < dim_; ++j) {
                        weights_[j] += learning_rate_ * error * current_input[j];
                    }

                    bias_ += learning_rate_ * error;
                }
            }
        }
    }

};
