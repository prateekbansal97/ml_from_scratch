//
// Created by Prateek Bansal on 8/1/25.
//

#ifndef ML_FROM_SCRATCH_MODEL_H
#define ML_FROM_SCRATCH_MODEL_H

#include "Eigen/Dense"
#include <memory>

class BatchNorm {
public:
    BatchNorm(int dim);

    Eigen::ArrayXXf forward(const Eigen::ArrayXXf& input, bool training);
    Eigen::ArrayXXf backward(const Eigen::ArrayXXf& grad_output);
    
    // Parameters
    Eigen::ArrayXf gamma;
    Eigen::ArrayXf beta;

    // Running stats (used in inference)
    Eigen::ArrayXf running_mean;
    Eigen::ArrayXf running_var;

    // Saved for backward
    Eigen::ArrayXXf input_centered;
    Eigen::ArrayXf std_inv;
    Eigen::ArrayXXf x_hat;

    // Gradients
    Eigen::ArrayXf dgamma;
    Eigen::ArrayXf dbeta;

    float momentum = 0.1;
    float eps = 1e-5;

private:
    int dim;
};


class LinearLayer
{

    /* Linear Layer that performs a forward pass on a batch of input features.
     * Input : Input dimension (d_inp) and output dimension (d_out)
     * Comes with a set of learnable weights and biases that are tuned during backpropogation.
     * Adam Optimizer matrices are used to further optimize learning. 
     * Separate matrices model the momentum and velocity for the weights and biases.
     * first, intermediate and last layers indicate whether the layer is the first, intermediate (hidden) or last (output) layer. Useful for tracking during backpropogation.
     * output matrix stores the output of the layer during forward pass, which is used during backpropogation.
     * del_bias stores the gradients associated with the bias terms for the layer.
     */


public:
    LinearLayer(int d_inp, int d_out, bool use_bn = false);

    Eigen::ArrayXXf forward(Eigen::ArrayXXf& input, bool training);

    virtual ~LinearLayer() = default;

    std::shared_ptr<BatchNorm> batchnorm;
    bool use_batchnorm = false;

private:
    friend class MLP;
    int d_inp;
    int d_out;
//    self.weights = np.random.randn(self.d_inp, self.d_out) * np.sqrt(2 / self.d_inp)
    Eigen::ArrayXXf weights;
    Eigen::ArrayXf biases;
    Eigen::ArrayXXf dropout_mask;

    Eigen::ArrayXXf mom_weights;
    Eigen::ArrayXf mom_biases;
    Eigen::ArrayXXf vel_weights;
    Eigen::ArrayXf vel_biases;
    Eigen::ArrayXXf generate_dropout_mask(int rows, int cols, float dropout_rate);

//    std::shared_ptr<Eigen::ArrayXXf> output;
//    std::shared_ptr<Eigen::ArrayXXf> del_bias;

    Eigen::ArrayXXf output;
    Eigen::ArrayXXf del_bias;
    bool first;
    bool intermediate;
    bool last;
};


class MLP {

    /* Multi-Layer Perceptron using a vector of LinearLayer instances.
     * Every layer except the final uses the supplied custom activation function.
     * Final layer activation function is applied for the output.
     * Current implementation uses ReLu activation for all layers except the final layer.
     * Softmax is used in the last layer for obtaining the probability distribution.
     * parameters array is used for saving model parameters.
     */

public:
    MLP(std::vector<std::shared_ptr<LinearLayer>> layers);

    Eigen::ArrayXXf forward(const Eigen::ArrayXXf& X,
                            const std::function<Eigen::ArrayXXf(const Eigen::ArrayXXf&)>& activation,
                            const std::function<Eigen::ArrayXXf(const Eigen::ArrayXXf&)>& final_activation,
                            float dropout_rate, bool training, bool do_dropout = true);
    
    void backward(double train_loss, Eigen::ArrayXXf& probs, Eigen::ArrayXXf one_hot_encoded, int epoch,
                  int batch_num, int total_batches, Eigen::ArrayXXf batch_input, double beta1,
                  double beta2, double alpha, double eps);
    
    std::vector<std::pair<Eigen::ArrayXXf*, Eigen::ArrayXf*>> parameters;

    Eigen::ArrayXXf dropout(const Eigen::ArrayXXf& input, float dropout_rate, bool training);
    void save_parameters(
            const std::vector<std::pair<Eigen::ArrayXXf*, Eigen::ArrayXf*>>& parameters,
            const std::string& filename); // To save model parameters

private:
    std::vector<std::shared_ptr<LinearLayer>> layers;
    int n_layers;
};
#endif //ML_FROM_SCRATCH_MODEL_H
