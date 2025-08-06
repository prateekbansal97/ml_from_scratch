//
// Created by Prateek Bansal on 8/1/25.
//

#ifndef ML_FROM_SCRATCH_MODEL_H
#define ML_FROM_SCRATCH_MODEL_H

#include "Eigen/Dense"
#include <memory>

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
    LinearLayer(int d_inp, int d_out);

    Eigen::ArrayXXf forward(Eigen::ArrayXXf& input);

    virtual ~LinearLayer() = default;

private:
    friend class MLP;
    int d_inp;
    int d_out;
//    self.weights = np.random.randn(self.d_inp, self.d_out) * np.sqrt(2 / self.d_inp)
    Eigen::ArrayXXf weights;
    Eigen::ArrayXf biases;


    Eigen::ArrayXXf mom_weights;
    Eigen::ArrayXf mom_biases;
    Eigen::ArrayXXf vel_weights;
    Eigen::ArrayXf vel_biases;

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
                            const std::function<Eigen::ArrayXXf(const Eigen::ArrayXXf&)>& final_activation);
    
    void backward(double train_loss, Eigen::ArrayXXf& probs, Eigen::ArrayXXf one_hot_encoded, int epoch,
                  int batch_num, int total_batches, Eigen::ArrayXXf batch_input, double beta1,
                  double beta2, double alpha, double eps);
    
    std::vector<std::pair<Eigen::ArrayXXf*, Eigen::ArrayXf*>> parameters;
    
    void save_parameters(
            const std::vector<std::pair<Eigen::ArrayXXf*, Eigen::ArrayXf*>>& parameters,
            const std::string& filename); // To save model parameters

private:
    std::vector<std::shared_ptr<LinearLayer>> layers;
    int n_layers;
};
#endif //ML_FROM_SCRATCH_MODEL_H
