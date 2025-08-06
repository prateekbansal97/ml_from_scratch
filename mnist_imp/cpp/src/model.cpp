//
// Created by Prateek Bansal on 8/1/25.
//

#include "model.h"
#include "Eigen/Dense"
#include "exception.h"
#include <cmath>
#include <fstream>

LinearLayer::LinearLayer(int d_inp, int d_out) : d_inp(d_inp), d_out(d_out)
{
    weights = Eigen::ArrayXXf::Random(d_inp, d_out) * std::sqrt(2.0f / d_inp); // He initialization
    biases = Eigen::ArrayXf::Random(d_out);
    
    // Useful during backpropogation, tracks position of layer in the network
    first = false;
    intermediate = false;
    last = false;
    
    // Adam Optimizer's momentum and velocities for updating model weights and biases
    mom_weights = Eigen::ArrayXXf::Zero(d_inp, d_out);
    mom_biases = Eigen::ArrayXf::Zero(d_out);
    
    vel_weights = Eigen::ArrayXXf::Zero(d_inp, d_out);
    vel_biases  = Eigen::ArrayXf::Zero(d_out);
}

Eigen::ArrayXXf LinearLayer::forward(Eigen::ArrayXXf &input) {
    // Linear transformation, y  = Ax + b;
    Eigen::ArrayXXf output_layer = (input.matrix() * this->weights.matrix()).array().rowwise() + this->biases.transpose();
    return output_layer;
}

MLP::MLP(std::vector<std::shared_ptr<LinearLayer>> layers) : layers(std::move(layers)) {

    int layernum = 0;

    // Initializing layers passed as layers within the MLP
    for (const auto& layer : this->layers) {
        if (layernum == 0) layer->first = true;
        else if (layernum == static_cast<int>(this->layers.size() - 1)) layer->last = true;
        else layer->intermediate = true;
        layernum++;
    }

    n_layers = static_cast<int>(this->layers.size());

    // Check if input of next layer has the shape as the output of the previous layer.
    for (int i = 0; i < n_layers - 1; i++) {
        const auto& layerprev = this->layers[i];
        const auto& layernext = this->layers[i+1];
        if (layerprev->d_out != layernext->d_inp)
        {
            throw ShapeMismatchError("Shape mismatch between output of layer" + std::to_string(i) + " : " + std::to_string(layerprev->d_out) + " and layer " + std::to_string(i + 1) + " : " + std::to_string(layernext->d_inp));
        }
        parameters.emplace_back(&(layerprev->weights), &(layerprev->biases));
    }
    const auto& last_layer = this->layers[n_layers - 1];
    parameters.emplace_back(&(last_layer->weights), &(last_layer->biases));
}

Eigen::ArrayXXf MLP::forward(const Eigen::ArrayXXf& X,
                        const std::function<Eigen::ArrayXXf(const Eigen::ArrayXXf&)>& activation,
                        const std::function<Eigen::ArrayXXf(const Eigen::ArrayXXf&)>& final_activation)
{
    Eigen::ArrayXXf out = X;
    for (int layernum = 0; layernum < n_layers - 1; layernum++)
    {
        const auto& layer = this->layers[layernum];
        out = layer->forward(out);
        out = activation(out);
        layer->output = out;
    }
    const auto& layerlast = this->layers[n_layers - 1];
    out = layerlast->forward(out);
    out = final_activation(out);
    layerlast->output = out;
    return out;

}

void MLP::backward(double train_loss, Eigen::ArrayXXf& probs, Eigen::ArrayXXf one_hot_encoded, int epoch, int batch_num,
                   int total_batches, Eigen::ArrayXXf batch_input, double beta1, double beta2, double alpha,
                   double eps) {

    std::vector<Eigen::ArrayXXf> original_weights;
    for (int i = 0; i < this->n_layers; i++)
    {
        const auto& layer = this->layers[i];
        original_weights.emplace_back(layer->weights);
    }

    long int t = epoch*total_batches + batch_num + 1;

    for (int i = this->n_layers - 1; i >= 0; i--)
    {
        int original_index = (int) i;
        const auto& layer = this->layers[i];
        Eigen::ArrayXXf del_bias;

        Eigen::ArrayXXf prev_layer_weights;
        if (i == n_layers - 1)
        {
            del_bias = probs - one_hot_encoded;
        }
        else
        {
            prev_layer_weights = original_weights[original_index + 1];
            del_bias = this->layers[original_index + 1]->del_bias.matrix() * prev_layer_weights.matrix().transpose();
            del_bias = (this->layers[original_index]->output == 0).select(0, del_bias);
        }

        layer->del_bias = del_bias;

        Eigen::ArrayXXf del_l_del_final;
        if (!layer->first)
        {
         del_l_del_final = (this->layers[original_index - 1]->output.matrix().transpose() * del_bias.matrix()).array();
        }
        else
        {
            del_l_del_final = (batch_input.matrix().transpose() * del_bias.matrix()).array();
        }

        float clip_norm = 5.0f;
        if (del_l_del_final.matrix().norm() > clip_norm)
            del_l_del_final = del_l_del_final * (clip_norm / del_l_del_final.matrix().norm());



        layer->mom_weights = layer->mom_weights*beta1 + (1 - beta1)*del_l_del_final;
        Eigen::ArrayXXf mom_cap_weights = layer->mom_weights/(1 - std::pow(beta1, t));

        layer->vel_weights = layer->vel_weights*beta2 + (1 - beta2)*del_l_del_final*del_l_del_final;
        Eigen::ArrayXXf vel_cap_weights = layer->vel_weights/(1 - std::pow(beta2, t));

        layer->weights = layer->weights - alpha*((mom_cap_weights)/(vel_cap_weights.sqrt() + eps));

        layer->mom_biases = layer->mom_biases*beta1 + (1 - beta1)*del_bias.sum();
        Eigen::ArrayXf mom_cap_bias = layer->mom_biases/(1 - std::pow(beta1, t));

        layer->vel_biases = layer->vel_biases*beta2 + (1 - beta2)*(del_bias*del_bias).sum();
        Eigen::ArrayXf vel_cap_bias = layer->vel_biases/(1 - std::pow(beta2, t));

        layer->biases = layer->biases - alpha*(mom_cap_bias)/(vel_cap_bias.sqrt() + eps);

        prev_layer_weights = layer->weights;

    }
}

void MLP::save_parameters(const std::vector<std::pair<Eigen::ArrayXXf *, Eigen::ArrayXf *>> &parameters,
                          const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open file for writing");

    int num_layers = static_cast<int>(parameters.size());
    out.write(reinterpret_cast<char*>(&num_layers), sizeof(int));

    for (const auto& [weights, biases] : parameters) {
        int rows = weights->rows(), cols = weights->cols();
        out.write(reinterpret_cast<char*>(&rows), sizeof(int));
        out.write(reinterpret_cast<char*>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char*>(weights->data()), sizeof(float) * rows * cols);

        int size = biases->size();
        out.write(reinterpret_cast<char*>(&size), sizeof(int));
        out.write(reinterpret_cast<const char*>(biases->data()), sizeof(float) * size);
    }
    out.close();
}

