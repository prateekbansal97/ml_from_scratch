//
// Created by Prateek Bansal on 8/1/25.
//

#include "model.h"
#include "Eigen/Dense"
#include "exception.h"
#include <cmath>
#include <fstream>

BatchNorm::BatchNorm(int dim) : dim(dim) {
    gamma = Eigen::ArrayXf::Ones(dim);
    beta = Eigen::ArrayXf::Zero(dim);
    running_mean = Eigen::ArrayXf::Zero(dim);
    running_var = Eigen::ArrayXf::Ones(dim);
}

Eigen::ArrayXXf BatchNorm::forward(const Eigen::ArrayXXf& input, bool training) {
    const int N = input.rows();

    if (training) {
        Eigen::ArrayXf batch_mean = input.colwise().mean();                    // (D)
        input_centered = input.rowwise() - batch_mean.transpose();             // (N,D)

        Eigen::ArrayXf batch_var =
                (input_centered.square().colwise().sum() / static_cast<float>(N)); // (D)

        // Update running stats
        running_mean = momentum * batch_mean + (1 - momentum) * running_mean;
        running_var  = momentum * batch_var  + (1 - momentum) * running_var;

        std_inv = (batch_var + eps).sqrt().inverse();                          // (D)

        // x_hat = centered * std_inv
        Eigen::ArrayXXf std_inv_row = std_inv.transpose().replicate(N, 1);     // (N,D)
        x_hat = input_centered * std_inv_row;                                  // (N,D)

        // y = gamma * x_hat + beta
        Eigen::ArrayXXf gamma_row = gamma.transpose().replicate(N, 1);         // (N,D)
        Eigen::ArrayXXf beta_row  = beta.transpose().replicate(N, 1);          // (N,D)
        return gamma_row * x_hat + beta_row;
    } else {
        Eigen::ArrayXXf centered =
                input.rowwise() - running_mean.transpose();                        // (N,D)
        Eigen::ArrayXXf inv_std_row =
                (running_var + eps).sqrt().inverse().transpose().replicate(N, 1);  // (N,D)
        Eigen::ArrayXXf xhat_infer = centered * inv_std_row;                   // (N,D)
        Eigen::ArrayXXf gamma_row  = gamma.transpose().replicate(N, 1);        // (N,D)
        Eigen::ArrayXXf beta_row   = beta.transpose().replicate(N, 1);         // (N,D)
        return gamma_row * xhat_infer + beta_row;
    }
}



Eigen::ArrayXXf BatchNorm::backward(const Eigen::ArrayXXf& grad_output) {
    const int N = grad_output.rows();

    // d_beta and d_gamma (vectors of length D)
    dbeta  = grad_output.colwise().sum();                 // (D)
    dgamma = (grad_output * x_hat).colwise().sum();       // (D)

    // scale = gamma * std_inv
    Eigen::ArrayXf scale = gamma * std_inv;               // (D)

    // Sums over batch
    Eigen::ArrayXf sum_dy      = grad_output.colwise().sum();           // (D)
    Eigen::ArrayXf sum_dy_xhat = (grad_output * x_hat).colwise().sum(); // (D)

    // Broadcast to (N,D)
    Eigen::ArrayXXf scale_row      = scale.transpose().replicate(N, 1);        // (N,D)
    Eigen::ArrayXXf sum_dy_row     = sum_dy.transpose().replicate(N, 1);       // (N,D)
    Eigen::ArrayXXf sum_dy_xhat_row= sum_dy_xhat.transpose().replicate(N, 1);  // (N,D)

    // dx = (1/N) * scale * (N*dy - sum(dy) - x_hat * sum(dy * x_hat))
    Eigen::ArrayXXf dx = (grad_output * static_cast<float>(N)
                          - sum_dy_row
                          - x_hat * sum_dy_xhat_row);

    dx = (dx * scale_row) / static_cast<float>(N); // (N,D)

    return dx;
}



LinearLayer::LinearLayer(int d_inp, int d_out, bool use_bn)
        : d_inp(d_inp), d_out(d_out), use_batchnorm(use_bn) {

    weights = Eigen::ArrayXXf::Random(d_inp, d_out) * std::sqrt(2.0f / d_inp);
    biases = Eigen::ArrayXf::Random(d_out);

    if (use_batchnorm)
        biases.setZero();

    mom_weights = Eigen::ArrayXXf::Zero(d_inp, d_out);
    mom_biases = Eigen::ArrayXf::Zero(d_out);
    vel_weights = Eigen::ArrayXXf::Zero(d_inp, d_out);
    vel_biases  = Eigen::ArrayXf::Zero(d_out);

    if (use_batchnorm) {

        batchnorm = std::make_shared<BatchNorm>(d_out);
    }
}


Eigen::ArrayXXf LinearLayer::forward(Eigen::ArrayXXf& input, bool training) {
    Eigen::ArrayXXf out = (input.matrix() * weights.matrix()).array().rowwise() + biases.transpose();
    if (use_batchnorm && batchnorm) {
        out = batchnorm->forward(out, training);
    }
    return out;
}

Eigen::ArrayXXf LinearLayer::generate_dropout_mask(int rows, int cols, float dropout_rate)
{
    Eigen::ArrayXXf random = (Eigen::ArrayXXf::Random(rows, cols) + 1.0f) / 2.0f; // [0,1]
    return (random > dropout_rate).cast<float>();
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
                        const std::function<Eigen::ArrayXXf(const Eigen::ArrayXXf&)>& final_activation,
                        float dropout_rate, bool training, bool do_dropout)
{
    Eigen::ArrayXXf out = X;
    for (int layernum = 0; layernum < n_layers - 1; layernum++)
    {
        const auto& layer = this->layers[layernum];
        out = layer->forward(out, training);
        out = activation(out);
        if (do_dropout && training && dropout_rate > 0.0f)
        {
            layer->dropout_mask = layer->generate_dropout_mask(out.rows(), out.cols(), dropout_rate);
            out = out * layer->dropout_mask / (1.0f - dropout_rate);
        }
        else
        {
            layer->dropout_mask.resize(0,0);
        }
        layer->output = out;
    }
    const auto& layerlast = this->layers[n_layers - 1];
    out = layerlast->forward(out, training);
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


        if (layer->dropout_mask.size() > 0)
        {
            del_bias *= layer->dropout_mask;
        }

        layer->del_bias = del_bias;


        if (layer->use_batchnorm && layer->batchnorm) {
            // First: apply ReLU mask already present in your code (done before this line)
            // Then backprop through BN
            del_bias = layer->batchnorm->backward(del_bias);

            // Simple SGD update for BN params (matches your alpha learning rate)
            layer->batchnorm->gamma -= alpha * layer->batchnorm->dgamma;
            layer->batchnorm->beta  -= alpha * layer->batchnorm->dbeta;
        }

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

        Eigen::ArrayXf grad_b = del_bias.colwise().sum();                 // (D)
        layer->mom_biases = layer->mom_biases*beta1 + (1 - beta1)*grad_b; // (D)
        Eigen::ArrayXf mom_cap_bias = layer->mom_biases/(1 - std::pow(beta1, t));

        Eigen::ArrayXf grad_b2 = (del_bias.square()).colwise().sum();     // (D)
        layer->vel_biases = layer->vel_biases*beta2 + (1 - beta2)*grad_b2;
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

