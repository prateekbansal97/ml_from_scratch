//
// Created by Prateek Bansal on 8/1/25.
//

#include "model.h"
#include "Eigen/Dense"

LinearLayer::LinearLayer(int d_inp, int d_out) : d_inp(d_inp), d_out(d_out)
{
    weights =   Eigen::ArrayXXf::Random(d_inp, d_out);
    biases = Eigen::ArrayXf::Random(d_out);
    first = false;
    intermediate = false;
    final = false;
    
    mom_weights = Eigen::ArrayXXf::Zero(d_inp, d_out);
    mom_biases = Eigen::ArrayXf::Zero(d_out);
    
    vel_weights = Eigen::ArrayXXf::Zero(d_inp, d_out);
    vel_biases  = Eigen::ArrayXf::Zero(d_out);
    
    output = nullptr;
    del_bias = nullptr;
}

Eigen::ArrayXXf LinearLayer::forward(Eigen::ArrayXXf &input) {
    Eigen::ArrayXXf output_layer = (input.matrix() * this->weights.matrix()).array().rowwise() + this->biases.transpose();
    return output_layer;
}

MLP::MLP(std::vector<std::shared_ptr<LinearLayer>> layers) : layers(std::move(layers)) {}