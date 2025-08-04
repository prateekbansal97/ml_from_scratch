//
// Created by Prateek Bansal on 8/1/25.
//
#include "Eigen/Dense"

Eigen::ArrayXXf ReLu(const Eigen::ArrayXXf& input)
{
    return input.max(0.0f);
}

Eigen::ArrayXXf softmax(const Eigen::ArrayXXf& X, bool clip = true) {
    Eigen::ArrayXXf X_shifted = X.colwise() - X.rowwise().maxCoeff();
    Eigen::ArrayXXf exps = X_shifted.exp();
    Eigen::ArrayXXf probs = exps.colwise() / exps.rowwise().sum();

    if (clip) {
        probs = probs.max(1e-10f).min(1.0f);
    }
    return probs;
}