//
// Created by Prateek Bansal on 8/1/25.
//

#ifndef ML_FROM_SCRATCH_ACTIVATION_H
#define ML_FROM_SCRATCH_ACTIVATION_H

#include "Eigen/Dense"

Eigen::ArrayXXf ReLu(const Eigen::ArrayXXf& input);
Eigen::ArrayXXf softmax(const Eigen::ArrayXXf& X, bool clip = true); /* softmax with gradient clipping */
#endif //ML_FROM_SCRATCH_ACTIVATION_H
