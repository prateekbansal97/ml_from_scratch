//
// Created by Prateek Bansal on 8/1/25.
//

#ifndef ML_FROM_SCRATCH_MODEL_H
#define ML_FROM_SCRATCH_MODEL_H

#include "Eigen/Dense"

class LinearLayer
{
public:
    LinearLayer(int d_inp, int d_out);

    Eigen::ArrayXXf forward(Eigen::ArrayXXf& input);


private:
    int d_inp;
    int d_out;
//    self.weights = np.random.randn(self.d_inp, self.d_out) * np.sqrt(2 / self.d_inp)
    Eigen::ArrayXXf weights;
    Eigen::ArrayXf biases;
    bool first;
    bool intermediate;
    bool final;

    Eigen::ArrayXXf mom_weights;
    Eigen::ArrayXf mom_biases;
    Eigen::ArrayXXf vel_weights;
    Eigen::ArrayXf vel_biases;

    Eigen::ArrayXXf* output;
    Eigen::ArrayXXf* del_bias;

};


class MLP {
public:
    MLP(std::vector<std::shared_ptr<LinearLayer>> layers);
private:
    std::vector<std::shared_ptr<LinearLayer>> layers;
};
#endif //ML_FROM_SCRATCH_MODEL_H
