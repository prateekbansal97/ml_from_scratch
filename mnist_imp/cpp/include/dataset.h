//
// Created by Prateek Bansal on 8/1/25.
//

#ifndef ML_FROM_SCRATCH_DATASET_H
#define ML_FROM_SCRATCH_DATASET_H

#include <vector>
#include <string>
#include <Eigen/Dense>

std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> load_mnist_images_labels(const std::string& image_path, const std::string& label_path);



#endif //ML_FROM_SCRATCH_DATASET_H
