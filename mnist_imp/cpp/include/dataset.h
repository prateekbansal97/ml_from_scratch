//
// Created by Prateek Bansal on 8/1/25.
//

#ifndef ML_FROM_SCRATCH_DATASET_H
#define ML_FROM_SCRATCH_DATASET_H

#include <vector>
#include <string>
#include <Eigen/Dense>

std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> load_mnist_images_labels(const std::string& image_path, const std::string& label_path);

class Dataset {
    /*    def __init__(self, features, labels, scale_feat=True, scale_label=False):*/
public:
    Dataset(std::vector<Eigen::ArrayXXf>& features, std::vector<uint8_t>& labels, bool scale_feat = true);


private:
    std::vector<Eigen::ArrayXXf> features;
    const std::vector<uint8_t> labels;
    bool scale_feat;
};


#endif //ML_FROM_SCRATCH_DATASET_H
