//
// Created by Prateek Bansal on 8/1/25.
//

#ifndef ML_FROM_SCRATCH_DATASET_H
#define ML_FROM_SCRATCH_DATASET_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <array>

std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> load_mnist_images_labels(const std::string& image_path, const std::string& label_path);

class Dataset {
    /*    def __init__(self, features, labels, scale_feat=True, scale_label=False):*/
public:
    Dataset(std::vector<Eigen::ArrayXXf>& features, std::vector<uint8_t>& labels, bool scale_feat = true);

    int get_length();
    std::pair<Eigen::ArrayXXf, uint8_t> get_element(int index);

private:
    std::vector<Eigen::ArrayXXf> features;
    const std::vector<uint8_t> labels;
    bool scale_feat;
};

/*train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)*/

std::pair<std::vector<Eigen::ArrayXXf>, std::vector<Eigen::ArrayXXf>> train_test_split_features(std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>>& input_data, float test_size);
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> train_test_split_labels(std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>>& input_data, float test_size);


#endif //ML_FROM_SCRATCH_DATASET_H
