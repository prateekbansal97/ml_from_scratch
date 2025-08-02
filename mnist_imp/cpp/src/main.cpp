
#include "dataset.h"
#include <Eigen/Dense>
#include <iostream>

int main()
{
    std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> input_data = load_mnist_images_labels("/Users/prateek/CLionProjects/ml_from_scratch/mnist_imp/dataset/raw/train-images-idx3-ubyte", "/Users/prateek/CLionProjects/ml_from_scratch/mnist_imp/dataset/raw/train-labels-idx1-ubyte");
    auto [train_features, valid_features] = train_test_split<Eigen::ArrayXXf>(input_data, 0.1f, false);
    auto [train_labels, valid_labels] = train_test_split<uint8_t>(input_data, 0.1f, true);
    Dataset train_dataset = Dataset(train_features, train_labels);
    Dataset valid_dataset = Dataset(valid_features, valid_labels);

//    std::cout << train_dataset.get_length() << std::endl;
    return 0;
}