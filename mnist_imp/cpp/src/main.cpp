
//#include "dataset.h"
#include "include/dataset.h"
#include <Eigen/Dense>

int main()
{

    std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> input_data = load_mnist_images_labels("/Users/prateek/CLionProjects/ml_from_scratch/mnist_imp/dataset/raw/train-images-idx3-ubyte", "/Users/prateek/CLionProjects/ml_from_scratch/mnist_imp/dataset/raw/train-labels-idx1-ubyte");
    Dataset train_dataset = Dataset(input_data.first, input_data.second);

    return 0;
}