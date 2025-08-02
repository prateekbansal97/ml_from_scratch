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
    std::vector<Eigen::ArrayXXf> features;
    const std::vector<uint8_t> labels;

private:
    bool scale_feat;
};

class DataLoader
{
public:
    DataLoader(Dataset& dataset, int batch_size);
    int get_length() const;
    std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> get_batch(int index);
    void shuffle();

    class Iterator {
    public:
        Iterator(DataLoader* loader, int pos) : loader(loader), pos(pos) {}

        bool operator!=(const Iterator& other) const {
            return pos != other.pos;
        }

        Iterator& operator++() {
            ++pos;
            return *this;
        }

        std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> operator*() {
            return loader->get_batch(pos);
        }

    private:
        DataLoader* loader;
        int pos;
    };

    Iterator begin() {
        shuffle();  // Reshuffle indices each time we begin iteration
        return Iterator(this, 0);
    }

    Iterator end() {
        return Iterator(this, n_batches);
    }

private:
    Dataset& dataset;
    int batch_size;
    std::vector<int> indices;
    int n_batches;
};
/*train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)*/

std::pair<std::vector<Eigen::ArrayXXf>, std::vector<Eigen::ArrayXXf>> train_test_split_features(std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>>& input_data, float test_size);
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> train_test_split_labels(std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>>& input_data, float test_size);

template<typename T>
std::pair<std::vector<T>, std::vector<T>> train_test_split(std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>>& input_data, float test_size, bool labels)
{
    const std::vector<T>* data_ptr;
    if (labels)
        data_ptr = reinterpret_cast<std::vector<T>*>(&input_data.second);
    else
        data_ptr = reinterpret_cast<std::vector<T>*>(&input_data.first);

    const auto& data = *data_ptr;

    std::pair<std::vector<T>, std::vector<T>> train_test_split_data;
    int final_train_index = static_cast<int>((1 - test_size) * data.size());

    train_test_split_data.first.assign(data.begin(), data.begin() + final_train_index);
    train_test_split_data.second.assign(data.begin() + final_train_index, data.end());

    return train_test_split_data;

}

#endif //ML_FROM_SCRATCH_DATASET_H
