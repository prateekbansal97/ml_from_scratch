//
// Created by Prateek Bansal on 8/1/25.
//

#include "dataset.h"
#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <algorithm> // For std::shuffle
#include <random>

std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> load_mnist_images_labels(const std::string& image_path, const std::string& label_path) {
    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file) throw std::runtime_error("Failed to open label file");

    uint32_t magic, num_labels;
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    std::vector<uint8_t> labels(num_labels);
    label_file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file) throw std::runtime_error("Failed to open image file");

    uint32_t num_images, rows, cols;
    image_file.read(reinterpret_cast<char*>(&magic), 4);
    image_file.read(reinterpret_cast<char*>(&num_images), 4);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::vector<Eigen::ArrayXXf> images;
    images.reserve(num_images);

    for (uint32_t i = 0; i < num_images; ++i) {
        Eigen::ArrayXXf img(rows, cols);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                uint8_t pixel;
                image_file.read(reinterpret_cast<char*>(&pixel), 1);
                img(r, c) = static_cast<float>(pixel);
            }
        }
        images.push_back(img);
    }

    return {images, labels};
}

Dataset::Dataset(std::vector<Eigen::ArrayXXf>& features, std::vector<uint8_t> &labels, bool scale_feat) :
features(std::move(features)), labels(std::move(labels)), scale_feat(scale_feat)
{
    if (scale_feat) {
        for (auto& img : this->features)
        {
            img /= 255.0;
        }
    }
}

std::pair<Eigen::ArrayXXf, uint8_t> Dataset::get_element(int index) {
    std::pair<Eigen::ArrayXXf, uint8_t> element;
    element.first = this->features[index];
    element.second = this->labels[index];
    return element;
}

int Dataset::get_length() {
    return static_cast<int>(this->features.size());
}

DataLoader::DataLoader(Dataset &dataset, int batch_size) : dataset(dataset), batch_size(batch_size)
{
    if (typeid(dataset) != typeid(Dataset))
    {
        throw std::invalid_argument("dataset should be of class Dataset");
    }

    indices = std::vector<int>(dataset.get_length());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd; // Obtain a random number from the OS
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    n_batches = static_cast<int>((std::ceil((this->dataset.get_length()) / this->batch_size)));
}

int DataLoader::get_length() const {
    return this->n_batches;
}

void DataLoader::shuffle() {
    std::random_device rd; // Obtain a random number from the OS
    std::mt19937 g(rd());
    std::shuffle(this->indices.begin(), this->indices.end(), g);
}


std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> DataLoader::get_batch(int index)
{
    int findex = index*this->batch_size;
    int lindex = std::min(findex + batch_size, this->dataset.get_length());
    int local_batch_size = lindex - findex;

    std::vector<Eigen::ArrayXXf> feature_batch;
    feature_batch.assign(this->dataset.features.begin() + findex, this->dataset.features.begin() + lindex);

    std::vector<uint8_t> label_batch;
    label_batch.assign(this->dataset.labels.begin() + findex, this-> dataset.labels.begin() + lindex);

    std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> batch = {feature_batch, label_batch};
    return batch;
}

Eigen::ArrayXXf flatten_and_stack(const std::vector<Eigen::ArrayXXf> &images) {
    int n_samples = static_cast<int>(images.size());
    int flattened_size = static_cast<int>(images[0].size());  // 28*28 = 784

    Eigen::ArrayXXf stacked(n_samples, flattened_size);

    for (int i = 0; i < n_samples; ++i) {

        if (images[i].size() != flattened_size) {
            std::cerr << "Image " << i << " has inconsistent size: "
                      << images[i].rows() << "x" << images[i].cols()
                      << " (expected flat size " << flattened_size << ")" << std::endl;
            std::exit(1);
        }

        if (images[i].data() == nullptr) {
            std::cerr << "Image " << i << " has null data pointer!" << std::endl;
            std::exit(1);
        }
// Flatten 2D image into row vector
        stacked.row(i) = Eigen::Map<const Eigen::RowVectorXf>(images[i].data(), flattened_size);
    }
    return stacked;
}
