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

std::pair<Eigen::ArrayXXf, Eigen::ArrayXi> load_mnist_images_labels(const std::string& image_path, const std::string& label_path) {
    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file) throw std::runtime_error("Failed to open label file");

    uint32_t magic, num_labels;
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    Eigen::ArrayXi labels(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        uint8_t lbl;
        label_file.read(reinterpret_cast<char*>(&lbl), 1);
        labels(i) = static_cast<int>(lbl);
    }

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

    Eigen::ArrayXXf images(num_images, rows * cols);
    //images.reserve(num_images);

    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t p = 0; p < rows * cols; ++p) {
            uint8_t pixel;
            image_file.read(reinterpret_cast<char*>(&pixel), 1);
            images(i, p) = static_cast<float>(pixel); // optional: /255.0f to normalize
        }
    }

    return {images, labels};
}

Dataset::Dataset(Eigen::ArrayXXf& features, Eigen::ArrayXi& labels, bool scale_feat) :
features(std::move(features)), labels(std::move(labels)), scale_feat(scale_feat)
{
    if (scale_feat) {
          this -> features /= 255.0;
    }
}

std::pair<Eigen::ArrayXf, int> Dataset::get_element(int index)  {
    return { features.row(index), labels(index) };
}

int Dataset::get_length() const {
    return static_cast<int>(this->features.rows());
}


DataLoader::DataLoader(Dataset& dataset, int batch_size)
    : dataset(dataset), batch_size(batch_size)
{
    if (batch_size <= 0) {
        throw std::invalid_argument("batch_size must be positive");
    }

    int length = dataset.get_length();

    // Allocate and initialize indices as Eigen::ArrayXi
    indices = Eigen::ArrayXi::LinSpaced(length, 0, length - 1);

    // Shuffle using Eigen and STL-compatible iterators
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.data(), indices.data() + indices.size(), g);

    // Calculate number of batches correctly
    n_batches = (length + batch_size - 1) / batch_size;
}


int DataLoader::get_length() const {
    return this->n_batches;
}

void DataLoader::shuffle() {
    std::random_device rd; // Obtain a random number from the OS
    std::mt19937 g(rd());
    std::shuffle(this->indices.begin(), this->indices.end(), g);
}


std::pair<Eigen::ArrayXXf, Eigen::ArrayXi> DataLoader::get_batch(int index)
{
    int findex = index*this->batch_size;
    int lindex = std::min(findex + batch_size, this->dataset.get_length());
    int local_batch_size = lindex - findex;

    Eigen::ArrayXXf feature_batch(local_batch_size, dataset.features.cols());

    
    //feature_batch.assign(this->dataset.features.begin() + findex, this->dataset.features.begin() + lindex);

    Eigen::ArrayXi label_batch(local_batch_size);
    //label_batch.assign(this->dataset.labels.begin() + findex, this-> dataset.labels.begin() + lindex);

    for (int i = 0; i < local_batch_size; ++i) {
        int data_index = indices(findex + i);
        feature_batch.row(i) = dataset.features.row(data_index);
        label_batch(i) = dataset.labels(data_index);
    }

    return {feature_batch, label_batch};
}

std::pair<std::pair<Eigen::ArrayXXf, Eigen::ArrayXi>,
          std::pair<Eigen::ArrayXXf, Eigen::ArrayXi>>
train_test_split(const Eigen::ArrayXXf& features, const Eigen::ArrayXi& labels, float test_size, unsigned int seed)
{
    if (features.rows() != labels.rows()) {
        throw std::invalid_argument("Number of feature rows and labels must match.");
    }
    if (test_size <= 0.0f || test_size >= 1.0f) {
        throw std::invalid_argument("test_size must be in (0, 1).");
    }

    int n_samples = static_cast<int>(features.rows());
    int n_test = static_cast<int>(n_samples * test_size);
    int n_train = n_samples - n_test;

    // Generate shuffled indices
    Eigen::ArrayXi indices = Eigen::ArrayXi::LinSpaced(n_samples, 0, n_samples - 1);
    std::mt19937 rng(seed);
    std::shuffle(indices.data(), indices.data() + n_samples, rng);

    // Allocate splits
    Eigen::ArrayXXf X_train(n_train, features.cols());
    Eigen::ArrayXXf X_test(n_test, features.cols());
    Eigen::ArrayXi y_train(n_train);
    Eigen::ArrayXi y_test(n_test);

    // Fill train/test sets
    for (int i = 0; i < n_train; ++i) {
        X_train.row(i) = features.row(indices(i));
        y_train(i) = labels(indices(i));
    }
    for (int i = 0; i < n_test; ++i) {
        X_test.row(i) = features.row(indices(n_train + i));
        y_test(i) = labels(indices(n_train + i));
    }

    return {{X_train, y_train}, {X_test, y_test}};
}
