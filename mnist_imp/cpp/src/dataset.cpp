//
// Created by Prateek Bansal on 8/1/25.
//

#include "include/dataset.h"
#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <iostream>

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
                img(r, c) = static_cast<float>(pixel) / 255.0f;
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