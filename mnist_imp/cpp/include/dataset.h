//
// Created by Prateek Bansal on 8/1/25.
//

#ifndef ML_FROM_SCRATCH_DATASET_H
#define ML_FROM_SCRATCH_DATASET_H

#include <random>
#include <string>
#include <Eigen/Dense>
#include <array>

std::pair<Eigen::ArrayXXf, Eigen::ArrayXi> load_mnist_images_labels(const std::string& image_path, const std::string& label_path);

class Dataset {
    /*    def __init__(self, features, labels, scale_feat=True, scale_label=False):*/
public:
    Dataset(Eigen::ArrayXXf& features, Eigen::ArrayXi& labels, bool scale_feat = true);

    int get_length() const ;
    std::pair<Eigen::ArrayXf, int> get_element(int index);
    Eigen::ArrayXXf features;
    Eigen::ArrayXi labels;

private:
    bool scale_feat;
};

class DataLoader
{
public:
    DataLoader(Dataset& dataset, int batch_size);
    int get_length() const;
    std::pair<Eigen::ArrayXXf, Eigen::ArrayXi> get_batch(int index);
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

        std::pair<Eigen::ArrayXXf, Eigen::ArrayXi> operator*() {
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
    int n_batches;

private:
    Dataset& dataset;
    int batch_size;
    Eigen::ArrayXi indices;
};


std::pair<std::pair<Eigen::ArrayXXf, Eigen::ArrayXi>, 
          std::pair<Eigen::ArrayXXf, Eigen::ArrayXi>>  
train_test_split(const Eigen::ArrayXXf& features, const Eigen::ArrayXi& labels, float test_size, unsigned int seed = 42);

#endif //ML_FROM_SCRATCH_DATASET_H
