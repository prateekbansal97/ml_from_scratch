
#include <memory>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

#include "dataset.h"
#include "model.h"
#include "activation.h"

int main()
{
    std::pair<std::vector<Eigen::ArrayXXf>, std::vector<uint8_t>> input_data = load_mnist_images_labels("../mnist_imp/dataset/raw/train-images-idx3-ubyte", "../mnist_imp/dataset/raw/train-labels-idx1-ubyte");
    auto [train_features, valid_features] = train_test_split<Eigen::ArrayXXf>(input_data, 0.1f, false);
    auto [train_labels, valid_labels] = train_test_split<uint8_t>(input_data, 0.1f, true);
    Dataset train_dataset = Dataset(train_features, train_labels);
    Dataset valid_dataset = Dataset(valid_features, valid_labels);

    DataLoader train_loader = DataLoader(train_dataset, 32);
    DataLoader valid_loader = DataLoader(valid_dataset, 32);

    std::vector<std::shared_ptr<LinearLayer>> layers = {
            std::make_shared<LinearLayer>(28 * 28, 128),
            std::make_shared<LinearLayer>(128, 32),
            std::make_shared<LinearLayer>(32, 10)
    };

    MLP model(layers);

    int num_epochs = 100;
    int num_classes = 10;

    double alpha = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;

    std::vector<double> train_loss_history;
    std::vector<double> valid_loss_history;

    double best_val_loss = std::numeric_limits<double>::infinity();

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double val_loss = 0.0f, train_loss = 0.0f;
        int correct_train = 0, correct_val = 0;

        std::vector<long int> indices(train_dataset.get_length());
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<long int> val_indices(valid_dataset.get_length());
        std::iota(val_indices.begin(), val_indices.end(), 0);

        std::random_device rd;   // Seed from hardware
        std::mt19937 g(rd());    // Mersenne Twister engine

        // Shuffle in place
        std::shuffle(indices.begin(), indices.end(), g);
        std::shuffle(val_indices.begin(), val_indices.end(), g);

        int train_batch_num = 0, valid_batch_num = 0;
        for (const auto& batch: train_loader)
        {
            auto& images = batch.first;
            auto& labels = batch.second;
            //auto start = std::chrono::high_resolution_clock::now();
            Eigen::ArrayXXf images_f = flatten_and_stack(images);
            //auto end = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> elapsed = end - start;
            //std::cout << "Time elapsed for train flattening: " << elapsed.count() << " seconds" << std::endl;

            Eigen::ArrayXXf probs_train = model.forward(images_f, ReLu, [](const Eigen::ArrayXXf& x) { return softmax(x, true); });

            int current_batch_size = images_f.rows();

            Eigen::ArrayXi predictions(current_batch_size);
            for (int i = 0; i < current_batch_size; ++i) {
                int maxIdx;
                probs_train.row(i).maxCoeff(&maxIdx);
                predictions(i) = maxIdx;
            }

            double batch_loss = 0.0;

            for (int i = 0; i < current_batch_size; ++i) {
                uint8_t true_label = labels[i];  // Ground truth
                float prob = probs_train(i, true_label);

                // Safety: avoid log(0)
                prob = std::max(prob, 1e-10f);

                batch_loss += -std::log(prob);
            }

            train_loss += batch_loss / current_batch_size;

            Eigen::ArrayXXf one_hot_encoded = Eigen::ArrayXXf::Zero(current_batch_size, num_classes);

            for (int i = 0; i < current_batch_size; ++i) {
                int label = static_cast<int>(labels[i]);

                if (label < 0 || label >= num_classes) {
                    std::cerr << "Invalid label " << label << " at index " << i << std::endl;
                    std::exit(1);
                }

                one_hot_encoded(i, label) = 1.0f;
            }

            model.backward(train_loss, probs_train, one_hot_encoded, epoch,
                           train_batch_num, train_loader.n_batches, images_f, beta1, beta2, alpha, eps);

            for (int i = 0; i < current_batch_size; ++i) {
                if (predictions(i) == static_cast<int>(labels[i])) {
                    correct_train++;
                }
            }
            train_batch_num++;
        }

        for (const auto& batch_valid: valid_loader)
        {
            auto& images_valid = batch_valid.first;
            auto& labels_valid = batch_valid.second;
            //auto start = std::chrono::high_resolution_clock::now();
            Eigen::ArrayXXf images_valid_f = flatten_and_stack(images_valid);
            //auto end = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> elapsed = end - start;
            //std::cout << "Time elapsed for valid flatten: " << elapsed.count() << " seconds" << std::endl;

            Eigen::ArrayXXf probs_valid = model.forward(images_valid_f, ReLu, [](const Eigen::ArrayXXf& x) { return softmax(x, true); });
            int current_batch_size = images_valid_f.rows();

            double batch_loss = 0.0;

            for (int i = 0; i < current_batch_size; ++i) {
                uint8_t true_label = labels_valid[i];  // Ground truth
                float prob = probs_valid(i, true_label);

                // Safety: avoid log(0)
                prob = std::max(prob, 1e-10f);

                batch_loss += -std::log(prob);
            }

            val_loss += batch_loss / current_batch_size;

            Eigen::ArrayXi predictions_val(current_batch_size);
            for (int i = 0; i < current_batch_size; ++i) {
                int maxIdx;
                probs_valid.row(i).maxCoeff(&maxIdx);
                predictions_val(i) = maxIdx;
            }

            for (int i = 0; i < current_batch_size; ++i) {
                if (predictions_val(i) == static_cast<int>(labels_valid[i])) {
                    correct_val++;
                }
            }

            valid_batch_num++;
        }

        train_loss /= train_loader.n_batches;
        val_loss /= valid_loader.n_batches;

        train_loss_history.emplace_back(train_loss);
        valid_loss_history.emplace_back(val_loss);

        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            std::vector<std::pair<Eigen::ArrayXXf *, Eigen::ArrayXf *>> best_parameters = model.parameters;

            std::cout << "Saving best model..." << std::endl;

             model.save_parameters(best_parameters, "best_parameters_epoch_" + std::to_string(epoch)+".pkl");
        }

        std::cout << "Epoch " << std::setw(2) << (epoch + 1)
                  << ": Train Loss = " << std::fixed << std::setprecision(4) << train_loss
                  << ", Train Accuracy = " << std::setprecision(2)
                  << (static_cast<float>(correct_train) / train_dataset.get_length()) * 100 << "%" << std::endl;

        std::cout << "Epoch " << std::setw(2) << (epoch + 1)
                  << ": Validation Loss = " << std::fixed << std::setprecision(4) << val_loss
                  << ", Validation Accuracy = " << std::setprecision(2)
                  << (static_cast<float>(correct_val) / valid_dataset.get_length()) * 100 << "%" << std::endl;

    }

//    std::cout << train_dataset.get_length() << std::endl;
    return 0;
}
