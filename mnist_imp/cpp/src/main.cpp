
#include <memory>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>

#include "dataset.h"
#include "model.h"
#include "activation.h"



int main()
{
    //Load Dataset
    auto [input_features, input_labels] = load_mnist_images_labels("../mnist_imp/dataset/raw/train-images-idx3-ubyte", "../mnist_imp/dataset/raw/train-labels-idx1-ubyte");
    
    // Train, Validation split
    auto [train_valid_data, test_data] = train_test_split(input_features, input_labels, 0.1f);
    auto [train_valid_features, train_valid_labels] = train_valid_data; //train_test_split<uint8_t>(input_data, 0.1f, true);
    auto [test_features, test_labels] = test_data;

    auto [train_data, valid_data] = train_test_split(train_valid_features, train_valid_labels, 0.77f);
    auto [train_features, train_labels] = train_data;
    auto [valid_features, valid_labels] = valid_data;

    //Generate Dataset Instances
    Dataset train_dataset = Dataset(train_features, train_labels);
    Dataset valid_dataset = Dataset(valid_features, valid_labels);
    Dataset test_dataset = Dataset(test_features, test_labels);

    //Generate DataLoader Instances
    DataLoader train_loader = DataLoader(train_dataset, 32);
    DataLoader valid_loader = DataLoader(valid_dataset, 32);
    DataLoader test_loader = DataLoader(test_dataset, 32);

    //MLP layers vector
    std::vector<std::shared_ptr<LinearLayer>> layers = {
            std::make_shared<LinearLayer>(28 * 28, 128, true),
            std::make_shared<LinearLayer>(128, 32, true),
            std::make_shared<LinearLayer>(32, 10, false)
    };

    //Initializing MLP
    MLP model(layers);
    
    #ifdef DEBUG_TRAINING
    int num_epochs = 1;
    #else
    int num_epochs = 100;  
    #endif

    // 10 digits
    int num_classes = 10;

    // Adam optimizer parameters
    double alpha = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;

    // Tracking Loss
    std::vector<double> train_loss_history;
    std::vector<double> valid_loss_history;
    std::vector<float> train_acc_history;
    std::vector<float> valid_acc_history;

    // For tracking best model
    double best_val_loss = std::numeric_limits<double>::infinity();

    // Main Training Loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        
        double val_loss = 0.0f, train_loss = 0.0f;
        
        // To calculate train and valid accuracies
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
        
        for (const auto& batch: train_loader) // Using custom iterator class for DataLoader
        {

            auto& images_f = batch.first;
            auto& labels = batch.second;

            // forward pass
            Eigen::ArrayXXf probs_train = model.forward(images_f, ReLu, [](const Eigen::ArrayXXf& x) { return softmax(x, true); }, 0.3f, true, true);

            int current_batch_size = images_f.rows();

            
            Eigen::ArrayXi predictions(current_batch_size);
            
            // equivalent to np.argmax
            for (int i = 0; i < current_batch_size; ++i) {
                int maxIdx;
                probs_train.row(i).maxCoeff(&maxIdx);
                predictions(i) = maxIdx;
            }

            double batch_loss = 0.0;

//            #pragma omp parallel for reduction(+:batch_loss)
            for (int i = 0; i < current_batch_size; ++i) {
                uint8_t true_label = labels[i];  // Ground truth
                float prob = probs_train(i, true_label);

                // Safety: avoid log(0)
                prob = std::max(prob, 1e-10f);

                // Multi class Cross entropy
                batch_loss += -std::log(prob);
            }

            train_loss += batch_loss / current_batch_size;

            // BackPropogation
            Eigen::ArrayXXf one_hot_encoded = Eigen::ArrayXXf::Zero(current_batch_size, num_classes);

//            #pragma omp parallel for
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

            //            #pragma omp parallel for reduction(+:correct_train)
            for (int i = 0; i < current_batch_size; ++i) {
                if (predictions(i) == static_cast<int>(labels[i])) {
                    correct_train++;
                }
            }

            train_batch_num++;
        }

        // Validation
        for (const auto& batch_valid: valid_loader)
        {
            auto& images_valid_f = batch_valid.first;
            auto& labels_valid = batch_valid.second;

            // Forward Pass
            Eigen::ArrayXXf probs_valid = model.forward(images_valid_f, ReLu, [](const Eigen::ArrayXXf& x) { return softmax(x, true); }, 0.0f,
                                                        false, false);
            int current_batch_size = images_valid_f.rows();

            double batch_loss = 0.0;

            // equivalent to np.argmax
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

        train_acc_history.push_back(static_cast<float>(correct_train) / train_dataset.get_length());
        valid_acc_history.push_back(static_cast<float>(correct_val) / valid_dataset.get_length());
    }


    std::ofstream loss_file("loss_history.csv");
    loss_file << "epoch,train_loss,valid_loss\n";
    for (int i = 0; i < train_loss_history.size(); ++i) {
        loss_file << i + 1 << "," << train_loss_history[i] << "," << valid_loss_history[i] << "\n";
    }
    loss_file.close();

    std::ofstream acc_file("accuracy_history.csv");
    acc_file << "epoch,train_acc,valid_acc\n";
    for (int i = 0; i < train_acc_history.size(); ++i) {
        acc_file << i + 1 << "," << train_acc_history[i] << "," << valid_acc_history[i] << "\n";
    }
    acc_file.close();

    int correct_test = 0;
    // Test loop
    for (const auto& batch_test: test_loader)
    {
        auto& images_test_f = batch_test.first;
        auto& labels_test = batch_test.second;

        // Forward Pass
        Eigen::ArrayXXf probs_test = model.forward(images_test_f, ReLu, [](const Eigen::ArrayXXf& x) { return softmax(x, true); },
                                                   0.0f, false, false);
        int current_batch_size = images_test_f.rows();


        Eigen::ArrayXi predictions_test(current_batch_size);
        for (int i = 0; i < current_batch_size; ++i) {
            int maxIdx;
            probs_test.row(i).maxCoeff(&maxIdx);
            predictions_test(i) = maxIdx;
        }

        for (int i = 0; i < current_batch_size; ++i) {
            if (predictions_test(i) == static_cast<int>(labels_test[i])) {
                correct_test++;
            }
        }


    }

    std::cout << ", Test Accuracy = " << std::setprecision(2)
                             << (static_cast<float>(correct_test) / test_dataset.get_length()) * 100 << "%" << std::endl;


    return 0;
}
