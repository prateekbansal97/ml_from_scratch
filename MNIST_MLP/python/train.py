import numpy as np
import pickle
import os
import struct
import numpy as np
from dataset import Dataset, DataLoader, load_mnist_images_labels
from exceptions import ShapeMismatchError
from model import LinearLayer, MLP
from activation import ReLu, softmax
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

train_images, train_labels = load_mnist_images_labels("/home/prateek/storage/ml_from_scratch/mnist_imp/dataset/raw/train-images-idx3-ubyte", "/home/prateek/storage/ml_from_scratch/mnist_imp/dataset/raw/train-labels-idx1-ubyte")
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

train_dataset = Dataset(train_images, train_labels)
valid_dataset = Dataset(val_images, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32)
valid_loader = DataLoader(valid_dataset, batch_size=32)

model = MLP(LinearLayer(28*28, 128), LinearLayer(128, 32), LinearLayer(32, 10))
random_input = np.random.rand(32, 28*28)
num_epochs = 100 
num_classes = 10

alpha = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

val_loss_history, train_loss_history = [], []
best_val_loss = np.inf

for epoch in range(num_epochs):
    val_loss, train_loss = 0, 0
    correct_val, correct = 0, 0
    indices = np.arange(len(train_labels))
    indices_val = np.arange(len(val_labels))
    np.random.shuffle(indices)
    np.random.shuffle(indices_val)
    for batch_num, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
        images, labels = batch
        probs_train = model(images, ReLu, final_activation=softmax)

        current_batch_size = len(images)

        prediction = np.argmax(probs_train, axis=1)
        train_loss += np.sum(-np.log(probs_train[np.arange(current_batch_size), labels])) / current_batch_size

        one_hot_encoded = np.zeros((current_batch_size, num_classes))
        one_hot_encoded[np.arange(current_batch_size, dtype=np.int64), labels] = 1

        model.backward(train_loss, probs_train, one_hot_encoded, epoch, batch_num, train_loader.n_batches, images, beta1, beta2, alpha, eps)
        correct += np.sum(prediction == labels)

    for batch in valid_loader:
        images_val, labels_val = batch
        probs_val = model(images_val, ReLu, final_activation=softmax)

        current_batch_size = len(images_val)
        val_loss += np.sum(-np.log(probs_val[np.arange(current_batch_size), labels_val])) / current_batch_size

        pred_val = np.argmax(probs_val, axis=1)
        correct_val += np.sum(pred_val == labels_val)

    train_loss /= train_loader.n_batches
    val_loss /= valid_loader.n_batches

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_parameters = model.parameters
        print("Saving best model...")
        pickle.dump(best_parameters, open(f"Best_model_parameters_epoch_{epoch}.pkl", "wb"))
    print(f"Epoch {epoch+1:2d}: Train Loss = {train_loss:.4f}, Train Accuracy = {correct/len(train_labels)*100:.2f}%")
    print(f"Epoch {epoch+1:2d}: Validation Loss = {val_loss:.4f}, Validation Accuracy = {correct_val/len(val_labels)*100:.2f}%")

pickle.dump(train_loss_history, open(f"Train_Loss_{num_epochs}_epochs.pkl", "wb"))
pickle.dump(val_loss_history, open(f"Val_Loss_{num_epochs}_epochs.pkl", "wb"))
plt.plot(train_loss_history, label="Train")
plt.plot(val_loss_history, label="Val")
plt.legend()
plt.title("Loss vs Epoch")
plt.savefig("Loss_vs_epoch.png", dpi=300)

test_images, test_labels = load_mnist_images_labels("/home/prateek/storage/ml_from_scratch/mnist_imp/dataset/raw/t10k-images-idx3-ubyte", "/home/prateek/storage/ml_from_scratch/mnist_imp/dataset/raw/t10k-labels-idx1-ubyte")
test_dataset = Dataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32)

test_acc = 0
for x, y in test_loader:
    probs = model(x, ReLu, final_activation=softmax)
    preds = np.argmax(probs, axis=1)
    test_acc += np.sum(preds == y)

test_acc /= len(test_labels)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
