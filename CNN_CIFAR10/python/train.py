from dataset import load_cifar10_batches_py, Dataset, DataLoader

input_features, input_labels = load_cifar10_batches_py("../dataset/cifar-10-batches-py")

# print(input_features.shape, input_labels.shape)

train_dataset = Dataset(input_features, input_labels)
train_loader = DataLoader(train_dataset, batch_size=32)

for train in train_loader:
    train_features, train_labels = train
    print(train_features.shape)