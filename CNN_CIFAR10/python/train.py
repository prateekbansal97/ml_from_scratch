from dataset import load_cifar10_batches_py, Dataset, DataLoader
import numpy as np
from model import convolve
# input_features, input_labels = load_cifar10_batches_py("../dataset/cifar-10-batches-py")
#
# # print(input_features.shape, input_labels.shape)
#
# train_dataset = Dataset(input_features, input_labels)
# train_loader = DataLoader(train_dataset, batch_size=32)

p = np.ones((3, 32, 32))
q = np.ones((3, 5, 5))*2
out = convolve(p, q, 1, col_stride=1, row_stride=1)
print(out.shape)
# for train in train_loader:
#     train_features, train_labels = train
#     print(train_features.shape)