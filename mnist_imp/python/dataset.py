import struct
import numpy as np


def load_mnist_images_labels(image_path, label_path):
    with open(label_path, 'rb') as lbl_file:
        magic, num = struct.unpack(">II", lbl_file.read(8))
        labels = np.frombuffer(lbl_file.read(), dtype=np.uint8)

    with open(image_path, 'rb') as img_file:
        magic, num, rows, cols = struct.unpack(">IIII", img_file.read(16))
        images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(len(labels), rows, cols)

    return images, labels



class dataset:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        assert self.features.shape[0] == len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataLoader:
    def __init__(self, dataset, batch_size=1):
        if not isinstance(dataset, Dataset):
            raise ValueError("dataset should be of class Dataset")
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        np.random.shuffle(self.indices)
        self.n_batches = int(np.ceil(len(self.dataset) / self.batch_size))

    def __len__(self):
        return self.n_batches

    def get_batch(self, idx):
        findex = idx*self.batch_size
        lindex = min(findex + self.batch_size, len(self.dataset))
        local_batch_size = lindex - findex
        feature_batch = self.dataset.features[self.indices[findex:lindex]].reshape(local_batch_size, -1)
        labels_batch = self.dataset.labels[self.indices[findex:lindex]]
        return feature_batch, labels_batch

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def __iter__(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_batches):
            yield self.get_batch(i)
