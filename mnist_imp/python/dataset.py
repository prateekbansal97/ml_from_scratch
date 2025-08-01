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
