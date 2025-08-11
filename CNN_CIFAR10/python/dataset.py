import pickle
import numpy as np
import os

def load_cifar10_batches_py(folder_path):
    """Load CIFAR-10 dataset from the Python version pickled batches.

    Args:
        folder_path (str): Path to cifar-10-batches-py/ directory.

    Returns:
        images: np.ndarray, shape (50000, 3, 32, 32), dtype=np.uint8
        labels: np.ndarray, shape (50000,), dtype=np.uint8
    """
    images_list = []
    labels_list = []
    # Training batches: data_batch_1 .. data_batch_5
    for batch_id in range(1, 6):
        batch_path = os.path.join(folder_path, f"data_batch_{batch_id}")
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")  # dict keys are bytes
            data = batch[b"data"]  # shape (10000, 3072)
            labels = batch[b"labels"]
            # Reshape to (N, 3, 32, 32)
            data = data.reshape(-1, 3, 32, 32)
            images_list.append(data)
            labels_list.extend(labels)

    images = np.vstack(images_list).astype(np.uint8)
    labels = np.array(labels_list, dtype=np.uint8)
    return images, labels

class Dataset:
    def __init__(self, features, labels, scale_feat=True, scale_label=False):
        self.features = features
        self.labels = labels
        if scale_feat:
            self.features = self.features.astype(np.float32) / np.max(self.features)
        if scale_label:
            self.labels /= np.max(self.labels)
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
        feature_batch = self.dataset.features[self.indices[findex:lindex]]#.reshape(local_batch_size, -1)
        labels_batch = self.dataset.labels[self.indices[findex:lindex]]
        return feature_batch, labels_batch

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def __iter__(self):
        np.random.shuffle(self.indices)
        for i in range(self.n_batches):
            yield self.get_batch(i)