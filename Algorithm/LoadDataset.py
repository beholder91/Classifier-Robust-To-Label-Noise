import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# split the train set into train and validation set
def train_val_random_split(X, S, train_ratio=0.8, random_seed=0):
    # set random seed
    np.random.seed(random_seed)

    # shuffle the data
    num_train = int(X.shape[0] * train_ratio)
    indices = np.random.permutation(X.shape[0])

    # split the data
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    return X[train_idx], S[train_idx], X[val_idx], S[val_idx]


class DatasetArray(Dataset):
    def __init__(self, X, Y=None, is_color=False, transform=None, to_onehot=False):
        """
        Args:
            X (numpy.ndarray): Image data of shape (num_samples, 28, 28) or (num_samples, 32, 32, 3)
            Y (numpy.ndarray, optional): Labels for the image data. Defaults to None.
            is_color (bool, optional): Whether the images are in color (3 channels). Defaults to False.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
            to_onehot (bool, optional): Whether to convert labels to one-hot vectors. Defaults to False.
        """

        self.X = np.asarray(X).astype(np.float32)
        
        # If is_color is False and data has 3 dimensions, expand to match PyTorch convention
        if not is_color and self.X.ndim == 3:
            self.X = np.expand_dims(self.X, axis=1)
        
        # If is_color is True and data has 4 dimensions, transpose to match PyTorch convention
        if is_color and self.X.ndim == 4:
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        
        if Y is not None:
            self.Y = np.asarray(Y).astype(np.int64)
        else:
            self.Y = None
        
        self.transform = transform
        self.to_onehot = to_onehot

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image_sample = self.X[index]
        
        if self.Y is not None:
            label_sample = self.Y[index]
            if self.to_onehot:
                onehot_label = np.zeros(3) # 3 classes classification problem
                onehot_label[label_sample] = 1
                label_sample = onehot_label
        else:
            label_sample = None
        
        if self.transform:
            image_sample = self.transform(torch.tensor(image_sample))
        
        return image_sample, label_sample


def load_data(path='../data/CIFAR.npz', train_ratio=0.8, random_seed=0, transform=None, batch_size=32, to_onehot=False, num_workers=0):
    """
    Prepares train and validation dataloaders from the given data.

    Args:
        train_ratio (float, optional): Ratio of training data. Defaults to 0.8.
        random_seed (int, optional): Random seed for shuffling, used for 10 times training.
        transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.

    Returns:
        DataLoader, DataLoader: Training and validation dataloaders.
    """
    
    data = np.load(path)
    # Split data into train and validation sets
    Xtr, Ytr, Xval, Yval = train_val_random_split(data['Xtr'], data['Str'], train_ratio, random_seed)  
    Xts, Yts = data['Xts'], data['Yts']

    # Check if data is color or grayscale
    if path == '../data/CIFAR.npz':
        is_color = True
    else:
        is_color = False

    # Create DatasetArray objects
    train_dataset = DatasetArray(Xtr, Ytr, is_color, transform, to_onehot=to_onehot)
    val_dataset = DatasetArray(Xval, Yval, is_color, transform, to_onehot=to_onehot)
    test_dataset = DatasetArray(Xts, Yts, is_color, transform, to_onehot=to_onehot)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

