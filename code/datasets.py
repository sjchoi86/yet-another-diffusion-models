import numpy as np
import torch
from sktime.datasets import load_from_arff_to_dataframe

class DatasetStandardizer:
    def __init__(self, axis=[0,1,2]):
        self.axis = axis
        pass

    def fit(self, dataset):
        self.mean = dataset[self.axis].mean()
        self.std = dataset[self.axis].std()

    def transform(self, dataset):
        y = dataset
        y = y - self.mean
        y = y / self.std
        return y

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def get_quadratic_function(x, M):
    '''
    x: [L x 1]
    M = number of trajectories
    '''
    if isinstance(x, np.ndarray):
        y = np.zeros(shape=(x.shape[0], M))

        for i in range(M):
            a = np.random.binomial(n=1, p=0.5, size=(1,)) * 2 - 1
            
            eps = np.random.normal(loc=0., scale=1., size=(1,))
            # y = a * (x[:] ** 2) + eps
            y[:, i] = a * (x.squeeze(-1) ** 2) + eps

    elif isinstance(x, torch.Tensor):
        y = torch.zeros(size=(x.shape[0], M), device=x.device)

        for i in range(M):
            a = torch.randint(low=0, high=2, size=(1,), device=x.device) * 2 - 1
            eps = torch.normal(mean=0., std=1., size=(1,), device=x.device)
            y[:, i] = a * (x[:] ** 2) + eps

    else:
        raise NotImplementedError('Unknown datatype for input x.')
    
    return y


def get_melbourne_dataset(p_train=0.8):
    # train_file_path = '/home/seungwoos/sp-sgm/data/MelbounePedetrian/MelbournePedestrian_TRAIN.arff'
    test_file_path = '/home/seungwoos/sp-sgm/data/MelbounePedetrian/MelbournePedestrian_TEST.arff'

    data = load_from_arff_to_dataframe(test_file_path)
    samples = []

    for i in range(data[0].shape[0]):
        samples.append(torch.tensor(data[0]['dim_0'][i]))

    y_full = torch.stack(samples, axis=0)
    ind = torch.isnan(y_full).sum(axis=1) == 0
    y_full = y_full[ind].float()
    x_full = torch.arange(y_full.shape[1], dtype=torch.float).unsqueeze(-1)

    num_train = int(y_full.shape[0] * p_train)

    train_indices = np.random.choice(y_full.shape[0], size=num_train, replace=False)
    eval_indices = np.setdiff1d(np.arange(y_full.shape[0]), train_indices)

    x_train = x_full.repeat(num_train, 1, 1)
    y_train = y_full[train_indices].unsqueeze(-1).transpose(1, 2)

    x_test = x_full.repeat(y_full.shape[0] - num_train, 1, 1)
    y_test = y_full[eval_indices].unsqueeze(-1).transpose(1, 2)

    scaler = DatasetStandardizer()
    
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)

    return x_train, y_train, x_test, y_test