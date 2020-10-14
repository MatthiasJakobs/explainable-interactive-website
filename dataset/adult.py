import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from os.path import join


class Adult(Dataset):

    def __init__(self, path, train=True, subset_size=None, transform=None):
        self.root_path = path
        self.train = train

        if self.train:
            self.data_path = join(self.root_path, 'adult_train_1h.pickle')
        else:
            self.data_path = join(self.root_path, 'adult_test_1h.pickle')

        self.data = pd.read_pickle(self.data_path)

        if subset_size is not None:
            self.subset_size = subset_size
        else:
            self.subset_size = len(self.data)

        perm = np.random.permutation(self.subset_size)
        self.data = self.data.iloc[list(perm)]
        self.y = self.data['income']
        self.X = self.data.drop(columns=['income'], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'x': torch.from_numpy(self.X.iloc[idx].to_numpy()).float(), 'y': torch.tensor(self.y.iloc[idx]).long()}

#a = Adult("/home/matty/code/interactive-website/dataset")

# class RescaleContinous(object):
#     """
#       Mean-center and unit variance for continous variables
#     """

#     def __init__(self, means, stddevs):
#         self.means = means
#         self.stddevs = stddevs
#         self.continous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain','capital-loss', 'hours-per-week'] # indices 0-5

#     def __call__(self, sample):
#         x = sample['x']
#         for i in range(len(self.continous_features)):
#             x[i] = (x[i] - self.means[i]) / self.stddevs[i]

#         return {'x': x, 'y': sample['y']}

