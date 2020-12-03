# based on code from https://github.com/saravrajavelu/Adult-Income-Analysis/blob/master/Adult_Income_Analysis.ipynb

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from os.path import join

import pandas as pd
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
continous_columns = ['age', 'fnlwgt', 'educational-num', 'capital-gain','capital-loss', 'hours-per-week']


class Adult(Dataset):

    def __init__(self, path, train=True, subset_size=None, transform=None):
        # TODO: Remove lines with missing values
        ds_train = pd.read_csv('dataset/adult.data', sep=",\s", header=None, names = column_names, engine = 'python')
        ds_test = pd.read_csv('dataset/adult.test', sep=",\s", header=None, names = column_names, engine = 'python')
        ds_test['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')

        adult_complete = pd.concat([ds_test,ds_train])
        adult_complete['income'] = adult_complete['income'].astype('category').cat.codes
        adult_complete = self.normalize(adult_complete)

        adult_onehot = self.to_one_hot(adult_complete, categorical_columns)
        #adult_onehot = self.normalize(adult_onehot)

        self.train = train

        if subset_size is not None:
            self.subset_size = subset_size
        else:
            self.subset_size = len(adult_complete)

        np.random.seed(200)
        perm = np.random.permutation(len(adult_complete))[:self.subset_size]

        if self.train:
            complete_set = adult_complete.iloc[perm][len(ds_test):]
            onehot_set = adult_onehot.iloc[perm][len(ds_test):]
        else:
            complete_set = adult_complete.iloc[perm][:len(ds_test)]
            onehot_set = adult_onehot.iloc[perm][:len(ds_test)]

        self.pd_X = complete_set.drop(columns=['income'], axis=1)
        self.pd_y = complete_set['income']

        self.pd_X_onehot = onehot_set.drop(columns=['income'], axis=1)
        self.pd_y_onehot = complete_set['income']
        self.np_X_onehot = self.pd_X_onehot.to_numpy()
        self.np_y_onehot = self.pd_y_onehot.to_numpy()        
        self.pt_X_onehot = torch.tensor(self.pd_X_onehot.to_numpy()).float()
        self.pt_y_onehot = torch.tensor(self.pd_y_onehot.to_numpy()).long()

        del adult_complete
        del adult_onehot
        del ds_train
        del ds_test

    def numpy(self):
        return self.np_X_onehot, self.np_y_onehot

    def torch(self):
        return self.pt_X_onehot, self.pt_y_onehot

    def pandas(self, onehot=False):
        if onehot:
            return self.pd_X_onehot, self.pd_y_onehot
        else:
            return self.pd_X, self.pd_y

        # create data that can be directly used with Pytorch
        self.X_pth = torch.zeros(len(self), *self[0]['x'].shape)
        self.y_pth = torch.zeros(len(self), *self[0]['y'].shape)

        for i in range(len(self)):
            self.X_pth[i] = self[i]['x']
            self.y_pth[i] = self[i]['y']

    def normalize(self, df):
        result = df.copy()
        self.mean_std = {}
        for feature_name in continous_columns:
            mean = df[feature_name].mean()
            std = df[feature_name].std()
            self.mean_std[feature_name] = {}
            self.mean_std[feature_name]['mean'] = mean
            self.mean_std[feature_name]['std'] = std
            result[feature_name] = (df[feature_name] - mean) / (std)
        return result

    def normalize_single(self, x):
        continuous_indices = [0, 2, 4, 10, 11, 12]
        for i, c_indx in enumerate(continuous_indices):
            name = continous_columns[i]
            mean = self.mean_std[name]['mean']
            std = self.mean_std[name]['std']
            x[c_indx] = (x[c_indx] - mean) / (std)
        return x

    def denormalize(self, x):
        # only works with numpy and pytorch vectors
        nr_features = x.shape[-1]
        if isinstance(x, type(torch.zeros(1))):
            round_fn = torch.round
        else:
            round_fn = np.round

        if nr_features == 14:
            continuous_indices = [0, 2, 4, 10, 11, 12]
            for i, c_indx in enumerate(continuous_indices):
                name = continous_columns[i]
                mean = self.mean_std[name]['mean']
                std = self.mean_std[name]['std']
                x[c_indx] = round_fn(x[c_indx] * std + mean)
        # else:
        #     # one-hot vectors
        #     for i, name in enumerate(continous_columns):
        #         mean = self.mean_std[name]['mean']
        #         std = self.mean_std[name]['std']
        #         x[:, i] = round_fn(x[:, i] * std + mean)

        return x

    def to_one_hot(self, df, df_cols):
        df_1 = df.drop(columns=df_cols, axis=1)
        df_2 = pd.get_dummies(df[df_cols])
        return pd.concat([df_1, df_2], axis=1, join='inner')

    def get_original_features(self):
        # remove label
        return self.pd_X

    def __len__(self):
        return len(self.pd_X)

    def __getitem__(self, idx):
        return {'x': self.pt_X_onehot[idx], 'y': self.pt_y_onehot[idx]}

    def as_json(self, idx):
        return {key:str(value) for key, value in self.pd_X.iloc[idx].to_dict().items()}

    def get_categorical_choices(self):
        categorical_choices = {}
        for category in categorical_columns:
           categorical_choices[category] = self.pd_X[category].unique()
        return categorical_choices

    def get_column_names(self):
        return column_names[:-1]

    def get_categorical_column_names(self):
        return categorical_columns

    def get_continous_column_names(self):
        return continous_columns
