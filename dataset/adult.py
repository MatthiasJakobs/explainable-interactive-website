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
        adult_onehot = self.normalize(adult_onehot)

        self.train = train

        if subset_size is not None:
            self.subset_size = subset_size
        else:
            self.subset_size = len(adult_complete)

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
        for feature_name in continous_columns:
            mean = df[feature_name].mean()
            std = df[feature_name].std()
            result[feature_name] = (df[feature_name] - mean) / (std)
        return result

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
