import shap
import torch
import numpy as np

from os.path import exists
from shap.utils._legacy import DenseData

from model import Net
from dataset.adult import Adult, categorical_columns, column_names, continous_columns

def get_categorical_indices(df, categories):
    # get the indicies in the feature vector for each categorical variable.
    # This is necessary because categorical variables have been one-hot encoded
    # and need to be summed together later
    indices = []
    for cat in categories:
        start = 1e10
        stop = -1
        labels = df.columns
        for i, lab in enumerate(labels):
            if (cat + "_") in lab:
                start = min(start, i)
                stop = max(stop, i)
        indices.append([start, stop])
    return indices

def sum_shapley_values(shaps, indices):
    # assumption: shape (batch_size, dims)
    new_shaps = torch.zeros((len(shaps), len(column_names)-1))
    nr_continuous = len(continous_columns)
    for item in range(len(shaps)):
        s = shaps[item]
        for i in range(0, nr_continuous):
            new_shaps[item][i] = s[i]
        for i in range(len(indices)):
            start, stop = indices[i]
            sliced = s[start:stop]
            new_shaps[item][i+nr_continuous] = torch.sum(sliced)            

    return new_shaps

# Model wrapper for KernelExplainer
def f(x):
    net.eval()
    x = torch.tensor(x).float()
    with torch.no_grad():
        return net(x).numpy()

net = Net()
net.load_state_dict(torch.load("model.pth"))

a_train = Adult("dataset", train=True)
a_test = Adult("dataset", train=False)

np_test_x, np_test_y = a_test.numpy()

indices_1h = get_categorical_indices(a_train.pd_X_onehot, categorical_columns)

# Restrict to a random subset for faster computation (KernelShap is slow)
subset_size = 100
subset_indices = np.random.permutation(len(np_test_x))[:subset_size]

if not exists('shaps.pt'):

    # Wrap data in a DenseData object, which allows one-hot categorical values to be treated as a group
    group_names = [[0],[0],[0],[0],[0],[0]] + [list(range(end-start+1)) for start, end in indices_1h]
    
    # This is not an error. They look into the first element of *args for group names...
    data = DenseData(shap.sample(np_test_x), group_names, group_names)

    e = shap.KernelExplainer(f, data)

    shaps = torch.tensor(e.shap_values(np_test_x[subset_indices]))
    shaps = shaps.permute(1,0,2)

    predictions = torch.argmax(torch.tensor(f(np_test_x[subset_indices])).float(), dim=-1)

    # get only shapley values for f_i(x) where i is prediction
    tmp = []
    for i in range(subset_size):
        s = shaps[i]
        s = s[predictions[i].item()].unsqueeze(0)
        tmp.append(s)
    shaps = torch.cat(tmp, 0)

    torch.save(shaps, 'shaps.pt')
else:
    shaps = torch.load('shaps.pt')

if shaps.requires_grad:
    shaps = shaps.detach().numpy()
else:
    shaps = shaps.numpy()

shap.summary_plot(shaps, a_test.pd_X.iloc[subset_indices])
