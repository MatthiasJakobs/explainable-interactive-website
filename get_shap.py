import shap
import torch
import numpy as np

from model import Net
from dataset.adult import Adult
from dataset.preprocess_adult import categorical_columns, column_names, continous_columns

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
               
net = Net()
net.load_state_dict(torch.load("model.pth"))

a_train = Adult("dataset", train=True)
a_test = Adult("dataset", train=False)

ds_train = torch.zeros(len(a_train), *a_train[0]['x'].shape)
ds_test = torch.zeros(len(a_test), *a_test[0]['x'].shape)
labels_test = torch.zeros(len(a_test), *a_test[0]['y'].shape)

indices_1h = get_categorical_indices(a_train.X, categorical_columns)

for i in range(len(a_train)):
    ds_train[i] = a_train[i]['x']
for i in range(len(a_test)):
    ds_test[i] = a_test[i]['x']
    labels_test[i] = a_test[i]['y']

e = shap.GradientExplainer(net, ds_train)

# For testing: first 3 datapoints
datapoints_to_test = ds_test[:3]

shaps = torch.tensor(e.shap_values(datapoints_to_test))
shaps = shaps.permute(1,0,2)

predictions = torch.argmax(net(datapoints_to_test), dim=-1)

# get only shapley values for f_i(x) where i is prediction
tmp = []
for i in range(3):
    s = shaps[i]
    s = s[predictions[i].item()].unsqueeze(0)
    tmp.append(s)
shaps = torch.cat(tmp, 0)

# sum all shapley values on one-hot encoded categorical variables
shaps = sum_shapley_values(shaps, indices_1h)
print(shaps)
