import shap
import torch
import numpy as np

from os.path import exists

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

if not exists('shaps.pt'):

    e = shap.GradientExplainer(net, ds_train)

    shaps = torch.tensor(e.shap_values(ds_test))
    shaps = shaps.permute(1,0,2)

    predictions = torch.argmax(net(ds_test), dim=-1)

    # get only shapley values for f_i(x) where i is prediction
    tmp = []
    for i in range(len(ds_test)):
        s = shaps[i]
        s = s[predictions[i].item()].unsqueeze(0)
        tmp.append(s)
    shaps = torch.cat(tmp, 0)


    torch.save(shaps, 'shaps.pt')
else:
    shaps = torch.load('shaps.pt')

shaps = sum_shapley_values(shaps, indices_1h)
if shaps.requires_grad:
    shaps = shaps.detach().numpy()
else:
    shaps = shaps.numpy()
ds = a_test.get_original_features()
shap.summary_plot(shaps, ds)
