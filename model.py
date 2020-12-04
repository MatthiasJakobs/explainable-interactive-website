import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path
import dice_ml
import pandas as pd

from dataset.adult import Adult, continous_columns
import matplotlib.pyplot as plt
from dataset.adult import Adult


class BaseModel(nn.Module):

    def predict (self, data):
        return torch.argmax(self.model.forward(data.torch()[0]), dim=-1).numpy() 

    def get_shap(self, data_rows, y, ds):
        shap_values = np.random.random((len(data_rows) , len(ds.get_column_names())))
        return shap_values

    def get_counterfactual(self, data_rows, y, ds):
        # TODO: What about y?
        #        - I think the model is called on X again, so no need to pass prediction in again?
        X, y = ds.pandas()
        df = pd.concat((X, y), axis=1)
        d = dice_ml.Data(dataframe=X, continuous_features=continous_columns, outcome_name='income')
        backend = 'PYT'
        m = dice_ml.Model(model=self, backend=backend)
        exp = dice_ml.Dice(d, m)

        instances = pd.DataFrame.to_dict(X.iloc[data_rows], orient='record')
        res = []
        for i in range(len(instances)):
            dice_exp = exp.generate_counterfactuals(instances[i], total_CFs=1, desired_class="opposite",
                                                    proximity_weight=0.5, diversity_weight=1, categorical_penalty=0.1, 
                                                    algorithm="DiverseCF", features_to_vary="all", yloss_type="hinge_loss", 
                                                    diversity_loss_type="dpp_style:inverse_dist", 
                                                    feature_weights="inverse_mad", optimizer="pytorch:adam", 
                                                    learning_rate=0.05, min_iter=500, max_iter=5000, project_iter=0, 
                                                    loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, 
                                                    init_near_query_instance=True, tie_random=False, 
                                                    stopping_threshold=0.5, posthoc_sparsity_param=0.1, 
                                                    posthoc_sparsity_algorithm="binary")
            res.append(dice_exp.final_cfs_df)
        return pd.concat(res).reset_index()
    

class FcNet(BaseModel):
    def __init__(self, checkpoint='fc_model.pt'):
        super(FcNet, self).__init__()
        self.build_model() 
        if os.path.isfile(checkpoint):
            self.model = torch.load(checkpoint)
        else:
            print('Model path does not exist!')
            
    def build_model(self):
        self.fc1 = nn.Linear(103, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(0.5)
        #self.fc3 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        #x = self.fc4(x)
        return x
        # return self.softmax(self.fc2(x))
        
    
class ConvNet(BaseModel):
    def __init__(self, checkpoint='conv_model.pt'):
        super(ConvNet, self).__init__()
        self.build_model() 
        if os.path.isfile(checkpoint):
            self.model = torch.load(checkpoint)
        else:
            print('Model path does not exist!')
            
    def build_model(self):
        self.conv1 = nn.Conv1d(103, 128, kernel_size=3)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
def train(model_name):
    learning_rate = 1e-3
    batch_size = 512
    epochs = 20

    ds_train = Adult('dataset', train=True)
    ds_test = Adult('dataset', train=False)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    if model_name == 'FcNet':
        path = 'fc_model.pt'
        net = FcNet()
    elif model_name == 'ConvNet':
        path = 'conv_model.pt'
        net = ConvNet()
       
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    logs_train_loss = []
    logs_val_loss = []
    for e in range(epochs):
        net.train()
        epoch_train_loss = 0.0
        for batch_idx, data in enumerate(dl_train):
            optimizer.zero_grad()
            x, y = data['x'], data['y']
            output = net(x)

            train_loss = criterion(output, y)
            epoch_train_loss += train_loss.item()

            train_loss.backward()
            optimizer.step()
        logs_train_loss.append(epoch_train_loss)

        net.eval()
        val_correct = 0
        val_accuracy = 0.0
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(dl_test):
                x, y = data['x'], data['y']
                output = net(x)
                
                val_loss = criterion(output, y)
                epoch_val_loss += val_loss.item()
                
                prediction = torch.argmax(output, axis=-1)
                val_correct += prediction.eq(y).sum().item()

            logs_val_loss.append(epoch_val_loss)

        print("Epoch {} train_loss {} val_accuracy {}".format(e, epoch_train_loss, val_correct / float(len(ds_test))))
    plotter(epochs, logs_train_loss, logs_val_loss)
    
    # torch.save(net.state_dict(), "fc_model.pth")
    torch.save(net, path)
        
    
def plotter(epochs, logs_train_loss, logs_val_loss):
    fig_1 = plt.figure()
    plt.plot(
        [epoch for epoch in range(epochs)], 
        [train_loss for train_loss in logs_train_loss])
    plt.plot(
        [epoch for epoch in range(epochs)], 
        [val_loss for val_loss in logs_val_loss])
    plt.legend(['total train loss', 'total val loss'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig_1.tight_layout()
    plt.grid(b=True, linestyle=':')

    plt.show()

if __name__ == "__main__":
    # train('FcNet')
    net = FcNet()
    data = Adult('', False, 500)
    predictions = net.predict(data)
