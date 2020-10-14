import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.adult import Adult

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(108, 256)
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.softmax(self.fc2(x))

learning_rate = 1e-2
batch_size = 128
epochs = 5

ds_train = Adult('dataset', train=True)
ds_test = Adult('dataset', train=False)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for e in range(epochs):
    net.train()
    epoch_loss = 0.0
    for batch_idx, data in enumerate(dl_train):
        optimizer.zero_grad()
        x, y = data['x'], data['y']
        output = net(x)

        loss = criterion(output, y)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()


    net.eval()
    val_correct = 0
    val_accuracy = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(dl_test):
            x, y = data['x'], data['y']
            output = net(x)
            prediction = torch.argmax(output, axis=-1)

            val_correct += prediction.eq(y).sum().item()

    print("Epoch {} train_loss {} val_accuracy {}".format(e, epoch_loss, val_correct / float(len(ds_test))))

torch.save(net.state_dict, "model.pth")