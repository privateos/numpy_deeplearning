import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import os

def get_mnist():
    current_file = os.path.relpath(__file__)
    current_path = os.path.split(current_file)[0]
    mnist_file = os.path.join(current_path, 'mnist.npz')
    data = np.load(mnist_file)
    x, y = data['x'], data['y']
    return x, y

class BP(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(BP, self).__init__()

        self.layer1 = nn.Linear(in_feature, 64)
        self.layer1_activation = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.layer2_activation = nn.ReLU()
        self.layer3 = nn.Linear(32, out_feature)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer1_activation(out)
        out = self.layer2(out)
        out = self.layer2_activation(out)
        out = self.layer3(out)
        return out

class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, y_hat, y_label):
        #print(y_hat.shape)
        log_softmax = F.log_softmax(y_hat)
        return F.nll_loss(log_softmax, y_label)

X, Y = get_mnist()
X = X/255.0

X = torch.from_numpy(X.reshape(X.shape[0], -1)).float()
Y = torch.tensor(np.argmax(Y, axis=1).tolist())

dataset = Data.TensorDataset(X, Y)
dataloader = Data.DataLoader(dataset, batch_size=100, shuffle=True)

network = BP(28*28, 10)
loss = SoftmaxCrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.01)#Adam
epochs = 40

for epoch in range(epochs):
    network.train()
    for (x, y_label) in dataloader:
        y_hat = network(x)
        loss_value = loss(y_hat, y_label)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    network.eval()

    y_hat = network(torch.autograd.Variable(X)).detach().numpy()
    y_hat = np.argmax(y_hat, axis=1)
    y_label = Y.detach().numpy()
    acc = y_hat == y_label
    acc = acc.mean()

    print(f'epoch={epoch}, accuracy={acc}')