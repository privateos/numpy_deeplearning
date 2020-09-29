import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import os

class Network(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Network, self).__init__()
        self.rnn1 = nn.LSTM(in_feature, 16, batch_first=True)
        self.linear = nn.Linear(16*28, out_feature)
    def forward(self, x):
        rnn_out1, (ht, ct) = self.rnn1(x)
        z = rnn_out1.reshape(-1, 28*16)
        return self.linear(z)

class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, y_hat, y_label):
        #print(y_hat.shape)
        log_softmax = F.log_softmax(y_hat)
        return F.nll_loss(log_softmax, y_label)

def get_mnist():
    current_file = os.path.relpath(__file__)
    current_path = os.path.split(current_file)[0]
    mnist_file = os.path.join(current_path, 'mnist.npz')
    data = np.load(mnist_file)
    x, y = data['x'], data['y']
    return x, y

def train_model():
    
    X,Y = get_mnist()
    X = X.reshape(X.shape[0], 28, 28)#(N, 28, 28)
    X = X/255.0
    #exit()
    #print(X.shape, Y.shape)
    #print(Y[0])
    #plt.imshow(X[0].reshape(28, 28))
    #plt.show()
    number = X.shape[0]
    X = torch.from_numpy(X.reshape(X.shape[0], 28, 28)).float()
    Y = torch.tensor(np.argmax(Y, axis=1).tolist())
    epochs = 20
    batch_size = 128
    dataset = Data.TensorDataset(X, Y)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    lr = 0.01
    model = Network(28, 10)
    loss = SoftmaxCrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for (x, y_label) in dataloader:
            y_hat = model(x)
            loss_value = loss(y_hat, y_label)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        model.eval()

        y_hat = model(torch.autograd.Variable(X)).detach().numpy()
        y_hat = np.argmax(y_hat, axis=1)
        y_label = Y.detach().numpy()
        acc = y_hat == y_label
        acc = acc.mean()

        print(f'epoch={epoch}, accuracy={acc}')

train_model()
