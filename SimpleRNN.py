import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt

class Linear:
    def __init__(self, in_feature, out_feature):
        self.w = torch.autograd.Variable(torch.randn(in_feature, out_feature)/(in_feature + out_feature), requires_grad=True)
        self.b = torch.autograd.Variable(torch.randn(out_feature)/out_feature, requires_grad=True)
    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def parameters(self):
        return [self.w, self.b]

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1/(1 + torch.exp(-x))
    
    def parameters(self):
        return []

class Tanh:
    def __init__(self):
        pass
    def forward(self, x):
        return torch.tanh(x)
    def parameters(self):
        return []

class Relu:
    def __init__(self):
        pass
    def forward(self, x):
        zeros = torch.zeros_like(x)
        return torch.where(x > zeros, x, zeros)
    
    def parameters(self):
        return []

class Softmax:
    def __init__(self):
        pass
    def forward(self, x):
        exp_x = torch.exp(x)
        return exp_x/torch.sum(exp_x, dim=1, keepdim=True)

    def parameters(self):
        return []

class MSELoss:
    def __init__(self):
        pass

    def forward(self, y, y_hat):
        t = torch.sum((y-y_hat)**2, dim=1)
        t = torch.mean(t, dim=0)
        return t

    def parameters(self):
        return []

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.softmax = Softmax()

    def forward(self, y, y_hat):
        softmax_y_hat = self.softmax.forward(y_hat)

        t = y * torch.log(softmax_y_hat)
        s = torch.sum(t, dim=1)
        m = torch.mean(s, dim=0)
        return -m

    def parameters(self):
        return []

class SimpleRNN:
    #[batch_size, time, feature]
    def __init__(self, feature, hiddens):
        self.w = torch.autograd.Variable(torch.randn(feature, hiddens)/(feature + hiddens), requires_grad=True)
        self.u = torch.autograd.Variable(torch.randn(hiddens, hiddens)/(hiddens + hiddens), requires_grad=True)
        self.b = torch.autograd.Variable(torch.randn(hiddens)/hiddens, requires_grad=True)
    
    def forward(self, x):
        time = x.size(1)
        x_t = x[:, 0, :]
        h_list = []
        h_t = torch.tanh(torch.matmul(x_t, self.w) + self.b)
        h_list.append(h_t.unsqueeze(1))
        h_t_1 = h_t
        for i in range(1, time):
            x_t = x[:, i, :]
            h_t = torch.tanh(torch.matmul(x_t, self.w) + torch.matmul(h_t_1, self.u) + self.b)
        
            h_t_1 = h_t
            h_list.append(h_t.unsqueeze(1))

        Y = torch.cat(h_list, dim=1)
        return Y

    def parameters(self):
        return [self.w, self.u, self.b]

class Network:
    def __init__(self, in_feature, out_feature):
        self.rnn = SimpleRNN(in_feature, 16)
        self.linear = Linear(16*28, out_feature)
    def forward(self, x):
        rnn_out = self.rnn.forward(x)
        z = rnn_out.view(-1, 28*16)
        return self.linear.forward(z)
    
    def parameters(self):
       p = []
       p.extend(self.rnn.parameters())
       p.extend(self.linear.parameters())
       return p


class SGD:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.data.zero_()

    def step(self):
        lr = self.lr
        for p in self.params:
            p.data -= lr * p.grad.data

def get_mnist():
    current_file = os.path.relpath(__file__)
    current_path = os.path.split(current_file)[0]
    mnist_file = os.path.join(current_path, 'mnist.npz')
    data = np.load(mnist_file)
    x, y = data['x'], data['y']
    return x, y

def train_model():
    
    X,Y = get_mnist()
    X = X.reshape(X.shape[0], 28, 28)
    X = X/255.0
    #exit()
    #print(X.shape, Y.shape)
    #print(Y[0])
    #plt.imshow(X[0].reshape(28, 28))
    #plt.show()
    number = X.shape[0]
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    epochs = 100
    batch_size = 128
    lr = 0.01
    model = Network(28, 10)
    loss = MSELoss()
    #loss = SoftmaxCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    for i in range(epochs):
        begin = 0
        end = begin + batch_size

        while end < number:
            train_x = torch.autograd.Variable(X[begin:end])
            train_y = torch.autograd.Variable(Y[begin:end])

            y_hat = model.forward(train_x)
            loss_value = loss.forward(train_y, y_hat)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            begin = end
            end += batch_size
        
        y_hat = model.forward(X)
        y_hat = y_hat.detach().numpy()
        Y_t = Y.detach().numpy()
        y_hat = np.argmax(y_hat, axis=1)
        Y_t = np.argmax(Y_t, axis=1)
        acc = y_hat == Y_t
        acc = acc.mean()
        print('epochs = ' + str(i) + ' accuracy= ' + str(acc))

train_model()
