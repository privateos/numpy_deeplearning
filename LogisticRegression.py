import numpy as np


#Y.shape = (n, 1)
def sigmoid(x):
    return 1/(1 + np.exp(-x))
#df/dx = sigomid(x)*(1 - sigmoid(x))

def train(X, Y):
    n, d = X.shape
    _, m = Y.shape
    w = np.random.randn(d, m)
    b = np.random.randn(m)

    alpha = 0.1
    epochs = 100
    for i in range(epochs):
        Y_t = np.dot(X, w) + b
        Y_hat = sigmoid(Y_t)
        z = (Y_hat - Y)**2
        A = np.sum(z, axis=1)
        B = np.mean(A, axis=0)

        #w = w - alpha*dB_dW
        #b = b - alpha*dB_db
        dB_dA = np.ones_like(A)/n
        dB_dA = dB_dA.reshape(dB_dA.shape[0], 1)
        dB_dz = np.tile(dB_dA, (1, z.shape[1]))
        dB_dY_hat = dB_dz * 2*(Y_hat - Y)

        #这是新加的一行代码
        dY_hat_dY_t = Y_hat *(1 - Y_hat)
        dB_dY_t = dB_dY_hat * dY_hat_dY_t

        dB_dw = np.dot(np.transpose(X), dB_dY_t)
        dB_db = np.sum(dB_dY_t, axis=0)

        w = w - alpha*dB_dw
        b = b - alpha*dB_db
        print('loss = ' + str(B))
X = np.random.randn(100, 3)
Y = np.random.randn(100, 1)
train(X, Y)

print("start pytorch")
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()
model = nn.Linear(X.shape[1], Y.shape[1])
# print(model.weight, model.bias)
epochs=10
lr = 0.01
mse = nn.MSELoss()
# exit()
optimizer = optim.SGD(model.parameters(), lr=lr)
for i in range(epochs):
    Y_t = model(X)#X@W + b
    Y_hat = torch.sigmoid(Y_t)
    B = mse(Y, Y_hat)
    optimizer.zero_grad()
    B.backward()
    optimizer.step()
    print('loss = ' + str(B.item()))