import numpy as np

def train(X, Y):
    n, d = X.shape
    _, m = Y.shape
    w = np.random.randn(d, m)
    b = np.random.randn(m)

    alpha = 0.01
    epochs = 100
    for i in range(epochs):
        Y_hat = np.dot(X, w) + b
        z = (Y_hat - Y)**2
        A = np.sum(z, axis=1)
        B = np.mean(A, axis=0)

        #w = w - alpha*dB_dW
        #b = b - alpha*dB_db
        dB_dA = np.ones_like(A)/n
        dB_dA = dB_dA.reshape(dB_dA.shape[0], 1)
        dB_dz = np.tile(dB_dA, (1, z.shape[1]))
        dB_dY_hat = dB_dz * 2*(Y_hat - Y)
        dB_dw = np.dot(np.transpose(X), dB_dY_hat)
        dB_db = np.sum(dB_dY_hat, axis=0)

        w = w - alpha*dB_dw
        b = b - alpha*dB_db
        print('loss = ' + str(B))

X = np.random.randn(100, 3)
Y = np.random.randn(100, 4)
train(X, Y)

print('start pytorch')
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()
model = nn.Linear(X.shape[1], Y.shape[1])
mse = nn.MSELoss()
epochs=10
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=0.001)
for i in range(epochs):
    Y_hat = model(X)#X@W + b
    B = mse(Y, Y_hat)
    optimizer.zero_grad()
    B.backward()#反向求导
    optimizer.step()#更新参数
    print('loss = ' + str(B.item()))
