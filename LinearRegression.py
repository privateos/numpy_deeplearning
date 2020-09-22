import numpy as np

def train(X, Y, w, b):
    n, d = X.shape
    _, m = Y.shape
    # w = np.random.randn(d, m)
    # b = np.random.randn(m)

    alpha = 0.01
    epochs = 10
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
        print(dB_dw)
        print(dB_db)
        #break
        #exit()

        w = w - alpha*dB_dw
        b = b - alpha*dB_db
        print('loss = ' + str(B))
        input()

X = np.random.randn(100, 3)
Y = np.random.randn(100, 4)
np_w = np.random.randn(X.shape[1], Y.shape[1])
np_b = np.random.randn(Y.shape[1])
np_w_copy = np.copy(np_w)
np_b_copy = np.copy(np_b)
train(X, Y, np_w, np_b)

print('start pytorch')
import torch
import torch.nn as nn
import torch.optim as optim

epochs=10
lr = 0.01
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

w = torch.autograd.Variable(torch.from_numpy(np_w_copy).float(), requires_grad=True)
b = torch.autograd.Variable(torch.from_numpy(np_b_copy).float(), requires_grad=True)
X = torch.autograd.Variable(X)
Y = torch.autograd.Variable(Y)

for i in range(epochs):
    Y_hat = torch.matmul(X, w) + b
    z = (Y_hat - Y)**2
    A = torch.sum(z, dim=1)
    B = torch.mean(A, dim=0)

    B.backward()
    print('loss=' + str(B.item()))
    print(w.grad, b.grad)

    w.data = w.data - lr * w.grad.data
    b.data = b.data - lr* b.grad.data
    w.grad.data.zero_()
    b.grad.data.zero_()
    input()
# model = nn.Linear(X.shape[1], Y.shape[1])
# mse = nn.MSELoss()


# optimizer = optim.SGD(model.parameters(), lr=0.001)
# for i in range(epochs):
#     Y_hat = model(X)#X@W + b
#     B = mse(Y, Y_hat)
#     optimizer.zero_grad()
#     B.backward()#反向求导
#     optimizer.step()#更新参数
#     print('loss = ' + str(B.item()))
