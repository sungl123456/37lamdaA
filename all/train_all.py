# this file was developed to train the whole dataset with limited descriptors


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, lr_scheduler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def norm(input1):
        max = input1.max(0)
        min = input1.min(0)
        scaler = max - min
        out_put = (input1 - min) / scaler
        return out_put


def train_ANN_all(target = float):
    dfx = pd.read_excel('VLE_input.xlsx', index_col=0)
    x = np.array(dfx, dtype=np.float32)
    dfy = pd.read_excel('VLE_output_'+target+'.xlsx', index_col=0)
    y = np.array(dfy, dtype=np.float32)

    y_max = y.max(0)
    y_min = y.min(0)
    y_scaler = y_max - y_min
    target_scaler = open(target+'_scaler.txt','w')
    target_scaler.write(str(y_scaler[0])+'\n')
    target_scaler.write(str(y_min[0]))
    target_scaler.close()

    x = norm(x)
    y = norm(y)

    x_train = torch.from_numpy(x)
    y_train = torch.from_numpy(y)
    batch_size = 50
    training_data = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    class NeuralNetwork(nn.Module):
        def __init__(self,fnn):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.fnn = fnn
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.fnn, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork(fnn = 5).to(device)

    #损失函数
    loss_fn = nn.MSELoss()
    #优化器
    optimizer = SGD(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [500,1000],gamma = 0.1)

        
    def train(dataloader, model, loss_fn, optimizer,epoch):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch %  200 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    epochs = 1000
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer,epoch)
        scheduler.step()
    print("Done!")
    torch.save(model.state_dict(), "model_all_"+target+".pth")

def train_RF_all(target = float):
    dfx = pd.read_excel('VLE_input.xlsx', index_col=0)
    x = np.array(dfx, dtype=np.float32)
    dfy = pd.read_excel('VLE_output_'+target+'.xlsx', index_col=0)
    y = np.array(dfy, dtype=np.float32)
    x = norm(x)
    y = norm(y)
    x_train = pd.DataFrame(x)
    y_train = pd.DataFrame(y)
    j = 200
    c = 5
    z = 14
    a = 1
    b = 2

    regr = RandomForestRegressor(n_estimators=j, max_features= c, max_depth = z, min_samples_leaf=a, min_samples_split = b, random_state = 42)
    regr.fit(x_train, y_train.values.ravel())

    with open('RF_'+target+'.pickle','wb') as f:
        pickle.dump(regr,f)



dfx = pd.read_excel('VLE_input.xlsx', index_col=0)
x = np.array(dfx, dtype=np.float32)

x_max = x.max(0)
x_min = x.min(0)
x_scaler = x_max - x_min
input_scaler = open('input_scaler.txt','w')
for i in range(x_scaler.shape[0]):
    input_scaler.write(str(x_scaler[i])+'\n')
    input_scaler.write(str(x_min[i])+'\n')

input_scaler.close()

train_ANN_all(target="T")
train_RF_all(target="T")
train_ANN_all(target="Y")
train_RF_all(target="Y")