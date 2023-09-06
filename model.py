from traceback import print_list
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.optim import SGD, lr_scheduler

def norm(input):
    max = torch.max(input,dim=0).values
    min = torch.min(input,dim=0).values
    scaler = max - min
    scaler = torch.tensor([1 if x == 0 else x for x in scaler])
    output = (input - min) / scaler
    return output

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
    
    
def train(dataloader, model, loss_fn, optimizer,epoch, device, writer, loss_data):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #把loss写入tensorboard
        writer.add_scalar("Loss/train", loss, epoch)
        #反向传播预测误差以调整模型的参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch %  200 == 0:
            loss, current = loss.item(), batch * len(X)
            loss_data.write(str(epoch)+' '+str(loss)+'\n')
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation(dataloader, model, loss_fn, epoch, device, validationloss_data):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    validation_loss0 = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validation_loss0 += loss_fn(pred[0], y[0]).item()
    validation_loss0 /= num_batches
    validationloss_data.write(str(epoch)+' '+str(validation_loss0)+'\n')
    print(f"validation : \n validation_loss_T: {validation_loss0:>8f}")