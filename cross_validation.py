import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.optim import SGD, lr_scheduler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import matplotlib.pyplot as plt

from data import mixtures_split,data_import
from model import NeuralNetwork,train, validation
from evaluate import RMSE_cal, MAE_cal, AADP_cal, R2_cal



def train_ANN(target = str, save_dir = str):

    writer = SummaryWriter()

    loss_data = open(save_dir+"\\ANN_epochs_loss_"+target+".txt", 'w')
    validationloss_data = open(save_dir+"\\ANN_val_loss_"+target+".txt", 'w')

    x_all, y_all, x_train, y_train, x_val, y_val, x_test, y_test, test_name = data_import(output='VLE_output_'+target+'.xlsx',save_dir=save_dir)
    x_1 = x_test
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val= torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    y_all= torch.from_numpy(y_all)

    y_max = torch.max(y_all,dim=0).values
    y_min = torch.min(y_all,dim=0).values
    y_scaler = y_max - y_min
    y_scaler = torch.tensor([1 if x == 0 else x for x in y_scaler])
    
    batch_size = 50

    training_data = TensorDataset(x_train, y_train) #将训练的input和output结合起来
    validation_data = TensorDataset(x_val, y_val)

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork(fnn = 23).to(device)

    #损失函数
    loss_fn = nn.MSELoss()
    #优化器
    optimizer = SGD(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [500,1000],gamma = 0.1)

    epochs = 1000
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(dataloader = train_dataloader,
            model = model,
            loss_fn = loss_fn,
            optimizer = optimizer,
            epoch = epoch,
            device = device,
            writer = writer,
            loss_data = loss_data)
        validation(dataloader = validation_dataloader,
                    model = model,
                    loss_fn = loss_fn,
                    epoch = epoch,
                    device = device,
                    validationloss_data = validationloss_data)
        scheduler.step()
    writer.flush()
    print("Done!")

    loss_data.close()
    validationloss_data.close()

    # 保存模型
    torch.save(model.state_dict(), save_dir+"\\model_"+target+".pth")

    model = NeuralNetwork(fnn = 21)
    model.load_state_dict(torch.load(save_dir+"\\model_"+target+".pth"))
    model.eval()

    # def norm(input):
    #     max = torch.max(input,dim=0).values
    #     min = torch.min(input,dim=0).values
    #     scaler = max - min
    #     scaler = torch.tensor([1 if x == 0 else x for x in scaler])
    #     output = (input - min) / scaler
    #     return output
    
    with torch.no_grad():
        pred = model(x_test)
        # pred = norm(pred)
    actual = y_test
    actual = actual * y_scaler + y_min
    

    pred = pred * y_scaler + y_min
    pred = pred.flatten()
    actual = actual.flatten()
    y_actual = np.array(actual)
    x_test = np.array(x_test.flatten)
    y_pred = np.array(pred)
    y_mean = np.mean(y_actual)
    
    if target == "Y":
        for i in range(y_pred.size):
            if y_pred[i] < 0:
                y_pred[i] = 0
            if y_pred[i] > 1:
                y_pred[i] = 1
        
    target_prediction = open(save_dir+"\\ANN_"+target+'_pred.csv','w')
    target_prediction.write('index'+','+'component1'+','+'component2'+','+'prediction value'+','+'actual value'+','+'x'+'\n')
    for i in range(y_pred.size):
        target_prediction.write('"{}","{}","{}","{}","{}","{}"'.format(str(i+1),test_name[i][0],test_name[i][1],
                                str(y_pred[i]),str(y_actual[i]),str(x_1[i][22])))
        target_prediction.write('\n')
        
    target_prediction.close()
    eval = open(save_dir+'\\eval_ANN_'+target+'.txt','w')

    RMSE = RMSE_cal(y_pred,y_actual)       
    eval.write('RMSE of '+target+':' + str(RMSE)+'\n')   

    MAE = MAE_cal(y_pred, y_actual)
    eval.write('MAE of '+target+':' + str(MAE)+'\n')
    if target == "T": 
        AADP = AADP_cal(y_pred,y_actual)
        eval.write('AADP of '+target+':' + str(AADP)+'\n')
    R2 = R2_cal(y_pred,y_actual,y_mean)
    eval.write('R2 of '+target+':' + str(R2)+'\n')

    eval.close()
    if target == "T":
        fig, ax = plt.subplots() 

        sizes = pred.size(0)

        x = np.arange(0,sizes)
        ax.scatter(y_actual ,y_pred , c='r')

        plt.xticks(np.linspace(300, 470, 10))
        plt.yticks(np.linspace(300, 470, 10))
        plt.plot([300,470],[300,470], c = 'SeaGreen')
    
        plt.xlabel('experimental data')
        plt.ylabel('T prediction value')
        plt.savefig(fname = save_dir+"\\T_ANN_pred.png")

    if target == "Y":
        fig, ax = plt.subplots() 

        sizes = pred.size(0)

        x = np.arange(0,sizes)
        ax.scatter(y_actual ,y_pred , c='b')

        plt.xticks(np.linspace(0, 1, 11))
        plt.yticks(np.linspace(0, 1.0, 8))
        plt.plot([0,1],[0,1], c = 'SeaGreen')
        # plt.title('result')
        plt.xlabel('data')
        plt.ylabel('prediction value')
        plt.legend()
        plt.savefig(fname = save_dir+"\\Y_ANN_pred.png")

def train_RF(target = str,save_dir = str):
        
    x_all, y_all, x_train, y_train, x_val, y_val, x_test, y_test, test_name = data_import(output='VLE_output_'+target+'.xlsx',save_dir=save_dir)

    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)


    j=200
    c=16
    z = 14
    a = 1
    b = 2
    regr = RandomForestRegressor(n_estimators=j, max_features= c, max_depth = z, min_samples_leaf=a, min_samples_split = b, random_state = 42)
    regr.fit(x_train, y_train.values.ravel())
    with open(save_dir+'\\RF_'+target+'.pickle','wb') as f:
            pickle.dump(regr,f)

    with open(save_dir+'\\RF_'+target+'.pickle','rb') as f:
        rf = pickle.load(f)

    y_max = y_all.max(0)
    y_min = y_all.min(0)
    y_scaler = y_max - y_min

    y_pred = rf.predict(x_test)
    y_pred = y_pred * y_scaler + y_min
    y_test = y_test * y_scaler + y_min
    y_test = np.nan_to_num(y_test.astype(np.float32))
    x_test = np.nan_to_num(x_test.astype(np.float32))
    y_pred = np.nan_to_num(y_pred.astype(np.float32))
    y_pred  = y_pred.flatten()
    y_test = y_test.flatten()

    y_mean = np.mean(y_test)

    target_prediction = open(save_dir+"\\RF_"+target+"_pred_RF.csv",'w')
    target_prediction.write('index'+','+'component1'+','+'component2'+','+'prediction value'+','+'actual value'+','+'x'+'\n')
    for i in range(y_pred.size):

        target_prediction.write('"{}","{}","{}","{}","{}","{}"'.format(str(i+1),
                                                                    test_name[i][0],
                                                                    test_name[i][1],
                                                                    str(y_pred[i]),
                                                                    str(y_test[i]),
                                                                    str(x_test[i][22])))
        target_prediction.write('\n')
    target_prediction.close()
    eval = open(save_dir+'\\eval_RF_'+target+'.txt','w')

    RMSE = RMSE_cal(y_pred,y_test)       
    eval.write('RMSE of '+target+':' + str(RMSE)+'\n')   

    MAE = MAE_cal(y_pred, y_test)
    eval.write('MAE of '+target+':' + str(MAE)+'\n')
    if target == "T": 
        AADP = AADP_cal(y_pred,y_test)
        eval.write('AADP of '+target+':' + str(AADP)+'\n')
    R2 = R2_cal(y_pred,y_test,y_mean)
    eval.write('R2 of '+target+':' + str(R2)+'\n')

    eval.close()

    if target == "T":
        fig, ax = plt.subplots() 

        sizes = y_pred.size

        x = np.arange(0,sizes)
        ax.scatter(y_test ,y_pred , c='r')

        plt.xticks(np.linspace(300, 470, 10))
        plt.yticks(np.linspace(300, 470, 10))
        plt.plot([300,470],[300,470], c = 'SeaGreen')
    
        plt.xlabel('experimental data')
        plt.ylabel('T prediction value')
        plt.savefig(fname = save_dir+"\\T_RF_pred.png")

    if target == "Y":
        fig, ax = plt.subplots() 

        sizes = y_pred.size

        x = np.arange(0,sizes)
        ax.scatter(y_test ,y_pred , c='b')

        plt.xticks(np.linspace(0, 1, 11))
        plt.yticks(np.linspace(0, 1.0, 8))
        plt.plot([0,1],[0,1], c = 'SeaGreen')

        plt.xlabel('data')
        plt.ylabel('prediction value')
        plt.legend()
        plt.savefig(fname = save_dir+"\\Y_RF_pred.png")

for i in range(10):
    save_dir = os.path.join(os.getcwd(),str(i+1))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    mixtures_split(save_dir = save_dir)
    train_ANN(target = "T",save_dir=save_dir)
    train_ANN(target = "Y",save_dir=save_dir)
    train_RF(target ="T",save_dir=save_dir)
    train_RF(target = "Y",save_dir=save_dir)
