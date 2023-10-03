from model import NeuralNetwork
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

def predict_ANN(target = str, x_test = np.array):
    #restore the ANN model
    model = NeuralNetwork(fnn=5)
    model.load_state_dict(torch.load("all\\model_all_"+target+".pth"))
    model.eval()

    x_test = torch.from_numpy(x_test)
    x_test = x_test.float()
    with torch.no_grad():
        pred = model(x_test)

    scaler_read = open('all\\'+target+'_scaler.txt','r')
    y_scaler = float(scaler_read.readline())
    y_min = float(scaler_read.readline())
    scaler_read.close()
    pred = pred * y_scaler + y_min    
    pred = np.array(pred)
    if target == "Y":
        for i in range(pred.size):
            if pred[i] < 0:
                pred[i] = 0
            if pred[i] > 1:
                pred[i] = 1
    y_pred = []
    for i in range(pred.shape[0]):
        y_pred.append(pred[i][0])
    return y_pred

def predict_RF(target = str, x_test= np.array):
    #restore the RF model
    with open('all\\RF_'+target+'.pickle','rb') as f:
        rf = pickle.load(f)
    
    x_test = pd.DataFrame(x_test)
    pred = rf.predict(x_test)
    scaler_read = open('all\\'+target+'_scaler.txt','r')
    y_scaler = float(scaler_read.readline())
    y_min = float(scaler_read.readline())
    scaler_read.close()
    y_pred = pred * y_scaler + y_min
    y_pred = np.nan_to_num(y_pred.astype(np.float32)).flatten()
    return y_pred


read = pd.read_csv("all\\Tc_Tb.csv",usecols = ['name','Tc','Tb'])
#调取基础数据，沸点和临界温度。Get the data from the database, boiling point and critical temperature.

print('This is a T-xy prediction for binary mixtures, please input the first component in the mixture')
name1 = input()
df1 = read[(read['name'] == name1)]
df1 = np.array(df1)
df1 = df1.tolist()

if len(df1) == 0:
    print(f"the {name1} is not involved in the dataset, please input its critical temperature and its boiling point directly, critical temperature:")
    Tc1 = float(input())
    print(f"boiling point:")
    Tb1 = float(input())
else:
    Tc1 = float(df1[0][1])
    Tb1 = float(df1[0][2])

print('please input the second component')
name2 = input()

df2 = read[(read['name'] == name2)]
df2 = np.array(df2)
df2 = df2.tolist()
if len(df2) == 0:
    print(f"the {name2} is not involved in the dataset, please input its critical temperature and its boiling point directly, critical temperature:")
    Tc2 = float(input())
    print(f"boiling point:")
    Tb2 = float(input())
else:
    Tc2 = float(df2[0][1])
    Tb2 = float(df2[0][2])



x_test = []
for i in range(21):
    g = []
    g.append(Tc1)
    g.append(Tb1)
    g.append(Tc2)
    g.append(Tb2)
    g.append(i*0.05)
    x_test.append(g)

x_test = np.array(x_test)

x_scaler_read = open('all\\input_scaler.txt','r')
x_min = []
x_scaler = []

for i in range(5):
    x_scaler.append(float(x_scaler_read.readline()))
    x_min.append(float(x_scaler_read.readline()))

x_scaler = np.array(x_scaler)
x_min = np.array(x_min)
x_test = (x_test - x_min) / x_scaler

T_ANN = predict_ANN(target = "T", x_test=x_test)
Y_ANN = predict_ANN(target = "Y", x_test=x_test)
T_ANN = np.array(T_ANN)
Y_ANN = np.array(Y_ANN)

T_RF = predict_RF(target = "T", x_test=x_test)
Y_RF = predict_RF(target = "Y", x_test=x_test)
T_RF = np.array(T_RF)
Y_RF = np.array(Y_RF)

VLE = open(name1+'_'+name2+"_VLE_predict.csv","w")
VLE.write('ANN_all_results'+'\n')
VLE.write('index'+','+'x'+','+"T"+','+"Y"+","+'\n')
for i in range(T_ANN.shape[0]):
    VLE.write('"{}","{}","{}","{}"'.format(str(i+1),i*0.05,str(T_ANN[i]),str(Y_ANN[i])))
    VLE.write('\n')
VLE.write('\n'+'RF_all_results'+'\n')
VLE.write('index'+','+'x'+','+"T"+','+"Y"+","+'\n')
for i in range(T_RF.shape[0]):
    VLE.write('"{}","{}","{}","{}"'.format(str(i+1),i*0.05,str(T_RF[i]),str(Y_RF[i])))
    VLE.write('\n')
VLE.write('\n')

print('the result has been saved in '+name1+'_'+name2+"_VLE_predict.csv")
VLE.close()

    
