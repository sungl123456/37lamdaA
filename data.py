import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

def mixtures_split(save_dir):
    components_train = open(save_dir+"\\components_train.txt","w")
    components_validation = open(save_dir+"\\components_validation.txt","w")
    components_test = open(save_dir+"\\components_test.txt","w")
    components = pd.read_excel('mixtures.xlsx', usecols = ['Substance 1','Substance 2'])
    comp = np.array(components)
    train_size = math.floor(len(comp)*0.6)
    val_size = math.floor(len(comp)*0.8)
    
    c = np.array(comp)
    s = set()
    for t in c:
        t=tuple(t)
        s.add(t)
    mixture_list = np.array(list(s))
    i = 0
    for i in range(train_size):
        components_train.write(str(mixture_list[i])+'\n')

    for i in range(train_size, val_size):
        components_validation.write(str(mixture_list[i])+'\n')

    for i in range(val_size, len(comp)):
        components_test.write(str(mixture_list[i])+'\n')
    components_train.close()
    components_validation.close()
    components_test.close()


def norm(input1):
    max = input1.max(0)
    min = input1.min(0)
    scaler = max - min
    out_put = (input1 - min) / scaler
    return out_put



def data_import(input = 'VLE_input.xlsx',
                output = 'VLE_output_T.xlsx',
                components_read = 'dataset.xlsx',
                save_dir = None):
    train_mixtures_read = save_dir+'\\components_train.txt'
    val_mixtures_read = save_dir+'\\components_validation.txt'
    test_mixtures_read = save_dir+'\\components_test.txt'

    dfx = pd.read_excel(input,index_col = 0)
    x_all = np.array(dfx, dtype=np.float32)
    dfy = pd.read_excel(output, index_col=0)
    y_all = np.array(dfy, dtype=np.float32)


    x = norm(x_all)
    y = norm(y_all)

    components = pd.read_excel(components_read, usecols =['Substance 1','Substance 2'])
    comp = np.array(components)
    components_train = open(train_mixtures_read,"r")
    components_val = open(val_mixtures_read,'r')
    components_test = open(test_mixtures_read,"r")

    c_train_line = components_train.read().splitlines()
    c_val_line = components_val.read().splitlines()
    c_test_line = components_test.read().splitlines()
    c_train = []
    c_val = []
    c_test = []

    for each_line in c_train_line:
        a = each_line.split("\'",1)
        b = a[1].split("\'",1)
        c = b[1].split("\'",1)
        d = c[1].split("\'",1)
        e = []
        e.append(b[0])
        e.append(d[0])
        f=[]
        f.append(d[0])
        f.append(b[0])
        c_train.append(e)
        c_train.append(f)

    for each_line in c_val_line:
        a = each_line.split("\'",1)
        b = a[1].split("\'",1)
        c = b[1].split("\'",1)
        d = c[1].split("\'",1)
        e = []
        e.append(b[0])
        e.append(d[0])
        f=[]
        f.append(d[0])
        f.append(b[0])
        c_val.append(e)
        c_val.append(f)

    for each_line in c_test_line:
        a = each_line.split("\'",1)
        b = a[1].split("\'",1)
        c = b[1].split("\'",1)
        d = c[1].split("\'",1)
        e = []
        e.append(b[0])
        e.append(d[0])
        f=[]
        f.append(d[0])
        f.append(b[0])
        c_test.append(e)
        c_test.append(f)
    

    x_train = []
    y_train = []

    for i in range(len(c_train)):
        for j in range(len(comp)):
            if comp[j][0] == c_train[i][0]:
                if comp[j][1] == c_train[i][1]:
                    x_train.append(list(x[j]))
                    y_train.append(list(y[j]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # 把要验证的数据存到列表里
    x_val = []
    y_val = []

    for i in range(len(c_val)):
        for j in range(len(comp)):
            if comp[j][0] == c_val[i][0]:
                if comp[j][1] == c_val[i][1]:
                    x_val.append(list(x[j]))
                    y_val.append(list(y[j]))

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    x_test = []
    y_test = []
    test_name = []
    for i in range(len(c_test)):
        for j in range(len(comp)):
            if comp[j][0] == c_test[i][0]:
                if comp[j][1] == c_test[i][1]:
                    x_test.append(list(x[j]))
                    y_test.append(list(y[j]))
                    test_name.append(c_test[i])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    test_name = np.array(test_name)

    components_train.close()
    components_val.close()
    components_test.close()

    return x_all, y_all, x_train, y_train, x_val, y_val, x_test, y_test, test_name