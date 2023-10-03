import math
import numpy as np

def RMSE_cal(y_pred, y_test):
    RMSE_square = 0
    for i in range(y_pred.size):
        RMSE_square1 = math.pow((y_pred[i] - y_test[i]),2)
        RMSE_square += RMSE_square1

    RMSE = math.sqrt(RMSE_square/y_pred.size)
    return RMSE

def MAE_cal(y_pred, y_test):
    MAE_sum = 0
    for i in range(y_pred.size):
        MAE1 = math.fabs(y_pred[i] - y_test[i])
        MAE_sum += MAE1

    MAE = MAE_sum/y_pred.size
    return MAE

def AADP_cal(y_pred, y_test):
    AADP = 0
    for i in range(y_pred.size):
        AADP1 = math.fabs((y_pred[i] - y_test[i])/y_test[i])
        AADP = AADP1+AADP

    AADP = AADP/y_pred.size
    return AADP

def R2_cal(y_pred, y_test, y_mean):
    R2_numerator = 0
    R2_denominator = 0
    for i in range(y_pred.size):
        R2_numerator1 = math.pow(y_pred[i] - y_test[i],2)
        R2_denominator1 = math.pow(y_test[i] - y_mean,2)
        R2_numerator += R2_numerator1
        R2_denominator += R2_denominator1

    R2 = 1 - R2_numerator/R2_denominator
    return R2


    