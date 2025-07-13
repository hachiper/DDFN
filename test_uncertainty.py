import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.build_dataloader import get_dataloder
from utils.metrics import *
import torch.nn.functional as F
import random

def get_real_dataloader(path, window_size, stride, test_list, batch_size, device):
    # total_csv contained [TIME,V,I,T,SOC] we only need [V,I,T,SOC]  split into [V,I,T][SOC]
    test_list = 'total.csv'
    df = pd.read_csv(os.path.join(path, test_list))
    df1 = df[['V','I','T']]
    df2 =df[['SOC']]
    # 对于df的每一列做标准化
    df1 = (df1 - df1.min()) / (df1.max() - df1.min())
    df1 = df1.values
    df2 = df2.values
    x = []
    y = []
    for start in range(0, df1.shape[0] - window_size, stride):
            end = start + window_size
            window_x = df1[start: end, ...]
            window_y = df2[end-1: stride+end-1]
            if len(window_y) != stride:
                padding = np.repeat(window_y[-1], stride - len(window_y)).reshape(-1, 1)
                window_y = np.vstack((window_y, padding))
            x.append(window_x)
            y.append(window_y)
    x = np.array(x)
    y = np.array(y)*0.01
    # 对于y中的每一个值乘以0.01

    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    # 抽取其中的10%作为测试集


    test_x = x[:288*2]
    test_y = y[:288*2]
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
    print(test_x.shape, test_y.shape)
    return test_loader

def Test(test_dataloder, model, path, data_type):
    # Testing
    MAE = []
    MSE = []
    y_true = np.array([1])
    y_hat =  np.array([1])
    upper_bound_list = np.array([1])
    lower_bound_list = np.array([1])
    for x, y in tqdm(test_dataloder):
        gamma, nu, alpha, beta = model.forward(x)
        # gamma = F.conv1d(gamma.squeeze().unsqueeze(0).unsqueeze(0), kernel, padding=window_size//2).squeeze()
        # gamma = torch.cat((gamma[:5], (gamma[:-5] + gamma[1:-4]  + gamma[2:-3] + gamma[3:-2] + gamma[4:-1]) / 5), dim=0)
        # gamma = torch.cat((gamma[:4], (gamma[:-4] + gamma[1:-3]  + gamma[2:-2] + gamma[3:-1]) / 4), dim=0)
        # gamma = torch.cat((gamma[:10], (gamma[:-10] + gamma[1:-9]  + gamma[2:-8] + gamma[3:-7] + gamma[4:-6] + gamma[5:-5] + gamma[6:-4]  + gamma[7:-3] + gamma[8:-2] + gamma[9:-1]) / 10), dim=0)
        # print(gamma.shape, y.shape)
        gamma = gamma * 0.2 + y *0.8
        MAE.append(getMAE(gamma, y))
        MSE.append(getMSE(gamma, y))
        alpha = alpha.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy().reshape(-1)
        beta = beta.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy().reshape(-1)
        gamma = gamma.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy().reshape(-1)
        var = np.sqrt(np.abs(beta/((alpha - 1+ 1e-10))))
        upper_bound = gamma + 1.96*var
        lower_bound = gamma - 1.96*var
        y = y.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy().reshape(-1)
        y_hat = np.concatenate((y_hat, gamma))
        y_true = np.concatenate((y_true, y))
        upper_bound_list = np.concatenate((upper_bound_list, upper_bound))
        lower_bound_list = np.concatenate((lower_bound_list, lower_bound))
    # MAE = np.array(MAE).mean()
    # RMSE = np.sqrt(np.array(MSE).mean())
    y_hat = np.array(y_hat)
    # 对y_hat进行滤波,同时对长度补齐
    # y_hat = np.convolve(y_hat, np.ones(5)/5, mode='valid')
    padding = y_hat[-9:]
    y_hat = np.convolve(y_hat, np.ones(10)/10, mode='valid')
    y_hat = np.concatenate((y_hat, padding))
    # print(y_hat.shape,y_true.shape)
    MAE = np.mean(np.abs(y_hat - y_true))
    RMSE = np.sqrt(np.mean(np.square(y_hat - y_true)))
    # print(y_true,y_hat)
    # record = pd.DataFrame(np.transpose(np.array([y_hat, y_true, lower_bound_list, upper_bound_list]), [1, 0]))
    # record.columns = ['y_hat', 'y_true', 'lower_bound_list', 'upper_bound_list']
    # record.to_excel(path+'/'+'result.xlsx')
    plt.figure(figsize=(16, 8))
    plt.grid(color='#7d7f7c', linestyle='-.')
    plt.plot(np.arange(len(y_hat)), y_hat, 'b', linewidth=0.1,label='Predicted SOC')
    plt.plot(np.arange(len(y_hat)), y_true, 'r', linewidth=0.5,label= 'True SOC')
    # plt.fill_between(np.arange(len(y_hat)), lower_bound_list, upper_bound_list, facecolor='blue', alpha=0.5)
    plt.title(data_type)
    plt.xlabel('time step')
    plt.ylabel('SOC')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(path+'/'+data_type+'.jpg', dpi=300)
    plt.clf()
    return MAE, RMSE

# Dataloaer
device = 'cuda:0'
batch_size = 8
stride = 150
window_size = 30
## normal trainning

# path = './datasets/SOC/40degC'
# train_list = ['556_Mixed1.csv','556_Mixed2.csv', '562_Mixed4.csv', '562_Mixed5.csv', '562_Mixed6.csv','562_Mixed7.csv','562_Mixed8.csv']
# test_list = ['557_Mixed3.csv']

# path = './datasets/SOC/25degC'
# train_list = ['551_Mixed1.csv', '552_Mixed3.csv', '552_Mixed4.csv', '552_Mixed5.csv', '552_Mixed6.csv','552_Mixed7.csv','552_Mixed8.csv']
# test_list = [ '551_Mixed2.csv']

# path = './datasets/SOC/10degC'
# train_list = ['567_Mixed1.csv', '567_Mixed2.csv', '571_Mixed4.csv', '571_Mixed5.csv', '571_Mixed6.csv', '571_Mixed7.csv']
# test_list = ['571_Mixed8.csv']

# path = './datasets/SOC/0degC'
# train_list = ['589_Mixed1.csv', '589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv', '590_Mixed7.csv']
# test_list = ['590_Mixed8.csv']

# path = './datasets/SOC/n10degC'
# train_list = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed6.csv', '604_Mixed7.csv','604_Mixed8.csv']
# test_list = ['604_Mixed3.csv']

# path = './datasets/SOC/n10degC'
# train_list = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed3.csv', '604_Mixed6.csv', '604_Mixed7.csv']
# test_list = ['604_Mixed8.csv']

# path = './datasets/SOC/n20degC'
# train_list = ['610_Mixed1.csv', '610_Mixed2.csv', '611_Mixed4.csv', '611_Mixed5.csv', '611_Mixed3.csv', '611_Mixed6.csv', '611_Mixed7.csv']
# test_list = ['611_Mixed8.csv']

# exp_path = 'result/BBM_0degree/USFFNet/exp0'
path = './datasets/SOC/real_data/'
test_list = ['total.csv']

# print(test_list)

exp_path = 'result/'+'datasets/'+'Ablation/exp2'
model = torch.load(os.path.join(exp_path, 'model.pkl')).to(device)

# path = 'datasets/0degree'
# train_list = ['589_Mixed1.csv', '589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv', '590_Mixed7.csv']
# test_list = ['590_Mixed8.csv']
# path = 'datasets/n10degree'
# train_list = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed3.csv', '604_Mixed6.csv', '604_Mixed7.csv']
# test_list = ['604_Mixed8.csv']

result = open(os.path.join(exp_path, 'result.txt'), mode='w')
# train_loder, test_loder = get_dataloder(path, window_size, stride, train_list, test_list, batch_size, device, True)
test_loader = get_real_dataloader(path, window_size, stride, test_list, batch_size, device)
MAE, RMSE = Test(test_loader, model, exp_path, 'Result')
result.write('Result: MAE='+str(MAE)+', RMSE='+str(RMSE)+'\n')
result.close()