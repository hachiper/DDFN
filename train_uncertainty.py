import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from model.sffnet import USFFNet
from utils.evaluate import Evaluate
from utils.build_dataloader import get_dataloder
from tools import EarlyStopping,adjust_learning_rate

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs')
from torchvision import utils as vutils
import os
import random
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

# Dataloader&&Evaluate
device = 'cuda:0'
batch_size = 32
stride = 150
window_size = 30

def get_real_dataloader(path, window_size, stride, file_list, batch_size, device):
    # total_csv contained [TIME,V,I,T,SOC] we only need [V,I,T,SOC]  split into [V,I,T][SOC]
    file_list = 'total.csv'
    df = pd.read_csv(os.path.join(path, file_list))
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
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    print(x.shape,y.shape)
    x = x[:int(len(x)*0.9)]
    y = y[:int(len(y)*0.9)]
    # split x into 0.9 and 0.1 as train_x and test_x
    # split y into 0.9 and 0.1 as train_y and test_y
    train_x = x[:int(len(x)*0.9)]
    test_x = x[int(len(x)*0.9):]
    train_y = y[:int(len(y)*0.9)]
    test_y = y[int(len(y)*0.9):]

    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
    # split dataloader into train and test
    
    return train_dataloader,test_dataloader



## normal training
# path = './datasets/SOC/0degC'
# train_list = ['589_Mixed1.csv', '589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv', '590_Mixed7.csv']
# test_list = ['590_Mixed8.csv']

# path = './datasets/SOC/10degC'
# train_list = ['567_Mixed1.csv', '567_Mixed2.csv', '571_Mixed4.csv', '571_Mixed5.csv', '571_Mixed6.csv', '571_Mixed7.csv']
# test_list = ['571_Mixed8.csv']

# path = './datasets/SOC/25degC'
# train_list = ['551_Mixed1.csv', '551_Mixed2.csv', '552_Mixed3.csv', '552_Mixed4.csv', '552_Mixed5.csv', '552_Mixed6.csv','552_Mixed7.csv']
# test_list = ['552_Mixed8.csv']

# path = './datasets/SOC/25degC'
# train_list = ['551_Mixed1.csv', '552_Mixed3.csv', '552_Mixed4.csv', '552_Mixed5.csv', '552_Mixed6.csv','552_Mixed7.csv','552_Mixed8.csv']
# test_list = [ '551_Mixed2.csv']

# path = './datasets/SOC/40degC'
# train_list = ['556_Mixed1.csv','556_Mixed2.csv', '557_Mixed3.csv', '562_Mixed4.csv', '562_Mixed5.csv', '562_Mixed6.csv','562_Mixed7.csv','562_Mixed8.csv']
# test_list = ['557_Mixed3.csv']


# path = './datasets/SOC/n10degC'
# train_list = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed6.csv', '604_Mixed7.csv','604_Mixed8.csv']
# test_list = ['604_Mixed3.csv']


# path = './datasets/SOC/n20degC'
# train_list = ['610_Mixed1.csv',  '611_Mixed3.csv', '611_Mixed4.csv', '611_Mixed5.csv',  '611_Mixed6.csv', '611_Mixed7.csv','611_Mixed8.csv']
# test_list = ['610_Mixed2.csv']

# path = './datasets/SOC/n10degC'
# train_list = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed3.csv', '604_Mixed6.csv', '604_Mixed7.csv']
# test_list = ['604_Mixed8.csv']

# path = './datasets/SOC/n20degC'
# train_list = ['610_Mixed1.csv', '610_Mixed2.csv', '611_Mixed4.csv', '611_Mixed5.csv', '611_Mixed3.csv', '611_Mixed6.csv', '611_Mixed7.csv']
# test_list = ['611_Mixed8.csv']

# robust trainning
'''
path = './datasets/SOC/real_data/'
train_list = []
test_list = []
file = os.listdir(path)
train_list = file
test_list = random.sample(file, int(len(file)*0.1))

with open('./result/output.txt', 'w') as f:
    f.write('train_list:\n')
    for item in train_list:
        f.write(item + '\n')  # 每个元素后加换行符
    f.write('test_list:\n')
    for item in test_list:
        f.write(item + '\n')  # 每个元素后加换行符
    f.close()

'''

#

# train_loader, test_loader = get_dataloder(path, window_size, stride, train_list, test_list, batch_size, device)
path = './datasets/SOC/real_data/'
file_list = ['total.csv']
train_loader, test_loader = get_real_dataloader(path, window_size, stride, file_list ,batch_size, device)
# print(train_loader.__len__())
# print(test_loader.__len__())
# Trainning&&Model Config
test_ratio = 1
epoches = 200
weight_decay = 1e-4
learning_rate = 1e-4
loss_funcation = nn.MSELoss()
evaluater = Evaluate(path.split('/')[1], 'Ablation', test_ratio)
block_num = 5
feature_num = 3

spa_ks_list = [3, 5, 7, 7, 7]
fre_ks_list = [3, 5, 7, 7, 7]
fus_ks_list = [3, 3, 7, 7, 7]
mid_channel_list = [32, 16, 8, 4, 4]
# spa_ks_list = [3, 5, 7]
# fre_ks_list = [3, 5, 7]
# fus_ks_list = [3, 3, 3]
# mid_channel_list = [32, 16, 8]
model = USFFNet(block_num, feature_num, window_size,stride, mid_channel_list, spa_ks_list, fre_ks_list, fus_ks_list).to(device)
# model.apply(weights_init)
optimizer = opt.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
evaluater.record_param_setting(window_size, stride, batch_size, learning_rate, weight_decay, model)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

# Trainning
lambda_coef = []
train_loss = []
vaild_loss = []
early_stopping = EarlyStopping(patience=5, verbose=True)
for epoch in range(epoches):
    model.train()
    epoch_loss = 0
    print('epoch: '+str(epoch))
    for x, y in tqdm(train_loader):
        # train
        gamma, nu, alpha, beta = model.forward(x)
        loss, nig_loss, nig_regularization = model.Uncertainty_Head.get_loss(y, gamma, nu, alpha, beta)
        # loss = loss_funcation(gamma, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.Uncertainty_Head.hyperparams_update(nig_regularization)
        # record
        # print(gamma.shape)
        _loss = loss_funcation(gamma, y)
        epoch_loss += _loss.item()

    epoch_loss /= train_loader.__len__()
    writer.add_scalar("Loss/train",epoch_loss,epoch)
    lambda_coef.append(model.Uncertainty_Head.lambda_coef.detach().cpu().numpy())
    train_loss.append(epoch_loss)
    print('trainning_loss = '+str(epoch_loss))
    if epoch%test_ratio == 0:
        model.eval()
        epoch_loss = 0
        for x, y in test_loader:
            gamma, nu, alpha, beta = model.forward(x)
            _loss = loss_funcation(gamma, y)
            epoch_loss += _loss.item()
        epoch_loss /= test_loader.__len__()
        writer.add_scalar('Loss/vali',epoch_loss,epoch)
        
        vaild_loss.append(epoch_loss)
        print('testing_loss = '+str(epoch_loss))
        
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #    break
    evaluater.visualize(train_loss, vaild_loss, model, None)
    # early_stopping(epoch_loss, model, path)
    # model = model.to(device)
    # evaluater.draw('lambda_coef', lambda_coef)
    # adjust_learning_rate(optimizer, epoch + 1, learning_rate,lradj='type1')


    '''
    mask_save = freq_mask[:,-1,:].clone().detach()
    x_save = x[:,-1,:].clone().detach()
    # print(mask_save.shape,x_save.shape)
    # print(mask_save.shape)
    mask_save = mask_save.to(torch.device('cpu')).repeat(1,1,3)
    x_save = x_save.to(torch.device('cpu'))
    # torch.save(mask_save, './mask/'+str(epoch)+'_mask.txt')
    # torch.save(x_save,'./mask/'+str(epoch)+'_x.txt')

    vutils.save_image(mask_save,'./mask/'+str(epoch)+'_mask.png')
    vutils.save_image(x_save,'./mask/'+str(epoch)+'_x.png')
    mask_save = mask_save.numpy()

# 写入到 .txt 文件
    with open('./mask/'+str(epoch)+'_mask.txt', 'w') as f:
        for row in mask_save:
        # 将每行的元素写入为空格分隔的字符串
            f.write(" ".join(map(str, row.flatten())) + '\n')
'''