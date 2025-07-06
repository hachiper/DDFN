import torch
import torch.nn as nn
from tqdm import tqdm
import data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as opt
import matplotlib.pyplot as plt
import numpy as np
from tools import EarlyStopping
from uncertainty_head import UncertaintyHead
from timesnet import Model
from Embed import DataEmbedding
from FKF import *


device = 'cuda:0'
batch_size = 128
epoches = 20
weight_decay = 1e-4
learning_rate = 5e-4
test_ratio = 1
# path = './result'

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.channel_mix = nn.Linear(3,1)
        self.linear1 = nn.Linear(150*3,75)
        self.linear2 = nn.Linear(75,10)
        self.linear3 = nn.Linear(10,1)
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        x = x.reshape(x.shape[0], -1)
        # output,_ = self.self_atten(x,x,x)
        # x = self.conv1(x)
        # out = self.channel_mix(x.transpose(2,1)).transpose(2,1)
        out = self.linear1(x)
        out = self.sigmoid(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        out = self.linear3(out)

        return out

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.d_model = 4
        self.lstm = nn.GRU(3, 10)
        self.fc = nn.Linear(10, 1)
        self.activate = nn.Sigmoid()
        # self.combine = nn.Linear(10, 1)
        self.embedding = DataEmbedding(3,self.d_model)
        # self.atten =  nn.MultiheadAttention(10,2)

    def forward(self,x):
        # print(x.shape)
        # embed = self.embedding(x)
        # print(embed.shape)
        # print(x.shape)
        out,_ = self.lstm(x.transpose(2,1))
        out = self.fc(out[:,-1,:])
        # out = self.bn(out)
        out = self.activate(out)
        
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.channel_confusion = nn.Conv1d(in_channels=3,out_channels=1,kernel_size=1)
        self.convs = nn.Sequential(
            nn.Conv1d(1,1,2,2,1),
            nn.Conv1d(1,1,2,2),
            nn.Conv1d(1,1,2,2,1),
            nn.Conv1d(1,1,2,2)
        )
        self.linear = nn.Linear(10,1)
        self.activation = nn.Sigmoid()
    def forward(self,x):
        out = self.channel_confusion(x)
        out = self.convs(out)
        # print(out.shape)
        out = self.linear(out)
        out = self.activation(out)
        
        return out


class LSTM_cell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(LSTM_cell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # gating
        self.w_i = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.w_f = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.w_c = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.w_o = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self,x,hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), 1)
        
        # 计算门的激活值
        i_t = self.sigmoid(self.w_i(combined))  
        f_t = self.sigmoid(self.w_f(combined))  
        o_t = self.sigmoid(self.w_o(combined))  
        c_t = self.tanh(self.w_c(combined))     
        
        # 当前时间步的记忆单元状态和隐状态
        c_t = f_t * c_prev + i_t * c_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM,self).__init__()
        self.hidden_size = configs.hidden_dim
        self.input_size = configs.nums
        self.layer_nums = configs.layers
        self.output_size = configs.pred_len 
        # self.lstm = LSTM_cell(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.tanh = nn.Tanh()
        cell_list = []
        for i in range(self.layer_nums):
            cur_input_dim = self.input_size if i == 0 else self.hidden_size
            cell_list.append(LSTM_cell(cur_input_dim, self.hidden_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self,x):
        h_t = torch.zeros(x.size(0),self.hidden_size,dtype=x.dtype).to(x.device)
        c_t = torch.zeros(x.size(0),self.hidden_size,dtype=x.dtype).to(x.device)
        
        cur_layer_input = x.transpose(1,2)

        for i in range(self.layer_nums):
            hidden = []
            for t in range(cur_layer_input.size(1)):
                h_t,c_t = self.cell_list[i](cur_layer_input[:,t,:],(h_t,c_t))
                hidden.append(h_t)
            layer_output = torch.stack(hidden,dim=1)
            cur_layer_input = layer_output
        
        output = self.tanh(self.fc(layer_output[:,-1,:]))
        return output,(h_t,c_t)

class Transformer(nn.Module):
    def __init__(self,feature,windows,d_model= 4):
        super(Transformer,self).__init__()
        
        self.input_embedding = DataEmbedding(feature,d_model)
        self.tsformer = nn.Transformer(d_model,1)
        self.projection = nn.Linear(d_model,1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)


    def forward(self,src,tgt):
        # print(src.shape,tgt.shape)
        tgt = tgt.unsqueeze(dim=2)
        embed_src = self.input_embedding(src)
        embed_tgt = self.input_embedding(tgt)
        # print(embed_src.shape,embed_tgt.shape)
        embed_src = embed_src.transpose(1,0)
        embed_tgt = embed_tgt.transpose(1,0)
        output = self.tsformer(embed_src,embed_tgt)
        out = self.dropout(self.activation(self.projection(output)))

        return out





def build_dataloader(configs):
    train_x,train_y = data.data(configs,configs.train_list,configs.window_size,configs.pred_len)
    vali_x,vali_y   = data.data(configs,configs.vali_list,configs.window_size,configs.pred_len)
    test_x,test_y   = data.data(configs,configs.test_list,configs.window_size,configs.pred_len)

    train_loader = DataLoader(TensorDataset(train_x.to(configs.device), train_y.to(configs.device)), batch_size=configs.batch_size, shuffle=False, drop_last=True)       
    vali_loader  = DataLoader(TensorDataset(vali_x.to(configs.device), vali_y.to(configs.device)), batch_size=configs.batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(TensorDataset(test_x.to(configs.device), test_y.to(configs.device)), batch_size=configs.batch_size, shuffle=False, drop_last=True)
    
    return train_loader, vali_loader, test_loader

def FKF(output, now, update):
    # torch.Size([32, 1, 1]) torch.Size([32, 1, 1]) torch.Size([32, 1, 1])
    output = output.unsqueeze(dim=1)
    now = now.repeat(batch_size, 1, 1)
    Q = (torch.eye(1).repeat(batch_size, 1, 1) * 0.1).to(device)
    R = (torch.eye(1).repeat(batch_size, 1, 1) * 0.1).to(device)
    x_pred, P_pred = ekf_predict(output, update, f, F_jacobian, Q)
    x_upd, P_upd = ekf_update(x_pred, P_pred, now, h, H_jacobian, R)
    return x_pred, P_upd


loss_func = nn.MSELoss()


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)     


"""

index = range(len(train_loss))
plt.figure(figsize=(12, 6))
plt.grid(color='#7d7f7c', linestyle='-.')
plt.plot(index, train_loss, 'c--', linewidth=1.5, label="train")
plt.plot(index, vaild_loss, '2b--', linewidth=1.5, label="vaild")
plt.title('Loss')
plt.xlabel('epoch')
plt.show()
"""

def train(configs, model,train_loader,vali_loader):
    early_stopping = EarlyStopping(configs,patience=configs.patience, verbose=True)
    optimizer = opt.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loss = []
    vali_loss = []
    for epoch in range(configs.epoches):
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_loader):
            output  = model.forward(x)
            loss = loss_func(y,output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= train_loader.__len__()
        train_loss.append(epoch_loss)
        print('epoch '+ str(epoch) + ' trainning_loss = '+ str(epoch_loss))
        value = vali(model,vali_loader)
        vali_loss.append(value)
        early_stopping(value, model)
        if early_stopping.early_stop:
            # torch.save(model, './result/'+configs.model+'/model.pth')
            print("Early stopping")
            break
        # torch.save(model, './result/'+configs.model+'/model.pth')
    index = np.arange(len(train_loss))
    plt.grid(color='#7d7f7c', linestyle='-.')
    plt.plot(index, train_loss, 'c', linewidth=1.5, label='train_loss')
    plt.plot(index, vali_loss, 'c', linewidth=1.5, label='vali_loss')
    plt.title('LOSS')
    plt.xlabel('epoch')
    plt.legend(loc=1)
    plt.savefig(configs.path+configs.model+'/loss.jpg', dpi=300)
    plt.show()
    # plt.clf()
        
    return model

def vali(model,vali_loader):
    model.eval()
    vali_loss = 0
    for x, y in tqdm(vali_loader):
        output  = model.forward(x)
        loss = loss_func(y,output)
        vali_loss += loss.item()
    vali_loss /= vali_loader.__len__()
    print('---vali_loss = '+str(vali_loss)+'---')
    return vali_loss

def test(configs,model,test_loader):
    
    y_true = np.array([1])
    y_hat =  np.array([1])
    model.eval()
    epoch_loss = 0
    for x, y in tqdm(test_loader):
        output = model.forward(x)
        # now = x [:,-1,-1] 
        # output, y_upd = FKF(output, now, update)
        # update = y_upd
        _loss = loss_func(output, y)
        output = output.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy()
        y = y.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy()
        # delta = y[-1] - np.mean(output[:5])
        # output[:5] = output[:5] + delta
        y_hat = np.concatenate((y_hat, output))
        y_true = np.concatenate((y_true, y))
        epoch_loss += _loss.item()
    epoch_loss /= test_loader.__len__()
    print('---testing_loss = '+str(epoch_loss)+'---')
    mae = np.mean(np.abs(y_true - y_hat))
    rmse = np.sqrt(np.mean((y_true - y_hat) ** 2))
    print('MAE:',mae,' RMSE: ',rmse)
    with open (configs.path+configs.model+'result.txt','w') as f:
        f.write('MAE:'+str(mae)+' RMSE: '+ str(rmse))
        f.close()
    # print(y_hat[2048:4096])
    plt.figure(figsize=(12, 6))
    plt.grid(color='#7d7f7c', linestyle='-.')
    plt.plot(np.arange(len(y_hat)), y_hat, 'b', linewidth=0.1, label="y_hat")
    plt.plot(np.arange(len(y_hat)), y_true, 'r', linewidth=0.5, label="y_true")
    plt.title('ratio')
    plt.legend()
    plt.xlabel('steps')
    plt.savefig(configs.path+'/'+configs.model+'/result.png',dpi=300, bbox_inches='tight')
    plt.show()
    

    
    


        
