import argparse
import os
import torch
import random
import numpy as np
from train import *

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Battery_prediction')

    # basic config
    parser.add_argument('--path',type = str, default = './result/10degC/')
    parser.add_argument('--file_path',type = str, default = './datasets/SOC/10degC/')
    parser.add_argument('--model',type = str, default = 'CNN')
    parser.add_argument('--epoches',    type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 256)

    parser.add_argument('--nums',       type = int, default = 3)
    parser.add_argument('--enc_dim',    type = int, default = 4)
    parser.add_argument('--cell_dim',   type = list,default = [16,32,64,64])
    parser.add_argument('--hidden_dim', type = int,default  = 16)
    parser.add_argument('--layers',     type = int, default = 3)
    parser.add_argument('--seq_len',    type = int, default = 150)
    parser.add_argument('--pred_len',   type = int, default = 1)
    parser.add_argument('--model_out',  type = int, default = 10)
    parser.add_argument('--dropout',    type = float,default = 0.1, help='dropout')
    parser.add_argument('--patience',   type = int, default = 10, help='early stop patience')
    parser.add_argument('--window_size',type = int, default = 150)
    parser.add_argument('--activation', type = str, default = 'sigmoid')
    # 0degreec
    # parser.add_argument('--train_list', type = list,default = ['589_Mixed1.csv', '589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv','590_Mixed7.csv'])
    # parser.add_argument('--vali_list',  type = list,default = ['590_Mixed8.csv'])
    # parser.add_argument('--test_list',  type = list,default = ['590_Mixed8.csv'])
    # 10degreec
    parser.add_argument('--train_list', type = list,default = ['567_Mixed1.csv', '567_Mixed2.csv', '571_Mixed4.csv', '571_Mixed5.csv', '571_Mixed6.csv', '571_Mixed7.csv'])
    parser.add_argument('--vali_list',  type = list,default = ['571_Mixed8.csv'])
    parser.add_argument('--test_list',  type = list,default = ['571_Mixed8.csv'])
    # 25degreec
    # parser.add_argument('--train_list', type = list,default = ['551_Mixed1.csv', '551_Mixed2.csv', '552_Mixed3.csv', '552_Mixed4.csv', '552_Mixed5.csv', '552_Mixed6.csv','552_Mixed7.csv'])
    # parser.add_argument('--vali_list',  type = list,default = ['552_Mixed8.csv'])
    # parser.add_argument('--test_list',  type = list,default = ['552_Mixed8.csv'])
    # 40degreec
    # parser.add_argument('--train_list', type = list,default = ['556_Mixed1.csv','556_Mixed2.csv', '557_Mixed3.csv', '562_Mixed4.csv', '562_Mixed5.csv', '562_Mixed6.csv','562_Mixed7.csv'])
    # parser.add_argument('--vali_list',  type = list,default = ['562_Mixed8.csv'])
    # parser.add_argument('--test_list',  type = list,default = ['562_Mixed8.csv'])
    # n10degreec
    # parser.add_argument('--train_list', type = list,default = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed6.csv', '604_Mixed7.csv','604_Mixed8.csv'])
    # parser.add_argument('--vali_list',  type = list,default = ['604_Mixed3.csv'])
    # parser.add_argument('--test_list',  type = list,default = ['604_Mixed3.csv'])
    # n20degreec
    # parser.add_argument('--train_list', type = list,default = ['610_Mixed1.csv',  '611_Mixed3.csv', '611_Mixed4.csv', '611_Mixed5.csv',  '611_Mixed6.csv', '611_Mixed7.csv','611_Mixed8.csv'])
    # parser.add_argument('--vali_list',  type = list,default = ['610_Mixed2.csv'])
    # parser.add_argument('--test_list',  type = list,default = ['610_Mixed2.csv'])
    
    parser.add_argument('--device',     type = str, default = 'cuda:0')


    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if os.path.exists(args.path):
        if os.path.exists(args.path+args.model):
            pass
        else:
            os.mkdir(args.path+args.model)
            
    else:
        os.mkdir(args.path)
        if os.path.exists(args.path+args.model):
            pass
        else:
            os.mkdir(args.path+args.model)

    print('>>>>>>>build_dataloader : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    train_loader, vali_loader, test_loader = build_dataloader(args)
    if args.model == 'FC':
        model = FC().to(device)
    elif args.model == 'RNN':
        model = RNN().to(device)
    elif args.model == 'CNN':
        model = CNN().to(device)
    elif args.model == 'LSTM':
        model = LSTM().to(device)
    # elif args.model == 'Transformer':
        # model = Transformer(feature=3,windows=150).to(device)
    elif args.model == 'xgboost':
        model = xgboost.to(device)
    # model = Model().to(device)  
    # model = PETNN(args).to(device)
    # model = LSTM (args).to(device)
    # torch.save(model,'model.pkt')  

    print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    train(args,model,train_loader,vali_loader)
    # torch.save(model, './result'+ar'/model.pth')
    model.load_state_dict(torch.load(args.path+args.model+'/model.pth'))

    print('>>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    test(args,model,test_loader)
    torch.cuda.empty_cache()
