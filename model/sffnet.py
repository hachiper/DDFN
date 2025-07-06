import torch
import math
import torch.nn.init as init
import torch.nn as nn
import torch.fft as fft
from model.uncertainty_head import UncertaintyHead
from model.basic_module import *
from Embed import *
from torchvision import utils as vutils
import time


class SpatialFlow(nn.Module):
    
    def __init__(self, window_size, feature_num, mid_channel, kernel_size):
        super(SpatialFlow, self).__init__()
        padding = int(kernel_size//2)
        self.up_conv1 = nn.Conv1d(feature_num  ,feature_num*2,kernel_size,1,padding)
        self.up_conv2 = nn.Conv1d(feature_num*2,feature_num*4,kernel_size,1,padding)

        self.c_conv = nn.Conv1d(feature_num*4, feature_num*4,kernel_size,1,padding)

        self.down_conv2 = nn.Conv1d(feature_num*4,feature_num*2,kernel_size,1,padding)
        self.down_conv1 = nn.Conv1d(feature_num*2,feature_num  ,kernel_size,1,padding)
        # self.icb = ICB(feature_num,mid_channel)
        # up_sampling 
        
        # down_sampling
        
        self.Spa_CNN = nn.Sequential(nn.Conv1d(feature_num, mid_channel, kernel_size, 1, padding), 
                                      nn.ReLU(),
                                      nn.Conv1d(mid_channel, feature_num, kernel_size, 1, padding),
                                      )
        self.act = nn.ReLU()


    def forward(self, x):
        # print(x.shape)
        Spa_feature = self.Spa_CNN(x)
        # Spa_feature = self.icb(x)

        # out1 = self.up_conv1(x)
        # out2 = self.up_conv2(out1)

        # mid = self.c_conv(out2)

        # up2 = self.down_conv2(mid+out2)
        # up1 = self.down_conv1(up2+out1)

        # Spa_feature = self.act(up1+x)

        return Spa_feature
    
    
class FrequencyFlow(nn.Module):
    
    def __init__(self, window_size, feature_num, mid_channel, kernel_size):
        super(FrequencyFlow, self).__init__()
        padding = int(kernel_size//2)
        self.pha_process = nn.Sequential(nn.Conv1d(feature_num, mid_channel, kernel_size, 1, padding), nn.ReLU(),
                                        nn.Conv1d(mid_channel, feature_num, kernel_size, 1, padding))
        self.amp_process = nn.Sequential(nn.Conv1d(feature_num, mid_channel, kernel_size, 1, padding), nn.ReLU(),
                                        nn.Conv1d(mid_channel, feature_num, kernel_size, 1, padding))
        
    def make_complex(self, phase, amplitude):
        real = amplitude * torch.cos(phase)
        im = amplitude * torch.sin(phase)
        complex_num = torch.complex(real, im)
        return complex_num

    def forward(self, x):
        frequency = fft.fft(x, dim=2, norm='backward')
        phase = torch.angle(frequency)
        magnitude = torch.abs(frequency)
        refine_phase = self.pha_process(phase)
        refine_magnitude = self.amp_process(magnitude)
        refine_spatial = self.make_complex(refine_phase, refine_magnitude)
        Fre_feature = torch.abs(fft.ifft(refine_spatial, dim=2, norm='backward'))
        return Fre_feature
        

class FusionBlock(nn.Module):
    
    def __init__(self, window_size, kernel_size, feature_num, r):
        super(FusionBlock, self).__init__()
        self.SA = SpatialAttention(kernel_size)
        self.CA = ChannelAttention(feature_num, r)

        
    def forward(self, fre_feature, spa_feature):
        spatial_refine_feature = self.SA(fre_feature - spa_feature)
        channel_refine_feature = self.CA(spa_feature + spatial_refine_feature)
        return channel_refine_feature
    
class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(in_features)

    def forward(self, x):
        # x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        out = self.conv3(out1 + out2)
        # print(out.shape)
        # out = self.act(self.bn(out))
        
        # x = x.transpose(1, 2)
        return out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        # trunc_normal_(self.complex_weight_high, std=.02)
        # trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        # print(x_fft.shape)

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in,adaptive_filter=True):
        x_in = x_in.transpose(1,2)
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.fft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        ###
        if adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            # print(x_fft.shape) 
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            
            # print(freq_mask.shape)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted = x_weighted2
            # x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.ifft(x_weighted, n=N, dim=1, norm='ortho')
        # print(x.shape,x_weighted.shape)

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape
        x = x.transpose(1,2)

        # print(freq_mask.shape)

        return x , freq_mask
    

class CRU(nn.Module):
    
    def __init__(self, window_size, kernel_size, feature_num, r):
        super(CRU, self).__init__()
        self.vc_channel = 2
        self.t_channel = 1
        ratio = 2
        self.vc_conv1   = nn.Conv1d(self.vc_channel,self.vc_channel*ratio,kernel_size) 
        self.t_conv1 = nn.Conv1d(self.t_channel,self.t_channel*ratio,kernel_size)

        self.vc_conv2 =nn.Conv1d(self.vc_channel*ratio,self.vc_channel,1) 
        self.t_conv2 = nn.Conv1d(self.t_channel*ratio,self.t_channel,1)

        self.vc_conv3 =nn.Conv1d(self.vc_channel*ratio,self.vc_channel,kernel_size) 
        self.t_conv3 = nn.Conv1d(self.t_channel*ratio,self.t_channel,kernel_size)

    def forward(self, x):
        vc,t = torch.split(x,[self.vc_channel,self.t_channel],dim=1)
        vc = self.vc_conv(vc) 
        t  = self.t_conv(t)

        vc = self.vc_conv2(vc) + self.vc_conv3(vc)
        t = self.t_conv2(t) + self.vc_conv3(t)

        feature = torch.cat([vc,t],dim=1)

        return feature + x

class PAIFILTER(nn.Module):

    def __init__(self,hidden_size,seq_len):
        super(PAIFILTER, self).__init__()
        self.seq_len = seq_len
        self.pred_len = 1
        self.scale = 0.02

        self.embed_size = self.seq_len
        self.hidden_size = hidden_size
        
        self.w = nn.Parameter(self.scale * torch.randn(3, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )
        self.softmx = nn.Softmax(dim=1)


    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x):
        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D
        x = self.fc(x)
        x = self.softmx(x)

        return x


class SFFBlock(nn.Module):
    
    def __init__(self, window_size, feature_num, mid_channel, spa_ks, fre_ks, fus_ks, r, fb, sf, ff):
        super(SFFBlock, self).__init__()
        self.fb, self.sf, self.ff = fb, sf, ff
        # if fb is True:
        #     self.FB = FusionBlock(window_size, fus_ks, feature_num, r)
        if sf is True:
            self.SF = SpatialFlow(window_size, feature_num, mid_channel, spa_ks)
        if ff is True:
            self.FF = FrequencyFlow(window_size, feature_num, mid_channel, fre_ks)
        # self.conv = nn.Conv1d(feature_num*2,feature_num,fus_ks)
        # self.icb = ICB(feature_num,  feature_num)
        # self.Norm = nn.BatchNorm1d(feature_num)
        # self.Norm = nn.LayerNorm(window_size)
        
    def forward(self, x):
        
        if self.sf is True:
            Spa_feature = self.SF(x)
        if self.ff is True:
            Fre_feature = self.FF(x)
        if self.fb is True:
            # feature = Spa_feature
            feature = Fre_feature 
            # feature = Fre_feature + Spa_feature
            # feature = self.FB(Fre_feature, Spa_feature)
            # feature = self.icb(torch.concat([Spa_feature , Fre_feature],dim=1))
            # print(feature.shape)
            
        else:
            if self.sf is True:
                feature = Spa_feature
                if self.ff is True:
                    feature += Fre_feature
            else:
                if self.ff is True:
                    feature = Fre_feature
        # feature = self.icb(feature)
        
        return feature + x
    

class USFFNet(nn.Module):
    
    def __init__(self, num_block, feature_num, window_size,stride, mid_channel_list, spa_ks_list, fre_ks_list, fus_ks_list):
        super(USFFNet, self).__init__()
        self.SFFBlock = nn.Sequential()
        for i in range(num_block):
             self.SFFBlock.add_module('SFFBlock'+str(i), SFFBlock(window_size, feature_num, mid_channel_list[i], spa_ks_list[i], fre_ks_list[i], fus_ks_list[i], 2, True, True, True))
        self.CNNI = nn.Sequential(nn.Conv1d(feature_num, 1, 3, 1, 1))
        self.Uncertainty_Head = UncertaintyHead(window_size,stride)
        # self.icb = ICB(feature_num, feature_num)
        self.embedding = DataEmbedding(feature_num,feature_num)
        # self.Adaptive_Spectral_Block = Adaptive_Spectral_Block(feature_num)
        self.paifilter = PAIFILTER(hidden_size=50,seq_len=window_size)
    
    def forward(self, x):
        x = x.transpose(2,1)
        embeded_x = self.embedding(x)
        # print(feature.shape)
        # feature, _  = self.Adaptive_Spectral_Block(embeded_x)
        # print(feature.shape,sum(freq_mask))
        # print(feature.shape)

        weight = self.paifilter(embeded_x)

        # feature = self.SFFBlock(feature + x )
        feature = self.SFFBlock(weight * embeded_x + x)
        # feature = self.CNNI(feature)
        # feature = self.icb(feature)
        # 
        
        feature = feature.reshape(feature.shape[0], feature.shape[1]*feature.shape[2])

        gamma, nu, alpha, beta = self.Uncertainty_Head.forward(feature)
        return gamma, nu, alpha, beta
    

