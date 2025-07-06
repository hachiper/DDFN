import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 定义一个软阈值函数
def soft_threshold(data, threshold):
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)


def load_data(configs,file_list, window_size, stride):
    window_data_x = []
    window_data_y = []
    path = configs.file_path
    # original_x = []
    for file_name in tqdm(file_list):
        data = pd.read_csv(path +  file_name, skiprows=30)
        data.columns = ['Time Stamp', 'Step', 'Status', 'Prog Time', 'Step Time', 'Cycle', 'Cycle Level', 'Procedure',
                        'Voltage', 'Current', 'Temperature', 'Capacity', 'WhAccu', 'Cnt', 'Empty']
        data = data[(data["Status"] == "TABLE") | (data["Status"] == "DCH")]
        # Normalize SOC
        max_discharge = abs(min(data["Capacity"]))
        data["SoC Capacity"] = max_discharge + data["Capacity"] 
        data["SoC Percentage"] = data["SoC Capacity"] / max(data["SoC Capacity"]) 
        y = data[["SoC Percentage"]].to_numpy()
        # Normalize Voltage, Current, Temperature 
        x = data[["Voltage", "Current", "Temperature"]].to_numpy()
        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
        # original_x+=x[150:].tolist()
        # print(x.shape)
        # Generate window trainning data
        for start in range(0, x.shape[0] - window_size, stride):
            end = start + window_size
            window_x = x[start:end, ...]
            window_y = y[end-1:end-1+stride]
            window_data_x.append(window_x)
            window_data_y.append(window_y)
    # print(len(original_x),len(original_x[0]))
    return window_data_x, window_data_y

# 小波阈值去噪函数
def wavelet_threshold_denoising(signal, wavelet='db1', level=None, threshold_method='universal'):    
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 计算阈值
    if threshold_method == 'universal':
        # 通用阈值法
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    else:
        raise ValueError("Unsupported threshold method.")

    # 对细节系数进行软阈值处理
    thresholded_coeffs = [coeffs[0]]  # 保留近似系数
    for detail_coeff in coeffs[1:]:
        thresholded_coeffs.append(soft_threshold(detail_coeff, threshold))

    # 小波重构
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)
    
    return denoised_signal
def furrier(data):
    freq_data = torch.fft.fft(data)
    filtered_freq_data = freq_data.clone()
    filtered_freq_data[10:-10] = 0

    reconstructed_time_series = torch.fft.ifft(filtered_freq_data)

    reconstructed_time_series = reconstructed_time_series.real
    return reconstructed_time_series

def wavelet_denoise_torch(data, wavelet='db8', level=1, threshold_type='soft', threshold_multiplier=1.0):
    """
    使用小波阈值去噪，结合 PyWavelets 和 PyTorch
    :param data: 需要去噪的时序数据，Tensor 类型
    :param wavelet: 使用的小波类型 (默认 'db8')
    :param level: 小波分解的级数 (默认 1)
    :param threshold_type: 阈值类型 ('soft' or 'hard')
    :param threshold_multiplier: 阈值系数，默认 1.0
    :return: 去噪后的信号，Tensor 类型
    """
    # 将数据转换为 NumPy 数组以使用 PyWavelets 进行小波分解
    data_np = data.detach().cpu().numpy()

    # 1. 使用小波分解信号
    coeffs = pywt.wavedec(data_np, wavelet, level=level)

    # 2. 计算全局阈值
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745  # 使用中值绝对偏差估计噪声标准差
    threshold = threshold_multiplier * sigma * np.sqrt(2 * np.log(len(data_np)))  # 通用阈值计算

    # 3. 对每个系数应用阈值处理
    denoised_coeffs = [coeffs[0]]  # 保留最顶层的近似系数
    for detail_coeffs in coeffs[1:]:
        if threshold_type == 'soft':
            denoised_coeffs.append(pywt.threshold(detail_coeffs, threshold, mode='soft'))
        elif threshold_type == 'hard':
            denoised_coeffs.append(pywt.threshold(detail_coeffs, threshold, mode='hard'))

    # 4. 小波重构信号
    denoised_data_np = pywt.waverec(denoised_coeffs, wavelet)

    # 将 NumPy 数组转换回 PyTorch Tensor
    denoised_data = torch.tensor(denoised_data_np, dtype=data.dtype, device=data.device)

    return denoised_data

def data(configs,datafile,windowsize,stride):
    x, y = load_data(configs,datafile,windowsize,stride)
    x, y = torch.Tensor(np.array(x)).float().transpose(1, 2), torch.Tensor(np.array(y)).float()
    # 
    # original_x = torch.Tensor(np.array(original_x)).float()
    # print(original_x.shape , x.shape, y.shape)
    # x = torch.Tensor(wavelet_threshold_denoising(x, wavelet='db1', level=3))
    # x = furrier(x)
    # print(x.shape)
    # x = wavelet_denoise_torch(x)
    return  x, y

"""

file_list = ['./589_Mixed1.csv']
x,y = load_data(file_list,1500,1)
train_x, train_y = torch.Tensor(np.array(x)).float().transpose(1, 2), torch.Tensor(np.array(y)).float()
# print(train_x.shape)
test_data = train_y[:,0].tolist()

x = range(len(test_data))

signal_denoised = wavelet_threshold_denoising(test_data, wavelet='db1', level=3)


plt.figure(figsize=(12, 6))
# plt.plot(x, signal_clean, label='Clean Signal', linewidth=1.5)
plt.plot(x, test_data, label='Noisy Signal', linewidth=1, alpha=0.6)
plt.plot(x, signal_denoised[:len(test_data)], label='Denoised Signal', linewidth=1.5, linestyle='--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Wavelet Threshold Denoising')
plt.show()


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

datafile = ['589_Mixed1.csv','589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv', '590_Mixed7.csv']
windowsize = 150
stride = 1 
original_x,x , y = data(datafile, windowsize,stride)
# y = y.repeat(1,1,x.shape[-1])
# processed_data = torch.cat([])
# print(x.shape,y.shape)


# Step 1: 降维 (使用 PCA 将数据降到 2 维)
# 先将数据调整为形状 [55388, 450]，即将每个 [3, 150] 展平
data_flattened = x.reshape(x.shape[0], -1)  # 将每个 3x150 的向量展平


# 使用 PCA 将数据从 450 维降到 2 维
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_flattened)
# print(data_flattened.shape,original_x.shape,data_pca.shape)
# tsne = TSNE(n_components=2, random_state=42)
# data_pca = tsne.fit_transform(data_flattened)

# Step 2: 聚类 (使用 K-Means 聚类)
kmeans = KMeans(n_clusters=3, random_state=42)  # 假设我们聚成 3 类
clusters = kmeans.fit_predict(data_pca)
labels = kmeans.labels_

# 使用 numpy 的 unique 函数来统计每个聚类的样本数量
unique, counts = np.unique(labels, return_counts=True)

# 打印每个聚类的样本数量
for cluster, count in zip(unique, counts):
    print(f"Cluster {cluster} contains {count} samples.")


centroids = kmeans.cluster_centers_

ax1 = plt.axes(projection='3d')
# 可视化降维后的数据点
plt.figure(figsize=(8, 6))
ax1.scatter3D(data_pca[:, 0], data_pca[:, 1],data_pca[:, 2], c=clusters, cmap='viridis')
ax1.scatter3D(centroids[:, 0], centroids[:, 1],centroids[:, 2],c='red', s=200, marker='X', label='Cluster Centroids')
plt.title("PCA")
# plt.xlabel("1")
# plt.ylabel("2")
# plt.colorbar(label="label")
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(range(len(data_pca[:, 0])), data_pca[:, 0]/10, label='p-dimension1', linewidth=1)
plt.plot(range(len(data_pca[:, 0])), data_pca[:, 1]/10, label='p-dimension2', linewidth=1)
plt.plot(range(len(data_pca[:, 0])), data_pca[:, 2]/10, label='p-dimension3', linewidth=1)
plt.title("performance of decomposition")
plt.legend()
# plt.colorbar(label="")
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(range(len(data_pca[:, 0])), original_x[:, 0], label='b-dimension1', linewidth=2)
plt.plot(range(len(data_pca[:, 0])), original_x[:, 1], label='b-dimension2', linewidth=2)
plt.plot(range(len(data_pca[:, 0])), original_x[:, 2], label='b-dimension3', linewidth=2)
plt.title("performance of before")
# plt.xlabel("1")
# plt.ylabel("2")
plt.legend()
# plt.colorbar(label="")
plt.show()

# 查看聚类结果
print(f"聚类中心：\n{kmeans.cluster_centers_}")
#
# print(x.shape)

x = x.transpose(2,1)
period_list, period_weight = FFT_for_Period(x,3)
seq_len = 150
pred_len = 1
B, T, N = x.size()
res = []
print(len(period_list))
for i in range(len(period_list)):
    period = period_list[i]
    # padding
    if (seq_len) % period != 0:
        length = (((seq_len) // period) + 1) * period
        padding = torch.zeros([x.shape[0], (length - (seq_len)), x.shape[2]]).to(x.device)
        out = torch.cat([x, padding], dim=1)
    else:
        length = (seq_len)
        out = x
        # reshape
    print(length // period, period)
    out = out.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous()
    ####  add conv_modular
    print(out.shape)
    res.append(out)
print(res.shape)
"""
"""

"""

'''
random_number = torch.randint(0, x.shape[2], (1,)).item()
print(random_number)
a = x[random_number,0,:]
b = x[random_number,1,:]
c = x[random_number,2,:]
plt.figure(figsize=(12, 6))
ax = range(len(a))
plt.plot(ax, a, label='Voltage', linewidth=1, alpha=0.6)
plt.plot(ax, b, label='Current', linewidth=1, alpha=0.6)
plt.plot(ax, c, label='Temperature', linewidth=1, alpha=0.6)
plt.legend()
plt.show()

'''

