
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR,LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tqdm import tqdm


# 假设有一组数据集，其中WhAccu是累计能量，Voltage、Current等是其他特征
def load_data(file_path,file_list, window_size, stride):
    window_data_x = []
    window_data_y = []
    path = file_path
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
    return np.array(window_data_x), np.array(window_data_y)

# 00
# file_path = './datasets/SOC/0degC/'
# file_list = ['589_Mixed1.csv', '589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv', '590_Mixed7.csv','590_Mixed8.csv']
# 10
# file_path = './datasets/SOC/10degC/'
# file_list = ['567_Mixed1.csv', '567_Mixed2.csv', '571_Mixed4.csv', '571_Mixed5.csv', '571_Mixed6.csv', '571_Mixed7.csv','571_Mixed8.csv']
# 25
# file_path = './datasets/SOC/25degC/'
# file_list = ['551_Mixed1.csv', '551_Mixed2.csv', '552_Mixed3.csv', '552_Mixed4.csv', '552_Mixed5.csv', '552_Mixed6.csv','552_Mixed7.csv','552_Mixed8.csv']
# 40
# file_path = './datasets/SOC/40degC/'
# file_list = ['556_Mixed1.csv','556_Mixed2.csv', '557_Mixed3.csv', '562_Mixed4.csv', '562_Mixed5.csv', '562_Mixed6.csv','562_Mixed7.csv','562_Mixed8.csv']
#n10
# file_path = './datasets/SOC/n10degC/'
# file_list = ['601_Mixed1.csv', '601_Mixed2.csv','604_Mixed3.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed6.csv', '604_Mixed7.csv','604_Mixed8.csv']
#n20
file_path = './datasets/SOC/n20degC/'
file_list = ['610_Mixed1.csv', '610_Mixed2.csv','611_Mixed3.csv', '611_Mixed4.csv', '611_Mixed5.csv',  '611_Mixed6.csv', '611_Mixed7.csv','611_Mixed8.csv']


window_size = 150 
stride = 1
# 将数据转换为DataFrame
x,y = load_data(file_path,file_list,window_size,stride)

# 特征：Voltage, Current, WhAccu, Temperature, Cycle
x,y= pd.DataFrame(x.reshape(x.shape[0],-1)), pd.DataFrame(y.reshape(y.shape[0],-1))
# print(x.shape,y.shape)
# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 使用线性回归进行训练
# model = LinearRegression()
# 00 mae: 0.033189722145955586 rmse: 0.04306679502941156
# 10 mae: 0.04611883466739926 rmse: 0.05548564109627825
# 25 mae: 0.03514794799226778 rmse: 0.048165251560090175
# 40 mae: 0.06467997873179307 rmse: 0.08949772601013287
#n10 mae: 0.050166576710022434 rmse: 0.06383671771174235
#n20 mae: 0.07531779829970224 rmse: 0.09362760935507264

model = LinearSVR()
# 00 mae: 0.04445399474012348 rmse: 0.05520270162347905
# 10 mae: 0.08133021059493468 rmse: 0.09723364971948903
# 25 mae: 0.0355967630689175 rmse: 0.0513321853416500
# 40 mae: 0.06364076826209657 rmse: 0.1091758929881837
#n10 mae: 0.0493404281527856 rmse: 0.06605398814361159
#n20 mae: 0.10206325938498846 rmse: 0.127605807865582
model.fit(X_train, y_train)

# 预测SOC
y_pred = model.predict(X_test)

### lasso 输出问题
# print(y_test.dtypes())
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("mae:", mae,"rmse:",rmse)