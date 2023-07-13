#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * np.pi * sinefreq
    t_sine = np.linspace(0, T, nsamples, endpoint=False)
    y_sine = np.sin(w * t_sine)
    result = pd.DataFrame({
        'data' : y_sine} ,index=t_sine)
    print (result)
    return result

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

fps = 30
sine_fq = 10 #Hz
duration = 10 #seconds
sine_5Hz = sine_generator(fps,sine_fq,duration)
sine_fq = 1 #Hz
duration = 10 #seconds
sine_1Hz = sine_generator(fps,sine_fq,duration)

sine = sine_5Hz + sine_1Hz


filtered_sine = butter_highpass_filter(sine.data,10,fps)

plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(range(len(sine)),sine)
plt.title('generated signal')
plt.subplot(212)
plt.plot(range(len(filtered_sine)),filtered_sine)
plt.title('filtered signal')
plt.show()


# In[9]:


sine.data


# In[1]:


import pandas as pd
import os
import re
from scipy import stats
import numpy as np
# Include this line in case the figures not show up in the web page
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt 


# In[2]:


path = r"D:\Project\Data\TE74-base oil-test1\ES&vib"  # 读取csv文件目录路径
# listdir()--返回path指定 的 文件夹中包含的文件或者文件夹名字 的 列表
#RMS
def rms(data):
    n=len(data)
    return np.sqrt(np.sum(data**2/n))

FileNames = os.listdir(path)# 因此Filename是一个列表
Mean_List=[]
for fn in FileNames:
     if re.search(r'\.csv$', fn):
        fullfilename = os.path.join(path, fn)
        frame = pd.read_csv(fullfilename)
        df1=frame.drop([0,2])
        df = df1.reset_index(drop=True)
        # 设置某一行为列索引【表头】
        c_list = df.values.tolist()[0]  # 得到想要设置为列索引【表头】的某一行提取出来
        df.columns = c_list   # 设置列索引【表头】
        df.drop([0], inplace=True) # 将原来的那一行删掉。
        # 这里的inplace=True，表示就在df这个数据表中进行修改，默认是False
        col = df.iloc[:, 5]
        df1 =pd.DataFrame(col)
        df2=df1.apply(pd.to_numeric)
        b=df2['S2'].mean()
        #b=rms(df2['S2'])
        Mean_List.append(b)


# In[3]:


#循环读取5个数并求均值（只有一个数据集需要，另一个不需要）
from scipy import stats
data_f=pd.DataFrame(Mean_List)
data_mean=[i.mean() for i in data_f[0].rolling(window=5)]
Mean_value=[]
a=0
for i in data_mean:
    a=a+1
    if a>=5:
        Mean_value.append(i)
        a=0
    else:
        continue
Mean_value


# In[4]:


Mean_per5h=pd.DataFrame(Mean_value)
from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.plot(Mean_per5h)
plt.show()


# In[5]:


Mean_per5h


# In[6]:


ts=Mean_per5h[0] 
ts


# In[10]:


#high pass filter
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
filtered_sine = butter_highpass_filter(Mean_per5h[0],10,fps)

plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(range(len(Mean_per5h)),Mean_per5h)
plt.title('generated signal')
plt.subplot(212)
plt.plot(range(len(filtered_sine)),filtered_sine)
plt.title('filtered signal')
plt.show()


# In[12]:


filtered_sine


# In[13]:


Signal=pd.DataFrame(filtered_sine)
Signal


# In[16]:


Signal.describe()


# In[17]:


Signal.skew()


# In[18]:


Signal.kurtosis()


# In[19]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(Signal)
plt.show()


# In[20]:


#平稳性检测
from statsmodels.tsa.stattools import adfuller
print('result：',adfuller(Signal))


# In[21]:


from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.DataFrame.ewm(timeSeries, span=size).mean()

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()

def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()

'''
　　Unit Root Test
   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
   root, with the alternative that there is no unit root. That is to say the
   bigger the p-value the more reason we assert that there is a unit root
'''
def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

# 自相关和偏相关图，默认阶数为14阶
def draw_acf_pacf(ts, lags=14):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=14, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=14, ax=ax2)
    plt.show()


# In[22]:


ts=Signal[0] 


# In[23]:


draw_trend(ts, 5)


# In[24]:


testStationarity(ts)


# In[25]:


draw_acf_pacf(Signal, lags=31)


# In[26]:


#对一阶差分后的序列做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The result of white noise verification：',acorr_ljungbox(Signal, lags= 1)) #返回统计量和 p 值
#p值大于0.05，说明是非白噪声序列


# In[27]:


#对模型进行定阶
import seaborn as sns
import statsmodels.api as sm 
pmax = int(len(ts) / 10)    #一般阶数不超过 length /10
qmax = int(len(ts) / 10)
bic_matrix = []
for p in range(pmax +1):
    temp= []
    for q in range(qmax+1):
        try:
            temp.append(sm.tsa.arima.ARIMA(ts,order=(p,1,q)).fit().bic)
            print("times", "p", p, "q", q)
        except:
            temp.append(None)
        bic_matrix.append(temp)
bic_matrix= pd.DataFrame(bic_matrix)   #将其转换成Dataframe 数据结构
#fig,ax=plt.subplots(figsize=(10,8))

bic_matrix.stack()

p,q=bic_matrix.stack().idxmin() #最小值的索引
print('Optimal value under BIC is p=%d,q value is q=%d'%(p,q))
#所以可以建立ARIMA 模型，ARIMA(0,1,1)
model = sm.tsa.arima.ARIMA(ts, order=(p,1,q)).fit()
print(model.summary())        #生成一份模型报告
print(model.predict(32,37))   #为未来5天进行预测，返回预测结果、标准误差和置信区间


# In[ ]:




