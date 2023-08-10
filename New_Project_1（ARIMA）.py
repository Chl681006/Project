#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import re
from scipy import stats
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import string


# In[2]:


path = r"D:\Project\Data\TE74-base oil-test4\ES&vib"  # 读取csv文件目录路径
# listdir()--返回path指定 的 文件夹中包含的文件或者文件夹名字 的 列表
#RMS
'''def rms(data):
    n=len(data)
    return np.sqrt(np.sum(data**2/n))
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    # Printing the values of order & cut-off frequency!
    #print("Order of the Filter=", b)  # N is the order
    # Wn is the cut-off freq of the filter
    #print("Cut-off frequency=  ",a)

    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
fps = 300'''
FileNames = os.listdir(path)# 因此Filename是一个列表
#Mean_List_S2=[]
#Mean_List_Vib=[]
data_all_vib =[]
data_all_s2 =[]
group=0
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
        #df2=df1.apply(pd.to_numeric)
        #filtered_sine = butter_highpass_filter(df2.S2,8,fps)
        #Signal=pd.DataFrame(filtered_sine)
        
        col1 = df.iloc[:, 7]
        df11 =pd.DataFrame(col1)
        #df21=df11.apply(pd.to_numeric)
        #filtered_sine1 = butter_highpass_filter(df21.Vib,8,fps)
        #Signal1=pd.DataFrame(filtered_sine1)
        
        col2 = df.iloc[:, 1]
        df12 =pd.DataFrame(col2)
        
        data_df=df12.join(df11.join(df1))#将encoder, Vib,S2合并到一个DataFrame
        a=len(data_df)
        ind=[]
        for i in range(a):#循环的目的是找出第一个值大于4的index
            if int(float(data_df.encoder[i+1]))>2:
                ind.append(i+1)
        data_vib=data_df['vib']#取出Vib变量并将其转换为list
        data_s2=data_df['s2']#取出Vib变量并将其转换为list
        #这里还需要提取出S2
        my_list_vib = list(data_vib)
        my_list_s2 = list(data_s2)
        my_float_list_vib = [float(i) for i in my_list_vib]#改变list中的数值类型变为FLOAT
        my_float_list_s2 = [float(i) for i in my_list_s2]
        step = 300
        #b_vib是从ind[7]开始将每300个数据分为一组，这一组的数据就是轴承转一圈的数据
        #VIB
        b_vib = [my_float_list_vib[i:i+step] for i in range(ind[7],len(data_df),step)]#ind7是第一个大于4的点，记为第一个开始的点，步长为300是因为每两个峰之间有300个数据（采样频率是3000hz(1s收集3000个样本),600r/min=10r/s,一转300个数据）
        c_vib =pd.DataFrame(b_vib )
        d_vib =c_vib .dropna(axis=0,how='any')#删除不满1圈的数据，由于数据量过大，可以忽略
        d_vib.loc['mean'] = d_vib.mean()
        data_vib =d_vib 
        e_vib=list(data_vib.loc['mean'])#取出每一个文件的数据的均值，这就代表在这个数据范围中一个平均情况（转一圈的平均情况）
        data_all_vib .append(e_vib)
        
        #S2
        b_s2 = [my_float_list_s2[i:i+step] for i in range(ind[7],len(data_df),step)]#ind7是第一个大于4的点，记为第一个开始的点，步长为300是因为每两个峰之间有300个数据（采样频率是3000hz(1s收集3000个样本),600r/min=10r/s,一转300个数据）
        c_s2 =pd.DataFrame(b_s2 )
        d_s2 =c_s2 .dropna(axis=0,how='any')#删除不满1圈的数据，由于数据量过大，可以忽略
        d_s2.loc['mean'] = d_s2.mean()
        data_s2 =d_s2 
        e_s2=list(data_s2.loc['mean'])#取出每一个文件的数据的均值，这就代表在这个数据范围中一个平均情况（转一圈的平均情况）
        data_all_s2 .append(e_s2)
        




        
        
        
        
        
        
        
       # df22=df12.apply(pd.to_numeric)
        #filtered_sine2 = butter_highpass_filter(df22.Vib,8,fps)
        #Signal2=pd.DataFrame(filtered_sine1)
        #plt.plot(df21.Vib,'red')
        #plt.plot(Signal1,'blue')
        
        #b=Signal[0].mean()
        #b=rms(Signal[0])
        #c=rms(Signal1[0])
        #d=rms(Signal2[0])
        #Mean_List_Vib.append(c)
        #Mean_List_S2.append(b)


# In[3]:


#前5个求平均
dada_vib=pd.DataFrame(data_all_vib)
dada_s2=pd.DataFrame(data_all_s2)

group=32
result_vib = np.array_split(dada_vib,group)#将文件均匀分为32份，每一份中有5个数据
result_s2 = np.array_split(dada_s2,group)


data_every5_all_vib=[]
data_every5_all_s2=[]
for i in range(group):
    result_vib[i].loc['mean'] = result_vib[i].mean()#在DataFrame上添加一行名为mean（是5个文件的均值）
    result_s2[i].loc['mean'] = result_s2[i].mean()
    
    data_every5_vib=list(result_vib[i].loc['mean'])#放到列表中
    data_every5_s2=list(result_s2[i].loc['mean'])
    
    data_every5_all_vib .append(data_every5_vib)#将每五个数据合为一个数据合并起来
    data_every5_all_s2 .append(data_every5_s2)
data_every5_all_32_vib=pd.DataFrame(data_every5_all_vib)
data_every5_all_32_s2=pd.DataFrame(data_every5_all_s2)
data_every5_all_32_vib
    


# In[4]:


data_every5_all_32_s2


# In[5]:


dada_vib


# In[6]:


dada_s2


# In[7]:


#VIB
ser=pd.DataFrame()

for i in range(32):
    ser=pd.concat([ser,data_every5_all_32_vib.iloc[i]])
ser_np=np.array(ser)
plt.plot(ser_np)


# In[8]:


#vib
for i in range(32):
    y = np.array(data_every5_all_32_vib.iloc[i])
    x = np.array(range(i*300+1,i*300+301))
    plt.plot(x,y)


# In[9]:


#VIB
ser_1=pd.DataFrame()

for i in range(160):
    ser_1=pd.concat([ser_1,dada_vib.iloc[i]])
ser_np_1=np.array(ser_1)
plt.plot(ser_np_1)


# In[10]:


#VIB
for i in range(160):
    y = np.array(dada_vib.iloc[i])
    x = np.array(range(i*300+1,i*300+301))
    plt.plot(x,y)


# In[11]:


#S2
ser_2=pd.DataFrame()

for i in range(32):
    ser_2=pd.concat([ser_2,data_every5_all_32_s2.iloc[i]])
ser_np_2=np.array(ser_2)
plt.plot(ser_np_2)


# In[12]:


#s2
for i in range(32):
    y = np.array(data_every5_all_32_s2.iloc[i])
    x = np.array(range(i*300+1,i*300+301))
    plt.plot(x,y)


# In[13]:


#S2
ser_3=pd.DataFrame()

for i in range(160):
    ser_3=pd.concat([ser_3,dada_s2.iloc[i]])
ser_np_3=np.array(ser_3)
plt.plot(ser_np_3)


# In[14]:


#S2
for i in range(160):
    y = np.array(dada_s2.iloc[i])
    x = np.array(range(i*300+1,i*300+301))
    plt.plot(x,y)


# In[15]:


ser_np


# In[17]:


#五个均值版本
vib_mean5=pd.DataFrame(ser_np,columns=['Vib_5mean'])
s2_mean5=pd.DataFrame(ser_np_2,columns=['S2_5mean'])


#VIB
vib_linear=pd.DataFrame(ser_np_1,columns=['Vib_linear'])
s2_linear=pd.DataFrame(ser_np_3,columns=['S2_linear'])


# In[ ]:





# In[18]:


s2_linear


# In[20]:


merged_df_mean=vib_mean5.join(s2_mean5)
merged_df_linear=vib_linear.join(s2_linear)
merged_df_mean


# In[21]:


merged_df_linear


# In[22]:


merged_df_mean.describe()


# In[23]:


merged_df_linear.describe()


# In[24]:


merged_df_mean.skew()


# In[25]:


merged_df_linear.skew()


# In[26]:


#对平均的S2做平稳性检测
from statsmodels.tsa.stattools import adfuller
print('result：',adfuller(merged_df_mean.S2_5mean))


# In[27]:


#对平均的Vib做平稳性检测
from statsmodels.tsa.stattools import adfuller
print('result：',adfuller(merged_df_mean.Vib_5mean))


# In[29]:


#对linear的S2做平稳性检测
from statsmodels.tsa.stattools import adfuller
print('result：',adfuller(merged_df_linear.S2_linear))


# In[30]:


#对linear的Vib做平稳性检测
from statsmodels.tsa.stattools import adfuller
print('result：',adfuller(merged_df_linear.Vib_linear))


# In[31]:


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


# In[33]:


#对平均的S2做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The result of white noise verification：',acorr_ljungbox(merged_df_mean.S2_5mean, lags= 1))


# In[34]:


#对平均的Vib做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The result of white noise verification：',acorr_ljungbox(merged_df_mean.S2_5mean, lags= 1))


# In[35]:


#对linear的S2做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The result of white noise verification：',acorr_ljungbox(merged_df_linear.S2_linear, lags= 1))


# In[32]:


#对linear的Vib做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The result of white noise verification：',acorr_ljungbox(merged_df_linear.S2_linear, lags= 1))


# In[ ]:


#对平均的S2模型进行定阶
import seaborn as sns
import statsmodels.api as sm 
pmax = int(len(merged_df_mean.S2_5mean) / 10)    #一般阶数不超过 length /10
qmax = int(len(merged_df_mean.S2_5mean) / 10)
bic_matrix = []
for p in range(pmax +1):
    temp= []
    for q in range(qmax+1):
        try:
            temp.append(sm.tsa.arima.ARIMA(merged_df_mean.S2_5mean,order=(p,1,q)).fit().bic)
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
model = sm.tsa.arima.ARIMA(merged_df_mean.S2_5mean, order=(p,1,q)).fit()
print(model.summary())        #生成一份模型报告
print(model.predict(32,37))   #为未来5天进行预测，返回预测结果、标准误差和置信区间


# In[ ]:




