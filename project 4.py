#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import re
from scipy import stats
import numpy as np
# Include this line in case the figures not show up in the web page
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt 


# In[3]:


path = r"D:\Project\Data\TE74-base oil-test4\ES&vib"  # 读取csv文件目录路径
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
        b=df2['s2'].mean()
        #b=rms(df2['S2'])
        Mean_List.append(b)


# In[4]:


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


# In[5]:


Mean_per5h=pd.DataFrame(Mean_value)
from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.plot(Mean_per5h)
plt.show()


# In[16]:


Mean_per5h.describe()


# In[6]:


#平稳性检测
from statsmodels.tsa.stattools import adfuller
print('result：',adfuller(Mean_per5h))


# In[7]:


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


# In[9]:


Signal=pd.DataFrame(Mean_per5h)
Signal


# In[11]:


ts=Mean_per5h[0] 
draw_trend(ts, 5)


# In[12]:


testStationarity(ts)


# In[13]:


draw_acf_pacf(Mean_per5h, lags=31)


# In[14]:


#对一阶差分后的序列做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The result of white noise verification：',acorr_ljungbox(Mean_per5h, lags= 1)) #返回统计量和 p 值
#p值大于0.05，说明是非白噪声序列


# In[15]:


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




