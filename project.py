#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[ ]:





# In[7]:


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


# In[8]:


Mean_per5h=pd.DataFrame(Mean_value)
from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.plot(Mean_per5h)
plt.show()


# In[21]:


Mean_per5h.describe()


# In[9]:


Mean_per5h.skew()


# In[10]:


Mean_per5h.kurtosis()


# In[11]:


#画出自相关性图
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(Mean_per5h)
plt.show()


# In[12]:


#平稳性检测
from statsmodels.tsa.stattools import adfuller
print('result：',adfuller(Mean_per5h))


# In[13]:


ts=Mean_per5h[0] 
ts


# In[14]:


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


# In[15]:


ts=Mean_per5h[0] 


# In[16]:


draw_trend(ts, 5)


# In[17]:


draw_ts(ts)


# In[18]:


testStationarity(ts)


# In[19]:


draw_acf_pacf(Mean_per5h, lags=31)


# In[20]:


ts_log = np.log(ts)
draw_ts(ts_log)
testStationarity(ts_log)


# In[16]:


testStationarity.draw_trend(ts_log, 12)


# In[23]:


#对一阶差分后的序列做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The result of white noise verification：',acorr_ljungbox(Mean_per5h, lags= 1)) #返回统计量和 p 值
#p值大于0.05，说明是非白噪声序列


# In[47]:


ts


# In[46]:


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


# In[1]:


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
print result

filtered_sine = butter_highpass_filter(sine.data,10,fps)

plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(range(len(sine)),sine)
plt.title('generated signal')
plt.subplot(212)
plt.plot(range(len(filtered_sine)),filtered_sine)
plt.title('filtered signal')
plt.show()


# In[21]:


#gaotong   不成功
import numpy as np
import matplotlib.pyplot as plt
def rc_high_pass(x_new, x_old, y_old, sample_rate_hz, highpass_cutoff_hz):
    dt = 1/sample_rate_hz
    rc = 1/(2*np.pi*highpass_cutoff_hz)
    alpha = rc/(rc + dt)
    y_new = alpha * (y_old + x_new - x_old)
    return y_new
def rc_low_pass(x_new, y_old, sample_rate_hz, lowpass_cutoff_hz):
    dt = 1/sample_rate_hz
    rc = 1/(2*np.pi*lowpass_cutoff_hz)
    alpha = dt/(rc + dt)
    y_new = x_new * alpha + (1 - alpha) * y_old
    return y_new
def rc_filters(xs, sample_rate_hz, 
               highpass_cutoff_hz, 
               lowpass_cutoff_hz):
    # Initialize. This can be improved to match wikipedia.
    x_prev = 0
    y_prev_high = 0
    y_prev_low = 0

    for x in xs:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz, 
                                   highpass_cutoff_hz)
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz, 
                                 lowpass_cutoff_hz)
        x_prev = x
        yield y_prev_high, y_prev_low


if __name__ == "__main__":
    """
    # RC filters for continuous signals
    """
    sample_rate = 2**13  # Close to 8 kHz
    duration_points = 2**10
    sec_duration = duration_points/sample_rate

    frequency_low = sample_rate/2**9
    frequency_high = sample_rate/2**3

    # Design the cutoff
    number_octaves = 3
    highpass_cutoff = frequency_high/2**number_octaves
    lowpass_cutoff = frequency_low*2**number_octaves

    print('Two-tone test')
    print('Sample rate, Hz:', sample_rate)
    print('Record duration, s:', sec_duration)
    print('Low, high tone frequency:', frequency_low, frequency_high)

    time_s = np.arange(duration_points)/sample_rate

    sig = np.sin(2*np.pi*frequency_low*time_s) + \
          np.sin(2*np.pi*frequency_high*time_s)

    filt_signals = np.array([[high, low]
                             for high, low in
                             rc_filters(ts, sample_rate,
                                        highpass_cutoff, lowpass_cutoff)])


# In[22]:


plt.plot(sig, label="Input signal")
plt.plot(filt_signals[:, 0], label="High-pass")
plt.plot(filt_signals[:, 1], label="Low-pass")
plt.title("RC Low-pass and High-pass Filter Response")
plt.legend()
plt.show()


# In[ ]:


#差分
diff_12 = ts_log.diff(12)
diff_12.dropna(inplace=True)
diff_12_1 = diff_12.diff(1)
diff_12_1.dropna(inplace=True)
test_stationarity.testStationarity(diff_12_1)


# In[ ]:





# In[3]:


Mean_List


# In[22]:


data_f=pd.DataFrame(Mean_List)
data_f.describe


# In[6]:


z=[1,2,3]
x=pd.DataFrame(z)
print(x)
x[0].mean()


# In[5]:


#循环读取5个数并求均值（只有一个数据集需要，另一个不需要）
from scipy import stats
data_f=pd.DataFrame(Mean_List)
data_mean=[i.mean() for i in data_f[0].rolling(window=5)]
data_mean
data_a=pd.DataFrame(data_mean)
data_a
  
    


# In[6]:


data_a.index = data_a.index + 1  
data_a


# In[7]:


a=[]
for i in range(0,len(df),6):##每隔6行取数据
     a.append(i)
for 
    data_a.drop(i)


# In[28]:


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


# In[ ]:





# In[3]:


path = r'D:\Project\Data\TE74-base oil-test1\ES&vib\20221121_183001.csv'
def rms(data):
    n=len(data)
    return np.sqrt(np.sum(data**2/n))
Mean_List=[]
frame = pd.read_csv(path)   # 直接使用 read_excel() 方法读取
df1=frame.drop([0,2])
df = df1.reset_index(drop=True)
# 设置某一行为列索引【表头】
c_list = df.values.tolist()[0]  # 得到想要设置为列索引【表头】的某一行提取出来
df.columns = c_list   # 设置列索引【表头】
df.drop([0], inplace=True) # 将原来的那一行删掉。
## 这里的inplace=True，表示就在df这个数据表中进行修改，默认是False
col = df.iloc[:, 5]
df1 =pd.DataFrame(col)
df2=df1.apply(pd.to_numeric)
#b=df2['S2'].mean()
b=rms(df2['S2'])
Mean_List.append(b)



# In[4]:


df2


# In[12]:


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


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
def rc_high_pass(x_new, x_old, y_old, sample_rate_hz, highpass_cutoff_hz):
    dt = 1/sample_rate_hz
    rc = 1/(2*np.pi*highpass_cutoff_hz)
    alpha = rc/(rc + dt)
    y_new = alpha * (y_old + x_new - x_old)
    return y_new
def rc_low_pass(x_new, y_old, sample_rate_hz, lowpass_cutoff_hz):
    dt = 1/sample_rate_hz
    rc = 1/(2*np.pi*lowpass_cutoff_hz)
    alpha = dt/(rc + dt)
    y_new = x_new * alpha + (1 - alpha) * y_old
    return y_new
def rc_filters(xs, sample_rate_hz, 
               highpass_cutoff_hz, 
               lowpass_cutoff_hz):
    # Initialize. This can be improved to match wikipedia.
    x_prev = 0
    y_prev_high = 0
    y_prev_low = 0

    for x in xs:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz, 
                                   highpass_cutoff_hz)
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz, 
                                 lowpass_cutoff_hz)
        x_prev = x
        yield y_prev_high, y_prev_low


if __name__ == "__main__":
    """
    # RC filters for continuous signals
    """
    sample_rate = 2**13  # Close to 8 kHz
    duration_points = 2**10
    sec_duration = duration_points/sample_rate

    frequency_low = sample_rate/2**9
    frequency_high = sample_rate/2**3

    # Design the cutoff
    number_octaves = 3
    highpass_cutoff = frequency_high/2**number_octaves
    lowpass_cutoff = frequency_low*2**number_octaves

    print('Two-tone test')
    print('Sample rate, Hz:', sample_rate)
    print('Record duration, s:', sec_duration)
    print('Low, high tone frequency:', frequency_low, frequency_high)

    time_s = np.arange(duration_points)/sample_rate

    sig = np.sin(2*np.pi*frequency_low*time_s) + \
          np.sin(2*np.pi*frequency_high*time_s)

    filt_signals = np.array([[high, low]
                             for high, low in
                             rc_filters(sig, sample_rate,
                                        highpass_cutoff, lowpass_cutoff)])
    


# In[8]:


plt.plot(sig, label="Input signal")
plt.plot(filt_signals[:, 0], label="High-pass")
plt.plot(filt_signals[:, 1], label="Low-pass")
plt.title("RC Low-pass and High-pass Filter Response")
plt.legend()
plt.show()


# In[5]:


plt(Mean_List)


# In[ ]:


path_file = 'D:\Project\Data\TE74-base oil-test1\ES&vib'
path_list=os.listdir(path_file)

for filename in path_list:
    path_filename=os.path.join(path,filename)
    path = path_filename
    frame = pd.read_csv(path)   # 直接使用 read_excel() 方法读取
    df1=frame.drop([0,2])
    df = df1.reset_index(drop=True)
    c_list = df.values.tolist()[0]  # 得到想要设置为列索引【表头】的某一行提取出来
    df.columns = c_list   # 设置列索引【表头】
    df.drop([0], inplace=True) # 将原来的那一行删掉。
    # 这里的inplace=True，表示就在df这个数据表中进行修改，默认是False
    col = df.iloc[:, 5]
    df1 =pd.DataFrame(col)
    df2=df1.apply(pd.to_numeric)
    b=df2['S2'].mean()
    Mean_List.append(b)


# In[3]:


path = 'D:\Project\Data\TE74-base oil-test1\ES&vib'
path_list=os.listdir(path)
for filename in path_list:
    print(os.path.join(path,filename))


# In[ ]:





# In[8]:


path = r"D:\Project\Data\TE74-base oil-test1\ES&vib"  # 读取csv文件目录路径
# listdir()--返回path指定 的 文件夹中包含的文件或者文件夹名字 的 列表
FileNames = os.listdir(path)# 因此Filename是一个列表

for fn in FileNames[0:1]:
     if re.search(r'\.csv$', fn):
        fullfilename = os.path.join(path, fn)
        df = pd.read_csv(fullfilename,encoding='utf-8',on_bad_lines='skip')
        


# In[ ]:


for fn in FileNames:
    # re.search(pattern, string, flags=0) 扫描整个字符串并返回第一个成功的匹配
    # pattern：匹配的正则表达式
    # string：要匹配的字符串
    # flags：表达式，用于控制正则表达式的匹配方式
    if re.search(r'\.csv$', fn):
        fullfilename = os.path.join(path, fn)
        df = pd.read_csv(fullfilename,encoding='utf-8',on_bad_lines='skip')
        print(fn)  # 文件名
        print(df)  # 数据


# In[13]:


path = 'D:\Project\Data\TE74-base oil-test1\ES&vib'
path_list=os.listdir(path)
for filename in path_list[0:1]:
    path_filename=os.path.join(path,filename)
    print( path_filename)


# In[16]:


path_file = 'D:\Project\Data\TE74-base oil-test1\ES&vib'
path_list=os.listdir(path_file)

for filename in path_list:
    path = os.path.join(path_file,path_list)
    print(path)


# In[5]:


a=[0,1] 
Mean_List=[1,2]
b=3
Mean_List.append(b)
Mean_List


# In[ ]:


import os
import pandas as pd
import re
path = r"./data/"  # 读取csv文件目录路径
# listdir()--返回path指定 的 文件夹中包含的文件或者文件夹名字 的 列表
FileNames = os.listdir(path)# 因此Filename是一个列表
for fn in FileNames:
    # re.search(pattern, string, flags=0) 扫描整个字符串并返回第一个成功的匹配
    # pattern：匹配的正则表达式
    # string：要匹配的字符串
    # flags：表达式，用于控制正则表达式的匹配方式
    if re.search(r'\.csv$', fn):
        fullfilename = os.path.join(path, fn)
        df = pd.read_csv(fullfilename,encoding='utf-8',on_bad_lines='skip')
        print(fn)  # 文件名
        print(df)  # 数据


# In[15]:


path ='D:\Project\Data\TE74-base oil-test1\ES&vib'

def get_file():                   #创建一个空列表
    files =os.listdir(path)
    list= []
    for file in files:
        if not  os.path.isdir(path +file):  #判断该文件是否是一个文件夹       
            f_name = str(file)        
#             print(f_name)
            tr = '\\'   #多增加一个斜杠
            filename = path + tr + f_name        
            list.append(filename)  
    return list 

list = get_file()
print('\\')  
print(list[:6])
data = pd.read_csv(list[2])
print(data.head())


# In[3]:


w


# In[9]:


a=[0,1] 
Mean_List=[]
path_file = 'D:\Project\Data\TE74-base oil-test1\ES&vib'
path_list=os.listdir(path_file)

for filename in path_list:
    path_filename=os.path.join(path,filename)
    path = path_filename
    frame = pd.read_csv(path)   # 直接使用 read_excel() 方法读取
    df1=frame.drop([0,2])
    df = df1.reset_index(drop=True)
    c_list = df.values.tolist()[0]  # 得到想要设置为列索引【表头】的某一行提取出来
    df.columns = c_list   # 设置列索引【表头】
    df.drop([0], inplace=True) # 将原来的那一行删掉。
    # 这里的inplace=True，表示就在df这个数据表中进行修改，默认是False
    col = df.iloc[:, 5]
    df1 =pd.DataFrame(col)
    df2=df1.apply(pd.to_numeric)
    b=df2['S2'].mean()
    Mean_List.append(b)
    


# In[ ]:




