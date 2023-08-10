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

for fn in FileNames[0:1]:
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


df1


# In[4]:


df11


# In[5]:


df12


# In[6]:


data_df=df12.join(df11.join(df1))
data_df


# In[10]:


a=len(data_df)

ind=[]
for i in range(a):
    if int(float(data_df.encoder[i+1]))>2:
        ind.append(i+1)
ind


# In[11]:


ind[7]


# In[12]:


len(data_df.encoder)


# In[13]:


w=data_df['vib']
my_list = list(w)
my_list


# In[14]:


my_float_list = [float(i) for i in my_list]
my_float_list


# In[15]:


step = 300
b = [my_float_list[i:i+step] for i in range(ind[7],len(data_df),step)]
print(b)


# In[16]:


c=pd.DataFrame(b)
c


# In[17]:


d=c.dropna(axis=0,how='any')
d


# In[18]:


d.loc['mean'] = d.mean()
data=d
data


# In[ ]:


step = 5
b = [my_float_list[i:i+step] for i in range(ind[7],len(data_df),step)]
print(b)


# In[23]:


data.loc['mean']


# In[26]:


www=list(data.loc['mean'])
www


# In[28]:


data_all=[]
data_all.append(www)
data_all


# In[29]:


wwwwww=pd.DataFrame(data_all)
wwwwww


# In[21]:


plt.plot(data.loc['mean'])


# In[ ]:





# In[ ]:


#循环读取5个数并求均值（只有一个数据集需要，另一个不需要）
from scipy import stats
data_f=pd.DataFrame(Mean_List_S2)
data_f1=pd.DataFrame(Mean_List_Vib)
data_mean=[i.mean() for i in data_f[0].rolling(window=5)]
data_mean1=[i.mean() for i in data_f1[0].rolling(window=5)]
Mean_value_S2=[]
Mean_value_Vib=[]
a=0
for i in data_mean:
    a=a+1
    if a>=5:
        Mean_value_S2.append(i)
        a=0
    else:
        continue
b=0
for i in data_mean1:
    b=b+1
    if b>=5:
        Mean_value_Vib.append(i)
        b=0
    else:
        continue
Mean_per5h_S2=pd.DataFrame(Mean_value_S2,columns=['S2'])
Mean_per5h_Vib=pd.DataFrame(Mean_value_Vib,columns=['Vib'])
merged_df=Mean_per5h_Vib.join(Mean_per5h_S2)
train=merged_df
X_train = np.array(train)


# In[30]:


import pandas as pd
 
 
dict=[[1,2,3,4,5,6],[2,3,4,5,6,7],[3,4,5,6,7,8],[4,5,6,7,8,9],[5,6,7,8,9,10]]
data=pd.DataFrame(dict)
print(data)


# In[31]:


for indexs in data.index:
    print(data.loc[indexs].values, data.loc[indexs].values[0], data.loc[indexs].values[1])


# In[ ]:




