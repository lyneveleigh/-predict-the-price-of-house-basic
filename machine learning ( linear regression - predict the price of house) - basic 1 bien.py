#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#du doan gia nha

# tao ra bo du lieu nho ( gia  su gia nha chi phu thuoc vao dien tich)
#huan luyen mo hinh hoi quy tuyen tinh xap xi tot nhat voi du lieu
#su dung mo hinh de du doan gia mot can nha co dien tich cho truowc 


# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# In[3]:


#tao data

data = np.array([[15,2504],[17,1608],[75,7394],[10,831],[50,5509],
                [35,3011],[90,8612],[80,9036],[5,614],[65,5832]])


# In[4]:


area = data[:,0]
price = data[:,1]


# In[5]:


plt.xlabel('dien tich')
plt.ylabel('gia nha')
plt.scatter(area,price, color = 'blue')


# In[6]:


reg = linear_model.LinearRegression()

reg.fit(area.reshape(-1,1), price)


# In[7]:


plt.xlabel('dien tich')
plt.ylabel('gia nha')
plt.scatter(area,price, color = 'blue')

plt.plot(area, reg.predict(area.reshape(-1,1)), color='red')


# In[16]:


#su dung mo hinh tren du doan gia cua 3 can nha co dien tich : 19m2, 56m2 ,102.5m2

need_prediction = [19, 56, 102.5]
for element in need_prediction:
    print(reg.predict([[element]]))


# In[ ]:




