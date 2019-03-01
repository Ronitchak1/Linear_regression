#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# In[ ]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[1]:


boston

df_x=pd.DataFrame()(boston.data, columns = boston.feature_names)
df_y=pd.DataFrame(boston.target)
# In[2]:


reg=linear_model.LinearRegression()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)

reg.fit(x_train,y_train)reg.coef_

# In[ ]:


a=reg.predict(x_test)


# In[ ]:


#mean square error
np.mean((a-y_test)**2)

