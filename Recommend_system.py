#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


# In[2]:


amazon_ratings = pd.read_csv("C:\\Users\\prateek.dubey\\Desktop\\ratings_Beauty.csv")


# In[3]:


amazon_ratings1 = amazon_ratings.head(10000)
# converting spreadsheet-style pivot table as a DataFrame.
ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
ratings_utility_matrix.head()


# In[4]:


# Transposing the matrix
X = ratings_utility_matrix.T
X.shape



# In[5]:


#performs linear dimensionality reduction 
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
print(decomposed_matrix.shape)
# finding matrix of correlation coefficients
correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape


# In[6]:


def recommend_sytem(item_id):
    product_names = list(X.index)
    product_ID = product_names.index(item_id)
    #finding Correlation for all items with the item purchased by this customer based on items rated by other 
    #customers people who bought the same product
    correlation_product_ID = correlation_matrix[product_ID]
    correlation_product_ID.shape
    Recommend = list(X.index[correlation_product_ID > 0.90])

    # Removes the item already bought by the customer
    Recommend.remove(item_id) 

    return Recommend[0:9]
print(recommend_sytem("0762451459"))
# we are taking id of items ,it can be anything from items  but for example i have taken this id.


# In[ ]:





# In[ ]:




