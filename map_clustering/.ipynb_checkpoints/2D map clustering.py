#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load libraries.py
import pandas as pd
from utils.sk_som.sklearn_som.som import SOM
# from sklearn.metrics import pairwise_distances_argmin_min
# from sklearn.metrics import silhouette_samples, silhouette_score
# from collections import Counter
from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
# from statsmodels.distributions.empirical_distribution import ECDF
# import plotly as py
# from plotly.subplots import make_subplots
# import plotly.express as px
# import plotly.figure_factory as ff
# import plotly.graph_objs as go
# import plotly.graph_objects as go

import os

if not os.path.exists("images"): os.mkdir("images")

# In[2]:


# sns.set(rc={'figure.figsize':(10,5)})
# sns.set(rc={'figure.dpi':100})

get_ipython().run_line_magic('matplotlib', 'inline')
# np.random.seed(42)


# In[3]:


sdbw_metric = ["chebyshev", "cityblock", "cosine", "euclidean"]


# In[4]:




# In[8]:


# maps = ['Env']#,'Freq','H','Paleo']
# maps = ['Freq']
# maps = ['H']
maps = ['Paleo']

# In[9]:


for item in maps:
    print(item)
    X = prepation_data(item)

# In[ ]:


sdbw_choice(X, 10, sdbw_metric, 'agg')

# In[10]:


n_clust = 4

model = SOM(m=n_clust,
            n=1,
            dim=1,
            random_state=42)
model.fit(X[1].iloc[:, 2].values.reshape(-1, 1))
predictions = model.predict(X[1].iloc[:, 2].values.reshape(-1, 1))

# In[11]:


pd.DataFrame(predictions).hist()

# In[ ]:


build_map(predictions, X, maps[0], [1000])  #

# In[ ]:


build_map(predictions, X, maps[0], [1000])

# In[ ]:


build_map(predictions, X, maps[0], [0, 5, 15, 25, 35, 40])

# In[13]:


build_map(predictions, X, maps[0], [50, 60, 70])

# In[ ]:


# In[ ]:


# maps = ['Env']#,'Freq','H','Paleo']
# maps = ['Freq']
# maps = ['H']
maps = ['Paleo']
for item in maps:
    print(item)
    X = prepation_data(item)

# In[ ]:


sdbw_choice(X, 10, sdbw_metric, 'km')

# In[ ]:


n_clust = 3

model_km = KMeans(
    n_clusters=n_clust,
    tol=0.01,
    n_init=500,
    random_state=42)

predictions = model_km.fit_predict(X[1].iloc[:, 2].values.reshape(-1, 1))

# In[ ]:


build_map(predictions, X, maps[0], [100])  #

# In[ ]:


build_map(predictions, X, maps[0], [100])  #

# In[ ]:


build_map(predictions, X, maps[0], [0, 5, 15, 25, 35, 40])

# In[ ]:


build_map(predictions, X, maps[0], [10, 20, 30, 40, 50, 60, 70])

# In[ ]:


# In[ ]:


maps = ['Env']  # ,'Freq','H','Paleo']
maps = ['Freq']
# maps = ['H']
# maps = ['Paleo']
for item in maps:
    print(item)
    X = prepation_data(item)

# In[ ]:


sdbw_choice(X, 10, sdbw_metric, 'agg')

# In[ ]:


n_clust = 5
'''
model_spec =  SpectralClustering(
    n_clusters=n_clust,
    affinity = 'nearest_neighbors',
    assign_labels='discretize',
    random_state=42)

predictions = model_spec.fit_predict(X[1].iloc[:,2].values.reshape(-1,1))  

'''

# In[ ]:


# In[ ]:


# In[ ]:


# maps = ['Env']#,'Freq','H','Paleo']
# maps = ['Freq']
maps = ['H']
maps = ['Paleo']
for item in maps:
    print(item)
    X = prepation_data(item)

# In[ ]:


sdbw_choice(X, 10, sdbw_metric, 'agg')

# In[ ]:


n_clust = 6

model_agl = AgglomerativeClustering(
    n_clusters=n_clust,
    affinity='cityblock',
    linkage='average')

predictions = model_agl.fit_predict(X[1].iloc[:, 2].values.reshape(-1, 1))

# In[ ]:


# In[ ]:


build_map(predictions, X, maps[0], [100])  #

# In[ ]:


build_map(predictions, X, maps[0], [10])  #

# In[ ]:


build_map(predictions, X, maps[0], [0, 5, 15, 25, 35, 40])

# In[ ]:


build_map(predictions, X, maps[0], [10, 20, 30, 40, 50, 60, 70])

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
