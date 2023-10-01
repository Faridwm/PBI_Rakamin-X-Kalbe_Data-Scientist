#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px

from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf, plot_predict
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, silhouette_score, silhouette_samples


# # Load Datasets

# In[2]:


df_customers = pd.read_csv("Case Study Data Scientist\Case Study - Customer.csv", sep=';')
df_products = pd.read_csv("Case Study Data Scientist\Case Study - Product.csv", sep=';')
df_stores = pd.read_csv("Case Study Data Scientist\Case Study - Store.csv", sep=';')
df_transactions = pd.read_csv("Case Study Data Scientist/Case Study - Transaction.csv", sep=';')


# ## Preview Datasets
# Dalam preview datasets, kita akan melihat lima data teratas dari masing-masing datasets. Kemudian akan dilkaukan pengecekkan tipe-tipe data pada masing-masing datasets beserta apakah terdapat atau tidaknya nilai null pada datasets. 

# ### Customers Datasets

# In[3]:


df_customers.head()


# In[4]:


df_customers.info()


# In[5]:


df_customers.isna().sum()


# In[6]:


df_customers['CustomerID'].duplicated().sum()


# ### Product Datasets

# In[153]:


df_products


# In[8]:


df_products.info()


# In[9]:


df_products.isna().sum()


# In[10]:


df_products['ProductID'].duplicated().sum()


# ### Stores Datasets

# In[11]:


df_stores.head()


# In[12]:


df_stores.info()


# In[13]:


df_stores.isna().sum()


# In[14]:


df_stores['StoreID'].duplicated().sum()


# ### Transactions Datasets

# In[15]:


df_transactions.head()


# In[16]:


df_transactions.info()


# In[17]:


df_transactions.isna().sum()


# In[18]:


df_transactions['TransactionID'].duplicated().sum()


# Dari preview keempat datasets diatas, dapat diketahui hal-hal berikut:
# - Pada datasets customers terdapat nilai null pada kolom 'Marital Status'. Pada datasets yang sama kolom 'Income' memiliki tipe object, seharusnya kolom tersebut bertipe data float dikarenakan kolom terebut merujuk pada pendapatan tiap customer.
# - Pada datasets stores kolom 'latitude' dan 'langitude' seharusnya beripe float bukan object.
# - Pada datasets transactions kolom 'Date' seharusnya bertipe datetime dikarenakan kolom tersebut menunjukkan waktu. Selain itu pada datasets ini juga memiliki baris/record yang duplikat, hal ini disimpulkan dengan adanya duplikasi pada kolom TransactionID yang seharusnya kolom ini bersifat unik pada setiap recordnya

# # Cleaning Datasets

# ## Customers Datasets

# In[19]:


# Ubah tipe data pada kolom Income
df_customers['Income'] = df_customers['Income'].replace('[,]', '.', regex=True).astype('float')


# In[20]:


# Isi data Marital Status yang kosong menjadi 'Unknown'
df_customers['Marital Status'] = df_customers['Marital Status'].fillna('Unknown')


# In[21]:


df_customers.info()


# Kita tidak dapat mmebuang data Customers yang memiliki nilai null pada salah satu kolomnya dikarenakan data customers tersebut terhubung dengan datasets transactions. Oleh karena itu, data yang null pada kolom 'Marital Status' diisi dengan 'Unknown' dikarenakan kita tidak mengetahui Status pernikahan dari customers tersebut.

# ## Stores Datasets

# In[22]:


# ubah tipe data kolom latitude dan longitude
df_stores['Latitude'] = df_stores['Latitude'].replace('[,]', '.', regex=True).astype('float')
df_stores['Longitude'] = df_stores['Longitude'].replace('[,]', '.', regex=True).astype('float')


# In[23]:


df_stores.info()


# ## Transactions Datasets

# In[24]:


df_transactions['Date'] = pd.to_datetime(df_transactions['Date'], format='%d/%m/%Y')


# In[25]:


df_transactions['TransactionID'].value_counts()


# In[26]:


df_transactions[df_transactions['TransactionID'] == 'TR71313']


# In[27]:


df_transactions = df_transactions.drop_duplicates(subset='TransactionID', keep='last')


# In[28]:


df_transactions.info()


# # Merge Datasets

# In[29]:


# Buat merge df_transactions dan df_customers
df_merge = pd.merge(df_transactions, df_customers, how='inner', on='CustomerID')

# merge hasil merge sebelumnya/df_merge dengan df_products
df_merge = pd.merge(df_merge, df_products[['ProductID', 'Product Name']], how='inner', on='ProductID')

# merge hasil merge sebelumnya/df_merge dengan df_stores
df_merge = pd.merge(df_merge, df_stores, how='inner', on='StoreID')


# In[30]:


df_all = df_merge.copy()


# # Descriptive Analysis

# In[31]:


df_all.head()


# In[32]:


df_all.info()


# In[33]:


print("Banyak baris dan kolom dari seluruh datasets adalah sebanyak {} baris dan {} kolom".format(df_all.shape[0], df_all.shape[1]))


# In[34]:


df_all.describe(include='number')


# In[35]:


df_all.describe(include='object')


# # Analisis Regresi

# ## Analisis Time Series

# In[36]:


df_all[['Date', 'Qty']].sort_values('Date')


# In[37]:


df_tsa = df_all[['Date', 'Qty']]


# In[38]:


df_tsa = df_tsa.groupby('Date').sum()


# In[39]:


df_tsa


# In[40]:


fig, ax = plt.subplots(figsize=(15, 5))
plt.xlabel('Date')
plt.ylabel('Number of Qty')
plt.tight_layout()
plt.plot(df_tsa)
plt.show()


# In[41]:


## plot graph
fig, ax = plt.subplots(figsize=(15, 5))
plt.xlabel('Date')
plt.ylabel('Number of Qty')
plt.tight_layout()
plt.plot(df_tsa.resample('M').mean())
plt.show()


# ### STL

# In[42]:


res = STL(df_tsa['Qty']).fit()
fig = res.plot()
fig.set_figwidth(20)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ### Decomposed Time Series

# In[43]:


decomposed = sm.tsa.seasonal_decompose(df_tsa['Qty'])


# In[44]:


fig, axes = plt.subplots(4, 1, figsize=(20, 8))

decomposed.observed.plot(ax=axes[0])
axes[0].set_title("Observed")

decomposed.trend.plot(ax=axes[1])
axes[1].set_title("Trend")

decomposed.seasonal.plot(ax=axes[2])
axes[2].set_title("Seasonal")

decomposed.resid.plot(ax=axes[3])
axes[3].set_title("Residual")

plt.tight_layout()
plt.show()


# In[45]:


fig, axes = plt.subplots(figsize=(20, 4))
decomposed.seasonal.plot()
plt.tight_layout()
plt.show()


# ### Data Stationary or Not

# #### Menggunakan rolling dan visualisasi

# In[46]:


rolmean = df_tsa.rolling(window=30, step=1).mean()
rolstd = df_tsa.rolling(window=30, step=1).std()


# In[47]:


fig, ax = plt.subplots(figsize=(15, 5))
plt.xlabel('Date')
plt.ylabel('Number of Qty')
plt.tight_layout()
plt.plot(df_tsa, label='Orignal Data')
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolstd, label='Rolling STD')
plt.legend(loc='best')
plt.show()


# #### Metode ADF

# In[48]:


adf_result = adfuller(df_tsa)


# In[49]:


print('ADF Statistik: {:.4f}'.format(adf_result[0]))
print('p-value: {:.4f}'.format(adf_result[1]))
print('Critical Values:')
for key, value in adf_result[4].items():
    print('\t{}: {:.3f}'.format(key, value))
if adf_result[1] > 0.05:
    print("Terima H0: Data Non-Stationary")
else:
    print("Tolak H0: Data Stationary")


# ## Split data

# In[50]:


# split data
split = int(np.round(df_tsa.shape[0] * 0.8))
df_tsa_train, df_tsa_test = df_tsa.iloc[:split], df_tsa.iloc[split:]


# In[51]:


fig, ax = plt.subplots(figsize=(20, 5))
plt.plot(df_tsa_train['Qty'], label='Train Data')
plt.plot(df_tsa_test['Qty'], label='Test Data')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ## Forecast Qty dengan ARIMA

# In[52]:


def evaluation(y_true, y_pred, title=None):
    if title:
        print(title)
    print("Mean Absolute Error (MAE): {:.2f}".format(mean_absolute_error(y_true,y_pred)))
    print("Root Mean Squared Error (RMSE): {:.2f}".format(np.sqrt(mean_squared_error(y_true, y_pred))))
    print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mean_absolute_percentage_error(y_true, y_pred) * 100))


# ### Auto ARIMA

# Menggunakan Auto ARIMA untuk menemukan nilai p, d dan q dan model dengan menguji coba kombinasi order dan memperkecil nilai AIC

# In[53]:


auto_model = auto_arima(df_tsa_train, trace=True, error_action='ignore', suppress_warnings=True)
auto_model.fit(df_tsa_train)


# In[54]:


auto_model.summary()


# In[55]:


y_pred_autoARIMA = auto_model.predict(len(df_tsa_test))
y_pred_autoARIMA


# In[56]:


fig, ax = plt.subplots(figsize=(20, 5))
plt.plot(df_tsa_test, label='Test Data')
plt.plot(y_pred_autoARIMA, color='red', linestyle='--')
plt.tight_layout()
plt.show()


# In[57]:


evaluation(df_tsa_test['Qty'], y_pred_autoARIMA)


# ### ARIMA

# #### Menemukan p, d dan q optimal

# ##### PACF dan ACF

# In[58]:


fig, axes = plt.subplots(2, 1, figsize=(8, 4))
plot_pacf(df_tsa['Qty'], lags=50, zero=False, auto_ylims=True, ax=axes[0])
plot_acf(df_tsa['Qty'], lags=50, zero=False, auto_ylims=True, ax=axes[1])
plt.tight_layout()
plt.show()


# ##### Pandas Auto Correlation

# In[59]:


fig, ax = plt.subplots(figsize=(12, 4))
pd.plotting.autocorrelation_plot(df_tsa['Qty'], ax=ax)
plt.ylim(-0.2, 0.2)
plt.xlim(0, 50)
plt.xticks(np.arange(0, 50, step=2))
plt.tight_layout()
plt.show()


# #### Membuat model ARIMA

# In[60]:


y = df_tsa_train['Qty']
ARIMA_model = ARIMA(y, order=(10, 0, 10)) 
ARIMA_model = ARIMA_model.fit()


# In[61]:


ARIMA_model.summary()


# In[62]:


y_pred_ARIMA = ARIMA_model.get_forecast(len(df_tsa_test))


# In[63]:


y_pred_ARIMA_df = y_pred_ARIMA.conf_int()
y_pred_ARIMA_df['predictions'] = ARIMA_model.predict(start=y_pred_ARIMA_df.index[0], end=y_pred_ARIMA_df.index[-1])
y_pred_ARIMA_df


# #### Evaluasi

# In[64]:


evaluation(df_tsa_test['Qty'], y_pred_ARIMA_df['predictions'])


# In[65]:


fig, ax = plt.subplots(figsize=(20, 5))
plt.plot(df_tsa_test, label='Test Data')
plt.plot(y_pred_ARIMA_df['predictions'], color='red', linestyle='--', label='Predict Data')
plt.legend()
plt.tight_layout()
plt.show()


# ### Kesimpulan

# In[66]:


fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(df_tsa_train, label='Train Data')
ax.plot(df_tsa_test, label='Test Data')
plot_predict(ARIMA_model, start=y_pred_ARIMA_df.index[0], end=pd.Timestamp('2023-02-28'), ax=ax)
plt.xticks(pd.date_range(start='2022-01-01', end='2023-03-01', freq='MS'))
plt.tight_layout()
plt.show()


# # Analisis Klasterisasi

# ## Persiapan Datasets

# In[67]:


df_all.head(10)


# In[68]:


# Cek korelasi antara feature
df_corr = df_all.select_dtypes(include='number').corr()


# In[69]:


# heatmap korelasi antar feature
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df_corr, cmap='Blues', fmt='.1f', annot=True, vmin=-1)
plt.show()


# In[108]:


df_cust_seg = df_all.groupby('CustomerID').agg(
    Total_Transactions = ('TransactionID', 'count'),
    Total_Qty = ('Qty', 'sum'),
    Total_Amount = ('TotalAmount', 'sum')
)


# In[109]:


df_cust_seg


# In[110]:


df_cust_seg.columns[0]


# ## Preprocess

# In[111]:


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):
    sns.kdeplot(data=df_cust_seg, x=df_cust_seg.columns[i], ax=ax)
plt.tight_layout()
plt.show()


# In[151]:


df_cust_seg[['Total_Amount']].boxplot()


# In[112]:


scaler = StandardScaler()


# In[113]:


scaler_data = scaler.fit_transform(df_cust_seg)


# In[114]:


scaler_data


# ## Pemilihan Jumlah Klaster

# In[115]:


k = range(2, 15)
kmeans_list = []
for i in k:
    cls_kmeans = KMeans(i, random_state=42, n_init='auto')
    cls_kmeans.fit(scaler_data)
    kmeans_list.append(cls_kmeans)


# In[116]:


inertia = []
sil_score = []

for i in kmeans_list:
    # cls_kmeans = KMeans(i, random_state=42, n_init='auto')
    # cls_kmeans.fit(scaler_data)
    labels = i.labels_
    inertia.append(i.inertia_)
    sil_score.append(silhouette_score(scaler_data, labels, random_state=42))


# In[117]:


print(inertia)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(k, inertia, marker='o')
ax.set_title('Elbow method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')
plt.tight_layout()
plt.show()


# In[118]:


fig, axes = plt.subplots(3, 2, figsize=(10, 10))
for j, kmeans in enumerate(kmeans_list[:6]):
    q, mod = divmod(j, 2)
    axes[q][mod].set_xlim([-0.1, 1])
    axes[q][mod].set_ylim([0, len(scaler_data) + (len(kmeans.cluster_centers_) + 1) * 10])
    cls_labels = kmeans.labels_
    silhouette_avg = silhouette_score(scaler_data, cls_labels)
   
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(scaler_data, cls_labels)

    y_lower = 10
    for i in range(len(kmeans_list)):
        ith_cluster_silhouette_values = sample_silhouette_values[cls_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / len(kmeans.cluster_centers_))
        axes[q][mod].fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # # Label the silhouette plots with their cluster numbers at the middle
        # axes[q][mod].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for the 0 samples

    axes[q][mod].set_title("The silhouette plot for the various clusters.")
    axes[q][mod].set_xlabel("The silhouette coefficient values")
    axes[q][mod].set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    axes[q][mod].axvline(x=silhouette_avg, color="red", linestyle="--")
    axes[q][mod].set_yticks([])  # Clear the yaxis labels / ticks
    axes[q][mod].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.tight_layout()
plt.show()


# ## Model K-Means dengan Optimum K

# In[119]:


cls_ml = KMeans(4, n_init='auto', random_state=42)
cls_ml.fit(scaler_data)


# In[120]:


seg_result = df_cust_seg.copy()
seg_result['Cluster'] = cls_ml.labels_
seg_result['Cluster'] = seg_result['Cluster'].astype('category')


# In[121]:


seg_result


# In[126]:


seg_result.groupby('Cluster').agg(['sum', 'mean']).round(2).applymap(lambda x: f"{x:,}")


# In[122]:


px.scatter_3d(data_frame=seg_result, 
              x='Total_Transactions', y='Total_Qty', z='Total_Amount', 
              color='Cluster', size_max=1, width=700, height=700)


# - Cluster 0: Pelanggan Reguler
# - Cluster 1: Pelanggan Premium
# - Cluster 2: Pelanggan Hemat
# - Cluster 3: Pelanggan Ekonomis

# In[134]:


cls_names = {
    0: 'Reguler',
    1: 'Premium',
    2: 'Hemat',
    3: 'Ekonomis'}


# In[154]:


plt.pie(seg_result['Cluster'].value_counts(), 
        labels=[cls_names[i] for i in seg_result['Cluster'].value_counts().index], 
        autopct='%.0f%%',
        colors=['#00CC96', '#EF553B', '#AB63FA', '#636EFA'])
plt.title('Percentage Number of Cluster')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# In[155]:


df_clustered = pd.merge(df_merge, seg_result.reset_index()[['CustomerID', 'Cluster']], how='inner', on='CustomerID')


# In[160]:


df_clustered


# In[156]:


uq_label = np.sort(df_clustered['Cluster'].unique())


# In[157]:


dfs_clustered = {i: df_clustered[df_clustered['Cluster'] == i] for i in uq_label}


# In[166]:


for cls in uq_label:
    print('Cluster', cls)
    prod_anlys = dfs_clustered[cls].groupby(['Product Name']).agg({'Qty':'sum'}).sort_values('Qty', ascending=False)
    print(prod_anlys)
    print('============')


# In[ ]:




