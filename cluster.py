import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# Load the csv file
df = pd.read_csv("storefront.csv")

# convert to date
df['OrderDate'] = pd.to_datetime(df['OrderDate'], infer_datetime_format=True)
df['ShipDate'] = pd.to_datetime(df['ShipDate'], infer_datetime_format=True)


# check min as max date
least_recent_date = df['OrderDate'].min()
most_recent_date = df['OrderDate'].max()


# consider only last year
df = df[df['OrderDate'].dt.year == 2017]


# Recency
date = '2018-01-01'
recency = pd.to_datetime(date).normalize() - df.groupby('CustomerID').agg(MaximumDate=('OrderDate', np.max))
recency = recency.astype(str).replace(r'\D+', '', regex=True)


# Frequency
frequency = df[pd.to_datetime(df['OrderDate']).dt.month == 12].groupby('CustomerID')['RowID'].count()


# Monetary
monetary = df[pd.to_datetime(df['OrderDate']).dt.month == 12].groupby('CustomerID')['Sales'].sum()


# fill null to 0
f_df = recency.join(frequency).join(monetary).rename(columns={'RowID': 'Count'}).reset_index()
convert_dict = {'CustomerID': str,
                'MaximumDate': int,
                'Count': float,
                'Sales': float
                }

f_df = f_df.astype(convert_dict)
f_df = f_df.fillna(0)


# k means
k_means = KMeans(n_clusters=2, random_state=42)

# model train
k_means.fit(f_df[['MaximumDate', 'Count', 'Sales']])


# Make pickle file of our model
pickle.dump(k_means, open("cluster.pkl", "wb"))