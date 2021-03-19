import numpy as np
import pandas as pd

df=pd.read_csv('skoda.csv')

df_1=df.copy()
df_1=df_1.drop(['model' ], axis=1)

labers=[0, 1]
bins_price=[0,12998,92000] #0 дешево 1- дорого
bins_engineSize=[-0.1, 1.45, 2.55]
bins_tax=[-0.1, 144 , 326]

df_1['year_2010'] = pd.cut(df_1['year'], bins=[2003,2010,2020], labels=labers)
df_1['year_2017'] = pd.cut(df_1['year'], bins=[2003,2017,2020], labels=labers)
df_1=df_1.drop(['year' ], axis=1)


df_1['mileage_1'] = pd.cut(df_1['mileage'], bins=[0,15000,300001], labels=labers)
df_1['mileage_2'] = pd.cut(df_1['mileage'], bins=[0,50000,300001], labels=labers)
df_1['mileage_3'] = pd.cut(df_1['mileage'], bins=[0,10000,300001], labels=labers)
df_1['mileage_4'] = pd.cut(df_1['mileage'], bins=[0,20000,300001], labels=labers)
df_1=df_1.drop(['mileage' ], axis=1)


df_1['mpg_1'] = pd.cut(df_1['mpg'], bins=[0,57,202], labels=labers)
df_1['mpg_2'] = pd.cut(df_1['mpg'], bins=[0,70,202], labels=labers)
df_1=df_1.drop(['mpg' ], axis=1)


df_1['price'] = pd.cut(df_1['price'], bins=bins_price, labels=labers)
df_1['engineSize'] = pd.cut(df_1['engineSize'], bins=bins_engineSize, labels=labers)
df_1['tax'] = pd.cut(df_1['tax'], bins=bins_tax, labels=labers)
df_1['fuelType'] = df_1['fuelType'].apply(lambda x: 1 if x=='Petrol' else 0)
df_1['transmission'] = df_1['transmission'].apply(lambda x: 1 if x=='Manual' else 0)
df_1['fuelType']=df_1.fuelType.astype('category')
df_1['transmission']=df_1.transmission.astype('category')
print(df_1.head(10))
cols = list(df_1.columns)
a, b = cols.index('price'), cols.index('mpg_2')
cols[b], cols[a] = cols[a], cols[b]
df_1 = df_1[cols]
# print(df_1.head(50))
df_1.to_csv('file_df_all', index=False)

df_1.to_csv('file_df_1', index=False)
