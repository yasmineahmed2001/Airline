#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import time
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pickle
import time

data= pd.read_csv("E:\\phase2 project machine\\airline-price-classification.csv")
label = LabelEncoder()

# column date
data5 = data[~data['date'].str.contains('/|-')==True]
dflist=data5['date'].values.tolist()
data['date'] = data['date'].replace(dflist,'20/2/2022')
data["date"].fillna("20/2/2022", inplace=True)
data['date'] = data['date'].str.replace("/", "-")
data['Date_Day'] = data['date'].apply(lambda x: x.split("-")[0]).astype(int)
data['Date_Month'] = data['date'].apply(lambda x: x.split("-")[1]).astype(int)
data['Date_Year'] = data['date'].apply(lambda x: x.split("-")[2]).astype(int)
del data['date']

# column airline
data['airline'] = label.fit_transform(data['airline'])


# column ch_code
del data['ch_code']

# column num_code
data['num_code']= pd.to_numeric(data['num_code'], errors='coerce').fillna(1).astype(int)
data['num_code'] = data['num_code'].astype(float)
data['num_code'] = (data['num_code'] - data['num_code'].min()) / (data['num_code'].max() - data['num_code'].min())

# colum dep_time
data6 = data[~data['dep_time'].str.contains(':')==True]
dflist=data6['dep_time'].values.tolist()
data['dep_time'] = data['dep_time'].replace(dflist,"5:45:00 AM")
data['dep_time'] = pd.to_datetime(data['dep_time']).dt.strftime("%-H:%M")
data['Dep_Hour'] = pd.DatetimeIndex(data['dep_time']).hour
data['Dep_Minute'] = pd.DatetimeIndex(data['dep_time']).minute
data = data.drop(['dep_time'], axis=1)

# colum time_taken

data6 = data[~data['time_taken'].str.contains('m')==True]
dflist=data6['time_taken'].values.tolist()
data['time_taken'] = data['time_taken'].replace(dflist,"10h 10m")
data[['taken_hours', 'taken_minutes']] = data.time_taken.str.split('h', expand=True)
data['taken_minutes'] = data['taken_minutes'].map(lambda x: str(x)[:-1])
data['taken_minutes'] = pd.to_numeric(data['taken_minutes'], errors='coerce').fillna(0, downcast='infer')
data['taken_hours'] = pd.to_numeric(data['taken_hours'], errors='coerce').fillna(0, downcast='infer')
data['taken_hours'] = data['taken_hours'].astype(int)
data['taken_hours'] = 60 * data['taken_hours']
columns_list = ['taken_hours', 'taken_minutes']
data['time_taken'] = data[columns_list].sum(axis=1)
data['time_taken'] = (data['time_taken'] - data['time_taken'].min()) / (
            data['time_taken'].max() - data['time_taken'].min())
data['time_taken'].fillna(data['time_taken'].median(), inplace=True)
data = data.drop(['taken_hours', 'taken_minutes'], axis=1)


# colum stop
def checkstopping(x):
    li = x.splitlines()
    if li[0] == "1-stop":
        return 1
    elif li[0] == "2+-stop":
        return 2
    elif li[0] == "3":
        return 3
    else:
        return 0


data['stop'].fillna("3", inplace=True)
data['stop'] = data['stop'].apply(checkstopping)

# colum arr_time
data6 = data[~data['arr_time'].str.contains(':')==True]
dflist=data6['arr_time'].values.tolist()
data['arr_time'] = data['arr_time'].replace(dflist,"5:45:00 AM")
data['arr_time'] = pd.to_datetime(data['arr_time']).dt.strftime("%-H:%M")
data['arr_hours'] = pd.DatetimeIndex(data['arr_time']).hour
data['arr_minutes'] = pd.DatetimeIndex(data['arr_time']).minute
data = data.drop(['arr_time'], axis=1)

#colum type
data['type'] = label.fit_transform(data['type'])
data['type'].fillna("2", inplace=True)

# column route
data['route'] = data['route'].str.replace("NAN", "{'source': 'NAN', 'destination': 'NAN'}")
data['route'].fillna("{'source': 'NAN', 'destination': 'NAN'}", inplace=True)
data['route_source'] = data['route'].apply(lambda x: x.split(",")[0]).apply(lambda x: x.split(":")[1])
ns = data['route'].apply(lambda x: x.split(",")[1]).apply(lambda x: x.split(":")[1])
data['route_distination'] = ns.str.split("}", n=1, expand=True)[0]
data['route_source'] = label.fit_transform(data['route_source'])
data['route_distination'] = label.fit_transform(data['route_distination'])
del data['route']

data[[ 'Dep_Hour', 'Dep_Minute', 'arr_hours', 'arr_minutes']] = data[['Dep_Hour', 'Dep_Minute', 'arr_hours',
     'arr_minutes']].fillna(value=data[['Dep_Hour', 'Dep_Minute', 'arr_hours', 'arr_minutes']].mean()).astype(int)
data['airline'] = data['airline'].fillna(value=data['airline'].mode()).astype(int)

#decision tree model
start = time.time()
data.dropna(how='any',inplace=True)
X= data.drop(["TicketCategory"],axis = 1)
Y=data['TicketCategory']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=10)
Tree_model = DecisionTreeClassifier(random_state=0,max_depth =16)
Tree_model.fit(X_train,y_train)
y_prediction = Tree_model.predict(X_test)
end = time.time()
filename = 'finalized_model.sav'
pickle.dump(Tree_model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
print("Time in  Decision tree:",end-start)

#test complete
data2= pd.read_csv("E:\\phase2 project machine\\ProjectTestSamples\\ProjectTestSamples\\Milestone 2\\airline-test-samples.csv")
label = LabelEncoder()
# column date
data5 = data2[~data2['date'].str.contains('/|-')==True]
dflist=data5['date'].values.tolist()
data2['date'] = data2['date'].replace(dflist,'20/2/2022')
data2["date"].fillna("20/2/2022", inplace=True)
data2['date'] = data2['date'].str.replace("/", "-")
data2['Date_Day'] = data2['date'].apply(lambda x: x.split("-")[0]).astype(int)
data2['Date_Month'] = data2['date'].apply(lambda x: x.split("-")[1]).astype(int)
data2['Date_Year'] = data2['date'].apply(lambda x: x.split("-")[2]).astype(int)
del data2['date']

# column airline
data2['airline'] = label.fit_transform(data2['airline'])

# column ch_code
del data2['ch_code']

# column num_code
data2['num_code']= pd.to_numeric(data2['num_code'], errors='coerce').fillna(1).astype(int)
data2['num_code'] = data2['num_code'].astype(float)
data2['num_code'] = (data2['num_code'] - data2['num_code'].min()) / (data2['num_code'].max() - data2['num_code'].min())


# colum dep_time
data6 = data2[~data2['dep_time'].str.contains(':')==True]
dflist=data6['dep_time'].values.tolist()
data2['dep_time'] = data2['dep_time'].replace(dflist,"05:45")
data2['dep_time'] = pd.to_datetime(data2['dep_time']).dt.strftime("%-H:%M")
data2['Dep_Hour'] = pd.DatetimeIndex(data2['dep_time']).hour
data2['Dep_Minute'] = pd.DatetimeIndex(data2['dep_time']).minute
data2 = data2.drop(['dep_time'], axis=1)

# colum time_taken
data6 = data2[~data2['time_taken'].str.contains('m')==True]
dflist=data6['time_taken'].values.tolist()
data2['time_taken'] = data2['time_taken'].replace(dflist,"10h 10m")
data2[['taken_hours', 'taken_minutes']] = data2.time_taken.str.split('h', expand=True)
data2['taken_minutes'] = data2['taken_minutes'].map(lambda x: str(x)[:-1])
data2['taken_minutes'] = pd.to_numeric(data2['taken_minutes'], errors='coerce').fillna(0, downcast='infer')
data2['taken_hours'] = pd.to_numeric(data2['taken_hours'], errors='coerce').fillna(0, downcast='infer')
data2['taken_hours'] = data2['taken_hours'].astype(int)
data2['taken_hours'] = 60 * data2['taken_hours']
columns_list = ['taken_hours', 'taken_minutes']
data2['time_taken'] = data2[columns_list].sum(axis=1)
data2['time_taken'] = (data2['time_taken'] - data2['time_taken'].min()) / (
            data2['time_taken'].max() - data2['time_taken'].min())
data2['time_taken'].fillna(data2['time_taken'].median(), inplace=True)
data2 = data2.drop(['taken_hours', 'taken_minutes'], axis=1)

# colum stop
def checkstopping(x):
    li = x.splitlines()
    if li[0] == "1-stop":
        return 1
    elif li[0] == "2+-stop":
        return 2
    elif li[0] == "3":
        return 3
    else:
        return 0


data2['stop'].fillna("3", inplace=True)
data2['stop'] = data2['stop'].apply(checkstopping)

# colum 
data6 = data2[~data2['arr_time'].str.contains(':')==True]
dflist=data6['arr_time'].values.tolist()
data2['arr_time'] = data2['arr_time'].replace(dflist,"5:45:00 AM")
data2['arr_time'] = pd.to_datetime(data2['arr_time']).dt.strftime("%-H:%M")
data2['arr_hours'] = pd.DatetimeIndex(data2['arr_time']).hour
data2['arr_minutes'] = pd.DatetimeIndex(data2['arr_time']).minute
data2 = data2.drop(['arr_time'], axis=1)

#colum type
data2['type'] = label.fit_transform(data2['type'])
data2['type'].fillna("2", inplace=True)

# column route

data2['route'] = data2['route'].str.replace("NAN", "{'source': 'NAN', 'destination': 'NAN'}")
data2['route'].fillna("{'source': 'NAN', 'destination': 'NAN'}", inplace=True)
data2['route_source'] = data2['route'].apply(lambda x: x.split(",")[0]).apply(lambda x: x.split(":")[1])
ns = data2['route'].apply(lambda x: x.split(",")[1]).apply(lambda x: x.split(":")[1])
data2['route_distination'] = ns.str.split("}", n=1, expand=True)[0]
data2['route_source'] = label.fit_transform(data2['route_source'])
data2['route_distination'] = label.fit_transform(data2['route_distination'])
del data2['route']

data2[[ 'Dep_Hour', 'Dep_Minute', 'arr_hours', 'arr_minutes']] = data2[['Dep_Hour', 'Dep_Minute', 'arr_hours',
     'arr_minutes']].fillna(value=data2[['Dep_Hour', 'Dep_Minute', 'arr_hours', 'arr_minutes']].mean()).astype(int)
data2['airline'] = data2['airline'].fillna(value=data2['airline'].mode()).astype(int)

data2.dropna(how='any',inplace=True)
X_train= data.drop(["TicketCategory"],axis = 1)
Y_train=data['TicketCategory']
X_test = data2.drop(["TicketCategory"],axis = 1)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
Tree_model.fit(X_train_std,Y_train)
y_prediction = Tree_model.predict(X_test)
print(y_prediction)

