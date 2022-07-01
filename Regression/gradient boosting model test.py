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
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import time

data= pd.read_csv("airline-price-prediction.csv")
label = LabelEncoder()
start = time.time()

# column date
data['date'] = data['date'].str.replace("NAN", "20/2/2022")
data["date"].fillna("20/2/2022", inplace=True)
data['date'] = data['date'].str.replace("/", "-")
data['Date_Day'] = data['date'].apply(lambda x: x.split("-")[0]).astype(int)
data['Date_Month'] = data['date'].apply(lambda x: x.split("-")[1]).astype(int)
data['Date_Year'] = data['date'].apply(lambda x: x.split("-")[2]).astype(int)
del data['date']
# column airline
data['airline'] = label.fit_transform(data['airline'])

# column ch_code
data['ch_code'] = label.fit_transform(data['ch_code'])

# column num_code

data['num_code'] = data['num_code'].astype(float)
data['num_code'] = (data['num_code'] - data['num_code'].min()) / (data['num_code'].max() - data['num_code'].min())
data['num_code'].fillna(data['num_code'].mean(), inplace=True)
# colum dep_time

data['dep_time'] = pd.to_datetime(data['dep_time']).dt.strftime("%-H:%M")
data['Dep_Hour'] = pd.DatetimeIndex(data['dep_time']).hour
data['Dep_Minute'] = pd.DatetimeIndex(data['dep_time']).minute
data = data.drop(['dep_time'], axis=1)

# colum time_taken
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
data['time_taken'].fillna(data['time_taken'].mean(), inplace=True)
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
data['arr_time'] = pd.to_datetime(data['arr_time']).dt.strftime("%-H:%M")
data['arr_hours'] = pd.DatetimeIndex(data['arr_time']).hour
data['arr_minutes'] = pd.DatetimeIndex(data['arr_time']).minute
data = data.drop(['arr_time'], axis=1)

# column type
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

# column price
data['price2'] = data['price'].str.replace(",", "").astype(int)
del data['price']

# nan value
data[['ch_code', 'airline', 'Dep_Hour', 'Dep_Minute', 'arr_hours', 'arr_minutes']] = data[
    ['ch_code', 'airline', 'Dep_Hour', 'Dep_Minute', 'arr_hours', 'arr_minutes']].fillna(
    value=data[['ch_code', 'airline', 'Dep_Hour', 'Dep_Minute', 'arr_hours', 'arr_minutes']].mean()).astype(int)

print(data)
# Gradient Boosting Regressor
data.dropna(how='any',inplace=True)
X=data.iloc[:,0:15]
Y=data['price2']
bhp = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=42, test_size=0.1)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
gbr_params = {'n_estimators': 1000,'max_depth': 3, 'min_samples_split': 5, 'learning_rate': 0.01, 'loss': 'ls'}
gbr = GradientBoostingRegressor(**gbr_params)
gbr.fit(X_train_std, y_train)
mse = mean_squared_error(y_test, gbr.predict(X_test_std))
print("The mean squared error (MSE) to Gradient Boosting Regressor: {:.4f}".format(mse))
execution = time.time()-start
print("Time execution ",execution)
