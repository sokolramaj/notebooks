# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()

train = pd.DataFrame()
test = pd.DataFrame()
store = pd.DataFrame() 

train_fn = 'train.csv'
test_fn = 'test.csv'
store_fn = 'store.csv'
 

os.chdir("C:\\Users\\sokol.ramaj\\Documents\\Kaggle\\Rossmann")
def read_files():
    global train
    train = pd.read_csv(train_fn, parse_dates = ['Date'], infer_datetime_format = True)
    global test
    test = pd.read_csv(test_fn, parse_dates = ['Date'], infer_datetime_format = True)
    global store
    store = pd.read_csv(store_fn)

def fillna():
    global store, test
    test.Open = test.Open.fillna(1)
    distmean = store.CompetitionDistance.median()
    store.CompetitionDistance = store.CompetitionDistance.fillna(distmean)
    
    def myPinterval(x):
        if x=='Feb,May,Aug,Nov':  return([0,1,0,0,1,0,0,1,0,0,1,0])
        elif x=='Jan,Apr,Jul,Oct':  return([1,0,0,1,0,0,1,0,0,1,0,0])
        elif x== 'Mar,Jun,Sept,Dec': return([0,0,1,0,0,1,0,0,1,0,0,1])
        else: return(np.repeat(0,12).tolist())

    proInt = store.PromoInterval.apply(myPinterval).tolist()
    proInt = pd.DataFrame(proInt, columns = ['ProInt'+ str(i) for i in range(1,13)]
                                , dtype=np.int8)
    store = store.drop('PromoInterval',1).join(proInt)
    store['CompetitionOpenSinceDay'] = 1
    store['CompetitionOpenSinceDT'] = pd.to_datetime(dict(year=store.CompetitionOpenSinceYear, month=store.CompetitionOpenSinceMonth, day=store.CompetitionOpenSinceDay))
    store = store.drop(['CompetitionOpenSinceYear','CompetitionOpenSinceMonth','CompetitionOpenSinceDay'], axis='columns')
    ifnulldt = pd.to_datetime('1970-01-01')
    store.CompetitionOpenSinceDT = store.CompetitionOpenSinceDT.fillna(ifnulldt)
    
    store['Promo2Mon'] = 1
    store['Promo2Day'] = 1
    store['Promo2SinceDT'] = pd.to_datetime(dict(year=store.Promo2SinceYear, month=store.Promo2Mon, day=store.Promo2Day))
    store = store.drop(['Promo2SinceYear','Promo2Mon','Promo2Day'], axis='columns')
    
    mask = store.Promo2SinceWeek.isnull() == False
    store.loc[mask, 'Promo2SinceWeek'] = store[mask].Promo2SinceWeek.apply(lambda x: np.timedelta64(np.int(x), 'W'))
    store['Promo2SinceDT'] = store['Promo2SinceDT'] + store['Promo2SinceWeek']
    store = store.drop(['Promo2SinceWeek'], axis='columns')
    store.Promo2SinceDT = store.Promo2SinceDT.fillna(ifnulldt)
    
def reduce_memsize():
    global train, test, store
    train.Store = train.Store.astype(np.int32)
    train.DayOfWeek = train.DayOfWeek.astype(np.int8)
    train.Sales = train.Sales.astype(np.int32)
    train.Customers = train.Customers.astype(np.int32)
    train.Open = train.Open.astype(np.int8)
    train.Promo = train.Promo.astype(np.int8)
    train.StateHoliday = pd.Categorical(train.StateHoliday.astype(str))
    train.SchoolHoliday = train.SchoolHoliday.astype(np.int8)
    
    test.Id = test.Id.astype(np.int32)
    test.Store = test.Store.astype(np.int32)
    test.DayOfWeek = test.DayOfWeek.astype(np.int8)
    test.Open = test.Open.astype(np.int8)
    test.Promo = test.Promo.astype(np.int8)
    test.StateHoliday = pd.Categorical(test.StateHoliday.astype(str))
    test.SchoolHoliday = test.SchoolHoliday.astype(np.int8)
    
    store.Store = store.Store.astype(np.int32)
    store.StoreType = pd.Categorical(store.StoreType)
    store.Assortment = pd.Categorical(store.Assortment)
    store.Promo2 = store.Promo2.astype(np.int8)
    store.CompetitionDistance = store.CompetitionDistance.astype(np.int32)
    
def add_missing_dates():
    global store, train, test
    all_stores = set(store.Store)
    train_m = pd.DataFrame()
    store_by_date = train.groupby('Date')['Store'].nunique().reset_index()
    for i in store_by_date.query('Store != 1115')['Date']:
        diff_stores = all_stores.difference(set(train[train['Date']==i].Store))
        s = list(diff_stores)
        d = [i]*len(s)
        missing = pd.DataFrame(data={
                                 'Date': d, 
                                 'Store': s, 
                                 'Customers': [0]*len(s),
                                 'Sales': [0]*len(s),
                                 'Open': [0]*len(s),
                                 'Promo': [0]*len(s),
                                 'SchoolHoliday': [0]*len(s),
                                 'StateHoliday': ['0']*len(s)
                                }) 
    
        train_m = train_m.append(missing)
        train_m['DayOfWeek'] = train_m.Date.dt.dayofweek+1
    
    store_by_date = train.groupby('Date')['Store'].nunique().reset_index()
    
    return train.append(train_m[['Store','DayOfWeek','Date','Sales','Customers','Open','Promo','SchoolHoliday','StateHoliday']])    
   
def create_dummies():
    global train_df
    train_df = train_df.drop('Assortment', axis='columns').join(pd.get_dummies(train_df.Assortment, prefix='Assortment'))
    train_df = train_df.drop('StoreType', axis='columns').join(pd.get_dummies(train_df.StoreType, prefix='StoreType'))
    train_df = train_df.drop('StateHoliday', axis='columns').join(pd.get_dummies(train_df.StateHoliday, prefix='StateHoliday'))
    return 0
    
read_files()
fillna()
train = add_missing_dates()
reduce_memsize()


train_df = train.set_index('Store').join(store.set_index('Store'), how='inner')
test_df = test.set_index('Store').join(store.set_index('Store'), how='inner')

train_df = train_df.assign(days_since_comp = train_df['Date'] - train_df['CompetitionOpenSinceDT'])
train_df = train_df.assign(days_since_promo = train_df['Date'] - train_df['Promo2SinceDT'])
test_df = test_df.assign(days_since_comp = test_df['Date'] - test_df['CompetitionOpenSinceDT'])
test_df = test_df.assign(days_since_promo = test_df['Date'] - test_df['Promo2SinceDT'])

train_df.days_since_comp = (train_df.days_since_comp / np.timedelta64(1, 'D')).astype(int)
train_df.days_since_promo = (train_df.days_since_promo / np.timedelta64(1, 'D')).astype(int) 

test_df.days_since_comp = (test_df.days_since_comp / np.timedelta64(1, 'D')).astype(int)
test_df.days_since_promo = (test_df.days_since_promo / np.timedelta64(1, 'D')).astype(int) 

train_df.loc[train_df.CompetitionOpenSinceDT.dt.year <= 1970,'days_since_comp']
train_df.loc[train_df.days_since_comp < 0,'days_since_comp'] = 0
train_df.loc[train_df.Promo2SinceDT.dt.year <= 1970,'days_since_promo'] = 0
train_df.loc[train_df.days_since_promo < 0, 'days_since_promo'] = 0
test_df.loc[test_df.CompetitionOpenSinceDT.dt.year <= 1970,'days_since_comp']
test_df.loc[test_df.days_since_comp < 0,'days_since_comp'] = 0
test_df.loc[test_df.Promo2SinceDT.dt.year <= 1970,'days_since_promo'] = 0
test_df.loc[test_df.days_since_promo < 0, 'days_since_promo'] = 0


train_df.drop(['CompetitionOpenSinceDT','Promo2SinceDT'], axis=1, inplace=True)
test_df.drop(['CompetitionOpenSinceDT','Promo2SinceDT'], axis=1, inplace=True)


train_df = train_df.reset_index().set_index(['Store', 'Date']).sort_index()
test_df = test_df.reset_index().set_index(['Store', 'Date']).sort_index()
store = store.set_index('Store').sort_index()

train_df.drop('index',axis=1, inplace=True)
test_df.drop('index',axis=1, inplace=True)

create_dummies()

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=200)

X = train_df.loc[2]

y1 = X.pop('Customers')
y2 = X.pop('Sales') 

clf.fit(X[500:-100], y1[500:-100])
y1_pred = clf.predict(X[-100:])
model_result = pd.DataFrame({'Y1_true':y1[-100:].values, 'y1_pred': y1_pred}, index=y1[-100:].index)

model_result.plot()
pd.DataFrame({'imp': clf.feature_importances_, 'feature': X.columns}).sort_values(by='imp')

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=200, n_jobs=4)

customer_score = []
customer_pred = []
customer_fimp = []
sales_score = []
sales_pred = []
sales_fimp = []

for i in store.index.values:
    print("Fitting store: ", i)
    X = train_df.loc[i]
    y1 = X.pop('Customers')
    y2 = X.pop('Sales')
    
    clf.fit(X[500:-100], y1[500:-100])
    y1_pred = clf.predict(X[-100:])
    y1_score = clf.score(X, y1)
    print("----Customer pred score: ", y1_score)
    
    customer_score.append(y1_score)
    customer_pred.append(y1_pred)
    customer_fimp.append(clf.feature_importances_)
    
    clf.fit(X[500:-100], y2[500:-100])
    y2_pred = clf.predict(X[-100:])
    y2_score = clf.score(X, y2)
    print("----Sales pred score: ", y2_score)
    
    sales_score.append(y2_score)
    sales_pred.append(y2_pred)
    sales_fimp.append(clf.feature_importances_)

plt.figure(figsize=(15,8))
plt.plot(store.index.values, sales_score)
plt.plot(store.index.values, customer_score)
plt.ylim(0,-1)
plt.show()