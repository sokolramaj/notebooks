# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from helper import load_data_file, drop_columns
from sklearn.preprocessing import StandardScaler

def split_date(df, date_col):
    n_date_year = df[date_col].dt.year
    n_date_month = df[date_col].dt.month
    n_date_weeknum = df[date_col].dt.weekofyear
    n_date_day = df[date_col].dt.day
    
    return df.assign(date_year=n_date_year, date_month=n_date_month, date_weeknum=n_date_weeknum, date_day=n_date_day)

def clean_state_holiday(df):
    df.StateHoliday = df.StateHoliday.map({'0':'0', 'a':'1', 'b': '2', 'c':'3'})
    return df

def set_dummies(df, columns = ['Assortment', 'StoreType', 'StateHoliday','DayOfWeek']):
    if columns.count('DayOfWeek') > 0:
        df.DayOfWeek = df.DayOfWeek.astype(str)
        
    dummies = pd.get_dummies(df[columns])
    df = df.join(dummies)
    df = df.drop(columns, axis=1)
    
    return df

def create_competition_date_features(df):
    df = df.assign(days_since_comp = df['Date'] - df['CompetitionOpenSinceDT'])
    df = df.assign(days_since_promo = df['Date'] - df['Promo2SinceDT'])
    
    df.days_since_comp = (df.days_since_comp / np.timedelta64(1, 'D')).astype(int)
    df.days_since_promo = (df.days_since_promo / np.timedelta64(1, 'D')).astype(int) 
    
    df.loc[df.CompetitionOpenSinceDT.dt.year <= 1970,'days_since_comp']
    df.loc[df.days_since_comp < 0,'days_since_comp'] = 0
    df.loc[df.Promo2SinceDT.dt.year <= 1970,'days_since_promo'] = 0
    df.loc[df.days_since_promo < 0, 'days_since_promo'] = 0
    
    df.drop(['CompetitionOpenSinceDT','Promo2SinceDT'], axis=1, inplace=True)
    
    return df

def remove_promo_interval_flag(df):
    df = df.assign(is_promo = ((df['date_month'] == 1)  & (df['ProInt1'] == 1))  |
                              ((df['date_month'] == 2)  & (df['ProInt2'] == 1))  | 
                              ((df['date_month'] == 3)  & (df['ProInt3'] == 1))  | 
                              ((df['date_month'] == 4)  & (df['ProInt4'] == 1))  | 
                              ((df['date_month'] == 5)  & (df['ProInt5'] == 1))  | 
                              ((df['date_month'] == 6)  & (df['ProInt6'] == 1))  | 
                              ((df['date_month'] == 7)  & (df['ProInt7'] == 1))  | 
                              ((df['date_month'] == 8)  & (df['ProInt8'] == 1))  | 
                              ((df['date_month'] == 9)  & (df['ProInt9'] == 1))  | 
                              ((df['date_month'] == 10) & (df['ProInt10'] == 1)) | 
                              ((df['date_month'] == 11) & (df['ProInt11'] == 1)) |
                              ((df['date_month'] == 12) & (df['ProInt12'] == 1)))
    
    df.is_promo = df.is_promo.astype(np.int8)
    df = df.drop(['ProInt1' ,'ProInt2' ,'ProInt3',
                   'ProInt4' ,'ProInt5' ,'ProInt6',
                   'ProInt7' ,'ProInt8' ,'ProInt9',
                   'ProInt10','ProInt11','ProInt12'], axis=1)
    
    return df

def build_features(df):
    return (df.pipe(split_date, 'Date')
            .pipe(clean_state_holiday)
            #.pipe(set_dummies)
            .pipe(create_competition_date_features)
            .pipe(remove_promo_interval_flag))
    
    
def decode_promo_interval(store):
    def myPinterval(x):
        if x=='Feb,May,Aug,Nov':  return([0,1,0,0,1,0,0,1,0,0,1,0])
        elif x=='Jan,Apr,Jul,Oct':  return([1,0,0,1,0,0,1,0,0,1,0,0])
        elif x== 'Mar,Jun,Sept,Dec': return([0,0,1,0,0,1,0,0,1,0,0,1])
        else: return(np.repeat(0,12).tolist())

    #Convert the Promointerval from a string column to a set of columns with flag [0/1]
    proInt = store.PromoInterval.apply(myPinterval).tolist()
    return pd.DataFrame(proInt, columns = ['ProInt'+ str(i) for i in range(1,13)] , dtype=np.int8)

def get_date_from_series(year, month=None, day=None, week=None):
    if year is not None and month is not None:
        if isinstance(year, pd.Series) and isinstance(month, pd.Series):
            if day is None:
                day = [1] * len(year)
            
            assert len(year) == len(month)
            assert len(year) == len(day)
            
            year = year.fillna('1970')
            month = month.fillna('01')
            return pd.to_datetime(pd.DataFrame({'year': year, 'month': month, 'day':day}))     
    elif year is not None and week is not None:
        month = [1] * len(year)
        day = [1] * len(year)
        year = year.fillna('1970')
        date_df = pd.to_datetime(pd.DataFrame({'year': year, 'month': month, 'day':day}))
        
        week = week.fillna(1).apply(lambda x: np.timedelta64(np.int(x), 'W'))
        return date_df + week

def create_agg_series(dataframe, by, metric='Sales', func='mean'):
    feature_agg = pd.DataFrame()
    for item in by:
        item_str = str('') 
        if isinstance(item, list):
            item_str = '_'.join(item)
        else:
            item_str = item
            
        feature_agg[metric + '_' + item_str + '_' + func] = dataframe.groupby(item)[metric].transform(func)
    
    return feature_agg   

        
def get_random_stores(numStores=1):
    return np.random.choice(data_store.Store, numStores)


def transform_logscale(df, columns):
    log_y = df[columns].astype(np.int).apply(np.log1p)
    df.update(log_y)

def transform_expscale(df, columns):
    exp_y = df[columns].apply(np.ep)
    df.update(exp_y)
    
print('Loading data ...')
data_dir = 'C:/Users/sokol.ramaj/Documents/git-repos/notebooks/rossman/'

data_train = load_data_file(data_dir + 'train.csv',
                            {'Id':np.int32,
                             'Store':np.int32,
                             'DayOfWeek':np.int8,
                             'Sales':np.int32,
                             'Customers':np.int32,
                             'Open':np.int8,
                             'Promo':np.int8,
                             'StateHoliday':np.object, # categorical
                             'SchoolHoliday':np.int8})

data_test = load_data_file(data_dir + 'test.csv',
                            {'Id':np.int32,
                             'Store':np.int32,
                             'DayOfWeek':np.int8,
                             'Open':np.object,         # there is some nan values
                             'Promo':np.int8,
                             'StateHoliday':np.object, # categorical
                             'SchoolHoliday':np.int8})

data_store = load_data_file(data_dir + 'store.csv',
                            {'Store':np.int32,
                             'StoreType':np.object,
                             'Assortment':np.object,
                             'CompetitionDistance':np.float32,
                             'CompetitionOpenSiceMonth':np.object, # categorical
                             'CompetitionOpenSiceYear':np.object,
                             'Promo2':np.int8,
                             'Promo2SinceWeek':np.object,
                             'Promo2SinceYear':np.object,
                             'PromoInterval':np.object}, False)
    
store_states = load_data_file(data_dir + 'store_states.csv',
                              {'Store':np.int32,
                               'State':np.object},
                              parsedate = False)

print('Normalize (Sales and Customers) features...')    
transform_logscale(data_train, ['Sales','Customers'])

sales_scaler = StandardScaler()
customers_scaler = StandardScaler()
data_train['Sales'] = sales_scaler.fit_transform(data_train['Sales'].values.reshape(-1,1))
data_train['Customers'] = customers_scaler.fit_transform(data_train['Customers'].values.reshape(-1,1))

rossmann = pd.concat([data_train,data_test])
print('Add some more features ...')    


data_store['CompetitionOpenSinceDT'] = get_date_from_series(data_store.CompetitionOpenSinceYear, data_store.CompetitionOpenSinceMonth)
data_store['Promo2SinceDT'] = get_date_from_series(data_store.Promo2SinceYear, week=data_store.Promo2SinceWeek)
data_store.CompetitionDistance = data_store.CompetitionDistance.fillna(data_store.CompetitionDistance.median())
data_store = data_store.join(decode_promo_interval(data_store))

data_store = data_store.pipe(drop_columns, ['PromoInterval','CompetitionOpenSinceYear','CompetitionOpenSinceMonth','Promo2SinceYear','Promo2SinceWeek'])

rossmann = rossmann.set_index('Store').join(data_store.set_index('Store')).reset_index()
rossmann = build_features(rossmann)    

rossmann = rossmann.assign(
     s1 = create_agg_series(rossmann, ['DayOfWeek']) \
    ,s2 = create_agg_series(rossmann, ['date_weeknum']) \
    ,s3 = create_agg_series(rossmann, ['date_month']) \
    ,s4 = create_agg_series(rossmann, ['date_day'])  \
        
    ,s5 = create_agg_series(rossmann, [['StoreType','DayOfWeek']])  \
    ,s6 = create_agg_series(rossmann, [['StoreType','date_weeknum']])  \
    ,s7 = create_agg_series(rossmann, [['StoreType','date_month']])  \
    ,s8 = create_agg_series(rossmann, [['StoreType','date_day']])  \
          
    ,c1 = create_agg_series(rossmann, ['DayOfWeek'], metric='Customers') \
    ,c2 = create_agg_series(rossmann, ['date_weeknum'], metric='Customers') \
    ,c3 = create_agg_series(rossmann, ['date_month'], metric='Customers') \
    ,c4 = create_agg_series(rossmann, ['date_day'], metric='Customers')  \
        
    ,c5 = create_agg_series(rossmann, [['StoreType','DayOfWeek']], metric='Customers')  \
    ,c6 = create_agg_series(rossmann, [['StoreType','date_weeknum']], metric='Customers')  \
    ,c7 = create_agg_series(rossmann, [['StoreType','date_month']], metric='Customers')  \
    ,c8 = create_agg_series(rossmann, [['StoreType','date_day']], metric='Customers') 
)

rossmann = set_dummies(rossmann)
rossmann = rossmann.sort_values(by=['Store','Date'])
train = rossmann[rossmann.Id.isnull()]
test  = rossmann[~rossmann.Id.isnull()]

Id = test.pop('Id')

test = test.pipe(drop_columns, ['Sales','Customers']) 
_ = train.pop('Id')

 
del rossmann 

"""
stores = get_random_stores()
sample_tr = train[train['Store'].isin(stores)]
sample_te = sample_tr['Sales']
trX, trY, teX, teY = train_test_split(sample_tr.drop('Date', axis=1), sample_te, 0.7)

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=150, verbose=True)
clf.fit(trX, trY)
clf.score(teX, teY)

plt.figure(figsize=(10,6))
plt.plot(teY.tolist())
plt.plot(clf.predict(teX) - teY)
plt.show()
"""