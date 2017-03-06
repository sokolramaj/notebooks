# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:09:05 2017

@author: sokol.ramaj
"""

import pandas as pd
import numpy as np

def load_data_file(filename,dtypes,parsedate = True):
    """
    Load file to dataframe.
    """ 
    if parsedate:
        return pd.read_csv(filename, parse_dates=['Date'], dtype=dtypes)
    else:
        return pd.read_csv(filename, dtype=dtypes)

def train_test_split(X, y, test_size=0.33):
    nRows = len(X)
    split = np.int(test_size*nRows)
    
    trStart = 0
    trEnd = nRows-split
    
    teStart = nRows-split
    teEnd = nRows
    
    return X.iloc[trStart:trEnd, :], y.iloc[trStart:trEnd], X.iloc[teStart:teEnd, :], y.iloc[teStart:teEnd]


def drop_columns(df, columns):
    return df.drop(columns, axis='columns')