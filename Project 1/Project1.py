# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:57:13 2017

@author: saeed
"""
import numpy as np
import pandas as pd

def covariance_prediction(ticker_returns):
    cov_matrix=np.cov(np.array(list(ticker_returns.values)))
    cor_matrix=np.corrcoef(np.array(list(ticker_returns.values)))
    print(cov_matrix)
    print("-----")
    print(cor_matrix)

ticker= pd.Series({'Google':[1,2,3], 'Apple': [3,6,9], 'AMZN':[8,10,12]})
covariance_prediction(ticker)

