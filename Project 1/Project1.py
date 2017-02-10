# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:57:13 2017

@author: saeed
"""
import numpy as np
import pandas as pd
from arch import arch_model

#def covariance_prediction(ticker_returns):
#    cov_matrix=np.cov(np.array(list(ticker_returns.values)))
#    cor_matrix=np.corrcoef(np.array(list(ticker_returns.values)))
#    print(cov_matrix)
#    print("-----")
#    print(cor_matrix)
#
#ticker= pd.Series({'Google':[1,2,3], 'Apple': [3,6,9], 'AMZN':[8,10,12]})
#covariance_prediction(ticker)

stock_returns = pd.Series({'Google':np.random.normal(0, .2, 10),
                           'Apple':np.random.normal(0, .2, 10),
                           'Ebay':np.random.normal(0, .2, 10)})


returns=np.array(list(stock_returns.values))
returns=returns[:,0:3]

#inititial condition setting esubt and q
esubt=[]
q_list=[]

cov_matrix=np.cov(returns)
diag=cov_matrix.diagonal()
diag_matrix=np.diag(np.diag(cov_matrix))
garch11 = arch_model(diag, p=1, q=1)
res = garch11.fit(update_freq=10)
alpha=res.params[2]
beta=res.params[3]

esubt.append(np.matmul(np.linalg.inv(diag_matrix),returns))
q_list.append(np.ones((3,3)))
pred_cov=[]


##############

def dccGARCH(returns):    
    cov_matrix=np.cov(returns)
    diag=cov_matrix.diagonal()
    diag_matrix=np.diag(np.diag(cov_matrix))
    
    garch11 = arch_model(diag, p=1, q=1)
    res = garch11.fit(update_freq=10)
    alpha=res.params[2]
    beta=res.params[3]    
    
    esubt.append(np.matmul(np.linalg.inv(diag_matrix),returns))
#     print(esubt)
#     print('\n')
#     print(np.matmul(esubt[-2],esubt[-2].T))
    qdash=np.cov(esubt[-2])
#    q_start=np.ones((3,3))
    
    q_list.append(((1-alpha-beta)*qdash) + 
             (alpha*np.matmul(esubt[-2],esubt[-2].T)) + ((beta)*q_list[-1]))
    
#     print("&&&&&")
    
    qstar=np.diag(np.diag(cov_matrix))
    qstar=np.sqrt(qstar)
    
    r=np.matmul(np.linalg.inv(qstar),q_list[-1],np.linalg.inv(qstar))
    
    pred_cov_temp=np.matmul(diag_matrix,r)
    pred_cov.append(np.matmul(pred_cov_temp,diag_matrix))
    
    print(pred_cov[-1])
    print(i)
    print('\n')
    

# dccGARCH(returns[:,0:4])
for i in range(4,10):
    dccGARCH(returns[:,0:i])

#print('\n')
#print(pred_cov)