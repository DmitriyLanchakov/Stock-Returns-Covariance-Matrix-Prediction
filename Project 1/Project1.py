    # -*- coding: utf-8 -*-
    """
    Created on Tue Feb  7 11:57:13 2017
    
    @author: saeed
    """
    import numpy as np
    import pandas as pd
    from arch import arch_model
    import math
    
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
    
    initial_numberof_returns=returns[1].size # size of any of the security
    
    #returns=returns[:,0:3]
    
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
    #garch 1,1 forecasting
    def simulate_GARCH(T, a0, a1, b1, sigma1):
        
        # Initialize our values
        X = np.ndarray(T)
        sigma = np.ndarray(T)
        sigma[0] = sigma1
        
        for t in range(1, T):
            # Draw the next x_t
            X[t - 1] = sigma[t - 1] * np.random.normal(0, 1)
            # Draw the next sigma_t
            sigma[t] = math.sqrt(a0 + b1 * sigma[t - 1]**2 + a1 * X[t - 1]**2)
            
        X[T - 1] = sigma[T - 1] * np.random.normal(0, 1)    
        
        return X, sigma
    
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
        
    #    print(pred_cov[-1])
    #    print(i)
    #    print('\n')
        
    
    # dccGARCH(returns[:,0:4])
    #performing garch11 to forecast the future returns which in turn is used for the dcc garch
    for n in returns:
        print(n)
        print('\n')
    # print(returns[1])
        garch11 = arch_model(n, p=1, q=1)
        res = garch11.fit(update_freq=10)
        omega=res.params[1]
        alpha=res.params[2]
        beta=res.params[3]
        # sig_ma=np.sqrt(np.mean(returns[1]))
        sig_ma=np.std(n)  # Assumption that I am making
        # print(sigma)
        # sig_ma=0
        # sigma = math.sqrt(omega / (1 - alpha - beta))
        return_forecast,sigma_forecast=simulate_GARCH(5,omega,alpha,beta,sig_ma)
        np.append(n,return_forecast)
        
        
        
    for i in range(initial_numberof_returns,returns[1].size):
        dccGARCH(returns[:,0:i])
    
    print(np.cov(returns))
    #print('\n')
    #print(pred_cov)