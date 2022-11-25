# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:12:49 2022

@author: Stefan Marinkovic
"""
# PROBATI SA FIKSNIM ALFA
import pandas as pd
import numpy as np
import random
#%%
class LinearnaRegresija:
        
    x_mean:float
    x_std: float
    
    def learn(data,class_atr,alfa,lambda_):
        y=data[[class_atr]].values        
        x=data.drop(class_atr,axis=1)
        LinearnaRegresija.x_mean=x.mean()
        LinearnaRegresija.x_std=x.std()
        x=(x-LinearnaRegresija.x_mean)/LinearnaRegresija.x_std
        x['x0']=1
        x=x.values
        m,n=np.shape(x)
        w=np.random.random((1,n))
        
        for iter in range(10000): # kod racunanja uzima nasumicno 70 instanci
            a=random.sample(range(m), 1)
            pred=x[a].dot(w.T)
            err=pred-y[a]
            grad=err.T.dot(x[a])/m + np.concatenate((2*lambda_*w[0][:-1],[0]))
            w=w-alfa*grad
               
               
            pred_ukupno=x.dot(w.T)
            err_ukupno=pred_ukupno-y
            grad_ukupno=err_ukupno.T.dot(x)/m + np.concatenate((2*lambda_*w[0][:-1],[0]))
            MSE_ukupno=err_ukupno.T.dot(err_ukupno)/m
            grad_norm_ukupno=abs(grad_ukupno).sum()
            if iter%100==0:
                decay=alfa/(iter+1)
                alfa= alfa * 1/(1 + decay * iter)   
            print(iter, grad_norm_ukupno, MSE_ukupno,w)
            if grad_norm_ukupno < 0.75: return w      
           
        return w
    
    def prediction(model,data):
        data=(data-LinearnaRegresija.x_mean)/LinearnaRegresija.x_std
        data['X0']=1
        pred=data.dot(model.T)
        return pred
        #%%
data=pd.read_csv('data/boston.csv')
model=LinearnaRegresija.learn(data,'MEDV', 1,0.00005) #smanjiti regularizaciju
new_data=data.iloc[-4:,:-1]
LinearnaRegresija.prediction(model, new_data)
data.iloc[-4:,-1]

x=data.drop('MEDV',axis=1)
x=(x-LinearnaRegresija.x_mean)/LinearnaRegresija.x_std
x['x0']=1
x=x.values
y=data[['MEDV']].values      
pred=x.dot(model.T)
err=pred-y
grad=err.T.dot(x)/506 + 2*0.005*model[0][:-1].sum()
MSE=err.T.dot(err)/506
grad_norm=abs(grad).sum()

