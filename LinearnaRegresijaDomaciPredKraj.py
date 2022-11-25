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
        
        for iter in range(10000):
            for i in random.sample(range(m), m):                
                pred=x[i].dot(w.T)
                err=pred-y[i]
                #grad=err*x[i]/m + 2*lambda_*w[0][:-1].sum()
                grad=err*x[i]/m + np.concatenate((2*lambda_*w[0][:-1],[0]))
                w=w-alfa*grad
                MSE=err**2/m
                grad_norm=abs(grad).sum()
                print(iter,i, grad_norm, MSE,w)
                if grad_norm < 0.012: return w            
        return w
    
    def prediction(model,data):
        data=(data-LinearnaRegresija.x_mean)/LinearnaRegresija.x_std
        data['X0']=1
        pred=data.dot(model.T)
        return pred
        #%%
        
data=pd.read_csv('data/boston.csv')
model=LinearnaRegresija.learn(data,'MEDV', 0.6,0.001)
new_data=data.iloc[-4:,:-1]
LinearnaRegresija.prediction(model, new_data)
data.iloc[-4:,-1]

x=data.drop('MEDV',axis=1)
x=(x-LinearnaRegresija.x_mean)/LinearnaRegresija.x_std
x['x0']=1
x=x.values
y=data[['MEDV']].values      
x.dot(model)
pred=LinearnaRegresija.prediction(model, x)
err=pred-y
grad=err.T.dot(x)/506 + 2*0.005*model[0][:-1].sum()
MSE=err.T.dot(err)/506
grad_norm=abs(grad).sum()