import pandas as pd
import numpy as np
data=pd.read_csv('data/house.csv')
y=data.iloc[:,[-1]]
y=y.values
x=data.drop('Price',axis=1)
x_mean=x.mean()
x_std=x.std()
x=(x-x_mean)/x_std

x['x0']=1

x=pd.DataFrame.to_numpy(x)

m,n=np.shape(x)
w=np.random.random((1,n))
pred=x.dot(w.T)
err=pred-y
grad_first=err.T.dot(x)/m + 2*0.01*w[0][:-1].sum()
grad_first=abs(grad_first).sum()

for iter in range(1000):
    pred=x.dot(w.T)
    err=pred-y
    grad=err.T.dot(x)/m    
    w=w-0.6*grad
    MSE=err.T.dot(err)/m
    grad_norm=abs(grad).sum()
    print(iter, grad_norm, MSE)
    
    
    
a=data[['Price']].values
data.values
import random
for i in random.sample(range(15),15):
    pred=x[i].dot(w.T)
    err=pred-y
    grad=err.T.dot(x)/m    
    w=w-0.6*grad
    MSE=err.T.dot(err)/m
    grad_norm=abs(grad).sum()
    print(iter, grad_norm, MSE)
x[0]
pred=x[0].dot(w.T)
err=pred-y[0]
grad=err*x[0]/m    
w=w-0.6*grad
MSE=err**2/m
grad_norm=abs(grad).sum()
print(iter, grad_norm, MSE)