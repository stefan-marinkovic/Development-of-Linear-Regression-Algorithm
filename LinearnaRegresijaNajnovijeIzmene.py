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
        random.seed(10)
        w=np.random.random((1,n))
        
        for iter in range(100):
            for i in random.sample(range(m), m):   #moze i 70 instanci 331 iteracija * 70 = 23170, da prolazi  kroz sve, m - 87 iteracija * 506 = 44022
                pred=x[i].dot(w.T)
                err=pred-y[i]
                grad=(err*x[i] + np.concatenate((2*lambda_*w[0][:-1],[0])))/m
                w=w-alfa*grad
                # grad_norm=abs(grad).sum()
               # if grad_norm < 0.01: break      
               
               
            pred_ukupno=x.dot(w.T)
            err_ukupno=pred_ukupno-y
            grad_ukupno=err_ukupno.T.dot(x)/m + np.concatenate((2*lambda_*w[0][:-1],[0]))
            MSE_ukupno=err_ukupno.T.dot(err_ukupno)/m
            grad_norm_ukupno=abs(grad_ukupno).sum()
            if iter%100==0:         # moze da bude parametar koji korisnik unosi
                decay=alfa/(iter+1)
                alfa= alfa * 1/(1 + decay * iter)   
               #alfa= alfa * 1/(1 + alfa)   
            print(iter, alfa,grad_norm_ukupno, MSE_ukupno,w)
            if grad_norm_ukupno < 0.5: return w      
           
        return w
    
    def prediction(model,data):
        data=(data-LinearnaRegresija.x_mean)/LinearnaRegresija.x_std
        data['X0']=1
        pred=data.dot(model.T)
        return pred
        #%%

data=pd.read_csv('data/boston.csv')
model=LinearnaRegresija.learn(data,'MEDV', 0.001,100)



