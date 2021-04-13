import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.animation as animation
import imageio
import os
import plotly.express as px

#Assureur auto
data1 = pd.read_csv('tab1.csv')
data1.columns = ['Assureurs','CA2019','CA2018','Variation1','PartSeg2019','NbreContrats2019','Variation2','PartAssAuto',"Unnamed1","Unnamed2"]
data1.pop("Unnamed1")
data1.pop("Unnamed2")
#data1 = data1.set_index('Assureurs')
#diff = ['69.0','26.0','68.0','41.3','22.0','45.0','93.2','72.0','41.0','34.8','45.3','9.7','15.3','19.3','1.2','4.5','6.1','13.8','5.0']
#data1['diffCAauto']=diff


data1['VarNbreContratAuto'] = data1['NbreContrats2019'] - data1['Variation2']
#data1['PartSeg2019'].fillna(0, inplace = True)
data1.tail(21)

import statsmodels.api as sm
X = np.array(data1['PartSeg2019']).reshape(-1,1)
Y = data1['NbreContrats2019']

regr = LinearRegression()
regr.fit(X,Y)
y_pred = regr.predict(X)

print(regr.coef_/100)

print('--------------------------')

print(regr.intercept_)



plt.figure(figsize=(23,15))

sns.barplot(x = data1['Assureurs'], y = data1['NbreContrats2019'], palette="Reds_r")
plt.xlabel('\nAssureurs', fontsize=15, color='#c0392b')
plt.ylabel("NbreContrats2019\n", fontsize=15, color='#c0392b')
plt.title("NbreContrats2019 par rapport aux assureurs\n", fontsize=18, color='#e74c3c')
plt.xticks(rotation= 45)
plt.tight_layout()