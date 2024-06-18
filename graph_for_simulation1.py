import pickle

import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler

n= 1
#min

e_time1= 60.0 *n
tr_time1=70.0 *n
c_time1= 17.9  *n
sum1=(e_time1+c_time1+tr_time1)
#equal

e_time2= 48.00048000480005 *n
tr_time2= 20.7*n
c_time2=  12.9*n
sum2=(e_time2+c_time2+tr_time2)

#EDGE ONLY
e_time3= 104.22602305985399*n
tr_time3= 0 *n
c_time3= 0 *n
sum3=(e_time3+c_time3+tr_time3)

#CLOUD ONLY

e_time4= 9.7692267195624 *n
tr_time4= 68.5 *n
c_time4= 17.8 *n
sum4=(e_time4+c_time4+tr_time4)

#original

e_time= 35.85482854494902*n
tr_time=17.2*n
c_time=12.2 *n
sum=(e_time+c_time+tr_time)

#early exit 고려 x 모델

e_time0= 36.877688998156124 *n
tr_time0= 18.0 *n
c_time0= 36.0*n
sum0=(e_time0+c_time0+tr_time0)

#Heuristic
e_time_0= 35.85482854494902*n
tr_time_0= 17.2* n
c_time_0=12.2*n
sum_0=(e_time_0+c_time_0+tr_time_0)

#col=['Algorithm','No Exit Prob','No Exit']
col=['ERDS-OSA','ERDS-HSA','Edgent-E','IAO','Cloud-Only','Edge-Only','Edgent-M']
coll=['Edge Node',"Tr","Cloud"]

tmp=np.zeros((7,3))
df_plot=pd.DataFrame(data=tmp,columns=coll)
df_plot.loc[0]=[e_time,tr_time,c_time]
df_plot.loc[1]=[e_time_0,tr_time_0,c_time_0]
df_plot.loc[2]=[e_time2,tr_time2,c_time2]
df_plot.loc[3]=[e_time0,tr_time0,c_time0]
df_plot.loc[4]=[e_time4,tr_time4,c_time4]
df_plot.loc[5]=[e_time3,tr_time3,c_time3]
df_plot.loc[6]=[e_time1,tr_time1,c_time1]

#for i in range(7):
#    for j in range(3):
#        df_plot.loc[i][j]=df_plot.loc[i][j]/sum1

plt.figure()
colors=['black','gray','silver']
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
df_plot.plot(kind="bar",stacked=True,figsize=(12, 6),color=colors)
plt.legend(loc='upper left')
plt.ylabel("Total Time Consumption(ms)",fontsize=22)
plt.xlabel("Algorithms",fontsize=20)
plt.xticks([0,1,2,3,4,5,6],labels=col,rotation=0,fontsize=17)
plt.ylim([0,15])
plt.plot()
plt.savefig('Sync_Graph.png')
plt.savefig('Sync_Graph.eps', format='eps')
plt.show()
