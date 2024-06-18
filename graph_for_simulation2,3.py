import pickle

import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler


#origin
#TR

origin0= 54.660933177838324 #0.5배
origin= 65.25482854494902 #1배
origin2=  74.71479850464682 #1.5배
origin3= 76.21391351892925 #2배
origin4=  84.31680171821404 #3배


#GPU
'''
origin0=  71.51074158551346 #0.5배
origin= 65.25482854494902 #1배
origin2= 63.517494308042586 #1.5배
origin3= 61.58416097470925 #2배
origin4= 57.12161412811912 #3배
'''


#noexit
#TR

noexit0= 79.59926063820446
noexit=  90.87768899815612
noexit2 = 102.75756772985764
noexit3= 113.83915925739043
noexit4= 131.47826086956522



#GPU
'''
noexit0= 110.82608695652173
noexit=  90.87768899815612
noexit2 = 78.26107288595003
noexit3= 72.09891101618433
noexit4= 66.49892756970797
'''


#edge_only
#TR

edge_only0= 104.22269936015621
edge_only= 104.22269936015621
edge_only2 = 104.22269936015621
edge_only3= 104.22269936015621
edge_only4=  104.22269936015621


#GPU
'''
edge_only0= 104.22269936015621
edge_only= 104.22269936015621
edge_only2 = 104.22269936015621
edge_only3= 104.22269936015621
edge_only4=  104.22269936015621
'''



#cloud_only
#TR

cloud_only0= 61.80756632495438
cloud_only= 96.06898047842027
cloud_only2= 130.6731282954825
cloud_only3= 165.4358958184045
cloud_only4= 235.39990625117187


#GPU
'''
cloud_only0= 104.98421178059621
cloud_only= 96.06898047842027
cloud_only2= 90.12799133134146
cloud_only3= 87.15486279839182
cloud_only4= 84.1893365675835
'''


#min
#TR

min0= 112.9
min = 147.9
min2 = 182.9
min3 =  214.3
min4= 235.0



#GPU
'''
min0= 156.85
min = 147.9
min2 = 141.93333333333334
min3 = 138.95
min4= 135.96666666666667
'''


#equal
#TR

equal0= 70.90018000180001
equal= 81.60048000480006
equal2 = 91.95048000480006
equal3= 102.30048000480005
equal4= 102.00085800858007


#GPU
'''
equal0= 88.05048000480005
equal= 81.60048000480006
equal2 = 77.30048000480005
equal3= 75.15048000480006
equal4= 73.00048000480005
'''

#Heuristic
#TR

h0= 54.81787605497917
h= 65.25482854494902
h2 = 75.75217993550609
h3= 76.21391351892925
h4= 84.31680171821404



#GPU
'''
h0= 71.51074158551346
h= 65.25482854494902
h2 = 63.517494308042586
h3=  61.58416097470925
h4= 57.12161412811912
'''








#col=['Algorithm','No Exit Prob','No Exit']
col=['0.5x','1x','1.5x','2x','3x']
#col=['2/3x','1x','1.5x','2x','3x']
x=[]

original = [origin0,origin,origin2,origin3,origin4]
x.append(original)
heu=[h0,h,h2,h3,h4]
x.append(heu)
no_exit = [noexit0,noexit,noexit2,noexit3,noexit4]
x.append(no_exit)
equalibrium=[equal0,equal,equal2,equal3,equal4]
x.append(equalibrium)
edgeonly=[edge_only0,edge_only,edge_only2,edge_only3,edge_only4]
x.append(edgeonly)
cloudonly=[cloud_only0,cloud_only,cloud_only2,cloud_only3,cloud_only4]
x.append(cloudonly)
minimum = [min0,min,min2,min3,min4]
x.append(minimum)

sum = [min0,min,min2,min3,min4]

for i in range(7):
    for j in range(5):
        x[i][j]=x[i][j]/sum[j]
print(x)

plt.plot(col,x[0],c='black',linestyle='-',marker='o',label='ERDS-OSA')
plt.plot(col,x[1],c='orange',linestyle='-',marker='^',label='ERDS-HSA')
plt.plot(col,x[2],c='red',linestyle='--',marker='x',label='IAO')
plt.plot(col,x[3],c='cyan',linestyle='--',marker='o',label='Edgent- E')
plt.plot(col,x[4],c='green',linestyle='--',marker='^',label='Edge-Only')
plt.plot(col,x[5],c='gray',linestyle='--',marker='s',label='Cloud-Only')
plt.plot(col,x[6],c='blue',linestyle='-',marker='s',label='Edgent-M')
plt.ylabel("Normalized Average Time Consumption",fontsize=12)
plt.xlabel("Transmission Time",fontsize=12)
plt.legend()
plt.savefig('Sync_Graph2.png')
plt.savefig('Sync_Graph2.eps', format='eps')
plt.show()