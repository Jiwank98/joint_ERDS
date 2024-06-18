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

origin0= 94.84 #0.5배
origin= 65.25482854494902 #1배 (기존)
origin2=  60.20  #1.25배
origin3= 54.38# 1.6배
origin4= 20.9  #2배
origin_43 =55.77#1.5배
origin_23=81.38 #2/3배
origin_no = 48.08


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

noexit0= 119.24
noexit=  90.87768899815612
noexit2 = 85.42
noexit3= 80.39
noexit4= 48.0
noexit_43 = 81.55
noexit_23 = 106.33
noexit_no=73.74


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

edge_only0= 212.56
edge_only= 104.22269936015621
edge_only2 = 85.03
edge_only3= 66.43
edge_only4=  20.9
edgeonly_43 =70.85
edgeonly_23 = 159.42
edgeonly_no = 53.14

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

cloud_only0= 107.9
cloud_only= 96.06898047842027
cloud_only2= 95.76
cloud_only3= 93.91
cloud_only4= 90.9
cloud_only_43 =94.39
cloud_only_23= 102.73
cloud_only_no= 92.68


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

min0= 207.9
min =  147.9
min2 = 135.9
min3 =  125.4
min4= 113.6
min_43=127.9
min_23 = 177.9
min_no=113.6



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

equal0= 123.94
equal= 81.60048000480006
equal2 = 72.04
equal3= 63.63
equal4= 20.9
equal_43 =65.63
equal_23 = 105.67
equal_no= 57.62


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

h0= 94.84
h= 65.25482854494902
h2 = 60.20
h3= 59.92
h4=  20.9
h_43 = 60.00
h_23 = 81.38
h_no = 48.08



#GPU
'''
h0= 71.51074158551346
h= 65.25482854494902
h2 = 63.517494308042586
h3=  61.58416097470925
h4= 57.12161412811912
'''







#col=['1x','7/8x','3/4x','5/8x','1/2x']
col=['2/3x','1x','1.5x','2x','Cloud']
x=[]

original = [origin_23, origin, origin_43,origin_no,origin4]
x.append(original)
heu=[h_23, h,h_43,h_no,h4]
x.append(heu)
no_exit = [noexit_23, noexit,noexit_43,noexit_no,noexit4]
x.append(no_exit)
equalibrium=[equal_23,equal,equal_43,equal_no,equal4]
x.append(equalibrium)
edgeonly=[edgeonly_23, edge_only,edgeonly_43,edgeonly_no,edge_only4]
x.append(edgeonly)
cloudonly=[cloud_only_23,cloud_only,cloud_only_43,cloud_only_no,cloud_only4]
x.append(cloudonly)
minimum = [min_23, min,min_43,min_no,min4]
x.append(minimum)

NATC_min = []


for i in range(5):
    NATC_min.append([original[i],heu[i], no_exit[i], equalibrium[i], edgeonly[i], cloudonly[i],  minimum[i] ])

sum =[max(NATC_min[0]), max(NATC_min[1]), max(NATC_min[2]), max(NATC_min[3]),max(NATC_min[4])]


for i in range(7):
    for j in range(5):
        x[i][j]=x[i][j]/sum[j]
print(x)

l1 = plt.plot(col,x[0],c='black',linestyle='-',marker='o',label='ERDS-OSA')
l2 = plt.plot(col,x[1],c='orange',linestyle='-',marker='^',label='ERDS-HSA')
l3 = plt.plot(col,x[2],c='red',linestyle='--',marker='x',label='IAO')
l4 = plt.plot(col,x[3],c='cyan',linestyle='--',marker='o',label='Edgent- E')
l5 = plt.plot(col,x[4],c='green',linestyle='--',marker='^',label='Edge-Only')
l6 = plt.plot(col,x[5],c='gray',linestyle='--',marker='s',label='Cloud-Only')
l7 = plt.plot(col,x[6],c='blue',linestyle='-',marker='s',label='Edgent-M')
plt.ylabel("Normalized Average Time Consumption",fontsize=12)
plt.xlabel("Computation Speed at the Edge Node",fontsize=12)
lines = l1+l2+l3
lines2 = l4+l5+l6+l7
labels1 = [l.get_label() for l in lines]
labels2 = [l.get_label() for l in lines2]
first_lgend = plt.legend(handles = lines,loc='lower left')
plt.gca().add_artist(first_lgend)
plt.legend(handles = lines2,loc='lower center')
plt.savefig('Sync_Graph2.png')
plt.savefig('Sync_Graph2.eps', format='eps',transparent=True)
plt.show()