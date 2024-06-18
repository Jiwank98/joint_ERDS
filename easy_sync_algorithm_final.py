from __future__ import print_function

import argparse
import copy
import pickle
import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import random
import cv2

import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import datasets
import sync_exit_edge_cloud as net
import torch.cuda.amp as amp
from itertools import product

import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import sys

import matplotlib.pyplot as plt


def Edge_time_function(partition_point,exit_point,batch_size,w,digital_twin):
    time = [0] * 6

    #n = edge computation speed varialbe, w = allocated copmuting resources
    n=2

    if digital_twin == 0:
        start_time = 1 * n * (1 / w)
        exit1_time = 2 * n * (1 / w)
        exit2_time = 4 * n * (1 / w)
        exit3_time = 6 * n * (1 / w)
        exit4_time = 8 * n * (1 / w)
        exit5_time = 10 * n * (1 / w)

    elif digital_twin == 1:
        start_time = 2 * n * (1 / w)
        exit1_time = 4 * n * (1 / w)
        exit2_time = 6 * n * (1 / w)
        exit3_time = 8 * n * (1 / w)
        exit4_time = 10 * n * (1 / w)
        exit5_time = 12 * n * (1 / w)

    else:
        start_time = 3 * n * (1 / w)
        exit1_time = 5 * n * (1 / w)
        exit2_time = 7 * n * (1 / w)
        exit3_time = 9 * n * (1 / w)
        exit4_time = 11 * n * (1 / w)
        exit5_time = 13 * n * (1 / w)


    time[0] = start_time * batch_size
    time[1] = exit1_time * batch_size
    time[2] = exit2_time * batch_size
    time[3] = exit3_time * batch_size
    time[4] = exit4_time * batch_size
    time[5] = exit5_time * batch_size

    edge_time = 0
    if partition_point < exit_point:
        for i in range(partition_point + 1):
            edge_time += time[i]
    else:
        for i in range(exit_point+1):
            edge_time += time[i]

    return edge_time



def Cloud_time_function(partition_point,exit_point,batch_size,digital_twin): #GPU

    # exit마다 layer크기에 따른 시간 측정
    time = [0] * 6

    n=1
    # n = cloud computation speed varialbe

    if digital_twin == 0:
        start_time = 1 * n  # ex_time[0] # 0.0015*2
        exit1_time = 2 * n  # ex_time[1]  # 0.02*2
        exit2_time = 4 * n  # ex_time[2] # 0.03*2
        exit3_time = 6 * n  # ex_time[3] # 0.06*2
        exit4_time = 8 * n  # ex_time[4]  # 0.16*2
        exit5_time = 10 * n  # ex_time[5]   # 0.11*2


    elif digital_twin == 1:
        start_time = 2 * n  # ex_time[0] # 0.0015*2
        exit1_time = 4 * n  # ex_time[1]  # 0.02*2
        exit2_time = 6 * n  # ex_time[2] # 0.03*2
        exit3_time = 8 * n  # ex_time[3] # 0.06*2
        exit4_time = 10 * n  # ex_time[4]  # 0.16*2
        exit5_time = 12 * n  # ex_time[5]   # 0.11*2

    else:
        start_time = 3 * n  # ex_time[0] # 0.0015*2
        exit1_time = 5 * n  # ex_time[1]  # 0.02*2
        exit2_time = 7 * n  # ex_time[2] # 0.03*2
        exit3_time = 9 * n  # ex_time[3] # 0.06*2
        exit4_time = 11 * n  # ex_time[4]  # 0.16*2
        exit5_time = 13 * n  # ex_time[5]   # 0.11*2


    time[0] = start_time * batch_size
    time[1]= exit1_time * batch_size
    time[2] = exit2_time * batch_size
    time[3]= exit3_time * batch_size
    time[4]= exit4_time * batch_size
    time[5] = exit5_time * batch_size

    cloud_time = 0
    if partition_point<exit_point:
        for i in range(partition_point + 1, exit_point+1):
            cloud_time+=time[i]

    return cloud_time


def transmission_time(partition_point,exit_point,batch_size,digital_twin):
    time = [0] * 6
    n = 1
    # n = transmission time varialbe

    if digital_twin == 0:
        start_time = 40 * n
        exit1_time = 13 * n
        exit2_time = 8 * n
        exit3_time = 6 * n
        exit4_time = 4 * n
        exit5_time = 2 * n

    elif digital_twin == 1:
        start_time = 55 * n
        exit1_time = 18 * n
        exit2_time = 9 * n
        exit3_time = 7 * n
        exit4_time = 5 * n
        exit5_time = 3 * n

    else:
        start_time = 70 * n
        exit1_time = 23 * n
        exit2_time = 16 * n
        exit3_time = 8 * n
        exit4_time = 6 * n
        exit5_time = 4 * n


    time[0] = start_time * batch_size
    time[1] = exit1_time * batch_size
    time[2] = exit2_time * batch_size
    time[3] = exit3_time * batch_size
    time[4] = exit4_time * batch_size
    time[5] = exit5_time * batch_size

    tr_time = 0
    if partition_point < exit_point:
        tr_time=time[partition_point]

    return tr_time



def softmax(x):
    e_x = np.exp(x - np.max(x))
    e_x = e_x / e_x.sum(axis=0)
    return e_x

def exit_softmax(a):
    ex = a - np.max(a)
    exp_a = np.exp(ex)
    sum_exp_a = np.sum(exp_a,axis=-1).reshape(-1, 1)
    y = exp_a / sum_exp_a
    return y


# ERDS-OSA Algorithm
def EOA(batch_size,model_probability,digital_twins,w_set):
    mm=sys.maxsize

    partitions = [0, 1, 2, 3, 4, 5]
    items = []
    for i in range(digital_twins):
        items.append(partitions)
    every_partitions = list(product(*items))

    optimal_partition_points=[14]*digital_twins

    for i in range(len(every_partitions)):
        p=every_partitions[i] #(0,0,0)
        exits = [1, 2, 3, 4, 5]
        e_items = []
        for i in range(digital_twins):
            e_items.append(exits)
        every_exits = list(product(*e_items))
        y = 0
        for j in range(len(every_exits)):
            e=every_exits[j]
            tmp=[]
            x_tmp=[]
            for twin in range(digital_twins):
                x=model_probability
                edge = Edge_time_function(p[twin], e[twin], batch_size,w_set[twin],twin)
                cloud = Cloud_time_function(p[twin], e[twin], batch_size,twin)
                transmission = transmission_time(p[twin], e[twin], batch_size,twin)
                t_u = edge + cloud + transmission
                tmp.append(t_u)
                x_tmp.append(x[e[twin]-1])

            x_for_y=1
            for k in range(digital_twins):
                x_for_y*=(x_tmp[k])
            y += (max(tmp)*x_for_y)

        if mm>=y:
            mm=y
            for o in range(digital_twins):
                optimal_partition_points[o]=p[o]

    return optimal_partition_points



#ERDS-HSA Algorithm
def HA(batch_size,model_probability,digital_twins,w_set):
    min=[sys.maxsize]*digital_twins
    partition=[14]*digital_twins
    for twin in range(digital_twins):
        x= model_probability
        for p in range(6):
            y=0
            for e in range(5):
                edge = Edge_time_function(p, e+1, batch_size, w_set[twin],twin)
                cloud = Cloud_time_function(p, e+1, batch_size, twin)
                transmission = transmission_time(p, e+1, batch_size,twin)
                t_u=(edge+cloud+transmission)
                y+=(x[e]*t_u)

            if y<min[twin]:
                min[twin]=y
                partition[twin]=p

    return partition



def Random_choice(digital_twins):

    digital_twin_partitions = []
    for i in range(digital_twins):
        partition_point = random.randint(0, 5)
        digital_twin_partitions.append(partition_point)

    return digital_twin_partitions


def Cloud_Algorithm(digital_twins):

    digital_twin_partitions = []
    for i in range(digital_twins):
        partition_point = 0
        digital_twin_partitions.append(partition_point)

    return digital_twin_partitions



def Edge_Algorithm(digital_twins):

    digital_twin_partitions = []
    for i in range(digital_twins):
        partition_point = 5
        digital_twin_partitions.append(partition_point)
    return digital_twin_partitions



def select_Algorithm(point,prob,digital_twins,w_set):
    if point == 0:
        partition_point = EOA(1,prob,digital_twins,w_set)
    elif point == 1:
        partition_point = Random_choice(digital_twins)
    elif point == 2:
        partition_point = HA(1,prob,digital_twins,w_set)
    elif point == 3:
        partition_point = Cloud_Algorithm(digital_twins)
    else:
        partition_point = Edge_Algorithm(digital_twins)

    return partition_point




def test(model, point,model_probability,model_time,digital_twins,w_set,exit_place):
    model.eval()
    correct_sum = []
    final_time = []
    expectation_point = []


    with torch.no_grad():
        index=0
        E_f_time = []
        C_f_time =[]
        tr_f_time=[]
        sim_E = []
        sim_tr = []
        sim_C = []
        for x in range(10): #0~1000 각 에포크(디지털 트윈5개씩)
            Digital_twins_Edge_Tr_time = [] #디지털 트윈 10개에 대해 EDGE+TR time max값을 뽑기 위해

            prob=model_probability
            #prob=[0,0,0,0,1] ->IAO

            ext_time = []

            for i in range(index,index+digital_twins):
                ext_time.append(model_time[i])

            exit_num = exit_place[index]
            #exit_num =5 ->IAO


            # 각 디지털 트윈 별로 독립적으로 알고리즘에 따라 partition point 결정!
            partition_point = select_Algorithm(point,prob,digital_twins,w_set)  # 최소 기댓값을 뽑아서 그떄 예상 시간 기댓값, partition point 결정
            final_wset = w_set


            expectation_point.append(partition_point)  # 알고리즘에 따른 예상하는 특정 partition point 저장
            E_final_time = []
            tr_final_time = []
            C_final_time = []
            for i in range(digital_twins):
                if i == 0:
                    exit_time = [1, 2, 4, 6, 8, 10]
                elif i == 1:
                    exit_time = [2, 4, 6, 8, 10, 12]
                else:
                    exit_time = [3, 5, 7, 9, 11, 13]


                #각 디지털 트윈의 알고리즘에 따른 partition point로 모델을 돌렸을 때 실제로 뽑은 시간을 Edge+TR시간과 Cloud 시간을 따로 저장
                E_time = 0
                C_time = 0

                nof= 2
                #nof = edge computation speed

                tr_time=transmission_time(partition_point[i],exit_num,1,i)
                if partition_point[i] < exit_num:
                    for j in range(partition_point[i]+1):
                        if i==1:
                            E_time += (exit_time[j] * nof * (1 / final_wset[i]))
                        elif i==2:
                            E_time += (exit_time[j] * nof * (1 / final_wset[i]))
                        else:
                            E_time += (exit_time[j] * nof * (1 / final_wset[i]))

                    for j in range(partition_point[i]+1,exit_num+1):
                        if i==1:
                            C_time += (exit_time[j])
                        elif i==2:
                            C_time += (exit_time[j])
                        else:
                            C_time += (exit_time[j])

                else:
                    for j in range(exit_num+1):
                        if i == 1:
                            E_time += (exit_time[j] * nof * (1 / final_wset[i]))
                        elif i == 2:
                            E_time += (exit_time[j] * nof * (1 / final_wset[i]))
                        else:
                            E_time += (exit_time[j] * nof * (1 / final_wset[i]))

                Digital_twins_Edge_Tr_time.append(E_time+C_time+tr_time)
                E_final_time.append(E_time)
                tr_final_time.append(tr_time)
                C_final_time.append(C_time)
                index += 1

            #10개의 디지털 트윈의 병렬적 진행된 EDGE+TR시간은 max로 뽑기, 10개의 디지털 트윈의 하나의 Cloud에서 진행된 시간은 모두 더해서 뽑기
            sync_time = np.max(Digital_twins_Edge_Tr_time)
            final_index= np.argmax(Digital_twins_Edge_Tr_time)


            #10개의 디지털 트윈에 대한 최종 디지털 트윈 동기화 시간은 final time에 저장
            final_time.append(sync_time)
            E_f_time.append(E_final_time[final_index])
            C_f_time.append(C_final_time[final_index])
            tr_f_time.append(tr_final_time[final_index])
            sim_E.append(E_final_time)
            sim_tr.append(tr_final_time)
            sim_C.append(C_final_time)

    return correct_sum, exit_place, final_time, expectation_point,E_f_time,C_f_time,tr_f_time, sim_E, sim_tr, sim_C





def test_model(model, model_path, lr, epochs,point,model_probability,model_time,digital_twins,w_set,exit_place):
    ps = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(ps, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    correct_sum = []
    exit_sum = []
    time_sum = []
    exp_point_sum=[]


    for epoch in range(1, epochs):
        correct, exit,exit_time,expectation_point,E_f_time,C_f_time,tr_f_time, sim_E, sim_tr, sim_C = test(model,point,model_probability,model_time,digital_twins,w_set,exit_place)
        correct_sum.append(correct)
        exit_sum.append(exit)
        time_sum.append(exit_time)
        exp_point_sum.append(expectation_point)
        torch.save(model, model_path)
        scheduler.step()


    return time_sum,correct_sum,E_f_time,C_f_time,tr_f_time, sim_E, sim_tr, sim_C


def load_model_data(i):

    if i==0:#CIFAR10
        with open("sync_test_exit_place_CIFAR10_2.pkl", "rb") as x:
            #sync_test_exit_place_CIFAR10_2.pkl
            #sync_test_exit_place_CIFAR10_dense
            test_exit_place = pickle.load(x)

        with open("sync_test_exit_time_CIFAR10_2.pkl", "rb") as v:
            #sync_test_exit_time_CIFAR10_2.pkl
            #sync_test_exit_time_CIFAR10_dense
            test_exit_time= pickle.load(v)

        with open("sync_dataset.pkl", "rb") as a:
            dataset= pickle.load(a)

        with open("sync_target.pkl", "rb") as b:
            target= pickle.load(b)

        #exit = [0.8,0.05,0.03,0.02,0.1]
        exit = [0, 0, 0, 0, 0]
        for i in range(10000):
            exit[test_exit_place[i]-1]+=1
        for i in range(5):
            exit[i]=exit[i]/10000

        #exit_place = []
        #for i in range(10000):
        #    p = random.choices(range(1,6),weights=[exit[0],exit[1],exit[2],exit[3],exit[4]])
        #    exit_place.append(p[0])

        #with open('exit_place_final_1.pkl', 'wb') as f:
        #    pickle.dump(exit_place, f)



        #exit_place_dense
        #exit_place_final_2
        with open('exit_place_final_2.pkl', 'rb') as c:
            exit_place = pickle.load(c)


        #exit_place = [5]*10000

        return exit, test_exit_time,dataset,target,exit_place


def sync_algorithm(digital_twins, w_set):

    parser = argparse.ArgumentParser(description='DDNN Example')

    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test_epochs', type=int, default=2, metavar='N',
                        help='number of epochs to test (default: 50)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--dataset', default='cifar10', help='dataset name')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--train_output1', default='models/sync_train_model_Cifar10.pth',
                        help='output directory')
    parser.add_argument('--train_output2', default='models/sync_train_model_Cifar100.pth',
                        help='output directory')
    parser.add_argument('--train_output3', default='models/sync_train_model_ImageNet.pth',
                        help='output directory')

    parser.add_argument('--test_output', default='models/sync_test_model_Cifar10_3.pth',
                        help='output directory')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    NGPU = torch.cuda.device_count()
    torch.manual_seed(args.seed)


    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    #train_dataset, train_loader, test_dataset, test_loader = data

    model = torch.load('models/sync_test_model_Cifar10_2.pth')
    # sync_test_model_Cifar10_2
    # sync_test_model_Cifar10_2_dense
    num_devices=1
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if NGPU > 1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        model = model.to(device)


    time= []
    E_time=[]
    C_time=[]
    tr_time=[]

    m=1


    for i in range(m):
        model_probability,model_time,data_set,target_set,exit_place = load_model_data(i)

        algo_time,correct_sum0,E_f_time,C_f_time,tr_f_time, sim_E, sim_tr, sim_C = test_model(model, args.test_output, args.lr, args.test_epochs,0, model_probability,model_time,digital_twins,w_set,exit_place)

        time.append(algo_time)
        E_time.append(E_f_time)
        C_time.append(C_f_time)
        tr_time.append(tr_f_time)

    real_time = (np.mean(E_f_time) + np.mean(C_f_time) + np.mean(tr_f_time))
    dt1 = [0] * 3
    dt2 = [0] * 3
    dt3 = [0] * 3
    # edge tr cloud

    for i in range(10):
        dt1[0] += sim_E[i][0]
        dt2[0] += sim_E[i][1]
        dt3[0] += sim_E[i][2]

        dt1[1] += sim_tr[i][0]
        dt2[1] += sim_tr[i][1]
        dt3[1] += sim_tr[i][2]

        dt1[2] += sim_C[i][0]
        dt2[2] += sim_C[i][1]
        dt3[2] += sim_C[i][2]

    print("{}+{}+{} = {}".format(np.mean(E_f_time), np.mean(C_f_time), np.mean(tr_f_time), real_time))
    print(
        "sub_DT 1 = {}+{}+{}={}".format(dt1[0] / 10, dt1[1] / 10, dt1[2] / 10, dt1[0] / 10 + dt1[1] / 10 + dt1[2] / 10))
    print(
        "sub_DT 2 = {}+{}+{}={}".format(dt2[0] / 10, dt2[1] / 10, dt2[2] / 10, dt2[0] / 10 + dt2[1] / 10 + dt2[2] / 10))
    print(
        "sub_DT 3 = {}+{}+{}={}".format(dt3[0] / 10, dt3[1] / 10, dt3[2] / 10, dt3[0] / 10 + dt3[1] / 10 + dt3[2] / 10))


    return E_time,tr_time,C_time,real_time, sim_E, sim_tr, sim_C



w_set = [0.188, 0.325, 0.487]
e_time,tr_time,c_time, real_time, sim_E, sim_tr, sim_C = sync_algorithm(3,w_set)

