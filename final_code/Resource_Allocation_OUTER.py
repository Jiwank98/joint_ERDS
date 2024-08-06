import torch
import pandas as pd
import numpy as np
import sys
from itertools import *

import easy_sync_algorithm_INNER as inner_algorithm
import pickle
Algorithm = inner_algorithm.sync_algorithm


def check_minimum(w_set,dimension,w_min):
    flag = [0]*dimension
    for i in range(dimension):
        if w_set[i] < w_min:
            flag[i]=1

    if np.sum(flag)>=1:
        return False
    else:
        return True


def run(args,digital_twins,nof,cof,tof):
    dimension = digital_twins
    # 초기화 과정
    iter_final_Q = []
    iter_final_upperbound = []

    w_min = 0.1  # 디지털 트윈에 줄 수 있는 자원 최소
    w_max = 1  # 디지털 트윈에 줄 수 있는 자원 최대

    w_minset = [w_min] * dimension
    w_maxset = [w_max] * dimension

    f_min = Algorithm(args,dimension, w_minset,nof,cof,tof)  # LBD : 최소 시간 구해주는 알고리즘에 모든 디지털 트윈 계산 컴퓨팅 자원을 최소한 주어 계산한 결과
    f_max = Algorithm(args,dimension, w_maxset,nof,cof,tof)  # UBD : 양의 무한대로 두기 ,
    print(f_min, f_max)

    brb_iter = 0  # 총 진행할 itertion
    zero_vector = [w_min] * dimension  # 디지털 트윈 개수 만큼 0 벡터 만들기
    e_vector = [w_max] * dimension  # 디지털 트윈 개수 만큼 1 벡터 만들기 (논문에서 e)

    Q = []  # M들을 담을 집합
    M_upperbound = []  # local upper bound 확인용
    # M_0 만들기 [0 vector, w_max * 1 vector]
    initial_M = []
    initial_cal_M = []
    for i in range(dimension):
        m = [zero_vector[i], e_vector[i]]
        m2 = w_max * e_vector[i]
        initial_M.append(m)
        initial_cal_M.append(m2)

    Q.append(initial_M)  # 집합에 M_0 넣기
    M_upperbound.append(Algorithm(args,dimension, initial_cal_M,nof,cof,tof))  # upperbound set에 local upper bound 넣기

    # BRB Algorithm
    while True:
        if brb_iter > 4000:
            break
        # if round(f_min, 3) == round(f_max, 3):
        #    cnt = 0
        #    for i in range(dimension):
        #        if (round(Q[max_M_idx][i][0], 3) == round(Q[max_M_idx][i][1], 3)):
        #            cnt += 1
        #    if cnt == dimension:
        #        break
        brb_iter += 1

        # 집합에서 가장 좋은 set 고르기 (Select best set)
        max_M_idx_list = list(filter(lambda x: M_upperbound[x] == np.min(M_upperbound), range(len(M_upperbound))))
        max_M_idx = np.max(max_M_idx_list)
        max_M = Q[max_M_idx]
        selected_index = 0
        selected_max = 0
        for i in range(dimension):
            k = Q[max_M_idx][i][1] - Q[max_M_idx][i][0]
            if selected_max < k:
                selected_max = k
                selected_index = i

        print("max_M_idx={},selected_index={}, {}".format(max_M_idx, selected_index, max_M))
        side = Q[max_M_idx][selected_index][1] - Q[max_M_idx][selected_index][0]
        side = side / 2

        # 가장 좋은 set를 반으로 하여 두 가지 set로 나누기
        M_setA = []
        M_setB = []
        cal_setA = []
        cal_setB = []

        for i in range(dimension):
            if i == selected_index:
                m1 = [round(Q[max_M_idx][i][0], 3), round(Q[max_M_idx][i][1] - side, 3)]
                m2 = [round(Q[max_M_idx][i][0] + side, 3), round(Q[max_M_idx][i][1], 3)]
                cal_setA.append(round(Q[max_M_idx][i][1] - side, 3))
            else:
                m1 = [round(Q[max_M_idx][i][0], 3), round(Q[max_M_idx][i][1], 3)]
                m2 = [round(Q[max_M_idx][i][0], 3), round(Q[max_M_idx][i][1], 3)]
                cal_setA.append(round(Q[max_M_idx][i][1], 3))

            M_setA.append(m1)
            M_setB.append(m2)
            cal_setB.append(round(Q[max_M_idx][i][1], 3))

        setA_local_ubd = Algorithm(args,dimension, cal_setA,nof,cof,tof)
        setB_local_ubd = Algorithm(args,dimension, cal_setB,nof,cof,tof)
        feasible_flag = [0] * 2

        # SetA 조건에 따라 Branching
        if setA_local_ubd > f_min:  # minimize 문제니까 기존과 다르게 f_min보다 클때로
            print(1, M_setA, setA_local_ubd)
            setA_local_ubd = 0
            M_setA.clear()
        else:
            feasible_flag[0] = 1

        # SetB 조건에 따라 Branching / Reduction
        if setB_local_ubd > f_min:  # minimize 문제니까 기존과 다르게 f_min보다 클때로
            print(2, M_setB, setB_local_ubd)
            setB_local_ubd = 0
            M_setB.clear()

        else:
            # 이제 최종 결정된 setB를 바탕으로 x prime feasibility check, 가능하다면 UBD/LBD 업데이트하기
            feasible_flag[1] = 1
            sum_resource = 0
            for i in range(dimension):
                sum_resource += M_setB[i][0]

            if sum_resource > w_max:  # infeasibility
                print(3, M_setB, setB_local_ubd)
                M_setB.clear()
                setB_local_ubd = 0
                feasible_flag[1] = 0

            else:  # feasibility -> bound
                feasible_solution = []
                tmp_f_bar_max = []
                z = [0] * dimension
                x_norm = 0
                difference_norm = 0
                for i in range(dimension):
                    x_norm += np.abs(M_setB[i][0])
                    difference_norm += np.abs(M_setB[i][1] - M_setB[i][0])

                # local lower bound calculation
                for i in range(dimension):
                    if i == selected_index:
                        difference = (M_setB[i][1] - M_setB[i][0])
                        y_x = difference / difference_norm
                        w_max_x_norm = w_max - x_norm
                        mul = y_x * w_max_x_norm
                        z[i] = (M_setB[i][0] + mul)
                    else:
                        z[i] = M_setB[i][0]

                f_bar_min = Algorithm(args,dimension, z,nof,cof,tof)

                # local upper bound calculation
                for i in range(dimension):
                    tmp_y = (M_setB[i][1] - z[i])
                    local_setB = []
                    for j in range(dimension):
                        if j == selected_index:
                            e = 1
                        else:
                            e = 0
                        local_y = M_setB[j][1] - (tmp_y * e)
                        local_setB.append(local_y)
                    tmp_f_bar_max.append(Algorithm(args,dimension, local_setB,nof,cof,tof))

                # minimize문제
                f_bar_max = (np.min(tmp_f_bar_max))

                # global local lower bound, global upper bound update
                print("min={},bar_min={}".format(f_min, f_bar_min))
                print("max={},bar_max={},local_ubd={}".format(f_max, f_bar_max, setB_local_ubd))
                f_min = min(f_min, f_bar_min)
                setB_local_ubd = max(setB_local_ubd, f_bar_max)

        # merge set list
        if feasible_flag[0] > 0:
            cal_setA = []
            for i in range(dimension):
                cal_setA.append(M_setA[i][1])

            if check_minimum(cal_setA, dimension, w_min) == True:
                Q.append(M_setA)
                M_upperbound.append(Algorithm(args,dimension, cal_setA,nof,cof,tof))

        if feasible_flag[1] > 0:
            cal_setB = []
            for i in range(dimension):
                cal_setB.append(M_setB[i][1])
            if check_minimum(cal_setB, dimension, w_min) == True:
                Q.append(M_setB)
                M_upperbound.append(setB_local_ubd)

        # 하나라도 branch가 존재하면 부모 노드 삭제
        if feasible_flag[0] > 0 or feasible_flag[1] > 0:
            del Q[max_M_idx]
            del M_upperbound[max_M_idx]
        else:
            if round(f_min, 3) != round(f_max, 3):
                del Q[max_M_idx]
                del M_upperbound[max_M_idx]

        if brb_iter % 10 == 0:
            iter_final_Q.append(Q[-1])
            iter_final_upperbound.append(M_upperbound[-1])

        print(M_upperbound)

        f_max = (np.min(M_upperbound))
        print("EPOCH={}, Now upper bound = {}, last Q = {}, last M_upper_bound = {} ".format(brb_iter, f_max, Q[-1],
                                                                                             M_upperbound[-1]))

    i = 0
    while True:
        Q_length = len(Q)
        if i == Q_length:
            break
        if len(Q[i]) == 0:
            del Q[i]
            del M_upperbound[i]
        elif len(Q[i]) != 0:
            i += 1

    for i in range(len(iter_final_Q)):
        print("iteration {} = Q: {}, UBD: {}".format(i, iter_final_Q[i], iter_final_upperbound[i]))

    print("LBD={},UBD={}".format(f_min, f_max))
    print("final_Q={} , final_upperbound = {}".format(Q[-1], M_upperbound[-1]))

    for i in range(len(Q)):
        print("Q={},final_upperbound ={}".format(Q[i], M_upperbound[i]))




#with open("final_Q_IAO.pkl","wb") as a:
#    pickle.dump(iter_final_Q,a)
#with open("final_upperbound_IAO.pkl","wb") as b:
#    pickle.dump(iter_final_upperbound,b)