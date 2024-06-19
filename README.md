# Early Exiting-Aware Joint Resource Allocation and DNN  Splitting for Multi-Sensor Digital Twin in Edge-Cloud  Collaborative System (2024)

In this paper, we address an edge computing resource allocation and deep neural network (DNN) splitting problem in an edge-cloud collaborative system to minimize the
task execution time of a multi-sensor digital twin (DT), where the constituent tasks of the multi-sensor DT are employed by DNN models with both split computing and early exit structures. 
To this end, we develop an early exiting-aware joint edge computing resource allocation and DNN splitting (ERDS) framework that optimally solves the problem. 

In the framework, the problem is reformulated into a nested optimization problem consisting of an outer edge computing resource allocation problem and an inner DNN splitting problem which considers early exiting.
Based on the nested structure, the framework can efficiently solve the problem without having to consider the edge computing resource allocation and DNN splitting jointly.
As components of the framework, we develop an edge computing resource allocation algorithm that exploits the mathematical structure of the outer problem; 
we also develop an optimal DNN splitting algorithm and a heuristic algorithm that identifies suboptimal solutions but has lower computational complexity. 

Through the simulation, we demonstrate that our proposed framework effectively outperforms other state-of-the-art baselines in terms of the task execution time of the multi-sensor DT in different environments, 
which shows that our proposed framework is applicable in practical multi-sensor DTs.

![image](https://github.com/Jiwank98/joint_ERDS/assets/67055711/842a7eee-891d-41fa-9550-ef606fd28c08)




# Simulation Results

# Simulation 1 (Total Execution Time for Multi-sensor DT)
![image](https://github.com/Jiwank98/joint_ERDS/assets/67055711/31a4eeb4-cb60-483a-a194-51100c040bf4)
![image](https://github.com/Jiwank98/joint_ERDS/assets/67055711/4f1e3706-0fd5-488b-89b2-fb985e512586)
![image](https://github.com/Jiwank98/joint_ERDS/assets/67055711/1790615c-0c4a-47c3-a0ea-b75e4e0f897a)
#


# Simulation 2 (Impact of Transmission Time)
![image](https://github.com/Jiwank98/joint_ERDS/assets/67055711/f5038e40-894b-45c3-a337-17e73ef1d27f)
#

# Simulation 3 (Impact of Cloud computing capabilities)
![image](https://github.com/Jiwank98/joint_ERDS/assets/67055711/bcf11fdc-9f7b-42c2-b05a-bec391e80255)
#


# Simulation 4 (Impact of Edge computing capabilities)
![image](https://github.com/Jiwank98/joint_ERDS/assets/67055711/bbaf987e-8eb1-420b-a760-67b4b2b22474)
#
