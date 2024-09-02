from __future__ import print_function

import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from torch.profiler import profile, record_function, ProfilerActivity


#CIFAR-10
def _layer(in_channels, out_channels, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # convolution block
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )




class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResLayer, self).__init__()
        self.c1 = _layer(in_channels, out_channels)
        self.c2 = _layer(out_channels, out_channels, activation=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)

        # residual connection
        if x.shape[1] == h.shape[1]:
            h += x

        h = self.activation(h)

        return h




class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        inner_channels = 4 * growth_rate

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,inner_channels,1,stride=1,padding=0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels,growth_rate, 3, stride=1, padding=1, bias= False)
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.cat((self.shortcut(x),self.residual(x)),1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels,reduction):
        super(TransitionLayer, self).__init__()

        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2,stride=2)
        )

    def forward(self, x):
        return self.downsample(x)



class DenseBlock(nn.Sequential):
    def __init__(self, nin, num_layers, growth_rate):
        super(DenseBlock, self).__init__()

        for i in range(num_layers):
            nin_bottleneck_layer = nin + growth_rate * i
            self.add_module('bottleneck_layer_%d' % i,
                            DenseLayer(nin_bottleneck_layer, growth_rate=growth_rate))



class DDNN_DenseNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDNN_DenseNet, self).__init__()
        self.exit_threshold = 0.3
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.growth_rate = 12
        self.reduction = 0.5
        self.num_layers = 2


        inner_channels = 2 * self.growth_rate
        edge_input_channels = self.in_channels

        #CIFAR10
        self.edge_start_channels = 32

        self.start_model = nn.Sequential(
            nn.Conv2d(edge_input_channels, inner_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        transition_1_layer = (inner_channels) + (self.growth_rate * self.num_layers)
        self.edge_model1 = nn.Sequential(
            DenseBlock(inner_channels,num_layers=self.num_layers,growth_rate = self.growth_rate),
        )

        transition_2_layer = int(transition_1_layer * self.reduction) + (self.growth_rate * self.num_layers)
        self.edge_model2 = nn.Sequential(
            TransitionLayer(transition_1_layer, self.reduction),
            DenseBlock(int(transition_1_layer*self.reduction), num_layers=self.num_layers, growth_rate=self.growth_rate)
        )

        transition_3_layer = int(transition_2_layer * self.reduction) + (self.growth_rate * self.num_layers)
        self.edge_model3 = nn.Sequential(
            TransitionLayer(transition_2_layer, self.reduction),
            DenseBlock(int(transition_2_layer * self.reduction), num_layers=self.num_layers,growth_rate=self.growth_rate)
        )

        transition_4_layer = int(transition_3_layer * self.reduction) + (self.growth_rate * self.num_layers)
        self.edge_model4 = nn.Sequential(
            TransitionLayer(transition_3_layer, self.reduction),
            DenseBlock(int(transition_3_layer * self.reduction), num_layers=self.num_layers,growth_rate=self.growth_rate)
        )

        self.cloud_model = nn.Sequential(
            TransitionLayer(transition_4_layer, self.reduction),
            DenseBlock(int(transition_4_layer * self.reduction), num_layers=self.num_layers,growth_rate=self.growth_rate)
        )

        final_layer = int(transition_4_layer*self.reduction) + (self.growth_rate * self.num_layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.edge_classifier1 = nn.Linear(transition_1_layer, self.out_channels)
        self.edge_classifier2 = nn.Linear(transition_2_layer, self.out_channels)
        self.edge_classifier3 = nn.Linear(transition_3_layer, self.out_channels)
        self.edge_classifier4 = nn.Linear(transition_4_layer, self.out_channels)
        self.cloud_classifier = nn.Linear(final_layer, self.out_channels)


    def softmax(self,a):
        ex = a - np.max(a)
        exp_a = np.exp(ex)
        sum_exp_a = np.sum(exp_a,axis=-1).reshape(-1, 1)
        y = exp_a / sum_exp_a
        return y

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)




    def exit_model(self, exit_num, h, input_size,batch_size,type):
        self.pool = nn.AdaptiveAvgPool2d(1)
        prediction = self.pool(h)
        if type=="train":
            if exit_num==1:
                exit_prediction = self.edge_classifier1(prediction.view(batch_size,-1))
            elif exit_num==2:
                exit_prediction = self.edge_classifier2(prediction.view(batch_size,-1))
            elif exit_num==3:
                exit_prediction = self.edge_classifier3(prediction.view(batch_size,-1))
            elif exit_num==4:
                exit_prediction = self.edge_classifier4(prediction.view(batch_size,-1))

            x = copy.copy(exit_prediction)
            x = x.cpu().detach().numpy()

            # Threshold구하기
            x = self.softmax(x)
            y = (x * np.log(x + 1e-9)) / np.log(self.out_channels)
            entropy = [0] * batch_size
            for i in range(batch_size):
                for j in range(self.out_channels):
                    entropy[i] += y[i][j]
                entropy[i] = -1 * entropy[i]

            # 2. Threshold와 각 디바이스에서 얻은 추론 결과 비교
            Local_Batch = []

            for i in range(batch_size):
                if (entropy[i] < self.exit_threshold):
                    Local_Batch.append([i, exit_prediction[i]])


            return exit_prediction,Local_Batch



        else:
            if exit_num == 1:
                exit_prediction = self.edge_classifier1(prediction.view(batch_size, -1))
            elif exit_num == 2:
                exit_prediction = self.edge_classifier2(prediction.view(batch_size, -1))
            elif exit_num == 3:
                exit_prediction = self.edge_classifier3(prediction.view(batch_size, -1))
            elif exit_num == 4:
                exit_prediction = self.edge_classifier4(prediction.view(batch_size, -1))


        return exit_prediction



    def Local_Batch_Calculate(self,final_batch,local_batch,batch_size):
        for i in range(batch_size):
            for j in range(len(local_batch)):
                if i == local_batch[j][0]:
                    final_batch[i] = local_batch[j][1]

        return final_batch



    def test_threshold(self,exit_prediction):
        x = self.softmax(exit_prediction.cpu().detach().numpy())
        y = (x * np.log(x + 1e-9)) / np.log(self.out_channels)
        entropy = 0
        for i in range(self.out_channels):
            entropy += y[0][i]
        entropy = -1 * entropy
        return entropy



    def forward(self, x , type, device):
        if type=="train":
            batch_size = x.shape[0]  # 16
            Final_Batch = [0] * batch_size
            Local_Batch = []
            timings = np.zeros((6, 1))
            exit_numbers=0

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # Edge 과정 진행
            starter.record()
            # WAIT FOR GPU SYNC
            x = self.start_model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            starter.record()
            h = self.edge_model1(x)
            exit_numbers+=1
            #FIRST EXIT-> local에서 exit 할 수있는 개수 뽑음
            exit1_prediction,Local_Batch1 = self.exit_model(exit_numbers,h,h.shape[1],batch_size,"train")

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            Final_Batch = self.Local_Batch_Calculate(Final_Batch,Local_Batch1,batch_size)

            #print("EXIT 1 = {}".format(len(Local_Batch1)))
            exit1 = len(Local_Batch1)
            if len(Local_Batch1)==batch_size:
                return exit1_prediction, exit1_prediction,exit1_prediction,exit1_prediction,exit1_prediction, timings


            starter.record()
            h = self.edge_model2(h)
            # SECOND EXIT-> local에서 exit 할 수있는 개수 뽑음
            exit_numbers+=1
            exit2_prediction, Local_Batch2 = self.exit_model(exit_numbers,h, h.shape[1], batch_size, "train")

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            Final_Batch = self.Local_Batch_Calculate(Final_Batch, Local_Batch2, batch_size)

            exit2 = len(Local_Batch2)
            #print("EXIT 2 = {}".format(exit2))
            if len(Local_Batch2)==batch_size:
                return exit1_prediction, exit2_prediction, exit2_prediction, exit2_prediction, exit2_prediction, timings


            starter.record()
            h = self.edge_model3(h)
            exit_numbers+=1
            # THIRD EXIT-> local에서 exit 할 수있는 개수 뽑음
            exit3_prediction, Local_Batch3 = self.exit_model(exit_numbers,h, h.shape[1], batch_size, "train")

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            Final_Batch = self.Local_Batch_Calculate(Final_Batch, Local_Batch3, batch_size)

            exit3 = len(Local_Batch3)
            #print("EXIT 3 = {}".format(exit3))
            if len(Local_Batch3) == batch_size:
                return exit1_prediction, exit2_prediction,exit3_prediction, exit3_prediction, exit3_prediction, timings



            starter.record()
            h = self.edge_model4(h)
            exit_numbers+=1
            # FOUTRTH EXIT-> local에서 exit 할 수있는 개수 뽑음
            exit4_prediction, Local_Batch4 = self.exit_model(exit_numbers,h, h.shape[1], batch_size, "train")

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            Final_Batch = self.Local_Batch_Calculate(Final_Batch, Local_Batch4, batch_size)
            exit4 = len(Local_Batch4)
            #print("EXIT 4 = {}".format(exit4))
            if len(Local_Batch4) == batch_size:
                return exit1_prediction, exit2_prediction,exit3_prediction, exit4_prediction, exit4_prediction, timings



            starter.record()
            h = self.cloud_model(h)  # Cloud convp
            exit_numbers+=1
            h = self.pool(h)
            cloud_prediction = self.cloud_classifier(h.view(batch_size, -1))  # Cloud 추론

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            exit5 = (batch_size - exit4)
            for j in range(batch_size):
                Final_Batch[j] = cloud_prediction[j]
            #print("EXIT FINAL CLOUD = {}".format(exit5))


            return exit1_prediction, exit2_prediction,exit3_prediction, exit4_prediction, cloud_prediction, timings

        else:
            test_threshold = 0.3
            timings = np.zeros((6,1))
            exit_numbers = 0
            check=0

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # Edge 과정 진행
            exit_place = 5
            batch_size = x.shape[0]
            entropy_sum=[]


            starter.record()
            x = self.start_model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            starter.record()
            h= self.edge_model1(x)
            exit_numbers += 1
            # EXIT 구조 추가 - Exit 1
            exit1_prediction = self.exit_model(1,h,h.shape[1],batch_size,"test")
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            entropy = self.test_threshold(exit1_prediction)
            entropy_sum.append(entropy)
            if entropy < test_threshold and check==0:
                exit_place = 1
                check+=1


            starter.record()
            h = self.edge_model2(h)
            exit_numbers += 1
            # EXIT 구조 추가 - Exit 1
            exit2_prediction = self.exit_model(2,h, h.shape[1],batch_size, "test")
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time


            entropy = self.test_threshold(exit2_prediction)
            entropy_sum.append(entropy)
            if entropy < test_threshold and check==0:
                exit_place = 2
                check+=1



            starter.record()
            h = self.edge_model3(h)
            exit_numbers += 1
            # EXIT 구조 추가 - Exit 1
            exit3_prediction = self.exit_model(3,h, h.shape[1],batch_size, "test")
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            entropy = self.test_threshold(exit3_prediction)
            entropy_sum.append(entropy)
            if entropy < test_threshold and check==0:
                exit_place = 3
                check+=1



            starter.record()
            h = self.edge_model4(h)
            # EXIT 구조 추가 - Exit 1
            exit_numbers += 1
            exit4_prediction = self.exit_model(4,h, h.shape[1],batch_size, "test")
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            entropy = self.test_threshold(exit4_prediction)
            entropy_sum.append(entropy)
            if entropy < test_threshold and check==0:
                exit_place = 4
                check += 1



            starter.record()
            h = self.cloud_model(h)  # Cloud convp
            h = self.pool(h)
            cloud_prediction = self.cloud_classifier(h.view(batch_size,-1))  # Cloud 추론
            entropy = self.test_threshold(cloud_prediction)
            exit_numbers += 1
            entropy_sum.append(entropy)
            exit_predictions = cloud_prediction
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time
            return exit_predictions,exit1_prediction,exit2_prediction,exit3_prediction,exit4_prediction,cloud_prediction,torch.tensor(exit_place).to(device), timings,entropy_sum




class DDNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDNN, self).__init__()
        self.exit_threshold = 0.3
        self.out_channels = out_channels
        self.in_channels = in_channels


        edge_input_channels = self.in_channels

        #CIFAR10
        self.edge_start_channels = 64

        self.start_model = nn.Sequential(
            nn.Conv2d(edge_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.edge_model1 = nn.Sequential(
            ResLayer(32, self.edge_start_channels),  # in,32
            ResLayer(self.edge_start_channels, self.edge_start_channels),  # 32,32
            # ResLayer(self.edge_start_channels, self.edge_start_channels),  # 32,32
        )

        self.edge_model2 = nn.Sequential(
            ResLayer(self.edge_start_channels, self.edge_start_channels * 2),  # 32, 64
            ResLayer(self.edge_start_channels * 2, self.edge_start_channels * 2),  # 64, 64
            # ResLayer(self.edge_start_channels * 2, self.edge_start_channels * 2),  # 64, 64
        )

        self.edge_model3 = nn.Sequential(
            ResLayer(self.edge_start_channels * 2, self.edge_start_channels * 4),  # 64,128
            ResLayer(self.edge_start_channels * 4, self.edge_start_channels * 4),  # 128, 128
            # ResLayer(self.edge_start_channels * 4, self.edge_start_channels * 4),  # 128, 128
        )

        self.edge_model4 = nn.Sequential(
            ResLayer(self.edge_start_channels * 4, self.edge_start_channels * 8),  # 128,256
            ResLayer(self.edge_start_channels * 8, self.edge_start_channels * 8),  # 256, 256
            # ResLayer(self.edge_start_channels * 8, self.edge_start_channels * 8),  # 256, 256
        )

        self.cloud_model = nn.Sequential(
            nn.AvgPool2d(2, 2),
            ResLayer(self.edge_start_channels * 8, self.edge_start_channels * 16),  # 256, 512
            ResLayer(self.edge_start_channels * 16, self.edge_start_channels * 16),  # 512, 512
            # ResLayer(self.edge_start_channels * 16, self.edge_start_channels * 16),  # 512, 512
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.edge_classifier1 = nn.Linear(self.edge_start_channels, self.out_channels)
        self.edge_classifier2 = nn.Linear(self.edge_start_channels * 2, self.out_channels)
        self.edge_classifier3 = nn.Linear(self.edge_start_channels * 4, self.out_channels)
        self.edge_classifier4 = nn.Linear(self.edge_start_channels * 8, self.out_channels)
        self.cloud_classifier = nn.Linear(self.edge_start_channels * 16, self.out_channels)


    def softmax(self,a):
        ex = a - np.max(a)
        exp_a = np.exp(ex)
        sum_exp_a = np.sum(exp_a,axis=-1).reshape(-1, 1)
        y = exp_a / sum_exp_a
        return y

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)




    def exit_model(self, exit_num, h, input_size,batch_size,type):
        self.pool = nn.AdaptiveAvgPool2d(1)
        prediction = self.pool(h)
        if type=="train":
            if exit_num==1:
                exit_prediction = self.edge_classifier1(prediction.view(batch_size,-1))
            elif exit_num==2:
                exit_prediction = self.edge_classifier2(prediction.view(batch_size,-1))
            elif exit_num==3:
                exit_prediction = self.edge_classifier3(prediction.view(batch_size,-1))
            elif exit_num==4:
                exit_prediction = self.edge_classifier4(prediction.view(batch_size,-1))

            x = copy.copy(exit_prediction)
            x = x.cpu().detach().numpy()

            # Threshold구하기
            x = self.softmax(x)
            y = (x * np.log(x + 1e-9)) / np.log(self.out_channels)
            entropy = [0] * batch_size
            for i in range(batch_size):
                for j in range(self.out_channels):
                    entropy[i] += y[i][j]
                entropy[i] = -1 * entropy[i]

            # 2. Threshold와 각 디바이스에서 얻은 추론 결과 비교
            Local_Batch = []

            for i in range(batch_size):
                if (entropy[i] < self.exit_threshold):
                    Local_Batch.append([i, exit_prediction[i]])


            return exit_prediction,Local_Batch



        else:
            if exit_num == 1:
                exit_prediction = self.edge_classifier1(prediction.view(batch_size, -1))
            elif exit_num == 2:
                exit_prediction = self.edge_classifier2(prediction.view(batch_size, -1))
            elif exit_num == 3:
                exit_prediction = self.edge_classifier3(prediction.view(batch_size, -1))
            elif exit_num == 4:
                exit_prediction = self.edge_classifier4(prediction.view(batch_size, -1))


        return exit_prediction



    def Local_Batch_Calculate(self,final_batch,local_batch,batch_size):
        for i in range(batch_size):
            for j in range(len(local_batch)):
                if i == local_batch[j][0]:
                    final_batch[i] = local_batch[j][1]

        return final_batch



    def test_threshold(self,exit_prediction):
        x = self.softmax(exit_prediction.cpu().detach().numpy())
        y = (x * np.log(x + 1e-9)) / np.log(self.out_channels)
        entropy = 0
        for i in range(self.out_channels):
            entropy += y[0][i]
        entropy = -1 * entropy
        return entropy



    def forward(self, x , type, device):
        if type=="train":
            batch_size = x.shape[0]  # 16
            Final_Batch = [0] * batch_size
            Local_Batch = []
            timings = np.zeros((6, 1))
            exit_numbers=0

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # Edge 과정 진행
            starter.record()
            # WAIT FOR GPU SYNC
            x = self.start_model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            starter.record()
            h = self.edge_model1(x)
            exit_numbers+=1
            #FIRST EXIT-> local에서 exit 할 수있는 개수 뽑음
            exit1_prediction,Local_Batch1 = self.exit_model(exit_numbers,h,h.shape[1],batch_size,"train")

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            Final_Batch = self.Local_Batch_Calculate(Final_Batch,Local_Batch1,batch_size)

            print("EXIT 1 = {}".format(len(Local_Batch1)))
            exit1 = len(Local_Batch1)
            if len(Local_Batch1)==batch_size:
                return exit1_prediction, exit1_prediction,exit1_prediction,exit1_prediction,exit1_prediction, timings


            starter.record()
            h = self.edge_model2(h)
            # SECOND EXIT-> local에서 exit 할 수있는 개수 뽑음
            exit_numbers+=1
            exit2_prediction, Local_Batch2 = self.exit_model(exit_numbers,h, h.shape[1], batch_size, "train")

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            Final_Batch = self.Local_Batch_Calculate(Final_Batch, Local_Batch2, batch_size)

            exit2 = len(Local_Batch2)
            print("EXIT 2 = {}".format(exit2))
            if len(Local_Batch2)==batch_size:
                return exit1_prediction, exit2_prediction, exit2_prediction, exit2_prediction, exit2_prediction, timings


            starter.record()
            h = self.edge_model3(h)
            exit_numbers+=1
            # THIRD EXIT-> local에서 exit 할 수있는 개수 뽑음
            exit3_prediction, Local_Batch3 = self.exit_model(exit_numbers,h, h.shape[1], batch_size, "train")

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            Final_Batch = self.Local_Batch_Calculate(Final_Batch, Local_Batch3, batch_size)

            exit3 = len(Local_Batch3)
            if len(Local_Batch3) == batch_size:
                return exit1_prediction, exit2_prediction,exit3_prediction, exit3_prediction, exit3_prediction, timings



            starter.record()
            h = self.edge_model4(h)
            exit_numbers+=1
            # FOUTRTH EXIT-> local에서 exit 할 수있는 개수 뽑음
            exit4_prediction, Local_Batch4 = self.exit_model(exit_numbers,h, h.shape[1], batch_size, "train")

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            Final_Batch = self.Local_Batch_Calculate(Final_Batch, Local_Batch4, batch_size)
            exit4 = len(Local_Batch4)
            print("EXIT 4 = {}".format(exit4))
            if len(Local_Batch4) == batch_size:
                return exit1_prediction, exit2_prediction,exit3_prediction, exit4_prediction, exit4_prediction, timings



            starter.record()
            h = self.cloud_model(h)  # Cloud convp
            exit_numbers+=1
            h = self.pool(h)
            cloud_prediction = self.cloud_classifier(h.view(batch_size, -1))  # Cloud 추론

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time
            print("CURR_TIME:{}".format(timings))

            exit5 = (batch_size - exit4)
            for j in range(batch_size):
                Final_Batch[j] = cloud_prediction[j]
            print("EXIT FINAL CLOUD = {}".format(exit5))


            return exit1_prediction, exit2_prediction,exit3_prediction, exit4_prediction, cloud_prediction, timings

        else:
            test_threshold = 0.3
            timings = np.zeros((6,1))
            exit_numbers = 0
            check=0

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # Edge 과정 진행
            exit_place = 5
            batch_size = x.shape[0]
            entropy_sum=[]


            starter.record()
            x = self.start_model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            starter.record()
            h= self.edge_model1(x)
            exit_numbers += 1
            # EXIT 구조 추가 - Exit 1
            exit1_prediction = self.exit_model(1,h,h.shape[1],batch_size,"test")
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            entropy = self.test_threshold(exit1_prediction)
            entropy_sum.append(entropy)
            if entropy < test_threshold and check==0:
                exit_place = 1
                check+=1


            starter.record()
            h = self.edge_model2(h)
            exit_numbers += 1
            # EXIT 구조 추가 - Exit 1
            exit2_prediction = self.exit_model(2,h, h.shape[1],batch_size, "test")
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time


            entropy = self.test_threshold(exit2_prediction)
            entropy_sum.append(entropy)
            if entropy < test_threshold and check==0:
                exit_place = 2
                check+=1



            starter.record()
            h = self.edge_model3(h)
            exit_numbers += 1
            # EXIT 구조 추가 - Exit 1
            exit3_prediction = self.exit_model(3,h, h.shape[1],batch_size, "test")
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            entropy = self.test_threshold(exit3_prediction)
            entropy_sum.append(entropy)
            if entropy < test_threshold and check==0:
                exit_place = 3
                check+=1



            starter.record()
            h = self.edge_model4(h)
            # EXIT 구조 추가 - Exit 1
            exit_numbers += 1
            exit4_prediction = self.exit_model(4,h, h.shape[1],batch_size, "test")
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time

            entropy = self.test_threshold(exit4_prediction)
            entropy_sum.append(entropy)
            if entropy < test_threshold and check==0:
                exit_place = 4
                check += 1



            starter.record()
            h = self.cloud_model(h)  # Cloud convp
            h = self.pool(h)
            cloud_prediction = self.cloud_classifier(h.view(batch_size,-1))  # Cloud 추론
            entropy = self.test_threshold(cloud_prediction)
            exit_numbers += 1
            entropy_sum.append(entropy)
            exit_predictions = cloud_prediction
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[exit_numbers] = curr_time
            return exit_predictions,exit1_prediction,exit2_prediction,exit3_prediction,exit4_prediction,cloud_prediction,torch.tensor(exit_place).to(device), timings,entropy_sum
