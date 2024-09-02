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

import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import datasets
import model as net
import torch.cuda.amp as amp
from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler

scaler = amp.GradScaler()


def softmax(a):
    ex = a - np.max(a)
    exp_a = np.exp(ex)
    sum_exp_a = np.sum(exp_a, axis=-1).reshape(-1, 1)
    y = exp_a / sum_exp_a
    return y


def train(model, train_loader, optimizer, num_devices,epoch,device):
    model.train()
    batch_size = 16
    loss_sum = []

    local_correct_sum = [[] for _ in range(4)]
    cloud_correct_sum = []

    train_index=0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
        train_index+=1
        if device=='cuda':
            data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)

        with amp.autocast():
            exit1_prediction, exit2_prediction, exit3_prediction, exit4_prediction, cloud_prediction, exit_time = model(data, 'train', device)

        exit1_loss = F.cross_entropy(exit1_prediction, target).to(device)
        exit2_loss = F.cross_entropy(exit2_prediction, target).to(device)
        exit3_loss = F.cross_entropy(exit3_prediction, target).to(device)
        exit4_loss = F.cross_entropy(exit4_prediction, target).to(device)
        cloud_loss = F.cross_entropy(cloud_prediction, target).to(device)

        target = target.cpu().detach().numpy()

        local_corr = [[0] * batch_size for _ in range(4)]
        cloud_corr = [0] * batch_size


        if exit2_prediction.shape[0]<=1:
            loss =(exit1_loss)
            exit1_pred = softmax(exit1_prediction.cpu().detach().numpy())
            local_corr = [0] * batch_size
            for i in range(batch_size):
                local_corr[i] = exit1_pred[i].argmax()
            local_correct = [0]
            for i in range(batch_size):
                if local_corr[0][i] == target[i]:
                    local_correct[0] += 1
            if train_index%100==0:
                print("Exit1_loss={}, Exit1_Accuracy={}%".format(exit1_loss, local_correct[0] / batch_size))
            local_correct_sum[0].append(local_correct[0] / batch_size)

        elif exit3_prediction.shape[0]<=1:
            loss = (exit1_loss) + (exit2_loss)
            exit1_pred = softmax(exit1_prediction.cpu().detach().numpy())
            exit2_pred = softmax(exit2_prediction.cpu().detach().numpy())
            local_corr = [[0] * batch_size for _ in range(2)]
            for i in range(batch_size):
                local_corr[0][i] = exit1_pred[i].argmax()
                local_corr[1][i] = exit2_pred[i].argmax()
            local_correct = [0] * 2
            for i in range(batch_size):
                if local_corr[0][i] == target[i]:
                    local_correct[0] += 1
                if local_corr[1][i] == target[i]:
                    local_correct[1] += 1
            if train_index % 100 == 0:
                print("Exit1_loss={}, Exit1_Accuracy={}%".format(exit1_loss, local_correct[0] / batch_size))
                print("Exit2_loss={}, Exit2_Accuracy={}%".format(exit2_loss, local_correct[1] / batch_size))
            for i in range(2):
                local_correct_sum[i].append(local_correct[i] / batch_size)

        elif exit4_prediction.shape[0]<=1:
            loss = (exit1_loss) + (exit2_loss) + (exit3_loss)
            exit1_pred = softmax(exit1_prediction.cpu().detach().numpy())
            exit2_pred = softmax(exit2_prediction.cpu().detach().numpy())
            exit3_pred = softmax(exit3_prediction.cpu().detach().numpy())
            local_corr = [[0] * batch_size for _ in range(3)]
            for i in range(batch_size):
                local_corr[0][i] = exit1_pred[i].argmax()
                local_corr[1][i] = exit2_pred[i].argmax()
                local_corr[2][i] = exit3_pred[i].argmax()

            local_correct = [0] * 3
            for i in range(batch_size):
                if local_corr[0][i] == target[i]:
                    local_correct[0] += 1
                if local_corr[1][i] == target[i]:
                    local_correct[1] += 1
                if local_corr[2][i] == target[i]:
                    local_correct[2] += 1
            if train_index % 100 == 0:
                print("Exit1_loss={}, Exit1_Accuracy={}%".format(exit1_loss, local_correct[0] / batch_size))
                print("Exit2_loss={}, Exit2_Accuracy={}%".format(exit2_loss, local_correct[1] / batch_size))
                print("Exit3_loss={}, Exit3_Accuracy={}%".format(exit3_loss, local_correct[2] / batch_size))
            for i in range(3):
                local_correct_sum[i].append(local_correct[i] / batch_size)

        elif cloud_prediction.shape[0]<=1:
            loss = (exit1_loss) + (exit2_loss) + (exit3_loss) + (exit4_loss)
            exit1_pred = softmax(exit1_prediction.cpu().detach().numpy())
            exit2_pred = softmax(exit2_prediction.cpu().detach().numpy())
            exit3_pred = softmax(exit3_prediction.cpu().detach().numpy())
            exit4_pred = softmax(exit4_prediction.cpu().detach().numpy())
            local_corr = [[0] * batch_size for _ in range(4)]
            for i in range(batch_size):
                local_corr[0][i] = exit1_pred[i].argmax()
                local_corr[1][i] = exit2_pred[i].argmax()
                local_corr[2][i] = exit3_pred[i].argmax()
                local_corr[3][i] = exit4_pred[i].argmax()
            local_correct = [0] * 4
            for i in range(batch_size):
                if local_corr[0][i] == target[i]:
                    local_correct[0] += 1
                if local_corr[1][i] == target[i]:
                    local_correct[1] += 1
                if local_corr[2][i] == target[i]:
                    local_correct[2] += 1
                if local_corr[3][i] == target[i]:
                    local_correct[3] += 1
            if train_index % 100 == 0:
                print("Exit1_loss={}, Exit1_Accuracy={}%".format(exit1_loss,local_correct[0]/batch_size))
                print("Exit2_loss={}, Exit2_Accuracy={}%".format(exit2_loss,local_correct[1]/batch_size))
                print("Exit3_loss={}, Exit3_Accuracy={}%".format(exit3_loss,local_correct[2]/batch_size))
                print("Exit4_loss={}, Exit4_Accuracy={}%".format(exit4_loss, local_correct[3] / batch_size))
            for i in range(4):
                local_correct_sum[i].append(local_correct[i] / batch_size)



        else:
            loss= (exit1_loss) + (exit2_loss) + (exit3_loss) + (exit4_loss)+ cloud_loss
            exit1_pred = softmax(exit1_prediction.cpu().detach().numpy())
            exit2_pred = softmax(exit2_prediction.cpu().detach().numpy())
            exit3_pred = softmax(exit3_prediction.cpu().detach().numpy())
            exit4_pred = softmax(exit4_prediction.cpu().detach().numpy())
            cloud_pred = softmax(cloud_prediction.cpu().detach().numpy())

            local_corr = [[0] * batch_size for _ in range(4)]
            cloud_corr = [0] * batch_size
            for i in range(batch_size):
                local_corr[0][i] = exit1_pred[i].argmax()
                local_corr[1][i] = exit2_pred[i].argmax()
                local_corr[2][i] = exit3_pred[i].argmax()
                local_corr[3][i] = exit4_pred[i].argmax()
                cloud_corr[i] = cloud_pred[i].argmax()
            local_correct = [0] *4
            cloud_correct = 0
            for i in range(batch_size):
                if local_corr[0][i] == target[i]:
                    local_correct[0] += 1
                if local_corr[1][i] == target[i]:
                    local_correct[1] += 1
                if local_corr[2][i] == target[i]:
                    local_correct[2] += 1
                if local_corr[3][i] == target[i]:
                    local_correct[3] += 1
                if cloud_corr[i] == target[i]:
                    cloud_correct += 1
            if train_index % 500 == 0:
                print("잘학습중")
                print("TRAI EPOCH ===={}".format(epoch))
                print("Exit1_loss={}, Exit1_Accuracy={}%".format(exit1_loss,local_correct[0]/batch_size))
                print("Exit2_loss={}, Exit2_Accuracy={}%".format(exit2_loss,local_correct[1]/batch_size))
                print("Exit3_loss={}, Exit3_Accuracy={}%".format(exit3_loss,local_correct[2]/batch_size))
                print("Exit4_loss={}, Exit4_Accuracy={}%".format(exit4_loss,local_correct[3]/batch_size))
                print("Cloud_loss={}, Cloud_Accuracy={}%".format(cloud_loss,cloud_correct/batch_size))
            for i in range(4):
                local_correct_sum[i].append(local_correct[i] / batch_size)
            cloud_correct_sum.append(cloud_correct / batch_size)


        loss_sum.append(loss)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    N = len(train_loader.dataset)

    print('Train Loss:: {}'.format(loss))

    return loss_sum, local_correct_sum, cloud_correct_sum, exit_time


def test(model, test_loader, optimizer, num_devices,device):
    model.eval()
    exit_sum = []
    prob = []
    exit_sm=[]
    image_time= []

    with torch.no_grad():
        for data, target in tqdm(test_loader, leave=False):
            if device=='cuda':
                data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)

            exit_predictions,exit1_prediction,exit2_prediction,exit3_prediction,exit4_prediction,cloud_prediction,exit_num,timing,entropy_sum = model(data,"test",device)
            exit_num = exit_num.cpu().detach().numpy()
            pred1 = softmax(exit1_prediction.reshape(1, 10).cpu().detach().numpy())
            pred2 = softmax(exit2_prediction.reshape(1, 10).cpu().detach().numpy())
            pred3 = softmax(exit3_prediction.reshape(1, 10).cpu().detach().numpy())
            pred4 = softmax(exit4_prediction.reshape(1, 10).cpu().detach().numpy())
            pred5 = softmax(cloud_prediction.reshape(1, 10).cpu().detach().numpy())
            target = target.cpu().detach().numpy()
            corr1 = pred1.argmax()
            corr2 = pred2.argmax()
            corr3 = pred3.argmax()
            corr4 = pred4.argmax()
            corr5 = pred5.argmax()
            pr=[0]*5
            sm=[0]*5
            if corr1==target:
                pr[0]+=1
            if corr2==target:
                pr[1]+=1
            if corr3==target:
                pr[2]+=1
            if corr4==target:
                pr[3]+=1
            if corr5==target:
                pr[4]+=1

            prob.append(pr)
            exit_sm.append(entropy_sum)
            exit_sum.append(exit_num)
            image_time.append(timing)

    N = len(test_loader.dataset)

    return exit_sum,prob,exit_sm,image_time

def train_model(model, model_path, train_loader,  lr, epochs, num_devices,device):
    ps = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(ps, lr=lr, weight_decay=1e-4,momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    loss_sum = []
    local_sum = []
    cloud_sum = []
    time_sum = []

    for epoch in range(1, epochs):
        print('[Epoch {}]'.format(epoch))
        loss,  local_correct, cloud_correct,exit_time = train(model, train_loader, optimizer, num_devices,epoch,device)
        loss_sum.append(loss)
        local_sum.append(local_correct)
        cloud_sum.append(cloud_correct)
        time_sum.append(exit_time)
        torch.save(model, model_path)
        scheduler.step()

    with open("sync_train_loss_CIFAR10_dense.pkl","wb") as a:
        pickle.dump(loss_sum,a, protocol=pickle.HIGHEST_PROTOCOL)
    with open("sync_train_exit_correct_CIFAR10_dense.pkl","wb") as c:
        pickle.dump(local_sum,c)
    with open("sync_train_cloud_correct_CIFAR10_dense.pkl","wb") as d:
        pickle.dump(cloud_sum,d)
    with open("sync_train_time_CIFAR10_dense.pkl","wb") as z:
        pickle.dump(time_sum,z)



def test_model(model, model_path, test_loader, lr, epochs, num_devices,device):
    ps = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(ps, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)


    for epoch in range(1, epochs):
        print('[Epoch {}]'.format(epoch))
        exit_sum,prob,exit_sm,image_time = test(model, test_loader, optimizer, num_devices,device)
        torch.save(model, model_path)
        scheduler.step()


    with open("sync_test_exit_place_CIFAR10_dense.pkl","wb") as j:
        pickle.dump(exit_sum,j)
        #각 데이터가 어디서 나갔는지 exit

    with open("sync_test_exit_correct_CIFAR10_dense.pkl","wb") as i:
        pickle.dump(prob,i)
        #각 데이터 별 이미지 확률

    with open("sync_test_exit_entropy_CIFAR10_dense.pkl","wb") as p:
        pickle.dump(exit_sm,p)
        #각 데이터 별 entropy

    with open("sync_test_exit_time_CIFAR10_dense.pkl","wb") as t:
        pickle.dump(image_time,t)
        #각 데이터 별 시간





if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Example')

    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')

    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epochs', type=int, default=101, metavar='N',
                        help='number of epochs to train (default: 50)')

    parser.add_argument('--test_epochs', type=int, default=2, metavar='N',
                        help='number of epochs to test (default: 50)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--dataset', default='cifar10', help='dataset name')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--train_output', default='models/sync_train_model_Cifar10_dense.pth',
                        help='output directory')

    parser.add_argument('--test_output', default='models/sync_test_model_Cifar10_2_dense.pth',
                        help='output directory')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    NGPU = torch.cuda.device_count()
    torch.manual_seed(args.seed)


    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    train_dataset, train_loader, test_dataset, test_loader = data
    x, _ = next(iter(train_loader))
    num_devices = 1
    in_channels = x.shape[1]
    print(len(test_loader))
    model = net.DDNN_DenseNet(in_channels, 10)
    # print(model)
    # model = torch.nn.DataParallel(model)


    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if NGPU > 1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            #model = torch.nn.parallel.DataParallel(model, device_ids=[0,1])
        #print(torch.multiprocessing.get_start_method())
        model = model.to(device)

        print('Using {} {} device'.format(NGPU,device))




    train_model(model, args.train_output, train_loader, args.lr, args.epochs, num_devices,device)
    model = torch.load(args.train_output)
    test_model(model, args.test_output, test_loader, args.lr, args.test_epochs, num_devices,device)


