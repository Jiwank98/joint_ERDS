import warnings

warnings.filterwarnings('ignore')

import argparse
import Resource_Allocation_OUTER as outer_algorithm


def get_parser():
    parser = argparse.ArgumentParser(description="Early Exiting-Aware Joint Resource Allocation and DNN Splitting "
                                                 "for Multi-Sensor Digital Twin in Edge-Cloud Collaborative System'")

    #For Outer Algrothm
    parser.add_argument('--scenario', required=True, help='types of scenario {original, dynamic_tr, dynamic_cloud, dynamic_edge}')
    parser.add_argument('--scenario2', required=True, help='types of scenario2 {0, 1, 2, 3, 4}')
    parser.add_argument('--digital_twin', required=True, help='The number of digital twins {1, 2, 3}')

    #For Inner Algorithm
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_epochs', type=int, default=2, metavar='N',
                        help='number of epochs to test (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dataset', default='cifar10', help='dataset name')
    parser.add_argument('--algorithm', type=int, default=0, metavar='N',
                        help='0:OSA, 1:Random, 2:HA, 3:Cloud, 4:Edge, 5:IAO')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--train_output1', default='models/sync_train_model_Cifar10:dense.pth',
                        help='output directory')
    parser.add_argument('--train_output2', default='models/sync_train_model_Cifar100.pth',
                        help='output directory')
    parser.add_argument('--train_output3', default='models/sync_train_model_ImageNet.pth',
                        help='output directory')
    parser.add_argument('--test_output', default='models/sync_test_model_Cifar10:dense.pth',
                        help='output directory')
    return parser



def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    scenario_list = ['original', 'dynamic_tr', 'dynamic_cloud', 'dynamic_edge']
    digital_twins = int(args.digital_twin)
    scenario2 = int(args.scenario2)

    if args.scenario not in scenario_list:
        raise ValueError('Invalid scenario.')

    if args.scenario == 'original':
        nof = 2
        tof = 1
        cof = 1
        outer_algorithm.run(args,digital_twins,nof, cof, tof)


    if args.scenario == 'dynamic_tr':
        tof_list = [1/2, 1, 3/2, 2, 3] #increasing_transmission_time, 원하는 시나리오로 설정
        nof = 2
        cof = 1
        tof = tof_list[scenario2]
        outer_algorithm.run(args,digital_twins,nof, cof, tof)


    if args.scenario == 'dynamic_cloud':
        cof_list = [3/2, 1, 2/3, 1/2, 1/3] #decreasing_cloud_time,원하는 시나리오로 설정
        nof = 2
        tof = 1
        cof = cof_list[scenario2]
        outer_algorithm.run(args,digital_twins,nof, cof, tof)


    if args.scenario == 'dynamic_edge':
        nof_list = [3, 2, 4/3, 1, 1] #decreasing_edge_time,원하는 시나리오로 설정
        nof = nof_list[scenario2]
        tof = 1
        cof = 1
        outer_algorithm.run(args,digital_twins,nof, cof, tof)





if __name__ == '__main__':
    main()