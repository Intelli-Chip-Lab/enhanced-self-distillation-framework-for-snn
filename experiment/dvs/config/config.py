# -*- coding: utf-8 -*-
import os
import argparse

parser = argparse.ArgumentParser(description='Training SNN')
parser.add_argument('--seed', default=60, type=int)
parser.add_argument('--arch', default="resnet18", type=str, help="")  # used
parser.add_argument('--dataset', default="CIFAR10_DVS_Aug", type=str, help="dataset")  # used
parser.add_argument('--data_path', default="/home/zxc/code/asgl_sum/experiment/dvs/dataset", type=str)  # used
parser.add_argument('--log_path', default="./log", type=str, help="log path")  # used
parser.add_argument('--auto_aug', default=True, action='store_true')  # used
parser.add_argument('--cutout', default=True, action='store_true')  # used
parser.add_argument('--resume', default=None, type=str)  # used
parser.add_argument('--train_batch_size', default=32, type=int)  # used
parser.add_argument('--val_batch_size', default=32, type=int)  # used
parser.add_argument('--lr', default=0.1, type=float)  # used
parser.add_argument('--wd', default=5e-4, type=float)  # used
parser.add_argument('--num_epoch', default=300, type=int)  # used # check
parser.add_argument('--num_workers', default=8, type=int)  # used
parser.add_argument('--optim', default='SGDM', type=str)  # used
parser.add_argument('--decay', default=0.2, type=float)  # used
parser.add_argument('--v_reset', default=None, type=float)
parser.add_argument('--thresh', default=1.0, type=float)  # used
parser.add_argument('--device', default='cuda:0', type=str)  # used
parser.add_argument('--T', default=10, type=int, help='num of time steps')  # used # check
parser.add_argument('--scheduler', default='COSINE', type=str)  # used
parser.add_argument('--detach_reset', default=True, action='store_true')  # used
parser.add_argument('--alp', default=1.0, type=float)
parser.add_argument('--beta', default=0.1, type=float)
args = parser.parse_args()
