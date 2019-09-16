#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(256)
        self.bn10 = nn.BatchNorm2d(256)
        self.bn11 = nn.BatchNorm2d(256)

        self.bn12 = nn.BatchNorm2d(512)
        self.bn13 = nn.BatchNorm2d(512)
        self.bn14 = nn.BatchNorm2d(512)
        self.bn15 = nn.BatchNorm2d(512)
        self.bn16 = nn.BatchNorm2d(512)
        self.bn17 = nn.BatchNorm2d(512)
        self.bn18 = nn.BatchNorm2d(512)
        self.bn19 = nn.BatchNorm2d(512)

        self.bn20 = nn.BatchNorm2d(args.emb_dims)
        self.bn21 = nn.BatchNorm2d(args.emb_dims)
        self.bn22 = nn.BatchNorm2d(args.emb_dims)
        self.bn23 = nn.BatchNorm2d(args.emb_dims)
        self.bn24 = nn.BatchNorm2d(args.emb_dims)
        self.bn25 = nn.BatchNorm2d(args.emb_dims)
        self.bn26 = nn.BatchNorm2d(args.emb_dims)
        self.bn27 = nn.BatchNorm2d(args.emb_dims)
        self.bn28 = nn.BatchNorm2d(args.emb_dims)
        self.bn28 = nn.BatchNorm2d(args.emb_dims)
        self.bn29 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn11,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv12 = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=1, bias=False),
                                   self.bn12,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv13 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn13,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv14 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn14,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv15 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn15,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv16 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn16,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv17 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn17,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv18 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn18,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv19 = nn.Sequential(nn.Conv2d(512*2, 512, kernel_size=1, bias=False),
                                   self.bn19,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv20 = nn.Sequential(nn.Conv2d(512*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn20,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv21 = nn.Sequential(nn.Conv2d(args.emb_dims*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn21,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv22 = nn.Sequential(nn.Conv2d(args.emb_dims*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn22,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv23 = nn.Sequential(nn.Conv2d(args.emb_dims*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn23,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv24 = nn.Sequential(nn.Conv2d(args.emb_dims*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn24,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv25 = nn.Sequential(nn.Conv2d(args.emb_dims*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn25,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv26 = nn.Sequential(nn.Conv2d(args.emb_dims*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn26,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv27 = nn.Sequential(nn.Conv2d(args.emb_dims*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn27,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv28 = nn.Sequential(nn.Conv2d(args.emb_dims*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn28,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv29 = nn.Sequential(nn.Conv1d(16640, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn29,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn30 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn31 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        # x1 = x + x1

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x2 = x1 + x2
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        #x3 = x + x3
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        #x4 = x + x4

        x = get_graph_feature(x4, k=self.k)
        x = self.conv5(x)
        x5 = x.max(dim=-1, keepdim=False)[0]
        x5 = x4 + x5
        
        x = get_graph_feature(x5, k=self.k)
        x = self.conv6(x)
        x6 = x.max(dim=-1, keepdim=False)[0]
        x6 = x5 + x6
        
        x = get_graph_feature(x6, k=self.k)
        x = self.conv7(x)
        x7 = x.max(dim=-1, keepdim=False)[0]
        x7 = x6 + x7

        x = get_graph_feature(x7, k=self.k)
        x = self.conv8(x)
        x8 = x.max(dim=-1, keepdim=False)[0]
        x8 = x7 + x8

        x = get_graph_feature(x8, k=self.k)
        x = self.conv9(x)
        x9 = x.max(dim=-1, keepdim=False)[0]
        x9 = x8 + x9

        x = get_graph_feature(x9, k=self.k)
        x = self.conv10(x)
        x10 = x.max(dim=-1, keepdim=False)[0]
        x10 = x9 + x10

        x = get_graph_feature(x10, k=self.k)
        x = self.conv11(x)
        x11 = x.max(dim=-1, keepdim=False)[0]
        x11 = x10 + x11
        
        x = get_graph_feature(x11, k=self.k)
        x = self.conv12(x)
        x12 = x.max(dim=-1, keepdim=False)[0]
        #x12 = x11 + x12

        x = get_graph_feature(x12, k=self.k)
        x = self.conv13(x)
        x13 = x.max(dim=-1, keepdim=False)[0]
        x13 = x12 + x13
        
        x = get_graph_feature(x13, k=self.k)
        x = self.conv14(x)
        x14 = x.max(dim=-1, keepdim=False)[0]
        x14 = x13 + x14
        
        x = get_graph_feature(x14, k=self.k)
        x = self.conv15(x)
        x15 = x.max(dim=-1, keepdim=False)[0]
        x15 = x14 + x15

        x = get_graph_feature(x15, k=self.k)
        x = self.conv16(x)
        x16 = x.max(dim=-1, keepdim=False)[0]
        x16 = x15 + x16

        x = get_graph_feature(x16, k=self.k)
        x = self.conv17(x)
        x17 = x.max(dim=-1, keepdim=False)[0]
        x17 = x16 + x17

        x = get_graph_feature(x17, k=self.k)
        x = self.conv18(x)
        x18 = x.max(dim=-1, keepdim=False)[0]
        x18 = x17 + x18

        x = get_graph_feature(x18, k=self.k)
        x = self.conv19(x)
        x19 = x.max(dim=-1, keepdim=False)[0]
        x19 = x18 + x19
        
        x = get_graph_feature(x19, k=self.k)
        x = self.conv20(x)
        x20 = x.max(dim=-1, keepdim=False)[0]
        #x20 = x19 + x20
        
        x = get_graph_feature(x20, k=self.k)
        x = self.conv21(x)
        x21 = x.max(dim=-1, keepdim=False)[0]
        x21 = x20 + x21
        
        x = get_graph_feature(x21, k=self.k)
        x = self.conv22(x)
        x22 = x.max(dim=-1, keepdim=False)[0]
        x22 = x21 + x22
        
        x = get_graph_feature(x22, k=self.k)
        x = self.conv23(x)
        x23 = x.max(dim=-1, keepdim=False)[0]
        x23 = x22 + x23
        
        x = get_graph_feature(x23, k=self.k)
        x = self.conv24(x)
        x24 = x.max(dim=-1, keepdim=False)[0]
        x24 = x23 + x24
        
        x = get_graph_feature(x24, k=self.k)
        x = self.conv25(x)
        x25 = x.max(dim=-1, keepdim=False)[0]
        x25 = x24 + x25
        
        x = get_graph_feature(x25, k=self.k)
        x = self.conv26(x)
        x26 = x.max(dim=-1, keepdim=False)[0]
        x26 = x25 + x26
        
        x = get_graph_feature(x26, k=self.k)
        x = self.conv27(x)
        x27 = x.max(dim=-1, keepdim=False)[0]
        x27 = x26 + x27
        
        x = get_graph_feature(x27, k=self.k)
        x = self.conv28(x)
        x28 = x.max(dim=-1, keepdim=False)[0]
        x28 = x27 + x28
        
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28), dim=1)

        x = self.conv29(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn30(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn31(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
