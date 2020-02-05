# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:12:47 2020

@author: ph
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from BCNN3_dataset import MyDataset, BCNN, weights_init

def sample_balance(df_train):
    df_count = dict(df_train['label'].value_counts())
    df_list = []
    for (label, frame) in df_count.items():
        df_list.append(df_count[label])
        
    print('训练集{}'.format(df_count))  
    sample_num = np.array(df_list).min()
    
    df_balance = pd.DataFrame(columns=df_train.columns)
    for (label, frame) in df_count.items():
        df_label = df_train[df_train['label'] == label].sample(sample_num).reset_index(drop=True)
        df_balance = df_balance.append(df_label, ignore_index=True)

    print('训练集样本数{}'.format(dict(df_balance['label'].value_counts())))
    return df_balance


def train_data(data_video, dataset, cl, subject_out, read_frame_root):
    '''
    获取训练样本和验证样本的图像路径和标签   
    :param data_video: 所有样本的视频统计
    :param dataset: 数据库（ME2 和 SAMM_MEGC）
    :param cl: 表情类型（micro 和 macro）
    :param subject_out: 受试者名称
    :param read_frame_root: 样本帧的存放文件夹
    :return: 
    '''
    
    # 图像路径文件
    read_images_root = './../datasets/processed_data/' + dataset + '_crop_faces/'

    data_train = pd.DataFrame()
    data_val = pd.DataFrame()

    for idx in range(len(data_video)):
        
        subject = data_video.iloc[idx]['subject']
        clip = data_video.iloc[idx]['clip']
        
        # 读取每个受试者每个视频中的样本帧的文件
        read_frame_path = read_frame_root + '{}_{}_{}.csv'.format(dataset, subject, clip)
        if os.path.exists(read_frame_path):
            data_sub = pd.read_csv(read_frame_path, converters={u'subject': str, u'clip': str})
        else:
            continue
        
        if subject == subject_out:
            data_val = data_val.append(data_sub, ignore_index=True)
        else:
            data_train = data_train.append(data_sub, ignore_index=True)

    data_train = shuffle(sample_balance(data_train)).reset_index(drop=True)
    data_val = shuffle(sample_balance(data_val)).reset_index(drop=True)

    train_paths = [read_images_root +
                   data_train.loc[i, 'subject'] + '/' +
                   data_train.loc[i, 'clip'] + '/img' +
                   str(data_train.loc[i, 'frame']).zfill(5) + '.jpg'
                   for i in range(len(data_train))]
    train_labels = list(data_train.label)

    val_paths = [read_images_root +
                 data_val.loc[i, 'subject'] + '/' +
                 data_val.loc[i, 'clip'] + '/img' +
                 str(data_val.loc[i, 'frame']).zfill(5) + '.jpg'
                 for i in range(len(data_val))]
    val_labels = list(data_val.label)
    
    return train_paths, train_labels, val_paths, val_labels

def train(data_video, dataset, cl, subject_out, read_frame_root, save_model_root):
    '''
    留一人交叉样本训练和验证    
    :param data_video: 所有样本的视频统计
    :param dataset: 数据库（ME2 和 SAMM_MEGC）
    :param cl:  表情类型（micro 和 macro）
    :param subject_out: 受试者名称
    :param read_frame_root: 样本帧的存放文件夹
    :param save_model_root: 模型的保存文件夹
    :return: 
    '''
    # 超参数
    LEARNING_RATE = 0.001
    EPISODE = 300
    BATCH_SIZE = 32
    CLASS_NUM = 3
    

    train_paths, train_labels, val_paths, val_labels = train_data(data_video, dataset, cl, subject_out, read_frame_root)
    # Step 1: init data folders 初始化数据文件夹
    print("Subject: {} —— init data folders".format(subject_out))
    
    if not os.path.exists(save_model_root):
        os.makedirs(save_model_root)

    save_model_path = save_model_root + '/{}_{}_{}_model.pkl'.format(dataset, cl, subject_out)
    save_loss_path = save_model_root + '/{}_{}_{}_loss.pkl'.format(dataset, cl, subject_out)

    # Step 2: init neural networks 初始化网络结构
    print("Subject: {} —— init neural networks".format(subject_out))

    model = BCNN()
    model.apply(weights_init) 
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # Step 3: build graph
    print("Subject: {} —— Training...".format(subject_out))

    transforms_train = transforms.Compose([
        transforms.Resize(244), transforms.RandomRotation(10),
        transforms.RandomCrop(224), transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.92206, 0.92206, 0.92206],
                             std=[0.08426, 0.08426, 0.08426])])

    transforms_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.92206, 0.92206, 0.92206],
                             std=[0.08426, 0.08426, 0.08426])])

    train_dataset = MyDataset(paths=train_paths, labels=train_labels, transform=transforms_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    val_dataset = MyDataset(paths=val_paths, labels=val_labels, transform=transforms_val)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    best_uf1 = 0.0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_uf1 = []

    for episode in range(EPISODE):
        optimizer_scheduler.step(episode)

        train_step_loss = 0.0
        train_step_acc = 0.0
        for train_step, (x_train, y_train) in enumerate(train_dataloader):
            optimizer.zero_grad()  # 梯度清零
            out_puts = model(Variable(x_train).cuda())

            mse = nn.MSELoss().cuda()

            one_hot_labels_train = Variable(
                torch.zeros(x_train.size(0), CLASS_NUM).scatter_(1, y_train.view(-1, 1).long(),
                                                                 1)).cuda()  # BATCH_SIZE * CLASS_NUM
            loss = mse(out_puts, one_hot_labels_train)

            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 梯度优化

            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)  # 梯度裁剪

            _, y_train_pred = torch.max(out_puts.data, 1)
            _, y_train_true = torch.max(one_hot_labels_train.data, 1)
            train_step_loss += loss.item() * x_train.size(0)
            train_step_acc += y_train_pred.eq(y_train_true).cpu().sum().item()

        train_step_loss /= float(len(train_dataloader.dataset))
        train_step_acc = float(train_step_acc) / float(len(train_dataloader.dataset))
        train_loss.append(train_step_loss)
        train_acc.append(train_step_acc)

        val_step_loss = 0.0
        val_step_acc = 0.0
        y_ture_all = []
        y_pred_all = []

        for val_step, (x_val, y_val) in enumerate(val_dataloader):
            out_puts = model(Variable(x_val).cuda())

            mse = nn.MSELoss().cuda()

            one_hot_labels_val = Variable(torch.zeros(x_val.size(0), CLASS_NUM).scatter_(1, y_val.view(-1, 1).long(),
                                                                                         1)).cuda()  # BATCH_SIZE * CLASS_NUM
            loss = mse(out_puts, one_hot_labels_val)

            _, y_val_pred = torch.max(out_puts.data, 1)
            _, y_val_true = torch.max(one_hot_labels_val.data, 1)
            val_step_loss += loss.item() * x_val.size(0)
            val_step_acc += y_val_pred.eq(y_val_true).cpu().sum().item()
            for i in range(len(y_val_true)):
                y_ture_all.append(y_val_true[i].cpu().item())
                y_pred_all.append(y_val_pred[i].cpu().item())

        val_step_loss /= float(len(val_dataloader.dataset))
        val_step_acc = float(val_step_acc) / float(len(val_dataloader.dataset))
        val_loss.append(val_step_loss)
        val_acc.append(val_step_acc)

        uf1 = f1_score(y_ture_all, y_pred_all, average='macro')
        # print("uf1: {:0.6f}  best_uf1: {:0.6f} ".format(uf1, best_uf1))
        val_uf1.append(uf1)
        
        data_pred = pd.DataFrame()
        data_pred['ture_label'] = y_ture_all
        data_pred['pred_label'] = y_pred_all
        if uf1 >= best_uf1:
            torch.save(model, save_model_path)
            best_uf1 = uf1
            print('subject: {} 真实值：{} 预测值：{} \n              {}\n              {}'.format(subject_out,
                    dict(data_pred['ture_label'].value_counts()), 
                    dict(data_pred['pred_label'].value_counts()), 
                    y_ture_all[:20], y_pred_all[:20],))

        if (episode + 1) % 1 == 0:
            print("subject: {} episode: {:3d} \n"
                  "              train loss: {:0.6f}  train acc: {:0.6f} \n"
                  "              val   loss: {:0.6f}  val   acc: {:0.6f} val uf1: {:0.6f}".format(
                    subject_out, episode + 1, train_step_loss, train_step_acc, val_step_loss, val_step_acc, uf1))
        
        if (episode + 1) % 20 == 0:            
            print('subject: {} 真实值：{} 预测值：{} \n              {}\n              {}'.format(subject_out,
                    dict(data_pred['ture_label'].value_counts()), 
                    dict(data_pred['pred_label'].value_counts()), 
                    y_ture_all[:20], y_pred_all[:20],))

    print('Finish dataset: {} subject: {}.'.format(dataset, subject_out))
    with open(save_loss_path, 'wb') as file:
        data = dict(train_loss=train_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc, val_uf1=val_uf1)
        pickle.dump(data, file)
    return train_loss, train_acc, val_loss, val_acc, val_uf1

def get_model(data_video, dataset, cl, read_frame_root, save_model_root):
    '''
    留一人交叉验证，获取模型
    :param data_video: 所有样本的视频统计
    :param dataset: 数据库（ME2 和 SAMM_MEGC）
    :param cl: 表情类型（micro 和 macro）
    :param read_frame_root: 样本帧的存放文件夹
    :param save_model_root: 模型的保存文件夹
    :return: 
    '''

    if not os.path.exists(save_model_root):
        os.makedirs(save_model_root)

    # 留一人交叉验证 (22 个受试者)
    for subject_out_idx in range(len(data_video['subject'].value_counts())):
        if dataset == 'ME2' and cl == 'macro' and subject_out_idx == 0:
            continue
        if dataset == 'ME2' and cl == 'macro' and subject_out_idx == 1:
            continue
        if dataset == 'ME2' and cl == 'macro' and subject_out_idx == 2:
            continue
        if dataset == 'ME2' and cl == 'macro' and subject_out_idx == 3:
            continue
        if dataset == 'ME2' and cl == 'macro' and subject_out_idx == 4:
            continue        
        subject_list = list(data_video['subject'].unique())
        subject_out = subject_list[subject_out_idx]
        # data_test = data[data[data_sub_column] == subject_out].reset_index(drop=True)
        train_loss, train_acc, val_loss, val_acc, val_uf1 = train(data_video, dataset, cl, subject_out, read_frame_root, save_model_root)
