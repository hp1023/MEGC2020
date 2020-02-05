# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:10:03 2020

@author: ph
"""

import os
import csv
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

def predict_result(data_test, dataset, cl, model, save_pred_root):

    # 读取图像文件夹
    read_images_root = './../datasets/processed_data/' + dataset + '_crop_gray/'

    if not os.path.exists(save_pred_root):
        os.makedirs(save_pred_root)

    transforms_test = transforms.Compose([
        transforms.Resize(244),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.92206, 0.92206, 0.92206],
                             std=[0.08426, 0.08426, 0.08426])])

    for idx in range(len(data_test)):

        data_pred = pd.DataFrame()  # 保存预测结果

        subject = data_test.iloc[idx]['subject']
        clip = data_test.iloc[idx]['clip']
        tail = data_test.iloc[idx]['tail']

        save_pred_path = save_pred_root + '/{}_{}_{}_{}.csv'.format(dataset, cl, subject, clip)

        # features = []
        test_pred = []
        for frame in range(1, tail + 1):
            img_path = read_images_root + subject + '/' + clip + '/img' + str(frame).zfill(5) + '.jpg'
            image = Image.open(img_path).convert('RGB')
            image = transforms_test(image).unsqueeze(0).cuda()
            out_put = model(image)
            
            _, y_pred = torch.max(out_put.data, 1)
            
            test_pred.append(y_pred.item())
            
            # feature = feature.cpu().detach().numpy()
            # feature = np.ravel(np.array(feature))
            # feature = str(list(feature))[1:-1]
            # features.append(feature)
            
        # data_pred['subject'] = [subject for i in range(len(test_pred))]
        # data_pred['clip'] = [clip for i in range(len(test_pred))]
        # data_pred['frame'] = [i+1 for i in range(len(test_pred))]
        data_pred['test_pred'] = test_pred
        # data_pred['features'] = features
        print(dict(data_pred['test_pred'].value_counts()))
        data_pred.to_csv(save_pred_path, index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC)
        print('Finish dataset: {} class: {} subject: {} video: {}.'.format(dataset, cl, subject, clip))

def get_predict(data_video, dataset, cl, read_model_root, save_pred_root):

    for subject_out_idx in range(len(data_video['subject'].value_counts())):
        data_sub_column = 'subject'
        subject_list = list(data_video[data_sub_column].unique())
        subject_out = subject_list[subject_out_idx]
        data_test = data_video[data_video[data_sub_column] == subject_out].reset_index(drop=True)

        read_model_path = read_model_root + '/{}_{}_{}_model.pkl'.format(dataset, cl, subject_out)
        model = torch.load(read_model_path)

        data_pred = predict_result(data_test, dataset, cl, model, save_pred_root)

        print('Finish dataset: {} class: {} subject: {}.'.format(dataset, cl, subject_out))
        

    return data_pred

    

if __name__ == '__main__':
    datasets = ['ME2', 'SAMM_MEGC']
    clss = ['micro', 'macro']
    for dataset in datasets:
        # 读取视频文件夹
        read_video_path = './../datasets/' + dataset + '_video.csv'
        data_video = pd.read_csv(read_video_path, converters={u'subject': str, u'clip': str})


        for cl in clss:
            # 读取模型文件夹
            read_model_root = './../datasets/BCNN/Model/{}_{}_Model/'.format(dataset, cl)
            # 保存预测结果文件
            save_pred_root = './../datasets/BCNN/Predict/{}_{}_Predict/'.format(dataset, cl)

            data_pred = get_predict(data_video, dataset, cl, read_model_root, save_pred_root)