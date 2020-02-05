# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:38:23 2020

@author: ph
"""

import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def get_clips(read_pred_root, save_pred_root, dataset, cl):

    read_pred_paths = [os.path.join(read_pred_root, predict_path) for predict_path in os.listdir(read_pred_root)]

    if not os.path.exists(save_pred_root):
        os.makedirs(save_pred_root)

    for i in range(len(read_pred_paths)):

        subject = read_pred_paths[i].split("/")[-1].split("_")[-2]
        clip = read_pred_paths[i].split("/")[-1].split("_")[-1].split(".")[0]


        test_pred = list(pd.read_csv(read_pred_paths[i]).pred_label)

        if test_pred == [0 for i in test_pred] or test_pred == [1 for i in test_pred]:
            print('Producss dataset: {} class: {} subject: {} video: {} 没有预测结果.'.format(dataset, cl, subject, clip))
            continue
        else:
            print('Producss dataset: {} class: {} subject: {} video: {}.'.format(dataset, cl, subject, clip))


        data_clip = pd.DataFrame()
        pred_clip = []

        s = 1
        while s <= len(test_pred)-1:
            if test_pred[s] == test_pred[s-1] == 0:
                flag = s - 1
                while test_pred[s] == test_pred[s-1] == 0:
                    s += 1
                    if s == len(test_pred):
                        break
                pred_clip.append([indx for indx in range(flag, s)])
            else:
                s += 1

        data_clip['onset_frame'] = [clip_s[0] for clip_s in pred_clip]
        data_clip['offset_frame'] = [clip_s[-1] for clip_s in pred_clip]
        save_file = save_pred_root + '/{}_{}_{}_{}.csv'.format(dataset, cl, subject, clip)
        data_clip.to_csv(save_file, index=False)

if __name__ == '__main__':
    # ME2 和 SAMM_MEGC 数据库的 LOSO 交叉验证
    # ME2_Samples        到 ME2_Features
    # SAMM_MEGC_Samples  到 SAMM_MEGC_Features
    datasets = ['ME2', 'SAMM_MEGC']
    clss = ['micro', 'macro']
    for dataset in datasets:
        for cl in clss:
            # 读取样本文夹
            read_pred_root = './../datasets/CNN/Predict/{}_{}_Predict/'.format(dataset, cl)
            save_pred_root = './../datasets/CNN/Predict_clip/{}_{}_Predict_clip/'.format(dataset, cl)
            get_clips(read_pred_root, save_pred_root, dataset, cl)
