# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:53:48 2020

@author: ph
"""

import pandas as pd
from BCNN3_train import get_model

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # ME2 和 SAMM_MEGC 数据库的 LOSO 交叉验证
    # ME2_Samples        到 ME2_Features
    # SAMM_MEGC_Samples  到 SAMM_MEGC_Features
    datasets = ['ME2', 'SAMM_MEGC']
    clss = ['micro', 'macro']
    for dataset in datasets:
        # 读取视频文件夹
        read_video_path = './../datasets/' + dataset + '_video.csv'
        data_video = pd.read_csv(read_video_path, converters={u'subject': str, u'clip': str})


        # 读取样本文夹
        read_frame_root = './../datasets/BCNN3/Sample/{}_frame/'.format(dataset)
        
        for cl in clss:
            # 保存模型文件夹
            save_model_root = './../datasets/BCNN3/Model/{}_{}_Model'.format(dataset, cl)
            get_model(data_video, dataset, cl, read_frame_root, save_model_root)
            
            
            print('Finish dataset: {} class: {}.'.format(dataset, cl))
            
        print('Finish dataset: {}.'.format(dataset))
        
    print('Finish all.')