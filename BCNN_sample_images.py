# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:50:35 2020

@author: ph
"""

import os
import csv
import numpy as np
import pandas as pd


def generate_samples(df):
    '''
    根据标记文件生成“微表情、宏表情、无表情”样本
    :param df: 标记文件
    :return: 样本文件
    '''
    df_samples = pd.DataFrame(columns=df.columns)  # 生成样本文件

    # 根据受试者的编号获取受试者样本的 DataFrame
    for subject_out_idx in range(len(df['subject'].value_counts())):
        data_sub_column = 'subject'
        subject_list = list(df[data_sub_column].unique())
        subject_out = subject_list[subject_out_idx]
        df_data_subs = df[df[data_sub_column] == subject_out]

        # 根据受试者视频文件夹的编号获取受试者视频文件夹下样本的 DataFrame
        for clip_out_idx in range(len(df_data_subs['clip'].value_counts())):
            data_clip_column = 'clip'
            clip_list = list(df_data_subs[data_clip_column].unique())
            clip_out = clip_list[clip_out_idx]
            df_data_clip = df_data_subs[df_data_subs[data_clip_column] == clip_out]

            # 获取该视频文件夹下所有微表情和宏表情样本的持续时间 offset_frame - onset_frame
            Duration_list = list(df_data_clip['offset_frame'] - df_data_clip['onset_frame'])
            # 获取最大微表情和宏表情样本的最大持续时间
            Duration = int(np.array(Duration_list).max())

            tail = int(df_data_clip.iloc[0]['tail_frame'])

            # 根据微表情和宏表情样本的最大持续时间，提取一些无表情的样本，从第 Duration 帧开始，每隔 Duration*3 提取一次无表情样本
            for i in range(Duration, tail - Duration, Duration * 10):

                # 获取的无表情样本
                no_expression = [no_e for no_e in range(i, i + Duration)]

                # 判断获取的无表情样本是否和微表情和宏表情样本有交集
                for j in range(len(df_data_clip)):
                    expression = [e for e in range(int(df_data_clip.iloc[j]['onset_frame']),
                                                   int(df_data_clip.iloc[j]['offset_frame']))]
                    intersection = list(set(no_expression) & set(expression))
                    # print(intersection)
                    if len(intersection):
                        break
                    else:
                        # print(i, i+Duration)
                        data = dict({"subject": subject_out, 'clip': clip_out,
                                     'onset_frame': i, 'apex_frame': int(i + Duration / 2),
                                     'offset_frame': i + Duration,
                                     'label': 2, 'tail_frame': tail})
                        df_data_clip = df_data_clip.append(data, ignore_index=True)
                        break
            df_data_clip = df_data_clip.sort_index(axis=0, ascending=True, by=['onset_frame'])

            df_samples = df_samples.append(df_data_clip, ignore_index=True)
    return df_samples


def frame_label(df, dataset, save_frame_path):
    '''
    给数据库的每帧图像定义标签，将无监督问题转化为有监督问题（分类、回归）
    :param df: 视频段标签
    :return: 帧标签
    '''
    for subject_out_idx in range(len(df['subject'].value_counts())):
        data_sub_column = 'subject'
        subject_list = list(df[data_sub_column].unique())
        subject_out = subject_list[subject_out_idx]
        df_data_subs = df[df[data_sub_column] == subject_out]

        # 根据受试者视频文件夹的编号获取受试者视频文件夹下样本的 DataFrame
        for clip_out_idx in range(len(df_data_subs['clip'].value_counts())):
            data_clip_column = 'clip'
            clip_list = list(df_data_subs[data_clip_column].unique())
            clip_out = clip_list[clip_out_idx]
            df_data_clip = df_data_subs[df_data_subs[data_clip_column] == clip_out].reset_index(drop=True)

            if dataset == 'ME2':
                subject = subject_out
                clip = clip_out[3:7]
            else:
                subject = str(subject_out).zfill(3)
                clip = clip_out.split('_')[1]

            # print(df_data_clip)
            df_frames = pd.DataFrame(columns=['subject', 'clip', 'frame', 'label', 'tail'])

            save_path = save_frame_path + '{}_{}_{}.csv'.format(dataset, subject, clip)
            print(save_path)

            for idx in range(len(df_data_clip)):
                tail = df_data_clip.iloc[idx]['tail_frame']
                onset = int(df_data_clip.iloc[idx]['onset_frame']) + 2
                offset = int(df_data_clip.iloc[idx]['offset_frame']) - 2
                label = df_data_clip.iloc[idx]['label']

                # if dataset == 'ME2':
                #     step = 1 if label == 0 else 5
                # else:
                #     step = 2 if label == 0 else 10 if label == 1 else 5
                # if offset - onset > 1000: step = 100
                #
                if offset - onset < 20:
                    step = 1
                else:
                    step = int((offset - onset) / 20)
                # print('时间间隔{}, 步长{}'.format(offset - onset, step))
                for i in range(onset, offset + 1, step):
                    data = dict({"subject": subject, 'clip': clip,
                                 'frame': i, 'label': label, 'tail': tail})

                    df_frames = df_frames.append(data, ignore_index=True)

            df_frames.to_csv(save_path, index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC)
    return df_frames


def print_min_max(data, dataset):
    Duration_micro = list(data[data['label'] == 0]['offset_frame'] - data[data['label'] == 0]['onset_frame'])
    Duration_macro = list(data[data['label'] == 1]['offset_frame'] - data[data['label'] == 1]['onset_frame'])
    Duration_notme = list(data[data['label'] == 2]['offset_frame'] - data[data['label'] == 2]['onset_frame'])
    print('数据库{}中，最长的微表情序列{}，最短的微表情序列{}, 微表情均值{}'
          .format(dataset, np.array(Duration_micro).max(), np.array(Duration_micro).min(), np.mean(Duration_micro)))
    print('数据库{}中，最长的宏表情序列{}，最短的宏表情序列{}, 宏表情均值{}'
          .format(dataset, np.array(Duration_macro).max(), np.array(Duration_macro).min(), np.mean(Duration_macro)))
    print('数据库{}中，最长的无表情序列{}，最短的无表情序列{}, 无表情均值{}'
          .format(dataset, np.array(Duration_notme).max(), np.array(Duration_notme).min(), np.mean(Duration_notme)))

if __name__ == '__main__':
    # 数据预处理第一步，生成各自的训练数据
    # ME2        到 ME2_Samples
    # SAMM_MEGC  到 SAMM_MEGC_Samples
    # 注意：
    # ME2 中有 98 个视频序列，但是其中只有 95 个视频序列有 微表情或者宏表情发生
    # SAMM_MEGC 中有 147 个视频序列，但是其中只有 146 个视频序列有 微表情或者宏表情发生, 012_4_1 表情发生的起始帧、峰值帧、终点帧全部超过文件视频帧数，因此去除
    datasets = ['ME2', 'SAMM_MEGC']
    for dataset in datasets:
        read_src_path = './../datasets/' + dataset + '.csv'
        save_sample_path = './../datasets/BCNN3/' + dataset + '_Samples.csv'

        data = pd.read_csv(read_src_path, converters={u'subject': str, u'slip': str})
        # 第 160 行 012_4_1 表情发生的起始帧、峰值帧、终点帧全部超过文件视频帧数
        # 第 328 行 020_6_5 终点帧超过文件视频帧数
        # 第 486 行 036_7_4 终点帧超过文件视频帧数
        if dataset == 'SAMM_MEGC':
            data.loc[328, 'offset_frame'] = data.loc[328, 'tail_frame']
            data.loc[486, 'offset_frame'] = data.loc[486, 'tail_frame']
            data = data.drop([160]).reset_index(drop=True)

        # 生成训练数据 微表情、宏表情、无表情
        data_samples = generate_samples(data)
        print_min_max(data_samples, dataset)

        # 提取视频序列的每一帧作为训练样本，存放在 dataset_frame 文件下
        save_frame_root = './../datasets/BCNN3/Sample/{}_frame/'.format(dataset)
        if not os.path.exists(save_frame_root):
            os.makedirs(save_frame_root)

        data_frames = frame_label(data_samples, dataset, save_frame_root)

        data_samples.to_csv(save_sample_path, index=False, quoting=csv.QUOTE_NONNUMERIC)


