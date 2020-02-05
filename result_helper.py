# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 22:03:10 2020

@author: ph
"""

import os
import numpy as np
import pandas as pd

def delete_short(clips, shortest_len):
    clps = []
    for i, clip in enumerate(clips):
        n = clip[1] - clip[0] + 1
        if n >= shortest_len:
            clps.append(clip)
    return clps

def delete_long(clips, longest_len):
    clps = []
    for i, clip in enumerate(clips):
        n = clip[1] - clip[0] + 1
        if n <= longest_len:
            clps.append(clip)
    return clps


def video_to_labels(df, sub, clip, dataset):
    for subject_out_idx in range(len(df['subject'].value_counts())):
        data_sub_column = 'subject'
        subject_list = list(df[data_sub_column].unique())
        subject_out = subject_list[subject_out_idx]
        # print(subject_out)
        if subject_out == sub:
            df_data_subs = df[df[data_sub_column] == subject_out]

            # 根据受试者视频文件夹的编号获取受试者视频文件夹下样本的 DataFrame
            for clip_out_idx in range(len(df_data_subs['clip'].value_counts())):
                data_clip_column = 'clip'
                clip_list = list(df_data_subs[data_clip_column].unique())
                clip_out = clip_list[clip_out_idx]

                if dataset == 'ME2':
                    clip_out_mark = clip_out[3:7]
                else:
                    clip_out_mark = clip_out[-1:]
                if clip_out_mark == clip:
                    df_data_clip = df_data_subs[df_data_subs[data_clip_column] == clip_out]
                    break
                else:
                    df_data_clip = pd.DataFrame()
    return df_data_clip

def calc_iou(clip1, clip2):
    intersection = max(0, min(clip1[1], clip2[1]) - max(clip1[0], clip2[0])) + 1
    union = (clip1[1] - clip1[0] + 1) + (clip2[1] - clip2[0] + 1) - intersection
    if clip1[1] <= clip1[0] or clip2[1] <= clip2[0] or union <= 0: return 0.0
    else: return float(intersection) / float(union)

def return_preds_labels(videos_path, dataset, cl, data_label, shortest_len=-1, longest_len=-1):
    pred_clip = []
    true_clip = []
    re_videos = []

    for video_path in videos_path:
        video = []
        if dataset == 'ME2':
            video.append(video_path.split('/')[-1].split("_")[-2])  # 受试者
            video.append(video_path.split('/')[-1].split("_")[-1][:4])  # 视频片段
        else:
            video.append(video_path.split('/')[-1].split("_")[-2])  # 受试者
            video.append(video_path.split('/')[-1].split("_")[-1][:1])  # 视频片段


        df_clips = pd.read_csv(video_path)
        clips = []
        for idx in range(len(df_clips)):
            clips.append([int(df_clips.loc[idx, 'onset_frame']), int(df_clips.loc[idx, 'offset_frame'])])

        if shortest_len != -1:
            clips = delete_short(clips, shortest_len)
        if longest_len != -1:
            clips = delete_long(clips, longest_len)

        lables = video_to_labels(data_label, video[0], video[1], dataset)
        

        lbs = []
        for lb in range(len(lables)):
            if cl == 'micro':
                if lables.iloc[lb]['label'] == 0:
                    lbs.append([lables.iloc[lb]['onset_frame'], lables.iloc[lb]['offset_frame']])
            else:
                if lables.iloc[lb]['label'] == 1:
                    lbs.append([lables.iloc[lb]['onset_frame'], lables.iloc[lb]['offset_frame']])

        pred_clip.append(clips)
        true_clip.append(lbs)
        re_videos.append(video)

    return pred_clip, true_clip, re_videos

def recall_precision_f1(A_clips, M_clips, N_clips):
    if M_clips != 0:
        recall = float(A_clips) / float(M_clips)                 # A / M
    else:
        recall = 0

    if N_clips != 0:
        precision = float(A_clips) / float(N_clips)             # A / N
    else:
        precision = 0

    if (recall + precision) != 0:
        F1_score = 2 * recall * precision / (recall + precision)
    else:
        F1_score = 0
        
    return recall, precision, F1_score

def pred_clips_analysis(pred_clip, true_clip, thresh):

    M_clips = 0
    N_clips = 0
    A_clips = 0 # TP = A ;  FP = N - A ; FN = M - A

    results_pred = []
    results_ture = []

    for i in range(len(pred_clip)):
        ture_labels = true_clip[i]                  # 每个视频中的样本标签

        num_clips_pred_batch = len(pred_clip[i])    # 每个视频预测 微表情（宏表情） 的clip个数
        num_clips_true_batch = len(true_clip[i])    # 每个视频真实 微表情（宏表情） 的clip个数

        result_pred = []
        result_ture = []

        for ture_label in ture_labels:             # 获取每个视频中的真实 微表情（宏表情）的个数，记为 M
            M_clips = M_clips + 1
            result_ture.append(0)

        for clip_pred_idx in range(num_clips_pred_batch):      # 获取每个视频中预测的 微表情（宏表情）的个数， 记为 N
            N_clips = N_clips + 1
            p_clip = pred_clip[i][clip_pred_idx]
            result_pred.append(0)

            for clip_ture_idx in range(num_clips_true_batch):
                t_clip = ture_labels[clip_ture_idx]
                if calc_iou(p_clip, t_clip) >= thresh:
                    result_pred[clip_pred_idx] = result_pred[clip_pred_idx] + 1
                    result_ture[clip_ture_idx] = result_ture[clip_ture_idx] + 1
                    A_clips = A_clips + 1

        results_pred.append(result_pred)
        results_ture.append(result_ture)
        
   

    # evaluation
    recall, precision, F1_score = recall_precision_f1(A_clips, M_clips, N_clips)
    
   

    return A_clips, M_clips, N_clips, results_pred, results_ture, recall, precision, F1_score

def get_result(cl, pred_clip, true_clip, re_videos, thresh):
    pred_info = {}
    pred_info[cl] = {"pred_clip": pred_clip, "true_clip": true_clip, "videos": re_videos}

    result_info = {'pred_result': [], "videos": []}
    info = pred_info[cl]
    pred_clip_reuslt = info['pred_clip']
    true_clip_result = info['true_clip']
    videos = info['videos']

    def true_pair(true, preds):
        for pred in preds:
            if calc_iou(true, pred) >= thresh:
                return [true, pred, 'TP']
        return [true, [], 'FN']

    def pred_pair(pred, trues):
        for true in trues:
            if calc_iou(pred, true) >= thresh:
                return [true, pred, 'TP']
        return [[], pred, 'FP']

    def sort_rule(pair):
        if pair[2] == 'TP' or pair[2] == 'FP':
            return pair[1][0]
        else:
            return pair[0][0]


    for video in videos:
        index = videos.index(video)
        # print(index)
        preds = pred_clip_reuslt[index]
        trues = true_clip_result[index]

        pairs = []
        for pred in preds:
            pairs.append(pred_pair(pred, trues))
        for true in trues:
            pair = true_pair(true, preds)
            if pair[2] == 'FN' or (pair not in pairs):
                pairs.append(pair)

        if video in result_info['videos']:
            index_result = result_info['videos'].index(video)
            result_info['pred_result'][index_result] += pairs
        else:
            result_info['pred_result'].append(pairs)
            result_info['videos'].append(video)

    pred_result = result_info['pred_result']
    for i in range(len(pred_result)):
        result_info['pred_result'][i] = sorted(pred_result[i], key=sort_rule)

    videos = result_info['videos']
    pred_result = result_info['pred_result']
    data_result = pd.DataFrame()
    for video in videos:
        df_result = pd.DataFrame(
            columns=['Video_ID', 'GT_onset', 'GT_offset', 'Predicted_onset', 'Predicted_offset', 'Result'])
        index = videos.index(video)
        pairs = pred_result[index]

        for pair in pairs:
            if pair[0]:
                GT_onset = int(pair[0][0])
                GT_offset = int(pair[0][1])
            else:
                GT_onset = '-'
                GT_offset = '-'
            if pair[1]:
                Predicted_onset = int(pair[1][0])
                Predicted_offset = int(pair[1][1])
            else:
                Predicted_onset = '-'
                Predicted_offset = '-'
            result = pair[2]

            Video_ID = '{}_{}'.format(video[0].replace('s', ''), video[1])
            data = dict({"Video_ID": Video_ID,
                         'GT_onset': GT_onset, 'GT_offset': GT_offset,
                         'Predicted_onset': Predicted_onset, 'Predicted_offset': Predicted_offset,
                         'Result': result})
            df_result = df_result.append(data, ignore_index=True)
        data_result = data_result.append(df_result, ignore_index=True)
    return data_result
