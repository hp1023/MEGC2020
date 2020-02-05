# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:35:38 2020

@author: ph
"""


import os
import pandas as pd
import result_helper as hp

if __name__ == '__main__':
    datasets = ['SAMM_MEGC', 'ME2']
    clss = ['micro', 'macro']
    df = pd.DataFrame()
    for dataset in datasets:

        if dataset == 'SAMM_MEGC':
            Video_ID = 1
            df_s = pd.DataFrame()
        else:
            Video_ID = 2
            df_m = pd.DataFrame()
        data = dict({"Video_ID": Video_ID,
                     'GT_onset': '', 'GT_offset': '',
                     'Predicted_onset': '', 'Predicted_offset': '',
                     'Result': ''})
        df = df.append(data, ignore_index=True)
    
        A_clips_all = 0
        M_clips_all = 0
        N_clips_all = 0
        for cl in clss:
            if dataset == 'ME2' and cl == 'micro':
                shortest_len = 6    # 7
                longest_len = 16
            elif dataset == 'ME2' and cl == 'macro':
                shortest_len = 17
                longest_len = -1
            elif (dataset == 'SAMM_MEGC' or dataset == 'SAMM') and cl == 'micro':
                shortest_len = 20 # 47
                longest_len = 105
            elif (dataset == 'SAMM_MEGC' or dataset == 'SAMM') and cl == 'macro':
                shortest_len = 106
                longest_len = -1

            folder = './../datasets/BCNN/eval_result'
            if not os.path.exists(folder):
                os.makedirs(folder)
            # 存放预测结果
            result_predict_file = '{}/{}_{}_result.txt'.format(folder, dataset, cl)
            data_root = './../datasets/BCNN/Predict_clip/{}_{}_Predict_clip/'.format(dataset, cl)
            videos_path = [os.path.join(data_root, predict_path) for predict_path in os.listdir(data_root)]
            data_label = pd.read_csv('./../datasets/' + dataset + '.csv', converters={u'subject': str, u'clip': str
                                                                                      })
            
            f = open(result_predict_file, 'w')
            pred_clip, true_clip, re_videos = hp.return_preds_labels(videos_path, dataset, cl, data_label, shortest_len, longest_len)
            A_clips, M_clips, N_clips, _, _, recall, precision, F1_score = hp.pred_clips_analysis(pred_clip, true_clip, thresh=0.5)
            # f.write('number of videos: {}\n'.format(len(true_clip)))
            # f.write('A = {} \nM = {} \nN = {} \nrecall = {:0.4} \nprecision = {:0.6} \nF1_score = {:0.6}\n\n'.format(A_clips, M_clips, N_clips, recall, precision, F1_score))
            # f.close()

            

            # 存放预测接
            result_log_file = '{}/{}_{}_result.csv'.format(folder, dataset, cl)
            data_result = hp.get_result(cl, pred_clip, true_clip, re_videos, thresh=0.5)
            # print(dict(data_result['Result'].value_counts()))
            print('Datasets: {}, expression: {}'.format(dataset, cl))
            print('TP: {} FP: {} FN: {} precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}\n'.format(A_clips, N_clips - A_clips, M_clips - A_clips, precision, recall, F1_score))
            A_clips_all += A_clips
            M_clips_all += M_clips
            N_clips_all += N_clips

            data_result.to_csv(result_log_file, index=False)
            if dataset == 'SAMM_MEGC':
                df_s = df_s.append(data_result, ignore_index=True)
            else:
                df_m = df_m.append(data_result, ignore_index=True)
            df = df.append(data_result, ignore_index=True)
            
        recall, precision, F1_score = hp.recall_precision_f1(A_clips_all, M_clips_all, N_clips_all)
        print('Datasets: {}'.format(dataset))
        print('TP: {} FP: {} FN: {} precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}\n'.format(A_clips_all, N_clips_all - A_clips_all, M_clips_all - A_clips_all, precision, recall, F1_score))
        print('{} Finish.'.format(dataset))
    
result_save_file = '{}/result.csv'.format(folder)
order = ['Video_ID', 'GT_onset', 'GT_offset', 'Predicted_onset', 'Predicted_offset', 'Result']
df = df[order]
df.to_csv(result_save_file, header=None, index=False)
print('Finish.')
