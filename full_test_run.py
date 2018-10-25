#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

datasets = [
    'check_1_r', 'check_2_r','check_3_r', 'check_4_c', 'check_5_c', 'check_6_c',
    'check_7_c', 'check_8_c'
]
result_dir = 'res'
data_dir = 'data'

total_res = {}

for data_set_path in datasets:
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if not os.path.exists('{}/{}'.format(result_dir, data_set_path)):
        os.mkdir('{}/{}'.format(result_dir, data_set_path))

    print('### Check dataset', data_set_path)

    os.system('python train.py --mode {} --train-csv {} --model-dir {}'.format(
        'regression' if data_set_path[-1] == 'r' else 'classification',
        '{}/{}/train.csv'.format(data_dir, data_set_path),
        '{}/{}/'.format(result_dir, data_set_path)
    ))

    os.system('python predict.py --prediction-csv {} --test-csv {} --model-dir {}'.format(
        '{}/{}/pred.csv'.format(result_dir, data_set_path),
        '{}/{}/test.csv'.format(data_dir, data_set_path),
        '{}/{}/'.format(result_dir, data_set_path)
    ))

    df = pd.read_csv('{}/{}/test-target.csv'.format(data_dir, data_set_path))
    df_pred = pd.read_csv('{}/{}/pred.csv'.format(result_dir, data_set_path))
    df = pd.merge(df, df_pred, on='line_id', left_index=True)

    score = roc_auc_score(df.target.values, df.prediction.values) if data_set_path[-1] == 'c' else \
            np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    print('Score {:0.4f}'.format(score))
    total_res[data_set_path] = score

print(total_res)

