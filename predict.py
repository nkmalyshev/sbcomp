import argparse
import os
import pickle
import time
import numpy as np
import pandas as pd
from utils_mk2 import load_data, preprocessing
from utils_model import xgb_predict_wrapper


# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    xgb_model = model_config['model']
    cols_to_use = model_config['cols_to_use']
    col_stats = model_config['col_stats']
    freq_stats = model_config['freq_stats']

    x_test, _, line_id_test, _, _ = load_data(args.test_csv, mode='test', input_cols=np.append(cols_to_use, ['line_id']))
    x_test_proc, _, _, _ = preprocessing(x=x_test, y=0, col_stats_init=col_stats, cat_freq_init=freq_stats)
    p_xgb_test = xgb_predict_wrapper(x_test_proc, xgb_model)

    line_id_test['prediction'] = p_xgb_test
    line_id_test[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {:0.2f}'.format(time.time() - start_time))
