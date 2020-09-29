import os
import logging
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from utils import (
    seed_everything, Timer,
    read_data
)

INPUT_DIR = os.environ.get('INPUT_DIR')
FOLD_DIR = os.environ.get('FOLD_DIR')

RANDOM_STATE = int(os.environ.get('RANDOM_STATE'))
FOLD_NAME = os.environ.get('FOLD_NAME')
FOLD_NUM = int(os.environ.get('FOLD_NUM'))

if __name__ == "__main__":

    t = Timer()
    with t.timer(f'FIX seed RANDOM_STATE:{RANDOM_STATE}'):
        seed_everything(RANDOM_STATE)

    with t.timer(f'READ data'):
        feature_names = pd.read_csv(f'{INPUT_DIR}/feature.name', sep='\t').columns.tolist()
        X = read_data(f'{INPUT_DIR}/train.data', feature_names)

    with t.timer(f'CREATE folds'):
        logging.info(f'FOLD_NAME: {FOLD_NAME}')
        logging.info(f'FOLD_NUM: {FOLD_NUM}')

        if FOLD_NAME == '1month_5fold':
            timestamp = []
            for x in X['timestamp'].values:
                timestamp.append(datetime.datetime.strptime(' '.join(x.split(' ')[1:]), '%b %d %H:%M:%S %z %Y'))
            X['ts_dt'] = timestamp
            val = X[['ts_dt']][X.ts_dt > "2020-04-01"].index
            kf = KFold(n_splits=FOLD_NUM, random_state=RANDOM_STATE, shuffle=True)
            splits = kf.split(val)
            X['kfold'] = -1
            for fold, (ftrn, fval) in enumerate(splits):
                X.loc[fval, 'kfold'] = fold
        elif FOLD_NAME == '2month_5fold':
            timestamp = []
            for x in X['timestamp'].values:
                timestamp.append(datetime.datetime.strptime(' '.join(x.split(' ')[1:]), '%b %d %H:%M:%S %z %Y'))
            X['ts_dt'] = timestamp
            val = X[['ts_dt']][X.ts_dt > "2020-03-01"].index
            kf = KFold(n_splits=FOLD_NUM, random_state=RANDOM_STATE, shuffle=True)
            splits = kf.split(val)
            X['kfold'] = -1
            for fold, (ftrn, fval) in enumerate(splits):
                X.loc[fval, 'kfold'] = fold

    with t.timer(f'SAVE folds'):
        logging.info(X.kfold.value_counts())
        X[['kfold']].to_csv(f'{FOLD_DIR}/train_folds_{FOLD_NAME}{FOLD_NUM}_RS{RANDOM_STATE}.csv', index=False)
