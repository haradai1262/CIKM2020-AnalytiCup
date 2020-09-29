import time
import random
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from contextlib import contextmanager
import torch
import logging


def read_data(file_path, feature_names):
    rows = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f.readlines())):
            line = line.strip()
            line = line.split('\t')
            rows.append(line)
    df = pd.DataFrame(rows, columns=feature_names)
    return df


def read_data_dbg(file_path, feature_names, nrows=None):
    rows = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f.readlines())):
            if i >= nrows:
                break
            line = line.strip()
            line = line.split('\t')
            rows.append(line)
    df = pd.DataFrame(rows, columns=feature_names)
    return df


def label_scaling(val):
    val = np.log(val + 1)
    scaler = MinMaxScaler()
    scaler.fit(val)
    val = scaler.transform(val)
    return scaler, val


def label_inverse_scaling(scaler, val):
    val = scaler.inverse_transform(val)
    val = np.exp(val) - 1
    return val


def seed_everything(seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for i, col in enumerate(df.columns):
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
        except ValueError:
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


class Timer:
    def __init__(self):
        self.processing_time = 0

    @contextmanager
    def timer(self, name):
        logging.info(f'[{name}] start')
        t0 = time.time()
        yield
        t1 = time.time()
        processing_time = t1 - t0
        self.processing_time += round(processing_time, 2)
        if self.processing_time < 60:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time:.2f} sec)')
        elif self.processing_time < 3600:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 60:.2f} min)')
        else:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 3600:.2f} hour)')

    def get_processing_time(self):
        return round(self.processing_time, 2)
