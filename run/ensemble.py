import ast
import numpy as np
import pandas as pd
import os
import sys
import zipfile
import shutil
from tqdm import tqdm
import logging

import mlflow
from sklearn.metrics import mean_squared_log_error
import torch
import torch.nn as nn
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.append('../src')
from utils import (
    seed_everything, Timer,
    label_scaling, label_inverse_scaling
)
from utils_feature import save_as_feather
from utils_model import (
    LogCoshLoss
)
from dataset import SimpleDataLoader
from model import (
    MLP
)

logging.basicConfig(level=logging.DEBUG)

WEIFGHT_DIR = os.environ.get('WEIFGHT_DIR')
ENSEMBLE_MODELS = {  # EXP_NAME: mlflow_run_id
    'MLP_5fold_v4': '6e6123efb1184c109379225946b4ed73',  # MLP (dropout_rate = 0.1)
    'MLP_5fold_v9': 'a3c175ab07d54740ab860ce8a674cedd',  # MLP (dropout_rate = 0.3)
    'MLP_5fold_v5': '38b5f50161b24d17bf90f3d65119b3fa',  # MLP (dropout_rate = 0.5)
    'MLP_5fold_v7': '42a5f61a29b34c189b8c6940b4ac58b8',  # MLP large (dropout_rate = 0.1)
    'MLP_5fold_v6': '6f15836a31d5406d9e8625b13d43d102',  # MLP large (dropout_rate = 0.3)
    'MLP_5fold_v8': 'e3e58d7de9b348ac97e594a51a9efc83',  # MLP large (dropout_rate = 0.5)
    'MLP_5fold_v10': '270fe5645163490f8ad71e58eb43cf05',  # MLP large (dropout_rate = 0.1), loss func = MAE
}

DEVICE = os.environ.get('DEVICE')
EXP_DIR = os.environ.get('EXP_DIR')
EXP_NAME = os.environ.get('EXP_NAME')

sys.path.append(f'{EXP_DIR}/{EXP_NAME}')
import config

INPUT_DIR = config.INPUT_DIR
FEATURE_DIR = config.FEATURE_DIR
FOLD_DIR = config.FOLD_DIR
SAVE_DIR = config.SAVE_DIR
SUB_DIR = config.SUB_DIR

FOLD_NAME = config.FOLD_NAME
FOLD_NUM = config.FOLD_NUM
RANDOM_STATE = config.RANDOM_STATE

dense_features = config.dense_features
sparse_features = config.sparse_features
varlen_sparse_features = config.varlen_sparse_features

VARLEN_MAX_LEN = config.VARLEN_MAX_LEN
SPARSE_EMBEDDING = config.SPARSE_EMBEDDING
LABEL_LOG_SCALING = config.LABEL_LOG_SCALING


def load_model(
    MODEL_NAME, BATCH_SIZE,
    DNN_HIDDEN_UNITS, DNN_DROPOUT, DNN_ACTIVATION, L2_REG, INIT_STD,
    SPAESE_EMBEDDING_DIM, VARLEN_MODE_LIST,
    feature_index,
    unique_num_dic,
):

    embedding_dict = nn.ModuleDict(
        {
            feat: nn.Embedding(
                unique_num_dic[feat], SPAESE_EMBEDDING_DIM, sparse=SPARSE_EMBEDDING
            ) for feat in sparse_features
        }
    )
    for mode in VARLEN_MODE_LIST:
        for feat in varlen_sparse_features:
            embedding_dict[f'{feat}__{mode}'] = nn.Embedding(
                unique_num_dic[feat], SPAESE_EMBEDDING_DIM, sparse=SPARSE_EMBEDDING
            )
    linear_embedding_dict = nn.ModuleDict(
        {
            feat: nn.Embedding(
                unique_num_dic[feat], 1, sparse=SPARSE_EMBEDDING
            ) for feat in sparse_features
        }
    )
    for mode in VARLEN_MODE_LIST:
        for feat in varlen_sparse_features:
            linear_embedding_dict[f'{feat}__{mode}'] = nn.Embedding(
                unique_num_dic[feat], 1, sparse=SPARSE_EMBEDDING
            )

    if MODEL_NAME == 'MLP':
        dnn_input_len = len(dense_features) + len(sparse_features) * SPAESE_EMBEDDING_DIM \
            + len(varlen_sparse_features) * len(VARLEN_MODE_LIST) * SPAESE_EMBEDDING_DIM
        model = MLP(
            dnn_input=dnn_input_len,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_dropout=DNN_DROPOUT,
            activation=DNN_ACTIVATION, use_bn=True, l2_reg=L2_REG, init_std=INIT_STD,
            device=DEVICE,
            feature_index=feature_index,
            embedding_dict=embedding_dict,
            dense_features=dense_features,
            sparse_features=sparse_features,
            varlen_sparse_features=varlen_sparse_features,
            varlen_mode_list=VARLEN_MODE_LIST,
            embedding_size=SPAESE_EMBEDDING_DIM,
            batch_size=BATCH_SIZE,
        )
    return model


def post_process(X):
    X['preds_submit'] = X['preds'].map(lambda x: float(Decimal(str(x)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))).astype('int')
    return X


def make_submission(validation_name, X, sub_name):

    sub_name = f'sub_{validation_name}_5fold_{sub_name}'
    valid_sub_dir = f'{SUB_DIR}/{sub_name}'
    if not os.path.exists(valid_sub_dir):
        os.mkdir(valid_sub_dir)

    X['preds_submit'].to_csv(f'{valid_sub_dir}/{validation_name}.predict', index=False, header=None)
    with zipfile.ZipFile(f'{SUB_DIR}/{sub_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        new_zip.write(f'{valid_sub_dir}/{validation_name}.predict', arcname=f'{validation_name}.predict')
    shutil.rmtree(valid_sub_dir)
    mlflow.log_artifact(f'{SUB_DIR}/{sub_name}.zip')
    return


def predict_cv(model, model_name, train_x, train_y, val_x, test_x, n_splits=5):

    preds = []
    preds_test = []
    preds_val = []
    va_idxes = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for i, (tr_idx, va_idx) in tqdm(enumerate(kf.split(train_x)), total=n_splits):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y[tr_idx], train_y[va_idx]

        model.fit(tr_x, tr_y)

        pred = model.predict(va_x)
        preds.append(pred)
        pred_val = model.predict(val_x)
        preds_val.append(pred_val)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)

        va_idxes.append(va_idx)

    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    preds_val = np.mean(preds_val, axis=0)
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_val, preds_test


def main():

    t = Timer()
    with t.timer(f'fix seed RANDOM_STATE:{RANDOM_STATE}'):
        seed_everything(RANDOM_STATE)

    with t.timer(f'read label'):
        y_train = pd.read_csv(f'{INPUT_DIR}/train.solution', header=None).T.values[0].reshape(-1, 1)

    if LABEL_LOG_SCALING is True:
        with t.timer(f'label log scaling (log->mms[0, 1]'):
            scaler, y_train = label_scaling(y_train)

    with t.timer(f'read features'):
        unique_num_dic = {}
        feature_index = {}

        X_train = pd.DataFrame()
        X_valid = pd.DataFrame()
        X_test = pd.DataFrame()
        fidx = 0
        for feat in dense_features:
            logging.info(f'[dense][{feat}] read feature ...')
            feature_index[feat] = fidx
            fidx += 1
            X_train = pd.concat([
                X_train, pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')
            ], axis=1)
            X_valid = pd.concat([
                X_valid, pd.read_feather(f'{FEATURE_DIR}/{feat}_valid.feather')
            ], axis=1)
            X_test = pd.concat([
                X_test, pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')
            ], axis=1)
        for feat in sparse_features:
            logging.info(f'[sparse][{feat}] read feature ...')
            feature_index[feat] = fidx
            fidx += 1
            X_train = pd.concat([
                X_train, pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')
            ], axis=1)
            X_valid = pd.concat([
                X_valid, pd.read_feather(f'{FEATURE_DIR}/{feat}_valid.feather')
            ], axis=1)
            X_test = pd.concat([
                X_test, pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')
            ], axis=1)
            unique_num = pd.concat([
                X_train[feat], X_valid[feat], X_test[feat]
            ]).nunique()
            unique_num_dic[feat] = unique_num
        for feat in varlen_sparse_features:
            logging.info(f'[varlen sparse][{feat}] read feature ...')
            feature_index[feat] = (fidx, fidx + VARLEN_MAX_LEN)
            fidx += VARLEN_MAX_LEN

            train_feat = pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather').values
            varlen_list = [i[0] for i in train_feat]
            varlen_list = pad_sequences(varlen_list, maxlen=VARLEN_MAX_LEN, padding='post', )
            X_train = pd.concat([
                X_train, pd.DataFrame(varlen_list)
            ], axis=1)

            valid_feat = pd.read_feather(f'{FEATURE_DIR}/{feat}_valid.feather').values
            varlen_list = [i[0] for i in valid_feat]
            varlen_list = pad_sequences(varlen_list, maxlen=VARLEN_MAX_LEN, padding='post', )
            X_valid = pd.concat([
                X_valid, pd.DataFrame(varlen_list)
            ], axis=1)

            test_feat = pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather').values
            varlen_list = [i[0] for i in test_feat]
            varlen_list = pad_sequences(varlen_list, maxlen=VARLEN_MAX_LEN, padding='post', )
            X_test = pd.concat([
                X_test, pd.DataFrame(varlen_list)
            ], axis=1)

            tmp = []
            for i in [i[0] for i in train_feat] + [i[0] for i in valid_feat] + [i[0] for i in test_feat]:
                tmp.extend(i)
            unique_num = len(set(tmp))
            unique_num_dic[feat] = unique_num
        X_train = X_train.fillna(0.0)
        X_valid = X_valid.fillna(0.0)
        X_test = X_test.fillna(0.0)

    with t.timer(f'READ folds'):
        folds = pd.read_csv(f'{FOLD_DIR}/train_folds_{FOLD_NAME}{FOLD_NUM}_RS{RANDOM_STATE}.csv')

    with t.timer(f'inference by saved models'):
        preds_train_val_models = pd.DataFrame()
        preds_valid_models = pd.DataFrame()
        preds_test_models = pd.DataFrame()

        for run_idx, (EXP_NAME, run_id) in enumerate(ENSEMBLE_MODELS.items()):

            mlflow.set_experiment(EXP_NAME)
            run = mlflow.get_run(run_id)

            MODEL_NAME = run.data.params['model']

            BATCH_SIZE = int(run.data.params['batch_size'])
            DNN_HIDDEN_UNITS = [int(i) for i in ast.literal_eval(run.data.params['dnn_hidden_layer'])]
            DNN_DROPOUT = float(run.data.params['dnn_dropout'])
            DNN_ACTIVATION = run.data.params['dnn_activation']
            L2_REG = float(run.data.params['l2_reg'])
            INIT_STD = float(run.data.params['init_std'])

            SPAESE_EMBEDDING_DIM = int(run.data.params['embedding_dim'])
            VARLEN_MODE_LIST = ast.literal_eval(run.data.params['varlen_mode_list'])

            preds_train_val = np.zeros(len(folds[folds.kfold != -1]))
            preds_valid = pd.DataFrame()
            preds_test = pd.DataFrame()
            for fold_idx in range(FOLD_NUM):

                val_idx = folds[folds.kfold == fold_idx].index.tolist()
                x_val = X_train.iloc[val_idx]
                y_val = y_train[val_idx]

                model = load_model(
                    MODEL_NAME, BATCH_SIZE,
                    DNN_HIDDEN_UNITS, DNN_DROPOUT, DNN_ACTIVATION, L2_REG, INIT_STD,
                    SPAESE_EMBEDDING_DIM, VARLEN_MODE_LIST,
                    feature_index=feature_index,
                    unique_num_dic=unique_num_dic,
                )
                weight_path = f'{WEIFGHT_DIR}/train_weights_mlflow-{run_id}_fold{fold_idx}.h5'
                model.load_state_dict(torch.load(weight_path))

                preds_train_val_fold = model.predict(x_val, BATCH_SIZE)
                preds_train_val[val_idx] = preds_train_val_fold

                preds_val_inv = label_inverse_scaling(scaler, preds_train_val_fold.reshape(-1, 1))
                y_val_inv = label_inverse_scaling(scaler, y_val)
                val_metric = mean_squared_log_error(y_val_inv, preds_val_inv)

                logging.info(f'{EXP_NAME}, FOLD{fold_idx}, MSLE:{val_metric}')

                preds_valid_fold = model.predict(X_valid, BATCH_SIZE)
                preds_valid[f'preds_{fold_idx}'] = preds_valid_fold

                preds_test_fold = model.predict(X_test, BATCH_SIZE)
                preds_test[f'preds_{fold_idx}'] = preds_test_fold

            val_idx = folds[folds.kfold != -1].index.tolist()

            preds_train_val_inv = label_inverse_scaling(scaler, preds_train_val.reshape(-1, 1))
            y_train_inv = label_inverse_scaling(scaler, y_train[val_idx])
            cv = mean_squared_log_error(y_train_inv, preds_train_val_inv)

            logging.info(f'RUN_ID:{run_id}, CV-MSLE:{cv}')

            preds_valid_mean = preds_valid[[f'preds_{fold_idx}' for fold_idx in range(FOLD_NUM)]].mean(axis=1).values
            preds_valid_mean_scaled = label_inverse_scaling(scaler, preds_valid_mean.reshape(-1, 1))
            preds_valid['preds_log'] = preds_valid_mean
            preds_valid['preds'] = preds_valid_mean_scaled

            preds_test_mean = preds_test[[f'preds_{fold_idx}' for fold_idx in range(FOLD_NUM)]].mean(axis=1).values
            preds_test_mean_scaled = label_inverse_scaling(scaler, preds_test_mean.reshape(-1, 1))
            preds_test['preds_log'] = preds_test_mean
            preds_test['preds'] = preds_test_mean_scaled

            preds_train_val_models[f'preds_{EXP_NAME}'] = preds_train_val_inv.reshape(-1)
            preds_valid_models[f'preds_{EXP_NAME}'] = preds_valid['preds']
            preds_test_models[f'preds_{EXP_NAME}'] = preds_test['preds']

            preds_train_val_models[f'preds_log_{EXP_NAME}'] = preds_train_val
            preds_valid_models[f'preds_log_{EXP_NAME}'] = preds_valid['preds_log']
            preds_test_models[f'preds_log_{EXP_NAME}'] = preds_test['preds_log']

        for run_idx, (EXP_NAME, run_id) in enumerate(ENSEMBLE_MODELS.items()):
            save_as_feather(
                f'preds_log_{EXP_NAME}', f'{SAVE_DIR}/model_predict',
                preds_train_val_models,
                preds_valid_models,
                preds_test_models
            )

    with t.timer(f'make each submission'):
        for model_preds_log_col in [f'preds_log_{EXP_NAME}' for EXP_NAME in ENSEMBLE_MODELS.keys()]:
            preds_log_valid_models = preds_valid_models[[model_preds_log_col]]
            preds_valid_models_mean = preds_log_valid_models.mean(axis=1).values
            preds_valid_models_mean_scaled = label_inverse_scaling(scaler, preds_valid_models_mean.reshape(-1, 1))
            preds_log_valid_models['preds_log'] = preds_valid_models_mean
            preds_log_valid_models['preds'] = preds_valid_models_mean_scaled
            preds_log_valid_models = post_process(preds_log_valid_models)

            sub_name = model_preds_log_col
            make_submission('validation', preds_log_valid_models, sub_name)

            preds_log_test_models = preds_test_models[[model_preds_log_col]]
            preds_test_models_mean = preds_log_test_models.mean(axis=1).values
            preds_test_models_mean_scaled = label_inverse_scaling(scaler, preds_test_models_mean.reshape(-1, 1))
            preds_log_test_models['preds_log'] = preds_test_models_mean
            preds_log_test_models['preds'] = preds_test_models_mean_scaled
            preds_log_test_models = post_process(preds_log_test_models)

            sub_name = model_preds_log_col
            make_submission('test', preds_log_test_models, sub_name)

    with t.timer(f'blend mean'):
        model_preds_log_col = [f'preds_log_{EXP_NAME}' for EXP_NAME in ENSEMBLE_MODELS.keys()]
        preds_log_valid_models = preds_valid_models[model_preds_log_col]
        preds_valid_models_mean = preds_log_valid_models.mean(axis=1).values
        preds_valid_models_mean_scaled = label_inverse_scaling(scaler, preds_valid_models_mean.reshape(-1, 1))
        preds_log_valid_models['preds_log'] = preds_valid_models_mean
        preds_log_valid_models['preds'] = preds_valid_models_mean_scaled
        preds_log_valid_models = post_process(preds_log_valid_models)

        sub_name = 'Blend__' + '__'.join([f'{EXP_NAME}' for EXP_NAME in ENSEMBLE_MODELS.keys()])
        make_submission('validation', preds_log_valid_models, sub_name)

        model_preds_log_col = [f'preds_log_{EXP_NAME}' for EXP_NAME in ENSEMBLE_MODELS.keys()]
        preds_log_test_models = preds_test_models[model_preds_log_col]
        preds_test_models_mean = preds_log_test_models.mean(axis=1).values
        preds_test_models_mean_scaled = label_inverse_scaling(scaler, preds_test_models_mean.reshape(-1, 1))
        preds_log_test_models['preds_log'] = preds_test_models_mean
        preds_log_test_models['preds'] = preds_test_models_mean_scaled
        preds_log_test_models = post_process(preds_log_test_models)

        sub_name = 'Blend__' + '__'.join([f'{EXP_NAME}' for EXP_NAME in ENSEMBLE_MODELS.keys()])
        make_submission('test', preds_log_test_models, sub_name)

    with t.timer(f'stacking (Ridge)'):
        model_preds_log_col = [f'preds_log_{EXP_NAME}' for EXP_NAME in ENSEMBLE_MODELS.keys()]
        preds_log_train_models = preds_train_val_models[model_preds_log_col]
        preds_log_valid_models = preds_valid_models[model_preds_log_col]
        preds_log_test_models = preds_test_models[model_preds_log_col]

        train_val_idx = folds[folds.kfold != -1].index.tolist()

        smodel = Ridge(alpha=1.0)
        smodel_name = 'ridge'
        stack_train, stack_valid, stack_test = predict_cv(
            smodel, smodel_name,
            preds_log_train_models,
            y_train[train_val_idx].reshape(-1),
            preds_log_valid_models,
            preds_log_test_models,
            n_splits=5
        )

        stack_valid_scaled = label_inverse_scaling(scaler, stack_valid.reshape(-1, 1))
        preds_log_valid_models['preds'] = stack_valid_scaled
        preds_log_valid_models['preds_log'] = stack_valid
        preds_log_valid_models = post_process(preds_log_valid_models)
        sub_name = 'Ridge__' + '__'.join([f'{EXP_NAME}' for EXP_NAME in ENSEMBLE_MODELS.keys()])
        make_submission('validation', preds_log_valid_models, sub_name)

        stack_test_scaled = label_inverse_scaling(scaler, stack_test.reshape(-1, 1))
        preds_log_test_models['preds'] = stack_test_scaled
        preds_log_test_models['preds_log'] = stack_test
        preds_log_test_models = post_process(preds_log_test_models)
        sub_name = 'Ridge__' + '__'.join([f'{EXP_NAME}' for EXP_NAME in ENSEMBLE_MODELS.keys()])
        make_submission('test', preds_log_test_models, sub_name)


if __name__ == "__main__":
    main()