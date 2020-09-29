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

from utils import (
    seed_everything, Timer,
    label_scaling, label_inverse_scaling
)
from utils_model import (
    LogCoshLoss
)
from dataset import SimpleDataLoader
from model import (
    MLP
)

logging.basicConfig(level=logging.INFO)

DEVICE = os.environ.get('DEVICE')
EXP_DIR = os.environ.get('EXP_DIR')
EXP_NAME = os.environ.get('EXP_NAME')

sys.path.append(f'{EXP_DIR}/{EXP_NAME}')
import config

IS_DEBUG = config.IS_DEBUG

INPUT_DIR = config.INPUT_DIR
FEATURE_DIR = config.FEATURE_DIR
FOLD_DIR = config.FOLD_DIR
SAVE_DIR = config.SAVE_DIR
SUB_DIR = config.SUB_DIR

FOLD_NAME = config.FOLD_NAME
FOLD_NUM = config.FOLD_NUM
RANDOM_STATE = config.RANDOM_STATE

MODEL_NAME = config.MODEL_NAME

EPOCH_NUM = config.EPOCH_NUM
BATCH_SIZE = config.BATCH_SIZE
DNN_HIDDEN_UNITS = config.DNN_HIDDEN_UNITS
DNN_DROPOUT = config.DNN_DROPOUT
DNN_ACTIVATION = config.DNN_ACTIVATION
L2_REG = config.L2_REG
INIT_STD = config.INIT_STD

SPAESE_EMBEDDING_DIM = config.SPAESE_EMBEDDING_DIM
SPARSE_EMBEDDING = config.SPARSE_EMBEDDING
VARLEN_MAX_LEN = config.VARLEN_MAX_LEN
VARLEN_MODE_LIST = config.VARLEN_MODE_LIST

LR = config.LR
OPTIMIZER = config.OPTIMIZER
LOSS = config.LOSS

LABEL_LOG_SCALING = config.LABEL_LOG_SCALING

dense_features = config.dense_features
sparse_features = config.sparse_features
varlen_sparse_features = config.varlen_sparse_features


def load_model(
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


def set_loss(loss_name):
    if type(loss_name) == str:
        if loss_name == 'MSE':
            loss_func = nn.MSELoss(reduction='mean')
        elif loss_name == 'MAE':
            loss_func = nn.L1Loss(reduction='mean')
        elif loss_name == 'Huber':
            loss_func = nn.SmoothL1Loss(reduction='mean')
        elif loss_name == 'LogCosh':
            loss_func = LogCoshLoss()
        loss_func_list = [loss_func]
    elif type(loss_name) == list:
        loss_func_list = []
        for ln in loss_name:
            if ln == 'MSE':
                loss_func = nn.MSELoss(reduction='mean')
            elif ln == 'MAE':
                loss_func = nn.L1Loss(reduction='mean')
            elif ln == 'Huber':
                loss_func = nn.SmoothL1Loss(reduction='mean')
            elif ln == 'LogCosh':
                loss_func = LogCoshLoss()
            loss_func_list.append(loss_func)
    return loss_func_list


def make_submission(validation_name, X, cv, run_id):

    X['preds'] = X['preds'].round().astype('int')
    sub_name = f'sub_{validation_name}_1fold_{cv}_{run_id}'
    valid_sub_dir = f'{SUB_DIR}/{sub_name}'
    if not os.path.exists(valid_sub_dir):
        os.mkdir(valid_sub_dir)

    X['preds'].to_csv(f'{valid_sub_dir}/{validation_name}.predict', index=False, header=None)
    with zipfile.ZipFile(f'{SUB_DIR}/{sub_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        new_zip.write(f'{valid_sub_dir}/{validation_name}.predict', arcname=f'{validation_name}.predict')
    shutil.rmtree(valid_sub_dir)
    mlflow.log_artifact(f'{SUB_DIR}/{sub_name}.zip')
    return


def save_mlflow(run_id, cv, fold_best_scores):

    mlflow.log_param("model", MODEL_NAME)

    mlflow.log_param("fold_name", FOLD_NAME)
    mlflow.log_param("fold_num", FOLD_NUM)

    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("loss", LOSS)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("learning rate", LR)
    mlflow.log_param("random_state", RANDOM_STATE)

    mlflow.log_param("dnn_hidden_layer", DNN_HIDDEN_UNITS)
    mlflow.log_param("dnn_dropout", DNN_DROPOUT)
    mlflow.log_param("dnn_activation", DNN_ACTIVATION)
    mlflow.log_param("l2_reg", L2_REG)
    mlflow.log_param("init_std", INIT_STD)

    mlflow.log_param("embedding_dim", SPAESE_EMBEDDING_DIM)
    mlflow.log_param("varlen_max_len", VARLEN_MAX_LEN)
    mlflow.log_param("varlen_mode_list", VARLEN_MODE_LIST)

    for feat in dense_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__dense__{feat}', 1)
    for feat in sparse_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__sparse__{feat}', 1)
    for feat in varlen_sparse_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__varspa__{feat}', 1)

    mlflow.log_metric("cv", cv)
    for fold_idx in range(FOLD_NUM):
        mlflow.log_metric(f'val_metric_{fold_idx}', fold_best_scores[fold_idx][0])

    return


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

    logging.info('SPARSE FEATURE UNIQUE_NUM')
    print(unique_num_dic)

    with t.timer(f'READ folds'):
        folds = pd.read_csv(f'{FOLD_DIR}/train_folds_{FOLD_NAME}{FOLD_NUM}_RS{RANDOM_STATE}.csv')

    mlflow.set_experiment(EXP_NAME)
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id

    fold_best_scores = {}  # fold_idx:best_cv_score
    for fold_idx in range(FOLD_NUM):

        trn_idx = folds[folds.kfold != fold_idx].index.tolist()
        val_idx = folds[folds.kfold == fold_idx].index.tolist()

        x_trn = X_train.iloc[trn_idx]
        y_trn = y_train[trn_idx]
        x_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]

        train_loader = SimpleDataLoader(
            [torch.from_numpy(x_trn.values), torch.from_numpy(y_trn)],
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        model = load_model(
            feature_index=feature_index,
            unique_num_dic=unique_num_dic,
        )
        loss_func = set_loss(loss_name=LOSS)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCH_NUM)

        loss_history = []

        steps_per_epoch = (len(x_trn) - 1) // BATCH_SIZE + 1
        best_score = 999.9
        for epoch in range(EPOCH_NUM):

            loss_history_epoch = []
            metric_history_epoch = []

            logging.info(f'[{DEVICE}][FOLD:{fold_idx}] EPOCH - {epoch} / {EPOCH_NUM}')
            model = model.train()
            for bi, (bx, by) in tqdm(enumerate(train_loader), total=steps_per_epoch):

                optim.zero_grad()

                bx = bx.to(DEVICE).float()
                by = by.to(DEVICE).float().squeeze()
                y_pred = model(bx).squeeze()

                loss = 0.0
                for loss_f in loss_func:
                    loss += loss_f(y_pred, by)
                loss = loss + model.reg_loss.item()

                loss.backward()
                optim.step()

                y_pred_np = y_pred.cpu().detach().numpy().reshape(-1, 1)
                y_np = by.cpu().detach().numpy().reshape(-1, 1)

                try:
                    if LABEL_LOG_SCALING is True:
                        y_pred_inv = label_inverse_scaling(scaler, y_pred_np)
                        y_inv = label_inverse_scaling(scaler, y_np)
                        mlse = mean_squared_log_error(y_inv, y_pred_inv)
                    else:
                        mlse = mean_squared_log_error(y_np, y_pred_np)
                    loss_history_epoch.append(loss.item())
                    metric_history_epoch.append(mlse)
                except:
                    continue

            scheduler.step()
            trn_loss_epoch = sum(loss_history_epoch) / len(loss_history_epoch)
            trn_metric_epoch = sum(metric_history_epoch) / len(metric_history_epoch)

            preds_val = model.predict(x_val, BATCH_SIZE)
            val_loss = 0.0
            for loss_f in loss_func:
                val_loss += loss_f(torch.from_numpy(preds_val.reshape(-1, 1)), torch.from_numpy(y_val)).item()

            try:
                if LABEL_LOG_SCALING is True:
                    preds_val_inv = label_inverse_scaling(scaler, preds_val.reshape(-1, 1))
                    y_val_inv = label_inverse_scaling(scaler, y_val)
                    val_metric = mean_squared_log_error(y_val_inv, preds_val_inv)
                else:
                    val_metric = mean_squared_log_error(y_val, preds_val)
            except:
                continue

            logging.info(f'Train - Loss: {trn_loss_epoch}, MSLE: {trn_metric_epoch}')
            logging.info(f'Valid - Loss: {val_loss}, MSLE: {val_metric}')
            loss_history.append([
                epoch, trn_loss_epoch, trn_metric_epoch, val_loss, val_metric
            ])

            if val_metric < best_score:
                best_score = val_metric
                weight_path = f'{SAVE_DIR}/model/train_weights_mlflow-{run_id}_fold{fold_idx}.h5'
                torch.save(model.state_dict(), weight_path)
                fold_best_scores[fold_idx] = (best_score, weight_path)
                mlflow.log_artifact(weight_path)

        history_path = f'{SAVE_DIR}/model/loss_history-{run_id}_fold{fold_idx}.csv'
        pd.DataFrame(loss_history, columns=['epoch', 'trn_loss', 'trn_metric', 'val_loss', 'val_metric']).to_csv(history_path)
        mlflow.log_artifact(history_path)

    cv = 0.0
    for fold_idx in range(FOLD_NUM):
        cv += fold_best_scores[fold_idx][0]
    cv /= FOLD_NUM

    preds_train_val = np.zeros(len(X_train))
    for fold_idx in range(FOLD_NUM):

        val_idx = folds[folds.kfold == fold_idx].index.tolist()
        x_val = X_train.iloc[val_idx]

        model = load_model(
            feature_index=feature_index,
            unique_num_dic=unique_num_dic,
        )
        weight_path = fold_best_scores[fold_idx][1]
        model.load_state_dict(torch.load(weight_path))

        preds_train_val_fold = model.predict(x_val, BATCH_SIZE)
        preds_train_val[val_idx] = preds_train_val_fold

        preds_valid = model.predict(X_valid, BATCH_SIZE)
        X_valid[f'preds_{fold_idx}'] = preds_valid

        preds_test = model.predict(X_test, BATCH_SIZE)
        X_test[f'preds_{fold_idx}'] = preds_test

    X_train['preds'] = preds_train_val
    X_valid['preds'] = X_valid[[f'preds_{fold_idx}' for fold_idx in range(FOLD_NUM)]].mean()
    X_test['preds'] = X_test[[f'preds_{fold_idx}' for fold_idx in range(FOLD_NUM)]].mean()

    save_path = f'{SAVE_DIR}/predict/preds_train_val_{run_id}.csv'
    X_train['preds'].to_csv(save_path, index=False, header=None)
    mlflow.log_artifact(save_path)

    save_path = f'{SAVE_DIR}/predict/preds_valid_{run_id}.csv'
    X_valid['preds'].to_csv(save_path, index=False, header=None)
    mlflow.log_artifact(save_path)

    save_path = f'{SAVE_DIR}/predict/preds_test_{run_id}.csv'
    X_test['preds'].to_csv(save_path, index=False, header=None)
    mlflow.log_artifact(save_path)

    save_mlflow(run_id, cv, fold_best_scores)
    mlflow.end_run()


if __name__ == "__main__":

    main()
