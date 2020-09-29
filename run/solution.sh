
export RANDOM_STATE='46'

'''
- create folds
  - `create_folds.py`
'''
export INPUT_DIR='../input/public_dat'
export FOLD_DIR='../folds'
export FOLD_NAME='1month_5fold'
export FOLD_NUM='5'
python ../src/create_folds.py

'''
- Feature extraction
  - `feature_extraction.py`
  - extract features from original data for training the model
'''
export FEATURE_DIR='../features'
python ../src/feature_extraction.py


'''
- Training the models
  - `train_5fold.py`
  - hyperparameters and evaluation scores of each model are saved in mlflow
  - refer to `config.py` to set the features and hyperparameters to be used to train the model
'''
export DEVICE='cuda:0'
export EXP_DIR='../exp'

export EXP_NAME='MLP_5fold_v4'
python ../src/train_5fold.py

export EXP_NAME='MLP_5fold_v9'
python ../src/train_5fold.py

export EXP_NAME='MLP_5fold_v5'
python ../src/train_5fold.py

export EXP_NAME='MLP_5fold_v7'
python ../src/train_5fold.py

export EXP_NAME='MLP_5fold_v6'
python ../src/train_5fold.py

export EXP_NAME='MLP_5fold_v8'
python ../src/train_5fold.py

export EXP_NAME='MLP_5fold_v10'
python ../src/train_5fold.py


'''
- Ensemble the trained models
  - `ensemble.py`
  - load the model by referring to run_id of mlflow, and outputs submission files
'''
export WEIFGHT_DIR='../save/model'
python ensemble.py