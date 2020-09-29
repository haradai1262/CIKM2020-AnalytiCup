IS_DEBUG = False
DEVICE = 'cuda:0'

INPUT_DIR = '../input/public_dat'
FEATURE_DIR = '../features'
FOLD_DIR = '../folds'
SAVE_DIR = '../save'
SUB_DIR = '../submission'

FOLD_NAME = '1month_5fold'
FOLD_NUM = 5
FOLD_IDX = 0
RANDOM_STATE = 46

MODEL_NAME = 'MLP'

EPOCH_NUM = 16
BATCH_SIZE = 512
DNN_HIDDEN_UNITS = (4096, 1024, 128)  # (2048, 512, 128)  # (4096, 1024, 128)
DNN_DROPOUT = 0.5
DNN_ACTIVATION = 'relu'
L2_REG = 1e-4
INIT_STD = 1e-4

SPAESE_EMBEDDING_DIM = 40
SPARSE_EMBEDDING = False
VARLEN_MAX_LEN = 10
VARLEN_MODE_LIST = ['mean']

LR = 0.001
OPTIMIZER = 'adam'
LOSS = 'MSE'  # 'MSE', 'MAE'

LABEL_LOG_SCALING = True

tweet_metrics_features = [
    '#followers', '#friends', '#favorites',
    '#followers__#favorites', '#friends__#favorites', '#followers__#friends__#favorites',
]
tweet_metrics_log_features = [f'{feat}_log' for feat in tweet_metrics_features]
tweet_metrics_cdf_features = [f'{feat}_cdf' for feat in tweet_metrics_features]
tweet_metrics_z_features = [f'{feat}_z' for feat in tweet_metrics_features]
tweet_metrics_rank_features = [f'{feat}_rank' for feat in tweet_metrics_features]

time_cat_features = ['weekday', 'hour', 'day', 'week_of_month']
time_num_features = ['diff_from_latest']

count_encording_source = [
    'sentiment_pos', 'sentiment_neg',
    'weekday', 'hour', 'day', 'week_of_month'
]
count_encording_features = [f'{feat}_ce' for feat in count_encording_source]

target_encording_source = [
    'username',
    'sentiment_pos', 'sentiment_neg',
    'weekday', 'hour', 'day', 'week_of_month',
    '#followers_qbin_10', '#friends_qbin_10', '#favorites_qbin_10', '#followers__#favorites_qbin_10'
]
target_encording_features = [f'{feat}_te' for feat in target_encording_source]

n_gram = 1
dr_name = "svd"
dr_dim = 5
text_tfidf_features = [f'TFIDF_{dr_name}_{i}' for i in range(dr_dim)]

nbin = 10
tweet_metrics_bin_features = [f'{feat}_qbin_{nbin}' for feat in tweet_metrics_features]

varlen_ori_features = ['entities', 'mentions', 'hashtags', 'urls']
varlen_count_features = [f'{feat}_count' for feat in varlen_ori_features]
url_factorized_features = ['url_http', 'url_domain', 'url_service_domain']

varlen_count_encording_source = [
    'entities', 'mentions', 'hashtags', 'urls', 'url_http', 'url_domain', 'url_service_domain'
]
varlen_count_encording_features = [f'{feat}_ce' for feat in varlen_count_encording_source]
varlen_target_encording_source = [
    'entities', 'mentions', 'hashtags', 'urls', 'url_http', 'url_domain', 'url_service_domain'
]
varlen_target_encording_features = [f'{feat}_te' for feat in varlen_target_encording_source]

user_features = ['username']
sentiment_features = [
    'sentiment_pos',
    'sentiment_neg'
]
user_stats_features = [
    'inday_fol_increase', 'prevday_fol_increase', 'inweek_fol_increase', 'prevweek_fol_increase',
    'inday_fri_increase', 'prevday_fri_increase', 'inweek_fri_increase', 'prevweek_fri_increase',
    'user_follow_mean', 'user_follow_std',
    'user_friend_mean', 'user_friend_std',
    'user_favorite_mean', 'user_favorite_std',
    'user_entities_unique', 'user_mentions_unique', 'user_hashtags_unique', 'user_urls_unique'
]

USER_CLUSTER_NUM = 1000
user_clustering_features = [
    f'user_stats_cluster_{USER_CLUSTER_NUM}',
    f'user_topic_cluster_{USER_CLUSTER_NUM}',
    f'user_stats_topic_cluster_{USER_CLUSTER_NUM}'
]

################################################################
dense_features = []
dense_features += tweet_metrics_features
dense_features += tweet_metrics_log_features
dense_features += tweet_metrics_cdf_features
dense_features += tweet_metrics_z_features
dense_features += tweet_metrics_rank_features
dense_features += time_num_features
dense_features += count_encording_features
dense_features += target_encording_features
dense_features += text_tfidf_features
dense_features += varlen_count_encording_features
dense_features += varlen_target_encording_features
dense_features += user_stats_features

sparse_features = []
sparse_features += user_features
sparse_features += sentiment_features
sparse_features += tweet_metrics_bin_features
sparse_features += time_cat_features
sparse_features += varlen_count_features
sparse_features += user_clustering_features

varlen_sparse_features = []
varlen_sparse_features += varlen_ori_features
varlen_sparse_features += url_factorized_features
################################################################