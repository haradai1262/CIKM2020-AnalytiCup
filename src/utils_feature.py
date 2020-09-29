import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
from datetime import timezone
from math import ceil
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def save_as_feather(feat, save_dir, X_train, X_valid, X_test):
    logging.info(f'[{feat}] save as feather, save_dir ... [{save_dir}]')
    X_train[[feat]].reset_index(drop=True).to_feather(f'{save_dir}/{feat}_train.feather')
    X_valid[[feat]].reset_index(drop=True).to_feather(f'{save_dir}/{feat}_valid.feather')
    X_test[[feat]].reset_index(drop=True).to_feather(f'{save_dir}/{feat}_test.feather')
    return


def url_isin_news(df):
    url_isin_news = []
    for urls in tqdm(df['urls'].values):
        isin_news = 0
        if not urls[0] == 'null':
            for url in urls:
                if 'news' in url:
                    isin_news = 1
        url_isin_news.append(isin_news)
    df['url_isin_news'] = url_isin_news
    return df


def tfidf_reduce(sentences, n_gram, n_components, dr_name):
    tfv = TfidfVectorizer(min_df=1, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, n_gram), use_idf=1, smooth_idf=1, sublinear_tf=1)
    if dr_name == "svd":
        dr_model = TruncatedSVD(n_components=n_components, random_state=1337)

    tfidf_col = tfv.fit_transform(sentences)
    dr_col = dr_model.fit_transform(tfidf_col)
    dr_col = pd.DataFrame(dr_col)
    return dr_col


def extract_text_tfidf(X_train, X_valid, X_test, n_gram=1, dr_name="svd", dr_dim=5):
    X_all = pd.concat([
        X_train, X_valid, X_test
    ], axis=0)

    tweet_url_publisher = []
    for urls in tqdm(X_all['urls'].values):
        publisher = ''
        if not urls[0] == 'null':
            for url in urls:
                if 'twitter.com' in url:
                    url = url.split('/')
                    if len(url) > 3:
                        publisher = url[3]
        tweet_url_publisher.append(publisher)

    X_all['tweet_url_publisher'] = tweet_url_publisher

    texts = []
    for m, e, h, tup in X_all[['mentions', 'entities', 'hashtags', 'tweet_url_publisher']].values:
        if len(m) == 0 or m[0] == 'null':
            m = []
        if len(e) == 0 or e[0] == 'null':
            e = []
        if len(h) == 0 or h[0] == 'null':
            h = []

        # factorize entities
        e_div = []
        for ent in e:
            e_div.append(ent)
            ent_div = ent.split('_')
            if len(ent_div) > 1:
                e_div.extend(ent_div)
        text = m + e_div + h + [tup]
        texts.append(' '.join(text))

    features = tfidf_reduce(texts, n_gram=n_gram, n_components=dr_dim, dr_name=dr_name)
    features = features.add_prefix('TFIDF_{}_'.format(dr_name))

    X_train = pd.concat([
        X_train, features.iloc[:len(X_train), :].reset_index(drop=True)
    ], axis=1)
    X_valid = pd.concat([
        X_valid, features.iloc[len(X_train):len(X_train) + len(X_valid), :].reset_index(drop=True)
    ], axis=1)
    X_test = pd.concat([
        X_test, features.iloc[len(X_train) + len(X_valid):, :].reset_index(drop=True)
    ], axis=1)

    return X_train, X_valid, X_test


def factorize_url(df):
    http_list, domain_list, service_domain_list = [], [], []
    for i in df['urls'].values:
        https, domains, service_domains = [], [], []
        if i[0] == 'null':
            https.append(i[0])
            domains.append(i[0])
            service_domains.append(i[0])
        else:
            for url in i:
                url = url.split('/')
                http = url[0]
                domain = url[2].split('.')[-1]
                service_domain = url[2]
                https.append(http)
                domains.append(domain)
                service_domains.append(service_domain)
        http_list.append(https)
        domain_list.append(domains)
        service_domain_list.append(service_domains)
    df['url_http'] = http_list
    df['url_domain'] = domain_list
    df['url_service_domain'] = service_domain_list
    return df


def quantile_binning(colname, X_train, X_valid, X_test, nbin):

    df = pd.concat([
        X_train, X_valid, X_test
    ], axis=0)

    s_cut, _ = pd.qcut(df[colname], nbin, retbins=True, duplicates='drop')
    # s_cut, _ = pd.qcut(df[colname], nbin, retbins=True)
    lbe = LabelEncoder()
    lbe.fit(s_cut)
    s_cut = lbe.transform(s_cut)

    X_train[f'{colname}_qbin_{nbin}'] = s_cut[:len(X_train)]
    X_valid[f'{colname}_qbin_{nbin}'] = s_cut[len(X_train):len(X_train) + len(X_valid)]
    X_test[f'{colname}_qbin_{nbin}'] = s_cut[len(X_train) + len(X_valid):]

    return X_train, X_valid, X_test


def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom / 7.0))


def extract_time_features(df):
    timestamp = []
    for x in df['timestamp'].values:
        timestamp.append(datetime.datetime.strptime(' '.join(x.split(' ')[1:]), '%b %d %H:%M:%S %z %Y'))
    weekday = [ts.weekday() for ts in timestamp]
    hour = [ts.hour for ts in timestamp]
    day = [ts.day for ts in timestamp]
    wom = [week_of_month(ts) for ts in timestamp]

    latest = datetime.datetime(2020, 6, 1, 0, 0, 0, 0, timezone.utc)
    diff_from_latest = [(latest - ts).seconds for ts in timestamp]

    df['weekday'] = weekday
    df['hour'] = hour
    df['day'] = day
    df['week_of_month'] = wom
    df['diff_from_latest'] = diff_from_latest

    return df


def label_encording_threshold(colname, X_train, X_valid, X_test, low_freq_th):
    tmp = pd.Series()
    tmp = pd.concat([tmp, X_train[colname]])
    tmp = pd.concat([tmp, X_valid[colname]])
    tmp = pd.concat([tmp, X_test[colname]])

    count = tmp.value_counts()
    low_freq_list = count[count <= low_freq_th].index.tolist()
    count = count[count > low_freq_th]

    key2index = {i: 0 for i in low_freq_list}
    idx = 1
    for i in count.index:
        key2index[i] = idx
        idx += 1
    unique_num = idx + 1

    X_train[colname] = X_train[colname].map(key2index)
    X_valid[colname] = X_valid[colname].map(key2index)
    X_test[colname] = X_test[colname].map(key2index)

    print(f'{colname} - Num. of use as low frequency category: {len(low_freq_list)}')
    print(f'{colname} - Num. of unique category: {unique_num}')

    return X_train, X_valid, X_test, key2index, unique_num


def label_encording(colname, X_train, X_valid, X_test):
    lbe = LabelEncoder()
    tmp = pd.concat([
        X_train[colname], X_valid[colname], X_test[colname]])
    lbe.fit(tmp)
    X_train[colname] = lbe.transform(X_train[colname])
    X_valid[colname] = lbe.transform(X_valid[colname])
    X_test[colname] = lbe.transform(X_test[colname])
    return X_train, X_valid, X_test, len(lbe.classes_)


def preprocess_entities(df):
    values = [e.split(';')[:-1] for e in df['entities'].values]
    rows = []
    for x in values:
        if x[0] == 'null':
            row = ['null']
        else:
            row = []
            for e in x:
                e = e.split(':')
                row.append(e[1])
        rows.append(row)
    df['entities'] = rows
    return df


def preprocess_varlen(df, colname):
    values = df[colname].values
    rows = []
    for x in values:
        if x == "null;":
            row = ['null']
        else:
            row = []
            for m in x.split():
                row.append(m)
        rows.append(row)
    df[colname] = rows
    return df


def varlen_count(x):
    if len(x) == 1 and x[0] == 'null':
        return 0
    return len(x)


def map_varlen(x, dict):
    return [dict[i] for i in x]


def varlen_label_encording_threshold(colname, X_train, X_valid, X_test, low_freq_th):

    entities = []
    for x in X_train[colname].values:
        entities.extend(x)
    for x in X_valid[colname].values:
        entities.extend(x)
    for x in X_test[colname].values:
        entities.extend(x)
    entities = pd.Series(entities)

    count = entities.value_counts()
    low_freq_list = count[count <= low_freq_th].index.tolist()
    count = count[count > low_freq_th]

    key2index = {i: 1 for i in low_freq_list}  # null: 0, low_freq: 1, ...
    idx = 2
    for i in count.index:
        key2index[i] = idx
        idx += 1
    unique_num = idx + 1

    X_train[colname] = X_train[colname].apply(map_varlen, dict=key2index)
    X_valid[colname] = X_valid[colname].apply(map_varlen, dict=key2index)
    X_test[colname] = X_test[colname].apply(map_varlen, dict=key2index)

    print(f'{colname} - Num. of use as low frequency category: {len(low_freq_list)}')
    print(f'{colname} - Num. of unique category: {unique_num}')

    return X_train, X_valid, X_test, key2index, unique_num


def mod_sentiment(df):
    senti = df['sentiment'].values
    senti = [i.split() for i in senti]
    senti_pos = [int(i[0]) for i in senti]
    senti_neg = [int(i[1]) for i in senti]
    df['sentiment_pos'] = senti_pos
    df['sentiment_neg'] = senti_neg
    return df


def varlen_count_encording(feat, X_train, X_valid, X_test):

    X_all = pd.concat([
        X_train, X_valid, X_test
    ], axis=0)

    counts = {}
    for varlen_list in X_train[feat].values:
        if len(varlen_list) == 0:
            continue
        if varlen_list[0] == 'null':
            continue
        for x in varlen_list:
            if x not in counts:
                counts[x] = 0
            counts[x] += 1

    varlen_ce = []
    for varlen_list in X_all[feat].values:
        if len(varlen_list) == 0:
            varlen_ce.append(0)
            continue
        if varlen_list[0] == 'null':
            varlen_ce.append(0)
            continue
        val = 0
        for x in varlen_list:
            if x in counts:
                val += counts[x]
            else:
                val += 0
        varlen_ce.append(val)

    X_train[f'{feat}_ce'] = varlen_ce[:len(X_train)]
    X_valid[f'{feat}_ce'] = varlen_ce[len(X_train):len(X_train) + len(X_valid)]
    X_test[f'{feat}_ce'] = varlen_ce[len(X_train) + len(X_valid):]

    return X_train, X_valid, X_test


def varlen_target_encording(feat, X_train, X_valid, X_test, y_train, fold):

    cat_target_mean = {}

    varlen_te_train = np.zeros(len(X_train))
    for fold_idx, (trn_idx, val_idx) in enumerate(fold.split(X_train)):
        X_fold_tra = X_train.iloc[trn_idx, :]
        cat_indexes = {}
        for i, varlen_list in enumerate(X_fold_tra[feat].values):
            if len(varlen_list) == 0:
                continue
            if varlen_list[0] == 'null':
                continue
            for x in varlen_list:
                if x not in cat_indexes:
                    cat_indexes[x] = []
                cat_indexes[x].append(i)

        for x, index in cat_indexes.items():
            if x not in cat_target_mean:
                cat_target_mean[x] = {i: 0.0 for i in range(fold.n_splits)}
            cat_target_mean[x][fold_idx] = y_train[index].mean()

        for i in val_idx:
            varlen_list = X_train[feat].values[i]
            val = 0
            for x in varlen_list:
                if x in cat_target_mean:
                    val += cat_target_mean[x][fold_idx]
                else:
                    val += 0.0
            varlen_te_train[i] = val
    X_train[f'{feat}_te'] = varlen_te_train

    varlen_te = []
    for varlen_list in X_valid[feat].values:
        if len(varlen_list) == 0:
            varlen_te.append(0.0)
            continue
        if varlen_list[0] == 'null':
            varlen_te.append(0.0)
            continue
        val = 0
        for x in varlen_list:
            if x in cat_target_mean:
                for fold_idx in range(fold.n_splits):
                    val += cat_target_mean[x][fold_idx]
                val /= fold.n_splits
            else:
                val += 0.0
        varlen_te.append(val)
    X_valid[f'{feat}_te'] = varlen_te

    varlen_te = []
    for varlen_list in X_test[feat].values:
        if len(varlen_list) == 0:
            varlen_te.append(0.0)
            continue
        if varlen_list[0] == 'null':
            varlen_te.append(0.0)
            continue
        val = 0
        for x in varlen_list:
            if x in cat_target_mean:
                for fold_idx in range(fold.n_splits):
                    val += cat_target_mean[x][fold_idx]
                val /= fold.n_splits
            else:
                val += 0.0
        varlen_te.append(val)
    X_test[f'{feat}_te'] = varlen_te

    return X_train, X_valid, X_test
