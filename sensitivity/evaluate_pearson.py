import os
import errno
from os import path
import pandas as pd
import numpy as np
import time
import seaborn.apionly as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation
from wordcloud import WordCloud
from scipy.stats import pearsonr

def load_data(data_path):
    df = pd.read_csv(data_path)
    print('load data shape:', df.shape)
    Y = df.Class.values
    df = df.drop(['Class'],
                         axis=1)  # drop  Class for the feature
    terms = df.columns
    X = df.values
    print('features shape:', X.shape)
    return X, Y, terms


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def print_top_words(model, terms, n_top_words):
    message = ''
    for topic_idx, topic in enumerate(model.components_):
        message += "\nTopic #%d: " % topic_idx
        message += ", ".join([terms[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)
    return message


def get_features_dict(terms, df_ph_dicts):
    features_dict = []
    for item in terms:
        if item.startswith("00"):
            item = item[2:]
        elif item.startswith("0"):
            item = item[1:]
        print('phecode:', item)
        try:
            phenotype_term = df_ph_dicts.loc[df_ph_dicts['PheCode'] == item]['Short_names'].item()
        except:
            phenotype_term = item
            pass
        features_dict.append(phenotype_term)
        print(phenotype_term)
        # if phenotype_term == 'Other tests':
        #     print(item)
        # if '&' in phenotype_term:
        #     phenotype_term = re.sub(r"&", "", phenotype_term)
    return features_dict

def topic_modeling(X, n_components, alpha=.1, l1_ratio=.5):
    nmf_f = NMF(n_components=n_components, random_state=10,
                alpha=alpha, l1_ratio=l1_ratio).fit(X)
    return nmf_f

def out_to_file(out_file, cotent):
    with open(out_file, 'w+') as f:
        f.write(cotent)
        f.write('\n\n')


n_top_words = 10
random_seed = 10

def calucate_pearson(topic_model, X, Y,k):
    X_topics = topic_model.transform(X)  # each patient's distribution or score on each topic
    columns = ['Topic_' + str(i) for i in range(0, k)]
    dtm_to_topic = pd.DataFrame(X_topics, columns=columns)  # patients distribution on each topic
    dtm_to_topic.head(10)
    message = ""

    for topic_idx in range(k):
        Cov_topic = 'Topic_' + str(topic_idx)
        X_topic = dtm_to_topic[Cov_topic].values
        message += '\nLPA SNPs and topic_{} {}'.format(str(topic_idx), pearsonr(X_topic, Y))
    return message

def run_pearson():
    data_path = '../data/data.csv'
    result_path = '../analysis_result'
    _mkdir_p(result_path)

    X, Y, terms = load_data(data_path)
    dict_path = path.join('../data', 'raw', 'JD_CODE_string.csv')
    df_ph_dicts = pd.read_csv(dict_path, usecols=['PheCode', 'Short_names'], dtype=str, header=0)
    features_dict = get_features_dict(terms, df_ph_dicts)
    k = 6

    alphas = [0.25, 0.5]
    l1_ratios = [0, 0.5, 1]

    # for k in n_components:
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            out_path = path.join('sensitive_analysis', 'nmf_alpha{}_l1_{}'.format(alpha, l1_ratio))
            _mkdir_p(out_path)

            topic_model = topic_modeling(X, k, alpha, l1_ratio)

            """output the topic descriptors (feature names) to files"""
            out_topics = print_top_words(topic_model, features_dict, n_top_words)
            output_file = path.join(out_path, 'topics.txt')
            out_to_file(output_file, out_topics)

            """pearson value to files"""
            result = calucate_pearson(topic_model, X, Y, k)
            output_file = path.join(out_path, 'person.txt')
            out_to_file(output_file, result)


if __name__ == '__main__':
    run_pearson()






