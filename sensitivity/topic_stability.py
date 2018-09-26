import numpy as np
import unsupervised.util
import unsupervised.rankings

from sklearn.decomposition import NMF, LatentDirichletAllocation
from wordcloud import WordCloud
from scipy.stats import pearsonr

import os
import errno
from os import path
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from wordcloud import WordCloud

from metrics import mean_pairwise_Jaccard_Similarity, avg_coherence_by_UMass
from Documents import Documents

def load_data(data_path):
    df = pd.read_csv(data_path)
    print('load data shape:', df.shape)
    df = df.drop(['Class'],
                         axis=1)  
    terms = df.columns
    X = df.values
    print('features shape:', X.shape)
    return X, df, terms


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def print_top_words(model, terms, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ", ".join([terms[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def get_top_words(model, terms, n_top_words):
    topic_list = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = []
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            top_words.append(terms[i])
        print(top_words)
        topic_list.append(top_words)
    return topic_list

def get_idx_for_top_words(model, n_top_words):
    topic_list = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = []
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            top_words_idx.append(i)
        topic_list.append(top_words_idx)
    return topic_list

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

def out_to_file(out_file, topics_per_year):
    with open(out_file, 'w+') as f:
        f.write(topics_per_year)
        f.write('\n\n')


n_top_words = 10
n_components = 100
random_seed = 10

def run():
    data_path = '../data/processed/for_topic_model/lpa_selected_count_to_binary(Adults).csv'
    result_path = '../analysis_result/sensitive_analysis'
    _mkdir_p(result_path)

    X, df, terms = load_data(data_path)
    print(len(terms))

    n_components = [6, 10, 20, 30]


    ref_alpha = 0.1
    ref_l1_ratio = 0.5
    records = []

    alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
    l1_ratios = [0, 0.5, 1]

    for k in n_components:
        ref_items_rankings = get_top_words(topic_modeling(X, k,ref_alpha, ref_l1_ratio), terms, n_top_words)

        all_items_rankings = []
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                topic_model = topic_modeling(X, k,alpha, l1_ratio)
                """evaluate the topics"""
                top_idxs_topics = get_top_words(topic_model, terms, n_top_words)
                all_items_rankings.append(top_idxs_topics)
        r = 0
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                metric = unsupervised.rankings.AverageJaccard()
                matcher = unsupervised.rankings.RankingSetAgreement(metric)
                score = matcher.similarity(ref_items_rankings, all_items_rankings[r])
                r += 1

                records.append({
                    'k': k,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'similarity_score': score,
                })

    pd.DataFrame(data=records).to_csv(path.join(result_path, 'topic_stability_report.csv'), index=None)

if __name__ == '__main__':
    run()






