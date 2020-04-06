#! /usr/bin/python
# -*- coding: utf-8 -*-
# Author: Wenyuan Wu, 18746867
# Date: 03.03.2020
# Additional Info:

import string
from collections import defaultdict
import pandas as pd
import numpy.ma as ma
from typing import List, TextIO
from helper_functions import hierarchical_clusters_draw, hierarchical_clusters_print, pca_plot, kmeans_clusters_print
# from simple_lemmatizer import simple_lemmatizer


def preprocess(raw_text: TextIO) -> List:
    """
    A function to handle raw input and preprocess text for further process.
    Uncomment line 12 and 37 to implement lemmatizer from Spacy. Caution: lemmatizer takes a lot of time.

    Parameters
    ----------
    raw_text: input raw file

    Returns
    -------
    A list of tokens
    """
    doc_list = []
    # special punctuation '’' in Shakespeare's works
    punctuation = string.punctuation + '’'
    for line in raw_text:
        line = line.rstrip('\n')
        word_list = line.split(' ')
        for token in word_list:
            token = token.lower()
            token = token.translate(str.maketrans('', '', punctuation))
            if token:
                # token = simple_lemmatizer(token)
                doc_list.append(token)
    return doc_list


def feature_matrix(base_list: List, target_list: List, doc_list: List) -> pd.DataFrame:
    """
    Function to calculate the feature matrix T x B using the context window size 5
    (two positions before and two after the target word). Use point-wise mutual information scores as weights.

    Parameters
    ----------
    base_list: list of base words
    target_list: list of target words
    doc_list: list of document

    Returns
    -------
    A Pandas DataFrame
    """
    temp_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for i in target_list:
        for j in base_list:
            temp_dict[i][j] = 0
    doc_len = len(doc_list)
    for index, token in enumerate(doc_list):
        # print('current index: {} token: {}'.format(index, token))
        if token in target_list:
            window = []
            if index-2 in range(0, doc_len):
                window.append(doc_list[index-2])
            if index-1 in range(0, doc_len):
                window.append(doc_list[index-1])
            if index+1 in range(0, doc_len):
                window.append(doc_list[index+1])
            if index+2 in range(0, doc_len):
                window.append(doc_list[index+2])
            # print('current window: {}'.format(window))
            for base in base_list:
                if base in window:
                    temp_dict[token][base] += window.count(base)
    df = pd.DataFrame.from_dict(temp_dict, orient='index')
    df_sum = df.sum(axis=1).sum()
    df["count_w"] = df.sum(axis=1)
    df.loc["count_c"] = df.sum(axis=0)
    df = df.astype(float)
    for word in target_list:
        for context in base_list:
            ppmi = joint_prob(word, context, df_sum, df)
            df[context][word] = ma.round(ppmi, decimals=2)
    df = df.drop('count_c')
    df = df.drop('count_w', axis=1)
    return df


def joint_prob(target: str, base: str, sum_c: int, df: pd.DataFrame) -> float:
    """
    To calculate PPMI for a token in a given DataFrame.

    Parameters
    ----------
    target: target word
    base: base word
    sum_c: total number of occurrence
    df: Pandas DataFrame

    Returns
    -------
    PPMI number, if less than 0, then return 0
    """
    prob_t_b = df[base][target] / sum_c
    prob_t = df['count_w'][target] / sum_c
    prob_c = df[base]['count_c'] / sum_c
    result = ma.log2((prob_t_b/(prob_t * prob_c)))
    if result > 0:
        return ma.round(result, decimals=4)
    else:
        return 0


def similarity_matrix(df: pd.DataFrame, target_list: List) -> pd.DataFrame:
    """
    Function to calculate cosine similarity matrix
    Parameters
    ----------
    df: Pandas DataFrame, feature matrix
    target_list: list of target words

    Returns
    -------
    DataFrame: cosine similarity matrix
    """
    cos_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for i in target_list:
        for j in target_list:
            x = df.loc[i, :]
            y = df.loc[j, :]
            cos_dict[i][j] = ma.round(cosine_similarity(x, y), decimals=3)
    cos_df = pd.DataFrame.from_dict(cos_dict, orient='index')
    return cos_df


def cosine_similarity(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate cosine similarity of two vectors.

    Parameters
    ----------
    x: Pandas Series
    y: Pandas Series

    Returns
    -------
    A float number
    """
    return ma.dot(x, y) / (ma.sqrt(ma.dot(x, x)) * ma.sqrt(ma.dot(y, y)))


def distance_matrix(df: pd.DataFrame, target_list: List) -> pd.DataFrame:
    """
    Function to calculate euclidean distance matrix
    Parameters
    ----------
    df: Pandas DataFrame, feature matrix
    target_list: list of target words

    Returns
    -------
    DataFrame: euclidean distance matrix
    """
    euc_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for i in target_list:
        for j in target_list:
            x = df.loc[i, :]
            y = df.loc[j, :]
            euc_dict[i][j] = ma.round(euclidean_distance(x, y), decimals=3)
    euc_df = pd.DataFrame.from_dict(euc_dict, orient='index')
    return euc_df


def euclidean_distance(x: pd.Series, y: pd.Series) -> float:
    """
     Calculate euclidean distance of two vectors.

     Parameters
     ----------
     x: Pandas Series
     y: Pandas Series

     Returns
     -------
     A float number
     """
    return 1 / cosine_similarity(x, y)
