#! /usr/bin/python
# -*- coding: utf-8 -*-
# Author: Wenyuan Wu, 18746867
# Date: 03.03.2020
# Additional Info:

import string
from simple_lemmatizer import simple_lemmatizer


def preprocess(raw_text):
    doc_list = []
    for line in raw_text:
        line = line.rstrip('\n')
        word_list = line.split(' ')
        for token in word_list:
            token = token.lower()
            token = token.translate(str.maketrans('', '', string.punctuation))
            if token:
                # token = simple_lemmatizer(token)
                doc_list.append(token)
    return doc_list


def feature_matrix(base_list, target_list, doc_list):
    pass


def similarity_matrix():
    pass


def main():
    raw_text = open('1984_test.txt', 'r', encoding='utf-8').readlines()
    base = open('B.txt', 'r', encoding='utf-8').readlines()
    target = open('T.txt', 'r', encoding='utf-8').readlines()
    doc_list = preprocess(raw_text)
    base_list = preprocess(base)
    target_list = preprocess(target)


if __name__ == '__main__':
    main()
