#! /usr/bin/python
# -*- coding: utf-8 -*-
# Author: Wenyuan Wu, 18746867
# Date: 03.03.2020
# Additional Info:

import argparse
import similarity
import helper_functions

# ### Distributional vectors and similarity
# This is the main script to run this python program.
# To get help, run this in command line:
# python3 clustering.py -h


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser with required options.

    Returns
    -------
    The argument parser with all arguments added.
    """
    parser = argparse.ArgumentParser(
        description='A Python script that takes as input raw text '
                    'and generates as output clusters of words with similar meanings.')
    parser.add_argument('-o', help='input file', required=True)
    parser.add_argument('-b', help='base file', required=True)
    parser.add_argument('-t', help='target file', required=True)
    return parser


def main():
    args = get_argument_parser().parse_args()
    raw_text = open(args.o, 'r', encoding='utf-8')
    base = open(args.b, 'r', encoding='utf-8')
    target = open(args.t, 'r', encoding='utf-8')
    doc_list = similarity.preprocess(raw_text)
    base_list = similarity.preprocess(base)
    target_list = similarity.preprocess(target)
    raw_text.close()
    base.close()
    target.close()
    df = similarity.feature_matrix(base_list, target_list, doc_list)
    print()
    print('Similarity Matrix:')
    print(similarity.similarity_matrix(df, target_list).to_csv(sep='\t'))
    print()
    print('Distance Matrix:')
    print(similarity.distance_matrix(df, target_list).to_csv(sep='\t'))
    # Plot and clustering
    arr = df.to_numpy()
    helper_functions.hierarchical_clusters_draw(arr, target_list)
    print()
    print('Hierarchical Clusters:')
    helper_functions.hierarchical_clusters_print(arr, target_list)
    print()
    print('K-means Clusters:')
    helper_functions.kmeans_clusters_print(arr, target_list)
    helper_functions.pca_plot(arr, target_list)


if __name__ == '__main__':
    main()
