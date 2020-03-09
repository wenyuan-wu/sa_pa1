#! /usr/bin/python
# -*- coding: utf-8 -*-
# Author: Wenyuan Wu, 18746867
# Date: 03.03.2020
# Additional Info:

import spacy

nlp = spacy.load('en_core_web_sm')


def simple_lemmatizer(word: str) -> str:
    """
    A simple lemmatizer from Spacy. -PRON- will be ignored, refer to:

    Parameters
    ----------
    word

    Returns
    -------

    """
    lemma = nlp(word)[0].lemma_
    if lemma != '-PRON-':
        return lemma
    else:
        return word
