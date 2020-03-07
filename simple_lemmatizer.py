import spacy

nlp = spacy.load('en_core_web_sm')


def simple_lemmatizer(word):
    lemma = nlp(word)[0].lemma_
    if lemma != '-PRON-':
        return lemma
    else:
        return word
