from similarity import preprocess
import itertools

raw_text = open('1984.txt', 'r', encoding='utf-8').readlines()
doc_list = preprocess(raw_text)

freq_dict = {word: doc_list.count(word) for word in doc_list}

new_dict = {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}

out = dict(itertools.islice(new_dict.items(), 100))

for k, v in out.items():
    print(k, v)
