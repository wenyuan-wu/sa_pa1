from similarity import preprocess
from collections import Counter

raw_text = open('shakespeare.txt', 'r', encoding='utf-8')
doc_list = preprocess(raw_text)
raw_text.close()

Counter = Counter(doc_list)
most_occr = Counter.most_common(1000)
for i, j in most_occr:
    print(i)
