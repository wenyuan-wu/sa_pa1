from similarity import preprocess
from collections import Counter
import sys

# ###
# This script is to output most n frequent words from a text
# usage:
# python3 most_freq.py [input_file] [number_of_n]

file_name = sys.argv[1]
most_com = int(sys.argv[2])

raw_text = open(file_name, 'r', encoding='utf-8')
doc_list = preprocess(raw_text)
raw_text.close()

counter = Counter(doc_list)
most_occr = counter.most_common(most_com)
for i, j in most_occr:
    print(i)
