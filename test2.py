import string
from collections import deque
import itertools

doc_list = ['a', 'very', 'funny', 'man', 'arrived',
            'in', 'the', 'house', 'and', 'put', 'on', 'his',
            'coat']

window = deque(maxlen=5)
for word in doc_list:
    window.append(word)

    print(window)
    window.popleft()

token = "accus’d"
punc = string.punctuation + '’'
token = token.translate(str.maketrans('', '', punc))
print(token)

# freq_dict = {word: doc_list.count(word) for word in doc_list}
#
# new_dict = {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}
#
# out = dict(itertools.islice(new_dict.items(), 100))
#
# for k, v in out.items():
#     print(k)
