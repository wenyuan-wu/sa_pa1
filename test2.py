
from collections import deque

doc_list = ['a', 'very', 'funny', 'man', 'arrived',
            'in', 'the', 'house', 'and', 'put', 'on', 'his',
            'coat']

window = deque(maxlen=5)
for word in doc_list:
    window.append(word)

    print(window)
    window.popleft()