from collections import defaultdict
import pandas as pd
import numpy.ma as ma

my_dict = defaultdict(lambda: defaultdict(lambda: 0))

my_dict['cherry']['computer'] = 2
my_dict['cherry']['data'] = 8
my_dict['cherry']['result'] = 9
my_dict['cherry']['pie'] = 442
my_dict['cherry']['sugar'] = 25

my_dict['strawberry']['computer'] = 0
my_dict['strawberry']['data'] = 0
my_dict['strawberry']['result'] = 1
my_dict['strawberry']['pie'] = 60
my_dict['strawberry']['sugar'] = 19

my_dict['digital']['computer'] = 1670
my_dict['digital']['data'] = 1683
my_dict['digital']['result'] = 85
my_dict['digital']['pie'] = 5
my_dict['digital']['sugar'] = 4

my_dict['information']['computer'] = 3325
my_dict['information']['data'] = 3982
my_dict['information']['result'] = 378
my_dict['information']['pie'] = 5
my_dict['information']['sugar'] = 13

df = pd.DataFrame.from_dict(my_dict, orient='index')

df_sum = df.sum(axis=1).sum()

df["count_w"] = df.sum(axis=1)
df.loc["count_c"] = df.sum(axis=0)

# print(df_sum)
# print(df["count_w"]["information"])

# df["count_w"]["information"] = 5555
# print(df)

base_list = ['computer', 'data', 'result', 'pie', 'sugar']
target_list = ['cherry', 'strawberry', 'digital', 'information']


def joint_prob(target, base, sum_c):
    prob_t_b = df[base][target] / sum_c
    prob_t = df['count_w'][target] / sum_c
    prob_c = df[base]['count_c'] / sum_c
    result = ma.log2((prob_t_b/(prob_t * prob_c)))
    if result > 0:
        return ma.round(result, decimals=4)
    else:
        return 0


# print(joint_prob('information', 'data', df_sum))

df = df.astype(float)

for word in target_list:
    for context in base_list:
        ppmi = joint_prob(word, context, df_sum)
        df[context][word] = ma.round(ppmi, decimals=2)

df = df.drop('count_c')
df = df.drop('count_w', axis=1)
print(df)

v1 = [0, 0, 4]
v2 = [0, 1, 2]

cherry = df.loc['cherry', :]
digital = df.loc['digital', :]
information = df.loc['information', :]
# print(cherry)


def euclidean_distance(x, y):
    return ma.sqrt(ma.sum((x - y) ** 2))


def cosine_similarity(x, y):
    return ma.dot(x, y) / (ma.sqrt(ma.dot(x, x)) * ma.sqrt(ma.dot(y, y)))


print(euclidean_distance(information, digital))
print(cosine_similarity(information, digital))

