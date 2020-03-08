from collections import defaultdict
import pandas as pd

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

df["count_w"] = df.sum(axis=1)
df.loc["count_c"] = df.sum(axis=0)
print(df["count_w"]["information"])

# df["count_w"]["information"] = 5555
print(df)
