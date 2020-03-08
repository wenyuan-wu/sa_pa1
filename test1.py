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


# print(euclidean_distance(information, digital))
# print(cosine_similarity(information, digital))

cos_dict = defaultdict(lambda: defaultdict(lambda: 0))
euc_dict = defaultdict(lambda: defaultdict(lambda: 0))

for i in target_list:
    for j in target_list:
        x = df.loc[i, :]
        y = df.loc[j, :]
        cos_dict[i][j] = ma.round(cosine_similarity(x, y), decimals=3)

# print(cos_dict)
cos_df = pd.DataFrame.from_dict(cos_dict, orient='index')
print(cos_df)


for i in target_list:
    for j in target_list:
        x = df.loc[i, :]
        y = df.loc[j, :]
        euc_dict[i][j] = ma.round(euclidean_distance(x, y), decimals=3)

# print(cos_dict)
euc_df = pd.DataFrame.from_dict(euc_dict, orient='index')
print(euc_df)

# not my code
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def hierarchical_clusters_draw(feature_matrix, target_words, max_d=0.5,
                               output_filename=None,
                               figheight=50, figwidth=20):
    """2d plot of 'feature_matrix' using hierarhical_clusters,
       with the words labeled by 'target_words'.

    Parameters
    ----------
    feature_matrix : 2d np.array,
    shape=(n_samples, n_features)
    (In our case n_samples is the target words)
    (Feature matrix is not the same as similarity/distance matrix!)

    target_words : list of str
        Names of the target words.

    max_d: float (default: 0.5)
        A threshold to apply when forming flat clusters.
        (Play around with this parameter.)

    output_filename : str (default: None)
        If not None, then the output image is written to this location.
        The filename suffix determines the image type. If None, then
        'plt.show()' is called.

    figheight : int (default: 40)
        Height in display units of the output.

    figwidth : int (default: 20)
        Width in display units of the output.
    """

    Z_spat = linkage(feature_matrix, 'complete', 'cosine')
    # You can try out with different linkage function here:
    # 'single','complete','average'

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(50)
    fig.set_figwidth(20)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
    plt.ylabel('Target words', fontsize=18)
    plt.xlabel('Distance', fontsize=18)
    dendrogram(
        Z_spat,
        leaf_font_size=17.,
        labels=target_words,
        orientation='right'
    )

    plt.axvline(x=max_d, color='k', linestyle='--')
    # This drows a vertical number,
    # choose the value where you think clusters should be cut

    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
        # bbox_inches='tight' - to make margins minimal
    else:
        plt.show()


arr = df.to_numpy()
print(arr.shape)
hierarchical_clusters_draw(arr, target_list)


def hierarchical_clusters_print(feature_matrix, target_words, max_d=0.5):
    """Prints clusters in 'feature_matrix' for 'target_words'
       using hierarhical_clusters.

    Parameters
    ----------
    feature_matrix : 2d np.array,
    shape=(n_samples, n_features)
    (In our case n_samples is the target words)
    (Feature matrix is not the same as similarity/distance matrix!)

    target_words : list of str
        Names of the target words.

    max_d: float (default: 0.5)
        A threshold to apply when forming flat clusters.
        (Play around with this parameter.)
    """

    Z_spat = linkage(feature_matrix, 'complete', 'cosine')
    # You can play around with different linkage function here:
    # 'single','complete','average'

    clusters = fcluster(Z_spat, max_d, criterion='distance')
    num_clusters = len(set(clusters))

    # Printing clusters
    for ind in range(1, num_clusters + 1):
        print("Cluster %d words:" % ind)

        for i, w in enumerate(target_words):
            if clusters[i] == ind:
                print(' %s' % w)
        print()  # add whitespace

hierarchical_clusters_print(arr, target_list)


def kmeans_clusters_print(feature_matrix, target_words,
                          num_clusters=5):
    """Prints clusters for vectors in 'feature_matrix' K-means algorithm.

    Parameters
    ----------
    feature_matrix : 2d np.array,
        shape=(n_samples, n_features)
        (In our case n_samples is the target words)
        (Feature matrix is not the same as similarity/distance matrix!)

    num_clusters : int (default: 5)
        Number of clusters.

    target_words : list of str
        Names of the target words.
    """

    # Fitting clusters
    km = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

    kmeans = km.fit(feature_matrix)

    cluster_labels = kmeans.labels_
    # the array of cluster labels
    # to which each input vector in n_samples belongs

    cluster_to_words = defaultdict(list)
    # which word belongs to which cluster
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(target_words[c])

    # Printing clusters
    for i in range(num_clusters):
        print("Cluster %d words:" % (i + 1))

        for w in cluster_to_words[i]:
            print(' %s' % w)
        print()  # add whitespace


def pca_plot(feature_matrix, target_words, output_filename=None, title="PCA decomposition"):
    """Plots first and second components of PCA decomposition for vectors in 'feature_matrix'.

        Parameters
        ----------
        feature_matrix : 2d np.array,
        shape=(n_samples, n_features)
        (In our case n_samples is the target words)
        (Feature matrix is not the same as similarity/distance matrix!)

        target_words : list of str
        Names of the target words.

        output_filename : str (default: None)
        If not None, then the output image is written to this location.
        The filename suffix determines the image type. If None, then
        'plt.show()' is called.

        title : str (default: "PCA decomposition")
        Title of the output image
        """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(feature_matrix)
    x = []
    y = []
    for value in pca_result:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(target_words[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title('Hierarchical Clustering Dendrogram', fontsize=20)

    if output_filename:
        plt.savefig(output_filename)
    else:
        plt.show()

    print('PCA done!')


kmeans_clusters_print(arr, target_list, num_clusters=4)
pca_plot(arr, target_list)
