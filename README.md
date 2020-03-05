# PA1: Distributional vectors and similarity

In this task, you will write a Python script that takes as input raw text and generates as output clusters of words with similar meanings.

Preprocessing
Before collecting the counts necessary for the calculation, the raw text needs to be preprocessed. You need to perform exactly these three steps: 1) separate words by white spaces, 2) lowercase, 3) remove all the punctuation.

Input
Define and describe a set B of words which will constitute your word vector basis. Store these words in a text file named B.txt, one word per line. Make sure that this file contains no spaces and no empty lines.
Define and describe the set T of words for which you will calculate cosine similarity and clustering. This set should be meaningful for the clustering task. Store this set in a text file named T.txt, in the same format as B.txt.

Feature matrix
Calculate the feature matrix T x B using the context window size 5 (two positions before and two after the target word). Use point-wise mutual information scores as weights.

Similarity matrix
Calculate the cosine similarity matrix T x T using the PMI feature matrix.
Convert the similarity score into distance. The output of this step are two matrices: one for similarity, one for distance. Print them to the command line.

Clustering
Group the most similar words together using two functions available in the Materials folder: hierarchical clustering and k-means. Be ready to discuss your output in class and propose potential improvements.

Specific requirements
Make sure that the formulas used to calculate feature weights and cosine similarity are transparent in your code. The raw text and the sets B and T should be read from a file given by the user as command line arguments when running the script. Please add any comments necessary to understand the elements of your code and provide instruction how to run it.

Submission

Upload to OLAT by 09.03.2010 at 15h:

1. your Python script
2. B.txt
3. T.txt
