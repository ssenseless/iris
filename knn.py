import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)


def count_occ(neighbors, data):
    """
    count occurrences of each type of flower
    :param neighbors: k-length list of nearest neighbors indices in data
    :param data: original dataframe of plants
    :return: classification of test plant
    """
    setosa = 0
    versicolor = 0
    virginica = 0

    # neighbor is the index in the dataframe
    for neighbor in neighbors:
        cur_neighbor = data.at[neighbor, 'class']

        # check and increment
        if cur_neighbor == 'Iris-setosa':
            setosa += 1
        elif cur_neighbor == 'Iris-versicolor':
            versicolor += 1
        else:
            virginica += 1

    # get maximum and return matching class to
    # classify test example
    maxval = max(setosa, versicolor, virginica)
    if maxval == setosa:
        return 'Iris-setosa'
    elif maxval == versicolor:
        return 'Iris-versicolor'
    return 'Iris-virginica'


def euclid(x, y):
    """
    compute l2 (Euclidean) distance for all
    parameters of test plant x and data plant y
    :param x: test plant [sepal length, sepal width, petal length, petal width]
    :param y: data plant [sepal length, sepal width, petal length, petal width]
    :return: Euclidean distance
    """
    squared_sum = 0

    for index, val in enumerate(x):
        squared_sum += (val - y[index]) ** 2  # sum over all parameters

    return np.sqrt(squared_sum)


def knn(data, test, k):
    """
    compute the k-nearest neighbors for each member of test with the members of data
    :param data: original dataframe [sepal length, sepal width, petal length, petal width, classification (target)]
    :param test: dataframe of test points [sepal length, sepal width, petal length, petal width, classification (target)]
    :param k: number of nearest neighbors (would work better with primes, due to three classification types)
    :return: altered dataframe with classifications filled
    """
    # deep copy
    test_copy = test.copy()

    # for each test plant
    for nn_index, test_row in tqdm(test_copy.iterrows(), total=len(test_copy)):
        # deep copy this too so as not to affect outer scope
        # (python is picky-choosy about when it is pass-by-ref or pass-by-val)
        data_copy = data.copy()
        distance = {}

        # get Euclidean distance w.r.t each existing data member
        for dist_index, data_row in data_copy.iterrows():
            distance[dist_index] = euclid(data_row.drop('class'), test_row.drop('class'))

        # python moment
        test_copy.at[nn_index, 'class'] = count_occ(list(dict(sorted(distance.items(), key=lambda neighbor: neighbor[1])[:k]).keys()), data)
    return test_copy


# fetch dataset
iris = fetch_ucirepo(id=53)
original_data = iris.data.original

# create some fourth-dimensional volume that
# has all the important ranges for the flowers
# in every direction, increasing by 0.2 each
# point. aptly name it a 'hyperpetal' because
# you're a nerd, and you think you're funny.
info_dict = {
    'sepal length': [],
    'sepal width': [],
    'petal length': [],
    'petal width': [],
    'class': []
}

# sepal len: [4.2 8.0]
# sepal wid: [2.0 4.4]
# petal len: [1.0 7.0]
# petal wid: [0.0 2.6]
for x in range(21, 41):
    for y in range(10, 22):
        for z in range(5, 36):
            for a in range(14):
                info_dict['sepal length'].append(x / 5.0)
                info_dict['sepal width'].append(y / 5.0)
                info_dict['petal length'].append(z / 5.0)
                info_dict['petal width'].append(a / 5.0)
                info_dict['class'].append(0)

# perform k-nearest neighbors
hyperpetal = pd.DataFrame(info_dict)
test_examples = knn(original_data, hyperpetal, 9)

# plot
sns.pairplot(test_examples, hue='class', height=2, palette=sns.color_palette("tab10"))
plt.show()
