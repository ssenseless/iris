import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)


def count_occ(neighbors, data):
    setosa = 0
    versicolor = 0
    virginica = 0

    for neighbor in neighbors:
        cur_neighbor = data.at[neighbor, 'class']

        if cur_neighbor == 'Iris-setosa':
            setosa += 1
        elif cur_neighbor == 'Iris-versicolor':
            versicolor += 1
        else:
            virginica += 1

    maxval = max(setosa, versicolor, virginica)
    if maxval == setosa:
        return 'Iris-setosa'
    elif maxval == versicolor:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'


def euclid(x, y):
    squared_sum = 0  # sum container
    for index, val in enumerate(x):
        squared_sum += (val - y[index]) ** 2  # sum over all

    return np.sqrt(squared_sum)


def knn(data, test, k):
    neighbors = [[] for _ in range(len(test))]
    test_copy = test.copy()

    for nn_index, test_row in test_copy.iterrows():
        data_copy = data.copy()

        while len(neighbors[nn_index]) < k:
            min_index = -1
            min_val = float("inf")

            for index, row in data_copy.iterrows():
                new_val = euclid(test_row.drop('class').values, row.drop('class').values)
                if new_val < min_val:
                    min_val = new_val
                    min_index = index

            if min_index != -1:
                data_copy.drop(min_index, inplace=True)
                neighbors[nn_index].append(min_index)

        test_copy.at[nn_index, 'class'] = count_occ(neighbors[nn_index], data)
    return test_copy


# fetch dataset
iris = fetch_ucirepo(id=53)
original_data = iris.data.original

test_dict = {
    'sepal length': [],
    'sepal width': [],
    'petal length': [],
    'petal width': [],
    'class': []
}

# sl [4.2 8.0]
# sw [2.0 4.4]
# pl [1.0 7.0]
# pw [0.0 2.6]
for x in range(21, 41):
    for y in range(10, 22):
        for z in range(5, 36):
            for a in range(14):
                test_dict['sepal length'].append(x / 5.0)
                test_dict['sepal width'].append(y / 5.0)
                test_dict['petal length'].append(z / 5.0)
                test_dict['petal width'].append(a / 5.0)
                test_dict['class'].append(0)

test_examples = pd.DataFrame(test_dict)
test_examples = knn(original_data, test_examples, 9)

sns.pairplot(test_examples, hue='class', height=2, palette=sns.color_palette("tab10"))
plt.show()
