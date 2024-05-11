import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm

sns.set(style="white", color_codes=True)


def col_perceptron(data, weights, eta, maxiter):
    """
    collate all the values for some given perceptron
    :param data: the original dataframe [sepal length, sepal width, petal length, petal width, classification (target)]
    :param weights: list of initial weights [bias, sl_param, sw_param, pl_param, pw_param]
    :param eta: same eta from gradient descent, just a scaling parameter
    :param maxiter: maximal number of iterations this function should run, regardless of convergence
    :return: the updated list of weights for the perceptron, again, regardless of convergence
    """
    # ensure we cap the convergence
    for __ in tqdm(range(maxiter), total=maxiter):
        # if exit flag is true at the end of a sweep through
        # all the rows, then there were no updates, and thus
        # the perceptron converged to some linear function
        # that linearly separates the data
        exit_flag = True

        for _, row in data.iterrows():
            # substitute values to check for update
            o = 1 * weights[0]
            for index, x in enumerate(row.drop('class').values):
                o += (weights[index + 1] * x)

            # if 0 then negative, otherwise just return sign
            if o == 0:
                o = -1
            else:
                o = o / np.abs(o)  # if this o is trivially small it could be a problem

            # if this is 0 then there is no need to update,
            # as it is classifying correctly
            update = int(row['class']) - o

            # otherwise update and flag false
            if update != 0:
                exit_flag = False
                weights[0] += eta * update
                for index, x in enumerate(row.drop('class').values):
                    weights[index + 1] += eta * update * x

        # convergence
        if exit_flag:
            return weights

    # non-convergence
    return weights


# fetch dataset
iris = fetch_ucirepo(id=53)
original = iris.data.original

# splitting data into three sets for each perceptron
p0_data = original.replace(to_replace={'Iris-setosa': 1, 'Iris-versicolor': -1, 'Iris-virginica': -1})
p1_data = original.replace(to_replace={'Iris-setosa': -1, 'Iris-versicolor': 1, 'Iris-virginica': -1})
p2_data = original.replace(to_replace={'Iris-setosa': -1, 'Iris-versicolor': -1, 'Iris-virginica': 1})

# weights [w_0 (bias), w_1 (sepal_l), w_2 (sepal_w), w_3 (petal_l), w_4 (petal_w)]
p0_weights = [0, 0, 0, 0, 0]
p1_weights = [0, 0, 0, 0, 0]
p2_weights = [0, 0, 0, 0, 0]

# perceptron 0 classifies setosa-type
# perceptron 1 classifies versicolor-type
# perceptron 2 classifies virginica-type
p0_weights = col_perceptron(p0_data, p0_weights, eta=0.1, maxiter=5000)
p1_weights = col_perceptron(p1_data, p1_weights, eta=0.1, maxiter=5000)
p2_weights = col_perceptron(p2_data, p2_weights, eta=0.1, maxiter=5000)

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

hyperpetal = pd.DataFrame(info_dict)

# iterate over the entire 4D object and let the perceptron
# decide what flower each point should be
# (this is so computationally inefficient, I apologize)
for dfindex, row in hyperpetal.iterrows():
    # bias
    p0 = p0_weights[0]
    p1 = p1_weights[0]
    p2 = p2_weights[0]

    # account for bias parameter and multiply each row value
    # with its corresponding perceptron parameter weight
    for index, x in enumerate(row.drop('class').values):
        p0 += p0_weights[index + 1] * x
        p1 += p1_weights[index + 1] * x
        p2 += p2_weights[index + 1] * x

    maxval = max(p0, p1, p2)

    # classify
    if p0 == maxval:
        hyperpetal.at[dfindex, 'class'] = 'Iris-setosa'
    elif p1 == maxval:
        hyperpetal.at[dfindex, 'class'] = 'Iris-versicolor'
    else:
        hyperpetal.at[dfindex, 'class'] = 'Iris-virginica'

# legend is out of order otherwise
hyperpetal = hyperpetal.sort_values('class')

# plot
sns.pairplot(hyperpetal, hue='class', height=2, palette=sns.color_palette("tab10"))
plt.show()