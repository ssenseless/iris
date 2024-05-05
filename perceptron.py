import pandas as pd
from ucimlrepo import fetch_ucirepo
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)


def col_perceptron(data, weights, eta, maxiter):
    for __ in range(maxiter):
        exit_flag = True

        for _, row in data.iterrows():
            o = 1 * weights[0]
            for index, x in enumerate(row.drop('class').values):
                o += (weights[index + 1] * x)

            if o == 0:
                o = -1
            else:
                o = o / np.abs(o)  # if this o is trivially small it could be a problem

            update = int(row['class']) - o
            if update != 0:
                exit_flag = False
                weights[0] += eta * update
                for index, x in enumerate(row.drop('class').values):
                    weights[index + 1] += eta * update * x

        if exit_flag:
            return weights

    print("did not converge")
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

testexamples = pd.DataFrame(
    {
        'sepal length': [5.4, 6.0, 6.8, 5.2, 5.9],
        'sepal width': [3.3, 2.7, 3.4, 4.0, 2.9],
        'petal length': [1.6, 4.2, 6.0, 5.0, 4.5],
        'petal width': [0.4, 1.2, 2.4, 0.3, 1.4],
        'class': ['Iris-test', 'Iris-test', 'Iris-test', 'Iris-test', 'Iris-test']
    }
)

plot_test = pd.concat([original, testexamples])
sns.pairplot(plot_test, hue='class', height=2, palette=sns.color_palette("tab10"))
plt.show()

for dfindex, row in testexamples.iterrows():
    p0 = p0_weights[0]
    p1 = p1_weights[0]
    p2 = p2_weights[0]

    for index, x in enumerate(row.drop('class').values):
        p0 += p0_weights[index + 1] * x
        p1 += p1_weights[index + 1] * x
        p2 += p2_weights[index + 1] * x

    maxval = max(p0, p1, p2)

    if p0 == maxval:
        testexamples.at[dfindex, 'class'] = 'Iris-setosa'
    elif p1 == maxval:
        testexamples.at[dfindex, 'class'] = 'Iris-versicolor'
    else:
        testexamples.at[dfindex, 'class'] = 'Iris-virginica'

plot_test = pd.concat([original, testexamples])
sns.pairplot(plot_test, hue='class', height=2, palette=sns.color_palette("tab10"))
plt.show()
