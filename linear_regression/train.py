# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from linear_regression import *


def show_data(x, y, w=None, b=None):
    plt.scatter(x, y, marker='.')
    if w is not None and b is not None:
        plt.plot(x, w * x + b, c='red')
    plt.show()


if __name__ == "__main__":
    # data generation
    np.random.seed(272)
    data_size = 100
    x = np.random.uniform(low=1.0, high=10.0, size=data_size)
    y = x * 20 + 10 + np.random.normal(loc=0.0, scale=10.0, size=data_size)

    plt.scatter(x, y, marker='.')
    plt.show()

    # train / test split
    shuffled_index = np.random.permutation(data_size)
    x = x[shuffled_index]
    y = y[shuffled_index]
    split_index = int(data_size * 0.7)
    # train dataset
    x_train = x[:split_index]
    y_train = y[:split_index]
    # test dataset
    x_test = x[split_index:]
    y_test = y[split_index:]

    # train the linear regression model
    regr = LinearRegression(learning_rate=0.01, max_iter=10, seed=314)
    regr.fit(x_train, y_train)
    print('cost: \t{:.3}'.format(regr.loss()))
    print('w: \t{:.3}'.format(regr.w))
    print('b: \t{:.3}'.format(regr.b))
    show_data(x_test, y_test, regr.w, regr.b)

    # plot the evolution of cost
    plt.scatter(np.arange(len(regr.loss_arr)), regr.loss_arr, marker='o', c='green')
    plt.show()
