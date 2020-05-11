import numpy as np

w = np.random.normal(1, 0.1)
b = np.random.normal(1, 0.1)


def predict(self, x=None):
    y_pred = f(x, w, b)
    return y_pred


def f(x, w, b):
    return x * w + b


x = [1.0, 2.0, 3.0]
print(w)
print(x * w)