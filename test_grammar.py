import numpy as np

w = np.random.normal(1, 0.1)
b = np.random.normal(1, 0.1)


def predict(self, x=None):
    y_pred = f(x, w, b)
    return y_pred


def f(x, w, b):
    return x * w + b


x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 2.0, 1.0])
print(len(x))


m =np.mean((y - x) ** 2)
print(m)
