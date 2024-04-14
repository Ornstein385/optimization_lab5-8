import numpy as np
from numpy.linalg import norm


def f(X):
    x, y = X
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def df(X):
    x, y = X
    dfx = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    dfy = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
    return np.array([dfx, dfy])


def grsearch(x0, tol):
    al = 0.01
    kmax = 1000
    coords = [x0]
    x = x0
    neval = 0

    for k in range(kmax):
        grad = df(x)  # Вычисляем градиент в текущей точке
        x_next = x - al * grad  # Обновляем позицию
        neval += 1
        coords.append(x_next)

        if norm(x_next - x) < tol or neval > 1000:  # Критерий остановки
            break
        x = x_next

    xmin = x
    fmin = f(xmin)
    return [xmin, fmin, neval, coords]
