import numpy as np


def goldensectionsearch(f, interval, tol):
    gr = (np.sqrt(5) + 1) / 2  # Золотое сечение
    a, b = interval
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    neval = 0

    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        neval += 1

    xmin = (b + a) / 2
    fmin = f(xmin)

    return [xmin, fmin, neval]


def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)

    return v


def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def sdsearch(f, df, x0, tol):
    kmax = 1000
    coords = [x0]
    x = np.array(x0, dtype=float)
    neval = 0

    for k in range(kmax):
        grad = df(x)
        neval += 1

        # Определяем функцию для одномерного поиска
        f1dim = lambda alpha: f(x - alpha * grad)
        interval = [0, 1]
        alpha, fmin, _ = goldensectionsearch(f1dim, interval, tol)

        x_next = x - alpha * grad
        coords.append(x_next)

        if np.linalg.norm(x_next - x) < tol:
            break

        x = x_next

    xmin = x
    fmin = f(xmin)
    return [xmin, fmin, neval, coords]
