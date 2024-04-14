import numpy as np
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
    a, b = interval
    gr = (np.sqrt(5) + 1) / 2

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


def bbsearch(f, df, x0, tol):
    D = 0.1
    deltaX = 1
    neval = 0
    interval = [0, 1]
    kmax = 1000
    coords = []
    f1dim = lambda x: f(x0 - x * df(x0))
    ak = goldensectionsearch(f1dim, interval, tol)[0]
    while (norm(deltaX) >= tol) and (neval < kmax):
        df_x0 = df(x0)
        x = x0 - ak * df_x0
        df_x = df(x)
        dg = df_x - df_x0
        dx = -ak * df_x0
        ak = min(np.dot(dx.transpose(), dx) / np.dot(dx.transpose(), dg), D / norm(df_x))
        deltaX = x - x0
        x0 = x
        neval += 1
        coords.append(x)

    xmin = x
    fmin = f(xmin)

    return [xmin, fmin, neval, coords]
