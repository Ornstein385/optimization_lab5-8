import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve

np.seterr(all='warn')


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


def H(X, tol, df):
    n = len(X)
    h = 0.1 * tol
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_plus_h = np.array(X, dtype=float)
            x_minus_h = np.array(X, dtype=float)
            x_plus_h[j] += h
            x_minus_h[j] -= h
            hessian[i, j] = (df(x_plus_h)[i] - df(x_minus_h)[i]) / (2 * h)
    return hessian


def nsearch(f, df, x0, tol):
    kmax = 1000
    x = np.array(x0, dtype=float)
    coords = [x.copy()]
    neval = 0

    for k in range(kmax):
        dx = -solve(H(x, tol, df), df(x))
        x += dx

        coords.append(x.copy())
        neval += 1

        if norm(dx) < tol:
            break

    return [x, f(x), neval, coords]
