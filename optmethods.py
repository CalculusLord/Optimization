"""
Written by
    Nathanael J. Reynolds
        SDSU, 2020 revised 2022
"""

import numpy as np
import math

# Global Variables
contr = 0.5 # rho for backtracking line search
c_val = 10**(-4) # c for backtracking line search
c2_val = 0.9 # New variable introduced for Homework 2

def converge(prev,now):
    """
    Determines rate of convergence returns a scalar
    """
    minim = np.array([[1],[1]])
    top = now - minim
    bottom = prev - minim
    P = math.log(np.linalg.norm(top))/math.log(np.linalg.norm(bottom))
    return P

def f(x):
    """
    Takes in a vector and returns the Rosenbrock Function
    """
    x1 = float(x[0])
    x2 = float(x[1])
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


def gradf(x):
    """
    Takes in a vector and returns gradient of f(x1,x2) as a column vector. The general
    solution for grad f was calculated by hand
    """
    x1 = float(x[0])
    x2 = float(x[1])

    a1 = -400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1)
    a2 = 200 * (x2 - x1 ** 2)
    grad = np.array([[a1], [a2]])
    return grad


def hessf(x):
    """
    Take in a vector and returns Hessian of f(x1,x2) as a column vector. The general
    solution for hess f was calculated by hand
    """
    x1 = float(x[0])
    x2 = float(x[1])

    a11 = -400 * (x2 - x1 ** 2) + 800 * x1 ** 2 + 2
    a12 = -400 * x1
    a21 = a12
    a22 = 200
    hess = np.array([[a11, a12], [a21, a22]])
    return hess


def norm(vec):
    """
    Returns the norm (magnitude) of any (1,2) vector
    """
    sum = 0
    for i in range(0, 2):
        sum = sum + vec[i] ** 2

    root = float(np.sqrt(sum))
    return root

def x_up(x, alpha, p):
    """
    Takes in a vector and updates it for iteration returns column vector
    """
    init_x1 = float(x[0]) + alpha * float(p[0])
    init_x2 = float(x[1]) + alpha * float(p[1])
    vec_x = np.array([[init_x1], [init_x2]])
    return vec_x


def step_dist(x, p, c):
    """
    Backtracking line search function
    """
    a = 1
    in1 = x + a * p
    fun = f(x)
    fin = f(in1)
    gfun = gradf(x)
    comp = fun + c * a * p.transpose().dot(gfun)

    while fin > comp:
        a = contr * a
        in1 = x + a * p
        fun = f(x)
        fin = f(in1)
        gfun = gradf(x)
        comp = fun + c * a * p.transpose().dot(gfun)
    return a
#
# def optlibrary(handle):
#     dictionary = {'steep': lambda x: -x / np.linalg.norm(x),
#                   'newton': lambda x, y: -np.linalg.inv(x).dot(y)}
#     return dictionary(handle)
#
# def optimizer(init_x):
#     func = f(init_x)
#     gf = gradf(init_x)
#     # steep = lambda x: -x / np.linalg.norm(x)  # returns p_k for steepest decent method
#     # newton = lambda x, y: -np.linalg.inv(x).dot(y)  # Returns p_k for Newton's Method
#     pk = steep(gf)
#     step = step_dist(x_bar, pk, c_val)
#     i = 0  # sets up a count for number of iterates
#     # %%
#     while (abs(func) > 10e-8 or abs(norm(gf)) < 10e-8):
#         """
#         The loop executes the algorithm to minimize the function
#         using steepest descent
#         """
#         if i < 6:
#             print('x', i + 1, '=\t\t', x_bar.transpose())
#         elif i > 6218 - 7:
#             print('x', i + 1, '=\t', x_bar.transpose())
#         x_bar = x_up(x_bar, step, pk)
#         func = f(x_bar)
#         gf = gradf(x_bar)
#         pk = steep(gf)
#         step = step_dist(x_bar, pk, c_val)
#         i = i + 1
#     print('minimized after', i, 'iterations to the function value', func, '\n')