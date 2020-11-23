# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sum.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/22 20:54:48 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 17:31:23 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import functools as ft

def elements(array):
    return array.ndim and array.size

def sum_(x, f):
    """
    Computes the sum of a non-empty numpy.ndarray onto wich a function is
    applied element-wise, using a for-loop.
    Args:
    x: has to be an numpy.ndarray, a vector.
    f: has to be a function, a function to apply element-wise to the
    vector.
    Returns:
    The sum as a float.
    None if x is an empty numpy.ndarray or if f is not a valid function.
    Raises:
    This function should not raise any Exception.
    c = 0
    for num in x:
        c += f(num)
    return (c)
    """
    if (not elements(x)):
        return (None)
    try:
        (f(-1))
    except:
        return (None)
    return (ft.reduce(lambda a, b: a + b, map(lambda l: f(l), x)))

if __name__ == "__main__":
    import numpy as np
    print("X = np.array([0, 15, -9, 7, 12, 3, -21])\nsum_(X, lambda x: x**2)")
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(sum_(X, lambda x: x**2))
