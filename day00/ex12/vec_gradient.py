# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_gradient.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/23 17:40:37 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 18:52:07 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex04.dot as d
import ex00.sum as s
import ex05.mat_vec_prod as v

def vec_mat_prod(y, x):
    if (not s.elements(x) or not s.elements(y)):
        return (None)
    if (len(x) == len(y)):
        x = np.rot90(x[::-1], 3)
        return (np.array(list(map(lambda l: d.dot(l, y), x))))

def vec_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrice of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector n * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg
    the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    x0y = (v.mat_vec_prod(x, theta) - y)
    return (vec_mat_prod(x0y, x) / len(y))

if __name__ == "__main__":
    X = np.array([
        [ -6, -7, -9],
        [ 13, -2, 14],
        [ -7, 14, -1],
        [ -8, -4, 6],
        [ -5, -9, 6],
        [ 1, -5, 11],
        [ 9, -11, 8]])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    Z = np.array([3,0.5,-6])
    print(vec_gradient(X, Y, Z))
    W = np.array([0,0,0])
    print(vec_gradient(X, Y, W))
    print(vec_gradient(X, v.mat_vec_prod(X,Z), Z))
