# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_mse.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/23 13:49:26 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 14:15:25 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex04.dot as d
import ex00.sum as s

def linear_mse(x, y, theta):
    """
    Computes the mean squared error of three non-empty numpy.ndarray,
    using a for-loop. The three arrays must have compatible dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    x: has to be an numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be an numpy.ndarray, a vector of dimension n * 1.
    Returns:
    The mean squared error as a float.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (not s.elements(x) or not s.elements(y) or not s.elements(theta)):
        return (None)
    if (len(list(filter(lambda l: len(l) == len(theta), x))) != len(x)):
        return (None)
    ss = s.sum_(np.array([(d.dot(theta, i) - j)**2 for i, j in zip(X, Y)]), lambda l: l)
    if (ss != None):
        return (ss/len(Y))
    return (0)


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
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
    print(linear_mse(X, Y, Z))
    W = np.array([0,0,0])
    print(linear_mse(X, Y, W))
