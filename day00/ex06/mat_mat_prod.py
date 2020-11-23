# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mat_mat_prod.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/23 11:39:11 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 16:37:33 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex04.dot as d
import ex00.sum as s

def mat_mat_prod(x, y):
    """
    Computes the product of two non-empty numpy.ndarray,
    for-loop. The two arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m
    y: has to be an numpy.ndarray, a vector of dimension n
    Returns:
    The product of the matrices as a matrix of dimension m
    None if x or y are empty numpy.ndarray.
    None if x and y does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (not s.elements(x) or not s.elements(y)):
        return (None)
    r = np.rot90(y[::-1], 3)
    res = []
    for row in x:
        res.append([d.dot(row, i) for i in r])
    return (np.array(res))

if __name__ == "__main__":
    W = np.array([
        [ -8, 8, -6, 14, 14, -9, -4],
        [ 2, -11, -2, -11, 14, -2, 14],
        [-13, -2, -5, 3, -8, -4, 13],
        [ 2, 13, -14, -15, -14, -15, 13],
        [ 2, -1, 12, 3, -7, -3, -6]])
    Z = np.array([
        [ -6, -1, -8, 7, -8],
        [ 7, 4, 0, -10, -10],
        [ 7, -13, 2, 2, -11],
        [ 3, 14, 7, 7, -4],
        [ -1, -3, -8, -4, -14],
        [ 9, -14, 9, 12, -7],
        [ -9, -4, -10, -3, 6]])
    print(mat_mat_prod(Z, W))
