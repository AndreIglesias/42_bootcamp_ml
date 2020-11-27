# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mat_vec_prod.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/23 11:11:21 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 18:46:16 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex04.dot as d
import ex00.sum as s

def mat_vec_prod(x, y):
    """
    Computes the product of two non-empty numpy.ndarray, using a
    for-loop. The two arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension n * 1.
    Returns:
    The product of the matrix and the vector as a vector of dimension m *
    1.
    None if x or y are empty numpy.ndarray.
    None if x and y does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (not s.elements(x) or not s.elements(y)):
        return (None)
    if (len(list(filter(lambda l: len(l) == len(y), x))) == len(x)):
        return (np.array(list(map(lambda l: d.dot(l, y), x))))
    return (None)

if __name__ == "__main__":
    W = np.array([
        [ -8, 8, -6, 14, 14, -9, -4],
        [ 2, -11, -2, -11, 14, -2, 14],
        [-13, -2, -5, 3, -8, -4, 13],
        [ 2, 13, -14, -15, -14, -15, 13],
        [ 2, -1, 12, 3, -7, -3, -6]])
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7,1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((7,1))
    print(mat_vec_prod(W, Y))
