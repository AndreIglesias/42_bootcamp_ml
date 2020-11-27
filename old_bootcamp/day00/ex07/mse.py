# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mse.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/23 12:58:02 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 19:10:35 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex00.sum as s

def mse(y, y_hat):
    """
    Computes the mean squared error of two non-empty numpy.ndarray, using
    a for-loop. The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector.
    y_hat: has to be an numpy.ndarray, a vector.
    Returns:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.ndarray.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (not s.elements(y) or not s.elements(y_hat)):
        return (None)
    f = s.sum_(np.array(list((x - y)**2 for x, y in zip(y, y_hat))), lambda l: l)
    if (f != None):
        return (f / len(y))
    return (0)

if __name__ == "__main__":
    print("X = np.array([0, 15, -9, 7, 12, 3, -21])\nY = np.array([2, 14, -13, 5, 12, 4, -19])\nmse(X,Y)")
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(mse(X, Y))
