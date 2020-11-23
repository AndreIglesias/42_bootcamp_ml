# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    dot.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/23 00:17:59 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 14:11:21 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex00.sum as s

def dot(x, y):
    """
    Computes the dot product of two non-empty numpy.ndarray, using a
    for-loop. The two arrays must have the same dimensions.
    Args:
    x: has to be an numpy.ndarray, a vector.
    y: has to be an numpy.ndarray, a vector.
    Returns:
    The dot product of the two vectors as a float.
    None if x or y are empty numpy.ndarray.
    None if x and y does not share the same dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (not s.elements(x) or not s.elements(y) or len(x) != len(y)):
        return (None)
    ss = s.sum_(np.array([i * j for i, j in zip(x,y)]), lambda X: X)
    if (ss != None):
        return (ss)
    return (0)

if __name__ == "__main__":
    print("X = np.array([0, 15, -9, 7, 12, 3, -21])\nY = np.array([2, 14, -13, 5, 12, 4, -19])\ndot(X,Y)")
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(dot(X, Y))
