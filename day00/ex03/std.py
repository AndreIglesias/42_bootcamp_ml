# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    std.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/23 00:09:52 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 19:11:46 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex02.variance as v
import ex00.sum as s

def std(x):
    """
    Computes the standard deviation of a non-empty numpy.ndarray, using a
    for-loop.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    The standard deviation as a float.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if (not s.elements(x)):
        return (None)
    return (np.sqrt(v.variance(x)))

if __name__ == "__main__":
    print("X = np.array([0, 15, -9, 7, 12, 3, -21])\nstd(X)")
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(std(X))
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print("Y = np.array([2, 14, -13, 5, 12, 4, -19])\nstd(Y)")
    print(std(Y))
