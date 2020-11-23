# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    variance.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/22 23:53:26 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 11:58:39 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex00.sum as s
import ex01.mean as m

def variance(x):
    """
    Computes the variance of a non-empty numpy.ndarray, using a for-loop.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    The variance as a float.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if (not x.any()):
        return (None)
    mean = m.mean(x)
    return (s.sum_(x, lambda X: (X - mean)**2) / len(x))

if __name__ == "__main__":
    import numpy as np
    print("X = np.array([0, 15, -9, 7, 12, 3, -21])\variance(X)")
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(variance(X))
