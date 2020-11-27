# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mean.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/22 21:03:12 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 19:09:13 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import ex00.sum as s

def mean(x):
    """
    Computes the mean of a non-empty numpy.ndarray, using a for-loop.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    The mean as a float.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if (not s.elements(x)):
        return (None)
    return (s.sum_(x, lambda X: X) / len(x))

if __name__ == "__main__":
    import numpy as np
    print("X = np.array([0, 15, -9, 7, 12, 3, -21])\nmean(X)")
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(mean(X ** 2))
