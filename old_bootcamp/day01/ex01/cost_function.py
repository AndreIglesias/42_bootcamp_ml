#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    cost_function.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/24 15:58:53 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/24 16:01:46 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
import day00.ex00.sum as s
import day01.ex00.pred as p


def cost_elem_(theta, X, Y):
    """
    Description:
        Calculates all the elements 0.5*M*(y_pred - y)^2 of the cost function.
    Args:
        theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
        X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
    Returns:
        J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
    Raises:
        This function should not raise any Exception.
    """
    if (not s.elements(theta) or not s.elements(X) or not s.elements(Y)):
        return (None)
    #try:
    return (np.power(np.subtract(p.predict_(theta, X), Y),2) / (2*len(Y)))
    #except:
    #    return (None)

def cost_(theta, X, Y):
    """
    Description:
        Calculates the value of cost function.
    Args:
        theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
        X: has to be a numpy.ndarray, a vector of dimension (number of training examples, number of features).
    Returns:
        J_value : has to be a float.
        None if X does not match the dimension of theta.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return (s.sum_(cost_elem_(theta, X, Y), lambda l: l)[0])
    except:
        return (None)
    
if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
    print(cost_elem_(theta1, X1, Y1))
    print(cost_(theta1, X1, Y1))
    X2 = np.array([[0.2, 2., 20.],
                   [0.4, 4., 40.], 
                   [0.6, 6., 60.],
                   [0.8, 8., 80.]])
    theta2 = np.array([[0.05], [1.], [1.], [1.]])
    Y2 = np.array([[19.], [42.], [67.], [93.]])
    print(cost_elem_(theta2, X2, Y2))
    print(cost_(theta2, X2, Y2))