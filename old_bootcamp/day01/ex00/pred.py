#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sum.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/22 20:54:48 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/23 17:31:23 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Created on Tue Nov 24 11:34:43 2020

@author: ciglesia
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
import day00.ex00.sum as s
import day00.ex05.mat_vec_prod  as v
import numpy as np


def predict_(theta, X):
    """
    Description:
        Prediction of output using the hypothesis function (linear model).
        Args:
            theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
            X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
        Returns:
            pred: numpy.ndarray, a vector of dimension (number of the training examples,1).
            None if X does not match the dimension of theta.
        Raises:
            This function should not raise any Exception.
    """
    if (not s.elements(X) or not s.elements(theta)):
        return (None)
    try:
        return (v.mat_vec_prod(np.append(np.full((len(X),1), fill_value=1), X, axis=1), (theta)))
    except:
        return (None)
    
    
if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    print(predict_(theta1, X1))
    X2 = np.array([[1], [2], [3], [5], [8]])
    theta2 = np.array([[2.]])
    print(predict_(theta2, X2))
    X3 = np.array([[0.2, 2., 20.],
                   [0.4, 4., 40.], 
                   [0.6, 6., 60.],
                   [0.8, 8., 80.]])
    theta3 = np.array([[0.05], [1.], [1.], [1.]])
    print(predict_(theta3, X3))