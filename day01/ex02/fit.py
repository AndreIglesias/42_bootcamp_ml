# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/24 17:56:57 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/24 17:57:48 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #
   
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
import day00.ex00.sum as s
import day00.ex06.mat_mat_prod  as m
import day01.ex00.pred as p
import day01.ex01.cost_function as c

def fit_(theta, X, Y, learning_rate = 0.001, n_cycle = 10000):
    """
    Description:
        Performs a fit of Y(output) with respect to X.
    Args:
        theta: has to be a numpy.ndarray, a vector of dimension (number of features + 1, 1).
        X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
        Y: has to be a numpy.ndarray, a vector of dimension (number of training examples, 1).
    Returns:
        new_theta: numpy.ndarray, a vector of dimension (number of the features +1,1).
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if (not s.elements(theta) or not s.elements(X) or not s.elements(Y)):
        return (None)
    for _ in range(n_cycle):
        pred = p.predict_(theta, X)
        diff = np.subtract(pred, Y)
        h = np.append(np.full((len(X), 1), fill_value=1), X, axis = 1)
        ss = np.sum(m.mat_mat_prod(h.T,diff),keepdims=True, axis=1)
        theta -= learning_rate * (0.5 / len(X)) * ss
    return (theta)
    
if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
    theta1 = np.array([[1.], [1.]])
    theta1 = fit_(theta1, X1, Y1, learning_rate = 0.01, n_cycle=2000)
    print(theta1)
    print(p.predict_(theta1, X1))
    X2 = np.array([[0.2, 2., 20.],
                   [0.4, 4., 40.], 
                   [0.6, 6., 60.],
                   [0.8, 8., 80.]])
    Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta2 = np.array([[42.], [1.], [1.], [1.]])
    theta2 = fit_(theta2, X2, Y2, learning_rate = 0.0005, n_cycle=42000)
    print(theta2)
    print(p.predict_(theta2, X2))
