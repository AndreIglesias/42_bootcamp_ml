# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mylinearregression.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/25 11:57:40 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/25 11:57:44 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
import day00.ex00.sum as s
import day00.ex06.mat_mat_prod  as m
import day00.ex05.mat_vec_prod  as v
import numpy as np

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, theta):
        """
        Description:
            generator of the class, initialize self.
        Args:
            theta: has to be a list or a numpy array, it is a vector of dimension (number of features + 1, 1).
        Raises:
            This method should noot raise any Exception.
        """
        self.theta = np.array(theta)
        
    def predict_(self, X):
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
        if (not s.elements(X) or not s.elements(self.theta)):
            return (None)
        try:
            return (v.mat_vec_prod(np.append(np.full((len(X),1), fill_value=1), X, axis=1), (self.theta)))
        except:
            return (None)
        
    def cost_elem_(self, X, Y):
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
        if (not s.elements(self.theta) or not s.elements(X) or not s.elements(Y)):
            return (None)
        try:
            return (np.power(np.subtract(self.predict_(X), Y),2) / (2*len(Y)))
        except:
            return (None)
        
    def cost_(self, X, Y):
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
            return (s.sum_(self.cost_elem_(X, Y), lambda l: l)[0])
        except:
            return (None)
    
    def fit_(self, X, Y, learning_rate = 0.001, n_cycle = 10000):
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
        if (not s.elements(self.theta) or not s.elements(X) or not s.elements(Y)):
            return (None)
        for _ in range(n_cycle):
            pred = self.predict_(X)
            diff = np.subtract(pred, Y)
            h = np.append(np.full((len(X), 1), fill_value=1), X, axis = 1)
            ss = np.sum(m.mat_mat_prod(h.T,diff),keepdims=True, axis=1)
            self.theta -= learning_rate * (0.5 / len(X)) * ss
        return (self.theta)

if __name__ == "__main__":
    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
    X = np.array([[1., 1., 2., 3.],
                  [5., 8., 13., 21.], 
                  [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    print(mylr.predict_(X))
    print(mylr.cost_elem_(X,Y))
    print(mylr.cost_(X,Y))
    print(mylr.fit_(X, Y, learning_rate = 1.6e-4, n_cycle=200000))
    print(mylr.predict_(X))
    print(mylr.cost_elem_(X,Y))
    print(mylr.cost_(X,Y))
