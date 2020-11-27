# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multi_linear_model.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/26 00:02:39 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/27 13:21:19 by ciglesia         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
from day01.ex03.mylinearregression import MyLinearRegression as MyLR
import day00.ex07.mse as m
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

import matplotlib.pyplot as pyplot


def mse_(X, Y):
    return (np.sum(np.power(np.subtract(X,Y), 2))) * (1 / len(X))


data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']]).reshape(-1,1)


print("age")
Xage = np.array(data[['Age']]).reshape(-1,1)
myLR_age = MyLR([[1000.0], [-1.0]])
print("fitting...")
myLR_age.fit_(Xage, Y, learning_rate = 2.5e-5, n_cycle = 1000)#00
y_model1 = myLR_age.predict_(Xage)
RMSE_age = m.mse(X[:,0].reshape(-1,1), Y)
print(RMSE_age)

print("plotting...")

pyplot.scatter(Xage, Y, label='age')
pyplot.scatter(Xage, y_model1, color='g')
pyplot.xlabel('age (in years)')
pyplot.ylabel('sell price (in keuros)')
pyplot.legend()
pyplot.show()


print("thrust power")
Xthrust = np.array(data[['Thrust_power']]).reshape(-1,1)
myLR_thrust = MyLR([[1000.0], [-1.0]])
print("fitting...")
myLR_thrust.fit_(Xthrust, Y, learning_rate = 2.5e-5, n_cycle = 10000)#0
y_model2 = myLR_thrust.predict_(Xthrust)

RMSE_thrust = m.mse(Xthrust, Y)
print(RMSE_thrust)

print("plotting...")

pyplot.scatter(Xthrust, Y, label='thrust', color='g')
pyplot.scatter(Xthrust, y_model2, color='y')
pyplot.xlabel('thrust power (in 10 Km/s)')
pyplot.ylabel('sell price (in Keuros')
pyplot.legend()
pyplot.show()

Xtera = np.array(data[['Terameters']]).reshape(-1,1)
myLR_tera = MyLR([[1000.0], [-1.0]])
print("fitting...")
myLR_tera.fit_(Xtera, Y, learning_rate = 2.5e-5, n_cycle = 10000)#0
y_model3 = myLR_tera.predict_(Xtera)

RMSE_tera = m.mse(Xtera, Y)
print(RMSE_tera)

print("plotting...")

pyplot.scatter(Xtera, Y, label='tmeters', color='purple')
pyplot.scatter(Xtera, y_model3, color='r')
pyplot.xlabel('distance totalizer value of spacecraft (in Tmeters)')
pyplot.ylabel('sell price (in Keuros')
pyplot.legend()
pyplot.show()
