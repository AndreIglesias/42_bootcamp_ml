# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_model.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ciglesia <ciglesia@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/25 13:01:53 by ciglesia          #+#    #+#              #
#    Updated: 2020/11/25 13:01:54 by ciglesia         ###   ########.fr        #
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

data = pd.read_csv("../resources/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)


print(m.mse(linear_model1.predict_(Xpill), Yscore))
# 57.60304285714282
print(mean_squared_error(Yscore, Y_model1))
# 57.603042857142825
print(m.mse(linear_model2.predict_(Xpill), Yscore))
# 232.16344285714285
print(mean_squared_error(Yscore, Y_model2))