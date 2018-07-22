# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:54:07 2018

@author: sankaran
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
x=np.array([[1,9],[2,8],[3,7],[4,6],[5,5],[6,4],[7,3],[8,2],[9,1],[10,0]])

y=np.array([1,1,1,1,1,0,0,0,0,0])

mlp=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto',
       early_stopping=False, 
       hidden_layer_sizes=(10), learning_rate='constant',
       learning_rate_init=0.1, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=10,
       warm_start=False)

mlp.fit(x,y)
g=int(input("Enter number of grey hair:"))
b=int(input("Enter number of black hair:"))

p=np.array([[g,b]])

predictions = mlp.predict(p)
if (predictions==1):
    print('younger than 40')
else:
    print("older than 40")     
