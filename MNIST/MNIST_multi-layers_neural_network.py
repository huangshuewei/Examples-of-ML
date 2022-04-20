# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:39:43 2022

Example of Digit Recogniser.
Dataset: MNIST
Tools: Tensorflow and Keras
Classifier: Multi-layer neural network.

@author: Shu-wei Huang
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam

'''
Read data.
MNIST dataset.
'''
train = pd.read_csv('C:/Users/User/Desktop/Programming/Python/Data/digit-recognizer/train.csv')
Test = pd.read_csv('C:/Users/User/Desktop/Programming/Python/Data/digit-recognizer/test.csv')

train_y=train['label']
train_x=train.drop(labels = ["label"],axis = 1)

'''
Show a part of data.
'''
plt.figure(figsize=(20, 10))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    xx = train_x.iloc[i].to_numpy()
    plt.imshow(xx.reshape((28,28)), cmap='gray')
    plt.title(train_y[i],size = 20)
plt.show()

'''
Transform data variances to arrays.
'''
train_y = train_y.to_numpy()
train_x = train_x.to_numpy()/255
test = Test.to_numpy()/255



train_x, test_x, train_y, test_y = train_test_split(train_x, 
                                                    train_y, 
                                                    test_size = 0.3)

train_x, val_x, train_y, val_y = train_test_split(train_x, 
                                                    train_y, 
                                                    test_size = 0.3)

train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)
val_y_true = val_y
val_y = np_utils.to_categorical(val_y)

'''
Built a model of multi-layer neural network.
'''
# network parameters
batch_size = 128
hidden_units_1 = 256
hidden_units_2 = 128
dropout = 0.5

model = Sequential()
model.add(Dense(hidden_units_1,
                input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units_2))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# Compilation of model can be done
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_x, 
                    train_y, 
                    epochs=30, 
                    batch_size=batch_size)

# Evaluating model performance
loss, acc = model.evaluate(test_x, 
                           test_y, 
                           batch_size=batch_size)

print('\n\nLoss     : {} \nAccuracy : {}'.format(history.history['loss'][-1],history.history['accuracy'][-1]))

'''
Evaluate on validation data
'''
Y_pred = model.predict(val_x)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.array(val_y_true)
rows = 4
cols = 6
f = plt.figure(figsize=(2*cols,2*rows))
print("Validation Predictions")
for i in range(rows*cols): 
    f.add_subplot(rows,cols,i+1)
    img = val_x[i+100]
    img = img.reshape((28,28))
    plt.imshow(img,
               cmap='gray')
    plt.axis("off")
    if Y_pred_classes[i+100] != Y_true[i+100]:
        plt.title("Prediction: {}\nTrue Value: {}".format(Y_pred_classes[i+100], Y_true[i+100]),
                  y=-0.35,color="red")
    else:
        plt.title("Prediction: {}\nTrue Value: {}".format(Y_pred_classes[i+100], Y_true[i+100]),
                  y=-0.35,color="green")
    
f.tight_layout()
f.show()

# Show confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

f,ax = plt.subplots(figsize=(6,6))
sns.heatmap(confusion_mtx, annot=True,
            linewidths=3,cmap="viridis",
            fmt= '.0f',ax=ax,
            cbar = False,
           annot_kws={"size": 16})
plt.yticks(rotation = 0)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix", size = 14)
plt.show()
