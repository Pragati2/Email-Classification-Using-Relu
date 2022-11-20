import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_spam_dat():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    dat = pd.read_csv(url, sep=',', header=None) 
    X = dat.iloc[:,:-1]
    Y = dat.iloc[:, -1:]
    X_train,X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.27, random_state= 47)
    return(X_train,y_train,X_test,y_test)

x_train, y_train,x_test,y_test = load_spam_dat()
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

batch_size = 40
epochs = 20
model_name = 'email_spam_classifier_trained_model.h5'

model = Sequential()
model.add(Dense(57, input_dim = 57, activation = 'relu'))
model.add(Dense(1,activation = 'relu')) #since this is binary class classification we can use sigmoid as activation function here
model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
print()
#model.add(Dense(s))

model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, shuffle = True)
