import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import utils
from sklearn.neighbors import KNeighborsClassifier

import time



from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

lab_enc = preprocessing.LabelEncoder()

df = pd.read_csv('/home/runner/kaggle/final/flower.csv')


for flower in df['Species'].unique():
    print(flower)
    temp_df = df[df['Species'] == flower]
    for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
        print("\t"+col + ": "+str(sum(temp_df[col])/len(temp_df[col])))



minmax = df[['Species','SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].copy()
for col in minmax.columns:
    if col != 'Species':
        minmax[col] = (minmax[col]-minmax[col].min())/(minmax[col].max() - minmax[col].min())

mimax = minmax.sample(frac=1).reset_index(drop=True)
halfway = len(minmax)//2
df_train = minmax[:halfway]
df_test = minmax[halfway:]

k_vals = [x for x in range(1,50)]

train_y = np.array([y for y in np.array(df_train)[:,0]])
train_x = np.array([[y for y in x] for x in np.array(df_train)[:,1:]])
test_y = np.array([y for y in np.array(df_test)[:,0]])
test_x = np.array([[y for y in x] for x in np.array(df_test)[:,1:]])
prediction_accuracies = []
for k in k_vals:
    KNN = KNeighborsClassifier(n_neighbors = k)
    KNN.fit(train_x,train_y)
    correct = 0
    for i in range(len(test_x)):
        if KNN.predict([test_x[i]]) == test_y[i]:
            correct += 1
    prediction_accuracies.append(correct/len(test_x))

plt.plot(k_vals, prediction_accuracies)
plt.show()

