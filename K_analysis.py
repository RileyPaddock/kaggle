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

df = pd.read_csv('/home/runner/kaggle/titanic/train.csv')
#data manipulation
def convert_sex_to_int(sex):
    if sex == 'male':
        return 0
    elif sex == 'female':
        return 1

df['Sex'] = df['Sex'].apply(convert_sex_to_int)

ages = [entry for entry in df['Age'] if not np.isnan(entry)]
avg = sum(ages)/len(ages)


df['Age'] = df['Age'].fillna(avg)


def indicator_greater_than_zero(x):
    if x > 0:
        return 1
    else:
        return 0

df['SibSp>0'] = df['SibSp'].apply(indicator_greater_than_zero)


df['Parch>0'] = df['Parch'].apply(indicator_greater_than_zero)

df['Cabin']= df['Cabin'].fillna('None')

def get_cabin_type(cabin):
    if cabin != 'None':
        return cabin[0]
    else:
        return cabin

df['CabinType'] = df['Cabin'].apply(get_cabin_type)

for cabin_type in df['CabinType'].unique():
    df['CabinType='+cabin_type] = df['CabinType'].apply(lambda entry: 1 if entry==cabin_type else 0)

del df['CabinType']


df['Embarked'] = df['Embarked'].fillna('None')

for embarked_loc in df['Embarked'].unique():
    df['Embarked='+embarked_loc] = df['Embarked'].apply(lambda entry: 1 if entry==embarked_loc else 0)

del df['Embarked']

df = df[["Survived", "Sex", "Pclass", "Fare", "Age", "SibSp"]][:100]

# df_book = pd.read_csv('/home/runner/kaggle/book_data.csv')
# def convert_book_type(book_type):
#     if book_type == "children's book":
#         return 0
#     elif book_type == 'adult book':
#         return 1

# df_book['book type'] = df_book['book type'].apply(convert_book_type)

def leave_one_out_validation(x,y, KNN, pred_col,iterations):
    correct = 0
    for i in range(iterations):
        removed_row = x[i]
        removed_result = y[i]
        leave_one_out_y = np.delete(y,i, axis = 0)
        leave_one_out_x = np.delete(x,i,axis = 0)
        
        if KNN.fit(leave_one_out_x, leave_one_out_y).predict([removed_row]) == removed_result:
            correct+=1
        
    return correct/len(df.index)


pred_col = 'Survived'



#Standard Normalization
stnd_norm = df.copy()
for col in stnd_norm.columns:
    if col != pred_col:
        stnd_norm[col] = stnd_norm[col]/stnd_norm[col].max()

#Minmax Normalization
minmax = df.copy()
for col in minmax.columns:
    if col != pred_col:
        minmax[col] = (minmax[col]-minmax[col].min())/(minmax[col].max() - minmax[col].min())

#Z-value Normalization
z_value = df.copy()
for col in z_value.columns:
    if col != pred_col:
        z_value[col] = (z_value[col] - z_value[col].mean())/(z_value[col].std())

begin_time = time.time()
k_vals = [x for x in range(100) if x%2 == 1]
for normal_method in [df, stnd_norm, minmax, z_value]:
    normal_method = normal_method[[pred_col]+[x for x in normal_method.columns if x != pred_col]]
    data_y = np.array([y for y in np.array(normal_method)[:,0]])
    data_x = np.array([[y for y in x] for x in np.array(normal_method)[:,1:]])
    prediction_accuracies = []
    for k in k_vals:
        KNN = KNeighborsClassifier(n_neighbors = k)
        prediction_accuracies.append(leave_one_out_validation(data_x, data_y, KNN, pred_col, len(data_x)))

    plt.plot(k_vals, prediction_accuracies)
end_time = time.time()
print('time taken(relative to Jusin):', 0.15/0.4 * (end_time - begin_time))


plt.legend(["unscaled", "simple scaling", "min-max", "z-score"])
plt.savefig('Titanic KNN accuracy.png')