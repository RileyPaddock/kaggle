import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

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


def leave_one_out_validation(df, k_val):
    correct = 0
    for i in range(len(df.index)):
        removed_row = df.loc[i]
        leave_one_out_df = df.drop(index = i)
        data = np.array(leave_one_out_df)
        data_y = [y for y in data[:,0]]
        data_x = [[y for y in x] for x in data[:,1:]]
        KNN = KNeighborsClassifier(n_neighbors = k_val)
        KNN.fit(data_x, data_y)

        test_y = np.array(removed_row)[0]
        test_x = [x for x in np.array(removed_row)][1:]
        if KNN.predict([test_x]) == test_y:
            correct+=1
        
    return correct/len(df.index)

#Standard Normalization
stnd_norm = df.copy()
for col in stnd_norm:
    stnd_norm[col] = stnd_norm[col]/stnd_norm[col].max()

#Minmax Normalization
minmax = df.copy()
for col in minmax:
    minmax[col] = (minmax[col]-minmax[col].min())/(minmax[col].max() - minmax[col].min())

#Z-value Normalization
z_value = df.copy()
for col in z_value:
    z_value[col] = (z_value[col] - z_value[col].mean())/z_value[col].std()

k_vals = [1,3,5,10,15,20,30,40,50,75]
prediction_accuracies = []
for k in k_vals:
    prediction_accuracies.append(leave_one_out_validation(df, k))
    print(k)

plt.plot([x for x in range(len(k_vals))],prediction_accuracies)

plt.show()