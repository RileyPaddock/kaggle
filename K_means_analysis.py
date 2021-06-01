import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import utils
from sklearn.cluster import KMeans

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

survived = df['Survived']
df = df[["Sex", "Pclass", "Fare", "Age", "SibSp"]]

def calc_euclidian_distance(p1, p2):
        return sum([(p1[i] - p2[i])**2 for i in range(len(p1))])**0.5

train_data = np.array(df)
x_coords = [x for x in range(1,26)]
y_coords = []
for k in [x for x in range(1,26)]:
    k_means = KMeans(n_clusters = k).fit(train_data)
    distance = 0
    for x in range(len(train_data)):
        distance += calc_euclidian_distance(k_means.cluster_centers_[k_means.labels_[x]],train_data[x])
    y_coords.append(distance)
best_k_means = KMeans(n_clusters = 4).fit(train_data)
df['cluster'] = [best_k_means.labels_[i] for i in range(len(train_data))]
df['Survived'] = survived
count = [[x for x in df['cluster']].count(i) for i in df['cluster'].unique()]
df = df.groupby(['cluster']).mean() 
df['count'] = count

#df['count'] = [df['cluster'].count(x) for x in df['cluster'].unique()]

print(df)

plt.plot(x_coords, y_coords)
plt.xlabel('k')
plt.ylabel('sum_squared_error')
plt.title('Best size k')
plt.savefig('Titanic_Elbow_Method.png')