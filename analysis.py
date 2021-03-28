import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df = pd.read_csv('/home/runner/kaggle/titanic/train.csv')

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

# Embarked

df['Embarked'] = df['Embarked'].fillna('None')

for embarked_loc in df['Embarked'].unique():
    df['Embarked='+embarked_loc] = df['Embarked'].apply(lambda entry: 1 if entry==embarked_loc else 0)

del df['Embarked']

keep_cols = ['Survived','Sex','Pclass','Fare','Age','SibSp', 'SibSp>0','Parch>0','Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
df_train = df[keep_cols][:500]
df_test = df[keep_cols][500:]



train = np.array(df_train)
test = np.array(df_test)

Y_train = train[:,0]
X_train = train[:,1:]

Y_test = test[:,0]
X_test = test[:,1:]

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

cols = ['Constant']+keep_cols[1:]
coefs = [regressor.intercept_]+[x for x in regressor.coef_]

print({cols[i]:round(coefs[i],4) for i in range(len(cols))})

for data in [(X_test,Y_test),(X_train,Y_train)]:
    predictions = regressor.predict(data[0])

    result = [0,0]
    for i in range(len(predictions)):
        output = 1 if predictions[i]>0.5 else 0
        result[1]+=1
        if output == data[1][i]:
            result[0]+=1

    print(result[0]/result[1])