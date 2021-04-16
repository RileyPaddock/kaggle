import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

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

#interaction terms

df = df[['Sex','Pclass','Fare','Age','SibSp', 'SibSp>0','Parch>0','Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']]

columns = [x for x in df.columns]

for col_1 in columns:
    for col_2 in columns:
        if '=' in col_1 and '=' in col_2:
            eq_i_1 = col_1.index('=')
            eq_i_2 = col_2.index('=')
            if col_1[:eq_i_1] != col_2[:eq_i_2] and columns.index(col_1) < columns.index(col_2):
                df[col_1 + ' * ' + col_2] = np.array([df[col_1][i]*df[col_2][i] for i in range(len(df[col_1]))])
        elif ('SibSp' in col_1 and 'SibSp' in col_2):
            continue
        else:
            if columns.index(col_1) < columns.index(col_2):
                df[col_1 + ' * ' + col_2] = np.array([df[col_1][i]*df[col_2][i] for i in range(len(df[col_1]))])

df['Survived'] = survived
df = df[['Survived']+[x for x in df.columns if x != 'Survived']]


#feature optimization


def run_model(df_train, df_test, pred_type, iter):
    train = np.array(df_train)
    test = np.array(df_test)

    train_y = train[:,0]
    train_x = train[:,1:]

    test_y = test[:,0]
    test_x = test[:,1:]

    regressor = LogisticRegression(max_iter = iter)
    regressor.fit(train_x, train_y)

    test = (test_x,test_y) if pred_type == 'test' else (train_x, train_y)
    predictions = regressor.predict(test[0])

    result = [0,0]
    for i in range(len(predictions)):
        output = 1 if predictions[i]>0.5 else 0
        result[1]+=1
        if output == test[1][i]:
            result[0]+=1
    return result[0]/result[1]

features = ['Survived']
accuracy = 0
while True:
    curr_lvl_max = (None,0)
    for feature in [x for x in df.columns]:
        if feature not in features:
            temp_cols = features + [feature]
            df_train = df[temp_cols][:500]
            df_test = df[temp_cols][500:]
            test_accuracy = run_model(df_train, df_test,'test', 1000)
            if test_accuracy > curr_lvl_max[1]:
                curr_lvl_max = (feature,test_accuracy)
    if curr_lvl_max[1] > accuracy:
        accuracy = curr_lvl_max[1]
        features.append(curr_lvl_max[0])
    else:
        break
print(features)
df_train = df[features][:500]
df_test = df[features][500:]
print(run_model(df_train, df_test, 'train',1000))
print(run_model(df_train, df_test, 'test',1000))



# train = np.array(df_train)
# test = np.array(df_test)

# Y_train = train[:,0]
# X_train = train[:,1:]

# Y_test = test[:,0]
# X_test = test[:,1:]

# regressor = LogisticRegression(max_iter = 10000)
# regressor.fit(X_train, Y_train)

# print("Logistic Regressor: \n")

# cols = ['Constant']+[x for x in df.columns if x != 'Survived']
# coefs = [regressor.intercept_[0]]+[x for x in regressor.coef_[0]]

# for data in [(X_train,Y_train),(X_test,Y_test)]:
#     predictions = regressor.predict(data[0])

#     result = [0,0]
#     for i in range(len(predictions)):
#         output = 1 if predictions[i]>0.5 else 0
#         result[1]+=1
#         if output == data[1][i]:
#             result[0]+=1

#     print("\t"+str(result[0]/result[1]))

# print(len(cols),len(coefs))
# print("\n\tcoefficients: "+str({cols[i]:round(coefs[i],4) for i in range(len(cols))})+"\n")

# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)

# print("\nLinear Regressor: \n")

# cols = ['Constant']+[x for x in df.columns][1:]
# coefs = [regressor.intercept_]+[x for x in regressor.coef_]

# print("\tfeatures: "+str(cols)+"\n")

# for data in [(X_train,Y_train),(X_test,Y_test)]:
#     predictions = regressor.predict(data[0])

#     result = [0,0]
#     for i in range(len(predictions)):
#         output = 1 if predictions[i]>0.5 else 0
#         result[1]+=1
#         if output == data[1][i]:
#             result[0]+=1

#     print("\t"+str(result[0]/result[1]))

# print("\n\tcoefficients: "+str({cols[i]:round(coefs[i],4) for i in range(len(cols))}))