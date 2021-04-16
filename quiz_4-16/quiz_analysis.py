import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/home/runner/kaggle/quiz_4-16/StudentsPerformance.csv')

print("What are the math scores in the last 3 rows of the data?")
print([df['math score'][i] for i in range(len(df['math score'])-3, len(df['math score']))])
print("\n")
print("What is the average math score across all students?")
print(sum(df['math score'])/len(df['math score']))
print("\n")

def convert_test_prep_to_int(status):
    if status == 'none':
        return 0
    elif status == 'completed':
        return 1

df["test preparation course"] = df["test preparation course"].apply(convert_test_prep_to_int)

print("What were the average math scores for students who did vs didn't complete the test preparation course?")
students_without_test_prep = [df['math score'][i] for i in range(len(df['math score'])) if df["test preparation course"][i] == 1]
students_with_test_prep = [df['math score'][i] for i in range(len(df['math score'])) if df["test preparation course"][i] == 0]

print("With prep: "+str(sum(students_with_test_prep)/len(students_with_test_prep)))
print("Without prep: "+str(sum(students_without_test_prep)/len(students_without_test_prep)))
print("\n")

distinct_educations = []
for x in df["parental level of education"]:
    if x not in distinct_educations:
        distinct_educations.append(x)
print("How many categories of parental level of education are there?")
print(distinct_educations)
print(str(len(distinct_educations))+" categories.")
print("\n")

def convert_parental_education_to_int(edu):
    educations = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
    return educations.index(edu)

df["parental level of education"] = df["parental level of education"].apply(convert_parental_education_to_int)

print("Create dummy variables for test preparation course and parental level of education. Then, fit a linear regression to all the data except for the last 3 rows, and use it to predict the math scores in the last 3 rows of the data. What scores do you get?")

df = df[['math score', 'parental level of education', 'test preparation course']]

df_train = df[:-3]
df_test = df[-3:]

train = np.array(df_train)
test = np.array(df_test)

Y_train = train[:,0]
X_train = train[:,1:]

Y_test = test[:,0]
X_test = test[:,1:]

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


cols = ['Constant']+[x for x in df.columns][1:]
coefs = [regressor.intercept_]+[x for x in regressor.coef_]

print("features: "+str(cols)+"\n")


predictions = regressor.predict(X_test)
print("Predictions")
print(predictions)

#print("\n\tcoefficients: "+str({cols[i]:round(coefs[i],4) for i in range(len(cols))}))