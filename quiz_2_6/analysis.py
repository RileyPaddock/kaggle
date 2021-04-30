import pandas as pd
import numpy as np

df = pd.read_csv('/home/runner/kaggle/quiz_2_6/dataset.csv')
print("Mean numner of training hours: ")
print(sum(df['training_hours']/len(df['training_hours'])))
print("\n")

print("What percent of students were looking to change jobs after the course?")
print(sum(df['target']/len(df['target'])))
#This works because the students who were looking to change jobs have value 1 and the others have value 0
print("\n")

print("Which city ID had the most students?")

city_students = {city_id:0 for city_id in df['city'].unique()}
for city in df['city']:
    city_students[city] += 1

best_city = (0,0)
for city in city_students:
    if city_students[city] > best_city[1]:
        best_city = (city, city_students[city])
print(best_city[0])
print("\n")

print("How many students did that city have?")
print(best_city[1])
print("\n")

print("What is the highest city ID?")
def get_id(city_id):
    split = city_id.split("_")
    return int(split[-1])

city_nums = [get_id(city) for city in df['city'].unique()]
print(max(city_nums))

print("\n")

print("How many companies in the data set had fewer than 10 employees?")

x = [1 for i in df['company_size'] if i == '<10']
print(sum(x))
print("\n")

print("How many companies in the data set had fewer than 100 employees?")

def less_than_100(i):
    if i == "<10":
        return True
    elif i == "50-99":
        return True
    elif i == '10/49':
        return True
    else:
        return False

x = [1 for i in df['company_size'] if less_than_100(i)]
print(sum(x))
print("\n")

