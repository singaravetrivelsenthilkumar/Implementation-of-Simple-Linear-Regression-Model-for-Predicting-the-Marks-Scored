# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## step 1:
Import pandas, numpy and sklearn.
## step 2:
Calculate the values for the training data set
## step 3:
Calculate the values for the test data set.
## step 4:
Plot the graph for both the data sets and calculate for MAE, MSE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SINGARAVETRIVEL S
RegisterNumber:  212222220048
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

df.tail()

X,Y=df.iloc[:,:-1].values, df.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression as lr
reg=lr()
reg.fit(Xtrain,Ytrain)
Ypred=reg.predict(Xtest)
print(Ypred)

plt.scatter(Xtrain,Ytrain,color="orange")
plt.plot(Xtrain,reg.predict(Xtrain),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(Xtest,Ytest,color="blue")
plt.plot(Xtest,reg.predict(Xtest),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Ytest,Ypred)
print("MSE= ",mse)

mae=mean_absolute_error(Ytest,Ypred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/12de31bc-efe5-468e-b300-92381a2d6c71)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/a2d17d87-ab45-4b9d-a999-cbb87a9f6d90)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/6cd09eda-6833-41eb-bf9f-8f25d5580894)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/19d37f6d-c10b-43b2-8f08-57e1aa41ff3e)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/f07ee377-a128-45e5-8a02-870e72216dd2)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/6c95449f-3509-4afd-a3ab-3da0005063c7)
### values of MSE, MAE, RMSE:
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/42a30f4a-4bfa-45b7-a7d9-7d262ee43dc0)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/5f1d8d44-f671-4ecf-93fe-ad5fe620045f)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/4511bbb1-d051-49ef-97f3-434130b651ca)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
