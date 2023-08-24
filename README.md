# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ganesh P
RegisterNumber: 212220040112
/*
```
```
/*
import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)
df.head()
df.tail()
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
```

## Output:
initial dataframe :

![m4](https://user-images.githubusercontent.com/94748389/192093643-be9a12e6-5867-477a-9662-6a41bbb248b3.png)
![mld1](https://user-images.githubusercontent.com/94748389/199940097-68c8323e-b4a1-4cfb-8ff1-2ddf1975b96f.png)
![mld2](https://user-images.githubusercontent.com/94748389/199940152-6291ee38-7512-485b-b161-55eca20b83ab.png)

Assigning hours to X and scores to Y :

![m5](https://user-images.githubusercontent.com/94748389/192093653-f3b88dca-c965-4279-a758-98ab506c2ac2.png)

Training set (Hours Vs Scores) :

![m6](https://user-images.githubusercontent.com/94748389/192093668-6d4b3884-8865-4a70-9672-9b0e25c84292.png)

Test set (Hours Vs Scores) :

![m7](https://user-images.githubusercontent.com/94748389/192093680-33a2b35f-58bc-42e7-8a97-8d95f897d4f3.png)

Finding the values of MSE , MAE and RMSE :

![m8](https://user-images.githubusercontent.com/94748389/192093688-56e3f998-4339-4558-9579-0f21e908af62.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
