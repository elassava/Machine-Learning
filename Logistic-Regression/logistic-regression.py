import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


datas = pd.read_csv("data.csv")
x = datas.iloc[:,1:4].values #independent variables
y = datas.iloc[:,4:].values #dependent variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) #split the data into test and train

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)

print(y_pred)
print(y_test)