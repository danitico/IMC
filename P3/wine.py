#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('basesDatosPr3IMC/wine.csv', header=None)
data = df.to_numpy(dtype=np.float32)

data_inputs = data[:, 1:]
data_outputs = data[:, 0]

scaler = MinMaxScaler()
data_inputs = scaler.fit_transform(data_inputs)


X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_outputs, test_size=0.4, stratify=data_outputs,
                                                    random_state=42)


for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)*100
    scoreTest = knn.score(X_test, y_test)*100

    print(i, " vecinos: CCR Train = ", scoreTrain, ", CCR Test = ", scoreTest)
    print(confusion_matrix(y_test, knn.predict(X_test)))


for j in [1e-1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]:
    logreg = LogisticRegression(solver='liblinear', C=j)
    logreg.fit(X_train, y_train)

    scoreTrain = logreg.score(X_train, y_train)*100
    scoreTest = logreg.score(X_test, y_test)*100

    print("C = ", j, ": CCR Train = ", scoreTrain, ", CCR Test = ", scoreTest)
    print(confusion_matrix(y_test, logreg.predict(X_test)))
