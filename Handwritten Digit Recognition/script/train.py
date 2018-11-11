import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").as_matrix()

clf = DecisionTreeClassifier()

#training dataset
xtrain = data[0:21000,1:]
train_label = data[0:21000,0]

clf.fit(xtrain,train_label)

#testing data
xtest = data[21000:,1:]
actual_label = data[21000:,0]

#Accuracy

p = clf.predict(xtest)
count = 0
for i in range(0,21000):
    count += 1 if p[i] == actual_label[i] else 0

print("Accuracy= ",(count/21000)*100)

d = xtest[50]
d.shape=(28,28)
plt.imshow(d,cmap='gray')
print(clf.predict([xtest[50]]))
plt.show()
