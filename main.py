# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


i = 0
j = 0
flag = False
predict = []
given = []
givenP = []
for line in sys.stdin:
    if flag:
        splitS = line.split(" ")
        predict.append((float(splitS[0]), float(splitS[1])))
        continue
    if i == 0:
        splitS = line.split(" ")
        f = int(splitS[0].strip())
        n = int(splitS[1].strip())
        i+=1
        continue
    
    splitS = line.split(" ")
    if (len(splitS) == 1):
        flag = True
        continue
    else:
        add1 = []
        j = 0
        while (j < len(splitS) -1):
            add1.append(float(splitS[j]))
            j+=1
        given.append(add1)
        givenP.append(float(splitS[len(splitS) -1]))
        
# print(given)
np_array_train = np.array(given)
np_array_test = np.array(predict)
y_train = np.array(givenP)


poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(np_array_train)
X_test_poly = poly.fit_transform(np_array_test)

    
model = linear_model.LinearRegression()
model.fit(X_train_poly, y_train)

preds = model.predict(X_test_poly)

print(*preds)
    
