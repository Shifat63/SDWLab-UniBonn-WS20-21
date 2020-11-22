from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt
from random import randint

#GBT Parameters
maxNumberOfTrees = 100
trainingDataFraction = 1.0 #Between (0,1]
learningRate = 0.1
maxTreeDepth = 3

def funcToBeApproximated(x):
    return x*x+4;

def trainGbtClassification(X, y):
    model = GradientBoostingClassifier(n_estimators=maxNumberOfTrees, subsample=trainingDataFraction, learning_rate=learningRate, max_depth=maxTreeDepth)
    model.fit(X, y)
    return model

def trainGbtRegression(X, y):
    model = GradientBoostingRegressor(n_estimators=maxNumberOfTrees, subsample=trainingDataFraction, learning_rate=learningRate, max_depth=maxTreeDepth)
    model.fit(X, y)
    return model

def testGbt(model, X):
    return model.predict(X)

# define dataset
x_values = [randint(0, 100) for i in range(50)]
x_values.sort()
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [funcToBeApproximated(i) for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

x_values = [randint(0, 100) for i in range(50)]
x_values.sort()
x_test = np.array(x_values, dtype=np.float32)
x_test = x_test.reshape(-1, 1)

y_values = [funcToBeApproximated(i) for i in x_values]
y_test = np.array(y_values, dtype=np.float32)
y_test = y_test.reshape(-1, 1)

#Training
model = trainGbtRegression(x_train, y_train)

plt.figure(figsize=(30,12))
prediction = testGbt(model, x_train)
plt.subplot(121)
plt.scatter(x_train, y_train, alpha=0.8, c='red')
plt.plot(x_train, prediction, 'g', label='Predictions', alpha=0.6)
plt.title('Training')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

prediction = testGbt(model, x_test)
plt.subplot(122)
plt.scatter(x_test, y_test, alpha=0.8, c='red')
plt.plot(x_test, prediction, 'g', label='Predictions', alpha=0.6)
plt.title('Test')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

plt.show()