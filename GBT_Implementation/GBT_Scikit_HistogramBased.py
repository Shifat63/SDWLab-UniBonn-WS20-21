# The primary benefit of the histogram-based approach to gradient boosting is speed.
# These implementations are designed to be much faster to fit on training data.
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt
from random import randint

#GBT Parameters
maxIteration = 100
learningRate = 0.1
maxLeafNode = 31

def funcToBeApproximated(x):
    return 2*x+4;

def trainGbtClassification(X, y):
    model = HistGradientBoostingClassifier(max_iter=maxIteration, learning_rate=learningRate, max_leaf_nodes=maxLeafNode)
    model.fit(X, y)
    return model

def trainGbtRegression(X, y):
    model = HistGradientBoostingRegressor(max_iter=maxIteration, learning_rate=learningRate, max_leaf_nodes=maxLeafNode)
    model.fit(X, y)
    return model

def testGbt(model, X):
    return model.predict(X)

# define dataset
x_values = [randint(0, 100) for i in range(80)]
x_values.sort()
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [funcToBeApproximated(i) for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

x_values = [randint(0, 100) for i in range(10)]
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