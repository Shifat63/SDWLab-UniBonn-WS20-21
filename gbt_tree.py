from xgboost import XGBClassifier, XGBRegressor, plot_tree
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def funcToBeApproximated(x):
    return x^3-x^2+4;

def trainGbtClassification(X, y):
    model = XGBClassifier()
    model.fit(X, y)
    return model

def trainGbtRegression(X, y):
    model = XGBRegressor()
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
model = trainGbtClassification(x_train, y_train)
# plot single tree
plot_tree(model)
plt.show()