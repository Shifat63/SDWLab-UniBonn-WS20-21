import numpy as np
import DiabetesNN as nn
import DiabetesGBT as gbt
import GetTrainAndTestData as data

train_dl, test_dl = data.prepare_data('diabetes_data_preprocessed.csv')
print('Training ', len(train_dl.dataset))
print('Test ', len(test_dl.dataset))

# define the NN
model = nn.MLP(16)

# train the NN
nn.train_model(train_dl, model)

# Generate soft labels from NN
xinputs, predictions = nn.get_soft_labels(train_dl, model)

# test the NN
acc = nn.evaluate_model(test_dl, model)
print('NN Accuracy: %.3f' % (acc*100.0))

# Train GBT on the soft labels
gbtModel = gbt.trainXGbtClassification(xinputs, predictions)

xTest, yTest = [], []
for i, (inputs, targets) in enumerate(test_dl):
    xTest.append(inputs.numpy().flatten())
    yTest.append(targets.numpy().flatten())

# Test GBT
acc = gbt.testGbt(gbtModel, np.array(xTest), yTest)
print('GBT(only soft labels) Accuracy: %.3f' % (acc*100.0))
# gbt.showTree(gbtModel)


xTrain, yTrain = [], []
for i, (inputs, targets) in enumerate(train_dl):
    xTrain.append(inputs.numpy().flatten())
    yTrain.append(targets.item())
xTrain = np.array(xTrain)
yTrain = np.array(yTrain)


# Train GBT on the hard labels
gbtModel2 = gbt.trainXGbtClassification(xTrain, yTrain)
# Test GBT
acc = gbt.testGbt(gbtModel2, np.array(xTest), yTest)
print('GBT(only hard labels) Accuracy: %.3f' % (acc*100.0))
# gbt.showTree(gbtModel2)


# xTrain = np.concatenate((xTrain, xinputs), axis=0)
# yTrain = np.concatenate((yTrain, predictions), axis=0)
# # Train GBT on the soft and hard labels
# gbtModel3 = gbt.trainXGbtClassification(xTrain, yTrain)
# # Test GBT
# acc = gbt.testGbt(gbtModel3, np.array(xTest), yTest)
# print('GBT(hard and soft labels mixed) Accuracy: %.3f' % (acc*100.0))
# # gbt.showTree(gbtModel3)
