import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


# Importing the dataset

df = pd.read_csv('cancer.csv')
df.replace('?', -99999, inplace=True) # replace the ? values for processing

X = np.array(df.drop(['classes'], 1)) # remove the Maligant or Benign condition from dataset
y = np.array(df['classes']) # store end conditions (Malignant or Benign) in Y
# print(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42) # test size is 20% of database
# print(X_test)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# adding models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluation of the Models
print("Results for the different models are shown below")
print("The first number is the mean and second is standard deviation")
results = []
names = []
first = 0
first_index = 0
second = 0
second_index = 0
index = -1
for name, model in models:
    index = index + 1
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    if(cv_results.mean() > first):
        first_index = index
    if(cv_results.mean() <= first and cv_results.mean() > second):
        second_index = index
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
print()

# Two best algorithms were chosen to evaluate the test set
print("Model 1: " + str(models[first_index]))
# First Model
model = models[first_index][1]
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
print("\n" + models[first_index][0] + " Model Results:\n")
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
# Second Model
print("model 2: " + str(models[second_index]))
model = models[second_index][1]
model.fit(X_train, Y_train)
predictions_2 = model.predict(X_test)
print('\n' + models[second_index][0] + ' Model Results:\n')
print(predictions_2)
print(accuracy_score(Y_test, predictions_2))
print(confusion_matrix(Y_test, predictions_2))
print(classification_report(Y_test, predictions_2))

# Compare Predictions and Predictions_2
comparison = []
differences = []
index = 0
for prediction in predictions:
    if(prediction == predictions_2[index]):
        comparison.append(1)
    else:
        comparison.append(0)
        differences.append(index)
    # print(comparison)
    # print(prediction)
    # print(predictions_2[index])
    index = index + 1
print("Differences: " + str(differences))
print()
# Create a part that can take user input and give a prediction
# Input values necessary for algorithm
"""
id = float(input("ID: "))
clump = float(input("Clump: "))
cell_size = float(input("cell size: "))
cell_shape = float(input("cell shape: "))
marg_adhesion = float(input("marg adhesion: "))
epith_cell_size = float(input('epithelial cell size: '))
bare_nuclei = float(input('bare_nuclei: '))
bland_chrom = float(input('bland chrom: '))
norm_nucleoli = float(input('norm nucleoli: '))
mitoses = float(input("mitsoes: "))

# Make a 2D array from user input
data = np.array([(id, clump, cell_size, cell_shape, marg_adhesion, epith_cell_size, bare_nuclei, bland_chrom, norm_nucleoli, mitoses)])
# df = pd.DataFrame(data)
print(data)
# create test set
X_test = sc.transform(data)
print(X_test)
# used the most accurate learning model
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
# Print predictions
print('\nKNN Model Results:\n')
print(predictions)
"""

# update part above to allow user to input a dataframe
dataframe_id = input("Name of File: ")

df_user = pd.read_csv(dataframe_id)
df_user.replace('?', -99999, inplace=True) # replace the ? values for processing

evaluate_user = sc.transform(df_user)
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(evaluate_user)

# Print Predictions
print('\nKNN Model Results:\n')
print(predictions)

# maybe in the future use the top 2 models code and comparison from up there down here as well
