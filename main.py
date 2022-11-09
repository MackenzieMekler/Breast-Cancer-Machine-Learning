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
y = np.array(df['classes']) # store end conditions in y
# print(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)
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

# Evaluation
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Two best algorithms were chosen to evaluate the test set
# SVC
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
print("\nSVC Model Results:\n")
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
# KNN
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
print('\nKNN Model Results:\n')
print(predictions)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

# Create a part that can take user input and give a prediction
# Input values necessary for algorithm
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