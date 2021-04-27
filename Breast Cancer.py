# CASE STUDY: BREAST CANCER CLASSIFICATION
# Dr. Ryan Ahmed
# STEP #1: PROBLEM STATEMENT
# Predicting if the cancer diagnosis is benign or malignant based on several observations/features
#
# 30 features are used, examples:
#
#   - radius (mean of distances from center to points on the perimeter)
#   - texture (standard deviation of gray-scale values)
#   - perimeter
#   - area
#   - smoothness (local variation in radius lengths)
#   - compactness (perimeter^2 / area - 1.0)
#   - concavity (severity of concave portions of the contour)
#   - concave points (number of concave portions of the contour)
#   - symmetry
#   - fractal dimension ("coastline approximation" - 1)
# Datasets are linearly separable using all 30 input features
#
# Number of Instances: 569
#
# Class Distribution: 212 Malignant, 357 Benign
#
# Target class:
#
#    - Malignant
#    - Benign
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)


################################################################

# Importing Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer)

cancer.keys()
print(cancer['DESCR'])
print(cancer['target'])
print(cancer['target_names'])
print(cancer['feature_names'])

print(cancer['data'].shape)

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))

print(df_cancer.head())
print(df_cancer.tail())

## Visualizing the Data

sns.pairplot(df_cancer,hue='target', vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter','mean smoothness'])
# plt.show()
# 0 - malignant

sns.countplot(df_cancer['target'])
# plt.show()
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue='target', data=df_cancer)
# plt.show()
plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot=True)
# plt.show()


## Step 4. MODEL TRAINING (FINDING A PROBLEM SOLUTION)

X = df_cancer.drop(['target'], axis = 1)
print(X)
y = df_cancer['target']
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
print(svc_model.fit(X_train, y_train))

## EVALUATING THE MODEL
y_pred = svc_model.predict(X_test)
print(y_pred)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
# plt.show()

## FEATURE SCALING
min_train = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train - min_train)/range_train
sns.scatterplot(x= X_train['mean area'], y= X_train['mean smoothness'], hue=y_train)
plt.show()
sns.scatterplot(x= X_train_scaled['mean area'], y= X_train_scaled['mean smoothness'], hue=y_train)
plt.show()

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

svc_model.fit(X_train_scaled, y_train)
y_pred = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_pred))


## IMPROVING THE MODEL PART 2
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
print(grid.fit(X_train_scaled, y_train))
print(grid.best_params_)
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, grid_predictions))






