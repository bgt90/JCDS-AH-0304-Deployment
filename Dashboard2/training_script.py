#library

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#ML models
from sklearn.linear_model import LogisticRegression

# Feature engineering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

# Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

#saving models

import pickle
import joblib

# data

df_wine=pd.read_csv('df_wine.csv')

# preprocessing

poly= PolynomialFeatures(degree=3,interaction_only=False, include_bias= False)
one_hot= OneHotEncoder(drop='first')

transformer= ColumnTransformer([
    ('poly', poly,['alcohol','density']),
    ('one hot', one_hot,['fixed_acidity_level','chlorides_level'])
])

# data splitting

X= df_wine.drop('label', axis=1)
y= df_wine['label']

X_train, X_test, y_train,y_test= train_test_split(
    X,y,
    stratify=y,
    random_state=2020
)
# evaluation 

model= LogisticRegression(solver='liblinear', random_state=2020)
estimator=Pipeline([('preprocess', transformer),('model', model)])

hyperparam_space= {
    'model__C':[100, 10, 1, 0.1, 0.01, 0.001],
    'model__solver':['liblinear','newton-cg']
}

skfold= StratifiedKFold(n_splits=5)

grid=GridSearchCV(
    estimator,
    param_grid= hyperparam_space,
    cv= skfold,
    scoring='f1',
    n_jobs=-1
)
grid.fit(X_train,y_train)

# saving model

grid.best_estimator_.fit(X,y) 


file_name='Model Final.sav'

pickle.dump(grid.best_estimator_,open(file_name,'wb'))