# Basic Operations
import pandas as pd
import numpy as np

# ML Models
from sklearn.linear_model import LogisticRegression

# Feature Engineering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

# Evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score

# Model
import pickle

# data
wine_train = pd.read_csv('wine_train.csv')

# preprocess
poly = PolynomialFeatures(degree = 3, interaction_only=False,include_bias=False)
one_hot = OneHotEncoder(drop = 'first')

transformer = ColumnTransformer([
                                 ('poly',poly,['alcohol','density']),
                                 ('one_hot',one_hot,['fixed acidity level','chlorides level'])
                                ])
# Data Splitting
X = wine_train.drop(columns = 'label')
y = wine_train['label']


# Model Selection
model = LogisticRegression(solver = 'liblinear', random_state = 2020)

estimator = Pipeline([
                      ('preprocess',transformer),
                      ('clf',model)
])

hyperparam_space = {
    'clf__C':[100, 10, 1, 0.1, 0.01, 0.001],
    'clf__solver':['liblinear','newton-cg']
}

skfold = StratifiedKFold(n_splits = 5)

grid_search = GridSearchCV(
    estimator, # model to tune
    param_grid = hyperparam_space, # hyperparameter space
    cv = skfold, # evaluation method
    scoring = 'f1', # metrics
    n_jobs = -1 # use all cores
)
grid_search.fit(X, y)

# Model Pickling
grid_search.best_estimator_.fit(X,y) # FINAL MODEL
filename = 'Model Final.sav'
pickle.dump(grid_search.best_estimator_,open(filename,'wb'))


#Buat dashboard untuk memprediksi data titanic, apakah seseorang survive atau tidak, berdasarkan best model yang kalian punya
