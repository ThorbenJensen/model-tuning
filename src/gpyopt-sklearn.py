"""Optimizing hyper parameters of sklearn model."""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from GPyOpt.methods import BayesianOptimization


# load data
train = pd.read_csv('data/train.csv')

# preprocessing
discrete_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
continuous_features = ['Age', 'Fare']

train = train[discrete_features + continuous_features + ['Survived']]
train = train.dropna(axis = 0, how = 'any')
train = pd.get_dummies(train, drop_first=True)

X = train.drop('Survived', axis=1)
y = train['Survived']

# EVALUATE MODEL QUALITY
def f(x):
    """Tune model."""
    x[0, 0] = x[0, 0] * 5000
    x[0, 1] = x[0, 1] * 20
    x = x + 1
    print('probing for x = {}'.format(x))
    n_estimators = int(x[0, 0])
    max_depth = int(x[0, 1])
    # print('running model with n_estimators = {}'.format(n_estimators))
    model = RandomForestClassifier(random_state=0,
                                   verbose=0,
                                   n_estimators=n_estimators,
                                   max_depth=max_depth)
    score = np.mean(cross_val_score(model, X, y, cv=4, n_jobs=-1,
                                    scoring='accuracy'))
    # print('Got CV score {}'.format(score))
    return (score)


domain = [{'name': 'n_estimators', 'type': 'continuous', 'domain': (0, 1)},
          {'name': 'max_depth', 'type': 'continuous', 'domain': (0, 1)},
          ]

batch_size = 4
num_cores = 4
opt = BayesianOptimization(f=f,
                           domain=domain,
                           acquisition_type = 'EI',              
                           maximize=True,
                           normalize_Y = True,
                           initial_design_numdata = 10,
                           evaluator_type = 'local_penalization',
                           batch_size = batch_size,
                           num_cores = num_cores,
                           acquisition_jitter = 0
                           )

for i in range(0,1):
    opt.run_optimization(max_iter=4)
    opt.plot_acquisition()
    
opt.plot_convergence()

-np.min(opt.Y)
opt.X[np.argmin(opt.Y)]


