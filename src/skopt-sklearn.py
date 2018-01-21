"""Sklearn example wit skopt."""

import numpy as np
from skopt import Optimizer
from skopt.plots import plot_convergence
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


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
    x0 = (x[0] * 5000) + 1
    x1 = (x[1] * 20) + 1
    print('probing for x = {}'.format(x))
    n_estimators = int(x0)
    max_depth = int(x1)
    # print('running model with n_estimators = {}'.format(n_estimators))
    model = RandomForestClassifier(random_state=0,
                                   verbose=0,
                                   n_estimators=n_estimators,
                                   max_depth=max_depth)
    score = np.mean(cross_val_score(model, X, y, cv=4, n_jobs=-1,
                                    scoring='accuracy'))
    print('Got CV score {}'.format(score))
    return (-score)

# f(np.array([1, 2]))

opt = Optimizer(dimensions=[(0, 1, "uniform"),
                            (0, 1, "uniform")],
                base_estimator='GP',
                acq_func="EI",
                acq_func_kwargs={'kappa': 5},
                n_initial_points=3,
                random_state=42)

res = opt.run(f, n_iter=32)

opt.Xi
opt.yi

# plot convergence
y = opt.yi
for i in range(0, len(y)):
    for j in range(i, len(y)):
        y[j] = min(y[i], y[j])
pd.DataFrame(y).plot()

# plot optimal parameters
df = pd.DataFrame(opt.Xi)
df['y'] = opt.yi
df = df.sort_values(['y'], ascending=False)

ax = parallel_coordinates(df, 'y', alpha=0.5, colormap='Reds')
ax.legend_.remove()

# get optimum
opt.Xi[np.argmin(opt.yi)]
np.min(opt.yi)
