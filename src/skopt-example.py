
import numpy as np
from skopt import gbrt_minimize, forest_minimize, Optimizer
from skopt.plots import plot_convergence
import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn


def f(x):
    return np.power(x[0], 2) + x[1]


f(np.array([1, 2]))

opt = Optimizer(dimensions=[(-10, 10, "uniform"),
                            (-10, 10, "uniform")],
                base_estimator='GP',
                acq_func="EI",
                acq_func_kwargs={'kappa': 5},
                n_initial_points=5,
                random_state=42)

res = opt.run(f, n_iter=50)

opt.Xi
opt.yi
opt.Xi[np.argmin(opt.yi)]

# plot converfence
y = opt.yi
for i in range(0, len(y)):
    for j in range(i, len(y)):
        y[j] = min(y[i], y[j])

pd.DataFrame(y).plot()

# plot optimal parameters
df = pd.DataFrame(opt.Xi)
df['y'] = opt.yi
df = df.sort_values(['y'], ascending=False)

ax = parallel_coordinates(df, 'y', colormap='Reds')
ax.legend_.remove()
