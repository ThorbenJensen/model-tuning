"""Example for bayesian hyperparameter tuning."""

from GPyOpt.methods import BayesianOptimization
import numpy as np
import math

# define your problem
def f(x):
    return (math.pow(x[0, 0], 2) + x[0, 1])

domain = [{'name': 'x', 'type': 'continuous', 'domain': (-1,1)},
          {'name': 'y', 'type': 'continuous', 'domain': (0,1)}]

# solve problem
opt = BayesianOptimization(f=f, domain=domain)
opt.run_optimization(max_iter=10)
opt.plot_acquisition()
opt.plot_convergence()

# show solution
opt_y = np.min(opt.Y)
opt_x = opt.X[np.argmin(opt.Y)]
f(opt_x)

# parallel computation
opt2 = \
    BayesianOptimization(f=f,  
                         domain = domain,                  
                         acquisition_type = 'EI',              
                         normalize_Y = True,
                         initial_design_numdata = 4,
                         evaluator_type = 'local_penalization',
                         batch_size = 4,
                         num_cores = 4,
                         acquisition_jitter = 0,
                         verbosity = True)

opt2.run_optimization(max_iter=4)
opt2.plot_acquisition()
