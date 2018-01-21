"""TESTING THE MOE PACKAGE:"""

# from moe.easy_interface.experiment import Experiment
# from moe.easy_interface.simple_endpoint import gp_next_points

from hyperopt import fmin, tpe, hp

best = fmin(fn=lambda x: x ** 2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=10)

print(best)

# RESULT: unfortunately, hyperopt only support parzen tree estimator
# no combinations between parameters are modeled
