import numpy as np
from scipy.optimize import minimize

def f(x):
    return x**2

initial_values = np.arange(-10, 10, 0.5)
bfgs_iters = []
newton_iters = []

for x0 in initial_values:
    res_bfgs = minimize(f, x0, method='BFGS')
    bfgs_iters.append(res_bfgs.nit)
    
    res_newton = minimize(f, x0, method='Newton-CG')
    newton_iters.append(res_newton.nit)

import matplotlib.pyplot as plt

plt.plot(initial_values, bfgs_iters, label='BFGS')
plt.plot(initial_values, newton_iters, label='Newton')
plt.xlabel('Initial Value of x')
plt.ylabel('Iterations')
plt.legend()
plt.title('Comparison of BFGS and Newton Optimization Algorithms')
plt.show()
