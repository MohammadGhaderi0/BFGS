import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x):
    '''
    FUNCTION TO BE OPTIMISED
    '''
    d = len(x)
    return sum(100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(d-1))

def grad_f(x):
    '''
    GRADIENT OF THE FUNCTION TO BE OPTIMISED
    '''
    d = len(x)
    grad = np.zeros(d)
    for i in range(d-1):
        grad[i] = -400 * (x[i+1] - x[i]**2) * x[i] + 2 * (x[i] - 1)
        grad[i+1] += 200 * (x[i+1] - x[i]**2)
    return grad

def hessian_f(x):
    '''
    HESSIAN MATRIX OF THE FUNCTION TO BE OPTIMISED
    '''
    d = len(x)
    hess = np.zeros((d, d))
    for i in range(d-1):
        hess[i, i] = 1200 * x[i]**2 - 400 * x[i+1] + 2
        hess[i, i+1] = -400 * x[i]
        hess[i+1, i] = -400 * x[i]
        hess[i+1, i+1] = 200
    return hess

def callback_store(xk, store=[]):
    store.append(np.copy(xk))
    return False

def newton_scipy(f, x0, max_it):
    '''
    Newton's Method using SciPy's built-in functions for optimization.
    '''
    callback_store.__defaults__ = ([],)
    result = minimize(f, x0, method='Newton-CG', jac=grad_f, hess=hessian_f,
                      options={'maxiter': max_it}, callback=callback_store)
    
    x_store = np.array(callback_store.__defaults__[0])
    return result.x, x_store

def BFGS(f, x0, max_it):
    '''
    BFGS Quasi-Newton Method using SciPy's built-in functions for optimization.
    '''
    callback_store.__defaults__ = ([],)
    result = minimize(f, x0, method='BFGS', jac=grad_f,
                      options={'maxiter': max_it}, callback=callback_store)
    
    x_store = np.array(callback_store.__defaults__[0])
    return result.x, x_store

# Compare BFGS and Newton methods
x0 = [-1.2, 1]
max_it = 50

print("Running BFGS method...")
x_opt_bfgs, x_store_bfgs = BFGS(f, x0, max_it)
print("Optimal solution (BFGS):", x_opt_bfgs)

print("Running Newton method...")
x_opt_newton, x_store_newton = newton_scipy(f, x0, max_it)
print("Optimal solution (Newton):", x_opt_newton)

# Plotting both methods on the same chart
if len(x0) == 2:
    x1 = np.linspace(min(np.min(x_store_bfgs[:,0]), np.min(x_store_newton[:,0])) - 0.5,
                     max(np.max(x_store_bfgs[:,0]), np.max(x_store_newton[:,0])) + 0.5, 30)
    x2 = np.linspace(min(np.min(x_store_bfgs[:,1]), np.min(x_store_newton[:,1])) - 0.5,
                     max(np.max(x_store_bfgs[:,1]), np.max(x_store_newton[:,1])) + 0.5, 30)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([[f([x1, x2]) for x1 in x1] for x2 in x2])
    
    plt.figure()
    plt.title('Optimization Paths of BFGS and Newton-CG')
    plt.contourf(X1, X2, Z, 30, cmap='jet')
    plt.colorbar()
    
    # Plot BFGS path
    plt.plot(x_store_bfgs[:,0], x_store_bfgs[:,1], 'w', label='BFGS', marker='o', color='blue')
    plt.scatter(x_store_bfgs[:,0], x_store_bfgs[:,1], color='blue')
    
    # Plot Newton-CG path
    plt.plot(x_store_newton[:,0], x_store_newton[:,1], 'w', label='Newton-CG', marker='x', color='red')
    plt.scatter(x_store_newton[:,0], x_store_newton[:,1], color='red')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.show()
