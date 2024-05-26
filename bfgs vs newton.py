import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define a more complex quadratic function and its gradient
def func(x):
    return 0.5 * np.dot(x.T, np.dot(A, x))

# Gradient of the function
def grad(x):
    return np.dot(A, x)

# Hessian of the function (constant for quadratic functions)
def hess(x):
    return A

# Initialize parameters
np.random.seed(0)
A = np.array([[3, 2], [2, 6]])  # A symmetric positive-definite matrix
x0 = np.array([100.0, -10.0])  # Starting point
tolerance = 1e-6
max_iter = 50

# Newton's Method
def newton_method(x0, grad, hess, tol, max_iter):
    x = x0
    errors = []
    for i in range(max_iter):
        grad_x = grad(x)
        hess_x = hess(x)
        step = np.linalg.solve(hess_x, -grad_x)
        x = x + step
        error = np.linalg.norm(grad_x)
        errors.append(error)
        if error < tol:
            break
    return errors

# BFGS Method
def bfgs_method(x0, func, grad, tol, max_iter):
    result = minimize(func, x0, method='BFGS', jac=grad, tol=tol, options={'maxiter': max_iter, 'disp': False, 'return_all': True})
    errors = result.allvecs  # Get all intermediate vectors
    errors = [np.linalg.norm(grad(x)) for x in errors]  # Compute the norm of the gradient at each step
    return errors

# Run both methods
newton_errors = newton_method(x0, grad, hess, tolerance, max_iter)
bfgs_errors = bfgs_method(x0, func, grad, tolerance, max_iter)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(newton_errors, label="Newton's Method", marker='o')
plt.plot(bfgs_errors, label='BFGS Method', marker='s')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm (Log Scale)')
plt.title('Convergence Comparison: Newton\'s Method vs. BFGS Method')
plt.legend()
plt.grid(True)
plt.show()
