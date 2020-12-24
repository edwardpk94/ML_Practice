import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pdb

def sklearn_soln(x,y):
    print('starting sklearn model solution')
    model = LinearRegression().fit(x,y)
    print('sklearn score was {}'.format(model.score(x,y)))
    print('sklearn coefficients were {},{}'.format(model.intercept_,model.coef_))
    return model

def matrix_soln(x,y):
    print('starting matrix direct solution')
    # Have to add a column to represent the bias/intercept
    A = np.ones([x.shape[0],x.shape[1]+1])
    A[:,1:] = x
    coef = np.matmul(inv(np.matmul(A.T,A)),np.matmul(A.T,y))
    print('direct matrix solution coefficients were \n{}'.format(coef))
    return coef

def gradient_descent_soln(x,y):
    print('starting gradient descent solution')
    y = y.squeeze()
    A = np.ones([x.shape[0],x.shape[1]+1])
    A[:,1:] = x
    num_iter = 100000
    alpha = 0.001
    print_frequency = 100000
    should_print = False

    theta = np.zeros(A.shape[1])
    theta_history = []

    cost = 0
    cost_history = []
    for iter_count, _ in enumerate(range(num_iter)):
        if ((iter_count % print_frequency) == 0 and should_print):
            print('iteration {} of {}'.format(iter_count, num_iter))
            print('theta = {}'.format(theta))
            print('cost = {}'.format(cost))

        # (n x 2) * (2 x 1) = (n x 1)
        h_x = np.matmul(A, theta)

        # (n x 1)
        error = h_x - y

        # (2 x 100) * (100 x 1) = (2 x 1)
        gradient = np.matmul(A.T, error) / A.shape[0]

        # scalar
        cost = 1/(2*A.shape[0]) * np.dot(error, error)

        # (2 x 1)
        theta = theta - alpha * gradient

        theta_history.append(theta)
    print('gradient descent coefficients were {}'.format(theta))
    return theta

if __name__ == '__main__':
    # X inputs are floats from random distribution. y = x + gaussian noise
    num_samples = 100
    input_range = [10,20]
    np.random.seed = 1
    x = np.random.uniform(input_range[0], input_range[1], [num_samples,1])
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    y = x + np.random.normal(0, 3, x.shape)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x, y)
    ax.set_title('Input Data')

    GD_coefficients = gradient_descent_soln(x,y)
    model = sklearn_soln(x,y)
    matrix_coef = matrix_soln(x,y)

    best_fit_x = np.linspace(input_range[0], input_range[1], x.size)
    best_fit_x = best_fit_x[:, np.newaxis]
    best_fit_y = np.concatenate([np.dot(x_,model.coef_) + model.intercept_ for x_ in best_fit_x])
    ax.plot(best_fit_x, best_fit_y, 'r-')
    # plt.show()
    pdb.set_trace()