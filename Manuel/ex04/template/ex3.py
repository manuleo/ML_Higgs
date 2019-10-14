import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    D = tx.shape[1]
    G = tx.T.dot(tx)
    if(np.linalg.matrix_rank(G)==D):
        w = np.linalg.inv(G).dot(tx.T).dot(y)
    else:
        w = np.linalg.solve(G,tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    train_size = round(x.shape[0] * ratio)
    indexes = np.arange(x.shape[0])
    np.random.shuffle(indexes)
    
    train_indexes = indexes[:train_size]
    test_indexes = np.setdiff1d(indexes, train_indexes)
    
    x_train = x[train_indexes]
    y_train = y[train_indexes]
    
    x_test = x[test_indexes]
    y_test = y[test_indexes]
    
    return x_train, x_test, y_train, y_test


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    G = tx.T.dot(tx)
    i = np.linalg.inv(G + 2*N*lambda_*np.eye(G.shape[0]))
    w_star = i.dot(tx.T).dot(y)
    return w_star


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    degrees = np.arange(0, degree+1)
    psi = np.power(x[:, np.newaxis], degrees)
    #psi = np.power(x[:, :, np.newaxis], degrees) multidimensions
    return psi