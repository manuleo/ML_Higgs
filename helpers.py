"""All helpers function"""
# -*- coding: utf-8 -*-
import numpy as np
import csv
from implementation_helpers import standardize, compute_loss, sigmoid
from implementations import ridge_regression

# GENERAL HELPERS FUNCTIONS
  
def y_for_logistic(y):
    """
    this function changes the -1 in 0 in the y vector to use it in the logistic regression
    """
    y_new = np.where(y==-1, 0, y)
    return y_new


def accuracy(y_real, y_pred):
    """
    compute the accuracy: percentage of correct predictions over the total number of predictions
    INPUTS: y_real := the y vector of true labels
            y_pred := the y vector of the predicted labels
    OUTPUTS: accuracy
    """
    N = len(y_real)
    if N!=len(y_pred):
        raise Exception('y_pred length wrong')
    correct = np.sum(y_real==y_pred)
    return correct/N
        

def split_data(y, x, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    N = x.shape[0]
    
    # produce an indexes vector and randomize it
    size_tr = round(N* ratio)
    index = np.arange(N)
    np.random.shuffle(index)
    
    # select indexes for train and test
    ind_tr = index[:size_tr]
    ind_te= np.setdiff1d(index, ind_tr)   
    
    # produce train and test
    x_train = x[ind_tr]
    y_train = y[ind_tr]
    x_test = x[ind_te]
    y_test = y[ind_te]
    
    return x_train, x_test, y_train, y_test


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=1 up to j=degree.
    INPUTS: x := the matrix
    OUTPUTS: the matrix with a new column for each degree and for each column of the old dataset
    Note: this modified version doesn't take care of the degree 0 because we already add a 1s column that will
    be repeated
    """
    d = np.arange(1, degree+1).repeat(x.shape[1])
    psi = np.tile(x, degree)
    psi = np.power(psi, d)
    return psi


# RUN HELPERS FUNCTION

def preprocessing(tX, y=[]):
    """
    This function executes the various steps for the cleaning and preparation of the dataset Higgs' Boson.
    INPUTS: X
            y (not mandatory): if present prduce test samples, if not doesn't build a y

    
    if len(y)!=0 (train set):
    OUTPUTS:y_new
            tX_new
            ids_new
            means 
            stds
    
    if len(y)==0 (test set):
    OUTPUTS:x_train 
            x_test 
            y_train 
            y_test

    """
    
    #build index to reconstruct later
    index = np.array(range(0,tX.shape[0])).reshape((tX.shape[0],1))
    #append index column at the end of tX
    tX = np.append(tX, index, axis=1)
    
    #divide based in jet num (col 22)
    tX_new1 = []
    for jet in range(0, 4):
        tX_new1.append(tX[tX[:,22] == jet])
    
    # dropping
    # we drop the column 22 in each "jet"
    for jet in range(0, 4):
        tX_new1[jet] = np.delete(tX_new1[jet],22,1) #drop jet num from all
    drops_0 = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28] # 28 is all zeros, others all NaN
    drops_1 = [4, 5, 6, 12, 25, 26, 27] #all NaN
    tX_new1[0] = np.delete(tX_new1[0],drops_0,1)
    tX_new1[1] = np.delete(tX_new1[1],drops_1, axis=1)
    
    
    # second diviion based on null in MASS_MMC column
    tX_new = []
    for jet in range(0, 4):
        indexes_nan = np.where(tX_new1[jet][:,0] == -999)
        indexes_not_nan = np.where(tX_new1[jet][:,0] != -999)
        tX_new.append(tX_new1[jet][indexes_nan])
        tX_new.append(tX_new1[jet][indexes_not_nan])
    
    #drop all NaN mass column
    for jet in range(0, 8):
        if (jet%2==0):
            tX_new[jet] = np.delete(tX_new[jet],0,1)
            
      
    #new datasets
    if len(y)!=0: #we are processing train dataset
        y_new = []
        for jet in range(0, 8):
            #generate y: use index column to save the correct y for each jet and use them for test later
            y_new.append(y[tX_new[jet][:,tX_new[jet].shape[1]-1].astype(int)])
    
    ids_new = []
    for jet in range(0, 8):
        ids_new.append(tX_new[jet][:,tX_new[jet].shape[1]-1].astype(int)) #save indexes for each jet to reconstruct later
        
    for jet in range(0,8):
        tX_new[jet] = np.delete(tX_new[jet],tX_new[jet].shape[1]-1,1) #delete index column
    
    #new 1s column
    for jet in range(0,8):
        tX_new[jet] = np.c_[np.ones(tX_new[jet].shape[0]), tX_new[jet]]
        
    #new processing discovered after visualizing data:
    #applying cube roots to skewed columns and delete columns with correlation 1
    
    cub0 = [1, 2, 4, 5, 6, 7, 8, 9, 12, 15, 17]
    tX_new[0][:, cub0] = np.cbrt(tX_new[0][:, cub0])
    tX_new[0] = np.delete(tX_new[0], 3, 1)
    
    cub1 = [1, 2, 3, 6, 7, 8, 9, 13, 16, 18]
    tX_new[1][:, cub1] = np.cbrt(tX_new[1][:, cub1])
    tX_new[1] = np.delete(tX_new[1], 4, 1)
    
    cub2 = [1,2,3,5,6,7,8,9,12,15,17,18]
    tX_new[2][:, cub2] = np.cbrt(tX_new[2][:, cub2])
    tX_new[2] = np.delete(tX_new[2], 21, 1)
    
    cub3 = [1,2,3,4,6,7,8,10,13,16,18,19,22]
    tX_new[3][:, cub3] = np.cbrt(tX_new[3][:, cub3])
    tX_new[3] = np.delete(tX_new[3], 22, 1)
    
    cub4 = [1,2,3,5,8,9,10,13,16,19,22,25,28]
    tX_new[4][:, cub4] = np.cbrt(tX_new[4][:, cub4])
    
    cub5 = [2,3,4,6,9,10,11,14,17,20,23,26,29]
    tX_new[5][:, cub5] = np.cbrt(tX_new[5][:, cub5])
    
    cub6 = [1,2,3,5,8,9,10,13,16,19,22,25,28]
    tX_new[6][:, cub6] = np.cbrt(tX_new[6][:, cub6])
    
    cub7 = [1,2,3,4,5,6,8,9,10,11,14,17,20,22,23,26,29]
    tX_new[7][:, cub7] = np.cbrt(tX_new[7][:, cub7])
    
    # standardize
    means, stds = [], []
    for jet in range (0, 8):
        tX_new[jet], mean, std = standardize(tX_new[jet])
        means.append(mean)
        stds.append(std)
        
    if len(y)!=0:
        return y_new, tX_new, ids_new, means, stds
    else:
        return tX_new, ids_new, means, stds
    

def build_predictions(tX, indexes, w, degrees=[], logistic=False):
    """
    build the predictions for a given dataset
    INPUTS: X
            indexes := indexes for the association (from the preprocessing, useful to assign the data from the right jet
            into the new predicted y)
            w := weights
            degrees := list of polynomial degree used per jet

    """
    N = 0
    for jet in range(0, 8):
        N += tX[jet].shape[0] #compute the length of the prediction
    
    y_pred = np.zeros(N)
    for jet in range (0, 8):
        if (len(degrees) != 0): #build polynomial basis if degrees is passed to the function
            x = build_poly(tX[jet], degrees[jet])
        else:
            x = tX[jet]
        # predict labels for each jet and insert them in the correct position of the full prediction
        y_p = predict_labels(w[jet], x, logistic) 
        index = indexes[jet]
        y_pred[index] = y_p

    return y_pred



def compute_rmse_ridge(y, tx, w, lambda_):
    """modified version of rmse to take care of the penalization
    used inside cross validation"""
    l = compute_loss(y, tx, w)
    l = l + lambda_ * np.linalg.norm(w)**2
    return np.sqrt(2*l)


def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression computed over a k-fold cross validation
    with polynomial degrees"""
    losses_tr = []
    losses_te = []
    
    for k_group in range(k):
        
        # divide in test and train set: 1 set for test all the others for train
        index_te = k_indices[k_group]
        index_tr = np.setdiff1d(np.arange(len(y)), index_te)
        x_te = x[index_te]
        x_tr = x[index_tr]
        y_te = y[index_te]
        y_tr = y[index_tr]
        
        # form data with polynomial degree
        x_te_poly = build_poly(x_te, degree)
        x_tr_poly = build_poly(x_tr, degree)
        
        # compute w with ridge regression
        w, _ = ridge_regression(y_tr, x_tr_poly, lambda_)
        
        # calculate the loss for train and test data
        rmse_tr = compute_rmse_ridge(y_tr, x_tr_poly, w, lambda_)
        rmse_te = compute_rmse_ridge(y_te, x_te_poly, w, lambda_)
        losses_tr.append(rmse_tr)
        losses_te.append(rmse_te)
        
    #return losses average
    loss_tr = np.mean(losses_tr)
    loss_te = np.mean(losses_te)
    return loss_tr, loss_te
    
def select_best_hypers_ridge(y, tX, max_degree, k_fold, min_lambda_pow, max_lambda_pow, seed=1):
    """this function returns the best pair (degree, lambda) for all the jet in our dataset
    INPUTS: Y, X
            max_degree := maximum degree of polynomial
            k_fold := k_fold to use in cross validation
            min_lambda_pow, max_lambda_pow := minimum and maximum exponents to use for lambda, min_lambda_pow should be negative
            (the check goes from 10^(min_lambda_pow) to 10^(max_lambda_pow))
    """
    lambdas = np.logspace(min_lambda_pow, max_lambda_pow, 50)
    degrees_star = []
    lambdas_star = []
    for jet in range(0, 8):
        #print("jet: {}".format(jet))
        # preliminar variables
        loss_min = np.inf
        degree_star = 0
        lambda_star = 0
        k_indices = build_k_indices(y[jet], k_fold, seed)
        for degree in range(1, max_degree+1):
            for lambda_ in lambdas:
                # perform cross validation
                loss_tr, loss_te = cross_validation_ridge(y[jet], tX[jet], k_indices, k_fold, lambda_, degree)
                #save parameters if loss is reduced
                if loss_te < loss_min:
                    loss_min = loss_te
                    degree_star = degree
                    lambda_star = lambda_
                    #print("New loss: {}, degree: {}, lambda: {}".format(loss_te, degree, lambda_))
        if (jet%2==0):
            print("Jet {} no mass -> best loss: {}, degree: {}, lambda: {}".format(jet, loss_min, degree_star, lambda_star))
        else:
            print("Jet {} with mass -> best loss: {}, degree: {}, lambda: {}".format(jet-1, loss_min, degree_star, lambda_star))
        degrees_star.append(degree_star)
        lambdas_star.append(lambda_star)
    return degrees_star, lambdas_star
    

#PROJ1 HELPERS WITH SMALL MODIFICATIONS TO ADAPT TO OUR CASE

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids



def predict_labels(weights, data, logistic=False):
    """Generates predictions given weights, and a test data matrix for classification with least squares or logistic regression"""
    if logistic:
        y_pred = sigmoid(np.dot(data, weights))
        y_pred[np.where(y_pred <= 0.5)] = 0
        y_pred[np.where(y_pred > 0.5)] = 1    
    else:
        y_pred = np.dot(data, weights)
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
        
    return y_pred


def create_csv_submission(ids, y_pred, name, logistic=False):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    if (logistic==True): #build logistic submission: we previously computed labels using 0-1: we must reconvert it
        y_pred = np.where(y_pred==0, -1, y_pred)
    
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, dialect='unix') #using 'unix' dialect to allowind building also under Windows
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
