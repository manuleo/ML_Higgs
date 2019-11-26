from helpers import *
from implementations import ridge_regression
import numpy as np
import argparse


def main(cross):
    
    # original data loading
    print("Loading training dataset....")
    DATA_TRAIN_PATH = 'train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Training dataset loaded!")
    
    print("Applying data preprocessing...")
    # split 80/20 dataset
    x_tr_tot, x_te_tot, y_tr_tot, y_te_tot = split_data(y,tX,0.8,1)
    
    # apply preprocessing to train and test sets
    y_tr, tX_tr, indexes_tr, means_tr, std_tr = preprocessing(x_tr_tot, y_tr_tot)
    y_te, tX_te, indexes_te, means_te, std_te = preprocessing(x_te_tot, y_te_tot)
    print("Preprocessing done.")

    #hardcoded parameter from ridge regression
    degrees_star = [7, 5, 5, 9, 6, 9, 5, 8]
    lambdas_star = [2.8117686979742307e-08, 1.757510624854793e-08, 3.088843596477485e-06, 2.94705170255181e-07, 1e-10, 1e-10, 1.67683293681101e-09, 1.0481131341546874e-09]

    # compute parameters by using cross validation on train set
    # !NOTE: this function needs more than 1 hour to run: the resulting parameters are hard coded for simplicity
    if (cross):
        print ("Starting cross validation...")
        degrees_star, lambdas_star = select_best_hypers_ridge(y_tr, tX_tr, max_degree = 9, k_fold = 12, min_lambda_pow = -10, max_lambda_pow = 0)
        print ("Cross validation ended! Hyper-parameters tuned")
    
    print("Computing best w...")
    #computing w with tuned hyper-parameters
    w = []
    for jet in range(0,8):
        x_tr_poly = build_poly(tX_tr[jet], degrees_star[jet])
        w_star, loss = ridge_regression(y_tr[jet], x_tr_poly, lambdas_star[jet])
        w.append(w_star)

    print("w ready!")
    #computing accuracy on our test set
    y_pred = build_predictions(tX_te, indexes_te, w, degrees_star)
    acc = accuracy(y_te_tot, y_pred)
    print("Best accuracy on our 20% test set is {}".format(acc))

    print("Loading test set and apply preprocessing...")
    #preparing data for submission
    DATA_TEST_PATH = 'test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    tX_test_new, indexes_test_new, means_test, stds_test = preprocessing(tX_test) # same function as train, when run without y doesn't compute nothing on that
    print("Test processed!")

    print("Building submission.csv...")
    #building submission
    y_pred_test = build_predictions(tX_test_new, indexes_test_new, w, degrees_star)
    OUTPUT_PATH = 'submission.csv'
    create_csv_submission(ids_test, y_pred_test, OUTPUT_PATH)
    print("Submission ready!")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cross', action='store_true')
    args = parser.parse_args()
    main(args.cross)