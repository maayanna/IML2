import numpy as np
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

def fit_linear_regression(X, y):
    """
    Task 9
    This function implements the linear regression
    :param X: numpy design matrix with p rows and n columns
    :param y: response vector with n rows
    :return: numpy array of coefficients vector w, numpy array of singular values of X
    """

    u, d, v = np.linalg.svd(X, full_matrices=False)
    my_diag = np.linalg.pinv(np.diag(d))
    w = np.dot(v.T, np.dot(my_diag, u.T))
    return np.dot(w.T, y), d


def predict(X, w):
    """
    Task 10
    :param X: numpy design matrix with p rows and n columns
    :param w: coefficients vector
    :return:
    """

    return np.dot(X.T, w)


def mse(response_v, prediction_v):
    """
    Task 11
    :param response_v: response vector
    :param prediction_v: prediction vector
    :return: returns the MSE over the received samples
    """

    diff = response_v - prediction_v
    sq = np.square(diff)
    return np.mean(sq)


def load_data(path):
    """
    Task 12
    This function is loading the data set and performing all the needed preprocessing so to get
    a design balid matrix
    :param path: path to the cv file that loads the my_dtet
    :return: valid design matrix
    """

    my_dt = pandas.read_csv("kc_house_data.csv")
    my_dt.dropna(how="any", inplace=True)  # If any NA values are present, drop that row or column.


    # Delete all corrupted data
    my_dt.drop(my_dt[my_dt['sqft_above'] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["price"] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["bedrooms"]<=0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["bathrooms"]<=0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["sqft_living"]<=0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["sqft_living15"] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["sqft_lot"]<=0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["sqft_lot15"] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["floors"]<=0].index, inplace=True)

    # Delete all no relevant data
    my_price = my_dt["price"]
    my_dt.drop(['price'], 1, inplace=True)

    my_dt.insert(loc=0, column="base", value=[1 for i in range(len(my_dt))])

    my_dt = pandas.get_dummies(my_dt, columns=["zipcode"])

    return my_dt, my_price
