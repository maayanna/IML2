import numpy as np
import pandas
import matplotlib.pyplot as plt


def fit_linear_regression(X, y):
    """
    Task 9 from linear_model.py
    This function implements the linear regression
    :param X: numpy design matrix with p rows and n columns
    :param y: response vector with n rows
    :return: numpy array of coefficients vector w, numpy array of singular values of X
    """

    u, d, v = np.linalg.svd(X, full_matrices=False)
    my_diag = np.linalg.pinv(np.diag(d))
    w = np.dot(v.T, np.dot(my_diag, u.T))
    return np.dot(w.T, y), d

def all_tasks(path):

    # Task 18
    my_dt = pandas.read_csv(path)

    # Task 19
    my_dt.insert(3, "log_detected", np.log(my_dt["detected"]))

    # Task 20

    my_dt.insert(4, "lst_one", [1 for i in range(len(my_dt))])

    new_dt = my_dt[["day_num", "lst_one"]]

    fit = fit_linear_regression(new_dt.T, my_dt["log_detected"])

    # Tak 21

    # First plot
    
    plt.plot(my_dt['day_num'], my_dt['log_detected'], 'o', label="DATA")
    plt.plot(my_dt['day_num'], np.dot(new_dt, fit[0]), label="ESTIMATION")
    plt.legend()
    plt.title("Q21 : Log detected graph as a function of day num")
    plt.xlabel("day_num")
    plt.ylabel("log_detected")
    plt.show()

    # Second plot

    plt.plot(my_dt['day_num'], my_dt['detected'], 'o', label="DATA")
    plt.plot(my_dt['day_num'], np.exp(np.dot(new_dt, fit[0])), label="ESTIMATION")
    plt.legend()
    plt.title("Q21 : Detected graph as a function of day num")
    plt.xlabel("day_num")
    plt.ylabel("detected")
    plt.show()

all_tasks("covid19_israel.csv")