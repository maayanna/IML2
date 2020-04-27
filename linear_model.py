import numpy as np
import pandas
import matplotlib.pyplot as plt
import random

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
    Task 12 and 13
    This function is loading the data set and performing all the needed preprocessing so to get
    a design balid matrix
    :param path: path to the cv file that loads the my_dtet
    :return: valid design matrix
    """

    my_dt = pandas.read_csv(path)
    my_dt.dropna(how="any", inplace=True)  # If any NA values are present, drop that row or column.


    # Delete all corrupted data
    my_dt.drop(my_dt[my_dt['sqft_basement'] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt['sqft_above'] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["price"] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["bedrooms"]<=0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["bathrooms"]<=0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["sqft_living"]<=0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["sqft_living15"] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["sqft_lot"]<=0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["sqft_lot15"] <= 0].index, inplace=True)
    my_dt.drop(my_dt[my_dt["floors"]<0].index, inplace=True)

    # Delete all no relevant data
    my_dt.drop(["id", "date"], axis=1, inplace=True)
    # my_dt.drop(['long', 'lat', 'id', 'date', 'yr_built', 'condition', 'sqft_lot', 'sqft_lot15',
    #                 'yr_renovated'], axis=1, inplace=True)

    # Q13
    my_price = my_dt["price"]
    my_dt.drop(['price'], 1, inplace=True)

    my_dt.insert(loc=0, column="base", value=[1 for i in range(len(my_dt))])

    my_dt = pandas.get_dummies(my_dt, columns=["zipcode"])
    # my_dt.drop(["zipcode"], 1, inplace=True)

    return my_dt, my_price


def plot_singular_values(collect_val):
    """
    Task 14
    :param collect_val: singular values
    :return: Plot the singular values in descending order
    """

    my_col = np.sort(collect_val)[::-1]
    plt.plot(my_col)
    plt.xlabel("Indexes")
    plt.ylabel("Singular Values")
    plt.title("Q15 : Singular values in descending order")
    plt.show()

# Task 15
# my_dt, price = load_data("kc_house_data.csv")
# mat = np.linalg.svd(my_dt, compute_uv=False)
# plot_singular_values(mat)


def train_test_set_random(data, price):
    """
    Task 16
    Fit a model and test it over the data. First split the data into train and test sets randomaly,
    such that the size of the test set is 1/4 of the total data.
    Over 3/4 of the data, fit a model based on the first p% of the training set.
    Then use the predict function.
    :param data: my data to test and fit over a model
    :return:
    """

    # print(data)

    all_mse = []
    index = int(len(data)*0.75)

    data.insert(0, "price", price.values)
    # sklearn.model_selection.train_test_split()

    random.shuffle(data.to_numpy())

    train_dt = data[:index] # 75 first percent
    test_dt = data[index:]

    test_price = test_dt["price"]
    test_dt.drop(["price"], 1, inplace=True)


    for p in range(1,101):

        index = int(p/100 * len(train_dt))
        new_train = train_dt[:index]

        my_price = new_train["price"]
        new_train.drop(["price"], 1, inplace=True)

        fit, d = fit_linear_regression(new_train.T, my_price)

        prediction = predict(test_dt.T, fit)

        all_mse.append(mse(test_price, prediction))


    plt.plot(np.arange(1,101),all_mse)
    plt.title("Q16 : MSE depending on the test set (func of p%)")
    plt.xlabel("% of training")
    plt.ylabel("MSE")
    plt.show()

#Task 16
# my_dt, _price = load_data("kc_house_data.csv")
# train_test_set_random(my_dt,_price)


def feature_evaluation(matrix, response_vect):
    """
    Task 17
    This function plots for every non-categorical feature, a graph of the feature values and the
    response values. Also computes and shows on the graph the Pearson Correlation between the feature
    and the response.
    :param matrix: design matrix
    :param response_vect: response vector
    :return:
    """

    matrix.drop(["base"], 1, inplace=True)
    # print(matrix.columns)

    for column in matrix.columns:
        first = np.cov(matrix[column], response_vect)
        scd = np.std(response_vect) * np.std(matrix[column])
        correlation = first / scd
        plt.scatter(matrix[column], response_vect)
        plt.title("Q17 : Scatter plot of the " + str(column) + " column and the response value.\nPearson Correlation : %f" % correlation[1][0])
        plt.xlabel(column)
        plt.ylabel("Response value")
        plt.show()


# Task 17
# my_dt, price = load_data("kc_house_data.csv")
# feature_evaluation(my_dt, price)