import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataframe = 'Data.csv'
def global_(dataframe):
    r2_score_dict = {}
    def open_dataset(dataframe):
        dataset = pd.read_csv(dataframe)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X, y, X_train, y_train, X_test, y_test

    def multiple_linear(dataframe):
        _, _, X_train, y_train, X_test, y_test = open_dataset(dataframe)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        np.set_printoptions(precision=2)
        np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
        r2_score_dict['multiple_linear'] = r2_score(y_test, y_pred)

    multiple_linear(dataframe)

    def polynomial(dataframe):
        _, _, X_train, y_train, X_test, y_test = open_dataset(dataframe)
        poly_reg = PolynomialFeatures(degree=4)
        X_poly = poly_reg.fit_transform(X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, y_train)
        y_pred = regressor.predict(poly_reg.transform(X_test))
        np.set_printoptions(precision=2)
        np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
        r2_score_dict['polynomial'] = r2_score(y_test, y_pred)

    polynomial(dataframe)

    # def svr(dataframe):
    #     _, _, X_train, y_train, X_test, y_test = open_dataset(dataframe)
    #     sc_X = StandardScaler()
    #     sc_y = StandardScaler()
    #     X_train = sc_X.fit_transform(X_train)
    #     y_train = sc_y.fit_transform(y_train)
    #     regressor = SVR(kernel='rbf')
    #     regressor.fit(X_train, y_train)
    #     y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
    #     np.set_printoptions(precision=2)
    #     np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
    #     r2_score_dict['svr'] = r2_score(y_test, y_pred)
    #
    # svr(dataframe)

    def decision_tree(dataframe):
        _, _, X_train, y_train, X_test, y_test = open_dataset(dataframe)
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        np.set_printoptions(precision=2)
        np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
        r2_score_dict['decision_tree'] = r2_score(y_test, y_pred)

    decision_tree(dataframe)

    def random_forest(dataframe):
        _, _, X_train, y_train, X_test, y_test = open_dataset(dataframe)
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        np.set_printoptions(precision=2)
        np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
        r2_score_dict['random_forest'] = r2_score(y_test, y_pred)

    random_forest(dataframe)

    print(r2_score_dict)

global_(dataframe)
