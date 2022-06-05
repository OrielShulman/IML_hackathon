from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

from source.data_preprocessing import apply_preprocessing_tumor_size
from source.split_data import split_train_dev_test
from constants import *


def train_cat_regressor(X_train: pd.DataFrame, y_train: pd.DataFrame) -> CatBoostRegressor:
    regressor = CatBoostRegressor()
    regressor.fit(X_train, y_train)
    return regressor


def test_regressor(regressor: CatBoostRegressor, X_test, y_test) -> None:
    y_predict = regressor.predict(X_test)
    score = mean_squared_error(y_pred=y_predict, y_true=y_test)
    print(f"\n{'-'*20}\nscore:\n{score}\n{'-'*20}")


def cat_regressor_predict(regressor: CatBoostRegressor, X_test):
    y_predict = regressor.predict(X_test)
    # score = mean_squared_error(y_pred=y_predict, y_true=y_test)
    # print(f"score:\n{score}")
    # csv_pred = pd.DataFrame(final_predict, columns=['אבחנה-Location of distal metastases'])
    y_predict = pd.DataFrame(y_predict, columns=['אבחנה-Tumor size'])
    y_predict.to_csv('part2/predicitions.csv', index=False)


def learn_tumor_size(test: bool = False) -> None:
    """ task 2 """
    # get learn features and labels:
    _X = apply_preprocessing_tumor_size(pd.read_csv(TRAIN_FEATURES_PATH))
    _y = pd.read_csv(TRAIN_LABELS_1_PATH)

    # split sets
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X_data=_X, y_labels=_y)

    # fit and predict / test
    if test:
        # get the test features:
        X_real_test = pd.read_csv(TEST_FEATURES_PATH)
        X_real_test.insert(loc=0, column='Unnamed', value=0)
        X_real_test = apply_preprocessing_tumor_size(df=X_real_test)

        # align sets for missing features
        X_real_test, _ = X_real_test.align(X_train, join="outer", axis=1)
        X_train, _ = X_train.align(X_real_test, join="outer", axis=1)

        # run catboost:
        _regressor = train_cat_regressor(X_train=X_train, y_train=y_train)
        cat_regressor_predict(regressor=_regressor, X_test=X_real_test)

    else:
        # run catboost:
        _regressor = train_cat_regressor(X_train=X_train, y_train=y_train)
        test_regressor(regressor=_regressor, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    learn_tumor_size()
