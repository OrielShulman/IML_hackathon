import sklearn as sklearn

from constants import*

from typing import Tuple
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# split each set to sick and not sick
# split to train - dev - test


def load_data():
    X = pd.read_csv(TRAIN_FEATURES_PATH)
    y_0 = pd.read_csv(TRAIN_LABELS_0_PATH)
    y_1 = pd.read_csv(TRAIN_LABELS_1_PATH)

    # return df.drop(labels='price', axis=1), df.price
    return X, y_0, y_1


def split_task_A():
    X_ = pd.read_csv(TRAIN_FEATURES_PATH)
    y_ = pd.read_csv(TRAIN_LABELS_0_PATH)
    print(f'before: {y_.nunique()}\n\n')
    print(f'values: {pd.unique(y_.squeeze())}\n{"-"*30}\n')

    # extract unique classes
    s = y_.squeeze()
    mlb = MultiLabelBinarizer()
    y_dummies = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=y_.index)
    # y_dummies = pd.get_dummies(y_.stack()).sum(level=0, axis=1)
    print(f'before: {y_dummies.nunique()}\n\n')
    print(f'values: {pd.unique(y_dummies.squeeze())}\n{"-"*30}\n')
    name = y_.columns[0]
    unique_y = y_[name].unique().index()

    y_s = y_.squeeze()
    X_train_, X_test, y_train_, y_test = train_test_split(X_, y_s, test_size=TEST_PERCENTAGE, random_state=SPLIT_SEED, stratify=y_s)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train_, y_train_, test_size=DEV_PERCENTAGE, random_state=SPLIT_SEED, stratify=y_train_)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


# def split_train_test(X: pd.DataFrame, y_0: pd.DataFrame, y_1: pd.DataFrame) -> \
#         Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Randomly split given sample to a training- and testing sample
#
#     Parameters
#     ----------
#     X : DataFrame of shape (n_samples, n_features)
#         Data frame of samples and feature values.
#
#     y : Series of shape (n_samples, )
#         Responses corresponding samples in data frame.
#
#     train_proportion: Fraction of samples to be split as training set
#
#     Returns
#     -------
#     train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
#         Design matrix of train set
#
#     train_y : Series of shape (ceil(train_proportion * n_samples), )
#         Responses of training samples
#
#     test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
#         Design matrix of test set
#
#     test_y : Series of shape (floor((1-train_proportion) * n_samples), )
#         Responses of test samples
#
#     """
#
#     # split to train | test:
#     test_per = np.ceil(len(X.index) * TRAIN_PERCENTAGE).astype(int)
#     train_X = X.sample(n=test_per)
#     test_X = X.drop(train_X.index)
#     test_y = y.drop(train_X.index).reindex_like(test_X)
#
#     train_y = y.drop(test_y.index).reindex_like(train_X)
#
#     return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_task_A()

    # train_set, dev_set, test_set = split_train_test(X=X, y=y, train_proportion=1-TEST_PERCENTAGE)
