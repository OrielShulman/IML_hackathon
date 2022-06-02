import sklearn as sklearn

from constants import*

from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# s = y_.squeeze()
# mlb = MultiLabelBinarizer()
# y_dummies = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=y_.index)
# # y_dummies = pd.get_dummies(y_.stack()).sum(level=0, axis=1)
# print(f'before: {y_dummies.nunique()}\n\n')
# print(f'values: {pd.unique(y_dummies.squeeze())}\n{"-"*30}\n')
# name = y_.columns[0]
# unique_y = y_[name].unique().index()
# y_s = y_.squeeze()

def load_data():
    X = pd.read_csv(TRAIN_FEATURES_PATH)
    y_0 = pd.read_csv(TRAIN_LABELS_0_PATH)
    y_1 = pd.read_csv(TRAIN_LABELS_1_PATH)

    # return df.drop(labels='price', axis=1), df.price
    return X, y_0, y_1


def split_data_tumor_size(mission) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # read relevant data
    if (mission == 0):
        y_ = pd.read_csv(TRAIN_LABELS_0_PATH)
        X_ = pd.read_csv(PROCESSED_DATA_0)
    else:
        y_ = pd.read_csv(TRAIN_LABELS_1_PATH)
        X_ = pd.read_csv(PROCESSED_DATA_1)
    print(f'before: {y_.nunique()}\n\n')
    print(f'values: {pd.unique(y_.squeeze())}\n{"-"*30}\n')

    # split data to [(train + dev), test]:
    X_train_, X_test, y_train_, y_test = train_test_split(X_, y_, test_size=TEST_PERCENTAGE, random_state=SPLIT_SEED)

    # split to -> [train, dev]:
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_, y_train_, test_size=DEV_PERCENTAGE, random_state=SPLIT_SEED)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def save_to_csv(df: Union[pd.DataFrame, pd.Series], path: str):
    path = path + r'.csv'
    df.to_csv(path_or_buf=path)


def examine_data(data: Union[pd.DataFrame, pd.Series]):
    # print(df['label'].corr(df['label']))
    print(f"feature:\n{data.columns}\n{'-' * 50}")
    # df.drop(axis=1, inplace=True, labels=['some label'])  # drop labels, inplace - should change the original df
    headers = data.head()  # 5 rows of each column
    labels = data.columns  # labels of dataset columns
    analysis = data.describe()  # the analysis of each column
    dtypes = data.dtypes
    print(f"X.dtypes:\n{data.dtypes}\n{'-' * 50}")


if __name__ == '__main__':
    np.random.seed(0)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data_tumor_size()

    # do not uncomment! just initial data save
    # for d, name in ((X_train, 'train_features'), (X_dev, 'dev_features'), (X_test, 'test_features'),
    #                 (y_train, 'train_labels'), (y_dev, 'dev_labels'), (y_test, 'test_labels')):
    #     save_to_csv(df=d, path=DATA_SAVE_PATH + name)

    for d in X_train, X_dev, X_test, y_train, y_dev, y_test:
        examine_data(d)




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
