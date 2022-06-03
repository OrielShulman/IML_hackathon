from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from constants import *
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from ast import literal_eval
from sklearn.preprocessing import OrdinalEncoder
import preprocessingEli
import preprocessing_oriel
from sklearn.preprocessing import MultiLabelBinarizer

import split_train_set


def labels_shape(list):
    res = 0
    if 'BON - Bones' in list:
        res += 1
    if 'SKI - Skin' in list:
        res += 10
    if 'PUL - Pulmonary' in list:
        res += 100
    if 'LYM - Lymph nodes' in list:
        res += 1000
    if 'HEP - Hepatic' in list:
        res += 10000
    if 'PER - Peritoneum' in list:
        res += 100000
    if 'OTH - Other' in list:
        res += 1000000
    if 'BRA - Brain' in list:
        res += 10000000
    if 'PLE - Pleura' in list:
        res += 100000000
    if 'ADR - Adrenals' in list:
        res += 1000000000
    if 'MAR - Bone Marrow' in list:
        res += 10000000000
    return res


def cat(X_train, X_test, y_train, y_test):
    cat = CatBoostClassifier()
    cat.fit(X_train, y_train, plot=True)
    y_predict = cat.predict(X_test)
    macro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='macro')
    micro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='micro')
    nums = [10000000000, 1000000000, 100000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10, 1]
    labels = ['MAR - Bone Marrow', 'ADR - Adrenals', 'PLE - Pleura', 'BRA - Brain', 'OTH - Other',
                'PER - Peritoneum', 'HEP - Hepatic', 'LYM - Lymph nodes', 'PUL - Pulmonary', 'SKI - Skin',
              'BON - Bones']
    final_predict = []
    for i in range(len(y_predict)):
        final_predict.append([])
        if y_predict[i] == 0:
            final_predict[i] = str(final_predict[i])
            continue
        for j in range(len(nums)):
            if y_predict[i] >= nums[j]:
                final_predict[i].append(labels[j])
                y_predict[i] -= nums[j]
        final_predict[i] = str(final_predict[i])
    csv_pred = pd.DataFrame(final_predict, columns=['אבחנה-Location of distal metastases'])
    csv_pred.to_csv('predicitions.csv', index=False)

    return micro_loss, macro_loss


def cat_regressor(X_train, X_test, y_train, y_test):
    cat = CatBoostRegressor()
    cat.fit(X_train, y_train, plot=True)
    y_predict = cat.predict(X_test)
    score = mean_squared_error(y_pred=y_predict, y_true=y_test)
    # csv_pred = pd.DataFrame(final_predict, columns=['אבחנה-Location of distal metastases'])
    y_predict = pd.DataFrame(y_predict, columns=['אבחנה-Tumor size'])
    y_predict.to_csv('part2/predicitions.csv', index=False)
    print(score)


def mission_zero():
    X_real_test = pd.read_csv('resorces/origin_data/test.feats.csv')
    X_real_test.insert(loc=0, column='Unnamed', value=0)
    X_real_test = preprocessingEli.basicPreprocessing(X_real_test)

    X = pd.read_csv('resorces/origin_data/train.feats.csv')
    # X = preprocessingEli.basicPreprocessing(X)
    X = preprocessing_oriel.apply_preprocessing_task1(X)
    X.to_csv(PROCESSED_DATA_0)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_set.split_data_tumor_size(0)
    X_real_test, right = X_real_test.align(X_train, join="outer", axis=1)
    X_train, right = X_train.align(X_real_test, join="outer", axis=1)
    y_train = y_train['אבחנה-Location of distal metastases'].apply(
        lambda x: labels_shape(literal_eval(x)))
    col_real_test = X_real_test.columns
    col_train = X_train.columns
    diff = [x for x in col_real_test if x not in col_train]
    y_test = y_test['אבחנה-Location of distal metastases'].apply(
        lambda x: labels_shape(literal_eval(x)))
    print(cat(X_train, X_test, y_train, y_test))

def mission_one():
    X_real_test = pd.read_csv('resorces/origin_data/test.feats.csv')
    X_real_test.insert(loc=0, column='Unnamed', value=0)
    X_real_test = preprocessingEli.basicPreprocessing1(X_real_test)

    X = pd.read_csv('resorces/origin_data/train.feats.csv')

    # todo: ose new processing
    X = preprocessingEli.basicPreprocessing1(X)
    X.to_csv(PROCESSED_DATA_1)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_set.split_data_tumor_size(1)
    X_real_test, right = X_real_test.align(X_train, join="outer", axis=1)
    X_train, right = X_train.align(X_real_test, join="outer", axis=1)
    # y_train = y_train['אבחנה-Location of distal metastases'].apply(
    #     lambda x: labels_shape(literal_eval(x)))
    col_real_test = X_real_test.columns
    col_train = X_train.columns
    # diff = [x for x in col_real_test if x not in col_train]
    # y_test = y_test['אבחנה-Location of distal metastases'].apply(
    #     lambda x: labels_shape(literal_eval(x)))
    print(cat_regressor(X_train, X_test, y_train, y_test))


if __name__== "__main__":
    # mission_zero()
    mission_one()
