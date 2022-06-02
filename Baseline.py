#Baseline
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from ast import literal_eval
from sklearn.preprocessing import OrdinalEncoder
import preprocessingEli

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
    # if 'PLE - Pleura' in list:

    # if 'ADR - Adrenals'
    #
    #     'MAR - Bone Marrow'


    return res


def baseline(X_train, X_test, y_train, y_test):
    dummy = DummyClassifier(strategy='stratified')
    dummy.fit(X_train, y_train)
    y_predict = dummy.predict(X_test)
    macro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='macro')
    micro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='micro')
    return micro_loss, macro_loss

if __name__== "__main__":
    X_train = pd.read_csv('/Users/yuvalpolinski/IML_hackathon/resorces/data/V_0/train_features.csv')
    X_test = pd.read_csv('/Users/yuvalpolinski/IML_hackathon/resorces/data/V_0/test_features.csv')
    y_train = pd.read_csv('/Users/yuvalpolinski/IML_hackathon/resorces/data/V_0/train_labels.csv')
    y_test = pd.read_csv('/Users/yuvalpolinski/IML_hackathon/resorces/data/V_0/test_labels.csv')
    X_train = preprocessingEli.basicPreprocessing(X_train)
    X_test = preprocessingEli.basicPreprocessing(X_test)
    y_train = preprocessingEli.basicPreprocessing(y_train)
    y_test = preprocessingEli.basicPreprocessing(y_test)
    y_train = y_train['אבחנה-Location of distal metastases'].apply(lambda x: literal_eval(str(x)))
    y_test = y_test['אבחנה-Location of distal metastases'].apply(lambda x: (literal_eval(str(x))))
    y_train = y_train.apply(lambda x: x.sort() if x is not None else [''])
    y_test = y_test.apply(lambda x: x.sort() if x is not None else [''])
    # ord_enc = OrdinalEncoder()
    # y_train = ord_enc.fit_transform(y_train)
    # y_test = ord_enc.fit_transform(y_test)
    # y_train = y_train.apply(labels_shape)
    # y_test = y_test.apply(labels_shape)

    print(baseline(X_train, X_test, y_train, y_test))





