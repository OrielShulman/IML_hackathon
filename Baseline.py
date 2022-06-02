#Baseline
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from ast import literal_eval
from sklearn.preprocessing import OrdinalEncoder
import preprocessingEli
from sklearn.preprocessing import MultiLabelBinarizer

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


def baseline(X_train, X_test, y_train, y_test):
    dummy = DummyClassifier(strategy='stratified')
    dummy.fit(X_train, y_train)
    y_predict = dummy.predict(X_test)
    macro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='macro')
    micro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='micro')
    return micro_loss, macro_loss

if __name__== "__main__":
    X_train = pd.read_csv('resorces/data/V_0/train_features.csv')
    X_test = pd.read_csv('resorces/data/V_0/test_features.csv')
    y_train = pd.read_csv('resorces/data/V_0/train_labels.csv')
    y_test = pd.read_csv('resorces/data/V_0/test_labels.csv')
    X_train = preprocessingEli.basicPreprocessing(X_train)
    X_test = preprocessingEli.basicPreprocessing(X_test)
    y_train = y_train['אבחנה-Location of distal metastases'].apply(lambda x: labels_shape(literal_eval(x)))
    y_test = y_test['אבחנה-Location of distal metastases'].apply(lambda x: labels_shape(literal_eval(x)))
    # print(y_test[1].cou)
    # y_train = y_train.apply(lambda x: sorted(x))
    # y_test = y_test.apply(lambda x: sorted(x))
    # y_test.replace(to_replace=[], value=[''])
    # y_train.replace(to_replace=[], value=[''])
    # ord_enc = OrdinalEncoder()
    # y_train = ord_enc.fit_transform(y_train)
    # y_test = ord_enc.fit_transform(y_test)
    # y_train = y_train.apply(labels_shape)
    # y_test = y_test.apply(labels_shape)
    # z= MultiLabelBinarizer().fit(y_train)
    # MultiLabelBinarizer().fit(y_test)
    print(baseline(X_train, X_test, y_train, y_test))





