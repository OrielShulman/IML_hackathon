from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
import pandas as pd

from source.data_preprocessing import apply_preprocessing_metastases, clean_metastases_labels
from source.split_data import split_train_dev_test
from constants import *


def test_baseline_classifier(X_train, X_test, y_train, y_test):
    dummy = DummyClassifier(strategy='stratified')
    dummy.fit(X_train, y_train)
    y_predict = dummy.predict(X_test)
    macro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='macro')
    micro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='micro')
    print(micro_loss, macro_loss)


def train_cat_classifier(X_train: pd.DataFrame, y_train: pd.DataFrame) -> CatBoostClassifier:
    classifier = CatBoostClassifier()
    classifier.fit(X_train, y_train)
    return classifier


def test_classifier(classifier: CatBoostClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    y_predict = classifier.predict(X_test)
    macro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='macro')
    micro_loss = f1_score(y_pred=y_predict, y_true=y_test, average='micro')
    print(f"micro: {micro_loss}\nmacro:{macro_loss}")


def cat_classifier_predict(classifier: CatBoostClassifier, X_test: pd.DataFrame,
                           save_path: str = r'part1/predicitions.csv') -> None:
    y_predict = classifier.predict(X_test)
    # re-transform y labels
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
    csv_pred.to_csv(path_or_buf=save_path, index=False)


def learn_metastases(test: bool = False) -> None:
    """ task 1 """
    # get learn features and labels:
    _X = apply_preprocessing_metastases(pd.read_csv(TRAIN_FEATURES_PATH))
    _y = clean_metastases_labels(df=pd.read_csv(TRAIN_LABELS_0_PATH))

    # split sets
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X_data=_X, y_labels=_y)

    # fit and predict / test
    if test:
        # get the test features:
        X_real_test = pd.read_csv(TEST_FEATURES_PATH)
        X_real_test.insert(loc=0, column='Unnamed', value=0)
        X_real_test = apply_preprocessing_metastases(df=X_real_test)

        # align sets for missing features
        X_real_test, _ = X_real_test.align(X_train, join="outer", axis=1)
        X_train, _ = X_train.align(X_real_test, join="outer", axis=1)

        # run catboost:
        _classifier = train_cat_classifier(X_train=X_train, y_train=y_train)
        cat_classifier_predict(classifier=_classifier, X_test=X_real_test)

    else:
        # run catboost:
        _classifier = train_cat_classifier(X_train=X_train, y_train=y_train)
        test_classifier(classifier=_classifier, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    learn_metastases()
