#Baseline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def baseline(X_train, X_test, y_train, y_test):
    dummy = DummyClassifier(strategy='stratified')
    dummy.fit(X_train, y_train)
    y_predict = dummy.predict(X_test)
    macro_loss = f1_score(y_pred=y_predict, y_true=y_test, average={'macro'})
    micro_loss = f1_score(y_pred=y_predict, y_true=y_test, average={'micro'})
    return y_predict, micro_loss, macro_loss




