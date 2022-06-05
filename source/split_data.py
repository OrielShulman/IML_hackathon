from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from source.constants import *


def split_train_dev_test(X_data: pd.DataFrame, y_labels: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    splits data according to the constant properties to [train, dev, test]
    :param X_data: features data to split
    :param y_labels: labels data to split
    :return: [X_train, X_dev, X_test, y_train, y_dev, y_test]
    """

    # split data to [(train + dev), test]:
    X_train_, X_test, y_train_, y_test = train_test_split(X_data, y_labels, test_size=TEST_PERCENTAGE,
                                                          random_state=SPLIT_SEED)

    # split to -> [train, dev]:
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_, y_train_, test_size=DEV_PERCENTAGE,
                                                      random_state=SPLIT_SEED)

    return X_train, X_dev, X_test, y_train, y_dev, y_test

