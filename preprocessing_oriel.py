import pandas as pd
import numpy as np
from constants import GENERAL_SEED

DATA_VERSION = r'V_0/'
TRAIN_F_PATH = r'resorces/data/' + DATA_VERSION + r'train_features.csv'
DEV_F_PATH = r'resorces/data/' + DATA_VERSION + r'dev_features.csv'
TEST_F_PATH = r'resorces/data/' + DATA_VERSION + r'test_features.csv'


def _print_column_values(df: pd.DataFrame) -> None:
    print(f"{len(df.unique())} values:\n{df.unique()}\n{'-'*30}\n")


def _f_basic_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Basic stage’: Carcinoma Basic stage - 4 Different stages (c, p , r, null)

    Mapping:
    Transforms to three continues features by stage value.
    p - Pathological: 3
    c - Clinical : 2
    r - Reccurent: 1
    None: 0

    Changes label name 'אבחנה-Basic stage' -> 'carcinoma_basic_stage'
    """
    # change feature name:
    df.rename(columns={'אבחנה-Basic stage': 'carcinoma_basic_stage'}, inplace=True)

    # rearrange values:
    category_map = {'p - Pathological': 3, 'c - Clinical': 2, 'r - Reccurent': 1, 'Null': 0}
    df.replace({'carcinoma_basic_stage': category_map}, inplace=True)
    return df


def _f_her2(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Her2': Tumor marker test that determines the number of copies of the HER2 gene or the
    amount of HER2 protein in a cancer cell - various formats
    1 -> negative to her2
    2 -> need to take FISH test
    3 -> positive to her2

    Mapping:
    ~unknown -> 0
    ~negative -> 1
    ~need to test for Fish -> 2
    ~positive -> 3
    """
    # change feature name:
    df.rename(columns={'אבחנה-Her2': 'her2'}, inplace=True)
    _print_column_values(df.her2)

    # deal with negative: (1/NEG..
    df.loc[df['her2'].str.contains('neg'), 'her2'] = '1'
    # df.loc['neg' in df['her2'].apply(str), 'her2'] = 1
    # df.loc['1' in df.her2.apply(str), 'her2'] = 1
    _print_column_values(df.her2)

    # df.loc[df['her2'] == "male", "gender"] = 1
    # rearrange values:
    # category_map = {'p - Pathological': 3, 'c - Clinical': 2, 'r - Reccurent': 1, 'Null': 0}
    # df.replace({'carcinoma_basic_stage': category_map}, inplace=True)
    return df


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # basic stage:
    print(f"types:\n{df.dtypes}\n{'-'*30}\n")

    _f_basic_stage(df)
    print(f"types:\n{df.dtypes}\n{'-'*30}\n")

    _f_her2(df)
    print(f"types:\n{df.dtypes}\n{'-'*30}\n")

    return df


if __name__ == '__main__':
    np.random.seed(GENERAL_SEED)

    train_set_ = pd.read_csv(TRAIN_F_PATH)
    # print(train_set_.dtypes, end=f"{'-'*30}\n")

    processed_train_set_ = apply_preprocessing(train_set_[['אבחנה-Basic stage', 'אבחנה-Her2']])
    # print(processed_train_set_.dtypes, end=f"{'-'*30}\n")



# df['carcinoma_basic_stage'] = df['carcinoma_basic_stage'].fillna(0)
# pd.unique(df.carcinoma_basic_stage)
# df.fillna(value={'carcinoma_basic_stage': 0})
# print(df.dtypes)
# print(df.columns)

