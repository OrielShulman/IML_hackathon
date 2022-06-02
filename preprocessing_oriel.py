import pandas as pd
import numpy as np
from constants import GENERAL_SEED
from typing import Union, Optional

DATA_VERSION = r'0'
TRAIN_F_PATH = r'resorces/data/V_' + DATA_VERSION + r'/train_features.csv'
DEV_F_PATH = r'resorces/data/V_' + DATA_VERSION + r'/dev_features.csv'
TEST_F_PATH = r'resorces/data/V_' + DATA_VERSION + r'/test_features.csv'

TASK_A_RELEVANT_FEATURES = ['אבחנה-M -metastases mark (TNM)',
                            'אבחנה-Stage',
                            'אבחנה-Surgery name1',
                            'אבחנה-Surgery name2',
                            'אבחנה-Age',
                            'אבחנה-Side',
                            'אבחנה-Positive nodes',
                            'אבחנה-Basic stage',
                            'אבחנה-Her2']


def _print_column_values(df: pd.DataFrame) -> None:
    print(f"{len(df.unique())} values:\n{df.unique()}\n{'-' * 30}\n")


def _her2_filter(val: str) -> int:
    """
    classifies HER2 possible values to legal values of [0, 1, 2, 3]
    0 ~ unknown             -->
    1 ~ negative            -->
    2 ~ mid (test for FISH) -->
    3 ~ positive            -->

    Note: positive and negative results are examined first
    :param val: original value as string (in lower case)
    """
    # definite negative matches:
    if any(x in val for x in ('+1', '1+', 'שלילי', 'neg', 'heg', 'nag')) or val == '1':
        return 1

    # definite positive matches:
    if any(x in val for x in ('+3', '3+', 'חיובי', 'pos')) or val == '3' or val == 'her2':
        return 3

    # mid-matches:
    if any(x in val for x in ('+2', '2+', 'fish')) or val == '2':
        return 2

    # all nan or ambiguous values
    return 0


def _stage_filter(val: str) -> float:
    """
    classifies cancer stage by rate
    """
    val = val.lower()
    stage_dictionary = {'la': 0, 'stage0': 0.25, 'stage0a': 0.5, 'stage0is': 0.75,
                        'stage1': 1, 'stage1a': 1.25, 'stage1b': 1.5, 'stage1c': 1.75,
                        'stage2': 2, 'stage2a': 2.33, 'stage2b': 2.66,
                        'stage3': 3, 'stage3a': 3.25, 'stage3b': 3.5, 'stage3c': 3.75,
                        'stage4': 4}
    if val in stage_dictionary:
        return stage_dictionary[val]

    return 0


def _metastases_filter(val: str) -> Optional[str]:
    """
    allowed values are M<x>, x in [1, 2, 3, 4, 5, 6]
    classifies HER2 possible values to legal values of [0, 1, 2, 3]
    """
    vals = 'M0', 'M1', 'M1A', 'M1B', 'M1C', 'MX'
    if val in ('M0', 'M1', 'M1A', 'M1B', 'M1C', 'MX'):
        return val

    return np.nan


def _general_filter(val: str, legal_vals: tuple) -> int:

    # definite negative matches:
    if any(x == val for x in legal_vals):
        return 1

    return 0


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    keeps only features that are relevant for the task and has been processed
    """
    return df[TASK_A_RELEVANT_FEATURES]


def f_metastases_mark(df: pd.DataFrame) -> pd.DataFrame:
    """
    'M -metastases mark (TNM)': Amount of existence of metastases - 6 values, M<x>
    """
    # change feature name:
    df.rename(columns={'אבחנה-M -metastases mark (TNM)': 'metastases_mark'}, inplace=True)

    # filter values
    df['metastases_mark'] = df['metastases_mark'].apply(lambda x: _metastases_filter(str(x).upper()))

    # fixme:
    #  - make sure that only 6 values are produced
    #  - FILL empty values (maybe insert extra value for null before dummies
    df = pd.get_dummies(df, columns=['metastases_mark'])
    return df


def f_surgery_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    fills null values within the columns of relevant surgeries
    'Surgery name1': Name of first surgery - 23 values, CONSTS (english caps)
    'Surgery name2': Name of second surgery - 18 values, CONSTS (english caps)
    'Surgery name3': Name of third surgery - 6 values, CONSTS (english caps)
    """
    df.rename(columns={'אבחנה-Surgery name1': 'surgery1',
                       'אבחנה-Surgery name2': 'surgery2'}, inplace=True)

    # TODO: why not use name 3?
    #   - examine number of different values

    df['surgery1'].fillna('no_surgery', inplace=True)
    df['surgery2'].fillna('no_surgery', inplace=True)

    df = pd.get_dummies(df, columns=['surgery1'])
    df = pd.get_dummies(df, columns=['surgery2'])
    return df


def f_cancer_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Stage': Stage of cancer - 17 values, english const
    """
    df.rename(columns={'אבחנה-Stage': 'cancer_stage'}, inplace=True)

    # fixme: null -> 0 may conflict with dictionary values
    #  - why dummies?

    # df['stage'].fillna(0, inplace=True)

    # stageDictionary = {'Stage2a': 2, 'Stage4': 4, 'Stage1': 1, 'Stage3b': 3, 'Stage2b': 2, 'LA': 0, 'Stage1c': 1,
    #                    'Stage2': 2, 'Stage3c': 3, 'Stage0': 0, 'Stage1b': 1, 'Stage3a': 3, 'Stage3': 3,
    #                    'Stage1a': 1, 'Stage0is': 0, 'Not yet Established': 0, 'Stage0a': 0}
    # df['stage'] = df['stage'].replace(stageDictionary)

    df['cancer_stage'] = df['cancer_stage'].apply(lambda x: _stage_filter(str(x).lower()))

    # df = pd.get_dummies(df, prefix='stage', columns=['stage'])
    return df


def f_positive_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    'positive nodes': How many of Lymph nodes contained carcinoma metastases - 28 values, integers
    """
    df.rename(columns={'אבחנה-Positive nodes': 'positive_nodes'}, inplace=True)

    df['positive_nodes'] = df['positive_nodes'].fillna(0).apply(int)
    return df


def f_basic_stage(df: pd.DataFrame) -> pd.DataFrame:
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


def f_her2(df: pd.DataFrame) -> pd.DataFrame:
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

    # map values to [0, 1, 2, 3]
    df['her2'] = df['her2'].apply(lambda x: _her2_filter(str(x).lower()))
    return df


def f_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    just feature name for convinients
    """
    # TODO: clear nan vals

    # change feature name:
    df.rename(columns={'אבחנה-Age': 'age'}, inplace=True)

    return df


def f_breast_side(df: pd.DataFrame) -> pd.DataFrame:
    """
    separates for left and right separately
    """
    # change feature name:
    df.rename(columns={'אבחנה-Side': 'side'}, inplace=True)

    df['left'] = df['side'].apply(lambda x: _general_filter(val=str(x), legal_vals=('left', 'both', 'שמאל', 'דו צדדי')))

    df['right'] = df['side'].apply(lambda x: _general_filter(val=str(x), legal_vals=('right', 'both', 'ימין', 'דו צדדי')))

    df.drop(['side'], axis=1, inplace=True)
    return df


def apply_preprocessing_task1(df: pd.DataFrame) -> pd.DataFrame:
    """
    applies preprocessing for the data that is relevant to the first task
    """
    df = drop_features(df)

    df = f_metastases_mark(df)

    df = f_surgery_names(df)

    df = f_cancer_stage(df)

    df = f_positive_nodes(df)

    df = f_basic_stage(df)

    df = f_her2(df)

    df = f_age(df=df)

    df = f_breast_side(df=df)

    return df


def generate_task1_data(level: str) -> None:
    f_train = apply_preprocessing_task1(df=pd.read_csv(TRAIN_F_PATH))
    f_dev = apply_preprocessing_task1(df=pd.read_csv(DEV_F_PATH))
    f_test = apply_preprocessing_task1(df=pd.read_csv(TEST_F_PATH))

    f_train.to_csv(path_or_buf=r'resorces/data/V_' + level + r'/train_features.csv')
    f_dev.to_csv(path_or_buf=r'resorces/data/V_' + level + r'/dev_features.csv')
    f_test.to_csv(path_or_buf=r'resorces/data/V_' + level + r'/test_features.csv')


if __name__ == '__main__':
    np.random.seed(GENERAL_SEED)

    generate_task1_data(level='1')

    # train_set_ = pd.read_csv(TRAIN_F_PATH)
    # processed_train_set_ = apply_preprocessing_task1(df=train_set_)
    # print(processed_train_set_.dtypes, end=f"{'-' * 30}\n")


# df['carcinoma_basic_stage'] = df['carcinoma_basic_stage'].fillna(0)
# pd.unique(df.carcinoma_basic_stage)
# df.fillna(value={'carcinoma_basic_stage': 0})
# print(df.dtypes)
# print(df.columns)

# df.loc[df['her2'].str.contains('neg'), 'her2'] = '1'
# df.loc['neg' in df['her2'].apply(str), 'her2'] = 1
# df.loc['1' in df.her2.apply(str), 'her2'] = 1
# df.loc[df['her2'] == "male", "gender"] = 1
# rearrange values:
# category_map = {'p - Pathological': 3, 'c - Clinical': 2, 'r - Reccurent': 1, 'Null': 0}
# df.replace({'carcinoma_basic_stage': category_map}, inplace=True)

# print(f"surgery1: {len(df['surgery1'].unique())} values:\n{'-'*30}")
# print(f"surgery2: {len(df['surgery2'].unique())} values:\n{'-'*30}")

# print(f"positive_nodes: {len(df['positive_nodes'].unique())} values:\n{df['positive_nodes'].unique()}\n"
#       f"{df['positive_nodes'].value_counts()}\n{'-'*30}")

# print(f"types:\n{df.dtypes}\n{'-' * 30}\n")
