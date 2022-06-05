import pandas as pd
import numpy as np
from typing import Optional
from ast import literal_eval


FEATURES_METASTASES_DICT = {'אבחנה-M -metastases mark (TNM)': 'metastases_mark',
                            'אבחנה-Stage': 'cancer_stage',
                            # 'אבחנה-Surgery name1': 'surgery1',
                            'אבחנה-Positive nodes': 'positive_nodes',
                            'אבחנה-Basic stage': 'carcinoma_basic_stage',
                            'אבחנה-Her2': 'her2',
                            'אבחנה-Age': 'age',
                            'אבחנה-Side': 'side'}

FEATURES_SIZE_DICT = {'אבחנה-M -metastases mark (TNM)': 'metastases_mark',
                      'אבחנה-Stage': 'cancer_stage',
                      'אבחנה-Surgery name1': 'surgery1',
                      # 'אבחנה-Surgery name2': 'surgery2',
                      'אבחנה-Positive nodes': 'positive_nodes',
                      'אבחנה-Basic stage': 'carcinoma_basic_stage',
                      'אבחנה-Side': 'side',
                      'אבחנה-Her2': 'her2',
                      'אבחנה-Age': 'age',
                      'אבחנה-Tumor width': 'width',
                      'אבחנה-Tumor depth': 'depth'}


# features cleaners and preprocessing:
def drop_features(df: pd.DataFrame, features_to_save: dict) -> pd.DataFrame:
    """
    keeps only features that are relevant for the task and has been processed
    """
    df = df[list(features_to_save.keys())]
    df.rename(columns=features_to_save, inplace=True)
    return df


def clean_metastases_mark(df: pd.DataFrame) -> pd.DataFrame:
    """
    'M -metastases mark (TNM)': Amount of existence of metastases - 6 values, M<x>
    """

    def _metastases_filter(val: str) -> Optional[str]:
        """
        allowed values are M<x>, x in [1, 2, 3, 4, 5, 6]
        classifies HER2 possible values to legal values of [0, 1, 2, 3]
        """
        if val in ('M0', 'M1', 'M1A', 'M1B', 'M1C', 'MX'):
            return val

        return np.nan

    # filter values
    df['metastases_mark'] = df['metastases_mark'].apply(lambda x: _metastases_filter(str(x).upper()))

    df = pd.get_dummies(df, columns=['metastases_mark'])
    return df


def clean_surgery_name_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    fills null values within the columns of relevant surgeries
    'Surgery name1': Name of first surgery - 23 values, CONSTS (english caps)
    'Surgery name2': Name of second surgery - 18 values, CONSTS (english caps)
    'Surgery name3': Name of third surgery - 6 values, CONSTS (english caps)
    """

    df['surgery1'].fillna('no_surgery', inplace=True)
    df = pd.get_dummies(df, columns=['surgery1'])

    return df


def clean_surgery_name_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    fills null values within the columns of relevant surgeries
    'Surgery name1': Name of first surgery - 23 values, CONSTS (english caps)
    'Surgery name2': Name of second surgery - 18 values, CONSTS (english caps)
    'Surgery name3': Name of third surgery - 6 values, CONSTS (english caps)
    """

    df['surgery2'].fillna('no_surgery', inplace=True)
    df = pd.get_dummies(df, columns=['surgery2'])

    return df


def clean_cancer_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Stage': Stage of cancer - 17 values, english const
    """

    def _stage_filter(val: str) -> float:
        """
        classifies cancer stage by rate
        """
        val = val.lower()
        stage_dictionary = {'la': 0, 'stage0': 0, 'stage0a': 0, 'stage0is': 0,
                            'stage1': 1, 'stage1a': 1, 'stage1b': 1, 'stage1c': 1,
                            'stage2': 2, 'stage2a': 2, 'stage2b': 2,
                            'stage3': 3, 'stage3a': 3, 'stage3b': 3, 'stage3c': 3,
                            'stage4': 4}
        if val in stage_dictionary:
            return stage_dictionary[val]

        return 0

    df['cancer_stage'] = df['cancer_stage'].apply(lambda x: _stage_filter(str(x).lower()))

    # fixme: consider using dummies
    # df = pd.get_dummies(df, prefix='stage', columns=['stage'])

    return df


def clean_positive_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    'positive nodes': How many of Lymph nodes contained carcinoma metastases - 28 values, integers
    """
    df['positive_nodes'] = df['positive_nodes'].fillna(0).apply(int)
    return df


def clean_basic_stage(df: pd.DataFrame) -> pd.DataFrame:
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

    # rearrange values:
    category_map = {'p - Pathological': 3, 'c - Clinical': 2, 'r - Reccurent': 1, 'Null': 0}
    df.replace({'carcinoma_basic_stage': category_map}, inplace=True)
    return df


def clean_her2(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Her2': Tumor marker test that determines the number of copies of the HER2 gene or the
    amount of HER2 protein in a cancer cell - various formats
    1 -> negative to her2
    2 -> need to take FISH test
    3 -> positive to her2

    Mapping:
    0 ~ unknown
    1 ~ negative
    2 ~ mid (test for FISH)
    3 ~ positive
    """

    def _her2_filter(val: str) -> int:
        """
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

    df['her2'] = df['her2'].apply(lambda x: _her2_filter(str(x).lower()))
    return df


def clean_breast_side(df: pd.DataFrame) -> pd.DataFrame:
    """
    separates for left and right separately
    """

    def _side_filter(val: str, legal_vals: tuple) -> int:
        # definite negative matches:
        if any(x == val for x in legal_vals):
            return 1

        return 0

    df['left'] = df['side'].apply(lambda x: _side_filter(val=str(x), legal_vals=('left', 'both', 'שמאל', 'דו צדדי')))

    df['right'] = df['side'].apply(lambda x: _side_filter(val=str(x), legal_vals=('right', 'both', 'ימין', 'דו צדדי')))

    df.drop(['side'], axis=1, inplace=True)

    return df


# - additional filtering: test
def clean_tumor_depth(df: pd.DataFrame) -> pd.DataFrame:
    """
    """

    df['depth'].fillna(df['depth'].mean(), inplace=True)
    # df['depth'].fillna(0, inplace=True)

    return df


def clean_tumor_width(df: pd.DataFrame) -> pd.DataFrame:
    """
    """

    # df['width'].fillna(df['width'].mean(), inplace=True)
    df['width'].fillna(0, inplace=True)

    return df


# labels value preprocessing:
def clean_metastases_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    reprocess y labels by the values of predictions
    """

    def _y_labels_reformat(val: list) -> int:
        """
        hashing the values in the labels list
        """
        res = 0
        if 'BON - Bones' in val:
            res += 1
        if 'SKI - Skin' in val:
            res += 10
        if 'PUL - Pulmonary' in val:
            res += 100
        if 'LYM - Lymph nodes' in val:
            res += 1000
        if 'HEP - Hepatic' in val:
            res += 10000
        if 'PER - Peritoneum' in val:
            res += 100000
        if 'OTH - Other' in val:
            res += 1000000
        if 'BRA - Brain' in val:
            res += 10000000
        if 'PLE - Pleura' in val:
            res += 100000000
        if 'ADR - Adrenals' in val:
            res += 1000000000
        if 'MAR - Bone Marrow' in val:
            res += 10000000000
        return res

    # df = df['אבחנה-Location of distal metastases'].apply(lambda x: _y_labels_reformat(literal_eval(x)))
    df['אבחנה-Location of distal metastases'] = \
        df['אבחנה-Location of distal metastases'].apply(lambda x: _y_labels_reformat(literal_eval(x)))
    return df


# preprocessing applications:
def apply_preprocessing_metastases(df: pd.DataFrame) -> pd.DataFrame:
    """
    applies preprocessing for the data that is relevant to the first task
    """
    df = drop_features(df=df, features_to_save=FEATURES_METASTASES_DICT)

    df = clean_metastases_mark(df)

    # df = clean_surgery_name(df)

    df = clean_cancer_stage(df)

    df = clean_positive_nodes(df)

    df = clean_basic_stage(df)

    df = clean_her2(df)

    df = clean_breast_side(df=df)

    return df


def apply_preprocessing_tumor_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    applies preprocessing for the data that is relevant to the second task
    """

    df = drop_features(df=df, features_to_save=FEATURES_SIZE_DICT)

    df = clean_tumor_depth(df=df)

    df = clean_tumor_width(df=df)

    df = clean_metastases_mark(df=df)

    df = clean_cancer_stage(df=df)

    df = clean_surgery_name_1(df=df)

    # df = clean_surgery_name_2(df=df)

    df = clean_positive_nodes(df=df)

    df = clean_basic_stage(df=df)

    df = clean_her2(df=df)

    df = clean_breast_side(df=df)

    return df
