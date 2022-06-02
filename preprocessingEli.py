import pandas as pd
import numpy as np

def dropFeatures(X: pd.DataFrame):
    irrelevant_features = [' Form Name', ' Hospital', 'User Name', 'אבחנה-Histological diagnosis', 'אבחנה-Histopatological degree', 'אבחנה-Ivi -Lymphovascular invasion'
    ,'אבחנה-N -lymph nodes mark (TNM)', 'אבחנה-Nodes exam', 'אבחנה-Positive nodes', 'אבחנה-Stage', 'אבחנה-Surgery date1', 'אבחנה-Surgery date2', 'אבחנה-Surgery date3'
    ,'אבחנה-Surgery sum', 'אבחנה-Tumor depth', 'אבחנה-Tumor width', 'surgery before or after-Activity date', 'surgery before or after-Actual activity']
    for feature in irrelevant_features:
        X = X.drop(feature, axis=1)
    return X


def basicPreprocessing(X: pd.DataFrame)->pd.DataFrame:
    X = X[['אבחנה-M -metastases mark (TNM)','אבחנה-Stage', 'אבחנה-Surgery name1', 'אבחנה-Surgery name2', 'אבחנה-Age', 'אבחנה-Side',
           'אבחנה-Positive nodes', 'surgery before or after-Actual activity']]

    # 'אבחנה-M -metastases mark (TNM)'
    # X['אבחנה-M -metastases mark (TNM)'].fillna('nan', inplace=True)
    # uniques = [elem for elem in X['אבחנה-M -metastases mark (TNM)'].unique()]
    # for elem in uniques:
    #     if type(elem) != str:
    #         uniques.remove(elem)
    X = pd.get_dummies(X, prefix='אבחנה-M -metastases mark (TNM)', columns=['אבחנה-M -metastases mark (TNM)'])
    # X.rename(columns={'אבחנה-M -metastases mark (TNM)_M0':'metastases mark (TNM)-M0', 'אבחנה-M -metastases mark (TNM)_M1':'metastases mark (TNM)-M1'}, inplace=True)
    # uniquesM1 = [elem for elem in uniques if '1' in elem]
    # print(uniques)
    # X['metastases mark (TNM)-M1'] = X[uniquesM1].max(axis=1)
    # print(uniques)
    # X.drop('אבחנה-M -metastases mark (TNM)_MX', axis=1, inplace=True)
    # X.drop('אבחנה-M -metastases mark (TNM)_Not yet Established', axis=1, inplace=True)
    # X.drop('אבחנה-M -metastases mark (TNM)_M1a', axis=1, inplace=True)
    # X.drop('אבחנה-M -metastases mark (TNM)_M1b', axis=1, inplace=True)

    # אבחנה-Surgery name1, אבחנה-Surgery name2
    X['אבחנה-Surgery name1'].fillna(0, inplace=True)
    X['אבחנה-Surgery name2'].fillna(0, inplace=True)
    X = pd.get_dummies(X, prefix='אבחנה-Surgery name1', columns=['אבחנה-Surgery name1'])
    X = pd.get_dummies(X, prefix='אבחנה-Surgery name2', columns=['אבחנה-Surgery name2'])

    # אבחנה-Stage
    X['אבחנה-Stage'].fillna(0, inplace=True)
    stageDictionary = {'Stage2a':2, 'Stage4':4, 'Stage1':1, 'Stage3b':3, 'Stage2b':2, 'LA':0, 'Stage1c':1, 'Stage2':2, 'Stage3c':3,
     'Stage0':0, 'Stage1b':1, 'Stage3a':3, 'Stage3':3, 'Stage1a':1, 'Stage0is':0,'Not yet Established':0, 'Stage0a':0}
    X['אבחנה-Stage'] = X['אבחנה-Stage'].replace(stageDictionary)

    # אבחנה-Side
    X = pd.get_dummies(X, prefix='אבחנה-Side', columns=['אבחנה-Side'])

    # אבחנה-Positive nodes
    X['אבחנה-Positive nodes'].fillna(0.0) # TODO: fill nan with mean value



    return X


def basicPreprocessing1(X: pd.DataFrame)->pd.DataFrame:
    X = X[['אבחנה-Tumor depth', 'אבחנה-Tumor width', 'אבחנה-Age', 'אבחנה-M -metastases mark (TNM)', 'אבחנה-Stage', 'אבחנה-Surgery name1', 'אבחנה-Surgery name2', 'אבחנה-Positive nodes']]

    # אבחנה-Tumor depth
    X['אבחנה-Tumor depth'].fillna(0, inplace=True)

    # אבחנה-Tumor width
    X['אבחנה-Tumor width'].fillna(0, inplace=True)

    # 'אבחנה-M -metastases mark (TNM)'
    X = pd.get_dummies(X, prefix='אבחנה-M -metastases mark (TNM)', columns=['אבחנה-M -metastases mark (TNM)'])

    # אבחנה-Stage
    X['אבחנה-Stage'].fillna(0, inplace=True)
    stageDictionary = {'Stage2a':2, 'Stage4':4, 'Stage1':1, 'Stage3b':3, 'Stage2b':2, 'LA':0, 'Stage1c':1, 'Stage2':2, 'Stage3c':3,
     'Stage0':0, 'Stage1b':1, 'Stage3a':3, 'Stage3':3, 'Stage1a':1, 'Stage0is':0,'Not yet Established':0, 'Stage0a':0}
    X['אבחנה-Stage'] = X['אבחנה-Stage'].replace(stageDictionary)

    # אבחנה-Surgery name1, אבחנה-Surgery name2
    X['אבחנה-Surgery name1'].fillna(0, inplace=True)
    X['אבחנה-Surgery name2'].fillna(0, inplace=True)
    X = pd.get_dummies(X, prefix='אבחנה-Surgery name1', columns=['אבחנה-Surgery name1'])
    X = pd.get_dummies(X, prefix='אבחנה-Surgery name2', columns=['אבחנה-Surgery name2'])

    # אבחנה-Positive nodes
    X['אבחנה-Positive nodes'].fillna(0.0) # TODO: fill nan with mean value





if __name__ == '__main__':
    X = pd.read_csv('/Users/elilevinkopf/Documents/Ex22B/IML/Hackathon/IML_hackathon/resorces/origin_data/train.feats.csv', low_memory=False)
    # dropFeatures(X)
    processed_X = basicPreprocessing(X)
    processed_X.to_csv('baseLine.csv')
    print(type(str([1,2,3])))
    # Y = pd.read_csv('/Users/elilevinkopf/Documents/Ex22B/IML/Hackathon/IML_hackathon/resorces/origin_data/train.labels.0.csv', low_memory=False)
    # print(Y['אבחנה-Location of distal metastases'].unique())
    
