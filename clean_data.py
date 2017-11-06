# -*- coding: utf-8 -*-
"""
Clean dataset

@author: gshai
"""

import numpy as np
import pandas as pd

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)


#%%
#####################################################


RANK_DICT = {
    "Capt":       "officer",
    "Col":        "officer",
    "Major":      "officer",
    "Rev":        "church",
    "Jonkheer":   "royalty",
    "Don":        "royalty",
    "Sir":       "royalty",
    "Lady":      "royalty",
    "Dona":       "royalty",
    "the Countess": "royalty",
    "Master":    "normal",
    "Dr":         "normal",
    "Mr":        "normal",
    "Mrs":       "normal",
    "Mme":        "normal",
    "Ms":         "normal",
    "Miss":      "normal",
    "Mlle":       "normal",
}

RELATIONSHIP_DICT = {
    "Mrs":       "married",
    "Mme":        "married",
    "Ms":         "unmarried",
    "Miss":      "unmarried",
    "Mlle":       "unmarried",
    "Master":    "unmarried",
}


#####################################################


def load_data():
    '''load_data'''
    df_train = pd.read_csv('dataset/train.csv')
    df_train.insert(
        0,
        'Train',
        np.ones((df_train.shape[0], 1))
    )

    df_test = pd.read_csv('dataset/test.csv')
    df_test.insert(
        0,
        'Train',
        np.zeros((df_test.shape[0], 1))
    )

    df_test.insert(
        df_train.columns.get_loc('Survived'),
        'Survived',
        -np.ones((df_test.shape[0], 1))
    )

    df = pd.concat([df_train, df_test], axis=0).reset_index().drop('index', 1)

    return df


def age_group(x):
    '''age_group'''
    if x['Title'] == 'Master' or x['Age'] < 16:
        return 'young'
    else:
        return 'old'


def create_discrete_features(df):
    '''create_discrete_features'''
    descrete_cols = [
        'AgeGroup',
        'Relationship',
        'Rank',
        'Sex',
        'Pclass',
        'Deck',
    ]

    df_discrete_features = pd.DataFrame()

    for name in descrete_cols:
        for category in set(df[name].values):
            df_discrete_features["{}_{}".format(
                name, category)] = (df[name] == category)

    df_discrete_features = df_discrete_features.astype('int32')

    df = df.drop(descrete_cols, 1)

    df = pd.concat([df, df_discrete_features], axis=1)

    return df


def create_datasets(df, split=0.7):
    '''create_datasets'''
    assert isinstance(split, float)

    m_train = round(df[df['Train'] == 1].shape[0] * split)
    m_val = round(df[df['Train'] == 1].shape[0] - m_train)
    m_test = round(df[df['Train'] == 0].shape[0])

    ms = [m_train, m_val, m_test]

    df_train = df.loc[:m_train - 1, :].copy()
    df_val = df.loc[m_train:m_train + m_val - 1, :].copy()
    df_test = df.loc[m_train + m_val:, :].copy()

    dfs = [df_train, df_val, df_test]

    y_train = df_train['Survived'].copy()
    y_val = df_val['Survived'].copy()

    ys = [y_train, y_val, None]

    for idx, item in enumerate(dfs):
        dfs[idx] = item.drop(['Survived', 'Train'], 1)

    return dfs, ys, ms


def normalize_dataset(df_train, dfs):
    '''normalize_dataset'''
    normalize_cols = {
        'Age',
        'Fare',
        'SibSp',
        'Parch',
    }

    for col in normalize_cols:

        # find mean and variance of the feature column in training dataset
        mean_val = np.mean(df_train[col])
        var_val = np.var(df_train[col])

        # normalize the feature column in all datasets
        for idx, df in enumerate(dfs):
            df[col] = (df[col] - mean_val) / np.sqrt(var_val)
            dfs[idx] = df

    return dfs


#%%
#####################################################


if __name__ == '__main__':
    # Load data
    df = load_data()

    # Data cleaning, feature engineerng
    titles = df['Name'].apply(lambda x: x
                              .split(', ')[1]
                              .split('. ')[0])

    df.insert(
        df.columns.get_loc('Name') + 1,
        'Title',
        titles
    )

    df.insert(
        df.columns.get_loc('Title') + 1,
        'Rank',
        df['Title'].apply(lambda x: RANK_DICT.get(x, 'NA'))
    )

    df.insert(
        df.columns.get_loc('Title') + 1,
        'Relationship',
        df['Title'].apply(lambda x: RELATIONSHIP_DICT.get(x, 'NA'))
    )

    df.insert(
        df.columns.get_loc('Title') + 1,
        'AgeGroup',
        df.apply(age_group, axis=1)
    )

    n_tickets = df.groupby('Ticket').apply(lambda x: x.shape[0])
    df['Fare'] = df.apply(lambda x: x['Fare'] / n_tickets[x['Ticket']], axis=1)

    def what_cabin(x):
        if not pd.isnull(x):
            if x[0] in cabin_list:
                return x[0]
        else:
            return 'unknown'

    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G']
    df['Deck'] = df['Cabin'].apply(what_cabin)

    def check_married(x):
        if x['Title'] == 'Mrs':
            name = x['Name']
            if '(' in name:
                name = name.split(' (')[0]
                name = name.replace('Mrs.', 'Mr.').strip()
                name = name.split(" ")[:3]
                name = " ".join(name)
                return name

    def is_married(x):
        if x['Relationship'] == 'NA' and x['Title'] == 'Mr':
            name = x['Name']
            name = name.split(" ")[:3]
            name = " ".join(name)
            if name in married:
                x['Relationship'] = 'married'
        return x

    married = set(df.apply(check_married, axis=1))
    df = df.apply(is_married, axis=1)

    #sns.swarmplot(x="Status", y="Age", hue="Survived", data=df, split=True)
    #sns.violinplot(x="Status", y="Pclass", hue="Survived", data=df, split=True)

    df = create_discrete_features(df)

    drop_cols = [
        'PassengerId',
        'Title',
        'Name',
        'Ticket',
        'Cabin',
        'Embarked',
    ]

    df_id = df[['PassengerId', 'Train', 'Survived']]

    df = df.drop(drop_cols, 1)

    for col in df.columns:
        if df[col].isnull().values.any():
            df[col] = df[col].fillna(df[col].mean())

    df.to_csv('dataset/df.csv', index=False)
    df_id.to_csv('dataset/df_id.csv', index=False)
