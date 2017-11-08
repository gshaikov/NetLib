# -*- coding: utf-8 -*-
"""
Clean dataset

@author: gshai
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", color_codes=True)


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


def feature_engineering():
    # Load data
    df = load_data()

    # shuffle dataframe 5 times
    for _ in range(5):
        df = df.sample(frac=1).reset_index(drop=True)

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

    def age_group(x):
        '''age_group'''
        if x['Title'] == 'Master' or x['Age'] < 15:
            return 'kid'
        elif x['Age'] < 30:
            return 'young'
        elif x['Age'] < 60:
            return 'adult'
        elif x['Age'] > 60:
            return 'old'
        else:
            return 'NA'

    df.insert(
        df.columns.get_loc('Age') + 1,
        'AgeGroup',
        df.apply(age_group, axis=1)
    )

    def fare_group(x):
        '''fare_group'''
        if x['Fare'] > 0:
            return np.ceil(x['Fare'] / 10.0) * 10.0

    n_tickets = df.groupby('Ticket').apply(lambda x: x.shape[0])
    df['Fare'] = df.apply(lambda x: x['Fare'] / n_tickets[x['Ticket']], axis=1)

    fare_avg = df[df['Fare'] > 0].groupby('Pclass')['Fare'].mean()

    df['Fare'] = df.apply(lambda x: fare_avg[x['Pclass']]
                          if not x['Fare'] > 0 else x['Fare'], axis=1)

    df.insert(
        df.columns.get_loc('Fare') + 1,
        'FareGroup',
        df.apply(fare_group, axis=1)
    )

    def what_cabin(x):
        if not pd.isnull(x):
            if x[0] in cabin_list:
                return x[0]
        else:
            return 'NA'

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

    return df


def feature_dropping(df):
    # drop some columns
    drop_cols = [
        'PassengerId',
        # 'Pclass',
        'Name',
        'Title',
        # 'Relationship',
        # 'Rank',
        # 'Sex',
        # 'Age',
        'AgeGroup',
        # 'SibSp',
        # 'Parch',
        'Ticket',
        # 'Fare',
        'FareGroup',
        'Cabin',
        # 'Deck',
        # 'Embarked',
    ]

    df = df.drop(drop_cols, 1)

    # deal with nan values
    for col in df.columns:
        if df[col].isnull().values.any():
            if col == 'Age':
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna('NA')

    return df


def create_discrete_features(df):
    '''create_discrete_features'''
    descrete_cols = [
        'Relationship',
        'Rank',
        'Sex',
        'AgeGroup',
        'Embarked',
        'Deck',
    ]

    df_discrete_features = pd.DataFrame()

    for name in descrete_cols:
        if name in df.columns:
            for category in set(df[name].values):
                df_discrete_features["{}_{}".format(
                    name, category)] = (df[name] == category)

    df_discrete_features = df_discrete_features.astype('int32')

    df = df.drop(descrete_cols, 1, errors='ignore')

    df = pd.concat([df, df_discrete_features], axis=1)

    return df

#%%
#####################################################


if __name__ == '__main__':
    df = feature_engineering()

#    df_plot = df[df['Train']==1]

#    sns.swarmplot(x="Pclass", y="Age", hue="Survived", data=df_plot, split=False)
#    sns.violinplot(x="Status", y="Pclass", hue="Survived", data=df, split=True)

#    plt.figure()
#    plt.scatter(x=df_plot['Age'], y=df_plot['Fare'], c=df_plot['Survived'])
#    plt.show()

    # create separate dataset with IDs
    df_id = df[['PassengerId', 'Train', 'Survived']]

    df = feature_dropping(df)

    df = create_discrete_features(df)

    df.to_csv('dataset/df.csv', index=False)
    df_id.to_csv('dataset/df_id.csv', index=False)


#%%
#####################################################

def create_datasets(df, split=0.7):
    '''create_datasets'''
    assert isinstance(split, float)

    m_train = round(df[df['Train'] == 1].shape[0] * split)
    m_val = round(df[df['Train'] == 1].shape[0] - m_train)
    m_test = round(df[df['Train'] == 0].shape[0])

    ms = [m_train, m_val, m_test]

    df_train = df[df['Train'] == 1].iloc[:m_train, :].copy()
    df_val = df[df['Train'] == 1].iloc[m_train:, :].copy()
    df_test = df[df['Train'] == 0].copy()

    dfs = [df_train, df_val, df_test]

    y_train = df_train[['Survived']].copy()
    y_val = df_val[['Survived']].copy()

    ys = [y_train, y_val, pd.DataFrame()]

    for idx, item in enumerate(dfs):
        dfs[idx] = item.drop(['Survived', 'Train'], 1)

    return dfs, ys, ms


def normalize_dataset(dfs):
    '''normalize_dataset'''
    cols_skewed = [
        'Fare',
    ]

    dfs[0][cols_skewed] = dfs[0][cols_skewed].apply(np.log, axis=0)

    df_train = dfs[0]

    for col in df_train.columns:
        # find mean and SD of the feature column in training dataset
        mean_val = np.mean(df_train[col])
        # sd_val = np.std(df_train[col])
        delta_val = np.max(df_train[col]) - np.min(df_train[col])

        # normalize the feature column in all datasets
        for idx, df in enumerate(dfs):
            df[col] = (df[col] - mean_val) / delta_val
            dfs[idx] = df

    return dfs
