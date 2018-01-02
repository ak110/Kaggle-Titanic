"""データの読み込み＆前処理。"""
import pathlib

import pandas as pd

import pytoolkit as tk

TRAIN_FILE = pathlib.Path('data/train.csv')
TEST_FILE = pathlib.Path('data/test.csv')


@tk.log.trace()
def load_data():
    """データの読み込み＆前処理。"""
    df_train = pd.read_csv(TRAIN_FILE).sort_values('PassengerId')
    df_test = pd.read_csv(TEST_FILE).sort_values('PassengerId')
    df_train['source'] = 'train'
    df_test['source'] = 'test'
    logger = tk.log.get()
    logger.info('train = {}, test = {}'.format(len(df_train), len(df_test)))

    # trainとtestをいったんくっつけてから特徴を作る
    # (統計量などはtestも含んだものを使ってしまう(本当は怪しいけど))
    df = pd.concat([df_train, df_test], ignore_index=True)

    # PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # https://www.kaggle.com/nicapotato/titanic-feature-engineering

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['NameLength'] = df['Name'].apply(len)

    df['Title'] = 0
    df['Title'] = df.Name.str.extract(r'([A-Za-z]+)\.', expand=False)  # lets extract the Salutations
    df['Title'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col',
                         'Rev', 'Capt', 'Sir', 'Don'],
                        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other',
                         'Other', 'Mr', 'Mr', 'Mr'], inplace=True)

    # df['Age'].fillna(df.Age.median(), inplace=True)
    df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'Age'] = df.Age[df.Title == 'Mr'].mean()
    df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'Age'] = df.Age[df.Title == 'Mrs'].mean()
    df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'Age'] = df.Age[df.Title == 'Master'].mean()
    df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'Age'] = df.Age[df.Title == 'Miss'].mean()
    df.loc[(df.Age.isnull()) & (df.Title == 'Other'), 'Age'] = df.Age[df.Title == 'Other'].mean()
    df.drop('Title', axis=1, inplace=True)

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
    df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2}).astype(int)

    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    # 使わない列の削除
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    logger.debug('df dtypes, describe, head:\n{}\n\n{}\n\n{}'.format(df.dtypes, df.describe(), df.head()))

    df_train = df.loc[df['source'] == 'train']
    df_test = df.loc[df['source'] == 'test']
    X_train = df_train.drop(['Survived', 'source'], axis=1).values
    y_train = df_train['Survived'].values.astype(int)
    X_test = df_test.drop(['Survived', 'source'], axis=1).values
    return (X_train, y_train), X_test
