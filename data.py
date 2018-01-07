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
    df_train['IsTrain'] = True
    df_train['IsTest'] = False
    df_test['IsTrain'] = False
    df_test['IsTest'] = True
    logger = tk.log.get()
    logger.info('train = {}, test = {}'.format(len(df_train), len(df_test)))

    # trainとtestをいったんくっつけてから特徴を作る
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
    df['Title_Mr'] = df.Title == 'Mr'
    df['Title_Mrs'] = df.Title == 'Mrs'
    df['Title_Master'] = df.Title == 'Master'
    df['Title_Miss'] = df.Title == 'Miss'
    df['Title_Other'] = df.Title == 'Other'

    # df['Age'].fillna(df.Age.median(), inplace=True)
    age_mr = df.Age[df.IsTest & df.Title_Mr].mean()
    age_mrs = df.Age[df.IsTest & df.Title_Mrs].mean()
    age_master = df.Age[df.IsTest & df.Title_Master].mean()
    age_miss = df.Age[df.IsTest & df.Title_Miss].mean()
    age_other = df.Age[df.IsTest & df.Title_Other].mean()
    logger.info('Age of Mr: {}'.format(age_mr))
    logger.info('Age of Mrs: {}'.format(age_mrs))
    logger.info('Age of Master: {}'.format(age_master))
    logger.info('Age of Miss: {}'.format(age_miss))
    logger.info('Age of Other: {}'.format(age_other))
    df.loc[df.Age.isnull() & df.Title_Mr, 'Age'] = age_mr
    df.loc[df.Age.isnull() & df.Title_Mrs, 'Age'] = age_mrs
    df.loc[df.Age.isnull() & df.Title_Master, 'Age'] = age_master
    df.loc[df.Age.isnull() & df.Title_Miss, 'Age'] = age_miss
    df.loc[df.Age.isnull() & df.Title_Other, 'Age'] = age_other
    df.drop('Title', axis=1, inplace=True)

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
    df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2}).astype(int)

    df['Fare'] = df['Fare'].fillna(df_test['Fare'].mean())

    # 使わない列の削除
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    logger.debug('df dtypes, describe, head:\n{}\n\n{}\n\n{}'.format(df.dtypes, df.describe(), df.head()))

    df_train = df.loc[df.IsTrain]
    df_test = df.loc[df.IsTest]
    X_train = df_train.drop(['Survived', 'IsTrain', 'IsTest'], axis=1).values.astype(float)
    y_train = df_train['Survived'].values.astype(int)
    X_test = df_test.drop(['Survived', 'IsTrain', 'IsTest'], axis=1).values.astype(float)
    return (X_train, y_train), X_test
