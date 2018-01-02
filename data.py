"""データの読み込み＆前処理。"""
import pathlib

import pandas as pd

import pytoolkit as tk

TRAIN_FILE = pathlib.Path('data/train.csv')
TEST_FILE = pathlib.Path('data/test.csv')


@tk.log.trace()
def load_data(data):
    """データの読み込み＆前処理。"""
    df = pd.read_csv(str(data)).sort_values('PassengerId')

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
    return df


@tk.log.trace()
def load_train_data():
    """訓練データの読み込み。"""
    df = load_data(TRAIN_FILE)
    X = df.drop('Survived', axis=1).values
    y = df['Survived'].values
    return X, y


@tk.log.trace()
def load_test_data():
    """テストデータの読み込み。"""
    df = load_data(TEST_FILE)
    X = df.values
    return X
