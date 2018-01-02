"""学習。"""
import pathlib

import pandas as pd
import sklearn.externals.joblib
import autosklearn.classification  # pip install auto-sklearn

import data
import pytoolkit as tk

SAMPLE_SUBMIT_FILE = pathlib.Path('data/gender_submission.csv')
MODEL_DIR = pathlib.Path('models/auto-sklearn')


def _main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    logger.addHandler(tk.log.file_handler(MODEL_DIR / 'train.log'))

    (X_train, y_train), X_test = data.load_data()

    with tk.log.trace_scope('train+predict'):
        estimator = autosklearn.classification.AutoSklearnClassifier()
        estimator.fit(X_train, y_train)
        sklearn.externals.joblib.dump(estimator, str(MODEL_DIR / 'model.pkl'))
        pred_test = estimator.predict(X_test)

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('PassengerId')
    df_submit['Survived'] = pred_test
    df_submit.to_csv(MODEL_DIR / 'submit.csv', index=False)


if __name__ == '__main__':
    _main()
