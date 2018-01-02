# 「KaggleのTitanicをやってみた」的コード

## さいしょ

適当にググって出てきたfeature engineeringと、RandomForest / auto-sklearnというやる気のない構成。

- RandomForest: 0.76555
- auto-sklearn: 0.77511

## bugfix

train / testで別々にmean()とかしてたのを全体で取るように修正。

- RandomForest: 0.74641 (oob score: 0.8215488215488216)

(あれー？)
