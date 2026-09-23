# library-dictionary

Python主要ライブラリの関数・API逆引き辞書。ライブラリごとに、よく使う関数/メソッド/クラスを「用途・シグネチャ・実際に動かした使用例・注意点」の形式でまとめる。

姉妹プロジェクトの[bayesian-analysis-index](https://github.com/karahashimanato/bayesian-analysis-index)がベイズ分析の技術・手法を横断的に索引化しているのに対し、こちらはライブラリ単位でAPIリファレンスを整理する。

すべてのエントリは、記載バージョンのライブラリを実際に実行して検証した内容(シグネチャ・出力とも)であり、記憶やトレーニングデータからの推測では書いていない。

## 収録ライブラリ

| ライブラリ | 検証バージョン | 辞書 |
|---|---|---|
| numpy | 2.4.6 | [numpy/README.md](numpy/README.md) |
| pandas | 3.0.5 | [pandas/README.md](pandas/README.md) |
| scikit-learn | 1.9.0 | [scikit-learn/README.md](scikit-learn/README.md) |
| matplotlib | 3.11.1 | [matplotlib/README.md](matplotlib/README.md) |
| scipy | 1.18.1 | [scipy/README.md](scipy/README.md) |
| polars | 1.44.1 | [polars/README.md](polars/README.md) |
| seaborn | 0.13.2 | [seaborn/README.md](seaborn/README.md) |
| statsmodels | 0.14.6 | [statsmodels/README.md](statsmodels/README.md) |
| imblearn | 0.14.2 | [imblearn/README.md](imblearn/README.md) |
| optuna | 4.9.0 | [optuna/README.md](optuna/README.md) |
| xgboost | 3.4.1 | [xgboost/README.md](xgboost/README.md) |
| catboost | 1.2.10 | [catboost/README.md](catboost/README.md) |
| lightgbm | 4.7.0 | [lightgbm/README.md](lightgbm/README.md) |
| pytorch | 2.13.0 | [pytorch/README.md](pytorch/README.md) |
| pymc | 6.3.1 | [pymc/README.md](pymc/README.md) |
| pytensor | 3.3.0 | [pytensor/README.md](pytensor/README.md) |
| jax | 0.11.1 | [jax/README.md](jax/README.md) |
| ruptures | 1.1.10 | [ruptures/README.md](ruptures/README.md) |
| scikit-survival | 0.28.0 | [scikit-survival/README.md](scikit-survival/README.md) |
| tsfresh | 0.21.2 | [tsfresh/README.md](tsfresh/README.md) |
| pyod | 3.6.5 | [pyod/README.md](pyod/README.md) |
| featuretools | 1.31.0 | [featuretools/README.md](featuretools/README.md) |

## 運用ルール

- 新しいライブラリを追加したら、上の表に追記し、`<library>/README.md`を同じ形式(用途・シグネチャ・使用例・実行結果・注意点)で作成する。
- エントリは必ず対象バージョンを実際にインストールして実行し、シグネチャ・出力を確認してから記載する(推測で書かない)。
- ライブラリのバージョンが上がり仕様が変わった場合は、該当エントリを再検証してから更新し、検証バージョンの表記も更新する。
