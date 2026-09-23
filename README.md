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

## 運用ルール

- 新しいライブラリを追加したら、上の表に追記し、`<library>/README.md`を同じ形式(用途・シグネチャ・使用例・実行結果・注意点)で作成する。
- エントリは必ず対象バージョンを実際にインストールして実行し、シグネチャ・出力を確認してから記載する(推測で書かない)。
- ライブラリのバージョンが上がり仕様が変わった場合は、該当エントリを再検証してから更新し、検証バージョンの表記も更新する。
