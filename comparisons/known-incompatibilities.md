# 既知の非互換・バージョン間の落とし穴まとめ

このドキュメントは、[library-dictionary](../README.md) の22ライブラリを1つの共有venv(`/home/manaty/library-practicing/.venv`)で検証する過程で見つかった、**バージョン依存の互換性問題**を1か所にまとめたものです。各ライブラリ単体のAPI仕様は各ライブラリの`README.md`に譲り、ここでは「複数ライブラリの組み合わせで問題が起きるもの」と「横断的に知っておくべき破壊的変更のサマリ」だけを扱います。

品質方針: ここに書く内容は、既存の各README.mdに実際に記載されている検証済みの注意点を集約したものか、本ドキュメント作成時に実際に`/home/manaty/library-practicing/.venv/bin/python`でコードを実行して確認したものだけです。推測や「起きそうな問題」の記載はしていません。

---

## 1. pandas 3.0系との非互換一覧

### 1.1 featuretools 1.31.0 は pandas 3.0系で動作しない(唯一の確認済みクロスライブラリ非互換)

| 項目 | 内容 |
|---|---|
| 症状 | `EntitySet.add_dataframe()` が必ず `woodwork.exceptions.WoodworkNotInitError` で失敗する |
| 原因 | pandas 3.0で `DataFrame` 拡張アクセサ(`pandas.core.accessor.Accessor.__get__`)のインスタンスキャッシュが廃止された。`df.ww` にアクセスするたびに毎回新しい `PandasTableAccessor` インスタンスが生成されるため、`df.ww.init(...)` で設定したschemaがそのインスタンス限りで消え、直後の `df.ww.schema` が `None` に戻ってしまう |
| 対象バージョン | featuretools 1.31.0 + woodwork 0.31.0(2026-09時点で最新)+ pandas 3.0.5 |
| 回避策 | featuretoolsを使う場合はpandas 2.x系を使う(`pip install featuretools` の依存指定が `pandas>=2.0.0` で上限がないため、誤ってpandas 3.0系を入れると機能しなくなる点に注意) |
| 再検証 | 本ドキュメント作成時に `/home/manaty/library-practicing/.venv/bin/python`(pandas 3.0.5, featuretools 1.31.0)で以下を実行し、再現を確認済み: |

```python
import pandas as pd, featuretools as ft
df = pd.DataFrame({"id": [1,2,3], "value": [10.0,20.0,30.0]})
es = ft.EntitySet(id="test")
es = es.add_dataframe(dataframe_name="df", dataframe=df, index="id")
# => WoodworkNotInitError: Woodwork not initialized for this DataFrame.
#    Initialize by calling DataFrame.ww.init
```

詳細・調査過程(woodworkのソースを読んで原因を特定した経緯や、隔離venvでの検証方法)は [featuretools/README.md](../featuretools/README.md) の冒頭注記を参照。

### 1.2 他のライブラリはpandas 3.0で問題なく動作することを確認した

以下は、pandas 3.0.5の主な仕様変更(デフォルト文字列dtypeが`object`ではなく`str`/`StringDtype`になったこと、Copy-on-Write(CoW)が常時有効であること)が、この venv 内の他のダウンストリームライブラリに影響しないか、実際にコードを実行して確認した結果です。**すべて問題なく動作しました。**

| ライブラリ | 確認内容 | 結果 |
|---|---|---|
| scikit-learn 1.9.0 | pandas 3.0の`str`dtype列を含むDataFrameを`ColumnTransformer`+`OneHotEncoder`に通し、`LogisticRegression.fit()`まで実行 | 問題なし |
| seaborn 0.13.2 | `sns.load_dataset("tips")`の読み込みと`scatterplot`/`barplot`の描画 | 問題なし(`load_dataset`はネットワーク経由でCSVを取得し`category`dtypeとして読み込まれる) |
| statsmodels 0.14.6 | `str`dtype列を含むDataFrameに対し、`smf.ols("y ~ x1 + C(grp)", data=df).fit()`という formula API を実行 | 問題なし |
| tsfresh 0.21.2 | long format(`id`/`time`/`value`列)のDataFrameを`extract_features()`に渡す | 問題なし |
| scikit-survival 0.28.0 | `load_whas500()`(`category`dtype列を含む)を`sksurv.preprocessing.OneHotEncoder`→`CoxPHSurvivalAnalysis.fit()`に通す | 問題なし |
| imblearn 0.14.2 | pandas DataFrame/Seriesを`SMOTE().fit_resample()`に通す | 問題なし |

結論: この venv 内では、pandas 3.0の「デフォルト文字列dtype」「常時CoW」が原因でダウンストリームライブラリが壊れる例は **featuretools 1.31.0(woodwork経由)以外には見つからなかった**。scikit-learn・statsmodels・seaborn・tsfresh・scikit-survival・imblearnはいずれも、pandasの`str`dtype列や`category`dtype列を素通りできる実装になっており、影響を受けない。

---

## 2. 各ライブラリ単体でのバージョン固有の破壊的変更・非推奨まとめ

以下は「他ライブラリとの組み合わせ」ではなく、各ライブラリ単体の中でのバージョン間の破壊的変更・非推奨化です。詳細はリンク先の各README.mdを参照してください。

| ライブラリ | 検証バージョン | 内容(サマリ) | 詳細 |
|---|---|---|---|
| pandas | 3.0.5 | 文字列列のデフォルトdtypeが`object`→`str`(StringDtype)に変更。CoWが常時有効(無効化不可)で`copy`引数が非推奨に。`fillna(method=...)`は廃止済みで`ffill()`/`bfill()`を使う。デフォルト時間解像度が`datetime64[ns]`→`datetime64[us]`に変更。`groupby.apply`で`include_groups`のデフォルトが`False`に変更。`Styler.applymap()`が完全廃止され`map()`に統一。`day_name()`の戻り値dtypeも`str`に変更 | [pandas/README.md](../pandas/README.md) |
| matplotlib | 3.11.1 | `Axes.boxplot()`の旧引数`labels=`が完全削除され`TypeError`になる(`tick_labels=`を使う)。`vert=`引数がmatplotlib 3.11で非推奨化(3.13で削除予定、`orientation=`を使う)。`ListedColormap`の`N`引数が非推奨化 | [matplotlib/README.md](../matplotlib/README.md) |
| scikit-learn | 1.9.0 | `LogisticRegression`の`penalty`、`SVC`の`probability`(1.11で削除予定、`CalibratedClassifierCV`を案内)、`GradientBoostingClassifier`の`criterion`がいずれも非推奨(デフォルト値が`'deprecated'`表示) | [scikit-learn/README.md](../scikit-learn/README.md) |
| polars | 1.44.1 | 旧`df.melt(id_vars=..., value_vars=...)`が非推奨、`df.unpivot(index=..., on=...)`に統一。`empty_as_null`のデフォルト挙動が将来変更予定 | [polars/README.md](../polars/README.md) |
| lightgbm | 4.7.0 | `eval_set=[(X, y)]`という旧来の書き方が`LGBMDeprecationWarning`。キーワード専用の`eval_X`/`eval_y`に移行中 | [lightgbm/README.md](../lightgbm/README.md) |
| xgboost | 3.4.1 | `XGBRFClassifier`が非推奨。ランダムフォレストとしての機能(early stopping等)が不完全なラッパーであるため、`num_parallel_tree`+`n_estimators=1`か`sklearn.ensemble`側の実装を推奨 | [xgboost/README.md](../xgboost/README.md) |
| imblearn | 0.14.2 | `RUSBoostClassifier`の`algorithm`引数が非推奨(デフォルト値が`'deprecated'`表示) | [imblearn/README.md](../imblearn/README.md) |
| pyod | 3.6.5 | `behaviour='old'`引数がシグネチャ上残存しているが、依拠していたscikit-learn側の`IsolationForest`の同名引数は既に廃止済みで、実質的に無意味な互換名残 | [pyod/README.md](../pyod/README.md) |
| pytorch | 2.13.0 | `softmax`で`dim`省略時に`UserWarning`(暗黙の次元選択が非推奨) | [pytorch/README.md](../pytorch/README.md) |
| pymc | 6.3.1 | `find_constrained_prior`が非推奨、PreliZの`maxent`関数への移行を推奨(`FutureWarning`) | [pymc/README.md](../pymc/README.md) |
| pytensor | 3.3.0 | `scan()`の戻り値仕様が変更中(将来`(outputs, updates)`タプルから`outputs`のみに)。`grad`/`L_op`の直接実装が非推奨化され新しい`pullback`フックに移行中。`Rop`/`Lop`も`pushforward`/`pullback`への改名が進行中 | [pytensor/README.md](../pytensor/README.md) |
| jax | 0.11.1 | `jax.random.PRNGKey()`は後方互換API。この検証バージョンではまだ`DeprecationWarning`は出ないが、公式は新規コードで`key()`を使うことを推奨 | [jax/README.md](../jax/README.md) |
| scipy | 1.18.1 | `interp1d`はレガシー的位置づけ(本バージョンでは廃止警告は出ないが、新規コードには`make_interp_spline`/`CubicSpline`を推奨) | [scipy/README.md](../scipy/README.md) |
| seaborn | 0.13.2 | 旧`ci=`引数が非推奨化され`errorbar=`に統合済み | [seaborn/README.md](../seaborn/README.md) |
| featuretools | 1.31.0 | 上記1.1参照(pandas 3.0系との非互換) | [featuretools/README.md](../featuretools/README.md) |

numpy・statsmodels・catboost・optuna・ruptures・scikit-survival・tsfreshについては、本ドキュメント作成時点で各README.mdに「バージョン固有の破壊的変更・非推奨」に該当する記載は見つからなかった(通常のAPI仕様の記述のみ)。

---

## 3. このリポジトリの検証環境

すべてのエントリは、以下の共有venv(`/home/manaty/library-practicing/.venv`)に同居させた状態で、実際にコードを実行して検証しています(featuretoolsのみ、1.1節の理由で別途pandas 2.x系の隔離venvでも検証)。

検証日: 2026-09-24

| ライブラリ | バージョン |
|---|---|
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| scipy | 1.18.1 |
| polars | 1.44.1 |
| seaborn | 0.13.2 |
| statsmodels | 0.14.6 |
| imblearn(imbalanced-learn) | 0.14.2 |
| optuna | 4.9.0 |
| xgboost | 3.4.1 |
| catboost | 1.2.10 |
| lightgbm | 4.7.0 |
| pytorch | 2.13.0 |
| pymc | 6.3.1 |
| pytensor | 3.3.0 |
| jax | 0.11.1 |
| ruptures | 1.1.10 |
| scikit-survival | 0.28.0 |
| tsfresh | 0.21.2 |
| pyod | 3.6.5 |
| featuretools | 1.31.0(※pandas 2.x系の隔離venvで検証。詳細は[featuretools/README.md](../featuretools/README.md)) |
| woodwork(featuretoolsの依存) | 0.31.0 |

バージョン一覧は[トップのREADME.md](../README.md)の収録ライブラリ表と同期しています。ライブラリのバージョンが上がった場合は、この表と各README.mdの両方を再検証のうえ更新してください。
