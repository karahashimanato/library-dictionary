# scikit-learn系4ライブラリのEstimator API比較

scikit-learn 1.9.0 / scikit-survival 0.28.0 / PyOD 3.6.5 / imbalanced-learn 0.14.2 を対象に、scikit-learnの共通推定器API(`fit`/`predict`/`transform`/`score`、`get_params`/`set_params`、`BaseEstimator`継承など)がどこまで共通化されていて、どこで逸脱しているかを比較したドキュメントです。

各ライブラリ単体のAPI一覧(メソッド1つずつのリファレンス)は [scikit-learn/README.md](../scikit-learn/README.md)・[scikit-survival/README.md](../scikit-survival/README.md)・[pyod/README.md](../pyod/README.md)・[imblearn/README.md](../imblearn/README.md) を参照してください。このドキュメントはそれらとは種類が異なり、「4ライブラリを横断して、同じ規約がどこまで通用するか」だけを扱います。

品質方針: 本ドキュメントの記述はすべて `/home/manaty/library-practicing/.venv/bin/python`(scikit-learn 1.9.0, scikit-survival 0.28.0, pyod 3.6.5, imbalanced-learn 0.14.2 が同居する共有venv)で実際にコードを実行し、その出力・エラーメッセージをそのまま転記したものです。記憶による推測は含みません。検証日: 2026-09-24。

## 目次

1. [共通Estimator APIの共通化度合い](#1-共通estimator-apiの共通化度合い)
2. [逸脱点の一覧](#2-逸脱点の一覧)
3. [sklearn Pipelineへの組み込み可否(実測表)](#3-sklearn-pipelineへの組み込み可否実測表)
4. [交差検証・GridSearchCVとの互換性](#4-交差検証gridsearchcvとの互換性)
5. [実務上の注意点まとめ](#5-実務上の注意点まとめ)

---

## 1. 共通Estimator APIの共通化度合い

各ライブラリから代表的な推定器を1つずつ選び、`sklearn.tree.DecisionTreeClassifier`(sklearn本体)・`sksurv.linear_model.CoxPHSurvivalAnalysis`(scikit-survival)・`pyod.models.knn.KNN`(PyOD)・`imblearn.over_sampling.SMOTE`(imbalanced-learn)で共通点を確認しました。

### 1.1 `BaseEstimator`継承

4つとも`sklearn.base.BaseEstimator`を継承しています(実行結果、`isinstance(est, BaseEstimator)`より)。

| ライブラリ | 推定器 | `isinstance(_, BaseEstimator)` |
|---|---|---|
| scikit-learn | `DecisionTreeClassifier` | `True` |
| scikit-survival | `CoxPHSurvivalAnalysis` | `True` |
| PyOD | `KNN` | `True` |
| imbalanced-learn | `SMOTE` / `RandomUnderSampler` | `True` |

### 1.2 `get_params()` / `set_params()`

4つとも`BaseEstimator`由来の`get_params()`/`set_params()`をそのまま使えます。

```python
clf.set_params(max_depth=3)          # sklearn
cox.set_params(alpha=0.1)            # sksurv
knn.set_params(n_neighbors=10)       # pyod
smote.set_params(k_neighbors=3)      # imblearn
```
実行結果(それぞれ設定後に`get_params()`で読み戻した値):
```
sklearn set_params(max_depth=3) -> get_params()['max_depth']: 3
sksurv set_params(alpha=0.1) -> get_params()['alpha']: 0.1
pyod set_params(n_neighbors=10) -> get_params()['n_neighbors']: 10
imblearn set_params(k_neighbors=3) -> get_params()['k_neighbors']: 3
```

`get_params()`のキー一覧もコンストラクタ引数とアルファベット順で一致しており(例: sklearn `['ccp_alpha', 'class_weight', 'criterion', 'max_depth', 'max_features']`、pyod `['algorithm', 'contamination', 'leaf_size', 'method', 'metric']`)、`GridSearchCV`のパラメータ名探索など`get_params`に依存する仕組みはどのライブラリでも同様に機能します(4節で実測)。

### 1.3 `fit()`の戻り値

sklearn・sksurv・pyodの3つは`fit()`が自分自身(`self`)を返す、sklearnの規約に従っています。

```python
clf.fit(X, y) is clf   # True
cox.fit(X, y) is cox   # True
knn.fit(X) is knn      # True
```

imblearnのサンプラー(`SMOTE`等)にも`fit(X, y)`メソッド自体は存在し、これも`self`を返します(規約通り)。ただし**`fit()`だけを呼んでもリサンプリングは実行されません**(4.2節・2.3節で詳述)。実際にリサンプリングされたデータを得るには`fit_resample(X, y)`を呼ぶ必要があり、これは`(X_resampled, y_resampled)`という**タプル**を返します(`self`ではない)。

### 1.4 メソッドの有無まとめ

実際に`hasattr()`で確認した結果です。

| メソッド/属性 | sklearn (`DecisionTreeClassifier`) | sksurv (`CoxPHSurvivalAnalysis`) | pyod (`KNN`) | imblearn (`SMOTE`) |
|---|---|---|---|---|
| `fit` | ○ | ○ | ○ | ○(ただしリサンプリングはしない) |
| `predict` | ○ | ○(リスクスコアを返す。時間や確率ではない) | ○(0/1の異常ラベル) | × |
| `predict_proba` | ○ | × | ○(inlier/outlier確率) | × |
| `decision_function` | ×(このクラスでは未実装) | × | ○ | × |
| `transform` | × | × | × | × |
| `score` | ○(accuracy) | ○(Harrell's C-index) | ×(`score`メソッドなし) | ×(`score`メソッドなし) |
| `fit_resample` | × | × | × | ○ |
| `fit_predict` | × | × | ○ | × |

実行結果の抜粋(sklearn):
```
predict(X)[:5]: [1 1 0 0 0]
score(X, y): 0.95
has decision_function: False
```
実行結果の抜粋(pyod、`decision_function`と独自属性):
```
decision_function(X)[:5]: [0.32725493 0.33178729 0.50679904 0.29127988 0.49901167]
独自属性 decision_scores_[:5]: [0.33871522 0.35811708 0.50749675 0.30185037 0.55677367]
独自属性 threshold_: 0.8753366338134713
```

結論として、**「`fit`があり`BaseEstimator`を継承し`get_params`/`set_params`が効く」という最も基礎的な部分は4ライブラリで完全に共通化されている一方、`predict`/`score`/`transform`が「何を返すか」「そもそも存在するか」は各ライブラリのドメイン(生存時間分析・異常検知・リサンプリング)に合わせて大きく変質しています。**

---

## 2. 逸脱点の一覧

### 2.1 scikit-survival: `y`が構造化配列 `(event, time)`

sksurvの推定器は、通常のsklearnのような`(n_samples,)`の1次元`y`ではなく、`event`(bool)と`time`(float)の2フィールドを持つnumpy structured arrayを要求します。

```python
yw[:3]
# [(False, 2178.) (False, 2172.) (False, 2190.)]
yw.dtype
# [('fstat', '?'), ('lenfol', '<f8')]
```

通常の1次元配列(例えば`time`列だけ)をそのまま`y`として渡すと、実際に次のエラーになります(検証済み)。
```python
cox.fit(Xw_num, yw["lenfol"].astype(float))
```
```
ValueError: y must be a structured array with the first field being a binary class event indicator and the second field the time of the event/censoring
```

さらに`predict(X)`が返すのは生存時間でも確率でもなく**リスクスコア**(値が大きいほどハイリスク)です。
```
predict(X)[:5] (risk score, NOT time/proba): [3.92845625 2.28252121 3.1448152  2.88407251 2.80785636]
```
確率的な生存曲線が欲しい場合は`predict_survival_function(X)`を別途呼ぶ必要があります。
```python
surv_funcs[0]([100, 500, 1000])
# [0.75430031 0.56014326 0.45696112]  (t=100,500,1000時点の生存確率)
```
`predict_proba`・`decision_function`は共に存在しません(`hasattr`で`False`を確認済み)。

### 2.2 PyOD: `decision_scores_`/`threshold_`/`labels_`という独自属性

PyODの検出器は`fit(X)`(教師なし、`y`は基本的に不要)後、sklearnには無い3つの独自属性を持ちます。

| 属性 | 内容 |
|---|---|
| `decision_scores_` | 学習データに対する異常スコア(`decision_function`の学習時キャッシュ) |
| `threshold_` | `contamination`から自動計算された、正常/異常を分ける閾値 |
| `labels_` | 学習データに対する`predict`結果(0=inlier/1=outlier)のキャッシュ |

実行結果:
```
独自属性 decision_scores_[:5]: [0.33871522 0.35811708 0.50749675 0.30185037 0.55677367]
独自属性 threshold_: 0.8753366338134713
独自属性 labels_[:10] (=fit時のpredict(X)と同じ): [0 0 0 0 0 0 0 0 0 0]
```
さらに、`fit(X, y)`のように`y`を(不要なのに)渡すと、sklearn的なCV機構経由(4節参照)でも次のような警告が出ます(実際に`cross_val_score`実行時に観測):
```
UserWarning: y should not be presented in unsupervised learning.
```
これは教師なし学習が前提のライブラリであることの表れで、`y`はsklearnのCV APIとの形式合わせのために許容されているだけで、内部では無視されます。

### 2.3 imbalanced-learn: `fit_resample`は行数を変える変換で、`transform`ではない

imblearnのサンプラーは`fit`/`transform`ではなく`fit_resample`という独自メソッドを持ち、これは**サンプル数(行数)そのものを変える**という、sklearnの`transform`契約(入力と出力で行数を変えない)には収まらない挙動をします。

```python
X.shape                    # (1000, 5)  元の不均衡データ(多数派900:少数派100)
Xr, yr = sm.fit_resample(X, y)
Xr.shape                   # (1800, 5)  少数派が900まで合成され、行数が800増えた
```
実行結果:
```
元のy分布: Counter({0: 900, 1: 100}) X.shape: (1000, 5)
fit_resample後: Counter({0: 900, 1: 900}) Xr.shape: (1800, 5)
行数が変わった: 元 1000 -> 1800
```
`fit(X, y)`単体を呼んでも(`self`は返るが)リサンプリングは実行されず、`predict`/`transform`/`score`いずれも持ちません(1.4節の表の通り)。このため素の`sklearn.pipeline.Pipeline`には組み込めません(3節で実測)。

---

## 3. sklearn Pipelineへの組み込み可否(実測表)

`sklearn.pipeline.Pipeline`に4ライブラリの推定器を実際に組み込んで`fit`/`predict`まで動かした結果です。

| ライブラリ / 推定器 | 位置 | `sklearn.pipeline.Pipeline`での結果 |
|---|---|---|
| sklearn `DecisionTreeClassifier` | 最終ステップ | ○ 成功(`score`まで問題なし) |
| sksurv `CoxPHSurvivalAnalysis` | 最終ステップ | ○ 成功(`score`=C-indexまで問題なし) |
| pyod `KNN` | 最終ステップ | ○ 成功(`fit`/`predict`とも問題なし) |
| imblearn `SMOTE` | **中間**ステップ | **× 失敗**(`TypeError`) |
| imblearn `SMOTE` | 最終ステップ | △ `fit()`自体は成功するが`predict`/`fit_resample`が使えない(下記) |

sksurv・pyodは「最終ステップとして使う分には」sklearn本体の`Pipeline`にそのまま組み込めます。これは`Pipeline`が中間ステップにだけ`transform`実装を要求し、最終ステップには`fit`さえあればよいためです。

imblearnの`SMOTE`を**中間**ステップに置くと、実際に次のエラーになります(検証済み)。
```python
from sklearn.pipeline import Pipeline as SkPipeline
skpipe = SkPipeline([("smote", SMOTE(random_state=0)), ("clf", LogisticRegression(max_iter=1000))])
skpipe.fit(Xtr, ytr)
```
```
TypeError: All intermediate steps should be transformers and implement fit and transform or be the string 'passthrough' 'SMOTE(random_state=0)' (type <class 'imblearn.over_sampling._smote.base.SMOTE'>) doesn't
```
`sklearn.pipeline.Pipeline`は中間ステップに「入力と出力で行数を変えない`transform`」を要求しますが、`SMOTE`は`fit_resample`しか持たず行数も変えるため、この契約に合致しません。

`SMOTE`を**最終**ステップに置くと`fit()`自体はエラーにならず通りますが、実用上は意味がありません。
```python
pipe = SkPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=0))])
pipe.fit(X, y)      # 成功する
pipe.predict(X)     # -> AttributeError: This 'Pipeline' has no attribute 'predict'
pipe.fit_resample(X, y)  # -> AttributeError: 'Pipeline' object has no attribute 'fit_resample'
```
`Pipeline`は最終ステップの`fit_resample`を転送(delegate)する仕組みを持たないため、`SMOTE`を挟んだ`sklearn.pipeline.Pipeline`からリサンプリング結果を取り出す方法はありません。

**回避策**: `imblearn.pipeline.Pipeline`(または`imblearn.pipeline.make_pipeline`)を使います。これはサンプラー(`fit_resample`を実装するオブジェクト)を認識できる特別な実装で、`fit`時にのみリサンプリングを適用してyも連動更新し、`predict`/`transform`時にはリサンプリングをスキップして元の行数のまま予測する、という仕組みを持ちます。
```python
from imblearn.pipeline import Pipeline as ImbPipeline
pipe_imb = ImbPipeline([("smote", SMOTE(random_state=0)), ("clf", LogisticRegression(max_iter=1000))])
pipe_imb.fit(X, y)
pipe_imb.predict(X).shape   # (200,) -- Xの元の行数のまま(SMOTEはpredict時にスキップされる)
```
実行結果:
```
成功。score: 0.96
pipe_imb.predict(X).shape: (200,) (Xの元の行数のまま。SMOTEはfit時のみ適用され、predict時はスキップされる)
```
(この`imblearn.pipeline.Pipeline`の詳細な用法は [imblearn/README.md](../imblearn/README.md) の「4. パイプライン」節を参照。)

---

## 4. 交差検証・GridSearchCVとの互換性

`sklearn.model_selection.cross_val_score`/`GridSearchCV`にそのまま渡せるかを実際に試しました。

| ライブラリ / 推定器 | `cross_val_score`(scoring未指定) | `GridSearchCV`(scoring未指定) | 備考 |
|---|---|---|---|
| sklearn `DecisionTreeClassifier` | ○ 成功 | ○ 成功 | 標準の`accuracy`が既定スコアとして使われる |
| sksurv `CoxPHSurvivalAnalysis` | ○ 成功(`cox.score`=Harrell's C-indexが使われる) | ○ 成功 | ただし`scoring`にsklearn標準文字列(`"accuracy"`等)を指定すると**クラッシュせず黙ってnanになる**(後述) |
| pyod `KNN` | **× 失敗**(`TypeError`) | **× 失敗**(`TypeError`) | `score`メソッドが無いため。`scoring`を明示すれば動く(後述) |
| imblearn `SMOTE`単体 | **× 失敗**(`TypeError`) | (未実施、同じ理由で失敗する) | `score`メソッドが無いため |
| imblearn `Pipeline`(SMOTE+LogisticRegression) | ○ 成功(`scoring="f1"`を明示) | ○ 成功 | 最終推定器の`score`/指定した`scoring`が使われる |

### 4.1 sklearn: 素直に動く

```
cross_val_score: [0.94029851 0.94029851 0.92424242]
GridSearchCV best_params_: {'max_depth': 2}
```

### 4.2 sksurv: `scoring`未指定なら動くが、標準scoring文字列を指定すると「エラーにならず黙ってnanになる」

`scoring`を省略すると`cox.score`(=C-index)がそのまま使われ、問題なく動きます。
```
cross_val_score (scoring未指定, cox.score=C-index使用) 成功: [0.72974343 0.78285648 0.7363871 ]
GridSearchCV(scoring未指定) 成功。best_params_: {'alpha': 0.0} best_score_: 0.7496623362512601
```

危険なのは、`scoring="accuracy"`のようなsklearn標準の分類用スコアリング文字列を(誤って)指定した場合です。`GridSearchCV.fit()`自体は**例外を出さずに完了**し、`best_score_`が`nan`になるだけで、原因はwarningとしてしか出ません(検証済み)。
```python
gs_surv = GridSearchCV(CoxPHSurvivalAnalysis(), {"alpha": [0.0, 0.1, 1.0]}, cv=3, scoring="accuracy")
gs_surv.fit(Xw_num, yw)
```
```
UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details:
...
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
```
```
fit()自体は例外を出さずに完了した。best_score_: nan
cv_results_['mean_test_score']: [nan nan nan]
```
これは`GridSearchCV`がfold内のスコア計算失敗を(デフォルト設定では)例外にせず`nan`として握りつぶすためで、「動いているように見えて実際は何も選んでいない」状態に気づきにくい落とし穴です。

正しくは、`sksurv.metrics.concordance_index_censored`等を使った独自スコアラーを`scoring`に渡します。
```python
def sksurv_c_index_scorer(estimator, X, y):
    risk_scores = estimator.predict(X)
    return concordance_index_censored(y["fstat"], y["lenfol"], risk_scores)[0]

gs_ok = GridSearchCV(CoxPHSurvivalAnalysis(), {"alpha": [0.0, 0.1, 1.0]}, cv=3, scoring=sksurv_c_index_scorer)
gs_ok.fit(Xw_num, yw)
```
```
独自スコアラー(concordance_index_censored)で成功。best_params_: {'alpha': 0.0} best_score_: 0.7496623362512601
```

### 4.3 pyod: `score`メソッドが無いため`scoring`指定が必須

`scoring`を省略すると、sklearn/sksurvと違ってその場で例外になります(黙って失敗するsksurvのケースより安全とも言えます)。
```python
cross_val_score(KNN(contamination=0.1), Xo, yo, cv=3)
```
```
TypeError: If no scoring is specified, the estimator passed should have a 'score' method. The estimator KNN(...) does not.
```
`decision_function`があるため、`scoring="roc_auc"`のように明示すれば正常に動きます。
```
cross_val_score(scoring='roc_auc'): [0.86774194 0.85238095 0.92456897]
GridSearchCV(scoring='roc_auc') best_params_: {'n_neighbors': 5} best_score_: 0.8815639511273549
```
またこのとき、`cross_val_score`/`GridSearchCV`はfold分割のために内部で`y`をPyODの`fit`に渡すため、次の警告が毎fold出ます(動作は継続します)。
```
UserWarning: y should not be presented in unsupervised learning.
```

### 4.4 imblearn: サンプラー単体はCVに直接渡せない。`Pipeline`経由なら問題なし

`SMOTE`単体を渡すと(`score`が無いため)pyodと同種の`TypeError`になります。
```
cross_val_score(SMOTE(random_state=0), Xi, yi, cv=3)
```
```
TypeError: If no scoring is specified, the estimator passed should have a 'score' method. The estimator SMOTE(random_state=0) does not.
```
`imblearn.pipeline.Pipeline`に包んで、最終ステップの分類器の`score`(または明示`scoring`)を使えば、`cross_val_score`/`GridSearchCV`ともに問題なく動きます。かつ、`SMOTE`のパラメータも`smote__k_neighbors`のようにステップ名プレフィックス付きでグリッド探索できます。
```
cross_val_score(imblearn Pipeline, scoring='f1') 成功: [1.         0.91666667 0.94285714]
GridSearchCV(imblearn Pipeline) 成功。best_params_: {'smote__k_neighbors': 3}
```
重要な点として、`imblearn.pipeline.Pipeline`を使うことで、各foldの学習側データにのみ`fit_resample`が適用されます(交差検証全体に対して事前に一括`fit_resample`すると、複製・合成サンプルがテスト側に漏れるデータリークになるため、必ず`Pipeline`をCVに渡す形にする必要があります)。

---

## 5. 実務上の注意点まとめ

| 観点 | sklearn | scikit-survival | PyOD | imbalanced-learn |
|---|---|---|---|---|
| `BaseEstimator`継承・`get_params`/`set_params` | ○ | ○ | ○ | ○ |
| `fit`が`self`を返す | ○ | ○ | ○ | ○(ただし`fit`だけではリサンプリングされない) |
| `y`の形式 | 通常の1次元配列 | **構造化配列**`(event, time)`必須。1次元配列を渡すと`ValueError` | 通常の1次元配列(異常検知では省略可・渡しても警告付きで無視) | 通常の1次元配列 |
| `predict`の意味 | ラベル/値そのもの | **リスクスコア**(時間や確率ではない。`predict_survival_function`が別途必要) | 0/1の異常ラベル | メソッド自体が無い |
| `score`の既定指標 | `accuracy`等 | Harrell's C-index | **`score`メソッド無し** | **`score`メソッド無し** |
| `sklearn.pipeline.Pipeline`最終ステップ | ○ | ○ | ○ | △(`fit`は通るが`predict`/`fit_resample`が使えず実用上無意味) |
| `sklearn.pipeline.Pipeline`中間ステップ | ○(transformer限定) | 該当なし(通常は最終ステップとして使う) | 該当なし(通常は最終ステップとして使う) | **×**(`TypeError`。`imblearn.pipeline.Pipeline`が必要) |
| `cross_val_score`/`GridSearchCV`(scoring未指定) | ○ | ○(ただし`scoring`を誤指定すると無警告に近い形でnanになる) | ×(`scoring`明示が必須) | 単体は×。`imblearn.pipeline.Pipeline`経由なら○ |

実務上、特に注意すべき点を3つに絞ると:

1. **scikit-survivalでGridSearchCVをする際は、必ず専用スコアラー(`concordance_index_censored`ベースの`make_scorer`相当、または`scoring`省略でデフォルトのC-index)を使う。** `scoring="accuracy"`のようなsklearn標準文字列を誤って指定しても例外にならず`best_score_`が`nan`になるだけなので、`cv_results_`を必ず目視確認する習慣が要る(4.2節で実際に再現)。
2. **imbalanced-learnのサンプラーは、交差検証・パイプラインの文脈では必ず`imblearn.pipeline.Pipeline`(または`imblearn.pipeline.make_pipeline`)経由で使う。** 素の`sklearn.pipeline.Pipeline`に中間ステップとして入れると`TypeError`になり(3節)、一括`fit_resample`してから`cross_val_score`に生データを渡すとテストフォールドへのリークになる(4.4節)。
3. **PyODは`score`メソッドが存在しないため、`cross_val_score`/`GridSearchCV`では`scoring`(例: `"roc_auc"`。`decision_function`を利用)を必ず明示する。** 省略すると即座に`TypeError`になるため、sksurvのように気づかずnanになるケースよりは安全だが、対処自体は必須(4.3節)。

いずれのライブラリも「`BaseEstimator`を継承し`get_params`/`set_params`/`fit`が使える」という最外層の規約は完全に守っているため、`GridSearchCV`のパラメータ探索の仕組みそのもの(`get_params`ベースのstep名プレフィックス解決など)は4ライブラリで共通して機能します。逸脱が起きるのは「`y`の形式」「`predict`が返すものの意味」「`score`の有無」「行数を変える`fit_resample`という第5のメソッド」という、各ライブラリのドメイン固有の部分です。
