# xgboost 逆引き辞書

xgboost 3.4.1 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(xgboost 3.4.1)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。

## 目次

1. [DMatrix・学習基礎](#1-dmatrix学習基礎)
2. [scikit-learn API](#2-scikit-learn-api)
3. [交差検証](#3-交差検証)
4. [正則化・過学習対策](#4-正則化過学習対策)
5. [特徴量重要度・可視化](#5-特徴量重要度可視化)
6. [カテゴリ変数・欠損値のネイティブ対応](#6-カテゴリ変数欠損値のネイティブ対応)
7. [モデルの保存・読み込み](#7-モデルの保存読み込み)
8. [ハイパーパラメータ・目的関数](#8-ハイパーパラメータ目的関数)
9. [その他ユーティリティ](#9-その他ユーティリティ)

## 応用・発展

10. [カスタム目的関数・カスタム評価関数](#10-カスタム目的関数カスタム評価関数)
11. [単調性制約・特徴量間相互作用制約](#11-単調性制約特徴量間相互作用制約)
12. [SHAP値の深掘り](#12-shap値の深掘り)
13. [ハイパーパラメータチューニングの実践パターン](#13-ハイパーパラメータチューニングの実践パターン)
14. [特殊なタスク・モデル診断の応用](#14-特殊なタスクモデル診断の応用)

---

## 1. DMatrix・学習基礎

### `xgboost.DMatrix(...)`

**用途**: xgboostの学習・予測用の内部データ構造。numpy配列やpandas DataFrameから作る。

**シグネチャ**: `xgboost.DMatrix(data, label=None, *, weight=None, base_margin=None, missing=None, silent=False, feature_names=None, feature_types=None, nthread=None, group=None, qid=None, label_lower_bound=None, label_upper_bound=None, feature_weights=None, enable_categorical=True, data_split_mode=0)`

**使用例**:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb

iris = load_iris()
X, y = iris.data, iris.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=iris.feature_names)
print(dtrain.num_row(), dtrain.num_col())
print(type(dtrain))
```
実行結果:
```
120 4
<class 'xgboost.core.DMatrix'>
```

**注意点・落とし穴**:
- `enable_categorical=True` がデフォルト(3.x系)。pandasの`category`型列があってもエラーにならず、内部でカテゴリ特徴量として扱われる(詳細は6章)。
- `label` を渡し忘れると学習(`xgb.train`)時にラベルなしと解釈され、目的関数によってはエラーになる。

### `xgboost.train(...)`

**用途**: `params`(辞書)と`DMatrix`から`Booster`(学習済みモデル)を作る、xgboostの低レベルAPIの中核関数。

**シグネチャ**: `xgboost.train(params, dtrain, num_boost_round=10, *, evals=None, obj=None, maximize=None, early_stopping_rounds=None, evals_result=None, verbose_eval=True, xgb_model=None, callbacks=None, custom_metric=None)`

**使用例**:
```python
import xgboost as xgb

dtest = xgb.DMatrix(Xte, label=yte, feature_names=iris.feature_names)
params = {"objective": "multi:softprob", "num_class": 3, "max_depth": 3, "eta": 0.3, "seed": 0}
bst = xgb.train(params, dtrain, num_boost_round=10, evals=[(dtest, "eval")], verbose_eval=False)
print(type(bst))
print("best_iteration:", bst.best_iteration if hasattr(bst, "best_iteration") else None)
print("num_boosted_rounds:", bst.num_boosted_rounds())
```
実行結果:
```
<class 'xgboost.core.Booster'>
best_iteration: None
num_boosted_rounds: 10
```

**注意点・落とし穴**:
- `num_boost_round`(デフォルト10)は木の本数。scikit-learn APIの`n_estimators`に相当するが、両者は別引数なので混同しない。
- `early_stopping_rounds`を指定しない場合、`best_iteration`属性は`None`のまま(4章参照)。
- 多クラス分類では`params`に`num_class`を明示的に指定する必要がある(忘れるとエラーになる)。

### `Booster.predict(...)`

**用途**: 学習済み`Booster`で予測する。

**シグネチャ**: `Booster.predict(self, data, *, output_margin=False, pred_leaf=False, pred_contribs=False, approx_contribs=False, pred_interactions=False, validate_features=True, training=False, iteration_range=(0, 0), strict_shape=False)`

**使用例**:
```python
pred = bst.predict(dtest)
print(pred.shape)
print(pred[:2].round(3))
```
実行結果:
```
(30, 3)
[[0.951 0.026 0.024]
 [0.026 0.947 0.027]]
```

**注意点・落とし穴**:
- `iteration_range=(0, 0)`(デフォルト)は「全ての木を使う」という意味であり、`early_stopping_rounds`で学習を打ち切っていても`Booster.predict()`はデフォルトでは全木を使う(scikit-learn APIの`predict`とは挙動が違う。4章の注意点参照)。
- `multi:softprob`ではクラス数分の列を持つ確率行列、`multi:softmax`では予測クラスのラベル配列(float型)が返る(8章参照)。

### `Booster.num_boosted_rounds()` / `DMatrix.num_row()` / `DMatrix.num_col()`

**用途**: 学習された木の本数、`DMatrix`の行数・列数を取得する。

**シグネチャ**: `Booster.num_boosted_rounds(self) -> int` / `DMatrix.num_row(self) -> int` / `DMatrix.num_col(self) -> int`

**使用例**:
```python
print(dtrain.num_row(), dtrain.num_col())
print(bst.num_boosted_rounds())
```
実行結果:
```
120 4
10
```

**注意点・落とし穴**:
- `num_boosted_rounds()`は早期終了(early stopping)後も打ち切り時点までの全木の本数を返す。ベストな木の本数は`best_iteration + 1`で別途確認する必要がある(4章参照)。

---

## 2. scikit-learn API

### `XGBClassifier(...)`

**用途**: scikit-learn互換の分類器インターフェース。`Pipeline`や`GridSearchCV`にそのまま組み込める。

**シグネチャ**: `xgboost.XGBClassifier(*, objective='binary:logistic', **kwargs)`(`max_depth`/`n_estimators`/`learning_rate`など主要パラメータは`kwargs`側に定義され、`get_params()`ではすべて`None`がデフォルトとして表示される)

**使用例**:
```python
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=0)
clf.fit(Xtr, ytr)
print("score:", clf.score(Xte, yte))
print("predict:", clf.predict(Xte[:5]))
print("classes_:", clf.classes_)
```
実行結果:
```
score: 0.9333333333333333
predict: [0 1 0 1 0]
classes_: [0 1 2]
```

**注意点・落とし穴**:
- `XGBClassifier().get_params()`を見ると`max_depth`/`n_estimators`/`learning_rate`など主要パラメータのデフォルトは軒並み`None`になっている。これは「未指定ならCの内部デフォルト(`max_depth=6`, `learning_rate=0.3`, `n_estimators=100`など)を使う」という意味で、`None`という値がそのまま使われるわけではない(8章で実測)。
- `objective`のデフォルトは`'binary:logistic'`だが、`fit`時に`y`のクラス数が3以上だと自動的に`multi:softprob`相当に切り替わる。

### `XGBRegressor(...)`

**用途**: scikit-learn互換の回帰器インターフェース。

**シグネチャ**: `xgboost.XGBRegressor(*, objective='reg:squarederror', **kwargs)`

**使用例**:
```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
Xr, yr = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=0)
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.2, random_state=0)
reg = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=0)
reg.fit(Xrtr, yrtr)
print("R2:", reg.score(Xrte, yrte))
print("predict[:3]:", reg.predict(Xrte[:3]).round(2))
```
実行結果:
```
R2: 0.8005910902019933
predict[:3]: [ -25.38 -195.33   27.03]
```

### `XGBClassifier.fit(...)` / `.predict()` / `.predict_proba()`

**用途**: scikit-learn API共通の学習・予測メソッド。

**シグネチャ**: `fit(self, X, y, *, sample_weight=None, base_margin=None, eval_set=None, verbose=True, xgb_model=None, sample_weight_eval_set=None, base_margin_eval_set=None, feature_weights=None)` / `predict(self, X, *, output_margin=False, validate_features=True, base_margin=None, iteration_range=None)` / `predict_proba(self, X, validate_features=True, base_margin=None, iteration_range=None)`

**使用例**:
```python
print("predict_proba[0]:", clf.predict_proba(Xte[:1]).round(3))
```
実行結果:
```
predict_proba[0]: [[0.977 0.016 0.007]]
```

**注意点・落とし穴**:
- `XGBClassifier`/`XGBRegressor`の`predict`/`predict_proba`は`iteration_range`を明示的に指定しない場合、学習時に早期終了していれば自動的に`best_iteration`までの木だけを使う。これは`Booster.predict()`(デフォルトで全木を使う)と挙動が異なるので、低レベルAPIとscikit-learn APIを混在させるときは要注意(4章で実測)。

### `XGBRFClassifier(...)` / `XGBRFRegressor(...)`

**用途**: ランダムフォレスト風の設定(`n_estimators=1`相当のブースティング1回・`subsample`/`colsample_bynode`をデフォルトで有効化)をあらかじめ組み込んだ分類器・回帰器。

**シグネチャ**: `xgboost.XGBRFClassifier(*, learning_rate=1.0, subsample=0.8, colsample_bynode=0.8, reg_lambda=1e-05, **kwargs)`

**使用例**:
```python
rf = xgb.XGBRFClassifier(n_estimators=100, random_state=0)
rf.fit(Xtr, ytr)
print("score:", rf.score(Xte, yte))
```
実行結果:
```
score: 0.9333333333333333
```
(実行時に以下の警告が出力される)
```
FutureWarning: `XGBRFClassifier` is deprecated and will be removed in a future release. The estimator is a thin wrapper over the boosting interface and does not implement a conventional random forest; features like early stopping are unsupported. Set `num_parallel_tree` along with `n_estimators=1` on the corresponding boosting estimator instead, or use a dedicated random forest implementation like those in `sklearn.ensemble`.
```

**注意点・落とし穴**:
- **バージョン固有の注意**: xgboost 3.4.1で`XGBRFClassifier`/`XGBRFRegressor`は非推奨(`FutureWarning`)。公式は`num_parallel_tree`パラメータ+`n_estimators=1`を通常の`XGBClassifier`/`XGBRegressor`に設定するか、`sklearn.ensemble`の実装を使うよう案内している(実際に警告メッセージを確認済み)。

---

## 3. 交差検証

### `xgboost.cv(...)`

**用途**: 低レベルAPI(`params`+`DMatrix`)でクロスバリデーションを行い、fold平均のスコア推移を返す。

**シグネチャ**: `xgboost.cv(params, dtrain, num_boost_round=10, *, nfold=3, stratified=False, folds=None, metrics=(), obj=None, maximize=None, early_stopping_rounds=None, fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True, seed=0, callbacks=None, shuffle=True, custom_metric=None)`

**使用例**:
```python
import xgboost as xgb
from sklearn.datasets import load_iris

iris = load_iris()
dtrain_all = xgb.DMatrix(iris.data, label=iris.target)
params = {"objective": "multi:softprob", "num_class": 3, "max_depth": 3, "eta": 0.3, "seed": 0}
res = xgb.cv(params, dtrain_all, num_boost_round=20, nfold=5, metrics="mlogloss",
             early_stopping_rounds=5, seed=0, as_pandas=True)
print(res.shape)
print(res.columns.tolist())
print(res.tail(3))
```
実行結果:
```
(13, 4)
['train-mlogloss-mean', 'train-mlogloss-std', 'test-mlogloss-mean', 'test-mlogloss-std']
    train-mlogloss-mean  train-mlogloss-std  test-mlogloss-mean  test-mlogloss-std
10             0.075845            0.012155            0.191307           0.100416
11             0.066662            0.011782            0.187938           0.102193
12             0.059483            0.011442            0.185375           0.103997
```

**注意点・落とし穴**:
- `num_boost_round=20`を指定しても、`early_stopping_rounds=5`により13行(0〜12ラウンド分)で打ち切られている。学習済みモデル自体は返らず、あくまでスコア推移(DataFrameまたは辞書)のみが返る点に注意(モデルが欲しい場合は改めて`xgb.train`を呼ぶ)。
- `stratified=False`がデフォルト。分類問題でクラス比率を保った分割をしたい場合は`stratified=True`を明示する。

---

## 4. 正則化・過学習対策

### `early_stopping_rounds`(scikit-learn API)/ `best_iteration` / `best_score`

**用途**: 検証データのスコアが指定ラウンド数だけ改善しなくなったら学習を打ち切る。

**シグネチャ**: `XGBClassifier(..., early_stopping_rounds=None, eval_metric=None, ...)`。`fit(X, y, eval_set=[...])`と組み合わせて使う。

**使用例**:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb

data = load_breast_cancer()
Xb, yb = data.data, data.target
Xbtr, Xbte, ybtr, ybte = train_test_split(Xb, yb, test_size=0.2, random_state=0, stratify=yb)

clf_es = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            early_stopping_rounds=10, eval_metric="logloss", random_state=0)
clf_es.fit(Xbtr, ybtr, eval_set=[(Xbte, ybte)], verbose=False)
print("best_iteration:", clf_es.best_iteration)
print("best_score:", clf_es.best_score)
print("実際に構築された木の本数:", clf_es.get_booster().num_boosted_rounds())
```
実行結果:
```
best_iteration: 33
best_score: 0.16547872173485526
実際に構築された木の本数: 44
```

**注意点・落とし穴**:
- `n_estimators=200`を指定していても、実際に構築される木の本数は`best_iteration + early_stopping_rounds`程度(この例では44本)で打ち切られる。`best_iteration`(33)と`num_boosted_rounds()`(44)は別物であり、前者が「そこから改善しなかった最良ラウンド」。
- `clf.predict()`/`predict_proba()`はデフォルトで`best_iteration`までの木だけを使う(1章・2章の注意点参照)。

### `xgboost.callback.EarlyStopping(...)`

**用途**: `xgb.train`の`callbacks`引数に渡す早期終了コールバック(低レベルAPI向け)。

**シグネチャ**: `xgboost.callback.EarlyStopping(self, *, rounds, metric_name=None, data_name=None, maximize=None, save_best=False, min_delta=0.0)`

**使用例**:
```python
import xgboost as xgb
# Xbtr/ybtr/Xbte/ybteは直前のearly_stopping_roundsの例(乳がんデータ)と同じもの
dtrain_b = xgb.DMatrix(Xbtr, label=ybtr)
dtest_b = xgb.DMatrix(Xbte, label=ybte)
es = xgb.callback.EarlyStopping(rounds=10, save_best=True)
bst_b = xgb.train({"max_depth": 4, "eta": 0.1, "objective": "binary:logistic"}, dtrain_b,
                   num_boost_round=200, evals=[(dtest_b, "eval")], callbacks=[es], verbose_eval=False)
print(bst_b.best_iteration, bst_b.num_boosted_rounds())
```
実行結果:
```
33 34
```

**注意点・落とし穴**:
- `save_best=True`にすると`Booster`自体が最良ラウンドの状態で切り詰められる(`num_boosted_rounds()`が`best_iteration+1`と一致する)。`save_best=False`(デフォルト)だと、`early_stopping_rounds`引数を直接`xgb.train`に渡した場合と同様、打ち切りまでの全木が残る。

### `reg_alpha` / `reg_lambda`

**用途**: L1正則化(`reg_alpha`)・L2正則化(`reg_lambda`)の強さを指定する。

**シグネチャ**: `XGBClassifier(..., reg_alpha=None, reg_lambda=None, ...)`(内部デフォルトは`reg_alpha=0`, `reg_lambda=1`)

**使用例**:
```python
def fit_score(alpha, lam):
    m = xgb.XGBClassifier(n_estimators=100, max_depth=4, reg_alpha=alpha, reg_lambda=lam, random_state=0)
    m.fit(Xbtr, ybtr)
    return m.score(Xbte, ybte)

print("alpha=0,lambda=1(既定相当):", fit_score(0, 1))
print("alpha=1,lambda=1:", fit_score(1, 1))
print("alpha=0,lambda=10:", fit_score(0, 10))
```
実行結果:
```
alpha=0,lambda=1(既定相当): 0.9649122807017544
alpha=1,lambda=1: 0.9649122807017544
alpha=0,lambda=10: 0.956140350877193
```

**注意点・落とし穴**:
- `reg_lambda`(L2)はデフォルト`1`で最初から有効だが、`reg_alpha`(L1)はデフォルト`0`で無効。scikit-learnの`Ridge`/`Lasso`と違い、xgboostは最初からL2正則化がかかっている点に注意。

### `subsample` / `colsample_bytree`

**用途**: 各木を作る際に使う行(`subsample`)・列(`colsample_bytree`)の割合をランダムに間引き、過学習を抑える。

**シグネチャ**: `XGBClassifier(..., subsample=None, colsample_bytree=None, ...)`(内部デフォルトはともに`1.0`)

**使用例**:
```python
m2 = xgb.XGBClassifier(n_estimators=100, subsample=0.8, colsample_bytree=0.8, random_state=0)
m2.fit(Xbtr, ybtr)
print("score:", m2.score(Xbte, ybte))
```
実行結果:
```
score: 0.956140350877193
```

**注意点・落とし穴**:
- デフォルトはどちらも`1.0`(間引きなし)。ランダムフォレストと違い、xgboostは明示的に指定しない限り毎回全データ・全特徴量を使う。

---

## 5. 特徴量重要度・可視化

### `XGBClassifier.feature_importances_`

**用途**: 学習済みscikit-learn APIモデルの特徴量重要度(デフォルトは`gain`ベース)を取得する。

**シグネチャ**: `XGBClassifier.feature_importances_`(プロパティ、引数なし、戻り値は`numpy.ndarray`)

**使用例**:
```python
# 2章で学習したclf(iris、n_estimators=50, max_depth=3, random_state=0)をそのまま使う
print(clf.feature_importances_.round(3))
```
実行結果:
```
[0.028 0.031 0.397 0.544]
```

**注意点・落とし穴**:
- scikit-learnの`RandomForestClassifier.feature_importances_`は分裂に使われた回数ベース(weight相当)だが、xgboostの`feature_importances_`は`importance_type='gain'`(平均的な損失改善量)がデフォルトで、両者は指標の意味が異なるため単純比較できない。

### `Booster.get_score(...)`

**用途**: `importance_type`を指定して特徴量重要度を辞書で取得する(低レベルAPI)。

**シグネチャ**: `Booster.get_score(self, fmap='', importance_type='weight')`

**使用例**:
```python
bst5 = clf.get_booster()
bst5.feature_names = iris.feature_names
print(bst5.get_score(importance_type="weight"))
print(bst5.get_score(importance_type="gain"))
```
実行結果:
```
{'sepal length (cm)': 49.0, 'sepal width (cm)': 53.0, 'petal length (cm)': 191.0, 'petal width (cm)': 136.0}
{'sepal length (cm)': 0.24152368307113647, 'sepal width (cm)': 0.2684745192527771, 'petal length (cm)': 3.4009242057800293, 'petal width (cm)': 4.664458274841309}
```

**注意点・落とし穴**:
- **`Booster.get_score`のデフォルトは`importance_type='weight'`(分裂に使われた回数)だが、`XGBClassifier.feature_importances_`のデフォルトは`'gain'`**。同じモデルでも呼び出し方によってデフォルトの重要度の種類が違うため、比較する際は`importance_type`を揃える必要がある。
- 分裂に一度も使われなかった特徴量は辞書のキーに現れない(0が入るわけではなく、キー自体が欠落する)。

### `xgboost.plot_importance(...)`

**用途**: 特徴量重要度を横棒グラフで可視化する(matplotlib必須)。

**シグネチャ**: `xgboost.plot_importance(booster, *, ax=None, height=0.2, xlim=None, ylim=None, title='Feature importance', xlabel='Importance score', ylabel='Features', fmap='', importance_type='weight', max_num_features=None, grid=True, show_values=True, values_format='{v}', **kwargs)`

**使用例**:
```python
import matplotlib
matplotlib.use("Agg")
import xgboost as xgb
ax = xgb.plot_importance(bst5, max_num_features=5)
print(type(ax))
```
実行結果:
```
<class 'matplotlib.axes._axes.Axes'>
```

**注意点・落とし穴**:
- `importance_type`のデフォルトは`'weight'`(`Booster.get_score`と同じ)。`XGBClassifier.feature_importances_`(`'gain'`)と見た目の順位が変わることがある。

### `xgboost.plot_tree(...)`

**用途**: 個々の決定木の構造を可視化する(graphviz必須)。

**シグネチャ**: `xgboost.plot_tree(booster, *, fmap='', num_trees=None, rankdir=None, ax=None, with_stats=False, tree_idx=0, **kwargs)`

**使用例**:
```python
ax2 = xgb.plot_tree(bst5, tree_idx=0)
print(type(ax2))
```
実行結果:
```
<class 'matplotlib.axes._axes.Axes'>
```

**注意点・落とし穴**:
- **バージョン固有の注意**: xgboost 3.4.1では`num_trees`引数は非推奨(`FutureWarning`)。代わりに`tree_idx`を使うよう案内される(`num_trees=0`を渡すと実際に警告が出ることを確認済み)。

### `Booster.trees_to_dataframe()`

**用途**: 全ての木の全ノード情報(分裂に使った特徴量・しきい値・ゲインなど)をDataFrameで取得する。

**シグネチャ**: `Booster.trees_to_dataframe(self, fmap='')`

**使用例**:
```python
df = bst5.trees_to_dataframe()
print(df.shape)
print(df.columns.tolist())
```
実行結果:
```
(1008, 12)
['Tree', 'Target', 'Node', 'ID', 'Feature', 'Split', 'Yes', 'No', 'Missing', 'Gain', 'Cover', 'Category']
```

**注意点・落とし穴**:
- 葉ノードの行は`Feature`列が`'Leaf'`、`Gain`列にはその葉の予測値(スコア)が入る(分裂ゲインではない)。集計する際は葉ノードとそれ以外を区別する必要がある。

### `Booster.predict(..., pred_contribs=True)`

**用途**: SHAP値相当の特徴量ごとの予測寄与度を取得する(TreeSHAPアルゴリズム)。

**シグネチャ**: `Booster.predict(self, data, *, pred_contribs=False, ...)`(1章参照)

**使用例**:
```python
# 4章の「EarlyStopping callback」で作ったbst_b/dtest_b(乳がんデータ、30特徴量)を再利用
contribs = bst_b.predict(dtest_b, pred_contribs=True)
pred_margin = bst_b.predict(dtest_b, output_margin=True)
print(contribs.shape)
print("各サンプルの寄与度合計 == マージン予測値:", (abs(contribs.sum(axis=1) - pred_margin) < 1e-5).all())
```
実行結果:
```
(114, 31)
各サンプルの寄与度合計 == マージン予測値: True
```

**注意点・落とし穴**:
- 出力の列数は特徴量数+1(最後の列がバイアス項/期待値)になる。全列の合計が`output_margin=True`の予測値(シグモイド適用前のロジット等)と一致する。

---

## 6. カテゴリ変数・欠損値のネイティブ対応

### `enable_categorical=True`

**用途**: pandasの`category`型列をワンホット化せずにそのまま木の分岐に使う(xgboostのネイティブカテゴリ対応)。

**シグネチャ**: `XGBClassifier(..., enable_categorical=True, ...)` / `DMatrix(..., enable_categorical=True)`

**使用例**:
```python
import numpy as np
import pandas as pd
import xgboost as xgb

df = pd.DataFrame({
    "color": pd.Categorical(["red", "blue", "green", "blue", "red", "green"]),
    "size": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
})
y = np.array([0, 1, 0, 1, 0, 1])
clf_cat = xgb.XGBClassifier(n_estimators=20, max_depth=2, enable_categorical=True, random_state=0)
clf_cat.fit(df, y)
print("feature_types:", clf_cat.get_booster().feature_types)

clf_cat2 = xgb.XGBClassifier(n_estimators=5, enable_categorical=False)
try:
    clf_cat2.fit(df, y)
except Exception as e:
    print("ERROR:", type(e).__name__)
```
実行結果:
```
feature_types: ['c', 'float']
ERROR: ValueError
```

**注意点・落とし穴**:
- `XGBClassifier`/`XGBRegressor`のデフォルトは`enable_categorical=True`(3.x系)。`category`型の列をそのまま渡せる。
- 逆に`enable_categorical=False`を明示すると、`category`型の列が含まれる時点で`ValueError`になる(「`enable_categorical`をTrueにせよ」という趣旨のメッセージ)。
- `OneHotEncoder`等で事前にエンコードする必要がなく、カテゴリ数が多い変数でも列数が増えない利点がある。

### `missing`(欠損値のネイティブ対応)

**用途**: 欠損値(デフォルトは`NaN`)を、木の各分岐でどちらに流すか学習時に自動決定させる。

**シグネチャ**: `DMatrix(data, ..., missing=None)`(`None`の場合`numpy.nan`が使われる)

**使用例**:
```python
import numpy as np
import xgboost as xgb

Xm = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 5.0], [4.0, 6.0]])
ym = np.array([0, 1, 0, 1])
dtrain_m = xgb.DMatrix(Xm, label=ym, missing=np.nan)
bst_m = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain_m, num_boost_round=5)
print(bst_m.predict(dtrain_m).round(3))
```
実行結果:
```
[0.5 0.5 0.5 0.5]
```

**注意点・落とし穴**:
- `SimpleImputer`のような事前の欠損値補完が不要。欠損があるサンプルがどちらの子ノードに進むかは学習データから決定される(欠損時のデフォルト方向として学習される)。
- サンプル数が極端に少ない例では予測値が全て同じ(0.5)になることもあるため、実運用では十分なデータ量で検証すること。

---

## 7. モデルの保存・読み込み

### `Booster.save_model(...)` / `Booster.load_model(...)`

**用途**: 学習済み`Booster`をファイルに保存・復元する(低レベルAPI)。

**シグネチャ**: `Booster.save_model(self, fname)` / `Booster.load_model(self, fname)`

**使用例**:
```python
import numpy as np
import xgboost as xgb

bst.save_model("/tmp/model.json")
bst2 = xgb.Booster()
bst2.load_model("/tmp/model.json")
print(np.allclose(bst.predict(dtest), bst2.predict(dtest)))
```
実行結果:
```
True
```

**注意点・落とし穴**:
- 拡張子`.json`または`.ubj`(バージョン1.6以降推奨のUBJSON形式)を使うとフォーマットが保証される。拡張子なしや未知の拡張子だとxgboost側が形式を推測する。
- 古い`.model`(バイナリ)形式は読めるが、Python版のバージョン間互換性は`.json`/`.ubj`の方が高い。

### `XGBClassifier.save_model(...)` / `XGBClassifier.load_model(...)`

**用途**: scikit-learn APIラッパーごと(ハイパーパラメータ含む)モデルを保存・復元する。

**シグネチャ**: `XGBModel.save_model(self, fname)` / `XGBModel.load_model(self, fname)`

**使用例**:
```python
clf.save_model("/tmp/clf.ubj")
clf2 = xgb.XGBClassifier()
clf2.load_model("/tmp/clf.ubj")
print(np.allclose(clf.predict_proba(Xte), clf2.predict_proba(Xte)))
print("classes_:", clf2.classes_)
```
実行結果:
```
True
classes_: [0 1 2]
```

**注意点・落とし穴**:
- `load_model`は「未学習の`XGBClassifier()`インスタンス」に対して呼ぶことで、`classes_`などscikit-learn API用の属性も復元される。逆に`Booster().load_model()`で読み込んだものはあくまで生の`Booster`であり、`classes_`や`.score()`のようなscikit-learn的メソッドは使えない。
- pickle/joblibでの保存も可能だが、xgboostのバージョンをまたぐ互換性は`save_model`/`load_model`(JSON/UBJ)の方が高いとされる。

---

## 8. ハイパーパラメータ・目的関数

### `objective`

**用途**: 学習する目的関数(タスクの種類)を指定する。分類・回帰・ランキングなどで異なる文字列を使う。

**シグネチャ**: `params["objective"]`(文字列)。代表例: `"binary:logistic"`(2値分類・確率出力)、`"multi:softmax"`(多クラス・ラベル出力)、`"multi:softprob"`(多クラス・確率出力)、`"reg:squarederror"`(回帰・二乗誤差)。

**使用例**:
```python
bst_sm = xgb.train({"objective": "multi:softmax", "num_class": 3, "max_depth": 3}, dtrain, num_boost_round=10)
bst_sp = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 3}, dtrain, num_boost_round=10)
print("softmax:", bst_sm.predict(dtest).shape, bst_sm.predict(dtest)[:3])
print("softprob:", bst_sp.predict(dtest).shape)
```
実行結果:
```
softmax: (30,) [0. 1. 0.]
softprob: (30, 3)
```

**注意点・落とし穴**:
- `multi:softmax`は予測クラス(float型のラベル)を1次元配列で返し、`multi:softprob`はクラスごとの確率を2次元配列で返す。`predict()`の戻り値の形が目的関数によって変わる点に注意。
- `XGBClassifier`は`objective`を自分で設定しなくても`fit`時のクラス数から自動選択するが、`xgb.train`(低レベルAPI)では`num_class`とセットで明示指定が必須。

### `eval_metric`

**用途**: 学習中に監視する評価指標を指定する(early stoppingの判定にも使われる)。

**シグネチャ**: `XGBClassifier(..., eval_metric=None, ...)`。複数指定する場合はリスト(例: `["logloss", "error"]`)。

**使用例**:
```python
clf_metric = xgb.XGBClassifier(n_estimators=20, eval_metric=["logloss", "error"], random_state=0)
clf_metric.fit(Xbtr, ybtr, eval_set=[(Xbte, ybte)], verbose=False)
res = clf_metric.evals_result()
print(list(res["validation_0"].keys()))
print([round(v, 4) for v in res["validation_0"]["logloss"][-3:]])
```
実行結果:
```
['logloss', 'error']
[0.162, 0.1636, 0.1651]
```

**注意点・落とし穴**:
- `logloss`は2値分類専用の指標であり、多クラス分類のモデルに`eval_metric="logloss"`を指定すると`XGBoostError`(`label and prediction size not match`)になる。多クラスでは`mlogloss`/`merror`を使う(実際にエラーを再現して確認済み)。
- 複数指標を指定した場合、early stoppingの判定に使われるのは`eval_metric`リストの**最後**の指標。

### `n_estimators` / `learning_rate` / `max_depth` の内部デフォルト値

**用途**: scikit-learn APIで`None`のまま(未指定)にした場合に実際に使われる内部デフォルト値を確認する。

**シグネチャ**: `XGBClassifier(n_estimators=None, learning_rate=None, max_depth=None, ...)`

**使用例**:
```python
import json
plain = xgb.XGBClassifier()
plain.fit(Xbtr, ybtr)
cfg = json.loads(plain.get_booster().save_config())
gbtree = cfg["learner"]["gradient_booster"]["tree_train_param"]
print("eta(learning_rate):", gbtree["eta"], "max_depth:", gbtree["max_depth"])
print("num_boosted_rounds(n_estimators):", plain.get_booster().num_boosted_rounds())
```
実行結果:
```
eta(learning_rate): 0.300000012 max_depth: 6
num_boosted_rounds(n_estimators): 100
```

**注意点・落とし穴**:
- 実測の結果、`n_estimators`未指定時は100本、`learning_rate`未指定時は0.3、`max_depth`未指定時は6が使われる。`get_params()`の表示(`None`)だけを見て「値が設定されていない」と誤解しないよう、`Booster.save_config()`で実際の設定をJSONとして確認できる。

### `scale_pos_weight`

**用途**: 2値分類でクラス不均衡がある場合に、正例(少数派)の重みを増やして再現率を上げる。

**シグネチャ**: `XGBClassifier(..., scale_pos_weight=None, ...)`(内部デフォルトは`1`)。目安は`負例数 / 正例数`。

**使用例**:
```python
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score, precision_score

Xi, yi = make_classification(n_samples=3000, n_features=10, n_informative=4, weights=[0.95, 0.05],
                              flip_y=0.05, class_sep=0.5, random_state=0)
Xitr, Xite, yitr, yite = train_test_split(Xi, yi, test_size=0.3, random_state=0, stratify=yi)
neg, pos = np.bincount(yitr)
ratio = neg / pos
clf_default = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=0).fit(Xitr, yitr)
clf_weighted = xgb.XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=ratio, random_state=0).fit(Xitr, yitr)
print("default: recall=%.3f precision=%.3f" % (recall_score(yite, clf_default.predict(Xite)), precision_score(yite, clf_default.predict(Xite))))
print("weighted: recall=%.3f precision=%.3f" % (recall_score(yite, clf_weighted.predict(Xite)), precision_score(yite, clf_weighted.predict(Xite))))
```
実行結果:
```
default: recall=0.060 precision=0.286
weighted: recall=0.134 precision=0.117
```
(`neg/pos ratio`は実測で約12.46倍)

**注意点・落とし穴**:
- `scale_pos_weight`を掛けると再現率(recall)が0.060→0.134に上がる一方、適合率(precision)は0.286→0.117に下がる(見逃し(偽陰性)を減らす代わりに誤検知(偽陽性)が増える典型的なトレードオフ)。「精度が上がる」魔法のパラメータではない。
- `scale_pos_weight`は「予測確率の較正(calibration)」を歪める副作用があり、`predict_proba`の値をそのまま確率として使う用途には向かない。`predict`(0.5しきい値の2値判定)の再現率・適合率のバランス調整に使うのが基本。

---

## 9. その他ユーティリティ

### `Booster.get_dump(...)`

**用途**: 各木の構造をテキスト(またはJSON)のリストとして取得する。

**シグネチャ**: `Booster.get_dump(self, fmap='', with_stats=False, dump_format='text')`

**使用例**:
```python
from sklearn.datasets import load_breast_cancer
import xgboost as xgb

data = load_breast_cancer()
dtrain_g = xgb.DMatrix(data.data, label=data.target, feature_names=list(data.feature_names))
bst_g = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain_g, num_boost_round=3)
dump = bst_g.get_dump()
print(len(dump))
print(dump[0])
```
実行結果:
```
3
0:[worst area<888.299988] yes=1,no=2,missing=2
	1:[worst concave points<0.160699993] yes=3,no=4,missing=4
		3:leaf=0.413357258
		4:leaf=-0.576436043
	2:[mean concavity<0.0715999976] yes=5,no=6,missing=6
		5:leaf=-0.216165826
		6:leaf=-0.784719884
```

**注意点・落とし穴**:
- `get_dump()`が返すリストの長さは「木の本数」(=`num_boost_round`。多クラス分類では`クラス数 × ラウンド数`になる)。
- `with_stats=True`にすると各分岐の`gain`/`cover`も出力される。`trees_to_dataframe()`の方が集計・分析には扱いやすいことが多い。

### `xgboost.config_context(...)` / `xgboost.get_config()`

**用途**: `verbosity`など、グローバル設定を一時的に(`with`ブロック内だけ)変更する。

**シグネチャ**: `xgboost.config_context(**new_config)` / `xgboost.get_config() -> dict`

**使用例**:
```python
with xgb.config_context(verbosity=0):
    print(xgb.get_config()["verbosity"])
print(xgb.get_config()["verbosity"])
```
実行結果:
```
0
1
```

**注意点・落とし穴**:
- `with`ブロックを抜けると自動的に元の設定(この例では`verbosity=1`)に戻る。グローバルに変更したままにしたい場合は`xgboost.set_config(**kwargs)`を使う。

### `Booster.attributes()` / `Booster.set_attr(...)`

**用途**: モデルに任意の文字列メタデータ(実験名やメモなど)を紐付けて保存する。

**シグネチャ**: `Booster.attributes(self) -> dict` / `Booster.set_attr(self, **kwargs) -> None`

**使用例**:
```python
bst.set_attr(memo="test-model")
print(bst.attributes())
```
実行結果:
```
{'memo': 'test-model'}
```

**注意点・落とし穴**:
- 値は文字列に変換されて保存される(数値を渡しても文字列になる)。`save_model`で保存したファイルにも属性は保持される。

### `XGBClassifier.get_params()` / `.set_params(...)`

**用途**: scikit-learnの`Pipeline`/`GridSearchCV`と連携するためのパラメータ取得・変更メソッド。

**シグネチャ**: `get_params(self, deep=True) -> dict` / `set_params(self, **params) -> self`

**使用例**:
```python
clf_gp = xgb.XGBClassifier(n_estimators=50, max_depth=3)
print(clf_gp.get_params()["max_depth"])
clf_gp.set_params(max_depth=5)
print(clf_gp.get_params()["max_depth"])
```
実行結果:
```
3
5
```

**注意点・落とし穴**:
- `get_params()`はコンストラクタに明示的に渡した値だけを保持し、未指定の項目は`None`で返る(8章参照)。`GridSearchCV`の`param_grid`でxgboostのパラメータをそのまま指定できる。

---

## 補足: 検証に使ったデータセット

上記の実行結果はscikit-learnの`load_iris`/`load_breast_cancer`/`make_classification`/`make_regression`で生成した練習用データに対する実測値であり、`random_state`固定でも将来のxgboost/scikit-learnのバージョン変更で数値が変わりうる。

---

## 応用・発展

以下はCPU環境(GPUなし)で実際に実行して検証した、より発展的・ニッチなAPI。GPU専用機能(`device="cuda"`でのみ意味を持つ設定など)は、この環境では実行結果を確認できないため含めていない。

## 10. カスタム目的関数・カスタム評価関数

### `obj=` / `custom_metric=`(`xgboost.train`)

**用途**: 組み込みの`objective`/`eval_metric`では表現できない独自の損失関数・評価指標を使う。低レベルAPI(`xgb.train`)に、勾配・ヘシアンを返す関数(`obj`)や評価値を返す関数(`custom_metric`)を渡す。

**シグネチャ**: `obj: Callable[[np.ndarray, DMatrix], Tuple[np.ndarray, np.ndarray]]`(予測値とDMatrixを受け取り`(grad, hess)`を返す)/ `custom_metric: Callable[[np.ndarray, DMatrix], Tuple[str, float]]`(予測値とDMatrixを受け取り`(名前, スコア)`を返す)。どちらも`xgboost.train(params, dtrain, ..., obj=None, ..., custom_metric=None)`のキーワード引数(1章の`xgb.train`シグネチャ参照)。

**使用例**:
```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=0, stratify=data.target)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dtest = xgb.DMatrix(Xte, label=yte)

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))     # binary:logisticを手動で再実装
    grad = p - labels
    hess = p * (1.0 - p)
    return grad, hess

def custom_error(preds, dtrain):
    labels = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    err = np.mean((p > 0.5).astype(float) != labels)
    return "custom_error", err

params = {"max_depth": 3, "eta": 0.3, "disable_default_eval_metric": 1}
bst = xgb.train(params, dtrain, num_boost_round=10, obj=logregobj,
                 evals=[(dtest, "eval")], custom_metric=custom_error,
                 verbose_eval=False, evals_result=(res := {}))
print("evals_result keys:", list(res["eval"].keys()))

pred_prob = 1.0 / (1.0 + np.exp(-bst.predict(dtest, output_margin=True)))
bst_builtin = xgb.train({"max_depth": 3, "eta": 0.3, "objective": "binary:logistic"}, dtrain, num_boost_round=10)
print("accuracy(自前logregobj):", accuracy_score(yte, (pred_prob > 0.5).astype(int)))
print("accuracy(組み込みbinary:logistic):", accuracy_score(yte, (bst_builtin.predict(dtest) > 0.5).astype(int)))
```
実行結果:
```
evals_result keys: ['custom_error']
accuracy(自前logregobj): 0.9298245614035088
accuracy(組み込みbinary:logistic): 0.9298245614035088
```

**注意点・落とし穴**:
- `obj`使用時、`Booster.predict()`はデフォルトでは`output_margin=False`でもロジスティック変換(シグモイド)を適用しない生のマージン値を返す(`objective`が`binary:logistic`ではなく`obj`が生スコアを直接最適化する扱いになるため)。この例のように`output_margin=True`で取り出して自分でシグモイドを適用する必要がある。
- `params`に`"objective"`を指定しないまま`obj`だけ渡すと、内部的にはデフォルトの`"reg:squarederror"`が設定される。`Booster.save_config()`で確認すると`learner.objective.name`が`"reg:squarederror"`のままになっており(実測済み)、実際には`obj`の勾配・ヘシアンで学習しているにもかかわらず設定名だけは紛らわしいので注意。
- `custom_metric`を指定しても、`params["objective"]`が(未指定により)`"reg:squarederror"`扱いのままだと既定の評価指標(`rmse`)も自動的に評価結果に混ざる。それを避けるには`disable_default_eval_metric=1`を`params`に加える(実測で`evals_result`のキーから`rmse`が消えることを確認)。

### `objective=<callable>`(scikit-learn API)

**用途**: `XGBClassifier`/`XGBRegressor`でも、文字列ではなく`(grad, hess)`を返す関数を`objective`に渡すことでカスタム目的関数を使える。

**シグネチャ**: `xgboost.XGBClassifier(*, objective: Union[str, Callable[[Any, Any], Tuple[np.ndarray, np.ndarray]], None] = 'binary:logistic', **kwargs)`(`XGBRegressor`も同様のUnion型で、デフォルトは`'reg:squarederror'`)

**使用例**:
```python
import numpy as np
import xgboost as xgb

def logregobj(preds, dtrain):
    labels = dtrain.get_label() if hasattr(dtrain, "get_label") else dtrain
    p = 1.0 / (1.0 + np.exp(-preds))
    return p - labels, p * (1.0 - p)

clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, objective=logregobj, random_state=0)
clf.fit(Xtr, ytr)
print("predict_proba[:2]:", clf.predict_proba(Xte[:2]).round(3))
print("objective属性:", clf.objective)
```
実行結果:
```
predict_proba[:2]: [[1. 0.]
 [1. 0.]]
objective属性: <function logregobj at 0x7736125b8400>
```
(`objective属性`のメモリアドレス部分は実行のたびに変わる)

**注意点・落とし穴**:
- `XGBClassifier.predict_proba()`は、`fit`時に検出したクラス数(2値分類)に基づいて内部で自動的にシグモイド変換を適用する。これは`objective`に何を渡したかに関係なく行われる挙動であり、`obj`が本当にロジスティック損失を再現していないカスタム関数の場合でも同じくシグモイドが適用されてしまう点に注意(`predict_proba`の出力を確率として信頼できるかは、渡した`objective`が実際にロジスティック損失に対応しているかどうかに依存する)。

---

## 11. 単調性制約・特徴量間相互作用制約

### `monotone_constraints`

**用途**: 特定の特徴量に対して、予測値が単調増加/単調減少になるよう木の分岐を制約する。ドメイン知識(例:「価格は面積が広いほど高くなるはず」)をモデルに強制したいときに使う。

**シグネチャ**: `XGBRegressor(..., monotone_constraints=None, ...)`(内部デフォルトは制約なし)。値はタプル/文字列で、各特徴量に対応する位置に`1`(単調増加)・`-1`(単調減少)・`0`(制約なし)を指定する(例: `(1, -1)`)。

**使用例**:
```python
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
n = 2000
x1 = rng.uniform(-3, 3, n)
x2 = rng.uniform(-3, 3, n)
y = 2.0 * x1 - 1.0 * x2 + rng.normal(0, 1.0, n)   # x1に対して増加、x2に対して減少
X = np.column_stack([x1, x2])

reg_free = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=0).fit(X, y)
reg_mono = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=0,
                             monotone_constraints=(1, -1)).fit(X, y)

grid = np.linspace(-3, 3, 50)
X_grid = np.column_stack([grid, np.zeros_like(grid)])

def is_monotonic_increasing(a):
    return np.all(np.diff(a) >= -1e-9)

print("制約なし: x1に対して単調増加か:", is_monotonic_increasing(reg_free.predict(X_grid)))
print("制約あり: x1に対して単調増加か:", is_monotonic_increasing(reg_mono.predict(X_grid)))
```
実行結果:
```
制約なし: x1に対して単調増加か: False
制約あり: x1に対して単調増加か: True
```

**注意点・落とし穴**:
- 制約は「学習データの範囲内外を問わず」予測関数の形に対してかかる。誤った方向を指定すると精度が大きく損なわれることがあるため、事前にドメイン知識で方向を確認してから使う。
- `tree_method`によって内部の制約実装(`exact`系と`hist`系)が異なるが、挙動としてはどちらも単調性を保証する。

### `interaction_constraints`

**用途**: 指定したグループ内の特徴量同士だけが同じ木の分岐パス上で組み合わさることを許可し、グループを跨いだ相互作用を禁止する。

**シグネチャ**: `XGBRegressor(..., interaction_constraints=None, ...)`。値は特徴量名のリストのリスト(例: `[["f0", "f1"], ["f2", "f3"]]`)、または旧来の整数インデックス形式をそのまま使いたい場合はJSON文字列(例: `"[[0,1],[2,3]]"`)を渡す。

**使用例**:
```python
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
n = 3000
X = rng.uniform(-2, 2, size=(n, 4))
y = (X[:, 0] * X[:, 1]) + (X[:, 2] + X[:, 3]) + rng.normal(0, 0.3, n)
dtrain = xgb.DMatrix(X, label=y, feature_names=["f0", "f1", "f2", "f3"])

def mixed_tree_count(bst):
    df = bst.trees_to_dataframe()
    count = 0
    for tid in df["Tree"].unique():
        feats = set(df[(df["Tree"] == tid) & (df["Feature"] != "Leaf")]["Feature"])
        if (feats & {"f0", "f1"}) and (feats & {"f2", "f3"}):
            count += 1
    return count, df["Tree"].nunique()

bst_free = xgb.train({"max_depth": 4, "eta": 0.3, "objective": "reg:squarederror"}, dtrain, num_boost_round=20)
bst_cons = xgb.train({"max_depth": 4, "eta": 0.3, "objective": "reg:squarederror",
                       "interaction_constraints": [["f0", "f1"], ["f2", "f3"]]}, dtrain, num_boost_round=20)

print("制約なし: 両グループを使う木の本数:", mixed_tree_count(bst_free))
print("制約あり: 両グループを使う木の本数:", mixed_tree_count(bst_cons))
```
実行結果:
```
制約なし: 両グループを使う木の本数: (18, 20)
制約あり: 両グループを使う木の本数: (0, 20)
```

**注意点・落とし穴**:
- **バージョン固有の注意**: xgboost 3.4.1で`interaction_constraints`にリストのリストを渡す場合、要素は整数インデックスではなく**特徴量名の文字列**でなければならない(内部で`DMatrix`/`XGBModel`の`feature_names`と突き合わせて整数に変換する実装になっている)。`[[0, 1], [2, 3]]`のように整数を直接渡すと`ValueError: Constrained features are not a subset of training data feature names`になる(実際にエラーを再現して確認済み)。整数インデックスをそのまま使いたい場合は、JSON文字列形式(`"[[0,1],[2,3]]"`)であれば`isinstance(value, str)`の分岐で変換をバイパスできる。
- 制約に含まれていない特徴量(グループのどれにも属さない特徴量)は無制限に使われる。「指定したグループの外の特徴量同士も自由に組み合わさる」点を誤解しないこと。

---

## 12. SHAP値の深掘り

### `Booster.predict(..., pred_interactions=True)`

**用途**: 特徴量ペアごとの相互作用まで分解したSHAP交互作用値(TreeSHAP interaction values)を取得する。

**シグネチャ**: `Booster.predict(self, data, *, pred_interactions=False, ...)`(1章の`Booster.predict`シグネチャ参照)

**使用例**:
```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=0, stratify=data.target)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dtest = xgb.DMatrix(Xte, label=yte)
bst = xgb.train({"max_depth": 4, "eta": 0.3, "objective": "binary:logistic"}, dtrain, num_boost_round=20)

inter = bst.predict(dtest, pred_interactions=True)
contribs = bst.predict(dtest, pred_contribs=True)
margin = bst.predict(dtest, output_margin=True)

print("pred_interactions shape:", inter.shape)
print("pred_contribs shape:", contribs.shape)
print("交互作用行列の全成分合計 == マージン予測値:", np.allclose(inter.sum(axis=(1, 2)), margin, atol=1e-4))
print("交互作用行列を1軸で合計 == pred_contribsと一致:", np.allclose(inter.sum(axis=2), contribs, atol=1e-4))
```
実行結果:
```
pred_interactions shape: (114, 31, 31)
pred_contribs shape: (114, 31)
交互作用行列の全成分合計 == マージン予測値: True
交互作用行列を1軸で合計 == pred_contribsと一致: True
```

**注意点・落とし穴**:
- 出力は`(サンプル数, 特徴量数+1, 特徴量数+1)`の3次元配列(最後の行・列はバイアス項)。特徴量数が多いモデルではメモリ消費・計算コストが`pred_contribs=True`より大幅に大きくなるため、全データに対して常用するのは避け、分析対象を絞って使う。
- 対角成分がその特徴量単独の寄与、非対角成分(`[i,j]`と`[j,i]`は等しい)が特徴量`i`と`j`の交互作用の寄与を表す。どちらか片方の軸だけを合計すると`pred_contribs`と一致する(実測で確認済み)。

### `Booster.predict(..., approx_contribs=True)`

**用途**: 厳密なTreeSHAP(デフォルト)の代わりに、より高速だが近似的な寄与度(Saabas法に近い近似)を計算する。

**シグネチャ**: `Booster.predict(self, data, *, pred_contribs=False, approx_contribs=False, ...)`

**使用例**:
```python
exact = bst.predict(dtest, pred_contribs=True)
approx = bst.predict(dtest, pred_contribs=True, approx_contribs=True)
print("exact と approx が完全一致するか:", np.allclose(exact, approx))
print("最大絶対差:", round(float(np.max(np.abs(exact - approx))), 4))
print("exact[0,:5]:", exact[0, :5].round(4))
print("approx[0,:5]:", approx[0, :5].round(4))
```
実行結果:
```
exact と approx が完全一致するか: False
最大絶対差: 1.0769
exact[0,:5]: [ 0.     -0.0927  0.      0.0104 -0.0554]
approx[0,:5]: [ 0.      0.      0.      0.     -0.0693]
```

**注意点・落とし穴**:
- `approx_contribs=True`は厳密なShapley値の公理(効率性・対称性など)を満たさない近似計算であり、結果は`approx_contribs=False`(デフォルト、厳密なTreeSHAP)と大きく異なりうる(実測で最大絶対差が1を超える例を確認)。精度が重要な分析用途では既定の厳密版を使い、`approx_contribs=True`は計算コストがボトルネックになる大規模データでの高速化目的に限定するのが安全。

---

## 13. ハイパーパラメータチューニングの実践パターン

### `RandomizedSearchCV` + `XGBClassifier`

**用途**: scikit-learnの`RandomizedSearchCV`で、xgboostのハイパーパラメータ(`max_depth`/`learning_rate`/`subsample`/`n_estimators`など)を分布からランダムサンプリングして探索する。

**シグネチャ**: `sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, return_train_score=False)`(scikit-learn 1.9.0で検証)

**使用例**:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform
import xgboost as xgb

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=0, stratify=data.target)

param_dist = {
    "max_depth": randint(2, 6),
    "learning_rate": uniform(0.01, 0.29),
    "subsample": uniform(0.6, 0.4),
    "n_estimators": randint(50, 200),
}
base = xgb.XGBClassifier(random_state=0, eval_metric="logloss")
search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=8, cv=3,
                             scoring="accuracy", random_state=0, n_jobs=1)
search.fit(Xtr, ytr)
print("best_params_:", search.best_params_)
print("best_score_:", round(search.best_score_, 4))
print("test score of best_estimator_:", round(search.best_estimator_.score(Xte, yte), 4))
```
実行結果:
```
best_params_: {'learning_rate': np.float64(0.14852288402319147), 'max_depth': 5, 'n_estimators': 137, 'subsample': np.float64(0.7919908689500229)}
best_score_: 0.9648
test score of best_estimator_: 0.9474
```

**注意点・落とし穴**:
- `RandomizedSearchCV`自体はxgboost固有の機能ではないが、`GridSearchCV`のように全組み合わせを総当たりしないため、探索空間が広いxgboostのチューニング(`max_depth`×`learning_rate`×`subsample`×`n_estimators`×...)では現実的な計算時間で回せる利点がある。`n_iter`が実際に試す組み合わせ数を決める。
- `XGBClassifier`に`eval_metric`だけ渡し`early_stopping_rounds`は渡していない(`RandomizedSearchCV`は`fit`に`eval_set`を自動では渡さないため、`CV`のループ内でearly stoppingを使うには`fit_params`経由で`eval_set`を明示的に渡すなど追加の工夫が必要)。

### `evals_result()`による学習曲線(訓練/検証の乖離を見る)

**用途**: `eval_set`に訓練データと検証データの両方を渡し、ラウンドごとの指標推移から過学習の兆候を確認する。

**シグネチャ**: `XGBClassifier.fit(self, X, y, *, eval_set=None, ...)` / `XGBClassifier.evals_result(self) -> dict`

**使用例**:
```python
clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         eval_metric=["logloss", "error"], random_state=0)
clf.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)
res = clf.evals_result()
train_logloss = res["validation_0"]["logloss"]
val_logloss = res["validation_1"]["logloss"]
print("train_logloss: 1ラウンド目=%.4f 最終ラウンド=%.4f" % (train_logloss[0], train_logloss[-1]))
print("val_logloss:   1ラウンド目=%.4f 最終ラウンド=%.4f" % (val_logloss[0], val_logloss[-1]))
```
実行結果:
```
train_logloss: 1ラウンド目=0.5767 最終ラウンド=0.0103
val_logloss:   1ラウンド目=0.5833 最終ラウンド=0.1763
```

**注意点・落とし穴**:
- `eval_set`の1番目に渡したデータが`validation_0`、2番目が`validation_1`という機械的な命名になる(データの中身が「訓練」か「検証」かはキー名からは分からず、渡した順番を自分で把握しておく必要がある)。
- この例では訓練データのlogloss(0.0103)が検証データのlogloss(0.1763)より大幅に低く、ラウンドが進むほど両者が乖離しており過学習の典型的なパターンが確認できる。`early_stopping_rounds`(4章)と組み合わせて、乖離が始まる直前で打ち切るのが実践的なパターン。

### `xgboost.callback.LearningRateScheduler(...)`

**用途**: ラウンドごとに学習率(`eta`)を動的に変化させる(例:ラウンドが進むにつれて減衰させる)。

**シグネチャ**: `xgboost.callback.LearningRateScheduler(self, learning_rates: Union[Callable[[int], float], Sequence[float]])`

**使用例**:
```python
import json
import xgboost as xgb

def lr_sched(epoch):
    return 0.3 * (0.95 ** epoch)

lrs = xgb.callback.LearningRateScheduler(lr_sched)
bst = xgb.train({"max_depth": 3, "objective": "binary:logistic"}, dtrain, num_boost_round=10,
                 callbacks=[lrs], evals=[(dtest, "eval")], verbose_eval=False,
                 evals_result=(res := {}))
cfg = json.loads(bst.save_config())
print("最終ラウンド時点のeta:", cfg["learner"]["gradient_booster"]["tree_train_param"]["eta"])
print("手計算(0.3*0.95**9):", round(0.3 * 0.95 ** 9, 9))
print("最終ラウンドのeval logloss:", round(res["eval"]["logloss"][-1], 4))
```
実行結果:
```
最終ラウンド時点のeta: 0.189074829
手計算(0.3*0.95**9): 0.189074823
最終ラウンドのeval logloss: 0.1847
```

**注意点・落とし穴**:
- `learning_rates`に渡す関数は「現在のラウンド番号(0始まり)」を受け取り、そのラウンドで使う`eta`を返す。固定値のリスト/タプル(`num_boost_round`と同じ長さ)を渡すこともできる。
- `XGBClassifier`側にも`callbacks`引数があるため、scikit-learn APIでも同様に使える(`xgb.XGBClassifier(..., callbacks=[lrs])`)。

---

## 14. 特殊なタスク・モデル診断の応用

### `Booster.predict(..., pred_leaf=True)`

**用途**: 各サンプルが各木でどの葉ノードに落ちたか(葉のインデックス)を取得する。得られたインデックスをワンホット化して別モデルの特徴量にする「木の葉によるスタッキング」的な使い方ができる。

**シグネチャ**: `Booster.predict(self, data, *, pred_leaf=False, ...)`

**使用例**:
```python
leaf_idx = bst.predict(dtest, pred_leaf=True)
print("pred_leaf shape:", leaf_idx.shape)
print("dtype:", leaf_idx.dtype)
print("1サンプル目の各木での葉インデックス:", leaf_idx[0])
```
実行結果:
```
pred_leaf shape: (114, 20)
dtype: float32
1サンプル目の各木での葉インデックス: [10. 10. 12. 12. 18. 14. 14. 10.  6. 12. 10.  6. 10.  6.  6. 10. 10.  2.
  6.  6.]
```

**注意点・落とし穴**:
- 出力の列数は木の本数(`num_boosted_rounds()`)と一致する。多クラス分類では「ラウンド数×クラス数」本の木があるため列数もその分だけ増える。
- 値は`float32`型だが中身は整数の葉インデックスなので、後続処理(ワンホットエンコードなど)では明示的に整数キャストしてから使うのが安全。

### `multi_strategy="multi_output_tree"`

**用途**: 複数の目的変数(マルチターゲット回帰)を、ターゲットごとに独立した木の集合を作るのではなく、1本の木が複数出力を同時に予測する構造で学習する。

**シグネチャ**: `XGBRegressor(..., multi_strategy=None, ...)`(内部デフォルトは`"one_output_per_tree"`。`"multi_output_tree"`を指定すると1本の木で複数出力を扱う)

**使用例**:
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

rng = np.random.default_rng(0)
X, y1 = make_regression(n_samples=300, n_features=5, noise=5.0, random_state=0)
y2 = y1 * 0.5 + rng.normal(0, 5, size=y1.shape[0])
Y = np.column_stack([y1, y2])
Xtr2, Xte2, Ytr2, Yte2 = train_test_split(X, Y, test_size=0.2, random_state=0)

reg_multi = xgb.XGBRegressor(n_estimators=50, max_depth=3, multi_strategy="multi_output_tree",
                              tree_method="hist", random_state=0).fit(Xtr2, Ytr2)
reg_default = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=0).fit(Xtr2, Ytr2)

print("predict shape (multi_output_tree):", reg_multi.predict(Xte2).shape)
print("R2 per output (multi_output_tree):", r2_score(Yte2, reg_multi.predict(Xte2), multioutput="raw_values").round(3))
print("R2 per output (既定 one_output_per_tree):", r2_score(Yte2, reg_default.predict(Xte2), multioutput="raw_values").round(3))
```
実行結果:
```
predict shape (multi_output_tree): (60, 2)
R2 per output (multi_output_tree): [0.958 0.96 ]
R2 per output (既定 one_output_per_tree): [0.961 0.941]
```

**注意点・落とし穴**:
- `multi_strategy="multi_output_tree"`は`tree_method="hist"`(またはGPU版)でのみサポートされる比較的新しい機能。既定の`"one_output_per_tree"`(出力ごとに個別の木群を学習)と比べて、出力間の相関を木構造自体で捉えられる可能性がある一方、常に精度が上がるわけではない(この実測例では片方の出力で改善、もう片方はほぼ同等)。

### `objective="reg:quantileerror"` + `quantile_alpha`

**用途**: 複数の分位点(例:10%点・中央値・90%点)を1回の学習でまとめて予測する分位点回帰。

**シグネチャ**: `XGBRegressor(objective="reg:quantileerror", quantile_alpha=None, ...)`(`quantile_alpha`に単一の分位点、またはリストで複数の分位点を指定)

**使用例**:
```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

X, y = make_regression(n_samples=500, n_features=5, noise=15.0, random_state=0)
Xtr3, Xte3, ytr3, yte3 = train_test_split(X, y, test_size=0.2, random_state=0)

reg_q = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=[0.1, 0.5, 0.9],
                          n_estimators=100, max_depth=3, random_state=0)
reg_q.fit(Xtr3, ytr3)
pred_q = reg_q.predict(Xte3)
print("predict shape:", pred_q.shape)
print("1サンプル目 [q0.1, q0.5, q0.9]:", pred_q[0].round(2))
print("q0.1<=q0.5<=q0.9が全サンプルで成立:", bool(np.all(pred_q[:, 0] <= pred_q[:, 1]) and np.all(pred_q[:, 1] <= pred_q[:, 2])))
coverage = np.mean((yte3 >= pred_q[:, 0]) & (yte3 <= pred_q[:, 2]))
print("実測値がq0.1〜q0.9に収まる割合:", round(coverage, 3))
```
実行結果:
```
predict shape: (100, 3)
1サンプル目 [q0.1, q0.5, q0.9]: [-129.65  -99.94  -81.34]
q0.1<=q0.5<=q0.9が全サンプルで成立: True
実測値がq0.1〜q0.9に収まる割合: 0.6
```

**注意点・落とし穴**:
- `quantile_alpha`にリストを渡すと`predict()`の出力が`(サンプル数, 分位点数)`の2次元配列になる(`quantile_alpha`を単一の浮動小数点数にすると1次元配列に戻る)。
- この実測例では「q0.1〜q0.9区間」の理論的なカバー率(80%)に対し、実測のカバー率は60%だった。ハイパーパラメータ(木の本数・深さなど)を調整せずにデフォルト寄りの設定で試した結果であり、分位点回帰は通常の回帰よりキャリブレーション(区間の妥当性)を別途検証する必要があることを示す一例。

### `xgboost.XGBRanker(...)`

**用途**: 検索結果のランキングなど、「グループ内での相対順序」を学習するランキングタスク用のscikit-learn APIラッパー。

**シグネチャ**: `xgboost.XGBRanker(*, objective='rank:ndcg', **kwargs)`。`fit(self, X, y, *, group=None, qid=None, ...)`(`qid`でグループ(クエリ)ごとにサンプルをまとめる)。

**使用例**:
```python
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr

rng = np.random.default_rng(0)
n_groups, docs_per_group = 20, 10
X = rng.uniform(0, 1, size=(n_groups * docs_per_group, 4))
true_rel = np.clip((X[:, 0] * 3).astype(int), 0, 3)   # 関連度はほぼ特徴量0で決まる
qid = np.repeat(np.arange(n_groups), docs_per_group)

ranker = xgb.XGBRanker(objective="rank:pairwise", n_estimators=50, max_depth=3, random_state=0)
ranker.fit(X, true_rel, qid=qid)
scores = ranker.predict(X)
print("scores shape:", scores.shape)
corr = spearmanr(scores[:docs_per_group], true_rel[:docs_per_group]).correlation
print("グループ0内の順位相関(予測スコア vs 真の関連度):", round(corr, 3))
```
実行結果:
```
scores shape: (200,)
グループ0内の順位相関(予測スコア vs 真の関連度): 0.967
```

**注意点・落とし穴**:
- `y`(この例では`true_rel`)は「絶対的なラベル」ではなく「同じ`qid`グループ内での相対順序」を学習するためだけに使われる。グループを跨いだスコアの大小比較には意味がない。
- `qid`はグループ番号でソート済みである必要がある(この例のように`np.repeat`で連続したブロックを作る)。バラバラの順序で渡すとエラーになるか意図しない学習結果になる。
- 既定の`objective`は`'rank:ndcg'`だが、この例では`'rank:pairwise'`を明示的に指定している(2章のようにデフォルトのまま使う場合は`objective`を省略できる)。

### `Booster.get_split_value_histogram(...)`

**用途**: 特定の特徴量について、全ての木でどのしきい値がどれだけ分岐に使われたかをヒストグラム(DataFrame)として取得する。

**シグネチャ**: `Booster.get_split_value_histogram(self, feature: str, fmap: PathLike = '', bins: Optional[int] = None, as_pandas: bool = True) -> Union[np.ndarray, pd.DataFrame]`

**使用例**:
```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
dtrain_h = xgb.DMatrix(data.data, label=data.target, feature_names=list(data.feature_names))
bst_h = xgb.train({"max_depth": 3, "objective": "binary:logistic"}, dtrain_h, num_boost_round=10)
hist = bst_h.get_split_value_histogram("worst radius", bins=5)
print(type(hist))
print(hist)
```
実行結果:
```
<class 'pandas.DataFrame'>
   SplitValue  Count
0   15.496667    1.0
1   17.730000    2.0
```
(`bins=5`を指定しても、実際に分岐に使われたしきい値の種類が少ない場合はそれより少ない行数になる)

**注意点・落とし穴**:
- `bins`は「最大でこの数まで」という上限であり、実際の出力行数は分岐に使われたユニークなしきい値の数以下になる(この例では5を指定しても2行しか返らない)。
- `trees_to_dataframe()`(5章)から自分で`Feature`列を絞り込んで集計しても同等の情報は得られるが、単一特徴量のしきい値分布だけを素早く確認したいときはこちらの方が簡潔。
