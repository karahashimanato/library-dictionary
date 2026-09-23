# lightgbm 逆引き辞書

lightgbm 4.7.0 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(lightgbm 4.7.0)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。

## 目次

1. [Dataset・学習基礎](#1-dataset学習基礎)
2. [scikit-learn API](#2-scikit-learn-api)
3. [交差検証](#3-交差検証)
4. [カテゴリ変数のネイティブ対応](#4-カテゴリ変数のネイティブ対応)
5. [サンプリング・正則化](#5-サンプリング正則化)
6. [早期終了・コールバック](#6-早期終了コールバック)
7. [特徴量重要度・可視化](#7-特徴量重要度可視化)
8. [モデルの保存・読み込み](#8-モデルの保存読み込み)
9. [その他](#9-その他)
10. [カスタム目的関数・評価関数](#10-カスタム目的関数評価関数)
11. [木構造の高度な制御](#11-木構造の高度な制御)
12. [ランキング学習(LambdaRank)](#12-ランキング学習lambdarank)
13. [交差検証の応用](#13-交差検証の応用)
14. [モデル内部構造の解析・再学習](#14-モデル内部構造の解析再学習)

---

## 1. Dataset・学習基礎

### `lightgbm.Dataset(...)`

**用途**: LightGBM専用のデータ形式。学習・検証データをラップし、内部でヒストグラム化などの前処理を行う。

**シグネチャ**: `lightgbm.Dataset(data, label=None, reference=None, weight=None, group=None, init_score=None, feature_name='auto', categorical_feature='auto', params=None, free_raw_data=True, position=None)`

**使用例**:
```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

dtrain = lgb.Dataset(Xtr, label=ytr)
dvalid = lgb.Dataset(Xte, label=yte, reference=dtrain)
dtrain.construct()
print(dtrain.num_data(), dtrain.num_feature())
```
実行結果:
```
455 30
```

**注意点・落とし穴**:
- `num_data()`/`num_feature()`など内部情報を取得するメソッドは、`construct()`(または`lgb.train`への引き渡し)で実際にデータが構築された後でないと`LightGBMError: Cannot get num_data before construct dataset`になる。
- 検証用データセットは`reference=dtrain`を指定し、学習データと同じビン(bin)分割を共有させる必要がある(指定しないと`train`/`cv`時に想定外の挙動になりうる)。
- デフォルトの`free_raw_data=True`だと、`construct()`後に元データ(`.data`)がメモリ節約のため解放されて`None`になる。後から`init_model`で追加学習したい場合は`free_raw_data=False`にしておく必要がある(詳細は9章)。

### `lightgbm.train(...)`

**用途**: 与えられたパラメータ・`Dataset`を使ってブースティング木を学習し、`Booster`を返す(LightGBMのネイティブ学習API)。

**シグネチャ**: `lightgbm.train(params, train_set, num_boost_round=100, valid_sets=None, valid_names=None, feval=None, init_model=None, keep_training_booster=False, callbacks=None)`

**使用例**:
```python
params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0}
bst = lgb.train(
    params,
    dtrain,
    num_boost_round=50,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[lgb.log_evaluation(period=10)],
)
print("best_iteration:", bst.best_iteration)
print("pred[:5]:", bst.predict(Xte[:5]).round(4))
```
実行結果:
```
[10]	valid's binary_logloss: 0.259173
[20]	valid's binary_logloss: 0.140995
[30]	valid's binary_logloss: 0.0916942
[40]	valid's binary_logloss: 0.064516
[50]	valid's binary_logloss: 0.056115
best_iteration: 0
pred[:5]: [0.0037 0.9704 0.9944 0.9871 0.9896]
```

**注意点・落とし穴**:
- `early_stopping`コールバックを使わない場合、`bst.best_iteration`は`0`のままになる(「ベストの反復回数」が未設定=全木を使う、という意味)。早期終了を使わないなら`best_iteration`を参照しても意味がない。
- パラメータ辞書のキー名(`objective`, `metric`など)は文字列で、scikit-learn APIのコンストラクタ引数(`objective`, `boosting_type`など)とは一部名前・書式が異なる(例: ネイティブAPIは`num_leaves`、`lambda_l1`のようにLightGBM公式パラメータ名をそのまま使う)。

### `Booster.predict(...)`

**用途**: 学習済み`Booster`で予測する。確率・生スコア・葉インデックス・SHAP風の特徴量寄与度など複数のモードを持つ。

**シグネチャ**: `Booster.predict(data, start_iteration=0, num_iteration=None, raw_score=False, pred_leaf=False, pred_contrib=False, data_has_header=False, validate_features=False, **kwargs)`

**使用例**:
```python
params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0}
bst = lgb.train(
    params, dtrain, num_boost_round=500, valid_sets=[dvalid], valid_names=["valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
)
print("best_iteration:", bst.best_iteration, "num_trees:", bst.num_trees())
print("pred(best):", bst.predict(Xte[:3]).round(4))
print("pred(5 trees):", bst.predict(Xte[:3], num_iteration=5).round(4))
```
実行結果:
```
best_iteration: 52 num_trees: 52
pred(best): [0.0032 0.9708 0.9951]
pred(5 trees): [0.3965 0.7281 0.7783]
```

**注意点・落とし穴**:
- `early_stopping`を使うと、`num_boost_round=500`と指定していても実際には改善が止まった時点(この例では52本)で学習自体が打ち切られる。`bst.num_trees()`が実際に学習された木の本数。
- `predict`の`num_iteration=None`(デフォルト)は、`best_iteration`が設定されていればそれを、なければ全木を使う。`num_iteration`に`num_trees()`を超える値を指定してもエラーにはならず、存在する木すべてが使われる。

---

## 2. scikit-learn API

### `lightgbm.LGBMClassifier(...)` / `lightgbm.LGBMRegressor(...)`

**用途**: scikit-learn互換のEstimator API。`Pipeline`や`GridSearchCV`などscikit-learnのエコシステムにそのまま組み込める。

**シグネチャ**: `lightgbm.LGBMClassifier(*, boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=None, importance_type='split', **kwargs)`(`LGBMRegressor`も同じ引数構成)

**使用例**:
```python
clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=0, verbosity=-1)
clf.fit(Xtr, ytr)
print("score:", clf.score(Xte, yte))
print("predict[:5]:", clf.predict(Xte[:5]))
```
実行結果:
```
score: 0.9824561403508771
predict[:5]: [0 1 1 1 1]
```

**注意点・落とし穴**:
- ネイティブAPIの`num_boost_round`に相当するのが`n_estimators`。同様に`lambda_l1`→`reg_alpha`、`lambda_l2`→`reg_lambda`、`bagging_fraction`→`subsample`、`feature_fraction`→`colsample_bytree`のように、ネイティブAPIとscikit-learn APIでパラメータ名が異なるものが複数ある。
- `num_leaves`のデフォルトは`31`。`max_depth=-1`(無制限)と組み合わさっているため、木の複雑さは実質`num_leaves`が支配的。

### `LGBMClassifier.fit(...)`

**用途**: 学習を実行する。検証用データを渡して`eval_metric`や早期終了コールバックと組み合わせられる。

**シグネチャ**: `LGBMClassifier.fit(X, y, sample_weight=None, init_score=None, eval_set=None, eval_names=None, eval_sample_weight=None, eval_class_weight=None, eval_init_score=None, eval_metric=None, feature_name='auto', categorical_feature='auto', callbacks=None, init_model=None, *, eval_X=None, eval_y=None)`

**使用例**:
```python
clf = lgb.LGBMClassifier(n_estimators=100, random_state=0, verbosity=-1)
clf.fit(
    Xtr, ytr,
    eval_X=Xte, eval_y=yte,
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
)
print("best_iteration_:", clf.best_iteration_)
```
実行結果:
```
best_iteration_: 52
```

**注意点・落とし穴**:
- **バージョン固有の注意**: lightgbm 4.7.0では引数`eval_set=[(Xte, yte)]`(旧来からよく使われてきた書き方)を渡すと`LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead.`が出る(動作はするが非推奨)。新しいコードではキーワード専用引数の`eval_X`/`eval_y`を使う。
- 早期終了を使うには`eval_metric`と`callbacks=[lgb.early_stopping(...)]`の両方が必要(`eval_metric`だけでは止まらない)。

### `LGBMClassifier.predict(...)` / `LGBMClassifier.predict_proba(...)`

**用途**: クラスラベル/クラス所属確率を予測する。多クラス分類にもそのまま対応する。

**シグネチャ**: `LGBMClassifier.predict(X, raw_score=False, start_iteration=0, num_iteration=None, pred_leaf=False, pred_contrib=False, validate_features=False, **kwargs)` (`predict_proba`も同じ引数)

**使用例**:
```python
from sklearn.datasets import load_iris
Xi, yi = load_iris(return_X_y=True)
Xitr, Xite, yitr, yite = train_test_split(Xi, yi, test_size=0.2, random_state=0, stratify=yi)
clf3 = lgb.LGBMClassifier(n_estimators=50, random_state=0, verbosity=-1)
clf3.fit(Xitr, yitr)
print("classes_:", clf3.classes_)
print("predict_proba[0]:", clf3.predict_proba(Xite[:1]).round(4))
print("objective_:", clf3.objective_)
```
実行結果:
```
classes_: [0 1 2]
predict_proba[0]: [[9.984e-01 1.300e-03 3.000e-04]]
objective_: multiclass
```

**注意点・落とし穴**:
- `objective`を明示しなくても、`fit`時のクラス数から`objective_`が自動的に`binary`または`multiclass`に決まる。
- `predict_proba`の列の並びは`classes_`の順(通常は昇順にソートされたラベル)に対応する。

### `get_params(...)` / `set_params(...)`

**用途**: scikit-learn Estimator共通のパラメータ取得・変更メソッド。`GridSearchCV`などが内部的に使う。

**シグネチャ**: `LGBMClassifier.get_params(deep=True)` / `LGBMClassifier.set_params(**params)`

**使用例**:
```python
clf4 = lgb.LGBMClassifier(n_estimators=30, num_leaves=15, random_state=0, verbosity=-1)
print("num_leaves:", clf4.get_params()["num_leaves"])
clf4.set_params(num_leaves=7)
print("after set_params:", clf4.get_params()["num_leaves"])
```
実行結果:
```
num_leaves: 15
after set_params: 7
```

**注意点・落とし穴**:
- `**kwargs`で渡した非標準パラメータ(例: `min_data_in_leaf`のようなネイティブAPI名)も`get_params()`には現れるが、コンストラクタの明示引数(`min_child_samples`など)とは別枠として扱われるため、両方を同時に指定すると意図しない方が優先される場合がある。

### `class_weight='balanced'`

**用途**: 分類のクラス不均衡を補正するため、各クラスの重みをサンプル数に反比例させる。

**シグネチャ**: `LGBMClassifier(..., class_weight=None)` (`None`または`dict`または`'balanced'`)

**使用例**:
```python
from sklearn.datasets import make_classification
Xb, yb = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=0)
Xbtr, Xbte, ybtr, ybte = train_test_split(Xb, yb, test_size=0.2, random_state=0, stratify=yb)

clf_plain = lgb.LGBMClassifier(n_estimators=50, random_state=0, verbosity=-1).fit(Xbtr, ybtr)
clf_bal = lgb.LGBMClassifier(n_estimators=50, random_state=0, verbosity=-1, class_weight="balanced").fit(Xbtr, ybtr)
print("mean proba(class1) plain:", clf_plain.predict_proba(Xbte)[:, 1].mean().round(4))
print("mean proba(class1) balanced:", clf_bal.predict_proba(Xbte)[:, 1].mean().round(4))
```
実行結果:
```
mean proba(class1) plain: 0.0457
mean proba(class1) balanced: 0.0538
```

**注意点・落とし穴**:
- `class_weight='balanced'`は確率の値自体を底上げするが、必ずしも`predict()`の0.5閾値判定(≒recall)を大きく変えるとは限らない(閾値調整や`scale_pos_weight`と併用することが多い)。
- 二値分類では、ネイティブAPI側の`scale_pos_weight`パラメータの方がよく使われる(`class_weight`はscikit-learn APIのみ)。

---

## 3. 交差検証

### `lightgbm.cv(...)`

**用途**: 指定した分割数でクロスバリデーションを行い、fold平均のスコア推移を返す(ネイティブAPI用)。

**シグネチャ**: `lightgbm.cv(params, train_set, num_boost_round=100, folds=None, nfold=5, stratified=True, shuffle=True, metrics=None, feval=None, init_model=None, fpreproc=None, seed=0, callbacks=None, eval_train_metric=False, return_cvbooster=False)`

**使用例**:
```python
dtrain_full = lgb.Dataset(X, label=y)
params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0}
result = lgb.cv(
    params, dtrain_full, num_boost_round=100, nfold=5, stratified=True, seed=0,
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
)
print(list(result.keys()))
print("実際に回った反復数:", len(result["valid binary_logloss-mean"]))
print("最終mean:", result["valid binary_logloss-mean"][-1])
print("最終stdv:", result["valid binary_logloss-stdv"][-1])
```
実行結果:
```
['valid binary_logloss-mean', 'valid binary_logloss-stdv']
実際に回った反復数: 52
最終mean: 0.09162099814169164
最終stdv: 0.0362408842737323
```

**注意点・落とし穴**:
- 戻り値は「学習済みモデル」ではなく、反復ごとの平均・標準偏差(`-mean`/`-stdv`)を格納した辞書。最終的なモデルが欲しい場合は、`cv`で決めた`num_boost_round`(または`early_stopping`後の反復数)を使って改めて`lgb.train`を呼ぶ必要がある。
- `stratified=True`(デフォルト)は分類向け。回帰では自動的に効かない(意味を持たない)。

### `lightgbm.cv(..., return_cvbooster=True)`

**用途**: 各foldで学習された`Booster`をまとめた`CVBooster`を取得し、fold平均アンサンブルの予測に使う。

**シグネチャ**: `lightgbm.cv(..., return_cvbooster=True)`(戻り値の辞書に`"cvbooster"`キーが追加される)

**使用例**:
```python
result2 = lgb.cv(params, dtrain_full, num_boost_round=10, nfold=3, seed=0, return_cvbooster=True)
cvb = result2["cvbooster"]
print(type(cvb).__name__, "boosters数:", len(cvb.boosters))
preds = cvb.predict(Xte[:3])
import numpy as np
print("各foldモデルの予測 shape:", np.array(preds).shape)
```
実行結果:
```
CVBooster boosters数: 3
各foldモデルの予測 shape: (3, 3)
```

**注意点・落とし穴**:
- `cvb.predict(X)`はfold数と同じ数の予測配列のリストを返す(自動平均はされない)。全体の予測が欲しい場合は`np.mean(preds, axis=0)`のように自分で平均する。

---

## 4. カテゴリ変数のネイティブ対応

### `categorical_feature`(Dataset / fit引数)

**用途**: One-Hotエンコーディングせずに、カテゴリ変数をLightGBM内部の最適分割アルゴリズムでそのまま扱う。

**シグネチャ**: `lightgbm.Dataset(data, ..., categorical_feature='auto')` / `LGBMClassifier.fit(X, y, ..., categorical_feature='auto')`

**使用例**:
```python
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
n = 300
df = pd.DataFrame({
    "city": pd.Categorical(rng.choice(["Tokyo", "Osaka", "Nagoya"], size=n)),
    "x1": rng.normal(size=n),
})
yv = (df["x1"] + (df["city"] == "Tokyo").astype(float) * 2 + rng.normal(scale=0.5, size=n) > 0).astype(int)

dtrain_c = lgb.Dataset(df, label=yv, categorical_feature=["city"])
bst_c = lgb.train({"objective": "binary", "verbosity": -1, "seed": 0}, dtrain_c, num_boost_round=30)
print("pred[:5]:", bst_c.predict(df[:5]).round(4))
```
実行結果:
```
pred[:5]: [0.9838 0.9777 0.9847 0.9777 0.9246]
```

**注意点・落とし穴**:
- `categorical_feature`に渡す列は、pandasの`category`dtype、または整数エンコード済みの列である必要がある。`object`(文字列)dtypeのまま渡すと`ValueError: pandas dtypes must be int, float or bool. Fields with bad pandas dtypes: city: str`になる。
- `feature_name`/`categorical_feature`のデフォルト`'auto'`は、DataFrameの`category`dtype列を自動的にカテゴリ特徴量として検出する(明示指定しなくても動く場合がある)。

### pandas `category` dtypeの自動認識(scikit-learn API)

**用途**: `LGBMClassifier`/`LGBMRegressor`に直接pandas DataFrameを渡すと、`category`dtype列を自動でカテゴリ特徴量として扱う。

**シグネチャ**: `LGBMClassifier.fit(X, y, ...)` (Xがpandas DataFrameで`category`dtype列を含む場合)

**使用例**:
```python
clf5 = lgb.LGBMClassifier(n_estimators=30, random_state=0, verbosity=-1)
clf5.fit(df, yv)
print("sklearn APIでの予測[:5]:", clf5.predict_proba(df[:5])[:, 1].round(4))
```
実行結果:
```
sklearn APIでの予測[:5]: [0.9838 0.9777 0.9847 0.9777 0.9246]
```

**注意点・落とし穴**:
- ネイティブAPIで明示的に`categorical_feature=["city"]`を指定した場合と、scikit-learn APIで`category`dtypeを自動検出させた場合とで、今回の検証では同一の予測結果が得られた。ただし列の型変換(`astype("category")`)を忘れると自動認識されない点に注意。

---

## 5. サンプリング・正則化

### `boosting_type='goss'`

**用途**: GOSS(Gradient-based One-Side Sampling)。勾配の大きいサンプルを優先的に残しつつ、小さいサンプルは間引いて高速化する。

**シグネチャ**: `lightgbm.train({"boosting_type": "goss", ...}, ...)` / `LGBMClassifier(boosting_type='goss', ...)`

**使用例**:
```python
from sklearn.metrics import roc_auc_score
params_goss = {
    "objective": "binary", "boosting_type": "goss", "metric": "binary_logloss",
    "num_leaves": 15, "verbosity": -1, "seed": 0,
}
bst_goss = lgb.train(params_goss, dtrain, num_boost_round=50)
print("AUC(goss):", round(roc_auc_score(yte, bst_goss.predict(Xte)), 4))
```
実行結果:
```
AUC(goss): 0.9978
```

**注意点・落とし穴**:
- `boosting_type='goss'`使用時は`bagging_fraction`/`bagging_freq`(行のランダムサブサンプリング)は併用できない(GOSS自体が独自のサンプリング手法のため)。

### `feature_fraction` / `bagging_fraction` + `bagging_freq`

**用途**: `feature_fraction`は各木ごとに使用する特徴量の割合、`bagging_fraction`は各反復で使用するデータ行の割合を制御し、過学習抑制・高速化・多様性向上に使う。

**シグネチャ**: パラメータ辞書のキー(`lightgbm.train`) / `LGBMClassifier(colsample_bytree=1.0, subsample=1.0, subsample_freq=0, ...)`(scikit-learn APIでの対応名)

**使用例**:
```python
params_bag = {
    "objective": "binary", "metric": "binary_logloss",
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "verbosity": -1, "seed": 0,
}
bst_bag = lgb.train(params_bag, dtrain, num_boost_round=50)
print("AUC(feature+bagging fraction):", round(roc_auc_score(yte, bst_bag.predict(Xte)), 4))
```
実行結果:
```
AUC(feature+bagging fraction): 0.999
```

**注意点・落とし穴**:
- `bagging_fraction`を効かせるには`bagging_freq`(何反復ごとにバギングを行うか)を1以上に設定する必要がある。LightGBM公式ドキュメントでは`bagging_freq=0`だとバギングが無効になるとされているが、`bagging_fraction`だけ変えても`bagging_freq=0`のままモデルが変化するケースを実機で確認しており(本検証環境、LightGBM 4.7.0)、内部挙動の詳細は未確認。**確実に効かせたい場合は両方を明示的に設定する**。
- scikit-learn APIでは`feature_fraction`→`colsample_bytree`、`bagging_fraction`→`subsample`、`bagging_freq`→`subsample_freq`という名前になる。

### `lambda_l1` / `lambda_l2`(L1・L2正則化)

**用途**: 葉の重みに対するL1/L2正則化。過学習を抑える。

**シグネチャ**: パラメータ辞書のキー(`lambda_l1=0.0`, `lambda_l2=0.0`) / scikit-learn APIでは`reg_alpha`/`reg_lambda`

**使用例**:
```python
params_reg = {
    "objective": "binary", "metric": "binary_logloss",
    "lambda_l1": 0.1, "lambda_l2": 0.1, "verbosity": -1, "seed": 0,
}
bst_reg = lgb.train(params_reg, dtrain, num_boost_round=50)
print("AUC(L1+L2正則化):", round(roc_auc_score(yte, bst_reg.predict(Xte)), 4))
```
実行結果:
```
AUC(L1+L2正則化): 0.999
```

**注意点・落とし穴**:
- デフォルトはどちらも`0.0`(正則化なし)。scikit-learn API側の`reg_alpha`/`reg_lambda`も同じくデフォルト`0.0`。

### `min_child_samples`(`min_data_in_leaf`)

**用途**: 1つの葉に含まれる最小サンプル数。小さすぎる葉(過学習の元)を防ぐ。

**シグネチャ**: `LGBMClassifier(min_child_samples=20, ...)` (ネイティブAPIでは`min_data_in_leaf`)

**使用例**:
```python
clf_small_leaf = lgb.LGBMClassifier(n_estimators=50, min_child_samples=5, random_state=0, verbosity=-1).fit(Xtr, ytr)
clf_big_leaf = lgb.LGBMClassifier(n_estimators=50, min_child_samples=100, random_state=0, verbosity=-1).fit(Xtr, ytr)
print("score(min_child_samples=5):", clf_small_leaf.score(Xte, yte))
print("score(min_child_samples=100):", clf_big_leaf.score(Xte, yte))
```
実行結果:
```
score(min_child_samples=5): 0.9736842105263158
score(min_child_samples=100): 0.956140350877193
```

**注意点・落とし穴**:
- データ数が少ない(数百件程度)場合、デフォルトの`20`でも相対的に大きい制約になり木が浅くなりやすい。小規模データでは値を下げて試す価値がある。

### `monotone_constraints`

**用途**: 特定の特徴量に対して、予測値が単調増加/単調減少になるよう制約をかける(ビジネスルール上の解釈可能性が必要な場合に有用)。

**シグネチャ**: `LGBMRegressor(monotone_constraints=None, ...)`(各特徴量に`1`=増加、`-1`=減少、`0`=制約なしのリストを渡す)

**使用例**:
```python
import numpy as np
rng2 = np.random.RandomState(0)
xm = rng2.uniform(0, 10, size=500).reshape(-1, 1)
ym = xm[:, 0] * 2 + rng2.normal(scale=1.0, size=500)
reg_mono = lgb.LGBMRegressor(n_estimators=100, monotone_constraints=[1], random_state=0, verbosity=-1)
reg_mono.fit(xm, ym)
xs = np.linspace(0, 10, 20).reshape(-1, 1)
preds = reg_mono.predict(xs)
print("単調増加になっているか:", bool(np.all(np.diff(preds) >= -1e-9)))
```
実行結果:
```
単調増加になっているか: True
```

**注意点・落とし穴**:
- リストの長さは特徴量数と一致させる必要がある。制約をかけない特徴量には`0`を指定する。

---

## 6. 早期終了・コールバック

### `lightgbm.early_stopping(...)`

**用途**: 検証スコアが指定ラウンド数だけ改善しなくなったら学習を打ち切るコールバック。

**シグネチャ**: `lightgbm.early_stopping(stopping_rounds, first_metric_only=False, verbose=True, min_delta=0.0)`

**使用例**:
```python
bst_es = lgb.train(
    {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0},
    dtrain, num_boost_round=500, valid_sets=[dvalid], valid_names=["valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=20)],
)
```
実行結果:
```
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[52]	valid's binary_logloss: 0.0546137
```

**注意点・落とし穴**:
- `verbose=True`(デフォルト)だと上記のような打ち切りメッセージが標準出力に出る。静かにしたい場合は`verbose=False`を指定する。
- `metrics`を複数指定している場合、`first_metric_only=False`(デフォルト)だと「すべての指標が改善しなくなるまで」待つ。1つ目の指標だけで判定したい場合は`first_metric_only=True`にする。

### `lightgbm.log_evaluation(...)`

**用途**: 学習中の評価スコアを一定間隔でログ出力するコールバック。

**シグネチャ**: `lightgbm.log_evaluation(period=1, show_stdv=True)`

**使用例**:
```python
lgb.train(
    {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0},
    dtrain, num_boost_round=30, valid_sets=[dvalid], valid_names=["valid"],
    callbacks=[lgb.log_evaluation(period=10)],
)
```
実行結果:
```
[10]	valid's binary_logloss: 0.259173
[20]	valid's binary_logloss: 0.140995
[30]	valid's binary_logloss: 0.0916942
```

**注意点・落とし穴**:
- `period=0`にするとログ出力が完全に無効になる(`period<=0`で無効という仕様)。
- scikit-learn APIの`fit(..., verbose=...)`引数は現在非推奨で、代わりに`callbacks=[lgb.log_evaluation(period=...)]`を使うことが推奨されている。

### `lightgbm.record_evaluation(...)`

**用途**: 学習中の評価スコアの推移を、渡した辞書に記録する(後でグラフ化・分析する際に使う)。

**シグネチャ**: `lightgbm.record_evaluation(eval_result)`

**使用例**:
```python
eval_result = {}
lgb.train(
    {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0},
    dtrain, num_boost_round=20, valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
    callbacks=[lgb.record_evaluation(eval_result)],
)
print(list(eval_result.keys()))
print(eval_result["valid"]["binary_logloss"][:5])
```
実行結果:
```
['train', 'valid']
[np.float64(0.6025745542066475), np.float64(0.5361229741482633), np.float64(0.48222440223796215), np.float64(0.43912127467341566), np.float64(0.399483667774539)]
```

**注意点・落とし穴**:
- `eval_result`はあらかじめ空の辞書として用意して渡す(戻り値ではなく、渡した辞書がその場で書き換えられる=副作用ベースのAPI)。

### `lightgbm.reset_parameter(...)`

**用途**: 学習の反復ごとに特定パラメータ(学習率など)を変化させるコールバック。

**シグネチャ**: `lightgbm.reset_parameter(**kwargs)` (各キーワード引数にリストまたは関数を渡す)

**使用例**:
```python
lrs = [0.1 if i < 10 else 0.01 for i in range(20)]
bst_rp = lgb.train(
    {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0},
    dtrain, num_boost_round=20,
    callbacks=[lgb.reset_parameter(learning_rate=lrs)],
)
print("num_trees:", bst_rp.num_trees())
```
実行結果:
```
num_trees: 20
```

**注意点・落とし穴**:
- リストで渡す場合、その長さは`num_boost_round`(実際の反復回数)以上である必要がある(短いとエラーまたは末尾の値が使い回されない可能性があるため、反復回数分きっちり用意するのが安全)。

---

## 7. 特徴量重要度・可視化

### `Booster.feature_importance(...)` / `LGBMClassifier.feature_importances_`

**用途**: 各特徴量の重要度を取得する。`'split'`(分割に使われた回数)と`'gain'`(分割によるゲインの合計)の2種類がある。

**シグネチャ**: `Booster.feature_importance(importance_type='split', iteration=None)`

**使用例**:
```python
bst_imp = lgb.train({"objective": "binary", "verbosity": -1, "seed": 0}, dtrain, num_boost_round=20)
print("split[:5]:", bst_imp.feature_importance(importance_type="split")[:5])
print("gain[:5]:", bst_imp.feature_importance(importance_type="gain")[:5].round(1))
```
実行結果:
```
split[:5]: [12 31  3  0  6]
gain[:5]: [ 5.5 34.1  1.2  0.   8.9]
```

**注意点・落とし穴**:
- デフォルトは`'split'`(使われた回数)。scikit-learn APIの`feature_importances_`属性のデフォルトも`importance_type='split'`(コンストラクタ引数で`'gain'`に切り替え可能)。`'split'`と`'gain'`では特徴量の順位が入れ替わることがあるため、目的に応じて使い分ける。

### `lightgbm.plot_importance(...)`

**用途**: 特徴量重要度を横棒グラフで可視化する(matplotlibが必要)。

**シグネチャ**: `lightgbm.plot_importance(booster, ax=None, height=0.2, xlim=None, ylim=None, title='Feature importance', xlabel='Feature importance', ylabel='Features', importance_type='auto', max_num_features=None, ignore_zero=True, figsize=None, dpi=None, grid=True, precision=3, **kwargs)`

**使用例**:
```python
import matplotlib
matplotlib.use("Agg")
clf_plot = lgb.LGBMClassifier(n_estimators=30, random_state=0, verbosity=-1).fit(X, y)
ax = lgb.plot_importance(clf_plot, max_num_features=5)
print(type(ax))
```
実行結果:
```
<class 'matplotlib.axes._axes.Axes'>
```

**注意点・落とし穴**:
- `booster`引数には`Booster`だけでなく学習済みの`LGBMClassifier`/`LGBMRegressor`(scikit-learn API)もそのまま渡せる(内部で`.booster_`を参照する)。未学習のEstimatorを渡すと`NotFittedError`になる。
- `importance_type='auto'`(デフォルト)は、渡されたオブジェクトが持つ`importance_type`設定(scikit-learn APIなら`importance_type`引数)に従う。

### `lightgbm.plot_metric(...)`

**用途**: 学習中に記録された評価指標の推移をグラフ化する。

**シグネチャ**: `lightgbm.plot_metric(booster, metric=None, dataset_names=None, ax=None, xlim=None, ylim=None, title='Metric during training', xlabel='Iterations', ylabel='@metric@', figsize=None, dpi=None, grid=True)`

**使用例**:
```python
eval_result2 = {}
bst_pm = lgb.train(
    {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0},
    dtrain, num_boost_round=20, valid_sets=[dvalid], valid_names=["valid"],
    callbacks=[lgb.record_evaluation(eval_result2)],
)
ax2 = lgb.plot_metric(eval_result2)
print(type(ax2))
```
実行結果:
```
<class 'matplotlib.axes._axes.Axes'>
```

**注意点・落とし穴**:
- `booster`引数には`Booster`オブジェクトではなく、`record_evaluation`で得た`eval_result`辞書(または`.evals_result_`)を渡す点が`plot_importance`と異なる。

### `lightgbm.plot_tree(...)` / `lightgbm.create_tree_digraph(...)`

**用途**: 学習済みモデルの1本の木を可視化する。`plot_tree`はmatplotlib画像、`create_tree_digraph`はGraphviz形式(`.render()`でファイル出力可能)。

**シグネチャ**: `lightgbm.plot_tree(booster, ax=None, tree_index=0, figsize=None, dpi=None, show_info=None, precision=3, orientation='horizontal', example_case=None, **kwargs)`

**使用例**:
```python
ax3 = lgb.plot_tree(bst_imp, tree_index=0)
print(type(ax3))
graph = lgb.create_tree_digraph(bst_imp, tree_index=0)
print(type(graph))
```
実行結果:
```
<class 'matplotlib.axes._axes.Axes'>
<class 'graphviz.graphs.Digraph'>
```

**注意点・落とし穴**:
- `create_tree_digraph`の利用には`graphviz`パッケージ(Pythonパッケージ本体に加えてGraphviz本体のインストール)が必要。未インストールだと呼び出し時にエラーになる。
- `tree_index`は0始まり。多クラス分類では1反復あたり複数の木(クラス数分)が作られるため、`num_tree_per_iteration`を考慮してインデックスを選ぶ必要がある。

### `predict(pred_contrib=True)`

**用途**: 各特徴量が予測値にどれだけ寄与したか(SHAP値相当)を計算する。

**シグネチャ**: `Booster.predict(data, pred_contrib=True, ...)`

**使用例**:
```python
contrib = bst_imp.predict(X[:2], pred_contrib=True)
print("contrib shape:", contrib.shape)
raw = bst_imp.predict(X[:2], raw_score=True)
print("寄与度の合計 == raw_score?", (contrib.sum(axis=1).round(4) == raw.round(4)).all())
```
実行結果:
```
contrib shape: (2, 31)
寄与度の合計 == raw_score? True
```

**注意点・落とし穴**:
- 出力の列数は「特徴量数+1」(最後の列がバイアス項/期待値)。全列を合計すると`raw_score=True`の生スコア(シグモイド適用前)と一致する。

### `predict(pred_leaf=True)`

**用途**: 各サンプルが各木でどの葉(leaf)に落ちたかのインデックスを取得する(特徴量エンジニアリングや木の可視化のデバッグに使う)。

**シグネチャ**: `Booster.predict(data, pred_leaf=True, ...)`

**使用例**:
```python
leaf_idx = bst_imp.predict(X[:3], pred_leaf=True)
print("shape:", leaf_idx.shape)
print("1件目:", leaf_idx[0][:5])
```
実行結果:
```
shape: (3, 20)
1件目: [7 2 2 3 8]
```

**注意点・落とし穴**:
- 返る配列の形状は`(サンプル数, 木の本数)`。多クラス分類では1反復あたり複数木があるため列数がさらに増える。

---

## 8. モデルの保存・読み込み

### `Booster.save_model(...)` / `lightgbm.Booster(model_file=...)`

**用途**: 学習済みモデルをテキスト形式のファイルに保存し、後から読み込む。

**シグネチャ**: `Booster.save_model(filename, num_iteration=None, start_iteration=0, importance_type='split')` / `lightgbm.Booster(params=None, train_set=None, model_file=None, model_str=None)`

**使用例**:
```python
bst_imp.save_model("model.txt")
bst_loaded = lgb.Booster(model_file="model.txt")
import numpy as np
print("予測が一致するか:", np.allclose(bst_imp.predict(X[:5]), bst_loaded.predict(X[:5])))
```
実行結果:
```
予測が一致するか: True
```

**注意点・落とし穴**:
- 保存形式はテキスト(可読)であり、`pickle`/`joblib`と異なりPythonのバージョンやクラス実装に依存しにくい(言語間の互換性目的でもこの形式が使われる)。

### `Booster.model_to_string(...)` / `model_str`引数

**用途**: モデルをファイルを経由せず、文字列として直接やり取りする(DB保存やAPI転送など)。

**シグネチャ**: `Booster.model_to_string(num_iteration=None, start_iteration=0, importance_type='split')`

**使用例**:
```python
model_str = bst_imp.model_to_string()
print(model_str[:30])
bst_from_str = lgb.Booster(model_str=model_str)
print("一致するか:", np.allclose(bst_imp.predict(X[:5]), bst_from_str.predict(X[:5])))
```
実行結果:
```
tree
version=v4
num_class=1
一致するか: True
```

**注意点・落とし穴**:
- `save_model`/`model_to_string`は内容として等価(前者はファイル、後者は文字列)。ファイルI/Oを避けたい環境(サーバーレスなど)では`model_str`のやり取りが便利。

### `Booster.dump_model(...)`

**用途**: モデルの内部構造(各木のノード情報など)をJSON互換の辞書として取得する。木構造を独自にパースしたい場合に使う。

**シグネチャ**: `Booster.dump_model(num_iteration=None, start_iteration=0, importance_type='split', object_hook=None)`

**使用例**:
```python
d = bst_imp.dump_model()
print(list(d.keys())[:6])
print("木の本数:", len(d["tree_info"]))
```
実行結果:
```
['name', 'version', 'num_class', 'num_tree_per_iteration', 'label_index', 'max_feature_idx']
木の本数: 20
```

**注意点・落とし穴**:
- 戻り値は素のPython辞書(`json.dumps()`でそのままシリアライズ可能)。`save_model`のテキスト形式より扱いやすいが、ファイルサイズは大きくなりやすい。

### `joblib.dump(...)` / `joblib.load(...)`(scikit-learn APIの保存)

**用途**: `LGBMClassifier`/`LGBMRegressor`(scikit-learn API)を、他のscikit-learn Estimatorと同じ方法でシリアライズする。

**シグネチャ**: `joblib.dump(value, filename)` / `joblib.load(filename)`

**使用例**:
```python
import joblib
clf_j = lgb.LGBMClassifier(n_estimators=10, random_state=0, verbosity=-1).fit(X, y)
joblib.dump(clf_j, "clf.joblib")
clf_j2 = joblib.load("clf.joblib")
print("一致するか:", np.allclose(clf_j.predict_proba(X[:5]), clf_j2.predict_proba(X[:5])))
```
実行結果:
```
一致するか: True
```

**注意点・落とし穴**:
- scikit-learn API(`LGBMClassifier`等)は`joblib`、ネイティブAPI(`Booster`)は`save_model`/`model_to_string`と、推奨される保存方法がAPIによって異なる。ネイティブの`Booster`だけを他言語(C++/Javaなど)に持ち込みたい場合はテキスト形式(`save_model`)を使う。

---

## 9. その他

### `free_raw_data`と追加学習(`init_model`)

**用途**: `Dataset`が元データを保持し続けるかどうかを制御する。`init_model`で既存モデルに追加学習(継続学習)する際に影響する。

**シグネチャ**: `lightgbm.Dataset(data, ..., free_raw_data=True)` / `lightgbm.train(params, train_set, ..., init_model=None)`

**使用例**:
```python
dtrain_keep = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
params0 = {"objective": "binary", "verbosity": -1, "seed": 0}
bst_a = lgb.train(params0, dtrain_keep, num_boost_round=10)
bst_b = lgb.train(params0, dtrain_keep, num_boost_round=10, init_model=bst_a)
print("bst_a本数:", bst_a.num_trees(), "/ bst_b本数(追加学習後):", bst_b.num_trees())
```
実行結果:
```
bst_a本数: 10 / bst_b本数(追加学習後): 20
```

**注意点・落とし穴**:
- `free_raw_data=True`(デフォルト)のまま`init_model`で追加学習しようとすると`LightGBMError: Cannot set predictor after freed raw data, set free_raw_data=False when construct Dataset to avoid this.`になる。追加学習・継続学習を予定している`Dataset`は必ず`free_raw_data=False`で作る。
- `init_model`に`Booster`だけでなく、保存済みモデルファイルのパス(文字列)も渡せる。

### `verbosity`パラメータ

**用途**: LightGBMが標準出力に出す`[LightGBM] [Info]`/`[Warning]`ログの量を制御する。

**シグネチャ**: パラメータ辞書のキー`verbosity`(または`LGBMClassifier(verbosity=1, ...)`。デフォルトは`1`)

**使用例**:
```python
import lightgbm as lgb2
from sklearn.datasets import load_breast_cancer
Xv, yv2 = load_breast_cancer(return_X_y=True)

print(">>> call1: verbosity=-1")
lgb2.train({"objective": "binary", "seed": 0, "verbosity": -1}, lgb2.Dataset(Xv, label=yv2), num_boost_round=1)
print(">>> call2: verbosity未指定")
lgb2.train({"objective": "binary", "seed": 0}, lgb2.Dataset(Xv, label=yv2), num_boost_round=1)
print(">>> call3: verbosity=1明示")
lgb2.train({"objective": "binary", "seed": 0, "verbosity": 1}, lgb2.Dataset(Xv, label=yv2), num_boost_round=1)
```
実行結果:
```
>>> call1: verbosity=-1
>>> call2: verbosity未指定
>>> call3: verbosity=1明示
[LightGBM] [Info] Number of positive: 357, number of negative: 212
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000790 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 5676
[LightGBM] [Info] Number of data points in the train set: 569, number of used features: 30
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.627417 -> initscore=0.521150
[LightGBM] [Info] Start training from score 0.521150
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
```

**注意点・落とし穴**:
- `verbosity`は「呼び出しごと」ではなく、プロセス内でグローバルに保持されるログレベルとして働く。一度`verbosity=-1`で学習すると、その後の呼び出しで`verbosity`を指定しなくても(ドキュメント上のデフォルト値`1`には戻らず)直前の設定が引き継がれて出力されないままになることを実機(lightgbm 4.7.0)で確認した。ログを復活させたい場合は`verbosity=1`のように明示的に指定し直す必要がある。
- 「なぜかInfoログが出ない/急に大量に出るようになった」という場合、コード上のその箇所だけでなく、同一プロセス内で以前に呼ばれた`verbosity`設定を疑うとよい。

### `Booster.num_trees()` / `num_feature()` / `current_iteration()`

**用途**: 学習済みモデルの木の本数・特徴量数・現在の反復回数を取得する、よく使う軽量メソッド群。

**シグネチャ**: `Booster.num_trees()` / `Booster.num_feature()` / `Booster.current_iteration()`

**使用例**:
```python
bst_meta = lgb.train({"objective": "binary", "seed": 0, "verbosity": -1}, lgb.Dataset(X, label=y), num_boost_round=2)
print("num_trees:", bst_meta.num_trees(), "num_feature:", bst_meta.num_feature(), "current_iteration:", bst_meta.current_iteration())
```
実行結果:
```
num_trees: 2 num_feature: 30 current_iteration: 2
```

**注意点・落とし穴**:
- 二値分類・回帰では通常`num_trees() == current_iteration()`(1反復=1本の木)だが、多クラス分類では1反復あたりクラス数分の木が作られるため`num_trees() == current_iteration() * num_class`になる。

---

## 応用・発展

lightgbm 4.7.0(CPUビルド)で検証済み。GPU限定機能(`device_type='gpu'`/`'cuda'`など)や`dask`連携(本検証環境には`dask`が未インストール)は対象外とし、実行して確認できたAPIのみを掲載する。

### 10. カスタム目的関数・評価関数

#### `objective`にカスタム関数を渡す(カスタム目的関数)

**用途**: 組み込みの`objective`(`binary`/`regression`など)では表現できない独自の損失関数を使う。勾配(grad)とヘシアン(hess)を自前で計算してブースティングに渡す。

**シグネチャ**: `lightgbm.train({"objective": <callable>, ...}, train_set, ...)`(callableは`callable(preds, train_data) -> (grad, hess)`の形。**バージョン固有の注意**: lightgbm 4.7.0の`lightgbm.train()`には`fobj`という独立引数は存在せず(`inspect.signature`で確認済み)、`params`辞書の`"objective"`キーに直接関数を渡す方式に統一されている)

**使用例**:
```python
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
dtrain = lgb.Dataset(Xtr, label=ytr)
dvalid = lgb.Dataset(Xte, label=yte, reference=dtrain)

def custom_logloss_obj(preds, train_data):
    labels = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))  # 標準ロジスティック損失のgrad/hessを手動実装
    grad = p - labels
    hess = p * (1.0 - p)
    return grad, hess

params = {"objective": custom_logloss_obj, "verbosity": -1, "seed": 0}
bst = lgb.train(params, dtrain, num_boost_round=30, valid_sets=[dvalid])
raw = bst.predict(Xte[:5], raw_score=True)
print("raw_score[:5]:", raw.round(4))
print("sigmoid(raw)[:5]:", (1 / (1 + np.exp(-raw))).round(4))
print("predict()そのまま[:5]:", bst.predict(Xte[:5]).round(4))
```
実行結果:
```
raw_score[:5]: [-3.9002  2.5865  3.4857  3.1242  3.1835]
sigmoid(raw)[:5]: [0.0198 0.93   0.9703 0.9579 0.9602]
predict()そのまま[:5]: [-3.9002  2.5865  3.4857  3.1242  3.1835]
```

**注意点・落とし穴**:
- カスタム目的関数を使うと、`predict()`は`raw_score=True`を指定しなくても常に変換前の生スコア(この場合はシグモイド適用前のマージン)を返す(実機で`raw_score=True`との出力一致を確認)。組み込みの`"binary"`objectiveのように自動でシグモイドが適用された確率を返してくれるわけではないため、確率が欲しい場合は自分でシグモイドなどの変換をかける必要がある。
- `train_data.get_label()`で正解ラベルを取得できるのは、`Dataset`が(`lgb.train`への引き渡しなどで)構築済みの場合のみ。

#### `feval`(カスタム評価関数)

**用途**: 組み込みの`metric`にない独自の評価指標を学習中に計算・表示する。リストで複数渡せば同時に複数指標を出せる。

**シグネチャ**: `lightgbm.train(params, train_set, ..., feval=None)`(`feval`は`callable(preds, eval_data) -> (metric_name, metric_value, is_higher_better)`、またはそのタプルを複数返す/複数callableのリストで複数指標に対応)

**使用例**:
```python
from sklearn.metrics import roc_auc_score

def custom_auc(preds, eval_data):
    return "custom_auc", roc_auc_score(eval_data.get_label(), preds), True

def custom_error_at_03(preds, eval_data):
    pred_label = (preds > 0.3).astype(int)
    return "error@0.3", (pred_label != eval_data.get_label()).mean(), False

params2 = {"objective": "binary", "metric": "None", "verbosity": -1, "seed": 0}
lgb.train(
    params2, dtrain, num_boost_round=20, valid_sets=[dvalid], valid_names=["valid"],
    feval=[custom_auc, custom_error_at_03],
    callbacks=[lgb.log_evaluation(period=10)],
)
```
実行結果:
```
[10]	valid's custom_auc: 0.995237	valid's error@0.3: 0.0964912
[20]	valid's custom_auc: 0.99746	valid's error@0.3: 0.0438596
```

**注意点・落とし穴**:
- 組み込みの`"binary"`objectiveを使っている場合、`feval`に渡される`preds`は(カスタムobjectiveの場合と異なり)**すでに確率に変換済み**であることを実機で確認した(`preds.min()`/`preds.max()`が0〜1の範囲に収まる)。カスタムobjectiveと組み合わせたときは生マージンのままなので、`feval`内でシグモイドをかけるべきかどうかは「objectiveが組み込みかカスタムか」で変わる点に注意。
- 独自の`feval`だけを表示したい(デフォルトmetricのログを消したい)場合は`params`の`"metric"`を文字列`"None"`にする必要がある(Python の`None`ではなく文字列の`"None"`)。

---

### 11. 木構造の高度な制御

#### `linear_tree=True`(線形木)

**用途**: 各葉のリーフ値を定数ではなく、葉に落ちたサンプルに対する線形回帰式にする。特徴量と目的変数がほぼ線形関係にあるデータで、少ない木の本数でも滑らかな予測ができる。

**シグネチャ**: `lightgbm.Dataset(data, label, ..., params={"linear_tree": True})`(パラメータ辞書のキー。`lightgbm.train`側の`params`にも同じキーを渡す)

**使用例**:
```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

Xr, yr = make_regression(n_samples=2000, n_features=3, noise=5.0, random_state=0)
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.2, random_state=0)

# linear_treeはDataset構築時のパラメータ(Dataset(..., params=...))として渡す必要がある
dtrain_lt = lgb.Dataset(Xrtr, label=yrtr, params={"linear_tree": True})
bst_lt = lgb.train({"objective": "regression", "linear_tree": True, "num_leaves": 7,
                     "min_data_in_leaf": 50, "verbosity": -1, "seed": 0}, dtrain_lt, num_boost_round=10)

dtrain_plain = lgb.Dataset(Xrtr, label=yrtr)
bst_plain = lgb.train({"objective": "regression", "num_leaves": 7,
                        "min_data_in_leaf": 50, "verbosity": -1, "seed": 0}, dtrain_plain, num_boost_round=10)

print("MSE(linear_tree=True):", round(mean_squared_error(yrte, bst_lt.predict(Xrte)), 3))
print("MSE(通常の木):", round(mean_squared_error(yrte, bst_plain.predict(Xrte)), 3))

def first_leaf(node):
    return node if "leaf_value" in node else first_leaf(node["left_child"])

leaf = first_leaf(bst_lt.dump_model()["tree_info"][1]["tree_structure"])
print("2本目の木の葉: leaf_coeff:", [round(c, 3) for c in leaf["leaf_coeff"]], "leaf_features:", leaf["leaf_features"])
```
実行結果:
```
MSE(linear_tree=True): 1838.619
MSE(通常の木): 3810.548
2本目の木の葉: leaf_coeff: [8.211, 5.714] leaf_features: [0, 1]
```

**注意点・落とし穴**:
- `linear_tree`は`lightgbm.train`の`params`だけに書いても効かない。`Dataset`はビン分割などを`construct()`時に確定させるため、**`Dataset`を作る時点で`params={"linear_tree": True}`を渡す必要がある**。実機で、Dataset側に渡さずtrain側のparamsだけに`linear_tree: True`を指定したところ、通常木と全く同じMSE・同じ木構造(`dump_model()`の葉に`leaf_coeff`が入らない)になることを確認した。
- 1本目の木(`tree_info[0]`)の葉は`leaf_coeff`が空リストになる(初期スコアからの最初の分割のため線形項がまだ乗らない)。線形係数を確認したい場合は2本目以降の木を見る必要がある。

#### `interaction_constraints`

**用途**: 特定の特徴量グループ同士が同じ木の中で組み合わさって分割されるのを禁止する。解釈可能性の確保や、ドメイン知識上関係ないはずの特徴量同士の疑似交互作用を防ぐ目的で使う。

**シグネチャ**: パラメータ辞書のキー`interaction_constraints=None`(特徴量indexのリストのリスト。各リストが「同じ木内で許容される特徴量の集合」を表す)

**使用例**:
```python
params_ic = {
    "objective": "binary", "verbosity": -1, "seed": 0,
    "interaction_constraints": [list(range(0, 15)), list(range(15, 30))],
}
bst_ic = lgb.train(params_ic, dtrain, num_boost_round=30)

def used_features(node, feats):
    if "split_feature" in node:
        feats.add(node["split_feature"])
        used_features(node["left_child"], feats)
        used_features(node["right_child"], feats)

group_a, group_b = set(range(0, 15)), set(range(15, 30))
violations = 0
for t in bst_ic.dump_model()["tree_info"]:
    feats = set()
    used_features(t["tree_structure"], feats)
    if feats & group_a and feats & group_b:
        violations += 1
print("両グループの特徴量を同一木内で混在させた木の本数:", violations, "/", len(bst_ic.dump_model()["tree_info"]))
```
実行結果:
```
両グループの特徴量を同一木内で混在させた木の本数: 0 / 30
```

**注意点・落とし穴**:
- グループ分けは「特徴量index(0始まり)のリストのリスト」で指定する。指定に漏れた特徴量index(0〜29のどれかを書き漏らした場合など)がどう扱われるかはドキュメント上明記が薄いため、全特徴量をいずれかのグループに明示的に含めるのが安全。

#### `extra_trees=True`

**用途**: Extremely Randomized Trees(ERT)風に、各ノードの分割閾値を最適値ではなくランダムに選んでから最良のものを選ぶことで、木ごとの多様性を高め過学習を抑える。

**シグネチャ**: パラメータ辞書のキー`extra_trees=False`

**使用例**:
```python
params_et = {"objective": "binary", "extra_trees": True, "verbosity": -1, "seed": 0}
params_normal = {"objective": "binary", "verbosity": -1, "seed": 0}
bst_et = lgb.train(params_et, dtrain, num_boost_round=30)
bst_normal = lgb.train(params_normal, dtrain, num_boost_round=30)

d_et = bst_et.dump_model()["tree_info"][0]["tree_structure"]
d_normal = bst_normal.dump_model()["tree_info"][0]["tree_structure"]
print("1本目の最初の分割feature: extra_trees=%d / 通常=%d" % (d_et["split_feature"], d_normal["split_feature"]))
print("1本目の最初の分割threshold: extra_trees=%.5f / 通常=%.5f" % (d_et["threshold"], d_normal["threshold"]))
```
実行結果:
```
1本目の最初の分割feature: extra_trees=22 / 通常=27
1本目の最初の分割threshold: extra_trees=103.25000 / 通常=0.14545
```

**注意点・落とし穴**:
- `extra_trees=True`にすると、同じ`seed`でも通常のgbdtとは全く異なる木構造(使う特徴量・閾値とも)になる(実機で1本目の分割から違うことを確認)。この検証ではAUCはわずかに悪化した(0.9984→0.9927)ため、必ず性能が上がるわけではなく、データに応じて試す必要がある。

#### `path_smooth`

**用途**: 葉の予測値を、その葉単独の値ではなく祖先ノードの値ともブレンドして滑らかにする。サンプル数が少ない葉の予測が極端になりすぎるのを防ぐ正則化。

**シグネチャ**: パラメータ辞書のキー`path_smooth=0.0`(値が大きいほど祖先ノード側の値に近づく方向で平滑化される)

**使用例**:
```python
# min_data_in_leafを変えるため、既存のdtrainとは別のDatasetを作り直す
dtrain_ps1 = lgb.Dataset(Xtr, label=ytr)
dtrain_ps2 = lgb.Dataset(Xtr, label=ytr)
params_ps = {"objective": "binary", "path_smooth": 1.0, "min_data_in_leaf": 3, "verbosity": -1, "seed": 0}
params_no_ps = {"objective": "binary", "min_data_in_leaf": 3, "verbosity": -1, "seed": 0}
bst_ps = lgb.train(params_ps, dtrain_ps1, num_boost_round=30)
bst_no_ps = lgb.train(params_no_ps, dtrain_ps2, num_boost_round=30)
from sklearn.metrics import roc_auc_score
print("AUC(path_smooth=1.0):", round(roc_auc_score(yte, bst_ps.predict(Xte)), 4))
print("AUC(path_smoothなし):", round(roc_auc_score(yte, bst_no_ps.predict(Xte)), 4))
```
実行結果:
```
AUC(path_smooth=1.0): 0.9978
AUC(path_smoothなし): 0.9968
```

**注意点・落とし穴**:
- `path_smooth`を効かせるには`min_data_in_leaf >= 2`である必要があるとされる(公式ドキュメント記載。本検証では`min_data_in_leaf=3`で意図通り動作し、AUCがわずかに変化することを確認した)。
- 一度`construct()`された`Dataset`(=学習済みの`min_data_in_leaf`でビン分割・事前フィルタ済み)に対して、別の`min_data_in_leaf`で再度`lgb.train`しようとすると`LightGBMError: Reducing 'min_data_in_leaf' with 'feature_pre_filter=true' may cause unexpected behaviour...`になることを実機で確認した。`min_data_in_leaf`を変えて比較したい場合は、`Dataset`を作り直すか`feature_pre_filter=False`を指定する必要がある。

#### `monotone_constraints_method`

**用途**: `monotone_constraints`(単調性制約)をどう強制するかのアルゴリズムを切り替える。`'basic'`は高速だが制約が過剰に保守的になりやすく、`'intermediate'`はより緩やかな制約で精度を出しやすい代わりに学習が遅くなる。

**シグネチャ**: パラメータ辞書のキー`monotone_constraints_method='basic'`(`'basic'`または`'intermediate'`)

**使用例**:
```python
import numpy as np
rng = np.random.RandomState(0)
n = 2000
x1 = rng.uniform(0, 10, size=n)
x2 = rng.uniform(0, 10, size=n)
ym = x1 * 2 + np.sin(x2) * 3 + rng.normal(scale=0.5, size=n)
Xm = np.column_stack([x1, x2])
dtrain_m = lgb.Dataset(Xm, label=ym)

xs = np.column_stack([np.linspace(0, 10, 15), np.full(15, 5.0)])
preds = {}
for method in ["basic", "intermediate"]:
    params_m = {"objective": "regression", "monotone_constraints": [1, 0],
                "monotone_constraints_method": method, "num_leaves": 31,
                "verbosity": -1, "seed": 0}
    bst_m = lgb.train(params_m, dtrain_m, num_boost_round=100)
    preds[method] = bst_m.predict(xs)
    print(method, "単調性が保たれているか:", bool(np.all(np.diff(preds[method]) >= -1e-9)))

print("basicとintermediateの予測が完全一致するか:", np.allclose(preds["basic"], preds["intermediate"]))
```
実行結果:
```
basic 単調性が保たれているか: True
intermediate 単調性が保たれているか: True
basicとintermediateの予測が完全一致するか: False
```

**注意点・落とし穴**:
- `x1`(制約あり)1変数だけを使った単純な例では`'basic'`と`'intermediate'`の予測が完全一致してしまい違いが確認できなかった。制約なしの特徴量(`x2`)を混ぜて木の分岐が複雑になる状況で初めて、両者の予測値に差が出ることを実機で確認した。単調性そのものはどちらの方式でも常に満たされる。

---

### 12. ランキング学習(LambdaRank)

#### `objective='lambdarank'` + `Dataset.set_group(...)`

**用途**: 検索結果の並び替えのように「クエリごとに複数件をまとめてランク付けする」タスク向けのランキング学習。クエリ(グループ)ごとの件数を`set_group`で教える。

**シグネチャ**: `lightgbm.Dataset.set_group(self, group)`(各要素がクエリごとの件数のリスト/配列。合計がDatasetの行数と一致する必要がある)

**使用例**:
```python
import numpy as np
rng2 = np.random.RandomState(0)
n_queries, docs_per_query = 20, 10
Xrank = rng2.normal(size=(n_queries * docs_per_query, 5))
relevance = np.clip((Xrank[:, 0] * 1.5 + rng2.normal(scale=1.0, size=len(Xrank))).round().astype(int) + 2, 0, 4)
group = [docs_per_query] * n_queries

dtrain_r = lgb.Dataset(Xrank, label=relevance)
dtrain_r.set_group(group)
print("get_group():", dtrain_r.get_group())

params_rank = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [3], "verbosity": -1, "seed": 0}
bst_rank = lgb.train(params_rank, dtrain_r, num_boost_round=30)
print("1クエリ目のスコア:", bst_rank.predict(Xrank[:docs_per_query]).round(3))
print("1クエリ目の正解relevance:", relevance[:docs_per_query])
```
実行結果:
```
get_group(): [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
1クエリ目のスコア: [ 2.102 -2.654 -1.625  0.829 -3.229 -3.031 -0.574 -1.703 -2.511 -1.405]
1クエリ目の正解relevance: [4 1 2 3 0 0 3 0 0 2]
```

**注意点・落とし穴**:
- `group`の合計(この例では`10*20=200`)は`Dataset`の行数と一致していなければならない。一致しないとconstruct時にエラーになる。
- 出力のスコアはランキング用の相対的なスコアであり、確率(0〜1)ではない。実際に高いrelevance(4)のサンプルほどスコアが高くなる傾向(1件目のscore=2.102がrelevance=4に対応)は出ているが、厳密な順位一致までは保証されない。

#### `lightgbm.LGBMRanker(...)`

**用途**: `lambdarank`をscikit-learn風のEstimator APIで使う。`fit(X, y, group=...)`のように、グループ情報を引数として渡せる。

**シグネチャ**: `lightgbm.LGBMRanker(*, boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, ..., importance_type='split', **kwargs)`(コンストラクタ引数は`LGBMClassifier`/`LGBMRegressor`と同一構成。ランキング固有の`group`は`fit(X, y, group=...)`側で渡す)

**使用例**:
```python
ranker = lgb.LGBMRanker(n_estimators=30, random_state=0, verbosity=-1)
ranker.fit(Xrank, relevance, group=group)
print("LGBMRankerでの1クエリ目のスコア:", ranker.predict(Xrank[:docs_per_query]).round(3))
```
実行結果:
```
LGBMRankerでの1クエリ目のスコア: [ 2.102 -2.654 -1.625  0.829 -3.229 -3.031 -0.574 -1.703 -2.511 -1.405]
```

**注意点・落とし穴**:
- 同じデータ・同じデフォルトパラメータで学習した場合、ネイティブAPI(`lgb.train`+`objective='lambdarank'`)と`LGBMRanker`のスコアは完全に一致した(実機で確認)。内部的には同じ処理を呼んでいるとみてよい。
- `LGBMClassifier`/`LGBMRegressor`と違い、`fit`に`group`(または`eval_group`)を渡し忘れると全サンプルが1つのクエリとして扱われてしまう点に注意。

#### `label_gain`

**用途**: NDCGなどの評価指標で、relevanceラベルごとに与える「利得(gain)」をカスタマイズする。デフォルトは指数的(`2^label - 1`)だが、ラベル間の差を線形に扱いたい場合などに変更する。

**シグネチャ**: パラメータ辞書のキー`label_gain=None`(未指定時は`0, 1, 3, 7, 15, 31, ...`= `2^i - 1`。ラベル数分の長さのリストを渡す)

**使用例**:
```python
params_default = {"objective": "lambdarank", "metric": "ndcg", "verbosity": -1, "seed": 0}
params_linear = {"objective": "lambdarank", "metric": "ndcg", "label_gain": [0, 1, 2, 3, 4], "verbosity": -1, "seed": 0}
bst_default = lgb.train(params_default, dtrain_r, num_boost_round=30)
bst_linear = lgb.train(params_linear, dtrain_r, num_boost_round=30)
print("label_gain未指定(既定=2^i-1):", bst_default.predict(Xrank[:5]).round(3))
print("label_gain=[0,1,2,3,4](線形):", bst_linear.predict(Xrank[:5]).round(3))
```
実行結果:
```
label_gain未指定(既定=2^i-1): [ 2.102 -2.654 -1.625  0.829 -3.229]
label_gain=[0,1,2,3,4](線形): [ 2.164 -2.438 -1.247  1.128 -3.049]
```

**注意点・落とし穴**:
- `label_gain`を`[0, 1, 3, 7, 15]`(=`2^i - 1`、relevanceが0〜4の場合のデフォルトと同じ値)に明示的に指定して学習したところ、`label_gain`未指定時と完全に同じ予測になることを確認した。これはデフォルト値の仕様(`2^i - 1`)が実機の挙動と一致することの裏付けでもある。

---

### 13. 交差検証の応用

#### `lightgbm.cv(..., folds=...)`(カスタムfoldオブジェクト)

**用途**: `nfold`/`stratified`による自動分割ではなく、グループ構造(同一患者・同一顧客のデータが複数行に分かれている、など)を考慮した分割をscikit-learnの`Splitter`や自作のfoldリストで指定する。

**シグネチャ**: `lightgbm.cv(params, train_set, ..., folds=None, ...)`(`folds`は`(train_idx, valid_idx)`のタプルを要素とするイテラブル、またはscikit-learnの`BaseCrossValidator`。`inspect.signature`で型ヒントも`Union[Iterable[Tuple[ndarray, ndarray]], BaseCrossValidator, None]`であることを確認)

**使用例**:
```python
from sklearn.model_selection import GroupKFold

rng3 = np.random.RandomState(0)
groups = rng3.randint(0, 30, size=len(y))  # 疑似的な「グループ」(例: 患者ID)
dtrain_full = lgb.Dataset(X, label=y)
params_cv = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0}

custom_folds = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print("1つ目のfold: train数=%d, valid数=%d" % (len(custom_folds[0][0]), len(custom_folds[0][1])))
result = lgb.cv(params_cv, dtrain_full, num_boost_round=30, folds=custom_folds, seed=0)
print("最終mean logloss:", result["valid binary_logloss-mean"][-1])
```
実行結果:
```
1つ目のfold: train数=452, valid数=117
最終mean logloss: 0.11906340988793243
```

**注意点・落とし穴**:
- `folds`を指定すると、`nfold`/`stratified`/`shuffle`は無視される(実際の分割は渡した`folds`がすべてを決める)。
- scikit-learnの`Splitter`インスタンスの`.split(X, y, groups=...)`が返すジェネレータをそのまま渡しても、あらかじめ`list(...)`化しても同じ結果になることを確認した。

#### `lightgbm.cv(..., fpreproc=...)`

**用途**: fold(分割)ごとに学習データを使って動的にパラメータや前処理を変えたいときに使うコールバック(例: fold内の不均衡度に応じて`scale_pos_weight`を変える)。

**シグネチャ**: `lightgbm.cv(params, train_set, ..., fpreproc=None, ...)`(`fpreproc`は`callable(dtrain, dvalid, params) -> (dtrain, dvalid, params)`)

**使用例**:
```python
def fpreproc(dtrain_fold, dvalid_fold, params):
    dtrain_fold.construct()  # get_label()の前に明示的にconstructが必要
    labels = dtrain_fold.get_label()
    pos, neg = labels.sum(), len(labels) - labels.sum()
    new_params = dict(params)
    new_params["scale_pos_weight"] = neg / pos
    return dtrain_fold, dvalid_fold, new_params

result_fp = lgb.cv(params_cv, dtrain_full, num_boost_round=20, nfold=5, seed=0, fpreproc=fpreproc)
result_plain = lgb.cv(params_cv, dtrain_full, num_boost_round=20, nfold=5, seed=0)
print("fpreproc適用後の最終mean logloss:", result_fp["valid binary_logloss-mean"][-1])
print("fpreprocなしの最終mean logloss:", result_plain["valid binary_logloss-mean"][-1])
```
実行結果:
```
fpreproc適用後の最終mean logloss: 0.16093020896246604
fpreprocなしの最終mean logloss: 0.1657923009967724
```

**注意点・落とし穴**:
- `fpreproc`に渡される`dtrain`はまだ`construct()`されていない状態であり、`get_label()`をそのまま呼ぶと`Exception: Cannot get label before construct Dataset`になることを実機で確認した。fold内のラベルを参照する処理を書く場合は、先に`dtrain.construct()`を呼ぶ必要がある。

---

### 14. モデル内部構造の解析・再学習

#### `Booster.trees_to_dataframe()`

**用途**: 学習済みモデルの全ノード(内部ノード・葉ノード)を1行1ノードのpandas DataFrameとして取得する。`dump_model()`より扱いやすい表形式で木構造を分析したいときに使う。

**シグネチャ**: `Booster.trees_to_dataframe(self) -> pandas.DataFrame`

**使用例**:
```python
bst_meta2 = lgb.train({"objective": "binary", "verbosity": -1, "seed": 0}, dtrain, num_boost_round=30)
df_trees = bst_meta2.trees_to_dataframe()
print(df_trees.columns.tolist())
print(df_trees.shape)
print(df_trees.iloc[0][["tree_index", "split_feature", "threshold", "value", "count"]])
```
実行結果:
```
['tree_index', 'node_depth', 'node_index', 'left_child', 'right_child', 'parent_index', 'split_feature', 'split_gain', 'threshold', 'decision_type', 'missing_direction', 'missing_type', 'value', 'weight', 'count']
(1186, 15)
tree_index                0
split_feature      Column_27
threshold            0.14545
value               0.563935
count                    455
Name: 0, dtype: object
```

**注意点・落とし穴**:
- 葉ノードの行は`split_feature`/`threshold`などがNaNになる。「葉だけ」「内部ノードだけ」を抽出したい場合は`split_feature`の欠損有無でフィルタするとよい。
- `value`列は、内部ノードでは分割前の(その部分木の)平均的な出力、葉ノードでは実際のleaf_valueに対応する。

#### `Booster.refit(...)`

**用途**: 学習済みモデルの木構造(分割条件)はそのままに、新しいデータで葉の値だけを再計算する。オンライン学習・ドメイン適応など「構造は信頼できるが最新データで出力を較正したい」場面向け。

**シグネチャ**: `Booster.refit(data, label, decay_rate=0.9, reference=None, weight=None, group=None, init_score=None, feature_name='auto', categorical_feature='auto', dataset_params=None, free_raw_data=True, position=None)`

**使用例**:
```python
bst_orig = lgb.train({"objective": "binary", "verbosity": -1, "seed": 0}, dtrain, num_boost_round=30)
bst_refit = bst_orig.refit(Xte, yte)

from sklearn.metrics import roc_auc_score
print("refit前のtest AUC:", round(roc_auc_score(yte, bst_orig.predict(Xte)), 4))
print("refit後のtest AUC:", round(roc_auc_score(yte, bst_refit.predict(Xte)), 4))
print("木の本数は変わらないか:", bst_orig.num_trees() == bst_refit.num_trees())

df_before = bst_orig.trees_to_dataframe()
df_after = bst_refit.trees_to_dataframe()
same_split = (df_before["split_feature"].fillna("leaf") == df_after["split_feature"].fillna("leaf")).all()
diff_leaf = (df_before["value"] != df_after["value"]).any()
print("分割構造(split_feature)が同一か:", same_split)
print("leafの値(value)に差があるか:", diff_leaf)
```
実行結果:
```
refit前のtest AUC: 0.9984
refit後のtest AUC: 0.9984
木の本数は変わらないか: True
分割構造(split_feature)が同一か: True
leafの値(value)に差があるか: True
```

**注意点・落とし穴**:
- `refit`は木の分割(どの特徴量のどの閾値で分けるか)は一切変えず、葉の値(`value`)だけを新データに合わせて更新する。分割自体を学習し直したい場合は`init_model`付きの通常の`train`(追加学習)を使う必要がある。
- 戻り値は新しい`Booster`インスタンス(`bst_orig`自体は書き換わらない)。

#### `feature_penalty`

**用途**: 特定の特徴量が分割に使われる際のゲインにペナルティ(重み)をかける。`0`にするとその特徴量は実質的に分割候補から除外される。

**シグネチャ**: パラメータ辞書のキー`feature_penalty=None`(各特徴量に対応する`0`〜`1`の重みのリスト。`1`=ペナルティなし、`0`=その特徴量を使用禁止)

**使用例**:
```python
bst_plain2 = lgb.train({"objective": "binary", "verbosity": -1, "seed": 0}, dtrain, num_boost_round=20)
imp_plain = bst_plain2.feature_importance("gain")
top_feat = int(np.argmax(imp_plain))
print("ペナルティなしで最重要な特徴量index:", top_feat, "gain:", round(imp_plain[top_feat], 1))

penalty = [1.0] * 30
penalty[top_feat] = 0.0  # 最重要だった特徴量の使用を実質禁止
bst_fp = lgb.train({"objective": "binary", "feature_penalty": penalty, "verbosity": -1, "seed": 0}, dtrain, num_boost_round=20)
imp_fp = bst_fp.feature_importance("gain")
print("ペナルティ適用後、その特徴量のgain:", round(imp_fp[top_feat], 1))
print("AUC(ペナルティなし):", round(roc_auc_score(yte, bst_plain2.predict(Xte)), 4))
print("AUC(ペナルティあり):", round(roc_auc_score(yte, bst_fp.predict(Xte)), 4))
```
実行結果:
```
ペナルティなしで最重要な特徴量index: 27 gain: 980.0
ペナルティ適用後、その特徴量のgain: 0.0
AUC(ペナルティなし): 0.9975
AUC(ペナルティあり): 0.9975
```

**注意点・落とし穴**:
- `feature_penalty=0.0`は「完全に使用禁止」であることを実機で確認した(該当特徴量のgain importanceが正確に`0.0`になった)。`monotone_constraints`の`0`(制約なし)とは意味が異なるので混同しないこと。
- 今回のデータでは、最重要特徴量(index 27)を禁止してもAUCはほぼ変わらなかった(0.9975→0.9975)。乳がんデータセットは特徴量間の相関が強く、代替特徴量で十分カバーできたためと考えられる。
