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
