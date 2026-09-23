# catboost 逆引き辞書

catboost 1.2.10 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(catboost 1.2.10)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。シグネチャは基本的に `inspect.signature()` で取得したもの(Cython実装のクラスも本バージョンでは問題なく取得できた)。

## 目次

1. [Pool・学習基礎](#1-pool学習基礎)
2. [カテゴリ変数・テキスト特徴量のネイティブ対応](#2-カテゴリ変数テキスト特徴量のネイティブ対応)
3. [交差検証・ハイパーパラメータ探索](#3-交差検証ハイパーパラメータ探索)
4. [Ordered Boosting・過学習対策](#4-ordered-boosting過学習対策)
5. [クラス不均衡への対応](#5-クラス不均衡への対応)
6. [特徴量重要度・特徴量選択](#6-特徴量重要度特徴量選択)
7. [モデルの保存・読み込み](#7-モデルの保存読み込み)
8. [評価指標・可視化](#8-評価指標可視化)
9. [ハイパーパラメータ・その他ユーティリティ](#9-ハイパーパラメータその他ユーティリティ)

## 応用・発展

10. [単調性制約](#10-単調性制約)
11. [埋め込み特徴量](#11-埋め込み特徴量)
12. [SHAP値の応用・モデル解釈の深掘り](#12-shap値の応用モデル解釈の深掘り)
13. [モデルの評価・比較](#13-モデルの評価比較)
14. [アンサンブル・スタッキングとの組み合わせ](#14-アンサンブルスタッキングとの組み合わせ)

---

## 1. Pool・学習基礎

### `CatBoostClassifier(...)`

**用途**: 勾配ブースティング(GBDT)による分類器。カテゴリ変数をそのまま(one-hotやラベルエンコード不要で)扱えるのが最大の特徴。

**シグネチャ**: `catboost.CatBoostClassifier(self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, model_size_reg=None, rsm=None, loss_function=None, border_count=None, feature_border_type=None, per_float_feature_quantization=None, input_borders=None, output_borders=None, fold_permutation_block=None, od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None, leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None, use_best_model=None, best_model_min_trees=None, verbose=None, silent=None, logging_level=None, metric_period=None, ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None, allow_const_label=None, target_border=None, classes_count=None, class_weights=None, auto_class_weights=None, class_names=None, one_hot_max_size=None, random_strength=None, random_score_type=None, name=None, ignored_features=None, train_dir=None, custom_loss=None, custom_metric=None, eval_metric=None, bagging_temperature=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None, allow_writing_files=None, final_ctr_computation_mode=None, approx_on_full_history=None, boosting_type=None, simple_ctr=None, combinations_ctr=None, per_feature_ctr=None, ctr_description=None, ctr_target_border_count=None, task_type=None, device_config=None, devices=None, bootstrap_type=None, subsample=None, mvs_reg=None, sampling_unit=None, sampling_frequency=None, dev_score_calc_obj_block_size=None, dev_efb_max_buckets=None, sparse_features_conflict_fraction=None, max_depth=None, n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, random_state=None, reg_lambda=None, objective=None, eta=None, max_bin=None, scale_pos_weight=None, gpu_cat_features_storage=None, data_partition=None, metadata=None, early_stopping_rounds=None, cat_features=None, grow_policy=None, min_data_in_leaf=None, min_child_samples=None, max_leaves=None, num_leaves=None, score_function=None, leaf_estimation_backtracking=None, ctr_history_unit=None, monotone_constraints=None, feature_weights=None, penalties_coefficient=None, first_feature_use_penalties=None, per_object_feature_penalties=None, model_shrink_rate=None, model_shrink_mode=None, langevin=None, diffusion_temperature=None, posterior_sampling=None, boost_from_average=None, text_features=None, tokenizers=None, dictionaries=None, feature_calcers=None, text_processing=None, embedding_features=None, callback=None, eval_fraction=None, fixed_binary_splits=None)`

**使用例**:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

clf = CatBoostClassifier(iterations=50, depth=4, learning_rate=0.1, verbose=False, random_seed=0)
clf.fit(Xtr, ytr)
print("score:", clf.score(Xte, yte))
```
実行結果:
```
score: 0.9466666666666667
```

**注意点・落とし穴**:
- ほぼ全引数のデフォルトが `None` になっている。`None` の場合は内部で自動決定される値が使われる(例: `depth` は実質6、`iterations` は実質1000、`loss_function` は`y`の中身から自動判定)。`clf.get_all_params()`(学習後)で実際に使われた値を確認できる。
- `verbose=False` を指定しないと1イテレーションごとに学習ログが標準出力に大量に出る。
- `random_state`/`random_seed`、`n_estimators`/`num_boost_round`/`num_trees`/`iterations`、`eta`/`learning_rate`、`max_depth`/`depth` のように、xgboost/lightgbm互換のためのエイリアス引数が多数存在する。

### `CatBoostRegressor(...)`

**用途**: 勾配ブースティングによる回帰。APIは`CatBoostClassifier`とほぼ共通。

**シグネチャ**: `catboost.CatBoostRegressor(self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, model_size_reg=None, rsm=None, loss_function='RMSE', border_count=None, feature_border_type=None, per_float_feature_quantization=None, input_borders=None, output_borders=None, fold_permutation_block=None, od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None, leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None, use_best_model=None, best_model_min_trees=None, verbose=None, silent=None, logging_level=None, metric_period=None, ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None, allow_const_label=None, target_border=None, one_hot_max_size=None, random_strength=None, random_score_type=None, name=None, ignored_features=None, train_dir=None, custom_metric=None, eval_metric=None, bagging_temperature=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None, allow_writing_files=None, final_ctr_computation_mode=None, approx_on_full_history=None, boosting_type=None, simple_ctr=None, combinations_ctr=None, per_feature_ctr=None, ctr_description=None, ctr_target_border_count=None, task_type=None, device_config=None, devices=None, bootstrap_type=None, subsample=None, mvs_reg=None, sampling_frequency=None, sampling_unit=None, dev_score_calc_obj_block_size=None, dev_efb_max_buckets=None, sparse_features_conflict_fraction=None, max_depth=None, n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, random_state=None, reg_lambda=None, objective=None, eta=None, max_bin=None, gpu_cat_features_storage=None, data_partition=None, metadata=None, early_stopping_rounds=None, cat_features=None, grow_policy=None, min_data_in_leaf=None, min_child_samples=None, max_leaves=None, num_leaves=None, score_function=None, leaf_estimation_backtracking=None, ctr_history_unit=None, monotone_constraints=None, feature_weights=None, penalties_coefficient=None, first_feature_use_penalties=None, per_object_feature_penalties=None, model_shrink_rate=None, model_shrink_mode=None, langevin=None, diffusion_temperature=None, posterior_sampling=None, boost_from_average=None, text_features=None, tokenizers=None, dictionaries=None, feature_calcers=None, text_processing=None, embedding_features=None, eval_fraction=None, fixed_binary_splits=None)`

**使用例**:
```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

Xr, yr = make_regression(n_samples=300, n_features=5, noise=5.0, random_state=0)
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.25, random_state=0)

reg = CatBoostRegressor(iterations=50, depth=4, learning_rate=0.1, verbose=False, random_seed=0)
reg.fit(Xrtr, yrtr)
print("predict[:5]:", reg.predict(Xrte[:5]).round(2))
print("R2:", reg.score(Xrte, yrte))
```
実行結果:
```
predict[:5]: [-64.94 113.12  39.39   8.71 163.03]
R2: 0.9450697855733522
```

**注意点・落とし穴**:
- `CatBoostClassifier`と違い `loss_function` のデフォルトは `'RMSE'` と明示されている(分類側は`None`で実行時に自動判定)。
- `class_weights`/`auto_class_weights`/`classes_count` など分類専用引数を持たない。

### `Pool(...)`

**用途**: catboost独自のデータコンテナ。特徴量・ラベルに加えて、カテゴリ変数の列やテキスト列、重みなどをまとめて保持できる。numpy配列/DataFrameを直接`fit`に渡すことも可能だが、大規模データや繰り返し利用時は`Pool`化してから渡す方が効率的。

**シグネチャ**: `catboost.Pool(self, data, label=None, cat_features=None, text_features=None, embedding_features=None, embedding_features_data=None, column_description=None, pairs=None, graph=None, delimiter='\t', has_header=False, ignore_csv_quoting=False, weight=None, group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, timestamp=None, feature_names=None, feature_tags=None, thread_count=-1, log_cout=None, log_cerr=None, data_can_be_none=False)`

**使用例**:
```python
from catboost import Pool

pool = Pool(Xtr, ytr)
print(type(pool), pool.shape)
print("num_row:", pool.num_row(), "num_col:", pool.num_col())
```
実行結果:
```
<class 'catboost.core.Pool'> (225, 6)
num_row: 225 num_col: 6
```

**注意点・落とし穴**:
- `shape`は`(行数, 列数)`のタプル(ラベル列は含まない)。
- `cv`・`grid_search`のクロスバリデーション系APIでは`Pool`(または`X, y`)を渡す設計になっており、`cat_features`はここで一度指定すれば以降使い回せる。

### `.predict()` / `.predict_proba()` / `.score()`

**用途**: 学習済みモデルで予測を行う。`predict`はクラスラベル(分類)または予測値(回帰)、`predict_proba`は分類確率、`score`は既定の評価指標(分類は正解率、回帰はR2)を返す。

**シグネチャ**:
- `CatBoostClassifier.predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU')`
- `CatBoostClassifier.predict_proba(self, X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU')`
- `CatBoostClassifier.score(self, X, y=None)`

**使用例**:
```python
print("Class:", clf.predict(Xte[:3], prediction_type="Class").ravel())
print("Probability:", clf.predict(Xte[:3], prediction_type="Probability").round(3))
print("RawFormulaVal:", clf.predict(Xte[:3], prediction_type="RawFormulaVal").round(3))
```
実行結果:
```
Class: [0 1 0]
Probability: [[0.818 0.182]
 [0.166 0.834]
 [0.84  0.16 ]]
RawFormulaVal: [-1.5    1.615 -1.661]
```

**注意点・落とし穴**:
- `prediction_type`は`'Class'`(既定)・`'Probability'`・`'RawFormulaVal'`(生のスコア、シグモイド適用前)などを選べる。`predict_proba`は内部的に`prediction_type='Probability'`相当。
- `predict()`の戻り値は`(n_samples, 1)`ではなく`(n_samples,)`の1次元配列(`.ravel()`は表示を揃えるためのもので必須ではない)。
- 汎用の`CatBoost`クラス(`CatBoostClassifier`/`Regressor`ではない)で読み込んだモデルの`predict()`はデフォルトで生スコア(margin)を返す点が異なる(後述「モデルの保存・読み込み」参照)。

---

## 2. カテゴリ変数・テキスト特徴量のネイティブ対応

### `cat_features`(カテゴリ変数の指定)

**用途**: どの列をカテゴリ変数として扱うかを指定する。指定した列は文字列のまま(one-hotエンコード不要で)学習に使える。

**シグネチャ**: `CatBoostClassifier(..., cat_features=None, ...)`(列名のリストまたは列インデックスのリストを渡す)

**使用例**:
```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

rng = np.random.default_rng(0)
n = 200
df = pd.DataFrame({
    "age": rng.integers(18, 65, n),
    "income": rng.normal(400, 100, n).round(1),
    "city": rng.choice(["Tokyo", "Osaka", "Nagoya"], n),
    "job": rng.choice(["Engineer", "Sales", "Teacher", "Doctor"], n),
})
y = ((df["age"] > 40) & (df["city"] == "Tokyo")).astype(int) ^ (rng.integers(0, 2, n) // 2)

clf = CatBoostClassifier(iterations=50, verbose=False, random_seed=0, cat_features=["city", "job"])
clf.fit(df, y)
print("predict[:5]:", clf.predict(df.iloc[:5]).ravel())
print("get_cat_feature_indices():", clf.get_cat_feature_indices())
```
実行結果:
```
predict[:5]: [1 0 0 0 0]
get_cat_feature_indices(): [2, 3]
```

**注意点・落とし穴**:
- `cat_features`に指定できるのは**整数(カテゴリID)または文字列**の列のみ。実数(float)の列を`cat_features`に含めるとエラーになる:実際に`float`型の列を`cat_features`に指定して`fit`すると`CatBoostError: Invalid type for cat_feature[...] : cat_features must be integer or string, real number values and NaN values should be converted to string.`が発生することを確認済み。数値ラベルのカテゴリ変数(例: 郵便番号)は事前に文字列化するか整数型のまま渡す必要がある。
- pandasのDataFrameを渡す場合、`cat_features`は列名でも列インデックスでも指定可能。numpy配列の場合はインデックスのみ。
- 欠損値(NaN)はそのまま「一つのカテゴリ」として扱われる。

### `Pool` + `cat_features` / `get_cat_feature_indices()`

**用途**: `Pool`側でカテゴリ変数を指定する場合の書き方。モデル側の`cat_features`と同じ列名・列インデックスを渡す。

**シグネチャ**: `catboost.Pool(data, label=None, cat_features=None, ...)` / `Pool.get_cat_feature_indices(self)`

**使用例**:
```python
from catboost import Pool

pool = Pool(df, y, cat_features=["city", "job"])
print(pool.get_cat_feature_indices())
```
実行結果:
```
[2, 3]
```

**注意点・落とし穴**:
- `Pool`作成時と`fit`時の両方で`cat_features`を指定できるが、`Pool`にすでに`cat_features`が設定されている場合、モデル側に別の`cat_features`を渡すとエラーまたは無視される。どちらか一方(`Pool`側を推奨)に統一する方が事故が少ない。

### `text_features`

**用途**: 自然文(自由記述テキスト)の列を、専用のテキスト特徴量として扱う。TF-IDFなどを事前に自分で行う必要がない。

**シグネチャ**: `CatBoostClassifier(..., text_features=None, tokenizers=None, dictionaries=None, feature_calcers=None, text_processing=None, ...)`

**使用例**:
```python
import pandas as pd
from catboost import CatBoostClassifier

text_df = pd.DataFrame({
    "review": ["great product loved it", "terrible waste of money",
               "not bad could be better", "amazing quality highly recommend",
               "worst purchase ever made"] * 20,
    "score_feat": rng.normal(0, 1, 100),
})
yt = [1, 0, 1, 1, 0] * 20

clf_text = CatBoostClassifier(iterations=30, verbose=False, random_seed=0, text_features=["review"])
clf_text.fit(text_df, yt)
print("predict[:5]:", clf_text.predict(text_df.iloc[:5]).ravel())
```
実行結果:
```
predict[:5]: [1 0 1 1 0]
```

**注意点・落とし穴**:
- `text_features`に指定する列はpandasの文字列(object)列である必要がある。
- デフォルトのトークナイザ・辞書・特徴量計算器(`feature_calcers`、既定は`BoW`相当)が自動適用されるため、細かく制御したい場合は`tokenizers`/`dictionaries`/`feature_calcers`を個別に指定する。
- GPUでは`text_features`のサポートに制限がある(CPUでの利用が基本)。

### `one_hot_max_size`

**用途**: カテゴリ数がこの値以下の特徴量については、catboost独自のターゲットエンコーディング(CTR)ではなく単純なone-hotエンコーディングを使うよう指定する。

**シグネチャ**: `CatBoostClassifier(..., one_hot_max_size=None, ...)`(既定は`None`。実行時、CPUでは2などの小さい値が自動選択される)

**使用例**:
```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

rng = np.random.default_rng(0)
df3 = pd.DataFrame({"cat3": rng.choice(["a", "b", "c"], 100), "num": rng.normal(size=100)})
yb = rng.integers(0, 2, 100)

c1 = CatBoostClassifier(iterations=10, verbose=False, one_hot_max_size=2, cat_features=["cat3"])
c1.fit(df3, yb)
c2 = CatBoostClassifier(iterations=10, verbose=False, one_hot_max_size=10, cat_features=["cat3"])
c2.fit(df3, yb)
```
実行結果: 両方ともエラーなく学習が完了することを確認(`cat3`は3カテゴリの列)。`one_hot_max_size=10`の方は3カテゴリ全てがone-hot化され、`one_hot_max_size=2`の方はカテゴリ数(3)がしきい値を超えるためCTR(統計量ベースのエンコーディング)が使われる。

**注意点・落とし穴**:
- 学習後のモデルから「どちらの方式が使われたか」を直接取得できるパブリックAPIは見つからなかった(`get_all_params()`には`one_hot_max_size`の値自体は出るが、各特徴量への適用結果は出ない)。挙動の詳細はcatboost公式ドキュメントの記載に基づく。
- カテゴリ数が非常に多い列(例: ユーザーID)を`one_hot_max_size`より大きくしたまま放置すると、CTR計算のコストが増える。

---

## 3. 交差検証・ハイパーパラメータ探索

### `cv(...)`

**用途**: `Pool`とパラメータ辞書を渡し、指定fold数でクロスバリデーションを行う関数(モデルのインスタンスメソッドではなくトップレベル関数)。

**シグネチャ**: `catboost.cv(pool=None, params=None, dtrain=None, iterations=None, num_boost_round=None, fold_count=None, nfold=None, inverted=False, partition_random_seed=0, seed=None, shuffle=True, logging_level=None, stratified=None, as_pandas=True, metric_period=None, verbose=None, verbose_eval=None, plot=False, plot_file=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, metric_update_interval=0.5, folds=None, type='Classical', return_models=False, log_cout=None, log_cerr=None)`

**使用例**:
```python
from sklearn.datasets import make_classification
from catboost import Pool, cv

X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=2, random_state=0)
pool = Pool(X, y)
params = {"loss_function": "Logloss", "iterations": 50, "depth": 4, "learning_rate": 0.1, "verbose": False}
scores = cv(pool, params, fold_count=3, seed=0, shuffle=True, verbose=False, plot=False)
print(type(scores))
print(scores.columns.tolist())
print(scores.tail(1).round(4))
```
実行結果:
```
<class 'pandas.DataFrame'>
['iterations', 'test-Logloss-mean', 'test-Logloss-std', 'train-Logloss-mean', 'train-Logloss-std']
    iterations  test-Logloss-mean  ...  train-Logloss-mean  train-Logloss-std
49          49             0.1977  ...               0.092             0.0217

[1 rows x 5 columns]
```

**注意点・落とし穴**:
- `cv()`関数を直接`verbose=False`で呼んでも、「Training on fold [0/3]」「bestTest = ...」といったfoldごとのヘッダーログは抑制されず標準出力に残ることを確認済み(パラメータ辞書側の`"verbose": False`とcv関数引数の`verbose=False`の両方を指定しても同様)。ログを完全に消したい場合は`log_cout`にダミーのストリームを渡す必要がある。
- 戻り値はデフォルト(`as_pandas=True`)でpandas DataFrame。イテレーションごとの行を持ち、最終行(`.tail(1)`)が全fold平均の最終スコアに相当する。
- `sklearn.model_selection.cross_val_score`とは別物で、`scoring`ではなく`params`辞書に`loss_function`/カスタム`custom_metric`を指定する。

### `.grid_search(...)`

**用途**: パラメータの全組み合わせをクロスバリデーションで評価し、最良の組み合わせを見つける(`CatBoostClassifier`/`Regressor`のインスタンスメソッド)。

**シグネチャ**: `CatBoostClassifier.grid_search(self, param_grid, X, y=None, cv=3, partition_random_seed=0, calc_cv_statistics=True, search_by_train_test_split=True, refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False, plot_file=None, log_cout=None, log_cerr=None)`

**使用例**:
```python
clf = CatBoostClassifier(iterations=30, verbose=False, random_seed=0)
grid = {"depth": [4, 6], "learning_rate": [0.05, 0.2]}
result = clf.grid_search(grid, X=X, y=y, cv=3, verbose=False, plot=False)
print(list(result.keys()))
print("best_params:", result["params"])
```
実行結果:
```
['params', 'cv_results']
best_params: {'depth': 6, 'learning_rate': 0.2}
```

**注意点・落とし穴**:
- `verbose=False`を指定しても各組み合わせの学習ログ(`bestTest = ...`)は出力される(`cv()`関数と同様)。
- `refit=True`(既定)により、`grid_search`実行後は呼び出し元の`clf`自体が最良パラメータで再学習された状態になる(戻り値を使わなくても`clf.predict()`がそのまま使える)。
- 戻り値の`"params"`キーには最良パラメータの辞書が、`"cv_results"`キーには各組み合わせのCV結果(DataFrame)が入る。

### `.randomized_search(...)`

**用途**: パラメータ空間からランダムに`n_iter`通りサンプリングして評価する(`grid_search`の効率化版)。

**シグネチャ**: `CatBoostClassifier.randomized_search(self, param_distributions, X, y=None, cv=3, n_iter=10, partition_random_seed=0, calc_cv_statistics=True, search_by_train_test_split=True, refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False, plot_file=None, log_cout=None, log_cerr=None)`

**使用例**:
```python
import numpy as np
from scipy.stats import uniform

np.random.seed(0)  # scipy.stats分布のサンプリングはnumpyのグローバル乱数状態に依存するため
clf2 = CatBoostClassifier(iterations=30, verbose=False, random_seed=0)
dist = {"depth": [4, 6, 8], "learning_rate": uniform(0.01, 0.3)}
res2 = clf2.randomized_search(dist, X=X, y=y, cv=3, n_iter=5, verbose=False, plot=False)
print("best_params:", res2["params"])
```
実行結果:
```
best_params: {'depth': 4, 'learning_rate': 0.22455680991172586}
```

**注意点・落とし穴**:
- `param_distributions`には固定リストだけでなく`scipy.stats`の確率分布オブジェクト(`uniform`など)を混在させられる(scikit-learnの`RandomizedSearchCV`と同様の考え方)。
- `randomized_search`のシグネチャには`random_state`/`seed`引数が無い。`partition_random_seed`はCVのfold分割のみを固定し、`param_distributions`から値をサンプリングする乱数はnumpyのグローバル状態に依存する。実際に同一コードを`np.random.seed()`無しで2回実行したところ`best_params`が`{'depth': 4, ...}`と`{'depth': 6, ...}`のように毎回変わることを確認した。再現性が必要な場合は呼び出し前に`np.random.seed(...)`を固定する。

---

## 4. Ordered Boosting・過学習対策

### `eval_set` / `use_best_model` / `early_stopping_rounds`

**用途**: 検証用データ(`eval_set`)でのスコアを監視し、一定ラウンド改善が無ければ学習を打ち切る(`early_stopping_rounds`)、または全イテレーション終了後に検証スコアが最良だった時点のモデルを採用する(`use_best_model`)。

**シグネチャ**: `CatBoostClassifier.fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None, graph=None, sample_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, plot_file=None, column_description=None, verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None, log_cout=None, log_cerr=None)`(`early_stopping_rounds`/`use_best_model`自体はコンストラクタ引数でもある)

**使用例**:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

X, y = make_classification(n_samples=300, n_features=10, n_informative=4, n_classes=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.25, random_state=0)

clf = CatBoostClassifier(
    iterations=1000, depth=6, learning_rate=0.1, random_seed=0,
    eval_metric="Logloss", early_stopping_rounds=20, use_best_model=True, verbose=False,
)
clf.fit(Xtr2, ytr2, eval_set=(Xval, yval))
print("tree_count_:", clf.tree_count_)
print("get_best_iteration():", clf.get_best_iteration())
print("get_best_score():", clf.get_best_score())
```
実行結果:
```
tree_count_: 47
get_best_iteration(): 46
get_best_score(): {'learn': {'Logloss': 0.02516485679717291}, 'validation': {'Logloss': 0.2733975837303226}}
```

**注意点・落とし穴**:
- `iterations=1000`を指定しても、`early_stopping_rounds=20`により実際には47本の木で学習が打ち切られている(`tree_count_`で確認可能)。指定したイテレーション数がそのまま木の本数になるとは限らない。
- `use_best_model=True`は`eval_set`が無いと効果を持たない(検証データが無ければ「最良」を判定できない)。

### `boosting_type`(Ordered / Plain)

**用途**: catboost独自の「Ordered Boosting」(各ステップで、その学習データを使っていないモデルで残差を推定し、target leakageを抑える手法)を使うか、通常のPlain(勾配ブースティングの標準的な実装)を使うかを指定する。

**シグネチャ**: `CatBoostClassifier(..., boosting_type=None, ...)`(`'Ordered'` または `'Plain'`。既定は`None`=自動選択)

**使用例**:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

X, y = make_classification(n_samples=300, n_features=10, n_informative=4, n_classes=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

clf_ordered = CatBoostClassifier(iterations=50, boosting_type="Ordered", verbose=False, random_seed=0)
clf_ordered.fit(Xtr, ytr)
clf_plain = CatBoostClassifier(iterations=50, boosting_type="Plain", verbose=False, random_seed=0)
clf_plain.fit(Xtr, ytr)
print("Ordered predict[:5]:", clf_ordered.predict(Xte[:5]).ravel())
print("Plain predict[:5]:", clf_plain.predict(Xte[:5]).ravel())
```
実行結果:
```
Ordered predict[:5]: [0 0 0 1 0]
Plain predict[:5]: [0 0 0 1 0]
```

**注意点・落とし穴**:
- 今回の検証環境(サンプル数100〜10万、CPU)では、`boosting_type`を明示しなかった場合に`get_all_params()`で確認すると小規模データ(100件)・大規模データ(10万件)のいずれでも実際に選ばれていたのは`'Plain'`だった。「小規模データではOrderedが自動選択される」という一般的な説明とは異なる結果だったため、実際に使われている値は推測せず`get_all_params()`で確認することを推奨する。
- `boosting_type='Ordered'`は`'Plain'`よりも計算コストが高く、大規模データやGPUでは`'Ordered'`が使えない(または自動的に`'Plain'`にフォールバックする)組み合わせがある。

### `od_type` / `od_wait`

**用途**: 過学習検出器(Overfitting Detector)の種類と待機ラウンド数。`early_stopping_rounds`はこの仕組みの簡易指定版に相当する。

**シグネチャ**: `CatBoostClassifier(..., od_type=None, od_wait=None, od_pval=None, ...)`(`od_type`は`'IncToDec'`または`'Iter'`)

**使用例**:
```python
X2, y2 = make_classification(n_samples=300, n_features=20, n_informative=3, n_redundant=2,
                              flip_y=0.3, n_classes=2, random_state=0)
Xtr2b, Xte2b, ytr2b, yte2b = train_test_split(X2, y2, test_size=0.25, random_state=0)
Xtr3, Xval2, ytr3, yval2 = train_test_split(Xtr2b, ytr2b, test_size=0.3, random_state=0)

clf = CatBoostClassifier(iterations=2000, depth=8, learning_rate=0.3,
                          od_type="Iter", od_wait=15, verbose=False, random_seed=0)
clf.fit(Xtr3, ytr3, eval_set=(Xval2, yval2))
print("tree_count_:", clf.tree_count_, "/ requested 2000")
```
実行結果:
```
tree_count_: 21 / requested 2000
```

**注意点・落とし穴**:
- ノイズの多いデータ(`flip_y=0.3`)・高い`learning_rate`・深い木という「過学習しやすい」設定にしたところ、要求した2000イテレーションのうちわずか21本で学習が打ち切られた。`od_type='Iter'`は`od_wait`ラウンド連続でスコア改善が無ければ即座に停止する単純な方式。
- `od_type='IncToDec'`(既定)は`od_pval`(p値のしきい値)を使ったより統計的な判定方式で、`od_wait`の意味合いも変わる。

### `.get_best_iteration()` / `.get_best_score()`

**用途**: `use_best_model=True`または過学習検出器で学習を打ち切った際に、実際に採用された最良イテレーション番号とそのスコアを取得する。

**シグネチャ**: `CatBoostClassifier.get_best_iteration(self)` / `CatBoostClassifier.get_best_score(self)`

**使用例**: 上記「`eval_set` / `use_best_model` / `early_stopping_rounds`」の実行結果を参照(`get_best_iteration(): 46`、`get_best_score(): {'learn': {...}, 'validation': {...}}`)。

**注意点・落とし穴**:
- `get_best_score()`は`{'learn': {...}, 'validation': {...}}`の2階層の辞書を返す。`eval_set`を渡していない場合は`'validation'`キーが存在しない。

---

## 5. クラス不均衡への対応

### `auto_class_weights` / `class_weights`

**用途**: 少数派クラスの重みを自動的に(または手動で)引き上げ、不均衡データでの少数派クラスの見逃しを減らす。

**シグネチャ**: `CatBoostClassifier(..., class_weights=None, auto_class_weights=None, ...)`(`auto_class_weights`は`'Balanced'`または`'SqrtBalanced'`)

**使用例**:
```python
from sklearn.metrics import recall_score

X3, y3 = make_classification(n_samples=1000, n_features=8, n_informative=4,
                              weights=[0.95, 0.05], flip_y=0.02, random_state=0)
Xtr3b, Xte3b, ytr3b, yte3b = train_test_split(X3, y3, test_size=0.25, random_state=0, stratify=y3)

clf0 = CatBoostClassifier(iterations=100, verbose=False, random_seed=0)
clf0.fit(Xtr3b, ytr3b)
print("重み付けなし recall(少数派):", recall_score(yte3b, clf0.predict(Xte3b)))

clf1 = CatBoostClassifier(iterations=100, verbose=False, random_seed=0, auto_class_weights="Balanced")
clf1.fit(Xtr3b, ytr3b)
print("auto_class_weights=Balanced recall(少数派):", recall_score(yte3b, clf1.predict(Xte3b)))
```
実行結果:
```
重み付けなし recall(少数派): 0.2
auto_class_weights=Balanced recall(少数派): 0.4666666666666667
```

**注意点・落とし穴**:
- 少数派クラス(全体の5%)のrecallが0.2→0.467に改善した一方、この設定は「少数派の見逃しを減らす」代わりに多数派クラスの誤検出(false positive)が増えるトレードオフがある(precisionとのバランスを別途確認する必要がある)。
- `class_weights`(手動指定、例: `[1, 10]`)と`auto_class_weights`は同時に指定できない。

---

## 6. 特徴量重要度・特徴量選択

### `.get_feature_importance()` / `.feature_importances_`

**用途**: 学習済みモデルの特徴量重要度を取得する。既定では`PredictionValuesChange`(その特徴量がある場合とない場合で予測値がどれだけ変化するか)を使う。

**シグネチャ**: `CatBoostClassifier.get_feature_importance(self, data=None, type=<EFstrType.FeatureImportance: 2>, prettified=False, thread_count=-1, verbose=False, fstr_type=None, shap_mode='Auto', model_output='Raw', interaction_indices=None, shap_calc_type='Regular', reference_data=None, sage_n_samples=128, sage_batch_size=512, sage_detect_convergence=True, log_cout=None, log_cerr=None)`

**使用例**:
```python
from sklearn.datasets import make_classification
from catboost import CatBoostClassifier

Xf, yf = make_classification(n_samples=300, n_features=6, n_informative=3, n_classes=2, random_state=0)
clf_fi = CatBoostClassifier(iterations=50, verbose=False, random_seed=0)
clf_fi.fit(Xf, yf)

imp = clf_fi.get_feature_importance()
print(imp.round(2))
print(clf_fi.feature_importances_.round(2))
```
実行結果:
```
[ 6.68 28.34 15.07 12.68  3.2  34.03]
[ 6.68 28.34 15.07 12.68  3.2  34.03]
```

**注意点・落とし穴**:
- `feature_importances_`属性は`get_feature_importance()`(既定パラメータ)の結果と完全に一致する(scikit-learn互換のためのエイリアス)。
- 値は合計100になる正規化はされていない(scikit-learnの`RandomForestClassifier.feature_importances_`とは正規化のされ方が異なる場合がある)。

### `.get_feature_importance(prettified=True)`

**用途**: 特徴量重要度を、特徴量名と重要度の対応が分かりやすいpandas DataFrame(重要度降順)で取得する。

**シグネチャ**: 上記`get_feature_importance`と同じ(`prettified=True`を指定)。

**使用例**:
```python
pretty = clf_fi.get_feature_importance(prettified=True)
print(pretty)
```
実行結果:
```
  Feature Id  Importances
0          5    34.028622
1          1    28.339549
2          2    15.068355
3          3    12.684215
4          0     6.682643
5          4     3.196616
```

**注意点・落とし穴**:
- `Feature Id`列は、明示的な列名を渡さずnumpy配列で学習した場合は列インデックスの文字列(`"0"`, `"1"`, ...)になる。DataFrameで学習していれば実際の列名が入る。

### `.get_feature_importance(type="ShapValues")`

**用途**: SHAP値(各特徴量が個々の予測にどれだけ寄与したか)を計算する。

**シグネチャ**: 上記`get_feature_importance`と同じ(`data`に`Pool`、`type="ShapValues"`を指定)。

**使用例**:
```python
from catboost import Pool

pool_fi = Pool(Xf, yf)
shap = clf_fi.get_feature_importance(pool_fi, type="ShapValues")
print(shap.shape)
print(shap[0].round(3))
```
実行結果:
```
(300, 7)
[ 0.184  0.118  0.235 -0.146 -0.003  1.354 -0.081]
```

**注意点・落とし穴**:
- 戻り値の列数は「特徴量数+1」になる(最後の列がベース値/期待値のオフセット)。6特徴量のデータで7列になっている。
- SHAP値の計算には`data`(`Pool`または`X, y`)の指定が必須。省略すると学習時のデータに対する重要度(`PredictionValuesChange`)に切り替わる。

### `.select_features(...)`

**用途**: SHAP値などに基づき重要度の低い特徴量を反復的に削除し、指定した個数まで特徴量選択を行う。

**シグネチャ**: `CatBoostClassifier.select_features(self, X, y=None, eval_set=None, features_for_select=None, num_features_to_select=None, algorithm=None, steps=None, shap_calc_type=None, train_final_model=True, verbose=None, logging_level=None, plot=False, plot_file=None, log_cout=None, log_cerr=None, grouping=None, features_tags_for_select=None, num_features_tags_to_select=None)`

**使用例**:
```python
from catboost import EFeaturesSelectionAlgorithm, EShapCalcType

X4, y4 = make_classification(n_samples=300, n_features=8, n_informative=3, n_classes=2, random_state=0)
Xtr4, Xte4, ytr4, yte4 = train_test_split(X4, y4, test_size=0.25, random_state=0)

clf4 = CatBoostClassifier(iterations=100, verbose=False, random_seed=0)
summary = clf4.select_features(
    Xtr4, ytr4, eval_set=(Xte4, yte4),
    features_for_select=list(range(8)), num_features_to_select=4, steps=2,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular, train_final_model=True,
    verbose=False, plot=False,
)
print(list(summary.keys()))
print("selected_features:", summary["selected_features"])
print("eliminated_features:", summary["eliminated_features"])
```
実行結果:
```
['selected_features', 'eliminated_features_names', 'loss_graph', 'eliminated_features', 'selected_features_names']
selected_features: [1, 3, 4, 7]
eliminated_features: [2, 6, 0, 5]
```

**注意点・落とし穴**:
- `verbose=False`を指定してもステップごとの「Feature #N eliminated」ログは出力される。
- `train_final_model=True`(既定)の場合、`select_features`実行後は`clf4`自体が選択後の特徴量のみで再学習された状態になる。

---

## 7. モデルの保存・読み込み

### `.save_model(...)` / `.load_model(...)`

**用途**: 学習済みモデルをファイルに保存し、後から読み込んで再利用する。

**シグネチャ**: `CatBoostClassifier.save_model(self, fname, format='cbm', export_parameters=None, pool=None)` / `CatBoostClassifier.load_model(self, fname=None, format='cbm', stream=None, blob=None)`

**使用例**:
```python
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

X, y = make_classification(n_samples=300, n_features=6, n_informative=3, n_classes=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
clf = CatBoostClassifier(iterations=30, verbose=False, random_seed=0)
clf.fit(Xtr, ytr)

clf.save_model("model.cbm")
print("file exists:", os.path.exists("model.cbm"), "size(bytes):", os.path.getsize("model.cbm"))

loaded = CatBoostClassifier()
loaded.load_model("model.cbm")
print("predict equal:", np.array_equal(clf.predict(Xte), loaded.predict(Xte)))
```
実行結果:
```
file exists: True size(bytes): 40496
predict equal: True
```

**注意点・落とし穴**:
- `load_model`は空の`CatBoostClassifier()`インスタンス(パラメータ未指定でOK)に対して呼び出す。読み込み後は保存時のパラメータ・木構造がすべて復元される。
- `format`は既定の`'cbm'`(catboost独自バイナリ)の他に`'json'`・`'coreml'`・`'onnx'`・`'python'`なども指定できる(用途に応じて選ぶ)。

### `CatBoost`(汎用クラスでの読み込み)

**用途**: `CatBoostClassifier`/`CatBoostRegressor`を区別しない汎用のモデルクラス。保存されたモデルが分類用か回帰用か分からない状況での読み込みなどに使う。

**シグネチャ**: `catboost.CatBoost(params=None)`

**使用例**:
```python
from catboost import CatBoost

generic = CatBoost()
generic.load_model("model.cbm")
print(type(generic), generic.predict(Xte[:3]))
```
実行結果:
```
<class 'catboost.core.CatBoost'> [ 1.02137276  1.28757333 -1.77143259]
```

**注意点・落とし穴**:
- 元は`CatBoostClassifier`で保存したモデルでも、`CatBoost`汎用クラスで読み込んで`predict()`すると**クラスラベルではなく生スコア(RawFormulaVal)が返る**(`CatBoostClassifier.predict()`のようにシグモイドを適用してクラス化する処理を持たないため)。分類結果が欲しい場合は`CatBoostClassifier`で読み込むこと。

---

## 8. 評価指標・可視化

### `.eval_metrics(...)`

**用途**: 学習済みモデルについて、指定した評価指標を(木の本数ごとの推移として)計算する。

**シグネチャ**: `CatBoostClassifier.eval_metrics(self, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None, plot=False, plot_file=None, log_cout=None, log_cerr=None)`

**使用例**:
```python
from catboost import Pool

test_pool = Pool(Xte, yte)
res = clf.eval_metrics(test_pool, metrics=["AUC", "Accuracy"])
print(list(res.keys()))
print("AUC(最終イテレーション):", round(res["AUC"][-1], 4))
print("Accuracy(最終イテレーション):", round(res["Accuracy"][-1], 4))
```
実行結果:
```
['AUC', 'Accuracy']
AUC(最終イテレーション): 0.9716
Accuracy(最終イテレーション): 0.9067
```

**注意点・落とし穴**:
- 戻り値は「指標名→木の本数分の値のリスト」の辞書。単一の値ではなく、`ntree_start`〜実際の木の本数まで各イテレーション時点でのスコアが入っている(最終値が欲しい場合は`[-1]`を取る)。

### `.get_evals_result()`

**用途**: `fit()`時に`eval_set`を渡して学習した場合の、学習曲線(イテレーションごとの指標推移)を取得する。

**シグネチャ**: `CatBoostClassifier.get_evals_result(self)`

**使用例**:
```python
clf5 = CatBoostClassifier(iterations=50, eval_metric="AUC", verbose=False, random_seed=0)
clf5.fit(Xtr, ytr, eval_set=(Xte, yte))
res = clf5.get_evals_result()
print(res.keys())
print("learnのキー:", res["learn"].keys())
print("validationのキー:", res["validation"].keys())
```
実行結果:
```
dict_keys(['learn', 'validation'])
learnのキー: dict_keys(['Logloss'])
validationのキー: dict_keys(['Logloss', 'AUC'])
```

**注意点・落とし穴**:
- `eval_metric="AUC"`を指定しても、`'learn'`側には目的関数(この場合`Logloss`)しか記録されず、`AUC`は`'validation'`側にしか現れないことを確認済み(学習データに対するAUC推移は既定では追跡されない)。学習曲線を`learn`/`validation`両方でAUC比較したい場合は、`custom_metric=["AUC"]`を併用するなど別途工夫が必要。

### `catboost.utils.eval_metric(...)`

**用途**: モデルを介さず、予測値と正解ラベルから直接指定した評価指標を計算するユーティリティ関数。

**シグネチャ**: `catboost.utils.eval_metric(label, approx, metric, weight=None, group_id=None, group_weight=None, subgroup_id=None, pairs=None, thread_count=-1)`

**使用例**:
```python
from catboost.utils import eval_metric

proba = clf5.predict_proba(Xte)[:, 1]
print("AUC:", eval_metric(yte, proba, "AUC"))
```
実行結果:
```
AUC: [0.9786628733997155]
```

**注意点・落とし穴**:
- 戻り値はスカラーではなく**要素数1のリスト**(内部的にはイテレーション/ブロック単位の値を返す設計のため)。単純に`float`として使いたい場合は`[0]`でアクセスする必要がある。
- `approx`(予測値)は確率ではなく生スコアを渡す指標もあるため、`metric`ごとに何を渡すべきかドキュメントで確認する。

### `.staged_predict(...)`

**用途**: 木の本数を増やしながら段階的に予測値を得る(学習過程での予測の変化を見たい場合などに使う)。戻り値はジェネレータ。

**シグネチャ**: `CatBoostClassifier.staged_predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None)`

**使用例**:
```python
preds = list(clf.staged_predict(Xte[:3], ntree_start=0, ntree_end=5, eval_period=2))
print(len(preds), [p.tolist() for p in preds])
```
実行結果:
```
3 [[1, 1, 0], [1, 1, 0], [1, 1, 0]]
```

**注意点・落とし穴**:
- `ntree_end=5`・`eval_period=2`のとき、木2本・4本・(端数の)5本時点の3段階分の予測が返る(`(ntree_end - ntree_start) / eval_period`を切り上げた回数になる)。
- 通常の`predict()`を複数回呼ぶより、木の再利用により効率的に段階的な予測が得られる。

---

## 9. ハイパーパラメータ・その他ユーティリティ

### `depth` / `iterations` / `learning_rate`

**用途**: catboostの最重要ハイパーパラメータ3点。`depth`は各木の深さ、`iterations`は木の本数、`learning_rate`は各木の寄与度。

**シグネチャ**: `CatBoostClassifier(iterations=None, learning_rate=None, depth=None, ...)`

**使用例**:
```python
X5, y5 = make_classification(n_samples=400, n_features=10, n_informative=4, n_classes=2, random_state=0)
Xtr5, Xte5, ytr5, yte5 = train_test_split(X5, y5, test_size=0.25, random_state=0)

for depth in [2, 6, 10]:
    m = CatBoostClassifier(iterations=100, depth=depth, verbose=False, random_seed=0)
    m.fit(Xtr5, ytr5)
    print(f"depth={depth}: train={m.score(Xtr5, ytr5):.4f} test={m.score(Xte5, yte5):.4f}")
```
実行結果:
```
depth=2: train=0.8833 test=0.9200
depth=6: train=0.9433 test=0.8900
depth=10: train=0.9767 test=0.8800
```

**注意点・落とし穴**:
- `depth`を大きくするほど学習データへの当てはまり(train score)は単調に上がったが、今回の検証データではテストスコアはむしろ`depth=2`が最良だった。深い木ほど良いとは限らず、過学習の兆候(train↑・test↓の乖離)を見ながら調整する必要がある。
- `depth`の最大値はCPUで16まで(GPUではさらに制限がある)。
- `iterations`・`depth`・`learning_rate`は`n_estimators`・`max_depth`・`eta`という別名でも指定できる(xgboost/lightgbm互換)。

### `l2_leaf_reg`

**用途**: 葉の重みに対するL2正則化係数。過学習抑制のために使う。

**シグネチャ**: `CatBoostClassifier(..., l2_leaf_reg=None, ...)`(既定は実質3)

**使用例**:
```python
for l2 in [1, 3, 20]:
    m = CatBoostClassifier(iterations=100, l2_leaf_reg=l2, verbose=False, random_seed=0)
    m.fit(Xtr5, ytr5)
    print(f"l2_leaf_reg={l2}: train={m.score(Xtr5, ytr5):.4f} test={m.score(Xte5, yte5):.4f}")
```
実行結果:
```
l2_leaf_reg=1: train=0.9400 test=0.8900
l2_leaf_reg=3: train=0.9267 test=0.8800
l2_leaf_reg=20: train=0.9000 test=0.8700
```

**注意点・落とし穴**:
- `l2_leaf_reg`を大きくするほど train/test ともにスコアが下がる傾向を確認した(正則化が強すぎると単純に学習不足になる)。`GridSearchCV`相当の`grid_search`/`randomized_search`で適切な範囲を探索するのが実用的。

### `.get_params()` / `.set_params()`

**用途**: モデルに設定したパラメータ(ユーザーが明示的に指定したもののみ)を取得・変更する(scikit-learn互換API)。

**シグネチャ**: `CatBoostClassifier.get_params(self, **_unused_kwargs)` / `CatBoostClassifier.set_params(self, **params)`(`set_params`は`inspect.signature`では`**params`のみ表示された)

**使用例**:
```python
clf6 = CatBoostClassifier(iterations=30, verbose=False, random_seed=0)
print("fit前 learning_rate:", clf6.get_params().get("learning_rate"))
clf6.set_params(learning_rate=0.05)
print("set_params後 learning_rate:", clf6.get_params().get("learning_rate"))

clf6.fit(Xtr, ytr)
try:
    clf6.set_params(learning_rate=0.2)
except Exception as e:
    print("学習済みモデルへのset_paramsはエラー:", type(e).__name__, e)
```
実行結果:
```
fit前 learning_rate: None
set_params後 learning_rate: 0.05
学習済みモデルへのset_paramsはエラー: CatBoostError You can't change params of fitted model.
```

**注意点・落とし穴**:
- `get_params()`は「コンストラクタで明示的に渡された値のみ」を返す。`iterations`のように指定していれば返るが、`depth`のように未指定の引数は含まれない・`None`になる(実際に使われた値をすべて見たい場合は`get_all_params()`を使う)。
- **`fit()`後のモデルに対して`set_params()`を呼ぶと`CatBoostError`になる**ことを確認済み。パラメータを変更したい場合は新しいインスタンスを作るか、`fit()`前に`set_params()`を呼ぶ必要がある。

### `.classes_` / `.feature_names_` / `.tree_count_`

**用途**: 学習済みモデルが持つメタ情報。`classes_`は分類クラスの一覧、`feature_names_`は特徴量名、`tree_count_`は実際に構築された木の本数。

**シグネチャ**: いずれも`fit()`後に参照できる属性(メソッドではない)。

**使用例**:
```python
clf7 = CatBoostClassifier(iterations=30, verbose=False, random_seed=0)
clf7.fit(Xtr, ytr)
print("classes_:", clf7.classes_)
print("feature_names_:", clf7.feature_names_)
print("tree_count_:", clf7.tree_count_)
```
実行結果:
```
classes_: [0 1]
feature_names_: ['0', '1', '2', '3', '4', '5']
tree_count_: 30
```

**注意点・落とし穴**:
- numpy配列で学習した場合、`feature_names_`は列インデックスを文字列化しただけの値(`'0'`, `'1'`, ...)になる。意味のある特徴量名を残したい場合はpandas DataFrameで学習する。
- `early_stopping_rounds`や過学習検出器で途中停止した場合、`tree_count_`は指定した`iterations`より小さくなる(前述の「Ordered Boosting・過学習対策」参照)。

---

## 10. 単調性制約

### `monotone_constraints`(リスト形式)

**用途**: 特定の特徴量について「その特徴量が増えるほど予測値は単調に増加(または減少)する」という制約を課す。解釈性が求められる場面(与信スコアなど、特徴量と予測値の関係が直感に反してはいけない場合)で使う。

**シグネチャ**: `CatBoostRegressor(..., monotone_constraints=None, ...)`(`CatBoostClassifier`にも同じ引数がある。特徴量の順序に対応する`1`(増加制約)・`-1`(減少制約)・`0`(制約なし)のリスト、または特徴量インデックス/列名をキーにした辞書、`"(1,-1,0)"`のような文字列でも指定可能)

**使用例**:
```python
import numpy as np
from catboost import CatBoostRegressor

rng = np.random.default_rng(0)
n = 150
x1 = rng.uniform(0, 10, n)
x2 = rng.uniform(0, 10, n)
noise = rng.normal(0, 3.0, n)
y = 2 * x1 - 1.5 * x2 + noise  # 本来 x1 に対して増加、x2 に対して減少するはずの関係
X = np.column_stack([x1, x2])

reg_free = CatBoostRegressor(iterations=300, depth=6, verbose=False, random_seed=0)
reg_free.fit(X, y)
reg_mono = CatBoostRegressor(iterations=300, depth=6, verbose=False, random_seed=0,
                              monotone_constraints=[1, -1])
reg_mono.fit(X, y)

# x2を5に固定し、x1だけを動かして予測値の変化を見る
x1_test = np.linspace(0, 10, 15)
x2_fixed = np.full(15, 5.0)
Xtest = np.column_stack([x1_test, x2_fixed])
diff_free = np.diff(reg_free.predict(Xtest))
diff_mono = np.diff(reg_mono.predict(Xtest))
print("制約なし: 単調増加か?", bool(np.all(diff_free >= -1e-9)))
print("制約なしの差分:", diff_free.round(3))
print("制約ありモデルは単調増加か?", bool(np.all(diff_mono >= -1e-9)))
print("制約ありの差分:", diff_mono.round(3))
```
実行結果:
```
制約なし: 単調増加か? False
制約なしの差分: [ 3.177  0.023  4.199  0.972  1.048  2.137  1.127  0.802  0.32   0.343
  1.596  2.015  3.392 -1.262]
制約ありモデルは単調増加か? True
制約ありの差分: [3.397 0.363 2.356 2.13  0.387 2.032 0.695 0.512 0.994 0.44  1.202 2.127
 2.374 1.581]
```

**注意点・落とし穴**:
- 制約なしモデルは、ノイズの影響で局所的に単調性が崩れている(最後の区間で差分が`-1.262`と負になっている)のに対し、`monotone_constraints=[1, -1]`を指定したモデルは検証した15点すべてで単調増加が保たれていることを実際に確認した。
- `depth`が浅い(単純な)データでは制約なしでも自然に単調になることがあるため、制約の効果を確認する際はある程度ノイズの多いデータ・深い木で検証しないと違いが見えにくい。

### `monotone_constraints`(辞書形式・列名指定)

**用途**: pandas DataFrameで学習する場合に、列名をキーにして単調性制約を指定する(リスト形式より可読性が高い)。

**シグネチャ**: 上記と同じ(`monotone_constraints={"列名": 1, ...}`の形式)。

**使用例**:
```python
import pandas as pd
from catboost import CatBoostRegressor

df = pd.DataFrame({"a": x1, "b": x2, "c": rng.uniform(0, 10, n)})
reg_dict = CatBoostRegressor(iterations=200, depth=6, verbose=False, random_seed=0,
                              monotone_constraints={"a": 1, "b": -1})
reg_dict.fit(df, y)
print(reg_dict.get_all_params().get("monotone_constraints"))
```
実行結果:
```
{'1': -1, '0': 1}
```

**注意点・落とし穴**:
- `get_all_params()`で確認すると、辞書形式・列名指定で渡した制約は内部的に「列インデックス(文字列)→制約値」の辞書に正規化される。指定していない列(`"c"`、インデックス`2`)はキーごと出力から省かれる(制約なし=`0`として扱われるが、辞書には現れない)。
- 文字列形式(`"(1,-1,0)"`)・リスト形式・辞書形式のいずれで渡しても、`get_all_params()`上は同じ正規化された辞書表現になることを確認した。

---

## 11. 埋め込み特徴量

### `embedding_features`(DataFrameの配列列で指定)

**用途**: 事前学習済み埋め込み(文章embeddingや画像embeddingなど、固定長ベクトル)をそのまま1つの特徴量列として扱う。ベクトルを自分でPCAなどで次元圧縮したり個々の要素に展開したりせずに済む。

**シグネチャ**: `catboost.Pool(data, label=None, cat_features=None, text_features=None, embedding_features=None, embedding_features_data=None, ...)` / `CatBoostClassifier(..., embedding_features=None, ...)`

**使用例**:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

rng = np.random.default_rng(0)
n = 300
num_feat = rng.normal(size=(n, 2))
emb = rng.normal(size=(n, 4))  # 4次元の「埋め込みベクトル」を模したデータ
y = (num_feat[:, 0] + emb[:, 0] - emb[:, 1] > 0).astype(int)

df = pd.DataFrame({"f0": num_feat[:, 0], "f1": num_feat[:, 1]})
df["emb"] = list(emb)  # 1列にベクトル(numpy配列)を格納

df_tr, df_te, y_tr, y_te = train_test_split(df, y, test_size=0.25, random_state=0)

clf = CatBoostClassifier(iterations=100, verbose=False, random_seed=0, embedding_features=["emb"])
clf.fit(df_tr, y_tr)
print("score:", clf.score(df_te, y_te))
print("get_embedding_feature_indices():", clf.get_embedding_feature_indices())
```
実行結果:
```
score: 0.8666666666666667
get_embedding_feature_indices(): [2]
```

**注意点・落とし穴**:
- `embedding_features`を指定せずにベクトル(配列)を含む列をそのまま`fit`すると、`CatBoostError: ... Cannot convert obj [...] to float`が発生することを確認済み。ベクトル列は通常の数値特徴量として自動認識されないため、必ず`embedding_features`で明示する必要がある。
- `clf.get_all_params()`の出力には`embedding_features`キー自体が(`None`のまま)現れず、実際に使われたかどうかは`get_embedding_feature_indices()`で確認する必要がある。

### `embedding_features_data`(numpy配列を分離して渡す)

**用途**: pandasを使わず、通常の特徴量行列とは別のnumpy 2次元配列として埋め込みベクトルを渡す方法。

**シグネチャ**: `catboost.Pool(data, label=None, ..., embedding_features=None, embedding_features_data=None, ...)`(`embedding_features_data`は2次元配列のリスト。`embedding_features`側には、通常特徴量の末尾に続く「仮想列インデックス」を指定する)

**使用例**:
```python
import numpy as np
from catboost import CatBoostClassifier, Pool

rng = np.random.default_rng(0)
n = 200
num_feat = rng.normal(size=(n, 2))
emb = rng.normal(size=(n, 4))
y = (num_feat[:, 0] + emb[:, 0] - emb[:, 1] > 0).astype(int)

# num_feat は2列なので、埋め込み列の仮想インデックスは2
pool = Pool(num_feat, y, embedding_features=[2], embedding_features_data=[emb])
print("pool.shape:", pool.shape)
clf = CatBoostClassifier(iterations=50, verbose=False, random_seed=0)
clf.fit(pool)
print("score:", clf.score(pool))
print("get_embedding_feature_indices():", clf.get_embedding_feature_indices())
```
実行結果:
```
pool.shape: (200, 3)
score: 0.96
get_embedding_feature_indices(): [2]
```

**注意点・落とし穴**:
- `embedding_features_data`だけを渡し`embedding_features`を省略すると、`CatBoostError: 'embedding_features_data' is not None, but 'embedding_features' parameter is not specified`になることを確認済み。両方をセットで指定する必要がある。
- `embedding_features`に指定するインデックスは「通常特徴量の列数」を起点にした仮想的な位置(この例では`num_feat`が2列なので`2`)であり、`embedding_features_data`内のリストの並び順と対応する。`pool.shape`の列数(`3`)は「通常特徴量2列+埋め込み列1列」で、埋め込みベクトルの次元数(4)はカウントされない。

---

## 12. SHAP値の応用・モデル解釈の深掘り

### `.get_feature_importance(type="Interaction")`

**用途**: 2つの特徴量の組み合わせがどれだけ強く相互作用しているかを、特徴量ペアごとにスコアリングする(SHAPではなく、木構造の分割パターンに基づく指標)。

**シグネチャ**: `CatBoostClassifier.get_feature_importance(self, data=None, type=<EFstrType.FeatureImportance: 2>, prettified=False, ...)`(`type="Interaction"`を指定)

**使用例**:
```python
from sklearn.datasets import make_classification
from catboost import CatBoostClassifier, Pool

Xf, yf = make_classification(n_samples=300, n_features=6, n_informative=3, n_classes=2, random_state=0)
clf_fi = CatBoostClassifier(iterations=50, verbose=False, random_seed=0)
clf_fi.fit(Xf, yf)
pool_fi = Pool(Xf, yf)

inter = clf_fi.get_feature_importance(pool_fi, type="Interaction")
print(inter[:3])
```
実行結果:
```
[[ 1.          5.         20.36029227]
 [ 2.          5.         14.24694255]
 [ 1.          3.         13.50224016]]
```

**注意点・落とし穴**:
- 戻り値は`(特徴量ペア数, 3)`のnumpy配列で、各行が「特徴量インデックス1, 特徴量インデックス2, 相互作用スコア」。スコアの降順にソート済みで返る(6特徴量なので`C(6,2)=15`行になる)。
- あくまで「相互作用の強さ」の指標であり、SHAP値のように個々の予測への寄与を分解するものではない(個々の予測レベルの相互作用が欲しい場合は次項の`ShapInteractionValues`を使う)。

### `.get_feature_importance(type="ShapInteractionValues")`

**用途**: 通常のSHAP値(各特徴量の寄与)をさらに「特徴量ペアごとの寄与」に分解する。ある特徴量の寄与が別の特徴量の値に依存して変わる場合(交互作用効果)を個々の予測レベルで確認できる。

**シグネチャ**: 上記`get_feature_importance`と同じ(`type="ShapInteractionValues"`を指定)。

**使用例**:
```python
shap = clf_fi.get_feature_importance(pool_fi, type="ShapValues")
shap_inter = clf_fi.get_feature_importance(pool_fi, type="ShapInteractionValues")
print("ShapInteractionValues shape:", shap_inter.shape)

import numpy as np
row0_sum = shap_inter[0].sum(axis=1)
print("通常のSHAP値[0]:", shap[0].round(4))
print("interaction値の行和[0]:", row0_sum.round(4))
print("ベース値の列を除いて一致するか:", np.allclose(shap[0][:-1], row0_sum[:-1], atol=1e-6))
```
実行結果:
```
ShapInteractionValues shape: (300, 7, 7)
通常のSHAP値[0]: [ 0.184   0.1178  0.2351 -0.1464 -0.0031  1.3541 -0.0811]
interaction値の行和[0]: [ 0.184   0.1178  0.2351 -0.1464 -0.0031  1.3541  0.    ]
ベース値の列を除いて一致するか: True
```

**注意点・落とし穴**:
- 戻り値の形状は`(サンプル数, 特徴量数+1, 特徴量数+1)`(`ShapValues`と同様、最後の行・列がベース値のオフセット用)。
- 各サンプルについて「interaction行列の行方向の和」は通常の`ShapValues`と(ベース値の列を除いて)一致することを実際に確認した。つまり`ShapInteractionValues`は`ShapValues`の各特徴量寄与を、どの特徴量との交互作用によるものかにさらに分解したものになっている。
- `(サンプル数, 特徴量数+1, 特徴量数+1)`の3次元配列は特徴量数が多いと急激にメモリを消費するため、大規模データでは`data`に一部サンプルのみの`Pool`を渡すなど注意が必要。

### `.get_object_importance(...)`(学習サンプルの影響度分析)

**用途**: テストデータの各予測が、学習データのどのサンプルにどれだけ強く影響されているかを求める(いわゆるinfluence functions/leave-one-out的な分析)。ノイズの多い・誤った学習データの特定に使える。

**シグネチャ**: `CatBoostClassifier.get_object_importance(self, pool, train_pool, top_size=-1, type='Average', update_method='SinglePoint', importance_values_sign='All', thread_count=-1, verbose=False, ostr_type=None, log_cout=None, log_cerr=None)`

**使用例**:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

X, y = make_classification(n_samples=300, n_features=6, n_informative=3, n_classes=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

clf = CatBoostClassifier(iterations=50, verbose=False, random_seed=0)
clf.fit(Xtr, ytr)

train_pool = Pool(Xtr, ytr)
test_pool = Pool(Xte, yte)
indices, scores = clf.get_object_importance(test_pool, train_pool, top_size=5)
print("影響度の大きい学習サンプルのインデックス:", indices)
print("影響度スコア:", [round(s, 4) for s in scores])
```
実行結果:
```
影響度の大きい学習サンプルのインデックス: [111, 35, 152, 122, 26]
影響度スコア: [0.0035, 0.0028, 0.0026, -0.0023, 0.0023]
```

**注意点・落とし穴**:
- 戻り値は`(インデックスのリスト, スコアのリスト)`のタプル。スコアの絶対値が大きい順に`top_size`件だけ返る(正負が混在し、正=予測に悪影響/負=良い影響、または逆の解釈になりうるため、`importance_values_sign`引数で符号を絞り込める)。
- `pool`(評価対象、通常はテストデータ)と`train_pool`(影響度を測る対象の学習データ)を取り違えないよう注意。

### `.virtual_ensembles_predict(...)`(予測の不確実性)

**用途**: 1つのモデルから複数の仮想アンサンブル(木の集合の一部分)を切り出し、予測のばらつきから「モデル自体の不確実性(knowledge uncertainty)」と「データ由来の不確実性(data uncertainty)」を分解して推定する。

**シグネチャ**: `CatBoostRegressor.virtual_ensembles_predict(self, data, prediction_type='VirtEnsembles', ntree_end=0, virtual_ensembles_count=10, thread_count=-1, verbose=None)`

**使用例**:
```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

Xr, yr = make_regression(n_samples=300, n_features=5, noise=5.0, random_state=0)
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.25, random_state=0)

reg = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.1, verbose=False, random_seed=0,
                         loss_function="RMSEWithUncertainty", posterior_sampling=True)
reg.fit(Xrtr, yrtr)

ve = reg.virtual_ensembles_predict(Xrte[:5], prediction_type="TotalUncertainty", virtual_ensembles_count=10)
print(ve.round(3))
```
実行結果:
```
[[-6.57830e+01  1.92000e-01  3.57630e+01]
 [ 1.02121e+02  5.40000e-02  6.43520e+01]
 [ 2.32820e+01  1.28400e+00  4.04990e+01]
 [ 1.77340e+01  2.27000e-01  1.48260e+01]
 [ 1.88006e+02  2.16000e+00  7.88820e+01]]
```

**注意点・落とし穴**:
- `loss_function="RMSEWithUncertainty"`と`posterior_sampling=True`をセットで指定しないと不確実性の分解ができない(通常の`"RMSE"`損失では使えない)。
- `prediction_type="TotalUncertainty"`の戻り値は各行が「予測平均値, knowledge uncertainty(モデル由来の不確実性), data uncertainty(データ由来の不確実性)」の3列。列の意味を取り違えると誤読するので注意。
- `virtual_ensembles_count`(既定10)は「木全体をいくつの仮想アンサンブルに分割するか」を指定する。`iterations`(この例では200)を`virtual_ensembles_count`で割り切れる必要はないが、数が少なすぎると不確実性の推定が粗くなる。

---

## 13. モデルの評価・比較

### `catboost.utils.get_roc_curve(...)`

**用途**: モデルを介さず、学習済みモデルとテストデータからROC曲線(FPR・TPR・閾値の組)を直接計算する。

**シグネチャ**: `catboost.utils.get_roc_curve(model, data, thread_count=-1, plot=False)`

**使用例**:
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve

X, y = make_classification(n_samples=300, n_features=6, n_informative=3, n_classes=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

clf = CatBoostClassifier(iterations=100, verbose=False, random_seed=0)
clf.fit(Xtr, ytr)
test_pool = Pool(Xte, yte)

fpr, tpr, thresholds = get_roc_curve(clf, test_pool)
print("点の数:", len(fpr))
print("fpr[:5]:", np.round(fpr[:5], 3))
print("tpr[:5]:", np.round(tpr[:5], 3))
```
実行結果:
```
点の数: 77
fpr[:5]: [0. 0. 0. 0. 0.]
tpr[:5]: [0.    0.027 0.054 0.081 0.108]
```

**注意点・落とし穴**:
- `data`には`Pool`(または`X, y`から自前で作った`Pool`)を渡す。2値分類専用で、多クラス分類のモデルに使うとエラーになる。
- 戻り値は`(fpr, tpr, thresholds)`の3つ組で、`sklearn.metrics.roc_curve`の戻り値の順序と同じ。

### `catboost.utils.get_confusion_matrix(...)` / `select_threshold(...)`

**用途**: `get_confusion_matrix`は既定の閾値(0.5)での混同行列を計算する。`select_threshold`は目標のFPR(またはFNR)を満たす分類閾値を逆算する。

**シグネチャ**: `catboost.utils.get_confusion_matrix(model, data, thread_count=-1)` / `catboost.utils.select_threshold(model=None, data=None, curve=None, FPR=None, FNR=None, thread_count=-1)`

**使用例**:
```python
from catboost.utils import get_confusion_matrix, select_threshold

cm = get_confusion_matrix(clf, test_pool)
print("confusion_matrix:\n", cm)

best_thr = select_threshold(clf, data=test_pool, FPR=0.05)
print("FPR=0.05を満たす閾値:", best_thr)
```
実行結果:
```
confusion_matrix:
 [[35.  3.]
 [ 6. 31.]]
FPR=0.05を満たす閾値: 0.642894118853287
```

**注意点・落とし穴**:
- `get_confusion_matrix`は`sklearn.metrics.confusion_matrix`と異なり、既定の閾値0.5を内部で固定して計算する(閾値を変えたい場合は`select_threshold`で得た閾値を使い、自前で`predict_proba`から二値化する必要がある)。
- `select_threshold`は`FPR`・`FNR`のどちらか一方しか指定できない(両方同時に指定すると意図通りに動かない可能性があるため、片方のみ指定するのが安全)。

### `catboost.utils.get_fpr_curve(...)` / `get_fnr_curve(...)`

**用途**: 分類閾値を変化させたときのFPR(偽陽性率)・FNR(偽陰性率)の推移を取得する。ROC曲線を「閾値対エラー率」の軸で見たいときに使う。

**シグネチャ**: `catboost.utils.get_fpr_curve(model=None, data=None, curve=None, thread_count=-1, plot=False)` / `catboost.utils.get_fnr_curve(model=None, data=None, curve=None, thread_count=-1, plot=False)`

**使用例**:
```python
from catboost.utils import get_fpr_curve, get_fnr_curve

fpr_curve, thr_fpr = get_fpr_curve(clf, test_pool)
fnr_curve, thr_fnr = get_fnr_curve(clf, test_pool)
print("fpr_curveの点数:", len(fpr_curve))
print("fnr_curveの点数:", len(fnr_curve))
```
実行結果:
```
fpr_curveの点数: 77
fnr_curveの点数: 77
```

**注意点・落とし穴**:
- どちらも戻り値は`(カーブの値, 対応する閾値)`の2つ組で、`get_roc_curve`で得た`(fpr, tpr, thresholds)`から`curve`引数経由で再計算させることも可能(`model`/`data`の代わりに既存の`curve`を渡す)。同じ`model`/`data`に対して`get_roc_curve`を何度も呼ぶより効率的。

### `catboost.utils.compute_wx_test(...)`(モデル間の統計的比較)

**用途**: 2つのモデルのサンプルごとの損失(誤差)を比較し、その差が統計的に有意かどうかをウィルコクソンの符号順位検定で判定する。「AUCが少し上がったが、それは誤差の範囲内では?」を確認するのに使える。

**シグネチャ**: `catboost.utils.compute_wx_test(baseline, test)`(`baseline`/`test`はサンプルごとの損失値のリスト)

**使用例**:
```python
import numpy as np
from catboost.utils import compute_wx_test

clf2 = CatBoostClassifier(iterations=50, depth=8, verbose=False, random_seed=0)
clf2.fit(Xtr, ytr)

proba1 = clf.predict_proba(Xte)[:, 1]
proba2 = clf2.predict_proba(Xte)[:, 1]
eps = 1e-15
loss1 = -(yte * np.log(np.clip(proba1, eps, 1 - eps)) + (1 - yte) * np.log(np.clip(1 - proba1, eps, 1 - eps)))
loss2 = -(yte * np.log(np.clip(proba2, eps, 1 - eps)) + (1 - yte) * np.log(np.clip(1 - proba2, eps, 1 - eps)))

result = compute_wx_test(loss1.tolist(), loss2.tolist())
print(result)
```
実行結果:
```
{'pvalue': 0.03699493644923635, 'wplus': 1030.0, 'wminus': 1820.0}
```

**注意点・落とし穴**:
- `baseline`/`test`には評価指標のスコアではなく、**サンプル(行)ごとの損失値のリスト**を渡す必要がある(catboostは分類・回帰の損失を自動計算しないため、この例のようにLoglossなどを自前で算出する必要がある)。
- `CatBoostClassifier.compare(...)`という同名のインスタンスメソッドも存在するが、実行してみたところ`ImportError: No module named 'traitlets'`となり、この検証環境(`ipywidgets`/`traitlets`未インストール)ではJupyterウィジェット依存のため動作しなかった。ノートブック環境かつ関連パッケージが入っていない場合は同様のエラーになりうるため、CLIやスクリプトからの比較には本項の`compute_wx_test`や`eval_metrics`を使う方が確実。

---

## 14. アンサンブル・スタッキングとの組み合わせ

### `catboost.sum_models(...)`(モデルのブレンディング)

**用途**: 複数の学習済みcatboostモデルを、木をそのまま結合する形で1つのモデルにブレンド(加重平均)する。再学習なしで複数モデルの予測を単一モデルとして扱いたい場合に使う。

**シグネチャ**: `catboost.sum_models(models, weights=None, ctr_merge_policy='IntersectingCountersAverage')`

**使用例**:
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, sum_models

Xr, yr = make_regression(n_samples=300, n_features=5, noise=5.0, random_state=0)
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.25, random_state=0)

reg1 = CatBoostRegressor(iterations=50, depth=4, random_seed=0, verbose=False)
reg1.fit(Xrtr, yrtr)
reg2 = CatBoostRegressor(iterations=50, depth=6, random_seed=1, verbose=False)
reg2.fit(Xrtr, yrtr)

blended = sum_models([reg1, reg2], weights=[0.5, 0.5])
print(type(blended))
pred_blend = blended.predict(Xrte[:5])
pred_manual = (reg1.predict(Xrte[:5]) + reg2.predict(Xrte[:5])) / 2
print("ブレンド予測:", pred_blend.round(3))
print("手動平均:", pred_manual.round(3))
print("一致するか:", np.allclose(pred_blend, pred_manual, atol=1e-6))
```
実行結果:
```
<class 'catboost.core.CatBoost'>
ブレンド予測: [-72.775 111.284   6.566  21.234 158.849]
手動平均: [-72.775 111.284   6.566  21.234 158.849]
一致するか: True
```

**注意点・落とし穴**:
- `weights=[0.5, 0.5]`での結果は、2モデルの予測値を単純に平均した値と完全に一致することを確認した(木を連結したうえで、各木の出力に重みを掛ける形でブレンドされている)。
- `sum_models`の戻り値は`CatBoostClassifier`/`Regressor`ではなく汎用の`CatBoost`クラス(前述「モデルの保存・読み込み」の`CatBoost`と同じ)。`.score()`メソッドを持たず(`hasattr(CatBoost, 'score')`は`False`)、分類モデルをブレンドした場合`predict()`はクラスラベルではなく生スコアを返す点に注意。

### `sklearn.ensemble.StackingClassifier` との組み合わせ

**用途**: catboostモデルを他のモデル(ランダムフォレストなど)と組み合わせ、それらの予測を入力として最終予測を行うメタモデル(スタッキング)を学習する。catboostはscikit-learn互換API(`fit`/`predict`/`predict_proba`)を持つため、`StackingClassifier`にそのまま渡せる。

**シグネチャ**: `sklearn.ensemble.StackingClassifier(self, estimators, final_estimator=None, *, cv=None, stack_method='auto', n_jobs=None, passthrough=False, verbose=0)`

**使用例**:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_classes=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

cat = CatBoostClassifier(iterations=100, verbose=False, random_seed=0)
rf = RandomForestClassifier(n_estimators=100, random_state=0)

stack = StackingClassifier(
    estimators=[("catboost", cat), ("rf", rf)],
    final_estimator=LogisticRegression(),
    cv=3,
)
stack.fit(Xtr, ytr)
print("StackingClassifier score:", stack.score(Xte, yte))

cat_only = CatBoostClassifier(iterations=100, verbose=False, random_seed=0)
cat_only.fit(Xtr, ytr)
print("catboost単体 score:", cat_only.score(Xte, yte))
```
実行結果:
```
StackingClassifier score: 0.888
catboost単体 score: 0.888
```

**注意点・落とし穴**:
- catboostのモデルインスタンスをそのまま`estimators`に渡せる(特別なラッパーは不要)。`StackingClassifier`はクロスバリデーション(`cv=3`)で各foldごとに`cat`を複製して学習するため、内部的にはcatboostが複数回学習される点でコストが増える。
- 今回の検証データではスタッキングによる改善は見られなかった(catboost単体と同スコア)。スタッキングが有効かどうかはデータ・ベースモデルの多様性に依存するため、必ず単体モデルとの比較検証が必要。

### `sklearn.ensemble.VotingClassifier` との組み合わせ

**用途**: catboostと他のモデルの予測確率を平均する、より単純なアンサンブル手法(ソフト投票)。スタッキングよりシンプルで過学習しにくい。

**シグネチャ**: `sklearn.ensemble.VotingClassifier(self, estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)`

**使用例**:
```python
from sklearn.ensemble import VotingClassifier

vote = VotingClassifier(estimators=[("catboost", cat), ("rf", rf)], voting="soft")
vote.fit(Xtr, ytr)
print("VotingClassifier(soft) score:", vote.score(Xte, yte))
```
実行結果:
```
VotingClassifier(soft) score: 0.88
```

**注意点・落とし穴**:
- `voting="soft"`を使うには、`estimators`に含める全モデルが`predict_proba`を持っている必要がある(catboostの`CatBoostClassifier`は対応済み)。
- `voting="hard"`(多数決)は`predict_proba`不要だが、モデル数が偶数だと同数決になりうる点に注意(今回は2モデルなので特に`soft`が無難)。
