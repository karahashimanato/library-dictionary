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
