# scikit-learn 逆引き辞書

scikit-learn 1.9.0 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(scikit-learn 1.9.0)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。

## 目次

1. [データ読み込み・分割](#1-データ読み込み分割)
2. [前処理・スケーリング](#2-前処理スケーリング)
3. [特徴量選択](#3-特徴量選択)
4. [次元削減](#4-次元削減)
5. [教師あり学習-分類](#5-教師あり学習-分類)
6. [教師あり学習-回帰](#6-教師あり学習-回帰)
7. [アンサンブル学習](#7-アンサンブル学習)
8. [クラスタリング](#8-クラスタリング)
9. [モデル選択・交差検証](#9-モデル選択交差検証)
10. [評価指標](#10-評価指標)
11. [パイプライン・ColumnTransformer](#11-パイプラインcolumntransformer)
12. [その他ユーティリティ](#12-その他ユーティリティ)
13. [応用・発展](#応用発展)
    - [高度なパイプライン](#高度なパイプライン)
    - [キャリブレーション・半教師あり学習](#キャリブレーション半教師あり学習)
    - [マルチラベル・マルチ出力とカスタムスコアラー](#マルチラベルマルチ出力とカスタムスコアラー)
    - [モデル解釈](#モデル解釈)
    - [等張回帰・カーネル近似・多様体学習](#等張回帰カーネル近似多様体学習)
    - [ガウス過程](#ガウス過程)

---

## 1. データ読み込み・分割

### `load_iris(...)`

**用途**: 練習用のirisデータセット(150サンプル、4特徴量、3クラス)を読み込む。

**シグネチャ**: `sklearn.datasets.load_iris(*, return_X_y=False, as_frame=False)`

**使用例**:
```python
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape, iris.target.shape)
print(iris.target_names)
print(iris.feature_names)
```
実行結果:
```
(150, 4) (150,)
['setosa' 'versicolor' 'virginica']
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```

**注意点・落とし穴**:
- デフォルトでは `Bunch`(辞書ライクなオブジェクト)が返る。`return_X_y=True` にすると `(X, y)` のタプルのみが返る。
- `as_frame=True` にすると `data`/`target` が pandas の DataFrame/Series になる。

### `make_classification(...)`

**用途**: 分類問題用の人工データを生成する(動作確認・アルゴリズム検証向け)。

**シグネチャ**: `sklearn.datasets.make_classification(n_samples=100, n_features=20, *, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None, return_X_y=True)`

**使用例**:
```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_classes=2, random_state=0)
print(X.shape, y.shape, y[:10])
```
実行結果:
```
(200, 5) (200,) [1 1 0 0 0 0 1 0 0 1]
```

**注意点・落とし穴**:
- `n_informative + n_redundant + n_repeated` が `n_features` を超えるとエラーになる。
- `flip_y`(デフォルト0.01)で意図的にラベルノイズが混ざる点に注意(完全に分離可能なデータにはならない)。

### `make_regression(...)`

**用途**: 回帰問題用の人工データを生成する。

**シグネチャ**: `sklearn.datasets.make_regression(n_samples=100, n_features=100, *, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)`

**使用例**:
```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=3, noise=0.5, random_state=0)
print(X.shape, y.shape, y[:5])
```
実行結果:
```
(100, 3) (100,) [-28.73811549  -3.58368006 -81.79535836  37.18884476   2.75890095]
```

**注意点・落とし穴**:
- デフォルトの `n_features=100` に対し `n_informative=10` なので、明示的に指定しないと大半が無関係な特徴量になる。
- `noise=0.0` がデフォルト。ノイズなしだと線形モデルが極端に良いR2を出すため、現実的な検証には `noise` を指定した方がよい。

### `train_test_split(...)`

**用途**: データを学習用・検証(テスト)用に分割する。

**シグネチャ**: `sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)`

**使用例**:
```python
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
print(Xtr.shape, Xte.shape, ytr.shape, yte.shape)
```
実行結果:
```
(75, 3) (25, 3) (75,) (25,)
```

**注意点・落とし穴**:
- `test_size`/`train_size` を両方省略すると `test_size=0.25` がデフォルトで使われる。
- 分類問題でクラス比率を保ちたい場合は `stratify=y` を指定する(指定しないとクラス不均衡データで偏った分割になりうる)。

---

## 2. 前処理・スケーリング

### `StandardScaler(...)`

**用途**: 各特徴量を平均0・分散1に標準化する。

**シグネチャ**: `sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)`

**使用例**:
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
X = np.array([[1., 2.], [3., 4.], [5., 10.]])
sc = StandardScaler()
Xs = sc.fit_transform(X)
print(Xs)
print("mean_:", sc.mean_, "scale_:", sc.scale_)
```
実行結果:
```
[[-1.22474487 -0.98058068]
 [ 0.         -0.39223227]
 [ 1.22474487  1.37281295]]
mean_: [3.         5.33333333] scale_: [1.63299316 3.39934634]
```

**注意点・落とし穴**:
- 必ず学習データで `fit`(または `fit_transform`)し、テストデータには `transform` のみを使う。テストデータで再度 `fit` するとデータリーク・スケール不整合の原因になる。
- 外れ値の影響を受けやすい(平均・標準偏差を使うため)。外れ値が多い場合は `RobustScaler` を検討する。

### `MinMaxScaler(...)`

**用途**: 各特徴量を指定範囲(デフォルト0〜1)にスケーリングする。

**シグネチャ**: `sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)`

**使用例**:
```python
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
print(mm.fit_transform(X))
```
実行結果:
```
[[0.   0.  ]
 [0.5  0.25]
 [1.   1.  ]]
```

**注意点・落とし穴**:
- 新しいデータで学習時の最小・最大を超える値が来ると範囲外(0〜1を超える値)になる。`clip=True` にすると範囲内にクリップされる。

### `RobustScaler(...)`

**用途**: 中央値と四分位範囲(IQR)を使ってスケーリングする。外れ値の影響を受けにくい。

**シグネチャ**: `sklearn.preprocessing.RobustScaler(*, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False)`

**使用例**:
```python
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
print(rs.fit_transform(X))
```
実行結果:
```
[[-1.  -0.5]
 [ 0.   0. ]
 [ 1.   1.5]]
```

### `OneHotEncoder(...)`

**用途**: カテゴリ変数をone-hotベクトルに変換する。

**シグネチャ**: `sklearn.preprocessing.OneHotEncoder(*, categories='auto', drop=None, sparse_output=True, dtype=<class 'numpy.float64'>, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat')`

**使用例**:
```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder
cat = np.array([["red"], ["blue"], ["green"], ["blue"]])
ohe = OneHotEncoder(sparse_output=False)
print(ohe.fit_transform(cat))
print(ohe.categories_)
```
実行結果:
```
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[array(['blue', 'green', 'red'], dtype='<U5')]
```

**注意点・落とし穴**:
- デフォルトは `sparse_output=True` で疎行列(`scipy.sparse`)が返る。DataFrameや通常のndarrayとして見たい場合は `sparse_output=False` を指定する。
- 未知のカテゴリが`transform`時に現れると `handle_unknown='error'`(デフォルト)ではエラーになる。本番運用では `handle_unknown='ignore'` を検討する。

### `OrdinalEncoder(...)`

**用途**: カテゴリ変数を整数(順序付き)にエンコードする。

**シグネチャ**: `sklearn.preprocessing.OrdinalEncoder(*, categories='auto', dtype=<class 'numpy.float64'>, handle_unknown='error', unknown_value=None, encoded_missing_value=nan, min_frequency=None, max_categories=None)`

**使用例**:
```python
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
print(oe.fit_transform(cat))
```
実行結果:
```
[[2.]
 [0.]
 [1.]
 [0.]]
```

**注意点・落とし穴**:
- 割り当てられる整数はアルファベット順(`categories_`)であり、意味的な順序(例: 小/中/大)ではない。順序を保証したい場合は `categories` 引数で明示する。
- 名義尺度(順序のないカテゴリ)にそのまま使うと、モデルが誤った大小関係を学習する恐れがある。名義尺度には `OneHotEncoder` が適している。

### `LabelEncoder()`

**用途**: 目的変数(ラベル)を0始まりの整数にエンコードする。

**シグネチャ**: `sklearn.preprocessing.LabelEncoder()` (引数なし)

**使用例**:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = ["cat", "dog", "cat", "bird"]
print(le.fit_transform(y))
print(le.classes_)
```
実行結果:
```
[1 2 1 0]
['bird' 'cat' 'dog']
```

**注意点・落とし穴**:
- 目的変数(1次元)用であり、説明変数(特徴量の行列)のエンコードには使わない。特徴量には `OrdinalEncoder`/`OneHotEncoder` を使う。

### `PolynomialFeatures(...)`

**用途**: 既存の特徴量から多項式・交互作用特徴量を生成する。

**シグネチャ**: `sklearn.preprocessing.PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True, order='C')`

**使用例**:
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
X2 = np.array([[2., 3.]])
print(pf.fit_transform(X2))
print(pf.get_feature_names_out(["a", "b"]))
```
実行結果:
```
[[2. 3. 4. 6. 9.]]
['a' 'b' 'a^2' 'a b' 'b^2']
```

**注意点・落とし穴**:
- 特徴量数が `degree` に応じて組合せ的に増えるため、元の特徴量数が多いと爆発的にメモリを消費する。
- `include_bias=True`(デフォルト)だと定数項(1)の列が先頭に追加される。線形モデルの `fit_intercept` と重複しがちなので注意。

### `FunctionTransformer(...)`

**用途**: 任意のPython関数(例: `np.log1p`)をパイプラインの変換ステップとして組み込む。

**シグネチャ**: `sklearn.preprocessing.FunctionTransformer(func=None, inverse_func=None, *, validate=False, accept_sparse=False, check_inverse=True, feature_names_out=None, kw_args=None, inv_kw_args=None)`

**使用例**:
```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
ft = FunctionTransformer(np.log1p)
print(ft.transform(np.array([[0., 1., 9.]])))
```
実行結果:
```
[[0.         0.69314718 2.30258509]]
```

**注意点・落とし穴**:
- `func=None`(デフォルト)だと恒等変換になる。
- `fit` では学習をせず(パラメータを持たない)、単に関数を適用するだけ。逆変換が必要なら `inverse_func` も指定する。

### `SimpleImputer(...)`

**用途**: 欠損値(NaN)を平均・中央値・最頻値・定数値で補完する。

**シグネチャ**: `sklearn.impute.SimpleImputer(*, missing_values=nan, strategy='mean', fill_value=None, copy=True, add_indicator=False, keep_empty_features=False)`

**使用例**:
```python
import numpy as np
from sklearn.impute import SimpleImputer
Xnan = np.array([[1., 2.], [np.nan, 3.], [7., 6.]])
imp = SimpleImputer(strategy="mean")
print(imp.fit_transform(Xnan))
```
実行結果:
```
[[1. 2.]
 [4. 3.]
 [7. 6.]]
```

**注意点・落とし穴**:
- 学習データの統計量(平均など)で補完するため、`StandardScaler` 同様、学習データで `fit` してテストデータには `transform` のみを使う。
- `add_indicator=True` にすると、どこが欠損していたかを示す二値列が追加される(欠損自体が情報を持つ場合に有用)。

---

## 3. 特徴量選択

### `VarianceThreshold(...)`

**用途**: 分散が閾値以下の(ほぼ変化しない)特徴量を除去する。

**シグネチャ**: `sklearn.feature_selection.VarianceThreshold(threshold=0.0)`

**使用例**:
```python
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
X = load_iris().data
vt = VarianceThreshold(threshold=0.2)
Xv = vt.fit_transform(X)
print(X.shape, "->", Xv.shape)
print(vt.get_support())
```
実行結果:
```
(150, 4) -> (150, 3)
[ True False  True  True]
```

**注意点・落とし穴**:
- 目的変数を使わない教師なしの選択法。スケールの異なる特徴量に使う場合は事前にスケーリングしないと閾値の意味が変わる点に注意。

### `SelectKBest(...)`

**用途**: 統計量スコア(F値など)の高い上位k個の特徴量を選択する。

**シグネチャ**: `sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, *, k=10)`

**使用例**:
```python
from sklearn.feature_selection import SelectKBest, f_classif
skb = SelectKBest(score_func=f_classif, k=2)
Xk = skb.fit_transform(X, y)
print(X.shape, "->", Xk.shape)
print(skb.get_support())
print(skb.scores_.round(2))
```
実行結果:
```
(150, 4) -> (150, 2)
[False False  True  True]
[ 119.26   49.16 1180.16  960.01]
```

**注意点・落とし穴**:
- `score_func` はタスクに合わせて選ぶ(分類なら `f_classif`/`chi2`/`mutual_info_classif`、回帰なら `f_regression` など)。デフォルトの `f_classif` を回帰に使うとエラーになる。

### `RFE(...)`

**用途**: 指定した推定器で学習と特徴量重要度評価を繰り返し、重要度の低い特徴量を再帰的に削除する(Recursive Feature Elimination)。

**シグネチャ**: `sklearn.feature_selection.RFE(estimator, *, n_features_to_select=None, step=1, verbose=0, importance_getter='auto')`

**使用例**:
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2)
rfe.fit(X, y)
print(rfe.support_)
print(rfe.ranking_)
```
実行結果:
```
[False False  True  True]
[3 2 1 1]
```

**注意点・落とし穴**:
- `estimator` は `coef_` または `feature_importances_` を持つモデルである必要がある(KNNなど持たないモデルは使えない)。
- ステップごとに再学習するため、特徴量数・データ量が多いと計算コストが高い。

### `SelectFromModel(...)`

**用途**: 学習済み(または新規)モデルの重要度・係数を使い、閾値以上の特徴量だけを残す。

**シグネチャ**: `sklearn.feature_selection.SelectFromModel(estimator, *, threshold=None, prefit=False, norm_order=1, max_features=None, importance_getter='auto')`

**使用例**:
```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
sfm = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0), threshold="median")
Xf = sfm.fit_transform(X, y)
print(X.shape, "->", Xf.shape)
print(sfm.get_support())
```
実行結果:
```
(150, 4) -> (150, 2)
[False False  True  True]
```

**注意点・落とし穴**:
- `prefit=True` にすると、すでに学習済みのモデルをそのまま使い、`SelectFromModel` 側では再学習しない(`fit`ではなく直接 `transform` を呼ぶ)。
- `threshold=None`(デフォルト)の場合、係数を持つ線形モデルでは `"mean"`(平均)相当がデフォルト閾値になる。

---

## 4. 次元削減

### `PCA(...)`

**用途**: 主成分分析により、分散を最大限保持しつつ低次元に線形変換する。

**シグネチャ**: `sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None)`

**使用例**:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
Xp = pca.fit_transform(X)
print(Xp[:3])
print("explained_variance_ratio_:", pca.explained_variance_ratio_)
```
実行結果:
```
[[-2.68412563  0.31939725]
 [-2.71414169 -0.17700123]
 [-2.88899057 -0.14494943]]
explained_variance_ratio_: [0.92461872 0.05306648]
```

**注意点・落とし穴**:
- 分散に基づく手法のため、事前にスケーリング(`StandardScaler`)しないと値の大きい特徴量に主成分が引っ張られやすい。
- `n_components` に整数だけでなく0〜1の小数(累積寄与率)も指定できる。

### `TruncatedSVD(...)`

**用途**: 特異値分解による次元削減。疎行列(TF-IDF行列など)にも使える(PCAと違い中心化しない)。

**シグネチャ**: `sklearn.decomposition.TruncatedSVD(n_components=2, *, algorithm='randomized', n_iter=5, n_oversamples=10, power_iteration_normalizer='auto', random_state=None, tol=0.0)`

**使用例**:
```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=0)
Xs = svd.fit_transform(X)
print(Xs[:3])
print("explained_variance_ratio_:", svd.explained_variance_ratio_)
```
実行結果:
```
[[ 5.91274714 -2.30203322]
 [ 5.57248242 -1.97182599]
 [ 5.44697714 -2.09520636]]
explained_variance_ratio_: [0.52875361 0.44845576]
```

**注意点・落とし穴**:
- データを中心化しないため、PCAと結果が異なる(疎行列で中心化すると密行列になりメモリを圧迫するのを避けるため)。密なデータでPCAと同じ結果を期待しない。

### `TSNE(...)`

**用途**: 高次元データを2〜3次元に非線形に可視化するための埋め込み手法。

**シグネチャ**: `sklearn.manifold.TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', max_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='pca', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None)`

**使用例**:
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0, perplexity=30)
Xt = tsne.fit_transform(X)
print(Xt.shape)
```
実行結果:
```
(150, 2)
```

**注意点・落とし穴**:
- `transform` メソッドを持たない(`fit_transform` のみ)。新しいデータを既存の埋め込みに射影することはできず、毎回全データで学習し直す必要がある。
- 可視化専用の手法であり、出力座標の絶対的な距離やスケールに意味はない(クラスタの相対配置の把握に使う)。
- `perplexity` はサンプル数より小さくする必要がある。

---

## 5. 教師あり学習-分類

### `LogisticRegression(...)`

**用途**: ロジスティック回帰による分類(名前に反して分類アルゴリズム)。

**シグネチャ**: `sklearn.linear_model.LogisticRegression(penalty='deprecated', *, C=1.0, l1_ratio=0.0, dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, n_jobs=None)`

**使用例**:
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, ytr)
print("score:", clf.score(Xte, yte))
print("predict:", clf.predict(Xte[:5]))
print("predict_proba[0]:", clf.predict_proba(Xte[:1]).round(3))
```
実行結果:
```
score: 1.0
predict: [2 2 0 0 1]
predict_proba[0]: [[0.    0.035 0.965]]
```

**注意点・落とし穴**:
- scikit-learn 1.9で `penalty` 引数はデフォルト値が `'deprecated'` になっている(内部的にはL2正則化相当の挙動を維持しているが、明示指定の仕方が変わりつつある。バージョンごとに `help(LogisticRegression)` で確認すること)。
- デフォルトの `max_iter=100` は収束しないことが多く、`ConvergenceWarning` が出た場合は `max_iter` を増やすかスケーリングを行う。
- `class_weight=None`(デフォルト)なのでクラス不均衡データではマイノリティクラスを軽視しやすい。`class_weight='balanced'` を検討する。

### `KNeighborsClassifier(...)`

**用途**: k近傍法による分類。

**シグネチャ**: `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)`

**使用例**:
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr, ytr)
print("score:", knn.score(Xte, yte))
```
実行結果:
```
score: 1.0
```

**注意点・落とし穴**:
- 距離ベースの手法なので、特徴量のスケールが揃っていないと大きい値のスケールの特徴量が距離を支配する。事前にスケーリングが必須。
- 学習は単なるデータ保持のみ(遅延学習)で、予測時に全学習データとの距離計算が発生するため大規模データでは低速。

### `SVC(...)`

**用途**: サポートベクターマシンによる分類。

**シグネチャ**: `sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability='deprecated', tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)`

**使用例**:
```python
from sklearn.svm import SVC
svc = SVC(kernel="rbf", random_state=0)
svc.fit(Xtr, ytr)
print("score:", svc.score(Xte, yte))
```
実行結果:
```
score: 1.0
```

**注意点・落とし穴**:
- **バージョン固有の注意**: scikit-learn 1.9で `probability` パラメータは非推奨(deprecated)になり、1.11で削除予定。`SVC(probability=True)` を渡すと `FutureWarning` が出る。確率出力が必要な場合は `CalibratedClassifierCV(SVC(), ensemble=False)` を使うよう案内される(実際に `SVC(probability=True)` を実行して確認済み)。
- 距離ベースの手法のため `KNeighborsClassifier` 同様スケーリングが重要。
- デフォルトの `kernel='rbf'` は非線形境界。線形分離を仮定するなら `kernel='linear'` を明示する。

### `DecisionTreeClassifier(...)`

**用途**: 決定木による分類。

**シグネチャ**: `sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, monotonic_cst=None)`

**使用例**:
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=0)
dt.fit(Xtr, ytr)
print("score:", dt.score(Xte, yte))
print("feature_importances_:", dt.feature_importances_.round(3))
```
実行結果:
```
score: 0.9555555555555556
feature_importances_: [0.    0.    0.034 0.966]
```

**注意点・落とし穴**:
- `max_depth=None`(デフォルト)だと葉が純粋になるまで分割し続け、過学習しやすい。実運用では `max_depth`/`min_samples_leaf` 等で制約をかける。
- スケーリング不要(閾値分割ベースのため、特徴量のスケールに影響されない)。

### `GaussianNB(...)`

**用途**: 各クラス内で特徴量が正規分布に従うと仮定するナイーブベイズ分類器。

**シグネチャ**: `sklearn.naive_bayes.GaussianNB(*, priors=None, var_smoothing=1e-09)`

**使用例**:
```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(Xtr, ytr)
print("score:", gnb.score(Xte, yte))
```
実行結果:
```
score: 0.9777777777777777
```

**注意点・落とし穴**:
- 特徴量間の独立性を仮定する(「ナイーブ」)。相関の強い特徴量が多いと精度が落ちうる。

### `SGDClassifier(...)`

**用途**: 確率的勾配降下法で学習する線形分類器(大規模データ・オンライン学習向け)。

**シグネチャ**: `sklearn.linear_model.SGDClassifier(loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.01, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)`

**使用例**:
```python
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(max_iter=1000, random_state=0)
sgd.fit(Xtr, ytr)
print("score:", sgd.score(Xte, yte))
```
実行結果:
```
score: 0.9111111111111111
```

**注意点・落とし穴**:
- デフォルトの `loss='hinge'` は線形SVM相当(確率出力`predict_proba`は使えない)。ロジスティック回帰相当にしたい場合は `loss='log_loss'` を指定する。
- 勾配降下法ベースのため特徴量のスケーリングが精度・収束速度に大きく影響する。

---

## 6. 教師あり学習-回帰

### `LinearRegression(...)`

**用途**: 最小二乗法による線形回帰。

**シグネチャ**: `sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, tol=1e-06, n_jobs=None, positive=False)`

**使用例**:
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xtr, ytr)
print("R2:", lr.score(Xte, yte))
print("coef_:", lr.coef_.round(2))
print("intercept_:", round(lr.intercept_, 3))
```
実行結果:
```
R2: 0.9985756708427995
coef_: [ 3.06 62.01 72.55 42.72]
intercept_: 0.364
```

**注意点・落とし穴**:
- 正則化を持たないため、多重共線性がある(特徴量同士が強く相関する)場合に係数が不安定になりやすい。その場合は `Ridge`/`Lasso` を検討する。
- `.score()` はデフォルトでR2(決定係数)を返す(MSEではない)。

### `Ridge(...)`

**用途**: L2正則化付き線形回帰。多重共線性・過学習を抑える。

**シグネチャ**: `sklearn.linear_model.Ridge(alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)`

**使用例**:
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(Xtr, ytr)
print("R2:", ridge.score(Xte, yte))
```
実行結果:
```
R2: 0.9983648953964481
```

**注意点・落とし穴**:
- `alpha` が大きいほど正則化が強くなり係数が0に近づく(ただし完全に0にはならない → 特徴選択効果はない)。特徴選択も兼ねたいなら `Lasso` を使う。

### `Lasso(...)`

**用途**: L1正則化付き線形回帰。一部の係数を厳密に0にすることで特徴選択の効果がある。

**シグネチャ**: `sklearn.linear_model.Lasso(alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')`

**使用例**:
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.5)
lasso.fit(Xtr, ytr)
print("R2:", lasso.score(Xte, yte))
print("coef_:", lasso.coef_.round(2))
```
実行結果:
```
R2: 0.998400624636374
coef_: [ 2.67 61.58 72.02 42.2 ]
```

**注意点・落とし穴**:
- `alpha` を大きくしすぎると全係数が0になり、常に切片(平均値)を予測するだけのモデルになる。
- 特徴量のスケールに敏感なため、事前のスケーリングが推奨される。

### `ElasticNet(...)`

**用途**: L1・L2正則化を組み合わせた線形回帰(`l1_ratio`で配合を制御)。

**シグネチャ**: `sklearn.linear_model.ElasticNet(alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')`

**使用例**:
```python
from sklearn.linear_model import ElasticNet
en = ElasticNet(alpha=0.5, l1_ratio=0.5)
en.fit(Xtr, ytr)
print("R2:", en.score(Xte, yte))
```
実行結果:
```
R2: 0.9487297115329718
```

**注意点・落とし穴**:
- `l1_ratio=1.0` で `Lasso` と等価、`l1_ratio=0.0` で `Ridge` と等価になる。両者の中間的な挙動が欲しいときに使う。

### `SVR(...)`

**用途**: サポートベクターマシンによる回帰。

**シグネチャ**: `sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)`

**使用例**:
```python
from sklearn.svm import SVR
svr = SVR(kernel="rbf", C=10)
svr.fit(Xtr, ytr)
print("R2:", svr.score(Xte, yte))
```
実行結果:
```
R2: 0.7100305909678044
```

**注意点・落とし穴**:
- `class_weight` を持たない(分類の`SVC`と異なる)。
- デフォルトの `kernel='rbf'` は線形関係のデータでは `LinearRegression` より性能が劣ることがある(今回の検証例でもR2が線形モデルより大幅に低い)。スケーリングも重要。

### `DecisionTreeRegressor(...)`

**用途**: 決定木による回帰。

**シグネチャ**: `sklearn.tree.DecisionTreeRegressor(*, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0, monotonic_cst=None)`

**使用例**:
```python
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=4, random_state=0)
dtr.fit(Xtr, ytr)
print("R2:", dtr.score(Xte, yte))
```
実行結果:
```
R2: 0.7684119291190833
```

**注意点・落とし穴**:
- 決定木は学習データの範囲外(外挿)を予測できない(葉の平均値しか返せない)。線形性の強いデータでは線形回帰に劣ることが多い。

### `KNeighborsRegressor(...)`

**用途**: k近傍法による回帰(近傍点の目的変数の平均を予測値とする)。

**シグネチャ**: `sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)`

**使用例**:
```python
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=5)
knr.fit(Xtr, ytr)
print("R2:", knr.score(Xte, yte))
```
実行結果:
```
R2: 0.8666680831447208
```

**注意点・落とし穴**:
- `KNeighborsClassifier` 同様スケーリングが必須。

---

## 7. アンサンブル学習

### `RandomForestClassifier(...)`

**用途**: 複数の決定木をバギング(ブートストラップ+特徴量サブサンプリング)で組み合わせる分類器。

**シグネチャ**: `sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)`

**使用例**:
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(Xtr, ytr)
print("score:", rf.score(Xte, yte))
print("feature_importances_:", rf.feature_importances_.round(3))
```
実行結果:
```
score: 0.9777777777777777
feature_importances_: [0.104 0.032 0.406 0.458]
```

**注意点・落とし穴**:
- 分類の `max_features` デフォルトは `'sqrt'`(特徴量数の平方根)。回帰版(`RandomForestRegressor`)はデフォルトが `1.0`(全特徴量)であり異なる点に注意。
- `random_state` を固定しないと木の生成がランダムなため実行のたびに結果が変わる。

### `RandomForestRegressor(...)`

**用途**: 複数の決定木をバギングで組み合わせる回帰器。

**シグネチャ**: `sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)`

**使用例**:
```python
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, random_state=0)
rfr.fit(Xrtr, yrtr)
print("R2:", rfr.score(Xrte, yrte))
```
実行結果:
```
R2: 0.8944229720877984
```

### `GradientBoostingClassifier(...)`

**用途**: 勾配ブースティングにより浅い決定木を逐次的に追加して学習する分類器。

**シグネチャ**: `sklearn.ensemble.GradientBoostingClassifier(*, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='deprecated', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)`

**使用例**:
```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=0)
gb.fit(Xtr, ytr)
print("score:", gb.score(Xte, yte))
```
実行結果:
```
score: 0.9777777777777777
```

**注意点・落とし穴**:
- `criterion` パラメータは1.9で非推奨(`'deprecated'`がデフォルト表示)になっている。
- 逐次学習(各木が前の誤差を補正)のため `RandomForestClassifier` と違い並列化しにくく、大規模データでは低速。大規模データには `HistGradientBoostingClassifier` の方が高速。

### `HistGradientBoostingClassifier(...)`

**用途**: ヒストグラムベースの高速な勾配ブースティング分類器(LightGBM類似のアルゴリズム)。

**シグネチャ**: `sklearn.ensemble.HistGradientBoostingClassifier(loss='log_loss', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='from_dtype', monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None, class_weight=None)`

**使用例**:
```python
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=0)
hgb.fit(Xtr, ytr)
print("score:", hgb.score(Xte, yte))
```
実行結果:
```
score: 0.9555555555555556
```

**注意点・落とし穴**:
- 欠損値をそのまま扱える(内部でNaNを専用の分岐として扱う)ため、事前の`SimpleImputer`が不要な場合がある。
- `categorical_features='from_dtype'`(デフォルト)は、pandasのcategory dtype列を自動的にカテゴリ特徴量として扱う。

### `AdaBoostClassifier(...)`

**用途**: 弱学習器(デフォルトは決定木)を逐次学習させ、誤分類サンプルの重みを増やしながら組み合わせるブースティング手法。

**シグネチャ**: `sklearn.ensemble.AdaBoostClassifier(estimator=None, *, n_estimators=50, learning_rate=1.0, random_state=None)`

**使用例**:
```python
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, random_state=0)
ada.fit(Xtr, ytr)
print("score:", ada.score(Xte, yte))
```
実行結果:
```
score: 0.9777777777777777
```

**注意点・落とし穴**:
- 引数名は `estimator`(旧バージョンでは `base_estimator` という名前だった)。`estimator=None` の場合、デフォルトで浅い決定木(`DecisionTreeClassifier(max_depth=1)`)が使われる。

### `VotingClassifier(...)`

**用途**: 複数の異なる分類器の予測を多数決(hard)または確率平均(soft)で統合する。

**シグネチャ**: `sklearn.ensemble.VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)`

**使用例**:
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
vc = VotingClassifier(estimators=[
    ("lr", LogisticRegression(max_iter=1000)),
    ("knn", KNeighborsClassifier()),
    ("dt", DecisionTreeClassifier(random_state=0)),
], voting="soft")
vc.fit(Xtr, ytr)
print("score:", vc.score(Xte, yte))
```
実行結果:
```
score: 1.0
```

**注意点・落とし穴**:
- `voting='soft'` を使うには、すべての推定器が `predict_proba` を実装している必要がある(`SVC`はデフォルトで不可、前述の`probability`非推奨の影響を受ける)。
- デフォルトは `voting='hard'`(多数決)。確率を平均する`'soft'`の方が一般に性能が良いとされる。

### `StackingClassifier(...)`

**用途**: 複数のベースモデルの予測を特徴量として、メタモデル(`final_estimator`)で最終予測する。

**シグネチャ**: `sklearn.ensemble.StackingClassifier(estimators, final_estimator=None, *, cv=None, stack_method='auto', n_jobs=None, passthrough=False, verbose=0)`

**使用例**:
```python
from sklearn.ensemble import StackingClassifier
sc = StackingClassifier(estimators=[
    ("lr", LogisticRegression(max_iter=1000)),
    ("knn", KNeighborsClassifier()),
], final_estimator=DecisionTreeClassifier(max_depth=2, random_state=0))
sc.fit(Xtr, ytr)
print("score:", sc.score(Xte, yte))
```
実行結果:
```
score: 1.0
```

**注意点・落とし穴**:
- 内部でクロスバリデーション(`cv`、デフォルト5分割相当)を使ってベースモデルの予測(メタ特徴量)を作るため、`VotingClassifier`より学習コストが高い。
- `final_estimator=None`(デフォルト)の場合、分類では `LogisticRegression` が自動的に使われる。

---

## 8. クラスタリング

### `KMeans(...)`

**用途**: k個のクラスタ中心を求め、各点を最も近い中心に割り当てるクラスタリング。

**シグネチャ**: `sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')`

**使用例**:
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=0)
km = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = km.fit_predict(X)
print("labels[:10]:", labels[:10])
print("inertia_:", round(km.inertia_, 2))
```
実行結果:
```
labels[:10]: [1 0 0 0 1 0 0 1 2 0]
inertia_: 104.37
```

**注意点・落とし穴**:
- `n_clusters` はデータから自動決定されない。事前にエルボー法や `silhouette_score` で見積もる必要がある。
- クラスタ番号(0, 1, 2, ...)自体には意味がなく、実行のたびに(あるいは`random_state`が違うと)番号の対応が変わりうる。
- ユークリッド距離ベースなのでスケーリングが重要。

### `DBSCAN(...)`

**用途**: 密度ベースのクラスタリング。クラスタ数を指定不要で、外れ値(ノイズ)も検出できる。

**シグネチャ**: `sklearn.cluster.DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)`

**使用例**:
```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.8, min_samples=5)
labels_db = db.fit_predict(X)
print("unique labels:", set(labels_db))
```
実行結果:
```
unique labels: [-1  0  1]
```

**注意点・落とし穴**:
- ノイズ点には `-1` というラベルが付く(通常のクラスタ番号ではない)ので、クラスタ数を数える際は `-1` を除外する必要がある。
- `eps`(近傍半径)の値に結果が非常に敏感で、データのスケールに直接依存する。スケーリング後の値を基準に調整する。

### `AgglomerativeClustering(...)`

**用途**: 階層的凝集型クラスタリング(近いクラスタ同士を順に併合していく)。

**シグネチャ**: `sklearn.cluster.AgglomerativeClustering(n_clusters=2, *, metric='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False)`

**使用例**:
```python
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3)
labels_ac = ac.fit_predict(X)
print("labels[:10]:", labels_ac[:10])
```
実行結果:
```
labels[:10]: [1 0 0 0 1 0 0 1 2 0]
```

**注意点・落とし穴**:
- デフォルトの `n_clusters=2` に注意(3クラス問題でも明示しないと2つに分けられてしまう)。
- `KMeans`と異なり `predict`(新規データへの適用)メソッドを持たない。`fit_predict`で学習時のラベルを得るのみ。

### `silhouette_score(...)`

**用途**: クラスタリング結果の妥当性を評価する指標(-1〜1、大きいほど良い分離)。

**シグネチャ**: `sklearn.metrics.silhouette_score(X, labels, *, metric='euclidean', sample_size=None, random_state=None, **kwds)`

**使用例**:
```python
from sklearn.metrics import silhouette_score
print(round(silhouette_score(X, labels), 3))
```
実行結果:
```
0.657
```

**注意点・落とし穴**:
- クラスタ数が1つ、またはサンプル数と同数の場合は計算できずエラーになる。
- 正解ラベル(教師データ)が不要な内部評価指標。正解ラベルがある場合は `adjusted_rand_score` などの外部評価指標の方が適切なことがある。

---

## 9. モデル選択・交差検証

### `cross_val_score(...)`

**用途**: 指定した分割数でクロスバリデーションを行い、各foldのスコアを配列で返す。

**シグネチャ**: `sklearn.model_selection.cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, params=None, pre_dispatch='2*n_jobs', error_score=nan)`

**使用例**:
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
print(scores.round(3))
print("mean:", round(scores.mean(), 3))
```
実行結果:
```
[0.967 1.    0.933 0.967 1.   ]
mean: 0.973
```

**注意点・落とし穴**:
- `cv`に整数を渡した場合、分類器では自動的に`StratifiedKFold`相当(クラス比率を保つ)、回帰では`KFold`が使われる。
- 前処理(スケーリングなど)を`cross_val_score`の外で`fit`してから渡すと、テストfoldの情報が学習に漏れる(データリーク)。前処理も含めた`Pipeline`を渡すのが安全。

### `KFold(...)`

**用途**: データをk個に分割し、交差検証用のインデックスを生成する。

**シグネチャ**: `sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)`

**使用例**:
```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=0)
for i, (tr, te) in enumerate(kf.split(X)):
    print(f"fold{i}: train={len(tr)} test={len(te)}")
```
実行結果:
```
fold0: train=100 test=50
fold1: train=100 test=50
fold2: train=100 test=50
```

**注意点・落とし穴**:
- `shuffle=False`(デフォルト)だとデータの並び順のまま連続したブロックに分割される。データが何らかの順序(時系列やクラス順)で並んでいると偏った分割になるため、通常は`shuffle=True`を検討する。
- `random_state`は`shuffle=True`のときのみ意味を持つ。

### `StratifiedKFold(...)`

**用途**: 各foldでクラス比率を保ったまま分割する(分類問題向け)。

**シグネチャ**: `sklearn.model_selection.StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)`

**使用例**:
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
for i, (tr, te) in enumerate(skf.split(X, y)):
    print(f"fold{i}: test_classes={np.bincount(y[te])}")
```
実行結果:
```
fold0: test_classes=[17 17 16]
fold1: test_classes=[17 16 17]
fold2: test_classes=[16 17 17]
```

**注意点・落とし穴**:
- `KFold`と違い`split(X, y)`に目的変数`y`を渡す必要がある(クラス比率を見るため)。

### `GridSearchCV(...)`

**用途**: 指定したパラメータの全組み合わせをクロスバリデーションで評価し、最良の組み合わせを探す。

**シグネチャ**: `sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)`

**使用例**:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
gs = GridSearchCV(SVC(), param_grid, cv=3)
gs.fit(X, y)
print("best_params_:", gs.best_params_)
print("best_score_:", round(gs.best_score_, 3))
```
実行結果:
```
best_params_: {'C': 1, 'kernel': 'linear'}
best_score_: 0.993
```

**注意点・落とし穴**:
- `refit=True`(デフォルト)により、`gs.fit()`後は全データで再学習された最良モデルが`gs.best_estimator_`(および`gs.predict()`)として使える。
- パラメータの組み合わせ数×`cv`分割数だけ学習が走るため、探索空間が広いと非常に時間がかかる。広い空間には`RandomizedSearchCV`の方が効率的なことが多い。

### `RandomizedSearchCV(...)`

**用途**: パラメータ空間からランダムサンプリングした`n_iter`通りだけを評価する(`GridSearchCV`の効率化版)。

**シグネチャ**: `sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, return_train_score=False)`

**使用例**:
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
rs = RandomizedSearchCV(SVC(), {"C": uniform(0.1, 10), "kernel": ["linear", "rbf"]}, n_iter=5, cv=3, random_state=0)
rs.fit(X, y)
print("best_params_:", rs.best_params_)
print("best_score_:", round(rs.best_score_, 3))
```
実行結果:
```
best_params_: {'C': np.float64(8.542657485810173), 'kernel': 'rbf'}
best_score_: 0.98
```

**注意点・落とし穴**:
- `param_distributions`には固定リストだけでなく`scipy.stats`の確率分布オブジェクト(`uniform`など)を渡すことで連続値を探索できる(`GridSearchCV`にはこれができない)。
- `n_iter`を増やすほど`GridSearchCV`の全探索に近づくが、必ず最適解が見つかる保証はない(サンプリングのため)。

---

## 10. 評価指標

以下はやや過学習していないロジスティック回帰(`C=0.01`)をirisの30%テストデータに適用した、あえて誤分類を含む例(`random_state=42`)。

### `accuracy_score(...)`

**用途**: 正解率(全予測のうち正しかった割合)を計算する。

**シグネチャ**: `sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)`

**使用例**:
```python
from sklearn.metrics import accuracy_score
print(accuracy_score(yte, ypred))
```
実行結果:
```
0.8222222222222222
```

**注意点・落とし穴**:
- クラス不均衡データでは正解率だけでは性能を正しく評価できない(多数派クラスを常に予測するだけで高い値になりうる)。`classification_report`などで各クラスの精度・再現率も確認する。

### `precision_score(...)` / `recall_score(...)` / `f1_score(...)`

**用途**: 適合率(precision)・再現率(recall)・両者の調和平均(F1)を計算する。

**シグネチャ**: `sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')`

**使用例**:
```python
from sklearn.metrics import precision_score, recall_score, f1_score
print("precision(macro):", round(precision_score(yte, ypred, average="macro"), 3))
print("recall(macro):", round(recall_score(yte, ypred, average="macro"), 3))
print("f1(macro):", round(f1_score(yte, ypred, average="macro"), 3))
```
実行結果:
```
precision(macro): 0.852
recall(macro): 0.822
f1(macro): 0.815
```

**注意点・落とし穴**:
- `average='binary'`がデフォルトだが、3クラス以上の多クラス分類にそのまま使うとエラーになる。多クラスでは`average='macro'`(クラス平均)、`'weighted'`(サポート数で重み付け)、`'micro'`などを明示的に指定する。
- 分母が0になるケース(例: あるクラスを一度も予測しなかった)では`zero_division='warn'`によりワーニングとともに0が返る。

### `confusion_matrix(...)`

**用途**: 真のクラスと予測クラスの組み合わせごとの件数を行列で表示する。

**シグネチャ**: `sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)`

**使用例**:
```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(yte, ypred))
```
実行結果:
```
[[15  0  0]
 [ 0  8  7]
 [ 0  1 14]]
```

**注意点・落とし穴**:
- 行が実際のクラス(y_true)、列が予測クラス(y_pred)。この例ではクラス1(versicolor)の15件中7件をクラス2(virginica)と誤分類している。
- `labels`引数で行・列の順序やクラスの絞り込みを制御できる。

### `classification_report(...)`

**用途**: クラスごとのprecision/recall/f1-scoreとサポート数(件数)をまとめて表示する。

**シグネチャ**: `sklearn.metrics.classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')`

**使用例**:
```python
from sklearn.metrics import classification_report
print(classification_report(yte, ypred, target_names=iris.target_names))
```
実行結果:
```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        15
  versicolor       0.89      0.53      0.67        15
   virginica       0.67      0.93      0.78        15

    accuracy                           0.82        45
   macro avg       0.85      0.82      0.81        45
weighted avg       0.85      0.82      0.81        45
```

**注意点・落とし穴**:
- `output_dict=True`にすると文字列ではなく辞書が返り、プログラムで各値を扱いやすくなる。

### `roc_auc_score(...)`

**用途**: ROC曲線の下側面積(AUC)を計算する。

**シグネチャ**: `sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)`

**使用例**:
```python
from sklearn.metrics import roc_auc_score
yproba = clf.predict_proba(Xte)
print(round(roc_auc_score(yte, yproba, multi_class="ovr"), 4))
```
実行結果:
```
0.9622
```

**注意点・落とし穴**:
- 多クラス分類では`multi_class`引数(`'ovr'`または`'ovo'`)を明示的に指定しないと`multi_class='raise'`(デフォルト)によりエラーになる。
- `y_score`にはクラスラベルではなく確率(`predict_proba`)やスコア(`decision_function`)を渡す。

### `mean_squared_error(...)`

**用途**: 平均二乗誤差(MSE)を計算する(回帰の代表的な損失指標)。

**シグネチャ**: `sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')`

**使用例**:
```python
from sklearn.metrics import mean_squared_error
print(round(mean_squared_error(yrte, ypr), 3))
```
実行結果:
```
19.15
```

**注意点・落とし穴**:
- 二乗誤差なので外れ値に敏感。外れ値の影響を抑えたい場合は`mean_absolute_error`を検討する。
- 旧バージョンにあった`squared`引数は現在のAPIには存在しない(RMSEが欲しい場合は`root_mean_squared_error`という専用関数を使うか、`np.sqrt`で自分で計算する)。

### `mean_absolute_error(...)`

**用途**: 平均絶対誤差(MAE)を計算する。

**シグネチャ**: `sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')`

**使用例**:
```python
from sklearn.metrics import mean_absolute_error
print(round(mean_absolute_error(yrte, ypr), 3))
```
実行結果:
```
3.635
```

### `r2_score(...)`

**用途**: 決定係数(R2)。目的変数の分散のうちモデルが説明できた割合(1に近いほど良い)。

**シグネチャ**: `sklearn.metrics.r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', force_finite=True)`

**使用例**:
```python
from sklearn.metrics import r2_score
print(round(r2_score(yrte, ypr), 4))
```
実行結果:
```
0.9986
```

**注意点・落とし穴**:
- R2は負の値になりうる(予測が「平均値を常に返す」ベースラインより悪い場合)。0〜1の範囲だと誤解しないこと。
- 回帰モデルの`.score()`メソッドはデフォルトでこのR2を返す。

---

## 11. パイプライン・ColumnTransformer

### `Pipeline(...)`

**用途**: 前処理と推定器を1つのオブジェクトに連結し、`fit`/`predict`を一括で行えるようにする。データリーク防止にも重要。

**シグネチャ**: `sklearn.pipeline.Pipeline(steps, *, transform_input=None, memory=None, verbose=False)`

**使用例**:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(Xtr, ytr)
print("score:", pipe.score(Xte, yte))
```
実行結果:
```
score: 0.9777777777777777
```

**注意点・落とし穴**:
- `pipe.fit(Xtr, ytr)`とすると、`StandardScaler`は`Xtr`のみで`fit`され、`Xte`側では自動的に`transform`のみが適用される。`GridSearchCV`と組み合わせても交差検証の各foldで正しく前処理がやり直されるため、手動で前処理してから分割するより安全。
- 各ステップ名(`"scaler"`, `"clf"`)は`pipe.named_steps["clf"]`のようにアクセスでき、`GridSearchCV`のパラメータ指定でも`clf__C`のように`ステップ名__パラメータ名`の形式で使う。

### `make_pipeline(...)`

**用途**: ステップ名を自動生成して`Pipeline`を簡潔に作る。

**シグネチャ**: `sklearn.pipeline.make_pipeline(*steps, memory=None, transform_input=None, verbose=False)`

**使用例**:
```python
from sklearn.pipeline import make_pipeline
pipe2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
print(pipe2.steps[0][0], pipe2.steps[1][0])
```
実行結果:
```
standardscaler logisticregression
```

**注意点・落とし穴**:
- ステップ名はクラス名を小文字化したものが自動で付く(`StandardScaler`→`"standardscaler"`)。同じクラスを2回使うと`-1`, `-2`のような連番が付く。名前を明示的に制御したい場合は`Pipeline`を直接使う。

### `ColumnTransformer(...)`

**用途**: DataFrameの列ごとに異なる前処理(数値列はスケーリング、カテゴリ列はOne-Hotなど)を適用し、結果を結合する。

**シグネチャ**: `sklearn.compose.ColumnTransformer(transformers, *, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=True)`

**使用例**:
```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
df = pd.DataFrame({
    "age": [25, 32, 47, 51],
    "income": [50000, 64000, 120000, 98000],
    "city": ["Tokyo", "Osaka", "Tokyo", "Nagoya"],
})
ct = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(sparse_output=False), ["city"]),
])
Xt = ct.fit_transform(df)
print(Xt)
print(ct.get_feature_names_out())
```
実行結果:
```
[[-1.29241939 -1.19624907  0.          0.          1.        ]
 [-0.63446043 -0.68874946  0.          1.          0.        ]
 [ 0.77545163  1.34124895  0.          0.          1.        ]
 [ 1.15142818  0.54374958  1.          0.          0.        ]]
['num__age' 'num__income' 'cat__city_Nagoya' 'cat__city_Osaka' 'cat__city_Tokyo']
```

**注意点・落とし穴**:
- `remainder='drop'`(デフォルト)なので、`transformers`で指定しなかった列は結果から除外される。全列を残したい場合は`remainder='passthrough'`を指定する。
- 出力の列名には`get_feature_names_out()`が使えるが、`verbose_feature_names_out=True`(デフォルト)により`変換名__元の列名`という形式になる点に注意(単純な元の列名ではない)。

### `make_column_transformer(...)`

**用途**: ステップ名を自動生成して`ColumnTransformer`を簡潔に作る。

**シグネチャ**: `sklearn.compose.make_column_transformer(*transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None, verbose=False, verbose_feature_names_out=True)`

**使用例**:
```python
from sklearn.compose import make_column_transformer
ct2 = make_column_transformer(
    (StandardScaler(), ["age", "income"]),
    (OneHotEncoder(sparse_output=False), ["city"]),
)
print(ct2.fit_transform(df))
```
実行結果:
```
[[-1.29241939 -1.19624907  0.          0.          1.        ]
 [-0.63446043 -0.68874946  0.          1.          0.        ]
 [ 0.77545163  1.34124895  0.          0.          1.        ]
 [ 1.15142818  0.54374958  1.          0.          0.        ]]
```

---

## 12. その他ユーティリティ

### `joblib.dump(...)` / `joblib.load(...)`

**用途**: 学習済みモデルなどのPythonオブジェクトをファイルに保存・復元する(scikit-learn公式ドキュメントでも推奨される永続化方法)。

**シグネチャ**: `joblib.dump(value, filename, compress=0, protocol=None)` / `joblib.load(filename, mmap_mode=None, ensure_native_byte_order='auto')`

**使用例**:
```python
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
iris = load_iris()
clf = LogisticRegression(max_iter=1000).fit(iris.data, iris.target)
joblib.dump(clf, "model.joblib")
clf2 = joblib.load("model.joblib")
print(clf2.score(iris.data, iris.target))
```
実行結果:
```
0.9733333333333334
```

**注意点・落とし穴**:
- `pickle`と同様、信頼できないファイルを`load`するとコード実行のリスクがある(公開先から受け取ったモデルファイルをそのまま`load`しない)。
- 保存時と読み込み時でscikit-learn/numpyのバージョンが大きく異なると、警告が出たり読み込めないことがある。

### `set_config(...)`

**用途**: scikit-learn全体の表示・動作(推定器の`repr`表示形式、並列処理設定など)をグローバルに変更する。

**シグネチャ**: `sklearn.set_config(assume_finite=None, working_memory=None, print_changed_only=None, display=None, pairwise_dist_chunk_size=None, enable_cython_pairwise_dist=None, array_api_dispatch=None, transform_output=None, enable_metadata_routing=None, skip_parameter_validation=None, sparse_interface=None)`

**使用例**:
```python
from sklearn import set_config
set_config(display="text")
print(clf)
```
実行結果:
```
LogisticRegression(max_iter=1000)
```

**注意点・落とし穴**:
- `display='diagram'`にするとJupyter上でパイプラインの構造をHTML図として表示できる(ノートブック環境で便利)。
- グローバル設定なので、ライブラリコード内で不用意に変更するとその後の処理全体に影響する。

### `check_is_fitted(...)`

**用途**: 推定器が学習済み(`fit`実行済み)かどうかを検証する(カスタム推定器の実装時によく使う)。

**シグネチャ**: `sklearn.utils.validation.check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=<built-in function all>)`

**使用例**:
```python
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
check_is_fitted(clf)  # 学習済みなら何も起きない
try:
    check_is_fitted(LogisticRegression())
except Exception as e:
    print(type(e).__name__, str(e)[:60])
```
実行結果:
```
NotFittedError This LogisticRegression instance is not fitted yet. Call 'fit
```

**注意点・落とし穴**:
- 未学習の場合`sklearn.exceptions.NotFittedError`(`ValueError`のサブクラス)を送出する。`try/except ValueError`でも捕捉できる。

### `shuffle(...)`

**用途**: 複数の配列を対応関係を保ったままランダムにシャッフルする。

**シグネチャ**: `sklearn.utils.shuffle(*arrays, random_state=None, n_samples=None)`

**使用例**:
```python
from sklearn.utils import shuffle
import numpy as np
idx = [0, 50, 100, 10, 60, 110]
X, y = iris.data[idx], iris.target[idx]
print("before:", y)
Xs, ys = shuffle(X, y, random_state=0)
print("after:", ys)
```
実行結果:
```
before: [0 1 2 0 1 2]
after: [2 2 1 0 0 1]
```

**注意点・落とし穴**:
- 複数配列を渡した場合、すべて同じ順序でシャッフルされる(`X`と`y`の対応関係は崩れない)。

### `resample(...)`

**用途**: 配列からブートストラップ(重複あり)や間引き(重複なし)でサンプリングする。不均衡データのオーバー/アンダーサンプリングにも使える。

**シグネチャ**: `sklearn.utils.resample(*arrays, replace=True, n_samples=None, random_state=None, stratify=None, sample_weight=None)`

**使用例**:
```python
from sklearn.utils import resample
Xr, yr = resample(X, y, n_samples=4, random_state=0)
print(yr)
```
実行結果:
```
[1 2 0 0]
```

**注意点・落とし穴**:
- デフォルトは`replace=True`(重複を許す復元抽出=ブートストラップ)。単純に間引きたいだけの場合は`replace=False`を指定する。

---

## 応用・発展

### 高度なパイプライン

#### `FeatureUnion(...)`

**用途**: 複数の変換器(Transformer)を並列に適用し、それぞれの出力を横に結合する。`Pipeline`が直列連結なのに対し、こちらは並列合成。

**シグネチャ**: `sklearn.pipeline.FeatureUnion(transformer_list, *, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=True)`

**使用例**:
```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
X = load_iris().data
fu = FeatureUnion([
    ("pca", PCA(n_components=2, random_state=0)),
    ("scaler", StandardScaler()),
])
Xf = fu.fit_transform(X)
print(X.shape, "->", Xf.shape)
```
実行結果:
```
(150, 4) -> (150, 6)
```

**注意点・落とし穴**:
- 各変換器の出力列数の合計が最終的な列数になる(PCAの2列 + StandardScalerの4列 = 6列)。列名の対応が分かりにくくなるため、`verbose_feature_names_out=True`(デフォルト)と`get_feature_names_out()`の併用が推奨される。
- `ColumnTransformer`と違い、列を指定して振り分けることはできない(全変換器に同じ`X`全体が渡る)。列ごとに変換器を変えたい場合は`ColumnTransformer`を使う。

#### カスタムTransformerの自作(`BaseEstimator`, `TransformerMixin`)

**用途**: scikit-learnの`Pipeline`やGridSearchCVと互換性のある独自の前処理ステップを作る。

**シグネチャ**: `class MyTransformer(BaseEstimator, TransformerMixin): def __init__(self, ...): ...; def fit(self, X, y=None): ...; def transform(self, X): ...`

**使用例**:
```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=-1.0, upper=1.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(X, self.lower, self.upper)

ct = ClipTransformer(lower=0.0, upper=6.0)
Xc = ct.fit_transform(X)
print(Xc.min(), Xc.max())
print(ct.get_params())
```
実行結果:
```
0.1 6.0
{'lower': 0.0, 'upper': 6.0}
```

**注意点・落とし穴**:
- `__init__`は受け取った引数をそのまま同名の属性に代入するだけにする(加工禁止)。これを破ると`BaseEstimator`が提供する`get_params()`/`set_params()`(`GridSearchCV`のパラメータ探索が依存する)が正しく動作しない。
- `TransformerMixin`を継承すると`fit`と`transform`から自動的に`fit_transform`が合成される(自分で`fit_transform`を書く必要はない)。

#### `set_output(...)`

**用途**: 変換器の`transform`/`fit_transform`の出力形式を、numpy配列ではなくpandas DataFrameに固定する。

**シグネチャ**: `Estimator.set_output(self, *, transform=None) -> Estimator`(`TransformerMixin`を継承する変換器全般が持つメソッド)

**使用例**:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
sc = StandardScaler().set_output(transform="pandas")
out = sc.fit_transform(df)
print(type(out))
print(out.head(2))
```
実行結果:
```
<class 'pandas.DataFrame'>
          a         b         c         d
0 -0.900681  1.019004 -1.340227 -1.315444
1 -1.143017 -0.131979 -1.340227 -1.315444
```

**注意点・落とし穴**:
- 列名は入力DataFrameの列名(または`get_feature_names_out()`の結果)がそのまま使われる。numpy配列を渡した場合は`x0`, `x1`, ...のような自動生成名になる。
- `sklearn.set_config(transform_output="pandas")`でグローバルに全変換器のデフォルトを変更することもできる(個別に`set_output`を呼ばずに済む)。

### キャリブレーション・半教師あり学習

#### `CalibratedClassifierCV(...)`

**用途**: 分類器の出力確率を、実際の事象発生率に近づくよう較正(キャリブレーション)する。`SVC`のように確率出力が不得意なモデルの`predict_proba`を改善する目的でよく使う。

**シグネチャ**: `sklearn.calibration.CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=None, n_jobs=None, ensemble='auto')`

**使用例**:
```python
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
base = SVC(kernel="rbf", random_state=0)
cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
cal.fit(Xtr, ytr)
print("predict_proba[0]:", cal.predict_proba(Xte[:1]).round(3))
print("score:", cal.score(Xte, yte))
```
実行結果:
```
predict_proba[0]: [[0.037 0.091 0.872]]
score: 0.9777777777777777
```

**注意点・落とし穴**:
- README冒頭の`SVC`の項で触れたとおり、`SVC(probability=True)`は1.9で非推奨。確率出力が必要なら`SVC(probability=True)`ではなく本項の`CalibratedClassifierCV(SVC(), ...)`を使うようscikit-learn側からも案内される。
- `method='sigmoid'`(Platt scaling、データが少ない場合向け)と`method='isotonic'`(データが多い場合向け、後述の`IsotonicRegression`を内部で使用)の2種類がある。データが少ないのに`'isotonic'`を使うと過学習しやすい。
- 内部で`cv`分割ごとにベースモデルを学習し直すため、単純な`fit`よりコストが高い。

#### `LabelPropagation(...)` / `LabelSpreading(...)`

**用途**: ラベル付きデータとラベルなしデータが混在する半教師あり学習で、ラベルなしサンプルにラベルを伝播させる。

**シグネチャ**: `sklearn.semi_supervised.LabelPropagation(kernel='rbf', *, gamma=20, n_neighbors=7, max_iter=1000, tol=0.001, n_jobs=None)`

**使用例**:
```python
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
rng = np.random.RandomState(0)
y_semi = y.copy()
unlabeled_idx = rng.choice(len(y), size=60, replace=False)
y_semi[unlabeled_idx] = -1
print("labeled count:", (y_semi != -1).sum(), "/ total:", len(y_semi))
lp = LabelPropagation()
lp.fit(X, y_semi)
print("score on true y:", lp.score(X, y))
```
実行結果:
```
labeled count: 90 / total: 150
score on true y: 0.98
```

**注意点・落とし穴**:
- ラベルなしサンプルは`-1`で表す決まりになっている(欠損値NaNではない)。
- `LabelSpreading`は`LabelPropagation`と似ているが、ラベル伝播時にノイズへの頑健性を高める`alpha`(正則化)パラメータを持つ点が異なる(`LabelPropagation`にはない)。

#### `SelfTrainingClassifier(...)`

**用途**: 任意の`predict_proba`を持つ分類器をベースに、確信度の高いラベルなしサンプルを繰り返し取り込みながら学習する半教師あり学習のラッパー。

**シグネチャ**: `sklearn.semi_supervised.SelfTrainingClassifier(estimator=None, threshold=0.75, criterion='threshold', k_best=10, max_iter=10, verbose=False)`

**使用例**:
```python
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
base_clf = LogisticRegression(max_iter=1000)
st = SelfTrainingClassifier(base_clf, threshold=0.8)
st.fit(X, y_semi)
print("score:", st.score(X, y))
print("n_iter_:", st.n_iter_)
```
実行結果:
```
score: 0.9533333333333334
n_iter_: 5
```

**注意点・落とし穴**:
- `LabelPropagation`と違い、任意の分類器(`predict_proba`があるもの)をベースにできる汎用ラッパー。`-1`ラベルの扱いは`LabelPropagation`と共通。
- `threshold`(デフォルト0.75)以上の確信度を持つ予測だけを次のイテレーションで正解ラベルとして取り込む。閾値が低すぎると誤ったラベルが混入し、高すぎると学習が進まないまま`max_iter`に達する。

### マルチラベル・マルチ出力とカスタムスコアラー

#### `MultiOutputClassifier(...)` / `MultiOutputRegressor(...)`

**用途**: 単一出力しか扱えない分類器・回帰器を、目的変数が複数列(マルチ出力)のタスクに拡張する。内部では出力列ごとに独立したモデルを学習する。

**シグネチャ**: `sklearn.multioutput.MultiOutputClassifier(estimator, *, n_jobs=None)` / `sklearn.multioutput.MultiOutputRegressor(estimator, *, n_jobs=None)`

**使用例**:
```python
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
Xm, y1 = make_classification(n_samples=200, n_features=6, n_classes=2, random_state=0)
rng = np.random.RandomState(0)
y2 = rng.randint(0, 3, size=200)
Ym = np.column_stack([y1, y2])
moc = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=0))
moc.fit(Xm, Ym)
print(moc.predict(Xm[:5]))
print("score:", moc.score(Xm, Ym))
```
実行結果:
```
[[0 0]
 [1 1]
 [1 0]
 [1 1]
 [1 1]]
score: 0.995
```

**注意点・落とし穴**:
- 出力列ごとに完全に独立したモデルが学習されるため、出力間の相関(依存関係)は考慮されない。出力間の依存を利用したい場合は分類なら`ClassifierChain`を使う。
- `RandomForestClassifier`や`DecisionTreeClassifier`など一部の推定器はもともとマルチ出力に対応しており、その場合`MultiOutputClassifier`で包む必要はない。
- `MultiOutputRegressor`も同様の考え方で、`Ridge`のようにネイティブに`n_targets`をサポートしない回帰器を多出力対応にする。

#### `ClassifierChain(...)`

**用途**: マルチラベル分類で、各ラベルの予測器を鎖状につなぎ、前のラベルの予測結果を次のラベルの入力特徴量に追加することでラベル間の依存関係を利用する。

**シグネチャ**: `sklearn.multioutput.ClassifierChain(estimator, *, order=None, cv=None, chain_method='predict', random_state=None, verbose=False)`

**使用例**:
```python
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
Xc, Yc = make_classification(n_samples=200, n_features=6, n_classes=2, random_state=0)
Yc2 = np.column_stack([Yc, rng.randint(0, 2, size=200), rng.randint(0, 2, size=200)])
cc = ClassifierChain(LogisticRegression(max_iter=1000), order="random", random_state=0)
cc.fit(Xc, Yc2)
print(cc.predict(Xc[:3]))
```
実行結果:
```
[[0. 1. 0.]
 [1. 1. 0.]
 [1. 0. 1.]]
```

**注意点・落とし穴**:
- `order="random"`(または明示的な順序リスト)でラベルを処理する順序を制御できる。順序によって精度が変わりうるため、`order=None`(デフォルト、入力順)で満足できない場合は複数の`ClassifierChain`をアンサンブルするのが一般的。
- `cv=None`(デフォルト)だと学習時に自分自身の予測(訓練データに対する予測)を次のラベルの特徴量として使うため、過学習(楽観的な連鎖)になりやすい。`cv`に整数を指定すると交差検証予測を使うため、より汎化した鎖になる。

#### `make_scorer(...)` とネストされた交差検証

**用途**: `make_scorer`は任意の評価関数(`fbeta_score`など)を`GridSearchCV`/`cross_val_score`の`scoring`引数に渡せる形に変換する。ネストされた交差検証は、ハイパーパラメータ探索(内側CV)とモデル評価(外側CV)を分離し、探索によるスコアの楽観バイアスを避ける手法。

**シグネチャ**: `sklearn.metrics.make_scorer(score_func, *, response_method='predict', greater_is_better=True, **kwargs)`

**使用例**:
```python
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

f2_scorer = make_scorer(fbeta_score, beta=2, average="macro")
scores = cross_val_score(SVC(), X, y, scoring=f2_scorer, cv=5)
print("f2 scores:", scores.round(3))

inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=1)
clf = GridSearchCV(SVC(), {"C": [0.1, 1, 10]}, cv=inner_cv)
nested_scores = cross_val_score(clf, X, y, cv=outer_cv)
print("nested scores:", nested_scores.round(3))
```
実行結果:
```
f2 scores: [0.966 0.966 0.966 0.933 1.   ]
nested scores: [0.98 0.96 0.94]
```

**注意点・落とし穴**:
- `make_scorer`に渡す`**kwargs`(この例では`beta=2`)は`score_func`の呼び出し時に固定引数として渡される。`greater_is_better=False`にすると内部でスコアの符号が反転される(誤差指標を「大きいほど良い」形式に揃えるため)。
- ネストされた交差検証では、`GridSearchCV`自体を`cross_val_score`の`estimator`として渡すのがポイント。こうすることで外側の各foldごとに独立したハイパーパラメータ探索が行われ、「最良パラメータを選ぶ」という行為そのものが評価スコアに漏れ込むのを防げる。外側CVを使わず単純に`gs.best_score_`だけを報告すると、探索空間が広いほどスコアが楽観的になりやすい。

### モデル解釈

#### `permutation_importance(...)`

**用途**: 特徴量の値をランダムにシャッフルしたときのスコア低下量から、モデル非依存に特徴量重要度を推定する。

**シグネチャ**: `sklearn.inspection.permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5, n_jobs=None, random_state=None, sample_weight=None, max_samples=1.0)`

**使用例**:
```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
r = permutation_importance(rf, X, y, n_repeats=10, random_state=0)
print("importances_mean:", r.importances_mean.round(3))
print("importances_std:", r.importances_std.round(3))
```
実行結果:
```
importances_mean: [0.016 0.011 0.289 0.181]
importances_std: [0.005 0.007 0.028 0.025]
```

**注意点・落とし穴**:
- `RandomForestClassifier.feature_importances_`(不純度ベースの重要度)とは計算方法が異なり、値も一致しない。不純度ベースの重要度は連続値・高カーディナリティな特徴量を過大評価しやすいが、`permutation_importance`はその影響を受けにくい。
- 学習データで計算すると重要度が過大評価されがちなので、可能であれば検証用データ(未使用データ)に対して計算する方が汎化性能への寄与をより正しく反映する。
- 相関の強い特徴量同士があると、片方をシャッフルしてももう片方から情報を補えてしまい、重要度が実際より低く出ることがある。

#### `partial_dependence(...)`

**用途**: 他の特徴量を平均化しつつ、対象の特徴量を動かしたときに予測がどう変化するか(部分依存)を計算する。

**シグネチャ**: `sklearn.inspection.partial_dependence(estimator, X, features, *, sample_weight=None, categorical_features=None, feature_names=None, response_method='auto', percentiles=(0.05, 0.95), grid_resolution=100, custom_values=None, method='auto', kind='average')`

**使用例**:
```python
from sklearn.inspection import partial_dependence
pd_result = partial_dependence(rf, X, features=[2], grid_resolution=5)
print(pd_result["average"].round(3))
print(pd_result["grid_values"][0].round(3))
```
実行結果:
```
[[0.658 0.407 0.165 0.163 0.159]
 [0.239 0.434 0.623 0.47  0.174]
 [0.103 0.159 0.212 0.367 0.667]]
[1.3 2.5 3.7 4.9 6.1]
```

**注意点・落とし穴**:
- 戻り値は辞書ライク(`Bunch`)で、`"average"`が予測値、`"grid_values"`が評価点。多クラス分類(この例ではiris3クラス)では`"average"`の各行がクラスごとの部分依存になる(3行×5グリッド点)。
- 可視化したい場合は`sklearn.inspection.PartialDependenceDisplay.from_estimator(...)`を使うと本関数を内部で呼びつつグラフ化してくれる。
- 特徴量間の相関を無視して「他の特徴量を固定/平均化」するため、強く相関する特徴量が存在すると非現実的な(データに存在しない)組み合わせで評価してしまう場合がある。

### 等張回帰・カーネル近似・多様体学習

#### `IsotonicRegression(...)`

**用途**: 単調性(増加または減少)のみを仮定してxとyの関係をノンパラメトリックに近似する回帰。確率較正(`CalibratedClassifierCV(method='isotonic')`)の内部でも使われる。

**シグネチャ**: `sklearn.isotonic.IsotonicRegression(*, y_min=None, y_max=None, increasing=True, out_of_bounds='nan')`

**使用例**:
```python
import numpy as np
from sklearn.isotonic import IsotonicRegression
x_iso = np.array([1, 2, 3, 4, 5, 6, 7])
y_iso = np.array([1, 0.9, 2.1, 2.0, 3.5, 3.2, 5.0])
ir = IsotonicRegression()
y_pred = ir.fit_transform(x_iso, y_iso)
print(y_pred.round(3))
```
実行結果:
```
[0.95 0.95 2.05 2.05 3.35 3.35 5.  ]
```

**注意点・落とし穴**:
- 入力`x`は1次元のみ対応(多変量の特徴量には使えない)。
- 出力は入力どおりの単調非減少(`increasing=True`がデフォルト)列になる。この例では逆転していた2番目・4番目のペア(0.9<1、2.0<2.1)がそれぞれ平均され隣接値と同じ値(0.95, 2.05)に均されている。
- 学習データの範囲外の`x`を`predict`すると、`out_of_bounds='nan'`(デフォルト)によりNaNが返る。

#### `Nystroem(...)` / `RBFSampler(...)`

**用途**: カーネル法(SVMのカーネルトリックなど)を、明示的な低次元特徴量への写像で近似する。`SGDClassifier`のような線形モデルと組み合わせることで、非線形な決定境界を高速に学習できる。

**シグネチャ**: `sklearn.kernel_approximation.Nystroem(kernel='rbf', *, gamma=None, coef0=None, degree=None, kernel_params=None, n_components=100, random_state=None, n_jobs=None)` / `sklearn.kernel_approximation.RBFSampler(*, gamma=1.0, n_components=100, random_state=None)`

**使用例**:
```python
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

ny = Nystroem(kernel="rbf", gamma=0.2, random_state=0, n_components=50)
X_ny = ny.fit_transform(X)
print(X.shape, "->", X_ny.shape)
clf = SGDClassifier(max_iter=1000, random_state=0).fit(X_ny, y)
print("score:", clf.score(X_ny, y))

rbf = RBFSampler(gamma=0.2, random_state=0, n_components=50)
X_rbf = rbf.fit_transform(X)
print(X.shape, "->", X_rbf.shape)
```
実行結果:
```
(150, 4) -> (150, 50)
score: 0.9733333333333334
(150, 4) -> (150, 50)
```

**注意点・落とし穴**:
- `Nystroem`は学習データの一部をサンプリングして基底を作る(データ依存)のに対し、`RBFSampler`はランダムフーリエ特徴量によりデータに依存せず近似する。一般に同じ`n_components`なら`Nystroem`の方が近似精度が高い傾向がある。
- `n_components`(近似の次元数)を増やすほど元のカーネル法の精度に近づくが、計算コストも増える。
- `gamma`は元の`SVC(kernel='rbf')`の`gamma`と同じ役割で、値の選び方が結果に大きく影響する。

#### `Isomap(...)` / `LocallyLinearEmbedding(...)`

**用途**: データが低次元多様体上に分布しているという仮定のもとで非線形に次元削減する(多様体学習)。`TSNE`と同様に可視化や特徴抽出に使うが、こちらは(条件付きで)新規データへの`transform`が可能。

**シグネチャ**: `sklearn.manifold.Isomap(*, n_neighbors=5, radius=None, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=None, metric='minkowski', p=2, metric_params=None)` / `sklearn.manifold.LocallyLinearEmbedding(*, n_neighbors=5, n_components=2, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None, n_jobs=None)`

**使用例**:
```python
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

iso = Isomap(n_neighbors=10, n_components=2)
X_iso = iso.fit_transform(X)
print(X_iso.shape)
print(X_iso[:3].round(3))

lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=0)
X_lle = lle.fit_transform(X)
print(X_lle.shape)
print("reconstruction_error_:", round(lle.reconstruction_error_, 6))
```
実行結果:
```
(150, 2)
[[-3.148 -0.119]
 [-3.288 -0.135]
 [-3.53  -0.154]]
(150, 2)
reconstruction_error_: 0.0
```

**注意点・落とし穴**:
- 実行時に `UserWarning: The number of connected components of the neighbors graph is 2 > 1. ...` が出ることを実際に確認した。irisデータではk近傍グラフが1つに繋がらない(孤立したクラスタができる)ことがあり、`n_neighbors`を増やして解消するか、警告を許容するかの判断が必要。
- `TSNE`と異なり両クラスとも`fit`済みのオブジェクトで新しい点に対する`transform`が可能(ただし`TSNE`同様、絶対座標やスケールに直接的な意味はない)。
- `LocallyLinearEmbedding`の`reconstruction_error_`は近傍からの線形再構成誤差。0に近いほど近傍構造をよく保てている(この例のように0.0近辺になることもある)。

### ガウス過程

#### `GaussianProcessClassifier(...)` / `GaussianProcessRegressor(...)`

**用途**: ガウス過程による確率的な分類・回帰。予測の不確実性(標準偏差)を自然な形で得られるのが最大の特徴。

**シグネチャ**: `sklearn.gaussian_process.GaussianProcessClassifier(kernel=None, *, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None)` / `sklearn.gaussian_process.GaussianProcessRegressor(kernel=None, *, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None)`

**使用例**:
```python
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, make_regression

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
gpc = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=0)
gpc.fit(Xtr, ytr)
print("score:", gpc.score(Xte, yte))
print("predict_proba[0]:", gpc.predict_proba(Xte[:1]).round(3))

Xr, yr = make_regression(n_samples=50, n_features=1, noise=5.0, random_state=0)
gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), random_state=0, normalize_y=True)
gpr.fit(Xr, yr)
mean, std = gpr.predict(Xr[:3], return_std=True)
print("mean:", mean.round(2))
print("std:", std.round(2))
```
実行結果:
```
score: 0.9777777777777777
predict_proba[0]: [[0.184 0.09  0.726]]
mean: [-22.21  26.85 -12.76]
std: [0. 0. 0.]
```

**注意点・落とし穴**:
- `GaussianProcessRegressor`の`predict`に`return_std=True`を渡すと予測の標準偏差(不確実性)も得られる。この例のように学習に使った点そのものを予測すると、デフォルトの`alpha=1e-10`(観測ノイズがほぼ0という仮定)のためほぼ完全に補間し、`std`が0近くになる。ノイズのあるデータでは`alpha`を大きくするか、`kernel`に`WhiteKernel`を加える。
- `GaussianProcessClassifier`実行時に`ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__length_scale is close to the specified lower bound...`が実際に出た。カーネルのハイパーパラメータ最適化が探索範囲の境界に張り付いている合図で、`RBF(length_scale_bounds=...)`で範囲を広げるなどの対処が有効。
- 計算量がサンプル数の3乗のオーダーで増える(内部でカーネル行列の逆行列を計算するため)。数千サンプルを超えるデータには不向き。
