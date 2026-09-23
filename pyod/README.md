# PyOD 逆引き辞書

PyOD 3.6.5 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(pyod 3.6.5)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。scikit-learnのestimator API(`fit`/`predict`等)の慣習に従っているため、シグネチャの読み方はscikit-learn逆引き辞書と共通。

## 目次

1. [データ生成・評価ユーティリティ](#1-データ生成評価ユーティリティ)
2. [基礎共通API](#2-基礎共通api)
3. [近傍ベース手法](#3-近傍ベース手法)
4. [統計的手法](#4-統計的手法)
5. [線形モデル系](#5-線形モデル系)
6. [クラスタリングベース手法](#6-クラスタリングベース手法)
7. [アンサンブル手法・スコア統合](#7-アンサンブル手法スコア統合)
8. [ニューラルネットワーク系](#8-ニューラルネットワーク系)

---

## 1. データ生成・評価ユーティリティ

### `generate_data(...)`

**用途**: 教師なし異常検知の動作確認用に、正規分布に基づく「正常データ+外れ値データ」を人工的に生成する。学習用・テスト用をまとめて返す。

**シグネチャ**: `pyod.utils.data.generate_data(n_train=1000, n_test=500, n_features=2, contamination=0.1, train_only=False, offset=10, behaviour='new', random_state=None, n_nan=0, n_inf=0)`

**使用例**:
```python
from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(
    n_train=200, n_test=100, n_features=2, contamination=0.1, random_state=42
)
print("X_train.shape:", X_train.shape, "y_train.shape:", y_train.shape)
print("y_train outlier count:", int(y_train.sum()))
```
実行結果:
```
X_train.shape: (200, 2) y_train.shape: (200,)
y_train outlier count: 20
```

**注意点・落とし穴**:
- 戻り値は`y`も含めて4つ(`X_train, X_test, y_train, y_test`)。`train_only=True`にすると返り値の個数が変わるため、呼び出し側のアンパック数に注意。
- ラベルは`0`=正常, `1`=異常。`contamination=0.1`は生成データにおける異常割合の目安(今回の`n_train=200`の例ではちょうど20件=10%と一致)。

### `generate_data_clusters(...)`

**用途**: 複数クラスタ(正常データの塊)+外れ値からなる、より複雑な形状の人工データを生成する。

**シグネチャ**: `pyod.utils.data.generate_data_clusters(n_train=1000, n_test=500, n_clusters=2, n_features=2, contamination=0.1, size='same', density='same', dist=0.25, random_state=None, return_in_clusters=False)`

**使用例**:
```python
from pyod.utils.data import generate_data_clusters

X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=200, n_test=50, n_clusters=3, n_features=2,
    contamination=0.1, random_state=42, return_in_clusters=False
)
print("X_train.shape:", X_train.shape, "X_test.shape:", X_test.shape)
print("y_train outliers:", int(y_train.sum()), "y_test outliers:", int(y_test.sum()))
```
実行結果:
```
X_train.shape: (200, 2) X_test.shape: (50, 2)
y_train outliers: 19 y_test outliers: 6
```

**注意点・落とし穴**:
- `n_test=0`を指定するとエラーになる(内部で`sklearn.model_selection.train_test_split`の`test_size`にそのまま渡されるため、0より大きい値が必須。実際に`n_test=0`で実行し`InvalidParameterError`を確認済み)。
- `return_in_clusters=True`にすると、通常の`(X_train, X_test, y_train, y_test)`という4タプルではなく、クラスタごとに分かれたリスト形式で返る。

### `get_outliers_inliers(...)`

**用途**: `X, y`からラベルに基づいて外れ値サンプルと正常サンプルを分離する。

**シグネチャ**: `pyod.utils.data.get_outliers_inliers(X, y)`

**使用例**:
```python
from pyod.utils.data import get_outliers_inliers

X_outliers, X_inliers = get_outliers_inliers(X_train, y_train)
print("X_outliers.shape:", X_outliers.shape, "X_inliers.shape:", X_inliers.shape)
```
実行結果:
```
X_outliers.shape: (20, 2) X_inliers.shape: (180, 2)
```

**注意点・落とし穴**:
- 戻り値の順序は「外れ値, 正常値」(outliers, inliers)。逆に受け取ると集計を取り違える。

### `evaluate_print(...)`

**用途**: 検知器の異常スコアを正解ラベルと突き合わせ、ROC-AUCとprecision @ rank nをまとめて標準出力に表示する(戻り値なし)。

**シグネチャ**: `pyod.utils.data.evaluate_print(clf_name, y, y_pred)`

**使用例**:
```python
from pyod.utils.data import evaluate_print
from pyod.models.knn import KNN

clf = KNN(contamination=0.1).fit(X_train)
evaluate_print("KNN", y_train, clf.decision_scores_)
evaluate_print("KNN", y_test, clf.decision_function(X_test))
```
実行結果:
```
KNN ROC:0.9992, precision @ rank n:0.95
KNN ROC:1.0, precision @ rank n:1.0
```

**注意点・落とし穴**:
- 第3引数は`y_pred`という名前だが、実際には0/1のラベルではなく連続値の異常スコア(`decision_scores_`や`decision_function()`の出力)を渡す。
- `print`するだけで戻り値はない。数値をプログラムで使うには`sklearn.metrics.roc_auc_score`等を別途呼ぶ必要がある。

### `precision_n_scores(...)` / `score_to_label(...)`

**用途**: `precision_n_scores`は上位n件(異常と疑われる件数)における適合率を計算する。`score_to_label`は連続スコアを、指定した外れ値割合で0/1ラベルに変換する。

**シグネチャ**:
- `pyod.utils.utility.precision_n_scores(y, y_pred, n=None)`
- `pyod.utils.utility.score_to_label(pred_scores, outliers_fraction=0.1)`

**使用例**:
```python
from pyod.utils.utility import precision_n_scores, score_to_label

print("precision_n_scores:", round(precision_n_scores(y_train, clf.decision_scores_), 4))
print("score_to_label[:5]:", score_to_label(clf.decision_scores_, outliers_fraction=0.1)[:5])
```
実行結果:
```
precision_n_scores: 0.95
score_to_label[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- `n=None`(デフォルト)の場合、`y`に含まれる異常ラベル数がそのまま上位n件の`n`として自動的に使われる。
- `score_to_label`はスコアの大小(値が大きいほど異常)で上位`outliers_fraction`をラベル1にする単純な閾値処理であり、`BaseDetector.threshold_`の内部計算と完全に同一とは限らない。

### ROC-AUC / PR-AUC(`sklearn.metrics`との組み合わせ)

**用途**: `evaluate_print`のように標準出力するだけでなく、ROC-AUCやPR-AUC(Average Precision)の数値をプログラムで直接利用したい場合、scikit-learnの評価関数をpyodの異常スコアにそのまま適用する。

**シグネチャ**: `sklearn.metrics.roc_auc_score(y_true, y_score)` / `sklearn.metrics.average_precision_score(y_true, y_score)`

**使用例**:
```python
from sklearn.metrics import roc_auc_score, average_precision_score

y_test_scores = clf.decision_function(X_test)
print("roc_auc_score(test):", round(roc_auc_score(y_test, y_test_scores), 4))
print("average_precision_score(test):", round(average_precision_score(y_test, y_test_scores), 4))
```
実行結果:
```
roc_auc_score(test): 1.0
average_precision_score(test): 1.0
```

**注意点・落とし穴**:
- pyodの`decision_scores_`/`decision_function()`は「値が大きいほど異常」という向きに全検知器で統一されているため、符号反転なしにそのまま`roc_auc_score`へ渡せる。

---

## 2. 基礎共通API

### `BaseDetector.fit(X)` と `decision_scores_` / `labels_` / `threshold_`

**用途**: pyodのすべての検知器(KNN, LOF, HBOSなど)はscikit-learn風の`fit`を共通APIとして持つ。学習後、訓練データに対する異常スコア`decision_scores_`、0/1判定`labels_`、判定に使われた閾値`threshold_`、指定した外れ値割合`contamination`が属性として得られる。

**シグネチャ**: `clf.fit(X, y=None)` (戻り値は`self`)

**使用例**:
```python
from pyod.utils.data import generate_data
from pyod.models.knn import KNN

X_train, X_test, y_train, y_test = generate_data(
    n_train=200, n_test=100, n_features=2, contamination=0.1, random_state=42
)
clf = KNN(contamination=0.1, n_neighbors=5)
clf.fit(X_train)

print("decision_scores_[:5]:", clf.decision_scores_[:5].round(3))
print("labels_[:5]:", clf.labels_[:5])
print("threshold_:", round(clf.threshold_, 3))
print("contamination:", clf.contamination)
```
実行結果:
```
decision_scores_[:5]: [0.344 0.801 0.235 0.18  0.216]
labels_[:5]: [0 0 0 0 0]
threshold_: 1.128
contamination: 0.1
```

**注意点・落とし穴**:
- `decision_scores_`の値そのもの(スケール・単位)はモデルによって全く異なる(KNNは近傍距離、HBOSはヒストグラム密度の対数など)。**「値が大きいほど異常」という向きだけは全検知器で統一されている**(本辞書ではABOD・CBLOF・IForest・OCSVMそれぞれについて、実際に外れ値群の平均スコアが正常群より高いことを実行して確認済み)。異なるモデル間でスコアを直接比較したい場合は後述の`standardizer`で正規化してから比較する。
- `contamination`(デフォルト0.1)は「訓練データに含まれる異常の割合」の事前知識であり、`threshold_`(`labels_`を決める閾値)はこの割合をもとに`decision_scores_`の分位点から自動計算される。実データの異常割合と大きくズレていると`labels_`の精度が落ちる。

### `decision_function(X)` / `predict(X)` / `predict_proba(X)` / `predict_confidence(X)`

**用途**: 学習済み検知器を新規データに適用する。`decision_function`は連続スコア、`predict`は0/1ラベル、`predict_proba`は外れ値である確率(のような値)、`predict_confidence`は予測の確信度を返す。

**シグネチャ**:
- `clf.decision_function(X)` → ndarray
- `clf.predict(X, return_confidence=False)` → ndarray(0/1)
- `clf.predict_proba(X, method='linear')` → ndarray(shape=(n, 2): [inlier側, outlier側])
- `clf.predict_confidence(X)` → ndarray

**使用例**:
```python
y_test_scores = clf.decision_function(X_test)
y_test_pred = clf.predict(X_test)
y_test_proba = clf.predict_proba(X_test)
print("decision_function[:5]:", y_test_scores[:5].round(3))
print("predict[:5]:", y_test_pred[:5])
print("predict_proba[:2]:", y_test_proba[:2].round(3))
print("predict_confidence[:5]:", clf.predict_confidence(X_test)[:5].round(3))
```
実行結果:
```
decision_function[:5]: [0.177 0.177 0.545 0.126 0.196]
predict[:5]: [0 0 0 0 0]
predict_proba[:2]: [[0.993 0.007]
 [0.993 0.007]]
predict_confidence[:5]: [1. 1. 1. 1. 1.]
```

**注意点・落とし穴**:
- `predict`の判定閾値には学習時に決まった`threshold_`(訓練データの`contamination`から計算)がそのまま使われる。テストデータの実際の異常割合が訓練時と異なっていても、閾値は自動で再計算されない。
- `predict_proba`はデフォルト`method='linear'`でスコアを0〜1に線形正規化した値であり、統計的に較正された確率ではない(`method='unify'`という代替方式もある)。

---

## 3. 近傍ベース手法

### `KNN(...)`

**用途**: 各サンプルからk個の近傍点までの距離を集約した値を異常スコアとする、最も基本的な近傍ベース検知器。

**シグネチャ**: `pyod.models.knn.KNN(contamination=0.1, n_neighbors=5, method='largest', radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=1)`

**使用例**:
```python
from pyod.models.knn import KNN

for method in ["largest", "mean", "median"]:
    clf = KNN(contamination=0.1, n_neighbors=5, method=method)
    clf.fit(X_train)
    print(method, clf.decision_scores_[:3].round(3))
```
実行結果:
```
largest [0.344 0.801 0.235]
mean [0.288 0.632 0.163]
median [0.312 0.646 0.201]
```

**注意点・落とし穴**:
- `method='largest'`(デフォルト)はk番目の近傍までの距離、`'mean'`はk個の近傍距離の平均、`'median'`は中央値を使う。実行結果の通りスコアの絶対値が変わるため、`threshold_`や他モデルとの比較時に`method`を混在させない。
- 距離ベースの手法なので、scikit-learnの`KNeighborsClassifier`同様、特徴量のスケールが揃っていないと距離計算が特定の特徴量に支配される。事前のスケーリングを推奨。

### `LOF(...)`

**用途**: 局所密度(Local Outlier Factor)を使い、周囲のサンプルと比べて相対的に密度が低い点を異常と判定する。scikit-learnの`LocalOutlierFactor`をpyodの共通API(`fit`/`decision_function`/`predict`)でラップしたもの。

**シグネチャ**: `pyod.models.lof.LOF(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination=0.1, n_jobs=1, novelty=True)`

**使用例**:
```python
from pyod.models.lof import LOF

lof = LOF(n_neighbors=20, contamination=0.1)
lof.fit(X_train)
print("LOF decision_scores_[:5]:", lof.decision_scores_[:5].round(3))
print("LOF labels_[:5]:", lof.labels_[:5])
print("LOF predict(test)[:5]:", lof.predict(X_test)[:5])
```
実行結果:
```
LOF decision_scores_[:5]: [1.093 1.673 1.033 0.968 1.   ]
LOF labels_[:5]: [0 0 0 0 0]
LOF predict(test)[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- 内部のscikit-learn `LocalOutlierFactor`は`novelty=False`の場合`negative_outlier_factor_`属性を持つが、pyodの`LOF`ラッパーは`novelty=True`固定相当で運用されており、`negative_outlier_factor_`属性は公開されていない(実際に`AttributeError`を確認済み)。スコアは`decision_scores_`/`decision_function()`経由で取得する。
- `n_neighbors`(デフォルト20)が小さすぎると局所密度の推定が不安定になり、大きすぎるとグローバルな密度と変わらなくなる。

### `COF(...)`

**用途**: LOFと同系統だが、局所密度の代わりに「連結性に基づく近傍距離」(chaining distance)を使う異常検知手法。密度が一様でない・鎖状に分布するデータでLOFより頑健とされる。

**シグネチャ**: `pyod.models.cof.COF(contamination=0.1, n_neighbors=20, method='fast')`

**使用例**:
```python
from pyod.models.cof import COF

cof = COF(contamination=0.1, n_neighbors=20)
cof.fit(X_train)
print("COF decision_scores_[:5]:", cof.decision_scores_[:5].round(3))
print("COF labels_[:5]:", cof.labels_[:5])
```
実行結果:
```
COF decision_scores_[:5]: [1.238 1.535 0.951 0.908 0.984]
COF labels_[:5]: [0 1 0 0 0]
```

**注意点・落とし穴**:
- `method='fast'`(デフォルト)は近似計算で高速化したバージョン。`method='memory'`にすると計算方法が変わり、大規模データではメモリ消費が大きくなる。
- LOFよりも計算コストが高い傾向があるため、大規模データにはサブサンプリングや近似設定を検討する。

### `ABOD(...)`

**用途**: 対象サンプルと他の点ペアがなす角度のばらつき(Angle-Based Outlier Detection)を使う。高次元データでLOFなど距離ベースの手法より頑健とされる。

**シグネチャ**: `pyod.models.abod.ABOD(contamination=0.1, n_neighbors=5, method='fast', algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=1)`

**使用例**:
```python
from pyod.models.abod import ABOD

abod = ABOD(contamination=0.1, n_neighbors=5, method="fast")
abod.fit(X_train)
print("ABOD decision_scores_[:5]:", abod.decision_scores_[:5].round(6))
print("ABOD mean score: outliers=%.4f inliers=%.4f" % (
    abod.decision_scores_[y_train == 1].mean(),
    abod.decision_scores_[y_train == 0].mean(),
))
```
実行結果:
```
ABOD decision_scores_[:5]: [-9.79272710e+01 -4.22133300e+00 -1.23655469e+03 -2.05172867e+04
 -4.84761630e+03]
ABOD mean score: outliers=-0.4227 inliers=-4396.5357
```

**注意点・落とし穴**:
- スコアはすべて負値になりうるが、**「値が大きい(0に近い)ほど異常」という向きは他モデルと同じ**(上の例でも外れ値群の平均が-0.42、正常群の平均が-4396.5と、外れ値の方が明らかに大きい=0に近い)。絶対値の大小で直感的に「異常度が高い」と誤解しないよう注意。
- デフォルトの`method='fast'`は近似計算(高速ABOD)。`method='default'`にすると全点ペアの角度を厳密に計算するため、サンプル数が多いと非常に遅い。

---

## 4. 統計的手法

### `HBOS(...)`

**用途**: 各特徴量ごとにヒストグラムを作り、独立性を仮定して密度の対数の合計を異常スコアとする(Histogram-Based Outlier Score)。高速で解釈しやすいのが特長。

**シグネチャ**: `pyod.models.hbos.HBOS(n_bins=10, alpha=0.1, tol=0.5, contamination=0.1)`

**使用例**:
```python
from pyod.models.hbos import HBOS

hbos = HBOS(n_bins=10, contamination=0.1)
hbos.fit(X_train)
print("HBOS decision_scores_[:5]:", hbos.decision_scores_[:5].round(3))
print("HBOS labels_[:5]:", hbos.labels_[:5])
```
実行結果:
```
HBOS decision_scores_[:5]: [2.477 4.547 2.477 3.33  2.23 ]
HBOS labels_[:5]: [0 1 0 0 0]
```

**注意点・落とし穴**:
- 特徴量間の独立性を仮定する(ナイーブベイズに近い発想)ため、特徴量間に強い相関があると精度が落ちる。
- `alpha`(デフォルト0.1)はビンの度数が0のときに割り当てる最小密度の平滑化パラメータ。0のままだとlogでエラーになるため必要な値。

### `ECOD(...)`

**用途**: 各特徴量の経験累積分布関数(Empirical Cumulative Distribution)の両側裾確率を集約して異常スコアとする、パラメータ調整がほぼ不要な統計的手法(Empirical-Cumulative-distribution-based Outlier Detection)。

**シグネチャ**: `pyod.models.ecod.ECOD(contamination=0.1, n_jobs=1)`

**使用例**:
```python
from pyod.models.ecod import ECOD

ecod = ECOD(contamination=0.1)
ecod.fit(X_train)
print("ECOD decision_scores_[:5]:", ecod.decision_scores_[:5].round(3))
print("ECOD predict(test)[:5]:", ecod.predict(X_test)[:5])
```
実行結果:
```
ECOD decision_scores_[:5]: [2.374 5.141 2.161 2.283 1.948]
ECOD predict(test)[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- ハイパーパラメータが`contamination`と`n_jobs`しかなく、`n_neighbors`や`n_bins`のようなチューニング対象がないのが最大の特徴。まず試す基準(ベースライン)として使いやすい。
- 各特徴量が独立かつ単峰(unimodal)であることを暗に仮定しており、多峰分布や強い特徴量間相関があるデータでは性能が落ちる場合がある。

### `COPOD(...)`

**用途**: ECODと同じCopula(コピュラ)ベースの統計的手法の先行研究。各特徴量の経験分布から得られる裾確率を、コピュラを用いて結合して異常スコアを計算する。

**シグネチャ**: `pyod.models.copod.COPOD(contamination=0.1, n_jobs=1)`

**使用例**:
```python
from pyod.models.copod import COPOD

copod = COPOD(contamination=0.1)
copod.fit(X_train)
print("COPOD decision_scores_[:5]:", copod.decision_scores_[:5].round(3))
print("COPOD predict(test)[:5]:", copod.predict(X_test)[:5])
```
実行結果:
```
COPOD decision_scores_[:5]: [1.873 3.401 2.116 2.283 1.7  ]
COPOD predict(test)[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- ECODと同じくパラメータ調整がほぼ不要。ECODとCOPODは同一著者による関連手法で、スコアの計算方法(裾確率の集約方法)が異なるため、同じデータでも`decision_scores_`の値はECODと一致しない(上の例でも[2.374, 5.141, ...]対[1.873, 3.401, ...]と異なる)。

### `KDE(...)`

**用途**: カーネル密度推定(Kernel Density Estimation)で全体のデータ密度を推定し、密度が低い点を異常とする。

**シグネチャ**: `pyod.models.kde.KDE(contamination=0.1, bandwidth=1.0, algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None)`

**使用例**:
```python
from pyod.models.kde import KDE

kde = KDE(contamination=0.1).fit(X_train)
print("KDE decision_scores_[:5]:", kde.decision_scores_[:5].round(3))
```
実行結果:
```
KDE decision_scores_[:5]: [2.571 3.673 2.595 2.593 2.479]
```

**注意点・落とし穴**:
- `bandwidth`(デフォルト1.0)がスコアの感度に直結する。データのスケールに対して固定値なので、事前にスケーリングしてから`bandwidth`を調整するのが実用的。
- 特徴量数が増える(次元が高くなる)ほど密度推定が不安定になる「次元の呪い」の影響を受けやすい。

---

## 5. 線形モデル系

### `PCA(...)`

**用途**: 主成分分析で低次元に射影し、元の空間への再構成誤差(または主成分空間でのマハラノビス距離相当)を異常スコアとする。

**シグネチャ**: `pyod.models.pca.PCA(n_components=None, n_selected_components=None, contamination=0.1, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None, weighted=True, standardization=True)`

**使用例**:
```python
from pyod.models.pca import PCA

pca = PCA(n_components=2, contamination=0.1, random_state=42)
pca.fit(X_train)
print("PCA decision_scores_[:5]:", pca.decision_scores_[:5].round(3))
print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3))
```
実行結果(`n_features=5`のデータ):
```
PCA decision_scores_[:5]: [12.037 10.385 10.748 14.869  8.661]
PCA explained_variance_ratio_: [0.67  0.131]
```

**注意点・落とし穴**:
- `standardization=True`(デフォルト)により内部で自動的に標準化されるため、scikit-learnの`PCA`と違い、事前の`StandardScaler`は必須ではない。
- `weighted=True`(デフォルト)は各主成分の寄与率で重み付けした異常スコアを計算する。`False`にすると単純な再構成誤差ベースになり、スコアの値が変わる。

### `OCSVM(...)`

**用途**: One-Class SVMにより正常データを囲む決定境界を学習し、境界外を異常とする。scikit-learnの`OneClassSVM`をpyodの共通API向けにラップしたもの。

**シグネチャ**: `pyod.models.ocsvm.OCSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, contamination=0.1)`

**使用例**:
```python
from pyod.models.ocsvm import OCSVM

ocsvm = OCSVM(contamination=0.1, kernel="rbf")
ocsvm.fit(X_train)
print("OCSVM decision_scores_[:5]:", ocsvm.decision_scores_[:5].round(3))
print("OCSVM labels_[:5]:", ocsvm.labels_[:5])
```
実行結果(`n_features=5`のデータ):
```
OCSVM decision_scores_[:5]: [ 2.36  -4.461  6.931  0.356 -7.522]
OCSVM labels_[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- scikit-learnの`OneClassSVM.decision_function()`は「正なら正常、負なら異常」という向きだが、pyodの`OCSVM`はこれを内部で符号反転し、「値が大きいほど異常」という他の検知器と統一された向きに変換している(実際に外れ値群の平均スコアが正常群より高いことを確認済み)。scikit-learn版の`OneClassSVM`を直接使う場合との符号の違いに注意。
- `nu`(デフォルト0.5)は訓練誤差の上限・サポートベクター数の下限を制御するパラメータで、`contamination`に近い意味を持つが厳密には別物。両方を無関係な値に設定すると挙動が食い違う。

### `MCD(...)`

**用途**: 最小共分散行列式(Minimum Covariance Determinant)によりロバストな平均・共分散を推定し、そこからのマハラノビス距離を異常スコアとする。

**シグネチャ**: `pyod.models.mcd.MCD(contamination=0.1, store_precision=True, assume_centered=False, support_fraction=None, random_state=None)`

**使用例**:
```python
from pyod.models.mcd import MCD

mcd = MCD(contamination=0.1, random_state=42)
mcd.fit(X_train)
print("MCD decision_scores_[:5]:", mcd.decision_scores_[:5].round(3))
print("MCD labels_[:5]:", mcd.labels_[:5])
```
実行結果(`n_features=5`のデータ):
```
MCD decision_scores_[:5]: [6.589 2.154 7.89  4.719 0.675]
MCD labels_[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- 共分散行列を推定するため、サンプル数が特徴量数に比べて少なすぎると推定が不安定になる(目安として`n_samples > 5 * n_features`程度は欲しい)。
- ロバスト推定である分、通常の共分散行列を使うマハラノビス距離より外れ値自体の影響を受けにくいが、計算コストは高め。

---

## 6. クラスタリングベース手法

### `CBLOF(...)`

**用途**: k-meansなどでクラスタリングした後、「大きいクラスタからの距離」と「小さい(異常)クラスタに属すること」を組み合わせて異常スコアとする(Cluster-Based Local Outlier Factor)。

**シグネチャ**: `pyod.models.cblof.CBLOF(n_clusters=8, contamination=0.1, clustering_estimator=None, alpha=0.9, beta=5, use_weights=False, check_estimator=False, random_state=None, n_jobs=1)`

**使用例**:
```python
from pyod.models.cblof import CBLOF

cblof = CBLOF(n_clusters=8, contamination=0.1, random_state=42)
cblof.fit(X_train)
print("CBLOF decision_scores_[:5]:", cblof.decision_scores_[:5].round(3))
print("CBLOF mean score: outliers=%.4f inliers=%.4f" % (
    cblof.decision_scores_[y_train == 1].mean(),
    cblof.decision_scores_[y_train == 0].mean(),
))
```
実行結果:
```
CBLOF decision_scores_[:5]: [0.459 1.343 0.691 0.444 0.212]
CBLOF mean score: outliers=7.8284 inliers=0.6301
```

**注意点・落とし穴**:
- `alpha`(クラスタを「大」「小」に分ける累積サイズの閾値、デフォルト0.9)と`beta`(大小の境界を決める比率、デフォルト5)の組み合わせ次第で、どのクラスタが「小さい(異常)クラスタ」扱いになるかが変わる。
- デフォルトの`clustering_estimator=None`の場合、内部でscikit-learnの`KMeans(n_clusters=n_clusters)`が使われる。データがクラスタ状に分布していない場合は他の手法の方が適することがある。

---

## 7. アンサンブル手法・スコア統合

### `IForest(...)`

**用途**: ランダムに軸分割する決定木(Isolation Tree)を多数作り、孤立させるまでの分割回数の平均を異常スコアとする(Isolation Forest)。scikit-learnの`IsolationForest`をpyodの共通APIでラップしたもの。

**シグネチャ**: `pyod.models.iforest.IForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=1, behaviour='old', random_state=None, verbose=0)`

**使用例**:
```python
from pyod.models.iforest import IForest

iforest = IForest(n_estimators=100, contamination=0.1, random_state=42)
iforest.fit(X_train)
print("IForest decision_scores_[:5]:", iforest.decision_scores_[:5].round(3))
print("IForest feature_importances_:", iforest.feature_importances_.round(3))
```
実行結果:
```
IForest decision_scores_[:5]: [-0.2   -0.06  -0.204 -0.207 -0.209]
IForest feature_importances_: [0.528 0.472]
```

**注意点・落とし穴**:
- `decision_scores_`が負値を含む点はOCSVMと似ているが、ここでも「値が大きいほど異常」という向きは統一されている(実際に外れ値群の平均スコア0.0662、正常群-0.1748と確認済み)。
- `behaviour='old'`という引数がシグネチャ上残っているが、scikit-learn側の`IsolationForest`ではこの引数はすでに廃止されており、pyod側の互換性維持のための名残(値を変えても挙動に影響しないため、通常は指定不要)。

### `LSCP(...)`

**用途**: 複数の検知器(通常はLOFやKNNなど局所的な手法)を、テスト点の局所領域での性能に応じて動的に重み付けして統合する(Locally Selective Combination in Parallel Outlier Ensembles)。

**シグネチャ**: `pyod.models.lscp.LSCP(detector_list, local_region_size=30, local_max_features=1.0, n_bins=10, random_state=None, contamination=0.1)`

**使用例**:
```python
from pyod.models.lscp import LSCP
from pyod.models.knn import KNN
from pyod.models.lof import LOF

detector_list = [KNN(n_neighbors=5), KNN(n_neighbors=10), LOF(n_neighbors=20), LOF(n_neighbors=30)]
lscp = LSCP(detector_list, contamination=0.1, random_state=42)
lscp.fit(X_train)
print("LSCP decision_scores_[:5]:", lscp.decision_scores_[:5].round(3))
```
実行結果:
```
LSCP decision_scores_[:5]: [-0.373  0.112 -0.385 -0.469 -0.442]
```

**注意点・落とし穴**:
- 第1引数`detector_list`は必須(デフォルトなし)。複数の未学習の検知器インスタンスのリストを渡すと、`LSCP.fit()`内部でそれぞれ`fit`される。
- `n_bins`(ヒストグラムのビン数、デフォルト10)が`detector_list`の要素数より多いと、`UserWarning: The number of histogram bins is greater than the number of classifiers, reducing n_bins to n_clf.`という警告が出て自動的に縮小される(4個の検知器・`n_bins=10`で実際に確認済み)。

### `FeatureBagging(...)`

**用途**: 特徴量のサブセット(ランダムに選んだ特徴量の一部)ごとにベース検知器(デフォルトLOF)を学習し、その予測を統合するアンサンブル手法。scikit-learnの`BaggingClassifier`に近い発想を異常検知に応用したもの。

**シグネチャ**: `pyod.models.feature_bagging.FeatureBagging(base_estimator=None, n_estimators=10, contamination=0.1, max_features=1.0, bootstrap_features=False, check_detector=True, check_estimator=False, n_jobs=1, random_state=None, combination='average', verbose=0, estimator_params=None)`

**使用例**:
```python
from pyod.models.feature_bagging import FeatureBagging

fb = FeatureBagging(contamination=0.1, n_estimators=10, random_state=42)
fb.fit(X_train)
print("FeatureBagging decision_scores_[:5]:", fb.decision_scores_[:5].round(3))
```
実行結果:
```
FeatureBagging decision_scores_[:5]: [1.045 1.357 1.01  0.998 1.006]
```

**注意点・落とし穴**:
- `base_estimator=None`(デフォルト)の場合、内部で`LOF()`が使われる。
- `combination='average'`(デフォルト)以外に、後述の`maximization`相当の統合方法も内部的にサポートされている。

### `XGBOD(...)`

**用途**: 複数の教師なし検知器のスコアを特徴量として元の特徴量に追加した上で、XGBoostによる教師あり分類器で最終判定する半教師あり手法(eXtreme Gradient Boosting Outlier Detection)。**pyodの中で唯一`fit(X, y)`に正解ラベル`y`が必須**な手法。

**シグネチャ**: `pyod.models.xgbod.XGBOD(estimator_list=None, standardization_flag_list=None, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, **kwargs)`

**使用例**:
```python
from pyod.models.xgbod import XGBOD

xgbod = XGBOD(random_state=42)
xgbod.fit(X_train, y_train)
print("XGBOD decision_scores_[:5]:", xgbod.decision_scores_[:5].round(3))
print("XGBOD predict(test)[:5]:", xgbod.predict(X_test)[:5])
```
実行結果:
```
XGBOD decision_scores_[:5]: [0.004 0.013 0.004 0.005 0.004]
XGBOD predict(test)[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- 他のpyod検知器は`fit(X)`のみで動く教師なし手法だが、XGBODは`fit(X, y)`と正解ラベルが必須。ラベルなしで`fit(X_train)`のみを呼ぶとエラーになる。
- 内部で複数の教師なし検知器を`fit`する際に`UserWarning: y should not be presented in unsupervised learning.`という警告が出ることを確認済み(内部処理の副作用であり、XGBOD自体の`fit(X, y)`呼び出しが誤りというわけではない)。

### `standardizer(...)`

**用途**: 複数の検知器の`decision_scores_`はスケールが異なるため、統合(アンサンブル)する前にZスコア標準化して揃える。

**シグネチャ**: `pyod.utils.utility.standardizer(X, X_t=None, keep_scalar=False)`

**使用例**:
```python
import numpy as np
from pyod.utils.utility import standardizer
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.cof import COF

clfs = [KNN().fit(X_train), LOF().fit(X_train), HBOS().fit(X_train), COF().fit(X_train)]
train_scores = np.column_stack([clf.decision_scores_ for clf in clfs])
train_scores_norm, _ = standardizer(train_scores, train_scores)
print("train_scores_norm[:3]:", train_scores_norm[:3].round(3))
```
実行結果:
```
train_scores_norm[:3]: [[-0.289 -0.335 -0.643  0.337]
 [ 0.132  0.443  1.237  1.464]
 [-0.39  -0.417 -0.643 -0.746]]
```

**注意点・落とし穴**:
- `X_t`を渡すと`(X_norm, X_t_norm)`のタプルが返る(訓練データの平均・分散でテストデータも正規化する用途)。`X_t=None`の場合は`X_norm`単体が返るため、戻り値の受け取り方が変わる点に注意。

### `average(...)` / `maximization(...)`

**用途**: 標準化済みの複数検知器のスコア行列(サンプル数×検知器数)を1列に統合する。`average`は単純平均、`maximization`は各サンプルごとの最大値を採用する。

**シグネチャ**:
- `pyod.models.combination.average(scores, estimator_weights=None)`
- `pyod.models.combination.maximization(scores)`

**使用例**:
```python
from pyod.models.combination import average, maximization

print("average[:5]:", average(train_scores_norm)[:5].round(3))
print("maximization[:5]:", maximization(train_scores_norm)[:5].round(3))
```
実行結果:
```
average[:5]: [-0.233  0.819 -0.549 -0.431 -0.59 ]
maximization[:5]: [ 0.337  1.464 -0.39   0.132 -0.408]
```

**注意点・落とし穴**:
- `maximization`は「複数の検知器のうち、どれか1つでも強く異常と判定すれば異常扱いにしたい」場合に向くが、誤検知(False Positive)も増えやすい。逆に外れ値と一致して全検知器が同意した場合のみ高スコアにしたいなら`average`や後述の`aom`/`moa`が向く。

### `aom(...)` / `moa(...)`

**用途**: 検知器を複数の「バケツ」に分けてバケツ内で統合(AOMはバケツ内最大値、MOAはバケツ内平均)した後、バケツ間で反対の統合(AOMはバケツ間平均、MOAはバケツ間最大値)を行う、`average`/`maximization`より頑健な統合方法(Average of Maximum / Maximum of Average)。

**シグネチャ**:
- `pyod.models.combination.aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None)`
- `pyod.models.combination.moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None)`

**使用例**:
```python
from pyod.models.combination import aom, moa

print("aom[:5]:", aom(train_scores_norm, n_buckets=2, random_state=42)[:5].round(3))
print("moa[:5]:", moa(train_scores_norm, n_buckets=2, random_state=42)[:5].round(3))
```
実行結果:
```
aom[:5]: [ 0.024  1.35  -0.403 -0.186 -0.434]
moa[:5]: [ 0.001  0.953 -0.516 -0.155 -0.542]
```

**注意点・落とし穴**:
- `method='static'`(デフォルト)では検知器の総数が`n_buckets`で割り切れる必要がある。割り切れない場合、`ValueError: n_estimators / n_buckets has a remainder. Not allowed in static mode.`が発生する(実際に4検知器・`n_buckets=2`はOK、3検知器・`n_buckets=2`はNGであることを確認済み)。割り切れない構成にしたい場合は`method='dynamic'`または`bootstrap_estimators=True`を使う。
- 大規模なアンサンブルを高速に統合したい場合、pyodには`SUOD`(Scalable Unsupervised Outlier Detection)という専用モジュールもあるが、これは`pip install suod`という追加パッケージが必要で、本検証環境には未インストールのため(`ImportError: pyod.models.suod requires the optional suod package`を実際に確認)、本辞書では動作例を割愛する。

---

## 8. ニューラルネットワーク系

### `AutoEncoder(...)`

**用途**: エンコーダ・デコーダ構造のニューラルネットワーク(PyTorchベース)で入力を再構成し、再構成誤差を異常スコアとする。

**シグネチャ**: `pyod.models.auto_encoder.AutoEncoder(contamination=0.1, preprocessing=True, lr=0.001, epoch_num=10, batch_size=32, optimizer_name='adam', device=None, random_state=42, use_compile=False, compile_mode='default', verbose=1, optimizer_params: dict = {'weight_decay': 1e-05}, hidden_neuron_list=[64, 32], hidden_activation_name='relu', batch_norm=True, dropout_rate=0.2)`

**使用例**:
```python
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(
    n_train=200, n_test=100, n_features=10, contamination=0.1, random_state=42
)
ae = AutoEncoder(hidden_neuron_list=[16, 8], epoch_num=10, contamination=0.1, random_state=42, verbose=0)
ae.fit(X_train)
print("AutoEncoder decision_scores_[:5]:", ae.decision_scores_[:5].round(3))
print("AutoEncoder predict(test)[:5]:", ae.predict(X_test)[:5])
```
実行結果:
```
AutoEncoder decision_scores_[:5]: [1.027 1.717 1.081 1.334 1.246]
AutoEncoder predict(test)[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- `preprocessing=True`(デフォルト)により、内部で自動的に`StandardScaler`相当の標準化が行われる(元の`X_train`をそのまま渡してよい)。
- 内部実装はPyTorch製(`device=None`ならGPUがあれば自動使用)。デフォルトの`hidden_neuron_list=[64, 32]`は次元数が少ないデータにはやや過剰な場合があり、今回の例のように小さいデータセットでは`hidden_neuron_list`を明示的に縮小した方が安定する。
- `epoch_num`(デフォルト10)が小さいと収束前に学習が終わる。`random_state=42`を指定していても、環境やPyTorchのバージョンによって厳密な再現性が保証されない場合がある点はニューラルネット系全般の注意点。
