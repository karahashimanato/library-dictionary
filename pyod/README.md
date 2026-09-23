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
9. [応用・発展](#応用発展)
   - 9.1 [閾値選択(pyod.models.thresholds)](#91-閾値選択pyodmodelsthresholds)
   - 9.2 [ニューラルネットワーク系検出器(応用)](#92-ニューラルネットワーク系検出器応用)
   - 9.3 [モデルの永続化](#93-モデルの永続化)
   - 9.4 [時系列向け異常検知](#94-時系列向け異常検知)
   - 9.5 [実データでのcontamination推定の実践パターン](#95-実データでのcontamination推定の実践パターン)

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

---

## 応用・発展

ここから先は、基礎編(1〜8章)では扱わなかった、より発展的・ニッチなAPIを扱う。標準の`pip install pyod`だけでは使えないものもあり、その場合は依存パッケージを明記する。

### 9.1 閾値選択(pyod.models.thresholds)

#### `pyod.models.thresholds` の概要と `FILTER(...)`

**用途**: `contamination`という事前知識(異常割合)を人手で指定する代わりに、`decision_scores_`の分布そのものから外れ値/正常値の閾値を自動推定するモジュール。内部的には追加パッケージ`pythresh`(本検証環境ではインストール済み、`pythresh==1.1.1`)のクラスをそのまま返す薄いラッパー関数になっている。`FILTER`はスコア列を信号とみなし、フィルタリング処理(デフォルトはSavitzky-Golayフィルタ)を通して外れ値境界を求める手法。

**シグネチャ**: `pyod.models.thresholds.FILTER(method='savgol', sigma='auto', random_state=1234)` (実体は`pythresh.thresholds.filter.FILTER`を返す関数。`AUCP`/`IQR`など他の閾値クラスも同様に`**kwargs`を`pythresh`側にそのまま渡す関数として実装されている)

**使用例**:
```python
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from pyod.models.thresholds import FILTER

X_train, X_test, y_train, y_test = generate_data(
    n_train=200, n_test=100, n_features=2, contamination=0.1, random_state=42
)
clf = KNN(contamination=0.1).fit(X_train)
scores = clf.decision_scores_

filt = FILTER(method="medfilt")
labels = filt.eval(scores)
print("FILTER labels_[:10]:", labels[:10])
print("推定contamination:", round(labels.mean(), 4))
print("thresh_:", round(filt.thresh_, 4))
```
実行結果:
```
FILTER labels_[:10]: [0 0 0 0 0 0 0 0 0 0]
推定contamination: 0.085
thresh_: 0.3927
```

**注意点・落とし穴**:
- `pyod.models.thresholds`配下のクラス(`FILTER`, `AUCP`, `IQR`など全30種)は`class`ではなく`def FILTER(**kwargs): ... return FILTER_thres(**kwargs)`という**関数**として実装されている(`inspect.signature`が`(*args, **kwargs)`しか返さず、詳細パラメータはdocstringでしか確認できない)。返ってくるインスタンスの実体は`pyod`ではなく`pythresh`パッケージのクラスであり、`pythresh`が未インストールの環境では`ImportError`になる。
- `eval(decision)`はスコア配列を渡すと即座にラベルを返す一括処理。sklearn風に`fit(X).predict(X)`と2段階で呼ぶAPIも用意されているが、`predict`は`fit`を先に呼んでいないと`NotFittedError`になる(実際に確認済み)。`eval`の方が手軽。

#### `AUCP(...)`

**用途**: スコアのカーネル密度推定(KDE)の曲線下面積(Area Under Curve)を使い、「平均+|平均-中央値|」を境に外れ値/正常値を分ける非パラメトリックな閾値手法。`contamination`のような割合の事前指定が不要。

**シグネチャ**: `pyod.models.thresholds.AUCP(random_state=1234)`

**使用例**:
```python
from pyod.models.thresholds import AUCP

aucp = AUCP()
labels = aucp.eval(scores)
print("AUCP labels_[:10]:", labels[:10])
print("推定contamination:", round(labels.mean(), 4))
print("thresh_:", round(aucp.thresh_, 4))
```
実行結果:
```
AUCP labels_[:10]: [0 0 0 0 0 0 0 0 0 0]
推定contamination: 0.095
thresh_: 0.1729
```

**注意点・落とし穴**:
- 今回のKNNスコア(良好に分離した人工データ)では真のcontamination(0.1)に近い0.095を推定できたが、手法によって推定結果が大きくブレることがある(9.5節で、同じデータでもECODスコアに対して`AUCP`と`FILTER`が0.305 vs 0.08という大きく異なる推定値を出す例を確認している)。単一の閾値手法の出力を無条件に信頼せず、複数手法を比較するか、ドメイン知識で妥当性を確認すべき。

#### 検出器への thresholder の直接組み込み(`contamination=`)

**用途**: `pyod`の各検出器(`KNN`, `HBOS`など)の`contamination`引数は、`float`だけでなく`pyod.models.thresholds`(=`pythresh`)の閾値インスタンスをそのまま渡せる。渡すと、固定の割合ではなく上記のような自動閾値推定ロジックで`labels_`/`threshold_`が計算される。

**シグネチャ**: `KNN(contamination=<float または pythresh.thresholds.base.BaseThresholder インスタンス>, ...)`(`BaseDetector`共通)

**使用例**:
```python
from pyod.models.knn import KNN
from pyod.models.thresholds import IQR

clf = KNN(contamination=IQR())
clf.fit(X_train)
print("labels_[:10]:", clf.labels_[:10])
print("labels_.sum():", int(clf.labels_.sum()))
print("threshold_:", round(clf.threshold_, 4))
print("contamination:", clf.contamination)
```
実行結果:
```
labels_[:10]: [0 0 0 0 0 0 0 0 0 0]
labels_.sum(): 25
threshold_: 0.1077
contamination: IQR()
```

**注意点・落とし穴**:
- `contamination=0.1`(float)で`KNN`を学習した場合は`labels_.sum()`が20件(2章で確認済み)だったが、`contamination=IQR()`にすると25件に変化する。`contamination`属性自体もfloatではなく渡した`IQR()`インスタンスがそのまま格納される点に注意(`type(clf.contamination)`は`pythresh.thresholds.iqr.IQR`)。
- 実務でどのthresholderを選ぶべきか自明ではない。9.5節の通り、`AUCP`/`FILTER`/`IQR`は同じスコアに対しても異なる推定contaminationを出すため、複数を試してレンジを把握するのが安全。

### 9.2 ニューラルネットワーク系検出器(応用)

#### `DeepSVDD(...)`

**用途**: One-Class SVMの発想をニューラルネットワークに拡張し、正常データを特徴空間内の1点(中心`c`)の周りに写像するよう学習する深層異常検知手法(Deep Support Vector Data Description)。中心からの距離を異常スコアとする。

**シグネチャ**: `pyod.models.deep_svdd.DeepSVDD(n_features, c=None, use_ae=False, hidden_neurons=None, hidden_activation='relu', output_activation='sigmoid', optimizer='adam', epochs=100, batch_size=32, dropout_rate=0.2, l2_regularizer=5e-07, validation_size=0.1, preprocessing=True, verbose=1, random_state=None, contamination=0.1, learning_rate=0.0001)`

**使用例**:
```python
from pyod.utils.data import generate_data
from pyod.models.deep_svdd import DeepSVDD

X_train, X_test, y_train, y_test = generate_data(
    n_train=200, n_test=100, n_features=10, contamination=0.1, random_state=42
)
clf = DeepSVDD(n_features=10, epochs=3, hidden_neurons=[16, 8],
                contamination=0.1, random_state=42, verbose=0)
clf.fit(X_train)
print("decision_scores_[:5]:", clf.decision_scores_[:5].round(3))
print("predict(test)[:5]:", clf.predict(X_test)[:5])
```
実行結果:
```
Epoch 1/3, Loss: 1.441234927624464
Epoch 2/3, Loss: 1.523838832974434
Epoch 3/3, Loss: 1.346273947507143
decision_scores_[:5]: [0.063 0.052 0.05  0.058 0.07 ]
predict(test)[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- 他の検出器と異なり、コンストラクタの第1引数`n_features`は**必須**(デフォルトなし)。渡さずに`DeepSVDD()`を呼ぶと`TypeError: DeepSVDD.__init__() missing 1 required positional argument: 'n_features'`になる(実際に確認済み)。`AutoEncoder`のように入力データから自動推論はされない。
- `verbose=0`を指定しても、上記実行結果の通り`Epoch i/N, Loss: ...`という学習ログは抑制されない。`pyod 3.6.5`同梱の`deep_svdd.py`のソースを確認したところ、学習ループ末尾の`print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss}")`が`self.verbose`の値を一切参照せず無条件に実行されているためで、`verbose`引数の実装漏れと考えられる(ログを抑制する公式な方法はない)。

#### `LUNAR(...)`

**用途**: グラフニューラルネットワーク(GNN)でk近傍情報を学習し、近傍距離を異常スコアに変換する深層学習ベースの近傍法(Learnable Unified Neighbourhood-based Anomaly Ranking)。古典的なKNN/LOFの「固定的な集約方法」をニューラルネットに置き換えたもの。

**シグネチャ**: `pyod.models.lunar.LUNAR(model_type='WEIGHT', n_neighbours=5, negative_sampling='MIXED', val_size=0.1, scaler=None, epsilon=0.1, proportion=1.0, n_epochs=200, lr=0.001, wd=0.1, verbose=0, contamination=0.1, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=1, random_state=None)`

**使用例**:
```python
from pyod.models.lunar import LUNAR

clf = LUNAR(n_neighbours=5, n_epochs=20, contamination=0.1, verbose=0, random_state=42)
clf.fit(X_train)
print("decision_scores_[:5]:", clf.decision_scores_[:5].round(3))
print("labels_[:5]:", clf.labels_[:5])
print("predict(test)[:5]:", clf.predict(X_test)[:5])
```
実行結果:
```
decision_scores_[:5]: [0.08  0.138 0.043 0.088 0.085]
labels_[:5]: [0 0 0 0 0]
predict(test)[:5]: [0 0 0 0 0]
```

**注意点・落とし穴**:
- `LUNAR`は`DeepSVDD`と対照的に、`verbose=0`にすると学習ログが正しく抑制される(実際に確認済み)。同じ「ニューラル系・`verbose`引数あり」でもモデルによって実装の徹底度に差があるため、ログ出力の挙動は個別に確認した方がよい。
- デフォルトの`n_epochs=200`は今回のような小規模データセットには重いため、検証目的では`n_epochs`を大きく減らして動作確認するのが現実的(上の例では20に縮小)。

### 9.3 モデルの永続化

#### `joblib.dump` / `joblib.load` によるモデル保存・復元

**用途**: 学習済みの`pyod`検出器を丸ごとファイルに保存し、後で(同じプロセスを再起動しても)`fit`をやり直さずに`predict`/`decision_function`を呼べるようにする。`pyod`の検出器は通常のPythonオブジェクトなので、scikit-learn同様`joblib`でシリアライズできる。

**シグネチャ**: `joblib.dump(value, filename)` / `joblib.load(filename)`(`joblib`パッケージ、`pyod`固有のAPIではない)

**使用例**:
```python
import numpy as np
from joblib import dump, load
from pyod.models.knn import KNN

clf = KNN(contamination=0.1).fit(X_train)

dump(clf, "/tmp/knn_model.joblib")
clf_loaded = load("/tmp/knn_model.joblib")

print("type:", type(clf_loaded).__name__)
print("predict一致:", np.array_equal(clf.predict(X_test), clf_loaded.predict(X_test)))
print("decision_function一致:", np.allclose(clf.decision_function(X_test), clf_loaded.decision_function(X_test)))
```
実行結果:
```
type: KNN
predict一致: True
decision_function一致: True
```

**注意点・落とし穴**:
- `KNN`のような古典的な検出器だけでなく、PyTorchベースの`AutoEncoder`でも同様に動作することを確認済み(次項参照)。`fit`済みの内部状態(近傍探索木、ニューラルネットの重みなど)を含めてそのまま復元される。
- 保存されるのは学習済みインスタンスそのものであり、`pyod`/`scikit-learn`/`torch`のバージョンが保存時と読み込み時で異なる環境間の互換性までは検証していない(本検証は同一環境内での保存・復元のみ)。

#### `pickle`(標準ライブラリ)との比較

**用途**: `joblib`を使わず、Python標準の`pickle`でも`pyod`検出器(PyTorchベースの`AutoEncoder`を含む)を保存・復元できるかを確認する。

**シグネチャ**: `pickle.dump(obj, file)` / `pickle.load(file)`(標準ライブラリ)

**使用例**:
```python
import pickle
import numpy as np
from pyod.models.auto_encoder import AutoEncoder

ae = AutoEncoder(hidden_neuron_list=[16, 8], epoch_num=5, contamination=0.1,
                  random_state=42, verbose=0)
ae.fit(X_train)  # n_features=10のX_train

with open("/tmp/ae_model.pkl", "wb") as f:
    pickle.dump(ae, f)
with open("/tmp/ae_model.pkl", "rb") as f:
    ae_loaded = pickle.load(f)

print("type:", type(ae_loaded).__name__)
print("decision_function一致:", np.allclose(ae.decision_function(X_test), ae_loaded.decision_function(X_test)))
```
実行結果:
```
type: AutoEncoder
decision_function一致: True
```

**注意点・落とし穴**:
- PyTorchベースの`AutoEncoder`であっても標準`pickle`だけで問題なく保存・復元できることを確認済み(GPU不使用・単一プロセス内の検証)。`joblib`は内部的に`pickle`ベースで大きなNumPy配列の扱いに最適化がある程度で、`pyod`検出器の保存自体に`joblib`が必須というわけではない。
- 同一環境・同一プロセス内での往復のみを確認しており、異なるマシン(特にGPU環境↔CPU環境間)での互換性は本辞書では未検証。

### 9.4 時系列向け異常検知

#### `TimeSeriesOD(...)`

**用途**: 通常の`pyod`検出器(`IForest`, `ECOD`など点ごとの異常検知手法)を時系列データに適用できるようにする「窓化(windowing)」のブリッジクラス。時系列をスライディングウィンドウに切り出して既存の検出器に食わせ、ウィンドウ単位のスコアを元のタイムスタンプ単位に写像し直す。

**シグネチャ**: `pyod.models.ts_od.TimeSeriesOD(detector='IForest', window_size=50, step=1, score_aggregation='max', contamination=0.1)`

**使用例**:
```python
import numpy as np
from pyod.models.ts_od import TimeSeriesOD

rng = np.random.RandomState(42)
ts = np.sin(np.linspace(0, 20 * np.pi, 500)) + rng.normal(0, 0.05, 500)
ts[250:255] += 5  # 異常スパイクを注入(インデックス250-254)

clf = TimeSeriesOD(detector="IForest", window_size=20, step=1, contamination=0.05)
clf.fit(ts)
print("decision_scores_.shape:", clf.decision_scores_.shape)
flagged = np.where(clf.labels_)[0]
print("異常フラグが立った範囲:", flagged.min(), "-", flagged.max())
print("labels_[248:258]:", clf.labels_[248:258])
```
実行結果:
```
decision_scores_.shape: (500,)
異常フラグが立った範囲: 237 - 258
labels_[248:258]: [1 1 1 1 1 1 1 1 1 1]
```

**注意点・落とし穴**:
- `decision_scores_`/`labels_`の長さは元の時系列の長さ(`n_timestamps`)と一致するよう自動的に写像し直される(内部でスライディングウィンドウのスコアを`score_aggregation`(デフォルト`'max'`)で集約)。
- 「窓のにじみ(window smearing)」に注意。250-254の5点にしか異常を注入していないのに、`window_size=20`では237-258という21点分に異常フラグが立った(注入区間の前後に`window_size`程度のマージンで広がる)。異常の正確な発生時刻をピンポイントで特定したい場合は`window_size`を小さくするか、後段で「区間の開始点」を別途特定するロジックが必要。
- `detector`引数には`'IForest'`のような文字列(内部のショートカット登録から解決)だけでなく、他の`pyod`検出器インスタンスをそのまま渡すこともできる(渡した場合は内部で`clone`される)。

#### 多変量時系列への適用

**用途**: `TimeSeriesOD`は1次元の時系列だけでなく、`(n_timestamps, n_channels)`形状の多変量時系列にもそのまま適用できることを確認する。

**シグネチャ**: `TimeSeriesOD.fit(X)` の `X` に `(n_timestamps, n_channels)` 形状の`ndarray`を渡す

**使用例**:
```python
import numpy as np
from pyod.models.ts_od import TimeSeriesOD

rng = np.random.RandomState(42)
t = np.linspace(0, 20 * np.pi, 500)
ts = np.column_stack([np.sin(t), np.cos(t)]) + rng.normal(0, 0.05, (500, 2))
ts[300:303] += 4  # 2チャンネル同時に異常を注入(インデックス300-302)

clf = TimeSeriesOD(detector="ECOD", window_size=15, step=1,
                     score_aggregation="mean", contamination=0.05)
clf.fit(ts)
print("ts.shape:", ts.shape, "-> decision_scores_.shape:", clf.decision_scores_.shape)
flagged = np.where(clf.labels_)[0]
print("異常フラグが立った範囲:", flagged.min(), "-", flagged.max())
```
実行結果:
```
ts.shape: (500, 2) -> decision_scores_.shape: (500,)
異常フラグが立った範囲: 289 - 313
```

**注意点・落とし穴**:
- 多チャンネルの時系列でも`decision_scores_`は`(n_timestamps,)`という1次元配列に集約される(チャンネルごとのスコアは返らない)。
- ここでも「窓のにじみ」が確認できる。300-302の3点にしか異常を注入していないが、`window_size=15`では289-313という25点分にフラグが立っており、`window_size`が大きいほどにじみ幅も広がる傾向が見て取れる。

### 9.5 実データでのcontamination推定の実践パターン

#### `contamination`は`decision_scores_`自体には影響しない

**用途**: `contamination`パラメータが検出器のどこに効いているのかを実際に確認する。実務では「正しいcontamination値が分からない」状態でまずスコアだけ計算し、後から閾値を調整したいことが多いため、この性質を理解しておくと効率的に試行できる。

**シグネチャ**: `ECOD(contamination=<float>)`(`BaseDetector`共通のパラメータ)

**使用例**:
```python
import numpy as np
from pyod.models.ecod import ECOD

ecod_a = ECOD(contamination=0.05).fit(X_train)
ecod_b = ECOD(contamination=0.3).fit(X_train)
print("decision_scores_が一致:", np.allclose(ecod_a.decision_scores_, ecod_b.decision_scores_))

for c in [0.05, 0.1, 0.2, 0.3]:
    clf = ECOD(contamination=c).fit(X_train)
    print("contamination=%.2f -> labels_.sum()=%d, threshold_=%.3f" % (c, clf.labels_.sum(), clf.threshold_))
```
実行結果:
```
decision_scores_が一致: True
contamination=0.05 -> labels_.sum()=10, threshold_=6.321
contamination=0.10 -> labels_.sum()=20, threshold_=5.648
contamination=0.20 -> labels_.sum()=40, threshold_=4.368
contamination=0.30 -> labels_.sum()=60, threshold_=3.775
```

**注意点・落とし穴**:
- `contamination`は`decision_scores_`(連続スコア)の計算には一切影響せず、`threshold_`と`labels_`(0/1判定)にのみ影響する。実務では、まず`contamination`のデフォルト値(0.1など仮の値)でモデルを`fit`してスコアを確認し、後から`threshold_`相当の閾値だけを別途探索する(9.1節の`thresholds`モジュールや、`labels_.sum()`をターゲットの異常件数に合わせて`contamination`を逆算する、など)方が、モデルの再学習コストを抑えられる。
- 上の実行結果からも分かる通り、`labels_.sum()`は`contamination × n_train`にほぼ比例する(200件中、0.05→10件、0.1→20件、0.2→40件、0.3→60件)。`contamination`は「割合」であって「件数」ではない点に注意。

#### thresholderによる自動contamination推定とその限界

**用途**: 正解ラベルが手元にない実データで、9.1節の`thresholds`モジュールを使って「妥当なcontamination」を推定するパターンと、その手法間のブレの大きさを確認する。

**シグネチャ**: `pyod.models.thresholds.AUCP().eval(decision_scores_)` / `pyod.models.thresholds.FILTER().eval(decision_scores_)`

**使用例**:
```python
from pyod.models.ecod import ECOD
from pyod.models.thresholds import AUCP, FILTER

ecod = ECOD().fit(X_train)
scores = ecod.decision_scores_

for name, thres in [("AUCP", AUCP()), ("FILTER", FILTER())]:
    labels = thres.eval(scores)
    print(name, "推定contamination:", round(labels.mean(), 4))
print("正解(参考):", round(y_train.mean(), 4))
```
実行結果:
```
AUCP 推定contamination: 0.305
FILTER 推定contamination: 0.08
正解(参考): 0.1
```

**注意点・落とし穴**:
- 同じ`ECOD`のスコアに対して、`AUCP`は0.305、`FILTER`は0.08と、真の値0.1を挟んで大きく異なる推定値を出した。同じデータに対して同じ検出器のスコアを使っても、どのthresholderを選ぶかで結果が数倍単位でブレうることが実際に確認できる。
- 実務でのパターンとしては、(1) 複数のthresholderを試してレンジ(この例なら0.08〜0.305)を把握する、(2) 可能であれば少数のラベル付きサンプルやドメイン知識(「過去の実績では異常は全体の数%程度」等)で検証・補正する、(3) 単一の自動推定値をそのまま本番の閾値として採用しない、という3点が安全側の運用になる。9.1節で見た通り`KNN`スコアに対する`AUCP`(0.095)は真値0.1に近かったが、これは検出器・データセットの組み合わせに依存する結果であり、一般に「この手法が最も正確」と言えるだけの根拠は今回の検証範囲にはない。
