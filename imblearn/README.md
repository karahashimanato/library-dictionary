# imbalanced-learn 逆引き辞書

imbalanced-learn 0.14.2 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(imbalanced-learn 0.14.2)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。scikit-learnの拡張ライブラリであるため、多くのクラスは`fit`/`fit_transform`ではなく`fit_resample`という独自メソッドを持つ点に注意。

## 目次

1. [オーバーサンプリング](#1-オーバーサンプリング)
2. [アンダーサンプリング](#2-アンダーサンプリング)
3. [複合サンプリング](#3-複合サンプリング)
4. [パイプライン](#4-パイプライン)
5. [アンサンブル学習](#5-アンサンブル学習)
6. [評価指標](#6-評価指標)
7. [その他ユーティリティ](#7-その他ユーティリティ)
8. [応用・発展](#8-応用発展)

---

## 1. オーバーサンプリング

以下1〜3節では共通して、次の不均衡な人工データ(多数派クラス900件・少数派クラス100件、比率9:1)を使う。

```python
from sklearn.datasets import make_classification
from collections import Counter

X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=0,
)
print(Counter(y))
```
実行結果:
```
Counter({np.int64(0): 900, np.int64(1): 100})
```

### `RandomOverSampler(...)`

**用途**: 少数派クラスのサンプルを単純に複製(ブートストラップ)して多数派に合わせる、最もシンプルなオーバーサンプリング。

**シグネチャ**: `imblearn.over_sampling.RandomOverSampler(*, sampling_strategy='auto', random_state=None, shrinkage=None)`

**使用例**:
```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
Xr, yr = ros.fit_resample(X, y)
print(Xr.shape, Counter(yr))
```
実行結果:
```
(1800, 5) Counter({np.int64(0): 900, np.int64(1): 900})
```

**注意点・落とし穴**:
- 単純複製のため、少数派の同じサンプルが何度も現れる。決定木系モデルでは同じサンプルに対して同じ分岐が繰り返し選ばれ、過学習の原因になりやすい。`shrinkage`(smoothed bootstrap)で複製にノイズを加えることもできる。
- `sampling_strategy`はデフォルト`'auto'`(2値分類では多数派件数に少数派を合わせる)のほか、floatで「少数派/多数派の目標比率」、dictで「クラスごとの目標件数」を直接指定できる(実際に`sampling_strategy=0.5`で少数派が450件に、`sampling_strategy={0: 200, 1: 100}`(`RandomUnderSampler`側)で狙った件数になることを確認済み)。

### `SMOTE(...)`

**用途**: 少数派サンプルとそのk近傍を線形補間して合成サンプルを生成するオーバーサンプリング(Synthetic Minority Oversampling Technique)。

**シグネチャ**: `imblearn.over_sampling.SMOTE(*, sampling_strategy='auto', random_state=None, k_neighbors=5)`

**使用例**:
```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
Xs, ys = sm.fit_resample(X, y)
print(Xs.shape, Counter(ys))
```
実行結果:
```
(1800, 5) Counter({np.int64(0): 900, np.int64(1): 900})
```

**注意点・落とし穴**:
- 数値特徴量のみを想定している。カテゴリ変数が混ざる場合は`SMOTENC`を使う必要がある(7節参照)。
- `k_neighbors`(デフォルト5)以下の少数派サンプル数しかないとエラーになる。
- 実在しない「合成」サンプルを生成するため、外れ値同士を補間すると不自然な(現実にはあり得ない)サンプルが生まれることがある。

### `ADASYN(...)`

**用途**: SMOTEと似るが、分類境界付近(誤分類されやすい=学習が難しい)の少数派サンプル周辺に重点的に合成サンプルを生成する(Adaptive Synthetic Sampling)。

**シグネチャ**: `imblearn.over_sampling.ADASYN(*, sampling_strategy='auto', random_state=None, n_neighbors=5)`

**使用例**:
```python
from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=0)
Xa, ya = ada.fit_resample(X, y)
print(Xa.shape, Counter(ya))
```
実行結果:
```
(1790, 5) Counter({np.int64(0): 900, np.int64(1): 890})
```

**注意点・落とし穴**:
- 生成数がサンプルごとの「学習の難しさ」に応じて動的に決まるため、`SMOTE`と違い`fit_resample`後の少数派件数が要求通り厳密に900件にならないことがある(今回の検証でも900ではなく890件)。

### `BorderlineSMOTE(...)`

**用途**: 境界線付近(多数派に近く「danger」と判定される)少数派サンプルのみを対象にSMOTE的補間を行う。

**シグネチャ**: `imblearn.over_sampling.BorderlineSMOTE(*, sampling_strategy='auto', random_state=None, k_neighbors=5, m_neighbors=10, kind='borderline-1')`

**使用例**:
```python
from imblearn.over_sampling import BorderlineSMOTE
bsm = BorderlineSMOTE(random_state=0, kind="borderline-1")
Xb, yb = bsm.fit_resample(X, y)
print(Xb.shape, Counter(yb))
```
実行結果:
```
(1800, 5) Counter({np.int64(0): 900, np.int64(1): 900})
```

**注意点・落とし穴**:
- `kind='borderline-1'`(デフォルト)と`'borderline-2'`で挙動が変わる(2は多数派側のサンプルとも補間しうる)。
- 安全域やノイズと判定された少数派サンプルは合成の対象にならない。データによっては「danger」判定されるサンプルが少なく、目標件数に届かない場合がある。

### `SVMSMOTE(...)`

**用途**: SVMの決定境界(サポートベクター)周辺を基準に少数派サンプルを合成するSMOTEの派生手法。

**シグネチャ**: `imblearn.over_sampling.SVMSMOTE(*, sampling_strategy='auto', random_state=None, k_neighbors=5, m_neighbors=10, svm_estimator=None, out_step=0.5)`

**使用例**:
```python
from imblearn.over_sampling import SVMSMOTE
svmsm = SVMSMOTE(random_state=0)
Xv, yv = svmsm.fit_resample(X, y)
print(Xv.shape, Counter(yv))
```
実行結果:
```
(1800, 5) Counter({np.int64(0): 900, np.int64(1): 900})
```

**注意点・落とし穴**:
- `svm_estimator=None`の場合、内部で`SVC`が学習されるため、他のSMOTE系よりも計算コストが高くなりがち。

---

## 2. アンダーサンプリング

以下も1節と同じ`X, y`(多数派900件・少数派100件)を使う。

### `RandomUnderSampler(...)`

**用途**: 多数派クラスのサンプルをランダムに間引いて少数派に合わせる、最もシンプルなアンダーサンプリング。

**シグネチャ**: `imblearn.under_sampling.RandomUnderSampler(*, sampling_strategy='auto', random_state=None, replacement=False)`

**使用例**:
```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
Xr, yr = rus.fit_resample(X, y)
print(Xr.shape, Counter(yr))
```
実行結果:
```
(200, 5) Counter({np.int64(0): 100, np.int64(1): 100})
```

**注意点・落とし穴**:
- ランダムに間引くため、多数派側の重要な情報を持つサンプルも失われうる(情報損失)。
- `replacement=False`(デフォルト)は非復元抽出。`sampling_strategy`にdictを渡すとクラスごとの目標件数を厳密に指定できる(例: `{0: 200, 1: 100}` → 検証済み)。

### `TomekLinks(...)`

**用途**: 異なるクラス同士で互いに最近傍になっている(Tomekリンクを形成する)サンプルペアを検出し、多数派側だけを除去してクラス境界を明確化する。

**シグネチャ**: `imblearn.under_sampling.TomekLinks(*, sampling_strategy='auto', n_jobs=None)`

**使用例**:
```python
from imblearn.under_sampling import TomekLinks
tl = TomekLinks()
Xt, yt = tl.fit_resample(X, y)
print(Xt.shape, Counter(yt))
```
実行結果:
```
(998, 5) Counter({np.int64(0): 898, np.int64(1): 100})
```

**注意点・落とし穴**:
- クラス数を揃える手法ではなく、境界が重複した曖昧なサンプルだけを除去するノイズ除去手法。単体では不均衡はほとんど解消されない(検証でも900→898件と2件しか減らない)。
- `random_state`を持たない(決定的なアルゴリズムのため)。`SMOTETomek`のように他の手法と組み合わせて使うのが一般的。

### `NearMiss(...)`

**用途**: 少数派サンプルとの距離に基づき多数派サンプルを選択的に間引く(`version`によって選択基準が異なる)。

**シグネチャ**: `imblearn.under_sampling.NearMiss(*, sampling_strategy='auto', version=1, n_neighbors=3, n_neighbors_ver3=3, n_jobs=None)`

**使用例**:
```python
from imblearn.under_sampling import NearMiss
nm = NearMiss(version=1)
Xn, yn = nm.fit_resample(X, y)
print(Xn.shape, Counter(yn))
```
実行結果:
```
(200, 5) Counter({np.int64(0): 100, np.int64(1): 100})
```

**注意点・落とし穴**:
- `version=1`は少数派k近傍との平均距離が最小の多数派サンプルを残す、`version=2`は最も遠い少数派とも近いサンプルを残す、`version=3`は2段階選択、と`version`ごとにアルゴリズムが異なる。目的に応じて明示的に指定すること。
- 距離ベースの手法なのでスケーリングが結果に影響する。

### `EditedNearestNeighbours(...)`

**用途**: 各サンプルのk近傍の多数決と自分のラベルが一致しない(≒ノイズ・境界にある)多数派サンプルを除去するクリーニング手法。

**シグネチャ**: `imblearn.under_sampling.EditedNearestNeighbours(*, sampling_strategy='auto', n_neighbors=3, kind_sel='all', n_jobs=None)`

**使用例**:
```python
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours()
Xe, ye = enn.fit_resample(X, y)
print(Xe.shape, Counter(ye))
```
実行結果:
```
(988, 5) Counter({np.int64(0): 888, np.int64(1): 100})
```

**注意点・落とし穴**:
- `TomekLinks`同様、クラス数を揃えるための手法ではなくノイズ除去が目的。
- `kind_sel='all'`(デフォルト、近傍全てが不一致なら除去)と`'mode'`(多数決で不一致なら除去。より緩い基準で除去件数が増えやすい)で挙動が変わる。

### `RepeatedEditedNearestNeighbours(...)`

**用途**: `EditedNearestNeighbours`を、変化がなくなるか`max_iter`に達するまで繰り返し適用する。

**シグネチャ**: `imblearn.under_sampling.RepeatedEditedNearestNeighbours(*, sampling_strategy='auto', n_neighbors=3, max_iter=100, kind_sel='all', n_jobs=None)`

**使用例**:
```python
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
renn = RepeatedEditedNearestNeighbours()
Xre, yre = renn.fit_resample(X, y)
print(Xre.shape, Counter(yre))
```
実行結果:
```
(988, 5) Counter({np.int64(0): 888, np.int64(1): 100})
```

**注意点・落とし穴**:
- 今回のデータでは1回目の適用で収束し、`EditedNearestNeighbours`単体と全く同じ結果(888件)になった。収束するまで反復するため、除去数は単発のENN以上にしかならない(減ることはない)。

### `AllKNN(...)`

**用途**: k近傍数を1から`n_neighbors`まで段階的に増やしながらENNを繰り返し適用する派生手法。

**シグネチャ**: `imblearn.under_sampling.AllKNN(*, sampling_strategy='auto', n_neighbors=3, kind_sel='all', allow_minority=False, n_jobs=None)`

**使用例**:
```python
from imblearn.under_sampling import AllKNN
aknn = AllKNN()
Xak, yak = aknn.fit_resample(X, y)
print(Xak.shape, Counter(yak))
```
実行結果:
```
(988, 5) Counter({np.int64(0): 888, np.int64(1): 100})
```

**注意点・落とし穴**:
- `allow_minority=True`にすると、多数派サンプルが少数派件数を下回るまで縮小することを許容する(デフォルト`False`では少数派件数が下限になる)。

### `CondensedNearestNeighbour(...)`

**用途**: 1-NN分類器を使って境界を保つ最小限の代表的サンプル集合を多数派から選び出す(圧縮近傍法)。

**シグネチャ**: `imblearn.under_sampling.CondensedNearestNeighbour(*, sampling_strategy='auto', random_state=None, n_neighbors=None, n_seeds_S=1, n_jobs=None)`

**使用例**:
```python
from imblearn.under_sampling import CondensedNearestNeighbour
cnn = CondensedNearestNeighbour(random_state=0)
Xc, yc = cnn.fit_resample(X, y)
print(Xc.shape, Counter(yc))
```
実行結果:
```
(148, 5) Counter({np.int64(1): 100, np.int64(0): 48})
```

**注意点・落とし穴**:
- 非常に強いアンダーサンプリングになりうる(今回は多数派900件→48件まで縮小)。多数派が少数派件数を下回ることもある(ここでは多数派48件<少数派100件)。
- 逐次的なアルゴリズムで、`RandomUnderSampler`等より計算コストが高い。

### `OneSidedSelection(...)`

**用途**: `TomekLinks`の除去と`CondensedNearestNeighbour`の代表点選択を組み合わせたアンダーサンプリング。

**シグネチャ**: `imblearn.under_sampling.OneSidedSelection(*, sampling_strategy='auto', random_state=None, n_neighbors=None, n_seeds_S=1, n_jobs=None)`

**使用例**:
```python
from imblearn.under_sampling import OneSidedSelection
oss = OneSidedSelection(random_state=0)
Xo, yo = oss.fit_resample(X, y)
print(Xo.shape, Counter(yo))
```
実行結果:
```
(805, 5) Counter({np.int64(0): 705, np.int64(1): 100})
```

**注意点・落とし穴**:
- `CondensedNearestNeighbour`ほど極端には削減されない(境界ノイズの除去のみ行い、代表点選択の基準がCNNより緩やか)。

### `ClusterCentroids(...)`

**用途**: 多数派クラスをKMeansでクラスタリングし、各クラスタの重心を代表サンプルとして置き換えるアンダーサンプリング。

**シグネチャ**: `imblearn.under_sampling.ClusterCentroids(*, sampling_strategy='auto', random_state=None, estimator=None, voting='auto')`

**使用例**:
```python
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
Xcc, ycc = cc.fit_resample(X, y)
print(Xcc.shape, Counter(ycc))
```
実行結果:
```
(200, 5) Counter({np.int64(0): 100, np.int64(1): 100})
```

**注意点・落とし穴**:
- 生成されるのは実サンプルではなく「重心」という人工的な点(`voting='hard'`にすると、重心に最も近い実データ点に置き換えられる)。
- 内部でKMeansを学習するため、他のアンダーサンプリング手法より計算コストが高い。

### `NeighbourhoodCleaningRule(...)`

**用途**: `EditedNearestNeighbours`をベースに、少数派の近傍で誤分類の原因となる多数派サンプルも追加的に除去するクリーニング手法。

**シグネチャ**: `imblearn.under_sampling.NeighbourhoodCleaningRule(*, sampling_strategy='auto', edited_nearest_neighbours=None, n_neighbors=3, threshold_cleaning=0.5, n_jobs=None)`

**使用例**:
```python
from imblearn.under_sampling import NeighbourhoodCleaningRule
ncr = NeighbourhoodCleaningRule()
Xncr, yncr = ncr.fit_resample(X, y)
print(Xncr.shape, Counter(yncr))
```
実行結果:
```
(973, 5) Counter({np.int64(0): 873, np.int64(1): 100})
```

**注意点・落とし穴**:
- ENN単体(888件)よりもやや多く除去される(873件)。`threshold_cleaning`は追加除去を行うかどうかを判定するクラス比率の閾値。

---

## 3. 複合サンプリング

こちらも1節と同じ`X, y`を使う。

### `SMOTEENN(...)`

**用途**: SMOTEでオーバーサンプリングした後、`EditedNearestNeighbours`で(合成サンプルを含む)ノイズ的なサンプルをクリーニングする複合手法。

**シグネチャ**: `imblearn.combine.SMOTEENN(*, sampling_strategy='auto', random_state=None, smote=None, enn=None, n_jobs=None)`

**使用例**:
```python
from imblearn.combine import SMOTEENN
se = SMOTEENN(random_state=0)
Xse, yse = se.fit_resample(X, y)
print(Xse.shape, Counter(yse))
```
実行結果:
```
(1780, 5) Counter({np.int64(1): 897, np.int64(0): 883})
```

**注意点・落とし穴**:
- SMOTE単体では900:900ちょうどになるが、後段のENNで両クラスから少しずつ削られるため、完全な1:1にはならない(検証では883:897)。
- `smote`/`enn`引数にそれぞれのインスタンスを渡してパラメータをカスタマイズできる(`None`ならデフォルト設定)。オーバーサンプリング+クリーニングの2段構成のため、`SMOTE`単体より計算コストが高い。

### `SMOTETomek(...)`

**用途**: SMOTEでオーバーサンプリングした後、`TomekLinks`で境界の重複サンプルを除去する複合手法。

**シグネチャ**: `imblearn.combine.SMOTETomek(*, sampling_strategy='auto', random_state=None, smote=None, tomek=None, n_jobs=None)`

**使用例**:
```python
from imblearn.combine import SMOTETomek
st = SMOTETomek(random_state=0)
Xst, yst = st.fit_resample(X, y)
print(Xst.shape, Counter(yst))
```
実行結果:
```
(1800, 5) Counter({np.int64(0): 900, np.int64(1): 900})
```

**注意点・落とし穴**:
- 今回のデータではSMOTE後にTomekLinksによる除去が発生せず、SMOTE単体と同じ1800件になった。TomekLinksの除去件数はデータ依存であり、必ずSMOTE単体と異なる結果になるとは限らない。
- 一般に`SMOTEENN`よりマイルドなクリーニングとされる(ENNの方が除去基準が厳しい)。

---

## 4. パイプライン

### `imblearn.pipeline.Pipeline(...)`

**用途**: 前処理・リサンプリング・推定器を1つにまとめる。scikit-learn本体の`Pipeline`と異なり、`fit_resample`を持つサンプラー(`SMOTE`等)をステップとして組み込める。

**シグネチャ**: `imblearn.pipeline.Pipeline(steps, *, transform_input=None, memory=None, verbose=False)`

**使用例**:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=0)),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(Xtr, ytr)
print("score:", pipe.score(Xte, yte))
print("steps:", [name for name, _ in pipe.steps])
```
実行結果:
```
score: 1.0
steps: ['scaler', 'smote', 'clf']
```

**注意点・落とし穴(重要)**:
- 同じ`SMOTE`ステップをscikit-learn本体の`sklearn.pipeline.Pipeline`に組み込んで`fit`すると、実際に次のエラーになる(検証済み)。
  ```python
  from sklearn.pipeline import Pipeline as SkPipeline
  skpipe = SkPipeline([("smote", SMOTE(random_state=0)), ("clf", LogisticRegression(max_iter=1000))])
  skpipe.fit(Xtr, ytr)
  ```
  実行結果:
  ```
  TypeError: All intermediate steps should be transformers and implement fit and transform or be the string 'passthrough' 'SMOTE(random_state=0)' (type <class 'imblearn.over_sampling._smote.base.SMOTE'>) doesn't
  ```
  sklearnの`Pipeline`は中間ステップに`transform`の実装(=入力と出力でサンプル数=行数を変えない変換)を要求する。`SMOTE`等のサンプラーは`fit_resample`でXとyの行数そのものを変えるため、この契約に合わずエラーになる。imblearnの`Pipeline`はサンプラー(`fit_resample`を実装するオブジェクト)を認識し、`fit`時のみリサンプリングを適用してyも連動して更新し、`predict`/`transform`時にはリサンプリングをスキップして元のX件数のまま予測する、という特別な仕組みを持っている。
- リサンプリングは必ず学習データ側だけに適用する。学習データ全体に対して一度だけ`fit_resample`し、その結果をそのまま交差検証や評価に流用すると、複製・合成されたサンプルがテスト側にも漏れる(データリーク)。`imblearn.pipeline.Pipeline`を`cross_val_score`等にそのまま渡し、各foldの学習側でのみ`fit_resample`が実行されるようにするのが安全。

### `imblearn.pipeline.make_pipeline(...)`

**用途**: `Pipeline`のステップ名を自動生成する簡易コンストラクタ(scikit-learnの`make_pipeline`のimblearn版)。

**シグネチャ**: `imblearn.pipeline.make_pipeline(*steps, memory=None, transform_input=None, verbose=False)`

**使用例**:
```python
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
pipe2 = make_pipeline(RandomUnderSampler(random_state=0), LogisticRegression(max_iter=1000))
pipe2.fit(Xtr, ytr)
print("score:", pipe2.score(Xte, yte))
print(pipe2.steps[0][0], pipe2.steps[1][0])
```
実行結果:
```
score: 0.9866666666666667
randomundersampler logisticregression
```

**注意点・落とし穴**:
- ステップ名はクラス名を小文字化して自動生成される(`Pipeline`と違い、任意の名前を明示的に指定することはできない)。

---

## 5. アンサンブル学習

以下は1節の`X, y`を`train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)`で分割し、学習・評価に使う。

```python
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print("train:", Counter(ytr), "test:", Counter(yte))
```
実行結果:
```
train: Counter({np.int64(0): 630, np.int64(1): 70}) test: Counter({np.int64(0): 270, np.int64(1): 30})
```

### `BalancedRandomForestClassifier(...)`

**用途**: 各決定木を学習する際にクラスごとに均等な件数のブートストラップサンプルを抽出する(内部でアンダーサンプリングする)ランダムフォレスト。

**シグネチャ**: `imblearn.ensemble.BalancedRandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, sampling_strategy='all', replacement=True, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)`

**使用例**:
```python
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
brf.fit(Xtr, ytr)
pred = brf.predict(Xte)
print("accuracy:", accuracy_score(yte, pred), "balanced_accuracy:", balanced_accuracy_score(yte, pred))
```
実行結果:
```
accuracy: 0.9466666666666667 balanced_accuracy: 0.9259259259259259
```

**注意点・落とし穴**:
- 通常の`sklearn.ensemble.RandomForestClassifier`とはデフォルト値が異なる(`bootstrap=False`、`sampling_strategy='all'`で全クラスを最小クラスの件数に合わせる、`replacement=True`)。インポート元も`imblearn.ensemble`であり、`sklearn.ensemble.RandomForestClassifier`とは別クラスである点に注意。

### `BalancedBaggingClassifier(...)`

**用途**: バギングの各反復でサブサンプルをアンダーサンプリングしてから弱学習器を学習させる分類器。

**シグネチャ**: `imblearn.ensemble.BalancedBaggingClassifier(estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, sampling_strategy='auto', replacement=False, n_jobs=None, random_state=None, verbose=0, sampler=None)`

**使用例**:
```python
from imblearn.ensemble import BalancedBaggingClassifier
bbc = BalancedBaggingClassifier(n_estimators=10, random_state=0)
bbc.fit(Xtr, ytr)
pred = bbc.predict(Xte)
print("estimator_:", bbc.estimator_)
print("accuracy:", accuracy_score(yte, pred), "balanced_accuracy:", balanced_accuracy_score(yte, pred))
```
実行結果:
```
estimator_: Pipeline(steps=[('sampler', RandomUnderSampler()),
                ('classifier', DecisionTreeClassifier())])
accuracy: 0.93 balanced_accuracy: 0.9166666666666667
```

**注意点・落とし穴**:
- `estimator=None`(デフォルト)の場合、内部で自動的に`DecisionTreeClassifier`が使われる。学習後の`estimator_`を見ると、各バギング反復が実際には`Pipeline(RandomUnderSampler -> DecisionTreeClassifier)`として実装されていることが確認できる(検証済み)。`sampler`引数を指定すれば`RandomUnderSampler`以外のサンプラーに差し替えられる。

### `EasyEnsembleClassifier(...)`

**用途**: 多数派クラスをランダムにアンダーサンプリングした複数のサブセットそれぞれでAdaBoostを学習し、その予測を統合するアンサンブル。

**シグネチャ**: `imblearn.ensemble.EasyEnsembleClassifier(n_estimators=10, estimator=None, *, warm_start=False, sampling_strategy='auto', replacement=False, n_jobs=None, random_state=None, verbose=0)`

**使用例**:
```python
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(n_estimators=10, random_state=0)
eec.fit(Xtr, ytr)
pred = eec.predict(Xte)
print("balanced_accuracy:", balanced_accuracy_score(yte, pred))
```
実行結果:
```
balanced_accuracy: 0.9537037037037037
```

**注意点・落とし穴**:
- `estimator=None`の場合、内部で`AdaBoostClassifier`が使われる(`BalancedBaggingClassifier`のデフォルト`DecisionTreeClassifier`とは異なる)。今回の検証データでは4手法中もっとも高いbalanced accuracyだったが、これはデータ依存であり常に最良とは限らない。

### `RUSBoostClassifier(...)`

**用途**: AdaBoostの各ブースティング反復の前に`RandomUnderSampler`でアンダーサンプリングを挟むブースティング手法。

**シグネチャ**: `imblearn.ensemble.RUSBoostClassifier(estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='deprecated', sampling_strategy='auto', replacement=False, random_state=None)`

**使用例**:
```python
from imblearn.ensemble import RUSBoostClassifier
rbc = RUSBoostClassifier(n_estimators=50, random_state=0)
rbc.fit(Xtr, ytr)
pred = rbc.predict(Xte)
print("balanced_accuracy:", balanced_accuracy_score(yte, pred))
```
実行結果:
```
balanced_accuracy: 0.8037037037037037
```

**注意点・落とし穴**:
- `algorithm`引数はimbalanced-learn 0.14.2時点で非推奨(デフォルト値が`'deprecated'`と表示される)。
- 今回の検証では4つのアンサンブル手法の中で最もbalanced accuracyが低かったが、データ・ハイパーパラメータに強く依存するため、常に他手法より劣るという意味ではない。

---

## 6. 評価指標

以下は、上記の`Xtr, ytr, Xte, yte`にロジスティック回帰(クラス重み調整なし)を適用した結果(`yte`, `pred`)を使う。

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
pred = clf.predict(Xte)
```

### `geometric_mean_score(...)`

**用途**: 各クラスの再現率(sensitivity)の幾何平均を計算する評価指標。多数派・少数派どちらかに偏った予測を罰する。

**シグネチャ**: `imblearn.metrics.geometric_mean_score(y_true, y_pred, *, labels=None, pos_label=1, average='multiclass', sample_weight=None, correction=0.0)`

**使用例**:
```python
from imblearn.metrics import geometric_mean_score
print(round(geometric_mean_score(yte, pred), 4))
```
実行結果:
```
0.9832
```

**注意点・落とし穴**:
- 通常のaccuracyと異なり、多数派クラスに極端に有利な予測をしても、少数派の再現率が低ければスコアも低くなる。

### `classification_report_imbalanced(...)`

**用途**: precision/recall/specificity/f1/geometric mean/index balanced accuracyをクラスごとに一覧表示する、不均衡データ向けのレポート。

**シグネチャ**: `imblearn.metrics.classification_report_imbalanced(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, alpha=0.1, output_dict=False, zero_division='warn')`

**使用例**:
```python
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(yte, pred))
```
実行結果:
```
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      1.00      0.97      1.00      0.98      0.97       270
          1       1.00      0.97      1.00      0.98      0.98      0.96        30

avg / total       1.00      1.00      0.97      1.00      0.98      0.97       300
```

**注意点・落とし穴**:
- scikit-learn本体の`classification_report`にはない`spe`(specificity, 特異度)・`geo`(geometric mean)・`iba`(index balanced accuracy)列が追加されている。`output_dict=True`で辞書としても取得できる。

### `sensitivity_specificity_support(...)`

**用途**: 再現率(sensitivity=recall)と特異度(specificity)をサポート数とともに計算する。

**シグネチャ**: `imblearn.metrics.sensitivity_specificity_support(y_true, y_pred, *, labels=None, pos_label=1, average=None, warn_for=('sensitivity', 'specificity'), sample_weight=None)`

**使用例**:
```python
from imblearn.metrics import sensitivity_specificity_support
print(sensitivity_specificity_support(yte, pred, average="binary"))
```
実行結果:
```
(np.float64(0.9666666666666667), np.float64(1.0), None)
```

**注意点・落とし穴**:
- 戻り値は`(sensitivity, specificity, support)`のタプル。`average='binary'`を指定してもサポート数は`None`のままになる(検証済み)。`scikit-learn`の`precision_recall_fscore_support`に近いインターフェース。

### `make_index_balanced_accuracy(...)`

**用途**: 任意の評価指標(例: `geometric_mean_score`)をラップし、優勢クラスに偏った評価にペナルティを与える"Index Balanced Accuracy(IBA)"版のスコア関数を作る高階関数(デコレータ)。

**シグネチャ**: `imblearn.metrics.make_index_balanced_accuracy(*, alpha=0.1, squared=True)`

**使用例**:
```python
from imblearn.metrics import make_index_balanced_accuracy
iba_gmean = make_index_balanced_accuracy(alpha=0.1, squared=True)(geometric_mean_score)
print(round(iba_gmean(yte, pred), 4))
```
実行結果:
```
0.9667
```

**注意点・落とし穴**:
- それ自体はスコアを計算せず、スコア関数を受け取ってラップ済みの新しい関数を返す(デコレータ的な使い方をする)。`classification_report_imbalanced`の`iba`列は内部でこの仕組みを使って計算されている。

---

## 7. その他ユーティリティ

### `SMOTENC(...)`

**用途**: カテゴリ変数と数値変数が混在するデータに対応したSMOTE。数値列のみSMOTE的に補間し、カテゴリ列は近傍探索時の距離計算を調整して扱う。

**シグネチャ**: `imblearn.over_sampling.SMOTENC(categorical_features, *, categorical_encoder=None, sampling_strategy='auto', random_state=None, k_neighbors=5)`

**使用例**:
```python
import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.datasets import make_classification

Xn, yn = make_classification(n_samples=200, n_features=4, n_informative=3, n_redundant=0,
                              n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=0)
rng = np.random.RandomState(0)
cat_col = rng.randint(0, 3, size=Xn.shape[0]).reshape(-1, 1).astype(float)
Xmix = np.hstack([Xn[:, :3], cat_col])  # 4列目をカテゴリ変数(0/1/2)にする
print("before:", Counter(yn))
smnc = SMOTENC(categorical_features=[3], random_state=0)
Xr, yr = smnc.fit_resample(Xmix, yn)
print("after:", Counter(yr))
print("resampled categorical column unique values:", np.unique(Xr[:, 3]))
```
実行結果:
```
before: Counter({np.int64(0): 180, np.int64(1): 20})
after: Counter({np.int64(0): 180, np.int64(1): 180})
resampled categorical column unique values: [0. 1. 2.]
```

**注意点・落とし穴**:
- `categorical_features`(カテゴリ列の位置。列インデックスまたはブールマスク)は必須の位置引数。
- 生成後もカテゴリ列の値は必ず元のカテゴリ値のいずれかになる(数値列のような連続値の補間はされない。検証でも生成後のユニーク値が`[0, 1, 2]`のまま変わらないことを確認)。全列がカテゴリ変数の場合は`SMOTEN`を使う。

### `imblearn.datasets.make_imbalance(...)`

**用途**: 既存の(通常はクラスバランスの取れた)データセットから、指定したクラスごとのサンプル数になるよう間引いて人工的に不均衡データを作る。

**シグネチャ**: `imblearn.datasets.make_imbalance(X, y, *, sampling_strategy=None, random_state=None, verbose=False, **kwargs)`

**使用例**:
```python
from sklearn.datasets import load_iris
from imblearn.datasets import make_imbalance
iris = load_iris()
print("original:", Counter(iris.target))
Xi, yi = make_imbalance(iris.data, iris.target, sampling_strategy={0: 10, 1: 30, 2: 40}, random_state=0)
print("after:", Counter(yi))
```
実行結果:
```
original: Counter({np.int64(0): 50, np.int64(1): 50, np.int64(2): 50})
after: Counter({np.int64(2): 40, np.int64(1): 30, np.int64(0): 10})
```

**注意点・落とし穴**:
- `sampling_strategy`に各クラスの元の件数を上回る値を指定するとエラーになる(間引くだけで増やすことはできない)。動作確認・アルゴリズム検証用にわざと不均衡データを作りたいときに便利。

### `imblearn.base.FunctionSampler(...)`

**用途**: 任意のPython関数を、imblearnの「サンプラー」(`fit_resample`を持つオブジェクト)としてラップし、`imblearn.pipeline.Pipeline`に組み込めるようにする。外れ値除去など、既存のSMOTE系にはない独自のサンプル数変更処理をパイプライン化したい場合に使う。

**シグネチャ**: `imblearn.base.FunctionSampler(*, func=None, accept_sparse=True, kw_args=None, validate=True)`

**使用例**:
```python
import numpy as np
from sklearn.datasets import make_classification
from imblearn.base import FunctionSampler

def outlier_rejection(X, y):
    # 各特徴量の平均から3標準偏差以上離れたサンプルを除去する(トイ例)
    mask = (np.abs(X - X.mean(axis=0)) < 3 * X.std(axis=0)).all(axis=1)
    return X[mask], y[mask]

X2, y2 = make_classification(n_samples=300, n_features=3, n_informative=2, n_redundant=0,
                              n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=0)
rng = np.random.RandomState(0)
outlier_idx = rng.choice(len(X2), size=5, replace=False)
X2[outlier_idx] += 50  # 明らかな外れ値を混入させる

fs = FunctionSampler(func=outlier_rejection)
Xf, yf = fs.fit_resample(X2, y2)
print("before:", X2.shape, "after:", Xf.shape)
```
実行結果:
```
before: (300, 3) after: (295, 3)
```

**注意点・落とし穴**:
- `func=None`(デフォルト)の場合、何もしない恒等サンプラーになる。
- scikit-learn本体の`FunctionTransformer`と異なり、戻り値でXとyのサンプル数(行数)を変えることが明示的に許可されている点が本質的な違い(`FunctionTransformer`はサンプル数を変えない変換を想定している)。これにより外れ値除去のような処理も`Pipeline`のステップとして組み込める。

---

## 8. 応用・発展

**8-1. カスタムサンプラーの自作**

### `imblearn.base.BaseSampler`

**用途**: 既存のSMOTE系・アンダーサンプリング系にない独自のリサンプリングロジックを、`fit_resample`を持つ「サンプラー」として自作し、`imblearn.pipeline.Pipeline`(4節)に組み込めるようにするための最も基底となるクラス。

**シグネチャ**: `imblearn.base.BaseSampler(sampling_strategy='auto')`

**使用例**:
```python
from sklearn.datasets import make_classification
from imblearn.base import BaseSampler

X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=0,
)

class Bad(BaseSampler):
    _sampling_type = "under-sampling"
    def _fit_resample(self, X, y):
        return X, y

try:
    Bad().fit_resample(X, y)
except AttributeError as e:
    print("AttributeError:", e)
```
実行結果:
```
AttributeError: 'Bad' object has no attribute '_parameter_constraints'
```

**注意点・落とし穴**:
- `BaseSampler`は`_sampling_type`クラス属性(`'under-sampling'`/`'over-sampling'`/`'clean-sampling'`)と`_fit_resample(self, X, y)`の実装を要求するが、それだけでは不十分で、`_parameter_constraints`(sklearnのパラメータ検証機構が参照する辞書)を自分で定義していないと`fit_resample`実行時に上記の`AttributeError`になる(検証済み)。
- 実務で自作サンプラーを書く場合は`BaseSampler`を直接継承するより、次項の`BaseUnderSampler`のように既に`_parameter_constraints`と`_sampling_type`が用意されているサブクラスを継承する方が簡単で安全。単純な関数をラップするだけなら7節の`FunctionSampler`の方が手軽。

### `imblearn.under_sampling.base.BaseUnderSampler`

**用途**: 独自のアンダーサンプリングアルゴリズムを実装する際の実用的な継承先。`_sampling_type = "under-sampling"`と、`sampling_strategy`の型チェック用`_parameter_constraints`があらかじめ定義されているため、`_fit_resample`を実装するだけで動くサンプラーが作れる。

**シグネチャ**: `imblearn.under_sampling.base.BaseUnderSampler(sampling_strategy='auto')`(継承先で`_fit_resample(self, X, y)`を実装する)

**使用例**:
```python
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling.base import BaseUnderSampler

X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=0,
)

class EveryOtherMajoritySampler(BaseUnderSampler):
    """多数派クラスのサンプルを1つおきに間引く自作アンダーサンプラー(デモ用)。"""
    def _fit_resample(self, X, y):
        classes = np.unique(y)
        majority_class = max(classes, key=lambda c: np.sum(y == c))
        keep_mask = np.ones(len(y), dtype=bool)
        majority_idx = np.flatnonzero(y == majority_class)
        keep_mask[majority_idx[1::2]] = False  # 多数派を1つおきに除去
        return X[keep_mask], y[keep_mask]

sampler = EveryOtherMajoritySampler()
Xr, yr = sampler.fit_resample(X, y)
print(Xr.shape, Counter(yr))
print(sampler.sampling_strategy_)
```
実行結果:
```
(550, 5) Counter({np.int64(0): 450, np.int64(1): 100})
OrderedDict({np.int64(0): np.int64(100)})
```

**注意点・落とし穴**:
- `_fit_resample`は検証・numpy配列化済みの`X`, `y`を受け取り、`(X_resampled, y_resampled)`を返すだけでよい。`fit_resample`実行後に自動でセットされる`self.sampling_strategy_`属性(クラスごとの「追加/削除すべき件数」の辞書、8-2節の`check_sampling_strategy`参照)を`_fit_resample`内で使えば、`sampling_strategy`引数を尊重したロジックも書ける。
- 同様にオーバーサンプリング系は`imblearn.over_sampling.base.BaseOverSampler`(`_sampling_type="over-sampling"`)、クリーニング系(`TomekLinks`等)は`imblearn.under_sampling.base.BaseCleaningSampler`(`_sampling_type="clean-sampling"`)を継承するのが定石。

**8-2. 多クラス不均衡・sampling_strategy の指定**

### `SMOTE(sampling_strategy={...})`(多クラスでのdict指定)

**用途**: 3クラス以上の多クラス不均衡データに対して、`sampling_strategy`にdictを渡し、クラスごとに目標件数(最終的な件数)を個別指定する。1節では2値分類のみを扱ったが、SMOTE系・アンダーサンプリング系のほとんどが多クラスにもそのまま対応している。

**シグネチャ**: `imblearn.over_sampling.SMOTE(*, sampling_strategy='auto', random_state=None, k_neighbors=5)`

**使用例**:
```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

X3, y3 = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, n_classes=3, weights=[0.7, 0.2, 0.1], flip_y=0, random_state=0,
)
print("before:", Counter(y3))

sm_dict = SMOTE(sampling_strategy={0: 700, 1: 300, 2: 300}, random_state=0)
_, yr = sm_dict.fit_resample(X3, y3)
print("after dict:", Counter(yr))

sm_auto = SMOTE(random_state=0)
_, ya = sm_auto.fit_resample(X3, y3)
print("after auto:", Counter(ya))
```
実行結果:
```
before: Counter({np.int64(0): 700, np.int64(1): 200, np.int64(2): 100})
after dict: Counter({np.int64(0): 700, np.int64(1): 300, np.int64(2): 300})
after auto: Counter({np.int64(0): 700, np.int64(1): 700, np.int64(2): 700})
```

**注意点・落とし穴**:
- `sampling_strategy`にdictを渡す場合、値は「そのクラスの最終的な目標件数」であり、元の件数以上でなければならない(オーバーサンプラーは間引けないため、元の件数を下回る値を指定するとエラーになる)。
- `sampling_strategy='auto'`は多クラスでも「最多クラス以外を全て最多クラスの件数に合わせる」("not majority"相当)という意味になる。今回、自動指定では3クラス全てが700件になったのに対し、dict指定では0(700件)はそのまま据え置きつつ1・2だけを300件に増やす、といった細かい制御ができる。

### `SMOTEN(...)`

**用途**: 全ての特徴量がカテゴリ変数であるデータ専用のSMOTE。数値変数とカテゴリ変数が混在する場合は`SMOTENC`(7節)、数値変数のみなら通常の`SMOTE`を使う。

**シグネチャ**: `imblearn.over_sampling.SMOTEN(categorical_encoder=None, *, sampling_strategy='auto', random_state=None, k_neighbors=5)`

**使用例**:
```python
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTEN

rng = np.random.RandomState(0)
n = 300
y = np.array([0] * 270 + [1] * 30)
X_cat = np.empty((n, 3), dtype=object)
for i in range(n):
    if y[i] == 0:
        X_cat[i] = rng.choice(["A", "B", "C"], size=3, p=[0.6, 0.3, 0.1])
    else:
        X_cat[i] = rng.choice(["A", "B", "C"], size=3, p=[0.1, 0.3, 0.6])

print("before:", Counter(y))
smn = SMOTEN(random_state=0)
Xr, yr = smn.fit_resample(X_cat, y)
print("after:", Counter(yr))
print("unique values col0:", np.unique(Xr[:, 0]))
```
実行結果:
```
before: Counter({np.int64(0): 270, np.int64(1): 30})
after: Counter({np.int64(0): 270, np.int64(1): 270})
unique values col0: ['A' 'B' 'C']
```

**注意点・落とし穴**:
- `SMOTENC`と異なり`categorical_features`引数は不要(全列がカテゴリという前提のため位置引数が無い)。
- 数値列の補間のような連続値は生成されず、合成後も各列の値は必ず元から存在するカテゴリ値のいずれかになる(検証でも生成後のユニーク値が`['A', 'B', 'C']`のまま変わらないことを確認)。

### `KMeansSMOTE(...)`

**用途**: 通常のSMOTEが全体の近傍情報だけを使うのに対し、事前にKMeansでクラスタリングしてから「少数派の密度が高い」クラスタ内で重点的にSMOTE的補間を行う派生手法。

**シグネチャ**: `imblearn.over_sampling.KMeansSMOTE(*, sampling_strategy='auto', random_state=None, k_neighbors=2, n_jobs=None, kmeans_estimator=None, cluster_balance_threshold='auto', density_exponent='auto')`

**使用例**:
```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import KMeansSMOTE

X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=0,
)
kms = KMeansSMOTE(random_state=0)
Xr, yr = kms.fit_resample(X, y)
print(Xr.shape, Counter(yr))
print(kms.kmeans_estimator_)
```
実行結果:
```
(1800, 5) Counter({np.int64(0): 900, np.int64(1): 900})
MiniBatchKMeans(random_state=0)
```

**注意点・落とし穴**:
- `k_neighbors`のデフォルトは`5`ではなく`2`(通常の`SMOTE`と異なる)。
- `kmeans_estimator=None`の場合、内部でクラスタリングに使われるのは`sklearn.cluster.KMeans`ではなく`sklearn.cluster.MiniBatchKMeans`である(`fit_resample`後の`kmeans_estimator_`属性で確認済み)。
- ソースコード(`imblearn/over_sampling/_smote/cluster.py`)を確認したところ、`cluster_balance_threshold`の条件を満たすクラスタが1つも見つからない場合は`RuntimeError("No clusters found with sufficient samples of ...")`になる実装になっている(今回検証した程度のデータ規模ではこのエラーは再現しなかったため、実際に発生する具体的な条件までは未検証)。

### `imblearn.utils.check_sampling_strategy(...)`

**用途**: `sampling_strategy`(`'auto'`・float・dict・callable)の指定を、実際に「クラスごとに何件追加/削除するか」を表す辞書に解決する内部ユーティリティ。自作サンプラー(8-1節)の`_fit_resample`内で`self.sampling_strategy_`が具体的に何を意味するかを確認するのに使える。

**シグネチャ**: `imblearn.utils.check_sampling_strategy(sampling_strategy, y, sampling_type, **kwargs)`

**使用例**:
```python
import numpy as np
from imblearn.utils import check_sampling_strategy

y3 = np.array([0] * 700 + [1] * 200 + [2] * 100)
print(check_sampling_strategy("auto", y3, "over-sampling"))
print(check_sampling_strategy({1: 300, 2: 300}, y3, "over-sampling"))

y_bin = np.array([0] * 900 + [1] * 100)
print(check_sampling_strategy(0.5, y_bin, "over-sampling"))
```
実行結果:
```
OrderedDict({np.int64(1): np.int64(500), np.int64(2): np.int64(600)})
OrderedDict({1: np.int64(100), 2: np.int64(200)})
OrderedDict({np.int64(1): np.int64(350)})
```

**注意点・落とし穴**:
- 戻り値は「最終的な目標件数」ではなく「追加で生成する件数」を表す辞書である。例えば`{1: 300, 2: 300}`(元は1が200件・2が100件)という最終件数の指定は、実際には「1に100件、2に200件を追加生成する」という意味の辞書に変換される。
- `dict`で指定した場合のキーはPythonの`int`のまま保持されるが、`'auto'`や`float`で指定した場合のキーは`y`のdtypeに合わせて`np.int64`になる、という型の違いがある(検証済み)。この関数はimblearn内部の全サンプラーの`fit_resample`から呼ばれている。

**8-3. `imblearn.metrics` の残りの指標**

### `macro_averaged_mean_absolute_error(...)`

**用途**: 順序のある多クラスラベル(例: 5段階評価などの序数)に対し、クラスごとにMAEを計算してからクラス数で単純平均する評価指標。クラスごとの出現頻度に依存しにくく、不均衡な順序ラベルの評価に向く。

**シグネチャ**: `imblearn.metrics.macro_averaged_mean_absolute_error(y_true, y_pred, *, sample_weight=None)`

**使用例**:
```python
import numpy as np
from sklearn.metrics import mean_absolute_error
from imblearn.metrics import macro_averaged_mean_absolute_error

y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5])
y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4])
print("macro MAE:", macro_averaged_mean_absolute_error(y_true, y_pred))
print("plain MAE:", mean_absolute_error(y_true, y_pred))
```
実行結果:
```
macro MAE: 0.34444444444444444
plain MAE: 0.2857142857142857
```

**注意点・落とし穴**:
- 通常の`sklearn.metrics.mean_absolute_error`(全サンプルにわたる単純平均)と異なり、先にクラスごとにMAEを計算してからクラス間で平均するため、サンプル数の少ないクラスの誤差も均等に反映される。今回のデータは件数の多いラベル1の予測がほぼ正確だったため、通常のMAE(0.286)よりmacro版(0.344)の方が大きい値になった(=通常のMAEだと少数ラベルの誤差が見えにくくなっていたことを意味する)。

### `sensitivity_score(...)`

**用途**: 再現率(recall)と同義のsensitivity(真陽性率)をスカラー値で返す、`sensitivity_specificity_support`(6節)の簡易版。

**シグネチャ**: `imblearn.metrics.sensitivity_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None)`

**使用例**:
```python
import numpy as np
from imblearn.metrics import sensitivity_score, specificity_score

yb_true = np.array([0] * 270 + [1] * 30)
rng = np.random.RandomState(0)
yb_pred = yb_true.copy()
flip_idx = rng.choice(len(yb_true), size=15, replace=False)
yb_pred[flip_idx] = 1 - yb_pred[flip_idx]

print("sensitivity:", sensitivity_score(yb_true, yb_pred, average="binary"))
```
実行結果:
```
sensitivity: 0.9666666666666667
```

**注意点・落とし穴**:
- 計算内容はsklearn本体の`recall_score`とほぼ同じで、`average`のデフォルトも`'binary'`で揃っている。`specificity_score`(次項)と対にして使うことを想定した命名になっている。

### `specificity_score(...)`

**用途**: 特異度(真陰性率、陰性クラスをどれだけ正しく陰性と判定できたか)をスカラー値で返す。scikit-learn本体には直接対応する関数がない。

**シグネチャ**: `imblearn.metrics.specificity_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None)`

**使用例**:
```python
print("specificity:", specificity_score(yb_true, yb_pred, average="binary"))
```
実行結果:
```
specificity: 0.9481481481481482
```

**注意点・落とし穴**:
- scikit-learn本体には特異度を直接計算する関数が無く、通常は`confusion_matrix`から自前で計算するか、この関数を使う必要がある。`sensitivity_score`と`specificity_score`を組み合わせれば`sensitivity_specificity_support`と同じ値をスカラー関数として個別に取得できる。

### `imblearn.metrics.pairwise.ValueDifferenceMetric`

**用途**: カテゴリ変数のみからなるデータに対して、特徴値ごとのクラスラベルとの共起確率の差に基づく距離(Value Difference Metric)を計算する。`NearMiss`や`EditedNearestNeighbours`など、通常はユークリッド距離を前提とするk近傍ベースの手法をカテゴリデータに適用する際の距離行列として使える。

**シグネチャ**: `imblearn.metrics.pairwise.ValueDifferenceMetric(*, n_categories='auto', k=1, r=2)`

**使用例**:
```python
import numpy as np
from imblearn.metrics.pairwise import ValueDifferenceMetric

rng = np.random.RandomState(0)
n = 200
y = np.array([0] * 180 + [1] * 20)
X_cat = np.empty((n, 2), dtype=int)
for i in range(n):
    if y[i] == 0:
        X_cat[i] = rng.choice([0, 1, 2], size=2, p=[0.6, 0.3, 0.1])
    else:
        X_cat[i] = rng.choice([0, 1, 2], size=2, p=[0.1, 0.3, 0.6])

vdm = ValueDifferenceMetric(n_categories="auto")
vdm.fit(X_cat, y)
dist = vdm.pairwise(X_cat[:4])
print(dist.shape)
print(np.round(dist, 3))
```
実行結果:
```
(4, 4)
[[0.    0.063 0.    0.   ]
 [0.063 0.    0.063 0.063]
 [0.    0.063 0.    0.   ]
 [0.    0.063 0.    0.   ]]
```

**注意点・落とし穴**:
- `n_categories`にint(例: `3`)を直接渡すと`sklearn.utils._param_validation.InvalidParameterError`になる(検証済み)。`'auto'`(列ごとに自動検出)か、列ごとのカテゴリ数を並べた配列で指定する必要がある。
- `SMOTE`などのように`fit_resample`を1回呼ぶ形式ではなく、`fit(X, y)`でラベルとの共起確率を学習した後に`pairwise(X)`で距離行列を計算する、scikit-learnの距離学習系クラスに近い2段階のAPIになっている。

**8-4. コスト考慮型学習との組み合わせ**

### `sklearn.utils.class_weight.compute_class_weight(...)`

**用途**: imblearn本体の関数ではないが、リサンプリングと同じ「不均衡データへの対処」を、データを複製・合成せずに損失関数側の誤分類コストを調整することで行う「コスト考慮型学習」のための重み計算に使う。リサンプリングとどちらが有効かを比較する目的でよく併記される。

**シグネチャ**: `sklearn.utils.class_weight.compute_class_weight(class_weight, *, classes, y, sample_weight=None)`

**使用例**:
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=0,
)
print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

pred_plain = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xte)
pred_cw = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr).predict(Xte)
pred_smote = Pipeline([("smote", SMOTE(random_state=0)),
                        ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr).predict(Xte)
pred_both = Pipeline([("smote", SMOTE(random_state=0)),
                       ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(Xtr, ytr).predict(Xte)

for name, pred in [("plain", pred_plain), ("class_weight", pred_cw),
                    ("SMOTE", pred_smote), ("SMOTE+class_weight", pred_both)]:
    print(f"{name:20s} balanced_accuracy={balanced_accuracy_score(yte, pred):.4f}")
```
実行結果:
```
[0.55555556 5.        ]
plain                balanced_accuracy=0.9833
class_weight         balanced_accuracy=0.9981
SMOTE                balanced_accuracy=1.0000
SMOTE+class_weight   balanced_accuracy=1.0000
```

**注意点・落とし穴**:
- `class_weight='balanced'`は`n_samples / (n_classes * np.bincount(y))`という式で重みを計算し、今回のデータ(多数派900・少数派100)では多数派に約0.556、少数派に約5.0という重みを与える(データそのものは複製・合成せず、損失関数側で少数派の誤分類コストを重くする)。
- 今回の検証データでは`class_weight='balanced'`のみでもSMOTEにかなり近い改善が得られ、`SMOTE`と`class_weight='balanced'`を併用しても単独の`SMOTE`からさらなる改善は見られなかった。リサンプリングとコスト考慮型学習は原理的には代替手段であり、両方を機械的に重ねがけしても常に上乗せ効果があるとは限らない(次項も参照)。

### `BalancedRandomForestClassifier(class_weight=...)`(リサンプリングとの併用)

**用途**: `imblearn.ensemble.BalancedRandomForestClassifier`(5節)は各決定木のブートストラップサンプリング自体が既にクラスごとの件数を揃える(内部的なアンダーサンプリング)。ここにsklearn由来の`class_weight`(コスト考慮型)をさらに重ねがけした場合に、性能が上乗せされるのか確認する(直前の`compute_class_weight`の例と同じ`Xtr, Xte, ytr, yte`をそのまま使う)。

**シグネチャ**: `imblearn.ensemble.BalancedRandomForestClassifier(n_estimators=100, *, ..., sampling_strategy='all', replacement=True, ..., class_weight=None, ...)`(全シグネチャは5節参照)

**使用例**:
```python
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

brf_default = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
brf_default.fit(Xtr, ytr)
pred1 = brf_default.predict(Xte)

brf_cw = BalancedRandomForestClassifier(n_estimators=100, random_state=0, class_weight="balanced_subsample")
brf_cw.fit(Xtr, ytr)
pred2 = brf_cw.predict(Xte)

print("sampling only:          ", balanced_accuracy_score(yte, pred1))
print("sampling + class_weight:", balanced_accuracy_score(yte, pred2))
```
実行結果:
```
sampling only:           0.9259259259259259
sampling + class_weight: 0.8981481481481481
```

**注意点・落とし穴**:
- リサンプリング(内部的なアンダーサンプリング)と`class_weight='balanced_subsample'`(コスト考慮)を併用しても必ず性能が上がるわけではない。今回の検証データでは、むしろ`class_weight`を追加した方がbalanced accuracyが下がった(0.926→0.898)。既にクラス比が調整済みのブートストラップサンプルに対してさらに重み付けを行う「二重補正」が、逆に予測を偏らせた可能性がある。
- リサンプリングとコスト考慮型学習のどちらか一方、あるいは両方の組み合わせが最適かはデータに強く依存するため、機械的に両方オンにするのではなく、検証データ(交差検証)で実際に比較して選ぶべきである。

