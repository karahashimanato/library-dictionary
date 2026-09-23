# scikit-survival 逆引き辞書

scikit-survival 0.28.0 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(scikit-survival 0.28.0)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。scikit-survivalはscikit-learnの推定器API(`fit`/`predict`/`score`)に準拠しているため、基本的な使い方は[scikit-learn/README.md](../scikit-learn/README.md)を参照。import名は`sksurv`。

## 目次

1. [データ形式・データセット](#1-データ形式データセット)
2. [前処理](#2-前処理)
3. [Cox回帰系](#3-cox回帰系)
4. [生存木・アンサンブル](#4-生存木アンサンブル)
5. [サポートベクターマシン系](#5-サポートベクターマシン系)
6. [ノンパラメトリック推定・群間比較](#6-ノンパラメトリック推定群間比較)
7. [評価指標](#7-評価指標)
8. [予測(生存関数・累積ハザード・リスクスコア)](#8-予測生存関数累積ハザードリスクスコア)
9. [応用・発展](#9-応用発展)
    - [ハイパーパラメータチューニング](#ハイパーパラメータチューニング)
    - [競合リスク・時間依存共変量への対応状況](#競合リスク時間依存共変量への対応状況)
    - [パイプライン統合](#パイプライン統合)
    - [モデルの永続化(joblib)](#モデルの永続化joblib)
    - [発展的な生存SVM・メタ推定器](#発展的な生存svmメタ推定器)

---

## 1. データ形式・データセット

### `Surv.from_arrays(...)`

**用途**: イベント発生有無の配列と観測時間の配列から、scikit-survivalが要求する構造化配列(structured array)形式の目的変数を作る。

**シグネチャ**: `sksurv.util.Surv.from_arrays(event, time, name_event=None, name_time=None)`

**使用例**:
```python
from sksurv.util import Surv

event = [True, False, True, True, False]
time = [5.0, 8.0, 3.0, 12.0, 6.0]
y = Surv.from_arrays(event=event, time=time)
print(y)
print(y.dtype)
```
実行結果:
```
[( True,  5.) (False,  8.) ( True,  3.) ( True, 12.) (False,  6.)]
[('event', '?'), ('time', '<f8')]
```

**注意点・落とし穴**:
- scikit-survivalの推定器・評価関数は、目的変数`y`として**構造化配列**(`(event, time)`の2フィールドを持つnumpy structured array)を要求する。通常のscikit-learnのように`(n_samples,)`の1次元配列ではない点が最大の違い。
- `event`(1フィールド目)は「イベントが観測された(＝打ち切りでない)か」を表すbool。`True`=イベント発生(death/failure)、`False`=打ち切り(censored)。`Surv.from_arrays`はデフォルトでフィールド名を`event`/`time`にするが、`name_event`/`name_time`で任意の名前(例: `"death"`/`"os_time"`)に変更できる。
- フィールド名が違っていても、推定器側はフィールドの**位置**(1番目=event, 2番目=time)ではなく`dtype.names`の中身を見て動作するため、後述の`concordance_index_censored`などに配列を渡すときはフィールド名で明示的に取り出す必要がある(`y["event"]`のように)。

### `Surv.from_dataframe(...)`

**用途**: pandas DataFrame中の「イベント列」「時間列」を指定して構造化配列の目的変数を作る。

**シグネチャ**: `sksurv.util.Surv.from_dataframe(event, time, data)`

**使用例**:
```python
import pandas as pd
from sksurv.util import Surv

df = pd.DataFrame({"status": [True, False, True], "time": [5.0, 8.0, 3.0]})
y = Surv.from_dataframe("status", "time", df)
print(y)
print(y.dtype.names)
```
実行結果:
```
[( True, 5.) (False, 8.) ( True, 3.)]
('status', 'time')
```

**注意点・落とし穴**:
- `event`/`time`は列名(文字列)を渡す点が`from_arrays`と異なる。生成される構造化配列のフィールド名は、DataFrameの元の列名(この例では`status`/`time`)がそのまま使われる。

### `get_x_y(...)`

**用途**: イベント列・時間列を含むDataFrameから、説明変数`X`(それらの列を除いたDataFrame)と目的変数`y`(構造化配列)を一度に取り出す。

**シグネチャ**: `sksurv.datasets.get_x_y(data_frame, attr_labels, pos_label=None, survival=True, competing_risks=False)`

**使用例**:
```python
import pandas as pd
from sksurv.datasets import get_x_y

df = pd.DataFrame({"status": [True, False, True, True], "time": [5.0, 8.0, 3.0, 9.0], "age": [50, 60, 70, 40]})
X, y = get_x_y(df, attr_labels=["status", "time"], pos_label=True)
print(X.columns.tolist())
print(y.dtype.names)
print(y)
```
実行結果:
```
['age']
('status', 'time')
[( True, 5.) (False, 8.) ( True, 3.) ( True, 9.)]
```

**注意点・落とし穴**:
- `attr_labels`は`[イベント列名, 時間列名]`の順で渡す。`pos_label`は、イベント列がbool型でなく文字列/数値などの場合に「イベント発生」を表す値を明示するために使う(この例のようにbool列なら省略しても動くことが多いが、`load_gbsg2`のような実データでは必要になる場合がある)。

### `load_whas500(...)`

**用途**: Worcester Heart Attack Study(WHAS500)の心筋梗塞後生存データを読み込む練習用データセット。500例・14特徴量。

**シグネチャ**: `sksurv.datasets.load_whas500(*, output_type='pandas')`

**使用例**:
```python
from sksurv.datasets import load_whas500

X, y = load_whas500()
print(type(X), X.shape)
print(y.dtype.names)
print(y[:3])
print("event rate:", y[y.dtype.names[0]].mean())
```
実行結果:
```
<class 'pandas.DataFrame'> (500, 14)
('fstat', 'lenfol')
[(False, 2178.) (False, 2172.) (False, 2190.)]
event rate: 0.43
```

**注意点・落とし穴**:
- 目的変数のフィールド名はデータセットごとに異なる(WHAS500では`fstat`=死亡有無, `lenfol`=追跡日数)。`load_gbsg2`では`cens`/`time`、`load_veterans_lung_cancer`では`Status`/`Survival_in_days`になっており、統一されていない。使う前に必ず`y.dtype.names`で確認する。
- `X`にはカテゴリ変数(`gender`, `mitype`など)がそのまま文字列/カテゴリ型で含まれており、Cox回帰などの数値のみを扱う推定器にそのまま渡すとエラーになる。後述の`encode_categorical`などで前処理してから使う。

### `load_gbsg2(...)`

**用途**: German Breast Cancer Study Group 2(GBSG2)の乳がん患者データを読み込む(686例・8特徴量)。

**シグネチャ**: `sksurv.datasets.load_gbsg2(*, output_type='pandas')`

**使用例**:
```python
from sksurv.datasets import load_gbsg2

X, y = load_gbsg2()
print(X.shape, X.columns.tolist())
print(y.dtype.names)
```
実行結果:
```
(686, 8) ['age', 'estrec', 'horTh', 'menostat', 'pnodes', 'progrec', 'tgrade', 'tsize']
('cens', 'time')
```

### `load_veterans_lung_cancer(...)`

**用途**: Veterans' Administration Lung Cancerの肺がん患者データを読み込む(137例・6特徴量)。

**シグネチャ**: `sksurv.datasets.load_veterans_lung_cancer(*, output_type='pandas')`

**使用例**:
```python
from sksurv.datasets import load_veterans_lung_cancer

X, y = load_veterans_lung_cancer()
print(X.shape, X.columns.tolist())
print(y.dtype.names)
```
実行結果:
```
(137, 6) ['Age_in_years', 'Celltype', 'Karnofsky_score', 'Months_from_Diagnosis', 'Prior_therapy', 'Treatment']
('Status', 'Survival_in_days')
```

---

## 2. 前処理

### `encode_categorical(...)`

**用途**: DataFrame中のカテゴリ変数をダミー変数(one-hot)にエンコードする。Cox回帰など数値のみを扱う推定器に渡す前の定番の前処理。

**シグネチャ**: `sksurv.column.encode_categorical(table, columns=None, **kwargs)`

**使用例**:
```python
import pandas as pd
from sksurv.column import encode_categorical

df = pd.DataFrame({
    "age": [50.0, 60.0, 70.0],
    "sex": pd.Categorical(["m", "f", "m"]),
    "grade": pd.Categorical(["low", "high", "low"]),
})
enc = encode_categorical(df)
print(enc)
```
実行結果:
```
    age  sex=m  grade=low
0  50.0    1.0        1.0
1  60.0    0.0        0.0
2  70.0    1.0        1.0
```

**注意点・落とし穴**:
- `sklearn.preprocessing.OneHotEncoder`と違い、各カテゴリ変数につき**カテゴリ数-1本**の列しか作らない(1つのカテゴリを基準として落とす、いわゆるdrop-first)。2値カテゴリの`sex`なら`sex=m`の1列だけになる。多重共線性を避けるための挙動であり、「全カテゴリの列が欲しい」場合は使えない。
- 対象は`pandas`の`category`dtype(または文字列)の列のみ。数値列はそのまま素通りする。

### `categorical_to_numeric(...)`

**用途**: カテゴリ変数を(one-hotではなく)0始まりの整数コードに変換する。

**シグネチャ**: `sksurv.column.categorical_to_numeric(table)`

**使用例**:
```python
print(categorical_to_numeric(df))
```
実行結果:
```
    age  sex  grade
0  50.0    1      1
1  60.0    0      0
2  70.0    1      1
```

**注意点・落とし穴**:
- `encode_categorical`(one-hot)と違い列数は変わらない(名義尺度を単に整数化するだけ)。木・アンサンブル系のように順序に強く依存しないモデルなら使えるが、Cox回帰など線形モデルにそのまま使うと、コード間に存在しないはずの大小関係を学習してしまう恐れがある(scikit-learnの`OrdinalEncoder`と同種の注意点)。

### `standardize(...)`

**用途**: 数値列を平均0・標準偏差1に標準化する。

**シグネチャ**: `sksurv.column.standardize(table, with_std=True)`

**使用例**:
```python
std = standardize(df[["age"]].copy())
print(std)
```
実行結果:
```
   age
0 -1.0
1  0.0
2  1.0
```

**注意点・落とし穴**:
- `sklearn.preprocessing.StandardScaler`と違い`fit`/`transform`を分けない関数型APIなので、学習データとテストデータに別々に適用すると基準となる平均・標準偏差がデータごとにずれる(データリークとは逆に、学習/推論で前処理基準が不整合になる)。テストデータには学習データの平均・標準偏差を手動で適用するか、`StandardScaler`を使う方が安全。

### `OneHotEncoder(...)`

**用途**: `encode_categorical`と同じ変換(カテゴリ数-1本のダミー変数化)を、scikit-learn互換の`fit`/`transform`を持つTransformerとして提供する(`Pipeline`に組み込みやすい)。

**シグネチャ**: `sksurv.preprocessing.OneHotEncoder(*, allow_drop=True)`

**使用例**:
```python
from sksurv.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
out = ohe.fit_transform(df)
print(out)
```
実行結果:
```
    age  sex=m  grade=low
0  50.0    1.0        1.0
1  60.0    0.0        0.0
2  70.0    1.0        1.0
```

**注意点・落とし穴**:
- `sklearn.preprocessing.OneHotEncoder`(全カテゴリ分の列を作り疎行列を返す)とは名前が同じでも挙動が異なるので混同注意。scikit-survivalのこちらは常に密なDataFrameを返し、デフォルトでカテゴリを1本落とす(`allow_drop=True`)。

---

## 3. Cox回帰系

### `CoxPHSurvivalAnalysis(...)`

**用途**: Cox比例ハザードモデル(セミパラメトリックな生存回帰の標準手法)。

**シグネチャ**: `sksurv.linear_model.CoxPHSurvivalAnalysis(alpha=0, *, ties='breslow', n_iter=100, tol=1e-09, verbose=0)`

**使用例**:
```python
from sksurv.datasets import load_whas500
from sksurv.column import encode_categorical
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis

X, y = load_whas500()
Xt = encode_categorical(X)
Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.25, random_state=0)

cph = CoxPHSurvivalAnalysis()
cph.fit(Xtr, ytr)
print("concordance index (score):", cph.score(Xte, yte))
print("coef_[:5]:", cph.coef_[:5].round(4))
print("predict (risk score)[:5]:", cph.predict(Xte[:5]).round(3))
```
実行結果:
```
concordance index (score): 0.8070936463383516
coef_[:5]: [ 0.1764  0.0436  1.3515 -0.0362  0.8066]
predict (risk score)[:5]: [2.464 1.823 1.355 2.944 0.306]
```

**注意点・落とし穴**:
- `.score(X, y)`は分類・回帰の指標(正解率・R2)ではなく**一致指数(concordance index、C-index)**を返す。1.0が完全一致、0.5がランダム予測相当。
- `.predict()`が返すのは生存時間そのものではなく**相対的なリスクスコア**(線形予測子)。値が大きいほどハザードが高い(イベントが早く起きやすい)ことを意味するが、絶対的な時間の予測ではない。
- `alpha=0`(デフォルト)は正則化なしのCox回帰。多重共線性がある場合や特徴量数がサンプル数に対して多い場合は`alpha`を大きくするか、正則化付きの`CoxnetSurvivalAnalysis`を使う。

### `CoxnetSurvivalAnalysis(...)`

**用途**: Elastic Net正則化(L1+L2)付きのCox回帰。特徴量選択を兼ねた正則化パスを一括で計算する。

**シグネチャ**: `sksurv.linear_model.CoxnetSurvivalAnalysis(*, n_alphas=100, alphas=None, alpha_min_ratio='auto', l1_ratio=0.5, penalty_factor=None, normalize=False, copy_X=True, tol=1e-07, max_iter=100000, verbose=False, fit_baseline_model=False)`

**使用例**:
```python
from sksurv.linear_model import CoxnetSurvivalAnalysis

coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, fit_baseline_model=True)
coxnet.fit(Xtr, ytr)
print("alphas_[:5]:", coxnet.alphas_[:5].round(5))
print("n alphas:", len(coxnet.alphas_))
print("coef_.shape:", coxnet.coef_.shape)
print("score at default alpha:", coxnet.score(Xte, yte))
```
実行結果:
```
alphas_[:5]: [5.20993 4.97313 4.74709 4.53133 4.32537]
n alphas: 100
coef_.shape: (14, 100)
score at default alpha: 0.80199030364889
```

**注意点・落とし穴**:
- `fit`は`alphas`(または`n_alphas`個の自動生成された正則化強度)ごとの係数パスを**一括で**推定するため、`coef_`は`(n_features, n_alphas)`の2次元配列になる。単一の`alpha`だけを見たい場合は`predict(X, alpha=...)`で指定する(未指定なら最後の`alphas_[-1]`、つまり最も正則化が弱いモデルが使われる)。
- `fit_baseline_model=False`(デフォルト)だとベースラインハザードを推定しないため`predict_survival_function`が使えない。生存関数まで必要な場合は`fit_baseline_model=True`を明示する。
- `l1_ratio=0.5`がデフォルト(Ridge寄りとLassoの中間)。`l1_ratio=1.0`でLasso相当、`l1_ratio`を0に近づけるほどRidge相当になる点はscikit-learnの`ElasticNet`と同じ。

### `IPCRidge(...)`

**用途**: Inverse Probability of Censoring weighting(IPCW)による重み付けで、打ち切りを考慮したRidge回帰により対数生存時間そのものを予測する。

**シグネチャ**: `sksurv.linear_model.IPCRidge(alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.001, solver='auto', positive=False, random_state=None)`

**使用例**:
```python
from sksurv.linear_model import IPCRidge

ipc = IPCRidge(alpha=1.0)
ipc.fit(Xtr, ytr)
print("predict (survival time)[:5]:", ipc.predict(Xte[:5]).round(3))
```
実行結果:
```
predict (survival time)[:5]: [ 272.914  318.063 1435.769  544.414 3192.183]
```

**注意点・落とし穴**:
- 他のscikit-survival推定器と違い、`.predict()`が返すのは相対リスクスコアではなく**生存時間そのもの**(の予測値)。予測値の意味がモデルによって異なる点に注意(前述のCox系は「リスクスコア」、こちらは「時間」)。
- 打ち切りが多いデータではIPCWの重みが不安定になりやすく、極端に大きい予測値(この例の3192日のような外れ値的な値)が出ることがある。

---

## 4. 生存木・アンサンブル

### `SurvivalTree(...)`

**用途**: log-rank統計量に基づく分割規則で生存時間データに特化した決定木を構築する。

**シグネチャ**: `sksurv.tree.SurvivalTree(*, splitter='best', max_depth=None, min_samples_split=6, min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, low_memory=False)`

**使用例**:
```python
from sksurv.tree import SurvivalTree

st = SurvivalTree(max_depth=4, random_state=0)
st.fit(Xtr, ytr)
print("score:", st.score(Xte, yte))
```
実行結果:
```
score: 0.739601939270222
```

**注意点・落とし穴**:
- `sklearn.tree.DecisionTreeClassifier/Regressor`と違い`min_samples_split=6`, `min_samples_leaf=3`がデフォルト(1ではない)。各葉でKaplan-Meier推定を行うため、極端に小さい葉ではノンパラメトリック推定が不安定になるための配慮。

### `RandomSurvivalForest(...)`

**用途**: `SurvivalTree`をバギングで多数組み合わせたアンサンブル(Random Survival Forest)。

**シグネチャ**: `sksurv.ensemble.RandomSurvivalForest(n_estimators=100, *, max_depth=None, min_samples_split=6, min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, max_samples=None, low_memory=False)`

**使用例**:
```python
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(n_estimators=100, random_state=0, n_jobs=-1)
rsf.fit(Xtr, ytr)
print("score:", rsf.score(Xte, yte))
```
実行結果:
```
score: 0.7833631028323552
```

**注意点・落とし穴**:
- `sklearn.ensemble.RandomForestClassifier/Regressor`と違い**`feature_importances_`属性を持たない**。実際に`rsf.feature_importances_`にアクセスすると`NotImplementedError`が送出される(実行確認済み)。特徴量重要度が必要な場合は`sklearn.inspection.permutation_importance`を使う。

### `ExtraSurvivalTrees(...)`

**用途**: `RandomSurvivalForest`のExtra-Trees版(分割点をランダム化してさらに分散を抑える)。

**シグネチャ**: `sksurv.ensemble.ExtraSurvivalTrees(n_estimators=100, *, max_depth=None, min_samples_split=6, min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, max_samples=None, low_memory=False)`

**使用例**:
```python
from sksurv.ensemble import ExtraSurvivalTrees

est = ExtraSurvivalTrees(n_estimators=100, random_state=0, n_jobs=-1)
est.fit(Xtr, ytr)
print("score:", est.score(Xte, yte))
```
実行結果:
```
score: 0.7744322531257974
```

### `GradientBoostingSurvivalAnalysis(...)`

**用途**: 勾配ブースティングで浅い木を逐次追加していく生存回帰モデル。

**シグネチャ**: `sksurv.ensemble.GradientBoostingSurvivalAnalysis(*, loss='coxph', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, random_state=None, max_features=None, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, dropout_rate=0.0, verbose=0, ccp_alpha=0.0)`

**使用例**:
```python
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

gbsa = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1, random_state=0)
gbsa.fit(Xtr, ytr)
print("score:", gbsa.score(Xte, yte))
print("feature_importances_[:5]:", gbsa.feature_importances_[:5].round(4))
```
実行結果:
```
score: 0.77507017096198
feature_importances_[:5]: [0.0072 0.3853 0.0006 0.1055 0.1234]
```

**注意点・落とし穴**:
- 木ベースの`RandomSurvivalForest`と違い、こちらは`feature_importances_`を実装している(実行確認済み)。
- デフォルトの`loss='coxph'`はCox部分尤度に基づく損失。`ipcwls`(IPCW付き最小二乗、対数生存時間を予測)も選択できるが、損失によって`predict`の出力の意味(リスクスコアか時間か)が変わる点に注意。

### `ComponentwiseGradientBoostingSurvivalAnalysis(...)`

**用途**: 各ブースティングステップで1つの特徴量(コンポーネント)だけを選んで更新する勾配ブースティング。結果的にスパースな線形モデルに近い解釈性を持つ。

**シグネチャ**: `sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis(*, loss='coxph', learning_rate=0.1, n_estimators=100, subsample=1.0, warm_start=False, dropout_rate=0, random_state=None, verbose=0)`

**使用例**:
```python
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

cgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, random_state=0)
cgb.fit(Xtr, ytr)
print("score:", cgb.score(Xte, yte))
print("coef_[:5]:", cgb.coef_[:5].round(4))
```
実行結果:
```
score: 0.7524878795611125
coef_[:5]: [0.     0.0461 0.0007 0.3857 0.    ]
```

**注意点・落とし穴**:
- 木ベースの`GradientBoostingSurvivalAnalysis`と違い`coef_`(線形モデルのような係数)を持つ。一度も選ばれなかった特徴量の係数は0のままになる(この例の1列目・3列目・5列目など)ため、暗黙的な特徴量選択の効果がある。

---

## 5. サポートベクターマシン系

### `FastSurvivalSVM(...)`

**用途**: 生存時間データのランキング(順序)学習に基づくSVM。Cox回帰と違い比例ハザードを仮定しない。

**シグネチャ**: `sksurv.svm.FastSurvivalSVM(alpha=1, *, rank_ratio=1.0, fit_intercept=False, max_iter=20, verbose=False, tol=None, optimizer=None, random_state=None, timeit=False)`

**使用例**:
```python
from sksurv.svm import FastSurvivalSVM

svm = FastSurvivalSVM(alpha=1, random_state=0, max_iter=1000)
svm.fit(Xtr, ytr)
print("score:", svm.score(Xte, yte))
```
実行結果:
```
score: 0.7994386323041592
```

**注意点・落とし穴**:
- デフォルトの`max_iter=20`は少なく、収束前に打ち切られることがある(この検証でも`max_iter=1000`に増やしている)。収束状況が不安な場合は`optimizer`や`tol`を調整する。
- `rank_ratio=1.0`(デフォルト)は完全にランキング学習(一致指数最大化寄り)。`rank_ratio`を1未満にすると回帰的な項(観測時間の回帰)も損失に混ざる。

### `FastKernelSurvivalSVM(...)`

**用途**: `FastSurvivalSVM`のカーネル版。非線形な関係を捉えられる。

**シグネチャ**: `sksurv.svm.FastKernelSurvivalSVM(alpha=1, *, rank_ratio=1.0, fit_intercept=False, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, max_iter=20, verbose=False, tol=None, optimizer=None, random_state=None, timeit=False)`

**使用例**:
```python
from sksurv.svm import FastKernelSurvivalSVM

ksvm = FastKernelSurvivalSVM(alpha=1, kernel="rbf", random_state=0, max_iter=1000)
ksvm.fit(Xtr, ytr)
print("score:", ksvm.score(Xte, yte))
```
実行結果:
```
score: 0.6565450369992345
```

**注意点・落とし穴**:
- カーネルSVMは`SVC`/`SVR`同様、特徴量のスケールに敏感。この検証例ではスケーリングをしていないため、線形版の`FastSurvivalSVM`より性能が低く出ている(スケーリングすれば改善する可能性が高い)。
- 計算量がサンプル数に対して重く(カーネル行列がサンプル数の2乗のサイズ)、大規模データには不向き。

---

## 6. ノンパラメトリック推定・群間比較

### `kaplan_meier_estimator(...)`

**用途**: Kaplan-Meier法により、打ち切りを考慮した生存関数(生存確率の推定値)を計算する。

**シグネチャ**: `sksurv.nonparametric.kaplan_meier_estimator(event, time_exit, time_enter=None, time_min=None, reverse=False, conf_level=0.95, conf_type=None)`

**使用例**:
```python
from sksurv.datasets import load_whas500
from sksurv.nonparametric import kaplan_meier_estimator

X, y = load_whas500()
event, time = y["fstat"], y["lenfol"]

time_km, surv_prob = kaplan_meier_estimator(event, time)
print("time_km[:5]:", time_km[:5])
print("surv_prob[:5]:", surv_prob[:5].round(4))

time_km2, surv_prob2, conf_int = kaplan_meier_estimator(event, time, conf_type="log-log")
print("conf_int shape:", conf_int.shape)
print("conf_int[:, :3]:", conf_int[:, :3].round(4))
```
実行結果:
```
time_km[:5]: [1. 2. 3. 4. 5.]
surv_prob[:5]: [0.984 0.968 0.962 0.958 0.954]
conf_int shape: (2, 395)
conf_int[:, :3]: [[0.9683 0.9483 0.9411]
 [0.992  0.9803 0.9756]]
```

**注意点・落とし穴**:
- `event`/`time`は構造化配列の`y`そのものではなく、`y["event列名"]`/`y["time列名"]`のように**各フィールドを別々に**渡す(第1引数がイベント有無、第2引数が時間)。
- `conf_type=None`(デフォルト)だと戻り値は`(time, prob)`の2つだが、`conf_type="log-log"`を指定すると信頼区間`conf_int`(shape `(2, n_times)`、1行目が下限・2行目が上限)が3つ目の戻り値として追加される。戻り値の個数が引数によって変わる点に注意。

### `nelson_aalen_estimator(...)`

**用途**: Nelson-Aalen法により、打ち切りを考慮した累積ハザード関数を計算する。

**シグネチャ**: `sksurv.nonparametric.nelson_aalen_estimator(event, time)`

**使用例**:
```python
from sksurv.nonparametric import nelson_aalen_estimator

time_na, cum_hazard = nelson_aalen_estimator(event, time)
print("time_na[:5]:", time_na[:5])
print("cum_hazard[:5]:", cum_hazard[:5].round(4))
```
実行結果:
```
time_na[:5]: [1. 2. 3. 4. 5.]
cum_hazard[:5]: [0.016  0.0323 0.0385 0.0426 0.0468]
```

**注意点・落とし穴**:
- `kaplan_meier_estimator`と異なり信頼区間オプションを持たない(常に`(time, cum_hazard)`の2つのみを返す)。

### `compare_survival(...)`

**用途**: 2群以上の生存曲線の差をlog-rank検定で比較する。

**シグネチャ**: `sksurv.compare.compare_survival(y, group_indicator, return_stats=False)`

**使用例**:
```python
from sksurv.compare import compare_survival

group = (X["age"] >= 70).to_numpy().astype(int)
chisq, pval = compare_survival(y, group)
print("chisq:", round(chisq, 4), "pvalue:", pval)

stats = compare_survival(y, group, return_stats=True)
print(stats[2])
```
実行結果:
```
chisq: 105.4475 pvalue: 9.744498397062832e-25
       counts  observed    expected  statistic
group                                         
0         224        39  113.296509 -74.296509
1         276       176  101.703491  74.296509
```

**注意点・落とし穴**:
- `y`は構造化配列(`Surv.from_arrays`などで作ったもの)をそのまま渡す。`group_indicator`は群を表す整数/カテゴリのラベル配列。
- `return_stats=False`(デフォルト)だと`(統計量, p値)`の2つ、`return_stats=True`にすると`(統計量, p値, 群ごとの集計DataFrame, 共分散行列)`の4要素タプルが返る(戻り値の個数が変わる点はサンプル数の少なさに注意して使う)。

---

## 7. 評価指標

以下は`CoxPHSurvivalAnalysis`をWHAS500データの75%で学習し、残り25%で評価した例(`random_state=0`)。

### `concordance_index_censored(...)`

**用途**: 打ち切りを考慮した一致指数(Harrellのconcordance index、C-index)を計算する。生存モデルで最も一般的な評価指標。

**シグネチャ**: `sksurv.metrics.concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-08)`

**使用例**:
```python
from sksurv.metrics import concordance_index_censored

risk = cph.predict(Xte)
cidx = concordance_index_censored(yte["fstat"], yte["lenfol"], risk)
print(cidx)
```
実行結果:
```
(np.float64(0.8070936463383516), np.int64(3163), np.int64(756), np.int64(0), np.int64(0))
```

**注意点・落とし穴**:
- 戻り値は`(c_index, concordant, discordant, tied_risk, tied_time)`の**5要素タプル**(C-index本体だけでなく、比較可能だったペアの内訳も返す)。C-indexだけ使う場合は`cidx[0]`のように取り出す。
- `estimate`(3引数目)は生存時間の予測値ではなく**リスクスコア**(値が大きいほどイベントが早い、と解釈される)を渡す。`CoxPHSurvivalAnalysis.predict()`の出力はこの向きに合っているが、`IPCRidge.predict()`(生存時間そのものを返す)をそのまま渡すと符号の解釈が逆転するので注意。

### `concordance_index_ipcw(...)`

**用途**: IPCW(逆確率重み付け)による一致指数。`concordance_index_censored`と違い、打ち切り分布の偏りによるバイアスを補正する。

**シグネチャ**: `sksurv.metrics.concordance_index_ipcw(survival_train, survival_test, estimate, tau=None, tied_tol=1e-08)`

**使用例**:
```python
from sksurv.metrics import concordance_index_ipcw

cidx_ipcw = concordance_index_ipcw(ytr, yte, risk)
print(cidx_ipcw)
```
実行結果:
```
(np.float64(0.7814220122407373), np.int64(3163), np.int64(756), np.int64(0), np.int64(0))
```

**注意点・落とし穴**:
- 打ち切り分布を推定するために**学習データの`y`(`survival_train`)も必要**(`concordance_index_censored`は不要)。今回の例のようにテストデータだけで打ち切り分布を推定すると標本数が少なく不安定になりやすいため、学習データ全体を使う設計になっている。
- `concordance_index_censored`と値が異なりうる(この例では0.807 vs 0.781)。打ち切りが多いデータや打ち切り分布に偏りがある場合は、単純な`concordance_index_censored`より`concordance_index_ipcw`の方が偏りの少ない評価とされる。

### `brier_score(...)`

**用途**: 指定した複数の時点それぞれで、予測生存確率と実際の生存状態のブライアスコア(二乗誤差、打ち切りをIPCWで補正)を計算する。

**シグネチャ**: `sksurv.metrics.brier_score(survival_train, survival_test, estimate, times)`

**使用例**:
```python
import numpy as np
from sksurv.metrics import brier_score

times = np.percentile(yte["lenfol"], np.linspace(5, 81, 15))
surv_funcs = cph.predict_survival_function(Xte)
preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])

t_bs, bs = brier_score(ytr, yte, preds, times)
print("t[:3]:", t_bs[:3].round(1))
print("bs[:3]:", bs[:3].round(4))
```
実行結果:
```
t[:3]: [ 11.6  48.  138. ]
bs[:3]: [0.0528 0.08   0.1079]
```

**注意点・落とし穴**:
- `estimate`は「各時点`times`における生存確率の予測値」の2次元配列(`shape=(n_samples, n_times)`)であり、C-index系のような1次元のリスクスコアではない。生存関数(`predict_survival_function`が返す`StepFunction`)を`times`の各時点で評価して作る必要がある。
- `times`はテストデータの最大追跡時間**未満**である必要がある。実際に範囲外の時点を渡すと`ValueError: all times must be within follow-up time of test data: [1.0; 2350.0[`のように明示的にエラーになる(実行確認済み)。テストデータの追跡時間の最大値ぎりぎりの時点は避け、パーセンタイルなどで余裕を持たせて`times`を決めるとよい。

### `integrated_brier_score(...)`

**用途**: 複数時点のブライアスコアを時間軸で積分した単一のスカラー指標(IBS)。時点を1つに絞らずモデル全体の予測精度を要約する。

**シグネチャ**: `sksurv.metrics.integrated_brier_score(survival_train, survival_test, estimate, times)`

**使用例**:
```python
from sksurv.metrics import integrated_brier_score

ibs = integrated_brier_score(ytr, yte, preds, times)
print(ibs)
```
実行結果:
```
0.1371433876657594
```

**注意点・落とし穴**:
- 値は小さいほど良い(0が完全予測)。C-indexなど「大きいほど良い」指標と評価の向きが逆になるので、複数指標を並べて表にする際は向きを揃えて解釈する。

### `cumulative_dynamic_auc(...)`

**用途**: 各時点でのROC-AUC(その時点までにイベントが起きるかどうかを予測する二値分類問題とみなしたAUC)を計算する時間依存AUC。

**シグネチャ**: `sksurv.metrics.cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tied_tol=1e-08)`

**使用例**:
```python
from sksurv.metrics import cumulative_dynamic_auc

auc, mean_auc = cumulative_dynamic_auc(ytr, yte, risk, times)
print("auc[:3]:", auc[:3].round(4))
print("mean_auc:", mean_auc)
```
実行結果:
```
auc[:3]: [0.8426 0.831  0.8395]
mean_auc: 0.8411107597305547
```

**注意点・落とし穴**:
- `brier_score`と違い`estimate`は生存確率の2次元配列ではなく、`concordance_index_*`と同じ**1次元のリスクスコア**(時点によらない単一の予測値)を渡す。関数によって`estimate`の形が異なる点に注意。
- 戻り値は`(各時点のAUC配列, 時間軸で重み付けした平均AUC)`のタプル。

---

## 8. 予測(生存関数・累積ハザード・リスクスコア)

### `predict_survival_function(...)`

**用途**: 学習済みモデルから、各サンプルの生存関数(時間 → 生存確率)を推定する。多くの推定器(`CoxPHSurvivalAnalysis`、`RandomSurvivalForest`など)が共通して持つメソッド。

**シグネチャ**: `estimator.predict_survival_function(X, return_array=False)`(例: `RandomSurvivalForest.predict_survival_function`)

**使用例**:
```python
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(n_estimators=100, random_state=0, n_jobs=-1)
rsf.fit(Xtr, ytr)

surv_funcs = rsf.predict_survival_function(Xte[:3])
sf0 = surv_funcs[0]
print(type(sf0))
print("sf0.x[:5]:", sf0.x[:5])
print("sf0.y[:5]:", sf0.y[:5].round(4))
print("sf0(100):", sf0(100))
```
実行結果:
```
<class 'sksurv.functions.StepFunction'>
sf0.x[:5]: [1. 2. 3. 4. 5.]
sf0.y[:5]: [0.9531 0.9006 0.9006 0.9006 0.8981]
sf0(100): 0.8144487046769652
```

**注意点・落とし穴**:
- デフォルト(`return_array=False`)では、サンプルごとに`sksurv.functions.StepFunction`オブジェクト(階段関数)が入った`numpy.ndarray`(`dtype=object`)を返す。各`StepFunction`は`.x`(時点)/`.y`(その時点での生存確率)を持ち、`sf(t)`のように呼び出すと任意の時刻`t`での生存確率を補間して返す。
- `return_array=True`にすると、代わりに`(n_samples, n_event_times_)`の2次元配列を直接返す(全サンプル共通の時間グリッド上での値)。`brier_score`などスカラー評価関数に渡す前処理としてはこちらの方が扱いやすい場合が多い。
- 学習時に観測された時点の範囲外(最大観測時間より先など)は外挿されず、範囲内でしか意味のある値にならない。

### `predict_cumulative_hazard_function(...)`

**用途**: 各サンプルの累積ハザード関数(時間 → 累積ハザード)を推定する。`predict_survival_function`のハザード版。

**シグネチャ**: `estimator.predict_cumulative_hazard_function(X, return_array=False)`(例: `RandomSurvivalForest.predict_cumulative_hazard_function`)

**使用例**:
```python
chf_funcs = rsf.predict_cumulative_hazard_function(Xte[:3])
chf0 = chf_funcs[0]
print("chf0.y[:5]:", chf0.y[:5].round(4))
```
実行結果:
```
chf0.y[:5]: [0.0469 0.1117 0.1117 0.1117 0.1142]
```

**注意点・落とし穴**:
- `predict_survival_function`同様`StepFunction`(または`return_array=True`で2次元配列)を返す。生存確率`S(t)`と累積ハザード`H(t)`はおおよそ`S(t) ≈ exp(-H(t))`の関係にある(Nelson-Aalen推定に基づく近似)。

### `predict(...)`(リスクスコア)

**用途**: 多くのscikit-survival推定器が共通して持つ、学習済みモデルによる**相対リスクスコア**の予測。値が大きいほどイベント(死亡・故障など)が早く起きやすいと解釈する。

**シグネチャ**: `estimator.predict(X)`(例: `CoxPHSurvivalAnalysis.predict`, `RandomSurvivalForest.predict`)

**使用例**:
```python
risk = rsf.predict(Xte[:5])
print("predict (risk score)[:5]:", risk.round(3))
```
実行結果:
```
predict (risk score)[:5]: [70.828 40.199 28.95  90.295  1.228]
```

**注意点・落とし穴**:
- モデルによって`predict()`が返す値のスケール・意味が異なる。`CoxPHSurvivalAnalysis`/木・アンサンブル系は「相対的なリスクスコア」(モデル間で値のスケールを比較できない)を返すが、`IPCRidge`だけは例外的に「生存時間そのもの」を返す(3章参照)。`concordance_index_*`などの評価関数に渡す際は、対象のモデルがどちらのタイプかを`predict()`のドキュメントで確認する必要がある。

---

## 9. 応用・発展

以下は、`load_whas500`をカテゴリ変数エンコード後に`train_test_split(test_size=0.25, random_state=0)`で分割した`Xtr, Xte, ytr, yte`(3章のCox回帰の例と同じ分割)を主に使う。

### ハイパーパラメータチューニング

#### `as_concordance_index_ipcw_scorer(...)`

**用途**: 打ち切りを考慮したIPCW一致指数(`concordance_index_ipcw`)を、scikit-learnの`GridSearchCV`などがそのまま使える「メタ推定器兼スコアラー」として提供する。推定器をラップし、`.fit`/`.score`をIPCW一致指数ベースに差し替える。

**シグネチャ**: `sksurv.metrics.as_concordance_index_ipcw_scorer(estimator, tau=None, tied_tol=1e-08)`

**使用例**:
```python
from sklearn.model_selection import GridSearchCV
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import as_concordance_index_ipcw_scorer

tau = ytr["lenfol"].max() * 0.9  # 打ち切り分布のIPCWが不安定になる最大時刻付近を避ける
est = as_concordance_index_ipcw_scorer(CoxPHSurvivalAnalysis(), tau=tau)
param_grid = {"estimator__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}
gcv = GridSearchCV(est, param_grid, cv=3)
gcv.fit(Xtr, ytr)
print("best_params_:", gcv.best_params_)
print("best_score_:", gcv.best_score_)
print("test score:", gcv.score(Xte, yte))
```
実行結果:
```
best_params_: {'estimator__alpha': 1.0}
best_score_: 0.7564699953687343
test score: 0.786443529012979
```

**注意点・落とし穴**:
- ラップ対象の推定器のハイパーパラメータは、`GridSearchCV`の`param_grid`では`estimator__`プレフィックスを付けて指定する(`as_concordance_index_ipcw_scorer`が返すオブジェクト自身のパラメータ名が`estimator`であるため)。
- **`tau`を指定しないと、CVの分割方法によっては`ValueError: censoring survival function is zero at one or more time points`で学習が失敗することがある**(実行確認済み: WHAS500を`cv=3`で分割すると、あるfoldの検証データの最大観測時刻がそのfoldの学習データの打ち切り分布の範囲外に出てしまうケースが起きる)。`GridSearchCV`は失敗したfoldのスコアを`nan`にして警告を出すだけで例外を上に伝播させないため、`best_score_`が`nan`にならないか結果を必ず確認すること。`tau`(評価対象時間の上限)を学習データの最大観測時刻より小さく設定すると回避できることが多い。

#### `as_integrated_brier_score_scorer(...)`

**用途**: 積分ブライアスコア(IBS)を`GridSearchCV`で使えるスコアラーにする。IBSは値が小さいほど良い指標だが、scikit-learnの「スコアは大きいほど良い」という規約に合わせるため、**符号を反転した値**(負のIBS)を返す。

**シグネチャ**: `sksurv.metrics.as_integrated_brier_score_scorer(estimator, times)`

**使用例**:
```python
import numpy as np
from sksurv.metrics import as_integrated_brier_score_scorer

times = np.percentile(ytr["lenfol"], np.linspace(5, 60, 10))
est = as_integrated_brier_score_scorer(CoxPHSurvivalAnalysis(), times)
param_grid = {"estimator__alpha": [0.01, 0.1, 1.0]}
gcv = GridSearchCV(est, param_grid, cv=3)
gcv.fit(Xtr, ytr)
print("best_params_:", gcv.best_params_)
print("best_score_:", gcv.best_score_)
```
実行結果:
```
best_params_: {'estimator__alpha': 1.0}
best_score_: -0.1637977618965549
```

**注意点・落とし穴**:
- `best_score_`が負の値(`-0.1638`)になっているのは`predict_survival_function`の予測誤差がゼロではないという意味であり、バグではない。「大きい(0に近い)ほど良い」向きにするための符号反転である点を理解していないと、一見スコアが悪化しているように誤読しやすい。
- `times`は`brier_score`/`integrated_brier_score`と同様、学習・検証それぞれのfoldの追跡時間の範囲内に収まっている必要がある。範囲外だと`as_concordance_index_ipcw_scorer`と同種のエラーで学習が失敗する。

#### `as_cumulative_dynamic_auc_scorer(...)`

**用途**: 時間依存AUC(`cumulative_dynamic_auc`の平均AUC)を`GridSearchCV`で使えるスコアラーにする。

**シグネチャ**: `sksurv.metrics.as_cumulative_dynamic_auc_scorer(estimator, times, tied_tol=1e-08)`

**使用例**:
```python
from sksurv.metrics import as_cumulative_dynamic_auc_scorer

est = as_cumulative_dynamic_auc_scorer(CoxPHSurvivalAnalysis(alpha=0.1), times)
est.fit(Xtr, ytr)
print("score (mean AUC) on test:", est.score(Xte, yte))
```
実行結果:
```
score (mean AUC) on test: 0.8333514023431929
```

**注意点・落とし穴**:
- 上の単純な`fit`/`score`は問題なく動くが、これを**そのまま`GridSearchCV(cv=3)`に渡すと、`times`の上限を`as_concordance_index_ipcw_scorer`の例と同程度に絞っても`ValueError: censoring survival function is zero at one or more time points`や`ValueError: time must be smaller than largest observed time point: ...`で失敗することを実行して確認した**(WHAS500・`cv=3`または`StratifiedKFold(4)`のいずれでも再現)。原因は各CV foldの検証データに含まれるイベント時刻が、そのfoldの学習データから推定した打ち切り分布(Kaplan-Meier)の観測範囲を超えてしまうこと。`as_concordance_index_ipcw_scorer`より`times`の制約が厳しく実務では扱いにくいため、`GridSearchCV`と組み合わせる場合は事前に`times`を学習データの最大観測時刻より十分小さく取る、またはCVのfold数を増やす・層化するなどの対策が要る。単一の学習・評価分割で使う分には問題ない。

---

### 競合リスク・時間依存共変量への対応状況

#### `cumulative_incidence_competing_risks(...)`

**用途**: 打ち切りに加えて、複数の排他的なイベント種別(競合リスク)がある場合に、ノンパラメトリックに各リスクの累積罹患確率(cumulative incidence function, CIF)を推定する。

**シグネチャ**: `sksurv.nonparametric.cumulative_incidence_competing_risks(event, time_exit, time_min=None, conf_level=0.95, conf_type=None, var_type='Aalen')`

**使用例**:
```python
from sksurv.datasets import load_bmt
from sksurv.nonparametric import cumulative_incidence_competing_risks

X_bmt, y_bmt = load_bmt()
event, time = y_bmt["status"], y_bmt["ftime"]  # event: 0=打ち切り, 1/2=競合する2種類のリスク
print("event unique:", set(event))

x, ci, conf_int = cumulative_incidence_competing_risks(event, time, conf_type="log-log")
print("x[:5]:", x[:5])
print("ci.shape:", ci.shape)
print("ci[:, :5]:")
print(ci[:, :5].round(4))
print("conf_int.shape:", conf_int.shape)
```
実行結果:
```
event unique: {np.int64(0), np.int64(1), np.int64(2)}
x[:5]: [0. 1. 2. 3. 4.]
ci.shape: (3, 21)
ci[:, :5]:
[[0.0571 0.1143 0.1429 0.2611 0.3498]
 [0.0286 0.0571 0.0571 0.1458 0.2049]
 [0.0286 0.0571 0.0857 0.1153 0.1448]]
conf_int.shape: (3, 2, 21)
```

**注意点・落とし穴**:
- `event`は`Surv.from_arrays`が使うbool配列ではなく、**0=打ち切り、1〜n_risks=各リスク種別を表す非負整数配列**を渡す(`kaplan_meier_estimator`などとは仕様が異なる)。
- 戻り値`ci`の1行目(`ci[0]`)は「いずれかのリスクが発生する」全体の累積罹患確率、2行目以降(`ci[1:]`)が各リスク個別の累積罹患確率。`conf_int`も同様に`(n_risks + 1, 2, n_times)`の形。
- **正直な確認結果として、scikit-survival 0.28.0がサポートする競合リスクへの対応は、このノンパラメトリックなCIF推定(および前述の`get_x_y`の`competing_risks`引数によるデータ整形)までであり、共変量を使って競合リスクを回帰する手法(Fine-Grayの部分分布ハザードモデルなど)は実装されていない。** `sksurv`パッケージ全体を`grep -ri "fine-gray\|fine gray"`で検索してもヒットせず(実行確認済み)、`sksurv.linear_model`/`sksurv.ensemble`/`sksurv.svm`にも競合リスク回帰専用のクラスは存在しない。共変量付きの競合リスク回帰が必要な場合は`lifelines`(`CoxPHFitter`のcompeting risks拡張や`AalenJohansenFitter`)など他ライブラリの検討が必要。
- **同様に、時間依存共変量(time-varying covariates)にも対応していない。** `inspect.signature(CoxPHSurvivalAnalysis.fit)`を実行すると`(self, X, y)`のみで、`lifelines.CoxTimeVaryingFitter`のような`start`/`stop`区間を渡す仕組みは存在しない(実行確認済み)。scikit-survivalの全推定器は「1サンプル=1行の固定共変量」を前提としている。

---

### パイプライン統合

#### `Pipeline` + `CoxPHSurvivalAnalysis`

**用途**: scikit-learnの`Pipeline`にscikit-survivalの前処理・推定器をそのまま組み込む。生データ(カテゴリ列を含むDataFrame)から前処理〜Cox回帰まで一つの`fit`/`predict`/`score`にまとめられる。

**シグネチャ**: `sklearn.pipeline.Pipeline(steps, *, transform_input=None, memory=None, verbose=False)`(scikit-survival固有の追加シグネチャはなし。scikit-learn互換の`Transformer`/`Estimator`であれば組み込める)

**使用例**:
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.datasets import load_whas500
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis

X_raw, y = load_whas500()
Xtr_raw, Xte_raw, ytr, yte = train_test_split(X_raw, y, test_size=0.25, random_state=0)

pipe = Pipeline([
    ("onehot", OneHotEncoder()),       # sksurv.preprocessing: カテゴリ列をダミー変数化
    ("scale", StandardScaler()),       # sklearn: 数値列を標準化
    ("cox", CoxPHSurvivalAnalysis()),
])
pipe.fit(Xtr_raw, ytr)
print("score:", pipe.score(Xte_raw, yte))
print("predict[:3]:", pipe.predict(Xte_raw[:3]).round(3))
```
実行結果:
```
score: 0.8070936463383516
predict[:3]: [ 0.221 -0.42  -0.888]
```

**注意点・落とし穴**:
- ここでの`score`(0.8070936463383516)は、3章で`encode_categorical`を使い標準化なしで学習した`CoxPHSurvivalAnalysis`のC-index(同じく0.8070936463383516)と**完全に一致する**(実行確認済み)。Cox回帰の一致指数(順位のみに依存する指標)は特徴量の線形スケーリングに対して不変なため。ただし係数`coef_`の値そのものはスケーリングの有無で変わるので、係数の解釈をする場合は標準化の有無を意識する必要がある。
- `sksurv.preprocessing.OneHotEncoder`は`fit_transform`時に学習データに存在したカテゴリを記憶する。テストデータに学習データにない新規カテゴリ値が含まれると、`ColumnTransformer`の`OneHotEncoder`のような`handle_unknown`オプションは無いため挙動に注意(未知カテゴリの扱いは明示的に確認していない)。

#### `GridSearchCV` + `Pipeline`(ネストしたチューニング)

**用途**: 前処理を含む`Pipeline`全体を`as_concordance_index_ipcw_scorer`でラップし、`GridSearchCV`で前処理込みのハイパーパラメータ(ここでは`CoxnetSurvivalAnalysis`の`alphas`)を探索する。

**シグネチャ**: `sklearn.model_selection.GridSearchCV(estimator, param_grid, ...)`(前掲の`as_concordance_index_ipcw_scorer`と`Pipeline`を組み合わせて使う)

**使用例**:
```python
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import as_concordance_index_ipcw_scorer

pipe2 = Pipeline([
    ("onehot", OneHotEncoder()),
    ("scale", StandardScaler()),
    ("model", CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True)),
])
tau = ytr["lenfol"].max() * 0.9
scored_pipe = as_concordance_index_ipcw_scorer(pipe2, tau=tau)
# Pipelineのステップ名(model)__ハイパーパラメータ名 の前に estimator__ を重ねて指定する
param_grid = {"estimator__model__alphas": [[0.05], [0.02], [0.01], [0.005]]}
gcv = GridSearchCV(scored_pipe, param_grid, cv=3)
gcv.fit(Xtr_raw, ytr)
print("best_params_:", gcv.best_params_)
print("best_score_:", round(gcv.best_score_, 4))
print("test score:", round(gcv.score(Xte_raw, yte), 4))
```
実行結果:
```
best_params_: {'estimator__model__alphas': [0.05]}
best_score_: 0.7617
test score: 0.7713
```

**注意点・落とし穴**:
- `param_grid`のキーは`estimator__model__alphas`のように、「スコアラーのパラメータ名(`estimator`)」→「`Pipeline`のステップ名(`model`)」→「推定器自身のパラメータ名(`alphas`)」の順に`__`で連結する(ネストが1段増えるごとに`__`も1段増える)。
- `CoxnetSurvivalAnalysis`本来の挙動(3章参照)は`alphas`未指定で正則化パス全体を一度に学習するが、ここでは`alphas=[0.05]`のように**単一の値をリストで**渡して1点だけを評価させている。`alphas`に複数値を含むリストを渡すと、`Coxnet`内部でそのパス全体を学習してしまい`GridSearchCV`が意図した「1パラメータ=1候補」の探索にならないため、チューニング目的では単一値のリストにする必要がある。

---

### モデルの永続化(joblib)

#### `joblib.dump(...)`

**用途**: 学習済みのscikit-survival推定器(scikit-learn互換オブジェクト)をファイルにシリアライズして保存する。

**シグネチャ**: `joblib.dump(value, filename, compress=0, protocol=None)`

**使用例**:
```python
import joblib
from sksurv.linear_model import CoxPHSurvivalAnalysis

cph = CoxPHSurvivalAnalysis()
cph.fit(Xtr, ytr)
joblib.dump(cph, "/tmp/cph_model.joblib")
print("saved. score on Xte:", cph.score(Xte, yte))
```
実行結果:
```
saved. score on Xte: 0.8070936463383516
```

**注意点・落とし穴**:
- scikit-survivalの推定器はscikit-learnの`BaseEstimator`を継承しているだけの通常のPythonオブジェクトであり、保存・読み込みの仕組みはscikit-learnの推定器と同じ(`joblib`が標準的に推奨される)。scikit-survival固有のシリアライズ機構は無い。
- `pickle.dump`でも動作はするが、`joblib`の方がnumpy配列を含む大きなオブジェクトの保存・読み込みが高速な場合が多く、scikit-learn公式ドキュメントでも慣例的に`joblib`が使われる。

#### `joblib.load(...)`

**用途**: `joblib.dump`で保存したモデルファイルを読み込み、学習済み推定器として復元する。

**シグネチャ**: `joblib.load(filename, mmap_mode=None, ensure_native_byte_order='auto')`

**使用例**(別プロセスで実行して検証):
```python
import joblib

loaded = joblib.load("/tmp/cph_model.joblib")
print("loaded type:", type(loaded))
print("score on Xte (reloaded model):", loaded.score(Xte, yte))
print("predict[:5]:", loaded.predict(Xte[:5]).round(3))
```
実行結果:
```
loaded type: <class 'sksurv.linear_model.coxph.CoxPHSurvivalAnalysis'>
score on Xte (reloaded model): 0.8070936463383516
predict[:5]: [2.464 1.823 1.355 2.944 0.306]
```

**注意点・落とし穴**:
- 別プロセスで`joblib.load`した後の`score`/`predict`の値が、保存前の元モデルの値(3章の`CoxPHSurvivalAnalysis`の例と同じ`[2.464 1.823 1.355 2.944 0.306]`)と完全に一致することを実際にプロセスを分けて確認した。学習済みパラメータ(`coef_`など)だけでなく、内部でベースラインハザード推定に使うオブジェクトなども含めて復元される。
- `joblib`はPythonのバージョンやscikit-learn/scikit-survivalのバージョンをまたいだ互換性を保証しない。保存時と異なるバージョン環境で読み込むと、警告なしに誤った結果になったり例外になったりするリスクがあるため、モデルを長期保存する場合は保存時のライブラリバージョンも一緒に記録しておくことが望ましい。

---

### 発展的な生存SVM・メタ推定器

#### `NaiveSurvivalSVM(...)`

**用途**: `FastSurvivalSVM`とは別の実装で、比較可能なサンプルペアの特徴量差分から新しいデータセットを作り、通常の線形SVM(liblinear)で学習する「素朴な」生存SVM。

**シグネチャ**: `sksurv.svm.NaiveSurvivalSVM(penalty='l2', loss='squared_hinge', *, dual=False, tol=0.0001, alpha=1.0, verbose=0, random_state=None, max_iter=1000)`

**使用例**:
```python
from sksurv.svm import NaiveSurvivalSVM

nsvm = NaiveSurvivalSVM(alpha=1.0, max_iter=1000, random_state=0)
nsvm.fit(Xtr, ytr)
print("score:", nsvm.score(Xte, yte))
print("predict[:3]:", nsvm.predict(Xte[:3]).round(3))
```
実行結果:
```
score: 0.7994386323041592
predict[:3]: [1.125 0.929 0.702]
```

**注意点・落とし穴**:
- クラスのdocstring(`NaiveSurvivalSVM.__doc__`)に明記されている通り、比較可能な全ペアの差分を計算して新しいデータセットを作るため**空間計算量が`O(n_samples^2)`**になる。`FastSurvivalSVM`はペアの差分データセットを明示的に作らない、より効率的な最適化アルゴリズムを採用していると公式docstringに説明されているが、具体的な計算量オーダーはdocstring中に明記されておらず未確認。「名前通り、`NaiveSurvivalSVM`は教育的・比較用の素朴な実装」という位置づけはdocstringの記述(class docstringの冒頭)に基づく。

#### `MinlipSurvivalAnalysis(...)`

**用途**: `FastKernelSurvivalSVM`と関連するが、最大マージンではなく「最小Lipschitz平滑性」を基準にした異なる目的関数を最小化する生存モデル。

**シグネチャ**: `sksurv.svm.MinlipSurvivalAnalysis(alpha=1.0, *, solver='ecos', kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, pairs='nearest', verbose=False, timeit=None, max_iter=None)`

**使用例**:
```python
from sksurv.svm import MinlipSurvivalAnalysis

minlip = MinlipSurvivalAnalysis(alpha=1.0, solver="ecos")
minlip.fit(Xtr, ytr)
print("score:", minlip.score(Xte, yte))
```
実行結果:
```
score: 0.7991834651696862
```

**注意点・落とし穴**:
- `solver='ecos'`がデフォルト(凸最適化ソルバーECOSを使う)。`FastSurvivalSVM`系の準ニュートン最適化とは異なる最適化バックエンドに依存しており、追加の依存パッケージ(`ecos`)が必要になる。
- `pairs='nearest'`がデフォルトで、`HingeLossSurvivalSVM`(後述、デフォルト`pairs='all'`)より考慮するペア数が少なく、計算量を抑える設計になっている。

#### `HingeLossSurvivalSVM(...)`

**用途**: `NaiveSurvivalSVM`のカーネル版(非線形カーネルに対応した素朴な生存SVM)。

**シグネチャ**: `sksurv.svm.HingeLossSurvivalSVM(alpha=1.0, *, solver='ecos', kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, pairs='all', verbose=False, timeit=None, max_iter=None)`

**使用例**(計算量が非常に大きいため、学習・評価データを意図的に小さく絞っている):
```python
from sksurv.svm import HingeLossSurvivalSVM

Xtr_small, ytr_small = Xtr.iloc[:40], ytr[:40]
Xte_small, yte_small = Xte.iloc[:20], yte[:20]

hsvm = HingeLossSurvivalSVM(alpha=1.0, kernel="linear")
hsvm.fit(Xtr_small, ytr_small)
print("score:", hsvm.score(Xte_small, yte_small))
print("predict[:3]:", hsvm.predict(Xte_small[:3]).round(3))
```
実行結果:
```
score: 0.7857142857142857
predict[:3]: [4.249 3.869 5.783]
```

**注意点・落とし穴**:
- **クラスdocstringに明記されている通り空間計算量`O(n_samples^4)`・時間計算量`O(n_samples^6 * n_features)`という極めて重い計算量を要求する**(`NaiveSurvivalSVM`の`O(n_samples^2)`よりさらに重い)。この検証でも全データ(375学習サンプル)ではなく先頭40サンプルに絞って実行している。実務でそのまま数百〜数千サンプル規模のデータに使うのは現実的でなく、`FastKernelSurvivalSVM`(準ニュートン法ベース)の方が実用的な選択肢になる。

#### `Stacking(...)`

**用途**: 複数の生存モデル(ベース推定器)の予測を、メタ推定器(通常はCox回帰など)でさらに統合するスタッキングアンサンブル。

**シグネチャ**: `sksurv.meta.Stacking(meta_estimator, base_estimators, *, probabilities=True)`

**使用例**:
```python
from sksurv.meta import Stacking
from sksurv.svm import FastSurvivalSVM

base = [
    ("coxph", CoxPHSurvivalAnalysis()),
    ("svm", FastSurvivalSVM(max_iter=1000, random_state=0)),
]
stack = Stacking(CoxPHSurvivalAnalysis(), base, probabilities=False)
stack.fit(Xtr, ytr)
print("score:", stack.score(Xte, yte))
print("meta coef_ (coxph, svmの重み):", stack.meta_estimator.coef_)
```
実行結果:
```
score: 0.8070936463383516
meta coef_ (coxph, svmの重み): [1.0000000e+00 1.3546381e-13]
```

**注意点・落とし穴**:
- `base_estimators`は`(名前, 推定器)`のタプルのリストで、scikit-learnの`VotingClassifier`などと同じ書式。`meta_estimator`は各ベース推定器の予測値(この例では`coxph`と`svm`の2列)を新たな特徴量として学習する。
- この例では`meta coef_`が`[1.0, ~0]`となっており、**メタ推定器がほぼ`coxph`の予測だけを採用し`svm`の予測をほぼ無視した**ことが分かる(実行確認済み)。結果として`Stacking`全体のスコア(0.8070936463383516)は単体の`CoxPHSurvivalAnalysis`のスコア(3章参照、同じく0.8070936463383516)と一致している。スタッキングは自動的に性能を改善する保証はなく、ベース推定器同士の予測が似ている(強い相関がある)場合はこのように実質的に1つのモデルへ縮退することがある点に注意。
- `probabilities=True`(デフォルト)は「ベース推定器が`predict_proba`を持っていればそれを使い、無ければ`predict()`にフォールバックする」という実装になっている(実行確認: `Stacking._predict_estimators`のソースを確認、および`predict_proba`を持たない`FastSurvivalSVM`を`probabilities=True`のまま`base_estimators`に含めても`fit`はエラーにならないことを実際に実行して確認した)。scikit-survivalの推定器はいずれも`predict_proba`を持たないため、この例のように`probabilities=False`を明示しても実際の挙動(`predict()`を使う)は変わらない。`probabilities`引数は分類器も扱える`sksurv.meta`の共通実装に由来するオプションで、生存モデルだけを使う場合は実質的に意味を持たない。
