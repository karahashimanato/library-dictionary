# tsfresh 逆引き辞書

tsfresh 0.21.2 で検証済み

本ドキュメントに掲載しているシグネチャ・実行結果は、すべて `/home/manaty/library-practicing/.venv/bin/python`(tsfresh 0.21.2)で実際にコードを実行して得たものです。tsfresh は「long format」(id列・time列・value列を持つ縦持ちDataFrame)を入力とし、pandas DataFrame を介して scikit-learn 等のパイプラインに組み込む前提のライブラリです。デフォルトの特徴量抽出(`ComprehensiveFCParameters`)は列数が数百〜数千に膨れ上がり、`n_jobs`(デフォルト8)による multiprocessing とプログレスバーが標準で有効になるため、検証コードでは `n_jobs=0`(メインプロセスで実行)・`disable_progressbar=True` を指定して出力を安定させています。

## 目次

1. [データ形式・基礎](#データ形式基礎)
2. [特徴量抽出](#特徴量抽出)
3. [設定・カスタマイズ](#設定カスタマイズ)
4. [特徴量選択](#特徴量選択)
5. [個別特徴量計算関数](#個別特徴量計算関数)
6. [ローリング特徴量](#ローリング特徴量)
7. [パイプライン統合](#パイプライン統合)
8. [ユーティリティ](#ユーティリティ)
9. [応用・発展](#応用発展)
   - [設定・分散処理](#設定分散処理)
   - [カスタム特徴量計算関数の自作](#カスタム特徴量計算関数の自作)
   - [`extract_features` の高度なオプション](#extract_features-の高度なオプション)
   - [学習・推論を意識したユーティリティ](#学習推論を意識したユーティリティ)

---

## データ形式・基礎

### long format(id / time / value)

**用途**: tsfresh の各種関数に渡す入力DataFrameの基本形。「どの系列か(id)」「時刻(time)」「値(value)」を1行1観測値で並べた縦持ち(long format)にする。

**シグネチャ**: 関数ではなくデータ形式の約束事。`extract_features` 等の `column_id` / `column_sort` / `column_value` 引数でこの3列の列名を指定する。

**使用例**:
```python
import pandas as pd

df = pd.DataFrame({
    "id": [1, 1, 1, 1, 2, 2, 2, 2],
    "time": [0, 1, 2, 3, 0, 1, 2, 3],
    "value": [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 4.0, 5.0],
})
print(df)
print(df.dtypes)
```
実行結果:
```
   id  time  value
0   1     0    1.0
1   1     1    2.0
2   1     2    3.0
3   1     3    4.0
4   2     0    2.0
5   2     1    4.0
6   2     2    4.0
7   2     3    5.0
id         int64
time       int64
value    float64
dtype: object
```

**注意点・落とし穴**:
- `id` は「1つの時系列サンプル」を識別するキーで、特徴量抽出後の出力DataFrameの行(インデックス)に対応する。`time` は `column_sort` で指定し、ソート済みでなくても関数側でソートされる。
- 列名は `id`/`time`/`value` 固定ではなく、`column_id="..."` のように任意の列名を指定できる。

### `column_kind` を使った複数種類の時系列

**用途**: 1つのidに対して複数種類のセンサー値(例: 気温・気圧)がある場合、`column_kind` で種類を表す列を指定し、種類ごとに特徴量を計算する。

**シグネチャ**: `extract_features(..., column_kind=None, column_value=None)`(`column_kind` を指定する場合は `column_value` も必須)

**使用例**:
```python
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

df_kind = pd.DataFrame({
    "id": [1, 1, 1, 1, 1, 1, 1, 1],
    "time": [0, 1, 2, 3, 0, 1, 2, 3],
    "kind": ["temp", "temp", "temp", "temp", "pressure", "pressure", "pressure", "pressure"],
    "value": [20.0, 21.0, 22.0, 23.0, 1.0, 1.1, 1.2, 1.3],
})
extracted = extract_features(
    df_kind, column_id="id", column_sort="time", column_kind="kind", column_value="value",
    default_fc_parameters=MinimalFCParameters(), n_jobs=0, disable_progressbar=True,
)
print(extracted.shape)
print(extracted.columns.tolist())
```
実行結果:
```
(1, 20)
['pressure__sum_values', 'pressure__median', 'pressure__mean', 'pressure__length', 'pressure__standard_deviation', 'pressure__variance', 'pressure__root_mean_square', 'pressure__maximum', 'pressure__absolute_maximum', 'pressure__minimum', 'temp__sum_values', 'temp__median', 'temp__mean', 'temp__length', 'temp__standard_deviation', 'temp__variance', 'temp__root_mean_square', 'temp__maximum', 'temp__absolute_maximum', 'temp__minimum']
```

**注意点・落とし穴**:
- 出力列名は `<kind>__<特徴量名>` の形式になり、kindの種類数だけ列数が倍増する(この例では kind が2種類なので、単一kind時の10列が20列になった)。
- id=1が1行だけ、つまり「1id・複数kind」を横に並べる形になる点に注意(`column_kind`なしの場合は列名が`value__...`のみ)。

---

## 特徴量抽出

### `extract_features(...)`

**用途**: long format の時系列DataFrameから、id列ごとに大量の統計的特徴量を自動計算し、id列をインデックスとする横持ちDataFrameを返す。tsfreshの中核関数。

**シグネチャ**: `extract_features(timeseries_container, default_fc_parameters=None, kind_to_fc_parameters=None, column_id=None, column_sort=None, column_kind=None, column_value=None, chunksize=None, n_jobs=8, show_warnings=False, disable_progressbar=False, impute_function=None, profile=False, profiling_filename='profile.txt', profiling_sorting='cumulative', distributor=None, pivot=True)`

**使用例**:
```python
import numpy as np
from tsfresh import extract_features

np.random.seed(0)
rows = []
for id_ in [1, 2]:
    for t in range(20):
        rows.append({"id": id_, "time": t, "value": np.sin(t / 3) + id_ + np.random.normal(0, 0.1)})
df_long = pd.DataFrame(rows)

extracted = extract_features(
    df_long, column_id="id", column_sort="time", n_jobs=0, disable_progressbar=True,
)
print("shape:", extracted.shape)
print(extracted.columns.tolist()[:5])
nan_cols = extracted.columns[extracted.isna().any()].tolist()
print("NaNを含む列数:", len(nan_cols))
print(nan_cols[:3])
```
実行結果:
```
shape: (2, 783)
['value__variance_larger_than_standard_deviation', 'value__has_duplicate_max', 'value__has_duplicate_min', 'value__has_duplicate', 'value__sum_values']
NaNを含む列数: 387
['value__ar_coefficient__coeff_0__k_10', 'value__ar_coefficient__coeff_1__k_10', 'value__ar_coefficient__coeff_2__k_10']
```

**注意点・落とし穴**:
- `default_fc_parameters` を省略するとデフォルトの `ComprehensiveFCParameters` が使われ、長さ20点の系列でも783列にまで膨れ上がる。実務では `MinimalFCParameters`/`EfficientFCParameters` や `kind_to_fc_parameters` で計算対象を絞るのが定石。
- `ar_coefficient` の `k_10` など、系列長よりウィンドウが大きい特徴量計算は自動的に `NaN` を返す(エラーにはならないが警告なく欠損が生じるため、後段で `impute()` が必要になることが多い)。
- `n_jobs` はデフォルト8(マルチプロセス)。小規模データやデバッグ時は `n_jobs=0` にするとオーバーヘッドが減り、進捗表示も `disable_progressbar=True` で消せる。
- 戻り値のインデックスは `column_id` の値そのもの(この例では `1, 2`)になる。

### `extract_relevant_features(timeseries_container, y, ...)`

**用途**: `extract_features` → `impute` → `select_features` を1回で行う便利関数。目的変数 `y` に対して統計的に関連のある特徴量だけを残した状態で返す。

**シグネチャ**: `extract_relevant_features(timeseries_container, y, X=None, default_fc_parameters=None, kind_to_fc_parameters=None, column_id=None, column_sort=None, column_kind=None, column_value=None, show_warnings=False, disable_progressbar=False, profile=False, profiling_filename='profile.txt', profiling_sorting='cumulative', test_for_binary_target_binary_feature='fisher', test_for_binary_target_real_feature='mann', test_for_real_target_binary_feature='mann', test_for_real_target_real_feature='kendall', fdr_level=0.05, hypotheses_independent=False, n_jobs=8, distributor=None, chunksize=None, ml_task='auto')`

**使用例**:
```python
import numpy as np
from tsfresh import extract_relevant_features

np.random.seed(1)
rows, y = [], {}
for id_ in range(1, 21):
    label = id_ % 2
    y[id_] = label
    base = 10 if label == 1 else 0
    for t in range(15):
        rows.append({"id": id_, "time": t, "value": base + np.random.normal(0, 1)})
df_rel = pd.DataFrame(rows)
y_series = pd.Series(y)

features_filtered = extract_relevant_features(
    df_rel, y_series, column_id="id", column_sort="time", n_jobs=0, disable_progressbar=True,
)
print(features_filtered.shape)
print(features_filtered.columns.tolist()[:5])
```
実行結果:
```
(20, 94)
['value__number_crossing_m__m_-1', 'value__range_count__max_0__min_-1000000000000.0', 'value__count_below__t_0', 'value__range_count__max_1000000000000.0__min_0', 'value__count_above__t_0']
```

**注意点・落とし穴**:
- `y` は `id` と同じ値をインデックスに持つ `pd.Series` である必要がある(この例では `y = pd.Series({id: label, ...})`)。
- ラベルと系列が弱くしか関連していないデータ(例えば `base` の差が小さい/idの数が少ない)では、有意な特徴量が0個になり `shape` が `(n, 0)` になることもある(実際に少数idかつ差が小さい設定で検証済み)。

---

## 設定・カスタマイズ

### `ComprehensiveFCParameters` / `EfficientFCParameters` / `MinimalFCParameters`

**用途**: `extract_features` の `default_fc_parameters` に渡す「どの特徴量計算関数を、どのパラメータで実行するか」の設定辞書。プリセットが3種類用意されている。

**シグネチャ**: いずれも `dict` のサブクラスで、引数なしでインスタンス化する(`ComprehensiveFCParameters()` など)。

**使用例**:
```python
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

c = ComprehensiveFCParameters()
e = EfficientFCParameters()
m = MinimalFCParameters()
print("Comprehensive:", len(c), "種類")
print("Efficient:", len(e), "種類")
print("Minimal:", list(m.keys()))
print("ComprehensiveのみでEfficientに無いもの:", set(c) - set(e))
```
実行結果:
```
Comprehensive: 75 種類
Efficient: 73 種類
Minimal: ['sum_values', 'median', 'mean', 'length', 'standard_deviation', 'variance', 'root_mean_square', 'maximum', 'absolute_maximum', 'minimum']
ComprehensiveのみでEfficientに無いもの: {'sample_entropy', 'approximate_entropy'}
```

**注意点・落とし穴**:
- `EfficientFCParameters` は `ComprehensiveFCParameters` から計算コストの高い `sample_entropy` / `approximate_entropy` のみを除いたもの(それ以外は同一)。
- `MinimalFCParameters` は10種類の基本統計量のみで高速だが、`length`(系列長)や `sum_values` など単純な特徴量しか得られない。まず `MinimalFCParameters` で動作確認し、必要に応じて `EfficientFCParameters`/`ComprehensiveFCParameters` に切り替えるのが実務的なワークフロー。
- 辞書のキーは `feature_calculators` モジュールの関数名と一致し、値はパラメータ付き計算の場合 `[{"引数名": 値, ...}, ...]` のリスト、パラメータ無し計算の場合 `None`。

### `from_columns(columns)`

**用途**: `extract_features` が出力した列名(`value__mean` など)から、その特徴量だけを再計算するための `fc_parameters` 辞書を逆生成する。特徴量選択後に「選ばれた特徴量だけ」を再計算したい場合に使う。

**シグネチャ**: `from_columns(columns, columns_to_ignore=None)`

**使用例**:
```python
from tsfresh.feature_extraction.settings import from_columns

cols = ["value__mean", "value__number_peaks__n_1", 'value__fft_coefficient__attr_"abs"__coeff_0']
result = from_columns(cols)
print(result)
```
実行結果:
```
{'value': {'mean': None, 'number_peaks': [{'n': 1}], 'fft_coefficient': [{'attr': 'abs', 'coeff': 0}]}}
```

**注意点・落とし穴**:
- 戻り値は `{kind: {feature_name: params, ...}}` の形になっており、これをそのまま `kind_to_fc_parameters=from_columns(cols)` として `extract_features` に渡せる。
- `select_features` などで選ばれた列名の文字列をそのまま解析するため、列名の命名規則(`__`区切り)を手動で書き換えていると解析に失敗する。

---

## 特徴量選択

### `select_features(X, y, ...)`

**用途**: `extract_features` で得た特徴量DataFrame `X` から、目的変数 `y` との統計的関連が有意な列だけを残す(多重検定補正付きの仮説検定によるフィルタリング)。

**シグネチャ**: `select_features(X, y, test_for_binary_target_binary_feature='fisher', test_for_binary_target_real_feature='mann', test_for_real_target_binary_feature='mann', test_for_real_target_real_feature='kendall', fdr_level=0.05, hypotheses_independent=False, n_jobs=8, show_warnings=False, chunksize=None, ml_task='auto', multiclass=False, n_significant=1)`

**使用例**:
```python
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

np.random.seed(1)
rows, y = [], {}
for id_ in range(1, 11):
    label = id_ % 2
    y[id_] = label
    base = 5 if label == 1 else 0
    for t in range(15):
        rows.append({"id": id_, "time": t, "value": base + np.random.normal(0, 1)})
df_sel = pd.DataFrame(rows)
y_series = pd.Series(y)

X = extract_features(df_sel, column_id="id", column_sort="time",
                      default_fc_parameters=MinimalFCParameters(), n_jobs=0, disable_progressbar=True)
impute(X)
X_filtered = select_features(X, y_series)
print("X shape:", X.shape, "-> X_filtered shape:", X_filtered.shape)
print(X_filtered.columns.tolist())
```
実行結果:
```
X shape: (10, 10) -> X_filtered shape: (10, 7)
['value__sum_values', 'value__median', 'value__mean', 'value__root_mean_square', 'value__maximum', 'value__absolute_maximum', 'value__minimum']
```

**注意点・落とし穴**:
- `X` に `NaN`/`inf` が含まれていると内部の統計検定でエラーになりやすいため、事前に `impute(X)` を呼んでおくのが定石(`extract_relevant_features` は内部でこれを自動実行している)。
- `y` が二値か連続値か、`X` の各列が二値か連続値かによって自動的に検定手法(fisher/mann/kendall)が選ばれる(`ml_task="auto"` のデフォルト動作)。

### `calculate_relevance_table(X, y, ...)`

**用途**: `select_features` の内部で使われている、各特徴量ごとのp値・有意判定を一覧できるDataFrameを直接取得する(選択前に検定結果を確認したい場合に使う)。

**シグネチャ**: `calculate_relevance_table(X, y, ml_task='auto', multiclass=False, n_significant=1, n_jobs=8, show_warnings=False, chunksize=None, test_for_binary_target_binary_feature='fisher', test_for_binary_target_real_feature='mann', test_for_real_target_binary_feature='mann', test_for_real_target_real_feature='kendall', fdr_level=0.05, hypotheses_independent=False)`

**使用例**:
```python
from tsfresh.feature_selection.relevance import calculate_relevance_table

table = calculate_relevance_table(X, y_series, n_jobs=0)
print(table[["feature", "type", "p_value", "relevant"]].head(5))
```
実行結果:
```
                                         feature  type   p_value  relevant
feature                                                                   
value__sum_values              value__sum_values  real  0.007937      True
value__median                      value__median  real  0.007937      True
value__mean                          value__mean  real  0.007937      True
value__root_mean_square  value__root_mean_square  real  0.007937      True
value__maximum                    value__maximum  real  0.007937      True
```

**注意点・落とし穴**:
- 返り値の `relevant` 列は `select_features` が残す列と一致する(`select_features(X, y)` は実質 `calculate_relevance_table` の `relevant=True` 行に絞る処理)。
- `p_value` は多重検定補正(Benjamini-Hochberg 型のFDR補正、`fdr_level`)後の値ではなく生のp値であり、`relevant` 列の判定に補正が反映されている点に注意。

---

## 個別特徴量計算関数

`tsfresh.feature_extraction.feature_calculators` モジュールには、`extract_features` が内部で呼び出す個々の特徴量計算関数(1次元 `numpy.ndarray` または `pandas.Series` を受け取りスカラー/タプル列を返す)が定義されている。単体の系列に対して特徴量を1つだけ試したい場合に直接呼び出せる。

### `mean` / `variance` / `standard_deviation`

**用途**: 平均・分散・標準偏差(いずれも `MinimalFCParameters` にも含まれる基本統計量)。

**シグネチャ**: `mean(x)` / `variance(x)` / `standard_deviation(x)`(いずれも引数はシリーズ1個のみ)

**使用例**:
```python
import numpy as np
import tsfresh.feature_extraction.feature_calculators as fc

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
print("mean:", fc.mean(x))
print("variance:", fc.variance(x))
print("standard_deviation:", fc.standard_deviation(x))
```
実行結果:
```
mean: 2.7777777777777777
variance: 1.7283950617283952
standard_deviation: 1.314684396244359
```

### `abs_energy(x)`

**用途**: 各値の2乗和(信号のエネルギー量)。

**シグネチャ**: `abs_energy(x)`

**使用例**:
```python
print("abs_energy:", fc.abs_energy(x))
```
実行結果:
```
abs_energy: 85.0
```

### `number_peaks(x, n)`

**用途**: 前後 `n` 点よりも大きい「ピーク」の個数を数える。

**シグネチャ**: `number_peaks(x, n)`

**使用例**:
```python
print("number_peaks(n=1):", fc.number_peaks(x, 1))
```
実行結果:
```
number_peaks(n=1): 1
```

**注意点・落とし穴**:
- `n` はピークの左右何点と比較するかを表す。`n` を大きくするほど「なだらかな山」しかピークとみなされなくなり、検出数は減りやすい。

### `longest_strike_above_mean(x)` / `count_above_mean(x)`

**用途**: 平均より上にある値が連続する最大長 / 平均より上にある値の総数。

**シグネチャ**: `longest_strike_above_mean(x)` / `count_above_mean(x)`

**使用例**:
```python
print("longest_strike_above_mean:", fc.longest_strike_above_mean(x))
print("count_above_mean:", fc.count_above_mean(x))
```
実行結果:
```
longest_strike_above_mean: 5
count_above_mean: 5
```

### `autocorrelation(x, lag)`

**用途**: 指定ラグでの自己相関係数を計算する。

**シグネチャ**: `autocorrelation(x, lag)`

**使用例**:
```python
print("autocorrelation(lag=1):", fc.autocorrelation(x, 1))
```
実行結果:
```
autocorrelation(lag=1): 0.6071428571428572
```

### `fft_coefficient(x, param)`

**用途**: 離散フーリエ変換(FFT)の指定係数(実部・虚部・絶対値・位相など)を取り出す。

**シグネチャ**: `fft_coefficient(x, param)`(`param` は `[{"coeff": int, "attr": "real"|"imag"|"abs"|"angle"}, ...]` のリスト)

**使用例**:
```python
result = list(fc.fft_coefficient(x, [{"coeff": 0, "attr": "abs"}, {"coeff": 1, "attr": "abs"}]))
print(result)
```
実行結果:
```
[('attr_"abs"__coeff_0', np.float64(25.0)), ('attr_"abs"__coeff_1', np.float64(8.29085936938159))]
```

**注意点・落とし穴**:
- 多くの `*_coefficient`/`*param`系関数(`fft_coefficient`, `linear_trend` など)は単一値ではなく `(名前, 値)` のタプルを要素とする**ジェネレータ**を返す。`list()` で展開しないと中身が見えない。

### `linear_trend(x, param)`

**用途**: 系列全体に対する線形回帰(時刻を説明変数とする単回帰)の傾き・切片・決定係数などを計算する。

**シグネチャ**: `linear_trend(x, param)`(`param` は `[{"attr": "slope"|"intercept"|"rvalue"|"pvalue"|"stderr"}, ...]`)

**使用例**:
```python
result = list(fc.linear_trend(x, [{"attr": "slope"}, {"attr": "intercept"}]))
print(result)
```
実行結果:
```
[('attr_"slope"', np.float64(-1.4802973661668754e-17)), ('attr_"intercept"', np.float64(2.7777777777777777))]
```

### `c3(x, lag)` / `cid_ce(x, normalize)`

**用途**: 系列の非線形性(`c3`)・複雑さ(`cid_ce`、値の変動の激しさ)を数値化する。

**シグネチャ**: `c3(x, lag)` / `cid_ce(x, normalize)`

**使用例**:
```python
print("c3(lag=1):", fc.c3(x, 1))
print("cid_ce(normalize=False):", fc.cid_ce(x, False))
print("cid_ce(normalize=True):", fc.cid_ce(x, True))
```
実行結果:
```
c3(lag=1): 37.142857142857146
cid_ce(normalize=False): 2.8284271247461903
cid_ce(normalize=True): 2.1514114968019085
```

### `binned_entropy(x, max_bins)`

**用途**: 値を `max_bins` 個の等幅ビンに分けたヒストグラムのエントロピーを計算する(値の分布の「散らばり具合」の指標)。

**シグネチャ**: `binned_entropy(x, max_bins)`

**使用例**:
```python
print("binned_entropy(max_bins=5):", fc.binned_entropy(x, 5))
```
実行結果:
```
binned_entropy(max_bins=5): 1.5810937501718236
```

**注意点・落とし穴**:
- `sample_entropy`/`approximate_entropy` など一部の複雑度指標は、系列が短い・左右対称すぎるなどの条件では `0/0` 相当の計算になり `RuntimeWarning: invalid value encountered in scalar divide` とともに `nan` を返すことを確認済み(`fc.sample_entropy(x)` で実際に発生)。`ComprehensiveFCParameters` にはこれらが含まれるため、`extract_features` の出力に説明なく `NaN` が混じる一因になる。

---

## ローリング特徴量

### `roll_time_series(df_or_dict, column_id, column_sort, ...)`

**用途**: 時系列を「時刻ごとの累積ウィンドウ」に分割し、各時点までの部分系列ごとに新しいidを振り直す。時点ごとの予測(1時点先予測など)向けに特徴量を作る前処理。

**シグネチャ**: `roll_time_series(df_or_dict, column_id, column_sort=None, column_kind=None, rolling_direction=1, max_timeshift=None, min_timeshift=0, chunksize=None, n_jobs=8, show_warnings=False, disable_progressbar=False, distributor=None)`

**使用例**:
```python
from tsfresh.utilities.dataframe_functions import roll_time_series

df_roll = pd.DataFrame({
    "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    "time": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    "value": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
})
rolled = roll_time_series(
    df_roll, column_id="id", column_sort="time", max_timeshift=2, min_timeshift=2,
    disable_progressbar=True,
)
print(rolled)
```
実行結果:
```
    time  value      id
12     0      1  (1, 2)
13     1      2  (1, 2)
14     2      3  (1, 2)
0      1      2  (1, 3)
1      2      3  (1, 3)
2      3      4  (1, 3)
6      2      3  (1, 4)
7      3      4  (1, 4)
8      4      5  (1, 4)
15     0     10  (2, 2)
16     1     20  (2, 2)
17     2     30  (2, 2)
3      1     20  (2, 3)
4      2     30  (2, 3)
5      3     40  (2, 3)
9      2     30  (2, 4)
10     3     40  (2, 4)
11     4     50  (2, 4)
```

**注意点・落とし穴**:
- 新しい `id` 列は `(元のid, ウィンドウ末尾の時刻)` というタプルになる。この出力をそのまま `extract_features(..., column_id="id")` に渡すと、ウィンドウごとの特徴量が計算できる。
- `min_timeshift` を指定しないと極端に短い(1点だけの)ウィンドウも生成され、特徴量計算で `NaN` が増える原因になる。

### `make_forecasting_frame(x, kind, max_timeshift, rolling_direction, min_timeshift=0)`

**用途**: 1本の時系列 `x` から、1時点先の値を予測するための「ローリングされた特徴量抽出用DataFrame」と「予測対象(教師)のSeries」を同時に作る。

**シグネチャ**: `make_forecasting_frame(x, kind, max_timeshift, rolling_direction, min_timeshift=0)`

**使用例**:
```python
from tsfresh.utilities.dataframe_functions import make_forecasting_frame

x = pd.Series([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0])
df_shift, y_shift = make_forecasting_frame(x, kind="fib", max_timeshift=3, rolling_direction=1)
print(y_shift)
```
実行結果:
```
(id, 1)     2.0
(id, 2)     3.0
(id, 3)     5.0
(id, 4)     8.0
(id, 5)    13.0
(id, 6)    21.0
Name: value, dtype: float64
```

**注意点・落とし穴**:
- 返り値 `y_shift` のインデックスは `df_shift` の `id` 列の値と一致するように作られており、`extract_features(df_shift, ...)` の出力にそのまま教師データとして使える設計になっている。
- `x` の最初の1点は「予測対象がない」ため `y_shift` には含まれない(7点の入力に対し `y_shift` は6点)。

---

## パイプライン統合

### `FeatureAugmenter` / `RelevantFeatureAugmenter`(sklearn連携)

**用途**: scikit-learn の `Pipeline` に組み込める Transformer。`FeatureAugmenter` は特徴量抽出のみ、`RelevantFeatureAugmenter` は抽出後に `y` との関連性フィルタ(`select_features`相当)まで行う。

**シグネチャ**: `FeatureAugmenter(default_fc_parameters=None, kind_to_fc_parameters=None, column_id=None, column_sort=None, column_kind=None, column_value=None, timeseries_container=None, chunksize=None, n_jobs=8, show_warnings=False, disable_progressbar=False, impute_function=None, profile=False, ...)` / `RelevantFeatureAugmenter(filter_only_tsfresh_features=True, default_fc_parameters=None, ..., timeseries_container=None, ..., fdr_level=0.05, ..., ml_task='auto', ...)`

**使用例**:
```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.feature_extraction import MinimalFCParameters

np.random.seed(2)
rows, y = [], {}
for id_ in range(1, 21):
    label = id_ % 2
    y[id_] = label
    base = 10 if label == 1 else 0
    for t in range(15):
        rows.append({"id": id_, "time": t, "value": base + np.random.normal(0, 1)})
timeseries = pd.DataFrame(rows)
y_series = pd.Series(y)
X_base = pd.DataFrame(index=y_series.index)  # 特徴量抽出前の「空」の特徴量行列

pipeline = Pipeline([
    ("augmenter", RelevantFeatureAugmenter(
        column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True, n_jobs=0,
    )),
    ("clf", RandomForestClassifier(random_state=0)),
])
pipeline.set_params(augmenter__timeseries_container=timeseries)
pipeline.fit(X_base, y_series)
print("selected:", pipeline.named_steps["augmenter"].feature_selector.relevant_features)
print("train score:", pipeline.score(X_base, y_series))
```
実行結果:
```
selected: ['value__sum_values', 'value__median', 'value__mean', 'value__root_mean_square', 'value__maximum', 'value__absolute_maximum', 'value__minimum']
train score: 1.0
```

**注意点・落とし穴**:
- `X_base` は `id` と同じインデックスを持つ「空の(あるいは他の特徴量を含む)」DataFrameを渡す必要がある。実際の時系列データは `timeseries_container` として別途 `set_params(augmenter__timeseries_container=...)` で渡す点が独特(コンストラクタ引数ではなく `set_params` 経由が一般的)。
- `RelevantFeatureAugmenter` は `fit` 時に `y` を使って関連特徴量を選ぶため、`transform` のみを他データに適用する場合は `fit` 時に選ばれた特徴量だけが使われる(新規の特徴量は追加されない)。

---

## ユーティリティ

### `impute(df_impute)` / `impute_dataframe_zero(df_impute)`

**用途**: 特徴量DataFrame中の `NaN`/`inf`/`-inf` を、列ごとの最小値・最大値・中央値(`impute`)またはゼロ(`impute_dataframe_zero`)で置き換える。**破壊的(inplace)に元のDataFrameを書き換える**。

**シグネチャ**: `impute(df_impute)` / `impute_dataframe_zero(df_impute)`

**使用例**:
```python
import numpy as np
from tsfresh.utilities.dataframe_functions import impute

X_imp = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.inf], "b": [-np.inf, 2.0, 3.0, 4.0]})
print("before:\n", X_imp)
impute(X_imp)
print("after:\n", X_imp)
```
実行結果:
```
before:
      a    b
0  1.0 -inf
1  NaN  2.0
2  3.0  3.0
3  inf  4.0
after:
      a    b
0  1.0  2.0
1  2.0  2.0
2  3.0  3.0
3  3.0  4.0
```

**注意点・落とし穴**:
- `impute` は戻り値を持たず(`None`)、引数の DataFrame を直接書き換える(`df.fillna()` のように新しいDataFrameを返す関数ではない)。呼び出し後は元の変数がそのまま更新済みDataFrameになる。
- `+inf` は列の最大値、`-inf` は列の最小値、`NaN` は列の中央値に置き換えられる(この例では `a` 列の `inf` が3.0の代わりに元データの最大有限値である3.0に、`-inf` は最小有限値2.0になっている)。

### `restrict_input_to_index(df_or_dict, column_id, index)`

**用途**: 長いlong-format DataFrameから、指定した `id` の集合(`index`)に該当する行だけを抜き出す。訓練用/テスト用にidを分割する際などに使う。

**シグネチャ**: `restrict_input_to_index(df_or_dict, column_id, index)`

**使用例**:
```python
from tsfresh.utilities.dataframe_functions import restrict_input_to_index

df_restrict = pd.DataFrame({"id": [1, 1, 2, 2, 3, 3], "time": [0, 1, 0, 1, 0, 1], "value": [1, 2, 3, 4, 5, 6]})
restricted = restrict_input_to_index(df_restrict, "id", pd.Index([1, 3]))
print(restricted)
```
実行結果:
```
   id  time  value
0   1     0      1
1   1     1      2
4   3     0      5
5   3     1      6
```

### `check_for_nans_in_columns(df, columns=None)`

**用途**: DataFrameの指定列(省略時は全列)に `NaN` が含まれていないかを検査し、含まれていれば例外を送出する。特徴量選択などの前段でのガード処理に使われる。

**シグネチャ**: `check_for_nans_in_columns(df, columns=None)`

**使用例**:
```python
from tsfresh.utilities.dataframe_functions import check_for_nans_in_columns

try:
    check_for_nans_in_columns(pd.DataFrame({"a": [1.0, None]}))
except ValueError as e:
    print("raised:", e)
```
実行結果:
```
raised: Columns ['a'] of DataFrame must not contain NaN values
```

**注意点・落とし穴**:
- 戻り値による通知ではなく `ValueError` の送出で異常を知らせる。`select_features` などを呼ぶ前に `impute()` を忘れると、この関数由来のエラーで処理が止まることがある。

---

## 応用・発展

### 設定・分散処理

#### `kind_to_fc_parameters` と `default_fc_parameters` の併用

**用途**: `column_kind` で種類分けした時系列に対し、種類(kind)ごとに異なる特徴量セットを `kind_to_fc_parameters` で個別指定する。指定されなかった kind には `default_fc_parameters` がフォールバックとして使われる。

**シグネチャ**: `extract_features(..., default_fc_parameters=None, kind_to_fc_parameters=None, ...)` の該当引数。`kind_to_fc_parameters` は `{kind名: fc_parameters辞書}` の形。

**使用例**:
```python
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

df_kind = pd.DataFrame({
    "id": [1] * 12,
    "time": list(range(4)) * 3,
    "kind": ["temp"] * 4 + ["pressure"] * 4 + ["humidity"] * 4,
    "value": [20.0, 21.0, 22.0, 23.0, 1.0, 1.1, 1.2, 1.3, 50.0, 51.0, 52.0, 53.0],
})
kind_to_fc_parameters = {"temp": {"mean": None}}

# default_fc_parameters を指定した場合: temp以外(pressure/humidity)はそちらにフォールバック
extracted_with_default = extract_features(
    df_kind, column_id="id", column_sort="time", column_kind="kind", column_value="value",
    default_fc_parameters=MinimalFCParameters(),
    kind_to_fc_parameters=kind_to_fc_parameters, n_jobs=0, disable_progressbar=True,
)
print("default_fc_parameters指定あり:", extracted_with_default.shape)

# default_fc_parameters を省略した場合
extracted_without_default = extract_features(
    df_kind, column_id="id", column_sort="time", column_kind="kind", column_value="value",
    kind_to_fc_parameters=kind_to_fc_parameters, n_jobs=0, disable_progressbar=True,
)
print("default_fc_parameters省略:", extracted_without_default.shape)
print(extracted_without_default.columns.tolist())
```
実行結果:
```
default_fc_parameters指定あり: (1, 21)
default_fc_parameters省略: (1, 1)
['temp__mean']
```

**注意点・落とし穴**:
- `kind_to_fc_parameters` を指定して `default_fc_parameters` を省略すると、`default_fc_parameters` は `ComprehensiveFCParameters()` ではなく**空辞書 `{}`** になる(`extraction.py` の実装で `kind_to_fc_parameters is not None` の場合はこの挙動)。結果として `kind_to_fc_parameters` に列挙されていない kind(この例では pressure/humidity)は特徴量が**1列も計算されない**まま静かに欠落する。全kindに何らかの特徴量を持たせたい場合は、`default_fc_parameters` を明示的に指定すること。

#### `MultiprocessingDistributor`(分散処理)

**用途**: `extract_features` の並列実行を担う既定の分散実行クラス。`n_jobs` 引数の裏側で暗黙的に使われているものを明示的にインスタンス化し、`distributor` 引数として渡せる。

**シグネチャ**: `MultiprocessingDistributor(n_workers, disable_progressbar=False, progressbar_title='Feature Extraction', show_warnings=True)`(`tsfresh.utilities.distribution` モジュール)

**使用例**:
```python
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.distribution import MultiprocessingDistributor

np.random.seed(0)
rows = []
for id_ in range(1, 6):
    for t in range(10):
        rows.append({"id": id_, "time": t, "value": np.sin(t / 3) + id_ + np.random.normal(0, 0.1)})
df = pd.DataFrame(rows)

distributor = MultiprocessingDistributor(n_workers=2, disable_progressbar=True, show_warnings=False)
extracted = extract_features(
    df, column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters(),
    distributor=distributor,
)
print(extracted.shape)
print(extracted.index.tolist())
```
実行結果:
```
(5, 10)
[1, 2, 3, 4, 5]
```

**注意点・落とし穴**:
- `distributor` を渡す場合、並列度は `distributor` 生成時の `n_workers` で決まり、`extract_features` 側の `n_jobs` 引数は使われない(両方渡しても `distributor` が優先される)。

#### `LocalDaskDistributor`(dask未インストール時の挙動)

**用途**: dask の分散実行基盤(`distributed`)上で特徴量抽出を並列化するための `Distributor`。大規模データをクラスタ/ローカルの複数プロセスに分散したい場合に使う。

**シグネチャ**: `LocalDaskDistributor(n_workers)`(`tsfresh.utilities.distribution` モジュール)

**使用例**:
```python
from tsfresh.utilities.distribution import LocalDaskDistributor

try:
    d = LocalDaskDistributor(n_workers=2)
    print("created", d)
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
ModuleNotFoundError No module named 'distributed'
```

**注意点・落とし穴**:
- 検証環境(tsfresh 0.21.2, Python 3.12)には `dask` 自体もインストールされておらず、`LocalDaskDistributor` は `dask.distributed`(`distributed` パッケージ)への依存を内部で `import` するため、未インストール環境では**インスタンス化した瞬間に** `ModuleNotFoundError` になることを確認した。tsfresh の基本インストール(`pip install tsfresh`)には dask 系パッケージは含まれないため、使う場合は別途 `pip install dask distributed` 等が必要。

---

### カスタム特徴量計算関数の自作

#### `@set_property("fctype", "simple")` による自作関数の登録

**用途**: 既存の特徴量計算関数だけでは足りない独自指標(例: 値の範囲=最大値-最小値)を自作し、`extract_features` の `default_fc_parameters`/`kind_to_fc_parameters` にそのまま渡して計算させる。

**シグネチャ**: `tsfresh.feature_extraction.feature_calculators.set_property(key, value)` はデコレータファクトリで、関数オブジェクトに属性を1つ設定するだけ(`func.fctype = "simple"` と同義)。`fctype` が `"simple"` の関数は `func(x)` の形(パラメータなし)または `func(x, **param)` の形(パラメータあり)で呼ばれる。

**使用例**:
```python
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction.feature_calculators import set_property

@set_property("fctype", "simple")
def peak_to_peak(x):
    x = np.asarray(x)
    return np.max(x) - np.min(x)

df = pd.DataFrame({
    "id": [1, 1, 1, 1, 2, 2, 2, 2],
    "time": [0, 1, 2, 3, 0, 1, 2, 3],
    "value": [1.0, 5.0, 2.0, 3.0, 10.0, 10.0, 10.0, 10.0],
})

# 関数名の文字列ではなく、関数オブジェクトそのものを辞書のキーにできる
fc_parameters = {peak_to_peak: None}
extracted = extract_features(
    df, column_id="id", column_sort="time",
    default_fc_parameters=fc_parameters, n_jobs=0, disable_progressbar=True,
)
print(extracted)
```
実行結果:
```
   value__peak_to_peak
1                  4.0
2                  0.0
```

**注意点・落とし穴**:
- `extract_features` の内部実装(`_do_extraction_on_chunk`)は `fc_parameters` 辞書のキーが `callable` であればそれをそのまま関数として使い、文字列であれば `tsfresh.feature_extraction.feature_calculators` モジュールから同名属性を `getattr` で探す。つまり自作関数を `feature_calculators` モジュールに登録(モンキーパッチ)しなくても、**関数オブジェクトを辞書のキーとして直接渡せば動く**(この例で検証済み)。
- 出力列名は `<kind>__<関数の__name__>` になる(`param` があれば `__<key>` が続く)。`fctype` を設定し忘れると `getattr(func, "fctype", None)` が `None` になり、`combiner` 用の呼び出し分岐に入らず `simple` 扱いされる(パラメータなし関数なら問題ないが、`param` を使う関数では `combiner` の指定が必須)。

#### `@set_property("fctype", "combiner")` によるパラメータ付き複数特徴量の自作

**用途**: 1回の計算で複数のパラメータ(例: 複数のパーセンタイル区間)に対応する複数の特徴量列をまとめて生成する自作関数を作る。`fft_coefficient` や `linear_trend` と同じ「combiner」型。

**シグネチャ**: `fctype="combiner"` の関数は `func(x, param)` の形で呼ばれ、`(名前文字列, 値)` のタプルを列挙する**ジェネレータ**を返す必要がある。

**使用例**:
```python
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction.feature_calculators import set_property

@set_property("fctype", "combiner")
def quantile_range(x, param):
    x = np.asarray(x)
    for p in param:
        lo, hi = np.percentile(x, p["low"]), np.percentile(x, p["high"])
        yield f'low_{p["low"]}__high_{p["high"]}', hi - lo

df = pd.DataFrame({
    "id": [1, 1, 1, 1, 2, 2, 2, 2],
    "time": [0, 1, 2, 3, 0, 1, 2, 3],
    "value": [1.0, 5.0, 2.0, 3.0, 10.0, 20.0, 10.0, 10.0],
})
fc_parameters = {quantile_range: [{"low": 10, "high": 90}]}
extracted = extract_features(
    df, column_id="id", column_sort="time",
    default_fc_parameters=fc_parameters, n_jobs=0, disable_progressbar=True,
)
print(extracted)
```
実行結果:
```
   value__quantile_range__low_10__high_90
1                                     3.1
2                                     7.0
```

**注意点・落とし穴**:
- `combiner` 型では `param`(この例では `[{"low": 10, "high": 90}]`)に含まれる各要素ごとに1つの `(名前, 値)` を `yield` する。複数要素を渡せば1回の関数呼び出しで複数列を一度に生成できる(`simple` 型は `param` の要素ごとに関数を毎回呼び直す点が異なる)。
- `simple` 型の関数に `param` 付きのリストを渡しても動くが、`combiner` 型の関数を `fctype` を `"simple"` のまま(デコレータを付け忘れた状態)で使うと `func(x, **param)` として呼ばれてしまい `TypeError`(`param` という名の引数がないため)になる。

---

### `extract_features` の高度なオプション

#### `impute_function`(抽出直後の欠損値処理)

**用途**: `extract_features` が返す前に、生成された特徴量DataFrameへ自動で欠損値補完関数を適用させる。`ar_coefficient` の `k` が系列長より大きい場合などに生じる `NaN` を、後段で別途 `impute()` を呼ばずにその場で解消できる。

**シグネチャ**: `extract_features(..., impute_function=None)`。`impute_function` には `impute(df)` のような「DataFrameを受け取りinplaceで書き換える(あるいは書き換えて返す)」関数を渡す。

**使用例**:
```python
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

df = pd.DataFrame({
    "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    "time": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    "value": [0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0],
})
fc = {"ar_coefficient": [{"coeff": 0, "k": 10}], "mean": None}

extracted_raw = extract_features(df, column_id="id", column_sort="time",
                                  default_fc_parameters=fc, n_jobs=0, disable_progressbar=True)
print("impute_functionなし:\n", extracted_raw)

extracted_imp = extract_features(df, column_id="id", column_sort="time", default_fc_parameters=fc,
                                  n_jobs=0, disable_progressbar=True, impute_function=impute)
print("impute_function=imputeあり:\n", extracted_imp)
```
実行結果:
```
impute_functionなし:
    value__ar_coefficient__coeff_0__k_10  value__mean
1                                    NaN          2.0
2                                    NaN          3.0
impute_function=imputeあり:
    value__ar_coefficient__coeff_0__k_10  value__mean
1                                    0.0          2.0
2                                    0.0          3.0
```

**注意点・落とし穴**:
- `ar_coefficient__k_10` は系列長5点に対し `k=10` のAR係数を要求しており計算不能なため常に `NaN` になる。`impute_function=impute` を渡すと、この `NaN` はその場で「列内の他の値の中央値」(ここでは他idの値も同じ列に1つしかないため中央値=最小値=最大値相当)に置き換えられる。`extract_relevant_features` はこのオプションと似た効果を内部の `impute` 呼び出しで実現している。

#### `chunksize`(並列化の粒度調整)

**用途**: マルチプロセス並列実行時に、1つのワーカープロセスへまとめて渡す「1id×1kindの時系列」の個数を制御する。大量のid/kindがある場合のプロセス間通信オーバーヘッドを調整するためのチューニング用パラメータ。

**シグネチャ**: `extract_features(..., chunksize=None)`。`None` の場合は distributor 側のヒューリスティックで自動決定される。

**使用例**:
```python
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

np.random.seed(0)
rows = []
for id_ in range(1, 7):
    for t in range(5):
        rows.append({"id": id_, "time": t, "value": float(t + id_)})
df = pd.DataFrame(rows)

extracted = extract_features(df, column_id="id", column_sort="time",
                              default_fc_parameters=MinimalFCParameters(),
                              n_jobs=0, disable_progressbar=True, chunksize=2)
print(extracted.shape)
print(extracted.index.tolist())
```
実行結果:
```
(6, 10)
[1, 2, 3, 4, 5, 6]
```

**注意点・落とし穴**:
- `chunksize` は計算結果そのものには影響しない(あくまで内部の並列化単位)。公式ドキュメント(docstring)によれば「1チャンク=1つのid・kindの時系列」を基準とし、`chunksize=10` なら1タスクで10系列分を計算する。メモリ不足が出る場合は値を小さくするとよい、とdocstringに明記されている。

#### `profile` / `profiling_filename`(内部プロファイリング)

**用途**: 特徴量抽出処理そのものを Python 標準の `cProfile` でプロファイリングし、関数ごとの呼び出し回数・所要時間をファイルへ出力する。どの特徴量計算関数がボトルネックかを調べたい場合に使う。

**シグネチャ**: `extract_features(..., profile=False, profiling_filename='profile.txt', profiling_sorting='cumulative')`

**使用例**:
```python
import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

df = pd.DataFrame({
    "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    "time": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    "value": [0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0],
})
profile_path = "/tmp/tsfresh_profile_example.txt"
extracted = extract_features(df, column_id="id", column_sort="time",
                              default_fc_parameters=MinimalFCParameters(),
                              n_jobs=0, disable_progressbar=True,
                              profile=True, profiling_filename=profile_path)
print("shape:", extracted.shape)
print("プロファイルファイル存在:", os.path.exists(profile_path))
with open(profile_path) as f:
    print(f.read().splitlines()[0])
os.remove(profile_path)
```
実行結果:
```
shape: (2, 10)
プロファイルファイル存在: True
         5010 function calls (4923 primitive calls) in 0.010 seconds
```

**注意点・落とし穴**:
- `profiling_filename` は実行時のカレントディレクトリからの相対パスでも絶対パスでも指定可能(この例では絶対パスを使用)。出力内容は `cProfile.Profile().dump_stats()` 相当ではなく `pstats.Stats` のテキストダンプ(`print_stats()` 出力)で、`profiling_sorting`(デフォルト `'cumulative'`)で並び替え基準を変更できる。

---

### 学習・推論を意識したユーティリティ

#### `get_range_values_per_column` / `impute_dataframe_range`(train/testで一貫した補完)

**用途**: 訓練データの各列の最大値・最小値・中央値を `get_range_values_per_column` で取得し、それをテストデータ側の `impute_dataframe_range` に渡すことで、「テストデータの補完に訓練データの統計量だけを使う」というデータリークを避けた補完が行える。

**シグネチャ**: `get_range_values_per_column(df)` は `(col_to_max, col_to_min, col_to_median)` の3つの `dict` を返す。`impute_dataframe_range(df_impute, col_to_max, col_to_min, col_to_median)` はその3つを受け取り `df_impute` をinplaceで補完する。

**使用例**:
```python
import numpy as np
import pandas as pd
from tsfresh.utilities.dataframe_functions import impute_dataframe_range, get_range_values_per_column

X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [10.0, np.nan, 30.0, 40.0]})
col_max, col_min, col_median = get_range_values_per_column(X_train)
print("max:", col_max)
print("min:", col_min)
print("median:", col_median)

X_test = pd.DataFrame({"a": [np.nan, 100.0, -100.0], "b": [np.inf, -np.inf, np.nan]})
impute_dataframe_range(X_test, col_max, col_min, col_median)
print(X_test)
```
実行結果:
```
max: {'a': np.float64(3.0), 'b': np.float64(40.0)}
min: {'a': np.float64(1.0), 'b': np.float64(10.0)}
median: {'a': np.float64(2.0), 'b': np.float64(30.0)}
       a     b
0    2.0  40.0
1  100.0  10.0
2 -100.0  30.0
```

**注意点・落とし穴**:
- `impute_dataframe_range` が置き換えるのは `NaN`(→中央値)・`+inf`(→最大値)・`-inf`(→最小値)の3種類のみで、範囲外の有限値(この例の `a` 列の `100.0`/`-100.0`)は**クリッピングされずそのまま残る**。「訓練データの範囲に収める」処理ではなく、あくまで欠損・無限大の補完専用である点に注意。
- 基底の `impute(df)` は列ごとに `df` 自身から範囲を計算する(訓練/テストを分けない)のに対し、この2関数の組み合わせは範囲の計算元(`X_train`)と補完対象(`X_test`)を分離できる点が異なる。

#### `add_sub_time_series_index`(固定長の非重複サブ系列への分割)

**用途**: 1つの時系列を、重なりのない固定長 `sub_length` のサブ系列に分割し、それぞれに新しい `id` を振り直す。`roll_time_series` が「累積的に伸びる重複ウィンドウ」を作るのに対し、こちらは「重ならない固定長の区間」に単純分割する。

**シグネチャ**: `add_sub_time_series_index(df_or_dict, sub_length, column_id=None, column_sort=None, column_kind=None)`

**使用例**:
```python
import pandas as pd
from tsfresh.utilities.dataframe_functions import add_sub_time_series_index

df = pd.DataFrame({
    "id": [1] * 4 + [2] * 4,
    "time": list(range(4)) * 2,
    "value": [10, 20, 30, 40, 100, 200, 300, 400],
})
result = add_sub_time_series_index(df, sub_length=2, column_id="id", column_sort="time")
print(result)
```
実行結果:
```
   time  value      id
0     0     10  (0, 1)
4     0    100  (0, 2)
5     1    200  (0, 2)
1     1     20  (0, 1)
6     2    300  (1, 2)
2     2     30  (1, 1)
3     3     40  (1, 1)
7     3    400  (1, 2)
```

**注意点・落とし穴**:
- 新しい `id` 列は `(サブ系列の連番, 元のid)` のタプルになる(`roll_time_series` の `(元のid, ウィンドウ末尾の時刻)` とは要素の順序も意味も異なるので混同しないこと)。
- 元の系列長が `sub_length` の倍数でない場合、端数分は切り捨てられずに短いサブ系列として残る(この関数自体には端数を捨てるオプションはなく、後段の `extract_features` 側で系列長依存の特徴量に `NaN` が出うる点は他のローリング系関数と同様)。
