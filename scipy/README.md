# scipy 逆引き辞書

scipy 1.18.1 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認)。

## 目次

1. [統計-検定](#統計-検定)
2. [統計-分布](#統計-分布)
3. [最適化(optimize)](#最適化optimize)
4. [補間(interpolate)](#補間interpolate)
5. [積分・微分方程式(integrate)](#積分微分方程式integrate)
6. [線形代数(linalg)](#線形代数linalg)
7. [疎行列(sparse)](#疎行列sparse)
8. [信号処理(signal)](#信号処理signal)
9. [空間データ・距離(spatial)](#空間データ距離spatial)
10. [クラスタリング(cluster)](#クラスタリングcluster)
11. [FFT(fft)](#fftfft)
12. [画像処理(ndimage)](#画像処理ndimage)
13. [その他(constants・special)](#その他constantsspecial)
14. [応用・発展](#応用発展)
    - [高度な最適化(optimize)](#高度な最適化optimize)
    - [高度な統計(stats)](#高度な統計stats)
    - [高度な疎行列(sparse)](#高度な疎行列sparse)
    - [常微分方程式の応用(integrate)](#常微分方程式の応用integrate)
    - [信号処理の応用(signal)](#信号処理の応用signal)
    - [特殊関数の深掘り(special)](#特殊関数の深掘りspecial)

---

## 統計-検定

### `scipy.stats.ttest_ind(a, b, ...)`

**用途**: 独立2標本のt検定(平均値に差があるかを検定)。

**シグネチャ**: `scipy.stats.ttest_ind(a, b, *, axis=0, equal_var=True, nan_policy='propagate', alternative='two-sided', trim=0, method=None, keepdims=False)`

**使用例**:
```python
from scipy import stats
a = [23, 21, 18, 30, 25, 27, 22]
b = [31, 28, 35, 29, 33, 30, 32]
print(stats.ttest_ind(a, b))
```
実行結果:
```
TtestResult(statistic=np.float64(-4.217756949399825), pvalue=np.float64(0.001193647787154938), df=np.float64(12.0))
```

**注意点・落とし穴**:
- 既定 `equal_var=True`(Student のt検定、等分散を仮定)。分散が等しいか分からない/異なる場合は `equal_var=False`(Welch のt検定)を指定する方が安全。
- 戻り値は `TtestResult`(名前付きタプル)。`.statistic`, `.pvalue`, `.df` で個別に取り出せる。

---

### `scipy.stats.ttest_rel(a, b, ...)`

**用途**: 対応のある(同一対象の前後比較など)2標本のt検定。

**シグネチャ**: `scipy.stats.ttest_rel(a, b, axis=0, nan_policy='propagate', alternative='two-sided', *, keepdims=False)`

**使用例**:
```python
from scipy import stats
before = [70, 72, 68, 75, 71]
after = [68, 71, 65, 74, 68]
print(stats.ttest_rel(before, after))
```
実行結果:
```
TtestResult(statistic=np.float64(4.47213595499958), pvalue=np.float64(0.011056493393450067), df=np.int64(4))
```

**注意点・落とし穴**:
- `a` と `b` は同じ長さ・同じ順序で対応している必要がある(対応がないデータには `ttest_ind` を使う)。

---

### `scipy.stats.mannwhitneyu(x, y, ...)`

**用途**: 独立2標本のノンパラメトリック検定(t検定の正規性を仮定しない版、Wilcoxon順位和検定の一種)。

**シグネチャ**: `scipy.stats.mannwhitneyu(x, y, use_continuity=True, alternative='two-sided', axis=0, method='auto', *, nan_policy='propagate', keepdims=False)`

**使用例**:
```python
from scipy import stats
a = [23, 21, 18, 30, 25, 27, 22]
b = [31, 28, 35, 29, 33, 30, 32]
print(stats.mannwhitneyu(a, b))
```
実行結果:
```
MannwhitneyuResult(statistic=np.float64(2.5), pvalue=np.float64(0.0059560157944908735))
```

**注意点・落とし穴**:
- 正規性が疑わしい/サンプル数が少ないデータで `ttest_ind` の代替として使う。順位に基づく検定のため外れ値の影響を受けにくい。

---

### `scipy.stats.wilcoxon(x, y=None, ...)`

**用途**: 対応のある2標本のノンパラメトリック検定(符号付き順位検定、`ttest_rel` のノンパラ版)。

**シグネチャ**: `scipy.stats.wilcoxon(x, y=None, zero_method='wilcox', correction=False, alternative='two-sided', method='auto', *, axis=0, nan_policy='propagate', keepdims=False)`

**使用例**:
```python
from scipy import stats
before = [70, 72, 68, 75, 71]
after = [68, 71, 65, 74, 68]
print(stats.wilcoxon(before, after))
```
実行結果:
```
WilcoxonResult(statistic=np.float64(0.0), pvalue=np.float64(0.0625))
```

**注意点・落とし穴**:
- サンプル数が少ないと(この例では n=5)有意水準0.05を下回りにくい。`method='auto'` は標本数に応じて厳密分布法か正規近似かを自動選択する。

---

### `scipy.stats.chi2_contingency(observed, ...)`

**用途**: クロス集計表(分割表)に対するカイ二乗独立性検定。

**シグネチャ**: `scipy.stats.chi2_contingency(observed, correction=True, lambda_=None, *, method=None)`

**使用例**:
```python
from scipy import stats
import numpy as np
table = np.array([[10, 20], [30, 15]])
print(stats.chi2_contingency(table))
```
実行結果:
```
Chi2ContingencyResult(statistic=np.float64(6.752232142857143), pvalue=np.float64(0.00936304763753395), dof=1, expected_freq=array([[16., 14.],
       [24., 21.]]))
```

**注意点・落とし穴**:
- 既定 `correction=True` で 2x2 表には Yates の連続性補正が自動的にかかる(この例で `correction=False` にすると統計量は `6.75` ではなく `8.04` になる)。2x2 以外の表には補正は適用されない。
- 戻り値の `expected_freq` は「独立だった場合に期待される度数」。観測値とのズレが大きいほど統計量が大きくなる。

---

### `scipy.stats.pearsonr(x, y, ...)` / `scipy.stats.spearmanr(a, b=None, ...)`

**用途**: `pearsonr` は線形関係の強さ(ピアソン相関係数)、`spearmanr` は単調関係の強さ(順位相関係数)を検定付きで計算する。

**シグネチャ**: `scipy.stats.pearsonr(x, y, *, alternative='two-sided', method=None, axis=0)` / `scipy.stats.spearmanr(a, b=None, axis=0, nan_policy='propagate', alternative='two-sided')`

**使用例**:
```python
from scipy import stats
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
print(stats.pearsonr(x, y))
print(stats.spearmanr(x, y))
```
実行結果:
```
PearsonRResult(statistic=np.float64(0.7745966692414835), pvalue=np.float64(0.1240270626575546))
SignificanceResult(statistic=np.float64(0.7378647873726218), pvalue=np.float64(0.15461852312844926))
```

**注意点・落とし穴**:
- `numpy.corrcoef` と違い、`pearsonr`/`spearmanr` は相関係数だけでなく検定の p値も同時に返す(`.statistic` が相関係数に相当)。
- 外れ値や非線形だが単調な関係には `spearmanr` の方が頑健。

---

### `scipy.stats.shapiro(x, ...)`

**用途**: シャピロ–ウィルク検定によるデータの正規性検定。

**シグネチャ**: `scipy.stats.shapiro(x, *, axis=None, nan_policy='propagate', keepdims=False)`

**使用例**:
```python
from scipy import stats
import numpy as np
rng = np.random.default_rng(0)
sample = rng.normal(size=30)
print(stats.shapiro(sample))
```
実行結果:
```
ShapiroResult(statistic=np.float64(0.9751758954926929), pvalue=np.float64(0.6879131893722795))
```

**注意点・落とし穴**:
- 帰無仮説は「データは正規分布に従う」。p値が小さい(例: <0.05)ときに正規性を棄却する。サンプル数が多いと僅かなズレでも棄却されやすくなる点に注意。

---

### `scipy.stats.kstest(rvs, cdf, ...)`

**用途**: コルモゴロフ–スミルノフ検定。標本が指定した理論分布に従うか、または2標本が同じ分布から来ているかを検定する。

**シグネチャ**: `scipy.stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', method='auto', *, axis=0, nan_policy='propagate')`

**使用例**:
```python
from scipy import stats
import numpy as np
rng = np.random.default_rng(0)
sample = rng.normal(size=30)
print(stats.kstest(sample, 'norm'))
```
実行結果:
```
KstestResult(statistic=np.float64(0.1403051230551312), pvalue=np.float64(0.549309647706063), statistic_location=np.float64(0.4116305363741328), statistic_sign=np.int8(1))
```

**注意点・落とし穴**:
- `cdf` に文字列(`'norm'` など)を渡すと標準正規分布(平均0, 標準偏差1)と比較される。標本自体の平均・標準偏差と比較したい場合は `args=(mean, std)` を指定するか、事前に標準化する必要がある。

---

### `scipy.stats.f_oneway(*samples, ...)`

**用途**: 3群以上の一元配置分散分析(ANOVA)。全群の平均が等しいかを検定する。

**シグネチャ**: `scipy.stats.f_oneway(*samples, axis=0, equal_var=True, nan_policy='propagate', keepdims=False)`

**使用例**:
```python
from scipy import stats
g1 = [1, 2, 3]
g2 = [2, 3, 4]
g3 = [5, 6, 7]
print(stats.f_oneway(g1, g2, g3))
```
実行結果:
```
F_onewayResult(statistic=np.float64(13.0), pvalue=np.float64(0.006591796875))
```

**注意点・落とし穴**:
- 有意差があると分かっても「どの群とどの群に差があるか」は分からない。事後検定(Tukey HSD: `scipy.stats.tukey_hsd`)が別途必要。

---

### `scipy.stats.zscore(a, ...)`

**用途**: 配列を標準化(平均0・標準偏差1)する。

**シグネチャ**: `scipy.stats.zscore(a, axis=0, ddof=0, nan_policy='propagate')`

**使用例**:
```python
from scipy import stats
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(stats.zscore(arr))
```
実行結果:
```
[-1.41421356 -0.70710678  0.          0.70710678  1.41421356]
```

**注意点・落とし穴**:
- 既定 `ddof=0`(母標準偏差で割る)。標本標準偏差で標準化したい場合は `ddof=1` を指定する。

---

## 統計-分布

### `scipy.stats.norm` の `pdf` / `cdf` / `ppf` / `rvs`

**用途**: 正規分布に関する確率密度・累積分布・分位点(累積分布の逆関数)・乱数生成。すべての連続分布(`uniform`, `expon` など)が同じメソッド名の共通APIを持つ。

**シグネチャ**: `scipy.stats.norm.pdf(x, *args, **kwds)` / `.cdf(x, *args, **kwds)` / `.ppf(q, *args, **kwds)` / `.rvs(*args, **kwds)`(共通で `loc`, `scale` キーワードを取る)

**使用例**:
```python
from scipy import stats
print(stats.norm.pdf(0))                 # x=0 での密度(標準正規)
print(stats.norm.cdf(1.96))               # P(X <= 1.96)
print(stats.norm.ppf(0.975))              # 累積確率0.975に対応するx
print(stats.norm.rvs(loc=0, scale=1, size=5, random_state=42))
```
実行結果:
```
0.3989422804014327
0.9750021048517795
1.959963984540054
[ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]
```

**注意点・落とし穴**:
- `inspect.signature` では `(x, *args, **kwds)` としか表示されないが、実際は `loc`(平均に相当する位置引数)、`scale`(標準偏差に相当する尺度引数)をキーワードで渡せる。
- 再現性が必要な場合は `rvs` に `random_state` を必ず指定する(省略するとグローバル乱数状態に依存し実行毎に変わる)。
- 特定のパラメータに固定した「凍結(frozen)分布」を作りたい場合は `dist = stats.norm(loc=5, scale=2)` のようにインスタンス化すると、以降 `dist.pdf(x)` のように `loc`/`scale` 省略で使える。

---

### `scipy.stats.binom` / `scipy.stats.poisson` の `pmf` / `cdf`

**用途**: 離散分布(二項分布・ポアソン分布)の確率質量関数・累積分布関数。

**シグネチャ**: `scipy.stats.binom.pmf(k, *args, **kwds)`(`n`, `p` をキーワードで指定) / `scipy.stats.poisson.pmf(k, *args, **kwds)`(`mu` をキーワードで指定)

**使用例**:
```python
from scipy import stats
print(stats.binom.pmf(3, n=10, p=0.5))    # 10回中3回成功する確率
print(stats.binom.cdf(3, n=10, p=0.5))    # 3回以下になる確率
print(stats.poisson.pmf(2, mu=3))         # 平均3のポアソン分布でk=2となる確率
```
実行結果:
```
0.1171875
0.171875
0.22404180765538775
```

**注意点・落とし穴**:
- 連続分布は `pdf`(確率密度)、離散分布は `pmf`(確率質量)とメソッド名が異なる点に注意(`cdf`/`ppf`/`rvs` は共通)。

---

### `scipy.stats.describe(a, ...)`

**用途**: 配列の要約統計量(件数・範囲・平均・分散・歪度・尖度)をまとめて計算する。

**シグネチャ**: `scipy.stats.describe(a, axis=0, ddof=1, bias=True, nan_policy='propagate')`

**使用例**:
```python
from scipy import stats
import numpy as np
data = np.array([1,2,3,4,5,6,7,8,9,10])
print(stats.describe(data))
```
実行結果:
```
DescribeResult(nobs=np.int64(10), minmax=(np.int64(1), np.int64(10)), mean=np.float64(5.5), variance=np.float64(9.166666666666668), skewness=np.float64(0.0), kurtosis=np.float64(-1.2242424242424244))
```

**注意点・落とし穴**:
- 既定 `ddof=1`(不偏分散/標本分散)。`numpy.var` の既定 `ddof=0`(母分散)とは異なるため、`numpy` の集計結果と単純比較すると値がずれる。

---

### `scipy.stats.gaussian_kde(dataset, ...)`

**用途**: カーネル密度推定(KDE)によりデータからなめらかな確率密度関数を推定する。

**シグネチャ**: `scipy.stats.gaussian_kde(self, dataset, bw_method=None, weights=None)`

**使用例**:
```python
from scipy import stats
import numpy as np
data = np.array([1,2,3,4,5,6,7,8,9,10])
kde = stats.gaussian_kde(data)
print(kde.evaluate([5.0, 5.5]))
```
実行結果:
```
[0.09896147 0.09918906]
```

**注意点・落とし穴**:
- `bw_method` でバンド幅(平滑化の強さ)を調整できる(省略時は Scott のルールで自動決定)。バンド幅が広すぎると分布の形が潰れ、狭すぎると過学習的にギザギザになる。

---

## 最適化(optimize)

### `scipy.optimize.minimize(fun, x0, ...)`

**用途**: 多変数のスカラー関数を最小化する汎用最適化関数(勾配法・準ニュートン法など多数のアルゴリズムを統一APIで扱う)。

**シグネチャ**: `scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)`

**使用例**:
```python
from scipy import optimize

def f(x):
    return (x[0]-3)**2 + (x[1]+1)**2

res = optimize.minimize(f, x0=[0, 0])
print(res.x, res.fun, res.success)
```
実行結果:
```
[ 3.00000004 -1.00000007] 7.036731787282697e-15 True
```

**注意点・落とし穴**:
- `method` を省略すると、制約や境界の有無に応じて `BFGS`/`L-BFGS-B`/`SLSQP` などが自動選択される。アルゴリズムによって `jac`(勾配)や `bounds`/`constraints` のサポート状況が異なる。
- 結果は厳密な最適値ではなく数値誤差を含む近似値(`res.success` で収束したかを必ず確認する)。

---

### `scipy.optimize.minimize_scalar(fun, ...)`

**用途**: 1変数のスカラー関数を最小化する。

**シグネチャ**: `scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(), method=None, tol=None, options=None)`

**使用例**:
```python
from scipy import optimize
res = optimize.minimize_scalar(lambda x: (x-2)**2)
print(res.x, res.fun)
```
実行結果:
```
1.9999999999999998 4.930380657631324e-32
```

---

### `scipy.optimize.curve_fit(f, xdata, ydata, ...)`

**用途**: 与えたモデル関数のパラメータを、観測データに最小二乗フィットさせる。

**シグネチャ**: `scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-inf, inf), method=None, jac=None, *, full_output=False, nan_policy=None, **kwargs)`

**使用例**:
```python
from scipy import optimize
import numpy as np

def model(x, a, b):
    return a * np.exp(-b * x)

xdata = np.linspace(0, 4, 10)
rng = np.random.default_rng(0)
ydata = model(xdata, 2.5, 1.3) + rng.normal(0, 0.02, size=xdata.size)
popt, pcov = optimize.curve_fit(model, xdata, ydata)
print(popt)
```
実行結果:
```
[2.50023433 1.29290738]
```

**注意点・落とし穴**:
- 戻り値の2番目 `pcov` はパラメータ推定値の共分散行列。標準誤差は `np.sqrt(np.diag(pcov))` で求める。
- 初期値 `p0` を省略すると全パラメータ `1.0` から開始する。非線形性が強いモデルでは初期値次第で収束しない/局所解に陥ることがある。

---

### `scipy.optimize.root(fun, x0, ...)`

**用途**: 非線形連立方程式 `fun(x) = 0` の解を求める。

**シグネチャ**: `scipy.optimize.root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None)`

**使用例**:
```python
from scipy import optimize

def g(x):
    return [x[0] + 2*x[1] - 3, x[0]**2 - x[1]]

r = optimize.root(g, x0=[1, 1])
print(r.x, r.success)
```
実行結果:
```
[1. 1.] True
```

**注意点・落とし穴**:
- 既定 `method='hybr'`(修正Powellハイブリッド法)。初期値 `x0` が解から遠いと収束しないことがあるため、`r.success` の確認が必須。

---

### `scipy.optimize.brentq(f, a, b, ...)`

**用途**: 1変数関数の根(f(x)=0となるx)をBrent法で求める(区間 `[a, b]` で符号が反転していることが前提)。

**シグネチャ**: `scipy.optimize.brentq(f, a, b, args=(), xtol=2e-12, rtol=np.float64(8.881784197001252e-16), maxiter=100, full_output=False, disp=True)`

**使用例**:
```python
from scipy import optimize
print(optimize.brentq(lambda x: x**3 - x - 2, 1, 2))
```
実行結果:
```
1.5213797068045676
```

**注意点・落とし穴**:
- `f(a)` と `f(b)` が同符号だと `ValueError: f(a) and f(b) must have different signs` になる。事前に符号が変わる区間を見つけておく必要がある。

---

### `scipy.optimize.linprog(c, ...)`

**用途**: 線形計画問題(線形の目的関数・制約のもとでの最小化)を解く。

**シグネチャ**: `scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None), method='highs', callback=None, options=None, x0=None, integrality=None)`

**使用例**:
```python
from scipy import optimize
c = [-1, -2]                      # 目的関数 -(x + 2y) を最小化 = x + 2y を最大化
A_ub = [[1, 1], [2, 1]]
b_ub = [4, 5]
res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)])
print(res.x, res.fun, res.success)
```
実行結果:
```
[0. 4.] -8.0 True
```

**注意点・落とし穴**:
- `linprog` は常に「最小化」を行う。最大化したい場合は目的関数の係数 `c` の符号を反転させる(この例では `x + 2y` の最大化を `-(x+2y)` の最小化として解いている)。
- 既定 `bounds=(0, None)` で全変数は非負が前提。負の値も許す変数には明示的に `bounds` を指定する。

---

## 補間(interpolate)

### `scipy.interpolate.interp1d(x, y, ...)`

**用途**: 1次元データ点から補間関数を作る(線形・多項式・スプラインなど)。

**シグネチャ**: `scipy.interpolate.interp1d(self, x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=nan, assume_sorted=False)`

**使用例**:
```python
from scipy import interpolate
import numpy as np
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])
f = interpolate.interp1d(x, y)
print(f(2.5))
f_cubic = interpolate.interp1d(x, y, kind='cubic')
print(f_cubic(2.5))
```
実行結果:
```
6.5
6.250000000000001
```

**注意点・落とし穴**:
- 既定 `kind='linear'`(区分線形補間)。`'cubic'`, `'quadratic'`, `'nearest'` なども指定できる。
- SciPy 公式ドキュメントでは新規コードには `make_interp_spline` や `CubicSpline` の使用が推奨されており、`interp1d` はレガシー的な位置づけ(ただし本バージョンでは廃止警告は出ない)。
- 範囲外の `x` を渡すと既定 `bounds_error=None`(実質 True 相当)で `ValueError` になる。範囲外を許容したい場合は `fill_value='extrapolate'` などを指定する。

---

### `scipy.interpolate.CubicSpline(x, y, ...)`

**用途**: 3次スプライン補間(区間ごとに3次多項式をなめらかに接続)。

**シグネチャ**: `scipy.interpolate.CubicSpline(self, x, y, axis=0, bc_type='not-a-knot', extrapolate=None)`

**使用例**:
```python
from scipy import interpolate
import numpy as np
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])
cs = interpolate.CubicSpline(x, y)
print(cs(2.5))
```
実行結果:
```
6.25
```

**注意点・落とし穴**:
- `cs(x, 1)` のように第2引数で微分階数を指定すると導関数の値も直接評価できる。
- `bc_type`(境界条件)の既定は `'not-a-knot'`。周期関数なら `'periodic'` を指定する。

---

### `scipy.interpolate.griddata(points, values, xi, ...)`

**用途**: 不規則(散布)な多次元データ点から、任意の座標での値を補間する。

**シグネチャ**: `scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan, rescale=False, simplex_tolerance=1.0)`

**使用例**:
```python
from scipy import interpolate
import numpy as np
points = np.array([[0,0],[1,0],[0,1],[1,1]])
values = np.array([0,1,1,2])
print(interpolate.griddata(points, values, [(0.5, 0.5)], method='linear'))
```
実行結果:
```
[1.]
```

**注意点・落とし穴**:
- `method='linear'`/`'cubic'` は補間点の凸包(convex hull)の外側では `fill_value`(既定 `nan`)を返す。`'nearest'` は外挿しても最も近い点の値を返す。

---

## 積分・微分方程式(integrate)

### `scipy.integrate.quad(func, a, b, ...)`

**用途**: 1変数関数の定積分を数値的に計算する(適応的な求積法)。

**シグネチャ**: `scipy.integrate.quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False)`

**使用例**:
```python
from scipy import integrate
import numpy as np
res, err = integrate.quad(lambda x: np.exp(-x**2), 0, np.inf)
print(res, err)
```
実行結果:
```
0.8862269254527579 7.10131839047246e-09
```

**注意点・落とし穴**:
- 戻り値はタプル `(積分値, 推定誤差)` の2つ。積分値だけを使いたい場合は `[0]` を明示的に取り出す。
- `np.inf` を積分区間に含めても正しく処理できる(変数変換して数値的に扱う)。

---

### `scipy.integrate.dblquad(func, a, b, gfun, hfun, ...)`

**用途**: 2重積分を数値的に計算する。

**シグネチャ**: `scipy.integrate.dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)`

**使用例**:
```python
from scipy import integrate
res, err = integrate.dblquad(lambda y, x: x * y, 0, 1, 0, 1)
print(res, err)
```
実行結果:
```
0.24999999999999997 5.539061329123429e-15
```

**注意点・落とし穴**:
- `func(y, x)` のように引数の順序が「内側の変数(y)が先」になる点に注意(`x` の積分区間が `a, b`、`y` の積分区間が `gfun, hfun`)。直感的な `(x, y)` の順とは逆。

---

### `scipy.integrate.solve_ivp(fun, t_span, y0, ...)`

**用途**: 常微分方程式の初期値問題を数値的に解く(SciPy 推奨の現行API)。

**シグネチャ**: `scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)`

**使用例**:
```python
from scipy import integrate

def decay(t, y):
    return -0.5 * y

sol = integrate.solve_ivp(decay, [0, 4], [10], t_eval=[0, 1, 2, 3, 4])
print(sol.t)
print(sol.y)
```
実行結果:
```
[0 1 2 3 4]
[[10.          6.06526852  3.67670143  2.2325709   1.35415994]]
```

**注意点・落とし穴**:
- `fun(t, y)` の引数順序は `(t, y)`(時刻が先)。古い `scipy.integrate.odeint` は逆の `(y, t)` なので混同注意(`odeint` は現在レガシー扱いで、新規コードには `solve_ivp` が推奨される)。
- `sol.y` は `(状態変数の数, 時刻の数)` の2次元配列で返る(1変数でも2次元になる)。

---

## 線形代数(linalg)

### `scipy.linalg.inv(a, ...)`

**用途**: 正方行列の逆行列を計算する(`numpy.linalg.inv` とほぼ同じだがLAPACKを直接呼ぶ点や追加オプションが異なる)。

**シグネチャ**: `scipy.linalg.inv(a, overwrite_a=False, check_finite=True, *, assume_a=None, lower=False)`

**使用例**:
```python
from scipy import linalg
import numpy as np
A = np.array([[1., 2.], [3., 4.]])
print(linalg.inv(A))
```
実行結果:
```
[[-2.   1. ]
 [ 1.5 -0.5]]
```

**注意点・落とし穴**:
- `assume_a`(`'sym'`, `'pos'` など)で行列の性質を事前指定すると、より高速・安定なLAPACKルーチンが選ばれる。

---

### `scipy.linalg.solve(a, b, ...)`

**用途**: 連立一次方程式 `a @ x = b` を解く。

**シグネチャ**: `scipy.linalg.solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None, transposed=False)`

**使用例**:
```python
from scipy import linalg
import numpy as np
A = np.array([[1., 2.], [3., 4.]])
b = np.array([5., 6.])
print(linalg.solve(A, b))
```
実行結果:
```
[-4.   4.5]
```

**注意点・落とし穴**:
- `numpy.linalg.solve` より引数が豊富(`assume_a='sym'`/`'pos'` で対称・正定値行列専用の高速解法を選べる)。逆行列を経由するより数値的に安定。

---

### `scipy.linalg.det(a, ...)`

**用途**: 正方行列の行列式を計算する。

**シグネチャ**: `scipy.linalg.det(a, overwrite_a=False, check_finite=True)`

**使用例**:
```python
from scipy import linalg
import numpy as np
A = np.array([[1., 2.], [3., 4.]])
print(linalg.det(A))
```
実行結果:
```
-2.0
```

---

### `scipy.linalg.eig(a, ...)`

**用途**: 正方行列の固有値・固有ベクトルを計算する(非対称行列にも対応し、複素数の固有値を返しうる)。

**シグネチャ**: `scipy.linalg.eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)`

**使用例**:
```python
from scipy import linalg
import numpy as np
A = np.array([[1., 2.], [3., 4.]])
w, v = linalg.eig(A)
print(w)
print(v)
```
実行結果:
```
[-0.37228132+0.j  5.37228132+0.j]
[[-0.82456484 -0.41597356]
 [ 0.56576746 -0.90937671]]
```

**注意点・落とし穴**:
- 戻り値の固有値は既定で複素数型(`+0.j`)になる(実数のみでも虚部0の複素数として返る)。
- `v` の列ベクトル `v[:, i]` が固有値 `w[i]` に対応する固有ベクトル。対称行列なら数値的に安定な `scipy.linalg.eigh` の方が適している。

---

### `scipy.linalg.svd(a, ...)`

**用途**: 特異値分解(SVD)。任意の行列を `U @ diag(s) @ Vt` に分解する。

**シグネチャ**: `scipy.linalg.svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')`

**使用例**:
```python
from scipy import linalg
import numpy as np
A = np.array([[1., 2.], [3., 4.]])
U, s, Vt = linalg.svd(A)
print(s)
```
実行結果:
```
[5.4649857  0.36596619]
```

**注意点・落とし穴**:
- `s` は特異値のみを1次元配列で返す(対角行列そのものではない)。対角行列が必要な場合は `scipy.linalg.diagsvd` で組み立てる。

---

### `scipy.linalg.lstsq(a, b, ...)`

**用途**: 最小二乗法で連立方程式(過剰決定系を含む)の近似解を求める。

**シグネチャ**: `scipy.linalg.lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None)`

**使用例**:
```python
from scipy import linalg
import numpy as np
A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([6, 0, 0])
result = linalg.lstsq(A, b)
print(result[0])
```
実行結果:
```
[ 8. -3.]
```

**注意点・落とし穴**:
- 戻り値はタプル `(解, 残差平方和, 行列のランク, 特異値)`。解だけが欲しい場合は `[0]` を取り出す。

---

### `scipy.linalg.norm(a, ...)`

**用途**: ベクトル・行列のノルムを計算する。

**シグネチャ**: `scipy.linalg.norm(a, ord=None, axis=None, keepdims=False, check_finite=True)`

**使用例**:
```python
from scipy import linalg
import numpy as np
v = np.array([3., 4.])
print(linalg.norm(v))
```
実行結果:
```
5.0
```

---

## 疎行列(sparse)

### `scipy.sparse.csr_matrix(arg1, ...)`

**用途**: 疎行列(ほとんどが0の行列)をCSR(圧縮行格納)形式で表現する。行方向のスライスや行列積が高速。

**シグネチャ**: `scipy.sparse.csr_matrix(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None)`

**使用例**:
```python
from scipy import sparse
import numpy as np
dense = np.array([[0,0,3],[4,0,0],[0,5,0]])
csr = sparse.csr_matrix(dense)
print(csr)
print(csr.toarray())
```
実行結果:
```
<Compressed Sparse Row sparse matrix of dtype 'int64'
	with 3 stored elements and shape (3, 3)>
  Coords	Values
  (0, 2)	3
  (1, 0)	4
  (2, 1)	5
[[0 0 3]
 [4 0 0]
 [0 5 0]]
```

**注意点・落とし穴**:
- 密行列に戻すには `.toarray()`(または `.todense()`)を使う。大きな疎行列で誤って密行列化するとメモリを大量消費するので注意。

---

### `scipy.sparse.coo_matrix((data, (row, col)), shape=...)`

**用途**: `(値, (行インデックス, 列インデックス))` の3つ組から疎行列を組み立てる(COO形式)。データ構築に向く。

**シグネチャ**: `scipy.sparse.coo_matrix(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None)`

**使用例**:
```python
from scipy import sparse
import numpy as np
row = np.array([0, 1, 2])
col = np.array([2, 0, 1])
data = np.array([3, 4, 5])
coo = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
print(coo.toarray())
```
実行結果:
```
[[0 0 3]
 [4 0 0]
 [0 5 0]]
```

**注意点・落とし穴**:
- COO形式は要素アクセスやスライスが遅い。計算に使う前に `.tocsr()` や `.tocsc()` に変換するのが一般的。

---

### `scipy.sparse.linalg.spsolve(A, b, ...)`

**用途**: 疎行列の連立一次方程式 `A @ x = b` を解く。

**シグネチャ**: `scipy.sparse.linalg.spsolve(A, b, permc_spec=None, use_umfpack=True)`

**使用例**:
```python
from scipy import sparse
from scipy.sparse import linalg as splinalg
import numpy as np
A = sparse.csr_matrix(np.array([[3., 1.], [1., 2.]]))
b = np.array([9., 8.])
print(splinalg.spsolve(A, b))
```
実行結果:
```
[2. 3.]
```

**注意点・落とし穴**:
- `A` は疎行列型(`csr_matrix` など)である必要がある。密行列を渡すと非効率、または型エラーになりうる。

---

## 信号処理(signal)

### `scipy.signal.find_peaks(x, ...)`

**用途**: 1次元配列からピーク(極大点)のインデックスを検出する。

**シグネチャ**: `scipy.signal.find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)`

**使用例**:
```python
from scipy import signal
import numpy as np
x = np.array([0, 1, 3, 1, 0, 2, 5, 2, 0, 1, 4, 1, 0])
peaks, props = signal.find_peaks(x, height=2)
print(peaks)
print(props)
```
実行結果:
```
[ 2  6 10]
{'peak_heights': array([3., 5., 4.])}
```

**注意点・落とし穴**:
- `height`/`distance`/`prominence` などの条件を指定しないと、ノイズによる微小な凹凸まで全てピークとして検出されてしまう。実データでは `prominence` や `distance` の指定が重要。

---

### `scipy.signal.convolve(in1, in2, ...)` / `scipy.signal.correlate(in1, in2, ...)`

**用途**: 畳み込み(convolve)・相互相関(correlate)を計算する。

**シグネチャ**: `scipy.signal.convolve(in1, in2, mode='full', method='auto')` / `scipy.signal.correlate(in1, in2, mode='full', method='auto')`

**使用例**:
```python
from scipy import signal
import numpy as np
a = np.array([1, 2, 3])
k = np.array([0, 1, 0.5])
print(signal.convolve(a, k, mode='full'))
print(signal.convolve(a, k, mode='same'))
print(signal.correlate([1, 2, 3, 4], [1, 2], mode='valid'))
```
実行結果:
```
[0.  1.  2.5 4.  1.5]
[1.  2.5 4. ]
[ 5  8 11]
```

**注意点・落とし穴**:
- 既定 `mode='full'` は出力長が `len(a)+len(k)-1` になる。元の配列と同じ長さにしたい場合は `mode='same'` を使う。
- `convolve` はカーネルを反転してから積をとる(数学的な畳み込み)のに対し、`correlate` は反転しない。カーネルが対称でない場合は結果が異なる。

---

### `scipy.signal.butter(N, Wn, ...)` + `scipy.signal.filtfilt(b, a, x, ...)`

**用途**: `butter` でバターワースフィルタの係数を設計し、`filtfilt` で位相遅れなしのゼロ位相フィルタリングを行う(ノイズ除去の定番の組み合わせ)。

**シグネチャ**: `scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)` / `scipy.signal.filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)`

**使用例**:
```python
from scipy import signal
import numpy as np
b, a = signal.butter(4, 0.2)   # 4次のローパスフィルタ、正規化カットオフ0.2
t = np.linspace(0, 1, 20)
noisy = np.sin(2*np.pi*2*t) + 0.3
filtered = signal.filtfilt(b, a, noisy)
print(np.round(filtered, 4))
```
実行結果:
```
[ 0.3025  0.5461  0.6856  0.6631  0.4886  0.2354  0.01   -0.0933 -0.0311
  0.1702  0.426   0.629   0.6942  0.5944  0.3715  0.1186 -0.0591 -0.0883
  0.0421  0.2774]
```

**注意点・落とし穴**:
- `Wn` は既定で「ナイキスト周波数(サンプリング周波数の半分)を1に正規化した」カットオフ周波数(`fs` を指定すれば実周波数で指定可能)。
- `filtfilt` は信号を前後2回通すため、通常のフィルタリング(`lfilter`)と違って位相のズレが生じない代わりに、実質のフィルタ次数は2倍になる。

---

### `scipy.signal.detrend(data, ...)`

**用途**: データからトレンド(線形または定数のバイアス)を除去する。

**シグネチャ**: `scipy.signal.detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False)`

**使用例**:
```python
from scipy import signal
import numpy as np
y = np.array([1., 2., 3., 4., 5.]) + np.array([0.1, -0.2, 0.05, 0.0, -0.1])
print(signal.detrend(y))
```
実行結果:
```
[ 0.09 -0.19  0.08  0.05 -0.03]
```

**注意点・落とし穴**:
- 既定 `type='linear'`(最小二乗の直線を引いて除去)。単に平均を引きたいだけなら `type='constant'` を指定する。

---

### `scipy.signal.resample(x, num, ...)`

**用途**: フーリエ変換ベースで信号のサンプル数を変更する(リサンプリング)。

**シグネチャ**: `scipy.signal.resample(x, num, t=None, axis=0, window=None, domain='time')`

**使用例**:
```python
from scipy import signal
import numpy as np
sig_in = np.sin(np.linspace(0, 2*np.pi, 10, endpoint=False))
res = signal.resample(sig_in, 5)
print(np.round(res, 4))
```
実行結果:
```
[-0.      0.9511  0.5878 -0.5878 -0.9511]
```

**注意点・落とし穴**:
- FFTベースのため、信号が周期的でない(端点が滑らかに繋がらない)場合は端の歪み(リンギング)が出やすい。単純な時間軸の間引き/補間には `scipy.signal.resample_poly` の方が適することがある。

---

## 空間データ・距離(spatial)

### `scipy.spatial.distance.euclidean(u, v, ...)`

**用途**: 2点間のユークリッド距離を計算する。

**シグネチャ**: `scipy.spatial.distance.euclidean(u, v, w=None)`

**使用例**:
```python
from scipy.spatial import distance
print(distance.euclidean([0, 0], [3, 4]))
```
実行結果:
```
5.0
```

---

### `scipy.spatial.distance.cdist(XA, XB, ...)` / `scipy.spatial.distance.pdist(X, ...)`

**用途**: `cdist` は2集合間の全点対距離行列、`pdist` は1集合内の全点対距離(圧縮形式)を計算する。

**シグネチャ**: `scipy.spatial.distance.cdist(XA, XB, metric='euclidean', *, out=None, **kwargs)` / `scipy.spatial.distance.pdist(X, metric='euclidean', *, out=None, **kwargs)`

**使用例**:
```python
from scipy.spatial import distance
import numpy as np
A = np.array([[0, 0], [1, 1]])
B = np.array([[1, 0], [0, 1], [2, 2]])
print(distance.cdist(A, B))

pts = np.array([[0, 0], [1, 0], [0, 1]])
print(distance.pdist(pts))
```
実行結果:
```
[[1.         1.         2.82842712]
 [1.         1.         1.41421356]]
[1.         1.         1.41421356]
```

**注意点・落とし穴**:
- `pdist` の戻り値は正方距離行列ではなく上三角部分だけを1次元に並べた「圧縮形式」。正方行列に戻すには `scipy.spatial.distance.squareform` を使う。
- `metric` 引数で `'cosine'`, `'cityblock'`(マンハッタン距離)など多数の距離を切り替えられる。

---

### `scipy.spatial.KDTree(data, ...)`

**用途**: 最近傍探索を高速に行うための空間分割木(k-d木)を構築する。

**シグネチャ**: `scipy.spatial.KDTree(self, data, leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)` / `.query(self, x, k=1, eps=0.0, p=2.0, distance_upper_bound=inf, workers=1)`

**使用例**:
```python
from scipy.spatial import KDTree
import numpy as np
pts = np.array([[0, 0], [1, 0], [0, 1]])
tree = KDTree(pts)
d, idx = tree.query([0.1, 0.1])
print(d, idx)
```
実行結果:
```
0.14142135623730953 0
```

**注意点・落とし穴**:
- 総当たり(`cdist` で全距離を計算)は点数が多いと `O(n^2)` になるが、`KDTree` は構築後の1回のクエリを平均 `O(log n)` にできる。点数が多く何度もクエリする場合に有効。

---

### `scipy.spatial.ConvexHull(points, ...)`

**用途**: 多次元点群の凸包(すべての点を含む最小の凸多角形/多面体)を計算する。

**シグネチャ**: `scipy.spatial.ConvexHull(self, points, incremental=False, qhull_options=None)`

**使用例**:
```python
from scipy.spatial import ConvexHull
import numpy as np
pts = np.array([[0,0], [1,0], [1,1], [0,1], [0.5,0.5]])
hull = ConvexHull(pts)
print(hull.vertices)
print(hull.area)
```
実行結果:
```
[0 1 2 3]
4.0
```

**注意点・落とし穴**:
- `hull.vertices` は凸包を構成する点の「元の配列内でのインデックス」(この例では内部の点 `[0.5,0.5]`(インデックス4)は凸包を構成しないため含まれない)。
- 2次元では `hull.area` が周長、`hull.volume` が面積を表す(次元によって意味が変わる)。

---

## クラスタリング(cluster)

### `scipy.cluster.vq.kmeans(obs, k_or_guess, ...)` / `scipy.cluster.vq.vq(obs, code_book, ...)`

**用途**: `kmeans` でk-means法によりクラスタ中心(セントロイド)を求め、`vq` で各データ点を最も近いクラスタに割り当てる。

**シグネチャ**: `scipy.cluster.vq.kmeans(obs, k_or_guess, iter=20, thresh=1e-05, check_finite=True, *, rng=None, seed=None)` / `scipy.cluster.vq.vq(obs, code_book, check_finite=True)`

**使用例**:
```python
from scipy.cluster import vq
import numpy as np
rng = np.random.default_rng(0)
data = np.vstack([rng.normal(0, 0.3, size=(20, 2)), rng.normal(5, 0.3, size=(20, 2))])
centroids, distortion = vq.kmeans(data, 2, seed=0)
print(np.round(np.sort(centroids, axis=0), 3))

codes, dists = vq.vq(data[:3], centroids)
print(codes)
```
実行結果:
```
[[-0.081  0.045]
 [ 5.121  5.06 ]]
[0 0 0]
```

**注意点・落とし穴**:
- `kmeans` は事前にデータの各次元を標準化しておくことが推奨される(内部では正規化せず、そのままユークリッド距離を使うため、スケールの違う特徴量があると偏った結果になる)。scikit-learn の `KMeans` の方が実務では使われることが多く、`scipy.cluster.vq` は軽量な代替。
- クラスタ数 `k_or_guess` に整数を渡すとランダム初期化されるため、`seed`(または `rng`)を固定しないと実行毎に結果が変わりうる。

---

### `scipy.cluster.hierarchy.linkage(y, ...)` / `scipy.cluster.hierarchy.fcluster(Z, t, ...)`

**用途**: `linkage` で階層的クラスタリングの結合過程を計算し、`fcluster` で任意の閾値からクラスタ番号を割り当てる。

**シグネチャ**: `scipy.cluster.hierarchy.linkage(y, method='single', metric='euclidean', optimal_ordering=False)` / `scipy.cluster.hierarchy.fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)`

**使用例**:
```python
from scipy.cluster import hierarchy
import numpy as np
pts = np.array([[0, 0], [1, 0], [0, 1]])
Z = hierarchy.linkage(pts, method='single')
print(Z)
clusters = hierarchy.fcluster(Z, t=1.5, criterion='distance')
print(clusters)
```
実行結果:
```
[[0. 1. 1. 2.]
 [2. 3. 1. 3.]]
[1 1 1]
```

**注意点・落とし穴**:
- `linkage` の戻り値 `Z` の各行は `[結合されたクラスタID1, クラスタID2, 距離, 新クラスタに含まれる要素数]`。元データ点は `0`〜`n-1`、新しく作られた結合クラスタは `n` 以降のIDが振られる。
- `fcluster` は `criterion` によって `t` の意味が変わる(`'distance'` なら距離の閾値、`'maxclust'` ならクラスタ数の上限など)。可視化してしきい値を決めたい場合は `dendrogram` と組み合わせる。

---

## FFT(fft)

### `scipy.fft.fft(x, ...)` / `scipy.fft.ifft(x, ...)`

**用途**: 離散フーリエ変換(DFT)とその逆変換を計算する。

**シグネチャ**: `scipy.fft.fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)`(`ifft` も同形)

**使用例**:
```python
from scipy import fft
import numpy as np
x = np.array([1.0, 2.0, 1.0, 0.0])
X = fft.fft(x)
print(X)
print(fft.ifft(X))
```
実行結果:
```
[4.-0.j 0.-2.j 0.-0.j 0.+2.j]
[1.+0.j 2.+0.j 1.-0.j 0.+0.j]
```

**注意点・落とし穴**:
- 実数入力でも結果は複素数配列になる。逆変換 `ifft(fft(x))` も理論上は実数に戻るが、浮動小数点誤差で微小な虚部(`-0.j` など)が残る。

---

### `scipy.fft.fftfreq(n, d=1.0, ...)`

**用途**: `fft` の出力の各要素に対応する周波数を計算する。

**シグネチャ**: `scipy.fft.fftfreq(n, d=1.0, *, xp=None, device=None)`

**使用例**:
```python
from scipy import fft
print(fft.fftfreq(4, d=0.5))
```
実行結果:
```
[ 0.   0.5 -1.  -0.5]
```

**注意点・落とし穴**:
- `d` はサンプリング間隔(周期)。`fft` の出力配列は「正の周波数→ナイキスト周波数→負の周波数」の順に並ぶため、`fftfreq` の値も同じ順序で正→負と並ぶ(単純な昇順ではない)。

---

### `scipy.fft.rfft(x, ...)`

**用途**: 実数入力専用の高速フーリエ変換。実数の対称性を利用し、非負周波数成分だけを返す。

**シグネチャ**: `scipy.fft.rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)`

**使用例**:
```python
from scipy import fft
import numpy as np
x = np.array([1.0, 2.0, 1.0, 0.0])
print(fft.rfft(x))
```
実行結果:
```
[4.+0.j 0.-2.j 0.+0.j]
```

**注意点・落とし穴**:
- 出力長は `n//2 + 1`(この例では入力長4に対し出力長3)。実数信号のスペクトル解析では `fft` より `rfft` の方がメモリ・計算量ともに効率的。

---

## 画像処理(ndimage)

### `scipy.ndimage.gaussian_filter(input, sigma, ...)`

**用途**: ガウシアンフィルタで画像・配列を平滑化(ぼかし)する。

**シグネチャ**: `scipy.ndimage.gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, *, radius=None, axes=None)`

**使用例**:
```python
from scipy import ndimage
import numpy as np
img = np.zeros((5, 5))
img[2, 2] = 1
print(np.round(ndimage.gaussian_filter(img, sigma=1), 4))
```
実行結果:
```
[[0.0034 0.0141 0.0233 0.0141 0.0034]
 [0.0141 0.0586 0.0966 0.0586 0.0141]
 [0.0233 0.0966 0.1592 0.0966 0.0233]
 [0.0141 0.0586 0.0966 0.0586 0.0141]
 [0.0034 0.0141 0.0233 0.0141 0.0034]]
```

**注意点・落とし穴**:
- `mode`(既定 `'reflect'`)が画像端の扱いを決める。端の処理方法によって境界付近の値が変わるので、用途に応じて `'constant'`, `'nearest'` などに変更する。

---

### `scipy.ndimage.rotate(input, angle, ...)`

**用途**: 配列(画像)を任意の角度で回転する。

**シグネチャ**: `scipy.ndimage.rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=3, mode='constant', cval=0.0, prefilter=True)`

**使用例**:
```python
from scipy import ndimage
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
print(ndimage.rotate(arr, 90, reshape=False, order=0))
```
実行結果:
```
[[3. 6. 9.]
 [2. 5. 8.]
 [1. 4. 7.]]
```

**注意点・落とし穴**:
- 既定 `reshape=True` は回転後に画像全体が収まるよう出力サイズを自動拡大する(元と同じサイズを保ちたい場合は `reshape=False`)。
- 既定 `order=3`(3次スプライン補間)で滑らかだが、ラベル画像(整数のカテゴリ値)を回転する場合は補間で値が混ざってしまうため `order=0`(最近傍)を使う必要がある。

---

### `scipy.ndimage.label(input, ...)`

**用途**: 二値画像の連結成分(かたまり)を検出し、それぞれに異なるラベル番号を振る。

**シグネチャ**: `scipy.ndimage.label(input, structure=None, output=None)`

**使用例**:
```python
from scipy import ndimage
import numpy as np
mask = np.array([[1, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]])
labeled, n = ndimage.label(mask)
print(labeled)
print(n)
```
実行結果:
```
[[1 0 2 2]
 [0 0 2 2]
 [3 0 0 0]]
3
```

**注意点・落とし穴**:
- 既定の連結性は上下左右のみ(4近傍)。斜め方向も連結とみなしたい場合は `structure=np.ones((3,3))` を指定する(8近傍)。

---

## その他(constants・special)

### `scipy.constants` の定数(`c`, `g`, `pi` など)/ `scipy.constants.convert_temperature(...)`

**用途**: 物理定数(光速・重力加速度など)や単位換算関数を提供するモジュール。

**シグネチャ**: `scipy.constants.convert_temperature(val: 'npt.ArrayLike', old_scale: str, new_scale: str) -> Any`

**使用例**:
```python
from scipy import constants
print(constants.c, constants.g, constants.pi)
print(constants.convert_temperature(25, 'Celsius', 'Fahrenheit'))
```
実行結果:
```
299792458.0 9.80665 3.141592653589793
77.0
```

**注意点・落とし穴**:
- `scipy.constants` の値は単なる定数(属性アクセス)であり関数ではない。名前は `scipy.constants.physical_constants` 辞書経由でも(単位・不確かさ付きで)取得できる。

---

### `scipy.special.gamma(x)`

**用途**: ガンマ関数(階乗の実数・複素数への一般化)を計算する。

**シグネチャ**: `scipy.special.gamma(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)`

**使用例**:
```python
from scipy import special
print(special.gamma(5))     # 4! = 24
print(special.gamma(0.5))   # sqrt(pi)
```
実行結果:
```
24.0
1.7724538509055159
```

**注意点・落とし穴**:
- 正の整数 `n` に対しては `gamma(n) == (n-1)!` となる(`gamma(5) = 4! = 24`)。ずれに注意。

---

### `scipy.special.comb(N, k, ...)` / `scipy.special.perm(N, k, ...)`

**用途**: 組み合わせ数(nCk)・順列数(nPk)を計算する。

**シグネチャ**: `scipy.special.comb(N, k, *, exact=False, repetition=False)` / `scipy.special.perm(N, k, exact=False)`

**使用例**:
```python
from scipy import special
print(special.comb(5, 2))
print(special.perm(5, 2))
```
実行結果:
```
10.0
20.0
```

**注意点・落とし穴**:
- 既定 `exact=False` では浮動小数点(`float`)で返る。大きな `N` で厳密な整数値が必要な場合は `exact=True` を指定する(内部でPythonの多倍長整数演算になる)。

---

### `scipy.special.erf(x)`

**用途**: 誤差関数(正規分布の累積分布関数と関係が深い特殊関数)を計算する。

**シグネチャ**: `scipy.special.erf(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)`

**使用例**:
```python
from scipy import special
print(special.erf(1.0))
```
実行結果:
```
0.8427007929497148
```

**注意点・落とし穴**:
- 標準正規分布の累積分布関数と `erf` は `norm.cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))` の関係で結ばれている(`scipy.stats.norm.cdf` を使えば直接計算できるため、通常は明示的に `erf` を呼ぶ場面は少ない)。

---

## 応用・発展

### 高度な最適化(optimize)

#### `scipy.optimize.differential_evolution(func, bounds, ...)`

**用途**: 差分進化法による大域最適化。勾配を使わず、局所解が多い(多峰性の)関数でも大域的最小値を探索できる。

**シグネチャ**: `scipy.optimize.differential_evolution(func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, rng=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=(), x0=None, *, integrality=None, vectorized=False, seed=None)`

**使用例**:
```python
from scipy import optimize
import numpy as np

def rastrigin(x):
    return 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)

res = optimize.differential_evolution(rastrigin, bounds=[(-5.12, 5.12)]*2, seed=42)
print(res.x, res.fun, res.success)
```
実行結果:
```
[ 1.30534318e-09 -1.03853438e-09] 0.0 True
```

**注意点・落とし穴**:
- `minimize`(`BFGS` など)は初期値付近の局所最小値に収束するが、`differential_evolution` は `bounds` で指定した範囲全体を探索するため、Rastrigin関数のような多数の局所最小値を持つ関数でも大域最小値(この例では原点)に到達できる。
- 母集団ベースの手法のため `minimize` より計算コストが高い。再現性が必要な場合は `seed`(または `rng`)を固定する。

---

#### `scipy.optimize.dual_annealing(func, bounds, ...)`

**用途**: 焼きなまし法(シミュレーテッドアニーリング)ベースの大域最適化。`differential_evolution` と同様に勾配不要で多峰性関数に強い。

**シグネチャ**: `scipy.optimize.dual_annealing(func, bounds, args=(), maxiter=1000, minimizer_kwargs=None, initial_temp=5230.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0, maxfun=10000000.0, rng=None, no_local_search=False, callback=None, x0=None, *, seed=None)`

**使用例**:
```python
from scipy import optimize
import numpy as np

def eggholder(x):
    x0, x1 = x
    return (-(x1 + 47) * np.sin(np.sqrt(abs(x0/2 + (x1 + 47))))
            - x0 * np.sin(np.sqrt(abs(x0 - (x1 + 47)))))

res = optimize.dual_annealing(eggholder, bounds=[(-512, 512), (-512, 512)], seed=42)
print(res.x, res.fun)
```
実行結果:
```
[439.48113565 453.97756804] -935.3379515569777
```

**注意点・落とし穴**:
- Eggholder関数は非常に多くの局所最小値を持つ有名なベンチマーク関数。既定では内部で局所探索(`minimize`)も併用する2段構え(dual)のアルゴリズムになっている(`no_local_search=True` で焼きなましのみに切り替え可能)。
- `differential_evolution` と同じく大域最適化なので計算コストは通常の `minimize` より高い。

---

#### `scipy.optimize.nnls(A, b, ...)`

**用途**: 非負制約付き最小二乗法(Non-Negative Least Squares)。`A @ x = b` を `x >= 0` の制約下で最小二乗近似する。

**シグネチャ**: `scipy.optimize.nnls(A, b, *, maxiter=None)`

**使用例**:
```python
from scipy import optimize
import numpy as np

A = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
b = np.array([2, 1, -1], dtype=float)
x, rnorm = optimize.nnls(A, b)
print(x, rnorm)
```
実行結果:
```
[1.5 0. ] 1.2247448713915892
```

**注意点・落とし穴**:
- 通常の `scipy.linalg.lstsq` では解に負の値が出うるが、`nnls` は常に `x >= 0` を満たす解を返す(この例では2つ目の成分が負になりうる制約なしの解を避け、`0` に押し込められている)。
- 戻り値の2番目 `rnorm` は残差の2乗和の平方根(`||Ax - b||`)。

---

#### `scipy.optimize.minimize(fun, x0, method='trust-constr', ...)` + `scipy.optimize.NonlinearConstraint`

**用途**: 非線形の等式・不等式制約付き最適化。`trust-constr` 法は信頼領域(trust-region)アルゴリズムで、非線形制約(`NonlinearConstraint`)や線形制約(`LinearConstraint`)を直接扱える。

**シグネチャ**: `scipy.optimize.NonlinearConstraint(fun, lb, ub, jac='2-point', hess=None, keep_feasible=False, finite_diff_rel_step=None, finite_diff_jac_sparsity=None)`

**使用例**:
```python
from scipy import optimize

def f(x):
    return (x[0]-1)**2 + (x[1]-2.5)**2

# 制約: x0^2 + x1^2 <= 1 (単位円の内側)
nlc = optimize.NonlinearConstraint(lambda x: x[0]**2 + x[1]**2, -1e9, 1)
res = optimize.minimize(f, x0=[0.5, 0.5], method='trust-constr', constraints=[nlc])
print(res.x, res.fun, res.success)
```
実行結果:
```
[0.37139054 0.92847634] 2.8648364728736206 True
```

**注意点・落とし穴**:
- 制約なしなら最小値は `(1, 2.5)` だが、`x0^2+x1^2<=1` の制約により単位円周上の境界に解が押し出される。
- `NonlinearConstraint` の `lb`/`ub` は下限・上限。片側のみ制約したい場合は反対側を `np.inf`/`-np.inf` にする(`method='SLSQP'` では `NonlinearConstraint` の代わりに辞書形式の `constraints` も使えるが、`trust-constr` は `NonlinearConstraint`/`LinearConstraint` オブジェクトが基本)。

---

#### `scipy.optimize.linear_sum_assignment(cost_matrix, ...)`

**用途**: 割当問題(ハンガリアン法)。コスト行列が与えられたとき、総コストが最小になるように行と列を1対1で対応付ける組み合わせを求める。

**シグネチャ**: `scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)`

**使用例**:
```python
from scipy import optimize
import numpy as np

cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
row_ind, col_ind = optimize.linear_sum_assignment(cost)
print(row_ind, col_ind)
print(cost[row_ind, col_ind].sum())
```
実行結果:
```
[0 1 2] [1 0 2]
5
```

**注意点・落とし穴**:
- `differential_evolution` などの連続最適化とは異なり、これは組み合わせ最適化(離散問題)専用の関数。正方行列でなくても(長方形の行列でも)動作する。
- `maximize=True` で総コスト最大化(利益最大化の割当)にも切り替えられる。

---

### 高度な統計(stats)

#### `scipy.stats.multivariate_normal(mean=None, cov=1, ...)`

**用途**: 多変量正規分布。共分散行列を指定して同時確率密度の評価や乱数生成を行う。

**シグネチャ**: `scipy.stats.multivariate_normal(mean=None, cov=1, allow_singular=False, seed=None, **kwds)`(呼び出すと凍結(frozen)分布インスタンスが得られる)

**使用例**:
```python
from scipy import stats

mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 2]])
print(mvn.pdf([0, 0]))
print(mvn.rvs(size=3, random_state=42))
```
実行結果:
```
0.12030982838508356
[[ 0.16865045  0.72887797]
 [ 1.62117105  0.36999679]
 [-0.32573873 -0.24160214]]
```

**注意点・落とし穴**:
- `cov` は分散共分散行列(対称行列)。非対角成分は変数間の共分散を表し、この例のように正の値だと2変数は正の相関を持つ乱数になる。
- `cov` が特異(ランク落ち)な場合は既定でエラーになる。特異行列でも扱いたい場合は `allow_singular=True` を指定する。

---

#### `scipy.stats.bootstrap(data, statistic, ...)`

**用途**: ブートストラップ法(リサンプリング)により、統計量(平均・中央値など任意の関数)の信頼区間を推定する。

**シグネチャ**: `scipy.stats.bootstrap(data, statistic, *, n_resamples=9999, batch=None, vectorized=None, paired=False, axis=0, confidence_level=0.95, alternative='two-sided', method='BCa', bootstrap_result=None, rng=None, random_state=None)`

**使用例**:
```python
from scipy import stats
import numpy as np

rng = np.random.default_rng(0)
data = (rng.normal(loc=5, scale=2, size=50),)
res = stats.bootstrap(data, np.mean, confidence_level=0.95, n_resamples=2000, rng=42)
print(res.confidence_interval)
```
実行結果:
```
ConfidenceInterval(low=np.float64(4.752494918934924), high=np.float64(5.748710031554587))
```

**注意点・落とし穴**:
- `data` はタプルで渡す(単一標本でも `(sample,)` のように1要素タプルにする必要がある)。
- 既定 `method='BCa'`(bias-corrected and accelerated)は分布の歪みを補正した信頼区間を計算する。正規分布を仮定しないため、理論分布が分からない統計量(中央値・分散など)にも使える。
- 乱数の再現性は `rng`(新しい `Generator` ベースAPI)または `random_state`(レガシー)で制御する。

---

#### `scipy.stats.permutation_test(data, statistic, ...)`

**用途**: 順列検定(ランダマイゼーション検定)。標本をランダムに入れ替えて検定統計量の帰無分布を経験的に構築し、p値を求める(分布を仮定しないノンパラメトリック検定)。

**シグネチャ**: `scipy.stats.permutation_test(data, statistic, *, permutation_type='independent', vectorized=None, n_resamples=9999, batch=None, alternative='two-sided', axis=0, rng=None, random_state=None)`

**使用例**:
```python
from scipy import stats
import numpy as np

x = [23, 21, 18, 30, 25, 27, 22]
y = [31, 28, 35, 29, 33, 30, 32]

def statistic(x, y):
    return np.mean(x) - np.mean(y)

res = stats.permutation_test((x, y), statistic, n_resamples=9999, rng=42)
print(res.statistic, res.pvalue)
```
実行結果:
```
-7.428571428571427 0.002913752913752914
```

**注意点・落とし穴**:
- `ttest_ind` は正規分布を仮定するが、`permutation_test` は「群のラベルを入れ替えても統計量の分布が変わらない」という帰無仮説のみを仮定するため、より仮定が緩い。標本数が少なく分布が疑わしい場合の代替になる。
- `n_resamples` が実際の順列総数より多い場合は、可能な組み合わせを網羅する厳密検定(exact test)に自動的に切り替わることがある。

---

#### `scipy.stats.ecdf(sample)`

**用途**: 経験累積分布関数(Empirical CDF)を計算する。ヒストグラムのようにビン幅を決める必要がなく、観測データそのものから分布を推定できる。

**シグネチャ**: `scipy.stats.ecdf(sample: 'npt.ArrayLike | CensoredData') -> scipy.stats._survival.ECDFResult`

**使用例**:
```python
from scipy import stats

sample = [2, 4, 4, 5, 7, 8, 8, 8, 10]
res = stats.ecdf(sample)
print(res.cdf.quantiles)
print(res.cdf.probabilities)
```
実行結果:
```
[ 2.  4.  5.  7.  8. 10.]
[0.11111111 0.33333333 0.44444444 0.55555556 0.88888889 1.        ]
```

**注意点・落とし穴**:
- `quantiles` はデータ中のユニークな値(この例では重複する `4` と `8` はそれぞれ1つにまとめられる)。`probabilities` は各値「以下」の累積相対度数。
- `stats.kstest` の内部でも同種の経験分布の考え方が使われている。打ち切りデータ(`CensoredData`)にも対応している点が `numpy` で自前計算するより優れる。

---

### 高度な疎行列(sparse)

#### `scipy.sparse.linalg.eigsh(A, k=6, ...)`

**用途**: 対称(またはエルミート)な疎行列の固有値・固有ベクトルを、指定した個数 `k` だけ計算する(ARPACK使用。行列全体を密行列化せずに上位/下位の固有値だけ求められる)。

**シグネチャ**: `scipy.sparse.linalg.eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal', rng=None)`

**使用例**:
```python
from scipy import sparse
from scipy.sparse import linalg as sla
import numpy as np

rng = np.random.default_rng(0)
A_dense = rng.normal(size=(50, 50))
A_sym = A_dense + A_dense.T
A = sparse.csr_matrix(A_sym)
vals, vecs = sla.eigsh(A, k=3, which='LA')
print(vals)
```
実行結果:
```
[16.0478559  17.41751611 19.32228514]
```

**注意点・落とし穴**:
- `scipy.linalg.eigh`(密行列版)は全固有値を計算するが、`eigsh` は `k` 個だけを反復法で近似的に求める。大規模疎行列で上位数個の固有値だけが欲しい場合に有効(`k` は行列サイズ未満である必要がある)。
- `which='LA'` は最大の代数的固有値(Largest Algebraic)。絶対値最大なら `'LM'`(既定)、最小なら `'SA'`/`'SM'` を指定する。

---

#### `scipy.sparse.linalg.splu(A, ...)`

**用途**: 疎行列のLU分解(SuperLU使用)。同じ行列 `A` で右辺 `b` を変えながら何度も `A @ x = b` を解く場合、分解結果を再利用して高速化できる。

**シグネチャ**: `scipy.sparse.linalg.splu(A, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None)`

**使用例**:
```python
from scipy import sparse
from scipy.sparse import linalg as sla
import numpy as np

A = sparse.csc_matrix(np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]]))
lu = sla.splu(A)
b = np.array([1., 2., 3.])
print(lu.solve(b))
```
実行結果:
```
[0.22222222 0.11111111 1.44444444]
```

**注意点・落とし穴**:
- `spsolve` は毎回分解からやり直すが、`splu` は分解結果(`lu` オブジェクト)を保持できるので、右辺 `b` だけを変えて何度も解く用途では `splu(A).solve(b)` を使い回す方が効率的。
- 入力は CSC形式が推奨される(内部でCSCに変換されるため、事前にCSCで渡すと変換コストを省ける)。

---

#### 疎行列フォーマット変換(`tocsr` / `tocsc` / `scipy.sparse.issparse`)

**用途**: 疎行列は用途によって最適な格納形式が異なる(COO=構築向き、CSR=行方向演算向き、CSC=列方向演算向き)。`.tocsr()` / `.tocsc()` などで相互変換し、`scipy.sparse.issparse` で疎行列かどうかを判定する。

**シグネチャ**: `scipy.sparse.issparse(x)`(各疎行列クラスの `.tocsr(copy=False)` / `.tocsc(copy=False)` などは引数がほぼ共通)

**使用例**:
```python
from scipy import sparse
import numpy as np

dense = np.array([[0, 0, 3], [4, 0, 0], [0, 5, 0]])
coo = sparse.coo_matrix(dense)
print(type(coo).__name__)
print(type(coo.tocsr()).__name__)
print(type(coo.tocsc()).__name__)
print(sparse.issparse(coo), sparse.issparse(dense))
```
実行結果:
```
coo_matrix
csr_matrix
csc_matrix
True False
```

**注意点・落とし穴**:
- `issparse` は通常の `numpy.ndarray` には `False` を返す(疎行列かどうかの判定に `isinstance` を直接使うより将来のクラス変更に頑健)。
- COO形式は要素の追加・構築が速いが演算には不向き。行スライス/行列積主体なら `tocsr()`、列スライス/列演算主体なら `tocsc()` に変換してから使うのが定石(`sparse.csr_matrix(dense)` のように最初から目的の形式で作ってしまってもよい)。

---

#### `scipy.sparse.linalg.svds(A, k=6, ...)`

**用途**: 疎行列の特異値分解(SVD)を、上位 `k` 個の特異値・特異ベクトルだけ計算する(密行列化せずに次元削減やLSAなどに使える)。

**シグネチャ**: `scipy.sparse.linalg.svds(A, k=6, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True, solver='arpack', rng=None, options=None, *, random_state=None)`

**使用例**:
```python
from scipy import sparse
from scipy.sparse import linalg as sla
import numpy as np

rng = np.random.default_rng(0)
M = sparse.csr_matrix(rng.normal(size=(30, 10)))
U, s, Vt = sla.svds(M, k=3, rng=0)
print(np.sort(s)[::-1])
```
実行結果:
```
[8.27541018 7.76557012 6.71594282]
```

**注意点・落とし穴**:
- `scipy.linalg.svd`(密行列版)は全特異値を降順で返すが、`svds` は反復法のため `s` が昇順に近い順不同で返ることがある(この例でも降順に並べ替えるため `np.sort(...)[::-1]` を使っている)。
- 内部で反復法の初期ベクトルに乱数を使うため、`rng`(または `v0`)を固定しないと実行毎に微妙に異なる結果になりうる(通常のNumPyのグローバルシードとは独立)。

---

### 常微分方程式の応用(integrate)

#### `scipy.integrate.solve_ivp(..., events=...)`

**用途**: `solve_ivp` の `events` 引数で「特定の条件を満たした瞬間」を検出する。イベント関数の符号が変わったタイミングを二分探索で特定し、`terminal=True` にすると積分をそこで打ち切れる(例: 物体が地面に着いた瞬間で計算を止める)。

**シグネチャ**: `scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)`

**使用例**:
```python
from scipy import integrate
import numpy as np

def falling(t, y):
    return [y[1], -9.8]

def hit_ground(t, y):
    return y[0]
hit_ground.terminal = True
hit_ground.direction = -1

sol = integrate.solve_ivp(falling, [0, 10], [10, 0], events=hit_ground)
print(sol.t_events)
print(np.round(sol.y_events[0], 4))
print(sol.status)
```
実行結果:
```
[array([1.42857143])]
[[ -0. -14.]]
1
```

**注意点・落とし穴**:
- イベント関数(この例では高さ `y[0]`)自体に `.terminal`(見つかったら積分を止めるか)、`.direction`(符号がどちら向きに変化したときだけ検出するか。負なら減少方向)を属性として設定する。
- `sol.status` は `1` が「イベントで終了」、`0` が「t_spanの終端まで到達」、`-1` が「積分失敗」を意味する。`sol.t_events`/`sol.y_events` はイベントごとのリストで、複数のイベント関数を渡した場合は要素数もそれに応じて増える。

---

#### `scipy.integrate.solve_bvp(fun, bc, x, y, ...)`

**用途**: 常微分方程式の境界値問題(BVP)を解く。初期値問題(`solve_ivp`)と異なり、両端の境界条件(`bc`)を満たす解を反復的に探す。

**シグネチャ**: `scipy.integrate.solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None, tol=0.001, max_nodes=1000, verbose=0, bc_tol=None)`

**使用例**:
```python
from scipy import integrate
import numpy as np

def fun(x, y):
    return np.vstack([y[1], -np.abs(y[0])])

def bc(ya, yb):
    return np.array([ya[0], yb[0] + 2])

x = np.linspace(0, 4, 5)
y0 = np.zeros((2, x.size))
res = integrate.solve_bvp(fun, bc, x, y0)
print(res.status, res.success)
print(np.round(res.sol(np.array([0, 1, 2, 3, 4]))[0], 4))
```
実行結果:
```
0 True
[ 0.      1.7394  1.8797  0.2921 -2.    ]
```

**注意点・落とし穴**:
- `fun(x, y)` は `y` の各成分をまとめた2次元配列(`(状態変数の数, メッシュ点数)`)を受け取り、同じ形の導関数を返す必要がある(`solve_ivp` の `fun(t, y)` とは引数の意味も形も異なる)。
- `y` の初期推定値(この例では全て0)は解に収束するための出発点。境界条件 `bc(ya, yb)` は「左端の状態 `ya`」「右端の状態 `yb`」を受け取り、満たすべき等式が0になるように定義する(この例は `ya[0]=0`, `yb[0]=-2` という境界条件)。
- 戻り値 `res.sol` は連続関数として解を評価できる補間オブジェクト(`dense_output` 相当)。メッシュ点 `x` 以外の任意の点でも `res.sol(x_new)` で評価できる。

---

### 信号処理の応用(signal)

#### `scipy.signal.spectrogram(x, fs=1.0, ...)`

**用途**: 信号を短時間区間ごとにフーリエ変換し、時間軸・周波数軸・パワースペクトル密度の3つ組(スペクトログラム)を返す。周波数成分が時間とともに変化する非定常信号の解析に使う。

**シグネチャ**: `scipy.signal.spectrogram(x, fs=1.0, window=('tukey_periodic', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')`

**使用例**:
```python
from scipy import signal
import numpy as np

fs = 100.0
t = np.arange(0, 2, 1/fs)
# 前半1秒は10Hz、後半1秒は25Hzの正弦波
x = np.where(t < 1, np.sin(2*np.pi*10*t), np.sin(2*np.pi*25*t))
f, tt, Sxx = signal.spectrogram(x, fs, nperseg=50)
print(f.shape, tt.shape, Sxx.shape)
print(np.round(f[np.argmax(Sxx, axis=0)], 1))
```
実行結果:
```
(26,) (4,) (26, 4)
[10. 10. 24. 24.]
```

**注意点・落とし穴**:
- `fft`(全区間を一度に変換)では「いつ」周波数が変化したかが分からないが、`spectrogram` は区間(セグメント)ごとに変換するため周波数の時間変化が追える。この例でも前半セグメントは10Hz付近、後半は25Hz付近がピークになっている。
- `nperseg`(セグメント長)を大きくすると周波数分解能は上がるが時間分解能が下がる(トレードオフ)。`Sxx` の形状は `(周波数ビン数, 時間セグメント数)`。

---

#### `scipy.signal.ShortTimeFFT(win, hop, fs, ...)`

**用途**: `stft`/`spectrogram` の後継となる短時間フーリエ変換(STFT)の実装。ウィンドウとホップ幅を明示的にオブジェクトとして保持し、STFT/ISTFT(逆変換)を一貫して扱える。

**シグネチャ**: `scipy.signal.ShortTimeFFT(win: numpy.ndarray, hop: int, fs: float, *, fft_mode='onesided', mfft=None, dual_win=None, scale_to=None, phase_shift=0)`

**使用例**:
```python
from scipy import signal
import numpy as np

fs = 100.0
t = np.arange(0, 2, 1/fs)
x = np.where(t < 1, np.sin(2*np.pi*10*t), np.sin(2*np.pi*25*t))

win = signal.windows.hann(50)
SFT = signal.ShortTimeFFT(win, hop=25, fs=fs)
Sx = SFT.stft(x)
print(Sx.shape)
print(np.round(SFT.f[np.argmax(np.abs(Sx), axis=0)], 1))
```
実行結果:
```
(26, 9)
[10. 10. 10. 10. 10. 26. 26. 24. 24.]
```

**注意点・落とし穴**:
- 公式ドキュメントでは、レガシーな `scipy.signal.stft`/`istft` 関数より `ShortTimeFFT` クラスの使用が推奨されている(パディングの扱いなどがより明確)。
- `win`(窓関数の配列そのもの)と `hop`(ホップ幅、サンプル数)を自分で用意する必要がある点が `spectrogram` の `nperseg`/`noverlap` 指定と異なる。窓は `scipy.signal.windows` モジュール(例: `windows.hann(N)`)から作るのが一般的。

---

#### `scipy.signal.hilbert(x, ...)`

**用途**: ヒルベルト変換により解析信号(analytic signal)を求める。信号の瞬時振幅(エンベロープ)や瞬時位相の抽出に使う。

**シグネチャ**: `scipy.signal.hilbert(x, N=None, axis=-1)`

**使用例**:
```python
from scipy import signal
import numpy as np

t = np.linspace(0, 1, 100, endpoint=False)
sig_in = (1 + 0.5*t) * np.sin(2*np.pi*5*t)   # 振幅が徐々に大きくなる正弦波
analytic = signal.hilbert(sig_in)
envelope = np.abs(analytic)
print(np.round(envelope[::20], 4))
```
実行結果:
```
[1.25   1.1212 1.2052 1.2948 1.3788]
```

**注意点・落とし穴**:
- `scipy.signal.hilbert` が返すのは実信号+虚部としてのヒルベルト変換を組み合わせた「解析信号」(複素数配列)であり、ヒルベルト変換そのもの(虚部だけ)ではない点に注意(名前から誤解しやすい)。
- `np.abs(analytic)` で振幅包絡線(エンベロープ)、`np.unwrap(np.angle(analytic))` で瞬時位相が得られる。この例では元の振幅係数 `1 + 0.5*t`(0〜1の間で1.0→1.5に増加)にほぼ一致したエンベロープが復元されている。

---

### 特殊関数の深掘り(special)

#### `scipy.special.jv(v, z)`

**用途**: 第一種ベッセル関数。円筒対称な波動方程式・振動問題などに現れる特殊関数。

**シグネチャ**: `scipy.special.jv(v, z, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)`(`inspect.signature` 上は汎用の `x1, x2` と表示されるが、実際の引数は次数 `v` と引数 `z`)

**使用例**:
```python
from scipy import special

print(special.jv(0, 1.0))
print(special.jv(1, 1.0))
print(special.jv([0, 1, 2], 5.0))
```
実行結果:
```
0.7651976865579666
0.44005058574493355
[-0.17759677 -0.32757914  0.04656512]
```

**注意点・落とし穴**:
- 第一引数 `v`(次数)は整数だけでなく実数(分数次)も指定できる。`v` に配列を渡すと複数の次数をまとめて計算できる(ブロードキャストされる)。

---

#### `scipy.special.iv(v, z)` / `scipy.special.kv(v, z)`

**用途**: 修正ベッセル関数(第一種 `iv`・第二種 `kv`)。熱伝導や拡散方程式など、振動しない指数関数的な解に現れる。

**シグネチャ**: `scipy.special.iv(v, z, /, out=None, ...)` / `scipy.special.kv(v, z, /, out=None, ...)`(`jv` と同様、引数は次数 `v` と引数 `z`)

**使用例**:
```python
from scipy import special

print(special.iv(0, 1.0))
print(special.kv(0, 1.0))
```
実行結果:
```
1.2660658777520084
0.42102443824070834
```

**注意点・落とし穴**:
- `jv`/`yv`(通常のベッセル関数)は振動しながら減衰するが、`iv` は `z` が大きくなると指数的に発散し、`kv` は指数的に減衰する。名前が似ているため混同しやすい(`i`=Increasing/`k`=Kelvin由来の慣習的な記号)。

---

#### `scipy.special.ellipk(m)` / `scipy.special.ellipe(m)`

**用途**: 第一種・第二種の完全楕円積分。振り子の周期の厳密解や楕円の弧長計算などに使われる。

**シグネチャ**: `scipy.special.ellipk(m, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)` / `scipy.special.ellipe(m, /, out=None, ...)`(同形)

**使用例**:
```python
from scipy import special

print(special.ellipk(0.5))
print(special.ellipe(0.5))
```
実行結果:
```
1.8540746773013719
1.3506438810476755
```

**注意点・落とし穴**:
- SciPyの引数 `m` は離心率的パラメータ `m = k^2`(母数)であり、楕円の「弾性率(modulus)」 `k` そのものではない(文献によっては `K(k)` と `K(m)` の記法が混在するので要注意)。`m=1` に近づくと `ellipk` は発散する。

---

#### `scipy.special.logsumexp(a, ...)` / `scipy.special.softmax(x, ...)`

**用途**: `logsumexp` はオーバーフローを避けながら `log(sum(exp(a)))` を数値的に安定して計算する。`softmax` は入力を確率分布(合計1の非負値)に変換する(`logsumexp` を内部で利用)。

**シグネチャ**: `scipy.special.logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False)` / `scipy.special.softmax(x, axis=None)`

**使用例**:
```python
from scipy import special
import numpy as np

a = np.array([1000.0, 1000.0, 1000.0])
print(special.logsumexp(a))          # 素朴に log(sum(exp(a))) を計算するとオーバーフローしてinfになる
print(special.softmax([1.0, 2.0, 3.0]))
```
実行結果:
```
1001.0986122886682
[0.09003057 0.24472847 0.66524096]
```

**注意点・落とし穴**:
- 実際に `np.log(np.sum(np.exp(a)))` を同じ入力(`a = [1000, 1000, 1000]`)で計算すると `exp(1000)` の時点でオーバーフローし `RuntimeWarning: overflow encountered in exp` とともに `inf` が返る(実行して確認済み)。`logsumexp` は最大値を引いてから計算する数値的トリックによりこれを回避し、正しく `1001.0986...` を返す。
- ロジスティック回帰やソフトマックス回帰、混合モデルの対数尤度計算など、指数と対数が組み合わさる箇所では素朴な実装よりこれらの関数を使う方が安全。
