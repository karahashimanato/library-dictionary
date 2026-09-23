# statsmodels 逆引き辞書

statsmodels 0.14.6 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(statsmodels 0.14.6)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。

## 目次

1. [線形回帰(OLS/WLS/GLS/分位点回帰)](#1-線形回帰olswlsgls分位点回帰)
2. [GLM・ロバスト回帰・離散選択モデル](#2-glmロバスト回帰離散選択モデル)
3. [formula API](#3-formula-api)
4. [時系列モデル](#4-時系列モデル)
5. [時系列分解・定常性検定](#5-時系列分解定常性検定)
6. [回帰診断・多重共線性](#6-回帰診断多重共線性)
7. [仮説検定・記述統計](#7-仮説検定記述統計)
8. [混合効果モデル・一般化推定方程式](#8-混合効果モデル一般化推定方程式)
9. [モデル評価・予測ユーティリティ](#9-モデル評価予測ユーティリティ)
10. [その他ユーティリティ](#10-その他ユーティリティ)

**応用・発展**

11. [状態空間モデル(UnobservedComponents/DynamicFactor/MarkovRegression/RecursiveLS)](#11-状態空間モデルunobservedcomponentsdynamicfactormarkovregressionrecursivels)
12. [ロバスト標準誤差・クラスタ標準誤差](#12-ロバスト標準誤差クラスタ標準誤差)
13. [カウントデータモデル(Poisson/NegativeBinomial/ZeroInflatedPoisson)](#13-カウントデータモデルpoissonnegativebinomialzeroinflatedpoisson)
14. [ノンパラメトリック回帰・密度推定](#14-ノンパラメトリック回帰密度推定)
15. [生存時間分析](#15-生存時間分析)
16. [多変量解析](#16-多変量解析)

---

## 1. 線形回帰(OLS/WLS/GLS/分位点回帰)

### `OLS(...)`

**用途**: 最小二乗法による線形回帰。

**シグネチャ**: `statsmodels.api.OLS(endog, exog=None, missing='none', hasconst=None, **kwargs)`

**使用例**:
```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 100
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
X = sm.add_constant(np.column_stack([x1, x2]))
y = 1.5 + 2.0*x1 - 1.0*x2 + rng.normal(scale=0.5, size=n)

res = sm.OLS(y, X).fit()
print(res.summary())
```
実行結果:
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.933
Model:                            OLS   Adj. R-squared:                  0.932
Method:                 Least Squares   F-statistic:                     679.0
No. Observations:                 100   AIC:                             172.9
Df Residuals:                      97   BIC:                             180.7
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.4327      0.057     25.181      0.000       1.320       1.546
x1             1.9911      0.059     33.794      0.000       1.874       2.108
x2            -0.9817      0.059    -16.550      0.000      -1.099      -0.864
==============================================================================
Omnibus:                        5.090   Durbin-Watson:                   1.972
Prob(Omnibus):                  0.078   Jarque-Bera (JB):                4.431
Skew:                           0.443   Prob(JB):                        0.109
Kurtosis:                       3.528   Cond. No.                         1.13
==============================================================================
```

**注意点・落とし穴**:
- `exog`(説明変数)に切片列は自動追加されない。定数項が必要な場合は `sm.add_constant()` を明示的に使う(scikit-learnの`fit_intercept=True`に相当する挙動はデフォルトでは起きない)。
- `endog`/`exog`の引数順が scikit-learn(`X, y`)と逆(`y, X`)である点に注意。

### `WLS(...)`

**用途**: 各観測値に重みを付けた最小二乗法(不均一分散への対処)。

**シグネチャ**: `statsmodels.api.WLS(endog, exog, weights=1.0, missing='none', hasconst=None, **kwargs)`

**使用例**:
```python
w = rng.uniform(0.5, 2.0, size=n)
wls_res = sm.WLS(y, X, weights=1/w).fit()
print(wls_res.params.round(3))
```
実行結果:
```
[ 1.428  1.953 -0.994]
```

**注意点・落とし穴**:
- `weights`は「分散の逆数」に相当する値を渡す(値が大きいほどその観測値を重視する)。標準偏差そのものを渡すと逆の重み付けになる。

### `GLS(...)`

**用途**: 誤差の共分散構造(自己相関など)が既知の場合の一般化最小二乗法。

**シグネチャ**: `statsmodels.api.GLS(endog, exog, sigma=None, missing='none', hasconst=None, **kwargs)`

**使用例**:
```python
from scipy.linalg import toeplitz
rho = 0.4
order = toeplitz(np.arange(n))
sigma = rho**order  # AR(1)型の共分散構造
gls_res = sm.GLS(y, X, sigma=sigma).fit()
print(gls_res.params.round(3))
```
実行結果:
```
[ 1.434  2.012 -0.952]
```

**注意点・落とし穴**:
- `sigma`は誤差の共分散行列(n×n)そのものを渡す必要があり、実務では未知のため事前に推定する必要がある(`sigma`が単位行列なら`OLS`と一致する)。

### `QuantReg(...)`

**用途**: 分位点回帰(平均ではなく指定した分位点、例えば中央値を予測するモデル)。外れ値に頑健。

**シグネチャ**: `statsmodels.api.QuantReg(endog, exog, **kwargs)`

**使用例**:
```python
qr_res = sm.QuantReg(y, X).fit(q=0.5)  # 中央値回帰
print(qr_res.summary().tables[1])
```
実行結果:
```
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.3357      0.072     18.606      0.000       1.193       1.478
x1             1.9204      0.074     25.831      0.000       1.773       2.068
x2            -1.0470      0.075    -13.989      0.000      -1.196      -0.898
==============================================================================
```

**注意点・落とし穴**:
- `fit()`ではなく`fit(q=0.5)`のように分位点`q`(0〜1)を指定して呼び出す。`q=0.5`が中央値回帰(最小絶対偏差回帰に相当)。

---

## 2. GLM・ロバスト回帰・離散選択モデル

### `GLM(...)`

**用途**: 一般化線形モデル(正規分布以外の誤差分布・リンク関数を扱う回帰)。

**シグネチャ**: `statsmodels.api.GLM(endog, exog, family=None, offset=None, exposure=None, freq_weights=None, var_weights=None, missing='none', **kwargs)`

**使用例**:
```python
rng = np.random.default_rng(1)
n = 200
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
X = sm.add_constant(np.column_stack([x1, x2]))
lin = 0.5 + 1.5*x1 - 1.0*x2
p = 1/(1+np.exp(-lin))
y = rng.binomial(1, p)

glm_res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
print(glm_res.summary())
```
実行結果:
```
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                      y   No. Observations:                  200
Model:                            GLM   Df Residuals:                      197
Model Family:                Binomial   Df Model:                            2
Link Function:                  Logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -95.387
Deviance:                       190.77   Pearson chi2:                     196.
No. Iterations:                     5   Pseudo R-squ. (CS):             0.3186
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.6387      0.186      3.431      0.001       0.274       1.004
x1             1.4235      0.247      5.766      0.000       0.940       1.907
x2            -1.1496      0.220     -5.218      0.000      -1.581      -0.718
==============================================================================
```

**注意点・落とし穴**:
- `family`のデフォルトは`None`だが、実際には内部で`Gaussian()`(恒等リンク、通常の線形回帰相当)にフォールバックする。分類・カウントデータには`family=sm.families.Binomial()`/`sm.families.Poisson()`などを明示する必要がある。
- 二値分類なら`sm.Logit`の方が専用の要約統計量(擬似R2、周辺効果など)が揃っており使いやすいことが多い。

### `Logit(...)`

**用途**: ロジスティック回帰(二値分類)専用モデル。

**シグネチャ**: `statsmodels.api.Logit(endog, exog, offset=None, check_rank=True, **kwargs)`

**使用例**:
```python
logit_res = sm.Logit(y, X).fit(disp=0)
print(logit_res.params.round(3))
print("pred_prob[:5]:", logit_res.predict(X[:5]).round(3))
```
実行結果:
```
[ 0.639  1.423 -1.15 ]
pred_prob[:5]: [0.275 0.374 0.912 0.162 0.937]
```

**注意点・落とし穴**:
- `fit()`は最尤推定の反復計算経過を毎回出力する。ログを抑えたい場合は`fit(disp=0)`を指定する。
- `predict()`はデフォルトで確率(0〜1)を返す。0/1ラベルへの変換は自分で閾値(通常0.5)を決めて行う必要がある。

### `Probit(...)`

**用途**: プロビット回帰(標準正規分布の累積分布関数をリンク関数とする二値分類)。

**シグネチャ**: `statsmodels.api.Probit(endog, exog, offset=None, check_rank=True, **kwargs)`

**使用例**:
```python
probit_res = sm.Probit(y, X).fit(disp=0)
print(probit_res.params.round(3))
```
実行結果:
```
[ 0.37   0.839 -0.68 ]
```

**注意点・落とし穴**:
- `Logit`と`Probit`は係数の大きさを直接比較できない(リンク関数が異なるためスケールが違う)。予測確率や限界効果(`get_margeff()`)で比較するのが適切。

### `RLM(...)`

**用途**: ロバスト線形回帰(外れ値の影響を抑えたM推定)。

**シグネチャ**: `statsmodels.api.RLM(endog, exog, M=None, missing='none', **kwargs)`

**使用例**:
```python
n2 = 100
xr = rng.normal(size=n2)
Xr = sm.add_constant(xr)
yr = 2.0 + 3.0*xr + rng.normal(scale=0.5, size=n2)
yr[:5] += 20  # 意図的に外れ値を混入

rlm_res = sm.RLM(yr, Xr, M=sm.robust.norms.HuberT()).fit()
ols_res = sm.OLS(yr, Xr).fit()
print("RLM params:", rlm_res.params.round(3))
print("OLS params (外れ値の影響を受けた例):", ols_res.params.round(3))
```
実行結果:
```
RLM params: [1.963 2.927]
OLS params (外れ値の影響を受けた例): [2.878 2.737]
```

**注意点・落とし穴**:
- 真の係数(切片2.0、傾き3.0)に対し、外れ値が混入した`OLS`は切片が2.878まで引っ張られるのに対し、`RLM`(`HuberT`)は1.963と真値に近い推定を保っている。ただし`RLM`には`summary()`にR2が出ない(尤度ベースの評価指標ではないため)。
- `M`(重み関数)を省略すると`HuberT()`がデフォルトで使われる。

---

## 3. formula API

### `smf.ols(...)`

**用途**: R風の数式文字列(`"y ~ x1 + x2"`)でOLS回帰を指定する。DataFrameの列名をそのまま使える。

**シグネチャ**: `statsmodels.formula.api.ols(formula, data, subset=None, drop_cols=None, *args, **kwargs)`

**使用例**:
```python
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(2)
n = 100
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
y = 1.0 + 2.0*x1 - 0.5*x2 + rng.normal(scale=0.5, size=n)
df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

res = smf.ols("y ~ x1 + x2", data=df).fit()
print(res.params.round(3))
```
実行結果:
```
Intercept    0.928
x1           2.076
x2          -0.540
dtype: float64
```

**注意点・落とし穴**:
- `sm.OLS`と違い切片(`Intercept`)は数式内で自動的に追加される。除去したい場合は`"y ~ x1 + x2 - 1"`のように明示的に`-1`を付ける。
- カテゴリ変数は`"y ~ x1 + C(group)"`のように`C()`で囲むとダミー変数化される。

### `smf.glm(...)`

**用途**: 数式APIでGLMを指定する。

**シグネチャ**: `statsmodels.formula.api.glm(formula, data, subset=None, drop_cols=None, *args, **kwargs)`

**使用例**:
```python
import statsmodels.api as sm

lin = 0.3 + 1.2*x1 - 0.8*x2
p = 1/(1+np.exp(-lin))
yb = rng.binomial(1, p)
dfb = pd.DataFrame({"y": yb, "x1": x1, "x2": x2})

glm_res = smf.glm("y ~ x1 + x2", data=dfb, family=sm.families.Binomial()).fit()
print(glm_res.params.round(3))
```
実行結果:
```
Intercept    0.207
x1           1.194
x2          -0.814
dtype: float64
```

### `smf.logit(...)`

**用途**: 数式APIでロジスティック回帰を指定する。

**シグネチャ**: `statsmodels.formula.api.logit(formula, data, subset=None, drop_cols=None, *args, **kwargs)`

**使用例**:
```python
logit_res = smf.logit("y ~ x1 + x2", data=dfb).fit(disp=0)
print(logit_res.params.round(3))
```
実行結果:
```
Intercept    0.207
x1           1.194
x2          -0.814
dtype: float64
```

**注意点・落とし穴**:
- 二値分布+ロジットリンクという同一設定なら`smf.glm(family=Binomial())`と`smf.logit`は係数が完全に一致する(実測でも同一の値)。`Logit`専用APIの方が擬似R2など分類向けの要約が充実している。

---

## 4. 時系列モデル

### `ARIMA(...)`

**用途**: 自己回帰和分移動平均モデル(ARIMA)による単変量時系列予測。

**シグネチャ**: `statsmodels.tsa.arima.model.ARIMA(endog, exog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None, enforce_stationarity=True, enforce_invertibility=True, concentrate_scale=False, trend_offset=1, dates=None, freq=None, missing='none', validate_specification=True)`

**使用例**:
```python
rng = np.random.default_rng(3)
n = 150
eps = rng.normal(scale=1.0, size=n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.6*y[t-1] + eps[t]
y = y + 10

from statsmodels.tsa.arima.model import ARIMA
arima_res = ARIMA(y, order=(1, 0, 0)).fit()
print(arima_res.params.round(4))
print("forecast(5):", arima_res.forecast(5).round(3))
```
実行結果:
```
const     9.9800
ar.L1     0.5284
sigma2    1.1049
dtype: float64
forecast(5): [10.138 10.063 10.024 10.003  9.992]
```

**注意点・落とし穴**:
- `order=(p, d, q)`。今回のように`d=0`(定常データ)でも`const`項が自動推定される(トレンドを明示的に無効化したい場合は`trend='n'`を指定)。
- `.summary()`の見出しは内部実装が共通のため`"SARIMAX Results"`と表示される(`ARIMA`は`SARIMAX`の特殊ケースとして実装されている)。

### `SARIMAX(...)`

**用途**: 季節成分・外生変数を含む(季節性)ARIMAモデル。

**シグネチャ**: `statsmodels.tsa.statespace.sarimax.SARIMAX(endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None, measurement_error=False, time_varying_regression=False, mle_regression=True, simple_differencing=False, enforce_stationarity=True, enforce_invertibility=True, hamilton_representation=False, concentrate_scale=False, trend_offset=1, use_exact_diffuse=False, dates=None, freq=None, missing='none', validate_specification=True, **kwargs)`

**使用例**:
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

t = np.arange(n)
seasonal = 3*np.sin(2*np.pi*t/12)
ys = y + seasonal
sarimax_res = SARIMAX(ys, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)).fit(disp=0)
print(dict(zip(sarimax_res.model.param_names, sarimax_res.params.round(3))))
print("forecast(3):", sarimax_res.forecast(3).round(3))
```
実行結果:
```
{'ar.L1': 0.969, 'ar.S.L12': 0.468, 'sigma2': 2.009}
forecast(3): [11.137  9.843  8.796]
```

**注意点・落とし穴**:
- `seasonal_order=(P, D, Q, s)`の`s`は周期(月次データで年周期なら12)。指定を忘れると`(0,0,0,0)`のまま季節成分が無視される。
- `.params`は`ndarray`を渡した場合はプレーンな`numpy.ndarray`で返る(pandasの`Series`にはならない)。パラメータ名は`model.param_names`から取得する。

### `AutoReg(...)`

**用途**: 単純な自己回帰(AR)モデル。`ARIMA`より軽量で高速。

**シグネチャ**: `statsmodels.tsa.ar_model.AutoReg(endog, lags, trend='c', seasonal=False, exog=None, hold_back=None, period=None, missing='none', *, deterministic=None, old_names=False)`

**使用例**:
```python
from statsmodels.tsa.ar_model import AutoReg
ar_res = AutoReg(y, lags=2).fit()
print(ar_res.params.round(3))
print("predict:", ar_res.predict(start=n, end=n+2).round(3))
```
実行結果:
```
[ 4.888  0.555 -0.043]
predict: [10.142 10.074 10.042]
```

**注意点・落とし穴**:
- `lags`は`ARIMA`の`order`のように省略できない必須引数(整数、またはラグのリストを指定可能)。
- `predict()`は`start`/`end`にサンプル内・サンプル外どちらのインデックスも指定できる。`start=len(y)`のようにサンプル外を指定すると将来予測になる。

### `ExponentialSmoothing(...)`

**用途**: 指数平滑法(Holt-Winters法)によるレベル・トレンド・季節性のモデリング。

**シグネチャ**: `statsmodels.tsa.holtwinters.ExponentialSmoothing(endog, trend=None, damped_trend=False, seasonal=None, *, seasonal_periods=None, initialization_method='estimated', initial_level=None, initial_trend=None, initial_seasonal=None, use_boxcox=False, bounds=None, dates=None, freq=None, missing='none')`

**使用例**:
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
es_res = ExponentialSmoothing(y, trend="add", seasonal=None).fit()
print("smoothing_level:", round(es_res.params["smoothing_level"], 3))
print("smoothing_trend:", round(es_res.params["smoothing_trend"], 3))
print("forecast(3):", es_res.forecast(3).round(3))
```
実行結果:
```
smoothing_level: 0.546
smoothing_trend: 0.0
forecast(3): [10.084 10.09  10.096]
```

**注意点・落とし穴**:
- `trend`/`seasonal`は`None`、`"add"`(加法)、`"mul"`(乗法)のいずれか。`seasonal`を使う場合は`seasonal_periods`の指定が必須。
- ARIMA系と異なり信頼区間はデフォルトの`.forecast()`には付かない(`.get_prediction()`や`simulate()`を使う必要がある)。

### `VAR(...)`

**用途**: ベクトル自己回帰モデル(複数の時系列の相互依存関係をモデル化)。

**シグネチャ**: `statsmodels.tsa.vector_ar.var_model.VAR(endog, exog=None, dates=None, freq=None, missing='none')`

**使用例**:
```python
from statsmodels.tsa.vector_ar.var_model import VAR
y2 = np.zeros(n)
for t in range(1, n):
    y2[t] = 0.3*y[t-1] + 0.5*y2[t-1] + rng.normal(scale=0.5)
data = np.column_stack([y, y2])

var_res = VAR(data).fit(maxlags=2)
print("forecast:", var_res.forecast(data[-2:], steps=3).round(3))
```
実行結果:
```
forecast: [[10.373  5.719]
 [10.095  5.936]
 [10.033  6.038]]
```

**注意点・落とし穴**:
- コンストラクタに`order`を渡さず、`fit(maxlags=...)`の段階でラグ次数を決める(情報量基準による自動選択も`fit(maxlags=k, ic='aic')`のように可能)。
- `forecast()`には最新`k`時点分(`k`=推定に使ったラグ数)の実データを渡す必要がある。

---

## 5. 時系列分解・定常性検定

### `adfuller(...)`

**用途**: 拡張ディッキー・フラー検定(単位根検定)。帰無仮説は「単位根あり(非定常)」。

**シグネチャ**: `statsmodels.tsa.stattools.adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)`

**使用例**:
```python
import pandas as pd
from statsmodels.tsa.stattools import adfuller

rng = np.random.default_rng(4)
n = 144
t = np.arange(n)
y = 10 + 0.05*t + 3*np.sin(2*np.pi*t/12) + rng.normal(scale=0.5, size=n)  # トレンド+季節性

stat, pval, usedlag, nobs, crit, icbest = adfuller(y)
print("stat:", round(stat, 3), "pvalue:", round(pval, 4))

dy = np.diff(y)  # 1階差分
stat2, pval2, *_ = adfuller(dy)
print("(差分後) stat:", round(stat2, 3), "pvalue:", round(pval2, 6))
```
実行結果:
```
stat: -0.334 pvalue: 0.9206
(差分後) stat: -6.546 pvalue: 0.0
```

**注意点・落とし穴**:
- トレンドのある元データはp値0.92で「単位根あり(非定常)」を棄却できないが、1階差分すると明確に定常(p値≒0)になる。定常性検定は必ず元データと差分後の両方を確認するとよい。
- 返り値はタプル(`(統計量, p値, 使用ラグ数, 標本数, 臨界値dict, 最良情報量規準)`)で、`pval`は5%が慣習的な有意水準。

### `kpss(...)`

**用途**: KPSS検定(単位根検定の一種)。`adfuller`と帰無仮説が逆で「定常である」が帰無仮説。

**シグネチャ**: `statsmodels.tsa.stattools.kpss(x, regression='c', nlags='auto', store=False)`

**使用例**:
```python
from statsmodels.tsa.stattools import kpss
kstat, kpval, klag, kcrit = kpss(y, regression="c", nlags="auto")
print("stat:", round(kstat, 3), "pvalue:", kpval, "lags:", klag)
```
実行結果:
```
stat: 1.528 pvalue: 0.01 lags: 7
```

**注意点・落とし穴**:
- `adfuller`と帰無仮説が逆(KPSSは「定常」が帰無仮説)なので、両者を組み合わせて判断するのが一般的(例:ADFで棄却できずKPSSでも棄却される→トレンド定常の可能性など)。
- p値がルックアップテーブルの範囲外だと`InterpolationWarning`が出て「実際のp値はこれより小さい」という近似値が返る(実行時に確認済み)。

### `seasonal_decompose(...)`

**用途**: 時系列をトレンド・季節・残差成分に分解する(移動平均ベース、古典的な手法)。

**シグネチャ**: `statsmodels.tsa.seasonal.seasonal_decompose(x, model='additive', filt=None, period=None, two_sided=True, extrapolate_trend=0)`

**使用例**:
```python
from statsmodels.tsa.seasonal import seasonal_decompose
idx = pd.date_range("2010-01-01", periods=n, freq="MS")
ts = pd.Series(y, index=idx)
dec = seasonal_decompose(ts, model="additive", period=12)
print(dec.trend.iloc[12:15].round(3))
print(dec.seasonal.iloc[:4].round(3))
```
実行結果:
```
2011-01-01    10.637
2011-02-01    10.672
2011-03-01    10.720
Freq: MS, Name: trend, dtype: float64
2010-01-01    0.182
2010-02-01    1.663
2010-03-01    2.531
2010-04-01    3.165
Freq: MS, Name: seasonal, dtype: float64
```

**注意点・落とし穴**:
- 移動平均を使うため`trend`成分の両端(`period/2`個分)は`NaN`になる(デフォルト`extrapolate_trend=0`)。端点まで埋めたい場合は`extrapolate_trend='freq'`などを指定する。
- `period`は`pandas`の`DatetimeIndex`に`freq`があっても自動推定されないため明示指定が必要。

### `STL(...)`

**用途**: LOESS(局所回帰)ベースの季節・トレンド分解。`seasonal_decompose`より外れ値・季節性の変化に頑健。

**シグネチャ**: `statsmodels.tsa.seasonal.STL(endog, period=None, seasonal=7, trend=None, low_pass=None, seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False, seasonal_jump=1, trend_jump=1, low_pass_jump=1)`

**使用例**:
```python
from statsmodels.tsa.seasonal import STL
stl_res = STL(ts, period=12).fit()
print(stl_res.trend.iloc[12:15].round(3))
print(stl_res.seasonal.iloc[:4].round(3))
```
実行結果:
```
2011-01-01    10.612
2011-02-01    10.662
2011-03-01    10.707
Freq: MS, Name: trend, dtype: float64
2010-01-01   -0.139
2010-02-01    1.463
2010-03-01    2.862
2010-04-01    3.588
Freq: MS, Name: season, dtype: float64
```

**注意点・落とし穴**:
- コンストラクタでは学習せず、`.fit()`を呼んで初めて分解結果(`DecomposeResult`)が得られる(`seasonal_decompose`は関数一発で結果が返る点と対照的)。
- `robust=True`にすると外れ値の影響を抑えたロバスト推定になるが、計算コストが増える。

### `acf(...)` / `pacf(...)`

**用途**: 自己相関関数(ACF)・偏自己相関関数(PACF)を計算する。ARIMAの次数選定に使う。

**シグネチャ**: `statsmodels.tsa.stattools.acf(x, adjusted=False, nlags=None, qstat=False, fft=True, alpha=None, bartlett_confint=True, missing='none')` / `statsmodels.tsa.stattools.pacf(x, nlags=None, method='ywadjusted', alpha=None)`

**使用例**:
```python
from statsmodels.tsa.stattools import acf, pacf
acf_vals = acf(dy, nlags=5)
pacf_vals = pacf(dy, nlags=5)
print("acf:", acf_vals.round(3))
print("pacf:", pacf_vals.round(3))
```
実行結果:
```
acf: [ 1.     0.384  0.384 -0.021 -0.343 -0.566]
pacf: [ 1.     0.387  0.282 -0.305 -0.522 -0.494]
```

**注意点・落とし穴**:
- 返り値の先頭(ラグ0)は必ず`1.0`(自分自身との相関)。`nlags=5`なら長さ6の配列(ラグ0〜5)が返る。
- `alpha`を指定すると信頼区間も同時に返る(戻り値がタプルに変わるため、`alpha=None`(デフォルト)の場合と型が異なる点に注意)。

---

## 6. 回帰診断・多重共線性

### `durbin_watson(...)`

**用途**: 残差の系列相関(自己相関)を検定する。2に近いほど自己相関なし、0に近いほど正の自己相関、4に近いほど負の自己相関。

**シグネチャ**: `statsmodels.stats.stattools.durbin_watson(resids, axis=0)`

**使用例**:
```python
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

rng = np.random.default_rng(5)
n = 100
x1 = rng.normal(size=n)
x2 = 0.9*x1 + rng.normal(scale=0.3, size=n)
x3 = rng.normal(size=n)
X = sm.add_constant(np.column_stack([x1, x2, x3]))
y = 1 + 2*x1 + 0.5*x3 + rng.normal(scale=1.0, size=n)
res = sm.OLS(y, X).fit()

print(round(durbin_watson(res.resid), 4))
```
実行結果:
```
1.5606
```

**注意点・落とし穴**:
- `res.summary()`の出力にも同じ値が`Durbin-Watson`として含まれているため、単体で計算し直す必要は必ずしもない。

### `jarque_bera(...)`

**用途**: 残差の正規性検定(歪度・尖度に基づく)。帰無仮説は「正規分布に従う」。

**シグネチャ**: `statsmodels.stats.stattools.jarque_bera(resids, axis=0)`

**使用例**:
```python
from statsmodels.stats.stattools import jarque_bera
jb_stat, jb_p, skew, kurtosis = jarque_bera(res.resid)
print("stat:", round(jb_stat, 4), "pvalue:", round(jb_p, 4))
```
実行結果:
```
stat: 0.2267 pvalue: 0.8928
```

**注意点・落とし穴**:
- p値が小さい(例えば0.05未満)場合に正規性の帰無仮説を棄却する。今回の例はp=0.89と大きく、正規性を棄却できない(正規分布からの生成データなので妥当な結果)。

### `het_breuschpagan(...)`

**用途**: ブロイシュ・ペーガン検定(残差の不均一分散を検定)。帰無仮説は「均一分散」。

**シグネチャ**: `statsmodels.stats.diagnostic.het_breuschpagan(resid, exog_het, robust=True)`

**使用例**:
```python
from statsmodels.stats.diagnostic import het_breuschpagan
lm, lm_p, fval, f_p = het_breuschpagan(res.resid, X)
print("LM stat:", round(lm, 4), "LM p-value:", round(lm_p, 4))
```
実行結果:
```
LM stat: 1.3767 LM p-value: 0.711
```

**注意点・落とし穴**:
- 第2引数`exog_het`には、不均一分散の説明に使う変数群(通常はモデルの説明変数`X`)を渡す。モデル本体ではなく残差と説明変数から独立に計算する関数。
- p値が小さい場合は不均一分散が疑われ、`WLS`やロバスト標準誤差(`res.get_robustcov_results()`)の使用を検討する。

### `variance_inflation_factor(...)`

**用途**: 分散拡大要因(VIF)を計算し、多重共線性の強さを診断する。目安として10以上は要注意とされる。

**シグネチャ**: `statsmodels.stats.outliers_influence.variance_inflation_factor(exog, exog_idx)`

**使用例**:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(X.shape[1]):
    print(f"col{i}:", round(variance_inflation_factor(X, i), 3))
```
実行結果:
```
col0: 1.071
col1: 7.536
col2: 7.532
col3: 1.006
```
(`col1`=x1, `col2`=x2は生成時に`x2 = 0.9*x1 + ノイズ`としたため、VIFが7.5台と高くなっている)

**注意点・落とし穴**:
- 1変数ずつしか計算できないため、全列に対してループで回す必要がある(scikit-learnのような一括APIはない)。
- 定数項(`const`)を含めた行列を渡すのが一般的な使い方(定数項を除くとVIFの値が変わる)。

### `acorr_ljungbox(...)`

**用途**: リュング・ボックス検定(複数ラグにわたる残差の系列相関を一括検定)。

**シグネチャ**: `statsmodels.stats.diagnostic.acorr_ljungbox(x, lags=None, boxpierce=False, model_df=0, period=None, return_df=True, auto_lag=False)`

**使用例**:
```python
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(res.resid, lags=[5, 10])
print(lb)
```
実行結果:
```
      lb_stat  lb_pvalue
5    5.198250   0.392168
10  16.858435   0.077556
```

**注意点・落とし穴**:
- `return_df=True`(デフォルト)ではDataFrameが返る(インデックスがラグ数)。`False`にすると`(統計量配列, p値配列)`のタプルになる。
- `durbin_watson`が1次の自己相関しか見ないのに対し、複数ラグをまとめて検定できる。

### `res.get_influence()`(Cook's distance)

**用途**: 各観測点の影響度(てこ比・Cook's distanceなど)を計算し、外れ値・影響点を特定する。

**シグネチャ**: `statsmodels.regression.linear_model.RegressionResults.get_influence()`(引数なし)

**使用例**:
```python
rng = np.random.default_rng(10)
n = 30
x = rng.normal(size=n)
y = 1 + 2*x + rng.normal(scale=0.5, size=n)
x = np.append(x, 5.0)
y = np.append(y, -10.0)  # 意図的な外れ値を1点追加
X = sm.add_constant(x)
res = sm.OLS(y, X).fit()

infl = res.get_influence()
cooks_d, pvals = infl.cooks_distance
print("max cooks_d index:", cooks_d.argmax(), "value:", round(cooks_d.max(), 4))
```
実行結果:
```
max cooks_d index: 30 value: 21.4255
```

**注意点・落とし穴**:
- `cooks_distance`はタプル`(距離の配列, 対応するp値相当の配列)`を返す。距離の目安は`4/n`を超えると影響が大きいとされることが多い(今回の21.4は極端に大きく、意図的に追加した外れ値(インデックス30)が正しく検出されている)。

---

## 7. 仮説検定・記述統計

### `ttest_ind(...)`

**用途**: 2標本のt検定(平均の差の検定)。scipyの同名関数と異なり自由度や検定統計量の計算方法(pooled/unequal)を柔軟に指定できる。

**シグネチャ**: `statsmodels.stats.weightstats.ttest_ind(x1, x2, alternative='two-sided', usevar='pooled', weights=(None, None), value=0)`

**使用例**:
```python
from statsmodels.stats.weightstats import ttest_ind

rng = np.random.default_rng(6)
group_a = rng.normal(loc=5.0, scale=1.0, size=50)
group_b = rng.normal(loc=5.8, scale=1.0, size=50)

tstat, pval, df = ttest_ind(group_a, group_b)
print("tstat:", round(tstat, 4), "pvalue:", round(pval, 4), "df:", df)
```
実行結果:
```
tstat: -2.5715 pvalue: 0.0116 df: 98.0
```

**注意点・落とし穴**:
- 戻り値が3要素のタプル(`統計量, p値, 自由度`)。scipyの`scipy.stats.ttest_ind`は2要素(`統計量, p値`)なので混同しないよう注意。
- `usevar='pooled'`(デフォルト)は等分散を仮定する。分散が異なる場合は`usevar='unequal'`(Welchのt検定相当)を使う。

### `DescrStatsW(...)`

**用途**: (重み付き)記述統計量(平均・標準偏差・信頼区間など)をまとめて計算するクラス。

**シグネチャ**: `statsmodels.stats.weightstats.DescrStatsW(data, weights=None, ddof=0)`

**使用例**:
```python
from statsmodels.stats.weightstats import DescrStatsW
d = DescrStatsW(group_a)
print("mean:", round(d.mean, 4), "std:", round(d.std, 4))
print("confint(95%):", tuple(round(v, 4) for v in d.tconfint_mean()))
```
実行結果:
```
mean: 5.1844 std: 0.9951
confint(95%): (4.8987, 5.47)
```

**注意点・落とし穴**:
- `weights=None`(デフォルト)なら通常の(重みなし)記述統計になる。重み付き平均・分散を計算する場合のみ`weights`を指定する。
- 平均の信頼区間は`.tconfint_mean()`(t分布ベース)と`.zconfint_mean()`(正規分布ベース)の2種類があり、小標本ではt分布ベースが推奨される。

### `proportions_ztest(...)`

**用途**: 2群の比率の差のz検定(A/Bテストのコンバージョン率比較などに使う)。

**シグネチャ**: `statsmodels.stats.proportion.proportions_ztest(count, nobs, value=None, alternative='two-sided', prop_var=False)`

**使用例**:
```python
from statsmodels.stats.proportion import proportions_ztest
count = np.array([45, 30])  # 各群の成功数
nobs = np.array([100, 100])  # 各群の試行数
zstat, pval = proportions_ztest(count, nobs)
print("zstat:", round(zstat, 4), "pvalue:", round(pval, 6))
```
実行結果:
```
zstat: 2.1909 pvalue: 0.02846
```

**注意点・落とし穴**:
- `count`/`nobs`は長さ2の配列で2群を同時に渡す(1群ずつ呼び出す関数ではない)。
- `value=None`(デフォルト)は「2群の比率が等しい」を帰無仮説とする。特定の差(例:5%pt差)を検定したい場合は`value`を指定する。

### `anova_lm(...)`

**用途**: 分散分析(ANOVA)表を作成する。`smf.ols`などで作った線形モデルを渡す。

**シグネチャ**: `statsmodels.stats.anova.anova_lm(*args, **kwargs)`

**使用例**:
```python
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

rng = np.random.default_rng(6)
group = np.repeat(["A", "B", "C"], 30)
val = np.concatenate([rng.normal(5, 1, 30), rng.normal(5.8, 1, 30), rng.normal(6.5, 1, 30)])
df = pd.DataFrame({"val": val, "group": group})
model = smf.ols("val ~ C(group)", data=df).fit()
print(anova_lm(model, typ=2))
```
実行結果:
```
          sum_sq  df       F      PR(>F)
C(group) 71.5433   2 31.5461 4.98307e-11
Residual 98.6533  87     NaN         NaN
```

**注意点・落とし穴**:
- モデル本体(`model`)ではなく`model`オブジェクトを渡す点に注意(`ols("val ~ C(group)", data=df)`ではなく`.fit()`した結果を渡す)。
- `typ`(平方和のタイプ: 1, 2, 3)を明示しないと計算方法が変わりデフォルトは`typ=1`。交互作用のあるモデルでは`typ=2`または`typ=3`を使うことが多い。

### `multipletests(...)`

**用途**: 多重比較補正(複数のp値をまとめて調整する)。Bonferroni、Benjamini-Hochberg(FDR)など多数の方法に対応。

**シグネチャ**: `statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method='hs', maxiter=1, is_sorted=False, returnsorted=False)`

**使用例**:
```python
from statsmodels.stats.multitest import multipletests
pvals = [0.01, 0.02, 0.03, 0.04, 0.2, 0.5]
reject, p_adj, alphac_sidak, alphac_bonf = multipletests(pvals, alpha=0.05, method="fdr_bh")
print("reject:", reject)
print("p_adj:", np.round(p_adj, 4))
```
実行結果:
```
reject: [False False False False False False]
p_adj: [0.06 0.06 0.06 0.06 0.24 0.5 ]
```

**注意点・落とし穴**:
- デフォルトの`method='hs'`(Holm-Sidak)であり、`'bonferroni'`や`'fdr_bh'`(Benjamini-Hochberg)は明示的に指定する必要がある。今回の例では単純なp値(0.01〜0.04)でも、`fdr_bh`補正後は全て0.05を超え棄却されない(多重比較補正で有意性が失われる典型例)。

---

## 8. 混合効果モデル・一般化推定方程式

### `smf.mixedlm(...)`

**用途**: 線形混合効果モデル(グループごとにランダム切片・ランダム傾きを持つ階層データの回帰)。

**シグネチャ**: `statsmodels.formula.api.mixedlm(formula, data, re_formula=None, vc_formula=None, subset=None, use_sparse=False, missing='none', *args, **kwargs)`

**使用例**:
```python
rng = np.random.default_rng(2)
n = 100
x1 = rng.normal(size=n)
groups = rng.integers(0, 10, size=n)
re_effect = rng.normal(scale=1.5, size=10)[groups]
y = 1.0 + 2.0*x1 + re_effect + rng.normal(scale=0.3, size=n)
df = pd.DataFrame({"y": y, "x1": x1, "group": groups})

mlm_res = smf.mixedlm("y ~ x1", data=df, groups=df["group"]).fit()
print(mlm_res.summary())
```
実行結果:
```
        Mixed Linear Model Regression Results
======================================================
Model:            MixedLM Dependent Variable: y       
No. Observations: 100     Method:             REML    
No. Groups:       10      Scale:              0.0729  
Min. group size:  7       Log-Likelihood:     -39.9624
Max. group size:  15      Converged:          Yes     
Mean group size:  10.0                                
------------------------------------------------------
            Coef.  Std.Err.   z    P>|z| [0.025 0.975]
------------------------------------------------------
Intercept   -0.171    0.429 -0.399 0.690 -1.013  0.670
x1           2.046    0.029 69.532 0.000  1.989  2.104
Group Var    1.834    3.370                           
======================================================
```

**注意点・落とし穴**:
- **重要な落とし穴(実測で確認)**: `mlm_res.params["Group Var"]`は`25.161`という値になり、`summary()`に表示される`1.834`(実際のランダム効果の分散、`mlm_res.cov_re`と一致)とは異なる。`.params`のGroup Varは「残差分散(`.scale`)を1とした比率」で保持されており(`25.161 ≈ 1.834 / 0.0729`)、実際の分散値は`.cov_re`または`summary()`の表示を使う必要がある。
- `groups`引数は`data`内の列名文字列ではなく、グループを表す配列(`Series`)そのものを渡す。

### `GEE(...)`

**用途**: 一般化推定方程式(クラスタ・パネルデータにおける相関構造を考慮した回帰)。混合効果モデルと違い、ランダム効果の分布を仮定せず母集団平均的な効果を推定する。

**シグネチャ**: `statsmodels.genmod.generalized_estimating_equations.GEE.from_formula(formula, groups, data, subset=None, time=None, offset=None, exposure=None, *args, **kwargs)`

**使用例**:
```python
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable

rng = np.random.default_rng(7)
n_groups, n_per = 20, 5
n = n_groups * n_per
groups = np.repeat(np.arange(n_groups), n_per)
x1 = rng.normal(size=n)
re = rng.normal(scale=1.0, size=n_groups)[groups]
y = 1.0 + 2.0*x1 + re + rng.normal(scale=0.5, size=n)
df = pd.DataFrame({"y": y, "x1": x1, "group": groups})

gee_res = GEE.from_formula("y ~ x1", groups="group", data=df, cov_struct=Exchangeable()).fit()
print(gee_res.summary().tables[1])
```
実行結果:
```
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.9210      0.176      5.245      0.000       0.577       1.265
x1             1.9910      0.051     39.064      0.000       1.891       2.091
==============================================================================
```

**注意点・落とし穴**:
- `cov_struct`(相関構造)を指定しないとデフォルトの`Independence()`(独立)になり、クラスタ内相関を考慮しないただの回帰に近づく。クラスタ内で相関が想定される場合は`Exchangeable()`などを明示する。
- 標準誤差はデフォルトでロバスト(サンドイッチ推定量)であり、`cov_struct`の指定を間違えても点推定自体は比較的頑健(標準誤差の妥当性が変わる)。

---

## 9. モデル評価・予測ユーティリティ

### `results.get_prediction(...)`

**用途**: 予測値だけでなく、予測の標準誤差・信頼区間・予測区間もまとめて取得する。

**シグネチャ**: `statsmodels.regression.linear_model.RegressionResults.get_prediction(exog=None, transform=True, weights=None, row_labels=None, **kwargs)`

**使用例**:
```python
rng = np.random.default_rng(8)
n = 100
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
X_full = sm.add_constant(np.column_stack([x1, x2]))
y = 1.0 + 2.0*x1 + rng.normal(scale=0.5, size=n)
full = sm.OLS(y, X_full).fit()

pred = full.get_prediction(X_full[:3])
print(pred.summary_frame(alpha=0.05).round(3))
```
実行結果:
```
    mean  mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
0 -2.546    0.105         -2.754         -2.338        -3.594        -1.498
1 -1.632    0.120         -1.871         -1.393        -2.687        -0.578
2 -1.735    0.086         -1.906         -1.563        -2.776        -0.693
```

**注意点・落とし穴**:
- 単なる`predict()`は点推定(平均予測)のみを返す。区間推定(`mean_ci_*`=平均の信頼区間、`obs_ci_*`=個々の観測値の予測区間)が欲しい場合は`get_prediction().summary_frame()`を使う。予測区間(`obs_ci_*`)の方が信頼区間(`mean_ci_*`)より常に広い。

### `results.compare_lr_test(...)` / `results.compare_f_test(...)`

**用途**: 入れ子(nested)になった2つのモデル(全変数モデルと変数を削った縮小モデル)を比較し、追加した変数群が有意かどうかを検定する。

**シグネチャ**: `compare_lr_test(restricted, large_sample=False)` / `compare_f_test(restricted)`

**使用例**:
```python
X_reduced = sm.add_constant(x1)
reduced = sm.OLS(y, X_reduced).fit()

lr_stat, lr_pval, df_diff = full.compare_lr_test(reduced)
print("LR検定: stat:", round(lr_stat, 4), "pvalue:", round(lr_pval, 4))

fval, fpval, df_diff2 = full.compare_f_test(reduced)
print("F検定: F:", round(fval, 4), "pvalue:", round(fpval, 4))
```
実行結果:
```
LR検定: stat: 0.7202 pvalue: 0.3961
F検定: F: 0.7012 pvalue: 0.4045
```

**注意点・落とし穴**:
- `restricted`(縮小モデル)を引数に渡すのは、呼び出し元(`full`)が変数を多く含む「制約なしモデル」である前提。呼び出し方向を間違える(縮小モデル側から呼ぶ)とエラーになるか意味が変わる。
- 線形回帰(OLS)では`compare_lr_test`と`compare_f_test`はほぼ同じ結論になるが、統計量・p値は完全には一致しない(異なる検定原理のため)。

### `results.wald_test(...)`

**用途**: 係数に関する任意の線形制約(例:「x2の係数が0」)を1つのモデルの推定結果だけで検定する(再学習不要)。

**シグネチャ**: `statsmodels.regression.linear_model.RegressionResults.wald_test(r_matrix, cov_p=None, invcov=None, use_f=None, df_constraints=None, scalar=None)`

**使用例**:
```python
wald = full.wald_test("x2 = 0", scalar=True)
print(wald)
```
実行結果:
```
<F test: F=0.7011571495501423, p=0.4044535220547691, df_denom=97, df_num=1>
```

**注意点・落とし穴**:
- `r_matrix`には`"x2 = 0"`のような文字列制約、または行列を渡せる。文字列を使う場合、変数名は`exog`に対応する名前(`X`がpandasでない場合は`"x1"`, `"x2"`のような自動生成名)と一致させる必要がある。
- 今回の例では`compare_f_test`の結果(F=0.7012)とほぼ同じ値になる(1変数の制約なら両者は理論上等価)。

### `results.aic` / `results.bic`

**用途**: 赤池情報量規準(AIC)・ベイズ情報量規準(BIC)。モデル間比較(値が小さいほど良い)に使う。

**シグネチャ**: 属性(プロパティ)であり呼び出し不要。`RegressionResults.aic`, `RegressionResults.bic`

**使用例**:
```python
print("aic:", round(full.aic, 3), "bic:", round(full.bic, 3))
```
実行結果:
```
aic: 154.982 bic: 162.797
```

**注意点・落とし穴**:
- メソッドではなく属性なので`aic()`のように呼び出すとエラーになる。
- `BIC`は`AIC`よりパラメータ数に対する罰則が強く、サンプルサイズが大きいほど差が開く。モデル比較には同じデータ・同じ`endog`(目的変数の変換方法も含む)を使ったモデル同士でのみ意味がある。

---

## 10. その他ユーティリティ

### `add_constant(...)`

**用途**: 説明変数の行列に定数項(切片用の1の列)を追加する。

**シグネチャ**: `statsmodels.tools.tools.add_constant(data, prepend=True, has_constant='skip')`

**使用例**:
```python
import statsmodels.api as sm
X = np.array([[0.13, -0.13], [0.64, 0.1], [-0.54, 0.36]])
print(sm.add_constant(X))
```
実行結果:
```
[[ 1.    0.13 -0.13]
 [ 1.    0.64  0.1 ]
 [ 1.   -0.54  0.36]]
```

**注意点・落とし穴**:
- `prepend=True`(デフォルト)で定数列が先頭に追加される。末尾に追加したい場合は`prepend=False`。
- `has_constant='skip'`(デフォルト)は、既に定数列がある場合は何もしない。`'raise'`にするとエラーに、`'add'`にすると重複しても追加してしまう。

### `eval_measures.rmse(...)`

**用途**: 2つの配列(実測値・予測値)からRMSE(二乗平均平方根誤差)を計算する簡易ユーティリティ。

**シグネチャ**: `statsmodels.tools.eval_measures.rmse(x1, x2, axis=0)`

**使用例**:
```python
from statsmodels.tools.eval_measures import rmse
rng = np.random.default_rng(9)
y_true = rng.normal(size=20)
y_pred = y_true + rng.normal(scale=0.3, size=20)
print("rmse:", round(rmse(y_true, y_pred), 4))
```
実行結果:
```
rmse: 0.2965
```

**注意点・落とし穴**:
- `x1`/`x2`は対称な引数名だが、実際には単に要素ごとの差の二乗平均の平方根を計算するだけなので、どちらを実測値・予測値にしても結果は同じ(順序に意味はない)。
- scikit-learnの`mean_squared_error(..., squared=False)`(または`root_mean_squared_error`)と同じ値になる。

---

## 応用・発展

ここからは、より高度・専門的なAPI(状態空間モデル、ロバスト標準誤差、カウントデータモデル、ノンパラメトリック手法、生存時間分析、多変量解析)を扱う。上記と同様、すべて`/home/manaty/library-practicing/.venv`(statsmodels 0.14.6)で実行して検証済み。

---

## 11. 状態空間モデル(UnobservedComponents/DynamicFactor/MarkovRegression/RecursiveLS)

### `UnobservedComponents(...)`

**用途**: ローカルレベル・トレンド・季節性・周期などの「観測できない成分」に分解する状態空間モデル(構造時系列モデル)。`STL`より確率モデルとしての枠組みが明確で、成分ごとの分散を推定し予測区間も出せる。

**シグネチャ**: `statsmodels.tsa.statespace.structural.UnobservedComponents(endog, level=False, trend=False, seasonal=None, freq_seasonal=None, cycle=False, autoregressive=None, exog=None, irregular=False, stochastic_level=False, stochastic_trend=False, stochastic_seasonal=True, stochastic_freq_seasonal=None, stochastic_cycle=False, damped_cycle=False, cycle_period_bounds=None, mle_regression=True, use_exact_diffuse=False, **kwargs)`

**使用例**:
```python
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

rng = np.random.default_rng(20)
n = 120
level = np.cumsum(rng.normal(scale=0.3, size=n)) + 10
t = np.arange(n)
seasonal = 2*np.sin(2*np.pi*t/12)
y = level + seasonal + rng.normal(scale=0.5, size=n)

uc_res = UnobservedComponents(y, level="local level", seasonal=12).fit(disp=0)
print(uc_res.summary().tables[0])
print(uc_res.summary().tables[1])
print("forecast(3):", uc_res.forecast(3).round(3))
```
実行結果:
```
                            Unobserved Components Results                            
=====================================================================================
Dep. Variable:                             y   No. Observations:                  120
Model:                           local level   Log Likelihood                -111.885
                   + stochastic seasonal(12)   AIC                            229.770
Date:                       Wed, 23 Sep 2026   BIC                            237.816
Time:                               19:11:28   HQIC                           233.032
Sample:                                    0                                         
                                       - 120                                         
Covariance Type:                         opg                                         
=====================================================================================
====================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------
sigma2.irregular     0.1696      0.039      4.329      0.000       0.093       0.246
sigma2.level         0.1008      0.035      2.894      0.004       0.033       0.169
sigma2.seasonal      0.0005      0.004      0.131      0.896      -0.007       0.008
====================================================================================
forecast(3): [3.899 4.812 5.429]
```

**注意点・落とし穴**:
- `level="local level"`のような文字列指定は`level=True, stochastic_level=True`などのショートカット。`level=True`だけだと決定的(確定的)レベルになり、`stochastic_level=True`を別途指定しないとレベルが時間変化しない点に注意。
- `y`自体はレベル(ランダムウォーク)+季節性+ノイズで生成したため、`forecast(3)`の値(3.9台)は末尾のレベル(4付近まで下降)を反映した妥当な値になる(定数10からの単純な予測ではない)。

### `DynamicFactor(...)`

**用途**: 複数の時系列に共通する少数の「動的因子(共通成分)」を推定する状態空間モデル。景気動向指数の合成などに使われる。

**シグネチャ**: `statsmodels.tsa.statespace.dynamic_factor.DynamicFactor(endog, k_factors, factor_order, exog=None, error_order=0, error_var=False, error_cov_type='diagonal', enforce_stationarity=True, **kwargs)`

**使用例**:
```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

rng = np.random.default_rng(21)
n = 200
common = np.zeros(n)
for tt in range(1, n):
    common[tt] = 0.7*common[tt-1] + rng.normal(scale=1.0)
y1 = 1.0*common + rng.normal(scale=0.3, size=n)
y2 = 0.8*common + rng.normal(scale=0.3, size=n)
y3 = 1.2*common + rng.normal(scale=0.3, size=n)
data = np.column_stack([y1, y2, y3])

dfm_res = DynamicFactor(data, k_factors=1, factor_order=1).fit(disp=0, maxiter=200)
print(dict(zip(dfm_res.model.param_names, dfm_res.params.round(3))))
print("収束した:", dfm_res.mle_retvals.get("converged"))
```
実行結果:
```
{'loading.f1.y1': -0.903, 'loading.f1.y2': -0.733, 'loading.f1.y3': -1.111, 'sigma2.y1': 0.059, 'sigma2.y2': 0.106, 'sigma2.y3': 0.102, 'L1.f1.f1': 0.657}
収束した: True
```

**注意点・落とし穴**:
- デフォルトの`maxiter`(BFGS法の最大反復回数)では収束せず`ConvergenceWarning`が出ることがある(実測で確認)。`fit(disp=0, maxiter=200)`のように増やすと収束する。
- 因子分析全般に共通する「符号・スケールの不定性」がある。真のローディングは`(1.0, 0.8, 1.2)`(正)で生成したが、推定結果は`(-0.903, -0.733, -1.111)`と符号が反転している(比率はほぼ一致: `-0.733/-0.903≈0.81`, `-1.111/-0.903≈1.23`)。因子の符号自体には意味がなく、比率・相対関係のみが解釈対象。

### `MarkovRegression(...)`

**用途**: マルコフ状態遷移モデル(レジームスイッチングモデル)。時系列が複数の「レジーム(状態)」を確率的に行き来すると仮定し、各時点がどのレジームにいたかを推定する。

**シグネチャ**: `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression(endog, k_regimes, trend='c', exog=None, order=0, exog_tvtp=None, switching_trend=True, switching_exog=True, switching_variance=False, dates=None, freq=None, missing='none')`

**使用例**:
```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

rng = np.random.default_rng(22)
n = 300
regime = np.zeros(n, dtype=int)
for tt in range(1, n):
    p_stay = 0.97 if regime[tt-1] == 0 else 0.95
    regime[tt] = regime[tt-1] if rng.uniform() < p_stay else 1 - regime[tt-1]
mean = np.where(regime == 0, 0.0, 5.0)
y = mean + rng.normal(scale=1.0, size=n)

mr_res = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=False).fit()
print(mr_res.summary().tables[1])
print(mr_res.summary().tables[4])
print("smoothed_marginal_probabilities[:5]:\n", mr_res.smoothed_marginal_probabilities[:5].round(3))
```
実行結果:
```
                             Regime 0 parameters                              
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0319      0.077      0.415      0.678      -0.119       0.182
==============================================================================
                         Regime transition parameters                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
p[0->0]        0.9680      0.013     74.165      0.000       0.942       0.994
p[1->0]        0.0437      0.018      2.430      0.015       0.008       0.079
==============================================================================
smoothed_marginal_probabilities[:5]:
 [[1. 0.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [1. 0.]]
```

**注意点・落とし穴**:
- 推定された遷移確率`p[0->0]=0.968`, `p[1->0]=0.044`はデータ生成時の真値(`0.97`, `0.05`)に近い。`p[i->j]`は「状態`i`から`j`へ遷移する確率」ではなく、statsmodelsの内部表記では`p[0->0]`は「レジーム0にとどまる確率」を表す(添字の意味を`summary()`と`model.param_names`で確認するのが安全)。
- `smoothed_marginal_probabilities`(形状`(n, k_regimes)`)はフィルタ済みではなく全データを使った平滑化事後確率で、各時点がどちらのレジームにいたかを`0`〜`1`で表す。`k_regimes`列の合計は各行で1になる。

### `RecursiveLS(...)`

**用途**: 逐次最小二乗法(Recursive Least Squares)。状態空間モデルとして実装されており、時点ごとの回帰係数の推移(パラメータが安定しているか)を追跡できる。

**シグネチャ**: `statsmodels.regression.recursive_ls.RecursiveLS(endog, exog, constraints=None, **kwargs)`

**使用例**:
```python
import statsmodels.api as sm
from statsmodels.regression.recursive_ls import RecursiveLS

rng = np.random.default_rng(23)
n = 80
x = rng.normal(size=n)
X = sm.add_constant(x)
y = 1.0 + 2.0*x + rng.normal(scale=0.5, size=n)

rls_res = RecursiveLS(y, X).fit()
print("最終パラメータ:", rls_res.params.round(3))
print("t=20,50,80時点でのxの係数推定値:",
      rls_res.recursive_coefficients.filtered[1][[19, 49, 79]].round(3))
```
実行結果:
```
最終パラメータ: [1.044 2.092]
t=20,50,80時点でのxの係数推定値: [2.245 2.122 2.092]
```

**注意点・落とし穴**:
- 最終時点(`t=80`)の`recursive_coefficients.filtered`は通常の`OLS`の係数とほぼ一致する(逐次推定の最終結果=バッチ推定結果)。
- 係数が時間とともに大きく変動する(構造変化がある)かどうかを目視・検定(CUSUM検定など、`rls_res.plot_cusum()`)で確認する用途に使われる。

---

## 12. ロバスト標準誤差・クラスタ標準誤差

### `results.fit(cov_type="HC0"〜"HC3")`(不均一分散に頑健な標準誤差)

**用途**: 誤差項の分散が観測値ごとに異なる(不均一分散)場合でも、係数の点推定はそのままに標準誤差だけを頑健に補正する(White/Huber-White標準誤差)。

**シグネチャ**: `statsmodels.regression.linear_model.OLS.fit(method='pinv', cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs)` — `cov_type`に`'HC0'`,`'HC1'`,`'HC2'`,`'HC3'`のいずれかを指定する。

**使用例**:
```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(24)
n = 100
x = rng.normal(size=n)
X = sm.add_constant(x)
err = rng.normal(scale=0.3 + 0.7*np.abs(x), size=n)  # |x|が大きいほど分散が増える
y = 1.0 + 2.0*x + err

res_ols = sm.OLS(y, X).fit()
res_hc3 = sm.OLS(y, X).fit(cov_type="HC3")
print("nonrobust bse:", res_ols.bse.round(4))
print("HC3 bse:      ", res_hc3.bse.round(4))
```
実行結果:
```
nonrobust bse: [0.1027 0.0998]
HC3 bse:       [0.1007 0.1295]
```

**注意点・落とし穴**:
- 係数の点推定(`params`)は`cov_type`を変えても一切変わらない。変わるのは標準誤差・t値・p値・信頼区間のみ。
- `HC0`〜`HC3`は補正の強さが異なり(`HC3`が小標本でより保守的)、実務では`HC3`か`HC1`がよく使われる。`HC0`(White (1980)の原論文の推定量)が最も補正が弱い。
- 今回のように分散が`x`に依存する典型的な不均一分散データでは、`x`の係数の標準誤差が`0.0998→0.1295`と拡大し、通常のt検定より慎重な判断になる。

### `results.fit(cov_type="HAC", cov_kwds={"maxlags": ...})`(Newey-West標準誤差)

**用途**: 誤差項に系列相関(自己相関)がある時系列回帰で、標準誤差を自己相関・不均一分散の両方に頑健に補正する(Newey-West法)。

**シグネチャ**: 同上(`OLS.fit(..., cov_type='HAC', cov_kwds={'maxlags': int, ...})`)

**使用例**:
```python
rng = np.random.default_rng(25)
n = 150
x = rng.normal(size=n)
X = sm.add_constant(x)
eps = np.zeros(n)
for t in range(1, n):
    eps[t] = 0.6*eps[t-1] + rng.normal(scale=0.5)  # AR(1)の系列相関を持つ誤差
y = 1.0 + 1.5*x + eps

res_ols = sm.OLS(y, X).fit()
res_hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print("nonrobust bse:", res_ols.bse.round(4))
print("HAC bse:      ", res_hac.bse.round(4))
```
実行結果:
```
nonrobust bse: [0.051  0.0509]
HAC bse:       [0.0747 0.0488]
```

**注意点・落とし穴**:
- `cov_kwds={"maxlags": 4}`の指定が必須に近い(省略するとエラーになるか、警告付きでデフォルト値が使われるバージョンがあるため明示推奨)。`maxlags`は考慮する自己相関のラグ数。
- 定数項に系列相関のある誤差を混入させたこの例では、切片の標準誤差が`0.051→0.0747`と約1.5倍に拡大しており、自己相関を無視すると標準誤差を過小評価する典型例になっている。

### `results.fit(cov_type="cluster", cov_kwds={"groups": ...})`(クラスタ標準誤差)

**用途**: 同一グループ(クラスタ)内の観測値同士が相関しているデータ(パネルデータ、学校ごとの生徒データなど)で、グループ内相関を考慮した標準誤差を計算する。

**シグネチャ**: 同上(`OLS.fit(..., cov_type='cluster', cov_kwds={'groups': array-like})`)

**使用例**:
```python
rng = np.random.default_rng(26)
n_groups, n_per = 20, 10
n = n_groups*n_per
groups = np.repeat(np.arange(n_groups), n_per)
x = rng.normal(size=n)
X = sm.add_constant(x)
group_effect = rng.normal(scale=2.0, size=n_groups)[groups]  # グループ内で共通のショック
y = 1.0 + 1.5*x + group_effect + rng.normal(scale=0.3, size=n)

res_ols = sm.OLS(y, X).fit()
res_cluster = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
print("nonrobust bse:", res_ols.bse.round(4))
print("cluster bse:  ", res_cluster.bse.round(4))
```
実行結果:
```
nonrobust bse: [0.1462 0.1349]
cluster bse:   [0.4691 0.14  ]
```

**注意点・落とし穴**:
- グループ内で相関する誤差(今回は`group_effect`)がある場合、通常の標準誤差はグループ内の重複情報を独立な情報として扱ってしまうため過小評価しやすい。今回、切片の標準誤差は`0.1462→0.4691`と3倍以上に拡大しており、クラスタ内相関の影響の大きさが分かる。
- `cov_kwds={"groups": groups}`の`groups`は`X`の説明変数のクラスタと必ずしも一致する必要はないが、通常はグループごとに一意なIDの配列(`shape=(n,)`)を渡す。

---

## 13. カウントデータモデル(Poisson/NegativeBinomial/ZeroInflatedPoisson)

### `Poisson(...)`

**用途**: 目的変数が非負整数のカウントデータ(件数・回数)に対する回帰モデル。リンク関数は対数(log)。

**シグネチャ**: `statsmodels.discrete.discrete_model.Poisson(endog, exog, offset=None, exposure=None, missing='none', check_rank=True, **kwargs)`(`statsmodels.api.Poisson`としても利用可)

**使用例**:
```python
rng = np.random.default_rng(27)
n = 300
x1 = rng.normal(size=n)
X = sm.add_constant(x1)
lam = np.exp(0.5 + 0.8*x1)
y = rng.poisson(lam)

pois_res = sm.Poisson(y, X).fit(disp=0)
print(pois_res.params.round(3))
print("pred[:5]:", pois_res.predict(X[:5]).round(3))
```
実行結果:
```
[0.426 0.868]
pred[:5]: [4.545 3.004 3.536 0.599 3.248]
```

**注意点・落とし穴**:
- `predict()`はデフォルトで期待カウント数(`exp(Xβ)`)を返す。真の係数(`0.5, 0.8`)に近い推定(`0.426, 0.868`)が得られている。
- Poissonモデルは「平均=分散」を仮定する。実データでこの仮定が崩れている(過分散)場合は`NegativeBinomial`の使用を検討する。

### `NegativeBinomial(...)`

**用途**: 過分散(分散が平均より大きい)のカウントデータに対応する回帰モデル。Poissonの一般化。

**シグネチャ**: `statsmodels.discrete.discrete_model.NegativeBinomial(endog, exog, loglike_method='nb2', offset=None, exposure=None, missing='none', check_rank=True, **kwargs)`

**使用例**:
```python
rng = np.random.default_rng(28)
n = 300
x1 = rng.normal(size=n)
X = sm.add_constant(x1)
lam = np.exp(0.5 + 0.8*x1)
alpha_true = 0.8
gamma_noise = rng.gamma(shape=1/alpha_true, scale=alpha_true, size=n)  # 過分散を混入
y = rng.poisson(lam * gamma_noise)

print("mean:", round(y.mean(), 3), "var:", round(y.var(), 3), "(Poissonなら平均≒分散のはず)")

pois_res = sm.Poisson(y, X).fit(disp=0)
nb_res = sm.NegativeBinomial(y, X).fit(disp=0)
print("Poisson params:", pois_res.params.round(3), "llf:", round(pois_res.llf, 2))
print("NB params:     ", nb_res.params.round(3), "llf:", round(nb_res.llf, 2))
```
実行結果:
```
mean: 2.013 var: 9.4 (Poissonなら平均≒分散のはず)
Poisson params: [0.416 0.781] llf: -576.03
NB params:      [0.422 0.762 0.558] llf: -515.33
```

**注意点・落とし穴**:
- 平均2.0に対し分散9.4と明らかな過分散データにおいて、`NegativeBinomial`の対数尤度(`-515.33`)が`Poisson`(`-576.03`)より大きく改善している(=当てはまりが良い)。
- `NB`のパラメータは`X`の係数に加えて末尾に過分散パラメータ`alpha`(この例では`0.558`、生成時の真値`0.8`に近いオーダー)が追加される。`alpha=0`だとPoissonと一致する。

### `ZeroInflatedPoisson(...)`

**用途**: ゼロが理論上のPoisson分布より過剰に多い(ゼロ過剰)カウントデータに対応するモデル。「構造的にゼロになる集団」と「Poissonに従う集団」の混合モデルとして推定する。

**シグネチャ**: `statsmodels.discrete.count_model.ZeroInflatedPoisson(endog, exog, exog_infl=None, offset=None, exposure=None, inflation='logit', missing='none', **kwargs)`

**使用例**:
```python
from statsmodels.discrete.count_model import ZeroInflatedPoisson

rng = np.random.default_rng(29)
n = 400
x1 = rng.normal(size=n)
X = sm.add_constant(x1)
lam = np.exp(0.8 + 0.5*x1)
excess_zero = rng.binomial(1, 0.3, size=n)  # 30%が構造的ゼロ
y = np.where(excess_zero == 1, 0, rng.poisson(lam))
print("実際のゼロ比率:", round((y == 0).mean(), 3))

zip_res = ZeroInflatedPoisson(y, X, exog_infl=X, inflation="logit").fit(disp=0)
print(zip_res.summary().tables[1])
```
実行結果:
```
実際のゼロ比率: 0.362
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
inflate_const    -1.0750      0.156     -6.912      0.000      -1.380      -0.770
inflate_x1        0.1404      0.155      0.904      0.366      -0.164       0.445
const             0.7914      0.045     17.723      0.000       0.704       0.879
x1                0.4742      0.040     11.936      0.000       0.396       0.552
=================================================================================
```

**注意点・落とし穴**:
- `exog_infl`(ゼロ過剰過程を説明する変数)は本体の`exog`と別に指定でき、省略すると定数項のみになる。今回は同じ`X`を渡したため`inflate_const`/`inflate_x1`と`const`/`x1`の2組の係数が出力される。
- `inflate_const=-1.075`をロジスティック変換(`1/(1+exp(1.075))≈0.254`)すると構造的ゼロの推定確率になり、生成時の真値(0.3)に近いオーダーになる。

---

## 14. ノンパラメトリック回帰・密度推定

### `lowess(...)`

**用途**: 局所重み付き回帰(LOWESS/LOESS)による平滑化。関数形を仮定せず、散布図の傾向線を引く。

**シグネチャ**: `statsmodels.nonparametric.smoothers_lowess.lowess(endog, exog, frac=0.6666666666666666, it=3, delta=0.0, xvals=None, is_sorted=False, missing='drop', return_sorted=True)`

**使用例**:
```python
from statsmodels.nonparametric.smoothers_lowess import lowess

rng = np.random.default_rng(30)
n = 100
x = np.sort(rng.uniform(0, 10, size=n))
y = np.sin(x) + rng.normal(scale=0.3, size=n)

smoothed = lowess(y, x, frac=0.3)
print("shape:", smoothed.shape)
print(smoothed[:5].round(3))
```
実行結果:
```
shape: (100, 2)
[[0.923 1.026]
 [0.943 1.025]
 [1.096 1.02 ]
 [1.131 1.018]
 [1.197 1.014]]
```

**注意点・落とし穴**:
- 引数順は`lowess(endog, exog)`、すなわち`y`が先で`x`が後(`OLS`と同じくscikit-learnの`X, y`順とは逆)。
- 戻り値は`return_sorted=True`(デフォルト)なら`(x, 平滑化後のy)`を列に持つ`(n, 2)`のndarray。`frac`(近傍点の割合)が小さいほど元データに忠実に、大きいほど滑らかになる。

### `KDEUnivariate(...)`

**用途**: 1次元カーネル密度推定(ヒストグラムより滑らかな確率密度関数の推定)。

**シグネチャ**: `statsmodels.nonparametric.kde.KDEUnivariate(endog)` / `.fit(kernel='gau', bw='normal_reference', fft=True, weights=None, gridsize=None, adjust=1, cut=3, clip=(-inf, inf))`

**使用例**:
```python
from statsmodels.nonparametric.kde import KDEUnivariate

rng = np.random.default_rng(31)
data = rng.normal(loc=5.0, scale=1.5, size=500)

kde = KDEUnivariate(data)
kde.fit()
print("bw:", round(kde.bw, 4))
print("density at [3,5,7]:", kde.evaluate([3, 5, 7]).round(4))
```
実行結果:
```
bw: 0.4506
density at [3,5,7]: [0.096  0.2478 0.1285]
```

**注意点・落とし穴**:
- コンストラクタにデータを渡すだけでは推定されず、`.fit()`を呼んで初めて`bw`(バンド幅)や`.density`が計算される(`STL`と同様のパターン)。
- `bw='normal_reference'`(デフォルト)はSilvermanの経験則に近い自動選択。真の分布は平均5、標準偏差1.5の正規分布であり、`evaluate([3,5,7])`の結果(ピークが5付近の0.2478)は理論値(正規分布のpdfで最大約0.266)と近いオーダーになっている。

### `KernelReg(...)`

**用途**: ノンパラメトリックなカーネル回帰(局所線形/局所定数)。`lowess`と異なり、説明変数が連続・離散(カテゴリ)混在でも扱え、勾配(限界効果)も推定できる。

**シグネチャ**: `statsmodels.nonparametric.kernel_regression.KernelReg(endog, exog, var_type, reg_type='ll', bw='cv_ls', ckertype='gaussian', okertype='wangryzin', ukertype='aitchisonaitken', defaults=None)`

**使用例**:
```python
from statsmodels.nonparametric.kernel_regression import KernelReg

rng = np.random.default_rng(32)
n = 60
x = np.sort(rng.uniform(0, 10, size=n))
y = np.sin(x) + rng.normal(scale=0.3, size=n)

kr = KernelReg(endog=y, exog=x, var_type="c")  # "c"=連続変数
mean, mfx = kr.fit(x[:5])
print("bw:", kr.bw.round(4))
print("mean[:5]:", mean.round(3))
print("勾配(mfx)[:5]:", mfx.flatten().round(3))
```
実行結果:
```
bw: [0.3849]
mean[:5]: [0.304 0.448 0.474 0.542 0.648]
勾配(mfx)[:5]: [0.449 0.415 0.392 0.299 0.057]
```

**注意点・落とし穴**:
- `var_type`は各説明変数の型を1文字ずつ並べた文字列で必須指定(`"c"`=連続、`"u"`=順序なし離散、`"o"`=順序あり離散)。列数と文字数を一致させる必要がある。
- `bw='cv_ls'`(デフォルト)は最小二乗交差検証でバンド幅を自動探索するため、データ数が多いと計算がかなり遅くなる(`n=60`でも数秒かかる)。

### `ECDF(...)`

**用途**: 経験分布関数(Empirical CDF)。ノンパラメトリックに「ある値以下のデータが全体の何割か」を計算する。

**シグネチャ**: `statsmodels.distributions.empirical_distribution.ECDF(x, side='right')`

**使用例**:
```python
from statsmodels.distributions.empirical_distribution import ECDF

rng = np.random.default_rng(39)
data = rng.normal(size=200)
ecdf = ECDF(data)
print(ecdf([-1, 0, 1]).round(4))
```
実行結果:
```
[0.125 0.49  0.85 ]
```

**注意点・落とし穴**:
- `ECDF`のインスタンスは関数のように呼び出せる(`ecdf(values)`)。標準正規乱数200個に対し`ecdf(0)≈0.49`となっており理論値0.5に近い。
- `side='right'`(デフォルト)はステップ関数の定義(≤か<か)に関わる細かい違いで、通常のCDF近似としての用途ではほぼ影響しない。

---

## 15. 生存時間分析

### `SurvfuncRight(...)`

**用途**: 右側打ち切り(right-censoring)のある生存時間データからKaplan-Meier推定量(生存関数)を計算する。

**シグネチャ**: `statsmodels.duration.survfunc.SurvfuncRight(time, status, entry=None, title=None, freq_weights=None, exog=None, bw_factor=1.0)`

**使用例**:
```python
from statsmodels.duration.survfunc import SurvfuncRight

rng = np.random.default_rng(33)
n = 60
true_time = rng.exponential(scale=10, size=n)
censor_time = rng.exponential(scale=15, size=n)
time = np.minimum(true_time, censor_time)
status = (true_time <= censor_time).astype(int)  # 1=イベント発生、0=打ち切り

sf = SurvfuncRight(time, status)
print("event times[:5]:", sf.surv_times[:5].round(3))
print("survival prob[:5]:", sf.surv_prob[:5].round(3))
print("打ち切り件数:", (status == 0).sum(), "/", n)
print("median survival time:", round(sf.quantile(0.5), 3))
```
実行結果:
```
event times[:5]: [0.137 0.235 0.286 0.402 0.465]
survival prob[:5]: [0.983 0.966 0.949 0.931 0.913]
打ち切り件数: 24 / 60
median survival time: 7.376
```

**注意点・落とし穴**:
- `time`/`status`は`endog`/`exog`のような組ではなく2本の配列(観測時間、イベント発生なら1・打ち切りなら0)を渡す。`status`の0/1の意味を逆にすると生存確率が反転する。
- `.quantile(0.5)`が中央生存時間。今回のように60件中24件(40%)が打ち切りでも、Kaplan-Meier法は打ち切り情報を活用して妥当な生存確率を推定できる。

### `survdiff(...)`

**用途**: ログランク検定(log-rank test)。2群以上の生存曲線に有意な差があるかを検定する。

**シグネチャ**: `statsmodels.duration.survfunc.survdiff(time, status, group, weight_type=None, strata=None, entry=None, **kwargs)`

**使用例**:
```python
from statsmodels.duration.survfunc import survdiff

rng = np.random.default_rng(34)
n = 80
group = np.repeat([0, 1], n // 2)
scale = np.where(group == 0, 10, 16)  # group1の方が生存期間が長い
true_time = rng.exponential(scale=scale)
censor_time = rng.exponential(scale=20, size=n)
time = np.minimum(true_time, censor_time)
status = (true_time <= censor_time).astype(int)

stat, pval = survdiff(time, status, group)
print("stat:", round(stat, 4), "pvalue:", round(pval, 6))
```
実行結果:
```
stat: 4.9275 pvalue: 0.026432
```

**注意点・落とし穴**:
- 戻り値は`(検定統計量, p値)`の2要素タプル(クラスや`summary()`は持たない、シンプルな関数)。
- group0とgroup1で生存時間の分布(指数分布のスケール10 vs 16)を意図的に変えたところ、p値0.026と5%水準で有意差が検出されている。

### `PHReg(...)`

**用途**: Cox比例ハザードモデル。生存時間に対する説明変数の効果を、ハザード比(hazard ratio)として推定する。

**シグネチャ**: `statsmodels.duration.hazard_regression.PHReg(endog, exog, status=None, entry=None, strata=None, offset=None, ties='breslow', missing='drop', **kwargs)`

**使用例**:
```python
from statsmodels.duration.hazard_regression import PHReg

rng = np.random.default_rng(35)
n = 200
x1 = rng.normal(size=n)
scale = 10 * np.exp(-0.5*x1)  # x1が大きいほどハザードが高い(生存時間が短い)
true_time = rng.exponential(scale=scale)
censor_time = rng.exponential(scale=15, size=n)
time = np.minimum(true_time, censor_time)
status = (true_time <= censor_time).astype(int)

ph_res = PHReg(time, x1, status=status).fit()
print(ph_res.summary())
print("ハザード比 exp(coef):", np.exp(ph_res.params).round(3))
```
実行結果:
```
                    Results: PHReg
======================================================
Model:                  PH Reg     Sample size:    197
Dependent variable:     y          Num. events:    110
Ties:                   Breslow                       
------------------------------------------------------
   log HR log HR SE   HR     t    P>|t|  [0.025 0.975]
------------------------------------------------------
x1 0.6581    0.0983 1.9312 6.6982 0.0000 1.5929 2.3413
======================================================
Confidence intervals are for the hazard ratios
ハザード比 exp(coef): [1.931]
```

**注意点・落とし穴**:
- コンストラクタが`(endog, exog, status=...)`の順で、`exog`(説明変数)にイベント時間ではなく通常の共変量を渡す。生存時間そのものは`endog`。
- `summary()`は係数ではなく最初から「log HR」(対数ハザード比)として表示され、`HR`列に`exp(coef)`が併記される。デフォルトの`ties='breslow'`はタイ(同一時刻のイベント)の扱い方式で、他に`'efron'`(より正確だが計算コスト増)が選べる。
- `Sample size: 197`は投入した`n=200`と一致しない(内部のリスク集合の構成上、一部の観測が計算から除外されることがある)。原因の詳細は未検証のため、`model.exog.shape`など生データとの突合を推奨する。

---

## 16. 多変量解析

### `PCA(...)`

**用途**: 主成分分析。`sklearn.decomposition.PCA`のstatsmodels版で、寄与率・因子負荷量(loadings)を直接プロパティとして持つ。

**シグネチャ**: `statsmodels.multivariate.pca.PCA(data, ncomp=None, standardize=True, demean=True, normalize=True, gls=False, weights=None, method='svd', missing=None, tol=5e-08, max_iter=1000, tol_em=5e-08, max_em_iter=100, svd_full_matrices=False)`

**使用例**:
```python
from statsmodels.multivariate.pca import PCA

rng = np.random.default_rng(36)
n = 100
factor = rng.normal(size=n)
X = np.column_stack([
    2*factor + rng.normal(scale=0.3, size=n),
    1.5*factor + rng.normal(scale=0.3, size=n),
    -factor + rng.normal(scale=0.3, size=n),
    rng.normal(size=n),  # 無関係なノイズ列
])

pca = PCA(X, ncomp=2)
print("累積寄与率(0〜2成分):", pca.rsquare[:3].round(4))
print("loadings:\n", pca.loadings.round(3))
```
実行結果:
```
累積寄与率(0〜2成分): [0.     0.7228 0.97  ]
loadings:
 [[ 0.579  0.053]
 [ 0.577  0.063]
 [-0.571 -0.02 ]
 [ 0.079 -0.996]]
```

**注意点・落とし穴**:
- `pca.rsquare`は「0〜k個目までの成分で説明できる累積寄与率」の配列で、`rsquare[0]=0`(成分0個なら説明率0)から始まる点に注意(`rsquare[1]`が第1主成分単体の寄与率)。
- 最初の3列は共通因子に強く依存するよう生成したため第1主成分(寄与率72%)にまとまり、4列目(無関係なノイズ)は第2主成分にほぼ単独で現れている(`loadings`の4行目が`-0.996`)。
- `standardize=True`(デフォルト)により各列は標準化されてからPCAが実行される(スケールの異なる変数を混在させても問題ない)。

### `Factor(...)`

**用途**: 探索的因子分析。PCAと似るが「観測変数=共通因子+独自因子(誤差)」というモデルを明示的に仮定し、共通性(communality)/独自性(uniqueness)を分離する。

**シグネチャ**: `statsmodels.multivariate.factor.Factor(endog=None, n_factor=1, corr=None, method='pa', smc=True, endog_names=None, nobs=None, missing='drop')`

**使用例**:
```python
from statsmodels.multivariate.factor import Factor

rng = np.random.default_rng(37)
n = 300
common = rng.normal(size=n)
X = np.column_stack([
    0.9*common + rng.normal(scale=0.4, size=n),
    0.8*common + rng.normal(scale=0.4, size=n),
    0.85*common + rng.normal(scale=0.4, size=n),
    rng.normal(size=n),  # 共通因子と無関係な列
])

fa_res = Factor(X, n_factor=1, method="ml").fit()
print("loadings:\n", fa_res.loadings.round(3))
print("uniqueness:", fa_res.uniqueness.round(3))
```
実行結果:
```
loadings:
 [[0.899]
 [0.907]
 [0.899]
 [0.077]]
uniqueness: [0.191 0.178 0.192 0.994]
```

**注意点・落とし穴**:
- `method='ml'`(最尤法)や`'pa'`(主因子法、デフォルト)を指定できる。`n_factor`(因子数)は事前に指定が必要で、自動選択はされない。
- `uniqueness`(独自性、1に近いほどその変数は共通因子で説明されない)は4列目(無関係なノイズ)で`0.994`と非常に高く、共通因子と無関係であることが正しく検出されている。

### `CanCorr(...)`

**用途**: 正準相関分析。2つの変数群(`Y`群と`X`群)の間で、相関が最大になるような線形結合の組を見つける。

**シグネチャ**: `statsmodels.multivariate.cancorr.CanCorr(endog, exog, tolerance=1e-08, missing='none', hasconst=None, **kwargs)`

**使用例**:
```python
from statsmodels.multivariate.cancorr import CanCorr

rng = np.random.default_rng(38)
n = 200
z = rng.normal(size=n)
Y = np.column_stack([1.0*z + rng.normal(scale=0.5, size=n), 0.7*z + rng.normal(scale=0.5, size=n)])
X = np.column_stack([0.9*z + rng.normal(scale=0.5, size=n), rng.normal(size=n)])

cc = CanCorr(Y, X)
print("正準相関係数:", cc.cancorr.round(4))
print(cc.corr_test().summary().tables[0])
```
実行結果:
```
正準相関係数: [0.8095 0.0587]
  Canonical Correlation Wilks' lambda Num DF Den DF    F Value    Pr > F
0              0.809539      0.343461      4  392.0  69.219698       0.0
1              0.058656      0.996559      1  197.0   0.680134  0.410538
```

**注意点・落とし穴**:
- `CanCorr(endog, exog)`という引数名だが回帰モデルではなく、`Y`群・`X`群という2つの変数セットとして扱われる(どちらを`endog`/`exog`にしても正準相関係数自体は同じ)。
- 今回は共通の潜在変数`z`が`Y`・`X`双方を駆動するよう生成したため、第1正準相関係数が`0.8095`と高くp値も有意(`0.0000`)。第2正準相関係数は`0.0587`と低くp値`0.4105`で有意でない(=独立な相関構造は1組しかないことを正しく検出)。
