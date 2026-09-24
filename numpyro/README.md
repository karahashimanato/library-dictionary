# NumPyro 逆引き辞書

NumPyro 0.21.0 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv/bin/python`(NumPyro 0.21.0、JAX 0.11.1、CPU実行、ArviZ 1.3.0)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。MCMC・SVIの数値(事後平均など)は乱数キーを固定していても、実行環境(OS・BLAS・JAXのバージョン等)によって末尾の桁がわずかに変動しうる点に留意すること。JAX自体の基本(`jax.random`・`jit`・`vmap`など)は[../jax/README.md](../jax/README.md)、姉妹ライブラリのPyMCは[../pymc/README.md](../pymc/README.md)を参照。

## 目次

1. [モデル記述の基礎](#1-モデル記述の基礎)
2. [確率分布](#2-確率分布)
3. [MCMCサンプリング](#3-mcmcサンプリング)
4. [変分推論(SVI)](#4-変分推論svi)
5. [予測とログ尤度](#5-予測とログ尤度)
6. [階層モデルと時系列](#6-階層モデルと時系列)
7. [制約と変数変換](#7-制約と変数変換)
8. [エフェクトハンドラ](#8-エフェクトハンドラ)
9. [ArviZ連携](#9-arviz連携)
10. [その他ユーティリティ](#10-その他ユーティリティ)

---

## 1. モデル記述の基礎

NumPyroのモデルは「`numpyro.sample`等を呼ぶ普通のPython関数」として書く。乱数を消費するサイトは推論エンジン(MCMC/SVI/Predictive)や`handlers.seed`が乱数キーを供給する。この章の例は次の共通コードを前提とする。

**共通コード**:
```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
```

### `numpyro.sample(...)`

**用途**: 確率変数(サンプルサイト)を宣言する。`obs=`を渡すと観測データとして扱われ、尤度の計算に使われる。

**シグネチャ**: `numpyro.sample(name, fn, obs=None, rng_key=None, sample_shape=(), infer=None, obs_mask=None)`

**使用例**:
```python
def model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)
    return mu

print(handlers.seed(model, rng_seed=0)())   # seedハンドラで乱数キーを供給して呼ぶ
try:
    model()                                  # seedなしで直接呼ぶ
except Exception as e:
    print(type(e).__name__, repr(str(e)))

tr = handlers.trace(handlers.seed(model, 0)).get_trace(y=jnp.array([0.5, 1.5]))
for name, site in tr.items():
    print(name, site["type"], site["is_observed"], site["value"])

def model_shape():
    return numpyro.sample("z", dist.Normal(0, 1), sample_shape=(3,))
print(handlers.seed(model_shape, 0)().shape)
```
実行結果:
```
-2.4424558
AssertionError ''
mu sample False -2.4424558
y sample True [0.5 1.5]
(3,)
```

**注意点・落とし穴**:
- 推論エンジンの外(素の関数呼び出し)では乱数キーが無いため、`numpyro.sample`は`AssertionError`(メッセージは空)になる。モデルを単体で動かして確認したいときは`handlers.seed(model, rng_seed=0)`で包む。
- `obs`を与えたサイトは`is_observed=True`になり、MCMC/SVIでは潜在変数として推定されない。`obs=None`のときは通常の確率変数(事前分布・予測に使われる)なので、同じモデルを学習にも予測にも使い回せる。
- `sample_shape`を指定すると返り値の形が`sample_shape + batch_shape`になる(上の例は`(3,)`)。

### `numpyro.sample(..., obs_mask=...)`

**用途**: 観測ベクトルの一部だけが欠損しているとき、`obs_mask`が`False`の要素を潜在変数として自動的に補完(推論)する。

**使用例**:
```python
def model_missing(y):
    numpyro.sample("y", dist.Normal(0.0, 1.0).expand([4]),
                   obs=y, obs_mask=jnp.array([True, False, True, False]))

tr = handlers.trace(handlers.seed(model_missing, 0)).get_trace(jnp.array([1.0, 2.0, 3.0, 4.0]))
for name, site in tr.items():
    print(name, site["type"], site["value"])

from numpyro.infer import MCMC, NUTS
mcmc = MCMC(NUTS(model_missing), num_warmup=50, num_samples=100, progress_bar=False)
mcmc.run(jax.random.key(0), jnp.array([1.0, 2.0, 3.0, 4.0]))
print({k: v.shape for k, v in mcmc.get_samples().items()})
```
実行結果:
```
y_observed sample [1. 2. 3. 4.]
y_unobserved sample [-2.4424558  -2.0356805   0.20554423 -0.3535502 ]
y deterministic [ 1.        -2.0356805  3.        -0.3535502]
{'y': (100, 4), 'y_unobserved': (100, 4)}
```

**注意点・落とし穴**:
- `obs_mask`を使うとサイトは`y_observed`(観測値)・`y_unobserved`(潜在変数)・`y`(両者を合成した`deterministic`)の3つに分割される。`y_unobserved`と`y`は元の形(上の例では4要素)を持ち、MCMCの`get_samples()`にもこの2つが含まれる(`y_observed`は含まれない)。
- 合成後の`y`は、`True`位置では観測値、`False`位置では`y_unobserved`(潜在変数)の値になる(上の出力の`y_unobserved`と`y`を比較)。

### `numpyro.param(...)`

**用途**: 学習可能なパラメータ(確率変数ではなく点推定される値)を宣言する。主にSVIのガイド関数や、モデル側の最適化対象パラメータに使う。

**シグネチャ**: `numpyro.param(name, init_value=None, **kwargs)`

**使用例**:
```python
def guide_like():
    s = numpyro.param("scale", 2.0, constraint=dist.constraints.positive)
    return s

tr = handlers.trace(handlers.seed(guide_like, 0)).get_trace()
print(tr["scale"]["type"], tr["scale"]["value"])
print(tr["scale"]["kwargs"])
```
実行結果:
```
param 2.0
{'constraint': Positive(lower_bound=0.0)}
```

**注意点・落とし穴**:
- `constraint=`を付けると、SVIは内部で制約なし空間の値を最適化し、`param`が返す値は常に制約を満たす。`svi.run(...).params`に格納されるのは制約後(constrained)の値。
- 第2引数`init_value`は初期値。`SVI`の`init`時に使われる(シグネチャ上は関数も渡せる)。

### `numpyro.plate(...)`

**用途**: 条件付き独立な次元(データ点、グループなど)を宣言する。`with`内のサイトは`plate`のサイズ分のバッチ次元を持ち、ベクトル化された確率計算になる。

**シグネチャ**: `numpyro.plate(name, size, subsample_size=None, dim=None)`

**使用例**:
```python
def model_plate():
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    with numpyro.plate("data", 5):
        x = numpyro.sample("x", dist.Normal(mu, 1))
    return x

print(handlers.seed(model_plate, 0)().shape)
tr = handlers.trace(handlers.seed(model_plate, 0)).get_trace()
print(tr["x"]["cond_indep_stack"])
print(tr["x"]["fn"].batch_shape, tr["mu"]["fn"].batch_shape)

with numpyro.plate("p", 4) as idx:
    print(idx)
```
実行結果:
```
(5,)
[CondIndepStackFrame(name='data', dim=-1, size=5)]
(5,) ()
[0 1 2 3]
```

**注意点・落とし穴**:
- `plate`は`with`ブロック内の全サイトに自動でブロードキャストをかける。上の例では`Normal(mu, 1)`の`batch_shape`は`()`だが、`plate`内では`(5,)`に拡張される。
- サイトの形が`plate`と合わないとエラーになる。例えば`plate("p", 5)`内で`dist.Normal(jnp.zeros(3), 1)`を使うと`ValueError: Incompatible shapes for broadcasting`(実測)。
- `plate`内で「各要素が3次元ベクトル」としたい場合は、`dist.Normal(jnp.zeros(3), 1).to_event(1)`とする(2章参照)。`to_event`なしだと3も`batch`次元と見なされ`plate`とぶつかる。

### `numpyro.plate(..., dim=...)` の入れ子と `numpyro.plate_stack`

**用途**: 複数の`plate`を入れ子にして多次元の独立構造を作る。`dim`で割り当てる次元(右端から負のインデックス)を明示できる。

**シグネチャ**: `numpyro.plate_stack(prefix, sizes, rightmost_dim=-1)`

**使用例**:
```python
def nested():
    with numpyro.plate("rows", 3, dim=-2):
        with numpyro.plate("cols", 4, dim=-1):
            return numpyro.sample("x", dist.Normal(0, 1))
print(handlers.seed(nested, 0)().shape)

def nested_auto():                     # dim省略: 内側のplateが左隣の次元を自動で使う
    with numpyro.plate("cols", 4):
        with numpyro.plate("rows", 3):
            return numpyro.sample("x", dist.Normal(0, 1))
print(handlers.seed(nested_auto, 0)().shape)

def stacked():
    with numpyro.plate_stack("stack", (3, 4)):
        return numpyro.sample("x", dist.Normal(0, 1))
print(handlers.seed(stacked, 0)().shape)
```
実行結果:
```
(3, 4)
(3, 4)
(3, 4)
```

**注意点・落とし穴**:
- `dim`を省略した入れ子では、外側の`plate`が`dim=-1`、内側が`dim=-2`と、内側ほど左の次元が使われる。上の`nested_auto`が`(3, 4)`になるのはこのため。
- 次元の割り当てを明示したいときは`dim=`(負のインデックス)を指定する。上の`nested`は`rows`が`dim=-2`、`cols`が`dim=-1`で、形は`(3, 4)`。

### `numpyro.plate(..., subsample_size=...)` と `numpyro.subsample`

**用途**: ミニバッチ学習(データの部分集合)を行う。`subsample_size`を指定した`plate`は、ランダムに選んだインデックスを返し、対数尤度を`size/subsample_size`倍にスケールして不偏推定にする。

**シグネチャ**: `numpyro.subsample(data, event_dim)`

**使用例**:
```python
data = jnp.arange(10.0)

def model_sub(data):
    with numpyro.plate("N", data.shape[0], subsample_size=3) as idx:
        batch = numpyro.subsample(data, event_dim=0)
        numpyro.sample("obs", dist.Normal(0, 1), obs=batch)
    return idx, batch

print(handlers.seed(model_sub, 0)(data))
tr = handlers.trace(handlers.seed(model_sub, 0)).get_trace(data)
print(tr["obs"]["scale"])
```
実行結果:
```
(Array([2, 4, 7], dtype=int32), Array([2., 4., 7.], dtype=float32))
3.3333333333333335
```

**注意点・落とし穴**:
- `numpyro.subsample(data, event_dim)`は、`plate`が選んだインデックスに従って`data`を切り出す。`event_dim`は`data`のうち「`plate`の次元を除いたイベント次元」の数(スカラー観測なら0)。
- サイトの`scale`が`10/3`になっている(上の出力)ため、ミニバッチの対数尤度が全データ相当にスケールされる。ミニバッチ学習はSVIで大規模データを扱うときに使う。

### `numpyro.deterministic(...)`

**用途**: 他の確率変数から決まる変換値(中間量)に名前を付け、推論結果(`get_samples()`)や`Predictive`の出力に含める。

**シグネチャ**: `numpyro.deterministic(name, value)`

**使用例**:
```python
def model_det():
    a = numpyro.sample("a", dist.Normal(0, 1))
    b = numpyro.deterministic("b", a ** 2)
    return b

tr = handlers.trace(handlers.seed(model_det, 0)).get_trace()
print({k: (v["type"], float(v["value"])) for k, v in tr.items()})

from numpyro.infer.util import log_density
def model_nodet():
    a = numpyro.sample("a", dist.Normal(0, 1))
    return a ** 2
lp1 = log_density(model_det, (), {}, {"a": 0.5})[0]
lp2 = log_density(model_nodet, (), {}, {"a": 0.5})[0]
print(lp1, lp2)
```
実行結果:
```
{'a': ('sample', -2.442455768585205), 'b': ('deterministic', 5.965590000152588)}
-1.0439385 -1.0439385
```

**注意点・落とし穴**:
- `deterministic`は対数確率(尤度)には寄与しない(上の`lp1`と`lp2`は一致)。単に値を記録するためのサイト。
- MCMCの`get_samples()`には`deterministic`サイトも含まれる(5章のPredictiveの注意点も参照)。`print_summary()`はデフォルトで`deterministic`を除外する(`exclude_deterministic=True`)。

### `numpyro.factor(...)`

**用途**: 任意の対数確率(ペナルティや、既存の分布で書けない尤度項)をモデルの対数密度に直接加算する。PyMCの`pm.Potential`に相当。

**シグネチャ**: `numpyro.factor(name, log_factor)`

**使用例**:
```python
def model_factor():
    x = numpyro.sample("x", dist.Normal(0, 1))
    numpyro.factor("penalty", -0.5 * (x - 3) ** 2 * 10)

from numpyro.infer.util import log_density
print(log_density(model_factor, (), {}, {"x": jnp.array(0.0)})[0])
print(dist.Normal(0, 1).log_prob(0.0) - 0.5 * 3 ** 2 * 10)

tr = handlers.trace(handlers.seed(model_factor, 0)).get_trace()
print(list(tr.keys()), tr["penalty"]["type"])
```
実行結果:
```
-45.918938
-45.918938
['x', 'penalty'] sample
```

**注意点・落とし穴**:
- `factor`は内部では「値を持たない特殊な分布(`Unit`)から観測」する形で実装されており、トレース上のサイト種別は`sample`になる。

### 乱数キー(`jax.random.key` と `jax.random.PRNGKey`)

**用途**: NumPyroの`MCMC.run`・`SVI.run`・`Predictive`・`Distribution.sample`は明示的なJAX乱数キーを受け取る。新式の`jax.random.key`と旧式の`jax.random.PRNGKey`のどちらも渡せる。

**シグネチャ**: `numpyro.prng_key()`

**使用例**:
```python
import warnings
warnings.simplefilter("error")          # 警告が出れば例外になるようにして確認
from numpyro.infer import MCMC, NUTS

def m(y):
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    numpyro.sample("y", dist.Normal(mu, 1), obs=y)

for name, k in [("key", jax.random.key(0)), ("PRNGKey", jax.random.PRNGKey(0))]:
    mc = MCMC(NUTS(m), num_warmup=50, num_samples=50, progress_bar=False)
    mc.run(k, jnp.array([0.5, 1.0]))
    print(name, mc.get_samples()["mu"].shape, k.dtype)

print(dist.Normal(0, 1).sample(jax.random.key(0)), dist.Normal(0, 1).sample(jax.random.PRNGKey(0)))
warnings.simplefilter("default")
```
実行結果:
```
key (50,) key<fry>
PRNGKey (50,) uint32
1.6226422 1.6226422
```

**注意点・落とし穴**:
- NumPyro 0.21.0 / jax 0.11.1では、`jax.random.key`(型付きキー、dtype `key<fry>`)と`jax.random.PRNGKey`(`uint32[2]`配列)のどちらを渡しても警告なく動作し、同じシード値なら同じ乱数が得られる(上の実行結果、および`warnings.simplefilter("error")`下で例外が出ないことで確認)。
- `handlers.seed(rng_seed=...)`も整数・`key`・`PRNGKey`のいずれも受け付ける。モデル関数の内側で現在のキーを取り出したい場合は`numpyro.prng_key()`を使う(`seed`の外では`None`を返す)。

---

## 2. 確率分布

`numpyro.distributions`(慣例で`dist`)の分布クラス。パラメータ名がPyMCと異なる点(例: `Normal(loc, scale)`、`Gamma(concentration, rate)`、`StudentT(df, loc, scale)`)に注意。この章の例は次の共通コードを前提とする。

**共通コード**:
```python
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
key = jax.random.key(0)
```

### `Distribution` の共通メソッド・属性

**用途**: 全分布に共通する、サンプリング(`sample`)・対数確率(`log_prob`)・要約統計量・形状の取得方法。

**シグネチャ**:
- `numpyro.distributions.Distribution.sample(key, sample_shape=())`
- `numpyro.distributions.Distribution.log_prob(value)`
- `numpyro.distributions.Distribution.cdf(value)`
- `numpyro.distributions.Distribution.icdf(q)`
- `numpyro.distributions.Distribution.entropy()`

**使用例**:
```python
d = dist.Normal(loc=0.0, scale=2.0)
print(d.sample(key, (5,)))
print(d.log_prob(1.0), d.mean, d.variance)
print(d.cdf(0.0), d.icdf(0.975))
print(d.entropy())
print(d.batch_shape, d.event_shape, d.support)
print(dist.Normal(jnp.zeros(3), 1.0).batch_shape)
print(dist.Normal(jnp.zeros((2, 3)), 1.0).sample(key, (4,)).shape)
print(dist.Normal(0.0, -1.0).log_prob(0.0))
```
実行結果:
```
[ 3.2452843  4.0505295 -0.8671889 -0.1572347  0.3521818]
-1.7370857 0.0 4.0
0.5 3.9199286
2.1120858
() () Real()
(3,)
(4, 2, 3)
nan
```

**注意点・落とし穴**:
- `sample(key, sample_shape)`は第1引数に乱数キーが必須。返る配列の形は`sample_shape + batch_shape + event_shape`(上の`(4, 2, 3)`)。
- `batch_shape`は「独立な分布の並び」、`event_shape`は「1つのサンプルの形」。`Normal(jnp.zeros(3), 1.0)`は`batch_shape=(3,)`、`event_shape=()`。多変量分布(`MultivariateNormal`、`Dirichlet`等)は`event_shape`を持つ。
- 既定では引数の検証が無効なので、範囲外のパラメータ(例: `Normal(0, -1)`)でもエラーにならず`nan`を返す(上の最終行)。検証を有効にする方法は10章参照。

### `Normal(...)`

**用途**: 正規分布。連続値のモデリングで最も基本的な分布。

**シグネチャ**: `numpyro.distributions.Normal(loc=0.0, scale=1.0, *, validate_args=None)`

**使用例**:
```python
d = dist.Normal(loc=0.0, scale=2.0)
print(d.sample(key, (5,)))
print(d.log_prob(jnp.array([0.0, 1.0, 2.0])))
```
実行結果:
```
[ 3.2452843  4.0505295 -0.8671889 -0.1572347  0.3521818]
[-1.6120857 -1.7370857 -2.1120858]
```

**注意点・落とし穴**:
- `scale`は分散ではなく標準偏差。PyMCの`sigma`に相当し、`tau`(精度)引数はない。

### `HalfNormal(...)` / `HalfCauchy(...)`

**用途**: 0以上の値だけを取る分布。標準偏差・スケール・階層モデルのグループ間分散などの事前分布によく使う。

**シグネチャ**: `numpyro.distributions.HalfNormal(scale=1.0, *, validate_args=None)` / `numpyro.distributions.HalfCauchy(scale=1.0, *, validate_args=None)`

**使用例**:
```python
print(dist.HalfNormal(2.0).sample(key, (5,)))
print(dist.HalfCauchy(1.0).sample(key, (5,)))
print(dist.HalfNormal(1.0).mean, dist.HalfCauchy(1.0).mean)
print(dist.HalfNormal(2.0).mean, 2.0 * (2 / jnp.pi) ** 0.5)
```
実行結果:
```
[3.2452843 4.0505295 0.8671889 0.1572347 0.3521818]
[ 6.0274944  14.837895    0.58172226  0.09874987  0.22315963]
0.7978845 inf
1.595769 1.5957691216057308
```

**注意点・落とし穴**:
- `HalfCauchy`は裾が非常に重く平均が存在しない(`mean`は`inf`を返す)。サンプルにも極端に大きな値が混じる(上の例では約14.8)。
- 引数名はどちらも`scale`。`HalfNormal(scale)`の平均は`scale*sqrt(2/pi)`で、`scale`そのものではない(上の最終行)。

### `StudentT(...)`

**用途**: 自由度`df`を持つt分布。正規分布より裾が重く、外れ値に頑健な誤差分布として使う。

**シグネチャ**: `numpyro.distributions.StudentT(df, loc=0.0, scale=1.0, *, validate_args=None)`

**使用例**:
```python
print(dist.StudentT(df=3, loc=0, scale=1).sample(key, (5,)))
print(dist.StudentT(df=1.0).mean, dist.StudentT(df=2.0).variance, dist.StudentT(df=5.0).variance)
```
実行結果:
```
[ 0.69022465 -3.2393365  -1.6745249  -1.8150756  -0.50429136]
inf inf 1.6666666
```

**注意点・落とし穴**:
- `df`は必須の第1引数(PyMCの`nu`)。`df<=1`で`mean`、`df<=2`で`variance`が`inf`になる(上の実行結果)。

### `Beta(...)`

**用途**: 0〜1の確率・比率のモデリング(コンバージョン率の事前分布など)。

**シグネチャ**: `numpyro.distributions.Beta(concentration1, concentration0, *, validate_args=None)`

**使用例**:
```python
d = dist.Beta(2.0, 5.0)
print(d.sample(key, (5,)), d.mean)
```
実行結果:
```
[0.24796297 0.12259734 0.36335078 0.3910233  0.40845913] 0.2857142857142857
```

**注意点・落とし穴**:
- 引数名は`alpha`/`beta`ではなく`concentration1`(=α)/`concentration0`(=β)。位置引数で`Beta(2.0, 5.0)`と書けばαβの順になる。PyMCの`mu`/`sigma`による指定はない。

### `Gamma(...)` / `Exponential(...)`

**用途**: 0以上の連続量(待ち時間・精度・分散パラメータなど)のモデリング。`Exponential`は`Gamma(1, rate)`に相当する。

**シグネチャ**: `numpyro.distributions.Gamma(concentration, rate=1.0, *, validate_args=None)` / `numpyro.distributions.Exponential(rate=1.0, *, validate_args=None)`

**使用例**:
```python
d = dist.Gamma(concentration=2.0, rate=1.0)
print(d.sample(key, (5,)), d.mean)
print(dist.Gamma(2.0, rate=2.0).mean)
print(dist.Exponential(2.0).mean)
```
実行結果:
```
[1.5897877 1.7599413 1.1290071 3.974334  2.9546   ] 2.0
1.0
0.5
```

**注意点・落とし穴**:
- `Gamma`の第2引数は`rate`(スケールではない)。平均は`concentration/rate`。`Exponential`の引数も`rate`で、平均は`1/rate`(上の実行結果: 0.5)。

### `LogNormal(...)` / `Uniform(...)`

**用途**: `LogNormal`は正の値で右に裾を引く量(所得・応答時間)に、`Uniform`は範囲のみ分かっている量に使う。

**シグネチャ**: `numpyro.distributions.LogNormal(loc=0.0, scale=1.0, *, validate_args=None)` / `numpyro.distributions.Uniform(low=0.0, high=1.0, *, validate_args=None)`

**使用例**:
```python
print(dist.LogNormal(0.0, 1.0).sample(key, (3,)))
print(dist.Uniform(0.0, 10.0).sample(key, (3,)))
print(dist.Uniform(0.0, 10.0).support)
```
実行結果:
```
[5.066459   7.578117   0.64817506]
[9.47667   9.785799  3.3229148]
Interval(lower_bound=0.0, upper_bound=10.0)
```

**注意点・落とし穴**:
- `LogNormal(loc, scale)`の`loc`/`scale`は「対数を取った後の正規分布」のパラメータ(`Normal`に`ExpTransform`を適用した分布と一致することは後述の`TransformedDistribution`の項で確認)。
- `Uniform(low, high)`の`support`は`Interval(low, high)`で、事前分布に使うと範囲外の値が禁止される(範囲に自信がないなら裾の広い分布の方が安全)。

### `Bernoulli(...)`

**用途**: 0/1の二値(成功・失敗)を表す分布。`probs`(確率)または`logits`(ロジット)のどちらか一方で指定する。

**シグネチャ**: `numpyro.distributions.Bernoulli(probs=None, logits=None, *, validate_args=None)`

**使用例**:
```python
d = dist.Bernoulli(probs=0.3)
print(d.sample(key, (10,)))
print(d.log_prob(1), d.support)
print(dist.Bernoulli(logits=0.0).probs)
print(type(dist.Bernoulli(probs=0.3)).__name__, type(dist.Bernoulli(logits=0.3)).__name__)
try:
    dist.Bernoulli(probs=0.3, logits=0.1)
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
[0 0 0 0 0 1 0 0 0 1]
-1.2039728 Boolean()
0.5
BernoulliProbs BernoulliLogits
ValueError: Exactly one of ['probs', 'logits'] must be specified; got ['probs', 'logits'].
```

**注意点・落とし穴**:
- `Bernoulli`は実際には`BernoulliProbs`または`BernoulliLogits`のインスタンスを返すファクトリ(上の出力)。ロジスティック回帰では`logits`指定(`dist.Bernoulli(logits=X @ beta)`)の方が数値的に安定で、`sigmoid`を自分で掛ける必要もない。
- `probs`と`logits`を同時に指定すると`ValueError: Exactly one of ['probs', 'logits'] must be specified`。
- 同じ形式が`Binomial`(`total_count`, `probs`/`logits`)にもある。

### `Binomial(...)`

**用途**: `total_count`回の試行のうちの成功回数のモデリング。

**シグネチャ**: `numpyro.distributions.Binomial(total_count=1, probs=None, logits=None, *, validate_args=None)`

**使用例**:
```python
d = dist.Binomial(total_count=10, probs=0.5)
print(d.sample(key, (8,)), d.support, d.mean)
print(dist.Binomial(1, probs=0.3).log_prob(1), dist.Bernoulli(probs=0.3).log_prob(1))
```
実行結果:
```
[10  7  5  6  4  2  5  7] IntegerInterval(lower_bound=0, upper_bound=10) 5.0
-1.2039733 -1.2039728
```

**注意点・落とし穴**:
- `total_count`は第1引数(既定値1)で、`Binomial(1, p)`の`log_prob`は`Bernoulli(p)`と一致する(上の最終行)。`probs`/`logits`はどちらか一方。

### `Poisson(...)` / `NegativeBinomial2(...)` / `ZeroInflatedPoisson(...)`

**用途**: カウントデータの分布。`Poisson`は平均=分散、`NegativeBinomial2`は過分散、`ZeroInflatedPoisson`はゼロが過剰なデータ向け。

**シグネチャ**:
- `numpyro.distributions.Poisson(rate, *, is_sparse=False, validate_args=None)`
- `numpyro.distributions.NegativeBinomial2(mean, concentration, *, validate_args=None)`
- `numpyro.distributions.ZeroInflatedPoisson(gate, rate=1.0, *, validate_args=None)`

**使用例**:
```python
print(dist.Poisson(4.0).sample(key, (8,)), dist.Poisson(4.0).log_prob(3))
print(dist.NegativeBinomial2(mean=3.0, concentration=2.0).sample(key, (8,)))
zip_ = dist.ZeroInflatedPoisson(gate=0.4, rate=3.0)
print(zip_.sample(key, (12,)))
print(jnp.exp(zip_.log_prob(0)), 0.4 + 0.6 * jnp.exp(-3.0))
print(dist.NegativeBinomial2(3.0, 2.0).variance, dist.NegativeBinomial2(3.0, 1000.0).variance)
```
実行結果:
```
[0 1 5 5 2 2 4 6] -1.6328773
[ 0  1  5  2 13  1  1  5]
[2 0 0 0 0 1 4 0 5 0 0 2]
0.4298722 0.42987224
7.499999 3.009
```

**注意点・落とし穴**:
- `Poisson`の引数は`rate`(平均)。`NegativeBinomial2(mean, concentration)`の分散は`mean + mean**2/concentration`で(上の`3+9/2=7.5`)、`concentration`が大きいほど分散が平均(=3)に近づき`Poisson`に近づく。
- `ZeroInflatedPoisson`の`gate`は「常に0になる確率」で、`P(0) = gate + (1-gate)*exp(-rate)`(上の2つの値が一致)。

### `Categorical(...)`

**用途**: `K`個のカテゴリの離散分布(サンプルは`0`〜`K-1`の整数)。`probs`または`logits`で指定する。

**シグネチャ**: `numpyro.distributions.Categorical(probs=None, logits=None, *, validate_args=None)`

**使用例**:
```python
d = dist.Categorical(probs=jnp.array([0.2, 0.5, 0.3]))
print(d.sample(key, (8,)), d.log_prob(1), d.support)
print(dist.Categorical(logits=jnp.array([0.0, 1.0, 2.0])).probs)
```
実行結果:
```
[2 2 1 1 1 0 1 1] -0.6931472 IntegerInterval(lower_bound=0, upper_bound=2)
[0.09003057 0.24472848 0.66524094]
```

**注意点・落とし穴**:
- カテゴリ数`K`は`probs`/`logits`の最終次元の長さで決まり、`support`は`IntegerInterval(0, K-1)`。カテゴリ番号は0始まり。

### `Dirichlet(...)`

**用途**: 確率ベクトル(合計1の非負ベクトル)の分布。カテゴリ確率や混合の重みの事前分布に使う。

**シグネチャ**: `numpyro.distributions.Dirichlet(concentration, *, validate_args=None)`

**使用例**:
```python
d = dist.Dirichlet(jnp.array([1.0, 2.0, 3.0]))
s = d.sample(key, (2,))
print(s)
print(s.sum(-1), d.event_shape, d.support)
```
実行結果:
```
[[0.14228562 0.4048826  0.4528318 ]
 [0.2847853  0.35153106 0.36368367]]
[1. 1.] (3,) Simplex()
```

**注意点・落とし穴**:
- `event_shape`が`(3,)`の多変量分布で、`support`は`Simplex()`(サンプルの`sum(-1)`が1になる)。`plate`と組み合わせるときは、最終次元(`event`次元)は`plate`のバッチ次元とは別物として扱われる。

### `MultivariateNormal(...)`

**用途**: 多変量正規分布。`covariance_matrix`・`precision_matrix`・`scale_tril`(Cholesky因子)のいずれかで指定する。

**シグネチャ**: `numpyro.distributions.MultivariateNormal(loc=0.0, covariance_matrix=None, precision_matrix=None, scale_tril=None, *, validate_args=None)`

**使用例**:
```python
cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
d = dist.MultivariateNormal(loc=jnp.zeros(2), covariance_matrix=cov)
print(d.sample(key, (2,)))
print(d.batch_shape, d.event_shape, d.log_prob(jnp.zeros(2)))
d2 = dist.MultivariateNormal(jnp.zeros(2), scale_tril=jnp.linalg.cholesky(cov))
print(d2.log_prob(jnp.zeros(2)))
```
実行結果:
```
[[ 1.6226422   2.5132725 ]
 [-0.43359444 -0.39404595]]
() (2,) -1.3270514
-1.3270514
```

**注意点・落とし穴**:
- 3つの共分散指定(`covariance_matrix`/`precision_matrix`/`scale_tril`)のうち指定するのは1つだけ。`covariance_matrix`と`scale_tril`(=`jnp.linalg.cholesky(cov)`)は同じ分布を表す(上の2つの`log_prob`が一致)。`scale_tril`を渡せば呼び出し側でCholesky因子を直接構築できる(LKJCholeskyと組み合わせる場合など)。

### `LKJCholesky(...)`

**用途**: 相関行列の事前分布(LKJ分布)を、相関行列のCholesky因子として表現する。多変量正規の共分散を階層的に構築するときの定番。

**シグネチャ**: `numpyro.distributions.LKJCholesky(dimension, concentration=1.0, sample_method='onion', *, validate_args=None)`

**使用例**:
```python
d = dist.LKJCholesky(dimension=3, concentration=2.0)
L = d.sample(key)
print(L)
print(L @ L.T)
print(d.support)
for c in (1.0, 10.0):
    Ls = dist.LKJCholesky(3, c).sample(key, (2000,))
    C = Ls @ jnp.swapaxes(Ls, -1, -2)
    print(c, float(jnp.abs(C[:, 0, 1]).mean()))
```
実行結果:
```
[[ 1.          0.          0.        ]
 [-0.68709725  0.7265655   0.        ]
 [-0.60607845  0.0611962   0.79304725]]
[[ 1.         -0.68709725 -0.60607845]
 [-0.68709725  1.          0.46089786]
 [-0.60607845  0.46089786  1.        ]]
CorrCholesky()
1.0 0.4176784157752991
10.0 0.1674545556306839
```

**注意点・落とし穴**:
- サンプルは相関行列そのものではなく下三角のCholesky因子`L`で、`L @ L.T`が相関行列(対角成分が1)になる。
- `concentration`が大きいほど非対角成分の絶対値の平均が小さくなる(上の最後の2行: 無相関寄りになる)。`MultivariateNormal(scale_tril=jnp.diag(sigma) @ L)`のように使う。

### `.to_event(...)` / `Independent(...)` / `.expand(...)`

**用途**: 分布の`batch_shape`と`event_shape`を調整する。`to_event(n)`は右端`n`個のバッチ次元をイベント次元に変換し、`expand`はバッチ形状を拡張する。

**シグネチャ**:
- `numpyro.distributions.Independent(base_dist, reinterpreted_batch_ndims, *, validate_args=None)`
- `numpyro.distributions.Distribution.to_event(reinterpreted_batch_ndims=None)`
- `numpyro.distributions.Distribution.expand(batch_shape)`

**使用例**:
```python
b = dist.Normal(jnp.zeros(3), 1.0)
print(b.batch_shape, b.event_shape, b.log_prob(jnp.zeros(3)))
e = b.to_event(1)
print(e.batch_shape, e.event_shape, e.log_prob(jnp.zeros(3)))
i = dist.Independent(b, 1)
print(i.batch_shape, i.event_shape)
x = dist.Normal(0.0, 1.0).expand([2, 3])
print(x.batch_shape, x.sample(key).shape)
y = x.to_event(1)
print(y.batch_shape, y.event_shape)
```
実行結果:
```
(3,) () [-0.9189385 -0.9189385 -0.9189385]
() (3,) -2.7568154
() (3,)
(2, 3) (2, 3)
(2,) (3,)
```

**注意点・落とし穴**:
- `to_event(1)`後は`log_prob`が最終次元で合算されたスカラーになる(上の`-2.7568154`は3要素の合計)。`Independent(b, 1)`は`b.to_event(1)`と同じ。
- `plate`内でベクトル値の確率変数を宣言するときは`to_event`でベクトル部分を`event`にして、`plate`が`batch`を担当するように分ける(1章の`plate`参照)。

### `.mask(...)`

**用途**: `mask`が`False`の要素の`log_prob`を0にする。欠損データや部分的に無効な観測を尤度から除外するのに使う。

**シグネチャ**: `numpyro.distributions.Distribution.mask(mask)`

**使用例**:
```python
m = dist.Normal(0.0, 1.0).expand([3]).mask(jnp.array([True, False, True]))
print(m.log_prob(jnp.array([0.0, 100.0, 0.0])))
```
実行結果:
```
[-0.9189385  0.        -0.9189385]
```

**注意点・落とし穴**:
- マスクされた位置の値(上の`100.0`)は`log_prob`に寄与しない(0になる)。モデル内のサイト群全体に適用する場合は`handlers.mask`(8章)を使う。

### `TruncatedNormal(...)`

**用途**: 範囲`[low, high]`で打ち切った正規分布。片側だけ指定することもできる。

**シグネチャ**: `numpyro.distributions.TruncatedNormal(loc=0.0, scale=1.0, *, low=None, high=None, validate_args=None)`

**使用例**:
```python
d = dist.TruncatedNormal(loc=0.0, scale=1.0, low=0.0)
print(d.sample(key, (5,)), d.support)
d2 = dist.TruncatedNormal(0.0, 1.0, low=-1.0, high=1.0)
print(d2.sample(key, (5,)))
```
実行結果:
```
[1.9403844  2.300496   0.42929506 0.6259747  0.78900117] GreaterThan(lower_bound=0.0)
[ 0.86185944  0.9412883  -0.29104835 -0.0536418   0.11988346]
```

**注意点・落とし穴**:
- `low`/`high`はキーワード専用引数(`TruncatedNormal(0, 1, -1, 1)`のように位置指定はできない)。片方を省略すると片側打ち切り(`support`が`GreaterThan(0.0)`になる)。

### `MixtureSameFamily(...)`

**用途**: 同じ分布族の成分の有限混合(混合正規分布など)。離散の潜在クラスを周辺化して(積分消去して)扱うので、NUTSでそのまま推論できる。

**シグネチャ**: `numpyro.distributions.MixtureSameFamily(mixing_distribution, component_distribution, *, validate_args=None)`

**使用例**:
```python
mix = dist.MixtureSameFamily(
    dist.Categorical(probs=jnp.array([0.3, 0.7])),
    dist.Normal(jnp.array([-2.0, 2.0]), jnp.array([0.5, 0.5])),
)
print(mix.sample(key, (6,)))
print(mix.mean, mix.log_prob(2.0), mix.batch_shape, mix.event_shape)

# 混合モデルをNUTSでそのまま推論できる(離散の潜在クラスは周辺化済み)
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
data = np.concatenate([np.random.default_rng(0).normal(-2, 0.5, 30), np.random.default_rng(1).normal(2, 0.5, 70)])
def mix_model(y):
    w = numpyro.sample("w", dist.Dirichlet(jnp.ones(2)))
    mu = numpyro.sample("mu", dist.Normal(jnp.array([-1.0, 1.0]), 3.0).to_event(1))
    numpyro.sample("y", dist.MixtureSameFamily(dist.Categorical(probs=w), dist.Normal(mu, 1.0)), obs=y)
mc = MCMC(NUTS(mix_model), num_warmup=200, num_samples=200, progress_bar=False)
mc.run(key, jnp.asarray(data))
print({k: v.shape for k, v in mc.get_samples().items()})
print(mc.get_samples()["mu"].mean(0), mc.get_samples()["w"].mean(0))
```
実行結果:
```
[-1.4979929 -2.3740861  2.294419   1.487201  -1.169186  -2.6444669]
0.79999995 -0.58246636 () ()
{'mu': (200, 2), 'w': (200, 2)}
[ 1.9549994 -2.0542936] [0.6981475  0.30185252]
```

**注意点・落とし穴**:
- 第1引数が混合重み(`Categorical`)、第2引数が成分の分布(最後のバッチ次元が成分数)。混合分布の`batch_shape`/`event_shape`は成分次元を消した形(上の例では`()`と`()`)になる。
- 混合モデルでは成分の順序(ラベル)が入れ替わりうる(label switching)。上の推論結果の`mu`の事後平均は`[1.95, -2.05]`で、事前分布の平均`[-1, 1]`を想定した順序とは逆になっている。結果を解釈するときは、どの成分がどのラベルに対応しているかを確認すること。

### `TransformedDistribution(...)`

**用途**: ベース分布に可逆変換(`Transform`)を適用した分布を作る。対数ヤコビアンは自動で補正される。

**シグネチャ**: `numpyro.distributions.TransformedDistribution(base_distribution, transforms, *, validate_args=None)`

**使用例**:
```python
from numpyro.distributions import transforms as T
td = dist.TransformedDistribution(dist.Normal(0.0, 1.0), T.ExpTransform())
print(td.sample(key, (3,)))
print(td.log_prob(1.0), dist.LogNormal(0.0, 1.0).log_prob(1.0))
aff = dist.TransformedDistribution(dist.Normal(0.0, 1.0), [T.AffineTransform(5.0, 2.0)])
print(aff.sample(key, (3,)))
```
実行結果:
```
[5.066459   7.578117   0.64817506]
-0.9189385 -0.9189385
[8.245284  9.0505295 4.132811 ]
```

**注意点・落とし穴**:
- 変換のリストを渡すと左から順に適用される。`Normal`に`ExpTransform`を適用した結果は`LogNormal`と一致する(上の`log_prob`が同値)。使える変換は7章参照。

---

## 3. MCMCサンプリング

`MCMC`(実行の枠組み)に`NUTS`などのカーネルを渡して事後分布からサンプリングする。この章の例は次の共通コード(単回帰: `y = a + b*x + ノイズ`)を前提とする。真値は`a=1, b=2, sigma=0.5`。

**共通コード**:
```python
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

rng = np.random.default_rng(0)
x = rng.normal(size=50)
y = 1.0 + 2.0 * x + rng.normal(scale=0.5, size=50)   # 真の値: a=1, b=2, sigma=0.5

def model(x, y=None):
    a = numpyro.sample("a", dist.Normal(0, 5))
    b = numpyro.sample("b", dist.Normal(0, 5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2))
    numpyro.sample("y", dist.Normal(a + b * x, sigma), obs=y)

from numpyro.infer import MCMC, NUTS, HMC
```

### `NUTS(...)`

**用途**: No-U-Turn Sampler。NumPyroの標準的なMCMCカーネルで、ステップ幅と質量行列をウォームアップ中に自動調整する。

**シグネチャ**: `numpyro.infer.NUTS(model=None, potential_fn=None, kinetic_fn=None, step_size=1.0, inverse_mass_matrix=None, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False, target_accept_prob=0.8, trajectory_length=None, max_tree_depth=10, init_strategy=init_to_uniform, find_heuristic_step_size=False, forward_mode_differentiation=False, regularize_mass_matrix=True)`

**使用例**:
```python
kernel = NUTS(model, target_accept_prob=0.9, max_tree_depth=8, dense_mass=True)
mcmc = MCMC(kernel, num_warmup=200, num_samples=200, progress_bar=False)
mcmc.run(jax.random.key(0), x, y)
print(mcmc.get_samples()["b"].shape)
print({k: v.shape for k, v in mcmc.last_state.adapt_state.inverse_mass_matrix.items()})   # dense_mass=True

mcmc_diag = MCMC(NUTS(model), num_warmup=200, num_samples=200, progress_bar=False)
mcmc_diag.run(jax.random.key(0), x, y)
print({k: v.shape for k, v in mcmc_diag.last_state.adapt_state.inverse_mass_matrix.items()})  # 既定は対角
```
実行結果:
```
(200,)
{('a', 'b', 'sigma'): (3, 3)}
{('a', 'b', 'sigma'): (3,)}
```

**注意点・落とし穴**:
- `NUTS`は「カーネル」にすぎず、単体ではサンプリングを実行しない。必ず`MCMC(kernel, ...)`に渡して`.run()`する。
- `target_accept_prob`の既定は0.8。発散(divergence)が出るときは0.9〜0.95に上げるのが定番の対処。`dense_mass=True`で質量行列が対角から密行列になる(上の出力で形が`(3,)`→`(3, 3)`に変わる)。
- モデルの引数は`NUTS(model)`の時点ではなく、後述の`mcmc.run(rng_key, *model_args)`で渡す。

### `MCMC(...)` と `MCMC.run(...)`

**用途**: MCMC実行の枠組み。ウォームアップ(`num_warmup`)と本サンプル(`num_samples`)の数を指定し、`run`で実行する。

**シグネチャ**: `numpyro.infer.MCMC(sampler, *, num_warmup, num_samples, num_chains=1, thinning=1, postprocess_fn=None, chain_method='parallel', progress_bar=True, progress_rate=None, jit_model_args=False)` / `numpyro.infer.MCMC.run(rng_key, *args, extra_fields=(), init_params=None, **kwargs)`

**使用例**:
```python
mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=500, progress_bar=False)
mcmc.run(jax.random.key(0), x, y)   # rng_keyの後ろの引数はそのままmodel(x, y)に渡る
s = mcmc.get_samples()
print({k: round(float(v.mean()), 3) for k, v in s.items()})
```
実行結果:
```
{'a': 1.022, 'b': 1.942, 'sigma': 0.52}
```

**注意点・落とし穴**:
- `num_warmup`と`num_samples`はキーワード専用の必須引数。`num_samples`はウォームアップを含まない本サンプル数(1チェーンあたり)。
- `mcmc.run(rng_key, *args, **kwargs)`の`rng_key`以降がモデル関数の引数として渡される。`extra_fields=`で発散などの追加情報を保存できる(後述)。
- `progress_bar=True`(既定)のままだとログにバーが大量に出るので、スクリプトや検証では`False`にする。

### `MCMC.get_samples(...)`

**用途**: 事後サンプルを`{サイト名: 配列}`の辞書で取得する。

**シグネチャ**: `numpyro.infer.MCMC.get_samples(group_by_chain=False)`

**使用例**:
```python
mcmc = MCMC(NUTS(model), num_warmup=200, num_samples=300, progress_bar=False)
mcmc.run(jax.random.key(0), x, y)
print({k: v.shape for k, v in mcmc.get_samples().items()})
print({k: v.shape for k, v in mcmc.get_samples(group_by_chain=True).items()})
```
実行結果:
```
{'a': (300,), 'b': (300,), 'sigma': (300,)}
{'a': (1, 300), 'b': (1, 300), 'sigma': (1, 300)}
```

**注意点・落とし穴**:
- デフォルトではチェーンを連結した形`(チェーン数*num_samples, ...)`が返る。チェーンごとの診断(r_hatなど)や`az.from_numpyro`では`group_by_chain=True`(形が`(chain, draw, ...)`)が必要になる。
- 戻り値には`numpyro.deterministic`サイトも含まれる。観測サイト(`obs=`付き)は含まれない。

### `MCMC.print_summary(...)`

**用途**: 各パラメータの事後平均・標準偏差・中央値・信用区間・有効サンプルサイズ`n_eff`・`r_hat`と、発散数を表形式で表示する。

**シグネチャ**: `numpyro.infer.MCMC.print_summary(prob=0.9, exclude_deterministic=True)`

**使用例**:
```python
mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=500, progress_bar=False)
mcmc.run(jax.random.key(0), x, y)
mcmc.print_summary()
```
実行結果:
```

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         a      1.02      0.08      1.02      0.89      1.14    323.67      1.00
         b      1.94      0.08      1.95      1.81      2.07    518.63      1.00
     sigma      0.52      0.05      0.51      0.43      0.61    204.44      1.00

Number of divergences: 0
```

**注意点・落とし穴**:
- 区間の既定は`prob=0.9`(列名`5.0%`/`95.0%`)。この区間は`numpyro.diagnostics.hpdi`による最高事後密度区間(HPDI)で、分位点区間(等尾区間)ではない(`summary`のソースで確認)。
- `n_eff`と`r_hat`は複数チェーンで真価を発揮する。1チェーンでも`r_hat`が表示されるが、それはチェーンを分割して計算した値(split r_hat)。
- 既定では`deterministic`サイトは表示されない(`exclude_deterministic=True`)。

### `MCMC.get_extra_fields(...)` と `extra_fields=`

**用途**: `run`時に`extra_fields=`で指定した診断情報(発散フラグ、ステップ数、ステップ幅など)を保存し、後から取り出す。

**シグネチャ**: `numpyro.infer.MCMC.get_extra_fields(group_by_chain=False)`

**使用例**:
```python
mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=500, progress_bar=False)
mcmc.run(jax.random.key(0), x, y,
         extra_fields=("potential_energy", "diverging", "num_steps", "accept_prob", "adapt_state.step_size"))
ef = mcmc.get_extra_fields()
print(list(ef.keys()))
print("divergences:", int(ef["diverging"].sum()))
print("mean accept_prob:", round(float(ef["accept_prob"].mean()), 3))
print("mean num_steps:", int(ef["num_steps"].mean()))

mcmc2 = MCMC(NUTS(model), num_warmup=100, num_samples=100, progress_bar=False)
mcmc2.run(jax.random.key(0), x, y)
print(list(mcmc2.get_extra_fields().keys()))
```
実行結果:
```
['accept_prob', 'adapt_state.step_size', 'diverging', 'num_steps', 'potential_energy']
divergences: 0
mean accept_prob: 0.942
mean num_steps: 5
['diverging']
```

**注意点・落とし穴**:
- `extra_fields`を指定しない場合でも`diverging`は自動で保存される(上の最後の出力)。発散が0でない場合は`target_accept_prob`を上げるか、モデルの再パラメータ化(6章)を検討する。
- `extra_fields`には`"adapt_state.step_size"`のようにドット区切りでネストした状態も指定できる。

### 複数チェーン(`num_chains` / `chain_method` / `numpyro.set_host_device_count`)

**用途**: 複数チェーンを並列/逐次で走らせて`r_hat`を有効にする。CPUで並列実行するには、JAXを初期化する前に`set_host_device_count`でデバイス数を増やす。

**シグネチャ**: `numpyro.set_host_device_count(n)`

**使用例**:
```python
import numpyro
numpyro.set_host_device_count(2)        # jaxのバックエンド初期化より前に呼ぶ

import jax, numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

x = np.random.default_rng(0).normal(size=30)
def model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    numpyro.sample("y", dist.Normal(mu, 1), obs=y)

print(jax.local_device_count())
for cm in ["parallel", "sequential", "vectorized"]:
    mc = MCMC(NUTS(model), num_warmup=100, num_samples=100, num_chains=2,
              chain_method=cm, progress_bar=False)
    mc.run(jax.random.key(0), x)
    print(cm, mc.get_samples(group_by_chain=True)["mu"].shape)
print(mc.get_samples()["mu"].shape)     # 既定(group_by_chain=False)はチェーンを連結した形
```
実行結果:
```
2
parallel (2, 100)
sequential (2, 100)
vectorized (2, 100)
(200,)
```

**注意点・落とし穴**:
- `set_host_device_count`はJAXがバックエンド(デバイス)を初期化する前に呼ぶ必要がある。`jax.local_device_count()`などを先に呼んだ後だと効果がなく、デバイス数は1のまま(実測: 呼び出し前後とも`1`)。
- デバイス数が足りないまま`num_chains=2`(既定`chain_method='parallel'`)にすると、`UserWarning: There are not enough devices to run parallel chains: expected 2 but got 1. Chains will be drawn sequentially.`と警告して逐次実行にフォールバックする。
- `chain_method`は`'parallel'`(デバイス並列)・`'sequential'`(逐次)・`'vectorized'`(1デバイス上で`vmap`)から選ぶ。

### `HMC(...)`

**用途**: 固定長の軌道(`num_steps`または`trajectory_length`)を使うハミルトニアンモンテカルロ。`NUTS`の軌道長自動調整がない版。

**シグネチャ**: `numpyro.infer.HMC(model=None, potential_fn=None, kinetic_fn=None, step_size=1.0, inverse_mass_matrix=None, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False, target_accept_prob=0.8, num_steps=None, trajectory_length=6.283185307179586, init_strategy=init_to_uniform, find_heuristic_step_size=False, forward_mode_differentiation=False, regularize_mass_matrix=True)`

**使用例**:
```python
mcmc = MCMC(HMC(model, num_steps=20, trajectory_length=None),
            num_warmup=200, num_samples=300, progress_bar=False)
mcmc.run(jax.random.key(0), x, y)
print({k: round(float(v.mean()), 3) for k, v in mcmc.get_samples().items()})
```
実行結果:
```
{'a': 1.024, 'b': 1.942, 'sigma': 0.52}
```

**注意点・落とし穴**:
- `num_steps`と`trajectory_length`(既定は`2π`)の両方を指定すると、`UserWarning: If both num_steps and trajectory_length are specified step size can't be adapted`となりステップ幅が自動調整されない(実測)。`num_steps`を使うときは`trajectory_length=None`を併せて渡す。
- 特に理由がなければ`NUTS`を使う。

### 初期値の指定(`init_strategy` / `init_to_value` / `init_params`)

**用途**: MCMCの初期値の決め方を変える。既定は`init_to_uniform`。

**シグネチャ**:
- `numpyro.infer.init_to_median(site=None, num_samples=15)`
- `numpyro.infer.init_to_value(site=None, values={})`
- `numpyro.infer.init_to_uniform(site=None, radius=2)`

**使用例**:
```python
from numpyro.infer import init_to_median, init_to_value

k1 = NUTS(model, init_strategy=init_to_median(num_samples=10))
k2 = NUTS(model, init_strategy=init_to_value(values={"a": 0.0, "b": 0.0, "sigma": 1.0}))
for k in (k1, k2):
    m = MCMC(k, num_warmup=100, num_samples=100, progress_bar=False)
    m.run(jax.random.key(0), x, y)
    print(m.get_samples()["b"].shape)

m3 = MCMC(NUTS(model), num_warmup=100, num_samples=100, progress_bar=False)
m3.run(jax.random.key(0), x, y, init_params={"a": 0.0, "b": 0.0, "sigma": 1.0})  # 制約付き空間の値
print(m3.get_samples()["sigma"].shape)
```
実行結果:
```
(100,)
(100,)
(100,)
```

**注意点・落とし穴**:
- `init_strategy`はカーネル側(`NUTS(model, init_strategy=...)`)、`init_params`は`mcmc.run(..., init_params=...)`側で指定する別経路。`init_strategy`の既定は`init_to_uniform`(`radius=2`)。
- 他に`init_to_feasible`・`init_to_sample`・`init_to_mean`がある(`numpyro.infer`から`init_to_*`として利用可能)。

### `MCMC.warmup(...)` / `last_state` / `thinning`

**用途**: ウォームアップだけを先に実行し(`warmup`)、その後の本サンプリングを`run`で継続する。ウォームアップ中のサンプルも取り出せる。`thinning`で間引く。

**シグネチャ**: `numpyro.infer.MCMC.warmup(rng_key, *args, extra_fields=(), collect_warmup=False, init_params=None, **kwargs)`

**使用例**:
```python
mcmc = MCMC(NUTS(model), num_warmup=200, num_samples=100, progress_bar=False)
mcmc.warmup(jax.random.key(0), x, y, collect_warmup=True)
print(mcmc.get_samples()["a"].shape)          # ウォームアップ中のサンプル
mcmc.run(jax.random.key(1), x, y)             # ウォームアップ状態から本サンプリングを継続
print(mcmc.get_samples()["a"].shape, type(mcmc.last_state).__name__)

thin = MCMC(NUTS(model), num_warmup=100, num_samples=100, thinning=2, progress_bar=False)
thin.run(jax.random.key(0), x, y)
print(thin.get_samples()["a"].shape)
```
実行結果:
```
(200,)
(100,) HMCState
(50,)
```

**注意点・落とし穴**:
- `warmup(collect_warmup=True)`直後の`get_samples()`はウォームアップ分(`num_warmup`個)を返し、続けて`run`を呼ぶと本サンプル(`num_samples`個)に置き換わる(上の出力)。
- `thinning=2`にすると保存されるサンプル数が半分(100→50)になる。

### `DiscreteHMCGibbs(...)`(離散潜在変数を含むモデル)

**用途**: 離散の潜在変数を、連続変数のNUTSと交互にGibbsサンプリングする複合カーネル。

**シグネチャ**: `numpyro.infer.DiscreteHMCGibbs(inner_kernel, *, random_walk=False, modified=False)`

**使用例**:
```python
from numpyro.infer import DiscreteHMCGibbs

rng2 = np.random.default_rng(0)
z_true = rng2.integers(0, 2, size=30)
y_mix = np.where(z_true == 1, 3.0, -3.0) + rng2.normal(size=30)

def model_discrete(y):
    p = numpyro.sample("p", dist.Beta(2, 2))
    with numpyro.plate("N", len(y)):
        z = numpyro.sample("z", dist.Bernoulli(p))
        numpyro.sample("y", dist.Normal(jnp.where(z == 1, 3.0, -3.0), 1.0), obs=y)

mcmc = MCMC(DiscreteHMCGibbs(NUTS(model_discrete)), num_warmup=100, num_samples=100, progress_bar=False)
mcmc.run(jax.random.key(0), y_mix)
s = mcmc.get_samples()
print({k: v.shape for k, v in s.items()}, s["z"].dtype)
print(round(float(s["p"].mean()), 3), z_true.mean())
print("z accuracy:", float((s["z"].mean(0).round() == z_true).mean()))

try:
    MCMC(NUTS(model_discrete), num_warmup=10, num_samples=10, progress_bar=False).run(jax.random.key(0), y_mix)
except Exception as e:
    print(type(e).__name__, str(e).split(".")[0])
```
実行結果:
```
{'p': (100,), 'z': (100, 30)} int32
0.631 0.6666666666666666
z accuracy: 1.0
ImportError Looking like you want to do inference for models with discrete latent variables
```

**注意点・落とし穴**:
- `NUTS`単体で離散の潜在変数を含むモデルを走らせると、上の最終行のように`ImportError`(`funsor`のインストールを求めるメッセージ。NumPyro 0.21.0では離散変数の自動周辺化に`funsor`が必要)になる。`DiscreteHMCGibbs`なら`funsor`なしで動く(上の実行結果)。
- 離散変数を周辺化して書き直せる場合は`MixtureSameFamily`(2章)を使う方法もあり、そのモデルは`NUTS`でそのまま推論できる。なお、上のように`NUTS(model_discrete)`を作る時点で`FutureWarning: Some algorithms will automatically enumerate the discrete latent site z of your model. In the future, enumerated sites need to be marked with infer={'enumerate': 'parallel'}`も出る(実測)。
- 上の例では、`z`の事後平均を四捨五入した値と真の`z`の一致率が`1.0`(全一致)だった。

### `BarkerMH(...)` / `SA(...)`(勾配を使わない/簡易なカーネル)

**用途**: `NUTS`以外のMCMCカーネル(`BarkerMH`はBarker提案のMetropolis-Hastings、`SA`はSample Adaptive MCMC)。

**シグネチャ**: `numpyro.infer.BarkerMH(model=None, potential_fn=None, step_size=1.0, adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False, target_accept_prob=0.4, init_strategy=init_to_uniform)` / `numpyro.infer.SA(model=None, potential_fn=None, adapt_state_size=None, dense_mass=True, init_strategy=init_to_uniform)`

**使用例**:
```python
from numpyro.infer import BarkerMH, SA

def model_simple(xs):
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    numpyro.sample("x", dist.Normal(mu, 1), obs=xs)
xs = jnp.array([0.5, 1.0, 1.5])
for K in (NUTS, BarkerMH, SA):
    m = MCMC(K(model_simple), num_warmup=200, num_samples=200, progress_bar=False)
    m.run(jax.random.key(0), xs)
    print(K.__name__, round(float(m.get_samples()["mu"].mean()), 2))
print("exact posterior mean:", 3 * 1.0 / 4)
```
実行結果:
```
NUTS 0.7
BarkerMH 0.74
SA -0.29
exact posterior mean: 0.75
```

**注意点・落とし穴**:
- この共役な例の事後平均の厳密値は`0.75`。`NUTS`と`BarkerMH`は近い値(0.7, 0.74)になったが、`SA`は今回の200サンプルでは大きくずれた(-0.29)。短い実行では`SA`は収束しないことがあるので、通常は`NUTS`で十分。

### `numpyro.diagnostics`(`summary` / `hpdi` / `effective_sample_size` / `split_gelman_rubin`)

**用途**: サンプル配列に対して、要約統計・最高事後密度区間・有効サンプルサイズ・`r_hat`を直接計算する。`MCMC`を使わない自前のサンプルにも使える。

**シグネチャ**:
- `numpyro.diagnostics.summary(samples, prob=0.9, group_by_chain=True)`
- `numpyro.diagnostics.hpdi(x, prob=0.9, axis=0)`
- `numpyro.diagnostics.effective_sample_size(x, bias=True)`
- `numpyro.diagnostics.split_gelman_rubin(x)`
- `numpyro.diagnostics.print_summary(samples, prob=0.9, group_by_chain=True)`

**使用例**:
```python
from numpyro.diagnostics import summary, hpdi, effective_sample_size, split_gelman_rubin, print_summary
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # 1デバイス環境の逐次実行に関する警告を抑制
    mc = MCMC(NUTS(model), num_warmup=200, num_samples=200, num_chains=2,
              chain_method="sequential", progress_bar=False)
    mc.run(jax.random.key(1), x, y)
g = mc.get_samples(group_by_chain=True)        # 形: (chain, draw)

st = summary(g, prob=0.9)
print({k: round(float(v), 3) for k, v in st["b"].items()})
print(hpdi(mc.get_samples()["b"], prob=0.9))
print(effective_sample_size(np.asarray(g["b"])), split_gelman_rubin(np.asarray(g["b"])))
print_summary(g, prob=0.9, group_by_chain=True)
```
実行結果:
```
{'mean': 1.939, 'std': 0.073, 'median': 1.942, '5.0%': 1.828, '95.0%': 2.056, 'n_eff': 365.969, 'r_hat': 1.003}
[1.8283077 2.0562947]
365.9687941937296 1.0034993

                mean       std    median      5.0%     95.0%     n_eff     r_hat
         a      1.02      0.07      1.02      0.91      1.14    306.45      1.00
         b      1.94      0.07      1.94      1.83      2.06    365.97      1.00
     sigma      0.52      0.06      0.52      0.43      0.61    398.23      1.00
```

**注意点・落とし穴**:
- `summary`/`print_summary`の入力は`{名前: 配列}`辞書(`group_by_chain=True`なら先頭軸がチェーン)。`n_eff`と`r_hat`はチェーンごとに分けた`(chain, draw)`形のサンプルから計算される。
- `hpdi`は最高事後密度区間`[下限, 上限]`を返す(`axis=0`がサンプル軸)。`effective_sample_size`と`split_gelman_rubin`は`(chain, draw, ...)`形の`numpy`配列を受け取る。

---

## 4. 変分推論(SVI)

`SVI`(確率的変分推論)は、事後分布を近似分布(ガイド)で置き換えてELBOを最大化する。MCMCより高速だが近似である。この章の例は3章と同じ単回帰データ・モデルを使う。

**共通コード**:
```python
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

rng = np.random.default_rng(0)
x = rng.normal(size=50)
y = 1.0 + 2.0 * x + rng.normal(scale=0.5, size=50)   # 真の値: a=1, b=2, sigma=0.5

def model(x, y=None):
    a = numpyro.sample("a", dist.Normal(0, 5))
    b = numpyro.sample("b", dist.Normal(0, 5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2))
    numpyro.sample("y", dist.Normal(a + b * x, sigma), obs=y)

import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
```

### `SVI(...)` と `SVI.run(...)`

**用途**: モデル・ガイド・オプティマイザ・損失(ELBO)を組み合わせ、`run`で最適化を実行する。

**シグネチャ**: `numpyro.infer.SVI(model, guide, optim, loss, **static_kwargs)` / `numpyro.infer.SVI.run(rng_key, num_steps, *args, progress_bar=True, stable_update=False, forward_mode_differentiation=False, init_state=None, init_params=None, **kwargs)`

**使用例**:
```python
guide = AutoNormal(model)
svi = SVI(model, guide, optim.Adam(0.05), Trace_ELBO())
result = svi.run(jax.random.key(0), 2000, x, y, progress_bar=False)
print(type(result).__name__, result._fields)
print(result.losses.shape, float(result.losses[0]), float(result.losses[-1]))
print({k: round(float(v), 3) for k, v in result.params.items()})
```
実行結果:
```
SVIRunResult ('params', 'state', 'losses')
(2000,) 160.53097534179688 47.556373596191406
{'a_auto_loc': 0.994, 'a_auto_scale': 0.069, 'b_auto_loc': 1.906, 'b_auto_scale': 0.075, 'sigma_auto_loc': -0.669, 'sigma_auto_scale': 0.096}
```

**注意点・落とし穴**:
- `svi.run(rng_key, num_steps, *model_args)`。`num_steps`は最適化のステップ数で、`result.losses`(各ステップの損失=−ELBO)が収束したか確認する。
- `result.params`の`*_auto_loc`/`*_auto_scale`は「制約なし空間」での近似分布のパラメータ。たとえば`sigma_auto_loc`は`log(sigma)`側の値で、`sigma`そのものではない(値を見るには下記の`sample_posterior`/`median`を使う)。
- 学習率や`num_steps`が不足すると収束しきらない。上の例では`b_auto_loc`が1.906で、3章のMCMCの事後平均(1.94)やMAP(下記`AutoDelta`の1.94)にまだ届いていない(学習率0.05・2000ステップ)。`losses`の推移を確認し、必要ならステップ数や学習率を調整する。

### `Trace_ELBO(...)` / `TraceMeanField_ELBO(...)`

**用途**: ELBO(変分下界)の推定器。`Trace_ELBO`はモンテカルロ推定、`TraceMeanField_ELBO`は解析的なKL項が利用できる場合にそれを使う。

**シグネチャ**: `numpyro.infer.Trace_ELBO(num_particles=1, vectorize_particles=True, multi_sample_guide=False, sum_sites=True)` / `numpyro.infer.TraceMeanField_ELBO(num_particles=1, vectorize_particles=True, sum_sites=True)`

**使用例**:
```python
from numpyro.infer import TraceMeanField_ELBO
for name, elbo in [("Trace_ELBO", Trace_ELBO()),
                   ("Trace_ELBO(num_particles=4)", Trace_ELBO(num_particles=4)),
                   ("TraceMeanField_ELBO", TraceMeanField_ELBO())]:
    r = SVI(model, AutoNormal(model), optim.Adam(0.05), elbo).run(
        jax.random.key(0), 500, x, y, progress_bar=False)
    print(name, round(float(r.losses[-1]), 2))
```
実行結果:
```
Trace_ELBO 48.16
Trace_ELBO(num_particles=4) 47.8
TraceMeanField_ELBO 47.2
```

**注意点・落とし穴**:
- `num_particles`を増やすと勾配のばらつきが減るが、1ステップの計算量が増える。
- `vectorize_particles`の既定は`True`で、複数particleを`jax.vmap`で並列に計算する(`False`なら`jax.lax.map`)。`losses`の値は最小化される損失(=−ELBO)。
- `TraceMeanField_ELBO`は解析的KLが使える場合にそれを使う唯一のELBO推定器だが、モデルとガイドの依存構造が平均場条件を満たさないと不正確な結果になりうる(NumPyroのdocstringの警告)。ガイドが`AutoNormal`などの平均場でモデルも素直な階層なら問題ない。

### `numpyro.optim.Adam` / `ClippedAdam` / `optax`

**用途**: SVIに渡すオプティマイザ。NumPyro自前のラッパーのほか、`optax`のオプティマイザもそのまま渡せる。

**シグネチャ**: `numpyro.optim.Adam(*args, **kwargs)` / `numpyro.optim.ClippedAdam(*args, clip_norm=10.0, **kwargs)`

**使用例**:
```python
import optax
for name, opt in [("numpyro Adam", optim.Adam(0.05)),
                  ("numpyro ClippedAdam", optim.ClippedAdam(0.05, clip_norm=10.0)),
                  ("optax.adam", optax.adam(0.05))]:
    r = SVI(model, AutoNormal(model), opt, Trace_ELBO()).run(
        jax.random.key(0), 500, x, y, progress_bar=False)
    print(name, round(float(r.losses[-1]), 2))
```
実行結果:
```
numpyro Adam 48.16
numpyro ClippedAdam 48.35
optax.adam 48.16
```

**注意点・落とし穴**:
- `optax`のオプティマイザ(上の`optax.adam(0.05)`)も`SVI`のオプティマイザ引数にそのまま渡せる(実行が成功することを確認済み)。
- `ClippedAdam`は勾配クリッピング付きのAdam(`clip_norm`)で、発散しやすいモデルの安定化に使う。

### `AutoNormal(...)`(平均場ガイド)

**用途**: 各潜在変数に独立な正規分布(制約なし空間)を当てはめる標準的な自動ガイド。`SVI`のガイドとして渡すだけで使える。

**シグネチャ**: `numpyro.infer.autoguide.AutoNormal(model, *, prefix='auto', init_loc_fn=init_to_uniform, init_scale=0.1, create_plates=None, forward_mode_differentiation=False)`

**使用例**:
```python
guide = AutoNormal(model)
result = SVI(model, guide, optim.Adam(0.1), Trace_ELBO()).run(
    jax.random.key(0), 1500, x, y, progress_bar=False)
post = guide.sample_posterior(jax.random.key(1), result.params, sample_shape=(1000,))
print({k: v.shape for k, v in post.items()})
print({k: round(float(v.mean()), 3) for k, v in post.items()})
print({k: round(float(v.std()), 3) for k, v in post.items()})
```
実行結果:
```
{'a': (1000,), 'b': (1000,), 'sigma': (1000,)}
{'a': 0.987, 'b': 1.903, 'sigma': 0.545}
{'a': 0.075, 'b': 0.086, 'sigma': 0.07}
```

**注意点・落とし穴**:
- `AutoNormal`は潜在変数を独立と見なす平均場近似のため、潜在変数間の相関を表現できない。相関を捉えたい場合は次項の`AutoMultivariateNormal`を使う。
- `sample_posterior(rng_key, params, sample_shape=...)`は、制約を元に戻した(constrained)スケール(例: `sigma`は正の値)でサンプルを返す。

### `AutoDiagonalNormal(...)` / `AutoMultivariateNormal(...)` / `AutoLowRankMultivariateNormal(...)`

**用途**: 全潜在変数を1つの連続ベクトルにまとめて正規分布で近似する自動ガイド。`Diagonal`は独立(対角共分散)、`MultivariateNormal`は共分散行列全体、`LowRank`は低ランク近似。

**シグネチャ**:
- `numpyro.infer.autoguide.AutoDiagonalNormal(model, *, prefix='auto', init_loc_fn=init_to_uniform, init_scale=0.1)`
- `numpyro.infer.autoguide.AutoMultivariateNormal(model, *, prefix='auto', init_loc_fn=init_to_uniform, init_scale=0.1)`
- `numpyro.infer.autoguide.AutoLowRankMultivariateNormal(model, *, prefix='auto', init_loc_fn=init_to_uniform, init_scale=0.1, rank=None)`

**使用例**:
```python
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal

g_diag = AutoDiagonalNormal(model)
r_diag = SVI(model, g_diag, optim.Adam(0.1), Trace_ELBO()).run(jax.random.key(0), 1500, x, y, progress_bar=False)
print(list(r_diag.params.keys()))
print(g_diag.get_posterior(r_diag.params).loc)

g_mvn = AutoMultivariateNormal(model)
r_mvn = SVI(model, g_mvn, optim.Adam(0.1), Trace_ELBO()).run(jax.random.key(0), 1500, x, y, progress_bar=False)
print(list(r_mvn.params.keys()))
q = g_mvn.get_posterior(r_mvn.params)
print(type(q).__name__, q.loc, q.covariance_matrix.shape)

from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
g_lr = AutoLowRankMultivariateNormal(model, rank=2)
r_lr = SVI(model, g_lr, optim.Adam(0.1), Trace_ELBO()).run(jax.random.key(0), 1500, x, y, progress_bar=False)
print(list(r_lr.params.keys()))
```
実行結果:
```
['auto_loc', 'auto_scale']
[ 0.9831638  1.9497685 -0.7020489]
['auto_loc', 'auto_scale_tril']
MultivariateNormal [ 0.97807187  1.9645947  -0.7202276 ] (3, 3)
['auto_cov_factor', 'auto_loc', 'auto_scale']
```

**注意点・落とし穴**:
- パラメータ名は`AutoNormal`と違い、全潜在変数を連結した`auto_loc`と`auto_scale`(`AutoMultivariateNormal`は`auto_scale_tril`)にまとめられる。`get_posterior(params)`で近似事後分布(制約なし空間の`Normal`/`MultivariateNormal`)を取得できる。
- `get_posterior`が返すのは制約なし空間の分布であり、`sigma`のような制約付き変数は`log`などの変換後の値になっている(上の`loc`の3番目が`sigma`の対数)。制約付き空間の値が欲しい場合は`sample_posterior`を使う。

### `AutoDelta(...)` / `AutoLaplaceApproximation(...)`

**用途**: `AutoDelta`は事後分布をデルタ関数(点推定=MAP推定)で近似、`AutoLaplaceApproximation`はMAPの周りのラプラス近似(ヘッセ行列から正規近似)を返す。

**シグネチャ**: `numpyro.infer.autoguide.AutoDelta(model, *, prefix='auto', init_loc_fn=init_to_median, create_plates=None, forward_mode_differentiation=False)` / `numpyro.infer.autoguide.AutoLaplaceApproximation(model, *, prefix='auto', init_loc_fn=init_to_uniform, create_plates=None, hessian_fn=None)`

**使用例**:
```python
from numpyro.infer.autoguide import AutoDelta, AutoLaplaceApproximation

g_map = AutoDelta(model)
r_map = SVI(model, g_map, optim.Adam(0.05), Trace_ELBO()).run(jax.random.key(0), 1500, x, y, progress_bar=False)
print({k: round(float(v), 3) for k, v in r_map.params.items()})
print({k: round(float(v), 3) for k, v in g_map.median(r_map.params).items()})

g_lap = AutoLaplaceApproximation(model)
r_lap = SVI(model, g_lap, optim.Adam(0.05), Trace_ELBO()).run(jax.random.key(0), 1500, x, y, progress_bar=False)
p_lap = g_lap.sample_posterior(jax.random.key(3), r_lap.params, sample_shape=(500,))
print({k: round(float(v.mean()), 3) for k, v in p_lap.items()})
```
実行結果:
```
{'a_auto_loc': 1.024, 'b_auto_loc': 1.94, 'sigma_auto_loc': 0.501}
{'a': 1.024, 'b': 1.94, 'sigma': 0.501}
{'a': 1.02, 'b': 1.941, 'sigma': 0.505}
```

**注意点・落とし穴**:
- `AutoDelta`のパラメータ名は`*_auto_loc`だが、その値は点推定値(MAP)。今回の例では`sigma`が制約付きなので`params`の値(制約後)と`median`が一致している。

### `guide.sample_posterior` / `guide.median` / `guide.quantiles`

**用途**: 学習済みの自動ガイドから、事後サンプル・中央値・分位数を取り出す。

**シグネチャ**:
- `numpyro.infer.autoguide.AutoNormal.sample_posterior(rng_key, params, *args, sample_shape=(), **kwargs)`
- `numpyro.infer.autoguide.AutoNormal.median(params)`
- `numpyro.infer.autoguide.AutoNormal.quantiles(params, quantiles)`

**使用例**:
```python
guide = AutoNormal(model)
result = SVI(model, guide, optim.Adam(0.1), Trace_ELBO()).run(jax.random.key(0), 1500, x, y, progress_bar=False)
print({k: round(float(v), 3) for k, v in guide.median(result.params).items()})
q = guide.quantiles(result.params, [0.05, 0.95])
print({k: np.round(np.asarray(v), 3) for k, v in q.items()})
print(list(guide.sample_posterior(jax.random.key(1), result.params).keys()))
```
実行結果:
```
{'a': 0.985, 'b': 1.901, 'sigma': 0.54}
{'a': array([0.862, 1.108], dtype=float32), 'b': array([1.752, 2.049], dtype=float32), 'sigma': array([0.444, 0.658], dtype=float32)}
['a', 'b', 'sigma']
```

**注意点・落とし穴**:
- `median`/`quantiles`は制約付き空間(例: `sigma`は正)の値で返る。`sample_posterior`は`sample_shape`を省略すると1サンプル分を返す。

### 手書きガイド(`numpyro.param` によるカスタム近似分布)

**用途**: 自動ガイドを使わず、`numpyro.sample`と`numpyro.param`でガイド関数を自分で書く。事後分布の形を自分で設計したいときに使う。

**使用例**:
```python
def guide_manual(x, y=None):
    a_loc = numpyro.param("a_loc", 0.0)
    a_scale = numpyro.param("a_scale", 1.0, constraint=dist.constraints.positive)
    b_loc = numpyro.param("b_loc", 0.0)
    b_scale = numpyro.param("b_scale", 1.0, constraint=dist.constraints.positive)
    s_loc = numpyro.param("s_loc", 0.0)
    s_scale = numpyro.param("s_scale", 0.1, constraint=dist.constraints.positive)
    numpyro.sample("a", dist.Normal(a_loc, a_scale))
    numpyro.sample("b", dist.Normal(b_loc, b_scale))
    numpyro.sample("sigma", dist.LogNormal(s_loc, s_scale))

r = SVI(model, guide_manual, optim.Adam(0.05), Trace_ELBO()).run(jax.random.key(0), 3000, x, y, progress_bar=False)
print({k: round(float(v), 3) for k, v in r.params.items()})
```
実行結果:
```
{'a_loc': 1.008, 'a_scale': 0.073, 'b_loc': 1.867, 'b_scale': 0.083, 's_loc': -0.683, 's_scale': 0.082}
```

**注意点・落とし穴**:
- ガイド内のサンプルサイトの名前は、モデル内の潜在変数の名前と完全に一致させる必要があり、観測サイト(`y`)はガイドに書かない。ガイドの分布のサポートはモデル側の事前分布のサポートと合わせる(`sigma`は正なのでガイドでも`LogNormal`を使っている)。
- ガイドの引数はモデルと同じシグネチャで呼ばれる(上の`guide_manual(x, y=None)`)。

### `SVI.init` / `SVI.update` / `SVI.evaluate`(手動ループ)

**用途**: `run`を使わず、1ステップずつ更新する。学習の途中で任意の処理を挟みたいとき、または`jax.lax.scan`/`jit`で自前のループを書くときに使う。

**シグネチャ**:
- `numpyro.infer.SVI.init(rng_key, *args, init_params=None, **kwargs)`
- `numpyro.infer.SVI.update(svi_state, *args, forward_mode_differentiation=False, **kwargs)`
- `numpyro.infer.SVI.evaluate(svi_state, *args, **kwargs)`
- `numpyro.infer.SVI.get_params(svi_state)`

**使用例**:
```python
svi = SVI(model, AutoNormal(model), optim.Adam(0.05), Trace_ELBO())
state = svi.init(jax.random.key(0), x, y)
for i in range(3):
    state, loss = svi.update(state, x, y)
    print(i, float(loss))
print(float(svi.evaluate(state, x, y)))
print(list(svi.get_params(state).keys()))
```
実行結果:
```
0 160.53097534179688
1 154.11305236816406
2 145.76263427734375
141.16622924804688
['a_auto_loc', 'a_auto_scale', 'b_auto_loc', 'b_auto_scale', 'sigma_auto_loc', 'sigma_auto_scale']
```

**注意点・落とし穴**:
- `svi.update(state, *args)`は`(新しいstate, 損失)`を返す。パラメータの取り出しは`svi.get_params(state)`。

---

## 5. 予測とログ尤度

`Predictive`は、モデルを事前分布・事後サンプル・変分ガイドのいずれかで条件付けて、観測サイトや`deterministic`サイトの値をシミュレーションする。この章では、3章と同じ単回帰に`mu`の`deterministic`サイトを加えたモデルを事前にMCMCで推論済みとする。

**共通コード**:
```python
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

rng = np.random.default_rng(0)
x = rng.normal(size=50)
y = 1.0 + 2.0 * x + rng.normal(scale=0.5, size=50)   # 真の値: a=1, b=2, sigma=0.5

def model(x, y=None):
    a = numpyro.sample("a", dist.Normal(0, 5))
    b = numpyro.sample("b", dist.Normal(0, 5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2))
    mu = numpyro.deterministic("mu", a + b * x)
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood

mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=400, progress_bar=False)
mcmc.run(jax.random.key(0), x, y)
samples = mcmc.get_samples()
```

### `Predictive(...)`(事前予測)

**用途**: `posterior_samples`を渡さずに`num_samples`だけ指定すると、事前分布からモデルをシミュレーションする(事前予測チェック)。

**シグネチャ**: `numpyro.infer.Predictive(model, posterior_samples=None, *, guide=None, params=None, num_samples=None, return_sites=None, infer_discrete=False, parallel=False, batch_ndims=None, exclude_deterministic=True)`

**使用例**:
```python
prior = Predictive(model, num_samples=100)(jax.random.key(1), x)
print({k: v.shape for k, v in prior.items()})
try:
    Predictive(model)
except ValueError as e:
    print("ValueError:", e)
print(list(Predictive(model, num_samples=10)(jax.random.key(1), x, y).keys()))   # yを渡した場合
```
実行結果:
```
{'a': (100,), 'b': (100,), 'mu': (100, 50), 'sigma': (100,), 'y': (100, 50)}
ValueError: Either posterior_samples or num_samples must be specified.
['a', 'b', 'mu', 'sigma', 'y']
```

**注意点・落とし穴**:
- `Predictive(model, ...)`はインスタンスを作った後に`(rng_key, *model_args)`で呼び出す。返り値は`{サイト名: 配列}`辞書で、先頭次元がサンプル数。
- `posterior_samples`も`num_samples`も指定しないと`ValueError: Either posterior_samples or num_samples must be specified.`。
- 予測したい観測サイトは`y=None`で呼ぶ必要がある(NumPyroのdocstringにも「観測変数は`None`にセットする」とある)。上の最終行のように`x, y`を両方渡すと、戻り値の`y`は予測ではなく観測値そのもの(を`num_samples`個並べたもの)になる(`Predictive(model, num_samples=3)`で確認すると、各行が観測`y`と一致する)。

### `Predictive(model, posterior_samples)(...)`(事後予測・新データへの予測)

**用途**: MCMCの事後サンプルを渡して、各サンプルでのモデル出力(観測値の予測分布)を得る。新しい説明変数を渡せば未知データへの予測になる。

**使用例**:
```python
pp = Predictive(model, posterior_samples=samples)(jax.random.key(2), x)
print({k: v.shape for k, v in pp.items()})
print(float(pp["y"].mean()), float(y.mean()))

xnew = jnp.linspace(-2, 2, 5)
pn = Predictive(model, samples)(jax.random.key(3), xnew)     # yを渡さない(=予測したい)
print(pn["y"].shape, pn["mu"].shape)
print(np.round(np.asarray(pn["mu"].mean(0)), 2))
```
実行結果:
```
{'mu': (400, 50), 'y': (400, 50)}
1.2680031061172485 1.2745913998352187
(400, 5) (400, 5)
[-2.86 -0.92  1.02  2.96  4.91]
```

**注意点・落とし穴**:
- 予測したい観測サイトは、`Predictive`呼び出し時にモデルへ`y`を渡さない(`y=None`)ことで確率変数として扱われる。`y`を渡してしまうと観測値に固定され、そのサイトの予測は得られない。
- 既定(`return_sites=None`)では、`posterior_samples`に含まれるサイト(`a`,`b`,`sigma`)は戻り値に含まれず、それ以外のサンプルサイト(`y`)と`deterministic`サイト(`mu`)が返る(上の出力)。含めたい場合は`return_sites`で明示する(次項)。

### `Predictive(..., return_sites=...)` と `exclude_deterministic`

**用途**: 返すサイトを指定する。`deterministic`サイトを含む事後サンプルで新しい入力に予測する際の落とし穴にも関わる。

**使用例**:
```python
xnew = jnp.linspace(-2, 2, 5)
print(list(samples.keys()))                     # MCMCの結果にはdeterministic 'mu'も含まれる
p1 = Predictive(model, samples, return_sites=["y"])(jax.random.key(3), xnew)
print(list(p1.keys()))
p2 = Predictive(model, samples, return_sites=["a", "mu", "y"])(jax.random.key(3), xnew)
print(list(p2.keys()))

for ed in (True, False):
    p = Predictive(model, samples, exclude_deterministic=ed)(jax.random.key(3), xnew)
    print("exclude_deterministic =", ed, {k: v.shape for k, v in p.items()})
```
実行結果:
```
['a', 'b', 'mu', 'sigma']
['y']
['a', 'mu', 'y']
exclude_deterministic = True {'mu': (400, 5), 'y': (400, 5)}
exclude_deterministic = False {'mu': (400, 50), 'y': (400, 50)}
```

**注意点・落とし穴**:
- **落とし穴(実測で確認)**: `mcmc.get_samples()`には訓練時の`numpyro.deterministic`サイト(ここでは`mu`、形`(400, 50)`)が含まれる。`exclude_deterministic=False`にすると、それが再計算されず古い値(訓練データ`x`に対する`(400, 50)`)のまま使われ、新しい`xnew`(5点)に対する予測にならない。既定の`True`なら`mu`は再計算されて`(400, 5)`になる。
- したがって新しい入力に対して予測するときは、`exclude_deterministic`は既定(`True`)のまま使う。
- `return_sites`に`posterior_samples`側のサイト(`a`など)を指定すると、そのサンプルもそのまま結果に含まれる。

### `Predictive(model, guide=..., params=..., num_samples=...)`(SVIの結果から予測)

**用途**: SVIで学習したガイドと`params`から、事後からのサンプリングと予測を一度に行う。

**使用例**:
```python
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

guide = AutoNormal(model)
res = SVI(model, guide, optim.Adam(0.1), Trace_ELBO()).run(jax.random.key(0), 1500, x, y, progress_bar=False)
xnew = jnp.linspace(-2, 2, 5)
pred = Predictive(model, guide=guide, params=res.params, num_samples=200)(jax.random.key(4), xnew)
print({k: v.shape for k, v in pred.items()})
pred2 = Predictive(model, guide=guide, params=res.params, num_samples=200,
                   return_sites=["a", "b", "sigma", "y"])(jax.random.key(4), xnew)
print({k: v.shape for k, v in pred2.items()})
```
実行結果:
```
{'mu': (200, 5), 'y': (200, 5)}
{'a': (200,), 'b': (200,), 'sigma': (200,), 'y': (200, 5)}
```

**注意点・落とし穴**:
- `guide`と`params`(`svi.run(...).params`)、および`num_samples`をセットで指定する(`posterior_samples`は不要)。既定ではガイドが生成した潜在変数(`a`,`b`,`sigma`)は返らず、モデルの出力(`mu`,`y`)だけが返る。潜在変数も欲しければ`return_sites`に列挙する。

### `Predictive(..., batch_ndims=...)` / `parallel=`

**用途**: 複数チェーンごとの形(`(chain, draw, ...)`)のサンプルをそのまま渡す、あるいは`vmap`で並列化する。

**使用例**:
```python
sg = mcmc.get_samples(group_by_chain=True)
print({k: v.shape for k, v in sg.items()})
xnew = jnp.linspace(-2, 2, 5)
p = Predictive(model, sg, batch_ndims=2)(jax.random.key(3), xnew)
print({k: v.shape for k, v in p.items()})
pp = Predictive(model, samples, parallel=True)(jax.random.key(3), xnew)
print(pp["y"].shape)
```
実行結果:
```
{'a': (1, 400), 'b': (1, 400), 'mu': (1, 400, 50), 'sigma': (1, 400)}
{'mu': (1, 400, 5), 'y': (1, 400, 5)}
(400, 5)
```

**注意点・落とし穴**:
- `batch_ndims`は`posterior_samples`(または`params`)の先頭にあるバッチ次元の数。docstringによれば`None`のときは、`guide`なしなら`1`(draw次元のみ)、`guide`ありなら`0`になる。`group_by_chain=True`のサンプルを渡すなら`2`を指定し、出力も`(chain, draw, ...)`形になる(上の出力)。

### `log_likelihood(...)`

**用途**: 各事後サンプルで、各観測点の対数尤度を計算する。LOO/WAICなどのモデル比較の入力になる。

**シグネチャ**: `numpyro.infer.log_likelihood(model, posterior_samples, *args, parallel=False, batch_ndims=1, **kwargs)`

**使用例**:
```python
ll = log_likelihood(model, samples, x, y)
print({k: v.shape for k, v in ll.items()})
lppd = jax.scipy.special.logsumexp(ll["y"], axis=0) - jnp.log(ll["y"].shape[0])
print(float(lppd.sum()))
```
実行結果:
```
{'y': (400, 50)}
-36.727317810058594
```

**注意点・落とし穴**:
- 戻り値は観測サイト名をキーとし、形が`(サンプル数, 観測数)`の辞書。上の`lppd`は各観測点の対数平均尤度(事後平均尤度の対数)で、その合計がlppd(対数点予測密度)。
- ArviZ連携(9章)では`az.from_numpyro(mcmc, log_likelihood=True)`が内部でこれを計算してくれる。

---

## 6. 階層モデルと時系列

グループごとのパラメータを共通の事前分布から生成する階層モデル、その再パラメータ化(non-centered)、および`scan`を使った時系列モデル。この章の例は次の共通コードを前提とする(6グループ・各10個の観測)。

**共通コード**:
```python
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

rng = np.random.default_rng(1)
J, n = 6, 10
true_mu = rng.normal(2.0, 1.0, size=J)
group = np.repeat(np.arange(J), n)
y = true_mu[group] + rng.normal(0, 0.5, size=J * n)
```

### 階層モデル(グループ別パラメータ、centered版)

**用途**: グループごとの平均`theta`を、共通の`mu`と`tau`から生成する。`plate`でグループ次元と観測次元を宣言し、`theta[group]`で観測ごとに対応するグループのパラメータを引く。

**使用例**:
```python
def hier(group, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfNormal(2))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    with numpyro.plate("groups", J):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
    with numpyro.plate("obs", len(group)):
        numpyro.sample("y", dist.Normal(theta[group], sigma), obs=y)

tr = handlers.trace(handlers.seed(hier, 0)).get_trace(group)
for k, v in tr.items():
    print(k, v["type"], v["fn"].batch_shape if v["type"] == "sample" else "-", [f.name for f in v["cond_indep_stack"]])

mcmc = MCMC(NUTS(hier), num_warmup=300, num_samples=300, progress_bar=False)
mcmc.run(jax.random.key(0), group, y, extra_fields=("diverging",))
print({k: v.shape for k, v in mcmc.get_samples().items()})
print("divergences:", int(mcmc.get_extra_fields()["diverging"].sum()))
mcmc.print_summary()
print("true mu:", np.round(true_mu, 2))
print("group means:", np.round(np.array([y[group == j].mean() for j in range(J)]), 2))
print("theta means:", np.round(np.asarray(mcmc.get_samples()["theta"].mean(0)), 2))
print("mu mean:", round(float(mcmc.get_samples()["mu"].mean()), 2))
```
実行結果:
```
mu sample () []
tau sample () []
sigma sample () []
groups plate - []
theta sample (6,) ['groups']
obs plate - []
y sample (60,) ['obs']
{'mu': (300,), 'sigma': (300,), 'tau': (300,), 'theta': (300, 6)}
divergences: 0

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      2.18      0.41      2.21      1.60      2.95    461.67      1.00
     sigma      0.43      0.05      0.42      0.34      0.49    331.80      1.00
       tau      1.04      0.38      0.96      0.51      1.56    305.72      1.00
  theta[0]      2.37      0.14      2.38      2.15      2.58    582.10      1.00
  theta[1]      2.62      0.11      2.62      2.45      2.80    486.33      1.00
  theta[2]      2.52      0.14      2.51      2.25      2.73    427.42      1.00
  theta[3]      0.50      0.13      0.49      0.31      0.73    657.81      1.00
  theta[4]      2.92      0.12      2.92      2.74      3.12    611.18      1.00
  theta[5]      2.31      0.13      2.31      2.08      2.52    459.82      1.00

Number of divergences: 0
true mu: [2.35 2.82 2.33 0.7  2.91 2.45]
group means: [2.37 2.63 2.52 0.46 2.93 2.31]
theta means: [2.37 2.62 2.52 0.5  2.92 2.31]
mu mean: 2.18
```

**注意点・落とし穴**:
- `plate`はサイトの`batch_shape`に反映される(`theta`は`(6,)`、`y`は`(60,)`)。`theta[group]`のようにインデックス配列でグループ値を観測ごとに展開する。
- 各グループの`theta[i]`は、そのグループの標本平均(上の`group means`)と全体平均`mu`の間に引き寄せられる(部分プーリング)。
- この合成データでは発散は0だった(上の出力)。一般に、グループ数が多くデータが少ない場合に`tau`が小さい領域で漏斗(funnel)状の事後分布となり発散が出やすくなるので、発散が出た場合は次項のnon-centered表現を試す。

### non-centered パラメータ化(手動)

**用途**: `theta = mu + tau * z`(`z ~ Normal(0,1)`)と書き換えて、`mu`/`tau`と`theta`の強い依存を分離し、漏斗構造による発散を避ける。

**使用例**:
```python
def hier_nc(group, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfNormal(2))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    with numpyro.plate("groups", J):
        z = numpyro.sample("z", dist.Normal(0, 1))
        theta = numpyro.deterministic("theta", mu + tau * z)
    with numpyro.plate("obs", len(group)):
        numpyro.sample("y", dist.Normal(theta[group], sigma), obs=y)

mcmc = MCMC(NUTS(hier_nc), num_warmup=300, num_samples=300, progress_bar=False)
mcmc.run(jax.random.key(0), group, y, extra_fields=("diverging",))
s = mcmc.get_samples()
print({k: v.shape for k, v in s.items()})
print(np.round(np.asarray(s["theta"].mean(0)), 2))
print(np.round(true_mu, 2))
print("divergences:", int(mcmc.get_extra_fields()["diverging"].sum()))
```
実行結果:
```
{'mu': (300,), 'sigma': (300,), 'tau': (300,), 'theta': (300, 6), 'z': (300, 6)}
[2.36 2.62 2.53 0.5  2.92 2.31]
[2.35 2.82 2.33 0.7  2.91 2.45]
divergences: 0
```

**注意点・落とし穴**:
- `theta`を`deterministic`にして結果に残す。`z`は標準正規なので、`mu`・`tau`と`theta`の依存が切れる。データが少ないグループが多いときには、一般にこの方が`centered`版より安定するとされる(この合成データでは、どちらも発散は0だった)。

### `handlers.reparam(..., config={...: LocScaleReparam(...)})`(自動 non-centered 化)

**用途**: モデルコードを書き換えずに、ハンドラで指定サイトをnon-centered化する。`LocScaleReparam(centered=0)`が完全なnon-centered表現。

**シグネチャ**: `numpyro.infer.reparam.LocScaleReparam(centered=None, shape_params=())` / `numpyro.handlers.reparam(fn=None, config=None)`

**使用例**:
```python
from numpyro.infer.reparam import LocScaleReparam

def hier(group, y=None):            # 前項のcentered版と同じ
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfNormal(2))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    with numpyro.plate("groups", J):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
    with numpyro.plate("obs", len(group)):
        numpyro.sample("y", dist.Normal(theta[group], sigma), obs=y)

reparam_model = handlers.reparam(hier, config={"theta": LocScaleReparam(centered=0)})
mcmc = MCMC(NUTS(reparam_model), num_warmup=300, num_samples=300, progress_bar=False)
mcmc.run(jax.random.key(0), group, y)
s = mcmc.get_samples()
print(list(s.keys()))
print(np.round(np.asarray(s["theta"].mean(0)), 2))
```
実行結果:
```
['mu', 'sigma', 'tau', 'theta', 'theta_decentered']
[2.36 2.62 2.53 0.5  2.92 2.31]
```

**注意点・落とし穴**:
- `config`のキーは対象サイト名(`"theta"`)、値は`Reparam`オブジェクト。結果には元の`theta`に加えて、内部で導入された`theta_decentered`(標準化された変数)が入る。
- `centered`は`0`(完全なnon-centered)〜`1`(元のcentered)の値を取る。省略(`None`)すると各サイト・各要素ごとに学習される係数(初期値0.5)になり、これを潜在変数として扱いたい場合はdocstringによれば`handlers.lift`で事前分布(例: `Uniform(0, 1)`)を与える。

### `scan(...)`(`numpyro.contrib.control_flow.scan`)

**用途**: 時系列など逐次的な構造を`jax.lax.scan`ベースで記述しつつ、ループ内で`numpyro.sample`を使える。

**シグネチャ**: `numpyro.contrib.control_flow.scan(f, init, xs, length=None, reverse=False, history=1)`

**使用例**:
```python
from numpyro.contrib.control_flow import scan

def ar1(T, y=None):
    phi = numpyro.sample("phi", dist.Uniform(-1, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    def step(prev, y_t):
        mu = phi * prev
        y_t = numpyro.sample("y", dist.Normal(mu, sigma), obs=y_t)
        return y_t, y_t                      # (次のcarry, 出力)
    _, ys = scan(step, 0.0, y, length=T)
    return ys

r = np.random.default_rng(0)
series = np.zeros(100)
for t in range(1, 100):
    series[t] = 0.7 * series[t - 1] + r.normal(scale=0.5)   # 真の値: phi=0.7, sigma=0.5

mcmc = MCMC(NUTS(ar1), num_warmup=200, num_samples=200, progress_bar=False)
mcmc.run(jax.random.key(0), 100, jnp.asarray(series))
s = mcmc.get_samples()
print(round(float(s["phi"].mean()), 2), round(float(s["sigma"].mean()), 2))
print(handlers.seed(ar1, 0)(5).shape)
tr = handlers.trace(handlers.seed(ar1, 0)).get_trace(5)
print(tr["y"]["value"].shape)
```
実行結果:
```
0.76 0.49
(5,)
(5,)
```

**注意点・落とし穴**:
- `scan(f, init, xs, length=None)`の`f`は`(carry, x) -> (carry, y)`の形。`xs=None`でも`length`を渡せば繰り返し回数を指定できる(上の`handlers.seed(ar1, 0)(5)`は観測なしで5ステップ生成)。
- ループ内で宣言した`numpyro.sample("y", ...)`は、時間軸方向に`T`個まとまった1つのサイトとしてトレースに現れる(上の最終行`(5,)`)。

### `GaussianRandomWalk(...)`

**用途**: ランダムウォーク(累積和)の分布。時系列のトレンド成分などに使える。

**シグネチャ**: `numpyro.distributions.GaussianRandomWalk(scale=1.0, num_steps=1, *, validate_args=None)`

**使用例**:
```python
d = dist.GaussianRandomWalk(scale=1.0, num_steps=5)
print(d.sample(jax.random.key(0)))
print(d.event_shape, d.batch_shape)
```
実行結果:
```
[1.6226422 3.6479068 3.2143123 3.135695  3.311786 ]
(5,) ()
```

**注意点・落とし穴**:
- `num_steps`が系列の長さで、`event_shape`は`(num_steps,)`。1ステップ目は`0`からの増分(`Normal(0, scale)`)。

---

## 7. 制約と変数変換

NumPyroは、制約付きパラメータ(正の値、0〜1、シンプレックスなど)を内部で「制約なし空間」に変換してからMCMC/SVIを行う。この変換は`constraints`(制約の定義)と`biject_to`(制約なし空間との全単射)で表現される。この章の例は次の共通コードを前提とする。

**共通コード**:
```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.distributions.transforms import biject_to
```

### `numpyro.distributions.constraints`

**用途**: 値が満たすべき制約(正、区間、シンプレックスなど)を表すオブジェクト。呼び出すと配列がその制約を満たすかの真偽値を返す。分布の`.support`や`numpyro.param(constraint=...)`で使う。

**使用例**:
```python
print(constraints.positive(jnp.array([-1.0, 0.0, 2.0])))
print(constraints.unit_interval(jnp.array([-1.0, 0.5, 2.0])))
print(constraints.simplex(jnp.array([0.2, 0.3, 0.5])))
print(constraints.interval(0.0, 10.0)(jnp.array([5.0, 11.0])))
print(constraints.greater_than(2.0), constraints.interval(0.0, 10.0), constraints.positive)
print(dist.Gamma(2.0, 1.0).support, dist.Beta(2.0, 3.0).support, dist.Dirichlet(jnp.ones(3)).support, dist.LKJCholesky(3).support)
```
実行結果:
```
[False False  True]
[False  True False]
True
[ True False]
GreaterThan(lower_bound=2.0) Interval(lower_bound=0.0, upper_bound=10.0) Positive(lower_bound=0.0)
Positive(lower_bound=0.0) UnitInterval(lower_bound=0.0, upper_bound=1.0) Simplex() CorrCholesky()
```

**注意点・落とし穴**:
- `constraints.positive`は厳密に0より大きい値のみ真(上の出力: `0.0`は`False`)。0以上を許す場合は`constraints.nonnegative`。
- 主な制約: `real`・`positive`・`unit_interval`・`interval(low, high)`・`greater_than(x)`・`less_than(x)`・`simplex`・`lower_cholesky`・`corr_cholesky`・`ordered_vector`・`positive_definite`(`dir(constraints)`で確認)。

### `biject_to(constraint)`

**用途**: 制約に対応する全単射(制約なし空間 → 制約付き空間)の`Transform`を返す。逆変換は`.inv`で得る。

**使用例**:
```python
t = biject_to(constraints.positive)
print(type(t).__name__)
u = jnp.array([-2.0, 0.0, 2.0])
print(t(u), t.inv(t(u)))
print(t.log_abs_det_jacobian(u, t(u)))

t2 = biject_to(constraints.interval(0.0, 10.0))
print(type(t2).__name__, t2(u))

t3 = biject_to(constraints.simplex)
v = t3(jnp.array([0.5, -0.5]))
print(type(t3).__name__, v, v.sum(), t3.forward_shape((2,)), t3.inverse_shape((3,)))

print(type(biject_to(constraints.lower_cholesky)).__name__, type(biject_to(constraints.corr_cholesky)).__name__)
print(biject_to(constraints.ordered_vector)(jnp.array([0.0, 1.0, 1.0])))
print(biject_to(dist.Beta(2.0, 3.0).support).inv(jnp.array(0.3)))
```
実行結果:
```
ExpTransform
[0.13533528 1.         7.389056  ] [-2.  0.  2.]
[-2.  0.  2.]
ComposeTransform [1.1920292 5.        8.80797  ]
StickBreakingTransform [0.45186275 0.20694411 0.3411931 ] 1.0 (3,) (2,)
LowerCholeskyTransform CorrCholeskyTransform
[0.        2.7182817 5.4365635]
-0.8472978
```

**注意点・落とし穴**:
- `positive`は`ExpTransform`、`interval`は複合変換(`ComposeTransform`)、`simplex`は`StickBreakingTransform`(次元が1つ減る: 制約なし空間`(2,)`↔シンプレックス`(3,)`)に対応する(上の出力)。
- `t.inv`をたどると浮動小数点の丸め誤差が出る(区間変換で`2.0`が`1.9999995`のように戻る例が実測されている)。厳密一致は期待しない。

### `Transform` クラス(`AffineTransform` / `ComposeTransform` など)

**用途**: 可逆変換の基本部品。`TransformedDistribution`(2章)やNormalizing Flowの構成要素になる。

**シグネチャ**: `numpyro.distributions.transforms.AffineTransform(loc, scale, domain=Real())`

**使用例**:
```python
aff = transforms.AffineTransform(loc=1.0, scale=2.0)
print(aff(jnp.array([0.0, 1.0])), aff.inv(jnp.array([1.0, 3.0])), aff.log_abs_det_jacobian(0.0, 1.0))

comp = transforms.ComposeTransform([transforms.ExpTransform(), transforms.AffineTransform(1.0, 2.0)])
print(comp(jnp.array([0.0, 1.0])))
print(transforms.SigmoidTransform()(jnp.array([0.0])), transforms.SoftplusTransform()(jnp.array([0.0])))
```
実行結果:
```
[1. 3.] [0. 1.] 0.6931472
[3.        6.4365635]
[0.5] [0.6931472]
```

**注意点・落とし穴**:
- `ComposeTransform`は左から右に順に適用される(上の`Exp`してから`Affine`)。`log_abs_det_jacobian(x, y)`は入力`x`と出力`y`を取る。

### `numpyro.infer.util.log_density(...)`

**用途**: モデルの結合対数密度(尤度+事前分布)を、パラメータ値を与えて直接計算する。

**シグネチャ**: `numpyro.infer.util.log_density(model, model_args, model_kwargs, params)`

**使用例**:
```python
from numpyro.infer.util import log_density

def model_s():
    s = numpyro.sample("s", dist.HalfNormal(1.0))
    numpyro.sample("y", dist.Normal(0.0, s), obs=jnp.array([0.5, -0.3]))

ld, trace = log_density(model_s, (), {}, {"s": 1.0})
print(ld, type(trace).__name__)
print(dist.HalfNormal(1.0).log_prob(1.0) + dist.Normal(0.0, 1.0).log_prob(jnp.array([0.5, -0.3])).sum())
```
実行結果:
```
-2.7336683 OrderedDict
-2.7336683
```

**注意点・落とし穴**:
- 引数は`(model, model_args, model_kwargs, params)`(第2・3引数は組`()`と辞書`{}`)。`params`は制約付き空間の値で、ヤコビアン補正は含まれない(手計算の`log_prob`の合計と一致する)。

### `potential_energy` / `constrain_fn` / `unconstrain_fn` / `initialize_model`

**用途**: NUTSが内部で使う関数群。制約なし空間の値から潜在エネルギー(−対数事後+ヤコビアン補正)を計算したり、制約付き空間との間で値を変換したりできる。

**シグネチャ**:
- `numpyro.infer.util.potential_energy(model, model_args, model_kwargs, params, enum=False)`
- `numpyro.infer.util.constrain_fn(model, model_args, model_kwargs, params, return_deterministic=False)`
- `numpyro.infer.util.unconstrain_fn(model, model_args, model_kwargs, params)`
- `numpyro.infer.util.initialize_model(rng_key, model, *, init_strategy=init_to_uniform, dynamic_args=False, model_args=(), model_kwargs=None, forward_mode_differentiation=False, validate_grad=True)`

**使用例**:
```python
from numpyro.infer.util import log_density, potential_energy, constrain_fn, unconstrain_fn, initialize_model

u = jnp.array(1.0)                                     # 制約なし空間の値(s = exp(1))
print(float(potential_energy(model_s, (), {}, {"s": u})))
print(float(-log_density(model_s, (), {}, {"s": jnp.exp(u)})[0]))
print(constrain_fn(model_s, (), {}, {"s": jnp.array(0.0)}))
print(unconstrain_fn(model_s, (), {}, {"s": jnp.array(2.0)}))

info = initialize_model(jax.random.key(0), model_s)
print(type(info).__name__, info._fields)
print(info.potential_fn({"s": jnp.array(0.0)}))
print(info.postprocess_fn({"s": jnp.array(0.0)}))
```
実行結果:
```
6.781202793121338
7.781202793121338
{'s': Array(1., dtype=float32, weak_type=True)}
{'s': Array(0.6931472, dtype=float32, weak_type=True)}
ModelInfo ('param_info', 'potential_fn', 'postprocess_fn', 'model_trace')
2.7336683
{'s': Array(1., dtype=float32, weak_type=True)}
```

**注意点・落とし穴**:
- `potential_energy`は「−対数密度」にヤコビアン補正(`s=exp(u)`なら`u`)を加えた値。上の1行目(6.78)は、2行目(`-log_density`=7.78)から`u=1`を引いた値になっている。
- `unconstrain_fn`は`s=2`に対し`log(2)=0.693...`を返す(`positive`制約なので)。MCMCの初期値やHMCの勾配計算を自作するときの基礎になる。

---

## 8. エフェクトハンドラ

`numpyro.handlers`のハンドラは、モデル関数の`sample`/`param`サイトの挙動を書き換える(乱数の供給、値の置換、記録など)。`MCMC`/`SVI`/`Predictive`も内部でこれらを使っている。この章の例は次の共通コードを前提とする。

**共通コード**:
```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.util import log_density

def model():
    a = numpyro.sample("a", dist.Normal(0, 1))
    b = numpyro.sample("b", dist.Normal(a, 1))
    numpyro.sample("y", dist.Normal(a + b, 0.1))
    return a + b
```

### `handlers.seed(...)`

**用途**: 乱数キーを供給する。`numpyro.sample`を含む関数を、推論エンジン抜きで直接実行できるようにする。

**シグネチャ**: `numpyro.handlers.seed(fn=None, rng_seed=None, hide_types=None)`

**使用例**:
```python
print(handlers.seed(model, rng_seed=0)())
print(handlers.seed(model, rng_seed=jax.random.key(0))())
print(handlers.seed(model, rng_seed=1)())
with handlers.seed(rng_seed=0):
    r1 = (model(), model())
print(r1)
with handlers.seed(rng_seed=0):
    r2 = (model(), model())
print(all(float(a) == float(b) for a, b in zip(r1, r2)))
```
実行結果:
```
-6.1423893
-6.1423893
0.45991147
(Array(-6.1423893, dtype=float32), Array(-5.6193132, dtype=float32))
True
```

**注意点・落とし穴**:
- `rng_seed`は整数でも`jax.random.key`でも渡せ、同じシードなら同じ結果になる(1行目と2行目が一致)。
- `with`文で使うと、ブロック内で複数回呼び出しても毎回異なる値が得られる(4行目)。同じシードで別の`with`ブロックに入ると、同じ列の値が再現される(最終行)。

### `handlers.trace(...)`

**用途**: モデルの実行を記録し、各サイトの情報(分布、値、観測の有無など)を辞書として取り出す。

**シグネチャ**: `numpyro.handlers.trace(fn=None)`

**使用例**:
```python
tr = handlers.trace(handlers.seed(model, 0)).get_trace()
print(type(tr).__name__, list(tr.keys()))
print(sorted(tr["a"].keys()))
print(tr["a"]["type"], tr["a"]["value"])
print(tr["a"]["fn"].batch_shape, tr["y"]["is_observed"])
lp = {k: round(float(v["fn"].log_prob(v["value"])), 4) for k, v in tr.items()}
print(lp)
```
実行結果:
```
OrderedDict ['a', 'b', 'y']
['args', 'cond_indep_stack', 'fn', 'infer', 'intermediates', 'is_observed', 'kwargs', 'name', 'scale', 'type', 'value']
sample -2.4424558
() False
{'a': -3.9017, 'b': -1.7096, 'y': 0.4207}
```

**注意点・落とし穴**:
- `get_trace()`は`trace`ハンドラのメソッドなので、`trace`を最も外側に置く(`handlers.trace(handlers.seed(model, 0)).get_trace()`)。逆順の`handlers.seed(handlers.trace(model), 0).get_trace()`は`AttributeError: 'seed' object has no attribute 'get_trace'`になる(実測)。
- 各サイトの辞書は`type`(`sample`/`param`/`deterministic`/`plate`)・`name`・`fn`(分布)・`value`・`is_observed`・`cond_indep_stack`などを持つ。`plate`は`type='plate'`のサイトとして現れる。

### `handlers.substitute(...)`

**用途**: 指定したサイトの値を固定値に置き換える。サイトは観測扱い(`is_observed=True`)にはならず、潜在変数のまま値だけ差し替わる。

**シグネチャ**: `numpyro.handlers.substitute(fn=None, data=None, substitute_fn=None)`

**使用例**:
```python
sub = handlers.substitute(handlers.seed(model, 0), data={"a": 1.0, "b": 2.0})
print(sub())
tr = handlers.trace(sub).get_trace()
print(tr["a"]["value"], tr["a"]["is_observed"], tr["y"]["is_observed"])
```
実行結果:
```
3.0
1.0 False False
```

**注意点・落とし穴**:
- `substitute`は「パラメータをこの値にしたときのモデルの挙動」を見る用途に向く。観測扱いにしたい(尤度に加えたい)場合は`condition`を使う。

### `handlers.condition(...)`

**用途**: 指定したサイトを観測値として条件付ける(`is_observed=True`にする)。同じモデルを、データありなし両方で使い回せる。

**シグネチャ**: `numpyro.handlers.condition(fn=None, data=None, condition_fn=None)`

**使用例**:
```python
cm = handlers.condition(handlers.seed(model, 0), data={"a": 1.0})
tr = handlers.trace(cm).get_trace()
print(tr["a"]["is_observed"], tr["a"]["value"], tr["b"]["value"])

tr2 = handlers.trace(handlers.seed(handlers.condition(model, {"y": 1.0}), 0)).get_trace()
print(tr2["y"]["is_observed"], tr2["y"]["value"])
```
実行結果:
```
True 1.0 -0.25747764
True 1.0
```

**注意点・落とし穴**:
- `condition`で観測扱いにしたサイトは、`substitute`と異なり推論(MCMC/SVI)では潜在変数として推定されない。`obs=`をモデルに直接書く代わりに、外側から後付けでデータを注入できる。

### `handlers.do(...)`(介入)

**用途**: 因果推論の`do`演算(介入)。指定サイトの値を固定値に置き換え、その固定値を下流のサイトにだけ伝える。

**シグネチャ**: `numpyro.handlers.do(fn=None, data=None)`

**使用例**:
```python
dm = handlers.do(handlers.seed(model, 0), data={"a": 10.0})
tr = handlers.trace(dm).get_trace()
print({k: round(float(v["value"]), 3) for k, v in tr.items()})
tr_plain = handlers.trace(handlers.seed(model, 0)).get_trace()
print({k: round(float(v["value"]), 3) for k, v in tr_plain.items()})
```
実行結果:
```
{'a': -2.442, 'b': 8.743, 'y': 18.604}
{'a': -2.442, 'b': -3.7, 'y': -6.281}
```

**注意点・落とし穴**:
- `do`で`a=10`に介入すると、下流の`b`と`y`は`a=10`を前提に計算される(介入なしの2行目に比べ、`b`が-3.7→8.7、`y`が-6.281→18.604に変わる)。一方トレース上の`a`は元のサンプル値(`-2.442`)のままで、介入値は下流にのみ伝わる。

### `handlers.block(...)`

**用途**: 指定したサイトを、外側のハンドラ(`trace`など)から見えなくする。

**シグネチャ**: `numpyro.handlers.block(fn=None, hide_fn=None, hide=None, expose_types=None, expose=None)`

**使用例**:
```python
bm = handlers.block(handlers.seed(model, 0), hide=["a"])
print(list(handlers.trace(bm).get_trace().keys()))
bm2 = handlers.block(handlers.seed(model, 0), expose=["a"])
print(list(handlers.trace(bm2).get_trace().keys()))
```
実行結果:
```
['b', 'y']
['a']
```

**注意点・落とし穴**:
- `hide`は隠すサイト名のリスト、`expose`は逆に「見せるサイト名」のリスト(それ以外は隠れる)。ガイドとモデルで同名サイトを分離する場合などに使う。

### `handlers.replay(...)`

**用途**: 別の実行で記録したトレースの値を再利用し、同じサイトの値を再現する。SVIで、ガイドからサンプルした潜在変数をモデルに渡すのに使われる。

**シグネチャ**: `numpyro.handlers.replay(fn=None, trace=None)`

**使用例**:
```python
tr_a = handlers.trace(handlers.seed(model, 0)).get_trace()
replayed = handlers.replay(handlers.seed(model, 5), trace=tr_a)   # 別のシードでもトレースの値が優先される
tr_b = handlers.trace(replayed).get_trace()
print(tr_b["a"]["value"] == tr_a["a"]["value"], tr_b["b"]["value"] == tr_a["b"]["value"])
```
実行結果:
```
True True
```

**注意点・落とし穴**:
- `replay`されるのは`trace`にあるサンプルサイトの値。`seed`が異なっていても、`trace`に存在するサイトの値が優先される。

### `handlers.scale(...)`

**用途**: 内包するサイトの対数確率を定数倍する(重み付き尤度、ミニバッチの補正など)。

**シグネチャ**: `numpyro.handlers.scale(fn=None, scale=1.0)`

**使用例**:
```python
def m_scale():
    with numpyro.plate("d", 3):
        numpyro.sample("x", dist.Normal(0, 1), obs=jnp.zeros(3))
print(log_density(m_scale, (), {}, {})[0])
print(log_density(handlers.scale(m_scale, scale=10.0), (), {}, {})[0])
```
実行結果:
```
-2.7568154
-27.568157
```

**注意点・落とし穴**:
- `plate(subsample_size=...)`が内部で`scale`を適用しているのと同じ仕組み(1章参照)。

### `handlers.mask(...)`

**用途**: 内包するサイトの対数確率に真偽マスクをかけ、`False`の位置(またはブロック全体)の寄与を0にする。

**シグネチャ**: `numpyro.handlers.mask(fn=None, mask=True)`

**使用例**:
```python
def m_mask():
    with numpyro.plate("d", 3):
        numpyro.sample("x", dist.Normal(0, 1), obs=jnp.array([0.0, 1.0, 2.0]))
print(log_density(m_mask, (), {}, {})[0])
print(log_density(handlers.mask(m_mask, mask=jnp.array([True, False, True])), (), {}, {})[0])
print(log_density(handlers.mask(m_mask, mask=False), (), {}, {})[0])
```
実行結果:
```
-5.256816
-3.8378773
0.0
```

**注意点・落とし穴**:
- `mask=False`とするとモデル全体の対数確率が0になる(上の最終行)。`Distribution.mask`(2章)は分布単位、こちらは関数(ブロック)単位で適用するハンドラ。

### `handlers.scope(...)`

**用途**: 内包するサイトの名前に接頭辞を付ける。同じサブモデルを複数回呼び出しても名前が衝突しない。

**シグネチャ**: `numpyro.handlers.scope(fn=None, prefix='', divider='/', *, hide_types=None)`

**使用例**:
```python
def sub_model():
    return numpyro.sample("w", dist.Normal(0, 1))

def outer():
    with handlers.scope(prefix="g1"):
        sub_model()
    with handlers.scope(prefix="g2"):
        sub_model()

print(list(handlers.trace(handlers.seed(outer, 0)).get_trace().keys()))
print(list(handlers.trace(handlers.seed(handlers.scope(sub_model, prefix="p", divider="."), 0)).get_trace().keys()))
```
実行結果:
```
['g1/w', 'g2/w']
['p.w']
```

**注意点・落とし穴**:
- 区切り文字の既定は`/`(`g1/w`)。`divider=`で変更できる。

### `handlers.lift(...)` / `handlers.uncondition(...)`

**用途**: `lift`は`numpyro.param`サイトを事前分布付きの確率変数(`sample`)に変換する。`uncondition`は観測を外して観測サイトを通常の確率変数に戻す。

**シグネチャ**: `numpyro.handlers.lift(fn=None, prior=None)` / `numpyro.handlers.uncondition(fn=None)`

**使用例**:
```python
def pm():
    s = numpyro.param("s", 1.0)
    numpyro.sample("x", dist.Normal(0, s), obs=0.5)

tr = handlers.trace(handlers.seed(pm, 0)).get_trace()
print({k: v["type"] for k, v in tr.items()})
lifted = handlers.lift(pm, prior={"s": dist.HalfNormal(1.0)})
tr = handlers.trace(handlers.seed(lifted, 0)).get_trace()
print({k: v["type"] for k, v in tr.items()})

uc = handlers.uncondition(handlers.seed(handlers.condition(model, {"y": 5.0}), 0))
print(handlers.trace(uc).get_trace()["y"]["is_observed"])
```
実行結果:
```
{'s': 'param', 'x': 'sample'}
{'s': 'sample', 'x': 'sample'}
False
```

**注意点・落とし穴**:
- `lift`は`prior`辞書で指定した名前の`param`サイトを`sample`サイトに変える(上の2つ目の出力)。`uncondition`は`condition`などで付いた観測を外し、サイトを通常の確率変数に戻す(上の最終行が`False`)。

---

## 9. ArviZ連携

`arviz.from_numpyro`でMCMC結果を`InferenceData`(ArviZ 1.3.0では`xarray.DataTree`)に変換し、`az.summary`や`az.loo`など可視化・診断・モデル比較を行う。ArviZ 1.3.0で検証した。この章の例は3章と同じ単回帰データ・モデルを使い、2チェーンを事前に実行している。

**共通コード**:
```python
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

rng = np.random.default_rng(0)
x = rng.normal(size=50)
y = 1.0 + 2.0 * x + rng.normal(scale=0.5, size=50)   # 真の値: a=1, b=2, sigma=0.5

def model(x, y=None):
    a = numpyro.sample("a", dist.Normal(0, 5))
    b = numpyro.sample("b", dist.Normal(0, 5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2))
    numpyro.sample("y", dist.Normal(a + b * x, sigma), obs=y)

import warnings
import arviz as az
from numpyro.infer import MCMC, NUTS, Predictive

with warnings.catch_warnings():
    warnings.simplefilter("ignore")            # 1デバイス環境での逐次実行の警告を抑制
    mcmc = MCMC(NUTS(model), num_warmup=200, num_samples=200, num_chains=2,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.key(0), x, y)
```

### `az.from_numpyro(...)`

**用途**: `MCMC`オブジェクトを`InferenceData`(`DataTree`)に変換する。

**シグネチャ**: `arviz.from_numpyro(posterior=None, *, prior=None, posterior_predictive=None, predictions=None, constant_data=None, predictions_constant_data=None, log_likelihood=False, index_origin=None, coords=None, dims=None, pred_dims=None, extra_event_dims=None, sample_dims=None, num_chains=None)`

**使用例**:
```python
idata = az.from_numpyro(mcmc)
print(type(idata).__name__)
print(list(idata.children.keys()))
print(dict(idata.posterior.sizes))
print(list(idata.posterior.data_vars))
print(az.summary(idata, kind="stats", round_to=2))
```
実行結果:
```
DataTree
['posterior', 'sample_stats', 'observed_data']
{'chain': 2, 'draw': 200}
['a', 'b', 'sigma']
       mean    sd  eti89_lb  eti89_ub
a      1.03  0.07      0.91      1.14
b      1.94  0.08      1.81      2.06
sigma  0.52  0.05      0.44      0.61
```

**注意点・落とし穴**:
- デフォルトでは`posterior`・`sample_stats`(`diverging`)・`observed_data`(モデルの`obs`から自動抽出)のグループができる。`prior`・`posterior_predictive`・`log_likelihood`は引数で渡さない限り含まれない。
- `posterior`の次元は`(chain, draw)`(上の出力)。SVIの結果からの変換は`az.from_numpyro_svi`(後述)。
- ArviZ 1.3.0の`az.summary`は既定の信用区間が`eti89`(89%等尾区間)で、列名は`eti89_lb`/`eti89_ub`になる(PyMC辞書7章と同じ挙動)。NumPyroの`print_summary`のHPDIとは区間の種類が異なる点に注意。

### `az.from_numpyro(..., prior=, posterior_predictive=, log_likelihood=, coords=, dims=)`

**用途**: 事前予測・事後予測・対数尤度・次元ラベルも含めて`InferenceData`にまとめる。

**使用例**:
```python
pp = Predictive(model, mcmc.get_samples())(jax.random.key(1), x)
prior = Predictive(model, num_samples=100)(jax.random.key(2), x)

idata = az.from_numpyro(mcmc, prior=prior, posterior_predictive=pp, log_likelihood=True,
                        coords={"obs_id": np.arange(50)}, dims={"y": ["obs_id"]})
print(list(idata.children.keys()))
print(dict(idata.posterior_predictive.sizes))
print(dict(idata.log_likelihood.sizes))
print(dict(idata.prior.sizes))
print(dict(idata.observed_data.sizes))
```
実行結果:
```
['posterior', 'sample_stats', 'log_likelihood', 'posterior_predictive', 'prior', 'prior_predictive', 'observed_data']
{'chain': 2, 'draw': 200, 'obs_id': 50}
{'chain': 2, 'draw': 200, 'obs_id': 50}
{'chain': 1, 'draw': 100}
{'obs_id': 50}
```

**注意点・落とし穴**:
- `Predictive`の結果(`{サイト名: 配列}`)をそのまま`prior=`/`posterior_predictive=`に渡せる。`log_likelihood=True`にすると内部で対数尤度を計算して`log_likelihood`グループに格納する。
- `dims={"y": ["obs_id"]}`と`coords={"obs_id": ...}`で次元に名前を付けられる(上の出力の`obs_id`)。名前を付けないと`y_dim_0`のような自動名になる。
- 上の`prior`グループは`chain=1, draw=100`で、`posterior`(2チェーン)とは形が異なる。`prior`に`y`を含む`Predictive`の結果を渡したので、`prior_predictive`グループも作られている(上の出力の`children`)。

### `az.loo(...)`(モデル比較)

**用途**: `log_likelihood`グループを持つ`InferenceData`から、PSIS-LOOによる予測精度(`elpd_loo`)を計算する。

**使用例**:
```python
idata = az.from_numpyro(mcmc, log_likelihood=True)
print(az.loo(idata))
```
実行結果:
```
Computed from 400 posterior samples and 50 observations log-likelihood matrix.

         Estimate       SE
elpd_loo   -39.25     3.91
p_loo        2.53        -
------

Pareto k diagnostic values:
                         Count   Pct.
(-Inf, 0.62]   (good)       50  100.0%
   (0.62, 1]   (bad)         0    0.0%
    (1, Inf)   (very bad)    0    0.0%
```

**注意点・落とし穴**:
- `az.loo`や`az.compare`を使うには、`az.from_numpyro`に`log_likelihood=True`を渡して`log_likelihood`グループを作っておく必要がある。ArviZ 1.3.0に`az.waic`はない(PyMC辞書7章参照)。

### `az.from_numpyro_svi(...)`(SVIの結果から)

**用途**: SVIの学習結果から近似事後サンプルを生成し、`InferenceData`にまとめる。

**シグネチャ**: `arviz.from_numpyro_svi(svi=None, *, svi_result=None, model_args=None, model_kwargs=None, prior=None, posterior_predictive=None, predictions=None, constant_data=None, predictions_constant_data=None, log_likelihood=False, index_origin=None, coords=None, dims=None, pred_dims=None, extra_event_dims=None, num_samples=1000)`

**使用例**:
```python
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

svi = SVI(model, AutoNormal(model), optim.Adam(0.1), Trace_ELBO())
res = svi.run(jax.random.key(0), 1000, x, y, progress_bar=False)
idata = az.from_numpyro_svi(svi, svi_result=res, model_args=(x, y), num_samples=300)
print(list(idata.children.keys()))
print(dict(idata.posterior.sizes))
```
実行結果:
```
['posterior', 'sample_stats', 'observed_data']
{'sample': 300}
```

**注意点・落とし穴**:
- SVI由来の`posterior`は`(chain, draw)`ではなく単一の`sample`次元(300)を持つ(上の出力)。`model_args`には学習時と同じ引数`(x, y)`を渡す。

---

## 10. その他ユーティリティ

グローバル設定(精度、検証)やモデルの構造確認に使うAPI。

**共通コード**:
```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
```

### `numpyro.enable_x64(...)`

**用途**: JAXを64ビット浮動小数点モードにする。数値精度が問題になるモデル(強い相関、極端なスケール)で使う。

**シグネチャ**: `numpyro.enable_x64(use_x64=True)`

**使用例**:
```python
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist

print(jnp.ones(1).dtype)
numpyro.enable_x64()
print(jnp.ones(1).dtype, dist.Normal(0.0, 1.0).sample(jax.random.key(0)).dtype)
numpyro.enable_x64(False)
print(jnp.ones(1).dtype)
```
実行結果:
```
float32
float64 float64
float32
```

**注意点・落とし穴**:
- `enable_x64()`はグローバル設定で、以降に生成される配列の既定dtypeが`float32`から`float64`になる(乱数の`sample`も`float64`)。`enable_x64(False)`で戻せる(上の出力)。

### `numpyro.enable_validation(...)` / `numpyro.validation_enabled()`

**用途**: 分布のパラメータ検証(範囲外の引数でのエラー送出)を有効にする。既定は無効。

**シグネチャ**: `numpyro.enable_validation(is_validate=True)` / `numpyro.validation_enabled(is_validate=True)`

**使用例**:
```python
print(dist.Normal(0.0, -1.0).log_prob(0.0))        # 既定は検証なし: nanになる
with numpyro.validation_enabled():
    try:
        dist.Normal(0.0, -1.0)
    except ValueError as e:
        print("ValueError:", e)
numpyro.enable_validation(True)                     # グローバルに有効化
try:
    dist.Normal(0.0, -1.0)
except ValueError as e:
    print("ValueError:", e)
numpyro.enable_validation(False)
```
実行結果:
```
nan
ValueError: Normal distribution got invalid scale parameter.
ValueError: Normal distribution got invalid scale parameter.
```

**注意点・落とし穴**:
- `validation_enabled()`はコンテキストマネージャで、`with`ブロック内だけ有効になる(真偽値を返す関数ではない)。個別の分布には`validate_args=True`引数でも有効化できる。
- 検証を有効にすると、範囲外のパラメータは分布の作成時に`ValueError`になる。既定の無効状態では`nan`のまま計算が進む(上の1行目)ので、モデルが`nan`を出して原因が分からないときのデバッグに有効化すると原因を特定しやすい。

### `numpyro.render_model(...)`

**用途**: モデルの確率的な依存関係をグラフ(graphviz)として描画する。

**シグネチャ**: `numpyro.render_model(model, model_args=None, model_kwargs=None, filename=None, render_distributions=False, render_params=False)`

**使用例**:
```python
def m2(x):
    a = numpyro.sample("a", dist.Normal(0, 1))
    with numpyro.plate("N", 3):
        numpyro.sample("y", dist.Normal(a, 1), obs=x)

g = numpyro.render_model(m2, model_args=(jnp.zeros(3),), render_distributions=True)
print(type(g))
print(g.source)
```
実行結果:
```
<class 'graphviz.graphs.Digraph'>
digraph {
	a [label=a fillcolor=white shape=ellipse style=filled]
	subgraph cluster_N {
		label=N labeljust=r labelloc=b
		y [label=y fillcolor=grey shape=ellipse style=filled]
	}
	a -> y
	distribution_description_node [label="a ~ Normal\ly ~ Normal\l" shape=plaintext]
}
```

**注意点・落とし穴**:
- 戻り値は`graphviz.Digraph`(上の出力)で、`.source`でDOT言語のテキストを取り出せる。`filename=`引数でファイル出力できる(シグネチャに存在。今回は実行していない)。観測サイトは灰色(`fillcolor=grey`)、`plate`は`cluster`のサブグラフで表現される。

### `numpyro.infer.inspect.get_model_relations(...)`

**用途**: モデルのサイト間の依存関係(どのサンプルサイトが他のどれに依存するか、`plate`との対応、観測サイト)を辞書で取得する。

**シグネチャ**: `numpyro.infer.inspect.get_model_relations(model, model_args=None, model_kwargs=None)`

**使用例**:
```python
from numpyro.infer.inspect import get_model_relations

rel = get_model_relations(m2, model_args=(jnp.zeros(3),))
for k, v in rel.items():
    print(k, v)
```
実行結果:
```
sample_sample {'a': [], 'y': ['a']}
sample_param {'a': [], 'y': []}
sample_dist {'a': 'Normal', 'y': 'Normal'}
param_constraint {}
plate_sample {'N': ['y']}
observed ['y']
```

**注意点・落とし穴**:
- `sample_sample`は「各サイト → それが依存する親サイト」の関係(`y`は`a`に依存)、`plate_sample`は`plate`ごとの所属サイト、`observed`は観測サイトの一覧。`render_model`が内部で使う情報。
