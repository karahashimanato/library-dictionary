# PyMC 逆引き辞書

PyMC 6.3.1 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(PyMC 6.3.1、ArviZ 1.3.0)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。MCMCの事後分布の要約値(平均・標準偏差など)は`random_seed`を固定していても、実行環境(OS・BLASライブラリ等)によって末尾の桁がわずかに変動しうる点に留意すること。

## 目次

1. [モデル定義の基礎](#1-モデル定義の基礎)
2. [分布(連続・離散)](#2-分布連続離散)
3. [決定論的変数・データ・制約](#3-決定論的変数データ制約)
4. [事前予測・事後予測チェック](#4-事前予測事後予測チェック)
5. [MCMCサンプリング](#5-mcmcサンプリング)
6. [変分推論(ADVI)](#6-変分推論advi)
7. [診断・モデル比較(ArviZ)](#7-診断モデル比較arviz)
8. [階層モデル・座標系(coords/dims)](#8-階層モデル座標系coordsdims)
9. [その他ユーティリティ](#9-その他ユーティリティ)

---

## 1. モデル定義の基礎

### `Model(...)`

**用途**: PyMCモデルの入れ物。`with`文のコンテキストマネージャとして使い、内部で定義した確率変数を自動的にモデルに登録する。

**シグネチャ**: `pymc.Model(name='', coords=None, check_bounds=True, *, model=UNSET)`

**使用例**:
```python
import pymc as pm

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=[1.0, 2.0, 1.5])

print(model.free_RVs)
print(model.observed_RVs)
```
実行結果:
```
[mu, sigma]
[obs]
```

**注意点・落とし穴**:
- `with pm.Model() as model:` ブロック内で `pm.Normal(...)` のように呼ぶと、戻り値の確率変数が自動的に`model`に登録される。ブロック外で`pm.Normal.dist(...)`を使うとモデルに登録されない単なる分布オブジェクトになる(2章参照)。
- `coords`引数に`{"group": [...]}`のような辞書を渡すと、8章の階層モデルのように次元にラベルを付けられる。
- `name`引数はモデルを入れ子にする場合のプレフィックスに使う(通常の単一モデルでは省略でよい)。

### モデルの構造を確認する(`model.str_repr()` / `basic_RVs` など)

**用途**: 定義したモデルの確率変数の構成(名前・分布・親子関係)を確認する。

**シグネチャ**: `Model.str_repr(self, formatting='plain', var_names=None, include_params=True)`(属性として `basic_RVs`, `free_RVs`, `observed_RVs`, `deterministics` も参照可能)

**使用例**:
```python
print(model.str_repr())
print("basic_RVs:", model.basic_RVs)
```
実行結果:
```
   mu ~ Normal(0, 1)
sigma ~ HalfNormal(0, 1)
  obs ~ Normal(mu, sigma)
basic_RVs: [mu, sigma, obs]
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: PyMC 6.3.1では単に`print(model)`(`str(model)`/`repr(model)`)としても`<pymc.model.core.Model object at 0x...>`としか表示されない。数式付きの読みやすい表示を得るには明示的に`model.str_repr()`を呼ぶ必要がある。
- `basic_RVs`は`free_RVs + observed_RVs`(決定論的変数`Deterministic`は含まない、別途`model.deterministics`で取得する)。

---

## 2. 分布(連続・離散)

### `Normal(...)`

**用途**: 正規分布。連続値のモデリングで最も基本的な分布。

**シグネチャ**: `pymc.Normal(name, *args, dims=None, initval=None, observed=None, **kwargs)`(内部の分布パラメータは`Normal.dist`と共通: `mu=0, sigma=None, tau=None`)

**使用例**:
```python
import pymc as pm
print(pm.draw(pm.Normal.dist(mu=0, sigma=1), draws=5, random_seed=0))
```
実行結果:
```
[ 1.44369095 -0.89594598  0.73595567  0.00587704  0.85338179]
```

**注意点・落とし穴**:
- `sigma`(標準偏差)と`tau`(精度=1/分散)のどちらかで散らばりを指定する。両方同時には指定できない。
- モデル内で使うときは`pm.Normal("mu", mu=0, sigma=1)`のように文字列の`name`が最初の引数として必須(`.dist()`にはない)。

### `Uniform(...)`

**用途**: 一様分布。範囲だけ分かっていて分布の形に情報がない場合の事前分布などに使う。

**シグネチャ**: `pymc.Uniform.dist(lower=0, upper=1, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.Uniform.dist(lower=0, upper=10), draws=5, random_seed=0))
```
実行結果:
```
[9.42937553 3.16337152 7.22342589 1.25603085 4.22976363]
```

**注意点・落とし穴**:
- 範囲外の値の確率密度は0になるため、事前分布として使うとその範囲外をNUTSが探索できず、事後分布が範囲の端に張り付く(打ち切られる)ことがある。範囲に自信がない場合は`Normal`や`HalfNormal`など裾の広い分布の方が安全なことが多い。

### `HalfNormal(...)`

**用途**: 半正規分布(0以上のみ)。標準偏差やスケールパラメータなど非負の量の事前分布によく使う。

**シグネチャ**: `pymc.HalfNormal.dist(sigma=None, tau=None, *args, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.HalfNormal.dist(sigma=2), draws=5, random_seed=0))
```
実行結果:
```
[2.88738191 1.79189195 1.47191134 0.01175408 1.70676358]
```

**注意点・落とし穴**:
- `sigma`は「元になる正規分布の標準偏差」であり、`HalfNormal`自体の標準偏差とは一致しない(半正規分布の標準偏差は`sigma * sqrt(1 - 2/pi)`)。

### `StudentT(...)`

**用途**: 自由度`nu`を持つt分布。正規分布よりも裾が重く、外れ値に頑健なモデリングに使う。

**シグネチャ**: `pymc.StudentT.dist(nu, mu=0, *, sigma=None, lam=None, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.StudentT.dist(nu=3, mu=0, sigma=1), draws=5, random_seed=0))
```
実行結果:
```
[ 2.66001511  0.00469289  0.66584316  1.62193575 -1.20673774]
```

**注意点・落とし穴**:
- `nu`(自由度)は必須の第一引数。`nu`が小さいほど裾が重く(外れ値を許容しやすく)なり、`nu`が大きくなるほど正規分布に近づく。
- 通常の回帰の誤差分布を`Normal`から`StudentT`に変えるだけで、外れ値に頑健なロバスト回帰になる(statsmodelsの`RLM`に近い効果)。

### `Beta(...)`

**用途**: ベータ分布。0〜1の確率・比率のモデリング(例: コンバージョン率の事前分布)に使う。

**シグネチャ**: `pymc.Beta.dist(alpha=None, beta=None, mu=None, sigma=None, nu=None, *args, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.Beta.dist(alpha=2, beta=5), draws=5, random_seed=0))
print(pm.draw(pm.Beta.dist(mu=0.3, sigma=0.1), draws=5, random_seed=0))
```
実行結果:
```
[0.40087077 0.31230465 0.3769816  0.10010578 0.2651989 ]
[0.37267552 0.31961526 0.3581455  0.1775717  0.29120364]
```

**注意点・落とし穴**:
- 従来の`alpha`/`beta`(形状パラメータ)だけでなく、`mu`(平均)/`sigma`(標準偏差)でも指定できる(内部で`alpha`/`beta`に変換される)。直感的に「平均◯%くらい」という事前知識がある場合は`mu`/`sigma`指定の方が分かりやすい。
- `mu`/`sigma`を使う場合、`sigma`は`mu*(1-mu)`(理論上の最大分散)より小さい必要があり、大きすぎるとエラーになる。

### `Gamma(...)`

**用途**: ガンマ分布。0以上の連続量(待ち時間、分散パラメータなど)のモデリングに使う。

**シグネチャ**: `pymc.Gamma.dist(alpha=None, beta=None, mu=None, sigma=None, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.Gamma.dist(alpha=2, beta=1), draws=5, random_seed=0))
print(pm.draw(pm.Gamma.dist(mu=5, sigma=2), draws=5, random_seed=0))
```
実行結果:
```
[4.31153613 2.80876074 3.02896093 2.96393468 1.96361884]
[8.13511492 6.31474648 6.5957329  6.51337387 5.16945373]
```

**注意点・落とし穴**:
- `Beta`と同様、形状-レート(`alpha`/`beta`)とモーメント(`mu`/`sigma`)の2通りの指定方法がある。`beta`は「レート」であり「スケール」(`1/beta`)ではない点に注意。

### `Exponential(...)`

**用途**: 指数分布。待ち時間・寿命など単純な非負の連続量のモデリングに使う(ガンマ分布の特殊ケース)。

**シグネチャ**: `pymc.Exponential.dist(lam=None, *, scale=None, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.Exponential.dist(lam=1), draws=5, random_seed=0))
print(pm.draw(pm.Exponential.dist(scale=2), draws=5, random_seed=0))
```
実行結果:
```
[3.29352779 0.76313074 1.25236712 0.2259495  0.22074504]
[6.58705558 1.52626148 2.50473425 0.45189899 0.44149008]
```

**注意点・落とし穴**:
- `lam`(レート、平均の逆数)と`scale`(=`1/lam`、平均そのもの)のどちらでも指定できる。両者を混同すると平均が意図と逆数になるバグになりやすい。

### `Bernoulli(...)`

**用途**: ベルヌーイ分布。二値(0/1)データのモデリング(ロジスティック回帰の観測モデルなど)。

**シグネチャ**: `pymc.Bernoulli.dist(p=None, logit_p=None, *args, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.Bernoulli.dist(p=0.3), draws=10, random_seed=0))
```
実行結果:
```
[0 0 0 1 0 0 1 0 1 0]
```

**注意点・落とし穴**:
- `p`(確率、0〜1)の代わりに`logit_p`(ロジット、実数全体)を指定できる。ロジスティック回帰でモデル内部の線形結合`a + b*x`をそのまま渡したい場合は`logit_p=a + b*x`の方が、`p=pm.math.invlogit(a + b*x)`より数値的に安定する。
- 離散分布のため`pm.sample()`はデフォルトのNUTSではなく、内部で(観測されていない離散変数がある場合)自動的に`Metropolis`など別のステップ手法に切り替わることがある。

### `Binomial(...)`

**用途**: 二項分布。`n`回の試行のうち成功した回数のモデリング。

**シグネチャ**: `pymc.Binomial.dist(n, p=None, logit_p=None, *args, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.Binomial.dist(n=10, p=0.5), draws=10, random_seed=0))
```
実行結果:
```
[7 4 6 3 5 6 3 6 4 6]
```

**注意点・落とし穴**:
- `n`(試行回数)が第一引数で必須。`n=1`の`Binomial`は`Bernoulli`と等価になる。

### `Poisson(...)`

**用途**: ポアソン分布。単位時間・単位面積あたりの発生回数(カウントデータ)のモデリング。

**シグネチャ**: `pymc.Poisson.dist(mu, *args, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.Poisson.dist(mu=4), draws=10, random_seed=0))
```
実行結果:
```
[4 3 5 2 4 5 5 1 6 2]
```

**注意点・落とし穴**:
- `mu`は平均であり分散でもある(ポアソン分布は平均=分散という制約がある)。実データが平均より分散が大きい「過分散」の場合は`NegativeBinomial`を検討する。

### `Categorical(...)`

**用途**: カテゴリカル分布。3クラス以上の多クラス分類の観測モデル。

**シグネチャ**: `pymc.Categorical.dist(p=None, logit_p=None, **kwargs)`

**使用例**:
```python
print(pm.draw(pm.Categorical.dist(p=[0.1, 0.2, 0.7]), draws=10, random_seed=0))
```
実行結果:
```
[2 2 2 1 2 2 0 2 1 2]
```

**注意点・落とし穴**:
- `p`は各カテゴリの確率のベクトル(合計が1になる必要がある)。`Bernoulli`/`Binomial`と同様、`logit_p`(多クラスロジット、softmax前の値)でも指定できる。

---

## 3. 決定論的変数・データ・制約

### `Deterministic(...)`

**用途**: サンプリング対象ではないが、事後分布と一緒に保存・追跡したい変数変換(例: 変換後のスケール、合成指標)を登録する。

**シグネチャ**: `pymc.Deterministic(name, var, model=None, dims=None)`

**使用例**:
```python
with pm.Model() as m:
    mu = pm.Normal("mu", mu=0, sigma=10)
    mu_sq = pm.Deterministic("mu_sq", mu ** 2)
```
実行結果(モデルの`deterministics`属性):
```
[mu_sq]
```

**注意点・落とし穴**:
- `Deterministic`を通さずに`mu ** 2`をただの中間変数として使っても計算自体は動くが、`pm.sample()`後の`idata.posterior`に保存されない。事後分布の要約(`az.summary`など)を見たい変換は必ず`Deterministic`で包む必要がある。
- サンプリング速度への影響はほぼない(計算グラフに変換を追加して結果を保存するだけ)。

### `Data(...)`

**用途**: モデルに入力する観測データ・共変量を、後から`pm.set_data()`で差し替え可能な形でモデルに登録する。

**シグネチャ**: `pymc.Data(name, value, *, dims=None, coords=None, infer_dims_and_coords=False, model=None, **kwargs)`

**使用例**:
```python
import numpy as np
with pm.Model() as m:
    x_data = pm.Data("x_data", np.array([1.0, 2.0, 3.0]))
    mu = pm.Normal("mu", 0, 10)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=x_data)
print(type(x_data))
```
実行結果:
```
<class 'pytensor.tensor.variable.TensorConstant'>
```

**注意点・落とし穴**:
- 単に`observed=[1.0, 2.0, 3.0]`と生の配列を渡してもモデルは動くが、`pm.Data`で包んでおくと後から`pm.set_data()`で値を差し替えて`sample_posterior_predictive(..., predictions=True)`による新規データへの予測(what-ifシミュレーション)ができるようになる。
- 事後予測チェック(`idata.constant_data`/`observed_data`グループ)にも`pm.Data`の値が自動的に記録される。

### `set_data(...)`

**用途**: `pm.Data`で登録済みの変数の値を、モデルを再構築せずに差し替える。

**シグネチャ**: `pymc.set_data(new_data, model=None, *, coords=None)`

**使用例**:
```python
with m:
    pm.set_data({"group_idx": np.array([0, 1, 2, 3])}, coords={"obs_id": np.arange(4)})
    preds = pm.sample_posterior_predictive(
        idata_h, var_names=["obs"], predictions=True, random_seed=0, progressbar=False
    )
print(preds.predictions["obs"].shape)
```
実行結果:
```
(2, 500, 4)
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: 新しいデータで対応する次元(`dims`)の長さが変わる場合、`coords`引数で新しい座標ラベルも同時に渡さないと`ValueError`(「次元の長さが変わったので新しい座標値が必要」)になる。単に値だけ差し替えて次元長も同じ場合は`coords`は不要。
- `predictions=True`を指定すると結果は`idata.posterior_predictive`ではなく`idata.predictions`グループに入る(元の観測データに対する事後予測と、新規データに対する予測を区別するため)。

### `Potential(...)`

**用途**: 特定の確率変数に紐づかない、任意の対数尤度(または制約)の項をモデルの目的関数に直接加える。

**シグネチャ**: `pymc.Potential(name, var, model=None, dims=None)`

**使用例**:
```python
import numpy as np
with pm.Model() as m:
    x = pm.Normal("x", mu=0, sigma=1)
    # xが正でなければ対数尤度に-infを加えて棄却する(ハードな制約)
    pot = pm.Potential("constraint", pm.math.switch(x > 0, 0.0, -np.inf))
    idata = pm.sample(draws=500, tune=500, chains=2, random_seed=0, progressbar=False)
print("x min:", float(idata.posterior["x"].min()))
```
実行結果:
```
There were 478 divergences after tuning. Increase `target_accept` or reparameterize.
x min: 0.006645857839666603
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `-np.inf`によるハードな打ち切り制約を`Potential`で表現すると、確認したとおり大量のダイバージェンス(478/1000)が発生する。NUTSは勾配ベースの手法のため、確率密度が不連続に0になる境界があると探索効率が大きく落ちる。制約自体は正しく反映される(`x`の事後サンプルは全て正)が、実務では`pm.Truncated`や、支持域が最初から正である`HalfNormal`等の分布を使う方が安全。
- `Deterministic`と違い、`Potential`の値自体はモデルの「変数」としては保存されない(対数尤度への加算項という扱い)。

---

## 4. 事前予測・事後予測チェック

### `sample_prior_predictive(...)`

**用途**: MCMCサンプリング前に、事前分布だけからパラメータ・観測値をシミュレーションする(モデル設定の妥当性チェックに使う)。

**シグネチャ**: `pymc.sample_prior_predictive(draws=500, model=None, var_names=None, random_seed=None, return_inferencedata=True, idata_kwargs=None, backend=None, compile_kwargs=None)`

**使用例**:
```python
with m:
    prior = pm.sample_prior_predictive(draws=100, random_seed=0)
print(prior.prior["mu"].shape)
print(prior.prior_predictive["obs"].shape)
```
実行結果:
```
(1, 100)
(1, 100, 50)
```

**注意点・落とし穴**:
- 戻り値は`InferenceData`(実体は`xarray.DataTree`)で、事前分布のパラメータサンプルは`prior`グループ、観測変数のシミュレーション値は`prior_predictive`グループに入る。
- `draws`のデフォルトは500(`sample`のデフォルト1000とは異なる)。

### `sample_posterior_predictive(...)`

**用途**: MCMCで得た事後分布のパラメータを使って、観測変数のシミュレーション値(事後予測)を生成する。モデルの当てはまり確認や新規データへの予測に使う。

**シグネチャ**: `pymc.sample_posterior_predictive(trace, model=None, *, var_names=None, sample_vars=None, freeze_vars=None, sample_dims=None, random_seed=None, progressbar=True, progressbar_theme=None, return_inferencedata=True, extend_inferencedata=False, predictions=False, idata_kwargs=None, backend=None, compile_kwargs=None)`

**使用例**:
```python
with m:
    ppc = pm.sample_posterior_predictive(idata, random_seed=0, progressbar=False)
print(ppc.posterior_predictive["obs"].shape)
```
実行結果:
```
(2, 500, 50)
```

**注意点・落とし穴**:
- 第一引数`trace`には`pm.sample()`の戻り値(`InferenceData`)をそのまま渡す。`chain`×`draw`の各事後サンプルごとに1回シミュレーションするため、結果の形状は`(chains, draws, 観測データの次元)`になる。
- `extend_inferencedata=True`にすると、新しい`idata`を作らず既存の`idata`に`posterior_predictive`グループを追加できる(`idata = pm.sample_posterior_predictive(idata, extend_inferencedata=True)`のように使う)。

---

## 5. MCMCサンプリング

### `sample(...)`

**用途**: MCMC(既定ではNUTS)で事後分布からサンプリングする、PyMCの中心的な関数。

**シグネチャ**: `pymc.sample(draws=1000, *, tune=None, chains=None, cores=None, random_seed=None, progressbar=True, step=None, var_names=None, nuts_sampler=None, init='auto', discard_tuned_samples=True, compute_convergence_checks=True, return_inferencedata=True, **kwargs)`(実際のフルシグネチャにはさらに`initvals`/`trace`/`idata_kwargs`等、合計20以上のキーワード引数がある)

**使用例**:
```python
with m:
    idata = pm.sample(draws=500, tune=500, chains=2, random_seed=0, progressbar=False)
```
実行結果(標準出力ログ):
```
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (2 chains in 2 jobs)
NUTS: [mu, sigma]
Sampling 2 chains for 500 tune and 500 draw iterations (1_000 + 1_000 draws total) took 0 seconds.
We recommend running at least 4 chains for robust computation of convergence diagnostics
```

**注意点・落とし穴**:
- デフォルトの`chains=None`は実際にはCPUコア数に応じて自動決定される(通常2以上)。今回の環境では`chains=2`と明示してもPyMCから「収束診断には最低4チェーン推奨」という警告が出る(実測で確認)。本番の分析では`chains=4`以上が推奨。
- `nuts_sampler=None`(デフォルト)ではPyMC純正のNUTS実装が使われる。`nuts_sampler='numpyro'`や`'nutpie'`を指定すると、対応ライブラリがインストールされていればより高速な実装に切り替えられる。
- `tune`(バーンイン相当)のサンプルは`discard_tuned_samples=True`(デフォルト)により結果には含まれない。

### `NUTS(...)`(明示的なステップ指定)

**用途**: HMCの発展形であるNUTS(No-U-Turn Sampler)のステップオブジェクトを明示的に作り、`target_accept`などのチューニングパラメータを調整する。

**シグネチャ**: `pymc.NUTS(vars=None, max_treedepth=10, early_max_treedepth=8, **kwargs)`(`target_accept`は`**kwargs`経由で親クラスの`HamiltonianMC`に渡される)

**使用例**:
```python
with pm.Model() as m2:
    mu = pm.Normal("mu", 0, 1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=y)
    step = pm.NUTS(target_accept=0.9)
    idata2 = pm.sample(draws=300, tune=300, chains=2, step=step, random_seed=0, progressbar=False)
```
実行結果:
```
    mean   sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
mu  1.86  0.2      1.6      2.2      215      223  1.00     0.013  0.0089
```

**注意点・落とし穴**:
- 通常`pm.sample()`は`step`を省略すると自動的に(連続変数には)NUTSを内部で構築するため、明示的に`pm.NUTS()`を渡す必要があるのは`target_accept`(既定0.8程度からの引き上げ)などを調整してダイバージェンスを減らしたい場合が中心。
- `target_accept`を上げる(例: 0.9〜0.99)とステップサイズが小さくなり、ダイバージェンスは減りやすいが1イテレーションあたりの計算コストが増える。

### `find_MAP(...)`

**用途**: 事後分布の最大点(MAP推定値)を数値最適化で求める。MCMCの初期値や簡易的な点推定に使う。

**シグネチャ**: `pymc.find_MAP(start=None, vars=None, method='L-BFGS-B', return_raw=False, include_transformed=True, progressbar=True, progressbar_theme=<rich.theme.Theme>, maxeval=5000, model=None, *args, seed=None, **kwargs)`

**使用例**:
```python
with m:
    map_est = pm.find_MAP(progressbar=False)
print(map_est)
```
実行結果:
```
{'mu': array(3.13599769), 'sigma_log__': array(0.13162892), 'sigma': array(1.14068496), 'mu_sq': array(9.8344815)}
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: 戻り値の辞書には、非負制約のある`sigma`について、制約なし空間に変換した`sigma_log__`(対数スケール)と、元のスケールに戻した`sigma`の両方が含まれる。`include_transformed=True`(デフォルト)によるもので、混同しないよう注意。
- MAP推定値は多峰性のある事後分布や高次元モデルでは事後分布の代表点として不適切なことが多い。多くの場合、完全なMCMC(`pm.sample()`)の方が信頼できる。

### `draw(...)`

**用途**: モデルコンテキストの外でも、分布オブジェクト(`.dist()`)から直接乱数サンプルを取り出す。

**シグネチャ**: `pymc.draw(vars, draws=1, random_seed=None, *, backend=None, **kwargs)`

**使用例**:
```python
a = pm.Normal.dist(0, 1)
b = pm.HalfNormal.dist(1)
vals = pm.draw([a, b], draws=3, random_seed=0)
print(vals)
```
実行結果:
```
[array([ 0.80508947, -1.91205922, -3.49664925]), array([1.44369095, 0.89594598, 0.73595567])]
```

**注意点・落とし穴**:
- `vars`にリストを渡すと、各要素ごとに独立した`ndarray`のリストが返る(まとめて1つの配列にはならない)。
- 本章のほとんどの分布の使用例で使っている`pm.draw(pm.Normal.dist(...), draws=n, random_seed=...)`は、まさにこの関数を使っている。

---

## 6. 変分推論(ADVI)

### `fit(...)`

**用途**: MCMCより高速な近似推論手法である変分推論(デフォルトはADVI)で事後分布を近似する。

**シグネチャ**: `pymc.fit(n=10000, method='advi', model=None, random_seed=None, start=None, start_sigma=None, inf_kwargs=None, *, backend=None, include_transformed=False, **kwargs)`

**使用例**:
```python
with m:
    approx = pm.fit(n=20000, random_seed=0, progressbar=False)
print(type(approx))
```
実行結果:
```
Finished [100%]: Average Loss = 85.597
<class 'pymc.variational.approximations.MeanField'>
```

**注意点・落とし穴**:
- `method='advi'`(デフォルト)は各パラメータを独立な正規分布で近似する(Mean-Field近似)。パラメータ間の相関が強いモデルでは近似精度が落ちる(`method='fullrank_advi'`だと相関も近似できるが計算コストが増える)。
- `fit()`の戻り値は事後サンプルそのものではなく`Approximation`オブジェクト。実際のサンプルを得るには次項の`.sample()`を呼ぶ必要がある。
- `n`(最適化イテレーション数)が少なすぎると収束前に打ち切られる。収束の目安は`approx.hist`(損失の推移)で確認できる。

### `Approximation.sample(...)`

**用途**: `pm.fit()`で得た近似分布から事後サンプルを生成し、`pm.sample()`と同じ`InferenceData`形式で扱えるようにする。

**シグネチャ**: `MeanField.sample(self, draws=500, *, random_seed=None, return_inferencedata=True, **kwargs)`

**使用例**:
```python
idata_advi = approx.sample(500)
print(az.summary(idata_advi, var_names=["mu", "sigma"]))
```
実行結果:
```
       mean    sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
mu     3.12  0.21      2.8      3.5      505      409   nan    0.0094  0.0061
sigma   1.2  0.14     0.99      1.4      445      448   nan    0.0066  0.0047
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `.sample()`が返す`InferenceData`は`chain`次元の長さが1(単一の近似分布から独立にサンプリングするため複数チェーンの概念がない)。そのため`r_hat`(複数チェーン間の収束診断)は計算できず`nan`になる。ADVIの結果を評価する際は`r_hat`ではなく、真のMCMC結果との比較や`approx.hist`の収束状況を見る。
- ADVIは事後分布の分散を過小評価する傾向が知られている(Mean-Field近似が相関を無視するため)。最終的な結論にはMCMC(`pm.sample()`)での検証を推奨。

---

## 7. 診断・モデル比較(ArviZ)

### `arviz.summary(...)`

**用途**: 事後分布の平均・標準偏差・信用区間・収束診断(r_hat, ESS)をまとめた表を作る。

**シグネチャ**: `arviz.summary(data, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, kind='all', fmt='wide', ci_prob=None, ci_kind=None, round_to='auto', skipna=False)`

**使用例**:
```python
import arviz as az
print(az.summary(idata, var_names=["mu", "sigma"]))
```
実行結果:
```
        mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
mu      3.14  0.173      2.9      3.4     1054      729  1.00    0.0053  0.0037
sigma  1.184  0.125        1      1.4     1072      637  1.00    0.0039  0.0034
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: ArviZ 1.3.0では信用区間のデフォルトが`ci_kind='eti'`(等尾区間)・`ci_prob=0.89`(89%)であり、表の列名も`hdi_3%`/`hdi_97%`ではなく`eti89_lb`/`eti89_ub`になる(`az.rcParams['stats.ci_kind']`/`az.rcParams['stats.ci_prob']`で確認)。94%HDIを期待していたコードを移植する場合は`ci_kind='hdi', ci_prob=0.94`を明示する必要がある。
- `r_hat`が1.01を明確に超える変数は収束不十分の疑いがあるとされる(目安)。

### `arviz.rhat(...)` / `arviz.ess(...)`

**用途**: 複数チェーンの収束を測る`r_hat`(潜在的尺度縮小因子)と、実質的な独立サンプル数を表す`ESS`(有効サンプルサイズ)を個別に計算する。

**シグネチャ**: `arviz.rhat(data, sample_dims=None, group='posterior', var_names=None, method='rank', chain_axis=0, draw_axis=1)` / `arviz.ess(data, sample_dims=None, group='posterior', var_names=None, method='bulk', relative=False, prob=None, chain_axis=0, draw_axis=1)`

**使用例**:
```python
print(az.rhat(idata, var_names=["mu", "sigma"]))
print(az.ess(idata, var_names=["mu", "sigma"]))
```
実行結果:
```
<xarray.DataTree 'posterior'>
Group: /posterior
    Dimensions:  ()
    Data variables:
        mu       float64 8B 1.001
        sigma    float64 8B 1.005
<xarray.DataTree 'posterior'>
Group: /posterior
    Dimensions:  ()
    Data variables:
        mu       float64 1.054e+03
        sigma    float64 1.072e+03
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: ArviZ 1.3.0ではこれらの戻り値は素の`xarray.Dataset`ではなく`posterior`グループを持つ`xarray.DataTree`になっている。値を取り出すには`az.rhat(idata)["posterior"]["mu"].item()`のようにグループを経由する。
- `method='rank'`(rhatのデフォルト)は、従来の分割Rhatより多峰性や裾の収束問題を検出しやすい改良版(rank-normalized R-hat)。

### `arviz.plot_trace(...)`

**用途**: 各パラメータの事後分布(周辺分布)とトレース(サンプル列)を並べて可視化し、収束やダイバージェンスを目視確認する。

**シグネチャ**: `arviz.plot_trace(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_trace(idata, var_names=["mu", "sigma"])
pc.savefig("trace.png")
print(type(pc))
```
実行結果:
```
<class 'arviz_plots.plot_collection.PlotCollection'>
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: ArviZ 1.3.0の`plot_trace`は、古いバージョンで一般的だった「matplotlibの`Axes`の2次元`ndarray`」ではなく、新しいプロット基盤(`arviz_plots`)の`PlotCollection`オブジェクトを返す。`axes[0, 0]`のようなインデックスアクセスは使えず、画像として保存するには`.savefig(path)`を使う。
- Jupyter上で単に評価すると自動的に描画されるが、スクリプト実行時はバックエンド(`matplotlib.use("Agg")`等)の設定と明示的な保存が必要。

### `pymc.compute_log_likelihood(...)`

**用途**: サンプリング済みの`InferenceData`に対して、各観測点ごとの対数尤度を計算して`log_likelihood`グループとして追加する(`loo`/`compare`の前処理として必須)。

**シグネチャ**: `pymc.compute_log_likelihood(idata, *, var_names=None, extend_inferencedata=True, model=None, sample_dims=('chain', 'draw'), progressbar=True, backend=None, compile_kwargs=None)`

**使用例**:
```python
with m1:
    idata1 = pm.sample(draws=500, tune=500, chains=2, random_seed=0, progressbar=False)
    pm.compute_log_likelihood(idata1)
print("log_likelihood" in idata1.groups())
```
実行結果:
```
True
```

**注意点・落とし穴**:
- `extend_inferencedata=True`(デフォルト)により、渡した`idata`オブジェクト自体が書き換えられて`log_likelihood`グループが追加される(新しいオブジェクトは返らない)。
- `az.loo()`や`az.compare()`を使う前に、比較したい全モデルの`idata`に対してこの関数を呼んでおく必要がある。

### `arviz.loo(...)` / `arviz.compare(...)`

**用途**: `loo`はPSIS-LOO(近似的なleave-one-out交差検証)によるモデルの予測性能(elpd)を計算し、`compare`は複数モデルのelpdを並べて順位付けする。

**シグネチャ**: `arviz.loo(data, pointwise=None, var_name=None, reff=None, model=None, ...)` / `arviz.compare(compare_dict, method='stacking', var_name=None, reference=None, round_to='auto')`

**使用例**:
```python
print(az.loo(idata1))
cmp = az.compare({"pooled": idata1, "hierarchical": idata2})
print(cmp)
```
実行結果:
```
Computed from 1000 posterior samples and 80 observations log-likelihood matrix.

         Estimate       SE
elpd_loo  -127.29     4.04
p_loo        1.53        -
------

Pareto k diagnostic values:
                         Count   Pct.
(-Inf, 0.67]   (good)       80  100.0%
   (0.67, 1]   (bad)         0    0.0%
    (1, Inf)   (very bad)    0    0.0%

              rank  elpd_diff  dse  p_worse  ...    p   elpd   se  weight
hierarchical     0        0.0  0.0      NaN  ...  5.2  -50.0  7.6     1.0
pooled           1      -80.0  8.2      1.0  ...  1.5 -127.0  4.0     0.0
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: ArviZ 1.3.0には`arviz.waic`が**存在しない**(呼び出すと`AttributeError: module 'arviz' has no attribute 'waic'`になる)。WAICはLOOに比べて実用上の弱点があるとされ非推奨化・削除されており、モデル比較には`az.loo`/`az.compare`を使う。
- `compare()`の結果の`rank=0`が最良モデル(elpdが最大)。今回の例では、真のデータ生成過程が群ごとに平均が異なる階層構造のため、`hierarchical`モデルの方が`pooled`モデルよりelpdが大幅に高い(`weight=1.0`)。
- `Pareto k`診断値が0.7を超える観測点が多い場合、PSIS近似の信頼性が低下している(`az.loo`の警告メッセージで通知される)。

---

## 8. 階層モデル・座標系(coords/dims)

### `Model(coords=...)` と `dims=...`

**用途**: グループ(例: 地域・個体)ごとに異なるパラメータを持つ階層モデルを、インデックス番号ではなくラベル付きの次元(`xarray`の座標)で分かりやすく管理する。

**シグネチャ**: `pymc.Model(coords: dict[str, Sequence] | None = None)` の`coords`引数、および各分布・`Data`の`dims: str | Sequence[str] | None`引数

**使用例**:
```python
import numpy as np
n_groups, n_per = 4, 20
group_idx = np.repeat(np.arange(n_groups), n_per)
y = ...  # 観測データ(省略)

coords = {"group": ["A", "B", "C", "D"], "obs_id": np.arange(len(y))}
with pm.Model(coords=coords) as hmodel:
    group_idx_data = pm.Data("group_idx", group_idx, dims="obs_id")
    mu_group = pm.Normal("mu_group", mu=0, sigma=10, dims="group")
    sigma = pm.HalfNormal("sigma", sigma=5)
    obs = pm.Normal("obs", mu=mu_group[group_idx_data], sigma=sigma, observed=y, dims="obs_id")
    idata_h = pm.sample(draws=500, tune=500, chains=2, random_seed=0, progressbar=False)

print(az.summary(idata_h, var_names=["mu_group", "sigma"]))
```
実行結果:
```
              mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
mu_group[A]   1.02  0.098     0.87      1.2      924      708  1.00    0.0032   0.0023
mu_group[B]  1.983  0.097      1.8      2.1     1103      757  1.00    0.0029   0.0021
mu_group[C]  2.926  0.098      2.8      3.1     1340      726  1.00    0.0027   0.0019
mu_group[D]  3.929  0.099      3.8      4.1     1210      890  1.00    0.0029   0.0019
sigma        0.438  0.036     0.38      0.5     1132      566  1.01    0.0011  0.00081
```

**注意点・落とし穴**:
- `dims="group"`を指定した`mu_group`は、`az.summary`や`idata.posterior["mu_group"]`で`mu_group[A]`, `mu_group[B]`...のように`coords`で与えたラベルで表示される。`dims`を指定しない場合は`mu_group[0]`, `mu_group[1]`のような番号になり可読性が落ちる。
- `mu_group[group_idx_data]`のように、パラメータをNumPy風のファンシーインデックスで観測データの数だけ複製するのが階層モデルの定石パターン。`group_idx_data`を`pm.Data`にしておくことで、3章の`pm.set_data()`を使って新しいグループ割り当てに対する予測もできる。

---

## 9. その他ユーティリティ

### `model_to_graphviz(...)`

**用途**: モデルの確率変数の依存関係を有向グラフとして可視化する(Graphvizが必要)。

**シグネチャ**: `pymc.model_to_graphviz(model=None, *, var_names=None, formatting='plain', save=None, figsize=None, dpi=300, include_dim_lengths=True)`

**使用例**:
```python
g = pm.model_to_graphviz(m2)
g.render(filename="model_graph", format="png", cleanup=True)
print(type(g))
```
実行結果:
```
<class 'graphviz.graphs.Digraph'>
```

**注意点・落とし穴**:
- 戻り値は`graphviz.Digraph`オブジェクト(PyMC自体の関数ではなく`graphviz`ライブラリのラッパー)。Jupyter上ではセルの最後に評価するだけで自動描画されるが、スクリプトでは`.render()`または`save`引数でファイル出力する必要がある。
- システムにGraphviz本体(`dot`コマンド)がインストールされていないと`.render()`実行時にエラーになる(Pythonパッケージの`graphviz`だけでは不十分)。

### `pymc.math`(`invlogit` / `switch` などの計算ユーティリティ)

**用途**: モデル内の確率変数(PyTensorのテンソル)に対して、NumPyに似た数式操作(シグモイド変換、条件分岐など)を行う。ロジスティック回帰などでよく使う。

**シグネチャ**: `pymc.math.invlogit(x)`(`pymc.math.sigmoid`のエイリアス)、`pymc.math.switch(cond, a, b)`

**使用例**:
```python
with pm.Model() as glm:
    a = pm.Normal("a", 0, 5)
    b = pm.Normal("b", 0, 5)
    p = pm.Deterministic("p", pm.math.invlogit(a + b * x))
    obs = pm.Bernoulli("obs", p=p, observed=y)
    idata = pm.sample(draws=500, tune=500, chains=2, random_seed=0, progressbar=False)
print(az.summary(idata, var_names=["a", "b"]))
```
実行結果:
```
    mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
a  -0.36  0.194    -0.67   -0.043      978      725  1.00    0.0062   0.004
b   2.34   0.33      1.8      2.9      800      618  1.00     0.012  0.0088
```
(真の係数 `a=-0.5, b=2.0` に対し、50件程度のノイズを含むデータからおおむね近い値が推定できている)

**注意点・落とし穴**:
- `pm.math.invlogit(a + b*x)`のように線形結合をシグモイド変換して`Bernoulli(p=...)`に渡す代わりに、`Bernoulli(logit_p=a + b*x)`と書くこともできる(2章参照)。後者の方が内部の数値計算(log-sum-expなど)が安定するため、実務では`logit_p`を使う方が無難なことが多い。
- `pymc.math`は基本的に`pytensor.tensor`の関数の再エクスポートであり、NumPyの関数(`np.log`など)をPyTensorの確率変数にそのまま使うとエラーになる場合、対応する`pymc.math`(`pm.math.log`等)に置き換える。
