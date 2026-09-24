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
10. [応用・発展](#10-応用発展)
    - 10-1. [カスタム分布・特殊な分布ラッパー](#10-1-カスタム分布特殊な分布ラッパー)
    - 10-2. [ガウス過程(pm.gp)](#10-2-ガウス過程pmgp)
    - 10-3. [時系列モデル](#10-3-時系列モデル)
    - 10-4. [逐次モンテカルロ(SMC)](#10-4-逐次モンテカルロsmc)
    - 10-5. [欠損データ・並列サンプリング設定](#10-5-欠損データ並列サンプリング設定)
    - 10-6. [事前分布設計ユーティリティ](#10-6-事前分布設計ユーティリティ)

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
- **バージョン固有の注意(実測で確認)**: ArviZ 1.3.0ではこれらの戻り値は素の`xarray.Dataset`ではなく、名前が`'posterior'`の`xarray.DataTree`になっている(変数`mu`・`sigma`は直下にある)。値は`az.rhat(idata)["mu"].item()`のように変数名で直接取り出す。`az.rhat(idata)["posterior"]["mu"]`のように`posterior`を経由すると`KeyError: 'Could not find node at posterior'`になる。
- `method='rank'`(rhatのデフォルト)は、従来の分割Rhatより多峰性や裾の収束問題を検出しやすい改良版(rank-normalized R-hat)。

### `arviz.plot_trace(...)`

**用途**: 各パラメータのトレース(チェーンごとのサンプル列)を可視化し、収束やチェーンの混ざり具合を目視確認する。

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
- **バージョン固有の注意(実測で確認)**: ArviZ 1.3.0の`plot_trace`はトレース(チェーンごとのサンプル列)だけを描き、0.x系の「左に周辺分布(密度)、右にトレース」という2列構成ではない(パラメータごとに1枚のトレースパネルが並ぶ)。周辺分布を見たい場合は別の関数を使う(詳しくは[arviz/README.md](../arviz/README.md))。
- Jupyter上で単に評価すると自動的に描画されるが、スクリプト実行時はバックエンド(`matplotlib.use("Agg")`等)の設定と明示的な保存が必要。

### `pymc.compute_log_likelihood(...)`

**用途**: サンプリング済みの`InferenceData`に対して、各観測点ごとの対数尤度を計算して`log_likelihood`グループとして追加する(`loo`/`compare`の前処理として必須)。

**シグネチャ**: `pymc.compute_log_likelihood(idata, *, var_names=None, extend_inferencedata=True, model=None, sample_dims=('chain', 'draw'), progressbar=True, backend=None, compile_kwargs=None)`

**使用例**:
```python
with m1:
    idata1 = pm.sample(draws=500, tune=500, chains=2, random_seed=0, progressbar=False)
    pm.compute_log_likelihood(idata1)
print("log_likelihood" in idata1.children)
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

---

## 10. 応用・発展

### 10-1. カスタム分布・特殊な分布ラッパー

#### `CustomDist(...)`

**用途**: 組み込みの分布にない任意の尤度(`logp`)や生成過程(`dist`/`random`)を自作し、通常の分布と同じようにモデルへ組み込む。

**シグネチャ**: `pymc.CustomDist(name, *dist_params, dist=None, random=None, logp=None, logcdf=None, support_point=None, ndim_supp=None, ndims_params=None, signature=None, dtype='floatX', **kwargs)`

**使用例**:
```python
import numpy as np
import pymc as pm
import arviz as az

rng = np.random.default_rng(0)
y_obs = rng.normal(loc=3.0, scale=1.5, size=60)

def logp_custom(value, mu, sigma):
    # 中身はNormalの対数尤度だが、任意の対数尤度式に置き換えられる
    return pm.logp(pm.Normal.dist(mu, sigma), value)

with pm.Model() as m:
    mu = pm.Normal("mu", 0, 10)
    sigma = pm.HalfNormal("sigma", 5)
    y = pm.CustomDist("y", mu, sigma, logp=logp_custom, observed=y_obs)
    idata = pm.sample(draws=500, tune=500, chains=2, random_seed=0, progressbar=False)

print(az.summary(idata, var_names=["mu", "sigma"]))
```
実行結果:
```
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (2 chains in 2 jobs)
NUTS: [mu, sigma]
Sampling 2 chains for 500 tune and 500 draw iterations (1_000 + 1_000 draws total) took 2 seconds.
We recommend running at least 4 chains for robust computation of convergence diagnostics
        mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
mu       3.1    0.2      2.8      3.4      842      529  1.00    0.0073  0.0062
sigma  1.394  0.136      1.2      1.6     1025      753  1.00    0.0044  0.0037
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `logp`だけを渡して`dist`/`random`を渡さない場合、MCMCサンプリング(`observed`ありの尤度評価)は問題なく動くが、`pm.sample_prior_predictive()`のようにこの変数から乱数生成しようとすると`NotImplementedError: Attempted to run random on the CustomDist '...', but this method had not been provided when the distribution was constructed.`になる。事前予測チェックまで行いたい場合は`random`(または`dist`)も併せて実装する必要がある。
- `logp`関数の第一引数は必ず`value`(観測値またはサンプル値)で、以降の引数が`*dist_params`で渡したパラメータ(ここでは`mu, sigma`)に対応する。

#### `Truncated(...)`

**用途**: 既存の任意の分布を「指定範囲内だけの値を取るように」切り詰める(密度を範囲内で再正規化する)。

**シグネチャ**: `pymc.Truncated.dist(dist, lower=None, upper=None, max_n_steps=10000, **kwargs)`

**使用例**:
```python
import pymc as pm

base = pm.Normal.dist(mu=0, sigma=1)
trunc = pm.Truncated.dist(base, lower=0, upper=None)
print(pm.draw(trunc, draws=10, random_seed=0))
```
実行結果:
```
[1.90283218 0.40746997 1.08559681 0.158076   0.55773751 0.93079058
 0.07109423 1.33742802 0.3433924  0.99291241]
```

**注意点・落とし穴**:
- 第一引数`dist`には`pm.Normal.dist(...)`のような「まだモデルに登録していない」分布オブジェクトを渡す(3章の`pm.Normal("mu", ...)`のようなモデル登録済み変数ではない)。
- 次項の`Censored`と混同しやすいが、`Truncated`は範囲外の確率密度自体を捨てて範囲内で再正規化するため、生成される値は必ず`lower`〜`upper`に収まり、境界に点質量は生じない。

#### `Censored(...)`

**用途**: 既存の分布からサンプリングした後、範囲外の値を境界値に切り詰める(打ち切り観測をモデル化する。例: センサーの測定上限)。

**シグネチャ**: `pymc.Censored.dist(dist, lower=-inf, upper=inf, **kwargs)`

**使用例**:
```python
base2 = pm.Normal.dist(mu=0, sigma=1)
cens = pm.Censored.dist(base2, lower=-1, upper=1)
print(pm.draw(cens, draws=10, random_seed=0))
```
実行結果:
```
[ 1.         -0.89594598  0.73595567  0.00587704  0.85338179  0.16094803
  0.81931469  0.80565568  0.21756732  0.97007874]
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `Truncated`と同じ乱数シードで比較すると挙動の違いがよくわかる。元の`Normal.dist(0, 1)`の1つ目のサンプルは`1.4437`だが、`Censored(lower=-1, upper=1)`では上限`1`にクリップされてちょうど`1.0`になっている(`Truncated`ならこの値自体が生成されない)。`Censored`は境界に点質量(確率の塊)を作るが、`Truncated`は境界に到達しない連続分布のままという違いがある。
- 用途に応じて使い分ける: 測定機器の上限/下限で観測値が丸められる(打ち切り)なら`Censored`、そもそも範囲外の値が物理的に存在しえない(切断)なら`Truncated`が適切。

#### `Mixture(...)`

**用途**: 複数の分布を重み付きで混合した分布(例: 2つの正規分布からなる二峰性データ)をモデル化する。

**シグネチャ**: `pymc.Mixture.dist(w, comp_dists, **kwargs)`

**使用例**:
```python
w = [0.3, 0.7]
comp_dists = [pm.Normal.dist(mu=-3, sigma=1), pm.Normal.dist(mu=3, sigma=1)]
mix = pm.Mixture.dist(w=w, comp_dists=comp_dists)
print(pm.draw(mix, draws=10, random_seed=0))
```
実行結果:
```
[ 4.44369095 -4.91205922  3.73595567  3.00587704 -1.70636569  3.16094803
  3.81931469  3.80565568  3.21756732 -2.24800945]
```

**注意点・落とし穴**:
- `w`(各コンポーネントの重み)は合計1になる必要があり、`comp_dists`のリストの長さと一致していなければならない。
- 出力を見ると生成値のほとんどが`mu=3`側(重み0.7)に偏っている。モデル内で`w`自体を`pm.Dirichlet`の事前分布を持つ確率変数にすれば、混合比率も一緒に推定できる。

#### `MvNormal(...)`

**用途**: 多変量正規分布。複数の確率変数間の相関を明示的にモデル化する(例: 複数指標の同時分布、相関のある回帰係数)。

**シグネチャ**: `pymc.MvNormal.dist(mu=0, cov=None, *, tau=None, chol=None, lower=True, **kwargs)`

**使用例**:
```python
mu = [0, 0]
cov = [[1.0, 0.7], [0.7, 1.0]]
mv = pm.MvNormal.dist(mu=mu, cov=cov)
print(pm.draw(mv, draws=5, random_seed=0))
```
実行結果:
```
[[1.44369095 0.37075026]
 [0.73595567 0.51936602]
 [0.85338179 0.71230714]
 [0.81931469 1.14887352]
 [0.21756732 0.84507191]]
```

**注意点・落とし穴**:
- 共分散行列は`cov`(共分散)・`tau`(精度行列)・`chol`(コレスキー分解)のいずれか1つで指定する。サンプリングの数値安定性・速度の観点では、モデル内で`chol`を`pm.LKJCholeskyCov`などから構築して渡す方が、生の`cov`を直接渡すより好まれる。
- 戻り値の各行が1サンプル(2次元ベクトル)になっており、`draws=5`の結果は`(5, 2)`の配列になる(1次元分布の`pm.draw`とは形状の次元数が異なる)。

### 10-2. ガウス過程(pm.gp)

#### `pm.gp.cov.ExpQuad(...)`

**用途**: ガウス過程(GP)の共分散関数(カーネル)のうち最も基本的なもの。2点間の距離が近いほど強く相関する滑らかな関数を表現する。

**シグネチャ**: `pymc.gp.cov.ExpQuad(input_dim: int, ls=None, ls_inv=None, active_dims=None)`

**使用例**:
```python
import numpy as np
import pymc as pm

X = np.array([[0.0], [1.0], [2.0]])
cov_func = 2.0 ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=1.0)
K = cov_func(X).eval()
print(K)
```
実行結果:
```
[[4.         2.42612264 0.54134113]
 [2.42612264 4.         2.42612264]
 [0.54134113 2.42612264 4.        ]]
```

**注意点・落とし穴**:
- `ls`(lengthscale)が長いほど遠くの点同士の相関が強く保たれ(なめらかな関数)、短いほど相関が急速に減衰する(ギザギザした関数)。`eta**2 * ExpQuad(...)`のように分散スケール(`eta`)を掛けて使うのが定石で、対角成分(自己共分散)が`eta**2`になる(上の例では`2.0**2=4.0`)。
- `input_dim`は入力`X`の列数(特徴量の次元数)と一致させる必要がある。`cov_func(X)`はPyTensorの計算グラフを返すだけなので、具体的な数値を見るには`.eval()`が必要。

#### `pm.gp.Marginal(...)` と `.marginal_likelihood(...)`

**用途**: 観測ノイズが正規分布であるガウス過程回帰を、GPの関数値自体をサンプリングせずに周辺化(積分消去)した尤度で効率よく推定する。

**シグネチャ**: `pymc.gp.Marginal(*, mean_func=Zero(), cov_func=Constant())`、`Marginal.marginal_likelihood(self, name, X, y, sigma, jitter=1e-06, is_observed=True, **kwargs)`

**使用例**:
```python
import numpy as np
import pymc as pm
import arviz as az

rng = np.random.default_rng(0)
X = np.linspace(0, 10, 20)[:, None]
y = np.sin(X[:, 0]) + rng.normal(scale=0.2, size=20)

with pm.Model() as gp_model:
    ell = pm.HalfNormal("ell", sigma=2)
    eta = pm.HalfNormal("eta", sigma=2)
    cov_func = eta ** 2 * pm.gp.cov.ExpQuad(1, ls=ell)
    gp = pm.gp.Marginal(cov_func=cov_func)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)
    idata = pm.sample(draws=300, tune=300, chains=2, random_seed=0, progressbar=False, target_accept=0.9)

print(az.summary(idata, var_names=["ell", "eta", "sigma"]))
```
実行結果:
```
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (2 chains in 2 jobs)
NUTS: [ell, eta, sigma]
Sampling 2 chains for 300 tune and 300 draw iterations (600 + 600 draws total) took 141 seconds.
We recommend running at least 4 chains for robust computation of convergence diagnostics
        mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
ell     1.66   0.43     0.97      2.3      203      344  1.00      0.03   0.019
eta     1.09   0.46     0.57      1.9      236      235  1.00     0.034   0.036
sigma  0.162  0.041     0.11     0.24      332      319  1.00    0.0022  0.0022
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `gp.Marginal`は内部で観測点数×観測点数の共分散行列のコレスキー分解を毎イテレーション計算するため、他章の単純な`Normal`モデルに比べて極端に遅い。上の例は観測点がわずか20点、`draws=300, tune=300, chains=2`という小規模設定でも実測141秒かかった(観測点数が増えると計算量は3乗で増大する)。データ数が多い場合は`pm.gp.MarginalApprox`や`pm.gp.HSGP`(Hilbert空間近似)などのスケーラブルな実装を検討する必要がある。
- `sigma`(観測ノイズの標準偏差)は`marginal_likelihood`の引数として渡し、GP自体の事前分布(`mean_func`/`cov_func`)とは別に指定する。

#### `Marginal.conditional(...)`(新しい入力点での予測)

**用途**: `marginal_likelihood`で学習したGPを使って、未観測の新しい入力点における関数値の事後予測分布を得る。

**シグネチャ**: `Marginal.conditional(self, name, Xnew, pred_noise=False, given=None, jitter=1e-06, **kwargs)`

**使用例**:
```python
Xnew = np.array([[10.5], [11.0]])
with gp_model:
    mu_pred = gp.conditional("mu_pred", Xnew)
    pred = pm.sample_posterior_predictive(idata, var_names=["mu_pred"], random_seed=0, progressbar=False)
print(pred.posterior_predictive["mu_pred"].mean(dim=["chain", "draw"]).values)
```
実行結果:
```
Sampling: [mu_pred]
[-0.50724903 -0.56592083]
```

**注意点・落とし穴**:
- `gp.conditional(...)`は学習済みの`gp_model`コンテキスト内で呼ぶ必要がある(元のモデルと同じ`with`ブロック、または`model=`引数で明示的に紐付ける)。呼び出すと新しい変数がモデルに追加されるだけなので、実際の予測値を得るには続けて`pm.sample_posterior_predictive`を呼ぶ。
- `pred_noise=False`(デフォルト)では観測ノイズ`sigma`を含まない「真の関数値」の予測になる。観測値そのものの予測区間(ノイズ込み)が欲しい場合は`pred_noise=True`を指定する。

#### `pm.gp.Latent(...)`

**用途**: 観測ノイズが正規分布ではない場合(例: ポアソン尤度のカウントデータ)にガウス過程を使う。GPの関数値そのものを潜在変数としてMCMCでサンプリングする。

**シグネチャ**: `pymc.gp.Latent(*, mean_func=Zero(), cov_func=Constant())`、`Latent.prior(self, name, X, n_outputs=1, reparameterize=True, jitter=1e-06, **kwargs)`

**使用例**:
```python
import numpy as np
import pymc as pm
import arviz as az

rng = np.random.default_rng(0)
X = np.linspace(0, 10, 10)[:, None]
f_true = np.sin(X[:, 0])
y = rng.poisson(np.exp(0.5 * f_true))

with pm.Model() as latent_model:
    ell = pm.HalfNormal("ell", sigma=2)
    cov_func = pm.gp.cov.ExpQuad(1, ls=ell)
    gp = pm.gp.Latent(cov_func=cov_func)
    f = gp.prior("f", X=X)
    y_obs = pm.Poisson("y_obs", mu=pm.math.exp(0.5 * f), observed=y)
    idata = pm.sample(draws=150, tune=150, chains=2, random_seed=0, progressbar=False, target_accept=0.9)

print(type(f))
print(az.summary(idata, var_names=["ell"]))
```
実行結果:
```
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (2 chains in 2 jobs)
NUTS: [ell, f_rotated_]
Sampling 2 chains for 150 tune and 150 draw iterations (300 + 300 draws total) took 125 seconds.
There was 1 divergence after tuning. Increase `target_accept` or reparameterize.
We recommend running at least 4 chains for robust computation of convergence diagnostics
The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
<class 'pytensor.tensor.variable.TensorVariable'>
    mean   sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
ell  1.4  1.2     0.12      3.7      253      224  1.00     0.069   0.057
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `NUTS: [ell, f_rotated_]`のログの通り、内部では`f`そのものではなく`reparameterize=True`(デフォルト)による非中心化パラメータ化`f_rotated_`がサンプリングされる(収束を改善するための標準的なテクニック)。`az.summary`で`f`自体の要約も見たい場合は`var_names=["f"]`のように明示すれば`Deterministic`経由で復元された値が参照できる。
- `pm.gp.Marginal`と異なりガウス尤度を仮定しないため任意の観測分布(ここでは`Poisson`)と組み合わせられるが、その分`f`(観測点数と同じ次元を持つ高次元の潜在変数)を直接NUTSでサンプリングすることになり、上の例のようにダイバージェンスが出やすく計算コストも高い。観測点数がわずか10点、`draws=150, tune=150`という小規模設定でも125秒かかった。

### 10-3. 時系列モデル

#### `AR(...)`

**用途**: 自己回帰(AR)過程。時点`t`の値が過去`p`時点の値の線形結合とノイズで決まる時系列データをモデル化する。

**シグネチャ**: `pymc.AR(name, rho, *args, steps=None, constant=False, ar_order=None, **kwargs)`(`.dist`版: `pymc.AR.dist(rho, sigma=None, tau=None, *, init_dist=None, steps=None, constant=False, ar_order=None, **kwargs)`)

**使用例**:
```python
import numpy as np
import pymc as pm
import arviz as az

rng = np.random.default_rng(0)
n = 100
true_rho = 0.7
y = np.zeros(n)
for t in range(1, n):
    y[t] = true_rho * y[t - 1] + rng.normal(scale=1.0)

with pm.Model() as ar_model:
    rho = pm.Normal("rho", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    ar = pm.AR("ar", rho=rho, sigma=sigma, init_dist=pm.Normal.dist(0, 1), observed=y)
    idata = pm.sample(draws=300, tune=300, chains=2, random_seed=0, progressbar=False)

print(az.summary(idata, var_names=["rho", "sigma"]))
```
実行結果:
```
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (2 chains in 2 jobs)
NUTS: [rho, sigma]
Sampling 2 chains for 300 tune and 300 draw iterations (600 + 600 draws total) took 1 seconds.
We recommend running at least 4 chains for robust computation of convergence diagnostics
        mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
rho     0.76  0.066     0.66     0.87      392      410  1.01    0.0032  0.0021
sigma  0.965  0.075     0.86      1.1      586      478  1.00    0.0031  0.0026
```
(真の`rho=0.7`に対し`0.76`程度と近い値が推定できている)

**注意点・落とし穴**:
- `rho`をスカラーの確率変数(単一の`pm.Normal`)として渡すとAR(1)になる。`rho`を長さ`p`のベクトル(または`shape=p`の分布)にして`ar_order=p`を指定すると、そのままAR(p)(p次の自己回帰)として扱える。
- `init_dist`(過程の最初の値の分布)を省略すると`UserWarning`とともに`Normal.dist(0, 100)`が自動的に使われる。過程の初期値についてある程度の知識がある場合は明示的に指定した方がよい。

#### `GaussianRandomWalk(...)`

**用途**: ガウシアン・ランダムウォーク。各時点の増分が正規分布に従う(自己回帰係数が常に1の特殊なAR)非定常な時系列の潜在トレンドなどに使う。

**シグネチャ**: `pymc.GaussianRandomWalk.dist(mu=0.0, sigma=1.0, *, init_dist=None, steps=None, **kwargs)`

**使用例**:
```python
import pymc as pm

grw = pm.GaussianRandomWalk.dist(mu=0, sigma=1, init_dist=pm.Normal.dist(0, 1), steps=5)
print(pm.draw(grw, draws=3, random_seed=0))
```
実行結果:
```
[[ 0.80508947  2.24878043  1.35283445  2.08879012  2.09466716  2.94804895]
 [-1.91205922 -1.75111118 -0.93179649 -0.12614081  0.09142651  1.06150525]
 [-3.49664925 -4.23576231 -3.64228438 -4.35774738 -5.11140475 -3.79773521]]
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `steps=5`を指定すると、出力される系列の長さは`steps`ではなく`steps + 1`(=6)になる(`init_dist`による初期値1点 + その後の5ステップの増分)。系列長を`n`にしたい場合は`steps=n-1`を指定する必要がある。
- `AR`の特殊ケース(`rho=1`固定)に相当し、単調に分散が増加していく(平均から離れやすくなる)非定常過程。株価やセンサー値の緩やかなトレンド成分のモデリングによく使われる。

### 10-4. 逐次モンテカルロ(SMC)

#### `sample_smc(...)`

**用途**: NUTSが苦手とする多峰性の強い事後分布や、離散変数を含むモデルにも比較的頑健な、逐次モンテカルロ(Sequential Monte Carlo)法でサンプリングする。

**シグネチャ**: `pymc.sample_smc(draws=2000, kernel=IMH, *, start=None, model=None, random_seed=None, chains=None, cores=None, blas_cores=None, compute_convergence_checks=True, return_inferencedata=True, idata_kwargs=None, progressbar=True, progressbar_theme=None, backend=None, compile_kwargs=None, mp_ctx=None, **kernel_kwargs)`

**使用例**:
```python
import numpy as np
import pymc as pm
import arviz as az

rng = np.random.default_rng(0)
y = rng.normal(loc=2.0, scale=1.0, size=50)

with pm.Model() as smc_model:
    mu = pm.Normal("mu", 0, 10)
    sigma = pm.HalfNormal("sigma", 5)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    idata_smc = pm.sample_smc(draws=500, chains=2, random_seed=0, progressbar=False)

print(az.summary(idata_smc, var_names=["mu", "sigma"]))
```
実行結果:
```
Initializing SMC sampler...
Sampling 2 chains in 2 jobs
We recommend running at least 4 chains for robust computation of convergence diagnostics
        mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
mu     2.126   0.13      1.9      2.3      948      925  1.00    0.0042  0.0032
sigma  0.946  0.099      0.8      1.1      973      834  1.00    0.0032  0.0024
```

**注意点・落とし穴**:
- `pm.sample()`(NUTS)とは異なる独立したサンプラーで、`draws`はNUTSの`tune`に相当する概念がなく、各世代でリサンプリング・重み付けを繰り返しながら事後分布に近づけていく(「chains」はSMCでは並列に動かす独立したパーティクル集団の数という意味合いが強い)。
- 戻り値の`idata_smc`には`log_likelihood`ではなく`sample_stats`グループにSMC特有の統計量(受理率など)が入る。通常のNUTSの結果と同じく`az.summary`や`az.plot_trace`はそのまま使えるが、`compute_log_likelihood`や`r_hat`の解釈はNUTSの場合と前提が異なる点に留意。

### 10-5. 欠損データ・並列サンプリング設定

#### 欠損データの自動インピュテーション(マスク配列)

**用途**: 観測データの一部が欠損している場合、`numpy.ma.masked_array`(またはNaNを含む配列)を`observed`にそのまま渡すだけで、PyMCが欠損値を自動的に未知パラメータとして扱いMCMCで補完(インピュテーション)する。

**シグネチャ**: 専用の関数はなく、`observed=`引数に`numpy.ma.MaskedArray`(または`pandas`のNaNを含むSeries/DataFrame)を渡す運用パターン。

**使用例**:
```python
import numpy as np
import numpy.ma as ma
import pymc as pm
import arviz as az

rng = np.random.default_rng(0)
y = rng.normal(loc=5.0, scale=2.0, size=20)
y_missing = y.copy()
y_missing[[3, 7, 15]] = np.nan
y_masked = ma.masked_invalid(y_missing)

with pm.Model() as impute_model:
    mu = pm.Normal("mu", 0, 10)
    sigma = pm.HalfNormal("sigma", 5)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_masked)
    print(sorted(impute_model.named_vars.keys()))
    idata = pm.sample(draws=300, tune=300, chains=2, random_seed=0, progressbar=False)

print(az.summary(idata, var_names=["obs_unobserved"]))
```
実行結果:
```
.../pymc/model/core.py:2061: ImputationWarning: Data in obs contains missing values and will be automatically imputed from the sampling distribution.
  warnings.warn(impute_message, ImputationWarning)
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (2 chains in 2 jobs)
NUTS: [mu, sigma, obs_unobserved]
['mu', 'obs', 'obs_observed', 'obs_unobserved', 'sigma']
Sampling 2 chains for 300 tune and 300 draw iterations (600 + 600 draws total) took 0 seconds.
We recommend running at least 4 chains for robust computation of convergence diagnostics
                  mean   sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
obs_unobserved[0]  4.5  2.1      1.3      7.9      791      479  1.00     0.073   0.053
obs_unobserved[1]  4.5    2      1.5      7.7      644      432  1.00     0.078    0.06
obs_unobserved[2]  4.5  2.2     0.92      7.9      600      383  1.00     0.091   0.077
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: マスクされた要素があると、元の`obs`という1つの確率変数が内部的に`obs_observed`(観測済みの値、定数)と`obs_unobserved`(欠損値、事後分布からサンプリングされる確率変数)の2つに自動分割される(`impute_model.named_vars`で確認可能)。`az.summary(idata, var_names=["obs"])`のように元の名前では欠損値の推定結果にアクセスできないので、`obs_unobserved`という名前を使う必要がある。
- `ImputationWarning`が出るのは仕様どおりの動作であり、エラーではない。ただし意図せずNaNが混入したデータをそのまま渡すと気づかずに欠損値補完が走ってしまうことがあるため、警告文は無視せず確認する習慣が重要。

#### `cores`/`chains`引数の挙動

**用途**: `pm.sample()`の`chains`(独立に走らせるMCMCチェーンの本数)と`cores`(実際に同時実行する並列プロセス数)の関係を理解し、意図通りの並列度でサンプリングする。

**シグネチャ**: `pymc.sample(..., chains=None, cores=None, ...)`(`chains`のデフォルトはNoneで内部的に2以上に、`cores`のデフォルトはNoneで`min(利用可能CPUコア数, chains)`に自動決定される)

**使用例**:
```python
import numpy as np
import pymc as pm

rng = np.random.default_rng(0)
y = rng.normal(loc=0, scale=1, size=20)

with pm.Model() as m:
    mu = pm.Normal("mu", 0, 10)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=y)
    idata_default = pm.sample(draws=200, tune=200, chains=4, random_seed=0, progressbar=False)

with m:
    idata_seq = pm.sample(draws=200, tune=200, chains=4, cores=1, random_seed=0, progressbar=False)
```
実行結果(標準出力ログ、環境は16コアCPU):
```
# cores省略(デフォルト)の場合
Multiprocess sampling (4 chains in 4 jobs)

# cores=1を明示した場合
Sequential sampling (4 chains in 1 job)
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `cores`を省略した場合、実行環境のCPUコア数(この検証環境では16)と`chains`の小さい方が使われ、`chains`本のチェーンがすべて別プロセスで同時に(マルチプロセスで)実行される。`cores=1`を明示すると、たとえ`chains=4`でも1プロセスずつ順番に(シーケンシャルに)実行される(ログの`Multiprocess sampling`↔`Sequential sampling`の違いで確認できる)。
- いずれの場合も結果の`idata.posterior`の形状(`chain`次元の長さ)は`chains`の値そのものであり、`cores`は計算の並列度(実行時間)にのみ影響し結果の統計的な意味には影響しない。ただしJupyter上で対話的に実行する場合、マルチプロセス実行(`cores>1`)はプラットフォームによってはpickle化の問題で失敗することがあり、その場合は`cores=1`で回避できる。

### 10-6. 事前分布設計ユーティリティ

#### `find_constrained_prior(...)`

**用途**: 「値は`lower`〜`upper`の範囲にだいたい`mass`(例: 90%)の確率で収まってほしい」という制約から、指定した分布族(例: `Gamma`)のパラメータを数値的に逆算する。事前分布の形を勘ではなくデータ的な制約から決めたい場合に使う。

**シグネチャ**: `pymc.find_constrained_prior(distribution, lower, upper, init_guess, mass=0.95, fixed_params=None, mass_below_lower=None, **kwargs)`

**使用例**:
```python
import pymc as pm
from scipy import stats

params = pm.find_constrained_prior(
    pm.Gamma, lower=1, upper=10, mass=0.9, init_guess={"alpha": 2, "beta": 0.5}
)
print(params)

alpha, beta = params["alpha"], params["beta"]
mass = stats.gamma(a=alpha, scale=1 / beta).cdf(10) - stats.gamma(a=alpha, scale=1 / beta).cdf(1)
print("mass in [1,10]:", mass)
```
実行結果:
```
<stdin>:3: FutureWarning: find_constrained_prior is deprecated and will be removed in a future version. Please use maxent function from PreliZ. https://preliz.readthedocs.io/en/latest/api_reference.html#preliz.unidimensional.maxent
{'alpha': np.float64(2.437273357641224), 'beta': np.float64(0.543787295811397)}
mass in [1,10]: 0.8999999998542045
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: PyMC 6.3.1では`find_constrained_prior`は**非推奨(`FutureWarning`)**であり、将来のバージョンで削除予定。メッセージが案内する通り、姉妹ライブラリ[PreliZ](https://preliz.readthedocs.io/)の`maxent`関数への移行が推奨されている。新規にコードを書く場合はPreliZの導入を検討した方がよい。
- 非推奨ではあるものの、`scipy.stats`で実測したとおり計算結果自体は正しく、返された`alpha`/`beta`のガンマ分布は`[1, 10]`の区間にちょうど`mass=0.9`(90%)の確率質量を持つ。`init_guess`(数値最適化の初期値)は指定した`distribution`が要求するパラメータ名の辞書で与える必要がある。
