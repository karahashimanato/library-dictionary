# ArviZ 逆引き辞書

ArviZ 1.3.0 で検証済み(0.x系からの大きな仕様変更に注意)

すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(ArviZ 1.3.0 = arviz_base 1.3.0 / arviz_stats 1.3.1 / arviz_plots 1.3.1、xarray 2026.2.0、PyMC 6.3.1、matplotlib 3.11.1、pandas 3.0.5)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。ArviZ 1.x は 0.x からの全面書き換えで、データモデルは `xarray.DataTree`、プロットは `arviz_plots` の `PlotCollection` に変わっている。シグネチャは型注釈を省いて表記している。MCMCの数値(ESSなど)は環境により末尾の桁が変動しうる。プロットはmatplotlib(Aggバックエンド)でPNGに保存して目視確認し、「目視確認」と記した箇所がその内容(画像自体はこのリポジトリに含めていない)。

## 目次

- [共通の準備](#共通の準備)
- [0.x から 1.x への主な変更点(実測の早見表)](#0x-から-1x-への主な変更点実測の早見表)
- [1. データ構造](#1-データ構造)
- [2. 入出力・他ライブラリ連携](#2-入出力他ライブラリ連携)
- [3. 要約統計・区間](#3-要約統計区間)
- [4. 収束診断](#4-収束診断)
- [5. 分布・情報量ユーティリティ](#5-分布情報量ユーティリティ)
- [6. モデル比較・予測評価](#6-モデル比較予測評価)
- [7. 可視化の基本(PlotCollection)](#7-可視化の基本plotcollection)
- [8. 可視化(事後分布・収束診断)](#8-可視化事後分布収束診断)
- [9. 可視化(事後予測・モデル比較)](#9-可視化事後予測モデル比較)
- [付録. 存在しない名前・未検証の名前](#付録-存在しない名前未検証の名前)

---

## 共通の準備

以降の使用例は、次のコードで作った変数を前提にする。`c8` は同梱の8 schools(ダイバージェンスを含む)、`dt` は自作の線形回帰(PyMCで作成、`log_likelihood`・事後/事前予測・`log_prior` グループ付き)、`dt0` は切片のみのモデル(比較用)。

```python
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
IMG = "."          # 画像の保存先ディレクトリ

c8 = az.load_arviz_data("centered_eight")
```

`dt` / `dt0` の作成(`import` と乱数シード固定込み。`cores=1` で逐次サンプリング):

```python
import numpy as np
import pymc as pm
import scipy.stats as st
import arviz as az

rng = np.random.default_rng(42)
N = 60
x = rng.normal(size=N)
y = 1.0 + 2.0 * x + rng.normal(scale=0.5, size=N)
coords = {"obs_id": np.arange(N)}

with pm.Model(coords=coords) as m_lin:                      # 線形回帰
    xd = pm.Data("x", x, dims="obs_id")
    a = pm.Normal("a", 0, 5)
    b = pm.Normal("b", 0, 5)
    sigma = pm.HalfNormal("sigma", 2)
    pm.Normal("y", a + b * xd, sigma, observed=y, dims="obs_id")
    dt = pm.sample(draws=500, tune=500, chains=4, cores=1, random_seed=1, progressbar=False)
    pm.sample_posterior_predictive(dt, extend_inferencedata=True, random_seed=2, progressbar=False)
    pp = pm.sample_prior_predictive(200, random_seed=3)
    dt["prior"] = pp["prior"]                                # DataTreeにはグループを代入で追加する
    dt["prior_predictive"] = pp["prior_predictive"]
    pm.compute_log_likelihood(dt, progressbar=False)

with pm.Model(coords=coords) as m_null:                     # 切片のみ(比較用)
    a = pm.Normal("a", 0, 5)
    sigma = pm.HalfNormal("sigma", 2)
    pm.Normal("y", a, sigma, observed=y, dims="obs_id")
    dt0 = pm.sample(draws=500, tune=500, chains=4, cores=1, random_seed=1, progressbar=False)
    pm.compute_log_likelihood(dt0, progressbar=False)

# log_prior グループは PyMC 6.3.1 に専用関数が見当たらないため手計算で追加(psense用)
P = dt.posterior
dt["log_prior"] = az.dict_to_dataset({
    "a": st.norm(0, 5).logpdf(P["a"].values),
    "b": st.norm(0, 5).logpdf(P["b"].values),
    "sigma": st.halfnorm(scale=2).logpdf(P["sigma"].values),
})
```

## 0.x から 1.x への主な変更点(実測の早見表)

各行は 1.3.0 で実際に確認した事実(詳細・出力は該当エントリ)。

| 0.x での書き方 | 1.3.0 での実際 |
|---|---|
| `az.InferenceData` | クラスは無く、`xarray.DataTree`。`az.InferenceData` にアクセスすると `MigrationWarning` を出して `DataTree` を返す |
| `idata.groups()` | `dt.groups` はメソッドではなく tuple プロパティ(`'/posterior'` 形式のパス)。有無確認は `"log_likelihood" in dt.children` |
| `idata.extend(other)` / `az.concat` | 存在しない。`dt["prior"] = other["prior"]` のようにグループ単位で代入 |
| `idata.posterior` が Dataset | 子の `DataTree`。`Dataset` は `.dataset` / `.to_dataset()` |
| `az.from_pymc(trace)` | 存在しない。`pm.sample` が最初から `DataTree` を返す |
| `az.to_netcdf(idata, path)` | 存在しない。`dt.to_netcdf(path)`(読み込みは `az.from_netcdf`) |
| `az.summary` の既定: 94% HDI(列 `hdi_3%`/`hdi_97%`) | 既定 89% ETI(列 `eti89_lb`/`eti89_ub`)。`hdi_prob=` は `TypeError`(`ci_prob`/`ci_kind` を使う) |
| `az.hdi(x, hdi_prob=0.9)` | `prob=0.9` に改名 |
| `az.waic` | 存在しない(`az.loo` / `az.compare`) |
| `az.compare` の列 `loo`, `loo_se`, `warning`, `scale` | 列は `rank / elpd_diff / dse / p_worse / diag_diff / diag_elpd / p / elpd / se / weight` |
| `az.plot_posterior` | 存在しない。`az.plot_dist` |
| `az.plot_ppc` / `az.plot_bpv` | `az.plot_ppc_dist` など(`plot_ppc_*`)。`plot_bpv` は無い |
| `az.plot_trace`(密度+トレース) | `plot_trace` はトレースのみ。密度も並べるなら `plot_trace_dist` |
| `az.plot_rank`(ヒストグラム棒) | チェーンごとの Δ-ECDF 曲線 |
| `az.plot_*` が Axes の ndarray を返す | `PlotCollection`(`plot_pair` は `PlotMatrix`)。保存は `pc.savefig(path)`。`figsize` は `figure_kwargs={"figsize": ...}` |
| `az.rhat`/`az.ess`/`az.mean`/`az.hdi` が Dataset を返す | `DataTree` を渡すと単一ノードの `DataTree` を返す(`r["a"]` で取り出し)。`Dataset` を渡すと `Dataset` |
| `az.mean` 等の値は丸められない | `round_to` 既定 `None` = rcParams `'2g'`(有効数字2桁に丸められる)。生の値は `round_to="none"` |
| `az.ecdf(idata, ...)` | `DataTree` には未対応(`NotImplementedError`)。`dt.posterior.dataset` を渡す |


## 1. データ構造

### `xarray.DataTree`(旧 `InferenceData`)

**用途**: 1.x では `az.InferenceData` クラスは無くなり、サンプリング結果は `xarray.DataTree`(グループ=ノードの木構造)で表現される。`posterior` / `sample_stats` / `posterior_predictive` / `log_likelihood` / `observed_data` などが子ノード(グループ)。

**シグネチャ**: `xarray.DataTree(dataset=None, children=None, name=None)`

**使用例**:
```python
print(type(dt))
print(dt.groups)
print(list(dt.children))
print(type(dt.posterior).__name__, type(dt["posterior"]).__name__)
print(type(dt.posterior.dataset).__name__, type(dt.posterior.to_dataset()).__name__)
print(dt.posterior["a"].dims, dt.posterior["a"].shape)
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    T = az.InferenceData
print(T, w[0].category.__name__)
```
実行結果:
```
<class 'xarray.core.datatree.DataTree'>
('/', '/posterior', '/sample_stats', '/observed_data', '/constant_data', '/posterior_predictive', '/prior', '/prior_predictive', '/log_likelihood', '/log_prior')
['posterior', 'sample_stats', 'observed_data', 'constant_data', 'posterior_predictive', 'prior', 'prior_predictive', 'log_likelihood', 'log_prior']
DataTree DataTree
DatasetView Dataset
('chain', 'draw') (4, 500)
<class 'xarray.core.datatree.DataTree'> MigrationWarning
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: `az.InferenceData` にアクセスすると `MigrationWarning` が出て `xarray.DataTree` が返るだけで、0.x の InferenceData クラスとしては使えない。`isinstance(x, az.InferenceData)` は `isinstance(x, xr.DataTree)` に置き換える。
- `dt.groups` は**メソッドではなくプロパティ(tuple)**で、ルート `'/'` と `'/posterior'` のようなパス文字列を含む。0.x の `idata.groups()`(呼び出し)は `TypeError: 'tuple' object is not callable` になる。グループの有無は `"log_likelihood" in dt.children`(または `in dt`)で確認する。
- `dt.posterior` や `dt["posterior"]` は `Dataset` ではなく子 `DataTree`。`Dataset` が必要なときは `.dataset`(読み取り専用ビュー)か `.to_dataset()`(コピー)を使う。変数へのアクセス `dt.posterior["a"]` はそのまま `DataArray` を返す。
- ルートで `dt.sel(chain=[0, 1])` のように選択すると、`prior` グループ(chain数が1)など全ノードに適用しようとして `KeyError: "not all values found in index 'chain'"` になる。グループ単位で `dt.posterior.sel(chain=[0, 1])` とすること。

### `arviz.load_arviz_data(...)` / `arviz.list_datasets()`

**用途**: ArviZ同梱のサンプルデータセット(8 schools、ロジスティック回帰など)を `DataTree` として読み込む。初回はリモートからダウンロードされキャッシュされる。

**シグネチャ**: `load_arviz_data(dataset=None, data_home=None, **kwargs)` / `list_datasets()`

**使用例**:
```python
lines = az.list_datasets().splitlines()
names = [l for l, nxt in zip(lines, lines[1:]) if nxt and set(nxt) == {"="}]
print(type(az.list_datasets()).__name__, len(names))
print(names)
print(list(c8.children))
print(dict(c8.posterior.sizes))
```
実行結果:
```
str 14
['centered_eight', 'non_centered_eight', 'radon', 'rugby', 'rugby_field', 'glycan_torsion_angles', 'crabs_poisson', 'crabs_hurdle_nb', 'sbc', 'anes', 'periwinkles', 'censored_cats', 'roaches_nb', 'roaches_zinb']
['posterior', 'posterior_predictive', 'log_likelihood', 'sample_stats', 'prior', 'prior_predictive', 'observed_data', 'constant_data']
{'chain': 4, 'draw': 500, 'school': 8}
```

**注意点・落とし穴**:
- `list_datasets()` は名前のリストではなく、各データセットの説明を連結した**文字列**を返す(上の例では下線行`====`を手掛かりに名前を抽出している)。
- `centered_eight` は0.20.0/PyMC 5.20 で作られた保存データで、ダイバージェンスを含む(収束診断・`plot_pair`のダイバージェンス表示のデモ向き)。属性 `arviz_version` は保存時のバージョンであり、1.3.0で読み込んでも `DataTree` になる。

### `arviz.from_dict(...)`

**用途**: `{グループ名: {変数名: ndarray}}` のネストした辞書から `DataTree` を作る。ArviZの規約(先頭軸が `chain`, `draw`)に沿って次元・座標を付与する。

**シグネチャ**: `from_dict(data, *, name=None, sample_dims=None, save_warmup=None, index_origin=None, coords=None, dims=None, pred_dims=None, pred_coords=None, check_conventions=True, attrs=None)`

**使用例**:
```python
rng = np.random.default_rng(0)
d = {"posterior": {"mu": rng.normal(size=(2, 100)),
                   "theta": rng.normal(size=(2, 100, 3))},
     "observed_data": {"y": rng.normal(size=3)}}
t = az.from_dict(d, dims={"theta": ["group"], "y": ["group"]},
                 coords={"group": ["a", "b", "c"]})
print(list(t.children))
print(dict(t.posterior.sizes))
print(t.posterior.dataset["theta"].dims)
print(t["observed_data"].attrs["sample_dims"], t["posterior"].attrs["sample_dims"])
try:
    az.from_dict({"posterior": {"mu": rng.normal(size=100)}})
except ValueError as e:
    print("ValueError:", str(e)[:90])
```
実行結果:
```
['posterior', 'observed_data']
{'chain': 2, 'draw': 100, 'group': 3}
('chain', 'draw', 'group')
[] ['chain', 'draw']
ValueError: In variable mu, there are more dims (2) given than existing ones (1). dims and shape shoul
```

**注意点・落とし穴**:
- 1次元配列(chain軸なし)を渡すと `ValueError: In variable mu, there are more dims (2) given than existing ones (1)` になる。`x[None, :]` のように `(chain, draw, ...)` の形にして渡す。
- グループ名に `data` を含むもの(`observed_data`/`constant_data`)は `sample_dims=[]` として扱われ、chain/drawが付かない。`log_likelihood` を含むグループはイベント次元をスキップする。`predictions` を含むグループは `pred_dims`/`pred_coords` が使われる。
- `dims` は「変数名→次元名リスト」で、sample次元(chain, draw)は含めない。`coords` の長さとデータの形が合わないと `CoordinateValidationError: conflicting sizes for dimension 'g': length 3 on the data but length 2 on coordinate 'g'` になる。

### `arviz.convert_to_datatree(...)` / `arviz.convert_to_dataset(...)` / `arviz.dict_to_dataset(...)`

**用途**: 各種オブジェクト(Dataset・ndarray・辞書・DataTree)を ArviZ 規約の `DataTree` / `Dataset` に変換する。`dict_to_dataset` は辞書1グループ分を `Dataset` にする下位関数。

**シグネチャ**: `convert_to_datatree(obj, **kwargs)` / `convert_to_dataset(obj, *, group='posterior', **kwargs)` / `dict_to_dataset(data, *, attrs=None, inference_library=None, coords=None, dims=None, sample_dims=None, index_origin=None, skip_event_dims=False, check_conventions=True)`

**使用例**:
```python
ds = az.dict_to_dataset({"mu": np.random.default_rng(0).normal(size=(2, 50))})
print(type(ds).__name__, dict(ds.sizes))

tree = az.convert_to_datatree(ds, group="posterior")
print(type(tree).__name__, list(tree.children))

arr_tree = az.convert_to_datatree(np.zeros((2, 50, 3)), group="posterior")
print(list(arr_tree.posterior.dataset.data_vars), arr_tree.posterior.dataset["x"].dims)

print(az.convert_to_datatree(dt) is dt)                 # 冪等
sub = az.convert_to_dataset(dt, group="posterior")
print(type(sub).__name__, list(sub.data_vars))
```
実行結果:
```
Dataset {'chain': 2, 'draw': 50}
DataTree ['posterior']
['x'] ('chain', 'draw', 'x_dim_0')
True
Dataset ['a', 'b', 'sigma']
```

**注意点・落とし穴**:
- `convert_to_datatree` は `DataTree` に対しては同じオブジェクトをそのまま返す(冪等)。`ndarray` を渡すと変数名は `x`、追加次元名は `x_dim_0` のように自動命名される。
- `convert_to_dataset(dt, group=...)` は指定グループを素の `xarray.Dataset`(`DatasetView`ではない)で返す。

### DataTreeのグループ操作(追加・削除・部分抽出)

**用途**: 0.x の `idata.extend()` / `az.concat()` / `idata.sel()` に相当する操作は、1.x では `DataTree` のインデックス代入・`drop_nodes`・`DataTree.from_dict` などxarrayのAPIで行う。

**シグネチャ**: `dt["group"] = Dataset` / `DataTree.drop_nodes(names, *, errors='raise')` / `DataTree.from_dict(data=None, coords=None, *, name=None, nested=False)` / `DataTree.copy(*, inherit=True, deep=False)`

**使用例**:
```python
new = dt.copy()
new["extra"] = az.dict_to_dataset({"z": np.zeros((4, 10))})     # グループ追加
print(list(new.children)[-2:])
del new["extra"]                                                  # グループ削除
print("extra" in new)

light = dt.drop_nodes(["prior", "prior_predictive"])              # 不要グループを除く
print(list(light.children))

keep = xr.DataTree.from_dict({g: dt[g].to_dataset() for g in ["posterior", "sample_stats"]})
print(list(keep.children))
print(hasattr(dt, "extend"), "concat" in dir(az))
```
実行結果:
```
['log_prior', 'extra']
False
['posterior', 'sample_stats', 'observed_data', 'constant_data', 'posterior_predictive', 'log_likelihood', 'log_prior']
['posterior', 'sample_stats']
False False
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: `dt.extend(...)` は `AttributeError: 'DataTree' object has no attribute 'extend'`、`az.concat` も存在しない。PyMCの `pm.sample_prior_predictive` の結果を既存のツリーに足すときは `dt["prior"] = pp["prior"]` のようにグループごとに代入する。
- `dt.copy()` は既定で**浅いコピー**(`deep=False`。グループ構造は複製されるが配列のメモリは元と共有)。グループの足し引きだけなら十分だが、値を書き換えるなら `deep=True` にする。

### `arviz.extract(...)`

**用途**: グループから変数を取り出し、`chain`/`draw` を1本の `sample` 次元にまとめる(または上限数までランダム抽出する)。プロットや散布図用の"平らな"サンプル取得に便利。

**シグネチャ**: `extract(data, group='posterior', sample_dims=None, *, combined=True, var_names=None, filter_vars=None, num_samples=None, weights=None, resampling_method=None, keep_dataset=False, random_seed=None)`

**使用例**:
```python
e = az.extract(dt, var_names=["a", "b"])
print(type(e).__name__, dict(e.sizes))
print(type(az.extract(dt, var_names="a")).__name__, az.extract(dt, var_names="a").shape)   # 変数1つ→DataArray
print(az.extract(dt, var_names="a", combined=False).dims)
print(az.extract(dt, var_names=["a", "b"], num_samples=50, random_seed=0).sizes["sample"])
print(list(az.extract(dt, var_names=["~sigma"]).data_vars))
pp = az.extract(dt, group="posterior_predictive", var_names="y", num_samples=10, random_seed=0)
print(pp.dims, pp.shape)
```
実行結果:
```
Dataset {'sample': 2000}
DataArray (2000,)
('chain', 'draw')
50
['a', 'b']
('obs_id', 'sample') (60, 10)
```

**注意点・落とし穴**:
- `var_names` を**文字列1つ**にすると `DataArray`、リストにすると `Dataset` が返る(リスト1要素でも Dataset)。常に Dataset がほしいときは `keep_dataset=True`。
- `combined=True`(デフォルト)で作られる `sample` 次元は `chain`/`draw` の MultiIndex 座標を持つ。`combined=False` なら元の `(chain, draw)` のまま返る。
- `var_names=["~sigma"]` のように `~` を前置すると除外指定になる。

### `arviz.thin(...)`

**用途**: `draw` 次元を間引く(自己相関の高いサンプルを薄める)。戻り値は**指定グループだけ**を持つ `DataTree`。

**シグネチャ**: `thin(data, sample_dims='draw', group='posterior', var_names=None, filter_vars=None, coords=None, factor='auto', chain_axis=0, draw_axis=1)`

**使用例**:
```python
th = az.thin(dt, factor=5)
print(type(th).__name__, th.name, dict(th.sizes))
print(th["a"].draw.values[:5])
print(dict(az.thin(dt, factor="auto").sizes))
```
実行結果:
```
DataTree posterior {'chain': 4, 'draw': 100}
[ 0  5 10 15 20]
{'chain': 4, 'draw': 250}
```

**注意点・落とし穴**:
- `factor='auto'`(デフォルト)は bulk/tail ESS から間引き幅を決める(Säilynoja et al. 2022 の方法)。この例では2になり、500→250 draw に減った。
- 戻り値は元の `dt` 全体ではなく、指定した `group`(既定 `posterior`)だけの単一ノードの `DataTree`。他のグループ(`log_likelihood` など)は含まれない。

### `arviz.get_log_likelihood(...)`

**用途**: `log_likelihood` グループから対数尤度の `DataArray` を取り出す(`loo`/`compare` 内部でも使われる)。

**シグネチャ**: `get_log_likelihood(idata, var_name=None)`

**使用例**:
```python
ll = az.get_log_likelihood(dt)
print(type(ll).__name__, ll.dims, ll.shape)
```
実行結果:
```
DataArray ('chain', 'draw', 'obs_id') (4, 500, 60)
```

**注意点・落とし穴**:
- `log_likelihood` グループに変数が複数あると `TypeError: Found several log likelihood arrays [...], var_name cannot be None`、グループが無いと `TypeError: log likelihood not found in inference data object` になる(いずれも実測)。前者は `var_name=` を指定し、後者はPyMCなら先に `pm.compute_log_likelihood(dt)` を呼ぶ。

### `arviz.dataset_to_dataframe(...)` / `arviz.dataset_to_dataarray(...)`

**用途**: `Dataset` を pandas の `DataFrame`、または変数を1本の `label` 次元に積んだ `DataArray` に変換する。

**シグネチャ**: `dataset_to_dataframe(ds, sample_dims=None, labeller=None, multiindex=False, new_dim='label')` / `dataset_to_dataarray(ds, sample_dims=None, labeller=None, add_coords=True, new_dim='label', label_type='flat')`

**使用例**:
```python
df = az.dataset_to_dataframe(dt.posterior.dataset[["a", "b"]])
print(df.shape); print(df.head(3))
print(df.index.names)
da = az.dataset_to_dataarray(dt.posterior.dataset[["a", "b"]])
print(da.dims, da.shape)
wide = az.dataset_to_dataframe(dt.posterior_predictive.dataset.isel(obs_id=slice(0, 2)))
print(wide.columns.tolist())
```
実行結果:
```
(2000, 2)
               a         b
(0, 0)  0.801092  2.057571
(0, 1)  0.879191  2.033461
(0, 2)  0.944475  1.905849
[None]
('chain', 'draw', 'label') (4, 500, 2)
['y[0]', 'y[1]']
```

**注意点・落とし穴**:
- 既定(`multiindex=False`)では行ラベルは `(chain, draw)` の**タプル**でインデックス名は `None`(`multiindex=True` にすると名前付きMultiIndex)。ベクトル変数の列名は `y[0]`, `y[1]` のように座標を含むラベルになる。
- 引数は `Dataset` を想定する。`DataTree` のグループは `dt.posterior.dataset`(または `.to_dataset()`)を渡す。

---

## 2. 入出力・他ライブラリ連携

### `DataTree.to_netcdf(...)` / `arviz.from_netcdf(...)`

**用途**: サンプリング結果を NetCDF ファイルに保存・復元する。0.x の `az.to_netcdf(idata, path)` / `idata.to_netcdf(path)` に相当し、1.x では保存は `DataTree` のメソッド、読み込みが `az.from_netcdf`。

**シグネチャ**: `DataTree.to_netcdf(filepath=None, mode='w', encoding=None, unlimited_dims=None, format=None, engine=None, group=None, write_inherited_coords=False, compute=True, **kwargs)` / `from_netcdf(filename_or_obj, *, engine=None, chunks=None, cache=None, decode_cf=None, mask_and_scale=None, decode_times=None, decode_timedelta=None, use_cftime=None, concat_characters=None, decode_coords=None, drop_variables=None, create_default_indexes=True, inline_array=False, chunked_array_type=None, from_array_kwargs=None, backend_kwargs=None, **kwargs)`

**使用例**:
```python
import os, tempfile
path = os.path.join(tempfile.mkdtemp(), "dt.nc")
dt.to_netcdf(path)
dt2 = az.from_netcdf(path)
print(type(dt2).__name__, list(dt2.children))
print(np.array_equal(dt2.posterior["a"].values, dt.posterior["a"].values))
print(az.summary(dt2, var_names=["a"], kind="stats").index.tolist())
dt2.close()
print(hasattr(az, "to_netcdf"))
```
実行結果:
```
DataTree ['posterior', 'sample_stats', 'observed_data', 'constant_data', 'posterior_predictive', 'prior', 'prior_predictive', 'log_likelihood', 'log_prior']
True
['a']
False
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: `az.to_netcdf` は存在しない(`AttributeError`)。保存は `dt.to_netcdf(path)`、`az.from_netcdf` の戻り値も `DataTree`。
- `from_netcdf` は遅延読み込みでファイルを開いたままにする(変数の repr が `[2000 values with dtype=float64]` のようになる)。開いたままの同名ファイルへ `to_netcdf` で上書きすると `OSError: ... unable to truncate a file which is already open` になる(実測)ので、先に `dt2.close()` するか `dt2.load()` でメモリに載せる。
- このvenvでは `netCDF4` は未インストールで `h5netcdf` が使われている(保存・読み込みとも動作確認済み)。

### `arviz.from_zarr(...)`

**用途**: Zarr ストア形式で保存された結果を `DataTree` として読み込む(書き出しは `DataTree.to_zarr`)。

**シグネチャ**: `from_zarr(filename_or_obj, *, engine='zarr', chunks=None, cache=None, decode_cf=None, mask_and_scale=None, decode_times=None, decode_timedelta=None, use_cftime=None, concat_characters=None, decode_coords=None, drop_variables=None, create_default_indexes=True, inline_array=False, chunked_array_type=None, from_array_kwargs=None, backend_kwargs=None, **kwargs)`

**使用例**:
```python
try:
    az.from_zarr("does_not_matter.zarr")
except ImportError as e:
    print("ImportError:", str(e)[:70])
```
実行結果:
```
ImportError: The zarr package is required for working with Zarr stores but could no
```

**注意点・落とし穴**:
- このvenvには `zarr` が入っておらず、上のように `ImportError`(`The zarr package is required for working with Zarr stores...`)になる。実際の読み書きはこの環境では**未検証**(`pip install zarr` が必要)。

### `arviz.from_numpyro(...)` / `arviz.from_emcee(...)` / `arviz.from_cmdstanpy(...)`

**用途**: 他の確率的プログラミングライブラリ(NumPyro・emcee・CmdStanPy)の結果を `DataTree` に変換する。PyMC の `pm.sample` は最初から `DataTree` を返すので変換不要(0.x の `az.from_pymc` は存在しない)。

**シグネチャ**: `from_numpyro(posterior=None, *, prior=None, posterior_predictive=None, predictions=None, constant_data=None, predictions_constant_data=None, log_likelihood=False, index_origin=None, coords=None, dims=None, pred_dims=None, extra_event_dims=None, sample_dims=None, num_chains=None)` / `from_emcee(sampler=None, var_names=None, slices=None, arg_names=None, arg_groups=None, blob_names=None, blob_groups=None, index_origin=None, coords=None, dims=None, check_conventions=True)` / `from_cmdstanpy(posterior=None, *, posterior_predictive=None, predictions=None, prior=None, prior_predictive=None, observed_data=None, constant_data=None, predictions_constant_data=None, log_likelihood=None, index_origin=None, coords=None, dims=None, save_warmup=None, dtypes=None)`

**使用例**:
```python
import jax, numpyro, numpyro.distributions as ndist
from numpyro.infer import MCMC, NUTS

y_obs = np.random.default_rng(0).normal(1.0, 1.0, size=30)
def model(y=None):
    mu = numpyro.sample("mu", ndist.Normal(0, 5))
    s = numpyro.sample("sigma", ndist.HalfNormal(2))
    numpyro.sample("y", ndist.Normal(mu, s), obs=y)

mcmc = MCMC(NUTS(model), num_warmup=200, num_samples=200, num_chains=2, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0), y=y_obs)
t = az.from_numpyro(mcmc)
print(type(t).__name__, list(t.children))
print(dict(t.posterior.sizes))
print(t.posterior.attrs["inference_library"], "from_pymc" in dir(az))
```
実行結果:
```
DataTree ['posterior', 'sample_stats', 'observed_data']
{'chain': 2, 'draw': 200}
numpyro False
```

**注意点・落とし穴**:
- 実行したのは `from_numpyro` のみ。`from_emcee`(`emcee` 未インストール)・`from_cmdstanpy`(`cmdstanpy` 未インストール)はシグネチャの確認だけで、変換自体はこの環境では**未検証**。
- `from_numpyro` は既定で `posterior` / `sample_stats` / `observed_data` を作り、`log_likelihood` は作らない。`log_likelihood=True`(既定は `False`)を渡すと `['posterior', 'sample_stats', 'log_likelihood', 'observed_data']` になる(実測)ので、`az.loo` に使うなら指定する。

---

## 3. 要約統計・区間

### `arviz.summary(...)`

**用途**: 事後分布の平均・標準偏差・信用区間・収束診断(ESS, r_hat, MCSE)をまとめた表を作る。

**シグネチャ**: `summary(data, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, kind='all', fmt='wide', ci_prob=None, ci_kind=None, round_to='auto', skipna=False)`

**使用例**:
```python
print(az.summary(dt, var_names=["a", "b", "sigma"]))
print(az.summary(dt, var_names=["a", "b"], kind="stats"))
print(az.summary(dt, var_names=["a", "b"], kind="diagnostics"))
print(az.summary(dt, var_names=["a", "b"], ci_kind="hdi", ci_prob=0.94, round_to=3))
print(az.summary(dt, var_names=["a", "b"], kind="stats_median"))
s = az.summary(dt, var_names=["a"])
print(type(s).__name__, s.loc["a", "mean"], s["mean"].dtype)          # 表示は0.909でも内部は生の値
print(az.summary(dt, var_names=["a"], round_to="none").loc["a", "mean"])
print(az.summary(dt, var_names=["a"], round_to=None).loc["a", "mean"])    # rcParamsの2gで丸め
```
実行結果:
```
        mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
a      0.909  0.052     0.83     0.99     2465     1565  1.00    0.0011  0.00077
b      1.962  0.065      1.9      2.1     2567     1810  1.00    0.0013  0.00095
sigma  0.397  0.039     0.34     0.46     2598     1671  1.00   0.00078  0.00064
   mean     sd eti89_lb eti89_ub
a  0.91  0.052     0.83     0.99
b     2  0.065      1.9      2.1
  ess_bulk ess_tail r_hat mcse_mean  mcse_sd
a     2465     1565  1.00    0.0011  0.00077
b     2567     1810  1.00    0.0013  0.00095
    mean     sd  hdi94_lb  hdi94_ub  ess_bulk  ess_tail  r_hat  mcse_mean  mcse_sd
a  0.909  0.052     0.809     1.008  2465.333  1565.996  1.000      0.001    0.001
b  1.962  0.065     1.836     2.088  2567.009  1810.526  0.999      0.001    0.001
  median    mad eti89_lb eti89_ub
a   0.91  0.034     0.83     0.99
b      2  0.043      1.9      2.1
SummaryDataFrame 0.9090007071629594 float64
0.9090007071629594
0.91
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: 既定は `ci_kind='eti'`・`ci_prob=0.89` で、列名は `eti89_lb` / `eti89_ub`(`hdi_3%`/`hdi_97%` ではない)。0.x の引数 `hdi_prob=` は削除されており `TypeError: summary() got an unexpected keyword argument 'hdi_prob'` になる。94%HDIは `ci_kind='hdi', ci_prob=0.94`(列名は `hdi94_lb`/`hdi94_ub`)。
- `kind` は `'all'`(既定) / `'stats'` / `'diagnostics'` / `'all_median'` / `'stats_median'` / `'diagnostics_median'` / `'mc_diagnostics'`。`*_median` 系は平均・SDの代わりに中央値・MADを出し、区間は常にETIに固定される(docstringの記述)。
- `round_to='auto'`(既定)の戻り値は `SummaryDataFrame`(pandas.DataFrameのサブクラス)で、**表示だけ**をMCSEに応じた桁(ESSは整数に切り捨て、r_hatは小数2桁など)に整形している。`s.loc['a', 'mean']` は表示が `0.909` でも内部の値は生の `0.9090007...`(dtypeも float64)であることを上の例で確認できる。
- 見た目通りに丸めた数値がほしければ `round_to=整数`、生の値を確実に得たければ `round_to='none'`(文字列)を渡す。`round_to=None`(Noneオブジェクト)は rcParams の `stats.round_to`('2g')で**値そのものが丸められる**(上の例の最終行が `0.91`)。docstringによると、表示整形は行の絞り込み・スライス・転置では保たれるが、構造を変える操作では失われる。

### `arviz.summary(...)` の `fmt` と `group` / `filter_vars`

**用途**: 出力形式(`wide`/`long`/`xarray`)、対象グループ、変数名フィルタを切り替える。ベクトル変数は `theta[Choate]` のように座標付きの行になる。

**シグネチャ**: `az.summary(data, var_names=None, filter_vars=None, group='posterior', coords=None, fmt='wide', ...)`

**使用例**:
```python
print(az.summary(c8, var_names=["theta"], kind="stats").head(3))
print(az.summary(c8, var_names=["th"], filter_vars="like", kind="stats").shape)
print(az.summary(c8, var_names=["mu", "tau"], kind="stats", fmt="long"))
print(az.summary(c8, var_names=["mu", "tau"], kind="stats", fmt="xarray"))
print(az.summary(dt, var_names=["y"], group="posterior_predictive", kind="stats").head(2))
```
実行結果:
```
                        mean   sd eti89_lb eti89_ub
theta[Choate]            6.4  5.9     -1.7       17
theta[Deerfield]           5  4.9       -3       12
theta[Phillips Andover]  3.4  5.4     -5.7       11
(8, 4)
            mu  tau
mean       4.2  4.3
sd         3.3    3
eti89_lb  -1.2  1.3
eti89_ub   9.2   10
<xarray.Dataset> Size: 96B
Dimensions:  (summary: 4)
Coordinates:
  * summary  (summary) object 32B 'mean' 'sd' 'eti89_lb' 'eti89_ub'
Data variables:
    mu       (summary) float64 32B 4.2 3.3 -1.2 9.2
    tau      (summary) float64 32B 4.3 3.0 1.3 10.0
      mean    sd eti89_lb eti89_ub
y[0]   1.5   0.4     0.85      2.1
y[1]  -1.1  0.41     -1.8    -0.49
```

**注意点・落とし穴**:
- `filter_vars='like'` は部分一致、`'regex'` は正規表現で `var_names` を解釈する(`var_names=["~^t"], filter_vars="regex"` のように `~` で除外も可)。
- `fmt='xarray'` は `xarray.Dataset`(次元 `summary`)を返し、このときの丸めは rcParams の `stats.round_to` に従う。`fmt='long'` は統計量が行・変数が列の転置形(実測)。

### `arviz.mean` / `median` / `mode` / `std` / `mad` / `iqr`

**用途**: 事後サンプルの点推定・散らばりを、`chain`/`draw` をまとめて計算する。次元名を `dim` で指定すればその次元だけ潰せる。

**シグネチャ**: `mean(data, dim=None, group='posterior', var_names=None, filter_vars=None, coords=None, round_to=None, skipna=False, **kwargs)` / `iqr(data, dim=None, group='posterior', var_names=None, filter_vars=None, coords=None, quantiles=(0.25, 0.75), round_to=None, skipna=False, **kwargs)`

**使用例**:
```python
for f in [az.mean, az.median, az.mode, az.std, az.mad, az.iqr]:
    r = f(c8, var_names=["mu", "tau"], round_to="none")
    print(f.__name__.ljust(6), {k: round(float(v), 3) for k, v in r.dataset.data_vars.items()})
r = az.mean(dt, var_names=["a"])
print(type(r).__name__, r["a"].item())
print(az.mean(dt, var_names=["a"], round_to="none")["a"].item())
print(az.mean(dt.posterior.dataset, var_names=["a"], round_to="none").__class__.__name__)
print(az.mean(c8, var_names=["theta"], dim="draw", round_to=3)["theta"].dims)
```
実行結果:
```
mean   {'mu': 4.171, 'tau': 4.321}
median {'mu': 4.063, 'tau': 3.511}
mode   {'mu': 3.485, 'tau': 2.241}
std    {'mu': 3.273, 'tau': 2.951}
mad    {'mu': 2.265, 'tau': 1.567}
iqr    {'mu': 4.539, 'tau': 3.478}
DataTree 0.91
0.9090007071629594
Dataset
('chain', 'school')
```

**注意点・落とし穴**:
- **注意**: `round_to` の既定は `None` = rcParams の `stats.round_to`(`'2g'`)なので、**既定のままだと有効数字2桁に丸められた値が返る**(上の例で平均 0.909... が `0.91`)。精度が必要なら `round_to='none'` か整数を指定する。
- `DataTree` を渡すと `DataTree`(単一ノード。変数へは `r["a"]`、`Dataset` は `r.dataset`)、`Dataset` を渡すと `Dataset` が返る。`mode` はKDEに基づく最頻値。
- `dim` を指定するとそれ以外の次元は残る(上の最終行は `chain` と `school` が残る)。`dim=None` は `rcParams['data.sample_dims']`(`chain`, `draw`)全体を潰す。

### `arviz.hdi(...)` / `arviz.eti(...)`

**用途**: 確信区間を計算する。`hdi` は最高密度区間、`eti` は等尾区間(両側から等確率を除く)。

**シグネチャ**: `hdi(data, prob=None, dim=None, group='posterior', var_names=None, filter_vars=None, coords=None, method='nearest', circular=False, max_modes=10, skipna=False, **kwargs)` / `eti(data, prob=None, dim=None, group='posterior', var_names=None, filter_vars=None, coords=None, method='linear', skipna=False, **kwargs)`

**使用例**:
```python
h = az.hdi(dt, var_names=["a", "b"])
print(type(h).__name__)
print(h.dataset)
print(az.hdi(dt, var_names=["a"], prob=0.94)["a"].values)
print(az.eti(dt, var_names=["a", "b"], prob=0.9)["a"].values)
th = az.hdi(c8, var_names=["theta"], prob=0.9)["theta"]
print(th.dims, th.coords["ci_bound"].values)
```
実行結果:
```
DataTree
<xarray.DatasetView> Size: 72B
Dimensions:   (ci_bound: 2)
Coordinates:
  * ci_bound  (ci_bound) <U5 40B 'lower' 'upper'
Data variables:
    a         (ci_bound) float64 16B 0.8274 0.9933
    b         (ci_bound) float64 16B 1.865 2.07
[0.81777163 1.01354303]
[0.82212261 0.99431538]
('school', 'ci_bound') ['lower' 'upper']
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: 確率の引数名は `hdi_prob` ではなく `prob`。`az.hdi(dt, hdi_prob=0.9)` は `TypeError: hdi got an unexpected keyword argument: 'hdi_prob'. The 'hdi_prob' argument was renamed to 'prob' in arviz-stats 1.0.` と、置換案内つきで失敗する。
- `prob` 省略時は rcParams の `stats.ci_prob`(**0.89**、94%ではない)。結果には長さ2の `ci_bound` 次元(`'lower'`/`'upper'`)が付く。
- `method` は `'nearest'`(既定・単一区間) / `'multimodal'` / `'multimodal_sample'`。2山分布(平均-3と+3のガウス混合)で試すと、`nearest` は `[-3.72, 3.57]` の1区間、`multimodal` は `m_mode` 次元が付いて `[[-3.84, -2.17], [2.17, 3.85]]` の2区間を返した(実測)。`max_modes`(既定10)はmultimodal時の最大山数。

### `arviz.ci_in_rope(...)`

**用途**: 確信区間(CI)内のサンプルのうち、ROPE(実質的同値領域)に入っている割合(%)を返す。

**シグネチャ**: `ci_in_rope(data, rope, var_names=None, filter_vars=None, group='posterior', dim=None, ci_prob=None, ci_kind=None, rope_dim='rope_dim')`

**使用例**:
```python
print(float(az.ci_in_rope(dt, rope=(1.9, 2.1), var_names=["b"], ci_kind="eti")["b"]))
print(float(az.ci_in_rope(dt, rope=(1.9, 2.1), var_names=["b"], ci_kind="hdi")["b"]))
print(float(az.ci_in_rope(dt, rope=(1.9, 2.1), var_names=["b"])["b"]))     # ci_kind省略
r = az.ci_in_rope(dt, rope={"a": (0.0, 1.0), "b": (1.9, 2.1)}, ci_prob=0.94, ci_kind="hdi")
print({k: round(float(v), 2) for k, v in r.data_vars.items()})
```
実行結果:
```
87.86516853932585
88.8265019651881
88.8265019651881
{'a': 97.98, 'b': 84.8}
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: docstringには `ci_kind` の既定は `rcParams['stats.ci_kind']`(`'eti'`)と書かれているが、実際に `ci_kind` を省略すると**HDIで計算された値**(`ci_kind='hdi'` と同じ 88.8...)になり、`'eti'` を明示した場合(87.9)と異なる。`plot_dist(..., rope=...)` の注釈(ETI基準)と数値を合わせたいときは `ci_kind='eti'` を明示する。
- 戻り値は `xarray.Dataset`。`rope` は全変数共通の `(下限, 上限)`、または `{変数名: (下限, 上限)}` の辞書。値は 0〜100 の**パーセント**。区間内サンプルに対する割合であり、事後分布全体に占める割合ではない。全体に占める割合が欲しければ `ci_prob=1` を渡す(この例では 82.0 で、`np.mean((b>=1.9)&(b<=2.1))*100` と一致することを確認した)。

### `arviz.rcParams` / `arviz.rc_context(...)`

**用途**: ArviZの既定値(信用区間の種類・確率、サンプル次元、プロットバックエンド、丸め桁など)を参照・変更する。`rc_context` は一時的な変更。

**シグネチャ**: `rc_context(rc=None, fname=None)`

**使用例**:
```python
print(az.rcParams["stats.ci_kind"], az.rcParams["stats.ci_prob"], az.rcParams["stats.round_to"])
print(az.rcParams["data.sample_dims"], az.rcParams["plot.backend"], az.rcParams["stats.ic_compare_method"])
with az.rc_context({"stats.ci_kind": "hdi", "stats.ci_prob": 0.94}):
    print(az.summary(dt, var_names=["a"], kind="stats"))
print(az.summary(dt, var_names=["a"], kind="stats"))
try:
    az.rcParams["bogus.key"] = 1
except KeyError as e:
    print("KeyError:", str(e)[:60])
print(sorted(az.rcParams.keys()))
```
実行結果:
```
eti 0.89 2g
('chain', 'draw') matplotlib stacking
   mean     sd hdi94_lb hdi94_ub
a  0.91  0.052     0.81        1
   mean     sd eti89_lb eti89_ub
a  0.91  0.052     0.83     0.99
KeyError: 'bogus.key is not a valid rc parameter (see rcParams.keys() 
['data.http_protocol', 'data.index_origin', 'data.sample_dims', 'data.save_warmup', 'plot.backend', 'plot.density_kind', 'plot.max_subplots', 'stats.ci_kind', 'stats.ci_prob', 'stats.envelope_prob', 'stats.ic_compare_method', 'stats.ic_pointwise', 'stats.ic_scale', 'stats.module', 'stats.point_estimate', 'stats.round_to']
```

**注意点・落とし穴**:
- `az.rcParams` は `RcParams`(辞書風)オブジェクトで、`with az.rc_context({...}):` の内側だけ値が変わる。上の例のように `summary` の区間種別・確率が既定で切り替わる。
- 存在しないキーへの代入は `KeyError: ... is not a valid rc parameter` になる(タイプミス防止)。`plot.max_subplots`(既定40)を超えるファセット数のプロットは `ValueError` になる(`plot_*` 節参照)。

---

## 4. 収束診断

### `arviz.rhat(...)`

**用途**: 複数チェーンの収束を測る R-hat(潜在的尺度縮小因子)。既定は rank 正規化した split R-hat。

**シグネチャ**: `rhat(data, sample_dims=None, group='posterior', var_names=None, filter_vars=None, coords=None, method='rank', chain_axis=0, draw_axis=1)`

**使用例**:
```python
for m in ["rank", "split", "folded", "z_scale", "identity"]:
    print(m.ljust(8), round(az.rhat(dt, var_names=["a"], method=m)["a"].item(), 5))
r = az.rhat(c8, var_names=["theta"])
print(type(r).__name__, r.name, list(r.children))
print(r["theta"].dims, r["theta"].values.round(3))
print(r.parent["posterior"]["theta"].shape, r.dataset["theta"].shape)
```
実行結果:
```
rank     1.00033
split    0.9996
folded   1.00033
z_scale  0.99957
identity 0.99933
DataTree posterior []
('school',) [1.007 1.011 1.01  1.01  1.019 1.012 1.012 1.012]
(8,) (8,)
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: 戻り値は素の `Dataset` ではなく、`posterior` グループ名を持つ単一ノードの `DataTree`(親に無名のルートがある)。値は `r["a"]`(または `r.dataset["a"]` / `r.to_dataset()["a"]`)で取り出す。`r["posterior"]["a"]` は自ノード配下に `posterior` が無いので `KeyError: 'Could not find node at posterior'` になる(ルート経由の `r.parent["posterior"]["a"]` なら動く)。
- `method` は `'rank'`(既定) / `'split'` / `'folded'` / `'z_scale'` / `'identity'`。r_hat > 1.01 を収束不十分の目安とする(`az.diagnose` の既定しきい値も 1.01)。
- チェーンが1本だけのデータ(`dt.posterior.sel(chain=[0])`)では、警告なしに `nan` が返る(実測)。収束判定には2本以上(推奨4本)のチェーンが必要。

### `arviz.ess(...)`

**用途**: 有効サンプルサイズ(ESS)。自己相関を考慮し、独立サンプル何個分の情報量かを表す。

**シグネチャ**: `ess(data, sample_dims=None, group='posterior', var_names=None, filter_vars=None, coords=None, method='bulk', relative=False, prob=None, chain_axis=0, draw_axis=1)`

**使用例**:
```python
for m in ["bulk", "tail", "mean", "sd", "median", "mad", "z_scale", "folded", "identity"]:
    print(m.ljust(8), round(float(az.ess(dt, var_names=["a"], method=m)["a"].item()), 1))
print("quantile", round(float(az.ess(dt, var_names=["a"], method="quantile", prob=0.1)["a"].item()), 1))
print("local   ", round(float(az.ess(dt, var_names=["a"], method="local", prob=(0.1, 0.3))["a"].item()), 1))
print("relative", round(az.ess(dt, var_names=["a"], relative=True)["a"].item(), 3))
```
実行結果:
```
bulk     2465.3
tail     1566.0
mean     2435.8
sd       1141.9
median   2792.1
mad      1285.2
z_scale  2465.3
folded   1157.9
identity 2396.4
quantile 1623.4
local    2019.0
relative 1.233
```

**注意点・落とし穴**:
- `method='bulk'`(既定)は分布の中心付近、`'tail'` は下側・上側の分位のESSの小さい方(ソース上 `min(qess(prob), qess(1-prob))`)。`prob` 省略時の値は 0.05 ではない(`prob=0.05` と `prob=(0.05, 0.95)` は 1555.3 で一致するが、省略すると 1566.0 と異なる)ので、明示的に指定したいときは `prob` を渡す。`quantile` は `prob`(単一の確率)、`local` は `prob=(下限, 上限)` の**2要素タプル**が必要(`prob=0.1` を渡すと `TypeError: object of type 'float' has no len()`)。
- `relative=True` は総サンプル数に対する比率で、上の例では 1.233 と **1を超える**(bulk ESS 2465 / 2000 draw)。負の自己相関のあるチェーンではESSが総サンプル数を超えることがある。
- 目安として bulk/tail ESS がどちらも 400 以上(4チェーンで各100以上)が推奨される(`plot_ess` のしきい線 `min_ess=400`、`diagnose` の `ess_threshold` 既定 `100×chain数` と整合)。

### `arviz.mcse(...)`

**用途**: モンテカルロ標準誤差(MCSE)。サンプリング誤差による推定値の不確かさを表す。

**シグネチャ**: `mcse(data, sample_dims=None, group='posterior', var_names=None, filter_vars=None, coords=None, method='mean', prob=None, circular=False, chain_axis=0, draw_axis=1)`

**使用例**:
```python
for m, kw in [("mean", {}), ("sd", {}), ("median", {}), ("quantile", {"prob": 0.9})]:
    print(m.ljust(8), round(float(az.mcse(dt, var_names=["a"], method=m, **kw)["a"].item()), 5))
print(az.summary(dt, var_names=["a"], kind="diagnostics", round_to="none")[["mcse_mean", "mcse_sd"]].round(5))
```
実行結果:
```
mean     0.00105
sd       0.00077
median   0.00107
quantile 0.00195
   mcse_mean  mcse_sd
a    0.00105  0.00077
```

**注意点・落とし穴**:
- `method` は `'mean'`(既定) / `'sd'` / `'median'` / `'quantile'`(`prob` 必須)。`az.summary` の `mcse_mean`/`mcse_sd` 列と同じ値(上の例で一致)。
- `summary` の docstring によれば、`round_to='auto'` では `mcse_*` を有効数字2桁で表示し、対応する統計量(mean など)の桁は 2×MCSE に基づいて決まる。

### `arviz.bfmi(...)`

**用途**: HMC/NUTS のエネルギーに基づく E-BFMI(ベイズ的欠損情報割合)をチェーンごとに計算する。0.3 を下回るとエネルギー遷移が悪く、事後の裾を探索できていない疑いがある。

**シグネチャ**: `bfmi(data, sample_dims=None, group='sample_stats', var_names='energy', filter_vars=None, coords=None, **kwargs)`

**使用例**:
```python
b = az.bfmi(c8)
print(type(b).__name__, b.name, dict(b.sizes))
print(b["energy"].values.round(3))
print(az.bfmi(dt)["energy"].values.round(3))
```
実行結果:
```
DataTree sample_stats {'chain': 4}
[0.287 0.378 0.346 0.324]
[1.158 1.133 1.226 1.018]
```

**注意点・落とし穴**:
- 既定で `group='sample_stats'`、`var_names='energy'` を使うので、`sample_stats` に `energy` が入っている(NUTS/HMCの結果)必要がある。戻り値のデータ変数名も `energy` で、`chain` 次元を持つ。
- 8 schools(centered)は 0.287 のチェーンが 0.3 を下回る一方、回帰モデルの例は全チェーン 1 前後で健全。

### `arviz.diagnose(...)`

**用途**: ダイバージェンス・最大木深さ・E-BFMI・ESS・R-hat を一括チェックし、CmdStan の `diagnose` 風のレポートを出力する。

**シグネチャ**: `diagnose(data, *, var_names=None, filter_vars=None, coords=None, sample_dims=None, group='posterior', rhat_max=1.01, ess_min_ratio=0.001, ess_threshold=None, bfmi_threshold=0.3, show_diagnostics=True, return_diagnostics=False)`

**使用例**:
```python
has_problem = az.diagnose(c8)
print("has_problem:", has_problem)
print(az.diagnose(dt, show_diagnostics=False))
err, info = az.diagnose(c8, return_diagnostics=True, show_diagnostics=False)
print(info["divergent"], info["bfmi"]["failed_chains"], info["rhat"]["bad_params"])
```
実行結果:
```
Divergences
19 of 2000 (0.95%) transitions ended with a divergence.
These divergent transitions indicate that HMC is not fully able to explore the posterior distribution.
Try increasing adapt delta closer to 1.
If this doesn't remove all divergences, try to reparameterize the model.

Tree depth
Treedepth satisfactory for all transitions.

E-BFMI
Chain 0: E-BFMI = 0.287
E-BFMI values are below the threshold 0.30 which suggests that HMC may have trouble exploring the target distribution.
If possible, try to reparameterize the model.

ESS
The following parameters have fewer than 400 effective samples:
  mu, theta, tau
This suggests that the sampler may not have fully explored the posterior distribution for this parameter.
Consider reparameterizing the model or increasing the number of samples.

R-hat
The following parameters have R-hat values greater than 1.01:
  mu, theta, tau
Such high values indicate incomplete mixing and biased estimation.
You should consider regularizing your model with additional prior information or a more effective parameterization.
has_problem: True
False
{'n_divergent': 19, 'pct': np.float64(0.95), 'total_samples': np.int64(2000)} [0] ['mu', 'theta', 'tau']
```

**注意点・落とし穴**:
- 戻り値の bool は `has_errors`(何らかの診断が失敗なら `True`)。`return_diagnostics=True` なら `(bool, dict)` で、dictのキーは `divergent` / `treedepth` / `bfmi` / `ess` / `rhat`。
- `show_diagnostics=True`(既定)は結果を標準出力に print する。ログとして扱いたいときは `False` にして戻り値を使う。
- **実測で確認した落とし穴**: 返却dictの `ess['bad_params']` は `ess_min_ratio`(既定0.001)基準の判定で、print されるメッセージ側の「400未満」(`ess_threshold` 既定=100×chain数)とは別基準。この例では print では `mu, theta, tau` が警告されるのに、dictの `bad_params` は `[]` だった。dictだけ見て「ESSは問題なし」と判断しないこと。

### `arviz.rhat_nested(...)`

**用途**: 短いチェーンを大量に走らせるとき(スーパーチェーン: 同じ初期値から始めたチェーンの群)用の nested R-hat。

**シグネチャ**: `rhat_nested(data, sample_dims=None, group='posterior', var_names=None, filter_vars=None, method='rank', coords=None, superchain_ids=None, chain_axis=0, draw_axis=1)`

**使用例**:
```python
n = az.rhat_nested(dt, var_names=["a"], superchain_ids=[0, 0, 1, 1])
print(round(float(n["a"].item()), 5), round(az.rhat(dt, var_names=["a"])["a"].item(), 5))
```
実行結果:
```
1.00029 1.00033
```

**注意点・落とし穴**:
- `superchain_ids` は各チェーンの所属スーパーチェーン番号のリスト(4チェーンなら `[0,0,1,1]` 等)。docstringによれば nested R-hat は 1 が下限で、スーパーチェーンあたり1チェーンの場合でも通常の R-hat と完全には一致しない。
- この例(収束済み)では通常の `rhat` とほぼ同じ値になる(1.00029 と 1.00033)。docstringでは「多数の短いチェーンを走らせるとき」に有用な診断とされている。

### `arviz.psense(...)` / `arviz.psense_summary(...)`

**用途**: 事前分布・尤度のべき乗スケーリングに対する事後分布の感度(power-scaling sensitivity)を、再サンプリングなしにPSISで計算する。事前分布が事後を支配していないか、尤度が情報を持っているかの診断。

**シグネチャ**: `psense(data, var_names=None, filter_vars=None, group='prior', coords=None, sample_dims=None, alphas=(0.99, 1.01), group_var_names=None, group_coords=None)` / `psense_summary(data, var_names=None, filter_vars=None, coords=None, sample_dims=None, threshold=0.05, alphas=(0.99, 1.01), prior_var_names=None, likelihood_var_names=None, prior_coords=None, likelihood_coords=None, round_to=3)`

**使用例**:
```python
print(az.psense(dt, var_names=["a", "b", "sigma"]))                       # 事前スケーリング
print(az.psense(dt, var_names=["a", "b", "sigma"], group="likelihood"))   # 尤度スケーリング
print(az.psense_summary(dt, var_names=["a", "b", "sigma"]))
```
実行結果:
```
<xarray.Dataset> Size: 24B
Dimensions:  ()
Data variables:
    a        float64 8B 0.0001978
    b        float64 8B 0.0008353
    sigma    float64 8B 0.0008765
<xarray.Dataset> Size: 24B
Dimensions:  ()
Data variables:
    a        float64 8B 0.0917
    b        float64 8B 0.1076
    sigma    float64 8B 0.1908
       prior  likelihood diagnosis
a      0.000       0.092         ✓
b      0.001       0.108         ✓
sigma  0.001       0.191         ✓
```

**注意点・落とし穴**:
- `log_prior` グループ(事前分布の対数密度)が必須。PyMC 6.3.1 には対応する `compute_log_prior` 相当の関数が見当たらなかったため、この辞書の共通データ作成では scipy で手計算して `dt["log_prior"] = az.dict_to_dataset({...})` として追加した。`log_likelihood` は `pm.compute_log_likelihood` で作れる。
- docstringの解釈: **事前感度が 0.05 超なら事前が情報的**、**尤度感度が 0.05 未満なら尤度が弱い/非情報的**。この例は事前感度がほぼ0(0.000〜0.001)、尤度感度は 0.09〜0.19 で、診断は `✓`(問題なし)。
- 戻り値は `psense` が `xarray.Dataset`、`psense_summary` が `DataFrame`(列 `prior` / `likelihood` / `diagnosis`)。

---

## 5. 分布・情報量ユーティリティ

### `arviz.kde(...)` / `arviz.ecdf(...)` / `arviz.histogram(...)` / `arviz.qds(...)`

**用途**: プロットの下請けにもなっている密度推定・経験累積分布・ヒストグラム・分位点ドットプロットの座標データを、`plot_axis` 次元(x/y)付きで返す。

**シグネチャ**: `kde(data, dim=None, group='posterior', var_names=None, filter_vars=None, coords=None, circular=False, **kwargs)` / `ecdf(data, dim=None, group='posterior', var_names=None, filter_vars=None, coords=None, pit=False, **kwargs)` / `histogram(data, dim=None, group='posterior', var_names=None, filter_vars=None, coords=None, bins=None, range=None, weights=None, density=True)` / `qds(data, dim=None, group='posterior', var_names=None, filter_vars=None, coords=None, nquantiles=100, binwidth=None, dotsize=1, stackratio=1, top_only=False, **kwargs)`

**使用例**:
```python
k = az.kde(dt, var_names=["a"])
print(type(k).__name__, dict(k.sizes), k["a"].plot_axis.values)
x, y = k["a"].values
print(x[:2].round(3), y.argmax())

ds = dt.posterior.dataset                        # ecdf は Dataset で渡す
e = az.ecdf(ds, var_names=["a"])
print(type(e).__name__, e["a"].sel(plot_axis="y").values[[0, -1]])

h = az.histogram(dt, var_names=["a"], bins=5)
print(h["a"].plot_axis.values, h["a"].sel(plot_axis="left_edges").values.round(3))
q = az.qds(dt, var_names=["a"], nquantiles=20)
print(dict(q.sizes))
xk, yk, bw = az.kde(np.random.default_rng(0).normal(size=1000))     # ndarrayも可
print(xk.shape, yk.shape, round(float(bw), 3))
```
実行結果:
```
DataTree {'plot_axis': 2, 'kde_dim': 512} ['x' 'y']
[0.705 0.706] 275
Dataset [5.e-04 1.e+00]
['histogram' 'left_edges' 'right_edges'] [0.705 0.783 0.86  0.938 1.016]
{'plot_axis': 2, 'qd_dim': 20}
(512,) (512,) 0.25
```

**注意点・落とし穴**:
- **実測で確認した落とし穴**: `az.ecdf(dt, ...)` に `DataTree` を渡すと `NotImplementedError: DataTree ecdf not available yet, use 'dt[group].dataset' instead.` になる(`kde`/`histogram`/`qds` は `DataTree` でも動く)。`dt.posterior.dataset` を渡す。
- `kde` は `plot_axis` 次元(座標 `'x'`/`'y'`)と `kde_dim`(512点)を持つ `DataTree`/`Dataset`。バンド幅は座標 `bw_a` に入る。`ndarray` を渡すと `(x, y, bw)` のタプルが返る。
- `histogram` は `density=True`(既定)で密度を返し、`plot_axis` 座標は `'histogram'`/`'left_edges'`/`'right_edges'`。

### `arviz.kl_divergence(...)` / `arviz.wasserstein(...)`

**用途**: 2つの `DataTree`(モデル)の事後分布間の差を、KLダイバージェンス(非対称)/ Wasserstein-1距離で測る。

**シグネチャ**: `kl_divergence(data1, data2, group='posterior', var_names=None, sample_dims=None, num_samples=500, round_to=None, random_seed=212480)` / `wasserstein(data1, data2, group='posterior', var_names=None, sample_dims=None, joint=True, num_samples=500, round_to=None, random_seed=212480)`

**使用例**:
```python
nc8 = az.load_arviz_data("non_centered_eight")
print(az.kl_divergence(c8, nc8, var_names="mu", round_to="none"))
print(az.kl_divergence(nc8, c8, var_names="mu", round_to="none"))      # 非対称
print(az.wasserstein(c8, nc8, var_names=["mu", "tau"], joint=False, round_to=3))
print(az.wasserstein(c8, nc8, var_names=["mu", "tau"], round_to=3))
print(az.kl_divergence(dt, dt, var_names="a"))
```
実行結果:
```
0.437292727600783
0.23292966554196834
1.298
1.179
0
```

**注意点・落とし穴**:
- 戻り値は変数ごとではなく**単一の float**(`var_names=['mu','tau']` と2変数を渡しても1つの値)。`wasserstein` の `joint=True`(既定)は同時分布、`joint=False` は周辺分布を独立とみなした計算で、後者の方が高速(docstring)。`num_samples=500` のサブサンプルと `random_seed` を使うので、`random_seed` を固定すると再現する。
- `round_to` の既定は rcParams の `stats.round_to`(`'2g'`)なので、生の値が必要なら `round_to='none'` を渡す。KLは `data1`/`data2` の順序で値が変わる。

### `arviz.bayes_factor(...)`

**用途**: Savage–Dickey 比でベイズファクターを計算する(事前分布と事後分布の、参照値における密度比をKDEで推定)。

**シグネチャ**: `bayes_factor(data, var_names, ref_vals=0, return_ref_vals=False, prior=None, circular=False)`

**使用例**:
```python
bf = az.bayes_factor(dt, var_names=["a"], ref_vals=1.0)
print(type(bf).__name__, bf["a"].bf_type.values, bf["a"].values.round(3))
bf2, dens = az.bayes_factor(dt, var_names=["a"], ref_vals=1.0, return_ref_vals=True)
print(dens["a"].density_type.values, dens["a"].values.round(3))
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    az.bayes_factor(dt, var_names=["b"], ref_vals=0)
print(str(w[0].message)[:78])
```
実行結果:
```
Dataset ['BF10' 'BF01'] [ 0.049 20.373]
['posterior' 'prior'] [1.709 0.084]
Reference value 0 for 'b' is outside the posterior range. This may overstate e
```

**注意点・落とし穴**:
- `prior` グループ(`dt["prior"]`)が必要で、無ければ `prior={変数名: 事前分布}` を辞書で渡す。`bf_type` 次元は `BF10` / `BF01` の2要素。`return_ref_vals=True` で得た参照値での密度は事後 1.709・事前 0.084 で、BF01 = 事後密度/事前密度 = 1.709/0.084 ≈ 20.4(上の出力と一致)、BF10 はその逆数 0.049。参照値 1.0 は `a` の真値に近いので参照値(H0)寄りの結果になる。
- **実測**: 参照値が事後分布の範囲外だと `UserWarning: Reference value 0 for 'b' is outside the posterior range. This may overstate evidence in favor of H1.` が出る。Savage–Dickey 比は事前密度を分母/分子に使うため、事前の設定(この例は N(0,5) の広い事前)に結果が左右される。

---

## 6. モデル比較・予測評価

### `arviz.loo(...)`

**用途**: PSIS-LOO(Pareto平滑化重点サンプリングによる近似leave-one-out交差検証)で予測性能 elpd を推定する。`log_likelihood` グループが必要。戻り値は `ELPDData`。

**シグネチャ**: `loo(data, pointwise=None, var_name=None, reff=None, log_lik_fn=None, log_weights=None, pareto_k=None, log_jacobian=None, mixture=False, moment_match=False, model=None)`

**使用例**:
```python
r = az.loo(dt)
print(r)
print(type(r).__name__, r.kind, r.scale, r.n_data_points, r.n_samples)
print(round(r.elpd, 3), round(r.se, 3), round(r.p, 3))
print(r.elpd_i.dims, r.pareto_k.dims, float(r.pareto_k.max()))
print(az.loo(dt, pointwise=False).elpd_i)
print("waic" in dir(az))
```
実行結果:
```
Computed from 2000 posterior samples and 60 observations log-likelihood matrix.

         Estimate       SE
elpd_loo   -30.16     4.54
p_loo        2.58        -
------

Pareto k diagnostic values:
                         Count   Pct.
(-Inf, 0.70]   (good)       60  100.0%
   (0.70, 1]   (bad)         0    0.0%
    (1, Inf)   (very bad)    0    0.0%

ELPDData loo log 60 2000
-30.162 4.537 2.581
('obs_id',) ('obs_id',) 0.22495648187648362
None
False
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: `az.waic` は存在しない。モデル比較は `loo` / `compare` を使う。戻り値の表示は `elpd_loo` の行と Pareto k の診断表で、しきい値は 0.7(`(-Inf, 0.70]` が good、`(0.70, 1]` が bad、`(1, Inf)` が very bad)。
- `ELPDData` は属性 `elpd`(合計) / `se` / `p`(有効パラメータ数) / `elpd_i`(点ごと) / `pareto_k` / `n_data_points` / `n_samples` / `scale` / `kind` などを持つ。`pointwise=False` にすると `elpd_i` は `None` になる(上の例で確認)。
- 観測変数が複数ある場合は `var_name=` を指定する。Pareto k が大きい点が多いときは `loo_moment_match` / `reloo` / `loo_kfold` 等の代替が用意されている(この辞書では動作未検証)。

### `arviz.loo_i(...)`

**用途**: 観測点 `i` 1点だけのLOO elpd を計算する(全点の計算コストを避けたいとき用)。

**シグネチャ**: `loo_i(i, data, var_name=None, reff=None, log_lik_fn=None, log_weights=None, pareto_k=None, log_jacobian=None)`

**使用例**:
```python
one = az.loo_i(0, dt)
print(type(one).__name__, one.n_data_points, round(one.elpd, 3), round(float(az.loo(dt).elpd_i[0]), 3))
```
実行結果:
```
ELPDData 1 -1.843 -1.843
```

**注意点・落とし穴**:
- 戻り値は `ELPDData`(観測数1)で、`az.loo(dt).elpd_i[0]` と同じ値になる(上の例で一致)。

### `arviz.compare(...)`

**用途**: 複数モデルの elpd(PSIS-LOO)を比較して順位・差・スタッキング重みなどの表を作る。

**シグネチャ**: `compare(compare_dict, method='stacking', var_name=None, reference=None, round_to='auto')`

**使用例**:
```python
cmp = az.compare({"linear": dt, "null": dt0})
print(type(cmp).__name__, cmp.columns.tolist())
print(cmp)
print(az.compare({"linear": dt, "null": dt0}, reference="null")[["rank", "elpd_diff"]])
print(az.compare({"linear": az.loo(dt), "null": az.loo(dt0)}, method="pseudo-BMA", round_to=2)[["rank", "elpd", "weight"]])
```
実行結果:
```
DataFrame ['rank', 'elpd_diff', 'dse', 'p_worse', 'diag_diff', 'diag_elpd', 'p', 'elpd', 'se', 'weight']
        rank  elpd_diff  dse  p_worse diag_diff diag_elpd    p   elpd   se  weight
linear     0        0.0  0.0      NaN                      2.6  -30.0  4.5     1.0
null       1      -80.0  7.1      1.0   N < 100            1.9 -110.0  5.6     0.0
        rank  elpd_diff
linear     0       80.0
null       1        0.0
        rank    elpd  weight
linear     0  -30.16     1.0
null       1 -114.47     0.0
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: 列は `rank / elpd_diff / dse / p_worse / diag_diff / diag_elpd / p / elpd / se / weight`(0.x の `loo`, `loo_se`, `warning`, `scale` 列は無い)。`rank=0` が最良で、`elpd_diff` は基準モデルとの差(基準は最良モデル)。
- `reference=` で基準モデルを指定できる(上の例では `null` が基準になり `elpd_diff` の符号が反転。このとき列名 `p_worse` は `p_better` に変わる、というのが docstring の記述)。`method` は `'stacking'`(既定) / `'BB-pseudo-BMA'` / `'pseudo-BMA'`。
- 入力は `DataTree` でも `az.loo` の結果(`ELPDData`)でもよい。`diag_diff` は docstring によると `N < 100`(データが少ない)、`|elpd_diff| < 4`(モデル差が小さい)、空文字(問題なし)のいずれかで、前2者のときは `p_worse` 等の解釈に注意が必要。上の表は60点なので `N < 100` が付いた。`diag_elpd` は Pareto k が高い点があると `K k_psis > threshold` になる。表示は `round_to='auto'` で丸められた見た目で、生の値は `round_to='none'` で得る。

### `arviz.loo_pit(...)`

**用途**: LOO-PIT値(LOO予測分布での観測値の累積確率)を観測点ごとに計算する。一様分布に近ければ予測分布が較正されている。

**シグネチャ**: `loo_pit(data, var_names=None, log_weights=None, pareto_k=None, random_state=None, pareto_pit=False)`

**使用例**:
```python
p = az.loo_pit(dt)
print(type(p).__name__, dict(p.sizes))
v = p["y"].values
print(v[:3].round(3), round(float(v.mean()), 3))
```
実行結果:
```
Dataset {'obs_id': 60}
[0.033 0.394 0.707] 0.506
```

**注意点・落とし穴**:
- `posterior_predictive` と `log_likelihood` が必要。戻り値は観測変数と同じ次元の `Dataset`。可視化は `plot_loo_pit`。

### `arviz.loo_r2(...)` / `arviz.bayesian_r2(...)` / `arviz.residual_r2(...)`

**用途**: ベイズ回帰のR²。`loo_r2` はLOO予測に基づく、`bayesian_r2`/`residual_r2` は事後のモデル分散/残差分散に基づく。

**シグネチャ**: `loo_r2(data, var_name, n_simulations=4000, summary=True, point_estimate=None, ci_kind=None, ci_prob=None, circular=False, round_to=None)` / `bayesian_r2(data, pred_mean, scale, scale_kind='sd', summary=True, group='posterior', point_estimate=None, ci_kind=None, ci_prob=None, circular=False, round_to=None)` / `residual_r2(data, pred_mean=None, obs_name=None, summary=True, group='posterior', point_estimate=None, ci_kind=None, ci_prob=None, circular=False, round_to=None)`

**使用例**:
```python
print(az.loo_r2(dt, var_name="y", n_simulations=1000))

rng = np.random.default_rng(1)
xs = rng.normal(size=50); ys = 1 + 2 * xs + rng.normal(scale=0.5, size=50)
a_ = rng.normal(1, 0.07, (4, 300)); b_ = rng.normal(2, 0.07, (4, 300)); s_ = np.abs(rng.normal(0.5, 0.05, (4, 300)))
mu = a_[..., None] + b_[..., None] * xs
t = az.from_dict({"posterior": {"mu": mu, "sigma": s_},
                  "posterior_predictive": {"y": mu + s_[..., None] * rng.normal(size=mu.shape)},
                  "observed_data": {"y": ys}})
print(az.bayesian_r2(t, pred_mean="mu", scale="sigma"))
print(az.residual_r2(t, pred_mean="mu", obs_name="y"))
print(az.bayesian_r2(t, pred_mean="mu", scale="sigma", summary=False).shape)
```
実行結果:
```
loo_R2(mean=0.94, eti_lb=0.91, eti_ub=0.96)
bayesian_R2(mean=0.93, eti_lb=0.9, eti_ub=0.95)
residual_R2(mean=0.95, eti_lb=0.94, eti_ub=0.95)
(1200,)
```

**注意点・落とし穴**:
- `bayesian_r2` は `posterior` 内に「予測平均」変数(`pred_mean`)と、`scale`(SD、`scale_kind='var'` なら分散)変数が必要。PyMCでは `pm.Deterministic("mu", ...)` で予測平均を記録しておく。`scale` は**必須引数**(ロジスティック回帰では `None` を明示)。
- 戻り値の `summary=True` は名前付きタプル(`bayesian_R2(mean=..., eti_lb=..., eti_ub=...)`)、`summary=False` はR²サンプルの ndarray(上の例で `(1200,)`)。点推定・区間の種類は rcParams(`stats.point_estimate`, `stats.ci_kind`, `stats.ci_prob`)に従う。
- `loo_r2` は `posterior_predictive` グループと `log_likelihood` から計算する(乱数を使うので `n_simulations` により値が揺れる)。

### `arviz.loo_expectations(...)` / `arviz.loo_metrics(...)` / `arviz.loo_score(...)`

**用途**: LOO重みで予測平均・分位を計算する(`loo_expectations`)/ RMSE・MAE等の予測誤差(`loo_metrics`)/ CRPS等のスコア(`loo_score`)。

**シグネチャ**: `loo_expectations(data, var_name=None, group='posterior_predictive', sample_dims=None, log_likelihood_var_name=None, kind='mean', probs=None, log_weights=None, pareto_k=None)` / `loo_metrics(data, kind='rmse', var_name=None, round_to=None)` / `loo_score(data, var_name=None, kind='crps', pointwise=False, round_to=None, log_weights=None, pareto_k=None)`

**使用例**:
```python
mean, sd = az.loo_expectations(dt, var_name="y")
print(type(mean).__name__, mean.dims, sd.dims, mean.values[:2].round(3))
print(az.loo_metrics(dt, kind="rmse"))
print(az.loo_metrics(dt, kind="mae"))
print(az.loo_score(dt, kind="crps"))
```
実行結果:
```
DataArray ('obs_id',) ('obs_id',) [ 1.503 -1.13 ]
rmse(mean=0.39, se=0.03)
mae(mean=0.32, se=0.029)
CRPS(mean=-0.2268162729895059, se=0.018383184487866373)
```

**注意点・落とし穴**:
- `loo_expectations` は `(loo_expec, khat)` の2つを返す(LOO重みによる観測点ごとの期待値と、その関数専用のPareto k)。`kind` は `'mean'`(既定) / `'median'` / `'var'` / `'sd'` / `'quantile'`(`probs=` 指定)/ `circular_*`。上の例の第2戻り値は `sd` ではなく `khat`。
- `loo_metrics`/`loo_score` は `(mean, se)` の名前付きタプル。`loo_metrics` の `kind` は `'rmse'`(既定) / `'mae'` / `'mse'` / `'acc'` / `'acc_balanced'`(後2者は分類用)で、`round_to` 既定は '2g' 丸め。一方 `loo_score` は同じ `round_to=None` でも**丸められず**多数桁の値が出る(上の出力で確認)。`observed_data`・`posterior_predictive`・`log_likelihood` が必要。

### `arviz.weight_predictions(...)`

**用途**: 複数モデルの事後予測分布を重み付きで混ぜ(モデル平均)、新しい `posterior_predictive` を持つ `DataTree` を返す。

**シグネチャ**: `weight_predictions(dts, weights=None, group='posterior_predictive', sample_dims=None, random_seed=None)`

**使用例**:
```python
try:
    az.weight_predictions([dt, dt0], weights=[0.7, 0.3])          # dt0 に posterior_predictive が無い
except ValueError as e:
    print("ValueError:", e)
nc8 = az.load_arviz_data("non_centered_eight")
w = az.weight_predictions([c8, nc8], weights=[0.7, 0.3], random_seed=0)
print(list(w.children), dict(w.posterior_predictive.sizes))
```
実行結果:
```
ValueError: All the objects must contain the `posterior_predictive` group
['posterior_predictive', 'observed_data'] {'school': 8, 'sample': 2000}
```

**注意点・落とし穴**:
- **実測**: 渡した全 `DataTree` に対象グループ(既定 `posterior_predictive`)が無いと `ValueError: All the objects must contain the `posterior_predictive` group` になる。`weights` 省略時の扱い(等重みか)はdocstringで未確認。`compare` の `weight` 列を渡す使い方が想定される用途。

---

## 7. 可視化の基本(PlotCollection)

### `arviz.PlotCollection` / `arviz.PlotMatrix`(プロット関数の戻り値)

**用途**: 1.x の `plot_*` 関数は matplotlib の `Axes` 配列ではなく、新しいプロット基盤 `arviz_plots` の `PlotCollection`(`plot_pair` だけ `PlotMatrix`)を返す。描画済みの図・軸・アーティストはこの中の `viz`(DataTree)に保持される。

**シグネチャ**: `PlotCollection.grid(data, cols=None, rows=None, backend=None, figure_kwargs=None, **kwargs)` / `PlotCollection.wrap(data, cols=None, col_wrap=4, backend=None, figure_kwargs=None, **kwargs)` / `PlotCollection.savefig(filename, **kwargs)`

**使用例**:
```python
pc = az.plot_dist(dt, var_names=["a", "b"])
print(type(pc).__name__, pc.backend)
print(list(pc.viz.children))
ax = pc.get_target("a", {})                      # 変数 a のmatplotlib Axes
print(type(ax).__name__, ax is pc.viz["plot"]["a"].item())
ax.axvline(1.0, color="red")                     # 通常のmatplotlib操作で加工できる
ax.set_title("a (custom)")
pc.savefig(IMG + "/pc_custom.png", bbox_inches="tight")
pm = az.plot_pair(c8, var_names=["mu", "tau"])
print(type(pm).__name__, hasattr(pm, "map_lower"))
```
実行結果:
```
PlotCollection matplotlib
['plot', 'row_index', 'col_index', 'dist', 'credible_interval', 'point_estimate', 'point_estimate_text', 'title']
Axes True
PlotMatrix True
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: 戻り値を `axes[0, 0]` のようにインデックスすることはできない。保存は `pc.savefig(path, **kwargs)`(`bbox_inches='tight'` 等はそのままmatplotlibに渡る)。`pc.viz` は `plot` / `title` / 各ビジュアル名のグループを持つ `DataTree` で、`pc.viz["plot"]["a"].item()` や `pc.get_target(変数名, 座標dict)` で個別の `Axes` が得られる。
- `backend="none"` を渡すと描画せず(`PlotCollection` の `backend` が `'none'`)構造だけ作れる。matplotlib以外に `plotly` はこのvenvにインストール済みだが、`bokeh` は未インストール。この辞書の検証はすべて matplotlib。
- 上のコードで保存した画像では、`a` の事後密度に赤い縦線(真値1.0)が引かれ、タイトルが `a (custom)` に変わっている(目視確認)。

### プロット共通の引数(`var_names` / `coords` / `visuals` / `figure_kwargs` / `col_wrap` ほか)

**用途**: `plot_*` 関数はほぼ共通の引数セットを持つ。変数の絞り込み(`var_names`/`filter_vars`/`coords`/`group`/`sample_dims`)に加え、要素ごとの見た目は `visuals` / `aes_by_visuals` / `stats`、図全体は `figure_kwargs` / `col_wrap`(`**pc_kwargs`)で調整する。

**シグネチャ**: `az.plot_dist(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, kind=None, point_estimate=None, rope=None, ci_kind=None, ci_prob=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_dist(
    dt, var_names=["a", "b", "sigma"],
    ci_kind="hdi", ci_prob=0.94, point_estimate="median",   # 区間・点推定
    col_wrap=2, figure_kwargs={"figsize": (9, 5)},           # 図のレイアウト
    visuals={"title": False},                                # 要素の非表示
)
pc.savefig(IMG + "/common_args.png", bbox_inches="tight")
print(sorted(pc.viz.children))

pc2 = az.plot_dist(c8, var_names=["theta"], coords={"school": ["Choate", "Deerfield"]})
print(pc2.viz["plot"].dataset["theta"].dims)

try:
    az.plot_dist(dt, var_names=["a"], figsize=(8, 3))
except ValueError as e:
    print("ValueError:", str(e)[:78])
try:
    az.plot_ecdf_pit(dt, group="posterior_predictive")
except ValueError as e:
    print("ValueError:", str(e)[:75])
```
実行結果:
```
['col_index', 'credible_interval', 'dist', 'plot', 'point_estimate', 'point_estimate_text', 'row_index']
('school',)
ValueError: Keyword arguments ['figsize'] have been passed as **kwargs but have no active 
ValueError: Requested 60 subplots, which exceeds rcParams['plot.max_subplots']=40. Redu
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: 0.x の `figsize=` や `hdi_prob=` のような直接引数はなく、`figsize` は `figure_kwargs={"figsize": (w, h)}`、確率は `ci_prob=`、区間の種類は `ci_kind=`。`figsize` をそのまま渡すと `ValueError: Keyword arguments ['figsize'] have been passed as **kwargs but have no active aesthetic mapped to them...` になる(余分な kwargs は aesthetic 指定として解釈されるため)。
- `visuals={"要素名": False}` でその要素を非表示、`{"要素名": {matplotlibのkwargs}}` でスタイル指定。使える要素名は関数ごとに異なる(署名の `visuals` の型に列挙されている。例: `plot_dist` は `dist/face/credible_interval/point_estimate/point_estimate_text/rope/rope_text/title/rug/remove_axis`)。
- `ci_kind`/`ci_prob`/`point_estimate` の既定は rcParams(`stats.ci_kind`=`'eti'`、`stats.ci_prob`=0.89、`stats.point_estimate`=`'mean'`)に従う。94%HDIにしたいコードは毎回指定するか `az.rc_context` で囲む。
- 1つの図に出せるサブプロット数の上限は `rcParams['plot.max_subplots']`(既定40)。超えると `ValueError: Requested 60 subplots, which exceeds rcParams['plot.max_subplots']=40...`(上の例)。観測点ごとのファセットになるプロットで発生しうる。
- 保存した画像の目視確認: `a`/`b`/`sigma` の3つの密度が2列で並び(3つ目は左下)、タイトル無し、各密度の下に中央値の点と94%HDIの線が描かれていた。

### `arviz.style.use(...)` / `arviz.style.available()`

**用途**: ArviZ同梱のスタイル(`arviz-variat`, `arviz-darkgrid` など)を適用する。matplotlib 用のスタイル一覧は `style.available()` で得られる。

**シグネチャ**: `style.use(name)` / `style.available()`

**使用例**:
```python
av = az.style.available()
print(list(av), [n for n in av["matplotlib"] if n.startswith("arviz")])
az.style.use("arviz-darkgrid")
pc = az.plot_dist(dt, var_names=["a", "b"])
pc.savefig(IMG + "/style_dark.png", bbox_inches="tight")
import matplotlib
matplotlib.rcdefaults()                           # matplotlibのグローバル設定を元に戻す
```
実行結果:
```
['matplotlib', 'plotly', 'common'] ['arviz-cetrino', 'arviz-darkgrid', 'arviz-tenui', 'arviz-tumma', 'arviz-variat', 'arviz-vibrant']
```

**注意点・落とし穴**:
- `style.use` は matplotlib のグローバルな rcParams を書き換えるので、以降の図すべてに影響する(上の例では最後に `matplotlib.rcdefaults()` で戻した)。
- 画像の目視確認: `arviz-darkgrid` 適用後は、背景が薄い灰色でグリッド線が白、密度曲線が青(既定より濃い青)、フォントがやや大きい図になった。

### `arviz.labels`(ラベラー: `labeller=` 引数)

**用途**: 軸・凡例に出る変数名・座標の表示文字列をカスタマイズする。`MapLabeller` は変数名の置換、他に `DimCoordLabeller` / `IdxLabeller` / `DimIdxLabeller` / `NoVarLabeller` / `mix_labellers`。

**シグネチャ**: `az.labels.MapLabeller(var_name_map=None, dim_map=None, coord_map=None, ...)`

**使用例**:
```python
print([n for n in dir(az.labels) if n.endswith("Labeller")])
lab = az.labels.MapLabeller(var_name_map={"mu": "μ", "tau": "τ"})
pc = az.plot_dist(c8, var_names=["mu", "tau"], labeller=lab)
pc.savefig(IMG + "/labeller.png", bbox_inches="tight")
```
実行結果:
```
['BaseLabeller', 'DimCoordLabeller', 'DimIdxLabeller', 'IdxLabeller', 'Labeller', 'MapLabeller', 'NoVarLabeller']
```

**注意点・落とし穴**:
- `labeller` は各 `plot_*` 関数の引数。上の例では画像のタイトルが `mu`/`tau` から `μ`/`τ` に置き換わった(目視確認)。MapLabeller の引数詳細(`dim_map` 等)はここでは未検証。

### `arviz.add_lines(...)` / `arviz.add_bands(...)` / `arviz.combine_plots(...)`

**用途**: 描画済みの `PlotCollection` に基準線・帯を後から追加する(`add_lines`/`add_bands`)。複数種類のプロットを1つの図に並べる(`combine_plots`)。

**シグネチャ**: `add_lines(plot_collection, values, orientation='vertical', aes_by_visuals=None, visuals=None, sample_dims=None, ref_dim='ref_dim', **kwargs)` / `add_bands(plot_collection, values, orientation='vertical', aes_by_visuals=None, visuals=None, sample_dims=None, ref_dim=None, **kwargs)` / `combine_plots(dt, plots, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, expand='column', plot_names=None, backend=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_dist(dt, var_names=["a", "b"])
az.add_lines(pc, xr.Dataset({"a": 1.0, "b": 2.0}))           # 真値の縦線
pc.savefig(IMG + "/add_lines.png", bbox_inches="tight")

pc = az.plot_dist(dt, var_names=["a"])
az.add_bands(pc, [(0.85, 0.95)])                             # 帯(1本)
pc.savefig(IMG + "/add_bands.png", bbox_inches="tight")

pc = az.combine_plots(dt, [(az.plot_dist, {}), (az.plot_rank, {})], var_names=["a", "b"])
print(type(pc).__name__)
pc.savefig(IMG + "/combine.png", bbox_inches="tight")
```
実行結果:
```
PlotCollection
```

**注意点・落とし穴**:
- `add_lines` の `values` には `Dataset`(変数名→位置)を渡して動作を確認した(docstringでは int/float/tuple/list/dict も可)。画像では `a` の密度に x=1.0、`b` の密度に x=2.0 の破線の縦線が入った(目視確認)。
- `combine_plots(dt, [(関数, kwargs), ...], expand='column')` は関数を列方向に並べる。この例(`plot_dist` + `plot_rank`)では左列が事後密度、右列がrankプロットになったが、rankプロットの `p=...` 注記がタイトルや隣のプロットと重なる箇所があった(目視確認)。`add_bands` は `[(0.85, 0.95)]` のように**タプルを要素とするリスト**で1本の帯を指定して動作した(画像では `a` の密度の x=0.85〜0.95 が灰色で塗られた)。単なるタプル `(0.85, 0.95)` や `{"a": (0.85, 0.95)}` は `ValueError: ref_dim length (2) does not match reference values length (1) for data variable a` になった(実測)。

---

## 8. 可視化(事後分布・収束診断)

### `arviz.plot_trace(...)` / `arviz.plot_trace_dist(...)`

**用途**: サンプル列(トレース)で収束・混合を目視確認する。`plot_trace` は**トレース(draw対値)のみ**、`plot_trace_dist` は周辺密度とトレースを並べる。

**シグネチャ**: `plot_trace(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, **pc_kwargs)` / `plot_trace_dist(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, compact=True, combined=False, kind=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_trace(dt, var_names=["a", "b"])
pc.savefig(IMG + "/trace.png", bbox_inches="tight")
print(type(pc).__name__, sorted(pc.viz.children))

pc = az.plot_trace_dist(dt, var_names=["a", "b"])
pc.savefig(IMG + "/trace_dist.png", bbox_inches="tight")
print(sorted(pc.viz.children))

pc = az.plot_trace(c8, var_names=["tau"])                # ダイバージェンスのあるデータ
pc.savefig(IMG + "/trace_div.png", bbox_inches="tight")
print("divergence" in pc.viz.children)
```
実行結果:
```
PlotCollection ['col_index', 'plot', 'row_index', 'title', 'trace', 'xlabel']
['col_index', 'dist', 'plot', 'row_index', 'trace', 'xlabel_trace']
True
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: 1.3.0 の `plot_trace` は**トレースだけ**を描く(変数ごとに1枚、x軸が draw、チェーンごとに色分け)。0.x のように左に周辺分布・右にトレースを並べたいときは `plot_trace_dist`(左に密度、右にトレース、x軸ラベル `Draw`)を使う。目視で確認した限り、`plot_trace` の図には密度パネルは無かった。
- ダイバージェンスは `sample_stats['diverging']` があると `divergence` ビジュアルとして自動描画される(`centered_eight` の `tau` トレースでは、トレースの下端に黒い短い縦線が並び、発散した draw の位置を示した。目視確認)。
- `plot_trace_dist` の画像では、左の密度がチェーンごとに別の線種(実線・破線・点線など)で4本重なって描かれた(`combined=False` が既定)。`compact=True`(既定)の詳細な挙動は未検証。

### `arviz.plot_dist(...)`

**用途**: 1変数の周辺事後密度(KDE/ヒストグラム/ECDF/ドット)に、点推定と信用区間(ROPE指定も可)を重ねる。0.x の `plot_posterior` に相当する主力関数。

**シグネチャ**: `plot_dist(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, kind=None, point_estimate=None, rope=None, ci_kind=None, ci_prob=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_dist(dt, var_names=["a", "b"])
pc.savefig(IMG + "/dist.png", bbox_inches="tight")
pc = az.plot_dist(dt, var_names=["a"], kind="hist")
pc.savefig(IMG + "/dist_hist.png", bbox_inches="tight")
pc = az.plot_dist(dt, var_names=["b"], rope=(1.9, 2.1))
pc.savefig(IMG + "/dist_rope.png", bbox_inches="tight")
print(sorted(pc.viz.children))
print(hasattr(az, "plot_posterior"))
```
実行結果:
```
['col_index', 'credible_interval', 'dist', 'plot', 'point_estimate', 'point_estimate_text', 'rope', 'rope_text', 'row_index', 'title']
False
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: `az.plot_posterior` は存在しない(`plot_dist` に置き換わった)。0.x の `hdi_prob` / `ref_val` / `point_estimate` のうち、現在は `ci_prob`/`ci_kind`/`point_estimate` と `rope`(区間)で指定する。参照値の縦線が必要なら `az.add_lines` で後から追加できる。
- 目視確認: 既定の図は変数ごとの滑らかな密度曲線の下に、点推定(`0.91 mean` のラベル付き点)と信用区間(既定89%ETI)の線が描かれる。`rope=(1.9, 2.1)` を指定した `b` では、1.9〜2.1に灰色の矩形が塗られ、`87.9% in ROPE` と注記された(信用区間内のうちROPEに入る割合。`ci_in_rope(..., ci_kind='eti')` の 87.87 と一致)。
- `kind` は `'auto'`(既定、rcParams `plot.density_kind`) / `'kde'` / `'hist'` / `'ecdf'` / `'dot'`。`kind='hist'` では階段状のヒストグラム線になった(目視確認)。この例の `'auto'` は KDE になったが、`plot_ppc_dist` の `'auto'` は連続データでECDFになる(関数ごとに解釈が異なる)。

### `arviz.plot_forest(...)` / `arviz.plot_ridge(...)`

**用途**: 複数の変数・座標(例: グループごとのパラメータ)の信用区間を縦に並べて比較する(`plot_forest`)/ 密度を縦に重ねる(`plot_ridge`)。

**シグネチャ**: `plot_forest(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, combined=False, point_estimate=None, ci_kind=None, ci_probs=None, labels=None, shade_label=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_ridge(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, combined=True, ridge_height=0.9, labels=None, shade_label=None, plot_collection=None, backend=None, labeller=None, kind=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_forest(c8, var_names=["theta"])
pc.savefig(IMG + "/forest.png", bbox_inches="tight")
pc = az.plot_forest(c8, var_names=["theta"], combined=True, ci_kind="hdi", ci_probs=(0.5, 0.94))
pc.savefig(IMG + "/forest_comb.png", bbox_inches="tight")
pc = az.plot_forest({"linear": dt, "null": dt0}, var_names=["a", "sigma"], combined=True)   # モデル比較
pc.savefig(IMG + "/forest_multi.png", bbox_inches="tight")
pc = az.plot_ridge(c8, var_names=["theta"])
pc.savefig(IMG + "/ridge.png", bbox_inches="tight")
print(type(pc).__name__)
```
実行結果:
```
PlotCollection
```

**注意点・落とし穴**:
- `combined=False`(既定)はチェーンごとに1本ずつ区間を描く(`theta` の各school行に4本の横線。目視確認)。`combined=True` でチェーンを統合した1本になる。`ci_probs` は `(内側=trunk, 外側=twig)` の2要素で、既定は `(0.5, rcParams['stats.ci_prob'])`。
- `dict` を渡すと(`{"linear": dt, "null": dt0}`)モデルごとに色分けして並べられる(目視確認: `a` と `sigma` の各行に、モデルごとの青・橙の区間が描かれ、`null` の `sigma`(約1.6)が `linear`(約0.4)より大きく右に離れた)。
- `plot_ridge` は school ごとの密度を縦にずらして塗りつぶし、左側に school 名(座標)のラベルが出る(目視確認)。

### `arviz.plot_pair(...)` / `arviz.plot_pair_focus(...)`

**用途**: 変数の組ごとの散布図と周辺分布の行列(`plot_pair`、戻り値は `PlotMatrix`)/ 1つの変数(`focus_var`)と他変数の散布図(`plot_pair_focus`)。

**シグネチャ**: `plot_pair(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, levels=None, marginal=True, marginal_kind=None, triangle='lower', plot_matrix=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_pair_focus(dt, focus_var, *, focus_var_coords=None, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, **pc_kwargs)`

**使用例**:
```python
pm = az.plot_pair(c8, var_names=["mu", "tau", "theta"], coords={"school": ["Choate", "Deerfield"]})
pm.savefig(IMG + "/pair.png", bbox_inches="tight")
pm = az.plot_pair(c8, var_names=["mu", "tau"], visuals={"divergence": True})
pm.savefig(IMG + "/pair_div.png", bbox_inches="tight")
pc = az.plot_pair_focus(c8, "tau", var_names=["mu", "tau", "theta"], visuals={"divergence": True})
pc.savefig(IMG + "/pair_focus.png", bbox_inches="tight")
print(type(pm).__name__, type(pc).__name__)
```
実行結果:
```
PlotMatrix PlotCollection
```

**注意点・落とし穴**:
- 既定では**下三角のみ**(`triangle='lower'`)に散布図、対角に周辺密度を描く。ダイバージェンスは**既定では表示されない**(`visuals` の `divergence` は既定 `False`)ので、`visuals={"divergence": True}` を指定する。指定すると、下三角の `mu`-`tau` 散布図の下部(`tau` が小さい領域)にオレンジの点が集まる(目視確認。漏斗の首でのダイバージェンス)。
- 散布図に等高線を重ねるには `visuals={"contour": True}`(既定 `False`)、`levels` で確率質量を指定する(等高線の描画そのものは未確認)。`marginal_kind` は `'kde'`/`'hist'`/`'ecdf'`/`'dot'`。
- `plot_pair_focus(dt, 'tau', ...)` は `tau` を縦軸に固定し、他の変数(`mu`, `tau` 自身, 各 `theta`)を横軸にした散布図を格子状に並べる(`tau` 自身との組は対角線状の点列になる。目視確認)。`focus_var` は位置引数(2番目)。

### `arviz.plot_parallel(...)`

**用途**: 全変数を横軸に並べ、各 draw を折れ線で結ぶ平行座標プロット。ダイバージェンスした draw が別色で重なり、漏斗状の領域などの検出に使う。

**シグネチャ**: `plot_parallel(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, norm_method=None, label_type='flat', plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_parallel(c8, var_names=["mu", "tau", "theta"])
pc.savefig(IMG + "/parallel.png", bbox_inches="tight")
print(type(pc).__name__)
```
実行結果:
```
PlotCollection
```

**注意点・落とし穴**:
- 目視確認: x軸に `mu, tau, theta[Choate], ...` が並び、薄い青の大量の折れ線の上に、ダイバージェンスした draw が橙色の線として重なる。`tau` の位置で線が細く絞られる(漏斗の首)のが見える。`norm_method`(正規化方法)は未検証。

### `arviz.plot_rank(...)` / `arviz.plot_rank_dist(...)`

**用途**: チェーン間の混合を、分数ランクのΔ-ECDF(ECDFと一様分布の差)で確認する。`plot_rank_dist` は周辺密度と並べる。

**シグネチャ**: `plot_rank(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, envelope_prob=None, method='mtc_c', thin=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_rank_dist(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, compact=True, combined=False, kind=None, envelope_prob=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_rank(dt, var_names=["a", "b"])
pc.savefig(IMG + "/rank.png", bbox_inches="tight")
pc = az.plot_rank_dist(dt, var_names=["a", "b"])
pc.savefig(IMG + "/rank_dist.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'dist', 'ecdf_lines', 'plot', 'row_index', 'suspicious_points', 'xlabel_rank']
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: 0.x の rank plot(チェーンごとのランクのヒストグラム棒)ではなく、**チェーンごとのΔ-ECDF曲線**(x軸 `Fractional ranks`、0付近で揺れる階段状の線)を描く。混合が良ければ全チェーンの線が0付近の帯に収まる。左上に `p=0.66(α=0.01)` のような多重検定のp値が注記された(既定 `method='mtc_c'`)。
- `method='envelope'` にすると信頼包絡線が描かれる(docstring。未検証)。

### `arviz.plot_autocorr(...)`

**用途**: チェーンごとの自己相関(ラグ0〜)。ゆっくり減衰するほど有効サンプルが少ない。

**シグネチャ**: `plot_autocorr(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, max_lag=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_autocorr(dt, var_names=["a", "b"])
pc.savefig(IMG + "/autocorr.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'credible_interval', 'lines', 'plot', 'ref_line', 'row_index', 'title', 'xlabel']
```

**注意点・落とし穴**:
- 目視確認: x軸 `Lag`(0〜99)、ラグ0で1から急落して、以降は灰色の帯(信頼帯 ±0.2程度)の中で0付近を揺れる4本の線(チェーンごとの色)。`max_lag` を省略した上の例ではラグ 0〜99 が描かれた。

### `arviz.plot_ess(...)` / `arviz.plot_ess_evolution(...)` / `arviz.plot_mcse(...)`

**用途**: ESSを分位点ごと(`plot_ess`)・総draw数ごと(`plot_ess_evolution`)に、またMCSEを分位点ごとに(`plot_mcse`)可視化する。

**シグネチャ**: `plot_ess(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, kind='local', relative=False, rug=False, rug_kind='diverging', n_points=20, extra_methods=False, min_ess=400, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_ess_evolution(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, relative=False, n_points=20, extra_methods=False, min_ess=400, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_mcse(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, rug=False, rug_kind='diverging', n_points=20, extra_methods=False, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_ess(dt, var_names=["a", "b"])
pc.savefig(IMG + "/ess.png", bbox_inches="tight")
pc = az.plot_ess_evolution(dt, var_names=["a", "b"])
pc.savefig(IMG + "/ess_evol.png", bbox_inches="tight")
pc = az.plot_mcse(dt, var_names=["a", "b"])
pc.savefig(IMG + "/mcse.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'mcse', 'plot', 'row_index', 'title', 'xlabel', 'ylabel']
```

**注意点・落とし穴**:
- `plot_ess` の `kind` は `'local'`(既定、区間確率の局所効率) / `'quantile'`(分位点の効率)。目視確認: x軸 `Quantile`、y軸 `ESS` の点が20個並び、破線+点の水平線が `min_ess=400` を示す(全点がこの線の上、約1350〜2100)。
- `plot_ess_evolution`: x軸 `Total Number of Draws` 100〜2000、y軸 `ESS`。青(bulk)とオレンジ(tail)の2本の折れ線が右上がりに伸び、水平の `min_ess` 線(400)を、bulkは約400 draw付近、tailは約600 draw付近で超える(目視確認)。収束確認は「ESSがdraw数に比例して増えているか」で見る。
- `plot_mcse`: x軸 `Quantile`、y軸 `mcse`。分位点ごとのMCSEが、`a` では両端が高く中央が低いU字状(0.001〜0.003)になった(目視確認)。

### `arviz.plot_energy(...)`

**用途**: HMCのエネルギー分布(周辺エネルギー vs エネルギー遷移)の重なりと、E-BFMIをチェーンごとに描く。

**シグネチャ**: `plot_energy(dt, *, sample_dims=None, kind=None, show_bfmi=True, threshold=0.3, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_energy(c8)
pc.savefig(IMG + "/energy.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['bfmi_points', 'dist', 'face', 'legend', 'ref_line', 'title', 'ylabel']
```

**注意点・落とし穴**:
- 目視確認: 左に `BFMI` パネル(チェーン番号 vs BFMI値の点、`threshold=0.3` の破線。チェーン0が約0.287で破線の左=閾値未満)、右に `marginal`(青)と `transition`(橙)のエネルギー密度が重ねて描かれ、遷移分布(橙)が周辺分布(青)より明確に狭かった。両者の広がりが大きくずれるとエネルギー探索の問題を示唆する。
- `sample_stats` に `energy` が必要。`show_bfmi=False` で左パネルを消せる。

### `arviz.plot_convergence_dist(...)`

**用途**: 全変数の ESS(bulk/tail)と R-hat の**値の分布**(ECDF)を1枚にまとめ、多数のパラメータがあるモデルの収束を俯瞰する。

**シグネチャ**: `plot_convergence_dist(dt, *, var_names=None, filter_vars=None, group='posterior', coords=None, sample_dims=None, diagnostics=None, grouped=True, ref_line=True, kind='ecdf', point_estimate=None, ci_kind=None, ci_prob=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_convergence_dist(dt, var_names=["a", "b", "sigma"])
pc.savefig(IMG + "/convergence_dist.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'dist', 'plot', 'ref_line', 'row_index', 'title']
```

**注意点・落とし穴**:
- 目視確認: `ess_bulk` / `ess_tail` / `rhat` の3パネルが横に並び、それぞれ3変数分のECDF階段(y軸は0.33〜1)と、しきい値の破線(ESSは400、rhatは1.01)が描かれた。このデータでは ESS は400より十分右、rhat は 1.01 より左にあり、全変数が基準内。
- `diagnostics` 引数で対象の診断量を選べる(既定は3種。`diagnostics` の選択肢は未検証)。

### `arviz.plot_prior_posterior(...)`

**用途**: 事前分布と事後分布の周辺密度を重ねて描く(事前分布から事後分布への更新の確認)。

**シグネチャ**: `plot_prior_posterior(dt, *, var_names=None, filter_vars=None, group=None, coords=None, sample_dims=None, kind=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_prior_posterior(dt, var_names=["a", "b"])
pc.savefig(IMG + "/prior_posterior.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'dist', 'legend', 'plot', 'row_index', 'title']
```

**注意点・落とし穴**:
- `prior` と `posterior` の両グループが必要(`group` を省略すると両方が描かれた)。目視確認: 凡例 `group`(`prior`=青、`posterior`=橙)。事前 N(0,5) は非常に平坦で、事後は狭いスパイクになるためスケールが合わず、事前は横に長い低い線に見える。
- 事前分布のドロー数は少ない(この例は200)とKDEが粗くなる。`pm.sample_prior_predictive` のドロー数を増やす。

### `arviz.plot_psense_dist(...)` / `arviz.plot_psense_quantities(...)`

**用途**: べき乗スケーリング感度の可視化。`plot_psense_dist` はスケール係数ごとの事後密度、`plot_psense_quantities` は統計量(mean, sd など)の係数依存性。

**シグネチャ**: `plot_psense_dist(dt, *, var_names=None, filter_vars=None, prior_var_names=None, likelihood_var_names=None, prior_coords=None, likelihood_coords=None, coords=None, sample_dims=None, alphas=None, kind=None, point_estimate=None, ci_kind=None, ci_prob=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_psense_quantities(dt, *, var_names=None, filter_vars=None, prior_var_names=None, likelihood_var_names=None, prior_coords=None, likelihood_coords=None, coords=None, sample_dims=None, alphas=None, quantities=None, mcse=True, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_psense_dist(dt, var_names=["a", "b"])
pc.savefig(IMG + "/psense_dist.png", bbox_inches="tight")
pc = az.plot_psense_quantities(dt, var_names=["a", "b"])
pc.savefig(IMG + "/psense_q.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'legend', 'likelihood_lines', 'likelihood_markers', 'mcse', 'plot', 'prior_lines', 'prior_markers', 'row_index', 'title', 'xlabel']
```

**注意点・落とし穴**:
- `log_prior` と `log_likelihood` が必要(`az.psense` 参照)。目視確認: `plot_psense_dist` は各変数について `prior` 列・`likelihood` 列に密度を描き、凡例 `Power Scale Factor`(0.8=青、1.0=黒、1.25=橙)。`prior` 列は3本がほぼ重なる(事前に鈍感)が、`likelihood` 列は係数で山の高さが変わる(尤度に敏感)。
- `plot_psense_quantities` は `mean` / `sd` ごとのパネルで、x軸が `Power-scaling α`(0.8, 1, 1.25)、青が prior・橙が likelihood のスケーリング。`sd` は尤度側の線が大きく右下がりになる。上下に水平の破線が2本描かれる(意味はここでは未確認)。多数の図を開くと matplotlib が `More than 20 figures have been opened` の警告を出す場合がある。

### `arviz.plot_bf(...)`

**用途**: Savage–Dickey ベイズファクターの可視化(事前・事後の密度と参照値の縦線、BF10の注記)。

**シグネチャ**: `plot_bf(dt, var_names, *, sample_dims=None, ref_val=0, kind=None, bf_type='BF10', plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_bf(dt, var_names="a", ref_val=1.0)
pc.savefig(IMG + "/bf.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'dist', 'legend', 'plot', 'ref_line', 'ref_value_text', 'row_index', 'title']
```

**注意点・落とし穴**:
- 目視確認: 事前(青、非常に平坦)と事後(橙、鋭いスパイク)が重なり、`ref_val=1.0` の縦線と `BF10=0.05` の注記が描かれた。`var_names` は必須で、`az.bayes_factor` と同じ計算に基づく。

---

## 9. 可視化(事後予測・モデル比較)

### `arviz.plot_ppc_dist(...)` / `arviz.plot_ppc_dist_pit(...)`

**用途**: 事後予測チェック。予測分布(事後予測サンプルの一部)と観測データの分布を重ねる(0.x の `plot_ppc` に相当)。`plot_ppc_dist_pit` は右にPIT Δ-ECDFを追加する。

**シグネチャ**: `plot_ppc_dist(dt, *, var_names=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, kind=None, num_samples=50, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_ppc_dist_pit(dt, *, var_names=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, kind=None, num_samples=50, method='pot_c', envelope_prob=None, coverage=False, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_ppc_dist(dt, var_names=["y"], num_samples=30)
pc.savefig(IMG + "/ppc_dist.png", bbox_inches="tight")
pc = az.plot_ppc_dist(dt, var_names=["y"], num_samples=30, kind="kde")
pc.savefig(IMG + "/ppc_kde.png", bbox_inches="tight")
pc = az.plot_ppc_dist_pit(dt, var_names=["y"], num_samples=30)
pc.savefig(IMG + "/ppc_dist_pit.png", bbox_inches="tight")
print(sorted(pc.viz.children), hasattr(az, "plot_ppc"))
```
実行結果:
```
['col_index', 'ecdf_lines', 'observed_dist', 'plot', 'predictive_dist', 'ref_line', 'row_index', 'suspicious_points', 'title', 'xlabel', 'ylabel'] False
```

**注意点・落とし穴**:
- **バージョン固有の注意(実測で確認)**: `az.plot_ppc` は存在しない(`plot_ppc_dist` に改名)。`posterior_predictive` と `observed_data` の両グループが必要で、`num_samples` の既定は50(署名) / docstring上は100と書かれており不一致だが、署名の値が実際の既定。
- `kind='auto'`(既定)は**連続データではECDF**になる(目視確認: 青い薄い階段状のECDFが30本、その上に観測データの黒いECDFが重なる)。密度を見たいときは `kind='kde'` を明示する。
- `plot_ppc_dist_pit` は左にこの予測分布vs観測、右に `PIT`(x軸)・`Δ ECDF`(y軸)の図を並べる(目視確認。`p=0.48(α=0.01)` の注記つき)。

### `arviz.plot_ppc_pit(...)` / `arviz.plot_loo_pit(...)`

**用途**: 予測分布の較正をPIT値のΔ-ECDFで確認する。PIT値が一様なら較正が取れており、曲線は0付近の帯に収まる。`plot_loo_pit` は LOO 版(`log_likelihood` が必要)。

**シグネチャ**: `plot_ppc_pit(dt, *, var_names=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, method='pot_c', envelope_prob=None, coverage=False, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_loo_pit(dt, *, var_names=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, method='pot_c', envelope_prob=None, coverage=False, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_ppc_pit(dt, var_names=["y"])
pc.savefig(IMG + "/ppc_pit.png", bbox_inches="tight")
pc = az.plot_loo_pit(dt, var_names=["y"])
pc.savefig(IMG + "/loo_pit.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'ecdf_lines', 'plot', 'ref_line', 'row_index', 'suspicious_points', 'title', 'xlabel', 'ylabel']
```

**注意点・落とし穴**:
- 目視確認: x軸 `PIT`(または `LOO-PIT`)、y軸 `Δ ECDF`。0の破線の周りを揺れる青い階段状の線で、±0.06程度の範囲に収まり、左上に `p=0.48(α=0.01)` が注記された。ここでは2つの図がほぼ同じ形になった(この例は予測が良く較正されているため)。
- `plot_ecdf_pit`(既定 `group='prior_sbc'`)に事後予測グループを与えると観測点の数だけサブプロットを作ろうとして `plot.max_subplots` 超過のエラーになる(共通引数の節参照)。

### `arviz.plot_ppc_tstat(...)`

**用途**: 検定統計量(平均・中央値・SDなど)について、事後予測サンプルでの分布と観測データの値を比べる(0.x の `plot_bpv` に近い用途)。

**シグネチャ**: `plot_ppc_tstat(dt, *, var_names=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, t_stat='median', kind=None, point_estimate=None, ci_kind=None, ci_prob=None, data_pairs=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_ppc_tstat(dt, var_names=["y"], t_stat="std")
pc.savefig(IMG + "/ppc_tstat.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'dist', 'observed_tstat', 'plot', 'row_index', 'title']
```

**注意点・落とし穴**:
- `t_stat` の既定は `'median'`。目視確認: タイトルが `y (std)` で、予測サンプルのSDの分布(滑らかな密度)の底に、観測データのSDが黒い点で打たれた。点が分布の中心付近にあれば、その統計量については観測が予測と整合している。
- `plot_bpv` は存在しない(`plot_ppc_tstat` や `plot_ppc_pit` を使う)。

### `arviz.plot_ppc_interval(...)` / `arviz.plot_loo_interval(...)`

**用途**: 観測点ごとに予測区間(太線=内側、細線=外側)を描き、観測値(黒点)と点推定(青点)を重ねる。`plot_loo_interval` はLOO版。

**シグネチャ**: `plot_ppc_interval(dt, *, var_names=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, point_estimate=None, ci_kind=None, ci_probs=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)` / `plot_loo_interval(dt, *, var_names=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, point_estimate=None, ci_kind=None, ci_probs=None, plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_ppc_interval(dt, var_names=["y"])
pc.savefig(IMG + "/ppc_interval.png", bbox_inches="tight")
pc = az.plot_loo_interval(dt, var_names=["y"])
pc.savefig(IMG + "/loo_interval.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['col_index', 'observed_markers', 'plot', 'prediction_markers', 'row_index', 'trunk', 'twig', 'xlabel', 'ylabel']
```

**注意点・落とし穴**:
- 目視確認: x軸 `data point (index)`(0〜59)、y軸 `y`。各点に青い縦の区間、青い点(予測の点推定)、黒い小点(観測値)が描かれ、観測の黒点がほとんど区間内に入っていた。区間から外れる観測が多いほど予測が不足(過小分散)。`ci_probs` で内側/外側の区間確率を指定できる。
- x軸は座標値ではなく観測点の通し番号(`data point (index)`)で、`dims`/`coords` のラベルは使われない。

### `arviz.plot_ppc_rootogram(...)`

**用途**: **離散(カウント)データ**の事後予測チェック。各カウント値の頻度を、予測(点と区間)と観測(黒点)で比べるルートグラム。

**シグネチャ**: `plot_ppc_rootogram(dt, *, var_names=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, ci_prob=None, point_estimate=None, yscale='sqrt', plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, **pc_kwargs)`

**使用例**:
```python
rng = np.random.default_rng(3)
lam = rng.gamma(20, 0.5, size=(4, 200, 1))
yrep = rng.poisson(np.broadcast_to(lam, (4, 200, 40)))
y_cnt = rng.poisson(10, size=40)
dtp = az.from_dict({"posterior": {"lam": lam[..., 0]},
                    "posterior_predictive": {"y": yrep},
                    "observed_data": {"y": y_cnt}})
pc = az.plot_ppc_rootogram(dtp, var_names=["y"])
pc.savefig(IMG + "/rootogram.png", bbox_inches="tight")
try:
    az.plot_ppc_rootogram(dt, var_names=["y"])
except ValueError as e:
    print("ValueError:", str(e).splitlines()[0])
```
実行結果:
```
ValueError: Variables y in 'observed' are continuous.
```

**注意点・落とし穴**:
- **実測**: 連続データ(`dt` の `y`)を渡すと `ValueError: Variables y in 'observed' are continuous.` になる(離散データ専用)。
- 目視確認: x軸 `counts`、y軸 `frequency`(既定 `yscale='sqrt'` で、目盛が 0,2,4,6,8 の平方根スケール)。予測の頻度が菱形マーカーと薄い青の縦バー(区間)で、観測の頻度が黒点で描かれ、黒点が概ねバーの範囲に入っていた。

### `arviz.plot_compare(...)`

**用途**: `az.compare` の結果を可視化する。各モデルのelpd差と標準誤差を点とエラーバーで描く。

**シグネチャ**: `plot_compare(cmp_df, *, relative_scale=True, rotated=False, hide_top_model=False, backend=None, visuals=None, **pc_kwargs)`

**使用例**:
```python
cmp = az.compare({"linear": dt, "null": dt0})
pc = az.plot_compare(cmp)
pc.savefig(IMG + "/compare.png", bbox_inches="tight")
print(type(pc).__name__, sorted(pc.viz.children))
```
実行結果:
```
PlotCollection []
```

**注意点・落とし穴**:
- 引数は `compare` の戻り値の `DataFrame`(第1引数名は `cmp_df`)。目視確認: タイトル `Model comparison / higher is better`、x軸 `ELPD (difference)`。最良モデル(`linear`)が x=0 の黒点と縦の破線、`null` が約-80 にエラーバー付きの黒点(区間 約-87〜-73)として描かれた。
- `relative_scale=True`(既定)は差(最良との相対値)、`False` で絶対elpdになる(後者は未検証)。

### `arviz.plot_khat(...)`

**用途**: PSIS-LOOのPareto k診断値を観測点ごとにプロットし、信頼できない点(k>0.7)を探す。

**シグネチャ**: `plot_khat(elpd_data, *, threshold=None, hover_format='{index}: {label}', legend=None, color=None, marker=None, hline_values=None, bin_format='{pct:.1f}%', plot_collection=None, backend=None, labeller=None, aes_by_visuals=None, visuals=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_khat(az.loo(dt))
pc.savefig(IMG + "/khat.png", bbox_inches="tight")
print(type(pc).__name__)
```
実行結果:
```
PlotCollection
```

**注意点・落とし穴**:
- 第1引数は `DataTree` ではなく `az.loo` の戻り値(`ELPDData`)。`DataTree`(`dt`)や `pointwise=False` の `ELPDData` を渡すと `ValueError: Could not find 'pareto_k' in the ELPDData object...` になる(実測)。目視確認: x軸 `Data Point`、y軸 `Shape parameter k` に60点が散布され、全点が約 -0.1〜0.23 の範囲(0.7を大きく下回る)。この例ではしきい値の水平線は描かれていなかった。

### `arviz.plot_lm(...)`

**用途**: 回帰系のデータの平均線と事後予測の信頼帯を、説明変数の関数として描き、観測値の散布図を重ねる。

**シグネチャ**: `plot_lm(dt, *, x=None, y=None, y_obs=None, plot_dim=None, filter_vars=None, group='posterior_predictive', coords=None, sample_dims=None, smooth=True, ci_kind=None, ci_prob=None, point_estimate=None, plot_collection=None, backend=None, xlabeller=None, ylabeller=None, aes_by_visuals=None, visuals=None, stats=None, **pc_kwargs)`

**使用例**:
```python
pc = az.plot_lm(dt, x="x", y="y")
pc.savefig(IMG + "/lm.png", bbox_inches="tight")
print(sorted(pc.viz.children))
```
実行結果:
```
['ci_band', 'col_index', 'observed_scatter', 'pe_line', 'plot', 'row_index', 'xlabel', 'ylabel']
```

**注意点・落とし穴**:
- `x` は `constant_data` グループの変数名(`pm.Data("x", ...)`)、`y` は `observed_data` の変数名。PyMCで説明変数を `pm.Data` にし、`dims` を付けておくと使える。
- 目視確認: x軸 `x`、y軸 `y`。灰色の観測点が右上がりに並び、その上に黒い平均線と、線を包む青い帯(信用区間)が描かれた。

---

## 付録. 存在しない名前・未検証の名前

### 0.x に存在し 1.3.0 に存在しない名前(実測)

**用途**: 0.x の記憶で書いたコードが動かない原因を素早く特定するため、`dir(az)` に無い名前を実測で洗い出した一覧。

**シグネチャ**: `dir(arviz)`

**使用例**:
```python
legacy = ["InferenceData", "from_pymc", "from_pystan", "from_pyro", "from_cmdstan", "concat",
          "to_netcdf", "waic", "psislw", "r2_score", "autocorr", "make_ufunc", "apply_ufunc",
          "plot_posterior", "plot_density", "plot_violin", "plot_kde", "plot_hdi", "plot_joint",
          "plot_ppc", "plot_bpv", "plot_elpd", "plot_dist_comparison", "plot_separation", "plot_bpv",
          "plot_forest", "plot_pair", "plot_trace", "plot_rank", "plot_khat", "plot_loo_pit", "plot_compare"]
print("無い:", sorted({n for n in legacy if n not in dir(az)}))
print("ある:", sorted({n for n in legacy if n in dir(az)}))
```
実行結果:
```
無い: ['InferenceData', 'apply_ufunc', 'autocorr', 'concat', 'from_cmdstan', 'from_pymc', 'from_pyro', 'from_pystan', 'make_ufunc', 'plot_bpv', 'plot_density', 'plot_dist_comparison', 'plot_elpd', 'plot_hdi', 'plot_joint', 'plot_kde', 'plot_posterior', 'plot_ppc', 'plot_separation', 'plot_violin', 'psislw', 'r2_score', 'to_netcdf', 'waic']
ある: ['plot_compare', 'plot_forest', 'plot_khat', 'plot_loo_pit', 'plot_pair', 'plot_rank', 'plot_trace']
```

**注意点・落とし穴**:
- `az.InferenceData` は `dir(az)` には出ないが、属性アクセスすると `MigrationWarning` 付きで `xarray.DataTree` を返す特別扱い(`__getattr__`)。上の「無い」に含まれるのはこのため。
- 「ある」側の `plot_forest` などは名前は残っているが**戻り値・引数が変わっている**(可視化の各節参照)。置き換え先: `plot_posterior`→`plot_dist`、`plot_ppc`→`plot_ppc_dist`、`plot_density`/`plot_violin`→`plot_dist`/`plot_forest`/`plot_ridge`、`plot_bpv`→`plot_ppc_tstat`/`plot_ppc_pit`(用途が近いもの)、`waic`→`loo`。

### この辞書で詳しく扱っていない 1.3.0 の公開名(存在確認のみ)

**用途**: 本辞書でコード実行による検証を行っていない公開APIの一覧。存在は `dir(az)` で確認済みだが、挙動・戻り値は検証していないので使う前に自分で確かめること。

**シグネチャ**: `dir(arviz)`

**使用例**:
```python
extra = ["loo_kfold", "reloo", "loo_moment_match", "loo_subsample", "loo_approximate_posterior",
         "loo_influence", "update_subsample", "SamplingWrapper", "kaplan_meier", "generate_survival_curves",
         "plot_ppc_censored", "plot_ppc_pava", "plot_ppc_pava_residuals", "plot_loo_pava",
         "plot_dgof", "plot_dgof_dist", "plot_ecdf_pit", "get_unconstrained_samples",
         "references_to_dataset", "citations", "explode_dataset_dims", "from_numpyro_svi",
         "MCMCAdapter", "SVIAdapter", "NumPyroInferenceAdapter", "PlotMatrix", "xarray_var_iter"]
print([n for n in extra if n in dir(az)] == extra)
```
実行結果:
```
True
```

**注意点・落とし穴**:
- `loo_kfold` / `reloo` / `loo_moment_match` / `loo_subsample` / `SamplingWrapper` は、署名(`wrapper`・`k`・`folds` などの引数)から見て `az.loo` の代替(K分割CV・再フィット・大規模データ向けの部分計算)と思われるが、モデルの再フィット処理を自作する必要があり、本辞書では動作を検証していない。
- `plot_ppc_pava`(と `plot_loo_pava`)は PAVA(等調回帰)による較正プロットで、`data_type='binary'` が既定。連続データでも図は出たが(目視で点線の対角線と青い帯を確認)、解釈は未検証。`plot_dgof` はΔ-ECDF-PITの診断図で、`plot_dgof(dt, var_names=['a'])` が動作し、`p=0.44(α=0.01)` が注記されたことのみ確認した。
