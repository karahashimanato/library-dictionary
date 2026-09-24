# xarray 逆引き辞書

xarray 2026.2.0 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認)。

本ドキュメントの実行結果は、`xarray 2026.2.0` / `numpy 2.4.6` / `pandas 3.0.5` / `scipy 1.18.1` / `h5netcdf 1.8.1` の環境で得たものです。**`netCDF4`・`zarr`・`dask`・`bottleneck` は未導入**のため、それらに依存する機能(`to_zarr`、`chunks=`/`open_mfdataset`、`ffill`/`bfill` など)は実行例を載せず、エラーになることの確認結果のみ記載しています。

各コード例は独立して実行できるようにしてあり、冒頭で以下を import 済みとします。

```python
import numpy as np
import pandas as pd
import xarray as xr
```

2026.2.0 で特に注意すべき挙動(いずれも本文で実行して確認)は以下のとおりです。

- **`attrs` が演算・集約の結果に引き継がれる**(`xr.get_options()["keep_attrs"]` の既定が `"default"`)。古い解説記事の「演算で attrs が消える」とは異なる。
- **`concat`/`merge` で座標が食い違うと `FutureWarning`**(将来 `join="outer"` から `"exact"` に、`compat` も変更予定)。`join=` を明示するのが安全。
- **`ffill`/`bfill`/`interpolate_na(max_gap=...)` は `bottleneck` が必要**。
- **`pd.date_range` が `datetime64[us]` を返す**(pandas 3.0)ため、`interp` に時刻を文字列で渡すと NaN になる。

## 目次

1. [DataArray/Dataset基礎](#dataarraydataset基礎)
2. [選択](#選択)
3. [演算・ブロードキャスト](#演算ブロードキャスト)
4. [集約](#集約)
5. [結合・整列](#結合整列)
6. [変形](#変形)
7. [欠損値](#欠損値)
8. [時系列・補間](#時系列補間)
9. [pandas連携](#pandas連携)
10. [入出力](#入出力)

---

## DataArray/Dataset基礎

### `xr.DataArray(...)`

**用途**: ラベル付き N 次元配列(データ + 次元名 `dims` + 座標 `coords` + 属性 `attrs`)を作る、xarray の中心クラス。

**シグネチャ**: xr.DataArray(data=<NA>, coords=None, dims=None, name=None, attrs=None, indexes=None, fastpath=False)

**使用例**:
```python
import numpy as np
import pandas as pd
import xarray as xr

da = xr.DataArray(
    np.arange(6).reshape(2, 3),
    coords={"y": ["a", "b"], "x": [10, 20, 30]},
    dims=("y", "x"),
    name="temp",
    attrs={"units": "degC"},
)
print(da)
```
実行結果:
```
<xarray.DataArray 'temp' (y: 2, x: 3)> Size: 48B
array([[0, 1, 2],
       [3, 4, 5]])
Coordinates:
  * y        (y) <U1 8B 'a' 'b'
  * x        (x) int64 24B 10 20 30
Attributes:
    units:    degC
```

**注意点・落とし穴**:
- `dims` を省略して `coords` を dict で渡すと、dict のキー順が次元名として使われる。dims を明示するほうが安全。
- 座標を持たない次元も作れる(`dims=("y", "x")` だけ渡した場合)。repr では `Dimensions without coordinates: y` と表示される。そうした次元に対する `sel(x=1)` は位置指定(`isel`)と同じ動きになる。
- 本書の以降のコードは、上記の `import` を済ませている前提(各例は独立して実行できるようにしてある)。

### `xr.Dataset(...)`

**用途**: 共通の座標・次元を共有する複数の DataArray(変数)を、辞書のように束ねたコンテナ。

**シグネチャ**: xr.Dataset(data_vars=None, coords=None, attrs=None)

**使用例**:
```python
ds = xr.Dataset(
    {
        "temp": (("y", "x"), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "pres": (("y", "x"), [[10, 20, 30], [40, 50, 60]]),
        "elev": ("x", [100, 200, 300]),
    },
    coords={"y": ["a", "b"], "x": [10, 20, 30]},
    attrs={"title": "demo"},
)
print(ds)
```
実行結果:
```
<xarray.Dataset> Size: 152B
Dimensions:  (y: 2, x: 3)
Coordinates:
  * y        (y) <U1 8B 'a' 'b'
  * x        (x) int64 24B 10 20 30
Data variables:
    temp     (y, x) float64 48B 1.0 2.0 3.0 4.0 5.0 6.0
    pres     (y, x) int64 48B 10 20 30 40 50 60
    elev     (x) int64 24B 100 200 300
Attributes:
    title:    demo
```

**注意点・落とし穴**:
- data_vars の値は `(dims, data)` のタプル、`(dims, data, attrs)`、または DataArray で与える。
- 変数ごとに持つ次元が違ってよい(上の `elev` は `x` のみ)。Dataset 全体の `sizes` は全次元の和集合になる。

### `da.dims / da.shape / da.sizes / da.coords / da.attrs / da.values / da.name`

**用途**: DataArray の構成要素(次元名・サイズ・座標・属性・中身の numpy 配列・名前)を取り出す。

**シグネチャ**: 属性(プロパティ)であり関数ではない

**使用例**:
```python
da = xr.DataArray(
    np.arange(6.0).reshape(2, 3),
    coords={"y": ["a", "b"], "x": [10, 20, 30]},
    dims=("y", "x"),
    name="temp",
    attrs={"units": "degC"},
)
print(da.dims)
print(da.shape)
print(dict(da.sizes))
print(da.name, da.dtype)
print(da.attrs)
print(da.values)
print(type(da.values))
print(da.coords)
print(da["x"].values)
print(da.indexes["x"])
```
実行結果:
```
('y', 'x')
(2, 3)
{'y': 2, 'x': 3}
temp float64
{'units': 'degC'}
[[0. 1. 2.]
 [3. 4. 5.]]
<class 'numpy.ndarray'>
Coordinates:
  * y        (y) <U1 8B 'a' 'b'
  * x        (x) int64 24B 10 20 30
[10 20 30]
Index([10, 20, 30], dtype='int64', name='x')
```

**注意点・落とし穴**:
- `da.dims` はタプル、`da.sizes` は次元名→長さの読み取り専用マッピング(`dict()` で変換すると見やすい)。
- `da.values` は numpy 配列(遅延配列なら計算が走る)。`da.to_numpy()` でも同じ。座標は `da["x"]`(DataArray)か `da.indexes["x"]`(pandas.Index)で取れる。

### `ds["var"] / ds.var / ds[["a", "b"]] / ds.data_vars`

**用途**: Dataset から変数を DataArray として、または複数変数を部分 Dataset として取り出す。

**シグネチャ**: Dataset の `__getitem__` とプロパティ

**使用例**:
```python
ds = xr.Dataset(
    {"temp": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]), "pres": (("y", "x"), [[10, 20], [30, 40]])},
    coords={"y": ["a", "b"], "x": [10, 20]},
)
print(list(ds.data_vars))
print(list(ds.coords))
print(ds["temp"])
print(ds[["pres"]])
```
実行結果:
```
['temp', 'pres']
['y', 'x']
<xarray.DataArray 'temp' (y: 2, x: 2)> Size: 32B
array([[1., 2.],
       [3., 4.]])
Coordinates:
  * y        (y) <U1 8B 'a' 'b'
  * x        (x) int64 16B 10 20
<xarray.Dataset> Size: 56B
Dimensions:  (y: 2, x: 2)
Coordinates:
  * y        (y) <U1 8B 'a' 'b'
  * x        (x) int64 16B 10 20
Data variables:
    pres     (y, x) int64 32B 10 20 30 40
```

**注意点・落とし穴**:
- `ds["temp"]` は DataArray、`ds[["pres"]]`(リスト指定)は Dataset を返す。混同しやすい。
- 属性アクセス(`ds.temp`)は、Dataset 自身の属性名と衝突する変数名では使えない。例えば変数名が `sizes` だと `ds.sizes` は変数ではなくプロパティ(次元サイズ)を返す。スクリプトでは `ds["temp"]` が安全。

### `da.assign_coords(...)`

**用途**: 座標を追加・置換した新しい DataArray を返す(元は変更されない)。次元座標だけでなく、次元に沿った補助座標(非次元座標)も付けられる。

**シグネチャ**: da.assign_coords(coords=None, **coords_kwargs)

**使用例**:
```python
da = xr.DataArray([1, 2, 3], dims="x", coords={"x": [0, 1, 2]})
da2 = da.assign_coords(x=[10, 20, 30], label=("x", ["p", "q", "r"]))
print(da2)
print(da)
print(da2.sel(label="q").values)
```
実行結果:
```
<xarray.DataArray (x: 3)> Size: 24B
array([1, 2, 3])
Coordinates:
  * x        (x) int64 24B 10 20 30
    label    (x) <U1 12B 'p' 'q' 'r'
<xarray.DataArray (x: 3)> Size: 24B
array([1, 2, 3])
Coordinates:
  * x        (x) int64 24B 0 1 2
2
```

**注意点・落とし穴**:
- 非次元座標 `label` は `sel` の対象にならない(インデックスがないため)。検索に使うなら `set_index(x="label")` で次元座標に昇格させる(「変形」の `set_index` を参照)。
- `da.coords["x"] = ...` のように in-place でも書き換えられるが、副作用を避けるなら `assign_coords` が無難。

### `da.assign_attrs(...) / da.rename(...)`

**用途**: 属性(`attrs`)を追加、または変数・座標・次元の名前を変更した新しいオブジェクトを返す。

**シグネチャ**: da.assign_attrs(*args, **kwargs) / da.rename(new_name_or_name_dict=None, **names)

**使用例**:
```python
da = xr.DataArray([1, 2, 3], dims="x", coords={"x": [0, 1, 2]}, name="v")
da2 = da.assign_attrs(units="m", source="test")
print(da2.attrs)
print((da2 + 1).attrs)                       # 演算後も attrs が残る(2026.2.0 の既定)
with xr.set_options(keep_attrs=False):
    print((da2 + 1).attrs)                   # keep_attrs=False で消える

r = da.rename("height").rename({"x": "position"})
print(r)
```
実行結果:
```
{'units': 'm', 'source': 'test'}
{'units': 'm', 'source': 'test'}
{}
<xarray.DataArray 'height' (position: 3)> Size: 24B
array([1, 2, 3])
Coordinates:
  * position  (position) int64 24B 0 1 2
```

**注意点・落とし穴**:
- `rename` は DataArray 自体の名前(文字列を渡す)と、次元・座標の名前(dict を渡す)の両方を扱える。
- Dataset の `rename` は変数名・次元名・座標名のいずれも dict で変更できる。次元だけ名前を変えるなら `rename_dims`、変数だけなら `rename_vars`。
- 2026.2.0 の既定(`xr.get_options()["keep_attrs"] == "default"`)では、`+`・`np.sqrt`・`sum`・`rolling().mean()`・`fillna`・`astype` など多くの演算で `attrs` が結果に引き継がれる。消したいときは `with xr.set_options(keep_attrs=False):` を使う。古い解説記事の「attrs は演算で消える」という記述とは挙動が異なる場合がある。

### `ds.assign(...) / ds.drop_vars(...)`

**用途**: Dataset に変数を追加(`assign`)、または変数・座標を削除(`drop_vars`)した新しい Dataset を返す。

**シグネチャ**: ds.assign(variables=None, **variables_kwargs) / ds.drop_vars(names, errors='raise')

**使用例**:
```python
ds = xr.Dataset({"a": ("x", [1, 2, 3])}, coords={"x": [0, 1, 2]})
ds2 = ds.assign(b=ds["a"] * 10, c=lambda d: d["a"] + d["a"])
print(list(ds2.data_vars))
print(list(ds2.drop_vars("a").data_vars))
```
実行結果:
```
['a', 'b', 'c']
['b', 'c']
```

**注意点・落とし穴**:
- `assign` のキーワード値には callable(引数は Dataset)も渡せ、pandas の `DataFrame.assign` と同じ感覚で連鎖できる。
- `ds["new"] = ...` は in-place の追加。パイプラインで元を壊したくないときは `assign` を使う。

### `xr.zeros_like(...) / xr.ones_like(...) / xr.full_like(...)`

**用途**: 既存オブジェクトと同じ次元・座標を持つ、0・1・任意の値で埋めた配列を作る。

**シグネチャ**: xr.zeros_like(other, dtype=None, chunks=None, chunked_array_type=None, from_array_kwargs=None) / xr.full_like(other, fill_value, dtype=None, chunks=None, chunked_array_type=None, from_array_kwargs=None)

**使用例**:
```python
da = xr.DataArray([[1.5, 2.5], [3.5, 4.5]], dims=("y", "x"), coords={"x": [10, 20]}, name="v", attrs={"units": "m"})
print(xr.zeros_like(da))
print(xr.full_like(da, 7, dtype=int).values)
```
実行結果:
```
<xarray.DataArray 'v' (y: 2, x: 2)> Size: 32B
array([[0., 0.],
       [0., 0.]])
Coordinates:
  * x        (x) int64 16B 10 20
Dimensions without coordinates: y
Attributes:
    units:    m
[[7 7]
 [7 7]]
```

**注意点・落とし穴**:
- 座標・次元名に加え、`name` と `attrs` も引き継がれる(2026.2.0 で確認)。dtype は元のものが使われ、変えたいときは `dtype=` を指定する。

---

## 選択

### `da.sel(...)`

**用途**: 座標の「ラベル」で要素を選ぶ。スカラー・リスト・`slice` を次元名のキーワード引数で渡す。`method="nearest"` で最近傍を選べる。

**シグネチャ**: da.sel(indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs) / ds.sel(indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs)

**使用例**:
```python
da = xr.DataArray(
    np.arange(12).reshape(3, 4),
    dims=("y", "x"),
    coords={"y": [1, 2, 3], "x": [10, 20, 30, 40]},
)
print(da.sel(y=2).values)                       # スカラー指定: 次元が消える
print(da.sel(x=[10, 30], y=2).values)           # リスト指定
print(da.sel(x=slice(15, 35)).x.values)         # スライス(端点を含む範囲)
print(da.sel(x=24, method="nearest").x.values)  # 最近傍
try:
    da.sel(x=25)
except KeyError as e:
    print("KeyError:", e)
```
実行結果:
```
[4 5 6 7]
[4 6]
[20 30]
20
KeyError: "not all values found in index 'x'. Try setting the `method` keyword argument (example: method='nearest')."
```

**注意点・落とし穴**:
- ラベルの `slice(a, b)` は **両端を含む**(位置指定のスライスと違う)。座標が昇順のとき `slice(35, 15)` のように逆順で書くと空(`(3, 0)` の shape)になる。
- 存在しないラベルを指定すると `KeyError`。`method="nearest"`(`ffill`/`pad` は手前側、`bfill`/`backfill` は後ろ側の最も近いラベル。`x=25` に対し `ffill` は 20、`bfill` は 30 になることを確認)と `tolerance=` を併用すると最近傍で許容範囲を制限できる(許容範囲外は `KeyError`)。
- スカラーで選ぶと、その次元は消えるが選んだ値がスカラー座標として結果に残る(`da.sel(y=2)` の結果には `y` 座標が残る)。`drop=True` を付けるとこのスカラー座標も取り除かれる。

### `da.isel(...)`

**用途**: 座標ではなく「整数位置(0始まり)」で要素を選ぶ。座標を持たない次元にも使える。

**シグネチャ**: da.isel(indexers=None, drop=False, missing_dims='raise', **indexers_kwargs)

**使用例**:
```python
da = xr.DataArray(
    np.arange(12).reshape(3, 4),
    dims=("y", "x"),
    coords={"y": [1, 2, 3], "x": [10, 20, 30, 40]},
)
print(da.isel(x=-1).values)                 # 最後の列
print(da.isel(x=slice(0, 4, 2)).x.values)   # 位置スライスは終端を含まない
print(da.isel(x=[0, 1], y=[0, 2]).values)   # リスト指定: 直交選択(格子状)
```
実行結果:
```
[ 3  7 11]
[10 30]
[[0 1]
 [8 9]]
```

**注意点・落とし穴**:
- `isel` の `slice(0, 4, 2)` は Python のスライスと同じく終端を含まない。`sel` の `slice` は含む。
- 範囲外の位置は `IndexError`(例: 長さ 4 の次元に `isel(x=10)`)。
- 複数次元にリストを渡すと「直交(outer)インデクシング」になり、numpy の fancy indexing のようにペア選択にはならない。ペアで選びたいときは次項の DataArray インデクサを使う。

### `da.loc[...]`

**用途**: `da.loc[y_label, x_label]` のように、次元の順序に従って位置引数風にラベル選択する。dict を渡して次元名指定もできる。

**シグネチャ**: `da.loc[...]`(インデクサ形式の `sel`)

**使用例**:
```python
da = xr.DataArray(
    np.arange(12).reshape(3, 4),
    dims=("y", "x"),
    coords={"y": [1, 2, 3], "x": [10, 20, 30, 40]},
)
print(da.loc[2, 20:30].values)
print(da.loc[dict(y=2, x=20)].values)
da.loc[dict(y=2, x=20)] = -1      # 代入にも使える
print(da.values)
```
実行結果:
```
[5 6]
5
[[ 0  1  2  3]
 [ 4 -1  6  7]
 [ 8  9 10 11]]
```

**注意点・落とし穴**:
- 位置引数の順序は `da.dims` の順序に依存する。次元の入れ替え(`transpose`)後にコードが壊れやすいので、可読性・堅牢性の面では `sel(y=..., x=...)` のキーワード形式のほうが安全。
- `da.loc[...] = value` は元の配列を書き換える。一方、`da.sel(y=[2])` のようにリストで選んだ結果は元のコピーなので、それに代入しても元は変わらない(スカラー指定の `da.sel(y=2)` は元のビューで、代入すると元も変わる。2026.2.0 で確認)。代入は `loc` か `da[dict(y=1)] = ...` で行うのが確実。

### ベクトル化(ポイント)インデクシング: DataArray をインデクサに渡す(`da.sel` / `da.isel`)

**用途**: 選択キーとして「新しい次元を持つ DataArray」を渡すと、点ごとの選択(numpy の `a[i_arr, j_arr]` 相当)になる。散在する観測点の値を取り出すときに使う。

**シグネチャ**: `da.sel(x=DataArray, y=DataArray)` / `da.isel(...)`

**使用例**:
```python
da = xr.DataArray(
    np.arange(12).reshape(3, 4),
    dims=("y", "x"),
    coords={"y": [1, 2, 3], "x": [10, 20, 30, 40]},
)
pts_x = xr.DataArray([10, 30, 40], dims="pts")
pts_y = xr.DataArray([1, 2, 3], dims="pts")

print(da.sel(x=pts_x, y=pts_y))                  # 3 点だけ
print(da.sel(x=[10, 30, 40], y=[1, 2, 3]).shape)  # 比較: リストなら 3x3 の格子
```
実行結果:
```
<xarray.DataArray (pts: 3)> Size: 24B
array([ 0,  6, 11])
Coordinates:
    y        (pts) int64 24B 1 2 3
    x        (pts) int64 24B 10 30 40
Dimensions without coordinates: pts
(3, 3)
```

**注意点・落とし穴**:
- 同じ新規次元名(ここでは `pts`)を共有する DataArray を各次元に渡すのがポイント。結果は `pts` 次元を持ち、元の `x`,`y` は `pts` に沿った座標として残る。
- リスト同士だと格子状(直交)になり、意図しない `3x3` が返る。点ごとに選ぶ意図なら必ず DataArray で渡す。

### `da.where(cond, other, drop=...)`

**用途**: 条件が真の要素だけを残し、偽の要素を NaN(または `other`)に置き換える。マスク処理の基本。

**シグネチャ**: da.where(cond, other=<NA>, drop=False)

**使用例**:
```python
da = xr.DataArray(
    np.arange(12).reshape(3, 4),
    dims=("y", "x"),
    coords={"y": [1, 2, 3], "x": [10, 20, 30, 40]},
)
print(da.where(da > 5))                   # 偽の要素は NaN(整数から float に変わる)
print(da.where(da > 5, drop=True))        # 全要素が偽の行・列を落とす
print(da.where(da > 5, 0).values)         # other を指定すると dtype は保たれる
```
実行結果:
```
<xarray.DataArray (y: 3, x: 4)> Size: 96B
array([[nan, nan, nan, nan],
       [nan, nan,  6.,  7.],
       [ 8.,  9., 10., 11.]])
Coordinates:
  * y        (y) int64 24B 1 2 3
  * x        (x) int64 32B 10 20 30 40
<xarray.DataArray (y: 2, x: 4)> Size: 64B
array([[nan, nan,  6.,  7.],
       [ 8.,  9., 10., 11.]])
Coordinates:
  * y        (y) int64 16B 2 3
  * x        (x) int64 32B 10 20 30 40
[[ 0  0  0  0]
 [ 0  0  6  7]
 [ 8  9 10 11]]
```

**注意点・落とし穴**:
- `other` を省略すると NaN 埋めになり、整数配列は float64 に変換される(上の例で確認)。整数のまま保ちたければ `other=0` のように値を指定する。
- `drop=True` は「条件が全て偽の座標ラベル」を落とす。1 つでも真があれば、その行・列は残り、NaN も残る。
- `da[cond]` のような真偽値の直接インデクシングは、次元 `x` に対する真偽値でも先頭の次元に適用されて `IndexError` になりやすい(例: `da[da.x > 15]` は長さ 4 の真偽値を長さ 3 の `y` に適用しようとして失敗)。`da.isel(x=cond.values)` や `da.where(...)` を使う。

### `xr.where(cond, x, y)`

**用途**: 条件に応じて `x` と `y` のどちらかを選ぶ、`np.where` の xarray 版。座標・次元名を保ったまま、次元名でブロードキャストする。

**シグネチャ**: xr.where(cond, x, y, keep_attrs=None)

**使用例**:
```python
da = xr.DataArray([1, 5, 9], dims="x", coords={"x": [0, 1, 2]})
print(xr.where(da > 4, "big", "small").values)
print(xr.where(da > 4, da, -da))
```
実行結果:
```
['small' 'big' 'big']
<xarray.DataArray (x: 3)> Size: 24B
array([-1,  5,  9])
Coordinates:
  * x        (x) int64 24B 0 1 2
```

**注意点・落とし穴**:
- `da.where(cond, other)` は「condが真なら自分、偽なら other」。`xr.where(cond, x, y)` は「真なら x、偽なら y」で、自分自身を主語にしない点が違う。
- `keep_attrs` を明示しない場合、属性は 2 番目の引数 `x` のものが引き継がれる(`x` がスカラーなら属性なし)。`keep_attrs=False` で消せる。
- `cond`・`x`・`y` の座標が一致しないと(例: `cond` の `x` 座標が `[1,2,3]`、値が `[0,1,2]`)`AlignmentError`(`join="exact"` 相当)になる。事前に `xr.align` か `reindex` で揃えること。

### `da.isin(...) / da.sortby(...) / da.drop_sel(...)`

**用途**: 値が集合に含まれるかの真偽値(`isin`)、座標(や他の DataArray)の値で並べ替え(`sortby`)、指定ラベルを除外(`drop_sel`)。

**シグネチャ**: da.isin(test_elements) / da.sortby(variables, ascending=True) / da.drop_sel(labels=None, errors='raise', **labels_kwargs)

**使用例**:
```python
da = xr.DataArray(
    np.arange(12).reshape(3, 4),
    dims=("y", "x"),
    coords={"y": [1, 2, 3], "x": [10, 20, 30, 40]},
)
print(da.isin([1, 5, 7]).values)
print(da.sortby("x", ascending=False).x.values)
print(da.drop_sel(x=[10, 20]).x.values)
```
実行結果:
```
[[False  True False False]
 [False  True False  True]
 [False False False False]]
[40 30 20 10]
[30 40]
```

**注意点・落とし穴**:
- `sortby` は座標名のほか、その次元に沿った 1 次元の DataArray も渡せ、その値の昇順(`ascending=False` で降順)に並べ替える。例: `da.sortby(da.isel(y=0), ascending=False)` は `y=0` 行の値の降順に `x` を並べ替える。
- `drop_sel` は存在しないラベルを指定すると `KeyError`。無視したければ `errors="ignore"`。

### `da.argmax(...) / da.idxmax(...)`

**用途**: 最大値の位置(整数, `argmax`)または最大値の座標ラベル(`idxmax`)を、指定次元に沿って返す。最小は `argmin` / `idxmin`。

**シグネチャ**: da.argmax(dim=None, axis=None, keep_attrs=None, skipna=None) / da.idxmax(dim=None, skipna=None, fill_value=<NA>, keep_attrs=None)

**使用例**:
```python
da = xr.DataArray(
    np.arange(12).reshape(3, 4),
    dims=("y", "x"),
    coords={"y": [1, 2, 3], "x": [10, 20, 30, 40]},
)
print(da.argmax(dim="x").values)   # 位置(0 始まり)
print(da.idxmax("x").values)       # ラベル(x の座標値)
print(da.idxmax("y"))
```
実行結果:
```
[3 3 3]
[40 40 40]
<xarray.DataArray 'y' (x: 4)> Size: 32B
array([3, 3, 3, 3])
Coordinates:
  * x        (x) int64 32B 10 20 30 40
```

**注意点・落とし穴**:
- `argmax()` に `dim` を渡さないと、将来変更される旨の `FutureWarning` が出る(2026.2.0 で確認)。必ず `dim=` を指定する。
- `idxmax` は座標値を返すので、時系列の「ピークの日付」を取り出す用途に便利。

---

## 演算・ブロードキャスト

### 次元名によるブロードキャスト(演算子 `+ - * /` など)

**用途**: numpy と違い、位置ではなく「次元名」で軸を対応づけて演算する。次元の並び順が違っても、次元名が同じなら正しく計算され、片方にしかない次元は自動でブロードキャストされる。

**シグネチャ**: 演算子 `+ - * /` など

**使用例**:
```python
a = xr.DataArray([1, 2, 3], dims="x", coords={"x": [0, 1, 2]})
b = xr.DataArray([10, 20], dims="y", coords={"y": ["p", "q"]})
print(a + b)                       # 別の次元同士 -> 2 次元に拡張

A = xr.DataArray(np.arange(6).reshape(2, 3), dims=("y", "x"))
B = xr.DataArray(np.arange(6).reshape(3, 2), dims=("x", "y"))   # 次元の順序が違う
print((A + B).dims, (B + A).dims)
print((A + B).values)
```
実行結果:
```
<xarray.DataArray (x: 3, y: 2)> Size: 48B
array([[11, 21],
       [12, 22],
       [13, 23]])
Coordinates:
  * x        (x) int64 24B 0 1 2
  * y        (y) <U1 8B 'p' 'q'
('y', 'x') ('x', 'y')
[[ 0  3  6]
 [ 4  7 10]]
```

**注意点・落とし穴**:
- 結果の次元の並び順は、左オペランドの次元順(左に無い次元は後ろ)に従う(上の `(A + B).dims` と `(B + A).dims` で確認)。並び順を固定したいときは `.transpose(...)` を使う。
- numpy なら形状が `(2,3)` と `(3,2)` で失敗する演算が、xarray では次元名で対応づくため通る。逆に、意図せず同名の次元が別の意味を持つと黙って混ざるので、次元名の設計が重要。
- 同名の次元で長さが違い、座標も無い場合は `AlignmentError`(座標があれば次項の自動整列が働く)。

### 座標による自動整列(inner join)と `xr.set_options(arithmetic_join=...)`

**用途**: 同名の次元どうしの演算では、座標ラベルで自動的に整列する。既定は共通ラベルだけを残す `inner` 結合。

**シグネチャ**: 演算子 + `xr.set_options(arithmetic_join="inner" | "outer" | "left" | "right" | "exact")`

**使用例**:
```python
a = xr.DataArray([1, 2, 3], dims="x", coords={"x": [0, 1, 2]})
c = xr.DataArray([10, 20, 30], dims="x", coords={"x": [1, 2, 3]})
print(a + c)                        # 共通ラベル x=1,2 のみ

with xr.set_options(arithmetic_join="outer"):
    print(a + c)                    # 和集合(欠ける側は NaN)

with xr.set_options(arithmetic_join="exact"):
    try:
        a + c
    except Exception as e:
        print(type(e).__name__)
```
実行結果:
```
<xarray.DataArray (x: 2)> Size: 16B
array([12, 23])
Coordinates:
  * x        (x) int64 16B 1 2
<xarray.DataArray (x: 4)> Size: 32B
array([nan, 12., 23., nan])
Coordinates:
  * x        (x) int64 32B 0 1 2 3
AlignmentError
```

**注意点・落とし穴**:
- 既定の `inner` では座標が食い違うとデータが黙って減る。想定外の行数減少に気づかないことがあるので、演算前後で `sizes` を確認するか、`arithmetic_join="exact"` で厳格に検証する。
- 座標を持たない次元(位置だけ)どうしなら、長さが同じであれば位置で対応づけられる。長さが違うと `AlignmentError`。
- `arithmetic_join="outer"` にすると整数配列でも欠損は NaN になり、float に昇格する。

### `np.sqrt(da)` など numpy ufunc / `xr.apply_ufunc(...)`

**用途**: numpy の ufunc(`np.sqrt`, `np.exp`, `np.add` など)は DataArray/Dataset にそのまま適用でき、座標・次元名が保たれる。ufunc 以外の numpy 関数は `xr.apply_ufunc` で包む。

**シグネチャ**: xr.apply_ufunc(func, *args, input_core_dims=None, output_core_dims=((),), exclude_dims=frozenset(), vectorize=False, join='exact', dataset_join='exact', ...)

**使用例**:
```python
a = xr.DataArray([1, 4, 9], dims="x", coords={"x": [0, 1, 2]})
print(np.sqrt(a).values)

A = xr.DataArray(np.arange(6).reshape(2, 3), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1, 2]})
# x 次元を「コア次元」として関数に渡し、x に沿って平均を取る(結果は y のみ)
res = xr.apply_ufunc(np.mean, A, input_core_dims=[["x"]], kwargs={"axis": -1})
print(res)
```
実行結果:
```
[1. 2. 3.]
<xarray.DataArray (y: 2)> Size: 16B
array([1., 4.])
Coordinates:
  * y        (y) int64 16B 0 1
```

**注意点・落とし穴**:
- `input_core_dims=[["x"]]` で指定した次元は、関数に渡される numpy 配列の「最後の軸」に移される。そのため `np.mean` などの軸を取る関数には `kwargs={"axis": -1}` を渡す。
- `vectorize=True` を付けると、要素(またはコア次元ごと)に関数を繰り返し適用する形になる。Python レベルのループなので大規模データでは遅い。`dask="parallelized"` 等の並列オプションは dask が必要で、本環境には未導入のため未検証。
- `apply_ufunc` の既定は `join="exact"` で、入力の座標が食い違うと通常の演算子(inner join)と違って `AlignmentError` になる(2026.2.0 で確認)。
- 単純な集約は `da.mean("x")` で十分。`apply_ufunc` は自前の numpy/scipy 関数を次元名つきで使いたいときの最終手段と考える。

### `xr.dot(...) / da.dot(...)`

**用途**: 共通する次元に沿って内積(縮約)をとる。`einsum` を次元名で書くイメージ。

**シグネチャ**: xr.dot(*arrays, dim=None, **kwargs) / da.dot(other, dim=None)

**使用例**:
```python
A = xr.DataArray(np.arange(6).reshape(2, 3), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1, 2]})
B = xr.DataArray(np.arange(6).reshape(3, 2), dims=("x", "y"), coords={"x": [0, 1, 2], "y": [0, 1]})
print(xr.dot(A, B, dim="x"))       # x だけ縮約して y が残る
print(A.dot(B))                    # 共通次元(x, y)を全て縮約 -> スカラー
```
実行結果:
```
<xarray.DataArray (y: 2)> Size: 16B
array([10, 40])
Coordinates:
  * y        (y) int64 16B 0 1
<xarray.DataArray ()> Size: 8B
array(50)
```

**注意点・落とし穴**:
- `dim` を省略すると、全オペランドに共通する次元をすべて縮約する(`A.dot(B)` は `x` と `y` の両方を縮約してスカラーになった)。行列積のつもりで意図しない次元まで縮約しないよう、`dim=` を明示するのが安全。
- 座標のラベルは自動整列される(inner join)。ラベルが揃っていない次元は共通部分だけが使われる。

### `da.cumsum(...) / da.diff(...) / da.shift(...) / da.roll(...)`

**用途**: 累積和(`cumsum`)、隣接差分(`diff`)、値のずらし(`shift`、空きは NaN)、循環シフト(`roll`)。

**シグネチャ**: da.cumsum(dim=None, skipna=None, keep_attrs=None, **kwargs) / da.diff(dim, n=1, label='upper') / da.shift(shifts=None, fill_value=<NA>, **shifts_kwargs) / da.roll(shifts=None, roll_coords=False, **shifts_kwargs)

**使用例**:
```python
a = xr.DataArray([1, 2, 4, 8], dims="x", coords={"x": [0, 1, 2, 3]})
print(a.cumsum("x").values)
print(a.diff("x"))
print(a.diff("x", label="lower").x.values)
print(a.shift(x=1).values)
print(a.shift(x=1, fill_value=0).values)
print(a.roll(x=1).values)
```
実行結果:
```
[ 1  3  7 15]
<xarray.DataArray (x: 3)> Size: 24B
array([1, 2, 4])
Coordinates:
  * x        (x) int64 24B 1 2 3
[0 1 2]
[nan  1.  2.  4.]
[0 1 2 4]
[8 1 2 4]
```

**注意点・落とし穴**:
- `diff` の結果は要素が 1 つ減り、座標は既定(`label="upper"`)で後ろ側のラベルになる(`label="lower"` で前側)。
- `shift` は座標をずらさず、値だけを動かす。空いた部分は NaN(整数は float に昇格)。`fill_value=` で埋め値を指定すると dtype を保てる。
- `roll` は循環シフト。既定では座標は動かず値だけが回る(`roll_coords=False`)。座標も一緒に回すなら `roll_coords=True`。
- `cumsum` は `dim` 指定なしだと全次元にわたって累積する(2 次元だと二重累積になる)。必ず `dim` を指定する。

### `ds.map(...) / da.pipe(...) / da.clip(...)`

**用途**: Dataset の全変数に関数を適用(`map`)、関数を method-chain 風に渡す(`pipe`)、値の上下限を切り詰める(`clip`)。

**シグネチャ**: ds.map(func, keep_attrs=None, args=(), **kwargs) / da.pipe(func, *args, **kwargs) / da.clip(min=None, max=None, keep_attrs=None)

**使用例**:
```python
ds = xr.Dataset({"u": ("x", [1, 2, 3]), "v": ("x", [4, 5, 6])})
print(ds.map(lambda v: v - v.mean()))     # 変数ごとに平均を引く

a = xr.DataArray([1, 2, 3], dims="x")
print(a.pipe(lambda d, k: d * k, 3).values)
print(a.clip(min=2).values)
```
実行結果:
```
<xarray.Dataset> Size: 48B
Dimensions:  (x: 3)
Dimensions without coordinates: x
Data variables:
    u        (x) float64 24B -1.0 0.0 1.0
    v        (x) float64 24B -1.0 0.0 1.0
[3 6 9]
[2 2 3]
```

**注意点・落とし穴**:
- `ds.map` は各変数(DataArray)に関数を適用して新しい Dataset を返す。関数は DataArray を受けて DataArray を返す必要がある。
- Dataset 全体に対する演算子(`ds * 2`, `ds - ds.mean()`)は全変数に効く。`ds.mean()` は各変数のスカラー平均を持つ Dataset を返し、それを引く演算もそのまま動く。

### `da.polyfit(...) / xr.polyval(...)`

**用途**: 指定次元に沿った最小二乗の多項式フィット(`polyfit`)と、得られた係数からの評価(`polyval`)。

**シグネチャ**: da.polyfit(dim, deg, skipna=None, rcond=None, w=None, full=False, cov=False) / xr.polyval(coord, coeffs, degree_dim='degree')

**使用例**:
```python
y = xr.DataArray([1.0, 3.1, 4.9, 7.2], dims="x", coords={"x": [0, 1, 2, 3]})
fit = y.polyfit("x", 1)                      # 1 次式
print(fit)
print(xr.polyval(y.x, fit.polyfit_coefficients).values)
```
実行結果:
```
<xarray.Dataset> Size: 32B
Dimensions:               (degree: 2)
Coordinates:
  * degree                (degree) int64 16B 1 0
Data variables:
    polyfit_coefficients  (degree) float64 16B 2.04 0.99
[0.99 3.03 5.07 7.11]
```

**注意点・落とし穴**:
- 戻り値は Dataset で、係数は変数 `polyfit_coefficients`(次元 `degree`)に入る。`degree` の並びは高次から低次(上の例では 1 次の係数 2.04、次に定数項 0.99)。
- `polyfit` は多項式の線形最小二乗フィット。scipy の `curve_fit` のような任意関数の非線形フィットではない。

### `da.equals(...) / da.identical(...) / xr.testing.assert_allclose(...)`

**用途**: 2 つのオブジェクトが同じか判定する。`equals` は値・次元・座標が同じか(名前や属性は無視)、`identical` は名前・属性まで同じか、`assert_allclose` は数値の許容誤差つき比較(テスト用)。

**シグネチャ**: da.equals(other) / da.identical(other) / xr.testing.assert_allclose(a, b, rtol=1e-05, atol=1e-08, decode_bytes=True, check_dim_order=True)

**使用例**:
```python
a = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [0, 1]}, name="n")
print(a.equals(a.rename("other")))
print(a.identical(a.rename("other")))
xr.testing.assert_allclose(a, a + 1e-12)      # 差が許容誤差内なら何も起きない
print("OK")
```
実行結果:
```
True
False
OK
```

**注意点・落とし穴**:
- `a == b` は要素ごとの比較結果(DataArray)を返し、`if a == b:` のような使い方はできない。全体の一致は `a.equals(b)`。
- `assert_allclose` は不一致のとき `AssertionError` を投げる。単体テストや検証コードで使う。

---

## 集約

### `da.mean(...) / da.sum(...) / da.std(...) / da.count(...)`

**用途**: 指定した次元(名前で指定)に沿って集約する。`dim` を省略すると全次元で集約してスカラーになる。NaN は既定で無視(`skipna=True`)される。`max`/`min`/`median`/`var`/`prod` なども同じ使い方。

**シグネチャ**: da.mean(dim=None, skipna=None, keep_attrs=None, **kwargs) / da.sum(dim=None, skipna=None, min_count=None, keep_attrs=None, **kwargs) / da.count(dim=None, keep_attrs=None, **kwargs)

**使用例**:
```python
da = xr.DataArray(
    [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]],
    dims=("y", "x"),
    coords={"y": ["a", "b"], "x": [0, 1, 2]},
    attrs={"units": "m"},
)
print(da.mean("x"))                      # x に沿って平均(y が残る)
print(da.mean().item())                  # 全体平均(NaN を除く)
print(da.mean("x", skipna=False).values) # NaN を含む行は NaN
print(da.sum("x", min_count=3).values)   # 有効値が 3 個未満なら NaN
print(da.count("x").values)              # NaN 以外の個数
print(da.std("x", ddof=1).values)
print(da.mean(["x", "y"]).item())        # 複数次元を同時に
```
実行結果:
```
<xarray.DataArray (y: 2)> Size: 16B
array([1.5, 5. ])
Coordinates:
  * y        (y) <U1 8B 'a' 'b'
Attributes:
    units:    m
3.6
[nan  5.]
[nan 15.]
[2 3]
[0.70710678 1.        ]
3.6
```

**注意点・落とし穴**:
- `dim` は次元名(文字列またはリスト)で指定する。`axis=` でも位置指定できる(`da.mean(axis=0)` も動く)が、次元の並びに依存するので次元名の指定を推奨。存在しない次元名を渡すと `ValueError`。
- `skipna` は float 型では既定 True(NaN を無視)。`sum` は全要素が NaN でも 0 を返す(`mean` は NaN)ため、`min_count=` で「有効値が足りなければ NaN」にできる。
- `std`/`var` の既定は `ddof=0`(母集団)。pandas(`ddof=1`)と違うので注意し、標本標準偏差が欲しいときは `ddof=1` を渡す。
- `attrs` は 2026.2.0 の既定で集約結果に引き継がれる(上の `mean("x")` の結果に `units` が残る)。消したいときは `keep_attrs=False`。
- Dataset に対して呼ぶと全変数に適用され、結果も Dataset になる。
- `da.mean()` のように `dim` を省略すると 0 次元の DataArray が返る。Python の数値にするには `.item()`(または `float(...)`)。

### `da.quantile(...)`

**用途**: 指定次元に沿った分位数を計算する。結果には新しい次元 `quantile` が付く。

**シグネチャ**: da.quantile(q, dim=None, method='linear', keep_attrs=None, skipna=None, interpolation=None)

**使用例**:
```python
da = xr.DataArray(
    [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]],
    dims=("y", "x"),
    coords={"y": ["a", "b"], "x": [0, 1, 2]},
)
print(da.quantile([0.25, 0.75], dim="x"))
print(da.quantile(0.5, dim="x").values)
```
実行結果:
```
<xarray.DataArray (quantile: 2, y: 2)> Size: 32B
array([[1.25, 4.5 ],
       [1.75, 5.5 ]])
Coordinates:
  * quantile  (quantile) float64 16B 0.25 0.75
  * y         (y) <U1 8B 'a' 'b'
[1.5 5. ]
```

**注意点・落とし穴**:
- `q` に配列を渡すと `quantile` 次元が付き、スカラーを渡すと次元は付かず、スカラー座標 `quantile` として残る(`da.quantile(0.5, dim="x")` で確認)。
- `skipna` の既定は NaN を無視する動作(上の行 `a` は有効値 `[1, 2]` から計算されている)。

### `da.groupby(...)`

**用途**: 座標(または DataArray、`"time.month"` のような日時属性)の値ごとにグループ化し、`mean`/`sum` などの集約や `map` を適用する。

**シグネチャ**: da.groupby(group=None, squeeze=False, restore_coord_dims=False, eagerly_compute_group=None, **groupers)

**使用例**:
```python
g = xr.DataArray(
    np.arange(6.0),
    dims="x",
    coords={"x": np.arange(6), "lab": ("x", ["a", "b", "a", "b", "a", "b"])},
)
print(g.groupby("lab").mean())
print(g.groupby("lab").max("x").values)
print(g.groupby("lab").map(lambda v: v - v.mean()))   # グループ内で平均を引く
for key, sub in g.groupby("lab"):
    print(key, sub.values)
```
実行結果:
```
<xarray.DataArray (lab: 2)> Size: 16B
array([2., 3.])
Coordinates:
  * lab      (lab) object 16B 'a' 'b'
[4. 5.]
<xarray.DataArray (x: 6)> Size: 48B
array([-2., -2.,  0.,  0.,  2.,  2.])
Coordinates:
  * x        (x) int64 48B 0 1 2 3 4 5
    lab      (x) <U1 24B 'a' 'b' 'a' 'b' 'a' 'b'
a [0. 2. 4.]
b [1. 3. 5.]
```

**注意点・落とし穴**:
- グループ化キーが非次元座標(`lab`)でも使える。結果はキー名の次元(`lab`)を持ち、元の次元(`x`)はグループ内で集約される。
- `map` は各グループに関数を適用し、結果を元の並びに結合し直す(元の次元の順序が保たれる)。
- `"time.month"` のように日時座標の属性でもグループ化できる(「時系列・補間」参照)。
- グルーパーオブジェクトを次元名のキーワードで渡す書き方(`g.groupby(x=xr.groupers.BinGrouper(...))`)もある(次項の例参照)。

### `da.groupby_bins(...)`

**用途**: 連続値の座標を区間(ビン)に分けてグループ化する。`pandas.cut` 相当。

**シグネチャ**: da.groupby_bins(group, bins, right=True, labels=None, precision=3, include_lowest=False, squeeze=False, restore_coord_dims=False, duplicates='raise', eagerly_compute_group=None)

**使用例**:
```python
g = xr.DataArray(np.arange(6.0), dims="x", coords={"x": np.arange(6)})
print(g.groupby_bins("x", bins=[0, 2, 6]).mean())
print(g.groupby_bins("x", bins=[0, 2, 6], right=False).mean())
# グルーパーを使う新しい書き方
print(g.groupby(x=xr.groupers.BinGrouper(bins=[0, 3, 6], right=False)).sum())
```
実行結果:
```
<xarray.DataArray (x_bins: 2)> Size: 16B
array([1.5, 4. ])
Coordinates:
  * x_bins   (x_bins) interval[int64, right] 32B (0, 2] (2, 6]
<xarray.DataArray (x_bins: 2)> Size: 16B
array([0.5, 3.5])
Coordinates:
  * x_bins   (x_bins) interval[int64, left] 32B [0, 2) [2, 6)
<xarray.DataArray (x_bins: 2)> Size: 16B
array([ 3., 12.])
Coordinates:
  * x_bins   (x_bins) interval[int64, left] 32B [0, 3) [3, 6)
```

**注意点・落とし穴**:
- 区間は既定で右閉じ `(0, 2]`(`right=True`)。そのため `x=0` はどの区間にも入らず結果から除外される(上の平均 1.5 は `x=1,2` の値、4.0 は `x=3,4,5` の値)。端を含めたいときは `right=False`(左閉じ `[0, 2)`。上の 2 つ目の例)か `include_lowest=True`(最初の区間だけ下端を含める)を指定する。
- 結果の座標は `x_bins`(`pandas.Interval`)になる。ラベル文字列にしたいときは `labels=[...]` を渡す。

### `da.rolling(...)`

**用途**: 指定次元に沿って移動窓(ウィンドウ)を作り、`mean`/`sum`/`max` などを適用する。

**シグネチャ**: da.rolling(dim=None, min_periods=None, center=False, **window_kwargs)

**使用例**:
```python
time = pd.date_range("2024-01-01", periods=10, freq="D")
v = xr.DataArray(np.arange(10.0), dims="time", coords={"time": time})
print(v.rolling(time=3).mean())                       # 最初の 2 点は NaN
print(v.rolling(time=3, min_periods=1).mean().values) # 窓が満たなくても計算
print(v.rolling(time=3, center=True).mean().values)   # 中心窓
print(v.rolling(time=3).construct("window").shape)    # 窓を新しい次元として展開
```
実行結果:
```
<xarray.DataArray (time: 10)> Size: 80B
array([nan, nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
Coordinates:
  * time     (time) datetime64[us] 80B 2024-01-01 2024-01-02 ... 2024-01-10
[0.  0.5 1.  2.  3.  4.  5.  6.  7.  8. ]
[nan  1.  2.  3.  4.  5.  6.  7.  8. nan]
(10, 3)
```

**注意点・落とし穴**:
- 窓の幅は要素数(整数)で指定する(`time=3` は 3 要素)。`time="3D"` のような時間幅の文字列は `TypeError` になる。時間ベースで集約したいときは `resample` を使う。
- 既定は窓の右端がラベル(`center=False`)。過去 3 点の平均が現在のラベルに付く。`center=True` だと前後に窓が広がり、両端が NaN になる。
- `construct("window")` は窓を新しい次元(ここでは `window`)に展開した配列(shape `(10, 3)`)を返す。`mean` 以外の自前の集計を書くときに使う。

### `da.coarsen(...)`

**用途**: 指定次元を、重ならない一定サイズのブロックにまとめて集約(ダウンサンプリング)する。

**シグネチャ**: da.coarsen(dim=None, boundary='exact', side='left', coord_func='mean', **window_kwargs)

**使用例**:
```python
time = pd.date_range("2024-01-01", periods=10, freq="D")
v = xr.DataArray(np.arange(10.0), dims="time", coords={"time": time})
print(v.coarsen(time=5).mean())
try:
    v.coarsen(time=3).mean()
except ValueError as e:
    print("ValueError:", e)
print(v.coarsen(time=3, boundary="trim").mean().values)   # 余りを切り捨て
print(v.coarsen(time=3, boundary="pad").mean().values)    # 余りを NaN で埋める
```
実行結果:
```
<xarray.DataArray (time: 2)> Size: 16B
array([2., 7.])
Coordinates:
  * time     (time) datetime64[us] 16B 2024-01-03 2024-01-08
ValueError: Could not coarsen a dimension of size 10 with window 3 and boundary='exact'. Try a different 'boundary' option.
[1. 4. 7.]
[1. 4. 7. 9.]
```

**注意点・落とし穴**:
- ブロックサイズで割り切れないと、既定(`boundary="exact"`)では `ValueError`。`boundary="trim"`(端を切り捨て)か `"pad"`(NaN 埋め)を選ぶ。
- `rolling` が窓を 1 つずつずらすのに対し、`coarsen` は重ならないブロックに集約するため、結果の長さが 1/サイズになる。
- 座標にも既定で平均が適用される(上の結果の `time` はブロック内の平均、つまり 5 日ブロックの中央日)。座標の集約方法は `coord_func=`(例: `"max"`)で変えられる。日時の間引きなら `resample` も使える(「時系列・補間」参照)。

### `da.weighted(...)`

**用途**: 重み付きの `mean`/`sum` などを計算する。緯度による面積補正付きの平均などで使う。

**シグネチャ**: da.weighted(weights)

**使用例**:
```python
values = xr.DataArray([1.0, 3.0], dims="x", coords={"x": [0, 1]})
w = xr.DataArray([1.0, 3.0], dims="x")
print(values.weighted(w).mean().item())     # (1*1 + 3*3) / (1 + 3)
print(values.weighted(w).sum().item())
```
実行結果:
```
2.5
10.0
```

**注意点・落とし穴**:
- 重みに NaN を含めると `ValueError`(メッセージは `weights.fillna(0)` を提案する)。欠損の重みは 0 に置き換えてから渡す。
- `weighted` の次元も、通常の集約と同様 `dim=` で指定できる。

### 気候値からの偏差: `da.groupby(...) - da.groupby(...).mean()`

**用途**: グループごとの平均(気候値)を、グループごとに元の並びへブロードキャストして引く。季節性の除去などに使う。

**シグネチャ**: 演算子 + groupby

**使用例**:
```python
g = xr.DataArray(
    np.arange(6.0),
    dims="x",
    coords={"x": np.arange(6), "lab": ("x", ["a", "b", "a", "b", "a", "b"])},
)
grp = g.groupby("lab")
print(grp - grp.mean())
```
実行結果:
```
<xarray.DataArray (x: 6)> Size: 48B
array([-2., -2.,  0.,  0.,  2.,  2.])
Coordinates:
  * x        (x) int64 48B 0 1 2 3 4 5
    lab      (x) <U1 24B 'a' 'b' 'a' 'b' 'a' 'b'
```

**注意点・落とし穴**:
- 結果は元と同じ形・座標の DataArray。`groupby(...).map(lambda v: v - v.mean())` と同じ結果になる(ここでは `[-2, -2, 0, 0, 2, 2]`)。

---

## 結合・整列

### `xr.concat(...)`

**用途**: 既存の次元に沿って連結する、または新しい次元を作って積み重ねる(numpy の `concatenate`/`stack` 相当)。

**シグネチャ**: xr.concat(objs, dim, data_vars=all, coords=different, compat=equals, positions=None, fill_value=<NA>, join=outer, combine_attrs='override', create_index_for_new_dim=True)

**使用例**:
```python
a = xr.DataArray([[1, 2], [3, 4]], dims=("y", "x"), coords={"y": [0, 1], "x": [10, 20]}, name="v")
b = xr.DataArray([[5, 6]], dims=("y", "x"), coords={"y": [2], "x": [10, 20]}, name="v")

print(xr.concat([a, b], dim="y"))                      # 既存の y 次元に連結
new = xr.concat([a, a + 10], dim="run")                # 新しい次元 run を作って積む
print(new.dims, new.sizes["run"])
new2 = xr.concat([a, a + 10], dim=pd.Index(["r1", "r2"], name="run"))   # ラベル付きの新次元
print(new2.run.values)
```
実行結果:
```
<xarray.DataArray 'v' (y: 3, x: 2)> Size: 48B
array([[1, 2],
       [3, 4],
       [5, 6]])
Coordinates:
  * y        (y) int64 24B 0 1 2
  * x        (x) int64 16B 10 20
('run', 'y', 'x') 2
['r1' 'r2']
```

**注意点・落とし穴**:
- `dim="run"` のように未存在の次元名を文字列で渡すと、座標を持たない新しい次元ができる。ラベルを付けたいときは `pd.Index([...], name="run")` を渡す。
- 連結対象の他の次元の座標が食い違うと、既定では和集合に揃えて NaN で埋める(`join="outer"`)。ただし 2026.2.0 では「将来 `join="exact"` に変わる」旨の `FutureWarning` が出るので、`join=` を明示するのが安全(下の `xr.merge` の項目で警告文を確認)。
- `inspect.signature` では `join=outer`・`data_vars=all`・`compat=equals` などが引用符なしで表示される。これらの既定値は `xarray.util.deprecation_helpers.CombineKwargDefault` 型の特別なオブジェクト(将来の既定値変更のための移行用)である。`xr.set_options(use_new_combine_kwarg_defaults=True)`(既定は False)を有効にすると新しい既定値で動作を先取りできる(`xr.merge` の警告文にもこの案内が出る)。

### `xr.merge(...) / ds.merge(...)`

**用途**: 名前の異なる変数を持つ Dataset/DataArray(名前付き)を、座標で整列して 1 つの Dataset にまとめる。

**シグネチャ**: xr.merge(objects, compat=no_conflicts, join=outer, fill_value=<NA>, combine_attrs='override') / ds.merge(other, overwrite_vars=frozenset(), compat=no_conflicts, join=outer, fill_value=<NA>, combine_attrs='override')

**使用例**:
```python
import warnings

ds1 = xr.Dataset({"t": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
ds2 = xr.Dataset({"p": ("x", [5.0, 6.0, 7.0])}, coords={"x": [1, 2, 3]})

print(xr.merge([ds1, ds2], join="outer"))
print(xr.merge([ds1, ds2], join="inner"))       # 共通の x のみ

# 座標が不一致なのに join を省略すると FutureWarning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    xr.merge([ds1, ds2])
print(w[0].category.__name__)
print(str(w[0].message)[:100])

# 同名変数で値が衝突すると MergeError
try:
    xr.merge([ds1, ds1.assign(t=("x", [9.0, 9.0]))])
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
<xarray.Dataset> Size: 96B
Dimensions:  (x: 4)
Coordinates:
  * x        (x) int64 32B 0 1 2 3
Data variables:
    t        (x) float64 32B 1.0 2.0 nan nan
    p        (x) float64 32B nan 5.0 6.0 7.0
<xarray.Dataset> Size: 24B
Dimensions:  (x: 1)
Coordinates:
  * x        (x) int64 8B 1
Data variables:
    t        (x) float64 8B 2.0
    p        (x) float64 8B 5.0
FutureWarning
In a future version of xarray the default value for join will change from join='outer' to join='exac
MergeError conflicting values for variable 't' on objects to be combined. You can skip this check by specifying compat='override'.
```

**注意点・落とし穴**:
- 同名の変数が両方にあるとき、値が食い違うと `MergeError`。`compat="override"` を渡すと先頭のオブジェクトの値を採用して検査を省略する(`compat=` の既定は `no_conflicts`: NaN 以外の値が衝突しなければ OK)。
- DataArray を渡すときは `name` が必要。名前のない DataArray を単独でまとめようとすると `ValueError: unable to convert unnamed DataArray to a Dataset ...`。
- `join=` の既定は `outer` だが、上で見たとおり座標不一致の場合は将来の既定値変更(`exact`)に関する `FutureWarning` が出る。座標が揃っていると分かっているデータ以外は `join=` を明示する。

### `da.combine_first(...)`

**用途**: 欠損(NaN)部分を、別のオブジェクトの値で埋める。座標の和集合に広げて結合する。

**シグネチャ**: da.combine_first(other)

**使用例**:
```python
a = xr.DataArray([1.0, np.nan, 3.0], dims="x", coords={"x": [0, 1, 2]}, name="t")
b = xr.DataArray([10.0, 20.0, 30.0, 40.0], dims="x", coords={"x": [1, 2, 3, 4]}, name="t")
print(a.combine_first(b))
```
実行結果:
```
<xarray.DataArray 't' (x: 5)> Size: 40B
array([ 1., 10.,  3., 30., 40.])
Coordinates:
  * x        (x) int64 40B 0 1 2 3 4
```

**注意点・落とし穴**:
- 自分の値が優先され、自分が NaN(または欠けている座標)の位置だけ相手の値が使われる。x=2 では自分の 3.0 が優先されている点に注意。
- 座標の和集合に広がるので、結果の長さは元より増える(上の例は x=0〜4 の 5 要素)。

### `xr.align(...)`

**用途**: 複数のオブジェクトの座標を揃えた(同じインデックスを持つ)コピーを返す。演算前の明示的な整列に使う。

**シグネチャ**: xr.align(*objects, join='inner', copy=True, indexes=None, exclude=frozenset(), fill_value=<NA>)

**使用例**:
```python
a = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [0, 1]}, name="t")
b = xr.DataArray([5.0, 6.0, 7.0], dims="x", coords={"x": [1, 2, 3]}, name="p")

a_in, b_in = xr.align(a, b, join="inner")
print(a_in.x.values, b_in.x.values)

a_out, b_out = xr.align(a, b, join="outer")
print(a_out.values)
print(b_out.values)

a_def, b_def = xr.align(a, b)   # join を省略(既定は "inner")
print(a_def.x.values)
```
実行結果:
```
[1] [1]
[ 1.  2. nan nan]
[nan  5.  6.  7.]
[1]
```

**注意点・落とし穴**:
- `xr.align` の既定は `join="inner"`(共通ラベルのみ残す)。`join="exact"` を指定すると座標が完全一致でなければ `AlignmentError` になるので、「揃っているはず」の検証に使える。演算子(`+` など)の既定も `inner` だが、`xr.where` や `xr.apply_ufunc` の既定は `exact` なので混同しないこと。
- `join="outer"` で拡張した欠損の埋め値は `fill_value=` で指定できる(既定は NaN で、整数は float に昇格)。
- 同じ目的で `Dataset`/`DataArray` の `.reindex_like(other)` を使う方法もある(次項)。

### `da.reindex(...) / da.reindex_like(...)`

**用途**: 指定した座標に合わせて配列を作り直す。存在しないラベルは NaN(または `fill_value`)、または最近傍などで埋める。

**シグネチャ**: da.reindex(indexers=None, method=None, tolerance=None, copy=True, fill_value=<NA>, **indexers_kwargs) / da.reindex_like(other, method=None, tolerance=None, copy=True, fill_value=<NA>)

**使用例**:
```python
da = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [0, 1]}, name="t")
print(da.reindex(x=[1, 2, 3]))
print(da.reindex(x=[1, 2, 3], fill_value=0).values)
print(da.reindex(x=[0.4, 1.6], method="nearest").values)
other = xr.DataArray([0, 0, 0], dims="x", coords={"x": [1, 2, 3]})
print(da.reindex_like(other).values)
```
実行結果:
```
<xarray.DataArray 't' (x: 3)> Size: 24B
array([ 2., nan, nan])
Coordinates:
  * x        (x) int64 24B 1 2 3
[2. 0. 0.]
[1. 2.]
[ 2. nan nan]
```

**注意点・落とし穴**:
- `sel` は存在しないラベルで `KeyError` だが、`reindex` は存在しないラベルを許し、NaN で埋める。ラベルの並べ替え・拡張・間引きを 1 回で行える。
- `method="nearest"` などは「最も近い既存ラベルの値」を持ってくるだけで、値の補間ではない。線形補間などをしたい場合は `interp`(「時系列・補間」)を使う。
- 元の座標に重複ラベルがあると `reindex` は `ValueError`(`the (pandas) index has duplicate values`)になる。

### `xr.broadcast(...)`

**用途**: 複数の配列を、次元名で同じ形にブロードキャストして返す(演算はしない)。

**シグネチャ**: xr.broadcast(*args, exclude=None)

**使用例**:
```python
a = xr.DataArray([1, 2], dims="x")
b = xr.DataArray([1, 2, 3], dims="y")
a2, b2 = xr.broadcast(a, b)
print(a2.dims, a2.shape)
print(a2.values)
print(b2.values)
```
実行結果:
```
('x', 'y') (2, 3)
[[1 1 1]
 [2 2 2]]
[[1 2 3]
 [1 2 3]]
```

**注意点・落とし穴**:
- ブロードキャスト後は全配列が同じ次元集合になる(次元の並びは引数の登場順)。2 つの配列を `np.stack` したいときなどに使う。
- 既存の配列を別の配列の形に合わせるだけなら `a.broadcast_like(b)` が簡単。

### `xr.combine_by_coords(...)`

**用途**: 座標値をもとに、タイル状に分割された Dataset(複数ファイルなど)を自動で並べ替えて 1 つに結合する。

**シグネチャ**: xr.combine_by_coords(data_objects=[], compat=no_conflicts, data_vars=all, coords=different, fill_value=<NA>, join=outer, combine_attrs='no_conflicts')

**使用例**:
```python
ds = xr.Dataset({"t": ("x", [1.0, 2.0, 3.0, 4.0])}, coords={"x": [0, 1, 2, 3]})
parts = [ds.isel(x=[2, 3]), ds.isel(x=[0, 1])]       # 順番がバラバラの断片
print(xr.combine_by_coords(parts))
```
実行結果:
```
<xarray.Dataset> Size: 64B
Dimensions:  (x: 4)
Coordinates:
  * x        (x) int64 32B 0 1 2 3
Data variables:
    t        (x) float64 32B 1.0 2.0 3.0 4.0
```

**注意点・落とし穴**:
- 隙間があってもそのまま連結される(`x=[0]` と `x=[3]` を渡すと `x=[0, 3]` になる)。重複する座標もチェックされず、重複したまま連結された(`x=[0,1]` と `x=[1,2]` から `x=[0,1,1,2]`)ので、断片が重ならないか自分で確認すること。座標がないデータや並び順を明示したいときは `xr.combine_nested(..., concat_dim=...)` を使う。
- 多数のファイルの結合には `xr.open_mfdataset` があるが、dask が必要(dask なしで呼ぶと `ImportError: chunk manager 'dask' is not available` になることを確認)なため例は載せていない。

---

## 変形

### `da.transpose(...) / da.T`

**用途**: 次元の並び順を入れ替える。`...`(Ellipsis)で「残りの次元」を指定できる。

**シグネチャ**: da.transpose(*dim, transpose_coords=True, missing_dims='raise')

**使用例**:
```python
da = xr.DataArray(
    np.arange(24).reshape(2, 3, 4),
    dims=("a", "b", "c"),
    coords={"a": [0, 1], "b": list("xyz"), "c": [10, 20, 30, 40]},
)
print(da.transpose("c", "a", "b").dims)
print(da.transpose(..., "a").dims)     # a を最後にし、残りは元の順序
print(da.T.dims)                       # 次元順を全て逆転
```
実行結果:
```
('c', 'a', 'b')
('b', 'c', 'a')
('c', 'b', 'a')
```

**注意点・落とし穴**:
- xarray の演算は次元名で整列するので、`transpose` が必要になるのは主に numpy や matplotlib に渡す直前(`.values` の軸順を揃えたいとき)や、次元順を明示的に固定したいとき。
- `da.transpose("c", "a")` のように全次元を列挙しない場合は `ValueError` になる。一部だけ指定したいときは `...` を使う。

### `da.stack(...) / da.unstack(...)`

**用途**: 複数の次元を 1 つの MultiIndex 次元にまとめる(`stack`)/ まとめた次元を元の複数次元に戻す(`unstack`)。

**シグネチャ**: da.stack(dim=None, create_index=True, index_cls=<class 'xarray.core.indexes.PandasMultiIndex'>, **dim_kwargs) / da.unstack(dim=None, fill_value=<NA>, sparse=False)

**使用例**:
```python
da = xr.DataArray(
    np.arange(24).reshape(2, 3, 4),
    dims=("a", "b", "c"),
    coords={"a": [0, 1], "b": list("xyz"), "c": [10, 20, 30, 40]},
)
s = da.stack(z=("a", "b"))
print(s)
print(s.sel(z=(0, "y")).values)          # MultiIndex のタプルで選択
print(s.unstack("z").dims)               # 元に戻すと次元順は (c, a, b)
print(s.unstack("z").transpose("a", "b", "c").equals(da))
```
実行結果:
```
<xarray.DataArray (c: 4, z: 6)> Size: 192B
array([[ 0,  4,  8, 12, 16, 20],
       [ 1,  5,  9, 13, 17, 21],
       [ 2,  6, 10, 14, 18, 22],
       [ 3,  7, 11, 15, 19, 23]])
Coordinates:
  * c        (c) int64 32B 10 20 30 40
  * z        (z) object 48B MultiIndex
  * a        (z) int64 48B 0 0 0 1 1 1
  * b        (z) <U1 24B 'x' 'y' 'z' 'x' 'y' 'z'
[4 5 6 7]
('c', 'a', 'b')
True
```

**注意点・落とし穴**:
- `stack` した新しい次元は末尾に置かれる(上の結果は `(c: 4, z: 6)`)。`unstack` すると次元順が元と違うので、必要なら `transpose` で戻す。
- MultiIndex を作りたくなければ `create_index=False`。元の `a`,`b` は `z` に沿った非インデックス座標として残る(値は保たれる)。
- scikit-learn などに渡す 2 次元(サンプル × 特徴)配列を作る用途にも使える(`da.stack(sample=(...)).transpose("sample", ...)`)。

### `da.expand_dims(...) / da.squeeze(...)`

**用途**: 長さ 1 の新しい次元を追加する(`expand_dims`)/ 長さ 1 の次元を削除する(`squeeze`)。

**シグネチャ**: da.expand_dims(dim=None, axis=None, create_index_for_new_dim=True, **dim_kwargs) / da.squeeze(dim=None, drop=False, axis=None)

**使用例**:
```python
da = xr.DataArray(
    np.arange(6).reshape(2, 3),
    dims=("a", "b"),
    coords={"a": [0, 1], "b": list("xyz")},
    name="v",
)
print(da.expand_dims("t").dims)                # 座標なしの新次元(先頭に追加)
print(da.expand_dims(t=[5]).coords["t"].values) # 座標つきの新次元
print(da.expand_dims({"t": 2}).shape)          # 長さ 2 に複製(ブロードキャスト)

one = da.isel(a=[0])
print(one.dims, one.squeeze().dims, one.squeeze("a").dims)
```
実行結果:
```
('t', 'a', 'b')
[5]
(2, 2, 3)
('a', 'b') ('b',) ('b',)
```

**注意点・落とし穴**:
- `expand_dims` は新しい次元を既定で先頭に追加する。`axis=` で挿入位置を指定できる。
- `expand_dims({"t": 2})` のように整数を渡すと長さ 2 に複製された次元になる(実体はコピーされずビューになっており、書き込もうとすると `ValueError: Assignment destination is a view` になる。書き込みたいときは `.copy()` してから)。
- `squeeze()` は長さ 1 の全次元を削除する。意図しない次元まで消えるのを避けるには `squeeze("a")` のように名前を指定する。スカラー選択(`sel(a=0)`)なら次元は最初から消えるので通常は不要。

### `da.swap_dims(...) / da.set_index(...) / da.reset_index(...)`

**用途**: 次元座標として使う座標を入れ替える(`swap_dims`/`set_index`)、または次元座標を通常の座標に降格する(`reset_index`)。

**シグネチャ**: da.swap_dims(dims_dict=None, **dims_kwargs) / da.set_index(indexes=None, append=False, **indexes_kwargs) / da.reset_index(dims_or_levels, drop=False)

**使用例**:
```python
da = xr.DataArray(
    np.arange(4),
    dims="c",
    coords={"c": [10, 20, 30, 40], "lab": ("c", list("pqrs"))},
    name="v",
)
print(da.swap_dims({"c": "lab"}))
print(da.set_index(c="lab").c.values)        # 次元 c の座標値を lab で置き換え
print(da.reset_index("c").coords)
```
実行結果:
```
<xarray.DataArray 'v' (lab: 4)> Size: 32B
array([0, 1, 2, 3])
Coordinates:
  * lab      (lab) <U1 16B 'p' 'q' 'r' 's'
    c        (lab) int64 32B 10 20 30 40
['p' 'q' 'r' 's']
Coordinates:
    c        (c) int64 32B 10 20 30 40
    lab      (c) <U1 16B 'p' 'q' 'r' 's'
```

**注意点・落とし穴**:
- `swap_dims({"c": "lab"})` は次元名自体が `lab` に変わり、元の座標 `c` は非次元座標として残る。`set_index(c="lab")` は次元名 `c` のまま、座標値だけ `lab` に置き換える(元の `c` の値は消える)。
- 次元座標にしたい座標は 1 次元で、その次元と同じ次元に沿っている必要がある。`swap_dims` 後は `da.sel(lab="q")` のようなラベル選択が使える。
- `reset_index("c")` は `c` のインデックスを外す。値は `c` という名前の、インデックスを持たない通常の座標として残る(上の出力で `*` が付かなくなる)ため、`sel(c=...)` は効率的なインデックス検索ではなくなる。

### `ds.to_array(...) / da.to_dataset(...)`

**用途**: Dataset の複数変数を新しい次元に積んで 1 つの DataArray にする(`to_array`)/ DataArray を Dataset に変換する(`to_dataset`、`dim=` で次元を変数に展開)。

**シグネチャ**: ds.to_array(dim='variable', name=None) / da.to_dataset(dim=None, name=None, promote_attrs=False)

**使用例**:
```python
ds = xr.Dataset({"u": ("x", [1, 2]), "v": ("x", [3, 4])})
arr = ds.to_array("variable")
print(arr)
print(arr.to_dataset(dim="variable"))
print(list(xr.DataArray([1, 2], dims="x", name="q").to_dataset().data_vars))
```
実行結果:
```
<xarray.DataArray (variable: 2, x: 2)> Size: 32B
array([[1, 2],
       [3, 4]])
Coordinates:
  * variable  (variable) object 16B 'u' 'v'
Dimensions without coordinates: x
<xarray.Dataset> Size: 32B
Dimensions:  (x: 2)
Dimensions without coordinates: x
Data variables:
    u        (x) int64 16B 1 2
    v        (x) int64 16B 3 4
['q']
```

**注意点・落とし穴**:
- `to_array` は全変数が同じ次元・dtype 互換であるほど扱いやすい(dtype が違うと共通の型に揃えられる(int64 と float64 の変数なら float64 になることを確認))。変数名が新次元(ここでは `variable`)の座標になる。
- `da.to_dataset(name=...)` は名前のない DataArray に名前を与えて Dataset にする。名前がなく `name=` も無いと `ValueError`(unable to convert unnamed DataArray ...)になる。

### `da.broadcast_like(...) / ds.rename_dims(...) / da.reset_coords(...)`

**用途**: 他の配列と同じ形にそろえる(`broadcast_like`)、次元名だけを変更する(Dataset の `rename_dims`)、座標を変数に降格する(`reset_coords`)。

**シグネチャ**: da.broadcast_like(other, exclude=None) / ds.rename_dims(dims_dict=None, **dims) / da.reset_coords(names=None, drop=False)

**使用例**:
```python
a = xr.DataArray([1, 2], dims="x", coords={"x": [0, 1]}, name="a")
b = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"), coords={"x": [0, 1], "y": [5, 6, 7]})
print(a.broadcast_like(b).dims)

da = xr.DataArray([1, 2], dims="x", coords={"x": [0, 1], "lab": ("x", ["p", "q"])}, name="v")
print(list(da.reset_coords("lab").data_vars))           # Dataset になり lab は変数に
print(list(da.reset_coords("lab", drop=True).coords))   # 捨てる

ds = xr.Dataset({"u": ("x", [1, 2])})
print(dict(ds.rename_dims(x="X").sizes))                # Dataset の次元名を変更
```
実行結果:
```
('x', 'y')
['lab', 'v']
['x']
{'X': 2}
```

**注意点・落とし穴**:
- `reset_coords(names, drop=False)`: `drop=False` のとき DataArray は Dataset になる(座標が変数になるため)。`drop=True` なら座標を単に捨てて DataArray のまま。
- DataArray には `rename_dims` がない(`AttributeError`)。DataArray の次元名は `da.rename({"x": "X"})` で変える。Dataset なら `ds.rename_dims(x="X")` が使える。

---

## 欠損値

### `da.isnull() / da.notnull()`

**用途**: 欠損値(NaN)かどうかの真偽値配列を返す。

**シグネチャ**: da.isnull(keep_attrs=None) / da.notnull(keep_attrs=None)

**使用例**:
```python
da = xr.DataArray([1.0, np.nan, np.nan, 4.0, np.nan], dims="x", coords={"x": [0, 1, 2, 3, 4]})
print(da.isnull().values)
print(da.notnull().sum().item())     # 有効値の個数
print(da.where(da.notnull(), -1).values)
```
実行結果:
```
[False  True  True False  True]
2
[ 1. -1. -1.  4. -1.]
```

**注意点・落とし穴**:
- 整数配列には NaN がないので `isnull()` は全て False になる。NaN を扱えるのは float などの dtype だけ。
- 有効値の個数は `da.count()` でも取れる。

### `da.fillna(...)`

**用途**: 欠損値を、スカラー・別の DataArray・Dataset なら変数ごとの dict で埋める。

**シグネチャ**: da.fillna(value)

**使用例**:
```python
da = xr.DataArray([1.0, np.nan, np.nan, 4.0, np.nan], dims="x", coords={"x": [0, 1, 2, 3, 4]})
print(da.fillna(0).values)
print(da.fillna(da.mean()).values)          # 平均で埋める

A = xr.DataArray([[1.0, np.nan], [np.nan, np.nan], [3.0, 4.0]], dims=("y", "x"))
print(A.fillna(A.mean("y")).values)         # 列(x)ごとの平均で埋める(次元名でブロードキャスト)

ds = xr.Dataset({"a": ("x", [1.0, np.nan]), "b": ("x", [np.nan, np.nan])})
print(ds.fillna({"a": 0, "b": -1}).to_array().values)
```
実行結果:
```
[1. 0. 0. 4. 0.]
[1.  2.5 2.5 4.  2.5]
[[1. 4.]
 [2. 4.]
 [3. 4.]]
[[ 1.  0.]
 [-1. -1.]]
```

**注意点・落とし穴**:
- `fillna` に別の DataArray を渡すと次元名・座標でブロードキャストされる(上の `A.mean("y")` は `x` に沿った配列)。
- Dataset の `fillna` は変数ごとに値を変えたいとき dict を渡す。

### `da.interpolate_na(...)`

**用途**: 欠損値を、前後の有効値から補間して埋める(既定は線形補間、座標の値を使う)。

**シグネチャ**: da.interpolate_na(dim=None, method='linear', limit=None, use_coordinate=True, max_gap=None, keep_attrs=None, **kwargs)

**使用例**:
```python
da = xr.DataArray([1.0, np.nan, np.nan, 4.0, np.nan], dims="x", coords={"x": [0, 1, 2, 3, 4]})
print(da.interpolate_na("x").values)                        # 端の NaN は埋まらない
print(da.interpolate_na("x", limit=1).values)               # 連続 NaN は最大 1 個まで
print(da.interpolate_na("x", fill_value="extrapolate").values)  # 端も外挿
print(da.interpolate_na("x", method="nearest").values)

db = xr.DataArray([1.0, np.nan, np.nan, 4.0, np.nan], dims="x", coords={"x": [0, 1, 2, 10, 11]})
print(db.interpolate_na("x").values)                        # 座標値(間隔)を考慮
print(db.interpolate_na("x", use_coordinate=False).values)  # 位置ベース(等間隔扱い)
```
実行結果:
```
[ 1.  2.  3.  4. nan]
[ 1.  2. nan  4. nan]
[1. 2. 3. 4. 5.]
[ 1.  1.  4.  4. nan]
[1.  1.3 1.6 4.  nan]
[ 1.  2.  3.  4. nan]
```

**注意点・落とし穴**:
- 既定では座標の値を横軸に使って補間する(上の `db` は `x=0`(値 1)から `x=10`(値 4)までの間にある `x=1, 2` が、座標の間隔に比例して 1.3, 1.6 になる。`use_coordinate=False` では位置ベースで 2, 3)。等間隔として扱いたいなら `use_coordinate=False`。
- 既定は外挿しない(先頭・末尾の欠損は残る)。埋めたいなら `fill_value="extrapolate"`。
- `limit=` は「連続する欠損のうち最大何個まで埋めるか」。`max_gap=` は docstring によると「埋める対象とする欠損の連続区間(gap)の最大の大きさ。gap の長さは座標値の差で定義される」で、本環境(bottleneck 未導入)では `max_gap=1` を指定すると `ModuleNotFoundError: No module named 'bottleneck'` になった。
- `method=` に `"cubic"` などを使うには scipy が必要(本環境では導入済み)。`"cubic"` は有効値が少ないと `ValueError`(有効値 2 点の例で `The number of derivatives at boundaries does not match` を確認)になる。

### `da.ffill(...) / da.bfill(...)`

**用途**: 欠損値を直前(`ffill`)/直後(`bfill`)の有効値で埋める。

**シグネチャ**: da.ffill(dim, limit=None) / da.bfill(dim, limit=None)

**使用例**:
```python
da = xr.DataArray([1.0, np.nan, np.nan, 4.0, np.nan], dims="x", coords={"x": [0, 1, 2, 3, 4]}, name="v")
try:
    da.ffill("x")
except ModuleNotFoundError as e:
    print(type(e).__name__, e)

# 代替: pandas 経由で前方埋めして戻す
print(da.to_series().ffill().to_xarray().values)
print(da.to_series().ffill(limit=1).to_xarray().values)
```
実行結果:
```
ModuleNotFoundError No module named 'bottleneck'
[1. 1. 1. 4. 4.]
[ 1.  1. nan  4.  4.]
```

**注意点・落とし穴**:
- xarray の `ffill`/`bfill` は docstring に「Requires bottleneck」とあり、`bottleneck` が必要。本環境では未導入のため、`ffill("x")` は `ModuleNotFoundError: No module named 'bottleneck'` になった。使いたい場合は `bottleneck` を導入する(導入後の出力は本書では未検証)。
- 1 次元なら `da.to_series().ffill().to_xarray()` のように pandas を経由して代替できる(上の例で確認)。Dataset でも `ds.to_dataframe().ffill().to_xarray()` の形で同様に前方埋めできることを確認した(x のみの 1 次元の例で確認。多次元では未検証)。
- 「直前の値で埋める」目的は、`interpolate_na(method="nearest")` とは異なる(最近傍は前後どちらか近いほう)。

### `da.dropna(...)`

**用途**: 欠損値を含む座標ラベル(行・列)を除去する。`how` と `thresh` で条件を指定する。

**シグネチャ**: da.dropna(dim, how='any', thresh=None)

**使用例**:
```python
A = xr.DataArray([[1.0, np.nan], [np.nan, np.nan], [3.0, 4.0]], dims=("y", "x"))
print(A.dropna("y").values)                 # 1 つでも NaN を含む y を落とす(how="any")
print(A.dropna("y", how="all").values)      # 全て NaN の y のみ落とす
print(A.dropna("y", thresh=2).values)       # 有効値が 2 個以上の y のみ残す
```
実行結果:
```
[[3. 4.]]
[[ 1. nan]
 [ 3.  4.]]
[[3. 4.]]
```

**注意点・落とし穴**:
- `dropna` は 1 つの次元を指定して、その次元のラベル単位(行・列)で除去する。配列の形は矩形のままなので、要素単位で NaN だけを消したいときは `A.stack(z=("y", "x")).dropna("z")` のように一度 1 次元にまとめる(この例では `[1., 3., 4.]` が残ることを確認)。
- Dataset の `dropna` は全変数を見て判定する(`how="all"` なら全変数が NaN のラベルだけ落とす)。

---

## 時系列・補間

### `da.interp(...)`

**用途**: 座標軸に沿って、新しい座標点での値を補間して求める(既定は線形補間)。内部で scipy を使う。

**シグネチャ**: da.interp(coords=None, method='linear', assume_sorted=False, kwargs=None, **coords_kwargs)

**使用例**:
```python
da = xr.DataArray([0.0, 10.0, 20.0, 30.0], dims="x", coords={"x": [0, 10, 20, 30]})
print(da.interp(x=[5, 15, 25]).values)
print(da.interp(x=[-5, 35]).values)                                   # 範囲外は NaN
print(da.interp(x=[-5, 35], kwargs={"fill_value": "extrapolate"}).values)   # 外挿

q = xr.DataArray([0.0, 1.0, 4.0, 9.0], dims="x", coords={"x": [0, 1, 2, 3]})
print(q.interp(x=[0.5, 1.5], method="linear").values)
print(q.interp(x=[0.5, 1.5], method="quadratic").values)   # x^2 に厳密に一致

g = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1, 2]})
print(g.interp(y=0.5, x=[0.5, 1.5]))                         # 2 次元補間

# 日時座標: 時刻は Timestamp / datetime64 で渡す
v = xr.DataArray(np.arange(6.0), dims="time", coords={"time": pd.date_range("2024-01-01", periods=6, freq="D")})
print(v.interp(time=pd.Timestamp("2024-01-02 12:00")).item())
print(v.interp(time="2024-01-02T12:00").item())              # 文字列だと NaN になる
```
実行結果:
```
[ 5. 15. 25.]
[nan nan]
[-5. 35.]
[0.5 2.5]
[0.25 2.25]
<xarray.DataArray (x: 2)> Size: 16B
array([2., 3.])
Coordinates:
  * x        (x) float64 16B 0.5 1.5
    y        float64 8B 0.5
1.5
nan
```

**注意点・落とし穴**:
- 範囲外は既定で NaN。外挿したいときは `kwargs={"fill_value": "extrapolate"}` を渡す。
- `interp` は scipy が必須(`method="linear"` でも、scipy を import できない状態では `ModuleNotFoundError: No module named 'scipy.interpolate'` になることを確認)。本環境は scipy 導入済み。`method="nearest"`/`"quadratic"`/`"cubic"` なども使える。`sel(method="nearest")` が最近傍の「既存ラベル」を返すのに対し、`interp` は新しい座標値そのものを結果の座標にする。
- 座標が数値(または日時)の場合のみ補間できる。文字列座標に対して呼ぶと `UFuncTypeError` になる。
- 2026.2.0(pandas 3 + `datetime64[us]` の時間座標)では、時刻を文字列で渡すと NaN になった。`da.interp(time="2024-01-02T12:00")` は `nan`、`pd.Timestamp(...)` や `np.datetime64(...)`、`pd.date_range(...)` で渡すと正しく補間される(上の例で確認)。

### `da.interp_like(...)`

**用途**: 別の配列の座標に合わせて補間する(`reindex_like` の補間版)。

**シグネチャ**: da.interp_like(other, method='linear', assume_sorted=False, kwargs=None)

**使用例**:
```python
g = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1, 2]})
target = xr.DataArray(np.zeros(2), dims="x", coords={"x": [0.5, 1.5]})
print(g.interp_like(target).values)     # x が target の座標に補間され、y はそのまま
```
実行結果:
```
[[0.5 1.5]
 [3.5 4.5]]
```

**注意点・落とし穴**:
- `target` が持たない次元(ここでは `y`)は元のまま残る。両者に共通の次元だけが補間対象になる。

### `da.resample(...)`

**用途**: 時間軸を新しい頻度(`"2D"`, `"W"`, `"ME"` など)にまとめ直し、`mean`/`sum` などで集約する(ダウン/アップサンプリング)。

**シグネチャ**: da.resample(indexer=None, skipna=None, closed=None, label=None, offset=None, origin='start_day', restore_coord_dims=None, **indexer_kwargs)

**使用例**:
```python
time = pd.date_range("2024-01-01", periods=6, freq="D")
v = xr.DataArray(np.arange(6.0), dims="time", coords={"time": time})

print(v.resample(time="2D").mean())                 # 2 日ごとの平均
print(v.resample(time="2D").sum().values)
print(v.resample(time="2D", closed="right", label="right").mean())
print(v.resample(time="ME").mean())                 # 月末ごと
print(v.resample(time="W").sum().time.values)       # 週ごと(日曜ラベル)
print(v.resample(time="12h").interpolate("linear").values)   # アップサンプリング + 補間
print(v.isel(time=[0, 1, 4]).resample(time="D").mean().values)  # 歯抜けを NaN で補完
```
実行結果:
```
<xarray.DataArray (time: 3)> Size: 24B
array([0.5, 2.5, 4.5])
Coordinates:
  * time     (time) datetime64[us] 24B 2024-01-01 2024-01-03 2024-01-05
[1. 5. 9.]
<xarray.DataArray (time: 4)> Size: 32B
array([0. , 1.5, 3.5, 5. ])
Coordinates:
  * time     (time) datetime64[us] 32B 2024-01-01 2024-01-03 ... 2024-01-07
<xarray.DataArray (time: 1)> Size: 8B
array([2.5])
Coordinates:
  * time     (time) datetime64[us] 8B 2024-01-31
['2024-01-07T00:00:00.000000']
[0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]
[ 0.  1. nan nan  4.]
```

**注意点・落とし穴**:
- `resample` は日時の座標を持つ次元が対象。`resample(time="2D")` のように次元名をキーワード、頻度文字列を値として渡す。
- 月末は `"ME"`(month end)。旧称の `"M"` は `ValueError: ... 'M' is no longer supported for offsets. Please use 'ME' instead.` になる(pandas 3.0.5 で確認)。
- アップサンプリングは `.asfreq()`(新しい時刻は NaN)、`.ffill()`(直前の値で埋める。`v.resample(time="12h").ffill()` は bottleneck なしで動作を確認)、`.interpolate("linear")` などを選ぶ。
- `closed`/`label` で区間の端の扱い・ラベル位置を変えられる。週次(`"W"`)は日曜終わりの週で、ラベルは区間の右端(日曜日)になる(上の例の 2024-01-01(月)〜01-06(土)の 6 日間は、ラベル `2024-01-07` の 1 週にまとまる)。

### `da.time.dt.(属性) / da.sel(time=...) / groupby("time.month")`

**用途**: 日時座標の要素(年・月・曜日など)を取り出す(`.dt`)、文字列で日時範囲を選ぶ、月・曜日などでグループ化して集約する。

**シグネチャ**: `da.time.dt` アクセサ、文字列/`slice` による日時選択、日時属性でのグループ化

**使用例**:
```python
time = pd.date_range("2024-01-01", periods=10, freq="D")
v = xr.DataArray(np.arange(10.0), dims="time", coords={"time": time})

print(v.time.dtype)                                 # pandas 3.0 の date_range は us 精度
print(v.time.dt.dayofweek.values)                   # 月曜=0
print(v.time.dt.strftime("%Y/%m/%d").values[:2])
print(v.sel(time="2024-01-03").values)              # 文字列で 1 日を選択
print(v.sel(time=slice("2024-01-02", "2024-01-04")).values)   # 両端を含む範囲
print(dict(v.sel(time="2024-01").sizes))            # 月単位の部分文字列
print(v.groupby("time.dayofweek").mean())
print(v.groupby("time.month").sum())
print(v.assign_coords(month=v.time.dt.month).month.values)
```
実行結果:
```
datetime64[us]
[0 1 2 3 4 5 6 0 1 2]
['2024/01/01' '2024/01/02']
2.0
[1. 2. 3.]
{'time': 10}
<xarray.DataArray (dayofweek: 7)> Size: 56B
array([3.5, 4.5, 5.5, 3. , 4. , 5. , 6. ])
Coordinates:
  * dayofweek  (dayofweek) int64 56B 0 1 2 3 4 5 6
<xarray.DataArray (month: 1)> Size: 8B
array([45.])
Coordinates:
  * month    (month) int64 8B 1
[1 1 1 1 1 1 1 1 1 1]
```

**注意点・落とし穴**:
- `sel(time="2024-01")` のような部分文字列(月単位など)でその月全体を選べる(この例では全 10 日が返る)。`slice("2024-01-02", "2024-01-04")` は両端を含む。
- `groupby("time.month")`(月)、`"time.dayofweek"`(曜日)、`"time.year"` などの日時属性名が使える。月・年のような文字列キーは結果の次元名がそのまま `month`/`dayofweek` になる。
- pandas 3.0 では `pd.date_range` が `datetime64[us]`(マイクロ秒)の時刻を返し、DataArray の座標もその dtype になる(上の例の `v.time.dtype` の出力)。`interp` に時刻を文字列で渡すと NaN になるなど、`datetime64[ns]` を前提にした古いコードと挙動が異なる場合がある。

### `da.differentiate(...) / da.integrate(...)`

**用途**: 座標を横軸として、数値微分(中心差分)・台形則の数値積分を行う。時間軸では `datetime_unit` で単位を指定する。

**シグネチャ**: da.differentiate(coord, edge_order=1, datetime_unit=None) / da.integrate(coord=None, datetime_unit=None)

**使用例**:
```python
time = pd.date_range("2024-01-01", periods=6, freq="D")
v = xr.DataArray(np.arange(6.0), dims="time", coords={"time": time})
print(v.differentiate("time", datetime_unit="D").values)   # 1 日あたりの変化率
print(v.integrate("time", datetime_unit="D").item())       # 台形則

x = xr.DataArray([0.0, 1.0, 4.0, 9.0], dims="x", coords={"x": [0, 1, 2, 3]})
print(x.differentiate("x").values)
```
実行結果:
```
[1. 1. 1. 1. 1. 1.]
12.5
[1. 2. 4. 5.]
```

**注意点・落とし穴**:
- 日時座標では `datetime_unit`(`"D"`, `"h"`, `"s"` など)が必要。省略するとナノ秒あたりで計算される(この例の `differentiate` は 1 日 1 増加のデータで `1.157e-11` = 1/86400e9 になった)。
- `differentiate` は中心差分で、両端は片側差分(`np.gradient([0., 1, 4, 9])` も同じ `[1, 2, 4, 5]` になることを確認)。

---

## pandas連携

### `da.to_series() / da.to_dataframe() / ds.to_dataframe() / da.to_pandas()`

**用途**: xarray の配列を pandas の Series / DataFrame に変換する。次元は行の MultiIndex(または列・行)になる。

**シグネチャ**: da.to_series() / da.to_dataframe(name=None, dim_order=None) / ds.to_dataframe(dim_order=None) / da.to_pandas()

**使用例**:
```python
da = xr.DataArray(
    np.arange(6).reshape(2, 3),
    dims=("y", "x"),
    coords={"y": ["a", "b"], "x": [10, 20, 30]},
    name="temp",
)
print(da.to_series())                          # 次元が MultiIndex の縦持ち
print(da.to_dataframe())                       # 変数名が列名(name が必須)
print(da.to_pandas())                          # 2 次元 -> 行=y, 列=x のワイド形式
print(type(da.isel(y=0).to_pandas()))          # 1 次元 -> Series

ds = xr.Dataset({"t": da, "p": da * 2})
print(ds.to_dataframe().reset_index().head(3)) # MultiIndex を通常の列に戻す
```
実行結果:
```
y  x 
a  10    0
   20    1
   30    2
b  10    3
   20    4
   30    5
Name: temp, dtype: int64
      temp
y x       
a 10     0
  20     1
  30     2
b 10     3
  20     4
  30     5
x  10  20  30
y            
a   0   1   2
b   3   4   5
<class 'pandas.Series'>
   y   x  t  p
0  a  10  0  0
1  a  20  1  2
2  a  30  2  4
```

**注意点・落とし穴**:
- `da.to_dataframe()` は DataArray に `name` がないと `ValueError`(メッセージは「cannot convert an unnamed DataArray to a DataFrame: use the name parameter」)になる。`da.to_dataframe(name="v")` で解決する。`to_series()` は名前なしでも動く。
- `to_pandas()` は次元数で型が変わる(1 次元→Series、2 次元→DataFrame)。0 次元は numpy のスカラー配列、3 次元以上は `ValueError: Cannot convert arrays with 3 dimensions into pandas objects` になる。3 次元以上を pandas に渡すなら `to_dataframe()`/`to_series()`(MultiIndex の縦持ち)を使う。
- 行の並びは `dims` の順序(最初の次元が最も外側のインデックス)で決まる。`dim_order=["x", "y"]` で並べ替えられる。
- 全次元の直積(全セルの組み合わせ)が行になるため、次元が多い・大きい配列では行数が爆発する。NaN のセルも行として残る(自動では除去されない)。

### `xr.Dataset.from_dataframe(...) / df.to_xarray() / xr.DataArray.from_series(...)`

**用途**: pandas の DataFrame / Series を xarray に変換する。インデックス(MultiIndex なら各レベル)が次元・座標になり、列が変数になる。

**シグネチャ**: xr.Dataset.from_dataframe(dataframe, sparse=False) / xr.DataArray.from_series(series, sparse=False)

**使用例**:
```python
df = pd.DataFrame(
    {"t": [1.0, 2.0, 3.0], "p": [10, 20, 30]},
    index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["y", "x"]),
)
print(xr.Dataset.from_dataframe(df))         # (b, 2) の組が無い部分は NaN

df2 = pd.DataFrame({"t": [1.0, 2.0, 3.0]}, index=pd.Index([10, 20, 30], name="x"))
print(df2.to_xarray())                       # pandas 側のメソッドも同じ結果

s = pd.Series([1, 2], index=pd.Index(["u", "v"], name="k"), name="val")
print(xr.DataArray.from_series(s))
```
実行結果:
```
<xarray.Dataset> Size: 96B
Dimensions:  (y: 2, x: 2)
Coordinates:
  * y        (y) object 16B 'a' 'b'
  * x        (x) int64 16B 1 2
Data variables:
    t        (y, x) float64 32B 1.0 2.0 3.0 nan
    p        (y, x) float64 32B 10.0 20.0 30.0 nan
<xarray.Dataset> Size: 48B
Dimensions:  (x: 3)
Coordinates:
  * x        (x) int64 24B 10 20 30
Data variables:
    t        (x) float64 24B 1.0 2.0 3.0
<xarray.DataArray 'val' (k: 2)> Size: 16B
array([1, 2])
Coordinates:
  * k        (k) object 16B 'u' 'v'
```

**注意点・落とし穴**:
- インデックスに名前がないと、次元名は `index`(単一 Index の場合)や `level_0`/`level_1`(MultiIndex の場合)になる。意味のある次元名にするには `names=[...]` を付けておく。
- MultiIndex の全組み合わせが揃っていなければ、欠けた組は NaN で埋められ、整数列は float64 に変換される(上の出力の `p` 列は整数から float64 に変わっている)。
- インデックスが重複しているデータは、変換自体は通るが重複ラベルのまま座標になる(2026.2.0 で確認)。そうすると後で `reindex` が `ValueError`(duplicate values)になるなど扱いづらいので、長い縦持ちデータは変換前に `df.groupby(...)` などで一意にしておく。
- pandas 3.0 の文字列 dtype(`str`)のインデックスは、xarray 側では座標の dtype が `object` になる(上の出力の `y`/`k` 座標で確認)。

### `da.to_dict(...) / xr.DataArray.from_dict(...)`

**用途**: DataArray / Dataset を、JSON 化しやすい辞書(データ・座標・次元・属性)に変換する / 辞書から復元する。

**シグネチャ**: da.to_dict(data='list', encoding=False) / xr.DataArray.from_dict(d)

**使用例**:
```python
da = xr.DataArray(
    np.arange(6).reshape(2, 3),
    dims=("y", "x"),
    coords={"y": ["a", "b"], "x": [10, 20, 30]},
    name="temp",
).isel(y=[0], x=[0, 1])

d = da.to_dict()
print(d)
print(xr.DataArray.from_dict(d))
print(list(da.to_dict(data=False)))         # データを含めず構造だけ
```
実行結果:
```
{'dims': ('y', 'x'), 'attrs': {}, 'data': [[0, 1]], 'coords': {'y': {'dims': ('y',), 'attrs': {}, 'data': ['a']}, 'x': {'dims': ('x',), 'attrs': {}, 'data': [10, 20]}}, 'name': 'temp'}
<xarray.DataArray 'temp' (y: 1, x: 2)> Size: 16B
array([[0, 1]])
Coordinates:
  * y        (y) <U1 4B 'a'
  * x        (x) int64 16B 10 20
['dims', 'attrs', 'dtype', 'shape', 'coords', 'name']
```

**注意点・落とし穴**:
- `data=False` にするとデータ本体を含めずに次元・座標・dtype・shape だけを返す(大きな配列のスキーマ確認に便利)。
- 数値データは Python の list/数値で返るので `json.dumps` にそのまま渡せる。ただし日時座標は `datetime.datetime` になり `TypeError: Object of type datetime is not JSON serializable` になる(確認済み)ため、文字列化などの前処理が必要。

---

## 入出力

### 実行環境で使える I/O バックエンド(`xr.backends.list_engines()`)

**用途**: 利用可能なバックエンド(読み書きエンジン)の一覧を確認する。どの形式の I/O 例が実行できるかの前提になる。

**シグネチャ**: xr.backends.list_engines()

**使用例**:
```python
print(list(xr.backends.list_engines()))
for mod in ["netCDF4", "h5netcdf", "zarr", "dask", "scipy", "bottleneck"]:
    try:
        __import__(mod)
        print(mod, "OK")
    except ImportError:
        print(mod, "なし")
```
実行結果:
```
['h5netcdf', 'scipy', 'store']
netCDF4 なし
h5netcdf OK
zarr なし
dask なし
scipy OK
bottleneck なし
```

**注意点・落とし穴**:
- 本書の検証環境では `h5netcdf`(と `scipy`)が使え、`netCDF4`・`zarr`・`dask`・`bottleneck` は未導入。そのため netCDF は `h5netcdf` エンジンで扱っており、Zarr(`to_zarr`/`open_zarr`)、dask による遅延・並列処理(`chunks=`、`open_mfdataset`)、netCDF4 エンジンの例は掲載していない(実行して動作確認できないため)。
- `to_zarr` を zarr なしで呼ぶと `ImportError: The zarr package is required for working with Zarr stores but could not be imported.`、`open_dataset(..., chunks={})` を dask なしで呼ぶと `ImportError: chunk manager 'dask' is not available.` になることを確認した。
- `engine=` を省略した場合の選択順は `xr.get_options()["netcdf_engine_order"]`(`('netcdf4', 'h5netcdf', 'scipy')`)。netCDF4 が無ければ h5netcdf が使われる。

### `ds.to_netcdf(...)`

**用途**: Dataset(または DataArray)を netCDF ファイルに書き出す。座標・属性・次元名・dtype が保存される。

**シグネチャ**: ds.to_netcdf(path=None, mode='w', format=None, group=None, engine=None, encoding=None, unlimited_dims=None, compute=True, invalid_netcdf=False, auto_complex=None)

**使用例**:
```python
ds = xr.Dataset(
    {"temp": (("time", "x"), np.arange(6.0).reshape(3, 2)), "pres": ("x", [1, 2])},
    coords={"time": pd.date_range("2024-01-01", periods=3), "x": [10, 20]},
    attrs={"title": "demo"},
)
ds["temp"].attrs["units"] = "degC"

ds.to_netcdf("sample.nc")                       # エンジンは自動選択(この環境では h5netcdf)
ds.to_netcdf("sample_h5.nc", engine="h5netcdf") # 明示指定
print(xr.open_dataset("sample.nc", engine="h5netcdf").identical(ds))   # 完全に復元されるか
```
実行結果:
```
True
```

**注意点・落とし穴**:
- ファイルは上書き(`mode="w"`)が既定。`mode="a"` で既存ファイルに変数を追加できる(既存の `temp` を残したまま `pres` を書き足せることを確認)。
- 同じファイルを `open_dataset` で開いたまま `to_netcdf` で上書きしようとすると `OSError: ... unable to truncate a file which is already open` になる。先に `.close()` するか、`.load()` してから閉じる。
- `engine="scipy"` は NetCDF3 形式で書く(そのため netCDF4/HDF5 形式のファイルは読めない)。読み込み側で `engine="scipy"` を指定すると `TypeError: ... is not a valid NetCDF 3 file` になった。
- `ds.to_netcdf()` のようにパスを省略すると、ファイルではなく `memoryview`(バイト列)が返る。
- `group="sub"` を指定すると netCDF4/HDF5 形式のグループに書ける。読み込みも `group=` で同じ名前を指定する(指定しないとルートグループだけが開かれ、`sizes` は空になった)。

### `xr.open_dataset(...)`

**用途**: netCDF ファイルを開いて Dataset として返す。データは遅延読み込み(必要になるまでディスクから読まない)。

**シグネチャ**: xr.open_dataset(filename_or_obj, engine=None, chunks=None, cache=None, decode_cf=None, mask_and_scale=None, decode_times=None, decode_timedelta=None, ...)

**使用例**:
```python
ds = xr.Dataset(
    {"temp": (("time", "x"), np.arange(6.0).reshape(3, 2))},
    coords={"time": pd.date_range("2024-01-01", periods=3), "x": [10, 20]},
)
ds.to_netcdf("sample.nc")

r = xr.open_dataset("sample.nc")
print(r)                                   # 変数の値は "..." (未読み込み)
print(r.temp.values[:1])                   # 値にアクセスして初めて読み込む
print(r.time.encoding["units"])            # ファイル上の時刻の単位が encoding に入る
r.close()

with xr.open_dataset("sample.nc") as f:    # with 構文で確実に close
    print(f.temp.mean().item())

print(xr.open_dataset("sample.nc", drop_variables=["temp"]))       # 読まない変数を指定
print(xr.open_dataset("sample.nc", decode_times=False).time.values) # 時刻をデコードしない
```
実行結果:
```
<xarray.Dataset> Size: 88B
Dimensions:  (time: 3, x: 2)
Coordinates:
  * time     (time) datetime64[ns] 24B 2024-01-01 2024-01-02 2024-01-03
  * x        (x) int64 16B 10 20
Data variables:
    temp     (time, x) float64 48B ...
[[0. 1.]]
days since 2024-01-01 00:00:00
2.5
<xarray.Dataset> Size: 40B
Dimensions:  (time: 3, x: 2)
Coordinates:
  * time     (time) datetime64[ns] 24B 2024-01-01 2024-01-02 2024-01-03
  * x        (x) int64 16B 10 20
Data variables:
    *empty*
[0 1 2]
```

**注意点・落とし穴**:
- `open_dataset` はファイルハンドルを開きっぱなしにする。使い終わったら `.close()` するか `with` で開く。開いたまま同じパスに `to_netcdf` すると上書きできない。
- 時刻座標は CF 規約(`units: "days since ..."`)に従って `datetime64` にデコードされる。読み込み後の `time.encoding["units"]` は `days since 2024-01-01 00:00:00`(検証環境で確認)。`decode_times=False` なら元の整数値(0, 1, 2)がそのまま出る。
- `chunks=` を指定すると dask 配列で遅延処理されるが、本環境は dask 未導入で `ImportError` になる。
- 存在しないパスは `FileNotFoundError`(h5netcdf エンジンでは h5py の `Unable to synchronously open file` 系のメッセージ)。

### `xr.load_dataset(...) / ds.load() / xr.open_dataarray(...)`

**用途**: `load_dataset` はファイルを全て読み込んでからファイルを閉じる(メモリに載せる)、`open_dataarray` は単一変数の netCDF を DataArray として開く。

**シグネチャ**: xr.load_dataset(filename_or_obj, **kwargs) / xr.open_dataarray(filename_or_obj, engine=None, chunks=None, cache=None, ...)

**使用例**:
```python
ds = xr.Dataset(
    {"temp": (("time", "x"), np.arange(6.0).reshape(3, 2)), "pres": ("x", [1, 2])},
    coords={"time": pd.date_range("2024-01-01", periods=3), "x": [10, 20]},
)
ds.to_netcdf("multi.nc")
ds["temp"].to_netcdf("single.nc")

loaded = xr.load_dataset("multi.nc")        # 読み込み済み。ファイルは閉じられている
print(loaded.temp.values[0])

print(xr.open_dataarray("single.nc"))       # 単一変数のファイル

try:
    xr.open_dataarray("multi.nc")           # 複数変数のファイルはエラー
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
[0. 1.]
<xarray.DataArray 'temp' (time: 3, x: 2)> Size: 48B
[6 values with dtype=float64]
Coordinates:
  * time     (time) datetime64[ns] 24B 2024-01-01 2024-01-02 2024-01-03
  * x        (x) int64 16B 10 20
ValueError: Given file dataset contains more than one data variable. Please read with xarray.open_dataset and then select the variable you want.
```

**注意点・落とし穴**:
- `open_dataset` の遅延読み込みに対し、`load_dataset` は即時に全データをメモリに読み込む(小さいファイルや、何度もアクセスするデータ向け)。既に開いた Dataset は `ds.load()` で読み込み、その後 `close()` してもデータは使える。
- `open_dataarray` は変数が 1 つだけのファイル用。複数変数があれば `ValueError`(メッセージは `open_dataset` で開いて変数を選ぶよう案内する)。`DataArray.to_netcdf` は変数を 1 つだけ持つファイルを作る。

### エンコーディング: `ds.to_netcdf(path, encoding=...)`

**用途**: 変数ごとに保存形式(dtype・欠損値マーカー・スケーリング・圧縮・時刻単位)を指定する。ファイルサイズを節約したいとき、CF 規約に合わせたいときに使う。

**シグネチャ**: `ds.to_netcdf(path, encoding={"var": {...}})`

**使用例**:
```python
ds = xr.Dataset(
    {"temp": (("time", "x"), np.arange(6.0).reshape(3, 2))},
    coords={"time": pd.date_range("2024-01-01", periods=3), "x": [10, 20]},
)

# 1) 保存 dtype と欠損値マーカー
ds.to_netcdf("b.nc", encoding={"temp": {"dtype": "float32", "_FillValue": -9999.0}})
r = xr.open_dataset("b.nc")
print(r.temp.dtype, r.temp.encoding["_FillValue"])
r.close()

# 2) 時刻の単位
ds.to_netcdf("c.nc", encoding={"time": {"units": "hours since 2024-01-01"}})
r = xr.open_dataset("c.nc")
print(r.time.encoding["units"])
r.close()

# 3) 整数へのスケーリング(scale_factor)
ds.to_netcdf("d.nc", encoding={"temp": {"dtype": "int16", "scale_factor": 0.1, "_FillValue": -32768}})
r = xr.open_dataset("d.nc")
print(r.temp.values[0], r.temp.encoding["scale_factor"])
r.close()

# 4) 圧縮(h5netcdf では zlib/complevel も、compression="gzip" も指定できる)
ds.to_netcdf("e.nc", engine="h5netcdf", encoding={"temp": {"compression": "gzip", "compression_opts": 4}})
r = xr.open_dataset("e.nc")
print(r.temp.encoding["zlib"], r.temp.encoding["complevel"])
r.close()
```
実行結果:
```
float32 -9999.0
hours since 2024-01-01
[0. 1.] 0.1
True 4
```

**注意点・落とし穴**:
- 読み込み後の `da.encoding` に、ファイルに保存されていた形式(dtype・`_FillValue`・`units` など)が入る。この `encoding` は演算(`r.temp + 1` など)や集約の結果には引き継がれない(空の dict になることを確認)。同じ形式で書き戻したいときは明示的に `encoding=` に渡す。
- float の欠損値(NaN)は既定で `_FillValue=NaN` として保存される。整数で保存する(`dtype="int16"` など)場合は `_FillValue` を必ず指定する。指定しないと `SerializationWarning: saving variable temp with floating point data as an integer dtype without any _FillValue to use for NaNs` が出て、NaN は 0 に化け(`[0.0, 1.23, nan]` が `[0, 1, 0]` になった)、小数部も切り捨てられる。
- `scale_factor` を使った整数保存は不可逆圧縮(丸めが入る)。`scale_factor=0.1` だと `1.26` は読み戻すと `1.3`、`1.23` は `1.2` になった。許容できる精度で使う。
- 圧縮の指定は `netCDF4` 系(`zlib=True, complevel=4`)と `h5netcdf`(`compression="gzip", compression_opts=4`)で書式が違うが、h5netcdf エンジンでは両方受け付けられる。上の例のように読み戻すと `zlib`/`complevel` の形で `encoding` に入っていた。
