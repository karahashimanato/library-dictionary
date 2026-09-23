# pandas 逆引き辞書

pandas 3.0.5 で検証済み(pandas 3.0以降の仕様変更に注意)

本ドキュメントに掲載しているシグネチャ・実行結果は、すべて `/home/manaty/library-practicing/.venv/bin/python`(pandas 3.0.5)で実際にコードを実行して得たものです。特に pandas 3.0 では以下のような大きな仕様変更があるため、古いバージョンの知識で判断しないよう注意してください。

- **文字列列のデフォルト dtype が `object` から `str`(`StringDtype`)に変更**。`pd.DataFrame({"b": ["x","y"]}).dtypes` は `b    str` と表示される。
- **Copy-on-Write (CoW) が常時有効になり、無効化できなくなった**(`pd.options.mode.copy_on_write` は非推奨警告付きで実質無視される)。スライスや列選択の結果を書き換えても元の DataFrame には影響しない。

## 目次

1. [入出力](#入出力)
2. [DataFrame/Series基礎](#dataframeseries基礎)
3. [インデックス操作・選択](#インデックス操作選択)
4. [欠損値処理](#欠損値処理)
5. [グループ化・集計](#グループ化集計)
6. [結合・連結](#結合連結)
7. [時系列処理](#時系列処理)
8. [文字列操作](#文字列操作)
9. [並べ替え・ランキング](#並べ替えランキング)
10. [ピボット・reshape](#ピボットreshape)
11. [型変換](#型変換)
12. [その他便利メソッド](#その他便利メソッド)

---

## 入出力

### `pd.read_csv(...)`

**用途**: CSVファイル(またはバッファ)を DataFrame として読み込む。

**シグネチャ**: `pd.read_csv(filepath_or_buffer, *, sep=<no_default>, header='infer', names=<no_default>, index_col=None, usecols=None, dtype=None, engine=None, na_values=None, parse_dates=None, ...)`(主要な引数のみ抜粋。実際は50個以上のキーワード引数を持つ)

**使用例**:
```python
import pandas as pd

df_io = pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "age": [30, 25, 35], "city": ["Tokyo", "Osaka", "Nagoya"]})
df_io.to_csv("/tmp/sample_io.csv", index=False)

df_read = pd.read_csv("/tmp/sample_io.csv")
print(df_read)
print(df_read.dtypes)
```
実行結果:
```
    name  age    city
0  Alice   30   Tokyo
1    Bob   25   Osaka
2  Carol   35  Nagoya
name      str
age     int64
city      str
dtype: object
```

**注意点・落とし穴**:
- pandas 3.0 では文字列列は `object` ではなく `str`(StringDtype)として読み込まれる。`dtype=object` を明示指定すれば従来通りの挙動にできる。
- `dtype_backend="pyarrow"` を指定すると Arrow バックエンドの型で読み込める。

### `df.to_csv(...)`

**用途**: DataFrame をCSV形式の文字列 or ファイルに出力する。

**シグネチャ**: `df.to_csv(path_or_buf=None, *, sep=',', na_rep='', header=True, index=True, encoding=None, ...)`

**使用例**:
```python
df_io.to_csv("/tmp/sample_io.csv", index=False)
print(open("/tmp/sample_io.csv").read())
```
実行結果:
```
name,age,city
Alice,30,Tokyo
Bob,25,Osaka
Carol,35,Nagoya
```

**注意点・落とし穴**:
- `index=False` を忘れるとインデックス列が1列目に余分に出力される。読み込み側と対で意識すること。

### `pd.read_json(...)` / `df.to_json(...)`

**用途**: JSON文字列・ファイルとDataFrameを相互変換する。

**シグネチャ**: `pd.read_json(path_or_buf, *, orient=None, typ='frame', convert_dates=True, lines=False, ...)` / `df.to_json(path_or_buf=None, *, orient=None, date_format=None, force_ascii=True, ...)`

**使用例**:
```python
json_str = df_io.to_json(orient="records", force_ascii=False)
print(json_str)

df_json = pd.read_json(pd.io.common.StringIO(json_str))
print(df_json)
```
実行結果:
```
[{"name":"Alice","age":30,"city":"Tokyo"},{"name":"Bob","age":25,"city":"Osaka"},{"name":"Carol","age":35,"city":"Nagoya"}]
    name  age    city
0  Alice   30   Tokyo
1    Bob   25   Osaka
2  Carol   35  Nagoya
```

**注意点・落とし穴**:
- `read_json` にファイルパス以外(文字列そのもの)を渡す場合は `pd.io.common.StringIO(json_str)` のようにバッファ化する必要がある(直接JSON文字列を渡すとファイルパスとして解釈されエラーになることがある)。
- `force_ascii=False` を指定しないと日本語などの非ASCII文字が `\uXXXX` にエスケープされる。

### `df.to_parquet(...)` / `pd.read_parquet(...)`

**用途**: 高速・省容量なParquet形式でDataFrameを保存・読込する(列指向、型情報を保持)。

**シグネチャ**: `df.to_parquet(path=None, *, engine='auto', compression='snappy', index=None, ...)` / `pd.read_parquet(path, engine='auto', columns=None, ...)`

**使用例**:
```python
df_io.to_parquet("/tmp/sample_io.parquet")
df_pq = pd.read_parquet("/tmp/sample_io.parquet")
print(df_pq)
print(df_pq.dtypes)
```
実行結果:
```
    name  age    city
0  Alice   30   Tokyo
1    Bob   25   Osaka
2  Carol   35  Nagoya
name      str
age     int64
city      str
dtype: object
```

**注意点・落とし穴**:
- `pyarrow` (または `fastparquet`) がインストールされている必要がある。本検証環境では `pyarrow` を使用。
- CSVと違い型情報(dtype)がそのまま保持されるため、読み込み後に型を再指定する手間がない。

### `df.to_pickle(...)` / `pd.read_pickle(...)`

**用途**: DataFrameをPythonのpickle形式でそのままシリアライズ・復元する。dtypeやインデックスも完全に保持される。

**シグネチャ**: `df.to_pickle(path, *, compression='infer', protocol=5, ...)` / `pd.read_pickle(filepath_or_buffer, compression='infer', ...)`

**使用例**:
```python
df_io.to_pickle("/tmp/sample_io.pkl")
df_pkl = pd.read_pickle("/tmp/sample_io.pkl")
print(df_pkl)
```
実行結果:
```
    name  age    city
0  Alice   30   Tokyo
1    Bob   25   Osaka
2  Carol   35  Nagoya
```

**注意点・落とし穴**:
- pickleはPython/pandasのバージョン間で互換性が保証されないため、長期保存や他システムとの連携には不向き(社内の一時キャッシュ用途などに限定するのが無難)。

---

## DataFrame/Series基礎

### `pd.DataFrame(...)`

**用途**: 表形式データ(行×列)を保持する pandas の中核データ構造を作成する。

**シグネチャ**: `pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)`

**使用例**:
```python
df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}, index=["x", "y", "z"])
print(df)
```
実行結果:
```
   a    b
x  1  4.0
y  2  5.0
z  3  6.0
```

### `pd.Series(...)`

**用途**: 1次元のラベル付き配列(DataFrameの1列に相当)を作成する。

**シグネチャ**: `pd.Series(data=None, index=None, dtype=None, name=None, copy=None)`

**使用例**:
```python
s = pd.Series([10, 20, 30], index=["x", "y", "z"], name="val")
print(s)
```
実行結果:
```
x    10
y    20
z    30
Name: val, dtype: int64
```

### `df.head(n=5)` / `df.tail(n=5)`

**用途**: DataFrame/Seriesの先頭・末尾n行を確認する。

**シグネチャ**: `df.head(n=5)` / `df.tail(n=5)`

**使用例**:
```python
df_ht = pd.DataFrame({"v": range(10)})
print(df_ht.head(3))
print(df_ht.tail(3))
```
実行結果:
```
   v
0  0
1  1
2  2
   v
7  7
8  8
9  9
```

### `df.info(...)`

**用途**: 列数・各列の非欠損数・dtype・メモリ使用量など、DataFrameの構造をまとめて確認する。

**シグネチャ**: `df.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None)`

**使用例**:
```python
df.info()
```
実行結果:
```
<class 'pandas.DataFrame'>
Index: 3 entries, x to z
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   a       3 non-null      int64  
 1   b       3 non-null      float64
dtypes: float64(1), int64(1)
memory usage: 75.0 bytes
```

**注意点・落とし穴**:
- 戻り値は `None`。標準出力に直接印字されるだけなので、結果を変数に受け取ることはできない(必要なら `buf=` にStringIOを渡す)。

### `df.describe(...)`

**用途**: 数値列の統計量(件数・平均・標準偏差・分位点など)を一括算出する。

**シグネチャ**: `df.describe(percentiles=None, include=None, exclude=None)`

**使用例**:
```python
print(df.describe())
```
実行結果:
```
         a    b
count  3.0  3.0
mean   2.0  5.0
std    1.0  1.0
min    1.0  4.0
25%    1.5  4.5
50%    2.0  5.0
75%    2.5  5.5
max    3.0  6.0
```

**注意点・落とし穴**:
- デフォルトでは数値列のみが対象。文字列/カテゴリ列も含めたい場合は `include="all"` を指定する。

### `df.copy(deep=True)`

**用途**: DataFrameの複製を作る。pandas 3.0のCopy-on-Write(CoW)の下での挙動理解が重要。

**シグネチャ**: `df.copy(deep=True)`

**使用例**:
```python
import numpy as np

df1 = pd.DataFrame({"a": [1, 2, 3]})
df_deep = df1.copy(deep=True)
df_shallow = df1.copy(deep=False)
print("deep shares memory:", np.shares_memory(df1["a"].to_numpy(), df_deep["a"].to_numpy()))
print("shallow shares memory:", np.shares_memory(df1["a"].to_numpy(), df_shallow["a"].to_numpy()))

df_shallow.iloc[0, 0] = 999
print("df1 after writing shallow copy:\n", df1)
print("df_shallow:\n", df_shallow)
```
実行結果:
```
deep shares memory: False
shallow shares memory: True
df1 after writing shallow copy:
    a
0  1
1  2
2  3
df_shallow:
      a
0  999
1    2
2    3
```

**注意点・落とし穴**:
- pandas 3.0 ではCoWが常時有効(無効化不可)。`copy(deep=False)` はメモリ上はコピー元と共有されるが、どちらかを書き換えた瞬間に内部的にコピーが発生するため、**元のDataFrameが意図せず変更される心配はない**(pandas 1.x/2.x以前の `SettingWithCopyWarning` の悩みが解消されている)。
- `df[["a"]]` のような列選択結果を書き換えても元のDataFrameには影響しない(CoWにより安全)。

### `df.rename(...)`

**用途**: 列名・行名(インデックス)を変更する。

**シグネチャ**: `df.rename(mapper=None, *, index=None, columns=None, axis=None, inplace=False, errors='ignore')`

**使用例**:
```python
df_r = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
print(df_r.rename(columns={"a": "alpha", "b": "beta"}))
```
実行結果:
```
   alpha  beta
0      1     3
1      2     4
```

**注意点・落とし穴**:
- デフォルトは非破壊的(新しいDataFrameを返す)。元を書き換えるには `inplace=True`。
- pandas 3.0で `copy` 引数は非推奨(CoWの下では常に効率的にコピーされるため意味を持たない)。

---

## インデックス操作・選択

### `df.loc[...]`

**用途**: ラベル(行名・列名)ベースでの選択・スライシング。

**シグネチャ**: `df.loc[row_indexer, col_indexer]`(プロパティのため関数シグネチャは持たない)

**使用例**:
```python
df_sel = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["r1", "r2", "r3"])
print(df_sel.loc["r2"])
print(df_sel.loc[["r1", "r3"], "b"])
print(df_sel.loc[df_sel["a"] > 1])
```
実行結果:
```
a    2
b    5
Name: r2, dtype: int64
r1    4
r3    6
Name: b, dtype: int64
    a  b
r2  2  5
r3  3  6
```

**注意点・落とし穴**:
- `loc` はラベルベースかつ**両端を含む**スライス(`df.loc["r1":"r2"]` は "r2" も含まれる)。`iloc` の位置ベース(片側排他)と混同しやすい。

### `df.iloc[...]`

**用途**: 整数の位置(0始まり)ベースでの選択・スライシング。

**シグネチャ**: `df.iloc[row_pos, col_pos]`

**使用例**:
```python
print(df_sel.iloc[0])
print(df_sel.iloc[0:2, 1])
print(df_sel.iloc[-1])
```
実行結果:
```
a    1
b    4
Name: r1, dtype: int64
r1    4
r2    5
Name: b, dtype: int64
a    3
b    6
Name: r3, dtype: int64
```

**注意点・落とし穴**:
- スライス `0:2` は Pythonの通常のスライスと同様に**終端を含まない**(位置0,1のみ)。

### `df.at[...]` / `df.iat[...]`

**用途**: 単一のスカラー値を高速に取得・設定する(`loc`/`iloc` より軽量)。

**シグネチャ**: `df.at[row_label, col_label]` / `df.iat[row_pos, col_pos]`

**使用例**:
```python
print(df_sel.at["r2", "a"])
print(df_sel.iat[0, 1])
```
実行結果:
```
2
4
```

**注意点・落とし穴**:
- 単一値の取得・書き換えに限定すれば `loc`/`iloc` より高速。複数セルの選択には使えない。

### `df.set_index(...)` / `df.reset_index(...)`

**用途**: 特定の列をインデックスに設定する/インデックスを列に戻して連番インデックスに振り直す。

**シグネチャ**: `df.set_index(keys, *, drop=True, append=False, inplace=False)` / `df.reset_index(level=None, *, drop=False, inplace=False)`

**使用例**:
```python
df_idx = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
df_set = df_idx.set_index("id")
print(df_set)
print(df_set.reset_index())
```
実行結果:
```
   val
id    
1    a
2    b
3    c
   id val
0   1   a
1   2   b
2   3   c
```

### `df.query(...)`

**用途**: 文字列式でDataFrameの行をフィルタする(SQLのWHERE句に近い書き心地)。

**シグネチャ**: `df.query(expr, *, parser='pandas', engine=None, local_dict=None, inplace=False)`

**使用例**:
```python
df_q = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
print(df_q.query("a > 2 and b < 40"))

thresh = 2
print(df_q.query("a > @thresh"))
```
実行結果:
```
   a   b
2  3  30
   a   b
2  3  30
3  4  40
```

**注意点・落とし穴**:
- ローカル変数を式内で使う場合は `@変数名` の記法が必要(そのまま書くと列名として解釈されエラーになる)。

### `df.isin(values)`

**用途**: 各要素が指定した値集合に含まれるかを真偽値で判定する(フィルタと組み合わせて使うことが多い)。

**シグネチャ**: `df.isin(values)`

**使用例**:
```python
df_isin = pd.DataFrame({"fruit": ["apple", "banana", "cherry"]})
print(df_isin[df_isin["fruit"].isin(["apple", "cherry"])])
```
実行結果:
```
    fruit
0   apple
2  cherry
```

---

## 欠損値処理

### `df.isna()` / `df.notna()`

**用途**: 各要素が欠損値(NaN/None/NaT)かどうかを判定する。

**シグネチャ**: `df.isna()` / `df.notna()`

**使用例**:
```python
import numpy as np
df_na = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]})
print(df_na.isna())
print(df_na.isna().sum())
```
実行結果:
```
       a      b
0  False   True
1   True  False
2  False  False
a    1
b    1
dtype: int64
```

### `df.dropna(...)`

**用途**: 欠損値を含む行(または列)を除外する。

**シグネチャ**: `df.dropna(*, axis=0, how=<no_default>, thresh=<no_default>, subset=None, inplace=False)`

**使用例**:
```python
print(df_na.dropna())
print(df_na.dropna(how="all"))
print(df_na.dropna(subset=["a"]))
```
実行結果:
```
     a    b
2  3.0  3.0
     a    b
0  1.0  NaN
1  NaN  2.0
2  3.0  3.0
     a    b
0  1.0  NaN
2  3.0  3.0
```

**注意点・落とし穴**:
- デフォルト(`how="any"`相当)は「1つでも欠損があれば行を落とす」ため、想定より多くの行が消えやすい。列単位で見たい場合は `subset=` を使う。
- `how="all"` は全列が欠損の行のみ除外(この例では全行が残った)。

### `df.fillna(value)`

**用途**: 欠損値を指定した値・辞書・前後の値などで埋める。

**シグネチャ**: `df.fillna(value, *, axis=None, inplace=False, limit=None)`

**使用例**:
```python
print(df_na.fillna(0))
print(df_na.fillna({"a": -1, "b": -2}))
print(df_na.ffill())
```
実行結果:
```
     a    b
0  1.0  0.0
1  0.0  2.0
2  3.0  3.0
     a    b
0  1.0 -2.0
1 -1.0  2.0
2  3.0  3.0
     a    b
0  1.0  NaN
1  1.0  2.0
2  3.0  3.0
```

**注意点・落とし穴**:
- 列ごとに異なる埋め方をしたい場合は辞書を渡す(`{"a": -1, "b": -2}`)。
- `method="ffill"` 引数は pandas 2.x で非推奨化され、pandas 3.0では `df.ffill()` / `df.bfill()` という専用メソッドを使う(`fillna(method=...)` は既に廃止済み)。

### `df.interpolate(method='linear')`

**用途**: 欠損値を前後の値から補間して埋める(時系列データの穴埋めなどに便利)。

**シグネチャ**: `df.interpolate(method='linear', *, axis=0, limit=None, inplace=False, limit_direction=None)`

**使用例**:
```python
df_interp = pd.DataFrame({"v": [1.0, np.nan, np.nan, 4.0]})
print(df_interp.interpolate())
```
実行結果:
```
     v
0  1.0
1  2.0
2  3.0
3  4.0
```

**注意点・落とし穴**:
- デフォルトの `method="linear"` はインデックスの間隔を無視して等間隔補間する。日時インデックスの間隔を考慮したい場合は `method="time"` を使う。

---

## グループ化・集計

### `df.groupby(by)`

**用途**: 指定したキーでグループ化し、グループごとに集計・変換処理を行う。

**シグネチャ**: `df.groupby(by=None, level=None, *, as_index=True, sort=True, group_keys=True, observed=True, dropna=True)`

**使用例**:
```python
df_g = pd.DataFrame({"team": ["A", "A", "B", "B"], "score": [10, 20, 30, 40]})
print(df_g.groupby("team")["score"].sum())
```
実行結果:
```
team
A    30
B    70
Name: score, dtype: int64
```

**注意点・落とし穴**:
- `groupby()` 自体は遅延評価の `DataFrameGroupBy` オブジェクトを返すだけで、`.sum()` 等の集計メソッドを呼ぶまで計算されない。

### `df.groupby(...).agg(...)`

**用途**: グループごとに複数の集計関数を同時に適用する。

**シグネチャ**: `df.agg(func=None, axis=0, *args, **kwargs)`

**使用例**:
```python
print(df_g.groupby("team")["score"].agg(["sum", "mean", "max"]))
print(df_g.groupby("team").agg(total=("score", "sum"), avg=("score", "mean")))
```
実行結果:
```
      sum  mean  max
team                
A      30  15.0   20
B      70  35.0   40
      total   avg
team             
A        30  15.0
B        70  35.0
```

**注意点・落とし穴**:
- `agg(名前=("列", "関数"))` の形式(named aggregation)を使うと、結果列に分かりやすい名前を付けられる。

### `df.groupby(...).transform(func)`

**用途**: グループごとの集計結果を、元のDataFrameと同じ行数・順序で返す(グループ平均との差分計算などに便利)。

**シグネチャ**: `df.transform(func, axis=0, *args, **kwargs)`

**使用例**:
```python
df_g2 = df_g.copy()
df_g2["team_mean"] = df_g2.groupby("team")["score"].transform("mean")
print(df_g2)
```
実行結果:
```
  team  score  team_mean
0    A     10       15.0
1    A     20       15.0
2    B     30       35.0
3    B     40       35.0
```

### `pd.pivot_table(...)`

**用途**: 集計関数を指定しながら、行・列にキーを配置したクロス集計表を作る。

**シグネチャ**: `pd.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True)`

**使用例**:
```python
df_pt = pd.DataFrame({
    "date": ["2024-01", "2024-01", "2024-02", "2024-02"],
    "team": ["A", "B", "A", "B"],
    "score": [10, 20, 15, 25],
})
print(pd.pivot_table(df_pt, values="score", index="date", columns="team", aggfunc="sum"))
```
実行結果:
```
team      A   B
date           
2024-01  10  20
2024-02  15  25
```

**注意点・落とし穴**:
- `df.pivot()`(集計なし・単純な reshape)と `pd.pivot_table()`(集計あり)は別物。重複するインデックス×列の組み合わせがある場合は `pivot()` はエラーになるが `pivot_table()` は `aggfunc` で集約する。

### `s.value_counts(...)`

**用途**: 値ごとの出現回数(または割合)を集計する。

**シグネチャ**: `s.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)`

**使用例**:
```python
s_vc = pd.Series(["a", "b", "a", "c", "a", "b"])
print(s_vc.value_counts())
print(s_vc.value_counts(normalize=True))
```
実行結果:
```
a    3
b    2
c    1
Name: count, dtype: int64
a    0.500000
b    0.333333
c    0.166667
Name: proportion, dtype: float64
```

**注意点・落とし穴**:
- 結果のSeries名は `count`(`normalize=True`時は `proportion`)に固定される。

---

## 結合・連結

### `pd.merge(left, right, how, on)`

**用途**: SQLのJOINに相当する、キー列に基づくDataFrame同士の結合。

**シグネチャ**: `pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, suffixes=('_x', '_y'), indicator=False, validate=None)`

**使用例**:
```python
left = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
right = pd.DataFrame({"id": [2, 3, 4], "val": [200, 300, 400]})
print(pd.merge(left, right, on="id", how="inner"))
print(pd.merge(left, right, on="id", how="outer"))
```
実行結果:
```
   id name  val
0   2    b  200
1   3    c  300
   id name    val
0   1    a    NaN
1   2    b  200.0
2   3    c  300.0
3   4  NaN  400.0
```

**注意点・落とし穴**:
- `how="outer"` で結合キーが一致しない行が発生すると、元は整数型だった列が欠損値(NaN)混入により `float64` に昇格することがある(上の例の `val` 列)。
- 結合キー以外に同名の列がある場合、自動的に `_x`/`_y` サフィックスが付く(`suffixes`で変更可)。

### `pd.concat(objs, axis=0)`

**用途**: 複数のDataFrame/Seriesを行方向または列方向に単純連結する。

**シグネチャ**: `pd.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, verify_integrity=False)`

**使用例**:
```python
df_c1 = pd.DataFrame({"a": [1, 2]})
df_c2 = pd.DataFrame({"a": [3, 4]})
print(pd.concat([df_c1, df_c2], ignore_index=True))

df_c3 = pd.DataFrame({"b": [5, 6]})
print(pd.concat([df_c1, df_c3], axis=1))
```
実行結果:
```
   a
0  1
1  2
2  3
3  4
   a  b
0  1  5
1  2  6
```

**注意点・落とし穴**:
- `axis=0`(デフォルト)で `ignore_index=True` を付けないと、元のインデックスがそのまま連結され重複することがある。

### `df.join(other, how='left')`

**用途**: インデックスをキーとしてDataFrame同士を結合する(`merge`のインデックス版に近い)。

**シグネチャ**: `df.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)`

**使用例**:
```python
left2 = pd.DataFrame({"name": ["a", "b"]}, index=[1, 2])
right2 = pd.DataFrame({"val": [100, 200]}, index=[2, 3])
print(left2.join(right2, how="left"))
print(left2.join(right2, how="inner"))
```
実行結果:
```
  name    val
1    a    NaN
2    b  100.0
  name  val
2    b  100
```

**注意点・落とし穴**:
- デフォルトの `how` は `merge` が `"inner"` なのに対し `join` は `"left"` なので、想定より欠損値(NaN)混じりの結果になりがち。

---

## 時系列処理

### `pd.to_datetime(arg, format=None)`

**用途**: 文字列や数値を `datetime64` 型に変換する。

**シグネチャ**: `pd.to_datetime(arg, errors='raise', dayfirst=False, utc=False, format=None, unit=None)`

**使用例**:
```python
print(pd.to_datetime(["2024-01-01", "2024-02-15"]))
print(pd.to_datetime("2024/03/10", format="%Y/%m/%d"))
```
実行結果:
```
DatetimeIndex(['2024-01-01', '2024-02-15'], dtype='datetime64[us]', freq=None)
2024-03-10 00:00:00
```

**注意点・落とし穴**:
- pandas 3.0ではデフォルトの時間解像度が `datetime64[us]`(マイクロ秒)。以前のpandas(1.x/2.x系)は `datetime64[ns]` 固定だったため、他ライブラリとの型比較で差異が出ることがある。
- 大量データを変換する場合、`format=` を明示した方が高速かつ曖昧さを排除できる。

### `pd.date_range(start, end, periods, freq)`

**用途**: 一定間隔の日時インデックスを生成する。

**シグネチャ**: `pd.date_range(start=None, end=None, periods=None, freq=None, tz=None, inclusive='both')`

**使用例**:
```python
print(pd.date_range("2024-01-01", periods=5, freq="D"))
print(pd.date_range("2024-01-01", "2024-01-10", freq="3D"))
```
実行結果:
```
DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'], dtype='datetime64[us]', freq='D')
DatetimeIndex(['2024-01-01', '2024-01-04', '2024-01-07', '2024-01-10'], dtype='datetime64[us]', freq='3D')
```

### `df.resample(rule)`

**用途**: 時系列データを別の時間間隔(例: 日次→3日ごと)に集約・変換する。

**シグネチャ**: `df.resample(rule, closed=None, label=None, on=None, origin='start_day')`

**使用例**:
```python
idx = pd.date_range("2024-01-01", periods=6, freq="D")
s_ts = pd.Series([1, 2, 3, 4, 5, 6], index=idx)
print(s_ts.resample("3D").sum())
```
実行結果:
```
2024-01-01     6
2024-01-04    15
Freq: 3D, dtype: int64
```

**注意点・落とし穴**:
- `resample()` を使うには index が `DatetimeIndex`(または `PeriodIndex`)である必要がある。通常のカラムに日時がある場合は `on="日時列名"` を指定する。

### `s.shift(periods=1)` / `s.diff()`

**用途**: 値を前後にずらす(ラグ特徴量の作成)/前の値との差分を取る。

**シグネチャ**: `s.shift(periods=1, freq=None, axis=0, fill_value=<no_default>)`

**使用例**:
```python
s_sh = pd.Series([1, 2, 3, 4])
print(s_sh.shift(1))
print(s_sh.diff())
```
実行結果:
```
0    NaN
1    1.0
2    2.0
3    3.0
dtype: float64
0    NaN
1    1.0
2    1.0
3    1.0
dtype: float64
```

**注意点・落とし穴**:
- `shift()` によって末尾または先頭に `NaN` が生じるため、元が整数型でも結果は `float64` になる(`fillna`で埋めるか `Int64`拡張型を使えば回避可)。

### `s.rolling(window).agg関数`

**用途**: 移動窓(直近n件)ごとの集計(移動平均など)を計算する。

**シグネチャ**: `s.rolling(window, min_periods=None, center=False, win_type=None)`

**使用例**:
```python
s_roll = pd.Series([1, 2, 3, 4, 5])
print(s_roll.rolling(window=3).mean())
```
実行結果:
```
0    NaN
1    NaN
2    2.0
3    3.0
4    4.0
dtype: float64
```

**注意点・落とし穴**:
- デフォルトでは窓が満たない先頭部分は `NaN` になる(`min_periods=1` にすると窓が満たなくても計算する)。

### `s.dt` アクセサ

**用途**: `datetime64` 型Seriesから年・月・曜日名などの要素を取り出す。

**シグネチャ**: プロパティアクセサ(`s.dt.year`, `s.dt.day_name()` など)

**使用例**:
```python
s_dt = pd.Series(pd.to_datetime(["2024-01-15", "2024-06-20"]))
print(s_dt.dt.year)
print(s_dt.dt.day_name())
```
実行結果:
```
0    2024
1    2024
dtype: int32
0      Monday
1    Thursday
dtype: str
```

**注意点・落とし穴**:
- pandas 3.0では `day_name()` の戻り値dtypeが `str`(StringDtype)になる(旧来は `object`)。

---

## 文字列操作

### `s.str.contains(pattern)`

**用途**: 各要素が指定パターン(部分文字列/正規表現)を含むか判定する。

**シグネチャ**: `s.str.contains(pat, case=True, flags=0, na=None, regex=True)`

**使用例**:
```python
s_str = pd.Series(["Apple", "banana", "Cherry", None])
print(s_str.str.contains("a", case=False, na=False))
```
実行結果:
```
0     True
1     True
2    False
3    False
dtype: bool
```

**注意点・落とし穴**:
- 欠損値(`None`)が含まれる場合、`na=` を指定しないとエラーや `NaN` を含む結果になることがあるため、フィルタ条件として使う際は `na=False` を明示するのが安全。

### `s.str.replace(pat, repl)`

**用途**: 文字列の一部を置換する。

**シグネチャ**: `s.str.replace(pat, repl, n=-1, case=None, flags=0, regex=False)`

**使用例**:
```python
print(s_str.str.replace("a", "@", case=False, regex=False))
```
実行結果:
```
0     @pple
1    b@n@n@
2    Cherry
3       NaN
dtype: str
```

**注意点・落とし穴**:
- pandas 2.0以降、`regex` のデフォルトは `False`(旧来は正規表現扱いだった)。正規表現を使いたい場合は明示的に `regex=True` を指定する。

### `s.str.split(pat, expand=False)`

**用途**: 文字列を区切り文字で分割する。`expand=True` で複数列に展開できる。

**シグネチャ**: `s.str.split(pat=None, *, n=-1, expand=False, regex=None)`

**使用例**:
```python
s_split = pd.Series(["a,b,c", "d,e"])
print(s_split.str.split(","))
print(s_split.str.split(",", expand=True))
```
実行結果:
```
0    [a, b, c]
1       [d, e]
dtype: object
   0  1    2
0  a  b    c
1  d  e  NaN
```

**注意点・落とし穴**:
- `expand=True` で列数が要素ごとに異なる場合、足りない箇所は `NaN` で埋められる。

### `s.str.extract(pattern)`

**用途**: 正規表現の捕捉グループ(`()`)にマッチした部分を列として取り出す。

**シグネチャ**: `s.str.extract(pat, flags=0, expand=True)`

**使用例**:
```python
s_ext = pd.Series(["item_001", "item_045"])
print(s_ext.str.extract(r"item_(\d+)"))
```
実行結果:
```
     0
0  001
1  045
```

**注意点・落とし穴**:
- マッチしない行は `NaN` になる。列名は正規表現に名前付きグループ(`(?P<name>...)`)を使うと自動的にその名前になる。

### `s.str.strip()`

**用途**: 文字列の前後の空白(または指定文字)を除去する。

**シグネチャ**: `s.str.strip(to_strip=None)`

**使用例**:
```python
s_strip = pd.Series(["  hello  ", "world  "])
print(s_strip.str.strip())
```
実行結果:
```
0    hello
1    world
dtype: str
```

---

## 並べ替え・ランキング

### `df.sort_values(by)`

**用途**: 指定した列の値でDataFrameを並べ替える。

**シグネチャ**: `df.sort_values(by, *, axis=0, ascending=True, inplace=False, na_position='last')`

**使用例**:
```python
df_sv = pd.DataFrame({"a": [3, 1, 2], "b": ["x", "y", "z"]})
print(df_sv.sort_values("a"))
print(df_sv.sort_values("a", ascending=False))
```
実行結果:
```
   a  b
1  1  y
2  2  z
0  3  x
   a  b
0  3  x
2  2  z
1  1  y
```

### `df.sort_index()`

**用途**: インデックス(行ラベル)でDataFrameを並べ替える。

**シグネチャ**: `df.sort_index(*, axis=0, level=None, ascending=True, inplace=False)`

**使用例**:
```python
df_si = pd.DataFrame({"a": [1, 2, 3]}, index=[3, 1, 2])
print(df_si.sort_index())
```
実行結果:
```
   a
1  2
2  3
3  1
```

### `s.rank(method='average')`

**用途**: 各値の順位を計算する。同順位の扱い方を `method` で制御できる。

**シグネチャ**: `s.rank(axis=0, method='average', na_option='keep', ascending=True, pct=False)`

**使用例**:
```python
s_rank = pd.Series([10, 20, 20, 30])
print(s_rank.rank())
print(s_rank.rank(method="min"))
print(s_rank.rank(method="dense"))
```
実行結果:
```
0    1.0
1    2.5
2    2.5
3    4.0
dtype: float64
0    1.0
1    2.0
2    2.0
3    4.0
dtype: float64
0    1.0
1    2.0
2    2.0
3    3.0
dtype: float64
```

**注意点・落とし穴**:
- デフォルト `method="average"` は同順位を平均順位にする(上の例では2位・3位が同値のため2.5)。`"min"`は最小順位、`"dense"`は順位が飛ばない(1,2,2,3)。

### `df.nlargest(n, columns)` / `df.nsmallest(n, columns)`

**用途**: 指定列で上位/下位n件を効率的に取得する(`sort_values().head()`より高速)。

**シグネチャ**: `df.nlargest(n, columns, keep='first')` / `df.nsmallest(n, columns, keep='first')`

**使用例**:
```python
df_nl = pd.DataFrame({"a": [5, 1, 9, 3]})
print(df_nl.nlargest(2, "a"))
print(df_nl.nsmallest(2, "a"))
```
実行結果:
```
   a
2  9
0  5
   a
1  1
3  3
```

---

## ピボット・reshape

### `df.pivot(index, columns, values)`

**用途**: 縦持ち(long)データを横持ち(wide)データに変換する(集計はしない単純な変形)。

**シグネチャ**: `df.pivot(*, columns, index=<no_default>, values=<no_default>)`

**使用例**:
```python
df_piv = pd.DataFrame({"date": ["d1", "d1", "d2", "d2"], "var": ["x", "y", "x", "y"], "val": [1, 2, 3, 4]})
print(df_piv.pivot(index="date", columns="var", values="val"))
```
実行結果:
```
var   x  y
date      
d1    1  2
d2    3  4
```

**注意点・落とし穴**:
- `index`×`columns`の組み合わせに重複があるとエラーになる(集計が必要な場合は `pivot_table` を使う)。

### `df.melt(id_vars, value_vars)`

**用途**: 横持ち(wide)データを縦持ち(long)データに変換する(`pivot`の逆操作)。

**シグネチャ**: `df.melt(id_vars=None, value_vars=None, var_name=None, value_name='value', ignore_index=True)`

**使用例**:
```python
df_melt = pd.DataFrame({"id": [1, 2], "x": [10, 20], "y": [30, 40]})
print(df_melt.melt(id_vars="id", value_vars=["x", "y"]))
```
実行結果:
```
   id variable  value
0   1        x     10
1   2        x     20
2   1        y     30
3   2        y     40
```

### `df.stack()` / `df.unstack()`

**用途**: 列インデックスを行インデックス(の最下層)に移す/その逆を行う。

**シグネチャ**: `df.stack(level=-1, dropna=<no_default>, sort=<no_default>, future_stack=True)` / `df.unstack(level=-1, fill_value=None, sort=True)`

**使用例**:
```python
df_su = pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=["r1", "r2"])
stacked = df_su.stack()
print(stacked)
print(stacked.unstack())
```
実行結果:
```
r1  x    1
    y    3
r2  x    2
    y    4
dtype: int64
    x  y
r1  1  3
r2  2  4
```

**注意点・落とし穴**:
- pandas 3.0では `future_stack=True` がデフォルトとなり、新しい(高速・欠損値の扱いがシンプルな)実装が標準になった。旧実装のオプション(`dropna`)は将来的に廃止予定。

### `df.explode(column)`

**用途**: リストなどのコレクションを含むセルを、要素ごとに複数行へ展開する。

**シグネチャ**: `df.explode(column, ignore_index=False)`

**使用例**:
```python
df_exp = pd.DataFrame({"id": [1, 2], "items": [["a", "b"], ["c"]]})
print(df_exp.explode("items"))
```
実行結果:
```
   id items
0   1     a
0   1     b
1   2     c
```

**注意点・落とし穴**:
- 展開後もインデックスは元の行のまま複製される(重複する)。連番に振り直したい場合は `ignore_index=True` を使う。

---

## 型変換

### `df.astype(dtype)`

**用途**: 列の型を明示的に変換する。

**シグネチャ**: `df.astype(dtype, copy=<no_default>, errors='raise')`

**使用例**:
```python
df_ast = pd.DataFrame({"a": ["1", "2", "3"]})
print(df_ast.dtypes)
df_ast2 = df_ast.astype({"a": "int64"})
print(df_ast2.dtypes)
```
実行結果:
```
a    str
dtype: object
a    int64
dtype: object
```

**注意点・落とし穴**:
- 変換できない値(例: `"abc"` を `int` に変換)があると `errors="raise"`(デフォルト)では例外になる。安全に変換したい場合は数値なら `pd.to_numeric(errors="coerce")` を使う方が柔軟。

### `pd.to_numeric(arg, errors='raise')`

**用途**: 文字列などを数値型に変換する。変換不能な値をエラーにするか欠損値にするか選べる。

**シグネチャ**: `pd.to_numeric(arg, errors='raise', downcast=None, dtype_backend=<no_default>)`

**使用例**:
```python
s_num = pd.Series(["1", "2", "abc"])
print(pd.to_numeric(s_num, errors="coerce"))
```
実行結果:
```
0    1.0
1    2.0
2    NaN
dtype: float64
```

**注意点・落とし穴**:
- `errors="coerce"` は変換できない値を `NaN` にする(データクレンジングで頻用)。`errors="raise"`(デフォルト)だと1件でも変換不能があれば例外で処理が止まる。

### `df.convert_dtypes()`

**用途**: 各列の内容を見て、pandasの拡張型(nullable Int64/string等)に自動変換する。

**シグネチャ**: `df.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True, dtype_backend='numpy_nullable')`

**使用例**:
```python
df_cv = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
print(df_cv.dtypes)
df_cv2 = df_cv.convert_dtypes()
print(df_cv2.dtypes)
print(df_cv2)
```
実行結果:
```
a    float64
b        str
dtype: object
a     Int64
b    string
dtype: object
      a  b
0     1  x
1     2  y
2  <NA>  z
```

**注意点・落とし穴**:
- 欠損値を含む整数列は通常 `float64` になってしまうが、`convert_dtypes()` を使うと nullable な `Int64`(先頭大文字、`<NA>`を保持できる拡張型)に変換できる。

---

## その他便利メソッド

### `df.apply(func, axis=0)`

**用途**: 行または列ごとに任意の関数を適用する。

**シグネチャ**: `df.apply(func, axis=0, raw=False, result_type=None, args=())`

**使用例**:
```python
df_apl = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(df_apl.apply(lambda row: row["a"] + row["b"], axis=1))
print(df_apl.apply(lambda col: col.max(), axis=0))
```
実行結果:
```
0    5
1    7
2    9
dtype: int64
a    3
b    6
dtype: int64
```

**注意点・落とし穴**:
- `axis=1`(行方向に適用)は内部的にPythonループに近い処理になるため、大規模データでは低速になりがち。ベクトル化できる処理は極力 `apply` を避けて直接演算する方が高速。

### `s.map(arg)`

**用途**: Seriesの各要素に関数または対応表(辞書/Series)を適用して変換する。

**シグネチャ**: `s.map(func, na_action=None)`

**使用例**:
```python
s_map = pd.Series(["a", "b", "c"])
print(s_map.map({"a": 1, "b": 2, "c": 3}))
print(s_map.map(str.upper))
```
実行結果:
```
0    1
1    2
2    3
dtype: int64
0    A
1    B
2    C
dtype: str
```

**注意点・落とし穴**:
- 辞書に存在しないキーに対応する値は `NaN` になる(`KeyError` にはならない)。

### `df.duplicated()` / `df.drop_duplicates()`

**用途**: 重複行を検出する/除去する。

**シグネチャ**: `df.duplicated(subset=None, keep='first')` / `df.drop_duplicates(subset=None, *, keep='first', inplace=False)`

**使用例**:
```python
df_dup = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
print(df_dup.duplicated())
print(df_dup.drop_duplicates())
```
実行結果:
```
0    False
1     True
2    False
dtype: bool
   a  b
0  1  3
2  2  4
```

**注意点・落とし穴**:
- `keep='first'`(デフォルト)は最初の出現を残し、以降の重複を `True`(除去対象)とする。`keep=False` にすると重複した行を全て(最初の出現も含め)対象にできる。

### `pd.cut(x, bins)` / `pd.qcut(x, q)`

**用途**: 連続値を区間(ビン)に分割してカテゴリ化する。`cut`は境界値を指定、`qcut`は分位点(件数が均等になるよう)で分割する。

**シグネチャ**: `pd.cut(x, bins, right=True, labels=None, include_lowest=False)` / `pd.qcut(x, q, labels=None, precision=3)`

**使用例**:
```python
s_cut = pd.Series([1, 7, 15, 22, 30])
print(pd.cut(s_cut, bins=[0, 10, 20, 30], labels=["low", "mid", "high"]))
print(pd.qcut(s_cut, q=2, labels=["lower_half", "upper_half"]))
```
実行結果:
```
0     low
1     low
2     mid
3    high
4    high
dtype: category
Categories (3, str): ['low' < 'mid' < 'high']
0    lower_half
1    lower_half
2    lower_half
3    upper_half
4    upper_half
dtype: category
Categories (2, str): ['lower_half' < 'upper_half']
```

**注意点・落とし穴**:
- `cut`はビンの境界を自分で決めるため各ビンの件数は不均等になりうる。`qcut`は件数が均等になるよう境界を自動計算する点が対照的。

### `df.sample(n=None, frac=None, random_state=None)`

**用途**: DataFrameから行(または列)をランダムに抽出する。

**シグネチャ**: `df.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)`

**使用例**:
```python
df_sample = pd.DataFrame({"a": range(10)})
print(df_sample.sample(n=3, random_state=42))
```
実行結果:
```
   a
8  8
1  1
5  5
```

**注意点・落とし穴**:
- `random_state` を固定しないと実行のたびに結果が変わる(再現性が必要な検証・デモでは必須)。

### `df.assign(**kwargs)`

**用途**: 既存のDataFrameを変更せず、新しい列を追加した新しいDataFrameを返す(メソッドチェーンに向く)。

**シグネチャ**: `df.assign(**kwargs)`

**使用例**:
```python
df_assign = pd.DataFrame({"a": [1, 2, 3]})
print(df_assign.assign(b=lambda d: d["a"] * 2, c=lambda d: d["a"] + d["b"]))
```
実行結果:
```
   a  b  c
0  1  2  3
1  2  4  6
2  3  6  9
```

**注意点・落とし穴**:
- `kwargs`内でラムダを使うと、同じ `assign()` 呼び出し内で直前に定義した列(この例の`b`)を後続の列(`c`)の計算に使える。
