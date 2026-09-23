# polars 逆引き辞書

polars 1.44.1 で検証済み

本ドキュメントに掲載しているシグネチャ・実行結果は、すべて `/home/manaty/library-practicing/.venv/bin/python`(polars 1.44.1)で実際にコードを実行して得たものです。pandasとの比較で押さえておくべき設計上の違いは次の通りです。

- **インデックス(行ラベル)が存在しない**。行は常に0始まりの位置でしかアクセスできず、`loc`/`iloc`に相当するものはない(位置アクセスは`df[i]`や`df.row(i)`、条件アクセスは`filter`を使う)。
- **式(Expression) API が中心**。`pl.col("a") + 1` のような式はそれ自体では評価されず、`select`/`with_columns`/`filter`などのコンテキストに渡して初めて実行される。pandasの「列オブジェクトを直接演算する」スタイルとは設計思想が異なる。
- **Eager API(`pl.DataFrame`)と Lazy API(`pl.LazyFrame`)の2本立て**。Lazy APIはクエリプランを構築し、`.collect()`するまで実際のデータは読まれない。クエリ最適化(述語プッシュダウンなど)が効くため、特に大規模データではLazy APIが推奨される。
- **Copy-on-Write のような曖昧さがない**。DataFrameの変更は基本的に非破壊的(新しいDataFrameを返す)であり、`inplace=True`という概念自体が存在しない。
- **デフォルトの文字列型は`str`(Utf8)一種類**。pandasのように`object`/`string`dtypeで揺れることはない。

## 目次

1. [入出力](#入出力)
2. [DataFrame/Series基礎](#dataframeseries基礎)
3. [式(Expressions)とselect/with_columns](#式expressionsとselectwith_columns)
4. [フィルタ・ソート・ユニーク](#フィルタソートユニーク)
5. [グループ化・集計とウィンドウ関数](#グループ化集計とウィンドウ関数)
6. [結合・連結](#結合連結)
7. [Lazy API](#lazy-api)
8. [文字列操作(.str)](#文字列操作str)
9. [日時処理(.dt)](#日時処理dt)
10. [欠損値処理](#欠損値処理)
11. [ピボット・reshape](#ピボットreshape)
12. [型変換・その他便利メソッド](#型変換その他便利メソッド)

---

## 入出力

### `pl.read_csv(...)` / `df.write_csv(...)`

**用途**: CSVファイルをDataFrameとして読み込む/DataFrameをCSVファイルに書き出す。

**シグネチャ**: `pl.read_csv(source, *, has_header=True, columns=None, new_columns=None, separator=',', comment_prefix=None, quote_char='"', skip_rows=0, schema=None, schema_overrides=None, null_values=None, ignore_errors=False, try_parse_dates=False, infer_schema_length=100, n_rows=None, encoding='utf8', ...)`(主要な引数のみ抜粋)/ `df.write_csv(file=None, *, include_bom=False, compression='uncompressed', include_header=True, separator=',', line_terminator='\n', quote_char='"', datetime_format=None, null_value=None, ...)`

**使用例**:
```python
import polars as pl

df_io = pl.DataFrame({"name": ["Alice", "Bob", "Carol"], "age": [30, 25, 35], "city": ["Tokyo", "Osaka", "Nagoya"]})
df_io.write_csv("/tmp/sample_io.csv")

df_read = pl.read_csv("/tmp/sample_io.csv")
print(df_read)
print(df_read.dtypes)
```
実行結果:
```
shape: (3, 3)
┌───────┬─────┬────────┐
│ name  ┆ age ┆ city   │
│ ---   ┆ --- ┆ ---    │
│ str   ┆ i64 ┆ str    │
╞═══════╪═════╪════════╡
│ Alice ┆ 30  ┆ Tokyo  │
│ Bob   ┆ 25  ┆ Osaka  │
│ Carol ┆ 35  ┆ Nagoya │
└───────┴─────┴────────┘
[String, Int64, String]
```

**注意点・落とし穴**:
- pandasと違い`index=False`のような引数は存在しない(インデックス自体がないため)。
- 日時列は自動推論されない(デフォルトでは文字列のまま)。`try_parse_dates=True`を指定するか、読み込み後に`.str.to_date()`/`.str.to_datetime()`で明示変換する。

### `pl.read_parquet(...)` / `df.write_parquet(...)`

**用途**: 列指向のParquet形式でDataFrameを高速・省容量に保存・読込する。

**シグネチャ**: `pl.read_parquet(source, *, columns=None, n_rows=None, row_index_name=None, parallel='auto', use_statistics=True, hive_partitioning=None, ...)` / `df.write_parquet(file, *, compression='zstd', compression_level=None, statistics=True, row_group_size=None, ...)`

**使用例**:
```python
df_io.write_parquet("/tmp/sample_io.parquet")
df_pq = pl.read_parquet("/tmp/sample_io.parquet")
print(df_pq)
```
実行結果:
```
shape: (3, 3)
┌───────┬─────┬────────┐
│ name  ┆ age ┆ city   │
│ ---   ┆ --- ┆ ---    │
│ str   ┆ i64 ┆ str    │
╞═══════╪═════╪════════╡
│ Alice ┆ 30  ┆ Tokyo  │
│ Bob   ┆ 25  ┆ Osaka  │
│ Carol ┆ 35  ┆ Nagoya │
└───────┴─────┴────────┘
```

**注意点・落とし穴**:
- pandasの`to_parquet`と違い、`pyarrow`が無くても動作する(polars独自のRust実装で読み書きする)。デフォルト圧縮は`zstd`。
- `write_parquet`にデフォルトの`index`引数は存在しない(インデックス列という概念自体がない)。

### `pl.read_json(...)` / `df.write_json(...)`

**用途**: JSON文字列・ファイルとDataFrameを相互変換する(レコード指向のJSON配列)。

**シグネチャ**: `pl.read_json(source, *, schema=None, schema_overrides=None, infer_schema_length=100)` / `df.write_json(file=None)`

**使用例**:
```python
import io

json_str = df_io.write_json()
print(json_str)

df_json = pl.read_json(io.StringIO(json_str))
print(df_json)
```
実行結果:
```
[{"name":"Alice","age":30,"city":"Tokyo"},{"name":"Bob","age":25,"city":"Osaka"},{"name":"Carol","age":35,"city":"Nagoya"}]
shape: (3, 3)
┌───────┬─────┬────────┐
│ name  ┆ age ┆ city   │
│ ---   ┆ --- ┆ ---    │
│ str   ┆ i64 ┆ str    │
╞═══════╪═════╪════════╡
│ Alice ┆ 30  ┆ Tokyo  │
│ Bob   ┆ 25  ┆ Osaka  │
│ Carol ┆ 35  ┆ Nagoya │
└───────┴─────┴────────┘
```

**注意点・落とし穴**:
- `write_json()`は常にレコード指向(`orient="records"`相当)の1形式のみで、pandasの`orient`引数のような選択肢はない。
- 日本語のような非ASCII文字はエスケープされず、そのままUTF-8で出力される(pandasの`force_ascii=False`相当の挙動がデフォルト)。

---

## DataFrame/Series基礎

### `pl.DataFrame(...)`

**用途**: 表形式データ(行×列)を保持するpolarsの中核データ構造を作成する。インデックスを持たない点がpandasと大きく異なる。

**シグネチャ**: `pl.DataFrame(data=None, schema=None, *, schema_overrides=None, strict=True, orient=None, infer_schema_length=100, nan_to_null=False)`

**使用例**:
```python
df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
print(df)
```
実行結果:
```
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ f64 │
╞═════╪═════╡
│ 1   ┆ 4.0 │
│ 2   ┆ 5.0 │
│ 3   ┆ 6.0 │
└─────┴─────┘
```

**注意点・落とし穴**:
- `pd.DataFrame`の`index=`に相当する引数はない。行番号でアクセスしたい場合は後述の`with_row_index()`で明示的に列を作る。

### `pl.Series(...)`

**用途**: 1次元配列(DataFrameの1列に相当)を作成する。こちらもインデックスを持たない。

**シグネチャ**: `pl.Series(name=None, values=None, dtype=None, *, strict=True, nan_to_null=False)`

**使用例**:
```python
s = pl.Series("val", [10, 20, 30])
print(s)
```
実行結果:
```
shape: (3,)
Series: 'val' [i64]
[
	10
	20
	30
]
```

### `df.head(n=5)` / `df.tail(n=5)`

**用途**: DataFrame/Seriesの先頭・末尾n行を確認する。

**シグネチャ**: `df.head(n=5)` / `df.tail(n=5)`

**使用例**:
```python
df_ht = pl.DataFrame({"v": range(10)})
print(df_ht.head(3))
print(df_ht.tail(3))
```
実行結果:
```
shape: (3, 1)
┌─────┐
│ v   │
│ --- │
│ i64 │
╞═════╡
│ 0   │
│ 1   │
│ 2   │
└─────┘
shape: (3, 1)
┌─────┐
│ v   │
│ --- │
│ i64 │
╞═════╡
│ 7   │
│ 8   │
│ 9   │
└─────┘
```

### `df.schema` / `df.dtypes` / `df.columns` / `df.shape`

**用途**: DataFrameの列名・型・行列数といったメタ情報を取得する。いずれもメソッドではなくプロパティ。

**シグネチャ**: プロパティアクセサ(引数なし)

**使用例**:
```python
print(df.schema)
print(df.dtypes)
print(df.columns)
print(df.shape)
```
実行結果:
```
Schema({'a': Int64, 'b': Float64})
[Int64, Float64]
['a', 'b']
(3, 2)
```

**注意点・落とし穴**:
- `df.schema`は列名→dtypeの`Schema`オブジェクト(辞書のように扱える)を返す。`df.dtypes`は型のリストのみで列名は含まれない。

### `df.describe(...)`

**用途**: 各列の統計量(件数・欠損数・平均・標準偏差・分位点など)を一括算出する。

**シグネチャ**: `df.describe(percentiles=(0.25, 0.5, 0.75), *, interpolation='nearest')`

**使用例**:
```python
print(df.describe())
```
実行結果:
```
shape: (9, 3)
┌────────────┬─────┬─────┐
│ statistic  ┆ a   ┆ b   │
│ ---        ┆ --- ┆ --- │
│ str        ┆ f64 ┆ f64 │
╞════════════╪═════╪═════╡
│ count      ┆ 3.0 ┆ 3.0 │
│ null_count ┆ 0.0 ┆ 0.0 │
│ mean       ┆ 2.0 ┆ 5.0 │
│ std        ┆ 1.0 ┆ 1.0 │
│ min        ┆ 1.0 ┆ 4.0 │
│ 25%        ┆ 2.0 ┆ 5.0 │
│ 50%        ┆ 2.0 ┆ 5.0 │
│ 75%        ┆ 3.0 ┆ 6.0 │
│ max        ┆ 3.0 ┆ 6.0 │
└────────────┴─────┴─────┘
```

**注意点・落とし穴**:
- 結果自体もDataFrameで返る(pandasと同じ)が、1列目`statistic`に統計量名が入る形式。`null_count`行がデフォルトで含まれ、pandasの`describe()`より欠損値の可視性が高い。

### `df.glimpse(...)`

**用途**: 各列の型と先頭数件の値を縦一覧で表示する。列数が多いDataFrameの概観を掴むのに便利(pandasに直接の対応物はない)。

**シグネチャ**: `df.glimpse(*, max_items_per_column=10, max_colname_length=50, return_type=None)`

**使用例**:
```python
print(df.glimpse(return_type="string"))
```
実行結果:
```
Rows: 3
Columns: 2
$ a <i64> 1, 2, 3
$ b <f64> 4.0, 5.0, 6.0

```

**注意点・落とし穴**:
- デフォルト(`return_type=None`)では戻り値を返さず直接標準出力に印字する。文字列として受け取りたい場合は`return_type="string"`を指定する。

---

## 式(Expressions)とselect/with_columns

### `df.select(...)`

**用途**: 指定した式(Expression)だけを評価して新しいDataFrameを作る(pandasの列選択+`assign`を合わせたようなもの)。

**シグネチャ**: `df.select(*exprs, **named_exprs)`

**使用例**:
```python
df2 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(df2.select(pl.col("a"), (pl.col("b") * 2).alias("b_double")))
```
実行結果:
```
shape: (3, 2)
┌─────┬──────────┐
│ a   ┆ b_double │
│ --- ┆ ---      │
│ i64 ┆ i64      │
╞═════╪══════════╡
│ 1   ┆ 8        │
│ 2   ┆ 10       │
│ 3   ┆ 12       │
└─────┴──────────┘
```

**注意点・落とし穴**:
- `select`は元の列のうち明示的に含めなかった列を落とす。既存の列を保持したまま追加したい場合は次の`with_columns`を使う。

### `df.with_columns(...)`

**用途**: 既存の全列を保持したまま、新しい列を追加・上書きする(`select`とセットで使う頻度が最も高いメソッド)。

**シグネチャ**: `df.with_columns(*exprs, **named_exprs)`

**使用例**:
```python
print(df2.with_columns((pl.col("a") + pl.col("b")).alias("sum")))
```
実行結果:
```
shape: (3, 3)
┌─────┬─────┬─────┐
│ a   ┆ b   ┆ sum │
│ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ 1   ┆ 4   ┆ 5   │
│ 2   ┆ 5   ┆ 7   │
│ 3   ┆ 6   ┆ 9   │
└─────┴─────┴─────┘
```

### `pl.col(...)` / `Expr.alias(...)`

**用途**: `pl.col("列名")`で式の起点となる列参照を作り、`.alias()`で結果列の名前を付ける。polarsの式APIの最も基本的な構成要素。

**シグネチャ**: `pl.col(...)`(引数の型に応じてオーバーロードされるため単一のシグネチャは持たない)/ `Expr.alias(self, name)`

**使用例**:
```python
print(df2.select(pl.col("a").alias("a_renamed"), pl.col("^b.*$")))
```
実行結果:
```
shape: (3, 2)
┌───────────┬─────┐
│ a_renamed ┆ b   │
│ ---       ┆ --- │
│ i64       ┆ i64 │
╞═══════════╪═════╡
│ 1         ┆ 4   │
│ 2         ┆ 5   │
│ 3         ┆ 6   │
└───────────┴─────┘
```

**注意点・落とし穴**:
- `pl.col()`は列名の完全一致だけでなく、正規表現(`"^b.*$"`のように`^`/`$`を含む文字列)や複数列名、dtype指定(`pl.col(pl.Int64)`)など多様な指定方法を受け付ける。

### `pl.when(...).then(...).otherwise(...)`

**用途**: 条件分岐(SQLのCASE WHENに相当)で値を作り分ける。pandasの`np.select`/`np.where`に相当。

**シグネチャ**: `pl.when(*predicates, **constraints) -> When`(`.then(expr)`で`Then`、`.otherwise(expr)`で`Expr`を返す)

**使用例**:
```python
df3 = pl.DataFrame({"score": [55, 80, 40, 95]})
print(df3.with_columns(
    pl.when(pl.col("score") >= 80).then(pl.lit("A"))
      .when(pl.col("score") >= 60).then(pl.lit("B"))
      .otherwise(pl.lit("C")).alias("grade")
))
```
実行結果:
```
shape: (4, 2)
┌───────┬───────┐
│ score ┆ grade │
│ ---   ┆ ---   │
│ i64   ┆ str   │
╞═══════╪═══════╡
│ 55    ┆ C     │
│ 80    ┆ A     │
│ 40    ┆ C     │
│ 95    ┆ A     │
└───────┴───────┘
```

**注意点・落とし穴**:
- `.when()`は複数回チェーンでき、最初にマッチした条件が採用される(上から順に評価される点はpandasの`np.select`と同じ)。`.otherwise()`を省略すると該当しない行は`null`になる。

### `polars.selectors`(`cs.numeric()` など)

**用途**: dtypeや命名パターンに基づいて列をまとめて選択する。`pl.col()`より宣言的に「数値列だけ」「文字列列だけ」といった選択ができる(pandasの`select_dtypes`に近いが式の中でシームレスに使える)。

**シグネチャ**: `cs.numeric() -> Selector` / `cs.string(*, include_categorical=False) -> Selector` など、dtypeカテゴリごとに関数が用意されている。

**使用例**:
```python
import polars.selectors as cs

df_cs = pl.DataFrame({"name": ["a", "b"], "age": [1, 2], "height": [1.5, 1.6]})
print(df_cs.select(cs.numeric()))
print(df_cs.select(cs.string()))
```
実行結果:
```
shape: (2, 2)
┌─────┬────────┐
│ age ┆ height │
│ --- ┆ ---    │
│ i64 ┆ f64    │
╞═════╪════════╡
│ 1   ┆ 1.5    │
│ 2   ┆ 1.6    │
└─────┴────────┘
shape: (2, 1)
┌──────┐
│ name │
│ ---  │
│ str  │
╞══════╡
│ a    │
│ b    │
└──────┘
```

**注意点・落とし穴**:
- `Selector`は集合演算(`cs.numeric() - cs.by_name("age")`など)が可能で、`pl.col()`と組み合わせて柔軟に列集合を組み立てられる。

### `pl.struct(...)` / `Expr.struct.field(...)`

**用途**: 複数列をまとめて1つの構造体(struct)型の列にする/structから特定フィールドを取り出す。pandasには直接の対応物がない、複数列をひとまとめに扱うための機能。

**シグネチャ**: `pl.struct(*exprs, schema=None, eager=False, **named_exprs) -> Expr | Series` / `Expr.struct.field(self, name, *more_names) -> Expr`

**使用例**:
```python
df_struct = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).with_columns(pl.struct(["a", "b"]).alias("s"))
print(df_struct)
print(df_struct.with_columns(pl.col("s").struct.field("a").alias("a_again")))
```
実行結果:
```
shape: (2, 3)
┌─────┬─────┬───────────┐
│ a   ┆ b   ┆ s         │
│ --- ┆ --- ┆ ---       │
│ i64 ┆ i64 ┆ struct[2] │
╞═════╪═════╪═══════════╡
│ 1   ┆ 3   ┆ {1,3}     │
│ 2   ┆ 4   ┆ {2,4}     │
└─────┴─────┴───────────┘
shape: (2, 4)
┌─────┬─────┬───────────┬─────────┐
│ a   ┆ b   ┆ s         ┆ a_again │
│ --- ┆ --- ┆ ---       ┆ ---     │
│ i64 ┆ i64 ┆ struct[2] ┆ i64     │
╞═════╪═════╪═══════════╪═════════╡
│ 1   ┆ 3   ┆ {1,3}     ┆ 1       │
│ 2   ┆ 4   ┆ {2,4}     ┆ 2       │
└─────┴─────┴───────────┴─────────┘
```

**注意点・落とし穴**:
- struct列は`group_by`のキーや複数列の一意性判定(`unique(subset=...)`の代替)にも使える、polars特有の便利な中間表現。

---

## フィルタ・ソート・ユニーク

### `df.filter(...)`

**用途**: 条件式に合致する行だけを抽出する。pandasの`df[condition]`や`df.query()`に相当。

**シグネチャ**: `df.filter(*predicates, **constraints)`

**使用例**:
```python
df4 = pl.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
print(df4.filter((pl.col("a") > 2) & (pl.col("b") < 40)))
```
実行結果:
```
shape: (1, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 3   ┆ 30  │
└─────┴─────┘
```

**注意点・落とし穴**:
- 複数条件は`&`/`|`で結合する(Python標準の`and`/`or`は使えない)。pandasの`query()`のような文字列式ではなく、常に式(Expression)を渡す。

### `Expr.is_in(...)`

**用途**: 各要素が指定した値集合に含まれるかを判定する。`filter`と組み合わせて使うことが多い。

**シグネチャ**: `Expr.is_in(self, other, *, nulls_equal=False) -> Expr`

**使用例**:
```python
df_isin = pl.DataFrame({"fruit": ["apple", "banana", "cherry"]})
print(df_isin.filter(pl.col("fruit").is_in(["apple", "cherry"])))
```
実行結果:
```
shape: (2, 1)
┌────────┐
│ fruit  │
│ ---    │
│ str    │
╞════════╡
│ apple  │
│ cherry │
└────────┘
```

### `df.sort(...)`

**用途**: 指定した列(式)でDataFrameを並べ替える。

**シグネチャ**: `df.sort(by, *more_by, descending=False, nulls_last=False, multithreaded=True, maintain_order=False)`

**使用例**:
```python
df5 = pl.DataFrame({"a": [3, 1, 2], "b": ["x", "y", "z"]})
print(df5.sort("a"))
print(df5.sort("a", descending=True))
```
実行結果:
```
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ y   │
│ 2   ┆ z   │
│ 3   ┆ x   │
└─────┴─────┘
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 3   ┆ x   │
│ 2   ┆ z   │
│ 1   ┆ y   │
└─────┴─────┘
```

**注意点・落とし穴**:
- pandasの`ascending=`ではなく`descending=`(意味が反転)である点に注意。

### `df.unique(...)`

**用途**: 重複行を除去する。

**シグネチャ**: `df.unique(subset=None, *, keep='any', maintain_order=False)`

**使用例**:
```python
df6 = pl.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
print(df6.unique(maintain_order=True))
```
実行結果:
```
shape: (2, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 3   │
│ 2   ┆ 4   │
└─────┴─────┘
```

**注意点・落とし穴**:
- デフォルトの`keep='any'`は「どの重複行が残るか保証しない」代わりに並列処理で高速。行の順序を保証したい場合は`maintain_order=True`を明示する必要がある(デフォルトでは順序が変わりうる)。

### `df.top_k(...)` / `df.bottom_k(...)`

**用途**: 指定列で上位/下位k件を取得する(pandasの`nlargest`/`nsmallest`に相当)。

**シグネチャ**: `df.top_k(k, *, by, reverse=False)` / `df.bottom_k(k, *, by, reverse=False)`

**使用例**:
```python
df_topk = pl.DataFrame({"a": [5, 1, 9, 3]})
print(df_topk.top_k(2, by="a"))
print(df_topk.bottom_k(2, by="a"))
```
実行結果:
```
shape: (2, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 9   │
│ 5   │
└─────┘
shape: (2, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 1   │
│ 3   │
└─────┘
```

**注意点・落とし穴**:
- `by`は必須のキーワード専用引数(pandasの`columns`位置引数と違い省略不可)。

---

## グループ化・集計とウィンドウ関数

### `df.group_by(...).agg(...)`

**用途**: 指定したキーでグループ化し、集計式を適用する。

**シグネチャ**: `df.group_by(*by, maintain_order=False, **named_by) -> GroupBy` / `GroupBy.agg(*aggs, **named_aggs) -> DataFrame`

**使用例**:
```python
df_g = pl.DataFrame({"team": ["A", "A", "B", "B"], "score": [10, 20, 30, 40]})
print(df_g.group_by("team", maintain_order=True).agg(
    total=pl.col("score").sum(),
    avg=pl.col("score").mean(),
    mx=pl.col("score").max(),
))
```
実行結果:
```
shape: (2, 4)
┌──────┬───────┬──────┬─────┐
│ team ┆ total ┆ avg  ┆ mx  │
│ ---  ┆ ---   ┆ ---  ┆ --- │
│ str  ┆ i64   ┆ f64  ┆ i64 │
╞══════╪═══════╪══════╪═════╡
│ A    ┆ 30    ┆ 15.0 ┆ 20  │
│ B    ┆ 70    ┆ 35.0 ┆ 40  │
└──────┴───────┴──────┴─────┘
```

**注意点・落とし穴**:
- デフォルト(`maintain_order=False`)ではグループの出力順序が保証されない(内部的に並列処理されるため)。pandasの`groupby(sort=True)`相当の挙動が欲しい場合は`maintain_order=True`を指定する。
- 集計名の指定はpandasの`agg(名前=("列","関数"))`と似た形式(`agg(名前=pl.col("列").関数())`)で行う。

### `Expr.over(...)`

**用途**: グループごとの集計値を、元のDataFrameと同じ行数・順序で返す(ウィンドウ関数。pandasの`groupby().transform()`に相当)。

**シグネチャ**: `Expr.over(self, partition_by=None, *more_exprs, order_by=None, descending=False, nulls_last=False, mapping_strategy='group_to_rows') -> Expr`

**使用例**:
```python
df_g2 = df_g.with_columns(pl.col("score").mean().over("team").alias("team_mean"))
print(df_g2)
```
実行結果:
```
shape: (4, 3)
┌──────┬───────┬───────────┐
│ team ┆ score ┆ team_mean │
│ ---  ┆ ---   ┆ ---       │
│ str  ┆ i64   ┆ f64       │
╞══════╪═══════╪═══════════╡
│ A    ┆ 10    ┆ 15.0      │
│ A    ┆ 20    ┆ 15.0      │
│ B    ┆ 30    ┆ 35.0      │
│ B    ┆ 40    ┆ 35.0      │
└──────┴───────┴───────────┘
```

**注意点・落とし穴**:
- `group_by().agg()`は行数がグループ数に集約されるのに対し、`.over()`は元の行数を保ったまま各行にグループ集計値を割り当てる。`with_columns`と組み合わせて使うのが定石。

### `df.group_by_dynamic(...)`

**用途**: 時間軸に沿って一定間隔の動的ウィンドウ(例: 3日ごと)でグループ化・集計する。pandasの`resample()`に近いが、任意のオフセット・重なりのあるウィンドウなどより柔軟な指定ができる。

**シグネチャ**: `df.group_by_dynamic(index_column, *, every, period=None, offset=None, include_boundaries=False, closed='left', label='left', group_by=None, start_by='window') -> DynamicGroupBy`

**使用例**:
```python
df_dyn = pl.DataFrame({
    "t": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 6), "1d", eager=True),
    "v": [1, 2, 3, 4, 5, 6],
})
print(df_dyn.group_by_dynamic("t", every="3d").agg(pl.col("v").sum()))
```
実行結果:
```
shape: (3, 2)
┌─────────────────────┬─────┐
│ t                   ┆ v   │
│ ---                 ┆ --- │
│ datetime[μs]        ┆ i64 │
╞═════════════════════╪═════╡
│ 2023-12-31 00:00:00 ┆ 3   │
│ 2024-01-03 00:00:00 ┆ 12  │
│ 2024-01-06 00:00:00 ┆ 6   │
└─────────────────────┴─────┘
```

**注意点・落とし穴**:
- `index_column`はソート済みである必要がある(未ソートだとエラーまたは誤った結果になりうる)。
- ウィンドウの起点は`start_by="window"`(デフォルト)により暦の境界(この例では日付の3日区切り)に自動整列されるため、上の例のように最初の窓が実データより前(`2023-12-31`)から始まることがある。

---

## 結合・連結

### `df.join(...)`

**用途**: SQLのJOINに相当する、キー列に基づくDataFrame同士の結合。

**シグネチャ**: `df.join(other, on=None, how='inner', *, left_on=None, right_on=None, suffix='_right', validate='m:m', nulls_equal=False, coalesce=None, ...)`

**使用例**:
```python
left = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
right = pl.DataFrame({"id": [2, 3, 4], "val": [200, 300, 400]})
print(left.join(right, on="id", how="inner"))
print(left.join(right, on="id", how="full"))
```
実行結果:
```
shape: (2, 3)
┌─────┬──────┬─────┐
│ id  ┆ name ┆ val │
│ --- ┆ ---  ┆ --- │
│ i64 ┆ str  ┆ i64 │
╞═════╪══════╪═════╡
│ 2   ┆ b    ┆ 200 │
│ 3   ┆ c    ┆ 300 │
└─────┴──────┴─────┘
shape: (4, 4)
┌──────┬──────┬──────────┬──────┐
│ id   ┆ name ┆ id_right ┆ val  │
│ ---  ┆ ---  ┆ ---      ┆ ---  │
│ i64  ┆ str  ┆ i64      ┆ i64  │
╞══════╪══════╪══════════╪══════╡
│ 2    ┆ b    ┆ 2        ┆ 200  │
│ 3    ┆ c    ┆ 3        ┆ 300  │
│ null ┆ null ┆ 4        ┆ 400  │
│ 1    ┆ a    ┆ null     ┆ null │
└──────┴──────┴──────────┴──────┘
```

**注意点・落とし穴**:
- pandasの`how="outer"`に相当するのは`how="full"`という名称(`"outer"`ではない)。
- `how="full"`かつ`coalesce`未指定の場合、結合キー列が`id`/`id_right`のように分裂したまま残る(pandasの`merge`のように自動でキーが1本化されない)。1本化したい場合は`coalesce=True`を指定する。

### `pl.concat(...)`

**用途**: 複数のDataFrame/Seriesを縦方向・横方向に連結する。

**シグネチャ**: `pl.concat(items, *, how='vertical', rechunk=False, parallel=True, strict=None)`

**使用例**:
```python
df_c1 = pl.DataFrame({"a": [1, 2]})
df_c2 = pl.DataFrame({"a": [3, 4]})
print(pl.concat([df_c1, df_c2]))

df_c3 = pl.DataFrame({"b": [5, 6]})
print(pl.concat([df_c1, df_c3], how="horizontal_extend"))
```
実行結果:
```
shape: (4, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 1   │
│ 2   │
│ 3   │
│ 4   │
└─────┘
shape: (2, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 5   │
│ 2   ┆ 6   │
└─────┴─────┘
```

**注意点・落とし穴**:
- デフォルトの`how='vertical'`は列名・型が一致している前提で単純に縦連結する(pandasの`ignore_index=True`相当がデフォルトの挙動で、元のインデックスという概念自体がないため気にする必要がない)。
- 横連結は`how="horizontal"`ではなく`how="horizontal_extend"`を使う(`"horizontal"`は将来的に高さが一致しないと使えなくなる予定で非推奨警告が出る、polars 1.42.1時点で確認)。

---

## Lazy API

### `pl.scan_csv(...)` / `lf.collect()`

**用途**: ファイルを即座に読み込まず、クエリプラン(LazyFrame)を構築する。`filter`・`group_by`などを繋げたあと`.collect()`した時点で初めて最適化された実行が走る。

**シグネチャ**: `pl.scan_csv(source, *, has_header=True, separator=',', schema=None, n_rows=None, ...)` / `LazyFrame.collect(self, *, type_coercion=True, predicate_pushdown=True, projection_pushdown=True, no_optimization=False, engine='auto', ...) -> DataFrame`

**使用例**:
```python
pl.DataFrame({"team": ["A", "A", "B", "B"], "score": [10, 20, 30, 40]}).write_csv("/tmp/sample_lazy.csv")
lf = pl.scan_csv("/tmp/sample_lazy.csv").filter(pl.col("score") > 15).group_by("team").agg(pl.col("score").sum())
print(type(lf))
print(lf.collect())
```
実行結果:
```
<class 'polars.lazyframe.frame.LazyFrame'>
shape: (2, 2)
┌──────┬───────┐
│ team ┆ score │
│ ---  ┆ ---   │
│ str  ┆ i64   │
╞══════╪═══════╡
│ A    ┆ 20    │
│ B    ┆ 70    │
└──────┴───────┘
```

**注意点・落とし穴**:
- `scan_csv`の時点ではファイルはまだ読まれていない(スキーマの軽量な推測のみ)。`.collect()`を呼ぶまで実データは取得されない。
- Lazy APIでは述語(`filter`条件)やプロジェクション(`select`で使う列)がファイル読み込みの段階まで押し下げられる(述語プッシュダウン)ため、大きなファイルの一部だけが必要な場合にEager API(`pl.read_csv`後に`filter`)より高速・省メモリになりうる。

### `lf.explain(...)`

**用途**: LazyFrameが実際にどう最適化・実行されるかのクエリプランを文字列で確認する。デバッグやパフォーマンスチューニングに使う(pandasには対応する概念がない)。

**シグネチャ**: `LazyFrame.explain(self, *, format='plain', optimized=True, streaming=False, engine='auto', ...) -> str`

**使用例**:
```python
print(lf.explain())
```
実行結果:
```
AGGREGATE[maintain_order: false]
  [col("score").sum()] BY [col("team")]
  FROM
  Csv SCAN [/tmp/sample_lazy.csv]
  PROJECT */2 COLUMNS
  SELECTION: col("score") > 15
  ESTIMATED ROWS: 6
```

**注意点・落とし穴**:
- プランは下から上に読む(`Csv SCAN` → `SELECTION`(フィルタ) → `AGGREGATE`)。`optimized=False`にすると最適化前の素朴なプランと比較できる。

### `df.lazy()`

**用途**: 既存のEager DataFrameをLazyFrameに変換する。逆に`lf.collect()`でEagerに戻る。EagerとLazyを行き来できる設計になっている。

**シグネチャ**: `df.lazy(self) -> LazyFrame`

**使用例**:
```python
df_lz = pl.DataFrame({"a": [1, 2, 3]})
print(type(df_lz.lazy()))
print(df_lz.lazy().filter(pl.col("a") > 1).collect())
```
実行結果:
```
<class 'polars.lazyframe.frame.LazyFrame'>
shape: (2, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 2   │
│ 3   │
└─────┘
```

**注意点・落とし穴**:
- 既にメモリ上にある(=構築コストが既に発生している)DataFrameを`.lazy()`にしても、ファイル読み込み自体の最適化(述語プッシュダウンなど)の恩恵はない。複数の変換をチェーンする際の式の最適化(共通部分式の除去など)には依然として意味がある。

---

## 文字列操作(.str)

### `Expr.str.contains(...)`

**用途**: 各要素が指定パターン(正規表現がデフォルト)を含むか判定する。

**シグネチャ**: `Expr.str.contains(self, pattern, *, literal=False, strict=True) -> Expr`

**使用例**:
```python
df_str = pl.DataFrame({"s": ["Apple", "banana", "Cherry", None]})
print(df_str.with_columns(pl.col("s").str.contains("(?i)a").alias("has_a")))
```
実行結果:
```
shape: (4, 2)
┌────────┬───────┐
│ s      ┆ has_a │
│ ---    ┆ ---   │
│ str    ┆ bool  │
╞════════╪═══════╡
│ Apple  ┆ true  │
│ banana ┆ true  │
│ Cherry ┆ false │
│ null   ┆ null  │
└────────┴───────┘
```

**注意点・落とし穴**:
- pandasの`case=False`のような専用引数はなく、大文字小文字を無視したい場合は正規表現のインラインフラグ`(?i)`を使う。
- `null`はそのまま`null`として伝播する(pandasの`na=`のような明示指定は不要で、比較・フィルタでは自動的に偽扱いになる)。

### `Expr.str.replace(...)`

**用途**: 文字列の一部を置換する(デフォルトは最初の1件のみ)。

**シグネチャ**: `Expr.str.replace(self, pattern, value, *, literal=False, n=1) -> Expr`

**使用例**:
```python
print(df_str.with_columns(pl.col("s").str.replace("a", "@", literal=True).alias("rep")))
```
実行結果:
```
shape: (4, 2)
┌────────┬────────┐
│ s      ┆ rep    │
│ ---    ┆ ---    │
│ str    ┆ str    │
╞════════╪════════╡
│ Apple  ┆ Apple  │
│ banana ┆ b@nana │
│ Cherry ┆ Cherry │
│ null   ┆ null   │
└────────┴────────┘
```

**注意点・落とし穴**:
- `str.replace`はデフォルトで**最初の1件だけ**置換する(`"banana"` → `"b@nana"`、2つ目以降の`a`は変わらない)。すべて置換したい場合は`str.replace_all`を使う。pandasの`str.replace`(デフォルトで全置換)と挙動が逆なので混同しやすい。
- `pattern`はデフォルトで正規表現として解釈される。単純な文字列置換なら`literal=True`を指定した方が意図が明確かつ高速。

### `Expr.str.to_uppercase(...)`

**用途**: 文字列を大文字に変換する。

**シグネチャ**: `Expr.str.to_uppercase(self) -> Expr`

**使用例**:
```python
print(df_str.with_columns(pl.col("s").str.to_uppercase().alias("upper")))
```
実行結果:
```
shape: (4, 2)
┌────────┬────────┐
│ s      ┆ upper  │
│ ---    ┆ ---    │
│ str    ┆ str    │
╞════════╪════════╡
│ Apple  ┆ APPLE  │
│ banana ┆ BANANA │
│ Cherry ┆ CHERRY │
│ null   ┆ null   │
└────────┴────────┘
```

### `Expr.str.split(...)`

**用途**: 文字列を区切り文字で分割し、リスト型の列にする。

**シグネチャ**: `Expr.str.split(self, by, *, inclusive=False, literal=True, strict=True) -> Expr`

**使用例**:
```python
df_split = pl.DataFrame({"s": ["a,b,c", "d,e"]})
print(df_split.with_columns(pl.col("s").str.split(",").alias("parts")))
```
実行結果:
```
shape: (2, 2)
┌───────┬─────────────────┐
│ s     ┆ parts           │
│ ---   ┆ ---             │
│ str   ┆ list[str]       │
╞═══════╪═════════════════╡
│ a,b,c ┆ ["a", "b", "c"] │
│ d,e   ┆ ["d", "e"]      │
└───────┴─────────────────┘
```

**注意点・落とし穴**:
- pandasの`expand=True`のように複数列に展開するのではなく、`list[str]`型の1列になる。個々の要素を列に展開したい場合は`.list.to_struct()`などを使う。

### `Expr.str.extract(...)`

**用途**: 正規表現の捕捉グループにマッチした部分を取り出す。

**シグネチャ**: `Expr.str.extract(self, pattern, group_index=1) -> Expr`

**使用例**:
```python
df_ext = pl.DataFrame({"s": ["item_001", "item_045"]})
print(df_ext.with_columns(pl.col("s").str.extract(r"item_(\d+)", 1).alias("num")))
```
実行結果:
```
shape: (2, 2)
┌──────────┬─────┐
│ s        ┆ num │
│ ---      ┆ --- │
│ str      ┆ str │
╞══════════╪═════╡
│ item_001 ┆ 001 │
│ item_045 ┆ 045 │
└──────────┴─────┘
```

**注意点・落とし穴**:
- `group_index`はキーワードではなく2番目の位置引数(デフォルト1)。pandasの`expand=True`のように複数グループを一度に複数列へ展開する機能はなく、必要なら`str.extract_groups()`を使う。

### `Expr.str.strip_chars(...)`

**用途**: 文字列の前後の空白(または指定文字)を除去する。

**シグネチャ**: `Expr.str.strip_chars(self, characters=None) -> Expr`

**使用例**:
```python
df_strip = pl.DataFrame({"s": ["  hello  ", "world  "]})
print(df_strip.with_columns(pl.col("s").str.strip_chars().alias("stripped")))
```
実行結果:
```
shape: (2, 2)
┌───────────┬──────────┐
│ s         ┆ stripped │
│ ---       ┆ ---      │
│ str       ┆ str      │
╞═══════════╪══════════╡
│   hello   ┆ hello    │
│ world     ┆ world    │
└───────────┴──────────┘
```

**注意点・落とし穴**:
- pandasの`str.strip()`に相当するが、名前が`strip_chars`になっている(引数なしの`str.strip()`は存在しない)。片側だけ除去する`strip_chars_start`/`strip_chars_end`もある。

---

## 日時処理(.dt)

### `Expr.str.to_date(...)` / `Expr.str.to_datetime(...)`

**用途**: 文字列を`Date`/`Datetime`型に変換する。pandasの`pd.to_datetime`に相当。

**シグネチャ**: `Expr.str.to_datetime(self, format=None, *, time_unit=None, time_zone=None, strict=True, exact=True, cache=True, ambiguous='raise') -> Expr`(`to_date`も同様の形式引数を持つ)

**使用例**:
```python
df_dt = pl.DataFrame({"d": ["2024-01-15", "2024-06-20"]}).with_columns(pl.col("d").str.to_date("%Y-%m-%d"))
print(df_dt)
```
実行結果:
```
shape: (2, 1)
┌────────────┐
│ d          │
│ ---        │
│ date       │
╞════════════╡
│ 2024-01-15 │
│ 2024-06-20 │
└────────────┘
```

**注意点・落とし穴**:
- polarsは`Date`(日付のみ)と`Datetime`(時刻を含む)を型として明確に区別する。時刻情報が不要なら`str.to_date`、必要なら`str.to_datetime`を使い分ける。

### `Expr.dt.year(...)` / `Expr.dt.strftime(...)`

**用途**: `Date`/`Datetime`型の列から年・曜日名などの要素を取り出す。

**シグネチャ**: `Expr.dt.year(self) -> Expr` / `Expr.dt.strftime(self, format) -> Expr`

**使用例**:
```python
print(df_dt.with_columns(pl.col("d").dt.year().alias("year"), pl.col("d").dt.strftime("%A").alias("weekday")))
```
実行結果:
```
shape: (2, 3)
┌────────────┬──────┬──────────┐
│ d          ┆ year ┆ weekday  │
│ ---        ┆ ---  ┆ ---      │
│ date       ┆ i32  ┆ str      │
╞════════════╪══════╪══════════╡
│ 2024-01-15 ┆ 2024 ┆ Monday   │
│ 2024-06-20 ┆ 2024 ┆ Thursday │
└────────────┴──────┴──────────┘
```

**注意点・落とし穴**:
- pandasの`.dt.day_name()`のような専用メソッドはなく、`strftime("%A")`で曜日名を得る(Cのstrftime書式に準拠)。

### `pl.date_range(...)`

**用途**: 一定間隔の日付・日時のSeriesを生成する。

**シグネチャ**: `pl.date_range(start, end, interval='1d', *, closed='both', eager=False) -> Series | Expr`

**使用例**:
```python
print(pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 5), "1d", eager=True))
```
実行結果:
```
shape: (5,)
Series: 'date' [date]
[
	2024-01-01
	2024-01-02
	2024-01-03
	2024-01-04
	2024-01-05
]
```

**注意点・落とし穴**:
- デフォルト(`eager=False`)では即座に値を持つSeriesではなく式(Expr)を返す。単独で値が欲しい場合は本例のように`eager=True`を指定する。
- `closed='both'`がデフォルトのため両端を含む(pandasの`date_range`と同じ挙動だが、`inclusive`ではなく`closed`という引数名)。

---

## 欠損値処理

### `Expr.is_null(...)` / `df.null_count(...)`

**用途**: 各要素が欠損値(`null`)かどうかを判定する/列ごとの欠損数を集計する。

**シグネチャ**: `Expr.is_null(self) -> Expr` / `df.null_count(self) -> DataFrame`

**使用例**:
```python
df_na = pl.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
print(df_na.select(pl.all().is_null()))
print(df_na.null_count())
```
実行結果:
```
shape: (3, 2)
┌───────┬───────┐
│ a     ┆ b     │
│ ---   ┆ ---   │
│ bool  ┆ bool  │
╞═══════╪═══════╡
│ false ┆ true  │
│ true  ┆ false │
│ false ┆ false │
└───────┴───────┘
shape: (1, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ u32 ┆ u32 │
╞═════╪═════╡
│ 1   ┆ 1   │
└─────┴─────┘
```

**注意点・落とし穴**:
- polarsには`NaN`(浮動小数点の非数)と`null`(欠損値)という区別がある。`is_null()`は`null`のみを検出し、`NaN`は検出しない(`NaN`を調べたい場合は`is_nan()`を使う)。pandasでは両者が`NaN`として一体化されがちなので注意。

### `Expr.fill_null(...)`

**用途**: 欠損値を指定した値・式・前後の値などで埋める。

**シグネチャ**: `Expr.fill_null(self, value=None, strategy=None, limit=None) -> Expr`

**使用例**:
```python
print(df_na.fill_null(0))
print(df_na.fill_null(strategy="forward"))
```
実行結果:
```
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 0   │
│ 0   ┆ 2   │
│ 3   ┆ 3   │
└─────┴─────┘
shape: (3, 2)
┌─────┬──────┐
│ a   ┆ b    │
│ --- ┆ ---  │
│ i64 ┆ i64  │
╞═════╪══════╡
│ 1   ┆ null │
│ 1   ┆ 2    │
│ 3   ┆ 3    │
└─────┴──────┘
```

**注意点・落とし穴**:
- `df.fill_null(0)`のようにDataFrameに直接呼べる(内部で全列に`pl.all().fill_null(0)`相当が適用される)。列ごとに異なる値で埋めたい場合は`with_columns`内で列ごとに`fill_null`を呼び分ける。
- `strategy="forward"`(前方埋め)でも列の先頭が欠損だと埋めようがなく`null`のまま残る(上の例の`b`列1行目)。

### `df.drop_nulls(...)`

**用途**: 欠損値を含む行を除外する。

**シグネチャ**: `df.drop_nulls(subset=None) -> DataFrame`

**使用例**:
```python
print(df_na.drop_nulls())
```
実行結果:
```
shape: (1, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 3   ┆ 3   │
└─────┴─────┘
```

**注意点・落とし穴**:
- pandasの`dropna(how="any"/"all")`のような`how`引数はない(常に「1つでも欠損があれば除外」)。列を限定したい場合は`subset=`を使う。

---

## ピボット・reshape

### `df.pivot(...)`

**用途**: 縦持ち(long)データを横持ち(wide)データに変換する。

**シグネチャ**: `df.pivot(on, on_columns=None, *, index=None, values=None, aggregate_function=None, maintain_order=True, sort_columns=False, separator='_', column_naming='auto') -> DataFrame`

**使用例**:
```python
df_piv = pl.DataFrame({"date": ["d1", "d1", "d2", "d2"], "var": ["x", "y", "x", "y"], "val": [1, 2, 3, 4]})
print(df_piv.pivot(on="var", index="date", values="val"))
```
実行結果:
```
shape: (2, 3)
┌──────┬─────┬─────┐
│ date ┆ x   ┆ y   │
│ ---  ┆ --- ┆ --- │
│ str  ┆ i64 ┆ i64 │
╞══════╪═════╪═════╡
│ d1   ┆ 1   ┆ 2   │
│ d2   ┆ 3   ┆ 4   │
└──────┴─────┴─────┘
```

**注意点・落とし穴**:
- pandasは「集計なしの`pivot()`」と「集計ありの`pivot_table()`」が別メソッドだが、polarsは`aggregate_function`引数を持つ1つの`pivot()`に統合されている。`index`×`on`の組み合わせに重複がある状態で`aggregate_function`を指定しないと、`ComputeError: aggregation 'item' expected no or a single value, got 2 values`という例外になる(実行して確認済み)。集計したい場合は`aggregate_function="sum"`のように明示する必要がある。
- 第1引数は`columns`ではなく`on`という名前である点に注意(引数名がpandasと異なる)。

### `df.unpivot(...)`

**用途**: 横持ち(wide)データを縦持ち(long)データに変換する(`pivot`の逆操作。pandasの`melt`に相当)。

**シグネチャ**: `df.unpivot(on=None, *, index=None, variable_name=None, value_name=None) -> DataFrame`

**使用例**:
```python
df_up = pl.DataFrame({"id": [1, 2], "x": [10, 20], "y": [30, 40]})
print(df_up.unpivot(index="id", on=["x", "y"]))
```
実行結果:
```
shape: (4, 3)
┌─────┬──────────┬───────┐
│ id  ┆ variable ┆ value │
│ --- ┆ ---      ┆ ---   │
│ i64 ┆ str      ┆ i64   │
╞═════╪══════════╪═══════╡
│ 1   ┆ x        ┆ 10    │
│ 2   ┆ x        ┆ 20    │
│ 1   ┆ y        ┆ 30    │
│ 2   ┆ y        ┆ 40    │
└─────┴──────────┴───────┘
```

**注意点・落とし穴**:
- 古い`df.melt(id_vars=..., value_vars=...)`は非推奨。実際に実行すると`` `DataFrame.melt` is deprecated; use `DataFrame.unpivot` instead, with `index` instead of `id_vars` and `on` instead of `value_vars` ``という`DeprecationWarning`が出ることを確認済み。引数名も`id_vars`→`index`、`value_vars`→`on`に変わっている。

### `df.explode(...)`

**用途**: リストなどのコレクションを含むセルを、要素ごとに複数行へ展開する。

**シグネチャ**: `df.explode(columns, *more_columns, empty_as_null=<不使用時のデフォルト>, keep_nulls=True) -> DataFrame`

**使用例**:
```python
df_exp = pl.DataFrame({"id": [1, 2], "items": [["a", "b"], ["c"]]})
print(df_exp.explode("items"))
```
実行結果:
```
shape: (3, 2)
┌─────┬───────┐
│ id  ┆ items │
│ --- ┆ ---   │
│ i64 ┆ str   │
╞═════╪═══════╡
│ 1   ┆ a     │
│ 1   ┆ b     │
│ 2   ┆ c     │
└─────┴───────┘
```

**注意点・落とし穴**:
- pandasの`explode()`と違い、行インデックスという概念がないため展開後に重複したインデックスを気にする必要はない。
- `empty_as_null`引数のデフォルト挙動が将来のバージョンで変わる旨の`DeprecationWarning`が出る(polars 1.44.1で実行確認済み)。空リストを`null`として扱いたいかどうかを明示したい場合は`empty_as_null=True/False`を指定するとよい。

### `Expr.list.len(...)` / `Expr.list.get(...)`

**用途**: リスト型の列の要素数を取得する/指定位置の要素を取り出す。

**シグネチャ**: `Expr.list.len(self) -> Expr` / `Expr.list.get(self, index, *, null_on_oob=False) -> Expr`

**使用例**:
```python
df_list = pl.DataFrame({"items": [["a", "b", "c"], ["d"]]})
print(df_list.with_columns(pl.col("items").list.len().alias("n"), pl.col("items").list.get(0).alias("first")))
```
実行結果:
```
shape: (2, 3)
┌─────────────────┬─────┬───────┐
│ items           ┆ n   ┆ first │
│ ---             ┆ --- ┆ ---   │
│ list[str]       ┆ u32 ┆ str   │
╞═════════════════╪═════╪═══════╡
│ ["a", "b", "c"] ┆ 3   ┆ a     │
│ ["d"]           ┆ 1   ┆ d     │
└─────────────────┴─────┴───────┘
```

**注意点・落とし穴**:
- `list.get()`はデフォルト(`null_on_oob=False`)だとインデックスが範囲外のときエラーになる。安全に`null`を返したい場合は`null_on_oob=True`を指定する。

---

## 型変換・その他便利メソッド

### `Expr.cast(...)`

**用途**: 列の型を明示的に変換する。

**シグネチャ**: `Expr.cast(self, dtype, *, strict=True, wrap_numerical=False) -> Expr`

**使用例**:
```python
df_cast = pl.DataFrame({"a": ["1", "2", "3"]})
print(df_cast.dtypes)
print(df_cast.with_columns(pl.col("a").cast(pl.Int64)).dtypes)
```
実行結果:
```
[String]
[Int64]
```

**注意点・落とし穴**:
- `df.astype(...)`のようなDataFrame側のメソッドはなく、必ず`with_columns(pl.col(...).cast(...))`のように式として書く。
- デフォルト(`strict=True`)では変換できない値があると例外になる。安全に`null`にしたい場合は`strict=False`を指定する。

### `df.to_pandas(...)`

**用途**: polarsのDataFrameをpandasのDataFrameに変換する(相互運用用)。

**シグネチャ**: `df.to_pandas(self, *, use_pyarrow_extension_array=False, **kwargs) -> pd.DataFrame`

**使用例**:
```python
df_r = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
print(type(df_r.to_pandas()))
print(df_r.to_pandas())
```
実行結果:
```
<class 'pandas.DataFrame'>
   a  b
0  1  3
1  2  4
```

**注意点・落とし穴**:
- 変換後のpandas DataFrameには自動的に0始まりの連番インデックスが振られる(polars側にインデックスはないため)。

### `df.to_dict(...)` / `df.to_numpy(...)`

**用途**: DataFrameをPython辞書やNumPy配列に変換する。

**シグネチャ**: `df.to_dict(self, *, as_series=True) -> dict[str, Series] | dict[str, list]` / `df.to_numpy(self, *, order='fortran', writable=False, allow_copy=True, structured=False, use_pyarrow=None) -> np.ndarray`

**使用例**:
```python
df_td = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
print(df_td.to_dict(as_series=False))
print(df_td.to_numpy())
```
実行結果:
```
{'a': [1, 2], 'b': [3, 4]}
[[1 3]
 [2 4]]
```

**注意点・落とし穴**:
- `to_dict()`はデフォルト(`as_series=True`)だと値がPythonのリストではなくpolarsの`Series`のまま返る。プレーンなリストが欲しい場合は`as_series=False`を明示する。

### `Expr.shift(...)` / `Expr.diff(...)`

**用途**: 値を前後にずらす(ラグ特徴量の作成)/前の値との差分を取る。

**シグネチャ**: `Expr.shift(self, n=1, *, fill_value=None) -> Expr` / `Expr.diff(self, n=1, null_behavior='ignore') -> Expr`

**使用例**:
```python
df_sh = pl.DataFrame({"v": [1, 2, 3, 4]})
print(df_sh.with_columns(pl.col("v").shift(1).alias("shifted"), pl.col("v").diff().alias("diff")))
```
実行結果:
```
shape: (4, 3)
┌─────┬─────────┬──────┐
│ v   ┆ shifted ┆ diff │
│ --- ┆ ---     ┆ ---  │
│ i64 ┆ i64     ┆ i64  │
╞═════╪═════════╪══════╡
│ 1   ┆ null    ┆ null │
│ 2   ┆ 1       ┆ 1    │
│ 3   ┆ 2       ┆ 1    │
│ 4   ┆ 3       ┆ 1    │
└─────┴─────────┴──────┘
```

**注意点・落とし穴**:
- pandasでは整数列を`shift`すると欠損混入で`float64`に昇格するが、polarsは整数型のまま`null`を保持できる(nullable性が型システムに組み込まれているため)。

### `Expr.rolling_mean(...)`

**用途**: 移動窓(直近n件)の平均を計算する。

**シグネチャ**: `Expr.rolling_mean(self, window_size, weights=None, *, min_samples=None, center=False) -> Expr`

**使用例**:
```python
df_roll = pl.DataFrame({"v": [1, 2, 3, 4, 5]})
print(df_roll.with_columns(pl.col("v").rolling_mean(window_size=3).alias("roll_mean")))
```
実行結果:
```
shape: (5, 2)
┌─────┬───────────┐
│ v   ┆ roll_mean │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 1   ┆ null      │
│ 2   ┆ null      │
│ 3   ┆ 2.0       │
│ 4   ┆ 3.0       │
│ 5   ┆ 4.0       │
└─────┴───────────┘
```

**注意点・落とし穴**:
- 引数名は`window`ではなく`window_size`。窓が満たない先頭部分は`null`になる点はpandasの`rolling().mean()`と同じ。

### `s.value_counts(...)`

**用途**: 値ごとの出現回数を集計する。

**シグネチャ**: `Expr.value_counts(self, *, sort=False, parallel=False, name=None, normalize=False) -> Expr`(Seriesにも同名メソッドがある)

**使用例**:
```python
s_vc = pl.Series("x", ["a", "b", "a", "c", "a", "b"])
print(s_vc.value_counts())
```
実行結果:
```
shape: (3, 2)
┌─────┬───────┐
│ x   ┆ count │
│ --- ┆ ---   │
│ str ┆ u32   │
╞═════╪═══════╡
│ c   ┆ 1     │
│ b   ┆ 2     │
│ a   ┆ 3     │
└─────┴───────┘
```

**注意点・落とし穴**:
- 戻り値はpandasのような1本のSeriesではなく、元の値の列(`x`)と`count`列を持つ**2列のDataFrame**。デフォルト(`sort=False`)では出現回数順に並ぶ保証がない。

### `Expr.n_unique(...)`

**用途**: ユニークな値の個数を数える。

**シグネチャ**: `Expr.n_unique(self) -> Expr`(Seriesにも同名メソッドがある)

**使用例**:
```python
print(s_vc.n_unique())
```
実行結果:
```
3
```

### `Expr.map_elements(...)`

**用途**: 各要素にPythonの任意関数を適用する(polarsのベクトル化された式で表現できない場合の最終手段)。

**シグネチャ**: `Expr.map_elements(self, function, return_dtype=None, *, skip_nulls=True, pass_name=False, strategy='thread_local', returns_scalar=False) -> Expr`

**使用例**:
```python
df_map = pl.DataFrame({"a": [1, 2, 3]})
print(df_map.with_columns(pl.col("a").map_elements(lambda x: x ** 2, return_dtype=pl.Int64).alias("sq")))
```
実行結果:
```
shape: (3, 2)
┌─────┬─────┐
│ a   ┆ sq  │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 1   │
│ 2   ┆ 4   │
│ 3   ┆ 9   │
└─────┴─────┘
```

**注意点・落とし穴**:
- 実行すると`PolarsInefficientMapWarning`(「`Expr.map_elements`はネイティブの式APIより著しく遅い。この式の代わりに`pl.col("a") ** 2`を使うこと」という趣旨)が出る(polars 1.44.1で実行確認済み)。行ごとにPythonコールバックを呼ぶため、可能な限り`pl.col()`のベクトル化演算で書き直すべき。
- `return_dtype`を指定しないと最初の要素から型を推測しようとして遅くなる/失敗することがあるため、明示指定が推奨される。

### `df.sample(...)`

**用途**: DataFrameから行をランダムに抽出する。

**シグネチャ**: `df.sample(n=None, *, fraction=None, with_replacement=False, shuffle=None, seed=None) -> DataFrame`

**使用例**:
```python
df_sample = pl.DataFrame({"a": range(10)})
print(df_sample.sample(n=3, seed=42))
```
実行結果:
```
shape: (3, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 6   │
│ 2   │
│ 9   │
└─────┘
```

**注意点・落とし穴**:
- 乱数シードの引数はpandasの`random_state`ではなく`seed`。

### `df.rename(...)` / `df.drop(...)`

**用途**: 列名を変更する/列を削除する。

**シグネチャ**: `df.rename(mapping, *, strict=True) -> DataFrame` / `df.drop(*columns, strict=True) -> DataFrame`

**使用例**:
```python
df_r = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
print(df_r.rename({"a": "alpha", "b": "beta"}))
print(df_r.drop("a"))
```
実行結果:
```
shape: (2, 2)
┌───────┬──────┐
│ alpha ┆ beta │
│ ---   ┆ ---  │
│ i64   ┆ i64  │
╞═══════╪══════╡
│ 1     ┆ 3    │
│ 2     ┆ 4    │
└───────┴──────┘
shape: (2, 1)
┌─────┐
│ b   │
│ --- │
│ i64 │
╞═════╡
│ 3   │
│ 4   │
└─────┘
```

**注意点・落とし穴**:
- どちらも`inplace=True`のような引数はない(常に新しいDataFrameを返す非破壊的な設計)。

### `df.with_row_index(...)`

**用途**: 0始まりの連番インデックス列を明示的に追加する。インデックスを持たないpolarsで「行番号」が必要なときに使う。

**シグネチャ**: `df.with_row_index(name='index', offset=0) -> DataFrame`

**使用例**:
```python
df_wri = pl.DataFrame({"a": [10, 20, 30]})
print(df_wri.with_row_index(name="idx", offset=1))
```
実行結果:
```
shape: (3, 2)
┌─────┬─────┐
│ idx ┆ a   │
│ --- ┆ --- │
│ u32 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 10  │
│ 2   ┆ 20  │
│ 3   ┆ 30  │
└─────┴─────┘
```

**注意点・落とし穴**:
- あくまで通常の`u32`型の**列**として追加されるだけで、pandasのインデックスのように`loc`でのラベル参照に使えるわけではない。

### `Expr.clip(...)`

**用途**: 値を指定した下限・上限の範囲に収める(範囲外の値を境界値に丸める)。

**シグネチャ**: `Expr.clip(self, lower_bound=None, upper_bound=None) -> Expr`

**使用例**:
```python
df_clip = pl.DataFrame({"v": [-5, 0, 5, 15, 25]})
print(df_clip.with_columns(pl.col("v").clip(0, 20).alias("clipped")))
```
実行結果:
```
shape: (5, 2)
┌─────┬─────────┐
│ v   ┆ clipped │
│ --- ┆ ---     │
│ i64 ┆ i64     │
╞═════╪═════════╡
│ -5  ┆ 0       │
│ 0   ┆ 0       │
│ 5   ┆ 5       │
│ 15  ┆ 15      │
│ 25  ┆ 20      │
└─────┴─────────┘
```

### `Expr.rank(...)`

**用途**: 各値の順位を計算する。

**シグネチャ**: `Expr.rank(self, method='average', *, descending=False, seed=None) -> Expr`

**使用例**:
```python
s_rank = pl.Series("v", [10, 20, 20, 30])
print(s_rank.rank())
print(s_rank.rank(method="min"))
print(s_rank.rank(method="dense"))
```
実行結果:
```
shape: (4,)
Series: 'v' [f64]
[
	1.0
	2.5
	2.5
	4.0
]
shape: (4,)
Series: 'v' [u32]
[
	1
	2
	2
	4
]
shape: (4,)
Series: 'v' [u32]
[
	1
	2
	2
	3
]
```

**注意点・落とし穴**:
- `method="average"`(デフォルト)の結果のみ`f64`型で、それ以外(`"min"`, `"dense"`など)は`u32`型になる(平均順位は小数になりうるが、それ以外は整数順位のため)。この型の違いはpandasの`rank()`(常に`float64`)にはない挙動。

### `Expr.cum_sum(...)`

**用途**: 累積和を計算する。

**シグネチャ**: `Expr.cum_sum(self, *, reverse=False) -> Expr`

**使用例**:
```python
df_cum = pl.DataFrame({"v": [1, 2, 3, 4]})
print(df_cum.with_columns(pl.col("v").cum_sum().alias("cumsum")))
```
実行結果:
```
shape: (4, 2)
┌─────┬────────┐
│ v   ┆ cumsum │
│ --- ┆ ---    │
│ i64 ┆ i64    │
╞═════╪════════╡
│ 1   ┆ 1      │
│ 2   ┆ 3      │
│ 3   ┆ 6      │
│ 4   ┆ 10     │
└─────┴────────┘
```

**注意点・落とし穴**:
- pandasの`cumsum()`と異なりアンダースコア入りの`cum_sum`という名前(`cum_max`/`cum_min`/`cum_prod`も同様の命名規則)。
