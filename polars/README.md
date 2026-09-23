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
13. [応用・発展](#応用発展)
    - [Lazy APIのストリーミング実行・プロファイリング](#lazy-apiのストリーミング実行プロファイリング)
    - [struct/list dtype操作の応用](#structlist-dtype操作の応用)
    - [高度なウィンドウ・rolling](#高度なウィンドウrolling)
    - [カスタムUDF(map_batches/map_elementsのパフォーマンス注意)](#カスタムudfmap_batchesmap_elementsのパフォーマンス注意)
    - [SQLコンテキスト(pl.SQLContext)](#sqlコンテキストplsqlcontext)
    - [高度な結合(join_asof・join_where・semi/anti join)](#高度な結合join_asofjoin_wheresemianti-join)

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

---

## 応用・発展

ここから先は、pandasとの対比というより、polars自体がLazy API・大規模データ処理・SQL互換性のために持つ独自機能を扱う。基礎編と同様、すべて`/home/manaty/library-practicing/.venv/bin/python`(polars 1.44.1)で実行して確認済み。

### Lazy APIのストリーミング実行・プロファイリング

#### `lf.collect(engine="streaming")`

**用途**: LazyFrameのクエリを新しいストリーミングエンジンで実行する。データをバッチ単位で処理するため、メモリに乗り切らない大規模データでもメモリ使用量を抑えて処理できる。

**シグネチャ**: `LazyFrame.collect(self, *, type_coercion=True, predicate_pushdown=True, projection_pushdown=True, simplify_expression=True, slice_pushdown=True, comm_subplan_elim=True, comm_subexpr_elim=True, cluster_with_columns=True, collapse_joins=True, no_optimization=False, engine='auto', background=False, optimizations=<QueryOptFlags>, **_kwargs) -> DataFrame | InProcessQuery`(主要引数のみ抜粋。`engine`は`'auto'`/`'in-memory'`/`'streaming'`/`'gpu'`などを取る)

**使用例**:
```python
import polars as pl

pl.DataFrame({"team": ["A", "A", "B", "B"], "score": [10, 20, 30, 40]}).write_csv("/tmp/sample_lazy2.csv")
lf = pl.scan_csv("/tmp/sample_lazy2.csv").filter(pl.col("score") > 15).group_by("team").agg(pl.col("score").sum())

print(lf.collect())
print(lf.collect(engine="streaming"))
```
実行結果:
```
shape: (2, 2)
┌──────┬───────┐
│ team ┆ score │
│ ---  ┆ ---   │
│ str  ┆ i64   │
╞══════╪═══════╡
│ A    ┆ 20    │
│ B    ┆ 70    │
└──────┴───────┘
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
- 旧バージョンにあった`collect(streaming=True)`は非推奨で、polars 1.44.1では`engine='streaming'`を使う(`explain()`の`streaming`引数はまだ残っているが、`collect()`側は`engine`に統一されている)。
- ストリーミングエンジンは全ての演算に対応しているわけではなく、非対応の演算が含まれる場合は自動的に`in-memory`エンジンにフォールバックする。
- `group_by`の出力行順序は(`maintain_order=False`がデフォルトのため)エンジンや実行タイミングによって変わりうる。本例ではたまたま両エンジンとも`A, B`の順になったが、順序に依存したコードを書くべきではない。

#### `lf.profile()`

**用途**: LazyFrameの各実行ノードにかかった時間を計測する。`explain()`がクエリ「計画」を見るのに対し、`profile()`は実際に実行して計測した結果を返す。

**シグネチャ**: `LazyFrame.profile(self, *, type_coercion=True, predicate_pushdown=True, projection_pushdown=True, simplify_expression=True, no_optimization=False, slice_pushdown=True, comm_subplan_elim=True, comm_subexpr_elim=True, cluster_with_columns=True, collapse_joins=True, show_plot=False, truncate_nodes=0, figsize=(18, 8), engine='auto', optimizations=<QueryOptFlags>, **_kwargs) -> tuple[DataFrame, DataFrame]`

**使用例**:
```python
result, timings = lf.profile()
print(result)
print(timings)
```
実行結果:
```
shape: (2, 2)
┌──────┬───────┐
│ team ┆ score │
│ ---  ┆ ---   │
│ str  ┆ i64   │
╞══════╪═══════╡
│ A    ┆ 20    │
│ B    ┆ 70    │
└──────┴───────┘
shape: (1, 3)
┌──────────────┬───────┬─────┐
│ node         ┆ start ┆ end │
│ ---          ┆ ---   ┆ --- │
│ str          ┆ u64   ┆ u64 │
╞══════════════╪═══════╪═════╡
│ optimization ┆ 0     ┆ 602 │
└──────────────┴───────┴─────┘
```

**注意点・落とし穴**:
- 戻り値は`(クエリ結果のDataFrame, 各ノードの所要時間を表すDataFrame)`のタプル。`explain()`と違い実際にクエリを実行するため相応のコストがかかる。
- `timings`の`start`/`end`はマイクロ秒単位の相対時刻。ノード数が少ない単純なクエリでは`optimization`ノード1件だけのように、粒度が粗く見えることがある(実行確認済み)。実行のたびに実測時間(上の例の`602`)は変動するため、値そのものではなく「どのノードが相対的に重いか」を見るのに使う。
- `result`の行順序(`group_by`の出力順)は`maintain_order=False`がデフォルトのため保証されない。実際に同じクエリを複数回実行すると`A, B`の順になったり`B, A`の順になったりすることを確認済み。

#### `lf.sink_parquet(...)` / `lf.sink_csv(...)`

**用途**: LazyFrameの実行結果をDataFrameとしてメモリに全展開せず、ファイルへ直接ストリーム書き出しする。巨大な結果をメモリに乗せずに保存したい場合に使う。

**シグネチャ**: `LazyFrame.sink_parquet(self, path, *, compression='zstd', compression_level=None, statistics=True, row_group_size=None, data_page_size=None, maintain_order=True, ..., lazy=False, engine='auto', ...) -> LazyFrame | None` / `LazyFrame.sink_csv(self, path, *, include_bom=False, compression='uncompressed', include_header=True, separator=',', ..., lazy=False, engine='auto', ...) -> LazyFrame | None`

**使用例**:
```python
lf2 = pl.scan_csv("/tmp/sample_lazy2.csv").filter(pl.col("score") > 15)
ret = lf2.sink_parquet("/tmp/sink_out.parquet")
print(ret)
print(pl.read_parquet("/tmp/sink_out.parquet"))
```
実行結果:
```
None
shape: (3, 2)
┌──────┬───────┐
│ team ┆ score │
│ ---  ┆ ---   │
│ str  ┆ i64   │
╞══════╪═══════╡
│ A    ┆ 20    │
│ B    ┆ 30    │
│ B    ┆ 40    │
└──────┴───────┘
```

**注意点・落とし穴**:
- デフォルト(`lazy=False`)では即座にファイルへ書き込みを実行して`None`を返す(`collect()`を呼ぶ必要はない)。`lazy=True`を指定すると書き込み自体を表す`LazyFrame`が返り、後で`.collect()`するまで実行されない。
- `write_parquet`/`write_csv`(Eager API)は結果を一度DataFrameとして完成させてから書き出すのに対し、`sink_*`はストリーミングエンジンでファイルI/Oまで含めてパイプライン化される点が異なる。

#### `pl.Config(...)` / `pl.thread_pool_size()`

**用途**: 表示フォーマット(表示行数・列数・文字列の省略幅など)や並列実行のスレッド数といった、polarsのグローバルな実行環境設定を調整する。

**シグネチャ**: `pl.Config(*, restore_defaults=False, apply_on_context_enter=False, **options) -> None`(`with pl.Config(...):`でコンテキストマネージャとしても使える)/ `pl.thread_pool_size() -> int`

**使用例**:
```python
print(pl.thread_pool_size())

df_cfg = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
with pl.Config(tbl_rows=2):
    print(df_cfg)
print(df_cfg)
```
実行結果:
```
16
shape: (5, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 1   │
│ …   │
│ 5   │
└─────┘
shape: (5, 1)
┌─────┐
│ a   │
│ --- │
│ i64 │
╞═════╡
│ 1   │
│ 2   │
│ 3   │
│ 4   │
│ 5   │
└─────┘
```

**注意点・落とし穴**:
- `pl.thread_pool_size()`の値(この検証環境では16)は実行マシンのCPUコア数などに依存し、環境によって変わる。スレッド数自体を変更したい場合は環境変数`POLARS_MAX_THREADS`をpolarsをインポートする前に設定する必要がある(インポート後の変更は反映されない)。
- `pl.Config`は`with`ブロックを抜けると設定が自動的に元に戻る(上の例で`tbl_rows=2`の効果が2回目の`print`には残っていない)。`with`を使わず`pl.Config.set_tbl_rows(2)`のようにクラスメソッドで呼ぶと、明示的に`pl.Config.restore_defaults()`するまで設定が残り続ける。

---

### struct/list dtype操作の応用

#### `Expr.list.eval(...)`

**用途**: リスト型の列の各要素(内側のリスト)に対して、`pl.element()`を起点とする式を適用する。`map_elements`のようにPython関数を呼ぶのではなく、polarsのネイティブ式で完結するため高速。

**シグネチャ**: `Expr.list.eval(self, expr, *, parallel=False) -> Expr`

**使用例**:
```python
df_le = pl.DataFrame({"scores": [[1, 5, 3], [9, 2, 8, 1]]})
print(df_le.with_columns(pl.col("scores").list.eval(pl.element() * 2).alias("doubled")))
```
実行結果:
```
shape: (2, 2)
┌─────────────┬──────────────┐
│ scores      ┆ doubled      │
│ ---         ┆ ---          │
│ list[i64]   ┆ list[i64]    │
╞═════════════╪══════════════╡
│ [1, 5, 3]   ┆ [2, 10, 6]   │
│ [9, 2, … 1] ┆ [18, 4, … 2] │
└─────────────┴──────────────┘
```

**注意点・落とし穴**:
- `pl.element()`はリストの「1つの要素」ではなく、リスト内の値全体を表す式の起点(`.rank()`や`.max()`のような集約も可能)。単純な四則演算だけなら`.list.eval()`を使わずとも後述の`list`名前空間の個別メソッド(`list.sort()`など)で足りることも多い。

#### `Expr.list.to_struct(...)` / `df.unnest(...)`

**用途**: 固定長のリスト列をフィールド名付きのstruct列に変換し(`list.to_struct`)、structを個別の列に展開する(`unnest`)。CSVなどでは表現しづらい「1セルに複数値」を通常の列構造に戻す定石。

**シグネチャ**: `Expr.list.to_struct(self, n_field_strategy=None, fields=None, upper_bound=None) -> Expr` / `DataFrame.unnest(self, columns=None, *more_columns, separator=None) -> DataFrame`

**使用例**:
```python
df_l2s = pl.DataFrame({"items": [["a", "b"], ["c", "d"]]})
out = df_l2s.with_columns(pl.col("items").list.to_struct(fields=["first", "second"]).alias("s"))
print(out)
print(out.unnest("s"))
```
実行結果:
```
shape: (2, 2)
┌────────────┬───────────┐
│ items      ┆ s         │
│ ---        ┆ ---       │
│ list[str]  ┆ struct[2] │
╞════════════╪═══════════╡
│ ["a", "b"] ┆ {"a","b"} │
│ ["c", "d"] ┆ {"c","d"} │
└────────────┴───────────┘
shape: (2, 3)
┌────────────┬───────┬────────┐
│ items      ┆ first ┆ second │
│ ---        ┆ ---   ┆ ---    │
│ list[str]  ┆ str   ┆ str    │
╞════════════╪═══════╪════════╡
│ ["a", "b"] ┆ a     ┆ b      │
│ ["c", "d"] ┆ c     ┆ d      │
└────────────┴───────┴────────┘
```

**注意点・落とし穴**:
- `fields`を省略する(`n_field_strategy`のデフォルト)と、先頭行のリスト長からフィールド数を推測し`field_0`, `field_1`, ...という名前が自動で振られる。リストの長さが行によって異なる場合、短い行の余ったフィールドは`null`になる。
- `unnest()`はstruct型の列を展開して元のDataFrameの列に混ぜ込む。struct列自体は`unnest`後には残らない(置き換えられる)。

#### `Expr.struct.rename_fields(...)`

**用途**: struct列内のフィールド名を変更する。`pl.struct()`で作った直後のフィールド名(元の列名がそのまま使われる)を、用途に応じて付け替えたいときに使う。

**シグネチャ**: `Expr.struct.rename_fields(self, names) -> Expr`

**使用例**:
```python
df_srf = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).with_columns(pl.struct(["a", "b"]).alias("s"))
print(df_srf.with_columns(pl.col("s").struct.rename_fields(["x", "y"]).alias("s2")).select("s2").unnest("s2"))
```
実行結果:
```
shape: (2, 2)
┌─────┬─────┐
│ x   ┆ y   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 3   │
│ 2   ┆ 4   │
└─────┴─────┘
```

**注意点・落とし穴**:
- `names`は位置で元のフィールドに対応する(名前で紐付けるわけではない)。元のフィールド数と`names`の長さが一致している必要がある。

#### `Expr.list.sort(...)` / `Expr.list.join(...)` / `Expr.list.contains(...)`

**用途**: リスト列そのものを扱う便利メソッド群。それぞれリスト内部の要素をソートする/文字列リストを区切り文字で1本の文字列に連結する/特定の値を含むか判定する。

**シグネチャ**: `Expr.list.sort(self, *, descending=False, nulls_last=False) -> Expr` / `Expr.list.join(self, separator, *, ignore_nulls=True) -> Expr` / `Expr.list.contains(self, item, *, nulls_equal=True) -> Expr`

**使用例**:
```python
df_lst = pl.DataFrame({"items": [[3, 1, 2], [9, 7]]})
print(df_lst.with_columns(
    pl.col("items").list.sort().alias("sorted"),
    pl.col("items").list.contains(7).alias("has7"),
))

df_words = pl.DataFrame({"words": [["a", "b", "c"], ["x", "y"]]})
print(df_words.with_columns(pl.col("words").list.join("-").alias("joined")))
```
実行結果:
```
shape: (2, 3)
┌───────────┬───────────┬───────┐
│ items     ┆ sorted    ┆ has7  │
│ ---       ┆ ---       ┆ ---   │
│ list[i64] ┆ list[i64] ┆ bool  │
╞═══════════╪═══════════╪═══════╡
│ [3, 1, 2] ┆ [1, 2, 3] ┆ false │
│ [9, 7]    ┆ [7, 9]    ┆ true  │
└───────────┴───────────┴───────┘
shape: (2, 2)
┌─────────────────┬────────┐
│ words           ┆ joined │
│ ---             ┆ ---    │
│ list[str]       ┆ str    │
╞═════════════════╪════════╡
│ ["a", "b", "c"] ┆ a-b-c  │
│ ["x", "y"]      ┆ x-y    │
└─────────────────┴────────┘
```

**注意点・落とし穴**:
- `list.join()`は文字列のリストにのみ使える(数値リストは事前に`.cast(pl.List(pl.String))`などで文字列化する必要がある)。区切り文字を列(式)で指定することもできる。

---

### 高度なウィンドウ・rolling

#### `Expr.rolling_mean_by(...)`

**用途**: 行数ベースではなく、時刻列の値そのもの(例: 過去3日間)を基準にした移動平均を計算する。`group_by_dynamic`が「時刻でバケット化して集計」するのに対し、こちらは各行ごとに「その時刻から遡った窓」の集計値を、元の行数のまま返す。

**シグネチャ**: `Expr.rolling_mean_by(self, by, window_size, *, min_samples=1, closed='right') -> Expr`

**使用例**:
```python
df_rmb = pl.DataFrame({
    "t": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 6), "1d", eager=True),
    "v": [1, 2, 3, 4, 5, 6],
})
print(df_rmb.with_columns(pl.col("v").rolling_mean_by("t", window_size="3d").alias("roll_mean_3d")))
```
実行結果:
```
shape: (6, 3)
┌─────────────────────┬─────┬──────────────┐
│ t                   ┆ v   ┆ roll_mean_3d │
│ ---                 ┆ --- ┆ ---          │
│ datetime[μs]        ┆ i64 ┆ f64          │
╞═════════════════════╪═════╪══════════════╡
│ 2024-01-01 00:00:00 ┆ 1   ┆ 1.0          │
│ 2024-01-02 00:00:00 ┆ 2   ┆ 1.5          │
│ 2024-01-03 00:00:00 ┆ 3   ┆ 2.0          │
│ 2024-01-04 00:00:00 ┆ 4   ┆ 3.0          │
│ 2024-01-05 00:00:00 ┆ 5   ┆ 4.0          │
│ 2024-01-06 00:00:00 ┆ 6   ┆ 5.0          │
└─────────────────────┴─────┴──────────────┘
```

**注意点・落とし穴**:
- `by`列は昇順にソート済みである必要がある(`group_by_dynamic`の`index_column`と同じ制約)。
- デフォルト`closed='right'`は「窓の右端(現在行の時刻)を含み、`window_size`だけ遡った範囲(左端は含まない)」を意味する。`group_by_dynamic`のデフォルトが`closed='left'`なのと非対称なので混同しやすい。

#### `Expr.rolling_map(...)`

**用途**: 移動窓に対して、`sum`/`mean`のような組み込み集計では表現できない任意のPython関数を適用する(`map_elements`の移動窓版)。

**シグネチャ**: `Expr.rolling_map(self, function, window_size, weights=None, *, min_samples=None, center=False) -> Expr`

**使用例**:
```python
df_rmap = pl.DataFrame({"v": [1, 2, 3, 4, 5]})
print(df_rmap.with_columns(
    pl.col("v").rolling_map(lambda s: s.max() - s.min(), window_size=3).alias("range3")
))
```
実行結果:
```
shape: (5, 2)
┌─────┬────────┐
│ v   ┆ range3 │
│ --- ┆ ---    │
│ i64 ┆ i64    │
╞═════╪════════╡
│ 1   ┆ null   │
│ 2   ┆ null   │
│ 3   ┆ 2      │
│ 4   ┆ 2      │
│ 5   ┆ 2      │
└─────┴────────┘
```

**注意点・落とし穴**:
- `function`は窓に含まれる値を`pl.Series`として受け取り、スカラーを返す必要がある。`map_elements`と同様にPythonコールバックを呼ぶため、`rolling_mean`などの組み込みメソッドで代替できないか先に検討すべき(遅い)。
- 窓が満たない先頭部分は`null`になる(`rolling_mean`などと同じ挙動)。

#### `Expr.cum_count(...)`

**用途**: 各行までの累積件数(1始まりの連番、`null`は数えない)を計算する。

**シグネチャ**: `Expr.cum_count(self, *, reverse=False) -> Expr`

**使用例**:
```python
df_cc = pl.DataFrame({"v": [10, 20, 30]})
print(df_cc.with_columns(pl.col("v").cum_count().alias("cum_n")))
```
実行結果:
```
shape: (3, 2)
┌─────┬───────┐
│ v   ┆ cum_n │
│ --- ┆ ---   │
│ i64 ┆ u32   │
╞═════╪═══════╡
│ 10  ┆ 1     │
│ 20  ┆ 2     │
│ 30  ┆ 3     │
└─────┴───────┘
```

**注意点・落とし穴**:
- `with_row_index()`が単なる連番の列を追加するのに対し、`cum_count()`は`null`値をスキップしてカウントする点が異なる(欠損を含む列で「有効値の累積個数」を数えたいときに使う)。

#### `Expr.over(order_by=..., mapping_strategy=...)`

**用途**: 基礎編で扱った`Expr.over()`をさらに応用し、`order_by`でグループ内の計算順序を明示したり、`mapping_strategy="join"`で集計結果を1行にまとめたリストとして返したりする。

**シグネチャ**: `Expr.over(self, partition_by=None, *more_exprs, order_by=None, descending=False, nulls_last=False, mapping_strategy='group_to_rows') -> Expr`

**使用例**:
```python
df_ov = pl.DataFrame({"team": ["A", "A", "B", "B"], "t": [3, 1, 2, 4], "score": [10, 20, 30, 40]})
print(df_ov.with_columns(pl.col("score").cum_sum().over("team", order_by="t").alias("cum_by_t")))
print(df_ov.with_columns(pl.col("score").over("team", mapping_strategy="join").alias("scores_in_team")))
```
実行結果:
```
shape: (4, 4)
┌──────┬─────┬───────┬──────────┐
│ team ┆ t   ┆ score ┆ cum_by_t │
│ ---  ┆ --- ┆ ---   ┆ ---      │
│ str  ┆ i64 ┆ i64   ┆ i64      │
╞══════╪═════╪═══════╪══════════╡
│ A    ┆ 3   ┆ 10    ┆ 30       │
│ A    ┆ 1   ┆ 20    ┆ 20       │
│ B    ┆ 2   ┆ 30    ┆ 30       │
│ B    ┆ 4   ┆ 40    ┆ 70       │
└──────┴─────┴───────┴──────────┘
shape: (4, 4)
┌──────┬─────┬───────┬────────────────┐
│ team ┆ t   ┆ score ┆ scores_in_team │
│ ---  ┆ --- ┆ ---   ┆ ---            │
│ str  ┆ i64 ┆ i64   ┆ list[i64]      │
╞══════╪═════╪═══════╪════════════════╡
│ A    ┆ 3   ┆ 10    ┆ [10, 20]       │
│ A    ┆ 1   ┆ 20    ┆ [10, 20]       │
│ B    ┆ 2   ┆ 30    ┆ [30, 40]       │
│ B    ┆ 4   ┆ 40    ┆ [30, 40]       │
└──────┴─────┴───────┴────────────────┘
```

**注意点・落とし穴**:
- `order_by`を指定しないと、`cum_sum`などの順序に依存する式は元のDataFrameの行順(この例では`t`列がバラバラな順)でグループ内計算されてしまう。時系列データで`over()`と累積系の式を組み合わせるときは`order_by`の指定を忘れないこと。
- `mapping_strategy`のデフォルト`'group_to_rows'`は各行にグループの集計値1つを割り当てるが、`'join'`はグループ内の全値を`list`型にまとめて各行に埋め込む(行ごとに「自分のグループの全メンバー」を持たせたい場合に使う)。

#### `df.rolling(...)`

**用途**: 時刻列を基準にした移動窓で`group_by`するための入口(`RollingGroupBy`)。`Expr.rolling_mean_by()`が式単体で完結するのに対し、`df.rolling()`は`group_by_dynamic`と同じく`.agg()`で複数の集計式をまとめて書ける。

**シグネチャ**: `DataFrame.rolling(self, index_column, *, period, offset=None, closed='right', group_by=None) -> RollingGroupBy`

**使用例**:
```python
df_roll2 = pl.DataFrame({
    "t": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 6), "1d", eager=True),
    "v": [1, 2, 3, 4, 5, 6],
})
print(df_roll2.rolling(index_column="t", period="2d").agg(pl.col("v").sum().alias("v_sum_2d")))
```
実行結果:
```
shape: (6, 2)
┌─────────────────────┬──────────┐
│ t                   ┆ v_sum_2d │
│ ---                 ┆ ---      │
│ datetime[μs]        ┆ i64      │
╞═════════════════════╪══════════╡
│ 2024-01-01 00:00:00 ┆ 1        │
│ 2024-01-02 00:00:00 ┆ 3        │
│ 2024-01-03 00:00:00 ┆ 5        │
│ 2024-01-04 00:00:00 ┆ 7        │
│ 2024-01-05 00:00:00 ┆ 9        │
│ 2024-01-06 00:00:00 ┆ 11       │
└─────────────────────┴──────────┘
```

**注意点・落とし穴**:
- `group_by_dynamic`が固定間隔(`every`)でウィンドウの起点を作るのに対し、`rolling()`は各行そのものを窓の右端(デフォルト`closed='right'`)として、そこから`period`分だけ遡る。そのため出力の行数は常に元のDataFrameと同じになる(`group_by_dynamic`は行数が変わりうる)。

---

### カスタムUDF(map_batches/map_elementsのパフォーマンス注意)

#### `Expr.map_batches(...)`

**用途**: Python関数を要素単位ではなく列(`Series`)単位で1回だけ呼び出す。`Series`全体の統計量(平均・標準偏差など)を使う処理や、NumPy/pandasなど他ライブラリのベクトル化関数をそのまま使いたい場合に向く。

**シグネチャ**: `Expr.map_batches(self, function, return_dtype=None, *, agg_list=False, is_elementwise=False, returns_scalar=False) -> Expr`

**使用例**:
```python
df_mb = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})

def zscore(s: pl.Series) -> pl.Series:
    return (s - s.mean()) / s.std()

print(df_mb.with_columns(pl.col("v").map_batches(zscore, is_elementwise=False).alias("z")))
```
実行結果:
```
shape: (4, 2)
┌─────┬───────────┐
│ v   ┆ z         │
│ --- ┆ ---       │
│ f64 ┆ f64       │
╞═════╪═══════════╡
│ 1.0 ┆ -1.161895 │
│ 2.0 ┆ -0.387298 │
│ 3.0 ┆ 0.387298  │
│ 4.0 ┆ 1.161895  │
└─────┴───────────┘
```

**注意点・落とし穴**:
- `function`が受け取るのはスカラーではなく`pl.Series`そのもの(この例のように`s.mean()`など`Series`全体に対する集計が使える)。`map_elements`と混同しないこと。
- `is_elementwise`(デフォルト`False`)は、公式ドキュメントいわく「入力に対して要素ごとに独立に(順不同のスライスに分割して実行しても結果が変わらない)処理である場合に`True`にすると最適化が効いて速くなる」引数。上の`zscore`例のように`Series`全体の統計量に依存する関数を誤って`is_elementwise=True`にすると、`group_by`後の集計などクエリオプティマイザがスライス分割・並列化する文脈で誤った結果になりうるため、要素ごとに完結する処理でない限り`False`のままにしておくのが安全。

#### パフォーマンス比較: `map_elements` vs `map_batches`

**用途**: 同じ「各値を2倍にする」処理を`map_elements`(要素ごとにPython関数を呼ぶ)と`map_batches`(列全体に対してベクトル化関数を1回呼ぶ)で実装し、実測で速度差を確認する。

**シグネチャ**: (ベンチマークのため両メソッドの再掲はしない。上記および基礎編の`Expr.map_elements(...)`を参照)

**使用例**:
```python
import time

n = 2_000_000
df_bench = pl.DataFrame({"a": range(n)})

t0 = time.perf_counter()
df_bench.with_columns(pl.col("a").map_elements(lambda x: x * 2, return_dtype=pl.Int64).alias("r"))
t_elem = time.perf_counter() - t0

t0 = time.perf_counter()
df_bench.with_columns(pl.col("a").map_batches(lambda s: s * 2, is_elementwise=True).alias("r"))
t_batch = time.perf_counter() - t0

print(f"map_elements: {t_elem:.3f}s")
print(f"map_batches:  {t_batch:.3f}s")
```
実行結果:
```
map_elements: 0.266s
map_batches:  0.050s
```

**注意点・落とし穴**:
- `map_elements`を実行すると`PolarsInefficientMapWarning`(「`Expr.map_elements`はネイティブ式APIより著しく遅い」という趣旨)が出る(実行確認済み。基礎編の該当箇所と同じ警告)。この検証(200万行、単純な`x * 2`)では`map_batches`が約5倍高速だった。ただし実際の速度差は処理内容・データ量に依存するため、この数値自体を一般的な倍率として鵜呑みにしないこと。
- 最も速いのは`map_elements`/`map_batches`のどちらでもなく、可能な限り`pl.col("a") * 2`のようなネイティブ式で書くこと。両者はいずれも「ネイティブ式で書けない場合の最終手段」という位置づけ。
- `map_elements`の`strategy`引数(デフォルト`'thread_local'`)を`'threading'`にすると別スレッドで並列実行できるが、公式ドキュメントに「Pythonの関数がGILを解放する(C拡張関数呼び出しなど)場合以外は効果が薄く、パフォーマンスが悪化しうる実験的機能」と明記されている(`help(pl.Expr.map_elements)`で確認済み)。

---

### SQLコンテキスト(pl.SQLContext)

#### `pl.SQLContext(...)` / `SQLContext.execute(...)`

**用途**: DataFrame/LazyFrameにSQLのテーブル名を割り当て、SQL文で問い合わせる。polarsの式APIに不慣れなメンバーとの共有や、既存のSQLロジックの移植に向く。

**シグネチャ**: `pl.SQLContext(frames=None, *, register_globals=False, eager=False, **named_frames) -> None`(コンストラクタ)/ `SQLContext.execute(self, query, *, eager=None) -> LazyFrame | DataFrame`

**使用例**:
```python
df_sql = pl.DataFrame({"team": ["A", "A", "B", "B"], "score": [10, 20, 30, 40]})
ctx = pl.SQLContext(players=df_sql, eager=True)
result = ctx.execute("SELECT team, SUM(score) AS total FROM players GROUP BY team ORDER BY team")
print(result)
```
実行結果:
```
shape: (2, 2)
┌──────┬───────┐
│ team ┆ total │
│ ---  ┆ ---   │
│ str  ┆ i64   │
╞══════╪═══════╡
│ A    ┆ 30    │
│ B    ┆ 70    │
└──────┴───────┘
```

**注意点・落とし穴**:
- コンストラクタの`eager=True`(またはコンストラクタでは指定せず`execute(eager=True)`)を指定しないと、`execute()`はDataFrameではなく未評価の`LazyFrame`を返す。標準ではLazyに倒す設計になっている。
- サポートされるSQL構文はpolarsが実装している範囲に限られ、DuckDBやPostgreSQLなど汎用RDBMSのSQL方言と完全互換ではない。

#### `with pl.SQLContext(...) as ctx:` / `ctx.tables()`

**用途**: 複数のDataFrameを一括登録してSQLのJOINを行い、登録済みテーブル名の一覧を確認する。`with`文で使うとブロックを抜けた際にコンテキストが後始末される。

**シグネチャ**: `SQLContext.tables(self) -> list[str]`

**使用例**:
```python
df_t1 = pl.DataFrame({"id": [1, 2], "name": ["x", "y"]})
df_t2 = pl.DataFrame({"id": [1, 2], "val": [10, 20]})
with pl.SQLContext(t1=df_t1, t2=df_t2, eager=True) as ctx:
    result = ctx.execute("SELECT t1.name, t2.val FROM t1 JOIN t2 ON t1.id = t2.id")
    print(result)
    print(ctx.tables())
```
実行結果:
```
shape: (2, 2)
┌──────┬─────┐
│ name ┆ val │
│ ---  ┆ --- │
│ str  ┆ i64 │
╞══════╪═════╡
│ x    ┆ 10  │
│ y    ┆ 20  │
└──────┴─────┘
['t1', 't2']
```

**注意点・落とし穴**:
- 複数フレームはコンストラクタのキーワード引数(`t1=df_t1, t2=df_t2, ...`)としてまとめて登録でき、そのキーワード名がSQL上のテーブル名になる。個別に追加したい場合は`ctx.register(name, frame)`を使う。

---

### 高度な結合(join_asof・join_where・semi/anti join)

#### `df.join_asof(...)`

**用途**: 完全一致ではなく「直近の値」で結合する(例: 取引時刻に対して直近の気配値を紐付ける、時系列データ特有の結合)。SQLの`ASOF JOIN`に相当。

**シグネチャ**: `df.join_asof(other, *, left_on=None, right_on=None, on=None, by_left=None, by_right=None, by=None, strategy='backward', suffix='_right', tolerance=None, allow_parallel=True, force_parallel=False, coalesce=True, allow_exact_matches=True, check_sortedness=True) -> DataFrame`

**使用例**:
```python
trades = pl.DataFrame({"t": [1, 3, 6, 9], "price": [100, 101, 103, 105]}).sort("t")
quotes = pl.DataFrame({"t": [0, 2, 4, 8], "quote": ["q0", "q2", "q4", "q8"]}).sort("t")
print(trades.join_asof(quotes, on="t", strategy="backward"))
```
実行結果:
```
shape: (4, 3)
┌─────┬───────┬───────┐
│ t   ┆ price ┆ quote │
│ --- ┆ ---   ┆ ---   │
│ i64 ┆ i64   ┆ str   │
╞═════╪═══════╪═══════╡
│ 1   ┆ 100   ┆ q0    │
│ 3   ┆ 101   ┆ q2    │
│ 6   ┆ 103   ┆ q4    │
│ 9   ┆ 105   ┆ q8    │
└─────┴───────┴───────┘
```

**注意点・落とし穴**:
- 両方のDataFrameが結合キー(`on`)で事前にソートされている必要がある(`check_sortedness=True`がデフォルトで、未ソートだと例外になる)。
- `strategy='backward'`(デフォルト)は「自分以下で最も近い」値に結合する。`'forward'`(自分以上で最も近い)、`'nearest'`(前後どちらでも最も近い)も選べる。
- `by`引数でグループごとの`asof`結合(例: 銘柄ごとに直近の気配値を引く)もできる。

#### `df.join_where(...)`

**用途**: 等価結合(`on`列が一致)ではなく、任意の不等号を含む条件式(例: 値が範囲内に収まるか)で結合する。SQLの`JOIN ... ON`に任意条件を書けるのと同じ発想(いわゆるtheta結合)。

**シグネチャ**: `df.join_where(other, *predicates, how='inner', suffix='_right') -> DataFrame`

**使用例**:
```python
df_left = pl.DataFrame({"id": [1, 2, 3], "lo": [0, 10, 20], "hi": [9, 19, 29]})
df_right = pl.DataFrame({"id2": [1, 2], "val": [5, 15]})
print(df_left.join_where(df_right, pl.col("val") >= pl.col("lo"), pl.col("val") <= pl.col("hi")))
```
実行結果:
```
shape: (2, 5)
┌─────┬─────┬─────┬─────┬─────┐
│ id  ┆ lo  ┆ hi  ┆ id2 ┆ val │
│ --- ┆ --- ┆ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╪═════╪═════╡
│ 2   ┆ 10  ┆ 19  ┆ 2   ┆ 15  │
│ 1   ┆ 0   ┆ 9   ┆ 1   ┆ 5   │
└─────┴─────┴─────┴─────┴─────┘
```

**注意点・落とし穴**:
- `on`ではなく複数の述語(`Expr`)を可変長引数で渡す(通常の`join`とは呼び出し方が異なる)。等価条件しか使わないなら通常の`join`の方が最適化が効きやすく高速。
- `how`に指定できるのは`'inner'`/`'left'`/`'right'`のみで、通常の`join`にある`'full'`/`'semi'`/`'anti'`は`join_where`では使えない(型定義`JoinWhereStrategy`で確認済み)。

#### `df.join(..., how="semi")` / `df.join(..., how="anti")`

**用途**: 相手側の値を列として持ち込まず、「相手に一致する行だけを残す(semi)」「相手に一致しない行だけを残す(anti)」というフィルタ的な結合を行う。SQLの`WHERE EXISTS`/`WHERE NOT EXISTS`に相当。

**シグネチャ**: `df.join(other, on=None, how='inner', ...)`(`how`に`'semi'`/`'anti'`を指定。基礎編の`df.join(...)`と同一メソッド)

**使用例**:
```python
df_left2 = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
df_right2 = pl.DataFrame({"id": [2, 3, 4]})
print(df_left2.join(df_right2, on="id", how="semi"))
print(df_left2.join(df_right2, on="id", how="anti"))
```
実行結果:
```
shape: (2, 2)
┌─────┬──────┐
│ id  ┆ name │
│ --- ┆ ---  │
│ i64 ┆ str  │
╞═════╪══════╡
│ 2   ┆ b    │
│ 3   ┆ c    │
└─────┴──────┘
shape: (1, 2)
┌─────┬──────┐
│ id  ┆ name │
│ --- ┆ ---  │
│ i64 ┆ str  │
╞═════╪══════╡
│ 1   ┆ a    │
└─────┴──────┘
```

**注意点・落とし穴**:
- `how='inner'`と`on`列だけを指定してから不要列を`drop`するのと違い、`semi`/`anti`は右側のDataFrameの列を一切結果に含めない(行のフィルタとしてのみ機能する)。右側に重複キーがあっても行が増殖しない点も通常の`inner`結合と異なる。

#### `df.join(..., validate=...)`

**用途**: 結合キーの重複関係(1:1・1:多・多:1・多:多)を事前に宣言し、想定と異なる場合に例外を出して早期に気付けるようにする。

**シグネチャ**: `df.join(other, on=None, how='inner', *, ..., validate='m:m', ...)`(`validate`は`'1:1'`/`'1:m'`/`'m:1'`/`'m:m'`のいずれか。デフォルトは検証なしの`'m:m'`)

**使用例**:
```python
df_left3 = pl.DataFrame({"id": [1, 1, 2], "name": ["a", "a2", "b"]})
df_right3 = pl.DataFrame({"id": [1, 2], "val": [100, 200]})
try:
    df_left3.join(df_right3, on="id", validate="1:1")
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
ComputeError join keys did not fulfill 1:1 validation
```

**注意点・落とし穴**:
- `validate`はキーの一意性を事前に検査するコストがかかる分、意図しない多重結合(結合後に行数が想定外に増えるバグ)を早期発見できる。デフォルトの`'m:m'`は検証を行わない(=何でも許容する)ので、想定される関係が分かっている場合は明示的に指定した方が安全。
