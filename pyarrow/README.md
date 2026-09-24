# PyArrow 逆引き辞書

PyArrow 25.0.1 で検証済み

本ドキュメントに掲載しているシグネチャ・実行結果は、すべて `/home/manaty/library-practicing/.venv/bin/python`(PyArrow 25.0.1)で実際にコードを実行して得たものです。連携例では pandas 3.0.5 / NumPy 2.4.6 / polars 1.44.1 を使っています。`csv.ReadOptions` など Cython 実装で `inspect.signature` が使えないクラスは、docstring の先頭にあるシグネチャ行を載せ、その旨を明記しています。

PyArrow は Arrow 列指向メモリフォーマットの Python バインディングで、pandas と polars の間(および Parquet / IPC / CSV などのファイル形式との間)を橋渡しする層です。押さえておくべき基本構造は次の通りです。

- **`Array`(1列分の連続メモリ)/ `ChunkedArray`(複数の `Array` をつないだ1列)/ `Table`(`ChunkedArray` を列に持つ表)/ `RecordBatch`(チャンク1つ分の表)** が基本の型。`Table` の各列は `ChunkedArray`(`t["a"]` は `Array` ではない)。
- データは**イミュータブル**。`Table` の列操作(`append_column` など)は新しい `Table` を返し、元は変わらない。
- 欠損は **null(有効性ビットマップ)** で表す。NaN とは別物で、`pa.array(np.array([1.0, np.nan]))` の NaN は null にならない。
- `pyarrow.compute`(`pc`)がベクトル化された計算関数群を提供し、`pyarrow.dataset`・Parquet の `filters=`・`Table.filter` は共通の式(`pc.field(...)` / `ds.field(...)`)を使う。

`pyarrow.dataset` / `parquet` / `compute` / `csv` / `feather` / `ipc` / `json` / `orc` / `fs` / `acero` / `flight` はすべて import できることを確認済み(本書では `orc` と `flight` は扱わない)。

各カテゴリの使用例は、そのカテゴリを上から順に実行した状態(`import` や変数 `t`・`dataset` など)を前提にしているものがあります。ファイルの書き出し例は `/tmp/pa_*` に書く形で記載していますが、実際の検証は一時ディレクトリで実行し、終了後に削除しています。

**25.0.1 で特に注意すべき点(本文の該当エントリに詳細)**

- `pc.sort_indices` の `null_placement` 引数は非推奨で `FutureWarning` が出る。列ごとの `sort_keys=[("x", "ascending", "at_start")]` 形式を使う。
- `Table.group_by(...).aggregate(...)` の結果ではキー列が**先頭**に来る。`first` / `last` は `use_threads=False` が必要。
- pandas 3.0 の `str` 列は Arrow では `large_string` になる。polars の `DataFrame` を `pa.table(df)`(PyCapsule 経由)で受け取ると文字列列は `string_view` になり、`pc.utf8_*` が使えない。
- `pq.read_table` でパーティション付きディレクトリを読むとパーティション列は `dictionary` 型になるが、`ds.dataset(..., partitioning="hive")` では推論された整数型になる。

## 目次

1. [型システム](#型システム)
    - [`pa.int64()` / `pa.string()` ほか基本型ファクトリ](#paint64--pastring-ほか基本型ファクトリ)
    - [`pa.timestamp(unit, tz=None)` / `pa.date32()` / `pa.duration(unit)`](#patimestampunit-tznone--padate32--padurationunit)
    - [`pa.decimal128(precision, scale)`](#padecimal128precision-scale)
    - [`pa.list_(value_type)` / `pa.large_list` / 固定長リスト](#palist_value_type--palarge_list--固定長リスト)
    - [`pa.struct(fields)`](#pastructfields)
    - [`pa.dictionary(index_type, value_type)`](#padictionaryindex_type-value_type)
    - [`pa.field(name, type)` / `pa.schema(fields)`](#pafieldname-type--paschemafields)
    - [`pa.unify_schemas(schemas)`](#paunify_schemasschemas)
2. [配列・テーブル](#配列テーブル)
    - [`pa.array(obj, type=None, ...)`](#paarrayobj-typenone-)
    - [`pa.scalar(value, type=None)` / `Array[i]` / `.as_py()`](#pascalarvalue-typenone--arrayi--as_py)
    - [`pa.chunked_array(arrays)` / `ChunkedArray`](#pachunked_arrayarrays--chunkedarray)
    - [`pa.table(data)` / `Table` の基本属性](#patabledata--table-の基本属性)
    - [`Table.from_pydict` / `Table.from_pylist` / `Table.from_arrays`](#tablefrom_pydict--tablefrom_pylist--tablefrom_arrays)
    - [`Table.to_pydict()` / `Table.to_pylist()`](#tableto_pydict--tableto_pylist)
    - [列の追加・削除・選択(`select` / `drop_columns` / `rename_columns` / `append_column` / `set_column`)](#列の追加削除選択select--drop_columns--rename_columns--append_column--set_column)
    - [行の選択(`Table.slice` / `Table.take` / `Table.filter` / `Table.sort_by`)](#行の選択tableslice--tabletake--tablefilter--tablesort_by)
    - [`pa.concat_tables(tables, promote_options=...)`](#paconcat_tablestables-promote_options)
    - [`Table.cast(target_schema)` / `Array.cast(target_type)`](#tablecasttarget_schema--arraycasttarget_type)
    - [`pa.record_batch` / `Table.to_batches` / `RecordBatchReader`](#parecord_batch--tableto_batches--recordbatchreader)
3. [計算(pyarrow.compute)](#計算pyarrowcompute)
    - [`pc.sum` / `pc.mean` / `pc.min_max` / `pc.count`(集約関数)](#pcsum--pcmean--pcmin_max--pccount集約関数)
    - [比較・論理関数と `pc.filter`(`pc.greater` / `pc.equal` / `pc.and_` / `pc.invert`)](#比較論理関数と-pcfilterpcgreater--pcequal--pcand_--pcinvert)
    - [`pc.take(data, indices)` / `pc.array_sort_indices` / `pc.sort_indices`](#pctakedata-indices--pcarray_sort_indices--pcsort_indices)
    - [`pc.is_in` / `pc.index_in`](#pcis_in--pcindex_in)
    - [`pc.if_else` / `pc.case_when` / `pc.fill_null` / `pc.coalesce`](#pcif_else--pccase_when--pcfill_null--pccoalesce)
    - [`pc.value_counts` / `pc.unique` / `pc.count_distinct` / `pc.mode`](#pcvalue_counts--pcunique--pccount_distinct--pcmode)
    - [文字列関数(`pc.utf8_upper` / `pc.match_substring` / `pc.replace_substring` / `pc.split_pattern` / `pc.utf8_length` ほか)](#文字列関数pcutf8_upper--pcmatch_substring--pcreplace_substring--pcsplit_pattern--pcutf8_length-ほか)
    - [日時関数(`pc.strptime` / `pc.strftime` / `pc.year` / `pc.month` / `pc.floor_temporal` ほか)](#日時関数pcstrptime--pcstrftime--pcyear--pcmonth--pcfloor_temporal-ほか)
    - [算術・丸め・累積(`pc.add` / `pc.multiply` / `pc.divide` / `pc.round` / `pc.cumulative_sum` / `pc.quantile`)](#算術丸め累積pcadd--pcmultiply--pcdivide--pcround--pccumulative_sum--pcquantile)
    - [list / struct 関数(`pc.list_flatten` / `pc.list_value_length` / `pc.list_element` / `pc.struct_field`)](#list--struct-関数pclist_flatten--pclist_value_length--pclist_element--pcstruct_field)
    - [`Table.group_by(keys).aggregate(aggs)`](#tablegroup_bykeysaggregateaggs)
    - [`Table.join(right_table, keys, ...)`](#tablejoinright_table-keys-)
    - [`pc.field` による式(Expression)](#pcfield-による式expression)
4. [Parquet入出力](#parquet入出力)
    - [`pq.write_table` / `pq.read_table`](#pqwrite_table--pqread_table)
    - [`pq.read_table(columns=..., filters=...)`(列選択・行フィルタ)](#pqread_tablecolumns-filters列選択行フィルタ)
    - [`pq.write_table` の主要オプション(`compression` / `row_group_size` / `use_dictionary` / `write_statistics`)](#pqwrite_table-の主要オプションcompression--row_group_size--use_dictionary--write_statistics)
    - [`pq.read_schema` / `pq.read_metadata` / `ParquetFile.metadata`](#pqread_schema--pqread_metadata--parquetfilemetadata)
    - [`pq.ParquetFile`(行グループ単位・バッチ単位の読み込み)](#pqparquetfile行グループ単位バッチ単位の読み込み)
    - [`pq.ParquetWriter`(追記書き込み・ストリーミング書き出し)](#pqparquetwriter追記書き込みストリーミング書き出し)
    - [`pq.write_to_dataset` / ディレクトリ読み込み(パーティション付き Parquet)](#pqwrite_to_dataset--ディレクトリ読み込みパーティション付き-parquet)
    - [スキーマメタデータの保存・pandas メタデータ(`replace_schema_metadata`)](#スキーマメタデータの保存pandas-メタデータreplace_schema_metadata)
5. [CSV・JSON・Feather・IPC](#csvjsonfeatheripc)
    - [`csv.read_csv` / `csv.write_csv`](#csvread_csv--csvwrite_csv)
    - [`csv.ReadOptions` / `csv.ParseOptions` / `csv.ConvertOptions`](#csvreadoptions--csvparseoptions--csvconvertoptions)
    - [`csv.open_csv`(ストリーミング読み込み)](#csvopen_csvストリーミング読み込み)
    - [`json.read_json`(改行区切り JSON)](#jsonread_json改行区切り-json)
    - [`feather.write_feather` / `feather.read_table`](#featherwrite_feather--featherread_table)
    - [`ipc.new_file` / `ipc.open_file`(Arrow IPC ファイル形式)](#ipcnew_file--ipcopen_filearrow-ipc-ファイル形式)
    - [`ipc.new_stream` / `ipc.open_stream`(Arrow IPC ストリーム形式)/ `Table` のシリアライズ](#ipcnew_stream--ipcopen_streamarrow-ipc-ストリーム形式-table-のシリアライズ)
6. [Dataset API(pyarrow.dataset)](#dataset-apipyarrowdataset)
    - [`ds.write_dataset`(パーティション付きで書き出し)](#dswrite_datasetパーティション付きで書き出し)
    - [`ds.dataset(source, format=..., partitioning=...)`(Dataset の作成と基本情報)](#dsdatasetsource-format-partitioningdataset-の作成と基本情報)
    - [`Dataset.to_table(columns=..., filter=...)`(列選択・フィルタ pushdown)](#datasetto_tablecolumns-filter列選択フィルタ-pushdown)
    - [`Dataset.scanner` / `Scanner.to_batches` / `Dataset.head` / `Dataset.take`](#datasetscanner--scannerto_batches--datasethead--datasettake)
    - [`ds.partitioning`(パーティション方式の指定)](#dspartitioningパーティション方式の指定)
    - [式による射影(`columns` に辞書を渡して列を計算)](#式による射影columns-に辞書を渡して列を計算)
    - [`write_dataset` の制御(`max_rows_per_file` / `existing_data_behavior` / `file_visitor`)](#write_dataset-の制御max_rows_per_file--existing_data_behavior--file_visitor)
    - [`Dataset.get_fragments` / `Fragment`(ファイル単位の操作)](#datasetget_fragments--fragmentファイル単位の操作)
    - [複数ファイル・インメモリデータから Dataset を作る(`ds.dataset` のリスト入力・`ds.InMemoryDataset`)](#複数ファイルインメモリデータから-dataset-を作るdsdataset-のリスト入力dsinmemorydataset)
7. [pandas・NumPy・polars 連携](#pandasnumpypolars-連携)
    - [`pa.Table.from_pandas` / `Table.to_pandas`](#patablefrom_pandas--tableto_pandas)
    - [`types_mapper=pd.ArrowDtype` / `pd.ArrowDtype`(Arrow 型を保ったまま pandas 化)](#types_mapperpdarrowdtype--pdarrowdtypearrow-型を保ったまま-pandas-化)
    - [日付・時刻の変換(`date_as_object` / `coerce_temporal_nanoseconds`)](#日付時刻の変換date_as_object--coerce_temporal_nanoseconds)
    - [`Array.to_numpy` / `pa.array(ndarray)`(NumPy とのゼロコピー)](#arrayto_numpy--paarrayndarraynumpy-とのゼロコピー)
    - [polars との相互変換(`pl.from_arrow` / `DataFrame.to_arrow`)](#polars-との相互変換plfrom_arrow--dataframeto_arrow)
    - [Arrow PyCapsule インターフェース(`__arrow_c_stream__`)による受け渡し](#arrow-pycapsule-インターフェース__arrow_c_stream__による受け渡し)
8. [メモリ管理・バッファ](#メモリ管理バッファ)
    - [`Array.nbytes` / `get_total_buffer_size()` / `buffers()`](#arraynbytes--get_total_buffer_size--buffers)
    - [`pa.default_memory_pool()` / `pa.total_allocated_bytes()`](#padefault_memory_pool--patotal_allocated_bytes)
    - [`pa.py_buffer` / `pa.allocate_buffer` / `Buffer`](#papy_buffer--paallocate_buffer--buffer)
    - [`pa.BufferOutputStream` / `pa.BufferReader`(メモリ上の入出力ストリーム)](#pabufferoutputstream--pabufferreaderメモリ上の入出力ストリーム)
    - [`pa.memory_map` / `memory_map=True`(メモリマップによるゼロコピー読み込み)](#pamemory_map--memory_maptrueメモリマップによるゼロコピー読み込み)
9. [ファイルシステム・UDF](#ファイルシステムudf)
    - [`pyarrow.fs`(`LocalFileSystem` / `FileSelector` / `FileSystem.from_uri`)](#pyarrowfslocalfilesystem--fileselector--filesystemfrom_uri)
    - [`pc.register_scalar_function`(Python 関数を計算カーネルとして登録)](#pcregister_scalar_functionpython-関数を計算カーネルとして登録)

---

## 型システム

### `pa.int64()` / `pa.string()` ほか基本型ファクトリ

**用途**: Arrow のデータ型(`DataType`)オブジェクトを生成する。整数・浮動小数・文字列・真偽値・日付などは、すべて `pa.<型名>()` という引数なしの関数で作る。

**シグネチャ**: `pa.int64()` / `pa.float64()` / `pa.string()` / `pa.large_string()` / `pa.bool_()` / `pa.binary()` / `pa.date32()` / `pa.null()` など(いずれも引数なし。例: `pa.int64()` のシグネチャは `()`)

**使用例**:
```python
import pyarrow as pa

for t in [pa.int8(), pa.int64(), pa.uint32(), pa.float32(), pa.float64(), pa.bool_(),
          pa.string(), pa.large_string(), pa.binary(), pa.date32(), pa.null()]:
    print(t)
print(pa.int64().bit_width, pa.float32().bit_width)
print(pa.int64() == pa.int64(), pa.int64() == pa.int32())
print(pa.utf8() == pa.string())
```
実行結果:
```
int8
int64
uint32
float
double
bool
string
large_string
binary
date32[day]
null
64 32
True False
True
```

**注意点・落とし穴**:
- `pa.float32()` の文字列表現は `float`、`pa.float64()` は `double` と表示される(Arrow 仕様上の名前)。`pa.float32()` と `pa.float16()` を取り違えないこと。
- `pa.string()`(= `pa.utf8()`)は32bitオフセット、`pa.large_string()` は64bitオフセットで、別の型として扱われる(相互変換は `cast` が必要)。1つの列の文字列の合計が2GiBを超える可能性があるときは `large_string` を使う。

### `pa.timestamp(unit, tz=None)` / `pa.date32()` / `pa.duration(unit)`

**用途**: 日時系の型を作る。`timestamp` は精度(`'s'`/`'ms'`/`'us'`/`'ns'`)とタイムゾーンを型に持つ。

**シグネチャ**: `pa.timestamp(unit, tz=None)` / `pa.duration(unit)` / `pa.date32()` / `pa.time64(unit)`

**使用例**:
```python
import datetime as dt

t_us = pa.timestamp("us")
t_tokyo = pa.timestamp("ms", tz="Asia/Tokyo")
print(t_us, "|", t_tokyo)
print(t_tokyo.unit, t_tokyo.tz)

arr = pa.array([dt.datetime(2024, 1, 1, 9, 0), None], type=pa.timestamp("s"))
print(arr)
print(arr.type)
print(pa.array([dt.timedelta(seconds=90)], type=pa.duration("s")))
```
実行結果:
```
timestamp[us] | timestamp[ms, tz=Asia/Tokyo]
ms Asia/Tokyo
[
  2024-01-01 09:00:00,
  null
]
timestamp[s]
[
  90
]
```

**注意点・落とし穴**:
- Python の `datetime` から型を推論すると単位は `us`(マイクロ秒)になる(`pa.array([datetime(2024,1,1)]).type` → `timestamp[us]`)。pandas 3.0 の `pd.to_datetime` の結果を `Table.from_pandas` で変換した場合も `timestamp[us]` だった。一方、`datetime64[ns]` の NumPy 配列や Series を `pa.array` に渡すと `timestamp[ns]` になる(確認済み)ので、元の単位がそのまま引き継がれる。
- `tz` を持つ型と持たない型は別の型で、`pa.concat_tables` などで混在させると型不一致エラーになる。

### `pa.decimal128(precision, scale)`

**用途**: 固定小数点数の型。金額など、浮動小数の丸め誤差を避けたい数値に使う。Python の `decimal.Decimal` と相互変換できる。

**シグネチャ**: `pa.decimal128(precision, scale=0)`

**使用例**:
```python
from decimal import Decimal

d = pa.decimal128(10, 2)
print(d, d.precision, d.scale)
arr = pa.array([Decimal("1.10"), Decimal("2.25"), None], type=d)
print(arr)
print(arr.to_pylist())
try:
    pa.array([Decimal("123456789.123")], type=d)
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
decimal128(10, 2) 10 2
[
  1.10,
  2.25,
  null
]
[Decimal('1.10'), Decimal('2.25'), None]
ArrowInvalid Rescaling Decimal value would cause data loss
```

**注意点・落とし穴**:
- 指定した scale より細かい桁を持つ値や、precision を超える桁数の値を入れるとエラーになる(上の例では `ArrowInvalid`)。暗黙の丸めは行われない。

### `pa.list_(value_type)` / `pa.large_list` / 固定長リスト

**用途**: 可変長リスト型(列の各要素がリスト)を作る。`list_size` を指定すると固定長リスト(`fixed_size_list`)になる。

**シグネチャ**: `pa.list_(value_type, list_size=-1)`

**使用例**:
```python
lt = pa.list_(pa.int64())
print(lt, lt.value_type)
arr = pa.array([[1, 2], [], None, [3]], type=lt)
print(arr)
print(arr.to_pylist())

fixed = pa.list_(pa.float32(), 3)
print(fixed)
print(pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type=fixed))
print(pa.large_list(pa.int32()))      # 64bitオフセット版
print(pa.array([[1, 2], [3]]).type)   # 型推論
```
実行結果:
```
list<item: int64> int64
[
  [
    1,
    2
  ],
  [],
  null,
  [
    3
  ]
]
[[1, 2], [], None, [3]]
fixed_size_list<item: float>[3]
[
  [
    1,
    2,
    3
  ],
  [
    4,
    5,
    6
  ]
]
large_list<item: int32>
list<item: int64>
```

**注意点・落とし穴**:
- 空リスト `[]` と null(`None`)は区別される(上の例の3要素目が null)。
- 固定長リストは要素数が合わない値を入れるとエラーになる(`pa.array([[1.0, 2.0]], type=pa.list_(pa.float32(), 3))` → `ArrowInvalid: Length of item not correct: expected 3 but got array of size 2`)。機械学習の埋め込みベクトルなど、長さが一定のデータに向く。

### `pa.struct(fields)`

**用途**: 複数のフィールドを持つ構造体型を作る。JSON のネストしたオブジェクトや、辞書のリストからの変換で登場する。

**シグネチャ**: `pa.struct(fields)`(`fields` は `pa.field` のリスト、または `(名前, 型)` のリスト・辞書)

**使用例**:
```python
st = pa.struct([("name", pa.string()), ("age", pa.int64())])
print(st)
arr = pa.array([{"name": "Alice", "age": 30}, {"name": "Bob", "age": None}, None], type=st)
print(arr.type)
print(arr.field("name"))
print(arr.flatten()[0])
print(arr.to_pylist())
print(pa.array([{"x": 1, "y": "a"}]).type)   # 型推論
```
実行結果:
```
struct<name: string, age: int64>
struct<name: string, age: int64>
[
  "Alice",
  "Bob",
  ""
]
[
  "Alice",
  "Bob",
  null
]
[{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': None}, None]
struct<x: int64, y: string>
```

**注意点・落とし穴**:
- 構造体自体が null の行と、フィールドだけが null の行は別物(上の例で `None` の行と `age=None` の行の `to_pylist()` 結果が異なる)。
- `arr.field("name")` は親(struct)の null を反映せず、子配列をそのまま返す(上の例では null 行が `""` と表示される)。親の null を子に伝播させたい場合は `arr.flatten()` を使う(上の例で `flatten()[0]` は null 行が `null` になる)。

### `pa.dictionary(index_type, value_type)`

**用途**: 辞書エンコード(カテゴリ型)。値の種類が少ない列を「辞書(ユニーク値)+整数インデックス」で表し、メモリと比較コストを削減する。pandas の `category` に相当する。

**シグネチャ**: `pa.dictionary(index_type, value_type, ordered=False)`

**使用例**:
```python
dt_ = pa.dictionary(pa.int8(), pa.string())
print(dt_)

arr = pa.array(["tokyo", "osaka", "tokyo", None, "tokyo"]).dictionary_encode()
print(arr.type)
print(arr.indices)
print(arr.dictionary)
print(arr.to_pylist())
print(arr.cast(pa.string()).type)
print(pa.array(["a", None, "a"]).dictionary_encode("encode").dictionary)   # null を辞書側に入れる
```
実行結果:
```
dictionary<values=string, indices=int8, ordered=0>
dictionary<values=string, indices=int32, ordered=0>
[
  0,
  1,
  0,
  null,
  0
]
[
  "tokyo",
  "osaka"
]
['tokyo', 'osaka', 'tokyo', None, 'tokyo']
string
[
  "a",
  null
]
```

**注意点・落とし穴**:
- `Array.dictionary_encode()` は index 型を自動で `int32` にする。null は既定(`null_encoding='mask'`)ではインデックス側の null として保持され、辞書側には入らない。`null_encoding='encode'` を指定すると null が辞書の値の1つとして入る(上の最終行の出力)。
- 辞書エンコードされた列と通常の `string` 列は型が異なるため、`concat_tables` 時は型を揃える必要がある。

### `pa.field(name, type)` / `pa.schema(fields)`

**用途**: 列名・型・null許容・メタデータを持つスキーマを定義する。`Table.from_pydict` や `pq.ParquetWriter`、`ds.dataset` などで型を明示するときの共通部品。

**シグネチャ**: `pa.field(name, type=None, nullable=None, metadata=None)` / `pa.schema(fields, metadata=None)`

**使用例**:
```python
schema = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("name", pa.string()),
    ("score", pa.float64()),            # (名前, 型) のタプルでも可
], metadata={"source": "demo"})
print(schema)
print(schema.names, schema.types)
print(schema.field("id").nullable, schema.field("name").nullable)
print(schema.get_field_index("score"))
print(schema.metadata)

schema2 = schema.append(pa.field("flag", pa.bool_()))
print(schema2.names)
print(schema.equals(schema2), schema.remove(0).names)

t = pa.table({"id": [1, 2], "name": ["a", "b"], "score": [0.5, 0.7]}, schema=schema)
print(t.schema.field("id"))
```
実行結果:
```
id: int64 not null
name: string
score: double
-- schema metadata --
source: 'demo'
['id', 'name', 'score'] [DataType(int64), DataType(string), DataType(double)]
False True
2
{b'source': b'demo'}
['id', 'name', 'score', 'flag']
False ['name', 'score']
pyarrow.Field<id: int64 not null>
```

**注意点・落とし穴**:
- `nullable=False` は宣言にすぎず、データ側の null は強制されない。`pa.table({"id": [1, None]}, schema=schema)` は例外なく作成でき、`validate(full=True)` も通ることを確認した(null を含むデータが `not null` 列として保持される)。必要なら自前で `pc.is_null` などで検査する。
- `schema.metadata` のキー・値は `bytes` で返る。

### `pa.unify_schemas(schemas)`

**用途**: 複数のスキーマを1つに統合する。列の追加・欠落があるファイル群を結合する前に共通スキーマを作るのに使う。

**シグネチャ**: `pa.unify_schemas(schemas, *, promote_options='default')`

**使用例**:
```python
s1 = pa.schema([("a", pa.int64()), ("b", pa.string())])
s2 = pa.schema([("b", pa.string()), ("c", pa.float64())])
print(pa.unify_schemas([s1, s2]))
print()
try:
    pa.unify_schemas([pa.schema([("a", pa.int64())]), pa.schema([("a", pa.string())])])
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[0])
print(pa.unify_schemas([pa.schema([("a", pa.int32())]), pa.schema([("a", pa.int64())])], promote_options="permissive"))
```
実行結果:
```
a: int64
b: string
c: double

ArrowTypeError Unable to merge: Field a has incompatible types: int64 vs string
a: int64
```

**注意点・落とし穴**:
- 同名で型が違うと既定ではエラーになる(`ArrowTypeError`)。`promote_options="permissive"` を指定すると、上の例のように `int32` と `int64` は `int64` に昇格して統合される。
- 統合結果の列順は、最初のスキーマの列順の後ろに、後続スキーマにだけある列が追加される形になる。

## 配列・テーブル

### `pa.array(obj, type=None, ...)`

**用途**: Python のリスト・NumPy 配列・pandas Series などから Arrow の `Array`(1列分の連続メモリ)を作る。

**シグネチャ**: `pa.array(obj, type=None, mask=None, size=None, from_pandas=None, safe=True, memory_pool=None)`

**使用例**:
```python
import pyarrow as pa
import numpy as np

a = pa.array([1, 2, None, 4])
print(a.type, len(a), a.null_count)
print(a)
print(pa.array([1, 2, 3], type=pa.float32()).type)
print(pa.array(np.array([1.5, np.nan, 3.0])).null_count)                 # NaN は null にならない
print(pa.array(np.array([1.5, np.nan, 3.0]), from_pandas=True).null_count)  # from_pandas=True で NaN→null
print(pa.array([10, 20, 30], mask=np.array([False, True, False])))
try:
    pa.array([300], type=pa.int8())
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
int64 4 1
[
  1,
  2,
  null,
  4
]
float
0
1
[
  10,
  null,
  30
]
ArrowInvalid Value 300 too large to fit in C integer type
```

**注意点・落とし穴**:
- 既定では NaN は null ではなく通常の浮動小数値として扱われる(`null_count` が 0)。NaN を null にしたい場合は `from_pandas=True` を渡すか `pc.if_else(pc.is_nan(...), None, ...)` を使う。
- 型を指定して範囲外の値を入れると `ArrowInvalid`。`safe=False` を渡しても `pa.array([300], type=pa.int8(), safe=False)` は同じ `ArrowInvalid`(Python の `int` 入力の範囲チェックは緩められなかった)。

### `pa.scalar(value, type=None)` / `Array[i]` / `.as_py()`

**用途**: 1つの値を型付きで表す `Scalar` を作る。配列の要素アクセス(`arr[0]`)の戻り値も `Scalar` で、`.as_py()` で Python の値に戻せる。

**シグネチャ**: `pa.scalar(value, type=None, *, from_pandas=None, memory_pool=None)` / `Scalar.as_py(self, *, maps_as_pydicts=None)`

**使用例**:
```python
s = pa.scalar(42)
print(repr(s), s.type, s.as_py(), s.is_valid)
n = pa.scalar(None, type=pa.int64())
print(repr(n), n.is_valid, n.as_py())

arr = pa.array(["x", None, "z"])
print(repr(arr[0]), arr[0].as_py(), arr[1].as_py())
print(arr[-1], arr[0:2].to_pylist())
print(pa.scalar(1) == pa.scalar(1), pa.scalar(1).equals(pa.scalar(1)))
```
実行結果:
```
<pyarrow.Int64Scalar: 42> int64 42 True
<pyarrow.Int64Scalar: None> False None
<pyarrow.StringScalar: 'x'> x None
z ['x', None]
True True
```

**注意点・落とし穴**:
- `Array` から取り出した値は Python の `int` や `str` ではなく `Scalar` オブジェクト。1要素ずつ `as_py()` する処理は遅いため、大量データでは `to_pylist()` / `to_numpy()` か `pyarrow.compute` を使う。
- `Scalar` 同士の `==` と `equals()` はいずれも値と型の一致を判定する(上の例の最終行は両方 `True`)。

### `pa.chunked_array(arrays)` / `ChunkedArray`

**用途**: 複数の `Array`(チャンク)を1本の列として扱う型。`Table` の各列は `ChunkedArray` で、ファイルの読み込みや `concat_tables` の結果は複数チャンクになる。

**シグネチャ**: `pa.chunked_array(arrays, type=None)`

**使用例**:
```python
ca = pa.chunked_array([[1, 2, 3], [4, 5], [6]])
print(type(ca).__name__, ca.type, len(ca), ca.num_chunks)
print([len(c) for c in ca.chunks])
print(ca[3], ca.to_pylist())
print(ca.slice(2, 3).to_pylist())

one = ca.combine_chunks()          # ChunkedArray -> 連続した1つの Array
print(type(one).__name__, one.to_pylist())
print(pa.chunked_array([[1], [2]]).null_count, ca.nbytes)
```
実行結果:
```
ChunkedArray int64 6 3
[3, 2, 1]
4 [1, 2, 3, 4, 5, 6]
[3, 4, 5]
Int64Array [1, 2, 3, 4, 5, 6]
0 48
```

**注意点・落とし穴**:
- `ChunkedArray` はチャンクをまたいだ連続バッファを持たない。`ChunkedArray.to_numpy()` は `zero_copy_only=True` を指定すると `ValueError: zero_copy_only must be False for pyarrow.ChunkedArray.to_numpy` になる(常にコピーして NumPy 配列を作る)。
- `combine_chunks()` の戻り値は `ChunkedArray` ではなく連結済みの `Array`(上の例では `Int64Array`)。

### `pa.table(data)` / `Table` の基本属性

**用途**: Arrow の表(列指向のテーブル)を作る。辞書・pandas DataFrame・配列のリストなどから作成でき、`schema`・`num_rows`・`column()` などで中身を確認する。

**シグネチャ**: `pa.table(data, names=None, schema=None, metadata=None, nthreads=None)`

**使用例**:
```python
t = pa.table({"id": [1, 2, 3], "name": ["a", "b", None], "score": [0.5, 1.5, 2.5]})
print(t)
print(t.shape, t.num_rows, t.num_columns)
print(t.schema)
print(t.column_names)
print(type(t.column("id")).__name__, t["name"].null_count)
print(t.column(2).to_pylist())
```
実行結果:
```
pyarrow.Table
id: int64
name: string
score: double
----
id: [[1,2,3]]
name: [["a","b",null]]
score: [[0.5,1.5,2.5]]
(3, 3) 3 3
id: int64
name: string
score: double
['id', 'name', 'score']
ChunkedArray 1
[0.5, 1.5, 2.5]
```

**注意点・落とし穴**:
- `t["id"]` / `t.column("id")` の戻り値は `Array` ではなく `ChunkedArray`。`t.column("id")[0]` は `Scalar`。
- `print(t)` は列ごとにチャンク構造(二重の `[[ ]]`)付きで表示される。行列形式で見たいときは `t.to_pandas()` にするのが手軽。

### `Table.from_pydict` / `Table.from_pylist` / `Table.from_arrays`

**用途**: 列の辞書、行(辞書)のリスト、配列のリストのそれぞれから `Table` を作る。`schema` で型を明示できる。

**シグネチャ**: `Table.from_pydict(mapping, schema=None, metadata=None)` / `Table.from_pylist(mapping, schema=None, metadata=None)` / `Table.from_arrays(arrays, names=None, schema=None, metadata=None)`

**使用例**:
```python
sch = pa.schema([("id", pa.int32()), ("name", pa.string())])

t1 = pa.Table.from_pydict({"id": [1, 2], "name": ["a", "b"]}, schema=sch)
t2 = pa.Table.from_pylist([{"id": 1, "name": "a"}, {"id": 2}], schema=sch)      # 欠けたキーは null
t3 = pa.Table.from_arrays([pa.array([1, 2]), pa.array(["a", "b"])], names=["id", "name"])
print(t1.schema.types)
print(t2.to_pylist())
print(t3.schema)
print(pa.Table.from_pylist([{"id": 1}, {"id": 2, "extra": "x"}]).column_names)   # 最初の行のキーだけが使われる
```
実行結果:
```
[DataType(int32), DataType(string)]
[{'id': 1, 'name': 'a'}, {'id': 2, 'name': None}]
id: int64
name: string
['id']
```

**注意点・落とし穴**:
- `from_pylist` は、`schema` を指定しない場合は **先頭行のキー** だけを列名として採用する。後続行にだけ存在するキー(上の `extra`)は無視される。
- `from_pylist` は `schema` 指定時に欠損キーを null で埋める。
- `from_arrays` は `names` か `schema` のどちらかが必須(どちらも無いと `ValueError: Must pass names or schema when constructing Table or RecordBatch.`)。

### `Table.to_pydict()` / `Table.to_pylist()`

**用途**: `Table` を Python の辞書(列名→値のリスト)/ 辞書のリスト(行)へ変換する。

**シグネチャ**: `Table.to_pydict(self)`(引数なし)/ `Table.to_pylist(self, *, maps_as_pydicts=None)`

**使用例**:
```python
print(t.to_pydict())
print(t.to_pylist())
print(t.slice(0, 1).to_pylist())
```
実行結果:
```
{'id': [1, 2, 3], 'name': ['a', 'b', None], 'score': [0.5, 1.5, 2.5]}
[{'id': 1, 'name': 'a', 'score': 0.5}, {'id': 2, 'name': 'b', 'score': 1.5}, {'id': 3, 'name': None, 'score': 2.5}]
[{'id': 1, 'name': 'a', 'score': 0.5}]
```

**注意点・落とし穴**:
- どちらも全データを Python オブジェクトに展開するため、大きなテーブルでは非常に遅く、メモリも多く使う。小さいテーブルの確認・テストに限って使う。

### 列の追加・削除・選択(`select` / `drop_columns` / `rename_columns` / `append_column` / `set_column`)

**用途**: `Table` の列を操作する。Arrow の `Table` はイミュータブルなので、いずれも新しい `Table` を返す(元は変わらない)。

**シグネチャ**: `Table.select(self, columns)` / `Table.drop_columns(self, columns)` / `Table.rename_columns(self, names)` / `Table.append_column(self, field_, column)` / `Table.set_column(self, i, field_, column)`

**使用例**:
```python
t = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"], "score": [0.5, 1.5, 2.5]})

print(t.select(["score", "id"]).column_names)              # 選択+並べ替え
print(t.drop_columns(["name"]).column_names)
print(t.rename_columns(["ID", "NAME", "SCORE"]).column_names)   # 全列分の新しい名前が必要
t2 = t.append_column("double", pa.array([1.0, 3.0, 5.0]))
print(t2.column_names)
t3 = t2.set_column(0, "id", pa.array([10, 20, 30]))          # 位置を指定して置換
print(t3.to_pydict()["id"], t.to_pydict()["id"])
```
実行結果:
```
['score', 'id']
['id', 'score']
['ID', 'NAME', 'SCORE']
['id', 'name', 'score', 'double']
[10, 20, 30] [1, 2, 3]
```

**注意点・落とし穴**:
- `rename_columns` は全列分の新しい名前のリストを取り、要素数が列数と違うと `ArrowInvalid`(`tried to rename a table of 2 columns but only 1 names were provided`)になる。25.0.1 では `{"a": "A"}` のような旧名→新名の辞書も受け付け、その場合は指定した列だけがリネームされる(`pa.table({'a':[1],'b':['x']}).rename_columns({'a':'A'}).column_names` → `['A', 'b']` を確認)。
- `set_column(i, field_, column)` の第1引数は列の位置(インデックス)。列名では指定できない。

### 行の選択(`Table.slice` / `Table.take` / `Table.filter` / `Table.sort_by`)

**用途**: 位置・インデックス・真偽値マスク・並べ替えによって行を選択する。`slice` はゼロコピー、`take`・`filter`・`sort_by` は新しいデータを作る。

**シグネチャ**: `Table.slice(self, offset=0, length=None)` / `Table.take(self, indices)` / `Table.filter(self, mask, null_selection_behavior='drop')` / `Table.sort_by(self, sorting, **kwargs)`

**使用例**:
```python
import pyarrow.compute as pc

t = pa.table({"id": [3, 1, 2, 4], "score": [30.0, 10.0, 20.0, 40.0]})
print(t.slice(1, 2).to_pydict())
print(t.take([3, 0]).to_pydict())
print(t.filter(pc.greater(t["score"], 15)).to_pydict())
print(t.sort_by("id").to_pydict())
print(t.sort_by([("score", "descending")]).to_pydict())
print(t.filter(pc.field("id") > 2).to_pydict())      # 式(Expression)でも絞り込める
```
実行結果:
```
{'id': [1, 2], 'score': [10.0, 20.0]}
{'id': [4, 3], 'score': [40.0, 30.0]}
{'id': [3, 2, 4], 'score': [30.0, 20.0, 40.0]}
{'id': [1, 2, 3, 4], 'score': [10.0, 20.0, 30.0, 40.0]}
{'id': [4, 3, 2, 1], 'score': [40.0, 30.0, 20.0, 10.0]}
{'id': [3, 4], 'score': [30.0, 40.0]}
```

**注意点・落とし穴**:
- `Table.filter` は、`Array`/`ChunkedArray` の真偽値マスクのほか、`pc.field("id") > 2` のような `Expression` も受け取る。
- マスクが null の行は、既定(`null_selection_behavior='drop'`)では除外される。`'emit_null'` を指定すると null 行として出力に残る(`pa.table({'a':[1,2,3]}).filter(pa.array([True,None,False]), null_selection_behavior='emit_null')` → `{'a': [1, None]}` を確認)。

### `pa.concat_tables(tables, promote_options=...)`

**用途**: 複数の `Table` を縦に連結する。データをコピーせず、チャンクを並べるだけなので高速。

**シグネチャ**: `pa.concat_tables(tables, memory_pool=None, promote_options='none', **kwargs)`

**使用例**:
```python
t1 = pa.table({"a": [1, 2], "b": ["x", "y"]})
t2 = pa.table({"a": [3], "b": ["z"]})
t = pa.concat_tables([t1, t2])
print(t.num_rows, t["a"].num_chunks)

t3 = pa.table({"a": [4], "c": [1.5]})
try:
    pa.concat_tables([t1, t3])
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[0])
print(pa.concat_tables([t1, t3], promote_options="default").to_pydict())

t4 = pa.table({"a": pa.array([5], pa.int32()), "b": ["w"]})
print(pa.concat_tables([t1, t4], promote_options="permissive").schema.field("a").type)
```
実行結果:
```
3 2
ArrowInvalid Schema at index 1 was different: 
{'a': [1, 2, 4], 'b': ['x', 'y', None], 'c': [None, None, 1.5]}
int64
```

**注意点・落とし穴**:
- 既定の `promote_options='none'` はスキーマが完全に一致しないとエラーになる。列が異なるとき(欠落列は null で埋まる)は `'default'`、`int32` と `int64` のような型違いを昇格させたいときは `'permissive'`。`'default'` でも `int64` と `string` のように昇格できない型の組はエラー(`ArrowTypeError: Unable to merge: Field a has incompatible types: int64 vs string`)。
- 連結後の列は複数チャンクになる(`num_chunks` が連結数)。以降の処理で連続メモリが必要なら `combine_chunks()` を呼ぶ。

### `Table.cast(target_schema)` / `Array.cast(target_type)`

**用途**: 列の型を変換する。`Table.cast` はスキーマ全体を渡す(列名・順序が一致する必要がある)。デフォルトは安全な変換(`safe=True`)。

**シグネチャ**: `Table.cast(self, target_schema, safe=None, options=None)` / `Array.cast(self, target_type=None, safe=None, options=None, memory_pool=None)`

**使用例**:
```python
t = pa.table({"id": [1, 2, 3], "v": ["1.5", "2.5", "3.5"]})
t2 = t.cast(pa.schema([("id", pa.int32()), ("v", pa.float64())]))
print(t2.schema)

print(pa.array([1.9, 2.1]).cast(pa.int64(), safe=False))
try:
    pa.array([1.9, 2.1]).cast(pa.int64())
except Exception as e:
    print(type(e).__name__, e)
try:
    pa.array([300]).cast(pa.int8())
except Exception as e:
    print(type(e).__name__, e)
print(pa.array(["a", "b"]).cast(pa.dictionary(pa.int8(), pa.string())).type)
```
実行結果:
```
id: int32
v: double
[
  1,
  2
]
ArrowInvalid Float value 1.900000 was truncated converting to int64
ArrowInvalid Integer value 300 not in range: -128 to 127
dictionary<values=string, indices=int8, ordered=0>
```

**注意点・落とし穴**:
- `safe=True`(既定)では、小数の切り捨てやオーバーフローは `ArrowInvalid` になる。`safe=False` にすると検査が外れ、`pa.array([300]).cast(pa.int8(), safe=False)` は `44`(下位ビットへの切り詰め)になるなど静かに値が壊れるため、範囲が保証できるときに限って使う。
- `Table.cast` に渡すスキーマは、元のテーブルと列名・順序が同じである必要がある(順序違いは `ValueError: Target schema's field names are not matching the table's field names`)。一部の列だけを変換したいときは `set_column` を使う。

### `pa.record_batch` / `Table.to_batches` / `RecordBatchReader`

**用途**: `RecordBatch`(チャンク1つ分の表)と、それを順次読み出すストリーム(`RecordBatchReader`)を扱う。メモリに載り切らないデータを処理するときの基本部品。

**シグネチャ**: `pa.record_batch(data, names=None, schema=None, metadata=None)` / `Table.to_batches(self, max_chunksize=None)` / `pa.RecordBatchReader.from_batches(schema, batches)` / `pa.Table.from_batches(batches)`

**使用例**:
```python
rb = pa.record_batch({"a": [1, 2, 3], "b": ["x", "y", "z"]})
print(type(rb).__name__, rb.num_rows, rb.schema.names)

t = pa.Table.from_batches([rb, rb])
print(t.num_rows, t["a"].num_chunks)
print([b.num_rows for b in t.to_batches(max_chunksize=4)])

def gen():
    for i in range(3):
        yield pa.record_batch({"a": [i, i + 10]})

reader = pa.RecordBatchReader.from_batches(pa.schema([("a", pa.int64())]), gen())
for batch in reader:
    print(batch.to_pydict())
```
実行結果:
```
RecordBatch 3 ['a', 'b']
6 2
[3, 3]
{'a': [0, 10]}
{'a': [1, 11]}
{'a': [2, 12]}
```

**注意点・落とし穴**:
- `RecordBatchReader` は一度読み切ると再利用できない(`read_all()` を2回呼ぶと2回目は 0 行になることを確認)。
- `Table.to_batches(max_chunksize=n)` は既存のチャンク境界をまたいで結合しない。上の例では 3 行のチャンク2つが、上限 4 行でも `[3, 3]` のまま返る。

## 計算(pyarrow.compute)

### `pc.sum` / `pc.mean` / `pc.min_max` / `pc.count`(集約関数)

**用途**: 配列全体を1つのスカラーに集約する。戻り値は `Scalar` で、`.as_py()` で Python の値にする。null はデフォルトで無視される。

**シグネチャ**: `pc.sum(array, /, *, skip_nulls=True, min_count=1, options=None, memory_pool=None)` / `pc.mean(...)`(同形)/ `pc.min_max(array, /, *, skip_nulls=True, min_count=1, ...)` / `pc.count(array, /, mode='only_valid', ...)`

**使用例**:
```python
import pyarrow as pa
import pyarrow.compute as pc

a = pa.array([1, 2, None, 4])
print(pc.sum(a), pc.mean(a).as_py())
print(pc.min_max(a).as_py())
print(pc.count(a).as_py(), pc.count(a, mode="only_null").as_py(), pc.count(a, mode="all").as_py())
print(pc.sum(a, skip_nulls=False).as_py())
print(pc.sum(pa.array([], type=pa.int64())).as_py())
print(pc.sum(pa.array([], type=pa.int64()), min_count=0).as_py())
print(pc.stddev(a, ddof=1).as_py(), pc.variance(pa.array([1.0, 2.0, 3.0])).as_py())
print(pc.count_distinct(pa.array(["a", "b", "a", None])).as_py())
```
実行結果:
```
7 2.3333333333333335
{'min': 1, 'max': 4}
3 1 4
None
None
0
1.5275252316519465 0.6666666666666666
2
```

**注意点・落とし穴**:
- `pc.sum` は空配列や全 null の配列に対して、既定の `min_count=1` では **null** を返す(0 ではない)。0 を得たい場合は `min_count=0` を指定する。
- `skip_nulls=False` にすると、1つでも null があれば結果は null になる。
- `pc.stddev`/`pc.variance` の `ddof` の既定は 0(母標準偏差)。標本標準偏差にしたい場合は `ddof=1`。

### 比較・論理関数と `pc.filter`(`pc.greater` / `pc.equal` / `pc.and_` / `pc.invert`)

**用途**: 比較関数で真偽値配列(マスク)を作り、`pc.filter` で該当要素だけを取り出す。Arrow の `Array` は `+` などの算術演算子は使えるが、比較演算子(`>` など)は使えない(`pa.array([1, 2]) > 1` は `TypeError`)ため、比較は関数呼び出しが基本になる。

**シグネチャ**: `pc.greater(x, y, /, *, memory_pool=None)` / `pc.equal(x, y, ...)` / `pc.and_(x, y, ...)` / `pc.and_kleene(x, y, ...)` / `pc.invert(x, ...)` / `pc.filter(input, selection_filter, /, null_selection_behavior='drop', ...)`

**使用例**:
```python
a = pa.array([5, 1, None, 8, 3])
mask = pc.greater(a, 2)
print(mask)
print(pc.filter(a, mask))
print(pc.filter(a, mask, null_selection_behavior="emit_null"))

both = pc.and_(pc.greater(a, 2), pc.less(a, 8))
print(both.to_pylist())
x = pa.array([False, None, True])
y = pa.array([None, None, None], pa.bool_())
print(pc.and_(x, y).to_pylist(), pc.and_kleene(x, y).to_pylist())
print(pc.invert(pa.array([True, False, None])).to_pylist())
print(pc.equal(pa.array(["a", "b"]), "a").to_pylist())
```
実行結果:
```
[
  true,
  false,
  null,
  true,
  true
]
[
  5,
  8,
  3
]
[
  5,
  null,
  8,
  3
]
[True, False, None, False, True]
[None, None, None] [False, None, None]
[False, True, None]
[True, False]
```

**注意点・落とし穴**:
- 比較結果は入力が null の位置で null になる(`False` ではない)。`pc.filter` は既定でマスクが null の行を捨てる。
- `pc.and_` は null が1つでもあれば null を返すが、`pc.and_kleene` は三値論理(`False AND null` は `False`)。上の例では `and_` が `[None, None, None]`、`and_kleene` が `[False, None, None]` になる(1行目が `False AND null`)。三値論理が必要なら型付きの bool 配列を渡すこと(`pa.array([None, None])` は null 型になり `and_kleene` が `no kernel matching input types (bool, null)` で失敗する)。

### `pc.take(data, indices)` / `pc.array_sort_indices` / `pc.sort_indices`

**用途**: インデックスによる要素の抽出と、並べ替えの順序を表すインデックス列の計算。ソートは「順序を表すインデックスを作って `take`」する2段階が基本。`Table` の並べ替えは `Table.sort_by` が簡便。

**シグネチャ**: `pc.take(data, indices, *, boundscheck=True, memory_pool=None)` / `pc.array_sort_indices(array, /, order='ascending', *, null_placement='at_end', ...)` / `pc.sort_indices(input, /, sort_keys=(), *, null_placement=None, options=None, ...)`

**使用例**:
```python
a = pa.array([30, 10, None, 20])
print(pc.array_sort_indices(a).to_pylist())
print(pc.take(a, pc.array_sort_indices(a, order="descending")).to_pylist())
print(pc.take(a, pc.array_sort_indices(a, null_placement="at_start")).to_pylist())
print(pc.take(a, pa.array([3, 0])).to_pylist())

t = pa.table({"g": ["b", "a", "b", "a"], "v": [1, 2, 3, 4]})
order = pc.sort_indices(t, sort_keys=[("g", "ascending"), ("v", "descending")])
print(order.to_pylist())
print(t.take(order).to_pydict())
try:
    pc.take(a, pa.array([10]))
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
[1, 3, 0, 2]
[30, 20, 10, None]
[None, 10, 20, 30]
[20, 30]
[3, 1, 2, 0]
{'g': ['a', 'a', 'b', 'b'], 'v': [4, 2, 3, 1]}
ArrowIndexError Index 10 out of bounds
```

**注意点・落とし穴**:
- `pc.sort_indices` の `null_placement` 引数(`SortOptions` 側の指定)は 25.0.0 で非推奨になり、`FutureWarning: Specifying null_placement in SortOptions is deprecated as of 25.0.0. Specify null_placement per sort_key instead.` が出る(`Table` でも同様)。列ごとに `sort_keys=[("x", "ascending", "at_start")]` のように 3 要素タプルで指定する(`Table.sort_by([("x", "ascending", "at_start")])` でも通ることを確認)。`pc.array_sort_indices` の `null_placement` は警告なしで使えた。
- null は既定で末尾に並ぶ。
- `take` の範囲外インデックスは既定の `boundscheck=True` で `ArrowIndexError` になる。`boundscheck=False` にすると検査が省かれ、範囲外を渡したときの結果は保証されない(手元では null が返ったが、その挙動に依存してはいけない)。

### `pc.is_in` / `pc.index_in`

**用途**: 各要素が指定した値の集合に含まれるかを判定する(`is_in`)/ 集合内の位置を返す(`index_in`)。SQL の `IN` 句に相当する。

**シグネチャ**: `pc.is_in(values, /, value_set, *, skip_nulls=False, options=None, ...)` / `pc.index_in(values, /, value_set, *, skip_nulls=False, options=None, ...)`

**使用例**:
```python
a = pa.array(["tokyo", "osaka", None, "nagoya"])
targets = pa.array(["osaka", "nagoya"])
print(pc.is_in(a, value_set=targets).to_pylist())
print(pc.index_in(a, value_set=targets).to_pylist())
print(pc.is_in(a, value_set=pa.array(["osaka", None])).to_pylist())
print(pc.is_in(a, value_set=pa.array(["osaka", None]), skip_nulls=True).to_pylist())

t = pa.table({"city": a, "n": [1, 2, 3, 4]})
print(t.filter(pc.is_in(t["city"], value_set=targets)).to_pydict())
```
実行結果:
```
[False, True, False, True]
[None, 0, None, 1]
[False, True, True, False]
[False, True, False, False]
{'city': ['osaka', 'nagoya'], 'n': [2, 4]}
```

**注意点・落とし穴**:
- `value_set` に Python のリストを渡すと `TypeError: "['osaka']" is not a valid value set` になる。必ず `pa.array(...)` で渡す。
- `is_in` は、入力の null が `value_set` に null を含む場合のみ True。`skip_nulls=True` にすると `value_set` の null を無視する(上の3・4つ目の出力を比較)。

### `pc.if_else` / `pc.case_when` / `pc.fill_null` / `pc.coalesce`

**用途**: 条件分岐(`if_else`, `case_when`)と欠損値の補完(`fill_null`, `coalesce`)を行う。`fill_null` は `Array` と `ChunkedArray`(`Table` の列)に使える(`Table` 全体を渡すと `AttributeError`)。

**シグネチャ**: `pc.if_else(cond, left, right, /, *, memory_pool=None)` / `pc.case_when(cond, /, *cases, memory_pool=None)` / `pc.fill_null(values, fill_value)` / `pc.coalesce(*values, memory_pool=None)`

**使用例**:
```python
a = pa.array([5, None, 15, 25])
print(pc.if_else(pc.greater(a, 10), "big", "small").to_pylist())      # cond が null → null
print(pc.fill_null(a, 0).to_pylist())
print(pc.coalesce(pa.array([None, 2, None]), pa.array([10, 20, 30])).to_pylist())

grade = pc.case_when(
    pc.make_struct(pc.less(a, 10), pc.less(a, 20)),
    "low", "mid", "high")
print(grade.to_pylist())
```
実行結果:
```
['small', None, 'big', 'big']
[5, 0, 15, 25]
[10, 2, 30]
['low', 'high', 'mid', 'high']
```

**注意点・落とし穴**:
- `pc.if_else` は `cond` が null の行を null にする(`fill_null` で後から埋める)。
- `pc.case_when` の第1引数は、条件配列を `pc.make_struct` で束ねた struct。続く引数は各条件に対応する値で、条件の数より1つ多い最後の値が「どれにも該当しない場合」の値(else)になる。先頭から最初に真になった条件が採用され、条件が null の行は偽として扱われて次の条件へ進む(上の例で入力が null の行は `"high"` になった)。

### `pc.value_counts` / `pc.unique` / `pc.count_distinct` / `pc.mode`

**用途**: 値の出現回数、ユニーク値、ユニーク数、最頻値を求める。`value_counts` は `values`/`counts` の struct 配列を返す。

**シグネチャ**: `pc.value_counts(array, /, *, memory_pool=None)` / `pc.unique(array, /, *, memory_pool=None)` / `pc.count_distinct(array, /, mode='only_valid', ...)` / `pc.mode(array, /, n=1, *, skip_nulls=True, min_count=0, ...)`

**使用例**:
```python
a = pa.array(["a", "b", "a", None, "a", "b"])
vc = pc.value_counts(a)
print(vc.type)
print(vc.to_pylist())
print(pc.unique(a).to_pylist())
print(pc.count_distinct(a).as_py(), pc.count_distinct(a, mode="all").as_py())
print(pc.mode(pa.array([1, 2, 2, 3, 3, 3])).to_pylist())

# 出現回数の多い順に並べる
counts = pa.Table.from_struct_array(vc)
print(counts.sort_by([("counts", "descending")]).to_pylist()[0])
```
実行結果:
```
struct<values: string, counts: int64>
[{'values': 'a', 'counts': 3}, {'values': 'b', 'counts': 2}, {'values': None, 'counts': 1}]
['a', 'b', None]
2 3
[{'mode': 3, 'count': 3}]
{'values': 'a', 'counts': 3}
```

**注意点・落とし穴**:
- `value_counts` の struct のフィールド名は `values` と `counts`(複数形)。
- `value_counts` は null も1つの値として数える(上の出力に `None` が現れる)。一方 `count_distinct` は既定で null を数えない(`mode='all'` で数える)。
- `value_counts` の結果の並び順は初出順で、頻度順ではない。頻度順にしたい場合は上のように `Table.from_struct_array` で表にして `sort_by` する。

### 文字列関数(`pc.utf8_upper` / `pc.match_substring` / `pc.replace_substring` / `pc.split_pattern` / `pc.utf8_length` ほか)

**用途**: 文字列配列に対するベクトル化された操作。`utf8_*` 系は Unicode 対応、それ以外の `ascii_*` 系は ASCII 前提。

**シグネチャ**: `pc.utf8_upper(strings, /, *, ...)` / `pc.utf8_length(strings, /, ...)` / `pc.match_substring(strings, /, pattern, *, ignore_case=False, ...)` / `pc.match_substring_regex(...)`(同形)/ `pc.starts_with(strings, /, pattern, *, ignore_case=False, ...)` / `pc.replace_substring(strings, /, pattern, replacement, *, max_replacements=None, ...)` / `pc.split_pattern(strings, /, pattern, *, max_splits=None, reverse=False, ...)` / `pc.extract_regex(strings, /, pattern, ...)` / `pc.utf8_slice_codeunits(strings, /, start, stop=None, step=1, ...)` / `pc.utf8_trim_whitespace(strings, /, ...)`

**使用例**:
```python
s = pa.array(["Apple pie", " banana ", None, "日本語テキスト"])
print(pc.utf8_upper(s).to_pylist())
print(pc.utf8_length(s).to_pylist())
print(pc.utf8_trim_whitespace(s).to_pylist())
print(pc.match_substring(s, "an").to_pylist())
print(pc.match_substring_regex(s, r"^[A-Z]").to_pylist())
print(pc.starts_with(s, "apple", ignore_case=True).to_pylist())
print(pc.replace_substring(s, "a", "_").to_pylist())
print(pc.split_pattern(pa.array(["a,b,c", "d"]), ",").to_pylist())
print(pc.utf8_slice_codeunits(s, 0, 3).to_pylist())
print(pc.extract_regex(pa.array(["id=12", "id=7", "x"]), r"id=(?P<num>\d+)").to_pylist())
```
実行結果:
```
['APPLE PIE', ' BANANA ', None, '日本語テキスト']
[9, 8, None, 7]
['Apple pie', 'banana', None, '日本語テキスト']
[False, True, None, False]
[True, False, None, False]
[True, False, None, False]
['Apple pie', ' b_n_n_ ', None, '日本語テキスト']
[['a', 'b', 'c'], ['d']]
['App', ' ba', None, '日本語']
[{'num': '12'}, {'num': '7'}, None]
```

**注意点・落とし穴**:
- `pc.utf8_length` は文字数(コードポイント数)で、`pc.binary_length` はバイト数。`'日本語'` は前者が 3、後者が 9(UTF-8 で1文字3バイト)になる。
- `match_substring` はリテラル一致、`match_substring_regex` は正規表現(RE2 構文)。`pc.extract_regex` は名前付きグループ(`(?P<name>...)`)を持つパターンを要求し(名前なしグループだと `ArrowInvalid: Regular expression contains unnamed groups`)、結果は struct 配列。マッチしなかった行は struct 全体が null になる(上の出力の `None`)。
- `Table` の列(`ChunkedArray`)にもそのまま渡せる(結果も `ChunkedArray`)。

### 日時関数(`pc.strptime` / `pc.strftime` / `pc.year` / `pc.month` / `pc.floor_temporal` ほか)

**用途**: 文字列→日時のパース、日時→文字列の整形、年月日などの要素の取り出し、切り捨て(週・月単位など)を行う。

**シグネチャ**: `pc.strptime(strings, /, format, unit, error_is_null=False, *, ...)` / `pc.strftime(timestamps, /, format='%Y-%m-%dT%H:%M:%S', locale='C', *, ...)` / `pc.year(values, /, ...)` / `pc.month(values, /, ...)` / `pc.day_of_week(values, /, *, count_from_zero=True, week_start=1, ...)` / `pc.floor_temporal(timestamps, /, multiple=1, unit='day', ...)`

**使用例**:
```python
s = pa.array(["2024-03-15 10:30:00", "2024-12-31 23:59:59", None])
ts = pc.strptime(s, format="%Y-%m-%d %H:%M:%S", unit="s")
print(ts.type)
print(pc.year(ts).to_pylist(), pc.month(ts).to_pylist(), pc.day_of_week(ts).to_pylist())
print(pc.strftime(ts, format="%Y/%m/%d").to_pylist())
print(pc.floor_temporal(ts, unit="month").to_pylist())
print(pc.strptime(pa.array(["2024-03-15", "oops"]), format="%Y-%m-%d", unit="s", error_is_null=True).to_pylist())
print(pc.subtract(ts[1], ts[0]))
```
実行結果:
```
timestamp[s]
[2024, 2024, None] [3, 12, None] [4, 1, None]
['2024/03/15', '2024/12/31', None]
[datetime.datetime(2024, 3, 1, 0, 0), datetime.datetime(2024, 12, 1, 0, 0), None]
[datetime.datetime(2024, 3, 15, 0, 0), None]
291 days, 13:29:59
```

**注意点・落とし穴**:
- `strptime` に渡す `unit` は必須(`'s'`/`'ms'`/`'us'`/`'ns'`)。フォーマットに合わない文字列があると既定で `ArrowInvalid: Failed to parse string: 'oops' as a scalar of type timestamp[s]` になり、`error_is_null=True` で null にできる。
- `pc.day_of_week` は既定で月曜=0(`count_from_zero=True, week_start=1`)。上の 2024-03-15 は金曜なので 4。
- `strftime`/`year` などは `timestamp` 型か `date` 型を要求する(`pc.year(pa.array([date(2024,1,1)]))` は `[2024]`)。文字列のまま渡すと `ArrowNotImplementedError: Function 'year' has no kernel matching input types (string)` になるので先に `strptime` する。

### 算術・丸め・累積(`pc.add` / `pc.multiply` / `pc.divide` / `pc.round` / `pc.cumulative_sum` / `pc.quantile`)

**用途**: 数値配列の要素ごとの演算(スカラーとも演算可)と、累積・分位点などの統計関数。

**シグネチャ**: `pc.add(x, y, /, *, ...)` / `pc.multiply(x, y, ...)` / `pc.divide(dividend, divisor, /, ...)` / `pc.abs(x, /, ...)` / `pc.round(x, /, ndigits=0, round_mode='half_to_even', *, ...)` / `pc.cumulative_sum(values, /, start=None, *, skip_nulls=False, ...)` / `pc.quantile(array, /, q=0.5, *, interpolation='linear', skip_nulls=True, min_count=0, ...)`

**使用例**:
```python
a = pa.array([1, 2, 3, 4])
print(pc.add(a, 10).to_pylist(), pc.multiply(a, a).to_pylist())
print(pc.divide(a, 2).to_pylist())                    # 整数同士は整数除算
print(pc.divide(pc.cast(a, pa.float64()), 2).to_pylist())
print(pc.round(pa.array([2.5, 3.5, -2.5])).to_pylist())
print(pc.round(pa.array([2.5, 3.5, -2.5]), round_mode="half_up").to_pylist())
print(pc.round(pa.array([2.5, 3.5, -2.5]), round_mode="half_towards_infinity").to_pylist())
print(pc.round(pa.array([1.234, 5.678]), ndigits=1).to_pylist())
print(pc.cumulative_sum(a).to_pylist())
print(pc.quantile(pa.array([1.0, 2.0, 3.0, 4.0, 5.0]), q=[0.25, 0.5, 0.9]).to_pylist())
try:
    pc.divide(a, 0)
except Exception as e:
    print(type(e).__name__, e)
print(pc.add(pa.array([2**62]), pa.array([2**62])))
```
実行結果:
```
[11, 12, 13, 14] [1, 4, 9, 16]
[0, 1, 1, 2]
[0.5, 1.0, 1.5, 2.0]
[2.0, 4.0, -2.0]
[3.0, 4.0, -2.0]
[3.0, 4.0, -3.0]
[1.2, 5.7]
[1, 3, 6, 10]
[2.0, 3.0, 4.6]
ArrowInvalid divide by zero
[
  -9223372036854775808
]
```

**注意点・落とし穴**:
- 整数同士の `pc.divide` は整数除算(`[1, 2, 3, 4] / 2` → `[0, 1, 1, 2]`。小数部は切り捨て)になる。小数の除算にしたければ先に `float64` へ `cast` する。整数の 0 除算は `ArrowInvalid: divide by zero`。
- `pc.round` の既定は銀行丸め(`half_to_even`)で `2.5 → 2`、`3.5 → 4`、`-2.5 → -2`。`half_up` は「ちょうど 0.5 のとき +∞ 方向」なので `-2.5 → -2`(負数では四捨五入と一致しない)。ゼロから遠ざかる方向の一般的な四捨五入にしたい場合は `round_mode="half_towards_infinity"`(`2.5→3`、`-2.5→-3`)を使う。
- `pc.add`/`pc.multiply` は整数オーバーフローを無視して回り込む(`pc.add_checked` / `pc.multiply_checked` を使うとオーバーフローで `ArrowInvalid` になる)。
- `pc.quantile` は `q` に配列を渡すと複数の分位点を一度に返す(戻り値は `double` 配列)。

### list / struct 関数(`pc.list_flatten` / `pc.list_value_length` / `pc.list_element` / `pc.struct_field`)

**用途**: リスト型・構造体型の列を平坦化したり、要素の長さ・n番目・フィールドを取り出す。ネストしたデータ(JSON 由来など)の処理で使う。

**シグネチャ**: `pc.list_flatten(lists, /, recursive=False, *, ...)` / `pc.list_value_length(lists, /, *, ...)` / `pc.list_element(lists, index, /, *, ...)` / `pc.list_parent_indices(lists, /, *, ...)` / `pc.struct_field(values, /, indices, *, ...)`

**使用例**:
```python
l = pa.array([[1, 2, 3], [], None, [4, 5]])
print(pc.list_value_length(l).to_pylist())
print(pc.list_flatten(l).to_pylist())
print(pc.list_parent_indices(l).to_pylist())      # 平坦化後の各要素が元の何行目に属するか
print(pc.list_element(pa.array([[1, 2], [3, 4]]), 0).to_pylist())

s = pa.array([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}])
print(pc.struct_field(s, "x").to_pylist())
print(pc.struct_field(s, [1]).to_pylist())          # 位置でも指定可

# 「1行のリスト→複数行」に展開(explode 相当)
t = pa.table({"id": [1, 2, 3, 4], "tags": l})
parent = pc.list_parent_indices(t["tags"])
print(pa.table({"id": pc.take(t["id"], parent), "tag": pc.list_flatten(t["tags"])}).to_pydict())
```
実行結果:
```
[3, 0, None, 2]
[1, 2, 3, 4, 5]
[0, 0, 0, 3, 3]
[1, 3]
[1, 2]
['a', 'b']
{'id': [1, 1, 1, 4, 4], 'tag': [1, 2, 3, 4, 5]}
```

**注意点・落とし穴**:
- `list_flatten` は null や空リストの行を結果から取り除く(出力に対応する要素が無い)。行との対応が必要なら `list_parent_indices` を併用する(上の explode 相当の例)。
- `list_value_length` と `list_element` は null のリストに対して null を返す。`list_element` の位置が範囲外の行があると `ArrowInvalid: Index 1 is out of bounds: should be in [0, 1)` になる。
- `struct_field` はフィールド名の文字列、または位置のリスト(`[1]`)で指定できる(`indices=` キーワードでも同じ)。

### `Table.group_by(keys).aggregate(aggs)`

**用途**: キー列ごとにグループ化して集約する(pandas の `groupby().agg()`、SQL の `GROUP BY` 相当)。

**シグネチャ**: `Table.group_by(self, keys, use_threads=True)` → `TableGroupBy.aggregate(self, aggregations)`(`aggregations` は `(列名, 関数名)` または `(列名, 関数名, オプション)` のリスト)

**使用例**:
```python
t = pa.table({
    "dept": ["A", "B", "A", "B", "A"],
    "sex": ["m", "f", "f", "f", "m"],
    "salary": [100, 200, 150, None, 120],
})
r = t.group_by("dept").aggregate([("salary", "sum"), ("salary", "mean"), ("salary", "count"), ("salary", "max")])
print(r.column_names)
print(r.sort_by("dept").to_pydict())

r2 = t.group_by(["dept", "sex"]).aggregate([("salary", "sum")]).sort_by([("dept", "ascending"), ("sex", "ascending")])
print(r2.to_pydict())
print(t.group_by("dept").aggregate([([], "count_all")]).sort_by("dept").to_pydict())
print(t.group_by("dept").aggregate([("salary", "count", pc.CountOptions(mode="all"))]).sort_by("dept").to_pydict())
```
実行結果:
```
['dept', 'salary_sum', 'salary_mean', 'salary_count', 'salary_max']
{'dept': ['A', 'B'], 'salary_sum': [370, 200], 'salary_mean': [123.33333333333333, 200.0], 'salary_count': [3, 1], 'salary_max': [150, 200]}
{'dept': ['A', 'A', 'B'], 'sex': ['f', 'm', 'f'], 'salary_sum': [150, 220, 200]}
{'dept': ['A', 'B'], 'count_all': [3, 2]}
{'dept': ['A', 'B'], 'salary_count': [3, 2]}
```

**注意点・落とし穴**:
- 結果の列名は `<列名>_<関数名>`(`salary_sum` など)で、25.0.1 ではキー列が**先頭**に来る(上の `column_names` の出力)。列名(`salary_sum` など)で参照し、位置に依存しないほうが安全。
- 結果のグループ順は保証されない(スレッド並列のため。上の例は比較のため `sort_by` している)。順序を安定させたい場合は `sort_by` するか、`use_threads=False` を指定する。
- `first` / `last` は順序依存の集約で、既定(`use_threads=True`)だと `ArrowNotImplementedError: Using ordered aggregator in multiple threaded execution is not supported` になる。`group_by(keys, use_threads=False)` を指定する(`v_first: [5, 7]`, `v_last: [6, 7]` を確認)。
- `count` は既定で null を数えない(`only_valid`)。null も含めた行数は `pc.CountOptions(mode="all")` か、`([], "count_all")`(行数)を使う。
- 使える集約関数名は `pc.list_functions()` の `hash_` で始まる関数から `hash_` を除いた名前で、`all` / `any` / `count` / `count_all` / `count_distinct` / `distinct` / `first` / `last` / `list` / `max` / `mean` / `min` / `one` / `product` / `stddev` / `sum` / `variance` などがある(`list` はグループごとの値のリスト、`distinct` はユニーク値のリストを返す)。

### `Table.join(right_table, keys, ...)`

**用途**: 2つのテーブルを結合する(内部結合・外部結合・semi/anti 結合)。結合キーの列型は一致している必要がある。

**シグネチャ**: `Table.join(self, right_table, keys, right_keys=None, join_type='left outer', left_suffix=None, right_suffix=None, coalesce_keys=True, use_threads=True, filter_expression=None)`

**使用例**:
```python
left = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
right = pa.table({"id": [2, 3, 4], "score": [20, 30, 40]})

print(left.join(right, keys="id", join_type="inner").sort_by("id").to_pydict())
print(left.join(right, keys="id").sort_by("id").to_pydict())                        # 既定は left outer
print(left.join(right, keys="id", join_type="full outer").sort_by("id").to_pydict())
print(left.join(right, keys="id", join_type="left anti").to_pydict())
print(left.join(right, keys="id", join_type="left semi").sort_by("id").to_pydict())

r2 = pa.table({"key": [1, 2], "name": ["X", "Y"]})
print(left.join(r2, keys="id", right_keys="key", right_suffix="_r").sort_by("id").to_pydict())
```
実行結果:
```
{'id': [2, 3], 'name': ['b', 'c'], 'score': [20, 30]}
{'id': [1, 2, 3], 'name': ['a', 'b', 'c'], 'score': [None, 20, 30]}
{'id': [1, 2, 3, 4], 'name': ['a', 'b', 'c', None], 'score': [None, 20, 30, 40]}
{'id': [1], 'name': ['a']}
{'id': [2, 3], 'name': ['b', 'c']}
{'id': [1, 2, 3], 'name': ['a', 'b', 'c'], 'name_r': ['X', 'Y', None]}
```

**注意点・落とし穴**:
- `join_type` は `'left semi'` / `'right semi'` / `'left anti'` / `'right anti'` / `'inner'` / `'left outer'` / `'right outer'` / `'full outer'` から選ぶ。既定が **`'left outer'`**(pandas の `merge` の既定 `inner` と異なる)。
- 行の順序は保証されない(上の例は比較のため `sort_by` している)。
- 結合キー以外に同名の列があり、`left_suffix` / `right_suffix` を指定しないと、同名の列がそのまま2つ並ぶ(`['id', 'name', 'name']`)。`to_pydict()` すると片方が消えるので、必ずサフィックスを指定する。
- キー列の型が違うと `ArrowInvalid: Incompatible data types for corresponding join field keys` になる(先に `cast` する)。

### `pc.field` による式(Expression)

**用途**: 評価を遅延した「式」を組み立てる。`Table.filter` や `ds.dataset` のフィルタ(`filter=`)・`pq.read_table(filters=...)` に共通して使われる。実データに触れずに条件だけを表現できるため、ファイルの述語プッシュダウンで使える。

**シグネチャ**: `pc.field(*name_or_index)` / `pc.scalar(value)`(定数の式)/ `pc.Expression.isin(values)` / `pc.Expression.is_valid()` / `pc.Expression.cast(self, type=None, safe=None, options=None)`

**使用例**:
```python
expr = (pc.field("age") >= 30) & (pc.field("city") == "Tokyo")
print(expr)
print(type(expr).__name__)

t = pa.table({"age": [25, 35, 40, None], "city": ["Tokyo", "Tokyo", "Osaka", "Tokyo"]})
print(t.filter(expr).to_pydict())
print(t.filter(pc.field("city").isin(["Osaka"])).to_pydict())
print(t.filter(pc.field("age").is_valid()).num_rows)
print(t.filter(~pc.field("age").is_null() & (pc.field("age") < 40)).to_pydict())
```
実行結果:
```
((age >= 30) and (city == "Tokyo"))
Expression
{'age': [35], 'city': ['Tokyo']}
{'age': [40], 'city': ['Osaka']}
3
{'age': [25, 35], 'city': ['Tokyo', 'Tokyo']}
```

**注意点・落とし穴**:
- 式の結合には `&`・`|`・`~` を使い、各比較は括弧で囲む。
- `Array`(マスク)へ比較演算子は使えないが、`Expression` に対しては `>` `>=` `==` などが使える(Expression 側でオーバーロードされている)。
- `pc.field("a") > 1 and pc.field("a") < 3` のように `and` を使うと `ValueError: An Expression cannot be evaluated to python True or False. ...` になる。null の判定は `is_null()` / `is_valid()` を使う。

## Parquet入出力

### `pq.write_table` / `pq.read_table`

**用途**: `Table` を Parquet ファイルに書き出す/読み込む。列指向・圧縮・スキーマ保持を備えた、Arrow で最も基本の永続化手段。

**シグネチャ**: `pq.write_table(table, where, row_group_size=None, version='2.6', use_dictionary=True, compression='snappy', write_statistics=True, ..., store_schema=True, write_page_index=False, ...)`(主要な引数のみ抜粋)/ `pq.read_table(source, *, columns=None, use_threads=True, schema=None, use_pandas_metadata=False, read_dictionary=None, memory_map=False, partitioning='hive', filesystem=None, filters=None, ...)`

**使用例**:
```python
import pyarrow as pa
import pyarrow.parquet as pq

t = pa.table({"id": [1, 2, 3, 4], "name": ["a", "b", None, "d"], "score": [0.5, 1.5, 2.5, 3.5]})
pq.write_table(t, "/tmp/pa_sample.parquet")

r = pq.read_table("/tmp/pa_sample.parquet")
print(r.schema)
print(r.equals(t))
print(r.to_pydict())
```
実行結果:
```
id: int64
name: string
score: double
True
{'id': [1, 2, 3, 4], 'name': ['a', 'b', None, 'd'], 'score': [0.5, 1.5, 2.5, 3.5]}
```

**注意点・落とし穴**:
- 上の例では往復後の `Table.equals` が `True` で、スキーマ(列名・型)も保持されている。
- 既定の圧縮は `snappy`、Parquet フォーマットのバージョンは `'2.6'`。

### `pq.read_table(columns=..., filters=...)`(列選択・行フィルタ)

**用途**: 読み込む列を絞る(列プルーニング)、行を条件で絞る(フィルタ)。Parquet は列指向なので、不要な列を読まないほど I/O が減る。

**シグネチャ**: `pq.read_table(source, *, columns=None, ..., filters=None, ...)`(`filters` は `pc.Expression`、または `[(列, 演算子, 値), ...]` のタプル形式)

**使用例**:
```python
r1 = pq.read_table("/tmp/pa_sample.parquet", columns=["id", "score"])
print(r1.column_names)

r2 = pq.read_table("/tmp/pa_sample.parquet", filters=[("score", ">", 1.0)])
print(r2.to_pydict())

import pyarrow.compute as pc
r3 = pq.read_table("/tmp/pa_sample.parquet", filters=(pc.field("id") >= 2) & (pc.field("name").is_valid()))
print(r3.to_pydict())

r4 = pq.read_table("/tmp/pa_sample.parquet", filters=[[("id", "=", 1)], [("id", "=", 4)]])   # OR(内側は AND)
print(r4.to_pydict())
```
実行結果:
```
['id', 'score']
{'id': [2, 3, 4], 'name': ['b', None, 'd'], 'score': [1.5, 2.5, 3.5]}
{'id': [2, 4], 'name': ['b', 'd'], 'score': [1.5, 3.5]}
{'id': [1, 4], 'name': ['a', 'd'], 'score': [0.5, 3.5]}
```

**注意点・落とし穴**:
- `filters` のタプル形式で内側のリストは AND、外側のリストは OR(DNF 形式)。式を使うほうが読みやすく、`ds.field` / `pc.field` のどちらでも通る。
- `columns` に含まれない列でも `filters` に使える(`columns=['name'], filters=[('score', '>', 1.0)]` は `{'name': ['b', None, 'd']}` を返すことを確認)。

### `pq.write_table` の主要オプション(`compression` / `row_group_size` / `use_dictionary` / `write_statistics`)

**用途**: 圧縮方式・行グループのサイズ・辞書エンコード・統計情報の有無を制御する。ファイルサイズと読み取り性能のトレードオフを調整する。

**シグネチャ**: `pq.write_table(table, where, row_group_size=None, version='2.6', use_dictionary=True, compression='snappy', write_statistics=True, compression_level=None, data_page_size=None, ...)`

**使用例**:
```python
import os, numpy as np
big = pa.table({"cat": np.random.default_rng(0).choice(["a", "b", "c"], 100_000),
                "val": np.arange(100_000)})
for comp in ["none", "snappy", "zstd", "gzip"]:
    pq.write_table(big, "/tmp/pa_big.parquet", compression=comp)
    print(comp.ljust(6), os.path.getsize("/tmp/pa_big.parquet"), pq.read_metadata("/tmp/pa_big.parquet").row_group(0).column(0).compression)

pq.write_table(big, "/tmp/pa_big.parquet", row_group_size=30_000)
md = pq.read_metadata("/tmp/pa_big.parquet")
print(md.num_row_groups, [md.row_group(i).num_rows for i in range(md.num_row_groups)])

pq.write_table(big, "/tmp/pa_big.parquet", compression={"cat": "zstd", "val": "snappy"})
print([pq.read_metadata("/tmp/pa_big.parquet").row_group(0).column(i).compression for i in range(2)])

small = pa.table({"name": ["a", "b", None, "d"], "score": [0.5, 1.5, 2.5, 3.5]})
pq.write_table(small, "/tmp/pa_opt.parquet", use_dictionary=False, write_statistics=False)
c = pq.read_metadata("/tmp/pa_opt.parquet").row_group(0).column(0)
print(c.has_dictionary_page, c.is_stats_set, c.statistics)
pq.write_table(small, "/tmp/pa_opt.parquet")
c = pq.read_metadata("/tmp/pa_opt.parquet").row_group(0).column(0)
print(c.has_dictionary_page, c.is_stats_set)
```
実行結果:
```
none   1029120 UNCOMPRESSED
snappy 629383 SNAPPY
zstd   318732 ZSTD
gzip   359170 GZIP
4 [30000, 30000, 30000, 10000]
['ZSTD', 'SNAPPY']
False False None
True True
```

**注意点・落とし穴**:
- `compression` は `'none'` / `'snappy'` / `'gzip'` / `'brotli'` / `'zstd'` / `'lz4'` から選べる(`lz4` と `brotli` も書き込めて、列メタデータの `compression` が `LZ4` / `BROTLI` になることを確認。`compression='foo'` のような未対応の値は `ArrowException: Unsupported compression: foo`)。上の例では `none`/`snappy`/`zstd`/`gzip` を実際に書いて、列メタデータの `compression` を確認した。列ごとに辞書で指定することもできる(上の最後の例)。速度重視なら `snappy`/`lz4`、サイズ重視なら `zstd`(`compression_level` で調整)が一般的な選択。
- `use_dictionary=False`(辞書エンコード無効)と `write_statistics=False`(統計情報無し)で書くと、上の例のように `has_dictionary_page` が `False`、`statistics` が `None` になる。統計情報が無いと `filters` による行グループスキップは効かない。
- `row_group_size` は「1行グループあたりの最大行数」(上の例では 100,000 行が 30000/30000/30000/10000 に分割された)。小さくしすぎるとメタデータのオーバーヘッドが増え、大きすぎるとフィルタによるスキップが効きにくくなる。

### `pq.read_schema` / `pq.read_metadata` / `ParquetFile.metadata`

**用途**: データ本体を読まずに、スキーマ・行数・行グループ数・各列の統計情報を調べる。大きなファイルの中身を確認するときの第一歩。

**シグネチャ**: `pq.read_schema(where, memory_map=False, decryption_properties=None, filesystem=None, arrow_extensions_enabled=True)` / `pq.read_metadata(where, memory_map=False, decryption_properties=None, filesystem=None, arrow_extensions_enabled=True)`

**使用例**:
```python
pq.write_table(t, "/tmp/pa_sample.parquet", row_group_size=2)
print(pq.read_schema("/tmp/pa_sample.parquet"))
md = pq.read_metadata("/tmp/pa_sample.parquet")
print(md.num_rows, md.num_columns, md.num_row_groups, md.format_version)
print(md.created_by)

rg = md.row_group(0)
print(rg.num_rows, rg.total_byte_size > 0)
col = rg.column(0)
print(col.path_in_schema, col.compression, col.physical_type)
st = col.statistics
print(st.has_min_max, st.min, st.max, st.null_count)
print(md.row_group(1).column(1).statistics.null_count)
```
実行結果:
```
id: int64
name: string
score: double
4 3 2 2.6
parquet-cpp-arrow version 25.0.1
2 True
id SNAPPY INT64
True 1 2 0
1
```

**注意点・落とし穴**:
- 統計情報(`min` / `max` / `null_count`)は行グループ×列ごとに保存される。行グループ単位のスキップに使われることは、同じファイル(2行グループ)に対し `ds.dataset(...)` の最初のフラグメントで `split_by_row_group(ds.field('id') > 3)` を呼ぶと行グループ 1 のみ、フィルタ無しの `split_by_row_group()` では 0 と 1 の両方が返ることで確認した。`write_statistics=False` で書いたファイルには統計情報が無い(`col.statistics` が `None`)。
- `md.created_by` は書き出したライブラリのバージョン文字列で、環境により異なる。

### `pq.ParquetFile`(行グループ単位・バッチ単位の読み込み)

**用途**: Parquet ファイルを開いたままにして、行グループ単位(`read_row_group`)またはバッチ単位(`iter_batches`)で少しずつ読む。メモリに載らない大きなファイルを扱うときに使う。

**シグネチャ**: `pq.ParquetFile(source, *, metadata=None, common_metadata=None, read_dictionary=None, memory_map=False, buffer_size=0, pre_buffer=True, ...)` / `ParquetFile.iter_batches(self, batch_size=65536, row_groups=None, columns=None, use_threads=True, use_pandas_metadata=False)` / `ParquetFile.read_row_group(self, i, columns=None, use_threads=True, use_pandas_metadata=False)`

**使用例**:
```python
pf = pq.ParquetFile("/tmp/pa_sample.parquet")     # row_group_size=2 で書いた 4 行(2 行グループ)
print(pf.metadata.num_row_groups, pf.schema_arrow.names)
print(pf.read_row_group(1).to_pydict())
print(pf.read_row_group(0, columns=["id"]).to_pydict())

for b in pf.iter_batches(batch_size=3):
    print(type(b).__name__, b.num_rows)
print(pf.read(columns=["score"]).num_rows)
print(sum(b.num_rows for b in pf.iter_batches(row_groups=[1])))
```
実行結果:
```
2 ['id', 'name', 'score']
{'id': [3, 4], 'name': [None, 'd'], 'score': [2.5, 3.5]}
{'id': [1, 2]}
RecordBatch 3
RecordBatch 1
4
2
```

**注意点・落とし穴**:
- `iter_batches(batch_size=n)` は行グループの境界とは無関係に n 行ずつ返す(上の例では 2 行グループが 2 つある 4 行のファイルで、`batch_size=3` は 3 行 + 1 行の2バッチになった)。行グループ単位で処理したい場合は `read_row_group` を使う。
- `ParquetFile` はファイルハンドルを保持する。使い終わったら `pf.close()`(シグネチャは `close(self, force=False)`)か、`with pq.ParquetFile(...) as pf:`(動作確認済み)で閉じる。

### `pq.ParquetWriter`(追記書き込み・ストリーミング書き出し)

**用途**: 複数の `Table`/`RecordBatch` を、1つの Parquet ファイルに順次書き込む(1 回の `write_table` ごとに行グループが増える)。メモリに載らないデータや、逐次生成されるデータの書き出しに使う。

**シグネチャ**: `pq.ParquetWriter(where, schema, filesystem=None, flavor=None, version='2.6', use_dictionary=True, compression='snappy', write_statistics=True, ...)`(`write_table(self, table, row_group_size=None)` / `write_batch(self, batch, row_group_size=None)` / `close()`)

**使用例**:
```python
schema = pa.schema([("id", pa.int64()), ("v", pa.float64())])
with pq.ParquetWriter("/tmp/pa_stream.parquet", schema) as w:
    for i in range(3):
        w.write_table(pa.table({"id": [i * 2, i * 2 + 1], "v": [i * 1.0, i * 1.0 + 0.5]}, schema=schema))

md = pq.read_metadata("/tmp/pa_stream.parquet")
print(md.num_rows, md.num_row_groups)
print(pq.read_table("/tmp/pa_stream.parquet").to_pydict())
try:
    with pq.ParquetWriter("/tmp/pa_stream2.parquet", schema) as w:
        w.write_table(pa.table({"id": ["x"], "v": [1.0]}))
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[0])
```
実行結果:
```
6 3
{'id': [0, 1, 2, 3, 4, 5], 'v': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]}
ValueError Table schema does not match schema used to create file: 
```

**注意点・落とし穴**:
- 書き込む `Table` のスキーマは `ParquetWriter` に渡したスキーマと一致している必要がある(不一致だと `ValueError: Table schema does not match schema used to create file`)。
- `with` を使う(または `close()` を呼ぶ)ことでフッターが書き込まれて有効な Parquet ファイルになる。`close()` 前のファイルを読むと `ArrowInvalid: ... Parquet magic bytes not found in footer.` になることを確認した。

### `pq.write_to_dataset` / ディレクトリ読み込み(パーティション付き Parquet)

**用途**: 列の値でディレクトリを分けて書き出す(Hive 形式: `col=value/`)。読み込み時にパーティション列が復元される。

**シグネチャ**: `pq.write_to_dataset(table, root_path, partition_cols=None, filesystem=None, schema=None, partitioning=None, basename_template=None, use_threads=None, file_visitor=None, existing_data_behavior=None, **kwargs)`

**使用例**:
```python
import os, glob, shutil
tt = pa.table({"year": [2023, 2023, 2024, 2024], "city": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
shutil.rmtree("/tmp/pa_part", ignore_errors=True)
pq.write_to_dataset(tt, "/tmp/pa_part", partition_cols=["year"])
for root, dirs, files in sorted(os.walk("/tmp/pa_part")):
    print(root.replace("/tmp/pa_part", "."), [f.rsplit(".", 1)[-1] for f in sorted(files)])
print(pq.ParquetFile(glob.glob("/tmp/pa_part/*/*.parquet")[0]).schema_arrow.names)   # ファイル内にはパーティション列が無い

r = pq.read_table("/tmp/pa_part")
print(r.schema)
print(r.sort_by("v").to_pydict())
print(pq.read_table("/tmp/pa_part", filters=[("year", "=", 2024)]).sort_by("v").to_pydict())
```
実行結果:
```
. []
./year=2023 ['parquet']
./year=2024 ['parquet']
['city', 'v']
city: string
v: int64
year: dictionary<values=int32, indices=int32, ordered=0>
{'city': ['a', 'b', 'a', 'b'], 'v': [1, 2, 3, 4], 'year': [2023, 2023, 2024, 2024]}
{'city': ['a', 'b'], 'v': [3, 4], 'year': [2024, 2024]}
```

**注意点・落とし穴**:
- パーティション列(`year`)はファイル内のデータからは除かれ、ディレクトリ名にだけ入る(上の `schema_arrow.names` に `year` が無い)。読み込み時にディレクトリ名から復元されるが、**型は `dictionary<values=int32, indices=int32>`(元は `int64`)に推論される**。元の型で読みたい場合は、`pq.read_table(path, partitioning=ds.partitioning(pa.schema([("year", pa.int64())]), flavor="hive"))` のようにスキーマを明示する(`int64` で読めることを確認)。読んだ後の `cast` でも戻せる。
- `partitioning=None` を指定して読むとディレクトリ名からの復元は行われず、列は `['v']` のみ(`pq.read_table(path, partitioning=None)`)。
- ファイル名は既定でランダムな英数字列 + `-0.parquet` になる。`basename_template="part-{i}.parquet"` で制御できる。

### スキーマメタデータの保存・pandas メタデータ(`replace_schema_metadata`)

**用途**: Parquet ファイルに任意のキー・値のメタデータを持たせる。pandas から書き出したファイルには、インデックス情報などの pandas メタデータが自動で格納される。

**シグネチャ**: `Table.replace_schema_metadata(self, metadata=None)` / `pq.read_schema(where, ...)`

**使用例**:
```python
import pandas as pd

meta_t = t.replace_schema_metadata({"source": "sensor-1", "version": "3"})
pq.write_table(meta_t, "/tmp/pa_meta.parquet")
print(pq.read_schema("/tmp/pa_meta.parquet").metadata[b"source"])
print({k: v for k, v in pq.read_table("/tmp/pa_meta.parquet").schema.metadata.items() if k in (b"source", b"version")})

df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="key"))
tp = pa.Table.from_pandas(df)
print(tp.column_names)              # インデックスも列として保存される
pq.write_table(tp, "/tmp/pa_pd.parquet")
print(pq.read_table("/tmp/pa_pd.parquet").to_pandas())
print(list(pq.read_schema("/tmp/pa_pd.parquet").metadata.keys()))
```
実行結果:
```
b'sensor-1'
{b'source': b'sensor-1', b'version': b'3'}
['a', 'key']
     a
key   
10   1
20   2
[b'pandas']
```

**注意点・落とし穴**:
- メタデータのキー・値は `bytes` で読み出される(`b"source"`)。
- pandas の `DataFrame` から作った `Table` には `b"pandas"` キーのメタデータが付き、`to_pandas()` 時にインデックスが復元される。メタデータがあることで、Parquet を往復しても pandas 側の型・インデックスが保たれる。

## CSV・JSON・Feather・IPC

### `csv.read_csv` / `csv.write_csv`

**用途**: CSV を `Table` として高速(マルチスレッド)に読み込む/`Table` を CSV に書き出す。型は自動推論される。

**シグネチャ**: `csv.read_csv(input_file, read_options=None, parse_options=None, convert_options=None, memory_pool=None)` / `csv.write_csv(data, output_file, write_options=None, memory_pool=None)`

**使用例**:
```python
import pyarrow as pa
import pyarrow.csv as pacsv

with open("/tmp/pa_sample.csv", "w") as f:
    f.write("id,name,score,joined\n1,Alice,0.5,2024-01-01\n2,Bob,,2024-02-15\n3,,2.5,2024-03-31\n")

t = pacsv.read_csv("/tmp/pa_sample.csv")
print(t.schema)
print(t.to_pydict())

pacsv.write_csv(t, "/tmp/pa_out.csv")
print(open("/tmp/pa_out.csv").read())
```
実行結果:
```
id: int64
name: string
score: double
joined: date32[day]
{'id': [1, 2, 3], 'name': ['Alice', 'Bob', ''], 'score': [0.5, None, 2.5], 'joined': [datetime.date(2024, 1, 1), datetime.date(2024, 2, 15), datetime.date(2024, 3, 31)]}
"id","name","score","joined"
1,"Alice",0.5,2024-01-01
2,"Bob",,2024-02-15
3,"",2.5,2024-03-31
```

**注意点・落とし穴**:
- `2024-01-01` 形式の列は `date32` に推論される。空欄は数値列では null になるが、文字列列では既定で**空文字列**のまま(上の `name` の3行目が `''`)。文字列列でも空欄や `NA` を null にしたい場合は `ConvertOptions(strings_can_be_null=True)` を指定する。
- `write_csv` は既定でヘッダと文字列値を `"` で囲んで出力し、null は空欄になる(上の出力)。`WriteOptions(quoting_style=...)` で変更できる。

### `csv.ReadOptions` / `csv.ParseOptions` / `csv.ConvertOptions`

**用途**: CSV の読み込み挙動を制御する3つのオプションクラス。読み取り(ヘッダ・エンコーディング・行スキップ)、構文解析(区切り文字・引用符)、変換(列の型・null 表現・対象列)に分かれている。

**シグネチャ**: `csv.ReadOptions(use_threads=None, *, block_size=None, skip_rows=None, skip_rows_after_names=None, column_names=None, autogenerate_column_names=None, encoding='utf8')` / `csv.ParseOptions(delimiter=None, *, quote_char=None, double_quote=None, escape_char=None, newlines_in_values=None, ignore_empty_lines=None, invalid_row_handler=None)` / `csv.ConvertOptions(check_utf8=None, *, column_types=None, default_column_type=None, null_values=None, true_values=None, false_values=None, decimal_point=None, strings_can_be_null=None, quoted_strings_can_be_null=None, include_columns=None, include_missing_columns=None, auto_dict_encode=None, auto_dict_max_cardinality=None, timestamp_parsers=None)`(`inspect.signature` では取れないため docstring のシグネチャ行)

**使用例**:
```python
with open("/tmp/pa_opt.tsv", "w") as f:
    f.write("# comment line\nid\tname\tflag\tamount\n001\tA\tyes\t1,5\n002\tB\tno\tNA\n")

t = pacsv.read_csv(
    "/tmp/pa_opt.tsv",
    read_options=pacsv.ReadOptions(skip_rows=1),
    parse_options=pacsv.ParseOptions(delimiter="\t"),
    convert_options=pacsv.ConvertOptions(
        column_types={"id": pa.string(), "amount": pa.float64()},
        null_values=["NA"],
        true_values=["yes"], false_values=["no"],
        decimal_point=",",
    ),
)
print(t.schema)
print(t.to_pydict())

# ヘッダなしCSV
with open("/tmp/pa_nohead.csv", "w") as f:
    f.write("1,x\n2,y\n")
print(pacsv.read_csv("/tmp/pa_nohead.csv", read_options=pacsv.ReadOptions(column_names=["n", "s"])).to_pydict())
print(pacsv.read_csv("/tmp/pa_nohead.csv", read_options=pacsv.ReadOptions(autogenerate_column_names=True)).column_names)
print(pacsv.read_csv("/tmp/pa_sample.csv", convert_options=pacsv.ConvertOptions(include_columns=["score", "id"])).column_names)
```
実行結果:
```
id: string
name: string
flag: bool
amount: double
{'id': ['001', '002'], 'name': ['A', 'B'], 'flag': [True, False], 'amount': [1.5, None]}
{'n': [1, 2], 's': ['x', 'y']}
['f0', 'f1']
['score', 'id']
```

**注意点・落とし穴**:
- `column_types` で明示しないと、`001` のような先頭ゼロ付き ID は整数として読まれ `1` になる(確認済み)。ID 列は文字列型で指定する。
- `null_values` を指定すると既定の null 表現が**置き換えられる**(追加ではない)。実際に、数値列 `1, null, 3, NA` は既定なら `[1, None, 3, None]`(`int64`)だが、`null_values=['NA']` を指定すると `null` が値として残り、列全体が `string` 型になった。
- 文字列列では `null_values` は `strings_can_be_null=True` を併用しないと効かない(併用しないと `'NA'` や `'-'` はそのまま文字列として残る)。
- `include_columns` で指定した順序が結果の列順になる(上の例は `['score', 'id']`)。
- `autogenerate_column_names=True` の列名は `f0`, `f1`, ... になる。

### `csv.open_csv`(ストリーミング読み込み)

**用途**: CSV を `RecordBatch` 単位で逐次読み込む(`CSVStreamingReader`)。メモリに収まらない大きな CSV を処理するときに使う。

**シグネチャ**: `csv.open_csv(input_file, read_options=None, parse_options=None, convert_options=None, memory_pool=None)`

**使用例**:
```python
with open("/tmp/pa_long.csv", "w") as f:
    f.write("a,b\n")
    for i in range(1000):
        f.write(f"{i},{i * 2}\n")

reader = pacsv.open_csv("/tmp/pa_long.csv", read_options=pacsv.ReadOptions(block_size=2000))
print(type(reader).__name__, reader.schema)
total, n = 0, 0
for batch in reader:
    n += 1
    total += batch.num_rows
print(n > 1, total)
print(pacsv.read_csv("/tmp/pa_long.csv")["b"].to_pylist()[-1])
```
実行結果:
```
CSVStreamingReader a: int64
b: int64
True 1000
1998
```

**注意点・落とし穴**:
- `open_csv` は型推論を**最初のブロックだけ**で行う。1000行の整数の後に `oops` がある CSV を `block_size=1000` で `open_csv` すると、スキーマは `int64` と推論され、イテレート中に `ArrowInvalid: In CSV column #0: CSV conversion error to int64: invalid value 'oops'` になった。同じファイルを `read_csv` すると全体を見て `string` 型になり成功した。ストリーミングでは `ConvertOptions(column_types=...)` で型を明示するのが安全。
- `block_size` は既定 1MiB。小さくするとバッチ数が増える(上の例では `n > 1` が `True`)。

### `json.read_json`(改行区切り JSON)

**用途**: 改行区切り JSON(NDJSON / JSON Lines)を `Table` として読み込む。ネストしたオブジェクトは struct、配列は list 型になる。

**シグネチャ**: `json.read_json(input_file, read_options=None, parse_options=None, memory_pool=None)`

**使用例**:
```python
import pyarrow.json as pajson

with open("/tmp/pa_data.jsonl", "w") as f:
    f.write('{"id": 1, "tags": ["a", "b"], "user": {"name": "Alice", "age": 30}}\n')
    f.write('{"id": 2, "tags": [], "user": {"name": "Bob"}}\n')
    f.write('{"id": 3, "tags": null, "extra": 1.5}\n')

t = pajson.read_json("/tmp/pa_data.jsonl")
print(t.schema)
print(t.to_pylist())
```
実行結果:
```
id: int64
tags: list<item: string>
  child 0, item: string
user: struct<name: string, age: int64>
  child 0, name: string
  child 1, age: int64
extra: double
[{'id': 1, 'tags': ['a', 'b'], 'user': {'name': 'Alice', 'age': 30}, 'extra': None}, {'id': 2, 'tags': [], 'user': {'name': 'Bob', 'age': None}, 'extra': None}, {'id': 3, 'tags': None, 'user': None, 'extra': 1.5}]
```

**注意点・落とし穴**:
- 1行に1つの JSON オブジェクトという形式(JSON Lines)のみ対応。全体が1つの JSON 配列(`[{"a":1},{"a":2}]`)のファイルは `ArrowInvalid: JSON parse error: Column() changed from object to array in row 0` になる。
- 行ごとにキーが違っても、全行のキーの和集合を列とし、欠けた値は null になる(上の例の `extra` や `user.age`)。

### `feather.write_feather` / `feather.read_table`

**用途**: Feather V2(= Arrow IPC ファイル形式)で書き出し/読み込みする。非圧縮で書けば、メモリマップによるゼロコピー読み込みができる(メモリ管理の章の `memory_map` を参照)。

**シグネチャ**: `feather.write_feather(df, dest, compression=None, compression_level=None, chunksize=None, version=2)` / `feather.read_table(source, columns=None, memory_map=False, use_threads=True)` / `feather.read_feather(source, columns=None, use_threads=True, memory_map=False, **kwargs)`(pandas を返す)

**使用例**:
```python
import pyarrow.feather as feather

t = pa.table({"id": [1, 2, 3], "name": ["a", "b", None]})
feather.write_feather(t, "/tmp/pa_sample.feather")
r = feather.read_table("/tmp/pa_sample.feather")
print(r.equals(t), r.schema.names)
print(feather.read_table("/tmp/pa_sample.feather", columns=["name"]).to_pydict())
print(type(feather.read_feather("/tmp/pa_sample.feather")).__name__)   # pandas.DataFrame を返す

feather.write_feather(t, "/tmp/pa_sample_z.feather", compression="zstd")
print(feather.read_table("/tmp/pa_sample_z.feather").equals(t))
```
実行結果:
```
True ['id', 'name']
{'name': ['a', 'b', None]}
DataFrame
True
```

**注意点・落とし穴**:
- `write_feather` の `compression` は `None`(既定)のとき、`lz4_frame` コーデックが使えれば `lz4` で圧縮される(`pyarrow/feather.py` のソースで確認)。指定できるのは `'uncompressed'` / `'lz4'` / `'zstd'` のみで、`'gzip'` などは `ValueError: compression="gzip" not supported, must be one of {...}` になる。
- 名前に反して `feather.read_feather` は pandas の `DataFrame` を返す。`Table` が欲しいときは `feather.read_table`。

### `ipc.new_file` / `ipc.open_file`(Arrow IPC ファイル形式)

**用途**: Arrow の IPC ファイル形式(ランダムアクセス可能)で書き出し/読み込みする。バッチ単位で `get_batch(i)` できる。プロセス間・言語間での Arrow データの受け渡しに使う。

**シグネチャ**: `ipc.new_file(sink, schema, *, options=None, metadata=None)` / `ipc.open_file(source, footer_offset=None, *, options=None, memory_pool=None)`

**使用例**:
```python
import pyarrow.ipc as ipc

schema = pa.schema([("id", pa.int64()), ("v", pa.string())])
with ipc.new_file("/tmp/pa_sample.arrow", schema) as w:
    w.write_batch(pa.record_batch({"id": [1, 2], "v": ["a", "b"]}, schema=schema))
    w.write_batch(pa.record_batch({"id": [3], "v": ["c"]}, schema=schema))
    w.write_table(pa.table({"id": [4, 5, 6], "v": ["d", "e", "f"]}, schema=schema))

with ipc.open_file("/tmp/pa_sample.arrow") as r:
    print(r.num_record_batches, r.schema.names)
    print(r.get_batch(1).to_pydict())
    print(r.read_all().num_rows)
```
実行結果:
```
3 ['id', 'v']
{'id': [3], 'v': ['c']}
6
```

**注意点・落とし穴**:
- `write_batch` を2回、`write_table` を1回(1チャンクの `Table`)呼ぶと、フッター上のバッチ数 `num_record_batches` は 3 だった。
- IPC ファイル形式はフッターを持つため、ランダムアクセス(`get_batch(i)`)と `read_all()` の両方が使える。ストリーム形式(下の `new_stream`)は前から順に読むしかできない。

### `ipc.new_stream` / `ipc.open_stream`(Arrow IPC ストリーム形式)/ `Table` のシリアライズ

**用途**: Arrow の IPC ストリーム形式で書き出し/読み込みする。ソケット・パイプ・メモリ上のバッファなど、順次読み書きするだけの用途に向く。`BufferOutputStream` と組み合わせれば、`Table` を `bytes` に直列化できる。

**シグネチャ**: `ipc.new_stream(sink, schema, *, options=None)` / `ipc.open_stream(source, *, options=None, memory_pool=None)`

**使用例**:
```python
import pyarrow.ipc as ipc

t = pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})

sink = pa.BufferOutputStream()
with ipc.new_stream(sink, t.schema) as w:
    w.write_table(t)
buf = sink.getvalue()
print(type(buf).__name__, buf.size > 0)

reader = ipc.open_stream(buf)
print(reader.schema.names)
for b in reader:
    print(b.num_rows, b.to_pydict())

print(ipc.open_stream(buf.to_pybytes()).read_all().equals(t))   # bytes からも読める
```
実行結果:
```
Buffer True
['id', 'v']
3 {'id': [1, 2, 3], 'v': ['a', 'b', 'c']}
True
```

**注意点・落とし穴**:
- 戻り値の `reader`(`RecordBatchStreamReader`)は反復可能で、`read_all()` で `Table` にまとめることもできる。読み切ったあとの `read_all()` は 0 行になる(先頭に戻せない)。
- `buf.to_pybytes()` で Python の `bytes` を得られる(コピーが発生する)。`buf` 自体は Arrow の `Buffer` で、そのまま `ipc.open_stream` に渡せる。

## Dataset API(pyarrow.dataset)

### `ds.write_dataset`(パーティション付きで書き出し)

**用途**: `Table`(または `RecordBatchReader` / `Dataset`)を、パーティション列ごとにディレクトリを分けたファイル群として書き出す。`pq.write_to_dataset` より高機能で、ファイル数・サイズの制御や `RecordBatchReader` からのストリーミング書き出しができる。

**シグネチャ**: `ds.write_dataset(data, base_dir, *, basename_template=None, format=None, partitioning=None, partitioning_flavor=None, schema=None, filesystem=None, file_options=None, use_threads=True, preserve_order=False, max_partitions=None, max_open_files=None, max_rows_per_file=None, min_rows_per_group=None, max_rows_per_group=None, file_visitor=None, existing_data_behavior='error', create_dir=True)`

**使用例**:
```python
import os, shutil
import pyarrow as pa
import pyarrow.dataset as ds

t = pa.table({
    "year": [2023, 2023, 2024, 2024, 2024],
    "city": ["tokyo", "osaka", "tokyo", "osaka", "tokyo"],
    "sales": [10, 20, 30, 40, 50],
})
shutil.rmtree("/tmp/pa_ds", ignore_errors=True)
ds.write_dataset(t, "/tmp/pa_ds", format="parquet",
                 partitioning=["year"], partitioning_flavor="hive",
                 basename_template="part-{i}.parquet")
for root, dirs, files in sorted(os.walk("/tmp/pa_ds")):
    print(root.replace("/tmp/pa_ds", "."), sorted(files))
```
実行結果:
```
. []
./year=2023 ['part-0.parquet']
./year=2024 ['part-0.parquet']
```

**注意点・落とし穴**:
- `partitioning=["year"]` のようにフィールド名のリストを渡した場合、`partitioning_flavor="hive"` を付けると `year=2023/` 形式に、付けないと `2023/` 形式(ディレクトリ名に値のみ)になる。
- `basename_template` には `{i}` を含める必要がある(ファイル番号に置換される)。
- `existing_data_behavior` の既定は `'error'` で、出力先に既存データがあると例外になる(次項参照)。

### `ds.dataset(source, format=..., partitioning=...)`(Dataset の作成と基本情報)

**用途**: ファイルやディレクトリ(複数ファイル・パーティション付き)を1つの論理テーブル(`Dataset`)として開く。この時点ではデータ本体は読まれず、スキーマとファイル一覧だけが解決される(遅延評価)。

**シグネチャ**: `ds.dataset(source, schema=None, format=None, filesystem=None, partitioning=None, partition_base_dir=None, exclude_invalid_files=None, ignore_prefixes=None)`

**使用例**:
```python
dataset = ds.dataset("/tmp/pa_ds", format="parquet", partitioning="hive")
print(type(dataset).__name__)
print(dataset.schema)
print(sorted(os.path.relpath(f, "/tmp/pa_ds") for f in dataset.files))
print(dataset.count_rows())
print(dataset.to_table().sort_by("sales").to_pydict())
```
実行結果:
```
FileSystemDataset
city: string
sales: int64
year: int32
['year=2023/part-0.parquet', 'year=2024/part-0.parquet']
5
{'city': ['tokyo', 'osaka', 'tokyo', 'osaka', 'tokyo'], 'sales': [10, 20, 30, 40, 50], 'year': [2023, 2023, 2024, 2024, 2024]}
```

**注意点・落とし穴**:
- `partitioning="hive"` を指定すると、ディレクトリ名 `year=2023` から `year` 列が復元される(型は推論され、上の出力では `int32`。`pq.read_table` で同じディレクトリを読むと `dictionary<values=int32, ...>` になるのと対照的)。`partitioning` を指定しない(既定 `None`)と、Hive 形式のディレクトリでも `year` 列は現れず `['city', 'sales']` だけになる。
- ファイル名が `_` または `.` で始まるもの(`_hidden.parquet`、`.tmp`)は既定で無視される(`ignore_prefixes`)ことを確認した。

### `Dataset.to_table(columns=..., filter=...)`(列選択・フィルタ pushdown)

**用途**: 列を絞り、`ds.field` の式で行を絞って読み込む。パーティション列に対する条件は、該当しないファイル(フラグメント)の除外に使われる(`get_fragments(filter=...)` の例を参照)。Parquet では行グループ統計による絞り込みも効く(`pq.read_schema` / `read_metadata` の項を参照)。

**シグネチャ**: `Dataset.to_table(self, columns=None, filter=None, batch_size=131072, batch_readahead=16, fragment_readahead=4, fragment_scan_options=None, use_threads=True, cache_metadata=True, memory_pool=None)` / `ds.field(*name_or_index)`

**使用例**:
```python
r = dataset.to_table(columns=["city", "sales"], filter=ds.field("year") == 2024)
print(r.column_names)
print(r.sort_by("sales").to_pydict())

expr = (ds.field("sales") >= 20) & ds.field("city").isin(["tokyo"])
print(dataset.to_table(filter=expr).sort_by("sales").to_pydict())

print(dataset.count_rows(filter=ds.field("year") == 2023))
print(dataset.to_table(filter=ds.field("year") == 1999).num_rows)
```
実行結果:
```
['city', 'sales']
{'city': ['tokyo', 'osaka', 'tokyo'], 'sales': [30, 40, 50]}
{'city': ['tokyo', 'tokyo'], 'sales': [30, 50], 'year': [2024, 2024]}
2
0
```

**注意点・落とし穴**:
- `ds.field("x")` と `pc.field("x")` は同じ式オブジェクトで、どちらでも使える。
- 条件に合う行が無ければ、0 行でスキーマ付きの `Table` が返る(上の最終行は `0`)。
- `columns` に含めない列でも `filter` で使える。

### `Dataset.scanner` / `Scanner.to_batches` / `Dataset.head` / `Dataset.take`

**用途**: 読み込み計画(`Scanner`)を作り、バッチ単位で逐次処理したり先頭だけを取ったりする。大きな Dataset をメモリに載せずに処理する基本パターン。

**シグネチャ**: `Dataset.scanner(self, columns=None, filter=None, batch_size=131072, batch_readahead=16, fragment_readahead=4, fragment_scan_options=None, use_threads=True, cache_metadata=True, memory_pool=None)` / `Dataset.to_batches(...)` / `Dataset.head(self, num_rows, columns=None, filter=None, ...)` / `Dataset.take(self, indices, columns=None, filter=None, ...)`

**使用例**:
```python
sc = dataset.scanner(columns=["sales"], filter=ds.field("sales") > 10, batch_size=2)
print(type(sc).__name__, sc.projected_schema)
print(sc.count_rows())
print([b.num_rows for b in sc.to_batches()])
print(dataset.head(2, columns=["city", "sales"]).to_pydict())
print(dataset.take([0, 4], columns=["sales"]).to_pydict())

total = 0
for batch in dataset.to_batches(columns=["sales"]):
    total += sum(batch.column(0).to_pylist())
print(total)
print(sc.to_reader().read_all().num_rows)
```
実行結果:
```
Scanner sales: int64
4
[1, 2, 1]
{'city': ['tokyo', 'osaka'], 'sales': [10, 20]}
{'sales': [10, 50]}
150
4
```

**注意点・落とし穴**:
- `Scanner` の `to_table()` / `to_batches()` は何度呼んでもその都度走査し直す(同じ `Scanner` で2回呼んで同じ結果になることを確認)。一方 `sc.to_reader()` が返す `RecordBatchReader` は読み切り型で、`read_all()` を2回呼ぶと2回目は 0 行だった。
- `batch_size` は各バッチの最大行数。上の例(2ファイル、`batch_size=2`、フィルタ後 4 行)のバッチ行数は `[1, 2, 1]` で、ファイルをまたいで詰め直されるわけではなかった。
- `dataset.take(indices)` の `indices` はデータセット全体での行位置(ファイル順に連結した通し番号)。

### `ds.partitioning`(パーティション方式の指定)

**用途**: パーティションのディレクトリ規則(Hive 形式か、ディレクトリ名のみの形式か)と、パーティション列の型を明示する。推論に任せると型が意図とずれるときに使う。

**シグネチャ**: `ds.partitioning(schema=None, field_names=None, flavor=None, dictionaries=None)`

**使用例**:
```python
shutil.rmtree("/tmp/pa_ds2", ignore_errors=True)
ds.write_dataset(t, "/tmp/pa_ds2", format="parquet",
                 partitioning=ds.partitioning(pa.schema([("year", pa.int16()), ("city", pa.string())])))   # flavor なし
for root, dirs, files in sorted(os.walk("/tmp/pa_ds2")):
    if not dirs:
        print(os.path.relpath(root, "/tmp/pa_ds2"))

d2 = ds.dataset("/tmp/pa_ds2", format="parquet",
                partitioning=ds.partitioning(pa.schema([("year", pa.int16()), ("city", pa.string())])))
print(d2.schema)

d3 = ds.dataset("/tmp/pa_ds", format="parquet",
                partitioning=ds.partitioning(pa.schema([("year", pa.int64())]), flavor="hive"))
print(d3.schema.field("year").type)
```
実行結果:
```
2023/osaka
2023/tokyo
2024/osaka
2024/tokyo
sales: int64
year: int16
city: string
int64
```

**注意点・落とし穴**:
- `flavor` を省略した `ds.partitioning(schema)` は「ディレクトリ名 = 値のみ」の形式(`2023/tokyo/`)。Hive 形式(`year=2023/`)にするには `flavor="hive"`。
- ディレクトリ名のみの形式では、読み込み時のフィールド順が書き込み時と同じでないと値が別の列に割り当てられる。`(city, year)` の順で読むと `ArrowInvalid: error parsing 'osaka' as scalar of type int16` になることを確認した。

### 式による射影(`columns` に辞書を渡して列を計算)

**用途**: `to_table` / `scanner` の `columns` に「新しい列名 → 式」の辞書を渡すと、読み込み時に列を計算できる。スキャン中に処理されるため、不要なデータを Python に持ち込まずに済む。

**シグネチャ**: `Dataset.to_table(columns={名前: 式, ...}, filter=...)`(`columns` にリストまたは辞書を渡せる)

**使用例**:
```python
import pyarrow.compute as pc

r = dataset.to_table(
    columns={
        "city_upper": pc.utf8_upper(ds.field("city")),
        "sales_x2": ds.field("sales") * 2,
        "year": ds.field("year"),
    },
    filter=ds.field("sales") > 30,
)
print(r.schema)
print(r.sort_by("sales_x2").to_pydict())
```
実行結果:
```
city_upper: string
sales_x2: int64
year: int32
{'city_upper': ['OSAKA', 'TOKYO'], 'sales_x2': [80, 100], 'year': [2024, 2024]}
```

**注意点・落とし穴**:
- 辞書のキーが出力の列名になる。式にはスカラー関数(要素ごとに計算する `pc.*`)が使える。集約関数は使えず、`columns={'s': pc.sum(ds.field('sales'))}` は `ArrowInvalid: ExecuteScalarExpression cannot Execute non-scalar expression sum(sales)` になる。

### `write_dataset` の制御(`max_rows_per_file` / `existing_data_behavior` / `file_visitor`)

**用途**: 1ファイルの最大行数、既存データがあるときの挙動、書き出したファイルごとのコールバックを指定する。

**シグネチャ**: `ds.write_dataset(data, base_dir, ..., max_rows_per_file=None, max_rows_per_group=None, existing_data_behavior='error', file_visitor=None, ...)`(`existing_data_behavior` は `'error'` / `'overwrite_or_ignore'` / `'delete_matching'`)

**使用例**:
```python
big = pa.table({"g": ["x"] * 5 + ["y"] * 3, "v": list(range(8))})
shutil.rmtree("/tmp/pa_ds3", ignore_errors=True)
written = []
ds.write_dataset(big, "/tmp/pa_ds3", format="parquet", partitioning=["g"], partitioning_flavor="hive",
                 max_rows_per_file=2, max_rows_per_group=2,
                 basename_template="p-{i}.parquet",
                 file_visitor=lambda f: written.append(os.path.relpath(f.path, "/tmp/pa_ds3")))
print(sorted(written))
print(ds.dataset("/tmp/pa_ds3", format="parquet", partitioning="hive").count_rows())

try:
    ds.write_dataset(big, "/tmp/pa_ds3", format="parquet", partitioning=["g"], partitioning_flavor="hive")
except Exception as e:
    print(type(e).__name__, str(e).replace("/tmp/pa_ds3", "<dir>").split(":")[0])

ds.write_dataset(big, "/tmp/pa_ds3", format="parquet", partitioning=["g"], partitioning_flavor="hive",
                 existing_data_behavior="delete_matching")
print(sorted(os.listdir("/tmp/pa_ds3/g=x")))
```
実行結果:
```
['g=x/p-0.parquet', 'g=x/p-1.parquet', 'g=x/p-2.parquet', 'g=y/p-0.parquet', 'g=y/p-1.parquet']
8
ArrowInvalid Could not write to <dir> as the directory is not empty and existing_data_behavior is to error
['part-0.parquet']
```

**注意点・落とし穴**:
- `existing_data_behavior='error'`(既定): 出力先に既存ファイルがあると `ArrowInvalid: Could not write to <dir> as the directory is not empty and existing_data_behavior is to error`(上の3つ目の出力)。`'overwrite_or_ignore'`: 既存ファイルはそのまま残して追加で書く(`basename_template` を変えて2回書くと、`g=x/a-0.parquet` と `g=x/b-0.parquet` が共存し、データが重複することを確認した。同名ファイルは上書きされる)。`'delete_matching'`: 今回書き込むパーティションのディレクトリ内の既存ファイルを削除してから書く(`g=x` だけを書き直すと `g=x` の旧ファイルだけが消え、`g=y` のファイルは残ることを確認した)。
- `max_rows_per_file` を小さくするときは `max_rows_per_group` も同じかそれ以下にする必要がある。`max_rows_per_file=1, max_rows_per_group=2` は `ArrowInvalid: max_rows_per_group must be less than or equal to max_rows_per_file` になる。既定の `max_rows_per_group` は大きいため、`max_rows_per_file` を単独で小さく指定するとこのエラーになりやすい。

### `Dataset.get_fragments` / `Fragment`(ファイル単位の操作)

**用途**: Dataset を構成するファイル(フラグメント)を個別に扱う。フィルタに該当するフラグメントだけを列挙したり、フラグメントごとにパーティション値やメタデータを取り出したりする。

**シグネチャ**: `FileSystemDataset.get_fragments(self, filter=None)` / `ParquetFileFragment.metadata` / `Fragment.partition_expression` / `Fragment.to_table(...)` / `Fragment.count_rows(...)`

**使用例**:
```python
for frag in dataset.get_fragments():
    print(os.path.relpath(frag.path, "/tmp/pa_ds"), frag.partition_expression, frag.count_rows())

matched = list(dataset.get_fragments(filter=ds.field("year") == 2024))
print(len(matched), matched[0].metadata.num_rows, matched[0].metadata.num_row_groups)
print(matched[0].to_table(columns=["sales"]).to_pydict())
```
実行結果:
```
year=2023/part-0.parquet (year == 2023) 2
year=2024/part-0.parquet (year == 2024) 3
1 3 1
{'sales': [30, 40, 50]}
```

**注意点・落とし穴**:
- `get_fragments(filter=...)` にパーティション列の条件を渡すと、該当しないディレクトリのフラグメントは列挙されない(ファイルを開く前の判定)。
- `frag.to_table()` の列は `['city', 'sales']` で、パーティション列 `year` は含まれない(ファイル内の物理列のみ。`frag.physical_schema` も同じ)。パーティション値は `frag.partition_expression` から取得する。`dataset.to_table()` ならパーティション列が付く。

### 複数ファイル・インメモリデータから Dataset を作る(`ds.dataset` のリスト入力・`ds.InMemoryDataset`)

**用途**: パスのリスト、複数のテーブルなどから `Dataset` を作る。ディレクトリ構造を持たない任意のファイルの組を1つのテーブルとして扱える。

**シグネチャ**: `ds.dataset(source, schema=None, format=None, ...)`(`source` はパス、パスのリスト、`Table` / `RecordBatch` のリスト)/ `ds.InMemoryDataset(source, Schema schema=None)`(docstring のシグネチャ行)

**使用例**:
```python
import pyarrow.parquet as pq
pq.write_table(pa.table({"a": [1, 2]}), "/tmp/pa_p1.parquet")
pq.write_table(pa.table({"a": [3], "b": ["x"]}), "/tmp/pa_p2.parquet")

d = ds.dataset(["/tmp/pa_p1.parquet", "/tmp/pa_p2.parquet"], format="parquet")
print(d.schema)                         # 先頭ファイルのスキーマが採用される
print(d.to_table().to_pydict())

d_mem = ds.dataset([pa.table({"a": [1, 2]}), pa.table({"a": [3]})])
print(type(d_mem).__name__, d_mem.count_rows())

unified = pa.unify_schemas([pq.read_schema("/tmp/pa_p1.parquet"), pq.read_schema("/tmp/pa_p2.parquet")])
d_u = ds.dataset(["/tmp/pa_p1.parquet", "/tmp/pa_p2.parquet"], format="parquet", schema=unified)
print(d_u.to_table().to_pydict())
```
実行結果:
```
a: int64
{'a': [1, 2, 3]}
InMemoryDataset 3
{'a': [1, 2, 3], 'b': [None, None, 'x']}
```

**注意点・落とし穴**:
- スキーマを指定しないと**最初のファイルのスキーマ**が全体に適用され、後続ファイルにだけある列(上の `b`)は無視される。ファイル間でスキーマが異なる場合は、`pa.unify_schemas` で統合したスキーマを `schema=` に渡す(欠けた列は null になる)。

## pandas・NumPy・polars 連携

### `pa.Table.from_pandas` / `Table.to_pandas`

**用途**: pandas の `DataFrame` と Arrow の `Table` を相互変換する。`from_pandas` はインデックスと dtype 情報を「pandas メタデータ」としてスキーマに保存し、`to_pandas` で復元する。

**シグネチャ**: `Table.from_pandas(df, schema=None, preserve_index=None, nthreads=None, columns=None, safe=True)` / `Table.to_pandas(self, memory_pool=None, categories=None, strings_to_categorical=False, zero_copy_only=False, integer_object_nulls=False, date_as_object=True, timestamp_as_object=False, use_threads=True, deduplicate_objects=True, ignore_metadata=False, safe=True, split_blocks=False, self_destruct=False, maps_as_pydicts=None, types_mapper=None, coerce_temporal_nanoseconds=False)`(主要な引数)

**使用例**:
```python
import numpy as np
import pandas as pd
import pyarrow as pa

df = pd.DataFrame({
    "i": [1, 2, 3],
    "f": [1.5, np.nan, 3.0],
    "s": ["a", None, "c"],
    "d": pd.to_datetime(["2024-01-01", "2024-01-02", None]),
    "n": pd.array([1, None, 3], dtype="Int64"),
})
t = pa.Table.from_pandas(df)
print(t.schema.remove_metadata())
print(t.to_pandas().dtypes)

# 素の Arrow テーブル(pandas メタデータ無し)を pandas にすると
raw = pa.table({"i": [1, None, 3], "s": ["a", "b", None]})
print(raw.to_pandas().dtypes)
print(raw.to_pandas())
```
実行結果:
```
i: int64
f: double
s: large_string
d: timestamp[us]
n: int64
i             int64
f           float64
s               str
d    datetime64[us]
n             Int64
dtype: object
i    float64
s        str
dtype: object
     i    s
0  1.0    a
1  NaN    b
2  3.0  NaN
```

**注意点・落とし穴**:
- pandas 3.0 の `str` 列(文字列)は Arrow では **`large_string`** に変換される(上の `s: large_string`)。`pa.string()` ではない点に注意(`pq.write_table` で保存したファイル側の型や、結合・`concat_tables` 時の型一致に影響する)。
- null を含む整数列は、pandas 側では既定で `float64` + NaN になる(上の `raw.to_pandas()` の `i` 列)。整数のまま保持したい場合は次項の `types_mapper=pd.ArrowDtype` を使う。
- `from_pandas` 由来のテーブルには `pandas` メタデータがあり、往復すると dtype(`str`, `Int64`, `datetime64[us]`)が復元される。メタデータの無い素のテーブルからは既定の対応で変換される。

### `types_mapper=pd.ArrowDtype` / `pd.ArrowDtype`(Arrow 型を保ったまま pandas 化)

**用途**: Arrow の型を保持した pandas の `ArrowDtype` 列として変換する。null を含む整数が NaN 化(float 化)されず、`int64[pyarrow]` のまま欠損は `<NA>` で表される。

**シグネチャ**: `Table.to_pandas(self, ..., types_mapper=None, ...)` / `pd.ArrowDtype(pyarrow_dtype)`

**使用例**:
```python
raw = pa.table({"i": [1, None, 3], "s": ["a", "b", None]})
pdf = raw.to_pandas(types_mapper=pd.ArrowDtype)
print(pdf.dtypes)
print(pdf)
print(pdf["i"].sum(), pdf["i"].isna().tolist())

# pandas -> Arrow でも型が保たれる
print(pa.Table.from_pandas(pdf).schema.types)
# 値ごとに対応を変える(整数だけ ArrowDtype)
print(raw.to_pandas(types_mapper={pa.int64(): pd.ArrowDtype(pa.int64())}.get).dtypes)
```
実行結果:
```
i     int64[pyarrow]
s    string[pyarrow]
dtype: object
      i     s
0     1     a
1  <NA>     b
2     3  <NA>
4 [False, True, False]
[DataType(int64), DataType(string)]
i    int64[pyarrow]
s               str
dtype: object
```

**注意点・落とし穴**:
- `types_mapper` は「Arrow の型 → pandas の dtype(または `None`)」を返す関数(辞書の `.get` も使える)。`None` を返した型は既定の変換になる(上の最終行で `s` 列が `str` のまま)。
- `ArrowDtype` 列の欠損は NaN ではなく `<NA>` で、`dtype` の表示も `int64[pyarrow]` のようになる。NumPy 系の dtype(`float64` など)と欠損値の扱いが異なるため、後続の処理で NaN 前提のコードがある場合は注意。

### 日付・時刻の変換(`date_as_object` / `coerce_temporal_nanoseconds`)

**用途**: `to_pandas` 時の日付・時刻型の変換方式を制御する。`date32` は既定で Python の `datetime.date` オブジェクトの列(`object`)になる。

**シグネチャ**: `Table.to_pandas(self, ..., date_as_object=True, timestamp_as_object=False, coerce_temporal_nanoseconds=False, ...)`

**使用例**:
```python
import datetime as dt

t = pa.table({
    "d": pa.array([dt.date(2024, 1, 1), None]),
    "ts": pa.array([dt.datetime(2024, 1, 1, 12), None], pa.timestamp("s")),
})
print(t.to_pandas().dtypes)
print(t.to_pandas(date_as_object=False).dtypes)
print(t.to_pandas(coerce_temporal_nanoseconds=True).dtypes)
print(pa.table({"t": pa.array([dt.datetime(3000, 1, 1)])}).to_pandas().dtypes)
```
実行結果:
```
d            object
ts    datetime64[s]
dtype: object
d     datetime64[ms]
ts     datetime64[s]
dtype: object
d             object
ts    datetime64[ns]
dtype: object
t    datetime64[us]
dtype: object
```

**注意点・落とし穴**:
- 既定では `date32` は `object` 列(要素は `datetime.date`)になる。`date_as_object=False` にすると `datetime64[ms]` の列になる。
- `timestamp[s]` は既定では `datetime64[s]` のまま変換される(上の出力)。ナノ秒に揃えたい場合は `coerce_temporal_nanoseconds=True`(`datetime64[ns]` になる)。ただし `date32` 列は `coerce_temporal_nanoseconds=True` でも `object` のままだった。
- 既定ではナノ秒への強制をしないため、`year 3000` のようにナノ秒表現の範囲(約1677〜2262年)を超える日時も `datetime64[us]` として変換できた(最終行)。

### `Array.to_numpy` / `pa.array(ndarray)`(NumPy とのゼロコピー)

**用途**: Arrow ⇔ NumPy 間の変換。null を含まない数値配列は、メモリをコピーせずに共有できる。

**シグネチャ**: `Array.to_numpy(self, zero_copy_only=True, writable=False)` / `pa.array(obj, ...)`

**使用例**:
```python
n = np.arange(5)
arr = pa.array(n)                       # ndarray -> Arrow(ゼロコピー)
n[0] = 100
print(arr.to_pylist())
print(np.shares_memory(n, arr.to_numpy()))
print(arr.to_numpy().flags.writeable)   # Arrow 由来の NumPy 配列は読み取り専用

a = pa.array([1, None, 3])
try:
    a.to_numpy()
except Exception as e:
    print(type(e).__name__, e)
print(a.to_numpy(zero_copy_only=False))         # null は NaN の float に変換(コピー)

s = pa.array(["a", "b"])
try:
    s.to_numpy()
except Exception as e:
    print(type(e).__name__, e)
print(s.to_numpy(zero_copy_only=False))
```
実行結果:
```
[100, 1, 2, 3, 4]
True
False
ArrowInvalid Needed to copy 1 chunks with 1 nulls, but zero_copy_only was True
[ 1. nan  3.]
ArrowInvalid Needed to copy 1 chunks with 0 nulls, but zero_copy_only was True
['a' 'b']
```

**注意点・落とし穴**:
- `to_numpy()` の既定は `zero_copy_only=True` で、ゼロコピーできない(null を含む/文字列/bool の `Array`。`bool` は `ArrowInvalid: Zero copy conversions not possible with boolean types`)場合は `ArrowInvalid: Needed to copy 1 chunks with 1 nulls, but zero_copy_only was True` のように例外になる。`zero_copy_only=False` にするとコピーして変換する(整数+null は `float64` + NaN になる)。
- `pa.array(ndarray)` はゼロコピーで、元の NumPy 配列を書き換えると Arrow 側にも反映される(上の例で先頭が `100`)。Arrow → NumPy の結果は読み取り専用(`writeable` が `False`)なので、書き換えたいときは `.copy()` するか、`to_numpy(zero_copy_only=False, writable=True)`(コピーして書き込み可能な配列を得る。元の Arrow 配列は変わらないことを確認)を使う。`zero_copy_only=True` のまま `writable=True` を指定すると `ValueError: Cannot return a writable array if asking for zero-copy` になる。
- `pa.array(np.array([1.0, np.nan]))` の NaN は null にならない(`null_count == 0`)が、`pa.array(pd.Series([1.0, np.nan]))` では NaN が null になる(`null_count == 1`)。同じ「NaN を含む float 配列」でも入力の型で扱いが変わる。

### polars との相互変換(`pl.from_arrow` / `DataFrame.to_arrow`)

**用途**: Arrow の `Table` と polars の `DataFrame` を相互変換する。両者とも Arrow 列指向のメモリレイアウトを基盤にしている。

**シグネチャ**: `pl.from_arrow(data, schema=None, *, schema_overrides=None, rechunk=True)` / `polars.DataFrame.to_arrow(self, *, compat_level=None)`

**使用例**:
```python
import polars as pl

t = pa.table({"i": [1, None, 3], "s": ["a", "b", None], "l": [[1], [2, 3], None]})
p = pl.from_arrow(t)
print(p.schema)
back = p.to_arrow()
print(back.schema)
print(pl.DataFrame(t).shape)                        # コンストラクタに Arrow テーブルを渡すのも可
print(pl.from_arrow(pa.chunked_array([[1, 2], [3]])).to_list())
```
実行結果:
```
Schema({'i': Int64, 's': String, 'l': List(Int64)})
i: int64
s: large_string
l: large_list<item: int64>
  child 0, item: int64
(3, 3)
[1, 2, 3]
```

**注意点・落とし穴**:
- polars 側の文字列は Arrow に戻すと **`large_string`**、リストは **`large_list`** になる(`pa.string()` / `pa.list_()` には戻らない)。往復すると元のスキーマと一致しない点に注意(上の例の `t` を往復した結果との `Table.equals` は `False` だが、元のスキーマへ `cast` し直すと `True` になることを確認)。
- polars の `Categorical` は `dictionary<values=large_string, indices=uint32>` になり、`_PL_CATEGORICAL2` というフィールドメタデータが付く。

### Arrow PyCapsule インターフェース(`__arrow_c_stream__`)による受け渡し

**用途**: `pa.table(obj)` は、`__arrow_c_stream__` を持つオブジェクト(pandas 3.0 の `DataFrame`、polars の `DataFrame` など)を、専用の変換コード無しで `Table` にできる。ライブラリ間の Arrow データ受け渡しの共通規格。

**シグネチャ**: `pa.table(data, ...)` / `pa.RecordBatchReader.from_stream(data, schema=None)`

**使用例**:
```python
import polars as pl
import pyarrow.compute as pc

p = pl.DataFrame({"s": ["a", "bb", None], "n": [1, 2, 3]})
print(hasattr(p, "__arrow_c_stream__"), hasattr(pd.DataFrame({"a": [1]}), "__arrow_c_stream__"),
      hasattr(pa.table({"a": [1]}), "__arrow_c_stream__"))

t_p = pa.table(p)                                   # polars -> Arrow(PyCapsule 経由)
print(t_p.schema)
print(p.to_arrow().schema)                          # 同じデータでも to_arrow() は別の型

r = pa.RecordBatchReader.from_stream(p)             # ストリームとして受け取る
print(r.read_all().num_rows)

try:
    pc.utf8_upper(t_p["s"])
except Exception as e:
    print(type(e).__name__, e)
print(pc.utf8_upper(t_p.cast(pa.schema([("s", pa.string()), ("n", pa.int64())]))["s"]).to_pylist())
```
実行結果:
```
True True True
s: string_view
n: int64
s: large_string
n: int64
3
ArrowNotImplementedError Function 'utf8_upper' has no kernel matching input types (string_view)
['A', 'BB', None]
```

**注意点・落とし穴**:
- polars の `DataFrame` を `pa.table(p)` で受け取ると、文字列列は **`string_view`** 型になる(`p.to_arrow()` は `large_string`)。25.0.1 では `pc.utf8_upper` / `pc.utf8_length` などの文字列カーネルが `string_view` に未対応で、`ArrowNotImplementedError: Function 'utf8_upper' has no kernel matching input types (string_view)` になる。`cast` で `pa.string()` に直してから使う。
- `string_view` の列は `pq.write_table` で書き込み・読み戻しでき(型は `string_view` のまま)、`to_pandas()` も可能(`str`)。ただし通常の `string` 列を持つテーブルとは `concat_tables` できない(スキーマ不一致エラー)。
- `pa.table(pandas_df)` も同様に動作する(この場合は `from_pandas` と同様に pandas メタデータが付く)。

## メモリ管理・バッファ

### `Array.nbytes` / `get_total_buffer_size()` / `buffers()`

**用途**: 配列が使うメモリ量と、その内訳(有効性ビットマップ・データ・オフセット)を調べる。`nbytes` は「その配列が参照するバイト範囲」、`get_total_buffer_size()` は「裏のバッファ全体のサイズ」。

**シグネチャ**: `Array.nbytes`(プロパティ)/ `Array.get_total_buffer_size(self)` / `Array.buffers(self)` / `Table.nbytes` / `Table.get_total_buffer_size(self)`

**使用例**:
```python
import numpy as np
import pyarrow as pa

a = pa.array(np.arange(1000, dtype="int64"))
print(a.nbytes)
print([None if b is None else b.size for b in a.buffers()])   # [有効性ビットマップ, データ]
print([None if b is None else b.size for b in pa.array([1, None, 3]).buffers()])

s = a.slice(10, 5)                          # ゼロコピーのスライス
print(len(s), s.nbytes, s.get_total_buffer_size())
print(pa.concat_arrays([s]).get_total_buffer_size())   # コピーして小さい配列にする
```
実行結果:
```
8000
[None, 8000]
[1, 24]
5 40 8000
40
```

**注意点・落とし穴**:
- null が無い配列の `buffers()` の先頭(有効性ビットマップ)は `None`。null があるとビットマップのバッファが付く(3要素で 1 バイト、データは `int64` × 3 = 24 バイト。上の3行目の出力 `[1, 24]`)。
- `Array.slice` は元のバッファを共有する(ゼロコピー)ため、5 要素のスライスでも `get_total_buffer_size()` は元の 8000 バイトのまま。`nbytes` は 40 バイト(スライス範囲のみ)。小さなスライスを長期間保持すると元の巨大バッファが解放されない。`pa.concat_arrays([s])` のようにコピーすれば 40 バイトになる。
- `Table.nbytes` も同様にオフセットを考慮した範囲の合計で、`Table.get_total_buffer_size()` は参照するバッファ全体の合計。

### `pa.default_memory_pool()` / `pa.total_allocated_bytes()`

**用途**: Arrow が確保しているメモリ量を確認する。NumPy との連携(ゼロコピー)でメモリを確保しているかどうかの検証や、メモリリーク調査に使える。

**シグネチャ**: `pa.default_memory_pool()` / `MemoryPool.bytes_allocated(self)` / `MemoryPool.backend_name` / `pa.total_allocated_bytes()` / `pa.supported_memory_backends()`

**使用例**:
```python
import numpy as np
import pyarrow as pa

pool = pa.default_memory_pool()
print(pool.backend_name, pa.supported_memory_backends())

base = pool.bytes_allocated()
x = pa.array(np.arange(1_000_000, dtype="int64"))     # NumPy から(ゼロコピー)
print(pool.bytes_allocated() - base)

y = pa.array(list(range(1_000_000)))                   # Python リストから(新規確保)
print(pool.bytes_allocated() - base)
del y
print(pool.bytes_allocated() - base, pa.total_allocated_bytes() - base)
```
実行結果:
```
mimalloc ['mimalloc', 'jemalloc', 'system']
0
8000000
0 0
```

**注意点・落とし穴**:
- 既定のメモリプールのバックエンドは環境依存(この環境では `mimalloc`)。環境変数 `ARROW_DEFAULT_MEMORY_POOL` に `system` / `jemalloc` / `mimalloc` を指定するとバックエンドを切り替えられる(それぞれ指定して `backend_name` が変わることを確認)。
- 上の例で、NumPy 配列からの変換は 0 バイト(ゼロコピー)、100万要素の Python リストからの変換は 8,000,000 バイト(`int64` × 1,000,000)を新規確保し、`del` で 0 に戻った。
- `bytes_allocated()` は Arrow のメモリプール経由で確保した分のみで、NumPy など外部のバッファを参照している場合は含まれない。
- 上の出力は他のオブジェクトを解放した影響を受けないよう、新しい Python プロセスで単独実行した結果。既に多くの Arrow オブジェクトが生きている(または解放される)セッションでは差分がずれることがある。

### `pa.py_buffer` / `pa.allocate_buffer` / `Buffer`

**用途**: Arrow の `Buffer`(連続したバイト列)を作る。Python の `bytes` や NumPy 配列を、コピーなしで `Buffer` として包む(`py_buffer`)。

**シグネチャ**: `pa.py_buffer(obj)` / `pa.allocate_buffer(size, memory_pool=None, resizable=False)` / `Buffer.to_pybytes()` / `Buffer.slice(offset=0, length=None)` / `Buffer.size` / `Buffer.address` / `Array.from_buffers(type, length, buffers, null_count=-1, offset=0, children=None)`

**使用例**:
```python
b = pa.py_buffer(b"hello world")
print(b.size, b.is_mutable, b.to_pybytes())
print(b.slice(6).to_pybytes(), b[0:5].to_pybytes())
print(memoryview(b).tobytes())

ba = bytearray(b"abc")
pb = pa.py_buffer(ba)                     # bytearray を共有
ba[0] = ord("X")
print(pb.to_pybytes(), pb.is_mutable)

buf = pa.allocate_buffer(8)
print(buf.size, buf.is_mutable)

# Buffer から配列を組み立てる(バリデーション無しの低レベル API)
data = pa.py_buffer(np.array([1, 2, 3], dtype="int32"))
print(pa.Array.from_buffers(pa.int32(), 3, [None, data]).to_pylist())
```
実行結果:
```
11 False b'hello world'
b'world' b'hello'
b'hello world'
b'Xbc' True
8 True
[1, 2, 3]
```

**注意点・落とし穴**:
- `bytes` から作った `Buffer` は読み取り専用(`is_mutable=False`)、`bytearray` や NumPy 配列から作ったものは書き込み可能で元のオブジェクトとメモリを共有する(上の例で `bytearray` の書き換えが `Buffer` に見える)。
- `Buffer.to_pybytes()` は `bytes` にコピーする。ゼロコピーで見たいときは `memoryview(buffer)` を使う。
- `Array.from_buffers` は低レベル API で、バッファの中身が型・長さと整合していることは呼び出し側の責任になる。通常は `pa.array` を使う。

### `pa.BufferOutputStream` / `pa.BufferReader`(メモリ上の入出力ストリーム)

**用途**: ファイルの代わりにメモリ上のバッファに読み書きするストリーム。Parquet の `write_table` / `read_table` や IPC の `new_stream` / `open_stream` に渡せるため、ファイルを作らないテストや、バイト列としてデータを受け渡したいときに使う。

**シグネチャ**: `pa.BufferOutputStream()`(`write(data)` / `tell()` / `getvalue()`)/ `pa.BufferReader(obj)`(`read(nbytes=None)` / `tell()`)

**使用例**:
```python
import pyarrow.parquet as pq

sink = pa.BufferOutputStream()
sink.write(b"abc")
sink.write(b"def")
print(sink.tell(), sink.getvalue().to_pybytes())

rd = pa.BufferReader(sink.getvalue())
print(rd.read(2), rd.tell(), rd.read())

# Parquet をメモリ上で往復
t = pa.table({"a": [1, 2, 3]})
mem = pa.BufferOutputStream()
pq.write_table(t, mem)
pbuf = mem.getvalue()
print(pbuf.size > 0)
print(pq.read_table(pa.BufferReader(pbuf)).equals(t))
```
実行結果:
```
6 b'abcdef'
b'ab' 2 b'cdef'
True
True
```

**注意点・落とし穴**:
- `getvalue()` は `Buffer` を返す(`bytes` ではない)。`bytes` が必要なら `.to_pybytes()`。`getvalue()` を呼ぶとストリームは閉じられ、以降の `write` は `ValueError: I/O operation on closed file` になる(確認済み)。書き終えてから呼ぶこと。

### `pa.memory_map` / `memory_map=True`(メモリマップによるゼロコピー読み込み)

**用途**: ファイルをメモリにマップして、Arrow データをコピー無しで読む。非圧縮の Feather/IPC ファイルで有効で、読み込みでメモリをほぼ消費しない。

**シグネチャ**: `pa.memory_map(path, mode='r')` / `feather.read_table(source, columns=None, memory_map=False, use_threads=True)` / `pq.read_table(..., memory_map=False, ...)`

**使用例**:
```python
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

t = pa.table({"a": np.arange(100_000)})
feather.write_feather(t, "/tmp/pa_mm.arrow", compression="uncompressed")   # 圧縮しないことが重要

before = pa.total_allocated_bytes()
with pa.memory_map("/tmp/pa_mm.arrow") as mm:
    r = ipc.open_file(mm).read_all()
    print(r.num_rows, pa.total_allocated_bytes() - before)

r2 = feather.read_table("/tmp/pa_mm.arrow", memory_map=True)
print(pa.total_allocated_bytes() - before)
r3 = feather.read_table("/tmp/pa_mm.arrow", memory_map=False)
print(pa.total_allocated_bytes() - before)
```
実行結果:
```
100000 0
0
800384
```

**注意点・落とし穴**:
- メモリマップ経由で読んだ場合、Arrow のメモリプールでの追加確保は 0 バイト、通常読み込み(`memory_map=False`)では 800,384 バイト(`int64` × 100,000 + α)増えた(他の `Table` を保持したまま実行すると差分がずれることがあるため、新しいプロセスで単独実行した結果)。データはファイルのページキャッシュを直接参照する。
- 圧縮された Feather(既定の `lz4`)や Parquet は、`memory_map=True` で読んでも Arrow のメモリプールに確保が発生する(同じ 100,000 行の `int64` で、非圧縮 Feather は 0 バイト、lz4 圧縮 Feather は 800,000 バイト、Parquet は 812,544 バイト)。ゼロコピーを狙うなら非圧縮の Feather/IPC にする。
- マップ元のファイルを読み込み中に書き換えたり削除したりしてはいけない。

## ファイルシステム・UDF

### `pyarrow.fs`(`LocalFileSystem` / `FileSelector` / `FileSystem.from_uri`)

**用途**: ローカル・S3・GCS・HDFS・Azure などを共通のインターフェースで扱う。`pq.read_table` や `ds.dataset` の `filesystem=` 引数に渡せる。

**シグネチャ**: `fs.LocalFileSystem(use_mmap=False)` / `fs.FileSelector(base_dir, allow_not_found=False, recursive=False)` / `FileSystem.get_file_info(paths_or_selector)` / `fs.FileSystem.from_uri(uri)`(`(filesystem, path)` を返す)/ `fs.S3FileSystem` / `fs.GcsFileSystem` など

**使用例**:
```python
import os
import pyarrow as pa
import pyarrow.fs as fs

os.makedirs("/tmp/pa_fsdir/sub", exist_ok=True)
open("/tmp/pa_fsdir/a.txt", "w").write("hi")
open("/tmp/pa_fsdir/sub/b.txt", "w").write("hello")

lf = fs.LocalFileSystem()
info = lf.get_file_info("/tmp/pa_fsdir/a.txt")
print(info.type.name, info.size, info.base_name)
print(lf.get_file_info("/tmp/pa_fsdir/none").type.name)

infos = lf.get_file_info(fs.FileSelector("/tmp/pa_fsdir", recursive=True))
print(sorted((i.base_name, i.type.name, i.size) for i in infos))

filesystem, path = fs.FileSystem.from_uri("file:///tmp/pa_fsdir/a.txt")
print(type(filesystem).__name__, path)

with lf.open_input_stream("/tmp/pa_fsdir/a.txt") as s:
    print(s.read())
```
実行結果:
```
File 2 a.txt
NotFound
[('a.txt', 'File', 2), ('b.txt', 'File', 5), ('sub', 'Directory', None)]
LocalFileSystem /tmp/pa_fsdir/a.txt
b'hi'
```

**注意点・落とし穴**:
- `FileSelector` は既定で非再帰(`recursive=False`)。`allow_not_found=False`(既定)のとき、存在しないディレクトリを指定すると例外になる。
- `from_uri` は S3 なら `s3://bucket/key` のような URI から適切な `FileSystem` と、その内側のパスに分解する。上の例では `file://` URI で `LocalFileSystem` が返る。
- 利用可能なファイルシステムクラスは `fs` モジュールに `S3FileSystem` / `GcsFileSystem` / `HadoopFileSystem` / `AzureFileSystem` として存在する(実際に使うにはそれぞれ認証情報や接続先が必要で、ここでは実行していない)。

### `pc.register_scalar_function`(Python 関数を計算カーネルとして登録)

**用途**: Python の関数をユーザー定義スカラー関数(UDF)として Arrow に登録し、`pc.call_function` から呼べるようにする。関数内では `pyarrow.compute` を使い、配列単位で処理するのが前提。

**シグネチャ**: `pc.register_scalar_function(func, function_name, function_doc, in_types, out_type, func_registry=None)`(`func` は `func(ctx, *args)` の形。`function_doc` は `{'summary': ..., 'description': ...}`、`in_types` は `{引数名: 型}`)

**使用例**:
```python
import pyarrow as pa
import pyarrow.compute as pc

def times2_plus(ctx, x, y):
    return pc.add(pc.multiply(x, 2), y)

pc.register_scalar_function(
    times2_plus, "times2_plus",
    {"summary": "2x + y", "description": "x を2倍して y を足す"},
    {"x": pa.int64(), "y": pa.int64()},
    pa.int64(),
)
print("times2_plus" in pc.list_functions())
print(pc.call_function("times2_plus", [pa.array([1, 2, None]), pa.array([10, 20, 30])]).to_pylist())
print(pc.call_function("times2_plus", [pa.chunked_array([[1], [2]]), pa.array([5, 6])]).to_pylist())

try:
    pc.call_function("times2_plus", [pa.array(["a"]), pa.array([1])])
except Exception as e:
    print(type(e).__name__, e)
try:
    pc.register_scalar_function(times2_plus, "times2_plus",
                                {"summary": "x", "description": "x"},
                                {"x": pa.int64(), "y": pa.int64()}, pa.int64())
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
True
[12, 24, None]
[7, 10]
ArrowNotImplementedError Function 'times2_plus' has no kernel matching input types (string, int64)
ArrowKeyError Already have a function registered with name: times2_plus
```

**注意点・落とし穴**:
- 登録した関数は、`pc.list_functions()` に現れ、`pc.call_function` から呼べる(`pc.times2_plus` のようなモジュール属性としては生えない)。
- 同名の関数を再登録すると `ArrowKeyError: Already have a function registered with name: ...` になる(同一プロセス内で登録は1回のみ)。
- 引数の型が `in_types` と合わない場合は `no kernel matching input types` エラー。
- `ChunkedArray` を渡しても呼び出せる(上の3行目)。関数内で要素ごとに Python ループを回すと遅くなるため、`pc.*` の組み合わせで書く。
