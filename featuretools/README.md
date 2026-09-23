# featuretools 逆引き辞書

featuretools 1.31.0 で検証済み

本ドキュメントに掲載しているシグネチャ・実行結果は、すべて検証専用の隔離venv(`featuretools==1.31.0` + `pandas<3.0` を新規インストールしたもの)で実際にコードを実行して得たものです。理由は次の通りです。

- **重要な既知の非互換**: 本来の検証対象である `/home/manaty/library-practicing/.venv`(pandas 3.0.5)では、`EntitySet.add_dataframe()` が **必ず例外を吐いて失敗します**。これは featuretools のバグではなく、pandas 3.0 が `DataFrame` 拡張アクセサ(`pandas.core.accessor.Accessor.__get__`)のインスタンスキャッシュを廃止したことが原因です。`df.ww` にアクセスするたびに毎回「新しい」`PandasTableAccessor` インスタンスが生成されるため、`df.ww.init(...)` で設定した schema がそのインスタンス限りで消えてしまい、直後の `df.ww.schema` や `df.ww.time_index` は `None`/未初期化のままになります(`woodwork.exceptions.WoodworkNotInitError` で失敗)。woodwork 最新版(0.31.0、2026-09時点で最新)でもこの問題は未修正であることを、実際に `pandas.core.accessor` のソースを読み `Accessor.__get__` が `object.__setattr__` によるキャッシュを行っていないことを確認し、さらに `df.ww.init(...)` 直後に `df.ww.schema` を取り直すと `None` に戻ることを実行して再現し、検証済みです。
- そのため、pandas 3.0.5 が入った共有venvを変更せず(他ライブラリの検証結果に影響するため)、`/tmp` 配下に featuretools 1.31.0 + pandas 2.3.3 の隔離venvを新規作成し、そちらで全エントリを実行・検証しました。**featuretools 1.31.0 を使う場合は pandas 2.x を使うこと**(`pip install featuretools` の依存指定は `pandas>=2.0.0` のみで上限がなく、pandas 3.0 系を誤って入れると `EntitySet` が機能しません)。

## 目次

1. [EntitySet構築](#entityset構築)
2. [特徴量生成(DFS)](#特徴量生成dfs)
3. [プリミティブ](#プリミティブ)
4. [特徴量選択・活用](#特徴量選択活用)
5. [時系列・リーク防止](#時系列リーク防止)
6. [特徴量の保存・読み込み](#特徴量の保存読み込み)
7. [その他](#その他)

---

## EntitySet構築

### `ft.EntitySet(id=None, dataframes=None, relationships=None)`

**用途**: 複数のDataFrame(テーブル)とそれらのリレーション(親子関係)をまとめて保持する、featuretoolsの中核オブジェクト。DFSはこのEntitySetを起点に特徴量を自動生成する。

**シグネチャ**: `ft.EntitySet(id=None, dataframes=None, relationships=None)`

**使用例**:
```python
import featuretools as ft
import pandas as pd

customers_df = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "join_date": pd.to_datetime(["2020-01-01", "2020-02-15", "2020-03-10"], format="%Y-%m-%d"),
    "region": ["east", "west", "east"],
})
transactions_df = pd.DataFrame({
    "transaction_id": range(1, 9),
    "customer_id": [1, 1, 1, 2, 2, 3, 3, 3],
    "transaction_time": pd.to_datetime([
        "2020-01-05", "2020-01-10", "2020-02-01",
        "2020-02-20", "2020-03-01",
        "2020-03-15", "2020-03-20", "2020-04-01",
    ], format="%Y-%m-%d"),
    "amount": [100.0, 50.5, 20.0, 300.0, 75.25, 60.0, 40.0, 90.0],
})

es = ft.EntitySet(id="transactions_es")
print(es)
```
実行結果:
```
Entityset: transactions_es
  DataFrames:
  Relationships:
    No relationships
```

**注意点・落とし穴**:
- `id` は必須ではないが、複数EntitySetを扱う際の識別用に指定しておくと `print(es)` の1行目に表示され分かりやすい。
- この時点ではDataFrameは1つも登録されていない。次に `add_dataframe` でテーブルを追加していく。

### `es.add_dataframe(...)`

**用途**: pandasのDataFrameをEntitySetに登録し、Woodwork(featuretoolsが内部で使う型システム)の型情報(index列・time_index列・論理型など)を付与する。

**シグネチャ**: `es.add_dataframe(dataframe, dataframe_name=None, index=None, logical_types=None, semantic_tags=None, make_index=False, time_index=None, secondary_time_index=None, already_sorted=False)`

**使用例**:
```python
es = ft.EntitySet(id="transactions_es")
es = es.add_dataframe(
    dataframe_name="customers",
    dataframe=customers_df,
    index="customer_id",
    time_index="join_date",
)
es = es.add_dataframe(
    dataframe_name="transactions",
    dataframe=transactions_df,
    index="transaction_id",
    time_index="transaction_time",
)
print(es)
print(es["transactions"].ww.schema)
```
実行結果:
```
Entityset: transactions_es
  DataFrames:
    customers [Rows: 3, Columns: 3]
    transactions [Rows: 8, Columns: 4]
  Relationships:
    No relationships
                 Logical Type Semantic Tag(s)
Column                                       
transaction_id        Integer       ['index']
customer_id           Integer     ['numeric']
transaction_time     Datetime  ['time_index']
amount                 Double     ['numeric']
```
(注: この時点では `add_relationship` をまだ呼んでいないため `customer_id` は単なる `numeric` 列であり、`Relationships` も `No relationships`。次の項目で `add_relationship` を呼ぶと `customer_id` に `foreign_key` タグが追加される)

**注意点・落とし穴**:
- `add_dataframe` は**戻り値のEntitySetを使うこと**(`es = es.add_dataframe(...)`)。メソッド自体は `self` を返す(`in-place` でも変更されるが、チェーンで書くのが公式スタイル)。
- `dataframe_name` は、渡すDataFrameがまだWoodwork初期化されていない場合は必須。
- **pandas 3.0系では、内部で `dataframe.ww.init(...)` を呼んだ直後に `dataframe.ww.time_index` を参照する処理があり、pandasのアクセサキャッシュ廃止(冒頭の注記参照)によりここで確実に `WoodworkNotInitError` が発生する。pandas 2.x を使うこと。**
- `logical_types` / `semantic_tags` を省略すると型は自動推論される。日付列は `pd.to_datetime` してから渡すと `format` を明示していないと `Could not infer format` という警告が出ることがある(実行結果には影響しないが、`format=` を明示するのが無難)。

### `es.add_relationship(...)` / `ft.Relationship(...)`

**用途**: 2つのDataFrame間の親子関係(外部キー)をEntitySetに登録する。DFSはこのリレーションを辿って `MEAN(transactions.amount)` のような集約特徴量を作る。

**シグネチャ**: `es.add_relationship(parent_dataframe_name=None, parent_column_name=None, child_dataframe_name=None, child_column_name=None, relationship=None)` / `ft.Relationship(entityset, parent_dataframe_name, parent_column_name, child_dataframe_name, child_column_name)`

**使用例**:
```python
es = es.add_relationship(
    parent_dataframe_name="customers",
    parent_column_name="customer_id",
    child_dataframe_name="transactions",
    child_column_name="customer_id",
)
print(es)

# Relationshipオブジェクトを直接渡す書き方も可能
rel = ft.Relationship(es, "customers", "customer_id", "transactions", "customer_id")
# es.add_relationship(relationship=rel)  # 上と同じ関係を重複追加するとエラーになるため実行はしない
```
実行結果:
```
Entityset: transactions_es
  DataFrames:
    customers [Rows: 3, Columns: 3]
    transactions [Rows: 8, Columns: 4]
  Relationships:
    transactions.customer_id -> customers.customer_id
```

**注意点・落とし穴**:
- 引数名を指定して呼ぶ(キーワード引数)か、`relationship=ft.Relationship(...)` のどちらかで呼ぶ。位置引数だけで両方混ぜることはできない。
- 子側の外部キー列(`transactions.customer_id`)は、リレーション追加後は自動的に `foreign_key` セマンティックタグが付与される。

### `es.normalize_dataframe(...)`

**用途**: 既存のDataFrameから新しいDataFrame(親テーブル)を切り出し、正規化する。例えば `transactions` の `product` 列から `products` テーブルを作り、リレーションも自動で張る。

**シグネチャ**: `es.normalize_dataframe(base_dataframe_name, new_dataframe_name, index, additional_columns=None, copy_columns=None, make_time_index=None, make_secondary_time_index=None, new_dataframe_time_index=None, new_dataframe_secondary_time_index=None)`

**使用例**:
```python
transactions_df2 = transactions_df.assign(product=["A", "B", "A", "A", "C", "B", "B", "A"])
es2 = ft.EntitySet(id="normalize_demo")
es2 = es2.add_dataframe(
    dataframe_name="transactions",
    dataframe=transactions_df2,
    index="transaction_id",
    time_index="transaction_time",
)
es2 = es2.normalize_dataframe(
    base_dataframe_name="transactions",
    new_dataframe_name="products",
    index="product",
)
print(es2)
print(es2["products"])
```
実行結果:
```
Entityset: normalize_demo
  DataFrames:
    transactions [Rows: 8, Columns: 5]
    products [Rows: 3, Columns: 2]
  Relationships:
    transactions.product -> products.product
  product first_transactions_time
A       A              2020-01-05
B       B              2020-01-10
C       C              2020-03-01
```

**注意点・落とし穴**:
- 元テーブルに `time_index` が設定されている場合、切り出した新テーブルには自動で `first_<元テーブル名>_time` という列(その`index`値が最初に出現した時刻)が追加される。
- `additional_columns` に指定した列は新テーブルに**移動**(元テーブルからは削除)され、`copy_columns` に指定した列は**両方に残る**という違いがある。

### `es.add_last_time_indexes(...)`

**用途**: 各DataFrameに「最後の関連イベント時刻」(last_time_index)を計算してセットする。`training_window` を使ったリーク防止DFSを行う前提条件。

**シグネチャ**: `es.add_last_time_indexes(updated_dataframes=None)`

**使用例**:
```python
es.add_last_time_indexes()
print(es["customers"].ww.schema)
print(es["customers"])
```
実行結果:
```
              Logical Type      Semantic Tag(s)
Column                                         
customer_id        Integer            ['index']
join_date         Datetime       ['time_index']
region             Unknown                   []
_ft_last_time     Datetime  ['last_time_index']
   customer_id  join_date region _ft_last_time
1            1 2020-01-01   east    2020-02-01
2            2 2020-02-15   west    2020-03-01
3            3 2020-03-10   east    2020-04-01
```

**注意点・落とし穴**:
- `_ft_last_time` という列が実際に追加され、その顧客に紐づく `transactions` の中で最も新しい `transaction_time`(=そのDataFrameの子孫全体の最終イベント時刻)が入る。顧客1なら最終取引が2020-02-01なので `_ft_last_time=2020-02-01` になる。
- `training_window` を `ft.dfs()` に渡すとき、事前に `add_last_time_indexes()` を呼んでいないと `UserWarning: Using training_window but last_time_index is not set for dataframe ...` が出た上、**集約特徴量がすべて0/NaNになる**(検証で実際に再現した。詳細は「時系列・リーク防止」章参照)。

### `es.query_by_values(...)`

**用途**: EntitySet内の特定DataFrameから、指定した列の値でフィルタした行を取得する。

**シグネチャ**: `es.query_by_values(dataframe_name, instance_vals, column_name=None, columns=None, time_last=None, training_window=None, include_cutoff_time=True)`

**使用例**:
```python
print(es.query_by_values("transactions", instance_vals=[1], column_name="customer_id"))
```
実行結果:
```
                transaction_id  customer_id transaction_time  amount
transaction_id                                                      
1                            1            1       2020-01-05   100.0
2                            2            1       2020-01-10    50.5
3                            3            1       2020-02-01    20.0
```

---

## 特徴量生成(DFS)

### `ft.dfs(...)`

**用途**: Deep Feature Synthesis。EntitySet内のリレーションを自動的に辿り、集約特徴量(`MEAN`, `COUNT` など)や変換特徴量(`MONTH`, `DAY` など)を大量に自動生成する、featuretoolsの中心機能。

**シグネチャ**: `ft.dfs(dataframes=None, relationships=None, entityset=None, target_dataframe_name=None, cutoff_time=None, instance_ids=None, agg_primitives=None, trans_primitives=None, groupby_trans_primitives=None, allowed_paths=None, max_depth=2, ignore_dataframes=None, ignore_columns=None, primitive_options=None, seed_features=None, drop_contains=None, drop_exact=None, where_primitives=None, max_features=-1, cutoff_time_in_index=False, save_progress=None, features_only=False, training_window=None, approximate=None, chunk_size=None, n_jobs=1, dask_kwargs=None, verbose=False, return_types=None, progress_callback=None, include_cutoff_time=True)`(主要引数のみ抜粋)

**使用例**:
```python
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    agg_primitives=["mean", "sum", "count"],
    trans_primitives=["month"],
    max_depth=2,
)
print(feature_matrix)
print([f.get_name() for f in feature_defs])
```
実行結果:
```
             COUNT(transactions)  MEAN(transactions.amount)  SUM(transactions.amount) MONTH(join_date)
customer_id                                                                                           
1                              3                  56.833333                    170.50                1
2                              2                 187.625000                    375.25                2
3                              3                  63.333333                    190.00                3
['COUNT(transactions)', 'MEAN(transactions.amount)', 'SUM(transactions.amount)', 'MONTH(join_date)']
```

**注意点・落とし穴**:
- `target_dataframe_name` に指定したテーブルの各行(この例では顧客)が特徴量行列の1行になり、`agg_primitives` はそのテーブルから見て子テーブル(`transactions`)方向の集約に、`trans_primitives` はテーブル自身の列(`join_date`)に適用される。
- 戻り値は `(feature_matrix, feature_defs)` のタプル。`feature_defs` は `FeatureBase` のリストで、`calculate_feature_matrix` に再利用したり `save_features` で保存できる。
- `features_only=True` を指定すると特徴量の計算をスキップし `feature_defs` のリストだけを高速に取得できる(型・列の組み合わせを事前確認したい時に有用)。

### `ft.calculate_feature_matrix(...)`

**用途**: `dfs()` の `features_only=True` などで事前に得た特徴量定義(`feature_defs`)を使って、実際に特徴量行列を計算する。同じ特徴量セットを異なる `cutoff_time` や異なるEntitySetに対して再計算したい場合に使う。

**シグネチャ**: `ft.calculate_feature_matrix(features, entityset=None, cutoff_time=None, instance_ids=None, dataframes=None, relationships=None, cutoff_time_in_index=False, training_window=None, approximate=None, save_progress=None, verbose=False, chunk_size=None, n_jobs=1, dask_kwargs=None, progress_callback=None, include_cutoff_time=True)`

**使用例**:
```python
fm2 = ft.calculate_feature_matrix(features=feature_defs, entityset=es)
print(fm2.head())
```
実行結果:
```
             COUNT(transactions)  MEAN(transactions.amount)  SUM(transactions.amount) MONTH(join_date)
customer_id                                                                                           
1                              3                  56.833333                    170.50                1
2                              2                 187.625000                    375.25                2
3                              3                  63.333333                    190.00                3
```

**注意点・落とし穴**:
- `features` に渡す `FeatureBase` のリストは、それを生成したEntitySetのスキーマ(列名・型)と整合している必要がある。別のEntitySetに使い回す場合はテーブル構成が一致している必要がある。

### `ft.describe_feature(...)`

**用途**: 生成された特徴量オブジェクトを、人間可読な自然文の説明に変換する。大量に自動生成された特徴量の意味を後から確認するのに便利。

**シグネチャ**: `ft.describe_feature(feature, feature_descriptions=None, primitive_templates=None, metadata_file=None)`

**使用例**:
```python
target_feat = [f for f in feature_defs if f.get_name() == "MEAN(transactions.amount)"][0]
print(ft.describe_feature(target_feat))
```
実行結果:
```
The average of the "amount" of all instances of "transactions" for each "customer_id" in "customers".
```

---

## プリミティブ

### `ft.list_primitives()`

**用途**: featuretoolsに標準搭載されている全プリミティブ(集約用・変換用の演算)を一覧表示する。

**シグネチャ**: `ft.list_primitives()`

**使用例**:
```python
lp = ft.list_primitives()
print(lp.shape)
print(lp.columns.tolist())
print(lp["type"].value_counts())
print(lp[lp["name"] == "mean"][["name", "type", "description", "valid_inputs", "return_type"]])
```
実行結果:
```
(203, 5)
['name', 'type', 'description', 'valid_inputs', 'return_type']
type
transform      138
aggregation     65
Name: count, dtype: int64
    name         type                                  description                                   valid_inputs                                    return_type
47  mean  aggregation  Computes the average for a list of values.  <ColumnSchema (Semantic Tags = ['numeric'])>  <ColumnSchema (Semantic Tags = ['numeric'])>
```

**注意点・落とし穴**:
- featuretools 1.31.0時点で203個(集約65 / 変換138)のプリミティブが標準搭載されている。`dfs()` の `agg_primitives`/`trans_primitives` にはここに出てくる `name` の文字列(小文字スネークケース)をそのまま渡せる。

### `ft.get_valid_primitives(...)`

**用途**: 指定したEntitySet・対象テーブルに対して、実際に適用可能な(列の型と整合する)プリミティブだけを絞り込んで取得する。

**シグネチャ**: `ft.get_valid_primitives(entityset, target_dataframe_name, max_depth=2, selected_primitives=None, **dfs_kwargs)`(内部で `(agg_primitives, trans_primitives)` の2リストを返す)

**使用例**:
```python
agg_valid, trans_valid = ft.get_valid_primitives(es, target_dataframe_name="customers")
print("agg count:", len(agg_valid), "trans count:", len(trans_valid))
print([p.name for p in agg_valid[:5]])
```
実行結果:
```
agg count: 65 trans count: 99
['count_outside_range', 'percent_unique', 'count_below_mean', 'skew', 'min']
```

**注意点・落とし穴**:
- 戻り値はプリミティブの**名前の文字列リストではなくクラスオブジェクトのリスト**(`p.name` でスネークケース名を取得する)。
- `customers` テーブルには `region`(カテゴリ)・`join_date`(日時)・`customer_id`(数値/index)しか列がないため、`amount` のような数値列を要求するプリミティブ(`sum`, `trend` など)はここには出てこない。

### プリミティブ引数(`agg_primitives` / `trans_primitives` / `primitive_options`)

**用途**: `dfs()` に渡す `agg_primitives`/`trans_primitives` で使うプリミティブの種類を絞り込み、`primitive_options` でプリミティブごとに適用対象の列やテーブルをさらに細かく制御する。

**シグネチャ**: `ft.dfs(..., agg_primitives=None, trans_primitives=None, primitive_options=None, ...)`

**使用例**:
```python
fm_opt, defs_opt = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    agg_primitives=["mean", "count"],
    trans_primitives=[],
    primitive_options={"mean": {"include_columns": {"transactions": ["amount"]}}},
    max_depth=1,
)
print(fm_opt)
```
実行結果:
```
             COUNT(transactions)  MEAN(transactions.amount)
customer_id                                                
1                              3                  56.833333
2                              2                 187.625000
3                              3                  63.333333
```

**注意点・落とし穴**:
- `primitive_options` を指定しない場合、`mean` は数値列すべてに適用されようとする。この例では `amount` しか数値列がないため見た目の差は出ないが、複数の数値列があるテーブルでは `include_columns`/`ignore_columns` で対象を絞らないと不要な特徴量が大量に生成される。

### カスタムプリミティブ(`TransformPrimitive` / `AggregationPrimitive` の継承)

**用途**: 標準搭載されていない独自の特徴量計算ロジックをプリミティブとして定義し、`dfs()` の `trans_primitives`/`agg_primitives` にクラスとして渡す。

**シグネチャ**: `class MyPrimitive(featuretools.primitives.TransformPrimitive): name = "..."; input_types = [...]; return_type = ...; def get_function(self): ...`

**使用例**:
```python
from featuretools.primitives import TransformPrimitive
from woodwork.column_schema import ColumnSchema
import numpy as np

class AmountRoundedUp(TransformPrimitive):
    name = "amount_rounded_up"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})

    def get_function(self):
        def amount_rounded_up(column):
            return np.ceil(column)
        return amount_rounded_up

fm_custom, defs_custom = ft.dfs(
    entityset=es,
    target_dataframe_name="transactions",
    agg_primitives=[],
    trans_primitives=[AmountRoundedUp],
    max_depth=1,
)
print(fm_custom)
```
実行結果:
```
                customer_id  amount  AMOUNT_ROUNDED_UP(amount)
transaction_id                                                
1                         1  100.00                      100.0
2                         1   50.50                       51.0
3                         1   20.00                       20.0
4                         2  300.00                      300.0
5                         2   75.25                       76.0
6                         3   60.00                       60.0
7                         3   40.00                       40.0
8                         3   90.00                       90.0
```

**注意点・落とし穴**:
- `input_types` は `ColumnSchema(logical_type=Double)` のように**論理型(logical_type)だけ**で指定すると、実際には(検証環境で確認した限り)DFSの列マッチングにヒットせず特徴量が1つも生成されなかった。標準プリミティブ(`Absolute` など)にならい `ColumnSchema(semantic_tags={"numeric"})` のように**セマンティックタグ**で指定するのが安全。
- `get_function()` が返す関数は、pandasの `Series`(または内部的にはnumpy配列)を受け取りベクトル化された演算を返す必要がある。`for`ループでの1件ずつの処理には対応していない。

---

## 特徴量選択・活用

### `featuretools.selection.remove_highly_null_features(...)`

**用途**: 特徴量行列から、欠損値の割合が閾値を超える列を除去する。

**シグネチャ**: `remove_highly_null_features(feature_matrix, features=None, pct_null_threshold=0.95)`

**使用例**:
```python
import pandas as pd
import numpy as np
from featuretools.selection import remove_highly_null_features

fm_demo = pd.DataFrame({
    "count_tx": [3, 2, 3, 5, 1],
    "sum_amount": [170.5, 375.25, 190.0, 900.0, 10.0],
    "mostly_null": [np.nan, np.nan, np.nan, np.nan, 5.0],
})
print(remove_highly_null_features(fm_demo, pct_null_threshold=0.7).columns.tolist())
```
実行結果:
```
['count_tx', 'sum_amount']
```

**注意点・落とし穴**:
- デフォルトの閾値は0.95(95%)と非常に高いため、`mostly_null`(欠損率80%)のような列でもデフォルトでは除去されない。用途に応じて `pct_null_threshold` を明示的に下げる必要がある。

### `featuretools.selection.remove_single_value_features(...)` / `remove_low_information_features(...)`

**用途**: 全行で同じ値しか持たない列(定数列)や、それに準じて情報量の少ない列を除去する。

**シグネチャ**: `remove_single_value_features(feature_matrix, features=None, count_nan_as_value=False)` / `remove_low_information_features(feature_matrix, features=None)`

**使用例**:
```python
fm_demo2 = pd.DataFrame({
    "count_tx": [3, 2, 3, 5, 1],
    "sum_amount": [170.5, 375.25, 190.0, 900.0, 10.0],
    "constant_flag": [1, 1, 1, 1, 1],
})
from featuretools.selection import remove_single_value_features, remove_low_information_features
print(remove_single_value_features(fm_demo2).columns.tolist())
print(remove_low_information_features(fm_demo2).columns.tolist())
```
実行結果:
```
['count_tx', 'sum_amount']
['count_tx', 'sum_amount']
```

**注意点・落とし穴**:
- `remove_low_information_features` は引数の閾値を取らず、内部的に「定数列」や「全欠損列」を自動判定して除去する。`remove_single_value_features` は定数列除去のみに特化した、より狭い機能。

### `featuretools.selection.remove_highly_correlated_features(...)`

**用途**: 特徴量同士の相関係数が閾値を超えるペアのうち、片方を除去して多重共線性を減らす。

**シグネチャ**: `remove_highly_correlated_features(feature_matrix, features=None, pct_corr_threshold=0.95, features_to_check=None, features_to_keep=None)`

**使用例**:
```python
fm_demo3 = pd.DataFrame({
    "sum_amount": [170.5, 375.25, 190.0, 900.0, 10.0],
    "sum_amount_x2": [341.0, 750.5, 380.0, 1800.0, 20.0],  # sum_amountの完全な線形結合
})
from featuretools.selection import remove_highly_correlated_features
print(remove_highly_correlated_features(fm_demo3).columns.tolist())
```
実行結果:
```
['sum_amount']
```

**注意点・落とし穴**:
- 2列が高相関の場合、`feature_matrix` に**後から出てくる列(右側の列)**が除去される(この例では `sum_amount_x2` が消える)。特定の列を必ず残したい場合は `features_to_keep` を使う。

### `ft.encode_features(...)`

**用途**: カテゴリ型特徴量をone-hotエンコーディングし、機械学習モデルにそのまま渡せる数値行列に変換する。

**シグネチャ**: `ft.encode_features(feature_matrix, features, top_n=10, include_unknown=True, to_encode=None, inplace=False, drop_first=False, verbose=False)`

**使用例**:
```python
fm_enc, defs_enc = ft.encode_features(feature_matrix, feature_defs)
print(fm_enc.columns.tolist())
```
実行結果:
```
['COUNT(transactions)', 'MEAN(transactions.amount)', 'SUM(transactions.amount)', 'MONTH(join_date) = 3', 'MONTH(join_date) = 2', 'MONTH(join_date) = 1', 'MONTH(join_date) is unknown']
```

**注意点・落とし穴**:
- `include_unknown=True`(デフォルト)だと、学習時に出現しなかったカテゴリ値に備えて `is unknown` 列が自動的に追加される。

---

## 時系列・リーク防止

### `cutoff_time` 引数

**用途**: 各対象インスタンス(この例では顧客)ごとに「その時刻より後のデータは見ない」という基準時刻を指定し、未来のデータが特徴量に混入する(データリーク)のを防ぐ。

**シグネチャ**: `ft.dfs(..., cutoff_time=None, cutoff_time_in_index=False, include_cutoff_time=True, ...)`(`cutoff_time` には対象のindex列名 + `time` 列を持つDataFrameを渡す)

**使用例**:
```python
cutoff_times = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "time": pd.to_datetime(["2020-02-01", "2020-03-01", "2020-03-25"], format="%Y-%m-%d"),
})
fm_cutoff, defs_cutoff = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    cutoff_time=cutoff_times,
    agg_primitives=["count", "sum"],
    trans_primitives=[],
    max_depth=1,
)
print(fm_cutoff)
```
実行結果:
```
             COUNT(transactions)  SUM(transactions.amount)
customer_id                                               
1                              3                    170.50
2                              2                    375.25
3                              2                    100.00
```

**注意点・落とし穴**:
- 顧客3は元々3件の取引(60.0+40.0+90.0=190.0)を持つが、`cutoff_time=2020-03-25` を指定したことで `2020-04-01` の取引(90.0)が除外され、`COUNT=2, SUM=100.0` になっている。これがまさにリーク防止の効果。
- `cutoff_time` のDataFrameは、`target_dataframe_name` のindex列名(この例では `customer_id`)と `time` 列を持つ必要がある。

### `training_window` 引数(+ `es.add_last_time_indexes()`)

**用途**: `cutoff_time` が「未来を見ない」ための上限だとすると、`training_window` は「どこまで過去に遡るか」を制限する下限。直近N日など、一定期間内のデータだけを集約したい場合に使う。

**シグネチャ**: `ft.dfs(..., training_window=None, ...)`(文字列 `"10 days"` や `ft.Timedelta(10, "d")` を渡す)

**使用例**:
```python
es.add_last_time_indexes()  # training_windowを使う前に必須

fm_tw, defs_tw = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    cutoff_time=cutoff_times,
    training_window="10 days",
    agg_primitives=["count", "sum"],
    trans_primitives=[],
    max_depth=1,
)
print(fm_tw)
```
実行結果:
```
             COUNT(transactions)  SUM(transactions.amount)
customer_id                                               
1                              1                     20.00
2                              1                     75.25
3                              1                     40.00
```

**注意点・落とし穴**:
- **`es.add_last_time_indexes()` を呼ばずに `training_window` を使うと、`UserWarning: Using training_window but last_time_index is not set for dataframe customers` という警告が出た上、集約結果が全行 `COUNT=0, SUM=0.0` になることを実際に確認した。** これは非常にハマりやすい落とし穴で、`training_window` を使う場合は必ず事前に `add_last_time_indexes()` を呼ぶこと。
- ウィンドウの境界は「下限(cutoff_time - window)は含まない・上限(cutoff_time)は含む」という挙動を確認した。例えば顧客2は `cutoff_time=2020-03-01`, `window=10日` で `2020-02-20`(ちょうど10日前)の取引(300.0)は含まれず、`2020-03-01`(cutoff当日)の取引(75.25)のみ含まれる。

---

## 特徴量の保存・読み込み

### `ft.save_features(...)` / `ft.load_features(...)`

**用途**: `dfs()` で生成した特徴量定義(`feature_defs`)をJSON形式でファイルに保存し、後で(学習時と推論時など)同じ特徴量セットを再利用する。

**シグネチャ**: `ft.save_features(features, location=None, profile_name=None)` / `ft.load_features(features, profile_name=None)`

**使用例**:
```python
ft.save_features(feature_defs, "/tmp/saved_features.json")
loaded = ft.load_features("/tmp/saved_features.json")
print(len(loaded), [f.get_name() for f in loaded])
```
実行結果:
```
4 ['COUNT(transactions)', 'MEAN(transactions.amount)', 'SUM(transactions.amount)', 'MONTH(join_date)']
```

**注意点・落とし穴**:
- 保存されるのは特徴量の「定義」(どのプリミティブをどの列に適用するか)であり、計算済みの値ではない。読み込んだ特徴量を実際に使うには、改めて `ft.calculate_feature_matrix(features=loaded, entityset=es)` を呼ぶ必要がある。
- `location` を省略するとJSON文字列がメモリ上に返るだけでファイルには保存されない。ファイルに保存する場合はパス文字列を渡す。

---

## その他

### `es.to_csv(...)` / `es.to_parquet(...)` / `es.to_pickle(...)`

**用途**: EntitySet全体(複数のDataFrameとリレーション定義)をディレクトリにシリアライズし、後で `ft.read_entityset(...)` で復元できるようにする。

**シグネチャ**: `es.to_csv(path, sep=',', encoding='utf-8', engine='python', compression=None, profile_name=None)`(`to_parquet`/`to_pickle` も同様の引数構成)

**使用例**:
```python
es.to_csv("/tmp/es_saved")
import os
print(sorted(os.listdir("/tmp/es_saved")))

es_loaded = ft.read_entityset("/tmp/es_saved")
print(es_loaded)
```
実行結果:
```
['data', 'data_description.json']
Entityset: transactions_es
  DataFrames:
    customers [Rows: 3, Columns: 4]
    transactions [Rows: 8, Columns: 5]
  Relationships:
    transactions.customer_id -> customers.customer_id
```
(注: `customers` が4列になっているのは、直前の「時系列・リーク防止」章で `es.add_last_time_indexes()` を呼んで `_ft_last_time` 列が追加された状態のままEntitySetを保存したため)

**注意点・落とし穴**:
- 単一ファイルではなく**ディレクトリ**が作られ、`data/` 以下に各DataFrameのデータファイル、`data_description.json` にスキーマ・リレーション情報がまとめて保存される。中身のサブディレクトリ名(`customers`/`transactions` など)は `data/` の下に作られる。

### `ft.Timedelta(value, unit=None, delta_obj=None)`

**用途**: `training_window` や `cutoff_time` 周りのAPIで期間を表現するためのfeaturetools独自クラス。多くの場面では `"10 days"` のような文字列でも代用できる。

**シグネチャ**: `ft.Timedelta(value, unit=None, delta_obj=None)`

**使用例**:
```python
td = ft.Timedelta(10, "d")
print(td.get_value("d"))
print(td.get_units())
```
実行結果:
```
10
['d']
```

**注意点・落とし穴**:
- `print(td)` としても `<featuretools.entityset.timedelta.Timedelta object at 0x...>` としか表示されず値が見えない。中身を確認したい場合は `get_value(unit)` を使う。
- 実用上は `training_window="10 days"` のように文字列で渡す方が可読性が高く、本ドキュメントの例でもすべて文字列表記を使っている。
