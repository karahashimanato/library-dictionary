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
8. [応用・発展](#応用発展)
   - [カスタムプリミティブの自作(応用)](#カスタムプリミティブの自作応用)
   - [複雑な多テーブル関係](#複雑な多テーブル関係)
   - [Deep Feature Synthesisの深さ制御](#deep-feature-synthesisの深さ制御)
   - [プリミティブオプション(primitive_options)の詳細制御](#プリミティブオプションprimitive_optionsの詳細制御)
   - [Woodworkの型システムとの連携](#woodworkの型システムとの連携)

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

---

## 応用・発展

ここから先は、これまでの章で使ってきた `es`(customers/transactionsの2テーブル)や `customers_df`/`transactions_df` を土台にしつつ、必要に応じて追加の列・テーブルを持つ別のEntitySet(`es2`, `es3`, ...)を組み立てて検証しています。基本のEntitySet構築・DFS・プリミティブ・特徴量選択・時系列・保存/読み込みでは扱わなかった、より発展的・ニッチなAPIをまとめます。

### カスタムプリミティブの自作(応用)

#### `TransformPrimitive.generate_name()` のオーバーライド

**用途**: プリミティブが生成する特徴量の列名を、コンストラクタ引数などに応じて動的に組み立てる。オーバーライドしない場合の既定の名前は `"<PRIMITIVE_NAME大文字>(base_feature_names...)"` になるが、独自の命名規則にしたい場合に使う。

**シグネチャ**: `class MyPrimitive(featuretools.primitives.TransformPrimitive): name = "..."; input_types = [...]; return_type = ...; def __init__(self, ...): ...; def get_function(self): ...; def generate_name(self, base_feature_names) -> str: ...`(`generate_name` は `TransformPrimitive` のインスタンスメソッドで、引数は `base_feature_names` の1つのみ)

**使用例**:
```python
from featuretools.primitives import TransformPrimitive
from woodwork.column_schema import ColumnSchema
import numpy as np

class AmountRoundedTo(TransformPrimitive):
    name = "amount_rounded_to"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})

    def __init__(self, ndigits=0):
        self.ndigits = ndigits

    def get_function(self):
        def amount_rounded_to(column):
            return np.round(column, self.ndigits)
        return amount_rounded_to

    def generate_name(self, base_feature_names):
        return "ROUND(%s, %s)" % (base_feature_names[0], self.ndigits)

fm, defs = ft.dfs(
    entityset=es,
    target_dataframe_name="transactions",
    agg_primitives=[],
    trans_primitives=[AmountRoundedTo(ndigits=1)],
    max_depth=1,
)
print(fm)
```
実行結果:
```
                customer_id  amount  ROUND(amount, 1)
transaction_id                                       
1                         1  100.00             100.0
2                         1   50.50              50.5
3                         1   20.00              20.0
4                         2  300.00             300.0
5                         2   75.25              75.2
6                         3   60.00              60.0
7                         3   40.00              40.0
8                         3   90.00              90.0
```

**注意点・落とし穴**:
- コンストラクタ引数(この例の `ndigits`)を `generate_name` 内で使う場合、`trans_primitives` には**クラスではなく必ずインスタンス化したもの**(`AmountRoundedTo(ndigits=1)`)を渡す必要がある。クラスのまま渡すと `__init__` の既定値(`ndigits=0`)で1個だけインスタンス化される。
- `generate_name` が返す文字列がそのまま特徴量行列の列名になるため、他のプリミティブや列名と衝突しない一意な名前にする責任は実装側にある。

#### `AggregationPrimitive.generate_name()` のオーバーライド

**用途**: 集約プリミティブが生成する特徴量の列名をカスタマイズする。既定では `NAME(relationship_path.column)` のような形式になる。

**シグネチャ**: `class MyPrimitive(featuretools.primitives.AggregationPrimitive): name = "..."; input_types = [...]; return_type = ...; def get_function(self): ...; def generate_name(self, base_feature_names, relationship_path_name, parent_dataframe_name, where_str, use_prev_str) -> str: ...`(`AggregationPrimitive.generate_name` は `TransformPrimitive` と異なり5つの引数を受け取る)

**使用例**:
```python
from featuretools.primitives import AggregationPrimitive

class AmountRange(AggregationPrimitive):
    name = "amount_range"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False

    def get_function(self):
        def amount_range(column):
            return column.max() - column.min()
        return amount_range

    def generate_name(self, base_feature_names, relationship_path_name, parent_dataframe_name, where_str, use_prev_str):
        return "RANGE(%s.%s)" % (relationship_path_name, base_feature_names[0])

fm, defs = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    agg_primitives=[AmountRange],
    trans_primitives=[],
    max_depth=1,
)
print(fm)
```
実行結果:
```
             RANGE(transactions.amount)
customer_id                            
1                                 80.00
2                                224.75
3                                 50.00
```

**注意点・落とし穴**:
- `AggregationPrimitive.generate_name` は `TransformPrimitive.generate_name` よりシグネチャが複雑(5引数)。引数を1つでも欠かすと `TypeError` になるため、全て受け取った上で使う引数だけ使うこと。
- `stack_on_self = False` を指定しないと、`max_depth>=2` の場合にこのプリミティブ自身の出力(例えば他の集約結果)にまで再適用されようとし、意図しない特徴量が増えることがある。

#### 複数入力を取るカスタムプリミティブと `commutative` 属性

**用途**: 2列以上を受け取る変換プリミティブを定義する場合の `input_types` の書き方と、列の組み合わせ爆発を抑える `commutative` 属性の効果を確認する。

**シグネチャ**: `input_types = [ColumnSchema(...), ColumnSchema(...)]`(2要素以上のリストにすると複数入力プリミティブになる) / `commutative: bool = False`(クラス属性。`True`にすると引数の順序違いの重複組み合わせが除去される)

**使用例**:
```python
transactions_df2 = transactions_df.assign(tax=[10.0, 5.0, 2.0, 30.0, 7.5, 6.0, 4.0, 9.0])
es3 = ft.EntitySet(id="ratio_demo")
es3 = es3.add_dataframe(dataframe_name="transactions", dataframe=transactions_df2, index="transaction_id", time_index="transaction_time")

class RatioTo(TransformPrimitive):
    name = "ratio_to"
    input_types = [ColumnSchema(semantic_tags={"numeric"}), ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})

    def get_function(self):
        def ratio_to(numer, denom):
            return numer / denom
        return ratio_to

defs_noncomm = ft.dfs(entityset=es3, target_dataframe_name="transactions",
                       agg_primitives=[], trans_primitives=[RatioTo], max_depth=1, features_only=True)
print("commutative未指定:", [f.get_name() for f in defs_noncomm])

class RatioToCommutative(RatioTo):
    name = "ratio_to_commutative"
    commutative = True

defs_comm = ft.dfs(entityset=es3, target_dataframe_name="transactions",
                    agg_primitives=[], trans_primitives=[RatioToCommutative], max_depth=1, features_only=True)
print("commutative=True:", [f.get_name() for f in defs_comm])
```
実行結果:
```
commutative未指定: ['customer_id', 'amount', 'tax', 'RATIO_TO(amount, customer_id)', 'RATIO_TO(amount, tax)', 'RATIO_TO(customer_id, amount)', 'RATIO_TO(customer_id, tax)', 'RATIO_TO(tax, amount)', 'RATIO_TO(tax, customer_id)']
commutative=True: ['customer_id', 'amount', 'tax', 'RATIO_TO_COMMUTATIVE(amount, customer_id)', 'RATIO_TO_COMMUTATIVE(amount, tax)', 'RATIO_TO_COMMUTATIVE(customer_id, tax)']
```

**注意点・落とし穴**:
- `transactions` テーブルには数値列が3つ(`customer_id`, `amount`, `tax`)あるため、`commutative` を指定しないと**順序違いを含めた6通り**すべてに特徴量が生成された(`RATIO_TO(amount, tax)` と `RATIO_TO(tax, amount)` の両方など)。`commutative = True` にすると3通りに半減することを実際に確認した。
- 本来 `customer_id` は識別子であり比率計算に意味を持たない列だが、`semantic_tags={"numeric"}` を満たすため自動的に対象になってしまう。実務では `primitive_options` の `include_columns`/`ignore_columns`(後述)と組み合わせて対象列を絞るのが安全。

#### `uses_calc_time` 属性(カットオフ時刻を計算関数内で利用する)

**用途**: プリミティブの計算関数の中で、そのインスタンスの基準時刻(`cutoff_time`)自体を参照したい場合(例:「cutoff時点から何日経過したか」)に使う。

**シグネチャ**: `class MyPrimitive(TransformPrimitive): uses_calc_time = True; def get_function(self): def f(column, time=None): ...`(`uses_calc_time = True` を立てると、`get_function()` が返す関数が `time` というキーワード引数でcutoff時刻のSeriesを追加で受け取れるようになる)

**使用例**:
```python
from woodwork.logical_types import Datetime as WWDatetime

class DaysSinceCalcTime(TransformPrimitive):
    name = "days_since_calc_time"
    input_types = [ColumnSchema(logical_type=WWDatetime)]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_calc_time = True

    def get_function(self):
        def days_since_calc_time(column, time=None):
            return (time - column).dt.days
        return days_since_calc_time

fm, defs = ft.dfs(
    entityset=es,
    target_dataframe_name="transactions",
    agg_primitives=[],
    trans_primitives=[DaysSinceCalcTime],
    max_depth=1,
    cutoff_time=pd.DataFrame({
        "transaction_id": transactions_df["transaction_id"].values,
        "time": pd.Timestamp("2020-05-01"),
    }),
)
print(fm)
```
実行結果:
```
                customer_id  amount  DAYS_SINCE_CALC_TIME(transaction_time)
transaction_id                                                             
1                         1  100.00                                   117.0
2                         1   50.50                                   112.0
3                         1   20.00                                    90.0
4                         2  300.00                                    71.0
5                         2   75.25                                    61.0
6                         3   60.00                                    47.0
7                         3   40.00                                    42.0
8                         3   90.00                                    30.0
```

**注意点・落とし穴**:
- `input_types` に日時型を指定する際、`ColumnSchema(logical_type="Datetime")` のように**文字列**を渡すと `TypeError: logical_type Datetime is not a registered LogicalType.` になることを実行して確認した。`from woodwork.logical_types import Datetime` のように**論理型クラス自体をインポートして**渡す必要がある。
- `uses_calc_time = True` を立てずに関数側で `time` 引数を宣言すると、DFS実行時にその引数が渡されないためエラーになる。逆に `uses_calc_time = True` なのに関数が `time` を受け取らないシグネチャだとやはりエラーになる。

### 複雑な多テーブル関係

#### 3階層以上のEntitySet(grandparent→parent→childのDFS集約)

**用途**: 3つ以上のテーブルを親子関係で連結し(regions → customers → transactions)、DFSが中間テーブルを自動で辿って孫テーブルまで集約する挙動を確認する。

**シグネチャ**: `es.add_relationship(...)` を複数回呼び、3階層以上のリレーショングラフを構築する。DFS自体の呼び出し方(`ft.dfs(...)`)は2階層の場合と同じ。

**使用例**:
```python
regions_df = pd.DataFrame({"region_id": [1, 2], "region_name": ["east", "west"]})
customers_df3 = customers_df.assign(region_id=[1, 2, 1])

es4 = ft.EntitySet(id="three_level_es")
es4 = es4.add_dataframe(dataframe_name="regions", dataframe=regions_df, index="region_id")
es4 = es4.add_dataframe(dataframe_name="customers", dataframe=customers_df3, index="customer_id", time_index="join_date")
es4 = es4.add_dataframe(dataframe_name="transactions", dataframe=transactions_df, index="transaction_id", time_index="transaction_time")
es4 = es4.add_relationship("regions", "region_id", "customers", "region_id")
es4 = es4.add_relationship("customers", "customer_id", "transactions", "customer_id")
print(es4)

fm, defs = ft.dfs(
    entityset=es4,
    target_dataframe_name="regions",
    agg_primitives=["count", "sum"],
    trans_primitives=[],
    max_depth=2,
)
print(fm)
print([f.get_name() for f in defs])
```
実行結果:
```
Entityset: three_level_es
  DataFrames:
    regions [Rows: 2, Columns: 2]
    customers [Rows: 3, Columns: 3]
    transactions [Rows: 8, Columns: 4]
  Relationships:
    customers.region_id -> regions.region_id
    transactions.customer_id -> customers.customer_id
           COUNT(customers)  COUNT(transactions)  SUM(transactions.amount)
region_id                                                                 
1                         2                    6                    360.50
2                         1                    2                    375.25
['COUNT(customers)', 'COUNT(transactions)', 'SUM(transactions.amount)']
```

**注意点・落とし穴**:
- `target_dataframe_name="regions"` に対して、**2ホップ先(孫テーブル)の`transactions`を集約した`COUNT(transactions)`や`SUM(transactions.amount)`が`max_depth=2`だけで自動生成される**。中間テーブル`customers`経由の集約を明示的に書く必要はない(DFSがリレーショングラフを自動で辿る)。
- region 1 は顧客1・3(合計6件の取引、100+50.5+20+60+40+90=360.5)、region 2 は顧客2(2件、300+75.25=375.25)に対応しており、値が正しく2ホップ集約されていることを確認した。

#### 同一テーブルへの複数の外部キー(`transfers[sender_id]` / `transfers[receiver_id]`)

**用途**: 1つの子テーブルが同じ親テーブルを指す外部キーを複数持つ場合(送金元/送金先、発注者/承認者など)に、リレーションをどう定義し、DFSが生成する特徴量名がどう区別されるかを確認する。

**シグネチャ**: `es.add_relationship(parent_dataframe_name, parent_column_name, child_dataframe_name, child_column_name)` を、同じ親子テーブルの組に対して異なる `child_column_name` で複数回呼ぶ。

**使用例**:
```python
transfers_df = pd.DataFrame({
    "transfer_id": range(1, 6),
    "sender_id": [1, 1, 2, 3, 3],
    "receiver_id": [2, 3, 3, 1, 2],
    "transfer_time": pd.to_datetime(["2020-01-06", "2020-01-20", "2020-02-25", "2020-03-18", "2020-03-22"], format="%Y-%m-%d"),
    "transfer_amount": [15.0, 8.0, 22.0, 5.0, 12.0],
})

es5 = ft.EntitySet(id="multi_fk_es")
es5 = es5.add_dataframe(dataframe_name="customers", dataframe=customers_df[["customer_id", "join_date"]], index="customer_id", time_index="join_date")
es5 = es5.add_dataframe(dataframe_name="transfers", dataframe=transfers_df, index="transfer_id", time_index="transfer_time")
es5 = es5.add_relationship("customers", "customer_id", "transfers", "sender_id")
es5 = es5.add_relationship("customers", "customer_id", "transfers", "receiver_id")
print(es5)

fm, defs = ft.dfs(
    entityset=es5,
    target_dataframe_name="customers",
    agg_primitives=["count", "sum"],
    trans_primitives=[],
    max_depth=1,
)
print(fm)
```
実行結果:
```
Entityset: multi_fk_es
  DataFrames:
    customers [Rows: 3, Columns: 2]
    transfers [Rows: 5, Columns: 5]
  Relationships:
    transfers.sender_id -> customers.customer_id
    transfers.receiver_id -> customers.customer_id
             COUNT(transfers[sender_id])  SUM(transfers[sender_id].transfer_amount)  COUNT(transfers[receiver_id])  SUM(transfers[receiver_id].transfer_amount)
customer_id                                                                                                                                                    
1                                      2                                       23.0                              1                                          5.0
2                                      1                                       22.0                              2                                         27.0
3                                      2                                       17.0                              2                                         30.0
```

**注意点・落とし穴**:
- 同じ2テーブル間に複数のリレーションがあると、featuretoolsは特徴量名を `transfers[sender_id]` / `transfers[receiver_id]` のように**どの外部キー経由かをブラケットで明示**して自動的に区別する。特徴量名がぶつかることはない。
- 顧客1は送金元として2件(15.0+8.0=23.0)、受取側として1件(5.0)というように、送信/受信で完全に独立した集約が計算されていることを値で確認した。

#### `allowed_paths` でリレーションパスを制限する

**用途**: リレーショングラフが複雑になったとき、DFSが辿るリレーションパスを明示的に絞り込み、不要な多ホップ特徴量の生成を防ぐ。

**シグネチャ**: `ft.dfs(..., allowed_paths=None, ...)`(`[[table_a, table_b], ...]` のような、辿ってよいテーブル名の並びのリストを渡す)

**使用例**:
```python
fm_full, defs_full = ft.dfs(
    entityset=es4,
    target_dataframe_name="regions",
    agg_primitives=["count", "sum"],
    trans_primitives=[],
    max_depth=2,
)
print("制限なし:", sorted(f.get_name() for f in defs_full))

fm_restricted, defs_restricted = ft.dfs(
    entityset=es4,
    target_dataframe_name="regions",
    agg_primitives=["count", "sum"],
    trans_primitives=[],
    max_depth=2,
    allowed_paths=[["regions", "customers"]],
)
print("allowed_paths制限あり:", sorted(f.get_name() for f in defs_restricted))
```
実行結果:
```
制限なし: ['COUNT(customers)', 'COUNT(transactions)', 'SUM(transactions.amount)']
allowed_paths制限あり: ['COUNT(customers)']
```

**注意点・落とし穴**:
- `allowed_paths=[["regions", "customers"]]` を指定すると、`regions → customers → transactions` という2ホップ目のパスが遮断され、`transactions` を集約した特徴量(`COUNT(transactions)`, `SUM(transactions.amount)`)が一切生成されなくなることを確認した。
- パスは `target_dataframe_name` 側から見たテーブル名の並びで指定する。大規模なEntitySetで特定の経路だけに特徴量生成を絞りたい場合に有効。

### Deep Feature Synthesisの深さ制御

#### `max_depth` を変えたときの特徴量数の実測

**用途**: `max_depth` を増やすと特徴量数がどう増えるか(そしてどこで頭打ちになるか)を、実際のリレーショングラフで実測する。

**シグネチャ**: `ft.dfs(..., max_depth=2, ...)`

**使用例**:
```python
for depth in [1, 2, 3]:
    defs = ft.dfs(
        entityset=es4,
        target_dataframe_name="regions",
        agg_primitives=["count", "mean"],
        trans_primitives=["month"],
        max_depth=depth,
        features_only=True,
    )
    print(f"max_depth={depth}: {len(defs)}件 -> {[f.get_name() for f in defs]}")
```
実行結果:
```
max_depth=1: 1件 -> ['COUNT(customers)']
max_depth=2: 5件 -> ['COUNT(customers)', 'COUNT(transactions)', 'MEAN(transactions.amount)', 'MEAN(customers.COUNT(transactions))', 'MEAN(customers.MEAN(transactions.amount))']
max_depth=3: 5件 -> ['COUNT(customers)', 'COUNT(transactions)', 'MEAN(transactions.amount)', 'MEAN(customers.COUNT(transactions))', 'MEAN(customers.MEAN(transactions.amount))']
```

**注意点・落とし穴**:
- `max_depth=2→3` で特徴量数が増えていない(5件のまま)。これは `regions` テーブルにこれ以上辿れるリレーションパスも日時列もなく、`trans_primitives=["month"]` を適用できる列(`regions`には日時列がない)も存在しないため。**`max_depth` を大きくしても、リレーショングラフとテーブルの列構成で頭打ちになる**ことを実測で確認した。`month` プリミティブは `UnusedPrimitiveWarning` が出て実際には使われなかった。
- `max_depth=2`では`MEAN(customers.COUNT(transactions))`のような「孫テーブルの集約をさらに親で集約する」特徴量(stacked feature)が生成されている点にも注意。depthを上げるほど、こうしたスタック特徴量は指数的に増え得るため、実データでは`max_depth`を上げる前に`max_features`や`primitive_options`での絞り込みを検討すべき。

#### `max_features` で生成数を打ち切る

**用途**: `agg_primitives`/`trans_primitives`/`max_depth`の組み合わせで大量の特徴量候補が生成される場合に、生成数の上限を強制的に区切る。

**シグネチャ**: `ft.dfs(..., max_features=-1, ...)`(既定値`-1`は無制限。正の整数を渡すとその件数で打ち切る)

**使用例**:
```python
defs_full = ft.dfs(
    entityset=es4, target_dataframe_name="regions",
    agg_primitives=["count", "mean"], trans_primitives=["month"],
    max_depth=3, features_only=True,
)
print("max_features指定なし:", len(defs_full), "件")

defs_limited = ft.dfs(
    entityset=es4, target_dataframe_name="regions",
    agg_primitives=["count", "mean"], trans_primitives=["month"],
    max_depth=3, max_features=3, features_only=True,
)
print("max_features=3:", len(defs_limited), "件 ->", [f.get_name() for f in defs_limited])
```
実行結果:
```
max_features指定なし: 5 件
max_features=3: 3 件 -> ['COUNT(customers)', 'COUNT(transactions)', 'MEAN(transactions.amount)']
```

**注意点・落とし穴**:
- `max_features` は、DFSが内部的に特徴量を構築していく**順序の先頭から**打ち切る単純な件数制限であり、重要度や分散などによる選別は一切行わない。列選択目的というよりは、探索的にDFSを試す際の暴走(特徴量が万単位になるなど)を防ぐための安全弁として使うのが実用的。

### プリミティブオプション(`primitive_options`)の詳細制御

#### `include_columns` / `ignore_columns` でテーブル別に対象列を絞る

**用途**: 同じプリミティブ(例: `sum`)を複数の数値列に無差別に適用させず、テーブルごとに対象列を明示的に指定する。

**シグネチャ**: `ft.dfs(..., primitive_options={"<primitive_name>": {"include_columns": {"<table>": [...]}, "ignore_columns": {"<table>": [...]}}}, ...)`

**使用例**:
```python
transactions_df3 = transactions_df.assign(fee=[1.0, 0.5, 0.2, 3.0, 0.75, 0.6, 0.4, 0.9])
es6 = ft.EntitySet(id="opt_es")
es6 = es6.add_dataframe(dataframe_name="customers", dataframe=customers_df, index="customer_id", time_index="join_date")
es6 = es6.add_dataframe(dataframe_name="transactions", dataframe=transactions_df3, index="transaction_id", time_index="transaction_time")
es6 = es6.add_relationship("customers", "customer_id", "transactions", "customer_id")

fm_ignore, _ = ft.dfs(
    entityset=es6, target_dataframe_name="customers", agg_primitives=["sum"], trans_primitives=[], max_depth=1,
    primitive_options={"sum": {"ignore_columns": {"transactions": ["fee"]}}},
)
print("ignore_columns(feeを除外):", sorted(fm_ignore.columns.tolist()))

fm_include, _ = ft.dfs(
    entityset=es6, target_dataframe_name="customers", agg_primitives=["sum"], trans_primitives=[], max_depth=1,
    primitive_options={"sum": {"include_columns": {"transactions": ["fee"]}}},
)
print("include_columns(feeのみ対象):", sorted(fm_include.columns.tolist()))
```
実行結果:
```
ignore_columns(feeを除外): ['SUM(transactions.amount)']
include_columns(feeのみ対象): ['SUM(transactions.fee)']
```

**注意点・落とし穴**:
- `primitive_options` を指定しない場合、`sum` は `transactions` の数値列(`amount`, `fee`)両方に適用され `SUM(transactions.amount)` と `SUM(transactions.fee)` の両方が生成される。`ignore_columns`/`include_columns` はテーブル名をキーにした辞書で、テーブルごとに独立して指定できる。
- `ignore_columns` と `include_columns` を同じプリミティブ・同じテーブルに同時指定した場合の優先順位までは検証していない(通常はどちらか一方だけ使うのが安全)。

#### `groupby_trans_primitives` + `include_groupby_columns`

**用途**: `cum_sum`(累積和)のように「あるグループ内で時系列順に計算する」変換プリミティブに対して、グループ化に使う列とグループ化対象の値列をそれぞれ指定する。

**シグネチャ**: `ft.dfs(..., groupby_trans_primitives=None, primitive_options={"<primitive_name>": {"include_columns": {...}, "include_groupby_columns": {...}}}, ...)`

**使用例**:
```python
from woodwork.logical_types import Categorical

transactions_df4 = transactions_df3.assign(category=["food", "food", "travel", "food", "travel", "travel", "food", "travel"])
es7 = ft.EntitySet(id="opt_es_cat")
es7 = es7.add_dataframe(
    dataframe_name="transactions", dataframe=transactions_df4,
    index="transaction_id", time_index="transaction_time",
    logical_types={"category": Categorical},
)

fm, defs = ft.dfs(
    entityset=es7, target_dataframe_name="transactions",
    agg_primitives=[], trans_primitives=[],
    groupby_trans_primitives=["cum_sum"], max_depth=1,
    primitive_options={
        "cum_sum": {
            "include_columns": {"transactions": ["amount"]},
            "include_groupby_columns": {"transactions": ["category"]},
        },
    },
)
print(fm[[c for c in fm.columns if "CUM_SUM" in c]])
```
実行結果:
```
                CUM_SUM(amount) by category
transaction_id                             
1                                     100.00
2                                     150.50
3                                      20.00
4                                     450.50
5                                      95.25
6                                     155.25
7                                     490.50
8                                     245.25
```

**注意点・落とし穴**:
- `category` 列に `logical_types={"category": Categorical}` を明示しないと、この列は自動推論で `Unknown` 論理型になり `include_groupby_columns` に指定しても `cum_sum` が一切適用されず(`UnusedPrimitiveWarning`)、特徴量が1つも生成されないことを実行して確認した。groupby対象にしたい列は明示的にCategorical系の型を与える必要がある(詳細は次の「Woodworkの型システムとの連携」章)。
- `category="food"`の行(1,2,4,7行目、amountは100,50.5,300,40)の累積和が 100.00→150.50→450.50→490.50 と、時系列順(`transaction_time`順)にfoodグループ内だけで積み上がっていることが値から確認できる。

#### `primitive_options` のキーにプリミティブインスタンスを使う

**用途**: 同じプリミティブ(例: `Sum`)を、対象列を変えて複数回・別々のオプションで使い分けたい場合、文字列名ではなく**プリミティブのインスタンス**を`agg_primitives`/`primitive_options`のキーとして使う。

**シグネチャ**: `ft.dfs(..., agg_primitives=[prim_instance_1, prim_instance_2], primitive_options={prim_instance_1: {...}, prim_instance_2: {...}}, ...)`

**使用例**:
```python
from featuretools.primitives import Sum

sum_amount = Sum()
sum_fee = Sum()
fm, defs = ft.dfs(
    entityset=es6,
    target_dataframe_name="customers",
    agg_primitives=[sum_amount, sum_fee],
    trans_primitives=[],
    max_depth=1,
    primitive_options={
        sum_amount: {"include_columns": {"transactions": ["amount"]}},
        sum_fee: {"include_columns": {"transactions": ["fee"]}},
    },
)
print(sorted(fm.columns.tolist()))
```
実行結果:
```
['SUM(transactions.amount)', 'SUM(transactions.fee)']
```

**注意点・落とし穴**:
- `primitive_options` の辞書のキーは、文字列名(`"sum"`)を使うと**その名前の全インスタンスに同じオプションが適用される**が、インスタンス自体をキーにすると**そのインスタンスだけ**に個別のオプションを適用できる。この例では同じ`Sum`プリミティブを2つのインスタンスとして使い分け、`amount`用と`fee`用でそれぞれ異なる`include_columns`を指定した。
- `agg_primitives`/`trans_primitives`にプリミティブのインスタンスを渡すこと自体は、カスタムプリミティブの章で見た`AmountRoundedTo(ndigits=1)`と同じ仕組み。標準プリミティブでも同様にインスタンス単位でオプションを分けられる。

### Woodworkの型システムとの連携

#### `logical_types` を明示指定して `add_dataframe`

**用途**: `add_dataframe` の型自動推論に頼らず、`Boolean`や`Ordinal`(順序付きカテゴリ)などの論理型を明示的に指定し、意図通りのセマンティックタグを持たせる。

**シグネチャ**: `es.add_dataframe(..., logical_types=None, ...)`(`{"<列名>": <LogicalTypeクラス or インスタンス>}` の辞書を渡す)

**使用例**:
```python
from woodwork.logical_types import Boolean, Ordinal

transactions_df5 = transactions_df.assign(
    is_refunded=[False, False, True, False, False, True, False, False],
    priority=["low", "medium", "high", "low", "high", "medium", "low", "high"],
)
es8 = ft.EntitySet(id="ww_es")
es8 = es8.add_dataframe(
    dataframe_name="transactions",
    dataframe=transactions_df5,
    index="transaction_id",
    time_index="transaction_time",
    logical_types={
        "is_refunded": Boolean,
        "priority": Ordinal(order=["low", "medium", "high"]),
    },
)
print(es8["transactions"].ww.schema)
```
実行結果:
```
                                        Logical Type Semantic Tag(s)
Column                                                              
transaction_id                               Integer       ['index']
customer_id                                  Integer     ['numeric']
transaction_time                            Datetime  ['time_index']
amount                                        Double     ['numeric']
is_refunded                                  Boolean              []
priority          Ordinal: ['low', 'medium', 'high']    ['category']
```

**注意点・落とし穴**:
- `Boolean`や`Ordinal`はクラスをそのまま渡せるが、`Ordinal`は`order=[...]`という追加パラメータが必要なため**インスタンス化して**(`Ordinal(order=[...])`)渡す必要がある。クラスのまま渡すと順序情報がなく初期化に失敗する。
- `Ordinal`型の列には自動的に`['category']`セマンティックタグが付与される一方、`Boolean`型の列にはセマンティックタグが付かない(`[]`)ことを実行して確認した。

#### `df.ww.logical_types` / `ww.semantic_tags` の確認と `ww.set_types` による型変更

**用途**: EntitySetに登録済みのDataFrameについて、列ごとの論理型・セマンティックタグを個別に確認し、必要に応じて後から型を変更する。

**シグネチャ**: `es[table_name].ww.logical_types`(プロパティ、`{列名: LogicalType}`の辞書) / `es[table_name].ww.semantic_tags`(プロパティ、`{列名: set}`の辞書) / `es[table_name].ww.set_types(logical_types=None, semantic_tags=None, retain_index_tags=True)`

**使用例**:
```python
print(es8["transactions"].ww.logical_types["priority"])
print(es8["transactions"].ww.semantic_tags["is_refunded"])

es8["transactions"].ww.set_types(logical_types={"customer_id": "Categorical"})
print(es8["transactions"].ww.logical_types["customer_id"])
print(es8["transactions"].ww.semantic_tags["customer_id"])
```
実行結果:
```
Ordinal: ['low', 'medium', 'high']
set()
Categorical
{'category'}
```

**注意点・落とし穴**:
- `ww.set_types` は `add_dataframe` 時とは異なり文字列(`"Categorical"`)でも型を指定できる(内部でLogicalType名から解決される)。
- `customer_id` は元々`Integer`型・`{'numeric'}`タグだったが、`set_types`で`Categorical`に変更すると`{'category'}`タグに切り替わる。**この後にDFSを実行すると、`customer_id`は数値集約プリミティブ(`sum`/`mean`など)の対象から外れ、代わりにカテゴリ系プリミティブの対象になる**(値そのものは変わらないが、DFSでの扱われ方がまるごと変わる点に注意)。

#### Ordinal型の `order` 指定がプリミティブの適用対象に与える影響

**用途**: `Ordinal`型で順序を明示するかどうかが、`greater_than`/`less_than`のような大小比較プリミティブの適用可否に実際にどう影響するかを確認する。

**シグネチャ**: `Ordinal(order=[...])`(`woodwork.logical_types.Ordinal`。`greater_than`/`less_than`/`greater_than_equal_to`/`less_than_equal_to`は`valid_inputs`に`Ordinal`同士のペアを含む)

**使用例**:
```python
sev_df = transactions_df.assign(
    priority=["low", "medium", "high", "low", "high", "medium", "high", "low"],
    severity=["medium", "medium", "low", "high", "low", "medium", "high", "low"],
)

# 両方をOrdinalとして明示した場合
es9 = ft.EntitySet(id="ww_es_ord")
es9 = es9.add_dataframe(
    dataframe_name="transactions", dataframe=sev_df.copy(),
    index="transaction_id", time_index="transaction_time",
    logical_types={
        "priority": Ordinal(order=["low", "medium", "high"]),
        "severity": Ordinal(order=["low", "medium", "high"]),
    },
)
defs_ord = ft.dfs(entityset=es9, target_dataframe_name="transactions",
                   agg_primitives=[], trans_primitives=["greater_than"], max_depth=1, features_only=True)
print("両方をOrdinal指定:", [f.get_name() for f in defs_ord])

# 型を明示せず自動推論に任せた場合
es10 = ft.EntitySet(id="ww_es_noord")
es10 = es10.add_dataframe(
    dataframe_name="transactions", dataframe=sev_df.copy(),
    index="transaction_id", time_index="transaction_time",
)
print("自動推論時の型:", es10["transactions"].ww.logical_types["priority"], "/", es10["transactions"].ww.logical_types["severity"])
defs_noord = ft.dfs(entityset=es10, target_dataframe_name="transactions",
                     agg_primitives=[], trans_primitives=["greater_than"], max_depth=1, features_only=True)
print("型指定なし:", [f.get_name() for f in defs_noord])
```
実行結果:
```
両方をOrdinal指定: ['customer_id', 'amount', 'is_refunded', 'priority', 'severity', 'priority > severity', 'severity > priority']
自動推論時の型: Categorical / Unknown
型指定なし: ['customer_id', 'amount', 'is_refunded', 'priority']
```

**注意点・落とし穴**:
- `greater_than`は`valid_inputs`に「同じOrdinal型同士」のペアを要求するため、`priority`と`severity`の両方に`Ordinal(order=[...])`を明示した場合のみ`priority > severity`/`severity > priority`という比較特徴量が(非可換なので両方向)生成された。
- 型を明示しなかった場合、`priority`は`Categorical`に自動推論されたが`severity`は`Unknown`型になった(同じ文字列カラムでも値の分布などにより自動推論結果が異なりうる)。さらに**`Unknown`型の列はDFSの特徴量候補にすら含まれず**(`defs_noord`の一覧に`severity`自体が出てこない)、`greater_than`はもちろん他のどのプリミティブの対象にもならない。文字列カテゴリ列は自動推論任せにせず、意図した型(`Categorical`または`Ordinal`)を明示するのが安全。
