# pandas vs polars 比較(DataFrameライブラリ)

検証バージョン: pandas 3.0.5 / polars 1.44.1
検証環境: `/home/manaty/library-practicing/.venv/bin/python`(Python 3.12.3)

このドキュメントは[pandas/README.md](../pandas/README.md)・[polars/README.md](../polars/README.md)がそれぞれ単体のAPI逆引き辞書であるのに対し、**両者を横断して「同じ処理をどう書くか」「どちらが速いか」「移行時に何に気をつけるべきか」を判断するための比較資料**として作成した。掲載しているコード・実行結果・ベンチマーク数値は、すべて本ドキュメント作成時に実際に上記Python環境で実行して得たものであり、記憶やドキュメントからの引用のみで書いた記述はない。

ベンチマークの実行環境は以下の通り(すべて単一マシン・単一セッションでの計測であり、権威的な公式ベンチマークではない点に注意):

- OS: Linux 6.6.87.2-microsoft-standard-WSL2(WSL2)
- CPU: AMD Ryzen AI 7 PRO 350(論理コア数 `os.cpu_count()` = 16)
- メモリ: 21GiB
- `pl.thread_pool_size()` = 16(polarsが認識しているスレッドプールサイズ)

## 目次

1. [設計思想の違い](#1-設計思想の違い)
2. [典型操作の対応表](#2-典型操作の対応表)
3. [パフォーマンス比較(実測ベンチマーク)](#3-パフォーマンス比較実測ベンチマーク)
4. [移行時の注意点](#4-移行時の注意点)
5. [どちらを使うべきかの判断基準](#5-どちらを使うべきかの判断基準)

---

## 1. 設計思想の違い

実際にコードを実行して確認できた設計上の違いは以下の通り。

### 1.1 インデックスの有無

```python
import pandas as pd
import polars as pl

df_pd = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
print(df_pd.index.tolist())          # ['x', 'y', 'z']

df_pl = pl.DataFrame({"a": [1, 2, 3]})
hasattr(df_pl, "index")              # False
df_pl.loc                            # AttributeError: 'DataFrame' object has no attribute 'loc'
```

pandasは行ラベル(インデックス)を持ち`loc`/`iloc`でアクセスする設計だが、polarsにはインデックスという概念自体が存在せず、`df_pl.loc`は実行時に`AttributeError`になることを確認した。polarsでは行は常に0始まりの位置(または`filter`の条件式)でアクセスする。

### 1.2 Eager/Lazy の2本立て

```python
lf = pl.LazyFrame({"a": [1, 2, 3]})
q = lf.filter(pl.col("a") > 1)
type(q)          # <class 'polars.lazyframe.frame.LazyFrame'>  ← filterを呼んだ時点ではまだ計算されない
q.collect()       # ここで初めて実行される
hasattr(pd.DataFrame, "lazy")   # False ← pandasにはLazy APIという概念がない
```

polarsは`filter`を呼んだ直後でも戻り値の型が`LazyFrame`のままであり、実際にクエリが実行されるのは`.collect()`を呼んだ時点であることをコード上の型確認で検証した。pandasには対応する仕組みがなく、各メソッド呼び出しはすべて即時(Eager)に実行される。

### 1.3 immutability(非破壊性)とCopy-on-Write

```python
import numpy as np

# pandas 3.0: CoWが常時有効
df1 = pd.DataFrame({"a": [1, 2, 3]})
df_shallow = df1.copy(deep=False)
np.shares_memory(df1["a"].to_numpy(), df_shallow["a"].to_numpy())  # True(コピー直後はメモリ共有)
df_shallow.iloc[0, 0] = 999
df1  # a: [1, 2, 3] のまま ← 書き換えても元は変化しない(CoWにより書き込み時に内部コピーが発生)

# polars: そもそも変更操作という概念がない
df_pl1 = pl.DataFrame({"a": [1, 2, 3]})
df_pl2 = df_pl1                 # 単純代入は同一オブジェクトの参照
df_pl2 is df_pl1                # True
df_pl3 = df_pl1.clone()         # clone()で明示的に別オブジェクトを作る
df_pl3 is df_pl1                # False
df_pl_new = df_pl1.with_columns((pl.col("a") * 2).alias("b"))
"b" in df_pl1.columns           # False ← 元のdf_pl1は変更されない
df_pl1 is df_pl_new             # False ← with_columnsは常に新しいDataFrameを返す
```

pandas 3.0はCoWが常時有効(無効化不可)になったことで「書き換えても元のDataFrameは変化しない」という点ではpolarsに近づいているが、pandasには依然として`inplace=True`という(効果は限定的だが)破壊的更新の構文自体が残っている。一方polarsには`inplace`という概念自体が存在せず、すべての変換メソッドは新しいDataFrame/LazyFrameを返す設計になっていることを`with_columns`の戻り値が別オブジェクトであることで確認した。

### 1.4 マルチスレッド実行

```python
import os
os.cpu_count()            # 16
pl.thread_pool_size()     # 16
hasattr(pd, "thread_pool_size")  # False
```

実際に`POLARS_MAX_THREADS`環境変数を変えて200万行のDataFrameに対する`group_by().agg()`を計測したところ、以下の通りスレッド数を絞ると明確に遅くなることを確認した(詳細は[3.4節](#34-マルチスレッドの効果pandasには存在しない設定軸)を参照)。

| スレッド数 | `group_by().agg()` 中央値(200万行) |
|---|---|
| 16(デフォルト) | 131.2ms |
| 1(`POLARS_MAX_THREADS=1`) | 411.1ms |

pandasは(NumPy/内部実装の一部を除き)基本的にシングルスレッドで動作し、`thread_pool_size`のような並列度を制御するAPIはpandas本体には存在しない。polarsはRust実装により、`group_by`のような集計処理で複数コアを積極的に使う設計になっていることを実測で確認した。

---

## 2. 典型操作の対応表

以下は代表的な操作についてpandas版・polars版のコードを両方実際に実行し、**結果が一致することをPythonの等価比較で検証済み**の対応表(検証スクリプトで15項目すべて`equal: True`を確認)。

| 分類 | pandas | polars | 検証結果 |
|---|---|---|---|
| 列選択 | `df[["team", "score"]]` | `df.select("team", "score")` | 一致 |
| フィルタ | `df[df["score"] > 20]` | `df.filter(pl.col("score") > 20)` | 一致 |
| 列追加 | `df.assign(score2=lambda d: d["score"]*2)` | `df.with_columns((pl.col("score")*2).alias("score2"))` | 一致 |
| groupby+複数集計 | `df.groupby("team").agg(total=("score","sum"), avg=("score","mean"))` | `df.group_by("team").agg(total=pl.col("score").sum(), avg=pl.col("score").mean())` | 一致 |
| グループ内集計をブロードキャスト | `df.groupby("team")["score"].transform("mean")` | `pl.col("score").mean().over("team")` | 一致 |
| 内部結合 | `pd.merge(left, right, on="id", how="inner")` | `left.join(right, on="id", how="inner")` | 一致 |
| 完全外部結合 | `pd.merge(left, right, on="id", how="outer")` | `left.join(right, on="id", how="full", coalesce=True)` | 一致(※下記注) |
| ピボット(集計あり) | `pd.pivot_table(df, values="val", index="date", columns="var", aggfunc="sum")` | `df.pivot(on="var", index="date", values="val", aggregate_function="sum")` | 一致 |
| 横→縦変形 | `df.melt(id_vars="id", value_vars=["x","y"])` | `df.unpivot(index="id", on=["x","y"])` | 一致 |
| 降順ソート | `df.sort_values("a", ascending=False)` | `df.sort("a", descending=True)` | 一致(引数の意味が反転する点に注意) |
| 欠損行除去 | `df.dropna()` | `df.drop_nulls()` | 一致 |
| 欠損値埋め | `df.fillna(0)` | `df.fill_null(0)` | 一致 |
| 値の出現回数 | `s.value_counts()` | `s.value_counts()`(結果は列`count`を持つDataFrame) | 一致 |
| 重複除去 | `df.drop_duplicates()` | `df.unique(maintain_order=True)` | 一致(`maintain_order`必須) |
| 文字列大文字化 | `s.str.upper()` | `s.str.to_uppercase()` | 一致 |

**完全外部結合の注**: 結果セットの値自体は一致したが、`pd.merge(..., how="outer")`は欠損値混入により`val`列が`int64`→`float64`に自動昇格するのに対し、`left.join(..., how="full", coalesce=True)`は`Int64`のまま`null`を保持する。実際の出力:

```
--- pandas how='outer' ---
   id name    val
0   1    a    NaN
1   2    b  200.0
2   3    c  300.0
3   4  NaN  400.0

--- polars how='full', coalesce=True ---
shape: (4, 3)
┌─────┬──────┬──────┐
│ id  ┆ name ┆ val  │
│ --- ┆ ---  ┆ ---  │
│ i64 ┆ str  ┆ i64  │
╞═════╪══════╪══════╡
│ 1   ┆ a    ┆ null │
│ 2   ┆ b    ┆ 200  │
│ 3   ┆ c    ┆ 300  │
│ 4   ┆ null ┆ 400  │
└─────┴──────┴──────┘
```

pandasは欠損値を表現するために数値列を`float64`へ昇格させる(`NaN`はfloat専用)のに対し、polarsは`null`を任意の型に持てるため型変換が発生しない。数値精度・メモリ効率の面でこれは無視できない差である。

---

## 3. パフォーマンス比較(実測ベンチマーク)

groupby+agg・filter・joinの3種類について、50万行と200万行の合成データで実測した。**各ケース3回実行した中央値**を掲載する(単一マシン・単一セッションでの計測であり、他のハードウェア・データ分布・ライブラリバージョンでは数値が変わりうる。あくまで傾向を掴むための参考値)。

データ生成:
```python
import numpy as np
rng = np.random.default_rng(42)
n = 500_000  # または 2_000_000
n_groups = 2000
group_ids = rng.integers(0, n_groups, size=n)
groups = np.array([f"g{gi:05d}" for gi in group_ids])
value1 = rng.normal(loc=100, scale=20, size=n)
value2 = rng.exponential(scale=5.0, size=n)
```
(join用に`group`列2000種をキーとする2000行のディメンションテーブルも別途生成)

### 3.1 groupby + 複数集計(sum/mean × 2列)

| 行数 | pandas(中央値) | polars Eager(中央値) | polars Lazy(中央値) | pandas/polars(Eager) |
|---|---|---|---|---|
| 500,000 | 23.1ms | 43.8ms | 33.0ms | **0.53x(pandasの方が速い)** |
| 2,000,000 | 96.4ms | 88.8ms | 79.5ms | 1.09x |

50万行では**pandasの方がpolars(Eager)より速かった**(pandas 23.1ms vs polars 43.8ms)。2000グループという中程度のカーディナリティでは、polarsのマルチスレッド化のオーバーヘッドがデータサイズに対して相対的に大きく、必ずしも「polarsが常に速い」わけではないことが実測から分かる。200万行になるとほぼ互角(pandas 96.4ms vs polars Eager 88.8ms、Lazy 79.5ms)まで差が縮まる。

### 3.2 フィルタ(2条件のAND)

| 行数 | pandas(中央値) | polars Eager(中央値) | polars Lazy(中央値) | pandas/polars(Eager) |
|---|---|---|---|---|
| 500,000 | 6.8ms | 2.3ms | 2.6ms | 2.96x |
| 2,000,000 | 16.1ms | 6.8ms | 9.6ms | 2.36x |

フィルタはどちらのサイズでもpolarsが2.4〜3.0倍高速だった。

### 3.3 結合(左テーブル50万/200万行 × 2000行のディメンションテーブル、left join)

| 行数 | pandas(中央値) | polars Eager(中央値) | polars Lazy(中央値) | pandas/polars(Eager) |
|---|---|---|---|---|
| 500,000 | 77.1ms | 23.1ms | 19.0ms | 3.33x |
| 2,000,000 | 312.4ms | 38.8ms | 30.6ms | **8.04x** |

結合はサイズが大きくなるほど差が開き、200万行では**pandasの8倍**(Lazy APIなら10.2倍)polarsが高速だった。今回計測した3操作の中で最もpolarsの優位が大きい処理だった。

### 3.4 マルチスレッドの効果(pandasには存在しない設定軸)

polars特有の設定として、環境変数`POLARS_MAX_THREADS`でスレッド数を変えて200万行の`group_by().agg()`と`sort()`を計測した(サブプロセスを分けて実行し、`pl.thread_pool_size()`で反映を確認):

| 処理 | 16スレッド(デフォルト) | 1スレッド | 倍率 |
|---|---|---|---|
| `group_by().agg()` | 131.2ms | 411.1ms | **3.1x** |
| `sort()` | 214.6ms | 246.1ms | 1.15x |

`group_by().agg()`はマルチスレッド化の恩恵が大きい(3.1倍)一方、`sort()`は今回のデータ・処理内容ではスレッド数を増やしてもほとんど変わらなかった(1.15倍)。すべての処理が均等にマルチスレッドの恩恵を受けるわけではない点は実測で確認しておく価値がある。

### まとめ

| 処理 | 傾向 |
|---|---|
| groupby+agg | データ規模・グループ数次第でpandasが優位になることもある(50万行では pandas優位、200万行ではほぼ互角〜微優位) |
| filter | 一貫してpolarsが2〜3倍程度高速 |
| join | データが大きいほどpolarsの優位が拡大(200万行で8〜10倍) |

「polarsは常にpandasより速い」と単純化はできず、少なくとも今回のベンチマーク条件では**操作の種類とデータ規模によって差の大きさ(場合によっては優劣)が変わる**ことが実測から分かった。

---

## 4. 移行時の注意点

pandas 3.0の仕様(CoW常時有効・文字列列のデフォルトdtypeが`StringDtype`になったこと、詳細は[pandas/README.md](../pandas/README.md)冒頭を参照)を踏まえた上で、pandas→polars移行特有の罠を実行検証した。

### 4.1 混在型リストの扱い: 黙示的`object`化 vs 構築時エラー

```python
mixed = [1, "two", 3]
pd.DataFrame({"a": mixed})["a"].dtype   # object ← エラーにならず黙って object 型になる
pl.DataFrame({"a": mixed})              # TypeError: unexpected value while building Series of type Int64; found value of type String: "two"
```

pandasは型が混在するリストを黙って`object`dtypeとして受け入れるが、polarsはデフォルト(`strict=True`)で先頭要素から推論した型に合わない値があると構築時に例外を送出する。pandasのコードをそのまま移植すると、データに型の混入があった場合にpolars側で初めて気づく(＝早期にエラーで検知できる)ことになる。

### 4.2 整数オーバーフロー: pandasは`astype`で「静かに」ラップアラウンドする

```python
s = pd.Series([1, 2, 300])          # デフォルト int64
s.astype("int8")                     # [1, 2, 44] ← 例外を出さずに 300 % 256 = 44 に丸め込まれる(サイレントなバグの温床)

pl.Series("x", [1, 2, 300], dtype=pl.Int8)
# TypeError: unexpected value while building Series of type Int8; found value of type Int64: 300
```

これは実行して初めて分かった、実務上もっとも注意すべき差である。**pandasの`astype("int8")`は範囲外の値を例外なくラップアラウンドさせる**(300→44)。一方、`pd.Series([1, 2, 300], dtype="int8")`のように**コンストラクタに直接`dtype`を渡した場合はpandasも`OverflowError`を送出する**ため、「pandasは常にオーバーフローを無視する」わけではなく、**`astype`による事後変換だけがこの罠に該当する**ことを確認した。polarsは構築時・`cast`時とも`strict=True`がデフォルトのため、範囲外の値があれば必ず例外になる(`strict=False`を明示すれば`null`に変換するモードを選べる)。

### 4.3 `NaN`と`null`は別物

```python
s_pd = pd.Series([1.0, float("nan"), None])
s_pd.isna().sum()   # 2 ← NaNもNoneも同じ「欠損」として扱われる

s_pl = pl.Series("s", [1.0, float("nan"), None])
s_pl.is_null().sum()  # 1 ← Noneのみ
s_pl.is_nan().sum()   # 1 ← float("nan")のみ
```

pandasはNaN・Noneをまとめて「欠損値」として扱う(`isna()`が両方拾う)が、polarsは`null`(欠損)と`NaN`(浮動小数点の非数)を明確に区別する。pandasの`isna()`ベースの欠損値処理コードをそのまま移植すると、polarsでは`is_null()`だけでは`NaN`を検出できず欠損判定が漏れる可能性がある(必要なら`is_null() | is_nan()`のように明示的に両方チェックする)。

### 4.4 `group_by()`の出力順序は保証されない(デフォルト)

```python
for _ in range(5):
    df.group_by("team").agg(pl.col("v").sum())["team"].to_list()[:5]
# 5回とも異なる順序が出力される(実測で確認)
```

`maintain_order=True`を指定しない`group_by()`は、内部的に並列処理されるため実行のたびに出力行順序が変わりうることを、同一データに対する5回の実行結果を比較して確認した(5回とも先頭5グループの並びが異なった)。pandasの`groupby(sort=True)`(デフォルト)がキーでソートされた安定した順序を返すのとは対照的で、テストコードで出力順序に依存した比較を書くと非決定的に失敗する原因になる。

### 4.5 join時のキー列dtype不一致は両者ともエラーにならない(想定と異なった点)

事前の想定では「polarsは型に厳格なのでdtypeが違う列同士のjoinはエラーになるのでは」と考えたが、実際に`Int64`列と`Int32`列を`on="id"`でjoinしたところ、**pandas・polarsとも例外を出さず正常に結合できる**ことを確認した(pandasは共通のより広い型に暗黙変換、polarsも同様に型を揃えて結合する)。型に厳しいのは主に「列構築時」の値の範囲チェックであり、結合時のキー型の互換変換は両者とも寛容だった。

### 4.6 その他、実行して再確認できた既知の差異

- `select`/`filter`/`with_columns`のような非集約系の操作では、polarsも行の順序を保持する(実測で確認)。順序が変わりうるのは`group_by`や`unique`(`maintain_order`未指定時)など集約・重複排除系の操作のみ。
- `sum()`はpandas・polarsともデフォルトで欠損値(NaN/null)を無視して計算する(`[1.0, NaN, 3.0]`→pandas `4.0`、`[1.0, null, 3.0]`→polars `4.0`で一致)。
- `df_pl.to_pandas()`で得られる文字列列のdtypeは`str`(pandas 3.0のStringDtype)になり、逆に`pl.from_pandas()`でpandasのStringDtype列を読み込むとpolars側は`String`になる。相互変換において文字列dtypeの不整合は発生しなかった。
- `str.replace`のデフォルト置換件数がpandasとpolarsで逆(pandasは全置換、polarsは最初の1件のみで全置換には`replace_all`が必要)である点は、[polars/README.md](../polars/README.md)に既出の情報だが本比較でも再確認した:pandas `"banana"→"b@n@n@"`(全置換)、polars `str.replace`は`"banana"→"b@nana"`(1件のみ)、`str.replace_all`で`"b@n@n@"`(全置換)。

---

## 5. どちらを使うべきかの判断基準

上記の実測・検証結果を踏まえた、決めつけない形での判断材料:

**polarsが有利と言えそうな場面(実測に基づく)**

- **結合(join)を含む処理**: 実測で200万行のjoinはpandasの8〜10倍高速だった。複数テーブルの結合を伴うETL・特徴量エンジニアリングではpolarsの優位が大きい。
- **フィルタ主体の処理**: 実測で一貫して2〜3倍高速。大きなCSV/Parquetから条件を絞り込む用途(特に`pl.scan_csv`のLazy APIと組み合わせた述語プッシュダウン)は利点が大きい。
- **マルチコアを活かしたい場合**: `group_by`で実測3.1倍のスレッド数依存効果を確認した通り、CPUコア数が多い環境ほど恩恵を受けやすい(ただし`sort`のように恩恵が小さい処理もある)。
- **データ品質を早期に検出したい場合**: 4.1・4.2で確認した通り、型混在やオーバーフローをpolarsは構築時に例外として検出する。pandasの`astype`のようなサイレントなバグ混入を避けたいコードベースでは有利。

**pandasが有利、または優位が小さい場面(実測に基づく)**

- **中規模データでの単純な集計**: 実測では50万行のgroupby+aggでpandasの方が速かった(23.1ms vs polars 43.8ms)。「大は小を兼ねない」場面が実際に存在する。
- **既存エコシステムとの連携**: 本リポジトリに収録されている scikit-learn・statsmodels・seaborn・matplotlib など多くのライブラリはpandas DataFrame/Seriesを第一級の入力として想定しており(polarsは`.to_pandas()`経由の変換が必要になることが多い)、これらとの組み合わせが中心のワークフローではpandasの方が変換コストが少ない。
- **インデックスベースの既存コード資産**: `loc`/`iloc`・`MultiIndex`など、pandas固有の機能に強く依存した既存コードを持つ場合、polarsへの書き換えはAPIの再設計に近いコストがかかる(1.1節で確認した通り`loc`相当のAPIはpolarsに存在しない)。

**総括**: 実測結果は「polarsは結合・フィルタで明確に優位、集計は規模次第、全体としてマルチスレッドとLazy APIの恩恵が大きいほどpolarsが有利になる」という傾向を示した。一方で「常にpolarsが速い」という単純化は本ベンチマークでは支持されず(50万行のgroupby+aggではpandasが優位)、既存のpandasエコシステムとの親和性も無視できない要素である。新規に大規模データ処理パイプラインを組む場合はpolars(特にLazy API)を第一候補にしつつ、既存資産や周辺ライブラリとの連携が多い場合はpandasを維持する、という使い分けが実測・実装の両面から妥当と考えられる。
