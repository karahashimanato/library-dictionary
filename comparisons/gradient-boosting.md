# 勾配ブースティング3種比較: xgboost / lightgbm / catboost

xgboost 3.4.1 / lightgbm 4.7.0 / catboost 1.2.10 で検証済み。すべて `/home/manaty/library-practicing/.venv` に実際にインストールされたバージョンで、CPU環境(WSL2, `x86_64`, 16 vCPU)で実行した結果に基づく。GPUは未検証。

このドキュメントは各ライブラリ単体のAPI逆引き([xgboost](../xgboost/README.md)、[lightgbm](../lightgbm/README.md)、[catboost](../catboost/README.md))とは性質が異なり、3ライブラリを横断して「木の育て方」「カテゴリ変数の扱い」「デフォルト値」「実測速度・精度」を比較し、使い分けの判断材料を提供することを目的とする。個別のAPI一覧はリンク先の各READMEを参照。

すべての数値・挙動は本ドキュメント作成時に実際にコードを実行して得たものであり、記憶からの推測は一切含まない(実行に使ったスクリプトの要点はコードブロックとして本文中に埋め込んでいる)。

## 目次

1. [木の成長戦略の違い](#1-木の成長戦略の違い)
2. [カテゴリ変数の扱いの違い](#2-カテゴリ変数の扱いの違い)
3. [デフォルトハイパーパラメータ対応表](#3-デフォルトハイパーパラメータ対応表)
4. [学習速度・精度の実測比較](#4-学習速度精度の実測比較)
5. [どちらを使うべきかの判断基準](#5-どちらを使うべきかの判断基準)

---

## 1. 木の成長戦略の違い

一般に「xgboostはdepth-wise、lightgbmはleaf-wise、catboostはoblivious tree(symmetric tree)」と言われる。実際に同一データ(`sklearn.datasets.make_classification`, 2000サンプル・10特徴量、`random_state=0`)で1本だけ木を学習させ、`trees_to_dataframe()`(xgboost/lightgbm)と`plot_tree()`(catboost)で実際のノード構造を確認した。

```python
X, y = make_classification(n_samples=2000, n_features=10, n_informative=6,
                            n_redundant=2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
```

### xgboost(`max_depth=4`, `grow_policy`未指定=`depthwise`)

`trees_to_dataframe()`で1本目の木を確認すると、ノード数は29個。ほとんどの葉がdepth3〜4に分布するが、1箇所(ノード`0-13`)だけdepth3で葉になっており、他の枝はdepth4まで伸びている。つまり depth-wise は「同じ深さのノードを一斉に展開する」戦略だが、各ノードごとに分割利得が閾値を下回れば個別に葉として打ち切るため、木全体としては完全対称にはならない(全ノードを機械的に埋めるわけではない)。

```
depth0: root(0-0, f2)
depth1: 0-1(f7), 0-2(f7)
depth2: 0-3(f2), 0-4(f2), 0-5(f5), 0-6(f1)
depth3: 0-7..0-12(分割継続), 0-13(Leaf ← ここだけ早期に葉化)
depth4: 0-14以降の分割から生じた葉(0-15〜0-28)
```

### lightgbm(`max_depth=4`, `num_leaves=31`, デフォルトのleaf-wise)

同条件で1本目の木は27ノード・葉14枚。`node_depth`列(root=1始まり)を見ると、葉は深さ3〜4(0始まり換算)に散らばっており、木の左右で深さが不揃い。例えば`0-S12`の下は深さ4で葉になる一方、`0-S7`の下は深さ3で葉になっている。これはleaf-wiseが「深さ」ではなく「その時点で最も利得の大きい葉」を優先して分割するためで、`num_leaves=31`という上限があっても、`max_depth=4`の制約に先に達して14枚で成長が止まった(全31枚は使い切っていない)。

### catboost(`depth=4`, デフォルトの`grow_policy=SymmetricTree`)

`plot_tree()`で出力したGraphvizを見ると、**同じ深さのノードは全て同一の分割条件(特徴量・閾値)を使っている**ことが直接確認できる。

- depth1: 全ノードが`feature 7, value>-0.325313`
- depth2: 全ノードが`feature 3, value>-1.78807`
- depth3: 全ノードが`feature 2, value>-0.254413`

これが oblivious tree(symmetric tree)の実体で、depth=4なら必ず2^4=16枚の葉・31ノードになる(実際に出力されたノード数は31で一致)。同一深さで分岐条件を共有するため、各サンプルがどの葉に落ちるかはビット列的に決まり、予測時のベクトル化・並列化が容易になる(4章の実測でcatboostの予測時間が最短だった一因と考えられる)。

**まとめ**: xgboost/lightgbmは「木の形がデータに応じて不揃いになる」のに対し、catboostは「深さで区切って強制的に対称にする」という設計思想の違いが、実際のノード出力からも確認できた。

---

## 2. カテゴリ変数の扱いの違い

「catboostはカテゴリ変数をそのまま渡せる」という説明はよく見るが、"何も指定しなくても自動処理されるか" を実際に確かめた。検証には文字列カテゴリ列(pandas `str`型、pandas 3.0.5のデフォルト文字列dtype)と`category`型列を含むDataFrameを使用。

```python
df = pd.DataFrame({
    "color": rng.choice(["red","blue","green","yellow"], size=n),  # dtype: str
    "size_cat": pd.Categorical(rng.choice(["S","M","L"], size=n)), # dtype: category
    "num": rng.normal(size=n),
})
```

実測結果:

| ライブラリ | 素の文字列列(`str`/objectのまま) | `category`型に変換した列 | 追加設定 |
|---|---|---|---|
| xgboost | `ValueError`(`enable_categorical`をTrueにせよ、という趣旨のメッセージ) | デフォルト(`enable_categorical=True`)でOK、`feature_types=['c','float']` | `category`型限定。`enable_categorical=False`を明示すると`category`型でもエラー |
| lightgbm | `ValueError: pandas dtypes must be int, float or bool`(`categorical_feature`を明示指定してもエラーは変わらず) | デフォルトで自動検出されOK | `category`型限定。列名を`categorical_feature`で指定しても、dtypeが`str`のままでは通らない |
| catboost | `cat_features`未指定だと`CatBoostError: Cannot convert 'red' to float`(数値列として扱おうとして失敗) | `cat_features`未指定だと同様にエラー | `cat_features=["color","size_cat"]`のように**明示指定すれば**、`str`型でも`category`型でも(one-hotやエンコード不要で)そのまま学習できた |

**分かったこと**: 3ライブラリとも「文字列型のまま何も指定せず学習できる」わけではない。

- xgboost・lightgbmは、pandasの`category`型に変換しておくことが前提(生の文字列/object列は明示指定してもエラーになる)。
- catboostは`category`型への変換は不要な代わりに、`cat_features`引数での明示指定が必須(何もしないとやはりエラーになる)。「catboostは何もしなくてもカテゴリを自動処理する」というのは不正確で、正しくは「**エンコード(one-hot/label encoding)なしで生の文字列を渡せる。ただしどの列がカテゴリかは自分で教える必要がある**」。
- 3ライブラリの中で「dtypeさえ`category`にしておけば追加引数なしで自動検出される」のはlightgbmのみ(実測で確認)。

---

## 3. デフォルトハイパーパラメータ対応表

xgboost・lightgbmはsklearn APIの`__init__`シグネチャにデフォルト値が明示されているが、xgboostはコンストラクタ上は全て`None`(未学習時の`get_params()`も`None`)で、実際にfitして`Booster.save_config()`のJSONを見ないと有効値が分からない。catboostも同様にコンストラクタは全て`None`で、`fit`後の`get_all_params()`で初めて実効値が判明する。以下は全て実機で確認した値。

```python
# xgboost: get_booster().save_config() のJSONから実効値を取得
# catboost: fit後の get_all_params() から実効値を取得(n_samples=5000, n_features=10のダミーデータで検証。
#           learning_rateなど一部の値はデータサイズ依存(後述)なので、この表の値はこの条件での実測値)
# lightgbm: get_params() がコンストラクタ時点でそのまま実効値
```

| 意味 | xgboost (`XGBClassifier`, 実効値) | lightgbm (`LGBMClassifier`, 実効値) | catboost (`CatBoostClassifier`, 実効値) |
|---|---|---|---|
| 木の本数 | `n_estimators`: **100** | `n_estimators`: **100** | `iterations`: **1000** |
| 学習率 | `learning_rate`(`eta`): **0.3** | `learning_rate`: **0.1** | `learning_rate`: **データ依存の自動値**(下記参照。固定デフォルト値は存在しない) |
| 木の深さ | `max_depth`: **6** | `max_depth`: **-1**(無制限、`num_leaves`で実質制御) | `depth`: **6** |
| 葉数上限 | `max_leaves`: **0**(無効、depth基準) | `num_leaves`: **31** | `max_leaves`: **64**(`get_all_params()`実測値。symmetric treeのため`2^depth`(depth=6→64)が自動的に設定される) |
| 葉の最小サンプル数系 | `min_child_weight`: **1** | `min_child_samples`: **20** | `min_data_in_leaf`: **1** |
| L1正則化 | `reg_alpha`: **0** | `reg_alpha`: **0.0** | (L1相当のパラメータなし。`get_all_params()`にも該当キーは存在しない) |
| L2正則化 | `reg_lambda`: **1** | `reg_lambda`: **0.0** | `l2_leaf_reg`: **3** |
| 行サブサンプリング | `subsample`: **1**(無効) | `subsample`: **1.0**(無効) | `bootstrap_type`: **MVS**、`subsample`実効値: **0.8**(`get_all_params()`実測。MVS bootstrap時のデフォルト比率) |
| 列サブサンプリング | `colsample_bytree`: **1** | `colsample_bytree`: **1.0** | `rsm`: **1**(無効) |
| 木の成長方針 | `grow_policy`: **depthwise** | (leaf-wiseのみ、切替パラメータなし) | `grow_policy`: **SymmetricTree** |
| ヒストグラムのbin数 | `max_bin`(内部): **256** | `max_bin`: **255**(`Dataset.feature_num_bin()`で連続値1本のダミー特徴量に対して実測) | `border_count`: **254** |
| ブースティング方式 | `booster`: **gbtree** | `boosting_type`: **gbdt** | `boosting_type`: **Plain**(⚠️名前は同じ「boosting_type」だが意味が違う。下記参照) |

**⚠️ 名前の罠(`boosting_type`)**: lightgbmの`boosting_type`は「gbdt/dart/goss」など勾配ブースティングのアルゴリズム種別を指すのに対し、catboostの`boosting_type`は「Ordered(順序付きブースティング、リーク防止のため各木を異なる並び順のデータで学習)」か「Plain(通常のブースティング)」かを指す、全く別の概念。同名パラメータだが指しているものが違う点は誤解しやすい。

**⚠️ catboostの`learning_rate`はデータサイズ依存**: 実際にサンプルサイズを変えて確認したところ、`learning_rate=None`(未指定)時の自動決定値は以下のように変化した(`n_features=10`, `iterations=1000`固定)。

| n_samples | 自動決定された learning_rate |
|---|---|
| 200 | 0.00518... |
| 5,000 | 0.02048... |
| 50,000 | 0.05475... |

xgboost(0.3固定)・lightgbm(0.1固定)がデータサイズによらず一定のデフォルト値を使うのに対し、catboostはデータ量が多いほど大きい学習率を自動選択する(小さいデータで大きい学習率を使うと過学習しやすいための調整と考えられる。ドキュメントの根拠は確認していないため、挙動のみ事実として記載)。

---

## 4. 学習速度・精度の実測比較

**注意**: 以下は合成データ1本(1シード)での1回限りの実測であり、厳密なベンチマークではない。データ・乱数・環境が変われば結果は変わりうる。

### データセット・設定

```python
X, y = make_classification(
    n_samples=300_000, n_features=30, n_informative=15, n_redundant=5,
    n_clusters_per_class=3, flip_y=0.02, random_state=0
)
# train:test = 240,000 : 60,000 (stratify=y)
```

3ライブラリともほぼ同等の設定で揃えた: `n_estimators`(`iterations`)=300, 木の深さ=6(lightgbmのみ`num_leaves=63`≒2^6で近似), `learning_rate=0.1`, `n_jobs`/`thread_count`=16(全コア), CPUのみ(`tree_method="hist"` / デフォルト / `task_type="CPU"`)。

### 実測結果(1回のみの実行)

| ライブラリ | 学習時間 | 予測時間(60,000件) | AUC(テスト) |
|---|---|---|---|
| xgboost | **66.64秒** | 0.078秒 | 0.98346 |
| lightgbm | **36.93秒** | 0.117秒 | 0.98297 |
| catboost | **6.01秒** | **0.017秒** | 0.98206 |

AUCは3者ともほぼ同水準(小数点2桁目までは横並び)で、この実測に関しては精度差はほぼ無いに等しい。一方で学習時間はcatboostが圧倒的に短く、xgboostの約1/11、lightgbmの約1/6だった。これは「catboostは遅い」という一般論(順序付きブースティングのコストに基づく)とは逆の結果だったため、原因を追加検証した。

### なぜcatboostが速かったか: `boosting_type`の自動切り替え

上記の学習後に`get_all_params()`で確認すると、`boosting_type: 'Plain'`が自動選択されていた。catboostは既定で「小さいデータではOrdered(順序付き)ブースティング、大きいデータではPlain(通常)ブースティング」を自動選択する(`boosting_type`を明示しなかった場合)。240,000行という今回のデータ規模ではPlainが選ばれ、Ordered特有の計算コストを払っていなかったことが速さの一因と考えられる。

これを検証するため、5万行のサブセットで`boosting_type`を`"Plain"`と`"Ordered"`に明示的に固定して比較した:

| `boosting_type` | 学習時間(5万行) | AUC |
|---|---|---|
| Plain | 2.22秒 | 0.98386 |
| Ordered | 8.86秒 | 0.98410 |

Ordered指定時はPlainの約4倍の時間がかかり、AUCはわずかに上回った(0.98410 vs 0.98386、この程度の差は今回の1回実行の範囲では誤差の可能性が高い)。つまり4章冒頭の速度差は、木の構造(oblivious tree)だけでなく、**catboostがデータ規模に応じて自動的に軽量なPlainモードを選んでいたこと**が大きく寄与している。同じデータでcatboostに`boosting_type="Ordered"`を強制すれば、この差はかなり縮まる可能性が高い(全データでの再検証はしていない)。

予測時間についてはoblivious tree(symmetric tree)の構造上、全サンプルが同じ分岐条件を通るためベクトル化・分岐予測がしやすく、catboostが最速だった点は1章の木構造の違いと整合する。

---

## 5. どちらを使うべきかの判断基準

一般論として言われることが多い基準と、今回の実測で裏付けられた/裏付けられなかった点を分けて書く。

| 一般論 | この実測での裏付け状況 |
|---|---|
| カテゴリ変数が多いならcatboost | 裏付けられた(2章)。ただし「何もしなくても自動」ではなく`cat_features`の明示指定は必要。lightgbmも`category`型に変換すれば自動検出される点は見落とされがち。 |
| 速度重視ならlightgbm | **今回の実測では裏付けられなかった**。むしろcatboostが最速(6.01秒 vs lightgbm 36.93秒)。ただしこれは中規模データでPlainブースティングが自動選択されたことが大きく、小規模データでOrdered boostingが選ばれる状況やGPU利用時、パラメータチューニング後は結果が変わりうる。「lightgbmは軽量」という評判は、ヒストグラムベースでメモリ効率が良いことに由来しており、今回計測していないメモリ使用量では別の結果になる可能性がある。 |
| xgboostは枯れていて情報が多い・安定 | 実測の範囲外(エコシステムの成熟度は速度・精度のベンチマークでは測れない)。今回の実測では3者中もっとも学習が遅かった(66.64秒)が、`tree_method`やパラメータチューニング次第で改善余地がある。 |
| 精度はほぼ横並び | 今回の1データセット・1設定では裏付けられた(AUC差は小数点3桁目)。ハイパーパラメータチューニングをしていないため、チューニング後の差は未検証。 |

**まとめ(今回の実測に基づく限定的な結論)**:
- カテゴリ変数を多く含み、エンコード前処理を省きたい → catboost(ただし`cat_features`の指定は必須)。
- 今回計測したような中規模(数十万行)・CPU環境での学習速度を最優先するなら、今回の実測ではcatboost(Plainモード)が優位だった。ただしこれは「catboost=常に速い」ではなく、「データ規模に応じた自動最適化がうまく働いた」結果である点に注意。
- 精度だけを見るなら、今回の設定・データでは3者に実質的な差はなかった。
- いずれも1回・1データセットの実測であり、実データやハイパーパラメータチューニングを行った場合の傾向は別途検証が必要。
