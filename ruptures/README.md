# ruptures 逆引き辞書

ruptures 1.1.10 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(ruptures 1.1.10)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。

## 目次

1. [合成データ生成](#1-合成データ生成)
2. [基礎検出アルゴリズム](#2-基礎検出アルゴリズム)
3. [コスト関数](#3-コスト関数)
4. [評価指標](#4-評価指標)
5. [可視化](#5-可視化)
6. [パラメータ・チューニング](#6-パラメータチューニング)
7. [応用・発展](#7-応用発展)
   - 7.1 [カスタムコスト関数の自作・拡張](#71-カスタムコスト関数の自作拡張)
   - 7.2 [多変量信号・欠損値のある信号への対応](#72-多変量信号欠損値のある信号への対応)
   - 7.3 [実データでの閾値選定の実践パターン](#73-実データでの閾値選定の実践パターン)
   - 7.4 [オンライン・ストリーミング風の逐次検出](#74-オンラインストリーミング風の逐次検出)

---

## 1. 合成データ生成

### `pw_constant(...)`

**用途**: 区分的に定数(平均がジャンプする)な人工信号と真の変化点を生成する。検出アルゴリズムの動作確認用。

**シグネチャ**: `ruptures.pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=None, delta=(1, 10), seed=None)`

**使用例**:
```python
import ruptures as rpt

signal, bkps = rpt.pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=1, seed=42)
print(signal.shape)
print(bkps)
print(signal[:5].ravel())
```
実行結果:
```
(200, 1)
[51, 103, 149, 200]
[8.09344484 7.64936184 7.94880328 7.11256051 8.84500241]
```

**注意点・落とし穴**:
- 返される変化点リスト `bkps` は必ず信号長(`n_samples`)を最後の要素として含む(ruptures全体の共通仕様。「最後のセグメントの終端」を表す)。実際の変化点は `bkps[:-1]` の3つ(51, 103, 149)。
- `n_bkps=3` を指定しても、乱数(`seed`)次第で実際に生成される信号形状(ジャンプ幅=`delta`)は変わる。

### `pw_linear(...)`

**用途**: 区分的に線形回帰の係数が変化する人工信号を生成する。`CostLinear` の検証に使う。

**シグネチャ**: `ruptures.pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=None, seed=None)`

**使用例**:
```python
import ruptures as rpt

signal, bkps = rpt.pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=0.5, seed=42)
print(signal.shape, bkps)
```
実行結果:
```
(200, 2) [51, 103, 149, 200]
```

**注意点・落とし穴**:
- `n_features=1` を指定しても出力の列数は `n_features + 1` になる(先頭列が目的変数、残りが説明変数)。docstring通りの仕様で、`CostLinear`/`CostCLinear` の入力形式(1列目=観測値、以降=共変量)と対応している。

### `pw_normal(...)`

**用途**: 2次元の区分的ガウス信号(平均・共分散が変化)を生成する。`CostNormal`/`CostMl` の検証に使う。

**シグネチャ**: `ruptures.pw_normal(n_samples=200, n_bkps=3, seed=None)`

**使用例**:
```python
import ruptures as rpt

signal, bkps = rpt.pw_normal(n_samples=200, n_bkps=3, seed=42)
print(signal.shape, bkps)
```
実行結果:
```
(200, 2) [51, 103, 149, 200]
```

**注意点・落とし穴**:
- `n_features` 引数を持たない。常に2次元(`shape=(n_samples, 2)`)の信号が生成される。

### `pw_wavy(...)`

**用途**: 区分的に周波数が変化する1次元の波状信号を生成する。

**シグネチャ**: `ruptures.pw_wavy(n_samples=200, n_bkps=3, noise_std=None, seed=None)`

**使用例**:
```python
import ruptures as rpt

signal, bkps = rpt.pw_wavy(n_samples=200, n_bkps=3, noise_std=0.5, seed=42)
print(signal.shape, signal.ndim, bkps)
```
実行結果:
```
(200,) 1 [51, 103, 149, 200]
```

**注意点・落とし穴**:
- **docstringとの相違(実行確認済み)**: docstringには `shape (n_samples, 1)` と書かれているが、実際に返るのは1次元配列 `shape=(200,)`(`ndim=1`)。`Pelt` などに渡す際、他のアルゴリズムが2次元入力を期待する場面では `signal.reshape(-1, 1)` が必要になることがある。

---

## 2. 基礎検出アルゴリズム

以降の例では共通して以下の信号を使う(`true_bkps: [51, 103, 149, 200]`)。

```python
import ruptures as rpt
signal, true_bkps = rpt.pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=1, seed=42)
```

### `Pelt(...)`

**用途**: ペナルティ`pen`を指定して最適な変化点数・位置を線形時間(平均的に)で探索する。変化点数が未知のときの定番アルゴリズム。

**シグネチャ**: `ruptures.Pelt(model='l2', custom_cost=None, min_size=2, jump=5, params=None)` / `.fit(signal) -> Pelt` / `.predict(pen)`

**使用例**:
```python
algo = rpt.Pelt(model="l2", min_size=2, jump=5).fit(signal)
print(algo.predict(pen=50))
```
実行結果:
```
[50, 105, 150, 200]
```

**注意点・落とし穴**:
- `predict()` は `pen`(ペナルティ)のみを受け取り、`n_bkps` は指定できない。`algo.predict(n_bkps=3)` は `TypeError: Pelt.predict() got an unexpected keyword argument 'n_bkps'` になる(実行確認済み)。変化点数を指定したい場合は `Binseg`/`BottomUp`/`Dynp`/`KernelCPD` を使う。
- `pen` が小さいほど変化点を検出しやすくなる(過検出)。同じ信号で `pen=1` だと19個もの変化点を検出したが、`pen=50` では真の変化点に近い4個(3変化点+終端)になった(実行確認済み)。

### `Binseg(...)`

**用途**: 二分探索的(Binary Segmentation)に、分散削減が最大となる点を逐次的に追加していく。

**シグネチャ**: `ruptures.Binseg(model='l2', custom_cost=None, min_size=2, jump=5, params=None)` / `.fit(signal) -> Binseg` / `.predict(n_bkps=None, pen=None, epsilon=None)`

**使用例**:
```python
algo = rpt.Binseg(model="l2").fit(signal)
print(algo.predict(n_bkps=3))
```
実行結果:
```
[50, 105, 150, 200]
```

**注意点・落とし穴**:
- `predict()` は `n_bkps`/`pen`/`epsilon` のいずれか1つを必ず指定する必要がある。何も渡さないと `AssertionError: Give a parameter.` になる(実行確認済み)。
- 貪欲法(逐次追加)のため `Dynp`(厳密解)より高速だが、最適解が保証されない。

### `BottomUp(...)`

**用途**: 信号を細かく分割した状態から始め、コスト増加が最小のペアを逐次的にマージしていく(Binsegと逆方向のアプローチ)。

**シグネチャ**: `ruptures.BottomUp(model='l2', custom_cost=None, min_size=2, jump=5, params=None)` / `.fit(signal) -> BottomUp` / `.predict(n_bkps=None, pen=None, epsilon=None)`

**使用例**:
```python
algo = rpt.BottomUp(model="l2").fit(signal)
print(algo.predict(n_bkps=3))
```
実行結果:
```
[50, 105, 150, 200]
```

**注意点・落とし穴**:
- `Binseg` 同様 `n_bkps`/`pen`/`epsilon` のいずれかが必須。
- このサンプル(`noise_std=1`)では `Binseg` と同じ結果になったが、信号によってはマージ方向(ボトムアップ)と分割方向(Binseg)で結果が異なりうる。

### `Window(...)`

**用途**: 信号上を固定幅の2つの窓をスライドさせ、窓間のコスト差が大きい点を変化点候補とするスライディングウィンドウ法。

**シグネチャ**: `ruptures.Window(width=100, model='l2', custom_cost=None, min_size=2, jump=5, params=None)` / `.fit(signal) -> Window` / `.predict(n_bkps=None, pen=None, epsilon=None)`

**使用例**:
```python
algo = rpt.Window(width=40, model="l2").fit(signal)
print(algo.predict(n_bkps=3))
```
実行結果:
```
[50, 105, 150, 200]
```

**注意点・落とし穴**:
- デフォルトの `width=100` は信号長200に対して大きすぎることがある。信号長に対して適切な `width` を明示的に指定すること。
- 返り値のリストの要素は `np.int64` になる箇所がある(実行確認済み。他のアルゴリズムでは素の `int` のことが多い)。`int()` でキャストしてから比較演算に使うと安全。

### `Dynp(...)`

**用途**: 動的計画法により、指定した変化点数に対する厳密な最適解(コスト最小)を求める。

**シグネチャ**: `ruptures.Dynp(model='l2', custom_cost=None, min_size=2, jump=5, params=None)` / `.fit(signal) -> Dynp` / `.predict(n_bkps)`

**使用例**:
```python
algo = rpt.Dynp(model="l2", min_size=2, jump=1).fit(signal)
print(algo.predict(n_bkps=3))
```
実行結果:
```
[51, 103, 149, 200]
```

**注意点・落とし穴**:
- `predict()` は `n_bkps` が必須の位置引数で、`pen` は使えない(厳密解を求めるアルゴリズムのため、ペナルティではなく変化点数を直接指定する設計)。
- `jump=1`(デフォルトは5)にしないと候補点が間引かれ、真の変化点(51, 103, 149)ちょうどを見つけられないことがある。今回は `jump=1` で真の変化点と完全一致した(実行確認済み)。
- 全候補点の組合せを評価するため、`model`がO(n)コストでも全体としては `O(n^2 * n_bkps)` 程度かかり、長い信号では低速。

### `KernelCPD(...)`

**用途**: カーネル法(線形・RBF・コサイン)によりノンパラメトリックに変化点を検出する。動的計画法ベースで `n_bkps`/`pen` どちらも指定可能。

**シグネチャ**: `ruptures.KernelCPD(kernel='linear', min_size=2, jump=1, params=None)` / `.fit(signal) -> KernelCPD` / `.predict(n_bkps=None, pen=None)`

**使用例**:
```python
algo = rpt.KernelCPD(kernel="rbf", min_size=2).fit(signal)
print(algo.predict(n_bkps=3))
```
実行結果:
```
[51, 103, 149, 200]
```

**注意点・落とし穴**:
- 使えるカーネルは `"linear"`(`kernel(x,y)=x^T y`)・`"rbf"`(`exp(-gamma*||x-y||^2)`)・`"cosine"` の3種類のみ。それ以外を渡すと `AssertionError` になる(公式docstringに明記、コード確認済み)。
- コンストラクタに `jump` 引数はあるが**常に1に固定され、指定値は無視される**(docstring: "not considered, set to 1")。他のアルゴリズム(Pelt等)と違い間引きは行われない。
- `pen` 指定時の返り値の要素は `np.int32` になることがある(実行確認済み)。

---

## 3. コスト関数

各コスト関数は `.fit(signal)` でデータを渡し、`.error(start, end)` でセグメント `[start, end)` のコスト(小さいほど当てはまりが良い)を計算する。`Pelt`等のコンストラクタに `model="l2"` のような文字列、または `custom_cost=CostL2()` のようにインスタンスを渡して使う。

### `CostL1()`

**用途**: 区分的定数信号に対するL1(絶対値誤差、中央値ベース)コスト。外れ値に頑健。

**シグネチャ**: `ruptures.costs.CostL1()`(引数なし。`model="l1"` 相当)

**使用例**:
```python
c = rpt.costs.CostL1().fit(signal)
print(c.error(0, 50))
print(c.error(0, 100))
```
実行結果:
```
28.226904837532366
442.3566378939532
```

**注意点・落とし穴**:
- セグメント内に変化点が実際に含まれる `[0, 100)` の方が `[0, 50)` より大幅にコストが高い(単調性はないが、変化点をまたぐと当てはまりが悪化する傾向がわかる)。

### `CostL2()`

**用途**: 区分的定数信号に対するL2(二乗誤差、平均ベース)コスト。ruptures全体のデフォルト(`model="l2"`)。

**シグネチャ**: `ruptures.costs.CostL2()`(引数なし)

**使用例**:
```python
c = rpt.costs.CostL2().fit(signal)
print(c.error(0, 50))
```
実行結果:
```
24.072071180781204
```

**注意点・落とし穴**:
- `CostL1`と同じセグメントでもコストの絶対値は直接比較できない(定義が異なるため)。同一コスト関数内でのみ大小比較する。

### `CostRbf(gamma=None)`

**用途**: RBFカーネルによるノンパラメトリックなコスト。分布形状に依存しない変化(平均以外の変化)も捉えやすい。

**シグネチャ**: `ruptures.costs.CostRbf(gamma=None)`

**使用例**:
```python
c = rpt.costs.CostRbf().fit(signal)
print(c.error(0, 50))
print("gamma:", c.gamma)
```
実行結果:
```
0.7623482976544622
0.011439436041543...
```

**注意点・落とし穴**:
- `gamma=None`(デフォルト)の場合、`fit()`時にデータのスケールから自動推定される(`c.gamma`で確認可能)。他の指標(L1/L2)とスケールがまったく異なるので、`pen`の値も流用できない。

### `CostLinear()`

**用途**: 説明変数に対する回帰係数が区分的に変化するデータの検出用コスト(最小二乗残差)。

**シグネチャ**: `ruptures.costs.CostLinear()`(引数なし)。`.fit(signal)` の入力は `shape=(n_samples, n_regressors+1)` で **1列目が目的変数、2列目以降が説明変数**。

**使用例**:
```python
signal_lin, bkps_lin = rpt.pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=0.5, seed=42)
algo = rpt.Pelt(custom_cost=rpt.costs.CostLinear()).fit(signal_lin)
print(algo.predict(pen=5))
# model="linear" という文字列指定でも同じ結果になる
algo2 = rpt.Pelt(model="linear").fit(signal_lin)
print(algo2.predict(pen=5) == algo.predict(pen=5))
```
実行結果:
```
[50, 100, 105, 150, 200]
True
```

**注意点・落とし穴**:
- `Pelt(model="linear")` と `Pelt(custom_cost=CostLinear())` は等価(`CostLinear.model = "linear"` として登録されているため)。実行結果が完全に一致することを確認済み。
- `pw_constant` などが返す通常の信号(観測値のみ)をそのまま渡すと `assert signal.ndim > 1` で失敗する。目的変数列を含む形式に整形する必要がある。

### `CostNormal(add_small_diag=True)`

**用途**: 多変量ガウス分布のパラメータ(平均・共分散)の変化を検出するコスト(負の対数尤度ベース)。

**シグネチャ**: `ruptures.costs.CostNormal(add_small_diag=True)`

**使用例**:
```python
signal_n, bkps_n = rpt.pw_normal(n_samples=200, n_bkps=3, seed=42)
c = rpt.costs.CostNormal().fit(signal_n)
print(c.error(0, 50))
```
実行結果:
```
-137.8956207181327
```
（`fit()`実行時に `UserWarning: New behaviour in v1.1.5: a small bias is added to the covariance matrix to cope with truly constant segments (see PR#198).` が出力される。実行確認済み)

**注意点・落とし穴**:
- 負の対数尤度ベースのため、`error()`の値は**負になりうる**(L1/L2のような非負コストではない)。他のコストと同じ感覚で「0に近いほど良い」とは限らない点に注意。
- `add_small_diag=True`(デフォルト)は上記のバージョン1.1.5からの挙動変更に対応するためのもので、共分散行列が特異になる(定数区間がある)場合の数値安定化のために小さなバイアスを加える。

### `CostAR(order=4)`

**用途**: 自己回帰(AR)モデルの係数が区分的に変化する時系列データを検出するコスト。

**シグネチャ**: `ruptures.costs.CostAR(order=4)`

**使用例**:
```python
signal_ar, bkps_ar = rpt.pw_normal(n_samples=200, n_bkps=3, seed=1)
c = rpt.costs.CostAR(order=4).fit(signal_ar[:, 0])
print(c.error(0, 50))
```
実行結果:
```
33.36777140155986
```

**注意点・落とし穴**:
- `order`(ARの次数)が大きいほど `min_size` も大きくなる(セグメント内のサンプル数が `order` を超えないと推定できない)。短いセグメントを検出したい場合は`order`を小さく保つ。

### `CostRank()`

**用途**: 各次元をランク変換してから評価するノンパラメトリックなコスト。外れ値・分布の裾の影響を受けにくい。

**シグネチャ**: `ruptures.costs.CostRank()`(引数なし)

**使用例**:
```python
c = rpt.costs.CostRank().fit(signal)
e = c.error(0, 50)
print(type(e), e)
```
実行結果:
```
<class 'numpy.ndarray'> [[-83.61382435]]
```

**注意点・落とし穴**:
- `error()` の返り値が他のコスト関数と異なり**スカラーではなく`(1, 1)`のndarray**になる(実行確認済み)。`Pelt`等の内部では問題なく動くが、`error()`を自前で使って数値として扱う場合は `float(e)` や `e.item()` で変換する必要がある。

### `CostMl(metric=None)`

**用途**: マハラノビス型の擬距離に基づくコスト。特徴量間の相関を考慮した変化検出に使う。

**シグネチャ**: `ruptures.costs.CostMl(metric=None)`

**使用例**:
```python
signal_n, bkps_n = rpt.pw_normal(n_samples=200, n_bkps=3, seed=42)
c = rpt.costs.CostMl().fit(signal_n)
print(c.error(0, 50))
```
実行結果:
```
126.58061159554622
```

**注意点・落とし穴**:
- `metric=None`(デフォルト)の場合、`fit()`時に信号全体の共分散行列の逆行列(マハラノビス計量)が自動的に推定される。独自の計量行列(PSD行列)を使いたい場合のみ `metric` を明示する。

---

## 4. 評価指標

`ruptures.metrics` は、真の変化点リストと検出結果を比較する関数群。以下は `Pelt(model="l2").fit(signal).predict(pen=10)` で得た `pred_bkps=[50, 55, 100, 105, 145, 150, 200]` を `true_bkps=[51, 103, 149, 200]` と比較した例。

### `precision_recall(...)`

**用途**: 許容誤差`margin`以内に一致する変化点の適合率・再現率を計算する。

**シグネチャ**: `ruptures.metrics.precision_recall(true_bkps, my_bkps, margin=10)`

**使用例**:
```python
import ruptures.metrics as m
print(m.precision_recall(true_bkps, pred_bkps, margin=10))
```
実行結果:
```
(0.5, 1.0)
```

**注意点・落とし穴**:
- 戻り値は `(precision, recall)` のタプル。この例では再現率1.0(真の変化点3つ全てが`margin=10`以内に検出された)だが、適合率0.5(検出した変化点のうち実際に一致したのは半分)と過検出気味であることがわかる。
- `margin`を3まで狭めても同じ結果だった(実行確認済み。`margin`は「一致とみなす許容距離」なので、真の変化点に近い検出点が複数あっても最も近いものだけが対応付けられる)。

### `hausdorff(...)`

**用途**: 2つの変化点集合間のハウスドルフ距離(最も離れた対応点間の距離)を計算する。

**シグネチャ**: `ruptures.metrics.hausdorff(bkps1, bkps2)`

**使用例**:
```python
print(m.hausdorff(true_bkps, pred_bkps))
```
実行結果:
```
4.0
```

**注意点・落とし穴**:
- 外れ値的な1点のズレが大きいと極端に悪化する(最大距離を見る指標のため、`precision_recall`のような平均的な評価とは性質が異なる)。

### `randindex(...)`

**用途**: 2つのセグメンテーション(区間分割)の一致度をRand indexで評価する(0〜1、大きいほど一致)。

**シグネチャ**: `ruptures.metrics.randindex(bkps1, bkps2)`

**使用例**:
```python
print(m.randindex(true_bkps, pred_bkps))
```
実行結果:
```
0.9653768844221106
```

**注意点・落とし穴**:
- 変化点の「個数」ではなく「同じ区間に属するサンプルペアがどれだけ一致するか」を見る指標なので、過検出があってもサンプル単位では高い値が出やすい(この例でも適合率0.5に対しRand indexは0.97と高め)。

### `hamming(...)`

**用途**: 2つのセグメンテーションの修正Hamming距離(0〜1、小さいほど一致)を計算する。

**シグネチャ**: `ruptures.metrics.hamming(bkps1, bkps2)`

**使用例**:
```python
print(m.hamming(true_bkps, pred_bkps))
```
実行結果:
```
0.03462311557788944
```

**注意点・落とし穴**:
- `randindex`とほぼ相補的な指標(`hamming ≈ 1 - randindex`に近い値になることが多い。今回も `0.0346 ≈ 1 - 0.9654`)。どちらか一方だけ見れば十分なことが多い。

### `meantime(...)`

**用途**: 真の変化点と検出変化点の平均時間差(検出位置のズレの平均)を計算する。

**シグネチャ**: `ruptures.metrics.meantime(true_bkps, my_bkps)`

**使用例**:
```python
pred_bkps_dynp = rpt.Dynp(model="l2").fit(signal).predict(n_bkps=3)
print(pred_bkps_dynp)
print(m.meantime(true_bkps, pred_bkps_dynp))
```
実行結果:
```
[50, 105, 150, 200]
1.3333333333333333
```

**注意点・落とし穴**:
- `precision_recall`等と異なり、真の変化点数と検出変化点数が一致している(対応付けがしやすい)場面向けの指標。数が大きく異なる場合は結果の解釈が難しくなる。

---

## 5. 可視化

### `display(...)`

**用途**: 信号に真の変化点(背景の色分け)と検出変化点(破線)を重ねて表示する。

**シグネチャ**: `ruptures.display(signal, true_chg_pts, computed_chg_pts=None, computed_chg_pts_color='k', computed_chg_pts_linewidth=3, computed_chg_pts_linestyle='--', computed_chg_pts_alpha=1.0, **kwargs)`

**使用例**:
```python
import matplotlib
matplotlib.use("Agg")
import ruptures as rpt

signal, true_bkps = rpt.pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=1, seed=42)
pred_bkps = rpt.Pelt(model="l2").fit(signal).predict(pen=50)
print(true_bkps, pred_bkps)

fig, axes = rpt.display(signal, true_bkps, pred_bkps)
fig.savefig("display_test.png")
```
実行結果:
```
[51, 103, 149, 200] [50, 105, 150, 200]
```
実際にPNGとして保存して描画を確認した。真のセグメント(`true_bkps`区切り)が背景の薄い青とピンクで交互に塗り分けられ、その上に検出変化点(`pred_bkps`)の位置に黒い太い破線が引かれる。信号の折れ線グラフと合わせて、真の変化点と検出結果のズレを一目で比較できる。

**注意点・落とし穴**:
- 戻り値は `(fig, axes)` で、`axes` は**信号の次元数(列数)ぶんのAxesを持つリスト**(1次元信号でも `[ax]` のリスト)。多変量信号では列ごとに別々のサブプロットが縦に並ぶ。
- `computed_chg_pts`(検出変化点)は省略可能(`None`がデフォルト)で、その場合は真の変化点による背景の色分けのみが表示される。

---

## 6. パラメータ・チューニング

### `pen`(ペナルティ) と `n_bkps`(変化点数)のトレードオフ

**用途**: `Pelt`/`KernelCPD`は`pen`で、`Binseg`/`BottomUp`/`Window`/`Dynp`は`n_bkps`(または`pen`)で変化点の「量」を制御する。どちらを使うかでワークフローが変わる。

**使用例**:
```python
algo = rpt.Pelt(model="l2", jump=5).fit(signal)
for pen in [1, 5, 10, 30, 50]:
    print(pen, algo.predict(pen=pen))
```
実行結果:
```
1  [20, 25, 45, 50, 55, 85, 95, 100, 105, 110, 120, 130, 135, 140, 145, 150, 170, 180, 200]
5  [50, 55, 100, 105, 130, 135, 145, 150, 200]
10 [50, 55, 100, 105, 145, 150, 200]
30 [50, 100, 105, 150, 200]
50 [50, 105, 150, 200]
```

**注意点・落とし穴**:
- `pen`が小さいほど「変化点を追加するコスト」が小さくなり過検出(多数の変化点)になる。`pen`が大きいほど過小検出(変化点数が減る)になる。この例では`pen=50`で初めて真の変化点数(3個+終端)と一致した。
- 真の変化点数を先に知っている場合は`n_bkps`を直接指定できる`Binseg`/`BottomUp`/`Dynp`/`KernelCPD`の方が使いやすいが、実運用(変化点数が未知)では`Pelt`+`pen`探索、または情報量規準に基づく`pen`設定(例: `pen = log(n) * dim * sigma^2`)が使われることが多い。

### `min_size`

**用途**: 変化点間の最小セグメント長を指定する。短すぎるセグメントの検出を防ぐ。

**シグネチャ**: 各アルゴリズム(`Pelt`等)のコンストラクタ引数。デフォルトは`2`。

**使用例**:
```python
for ms in [2, 30]:
    algo = rpt.Pelt(model="l2", min_size=ms).fit(signal)
    print(ms, algo.predict(pen=10))
```
実行結果:
```
2  [50, 55, 100, 105, 145, 150, 200]
30 [50, 105, 150, 200]
```

**注意点・落とし穴**:
- `min_size`を大きくすると近接した誤検出(例: 50と55の両方)が自然に排除される。ノイズによる過検出を抑えたい場合に有効なパラメータの一つ(`pen`を大きくするのとは異なるアプローチ)。
- コスト関数自身も`min_size`属性を持つ(例: `CostAR(order=4)`は次数分のサンプルが必要なため`min_size`が自動的に大きくなる)。`Pelt(min_size=2)`と指定しても、内部的にはコスト関数側の`min_size`との大きい方が使われる。

### `jump`

**用途**: 変化点候補を何サンプルおきに評価するかを指定する(間引き)。大きいほど高速だが精度が粗くなる。

**シグネチャ**: 各アルゴリズム(`Pelt`等)のコンストラクタ引数。デフォルトは`5`(`KernelCPD`のみ`1`固定)。

**使用例**:
```python
for jump in [1, 5, 20]:
    algo = rpt.Pelt(model="l2", jump=jump).fit(signal)
    print(jump, algo.predict(pen=30))
```
実行結果:
```
1  [51, 103, 149, 200]
5  [50, 100, 105, 150, 200]
20 [40, 60, 100, 140, 160, 200]
```

**注意点・落とし穴**:
- `jump=1`にすると候補点が間引かれず最も精度が高くなる(この例では真の変化点`[51, 103, 149]`とほぼ一致)。`jump=20`のように粗くすると検出位置のズレが大きくなる(`40`や`160`のように真の値から大きくずれる)。
- `KernelCPD`のみ、コンストラクタに`jump`引数はあるものの**常に1に固定され、指定しても無視される**(公式docstringに明記、ソースコードでも`self.jump = 1  # set to 1`と確認済み)。

---

## 7. 応用・発展

### 7.1 カスタムコスト関数の自作・拡張

#### `BaseCost` を継承したカスタムコスト関数

**用途**: 既存のコスト関数(L1/L2/RBF等)にない独自の当てはまり基準を実装し、`Pelt`等の検出アルゴリズムにそのまま組み込む。`ruptures.base.BaseCost`は`fit`/`error`/`model`(プロパティ)を実装すれば良い抽象基底クラス。

**シグネチャ**: `ruptures.base.BaseCost`(抽象基底クラス)。必須実装: `fit(self, *args, **kwargs)` / `error(self, start, end)` / `model`(クラス変数としての文字列プロパティ)。組み込み実装(`sum_of_costs(self, bkps)`)は変化点リストからセグメント別コストの合計を計算する。

**使用例**:
```python
import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost
from ruptures.exceptions import NotEnoughPoints

class CostMAD(BaseCost):
    """Median Absolute Deviation ベースのロバストなコスト(外れ値に強い)。"""
    model = "custom_mad"

    def __init__(self):
        self.signal = None
        self.min_size = 2

    def fit(self, signal):
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        self.signal = signal
        return self

    def error(self, start, end):
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub = self.signal[start:end]
        med = np.median(sub, axis=0)
        return np.abs(sub - med).sum()

signal, true_bkps = rpt.pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=1, seed=42)

algo = rpt.Pelt(custom_cost=CostMAD(), min_size=2, jump=5).fit(signal)
print(algo.predict(pen=30))
print(algo.cost.model)
```
実行結果:
```
[50, 105, 150, 200]
custom_mad
```

**注意点・落とし穴**:
- `error()`が`min_size`未満のセグメントに対して`NotEnoughPoints`(`ruptures.exceptions`)を送出する規約は、既存の`CostL2`等の実装と同じ。これを守らないと`Pelt`等の内部探索でセグメント長の制約が効かなくなる。
- `custom_cost=CostMAD()`のように**インスタンス**を渡す(`model="custom_mad"`という文字列ではなく)のが基本の使い方。

#### 文字列`model=`によるカスタムコストの自動登録

**用途**: 一度`BaseCost`のサブクラスを定義してインポートしておけば、`custom_cost=`でインスタンスを渡さなくても`model="<自作のmodel文字列>"`という**文字列だけ**で呼び出せる。`ruptures.costs.cost_factory`の実装に由来する挙動。

**シグネチャ**: `ruptures.costs.cost_factory(model, *args, **kwargs)` — 内部で`BaseCost.__subclasses__()`を走査し、`cls.model == model`に一致するクラスをインスタンス化する。

**使用例**:
```python
# 前の例で定義した CostMAD クラスが同一プロセス内に存在する状態で、
# custom_cost= を渡さず、model="custom_mad" という文字列だけで動くか
algo = rpt.Pelt(model="custom_mad", min_size=2, jump=5).fit(signal)
print(algo.predict(pen=30))
```
実行結果:
```
[50, 105, 150, 200]
```

**注意点・落とし穴**:
- `cost_factory`は`BaseCost.__subclasses__()`(Pythonの標準機能、**そのクラスを継承した全サブクラスをインポート時点で自動収集**)を使っている。そのため、**クラス定義(=モジュールのインポート)さえ実行されていれば**、`custom_cost=`を渡さなくても文字列指定だけで動いてしまう(実行確認済み)。裏を返すと、`model`文字列が既存のコスト名(`"l2"`等)や他で定義済みの自作コストと衝突すると、意図しないクラスが選ばれる危険がある。命名は具体的にすること。

### 7.2 多変量信号・欠損値のある信号への対応

#### 3次元以上の多変量信号

**用途**: `pw_constant`の`n_features`を3以上にした多変量信号でも、`CostL2`等は追加の設定なしにそのまま動く(`error()`内部で`.var(axis=0).sum()`のように各次元のコストを合算しているため)。特徴量を絞るかどうかで検出結果が変わりうる点を確認する。

**使用例**:
```python
signal, true_bkps = rpt.pw_constant(n_samples=200, n_features=5, n_bkps=3, noise_std=1, seed=42)
print(signal.shape, true_bkps)

algo = rpt.Pelt(model="l2", min_size=2, jump=5).fit(signal)
print("5特徴量:", algo.predict(pen=50))

algo1 = rpt.Pelt(model="l2", min_size=2, jump=5).fit(signal[:, :1])
print("1特徴量のみ:", algo1.predict(pen=50))
```
実行結果:
```
(200, 5) [51, 103, 149, 200]
5特徴量: [50, 100, 105, 145, 150, 200]
1特徴量のみ: [50, 105, 150, 200]
```

**注意点・落とし穴**:
- 同じ`pen=50`でも、特徴量数を増やすと検出変化点が増える(この例では5特徴量で6個、1特徴量で4個)。`CostL2`のコストは次元ごとの分散の合計であり次元数に応じて総コストのスケールも変わるため、多変量にする際は`pen`を次元数に応じて調整し直す必要がある(6.節で触れた`pen ≈ log(n) * dim * sigma^2`の目安がここでも有効)。

#### 欠損値(NaN)を含む信号

**用途**: ruptures本体には欠損値の自動処理機構がないため、`NaN`を含む信号を渡すとどうなるかを実際に確認し、実務での前処理(補間)パターンを示す。

**使用例**:
```python
import numpy as np
import pandas as pd

signal, bkps = rpt.pw_constant(n_samples=200, n_features=3, n_bkps=3, noise_std=1, seed=42)

sig_nan = signal.copy()
rng = np.random.default_rng(0)
missing_idx = rng.choice(200, size=10, replace=False)
sig_nan[missing_idx, 0] = np.nan
print("NaN count:", np.isnan(sig_nan).sum())

algo0 = rpt.Pelt(model="l2").fit(signal)
print("NaNなし:", algo0.predict(pen=50))

algo1 = rpt.Pelt(model="l2").fit(sig_nan)
print("NaNあり(未対応):", algo1.predict(pen=50))

df_interp = pd.DataFrame(sig_nan).interpolate(limit_direction="both").to_numpy()
algo2 = rpt.Pelt(model="l2").fit(df_interp)
print("線形補間後:", algo2.predict(pen=50))
```
実行結果:
```
NaN count: 10
NaNなし: [50, 100, 105, 150, 200]
NaNあり(未対応): [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
線形補間後: [50, 55, 100, 105, 150, 200]
```

**注意点・落とし穴**:
- **重大な落とし穴(実行確認済み)**: `Pelt`は`NaN`を含む信号を渡しても例外を出さない。`CostL2.error()`内部の`.var()`計算が`NaN`を伝播させ、実質すべての区間のコストが`NaN`(比較不能)になり、結果として`jump`刻みのほぼ全候補点が変化点として検出される(この例では39個)という**静かな暴走**が起きる。エラーにならないため気付きにくい。
- 対処としては、`fit()`に渡す前に`pandas.DataFrame.interpolate()`等で欠損値を補間しておく必要がある。補間後は元のパターン(`[50, 100, 105, 150, 200]`)にかなり近い結果(`[50, 55, 100, 105, 150, 200]`)に戻ることを確認した。

### 7.3 実データでの閾値選定の実践パターン

#### `pen`候補を複数試して比較する

**用途**: 実データでは真の変化点数が未知なため、複数の`pen`候補で`Pelt`を実行し、検出変化点数とセグメンテーション全体のコスト(`algo.cost.sum_of_costs(bkps)`)の推移を見比べる。

**シグネチャ**: `ruptures.base.BaseCost.sum_of_costs(self, bkps)` — 変化点リスト`bkps`(`bkps[-1]==n_samples`)から、各セグメントの`error()`の合計を返す。

**使用例**:
```python
signal, true_bkps = rpt.pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=1, seed=42)
n = signal.shape[0]
algo = rpt.Pelt(model="l2", min_size=2, jump=5).fit(signal)

print(f"{'pen':>4} {'n_bkps':>7} {'sum_of_costs':>13}")
for pen in [1, 3, 5, 10, 20, 30, 50, 80, 120]:
    bkps = algo.predict(pen=pen)
    cost = algo.cost.sum_of_costs(bkps)
    print(f"{pen:>4} {len(bkps)-1:>7} {cost:>13.3f}")
```
実行結果:
```
 pen  n_bkps  sum_of_costs
   1      18       288.232
   3       8       305.992
   5       8       305.992
  10       6       316.363
  20       5       331.451
  30       4       353.435
  50       3       402.463
  80       3       402.463
 120       3       402.463
```

**注意点・落とし穴**:
- `pen`を増やすほど`n_bkps`は単調非増加、`sum_of_costs`は単調非減少になる(「変化点を増やすほど当てはまりは良くなるが、増やすコストが上がる」というトレードオフそのもの)。`pen=50`以上では結果が飽和する(同じセグメンテーションに収束する)ことも確認できる。

#### BIC的な基準による`pen`選定

**用途**: 上の`pen`候補一覧に対して、情報量規準に似た「当てはまりの良さ(コスト)+複雑さへの罰則」の合計が最小になる`pen`を選ぶ、という実務でよく使われる簡易パターン。

**使用例**:
```python
sigma2 = 1.0  # 既知または事前に推定したノイズ分散
print(f"{'pen':>4} {'n_bkps':>7} {'cost':>10} {'BIC近似':>10}")
best_pen, best_bic = None, np.inf
for pen in [1, 3, 5, 10, 20, 30, 50, 80, 120]:
    bkps = algo.predict(pen=pen)
    k = len(bkps) - 1
    cost = algo.cost.sum_of_costs(bkps)
    bic = cost + k * np.log(n) * sigma2
    print(f"{pen:>4} {k:>7} {cost:>10.3f} {bic:>10.3f}")
    if bic < best_bic:
        best_bic, best_pen = bic, pen
print("BIC最小のpen:", best_pen)
```
実行結果:
```
 pen  n_bkps       cost      BIC近似
   1      18    288.232    383.602
   3       8    305.992    348.379
   5       8    305.992    348.379
  10       6    316.363    348.153
  20       5    331.451    357.942
  30       4    353.435    374.629
  50       3    402.463    418.358
  80       3    402.463    418.358
 120       3    402.463    418.358
BIC最小のpen: 10
```

**注意点・落とし穴**:
- この基準(`cost + n_bkps * log(n) * sigma^2`)はruptures組み込みのものではなく、あくまで「情報量規準の考え方を`sum_of_costs`に当てはめた」自作の目安。BIC最小の`pen=10`(6変化点)は、真の変化点数3個より多めに出ている(実行確認済み)。`sigma2`の推定値やペナルティの係数次第で最適`pen`は変わるため、この方法だけで機械的に決め打ちせず、`display()`等で目視確認するのが実務では無難。
- `sum_of_costs`はコスト関数の種類(L1/L2/RBF等)によってスケールがまったく異なるため、この基準を使い回す場合はコスト関数ごとに`sigma2`やペナルティ係数を調整し直す必要がある。

### 7.4 オンライン・ストリーミング風の逐次検出

**補足**: 検証した ruptures 1.1.10 には、オンライン変化点検出(例: BOCPD)専用のクラス(`rpt.Online`等)は存在しない(`dir(ruptures)`で確認済み、`Binseg`/`BottomUp`/`Dynp`/`KernelCPD`/`Pelt`/`Window`の6アルゴリズムのみ)。そのため、新しいデータが到着するたびに既存アルゴリズムを再実行する疑似オンラインパターンで代用する。

#### 逐次再実行によるオンライン検出パターン

**用途**: データが一定間隔で到着する状況を想定し、到着のたびに直近までの信号全体で`Pelt`を再実行し、前回までに検出済みでない新規の変化点だけを「アラーム」として報告する。

**使用例**:
```python
signal, true_bkps = rpt.pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=1, seed=42)

step = 10
pen = 20
detected_so_far = set()
alarms = []

for t in range(30, len(signal) + 1, step):
    sub_signal = signal[:t]
    algo = rpt.Pelt(model="l2", min_size=2, jump=5).fit(sub_signal)
    bkps = algo.predict(pen=pen)
    new_bkps = set(bkps[:-1]) - detected_so_far
    if new_bkps:
        for b in sorted(new_bkps):
            alarms.append((t, b))
        detected_so_far |= new_bkps

print(alarms)
print("真の変化点:", true_bkps[:-1])
```
実行結果:
```
[(60, 50), (110, 100), (110, 105), (150, 145), (160, 150)]
真の変化点: [51, 103, 149]
```

**注意点・落とし穴**:
- 「アラーム時刻(t)」と「実際の変化点位置」は一致しない。例えば真の変化点51は`t=60`まで到着データが増えて初めて検出されており、**検出には`step`刻み分(最大10サンプル)のラグが必ず生じる**(実行確認済み)。オンライン検知の即時性を求める場合は`step`を小さくする必要があるが、その分下記の計算コストが増える。
- `pen=20`は本文6節のバッチ処理と同じ値を使ったが、疑似オンライン(部分信号に対する`Pelt`)では信号長`t`ごとにコストのスケールが変わるため、バッチと同じ`pen`が常に最適とは限らない。

#### 逐次検出の限界(計算コスト・検出ラグ)

**用途**: 「毎回全データで再`fit`する」方式が、データが増えるにつれてどれだけ非効率になるかを実測する。

**使用例**:
```python
import time

signal, true_bkps = rpt.pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=1, seed=42)

# 疑似オンライン: 毎回全データで再fit
t0 = time.perf_counter()
for t in range(30, len(signal) + 1, 10):
    algo = rpt.Pelt(model="l2", min_size=2, jump=5).fit(signal[:t])
    algo.predict(pen=20)
t1 = time.perf_counter()
print(f"逐次再fit(18回): {t1 - t0:.4f} 秒")

# バッチ: 全データに対して1回だけfit
t2 = time.perf_counter()
algo_batch = rpt.Pelt(model="l2", min_size=2, jump=5).fit(signal)
algo_batch.predict(pen=20)
t3 = time.perf_counter()
print(f"バッチ1回fit: {t3 - t2:.4f} 秒")
```
実行結果:
```
逐次再fit(18回): 0.0269 秒
バッチ1回fit: 0.0024 秒
```

**注意点・落とし穴**:
- 信号長200という小さなデータでも、疑似オンライン方式(18回の再`fit`)はバッチ1回`fit`よりおよそ11倍遅い(実行確認済み、実行環境依存で倍率は変動しうる)。`Pelt`自体は平均的に線形時間だが、「到着のたびに先頭から全部再計算する」実装のため全体では信号長に対して二次的にコストが増える。信号が長くなる・到着頻度が高くなるオンライン用途では、直近`W`サンプルだけを対象にする固定長スライディングウィンドウ(`signal[t-W:t]`)に変えるか、専用のオンラインアルゴリズム(ruptures外のライブラリ、例: `bayesian_changepoint_detection`等)を検討する必要がある。
