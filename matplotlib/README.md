# matplotlib 逆引き辞書

matplotlib 3.11.1 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で `matplotlib.use("Agg")` によりGUI無しで実際に実行して確認。図の実行結果は画像として生成・目視確認した内容を記述している)。

## 目次

1. [基本プロット](#基本プロット)
2. [Figure・Axes管理・レイアウト](#figureaxes管理レイアウト)
3. [軸の設定(目盛り・範囲・スケール)](#軸の設定目盛り範囲スケール)
4. [凡例・タイトル・ラベル](#凡例タイトルラベル)
5. [色・スタイル・カラーマップ](#色スタイルカラーマップ)
6. [注釈・テキスト](#注釈テキスト)
7. [複数プロット種](#複数プロット種)
8. [3Dプロット](#3dプロット)
9. [保存・表示](#保存表示)
10. [スタイル・テーマ(rcParams)](#スタイルテーマrcparams)
11. [その他](#その他)

---

## 基本プロット

### `ax.plot(...)` / `plt.plot(...)`

**用途**: 折れ線グラフ(連続する点を線でつなぐ)を描く、matplotlib で最も基本的な関数。

**シグネチャ**: `Axes.plot(self, *args, scalex=True, scaley=True, data=None, **kwargs)`

**使用例**:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), label="cos", linestyle="--")
ax.legend()
fig.savefig("01_plot.png")
```
実行結果: 同じ Axes 上に sin(青の実線)と cos(オレンジの破線)の2本の曲線が描かれ、右上に凡例("sin"/"cos")が表示される PNG が生成される(例外なく完了)。

**注意点・落とし穴**:
- `plt.plot(...)` は内部で `plt.gca().plot(...)` を呼んでいるだけで、`ax.plot(...)` と等価。複数 Axes を扱うコードでは `ax.plot` を明示する方が事故が少ない。
- `*args` は `plot(x, y)`, `plot(y)`(x は自動で 0,1,2,... になる), `plot(x, y, fmt)`(`'ro-'` のような書式文字列)など複数の呼び出し形式を許容する可変長引数。

---

### `ax.scatter(...)`

**用途**: 散布図を描く。点ごとにサイズ・色を変えられるのが `plot` との違い。

**シグネチャ**: `Axes.scatter(self, x, y, s=None, c=None, *, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None, colorizer=None, plotnonfinite=False, data=None, **kwargs)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(0)
fig, ax = plt.subplots()
x = rng.random(50)
y = rng.random(50)
sizes = rng.random(50) * 300
ax.scatter(x, y, s=sizes, c=x, cmap="viridis", alpha=0.7)
fig.savefig("02_scatter.png")
```
実行結果: 50個の円が散布され、`c=x` によって色が viridis カラーマップでx座標に応じて紫→黄に変化し、`s=sizes` によって円の大きさもランダムに変わる散布図が生成される。

**注意点・落とし穴**:
- `c` に数値配列を渡すと `cmap` でマッピングされるが、`c` に単色("red" など)や RGBA を渡した場合は `cmap` は無視される。`c=[[1,0,0]]` のような単一の色配列と数値配列の区別が曖昧になりがちなので注意。

---

### `ax.bar(...)` / `ax.barh(...)`

**用途**: 縦棒グラフ(`bar`)・横棒グラフ(`barh`)を描く。

**シグネチャ**:
- `Axes.bar(self, x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)`
- `Axes.barh(self, y, width, height=0.8, left=None, *, align='center', data=None, **kwargs)`

**使用例**:
```python
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
cats = ["A", "B", "C"]
vals = [3, 7, 5]
axes[0].bar(cats, vals, color="tab:blue")
axes[1].barh(cats, vals, color="tab:orange")
fig.savefig("03_bar.png")
```
実行結果: 左側に A/B/C の縦棒グラフ、右側に同じデータの横棒グラフが並んで表示される(縦棒は下から、横棒は左から伸びる)。

**注意点・落とし穴**:
- `bar` の第1引数は「棒の中心 x 座標」(`align='center'` が既定)。棒の左端を基準にしたい場合は `align='edge'` を指定する。
- `barh` は `bar` と引数の意味が微妙に異なり(`height`/`width` の役割が入れ替わる)、コピペで書き換える際に混同しやすい。

---

### `ax.fill_between(...)`

**用途**: 2本の曲線(または曲線と定数)の間を塗りつぶす。信頼区間の可視化などによく使う。

**シグネチャ**: `Axes.fill_between(self, x, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)`

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(x) + 0.5
ax.fill_between(x, y1, y2, alpha=0.3, color="green")
fig.savefig("04_fill_between.png")
```
実行結果: sin(x) と sin(x)+0.5 の間の帯状の領域が半透明の緑で塗りつぶされた図が生成される。

**注意点・落とし穴**:
- `y2` の既定値は `0`(x軸との間を塗る)。2曲線の間を塗りたい場合は明示的に `y2` を渡す必要がある。
- `where` で条件を満たす区間だけ塗りつぶせるが、2曲線が交差する境界を滑らかにしたい場合は `interpolate=True` を追加しないと交点付近がギザギザになる。

---

### `ax.step(...)`

**用途**: 階段状(ステップ)のグラフを描く。離散的な値の変化を表すのに適する。

**シグネチャ**: `Axes.step(self, x, y, *args, where='pre', data=None, **kwargs)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(0)
fig, ax = plt.subplots()
x = np.arange(10)
y = rng.integers(0, 10, 10)
ax.step(x, y, where="mid")
fig.savefig("05_step.png")
```
実行結果: 10点の値が階段状の線で結ばれた図が生成される(`where="mid"` のため各点の中間で値が切り替わる)。

**注意点・落とし穴**:
- 既定は `where='pre'`(各点の手前で値が変わる)。直感的に「その点を中心に段差がある」ようにしたい場合は `where='mid'`、逆に「その点の後ろで変わる」場合は `where='post'` を指定する。既定値を知らずに使うと段差の位置がずれて見える。

---

### `ax.stem(...)`

**用途**: 各点から基準線(既定は0)まで縦線を引き、先端にマーカーを打つ「幹(ステム)」グラフ。離散信号の可視化などに使う。

**シグネチャ**: `Axes.stem(self, *args, linefmt=None, markerfmt=None, basefmt=None, bottom=0, label=None, orientation='vertical', data=None)`

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)
ax.stem(x, y)
fig.savefig("06_stem.png")
```
実行結果: sin波の20点それぞれについて、水平の基準線(0)から各点まで縦線が伸び、先端に丸いマーカーが付いた図が生成される。

---

### `ax.errorbar(...)`

**用途**: 誤差範囲(標準偏差・信頼区間など)付きの折れ線・散布図を描く。

**シグネチャ**: `Axes.errorbar(self, x, y, yerr=None, xerr=None, fmt='', *, ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, elinestyle=None, data=None, **kwargs)`

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
x = np.arange(5)
y = np.array([1, 3, 2, 5, 4])
yerr = np.array([0.5, 0.2, 0.4, 0.3, 0.6])
ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=4)
fig.savefig("07_errorbar.png")
```
実行結果: 5点の折れ線グラフに、各点で上下に伸びるエラーバー(先端にキャップ付き)が重ねて表示される。

**注意点・落とし穴**:
- `fmt` の既定は空文字列で、マーカーも線も明示しないと点だけ・線だけの見た目になりがち。`fmt='o-'` のように書式文字列で線種とマーカーを同時指定するのが一般的。
- `capsize` の既定は `None`(実質0、キャップなし)。誤差の端が見やすいキャップを付けたい場合は明示的に指定する必要がある。

---

## Figure・Axes管理・レイアウト

### `plt.figure(...)`

**用途**: 新しい Figure(描画キャンバス全体)を作成する。

**シグネチャ**: `plt.figure(num=None, figsize=None, dpi=None, *, facecolor=None, edgecolor=None, frameon=True, clear=False, **kwargs) -> Figure`

**使用例**:
```python
fig = plt.figure(figsize=(5, 4), dpi=100)
print(fig.get_size_inches(), fig.dpi)
```
実行結果:
```
[5. 4.] 100
```

**注意点・落とし穴**:
- `figsize`/`dpi` を省略すると `rcParams['figure.figsize']`(既定 `[6.4, 4.8]`)・`rcParams['figure.dpi']`(既定 `100.0`)が使われる。
- `num` に既存の番号/文字列を渡すと、新規作成ではなく同じ Figure を再利用(既存内容の上にプロット追加、または `clear=True` でクリアしてから利用)する点に注意。

---

### `plt.subplots(...)`

**用途**: Figure と複数の Axes をまとめて生成する、最も一般的なレイアウト作成関数。

**シグネチャ**: `plt.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw) -> tuple[Figure, Any]`

**使用例**:
```python
import numpy as np
fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    ax.plot(np.arange(10), np.arange(10) * (i + 1))
    ax.set_title(f"ax{i}")
fig.savefig("08_subplots.png")
```
実行結果: 2x2に並んだ4つの Axes それぞれに傾きの異なる直線が描かれ、各タイトル(ax0〜ax3)が表示される。`sharex=True`/`sharey=True` のため軸範囲が全パネルで揃う。

**注意点・落とし穴**:
- `squeeze=True`(既定)のとき、`nrows=ncols=1` なら `axes` は単一の `Axes` オブジェクト(配列ではない)になる。行または列が1つだけの場合は1次元配列、両方2以上なら2次元配列になり、コードの汎用性を保つには `squeeze=False` を使うか `axes.flat` でイテレートするのが安全。

---

### `fig.add_subplot(...)`

**用途**: 既存の Figure に1つずつ Axes を追加する(`plt.subplots` と違い、レイアウトの異なる Axes を個別に配置できる)。

**シグネチャ**: `Figure.add_subplot(self, *args, **kwargs)`

**使用例**:
```python
fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot([1, 2, 3])
ax2.scatter([1, 2, 3], [3, 2, 1])
fig.savefig("09_add_subplot.png")
```
実行結果: 横に並んだ2つの Axes に、左は折れ線、右は散布図がそれぞれ描かれる。

**注意点・落とし穴**:
- `add_subplot(2, 2, 1)` のように3引数、または `add_subplot(221)`(3桁の整数、10x10未満限定)のどちらの書き方も可能。
- `projection='3d'` や `projection='polar'` を渡すと通常の `Axes` ではなく `Axes3D`/`PolarAxes` が返る(3Dプロットの節を参照)。

---

### `plt.subplot_mosaic(...)`

**用途**: 文字列やネストしたリストで直感的にレイアウトを指定し、複数 Axes を一括生成する(不揃いなグリッドを組みやすい)。

**シグネチャ**: `plt.subplot_mosaic(mosaic, *, sharex=False, sharey=False, width_ratios=None, height_ratios=None, empty_sentinel='.', subplot_kw=None, gridspec_kw=None, per_subplot_kw=None, **fig_kw) -> tuple[Figure, dict[...]]`

**使用例**:
```python
fig, axd = plt.subplot_mosaic(
    [["left", "right_top"], ["left", "right_bottom"]], figsize=(6, 4)
)
axd["left"].set_title("left")
axd["right_top"].set_title("right_top")
axd["right_bottom"].set_title("right_bottom")
fig.tight_layout()
fig.savefig("10_subplot_mosaic.png")
```
実行結果: 左側に縦に長い1つの Axes("left")、右側に上下2分割された2つの Axes("right_top"/"right_bottom")が配置された図が生成される。戻り値 `axd` は `{"left": Axes, "right_top": Axes, "right_bottom": Axes}` の辞書。

**注意点・落とし穴**:
- 同じラベルを複数マスに書くとその領域を1つの Axes が占有する(上記の `"left"` が2行分)。`empty_sentinel='.'`(既定)を書いたマスは Axes を作らない空白になる。

---

### `fig.tight_layout(...)` / `plt.tight_layout(...)`

**用途**: タイトル・軸ラベル・目盛りラベル同士の重なりを自動調整して余白を詰める。

**シグネチャ**: `Figure.tight_layout(self, *, pad=1.08, h_pad=None, w_pad=None, rect=None)`

**使用例**:
```python
fig, axes = plt.subplots(1, 2, figsize=(4, 3))
axes[0].set_title("long title that might overlap")
axes[1].set_ylabel("very long ylabel text")
fig.tight_layout()
fig.savefig("11_tight_layout.png")
```
実行結果: `tight_layout()` を呼んでも、Figure サイズ (4, 3) に対してタイトル・ylabel の文字列が長すぎるため、依然としてタイトルの一部が Figure 端で見切れる図になった(完全には重なりを解消しきれない実例)。

**注意点・落とし穴**:
- `tight_layout` は「重なりを完全になくす」ことを保証しない。極端に小さい `figsize` や長いラベルでは、上記の検証例のように文字が見切れることがある。`figsize` を大きくするか、`constrained_layout=True`(`plt.subplots(..., constrained_layout=True)`)を使う方が安定するケースが多い。
- `Axes.set_aspect` で縦横比を固定している場合や `colorbar` を使っている場合、`tight_layout` とレイアウトが競合し警告が出ることがある。

---

### `fig.subplots_adjust(...)`

**用途**: サブプロット間の余白(左右上下・行間・列間)を手動で調整する。

**シグネチャ**: `Figure.subplots_adjust(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)`

**使用例**:
```python
fig, axes = plt.subplots(1, 2)
fig.subplots_adjust(wspace=0.5, left=0.2)
fig.savefig("12_subplots_adjust.png")
```
実行結果: 2つの Axes の間隔(`wspace`)が広がり、左側の余白(`left`)も広がった図が生成される。

**注意点・落とし穴**:
- `tight_layout()` や `constrained_layout=True` と同時に使うと、後から呼んだ方・有効になっている方が優先されて意図した余白にならないことがある。どちらか一方の方式に統一するのが無難。

---

### `fig.add_gridspec(...)`

**用途**: 行・列比率を指定できる `GridSpec` を作り、`gs[row, col]` のスライスで複数セルにまたがる Axes を柔軟に配置する。

**シグネチャ**: `Figure.add_gridspec(self, nrows=1, ncols=1, **kwargs)`

**使用例**:
```python
fig = plt.figure(figsize=(6, 4))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
ax1.set_title("1"); ax2.set_title("2"); ax3.set_title("3 (span)")
fig.tight_layout()
fig.savefig("13_gridspec.png")
```
実行結果: 上段が幅2:1の2分割、下段が `gs[1, :]` により横いっぱいに広がった1つの Axes になった、不均等なグリッドレイアウトの図が生成される。

---

## 軸の設定(目盛り・範囲・スケール)

### `ax.set_xlim(...)` / `ax.set_ylim(...)`

**用途**: 表示する x軸・y軸の範囲を指定する。

**シグネチャ**: `Axes.set_xlim(self, left=None, right=None, *, emit=True, auto=False, xmin=None, xmax=None)`(`set_ylim` も同形)

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
ax.plot(np.arange(10), np.arange(10)**2)
ax.set_xlim(2, 8)
ax.set_ylim(0, 50)
print(ax.get_xlim(), ax.get_ylim())
fig.savefig("14_xlim_ylim.png")
```
実行結果:
```
(np.float64(2.0), np.float64(8.0)) (np.float64(0.0), np.float64(50.0))
```
x軸が2〜8、y軸が0〜50の範囲だけを表示するようにトリミングされた図になる。

**注意点・落とし穴**:
- `auto=False`(既定)が渡ると以後の自動スケーリングが無効化される。動的にデータを追加してもその都度 `set_xlim` しない限り範囲が固定されたままになる。

---

### `ax.set_xticks(...)` / `ax.set_xticklabels(...)`

**用途**: 目盛りの位置(`set_xticks`)とその表示ラベル(`set_xticklabels`)を明示的に指定する。

**シグネチャ**: `Axes.set_xticks(self, ticks, labels=None, *, minor=False, **kwargs)` / `Axes.set_xticklabels(self, labels, *, minor=False, fontdict=None, **kwargs)`

**使用例**:
```python
fig, ax = plt.subplots()
ax.plot(range(5), [1, 3, 2, 5, 4])
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(["a", "b", "c", "d", "e"], rotation=45)
fig.savefig("15_xticks.png")
```
実行結果: x軸の目盛りが0〜4の5点に固定され、そのラベルが "a"〜"e" に置き換わり、45度回転して表示される図が生成される。

**注意点・落とし穴**:
- `set_xticks` は第2引数 `labels` でラベルも同時に指定できる(`set_xticks([0,1,2], labels=["a","b","c"])`)。別々に `set_xticklabels` を呼ぶより、目盛り位置とラベルの対応がずれるリスクが少ないため公式にも推奨されている。
- `set_xticklabels` だけを単独で呼ぶと、既存の目盛り位置(`FixedLocator` でない場合)とラベル数が一致せずズレることがある。

---

### `ax.set_xscale(...)` / `ax.set_yscale(...)`

**用途**: 軸のスケールを線形以外(対数など)に変更する。

**シグネチャ**: `Axes.set_xscale(self, value, **kwargs)`(`set_yscale` も同形)

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
x = np.linspace(1, 1000, 100)
ax.plot(x, x**2)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("16_log_scale.png")
```
実行結果: x軸・y軸ともに対数目盛り(1, 10, 100, 1000, ... / 1, 10², 10⁴, 10⁶)になり、`y = x**2` の曲線が両対数プロット上では直線として表示される。

**注意点・落とし穴**:
- `value` には `'linear'`, `'log'`, `'symlog'`(正負両方を扱える対数), `'logit'` などを指定できる。`'log'` スケールで0以下の値を含むデータを渡すと警告が出て、その点は描画されない。

---

### `ax.twinx(...)` / `ax.twiny(...)`

**用途**: x軸(`twinx`)またはy軸(`twiny`)を共有しつつ、もう一方の軸だけ独立にスケールが異なる2つ目の Axes を重ねる。

**シグネチャ**: `Axes.twinx(self, axes_class=None, *, delta_zorder=0.0, **kwargs)`

**使用例**:
```python
import numpy as np
fig, ax1 = plt.subplots()
x = np.arange(10)
ax1.plot(x, x, color="tab:blue")
ax1.set_ylabel("linear", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(x, np.exp(x / 3), color="tab:red")
ax2.set_ylabel("exp", color="tab:red")
fig.savefig("17_twinx.png")
```
実行結果: 左のy軸(青)に線形の直線、右のy軸(赤)に指数関数の曲線が、同じx軸を共有しつつ別スケールで重ねて描かれた図が生成される。

**注意点・落とし穴**:
- `ax1.legend()` と `ax2.legend()` を別々に呼ぶと凡例が2つ重なって表示される。1つにまとめたい場合は `handles`/`labels` を両方の Axes から集めて `ax1.legend(handles1 + handles2, labels1 + labels2)` のように結合する必要がある。

---

### `ax.invert_yaxis(...)` / `ax.invert_xaxis(...)`

**用途**: 軸の向きを反転させる(値が大きいほど下/左になるようにする)。ランキング表示などでよく使う。

**シグネチャ**: `Axes.invert_yaxis(self)`(引数なし。`invert_xaxis` も同様)

**使用例**:
```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [3, 2, 1])
ax.invert_yaxis()
fig.savefig("18_invert_yaxis.png")
```
実行結果: y軸の目盛りが上から下に向かって増加する(3, 2, 1 ではなく 1 が下、3 が上に来る通常表示から反転し、3 が上端付近に来る)向きになった折れ線グラフが生成される。

---

### `ax.set_aspect(...)`

**用途**: x軸とy軸のデータ単位あたりの表示比率を固定する(円を真円に見せる、など)。

**シグネチャ**: `Axes.set_aspect(self, aspect, adjustable=None, anchor=None, share=False)`

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
theta = np.linspace(0, 2 * np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta))
ax.set_aspect("equal")
fig.savefig("19_set_aspect.png")
```
実行結果: `set_aspect("equal")` により、単位円がゆがまず真円として表示される図が生成される(指定しないと Figure の縦横比に応じて楕円に見えることがある)。

**注意点・落とし穴**:
- `aspect` には `'equal'`(1:1)、`'auto'`(既定、自動調整)のほか、数値(y方向1単位に対するx方向単位数)も指定できる。

---

### `ax.grid(...)`

**用途**: 目盛り線(グリッド線)を表示する。

**シグネチャ**: `Axes.grid(self, visible=None, which='major', axis='both', **kwargs)`

**使用例**:
```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.grid(True, linestyle="--", alpha=0.6)
fig.savefig("20_grid.png")
```
実行結果: プロット領域に薄い破線のグリッド線(縦横両方)が重ねて表示される図が生成される。

**注意点・落とし穴**:
- `which='major'`(既定)は主目盛りのみに線を引く。副目盛り(`minor`)にも線を引きたい場合は `which='minor'` または `which='both'` を指定し、かつ `ax.minorticks_on()` で副目盛り自体を有効化しておく必要がある。

---

## 凡例・タイトル・ラベル

### `ax.legend(...)`

**用途**: 各系列(`label=` を指定した `plot`/`scatter` など)の凡例を表示する。

**シグネチャ**: `Axes.legend(self, *args, **kwargs)`(多数の呼び出しパターンに対応するため実質可変長引数)

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
x = np.linspace(0, 10, 50)
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), label="cos")
ax.legend(loc="upper right", ncol=2, frameon=True)
fig.savefig("21_legend.png")
```
実行結果: プロット右上に "sin"・"cos" の2つの凡例項目が横に並んで(`ncol=2`)、枠線付きで表示される図が生成される。

**注意点・落とし穴**:
- 各アーティストに `label` を渡していないと `legend()` は空か何も表示されない(`label` が `"_"` で始まる場合も自動的に凡例から除外される)。
- `loc="best"`(既定)は毎回データと重ならない位置を探索するため、データ量が多いと描画が遅くなることがある。位置が分かっている場合は `loc` を明示した方が高速。

---

### `ax.set_title(...)` / `fig.suptitle(...)`

**用途**: 個別 Axes のタイトル(`set_title`)、Figure 全体の大見出し(`suptitle`)を設定する。

**シグネチャ**: `Axes.set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs)` / `Figure.suptitle(self, t, **kwargs)`

**使用例**:
```python
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].plot([1, 2, 3]); axes[0].set_title("subplot A")
axes[1].plot([3, 2, 1]); axes[1].set_title("subplot B")
fig.suptitle("Overall Title", fontsize=14)
fig.savefig("22_title_suptitle.png")
```
実行結果: 左右2つの Axes それぞれに "subplot A"/"subplot B" の個別タイトルが付き、さらに Figure 上部中央に大きめの "Overall Title" が表示される図が生成される。

**注意点・落とし穴**:
- `set_title` は既定で Axes 上部中央(`loc='center'`)に付くが、`loc='left'`/`'right'` で左右寄せにできる。`suptitle` はあくまで Figure レベルなので、`tight_layout()` と併用すると Axes のタイトルと近すぎたり重なったりすることがあり、`fig.tight_layout(rect=[0, 0, 1, 0.95])` のような調整が必要になる場合がある。

---

### `ax.set_xlabel(...)` / `ax.set_ylabel(...)`

**用途**: x軸・y軸のラベル(軸名)を設定する。

**シグネチャ**: `Axes.set_xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs)`(`set_ylabel` も同形)

**使用例**:
```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_xlabel("X axis", fontsize=12)
ax.set_ylabel("Y axis", fontsize=12)
fig.savefig("23_xlabel_ylabel.png")
```
実行結果: x軸の下に "X axis"、y軸の左に "Y axis" というラベルが12ポイントで表示される図が生成される。

**注意点・落とし穴**:
- **日本語ラベルの文字化け**: `ax.set_xlabel("X軸")` のように日本語を渡すと、matplotlib既定のフォント(DejaVu Sans)には漢字グリフが無いため `UserWarning: Glyph ... missing from font(s) DejaVu Sans` が出て、実際に生成される画像上でもその文字が四角い豆腐(tofu)表示になることを実機で確認した。日本語を表示するには IPAexGothic や Noto Sans CJK JP など日本語対応フォントをインストールし `plt.rcParams['font.family']` に設定する必要がある。

---

### `ax.set(...)`

**用途**: `set_xlabel`/`set_ylabel`/`set_title`/`set_xlim` など多数の `set_*` メソッドを、1回の呼び出しでキーワード引数としてまとめて設定できるショートカット。

**使用例**:
```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set(xlabel="x", ylabel="y", title="set() shortcut", xlim=(0, 4))
fig.savefig("24_ax_set.png")
```
実行結果: x軸ラベル "x"、y軸ラベル "y"、タイトル "set() shortcut"、x軸範囲 0〜4 がすべて1回の `ax.set(...)` 呼び出しで反映された図が生成される。

**注意点・落とし穴**:
- `ax.set(...)` のキーワード名は `set_xlabel` なら `xlabel=`、`set_xlim` なら `xlim=` のように「`set_` を除いた属性名」に対応する。存在しないプロパティ名を渡すと `AttributeError` になる。

---

## 色・スタイル・カラーマップ

### `matplotlib.colormaps[name]`

**用途**: 登録済みのカラーマップ(`"viridis"`, `"plasma"`, `"coolwarm"` など)をレジストリから取得する。

**使用例**:
```python
import matplotlib as mpl
cmap = mpl.colormaps["viridis"]
print(type(cmap))
print(cmap(0.5))
```
実行結果:
```
<class 'matplotlib.colors.ListedColormap'>
(np.float64(0.127568), np.float64(0.566949), np.float64(0.550556), np.float64(1.0))
```
`cmap(0.5)` のように0〜1の値を渡すと、その位置に対応する RGBA タプルが返る。

**注意点・落とし穴**:
- 古い `matplotlib.cm.get_cmap(name)` は非推奨で、`matplotlib.colormaps[name]`(辞書のような添字アクセス)または `matplotlib.pyplot.get_cmap(name)` を使うのが現在の推奨方法。

---

### `matplotlib.colors.Normalize(vmin, vmax, clip=False)`

**用途**: データ値を `[0, 1]` の範囲に線形正規化する。カラーマップに渡す前処理として使われる。

**シグネチャ**: `Normalize(vmin=None, vmax=None, clip=False)`

**使用例**:
```python
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=100)
print(norm(50))
```
実行結果:
```
0.5
```

**注意点・落とし穴**:
- `clip=False`(既定)では `vmin`/`vmax` の範囲外の値も外挿されて0未満・1超の値になり得る。カラーマップに渡すと範囲外の色(`cmap.get_under()`/`get_over()`)にマッピングされる場合がある。

---

### `fig.colorbar(...)`

**用途**: `imshow`/`scatter`/`pcolormesh`/`contourf` などが返す `mappable` オブジェクトに対応するカラーバー(色の凡例)を Figure に追加する。

**シグネチャ**: `Figure.colorbar(self, mappable, cax=None, ax=None, use_gridspec=True, **kwargs)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(1)
fig, ax = plt.subplots()
data = rng.random((10, 10))
im = ax.imshow(data, cmap="plasma")
fig.colorbar(im, ax=ax, label="value")
fig.savefig("25_colorbar.png")
```
実行結果: 10x10のランダム値のヒートマップの右側に、"value" というラベル付きの縦カラーバー(黒紫→黄色のplasmaカラーマップの凡例)が追加された図が生成される。

**注意点・落とし穴**:
- `mappable`(第1引数)は `imshow` などの戻り値であって、`Axes` や `Figure` そのものではない。`ax` キーワードで「どの Axes の隣にカラーバー用のスペースを確保するか」を指定する。

---

### `ax.set_prop_cycle(...)`

**用途**: そのAxesで使われる色・線種などの自動サイクル順序を上書きする。

**シグネチャ**: `Axes.set_prop_cycle(self, *args, **kwargs)`

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
ax.set_prop_cycle(color=["red", "green", "blue"])
for i in range(3):
    ax.plot(np.arange(5) + i, label=f"line{i}")
ax.legend()
fig.savefig("26_prop_cycle.png")
```
実行結果: 通常のデフォルト配色(青・オレンジ・緑…)ではなく、指定した通り赤→緑→青の順で3本の線が描かれる図が生成される。

**注意点・落とし穴**:
- 既定の色サイクルは `rcParams['axes.prop_cycle']` で定義されている。個々の Axes だけでなく全体を変えたい場合は `plt.rcParams['axes.prop_cycle']` を書き換える方法もある。

---

## 注釈・テキスト

### `ax.annotate(...)`

**用途**: 矢印付きで特定のデータ点にラベル・説明を付ける。

**シグネチャ**: `Axes.annotate(self, text, xy, xytext=None, xycoords='data', textcoords=None, arrowprops=None, annotation_clip=None, **kwargs)`

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.annotate("peak", xy=(1.57, 1.0), xytext=(3, 1.2),
            arrowprops=dict(facecolor="black", arrowstyle="->"))
fig.savefig("27_annotate.png")
```
実行結果: sin波の山の頂点(約x=1.57)に向かって、少し右上のテキスト "peak" から黒い矢印が伸びている図が生成される。

**注意点・落とし穴**:
- `xy`(矢印の先端=対象点)と `xytext`(テキストの位置)を混同しやすい。`xytext` を省略すると矢印は描かれず、`xy` の位置にそのままテキストが置かれる。

---

### `ax.text(...)`

**用途**: 任意の位置に自由なテキストを配置する(`annotate` と違い矢印なし)。

**シグネチャ**: `Axes.text(self, x, y, s, fontdict=None, **kwargs)`

**使用例**:
```python
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 0])
ax.text(1, 0.5, "peak label", ha="center", fontsize=12, color="red")
fig.savefig("28_text.png")
```
実行結果: 山型の折れ線の中腹あたり(座標 (1, 0.5))に赤字で "peak label" という文字列が中央揃えで配置された図が生成される。

**注意点・落とし穴**:
- 水平方向の基準点(`ha`、既定 `'left'`)・垂直方向の基準点(`va`、既定 `'baseline'`)を指定しないと、指定した座標がテキストの左端/ベースラインになり、意図した位置からずれて見えることがある。

---

### `ax.axhline(...)` / `ax.axvline(...)`

**用途**: Axes全体を貫く水平線(`axhline`)・垂直線(`axvline`)を引く。基準線やしきい値の表示に使う。

**シグネチャ**: `Axes.axhline(self, y=0, xmin=0, xmax=1, **kwargs)` / `Axes.axvline(self, x=0, ymin=0, ymax=1, **kwargs)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(2)
fig, ax = plt.subplots()
ax.plot(np.arange(10), rng.normal(size=10))
ax.axhline(0, color="gray", linestyle="--")
ax.axvline(5, color="red", linestyle=":")
fig.savefig("29_axhline_axvline.png")
```
実行結果: ランダムな折れ線グラフに、y=0の灰色破線とx=5の赤い点線が、それぞれAxes全体を貫くように重ねて表示される図が生成される。

**注意点・落とし穴**:
- `xmin`/`xmax`(`axhline`)や `ymin`/`ymax`(`axvline`)は**データ座標ではなくAxes内の相対位置(0〜1)**を表す。データ座標で線の長さを制限したい場合はこの引数ではなく、通常の `ax.plot([x1, x2], [y, y])` を使う方が直感的。

---

### `ax.axhspan(...)` / `ax.axvspan(...)`

**用途**: Axesを貫く水平方向・垂直方向の帯(範囲)を塗りつぶす。

**シグネチャ**: `Axes.axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs)` / `Axes.axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs)`

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
ax.plot(np.arange(10), np.arange(10))
ax.axhspan(3, 6, color="yellow", alpha=0.3)
ax.axvspan(2, 4, color="blue", alpha=0.2)
fig.savefig("30_axhspan_axvspan.png")
```
実行結果: 直線グラフに、y=3〜6の水平な黄色い帯とx=2〜4の垂直な青い帯が重ねて表示され、2つの帯が交差する部分は色が混ざって見える図が生成される。

**注意点・落とし穴**:
- `axhline`/`axvline` と同様、`axhspan` の `xmin`/`xmax`(および `axvspan` の `ymin`/`ymax`、つまり帯の"長さ方向"ではない側の引数)はAxes相対座標(0〜1)である点に注意。

---

## 複数プロット種

### `ax.hist(...)`

**用途**: ヒストグラム(度数分布)を描く。

**シグネチャ**: `Axes.hist(self, x, bins=None, *, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, data=None, **kwargs)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(42)
fig, ax = plt.subplots()
data = rng.normal(size=1000)
n, bins, patches = ax.hist(data, bins=30, color="steelblue", edgecolor="white")
fig.savefig("31_hist.png")
```
実行結果: 標準正規分布に従う1000個の乱数の、中央付近が高く両端が低い釣鐘型のヒストグラム(30ビン、白い縁取り付きの青い棒)が生成される。

**注意点・落とし穴**:
- 戻り値は `(度数の配列, ビン境界の配列, 描画されたパッチ)` の3つ組。`density=True` にすると縦軸は「度数」ではなく「面積の合計が1になる密度」に変わり、既定(`density=False`)の度数のグラフと縦軸のスケールが大きく異なる点に注意。

---

### `ax.boxplot(...)`

**用途**: 箱ひげ図(四分位範囲・中央値・外れ値)を描く。

**シグネチャ**: `Axes.boxplot(self, x, *, notch=None, sym=None, orientation='vertical', whis=None, positions=None, widths=None, patch_artist=None, tick_labels=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, flierprops=None, medianprops=None, meanprops=None, manage_ticks=True, autorange=False, zorder=None, capwidths=None, label=None, data=None)`(一部の高度な引数は省略)

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(42)
fig, ax = plt.subplots()
data = [rng.normal(size=100), rng.normal(loc=2, size=100), rng.normal(loc=-1, scale=2, size=100)]
ax.boxplot(data, tick_labels=["A", "B", "C"])
fig.savefig("32_boxplot.png")
```
実行結果: A/B/C 3群それぞれについて、中央値の線・箱(四分位範囲)・ひげ・外れ値の丸印からなる箱ひげ図が並んで表示される(Bは全体的に高く、Cはばらつきが大きい)。

**注意点・落とし穴**:
- **バージョン依存の破壊的変更を実機で確認**: matplotlib 3.11.1 では旧引数 `labels=[...]` を渡すと `TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'` になる(完全に削除済み)。ラベルを付けるには `tick_labels=` を使う必要がある。
- 同様に `vert=False`(縦横切り替え)を渡すと `MatplotlibDeprecationWarning: vert: bool was deprecated in Matplotlib 3.11 and will be removed in 3.13. Use orientation instead.` という非推奨警告が出る(3.13で削除予定)。新しいコードでは `orientation='horizontal'` を使うべき。

---

### `ax.violinplot(...)`

**用途**: バイオリンプロット(分布の形状をカーネル密度推定で可視化する、箱ひげ図の拡張)を描く。

**シグネチャ**: `Axes.violinplot(self, dataset, positions=None, *, orientation='vertical', widths=0.5, showmeans=False, showextrema=True, showmedians=False, quantiles=None, points=100, bw_method=None, side='both', facecolor=None, linecolor=None, data=None)`

**使用例**:
```python
fig, ax = plt.subplots()
ax.violinplot(data, showmeans=True)
fig.savefig("33_violinplot.png")
```
(`data` は `boxplot` の例と同じ3群のリスト)

実行結果: 3群それぞれについて、分布の密度に応じて左右に膨らんだバイオリン形状の図形が並び、平均値の横線(`showmeans=True`)が表示される。

**注意点・落とし穴**:
- `boxplot` と同様に `vert` 引数は matplotlib 3.11 で非推奨化されており、`orientation` を使う必要がある(実機で同じ `MatplotlibDeprecationWarning` を確認)。
- 既定では平均線・中央値線は表示されない(`showmeans`/`showmedians` をそれぞれ `True` にする必要がある)。

---

### `ax.imshow(...)`

**用途**: 2次元配列(画像やヒートマップ)をピクセルの色として表示する。

**シグネチャ**: `Axes.imshow(self, X, cmap=None, norm=None, *, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, **kwargs)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(42)
fig, ax = plt.subplots()
img = rng.random((20, 20))
im = ax.imshow(img, cmap="viridis")
fig.colorbar(im, ax=ax)
fig.savefig("34_imshow.png")
```
実行結果: 20x20のランダム値がviridisカラーマップ(紫〜黄)のマス目として表示され、右にカラーバーが付いたヒートマップが生成される。

**注意点・落とし穴**:
- 既定 `origin='upper'` のため、配列の1行目(インデックス0)は画像の**上端**に表示される(数学的なグラフのように下端にしたい場合は `origin='lower'` を指定)。
- y軸の目盛りが上から下に増加する(通常のプロットと逆向き)ため、他のプロットと重ねる際に軸の向きの整合性に注意が必要。

---

### `ax.pcolormesh(...)`

**用途**: `X`, `Y` 格子座標を伴う2次元データを疑似カラー(色分けされたメッシュ)で表示する。`imshow` と違い不等間隔の格子にも対応する。

**シグネチャ**: `Axes.pcolormesh(self, *args, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, shading=None, antialiased=False, **kwargs)`

**使用例**:
```python
import numpy as np
fig, ax = plt.subplots()
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)
pcm = ax.pcolormesh(X, Y, Z, cmap="coolwarm", shading="auto")
fig.colorbar(pcm, ax=ax)
fig.savefig("35_pcolormesh.png")
```
実行結果: `sin(X)*cos(Y)` の値に応じて赤(正)〜白(0)〜青(負)へ滑らかに色分けされた市松模様状のヒートマップが生成される。

**注意点・落とし穴**:
- `shading` を指定しないと matplotlib のバージョンによって既定挙動(`'flat'` か `'auto'` か)が変わり、`X`/`Y`/`Z` の shape 関係(`Z` が `X`,`Y` よりそれぞれ1小さい必要があるか等)でエラーや警告になることがある。迷ったら `shading='auto'` を明示するのが無難。

---

### `ax.contour(...)` / `ax.contourf(...)`

**用途**: 等高線(`contour`、線のみ)・塗りつぶし等高線(`contourf`)を描く。

**シグネチャ**: `Axes.contour(self, *args, data=None, **kwargs)` / `Axes.contourf(self, *args, data=None, **kwargs)`(`*args` は `(Z)` または `(X, Y, Z)`)

**使用例**:
```python
import numpy as np
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
cs = axes[0].contour(X, Y, Z, levels=10)
axes[0].clabel(cs, inline=True, fontsize=8)
cf = axes[1].contourf(X, Y, Z, levels=10, cmap="RdBu")
fig.colorbar(cf, ax=axes[1])
fig.savefig("36_contour.png")
```
実行結果: 左側は等高線とその数値ラベル(`clabel`)が描かれた図、右側は同じ等高線を10段階で塗りつぶした(`RdBu`カラーマップで赤〜青のグラデーション)図が並んで生成される。

**注意点・落とし穴**:
- `levels` に整数を渡すとその数だけ自動でレベルを決めるが、境界値が期待通りにならないことがある。特定の値で区切りたい場合は `levels=[...]` のように配列で明示する。

---

### `ax.pie(...)`

**用途**: 円グラフを描く。

**シグネチャ**: `Axes.pie(self, x, *, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=0, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, rotatelabels=False, normalize=True, hatch=None, data=None)`

**使用例**:
```python
fig, ax = plt.subplots()
sizes = [30, 25, 20, 25]
labels = ["A", "B", "C", "D"]
ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
fig.savefig("37_pie.png")
```
実行結果: A(30%)・B(25%)・C(20%)・D(25%)の4分割の円グラフが、それぞれの扇形に外側のラベルと内側にパーセント表示(`autopct`)付きで生成される。真上(12時の方向、`startangle=90`)からAが始まり反時計回りに並ぶ。

**注意点・落とし穴**:
- `normalize=True`(既定)のため、`x` の合計が100でなくても自動的に比率に正規化される(合計が100を超えても以下でも円グラフとして成立してしまう)。合計が100になっているかどうかのチェックは別途必要。

---

### `ax.hist2d(...)`

**用途**: 2変数の同時分布を2次元ヒストグラム(ビンごとの度数を色で表現)として可視化する。

**シグネチャ**: `Axes.hist2d(self, x, y, bins=10, *, range=None, density=False, weights=None, cmin=None, cmax=None, data=None, **kwargs)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(42)
fig, ax = plt.subplots()
x = rng.normal(size=2000)
y = rng.normal(size=2000) + x * 0.5
h = ax.hist2d(x, y, bins=30, cmap="Blues")
fig.colorbar(h[3], ax=ax)
fig.savefig("38_hist2d.png")
```
実行結果: x, y に正の相関がある2000点のデータについて、密集している中心付近ほど濃い青になる30x30ビンのヒートマップが生成され、右にカラーバー(度数の最大値25程度まで)が付く。

**注意点・落とし穴**:
- 戻り値は `(度数の2次元配列, x方向ビン境界, y方向ビン境界, QuadMesh画像オブジェクト)` の4つ組。カラーバーを付けるには4番目の要素(`h[3]`)を `colorbar` に渡す必要がある。

---

## 3Dプロット

### `fig.add_subplot(projection="3d")`

**用途**: 3D描画用の `Axes3D` オブジェクトを作成する(`mpl_toolkits.mplot3d` は matplotlib 本体に組み込み済みで、追加インポートなしで `projection="3d"` を指定するだけで使える)。

**使用例**:
```python
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
print(type(ax))
```
実行結果:
```
<class 'mpl_toolkits.mplot3d.axes3d.Axes3D'>
```

**注意点・落とし穴**:
- 昔のコード例に見られる `from mpl_toolkits.mplot3d import Axes3D` の明示的インポートは、`projection="3d"` を使うだけであれば現在は不要(登録のためのインポートのみが目的だったため)。ただし型ヒントなどで `Axes3D` クラス自体を参照したい場合はインポートが必要。

---

### `ax.plot_surface(X, Y, Z, ...)`

**用途**: 3次元のサーフェス(曲面)をメッシュとして描画する。

**シグネチャ**: `Axes3D.plot_surface(self, X, Y, Z, *, norm=None, vmin=None, vmax=None, lightsource=None, axlim_clip=False, **kwargs)`

**使用例**:
```python
import numpy as np
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, cmap="viridis")
fig.colorbar(surf, ax=ax, shrink=0.5)
fig.savefig("39_plot_surface.png")
```
実行結果: `sin(sqrt(x²+y²))` の同心円状の波紋のような曲面が、viridisカラーマップで色付けされた3D曲面として生成され、右に縮小されたカラーバーが付く。

**注意点・落とし穴**:
- `cmap` を指定すると `plot_surface` の戻り値を `colorbar` にそのまま渡せる(内部的に `Poly3DCollection` を返すが `mappable` として扱える)。

---

### `ax.scatter(xs, ys, zs, ...)`(3D散布図)

**用途**: 3次元空間内に点を散布する。2Dの `scatter` と同名だが `Axes3D` 上で呼ぶと3次元になる。

**シグネチャ**: `Axes3D.scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=None, *args, depthshade_minalpha=None, axlim_clip=False, data=None, **kwargs)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(3)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
xs = rng.random(100)
ys = rng.random(100)
zs = rng.random(100)
ax.scatter(xs, ys, zs, c=zs, cmap="plasma")
fig.savefig("40_scatter3d.png")
```
実行結果: 単位立方体内にランダムに散らばった100個の点が、z座標に応じてplasmaカラーマップ(紫〜黄)で色分けされた3D散布図が生成される。

**注意点・落とし穴**:
- `zs` を省略すると既定値 `0`(全点が z=0 の平面上)になる。3D散布図のつもりが `zs` を渡し忘れて平面になってしまうミスに注意。

---

### `ax.plot_wireframe(X, Y, Z, ...)`

**用途**: `plot_surface` の塗りつぶし版に対し、格子線(ワイヤーフレーム)だけで曲面を表現する。

**シグネチャ**: `Axes3D.plot_wireframe(self, X, Y, Z, *, axlim_clip=False, **kwargs)`

**使用例**:
```python
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
fig.savefig("41_plot_wireframe.png")
```
(`X`, `Y`, `Z` は `plot_surface` の例と同じ)

実行結果: 同じ波紋状の曲面が、塗りつぶしなしの格子状の線(`rstride`/`cstride=5` により5点おきの間引き)だけで表現された3D図が生成される。

**注意点・落とし穴**:
- `rstride`/`cstride`(既定は共に1)を大きくすると格子が粗くなり描画が軽くなるが、細かい形状の情報は失われる。大きなメッシュをそのまま `plot_surface`/`plot_wireframe` に渡すと非常に重くなるため、間引きが実用上重要。

---

## 保存・表示

### `fig.savefig(...)` / `plt.savefig(...)`

**用途**: Figure をファイル(PNG, PDF, SVG など)として保存する。

**シグネチャ**: `Figure.savefig(self, fname, *, transparent=None, **kwargs)`(`dpi`, `bbox_inches`, `facecolor` などの旧来お馴染みのオプションは、このバージョンでは明示引数ではなく `**kwargs` 経由で受け取る形に整理されている)

**使用例**:
```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
fig.savefig("42_savefig.png", dpi=150, bbox_inches="tight", transparent=False)
```
実行結果: dpi 150、余白を自動トリミング(`bbox_inches="tight"`)した不透明背景のPNGファイルが例外なく生成される(実行確認済み、ファイルサイズ 28,568 バイト)。

**注意点・落とし穴**:
- 拡張子(`.png`, `.pdf`, `.svg`, `.jpg` 等)から保存形式が自動判定される。ベクター形式(PDF/SVG)は拡大しても劣化しないため論文用途などに向くが、`imshow` のような大きな配列を描画した図では容量が肥大化しやすい。
- `bbox_inches="tight"` を使うと、複数回保存したときに(タイトルの有無などで)Figureごとに余白の切り取られ方が微妙に変わり、複数図の縦横サイズを完全に揃えたい場合には不向きなことがある。

---

### `plt.show(...)` / `plt.close(...)`

**用途**: `show` はインタラクティブなウィンドウに図を表示する(GUIバックエンド時)。`close` はメモリ上のFigureを破棄する。

**シグネチャ**: `plt.show(*, block=None)` / `plt.close(fig=None)`

**使用例**:
```python
fig = plt.figure()
print(len(plt.get_fignums()))
plt.close(fig)
print(len(plt.get_fignums()))
```
実行結果:
```
1
0
```

**注意点・落とし穴**:
- `Agg` バックエンド(GUIなし、サーバー/バッチ処理向け)では `plt.show()` は実質何もしない。画面表示が目的でないスクリプトでは `savefig` だけで十分。
- `plt.close()` を引数なしで呼ぶと「現在アクティブな」Figureだけを閉じる。ループで大量にFigureを作る処理で `close` を忘れると、メモリを大量に消費し続ける(「too many open figures」警告)ので、`plt.close(fig)` または `plt.close("all")` を明示するのが安全。

---

## スタイル・テーマ(rcParams)

### `plt.style.use(...)` / `plt.style.context(...)`

**用途**: あらかじめ用意されたスタイルシート(配色・フォント・グリッドなどのプリセット)を適用する。

**シグネチャ**: `plt.style.use(style)`

**使用例**:
```python
print(plt.style.available[:5])
with plt.style.context("ggplot"):
    fig, ax = plt.subplots()
    ax.plot(range(10))
    ax.set_title("ggplot style")
    fig.savefig("43_style_ggplot.png")
```
実行結果:
```
['Solarize_Light2', 'bmh', 'classic', 'dark_background', 'fast']
```
`with plt.style.context("ggplot")` のブロック内で作った図は、灰色の背景に白いグリッド線、オレンジがかった線色というggplot風の見た目になる。

**注意点・落とし穴**:
- `plt.style.use(...)` はグローバルな `rcParams` を書き換えるため、以降作成する全ての図に影響し続ける。一時的に使いたいだけなら `with plt.style.context(...):` を使う方が安全(ブロックを抜けると元の設定に戻る)。

---

### `plt.rcParams` / `matplotlib.rcParams`

**用途**: フォントサイズ・線の太さ・デフォルト色など、matplotlibの全体設定を保持する辞書ライクなオブジェクト。

**使用例**:
```python
import matplotlib as mpl
print(mpl.rcParams["lines.linewidth"])
mpl.rcParams["lines.linewidth"] = 3
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 2, 1])
fig.savefig("44_rcparams.png")
```
実行結果:
```
1.5
```
既定の線の太さ1.5から3に変更した結果、通常より太い線でプロットされた図が生成される。

**注意点・落とし穴**:
- `rcParams` はプロセス全体でグローバルに共有される状態。一度書き換えると `plt.rcdefaults()` で明示的にリセットするか、プロセスを再起動しない限り以後全ての図に影響し続ける。

---

### `plt.rc(...)` / `plt.rcdefaults()`

**用途**: `rcParams` の特定グループ(`"font"`, `"lines"` など)の設定をキーワード引数でまとめて変更する(`rc`)、全設定をデフォルトに戻す(`rcdefaults`)。

**シグネチャ**: `plt.rc(group, **kwargs)` / `plt.rcdefaults()`

**使用例**:
```python
plt.rc("font", size=14, family="sans-serif")
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [3, 1, 2])
ax.set_title("rc font size 14")
fig.savefig("45_rc.png")
plt.rcdefaults()
```
実行結果: タイトル・軸目盛りなど図中のすべての文字が通常より大きい14ポイントで表示された図が生成される。その後 `plt.rcdefaults()` で全設定が既定値に戻る。

---

## その他

### `plt.gca()` / `plt.gcf()`

**用途**: 現在アクティブな Axes(`gca` = get current axes)・Figure(`gcf` = get current figure)を取得する。

**シグネチャ**: `plt.gca() -> Axes` / `plt.gcf() -> Figure`

**使用例**:
```python
plt.figure()
plt.plot([1, 2, 3])
ax = plt.gca()
fig = plt.gcf()
print(type(ax), type(fig))
```
実行結果:
```
<class 'matplotlib.axes._axes.Axes'> <class 'matplotlib.figure.Figure'>
```

**注意点・落とし穴**:
- オブジェクト指向スタイル(`fig, ax = plt.subplots()` を保持して使い回す)の方が、複数Axesを扱うコードでは「今どれがアクティブか」を気にせず済み事故が少ない。`plt.gca()`/`plt.gcf()` は対話環境でのクイックな操作や、外部ライブラリが暗黙に描画したAxesを後から取得したい場合に便利。

---

### `matplotlib.ticker.MultipleLocator` / `matplotlib.ticker.FuncFormatter`

**用途**: 目盛り位置を任意の間隔に固定する(`MultipleLocator`)、目盛りラベルの表示形式を任意の関数で変換する(`FuncFormatter`)。

**シグネチャ**: `MultipleLocator(base=1.0, offset=0.0)` / `FuncFormatter(func)`(`func` は `(value, position) -> str` の関数)

**使用例**:
```python
import numpy as np
import matplotlib.ticker as mticker
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.1f}$"))
fig.savefig("46_ticker.png")
```
実行結果: x軸の目盛りが2刻み(0, 2, 4, ...)に固定され、y軸の目盛りラベルが "1.0$" のように末尾に "$" が付いた文字列に変換された図が生成される。

**注意点・落とし穴**:
- `FuncFormatter` に渡す関数は `(value, position)` の2引数を取る必要がある(`position` はその目盛りが何番目かの整数で、通常は使わなくても受け取る引数として必須)。引数の数を間違えると `TypeError` になる。

---

### `ax.spines`

**用途**: Axesの四辺の枠線(上下左右、"spine")を個別に取得し、表示/非表示や色・太さを制御する。

**使用例**:
```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("gray")
fig.savefig("47_spines.png")
print(list(ax.spines.keys()))
```
実行結果:
```
['left', 'right', 'bottom', 'top']
```
上と右の枠線が消え、左の枠線だけ灰色になった、いわゆる「オープンスタイル」の見た目の図が生成される。

**注意点・落とし穴**:
- `ax.spines` は辞書のように `["top"]`, `["right"]`, `["left"]`, `["bottom"]` の4キーでアクセスするオブジェクト(`Spines` クラス)。`ax.spines.top` のような属性アクセスも可能だが、キー名を間違えると `KeyError` になる。
