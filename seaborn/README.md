# seaborn 逆引き辞書

seaborn 0.13.2 で検証済み

本ドキュメントに掲載しているシグネチャ・実行結果は、すべて `/home/manaty/library-practicing/.venv/bin/python`(seaborn 0.13.2、matplotlib 3.11.1、`Agg`バックエンド)で実際にコードを実行して得たものです。シグネチャは `inspect.signature()` で取得した実際の値であり、記憶や公式ドキュメントからの転記ではありません。seaborn の出力はグラフ(画像)であるため、各項目では実際にコードを実行してPNGに保存し、目視で確認した内容を「実行結果」として1〜2文で記述しています(標準出力の文字列ではありません)。データは seaborn 同梱のサンプルデータセット(`tips` / `iris` / `flights` / `penguins`)をインターネット経由で取得して使用しました。

## 目次

1. [関係性の可視化](#関係性の可視化)
2. [分布の可視化](#分布の可視化)
3. [カテゴリカルデータ](#カテゴリカルデータ)
4. [回帰・線形モデル](#回帰線形モデル)
5. [ヒートマップ・行列](#ヒートマップ行列)
6. [ペアプロット・グリッド](#ペアプロットグリッド)
7. [スタイル・テーマ・カラーパレット](#スタイルテーマカラーパレット)
8. [データセット・ユーティリティ](#データセットユーティリティ)
9. [応用・発展](#応用発展)
    - [新しいオブジェクト指向インターフェース(seaborn.objects)](#新しいオブジェクト指向インターフェースseabornobjects)
    - [FacetGridの高度なカスタマイズ](#facetgridの高度なカスタマイズ)
    - [統計的注釈の応用](#統計的注釈の応用)
    - [カラーパレットの高度な作成](#カラーパレットの高度な作成)

---

## 関係性の可視化

### `sns.scatterplot(...)`

**用途**: 2変数の散布図を描く。`hue`/`size`/`style` で第3〜5の変数を色・大きさ・マーカー形状にマッピングできる。

**シグネチャ**: `sns.scatterplot(data=None, *, x=None, y=None, hue=None, size=None, style=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, markers=True, style_order=None, legend='auto', ax=None, **kwargs)`

**使用例**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
tips = sns.load_dataset("tips")
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="smoker", size="size")
plt.savefig("scatterplot.png")
```
実行結果:
散布図が描画され、`hue="time"` で点の色が Lunch/Dinner の2色に、`style="smoker"` でマーカー形状が○/×に、`size="size"` で点の大きさが人数に応じて変化する。凡例には time・size・smoker の3つが積み重ねて表示される。

**注意点・落とし穴**:
- `hue`/`size`/`style` を同時に使うと凡例が縦に長くなる(3種の凡例が結合されて1つのボックスに表示される)。

### `sns.lineplot(...)`

**用途**: 折れ線グラフを描く。同じx値に複数行がある場合、デフォルトで平均値と信頼区間(バンド)を自動計算して描画する。

**シグネチャ**: `sns.lineplot(data=None, *, x=None, y=None, hue=None, size=None, style=None, units=None, weights=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, seed=None, orient='x', sort=True, err_style='band', err_kws=None, legend='auto', ci='deprecated', ax=None, **kwargs)`

**使用例**:
```python
flights = sns.load_dataset("flights")
sns.lineplot(data=flights, x="year", y="passengers", hue="month", legend=False)
plt.savefig("lineplot.png")
```
実行結果:
年ごとの乗客数の推移を月別に色分けした折れ線が12本描かれ、右肩上がりの増加傾向が確認できる。`hue="month"` で12色に塗り分けられている。

**注意点・落とし穴**:
- `errorbar=('ci', 95)` がデフォルトで、同一x値に対して複数の観測値がある場合は自動的に平均線+信頼区間バンドになる(この例のように `hue` でグループが1本ずつしかない場合はバンドは出ない)。旧バージョンの `ci=` 引数は非推奨(`errorbar=`に統合)。

### `sns.relplot(...)`

**用途**: `scatterplot`/`lineplot` を `FacetGrid` でラップし、`row`/`col` によるファセット分割(小さい図の格子表示)を組み合わせられる関数レベルインターフェース。

**シグネチャ**: `sns.relplot(data=None, *, x=None, y=None, hue=None, size=None, style=None, units=None, weights=None, row=None, col=None, col_wrap=None, row_order=None, col_order=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, markers=None, dashes=None, style_order=None, legend='auto', kind='scatter', height=5, aspect=1, facet_kws=None, **kwargs)`

**使用例**:
```python
g = sns.relplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker", kind="scatter")
g.savefig("relplot.png")
```
実行結果:
`time`(Lunch/Dinner)ごとに2つのパネルへ分割された散布図が横並びで描かれ、各パネル内で `smoker` により点の色が塗り分けられる。戻り値は `FacetGrid` オブジェクト。

**注意点・落とし穴**:
- `kind="scatter"`(デフォルト)と `kind="line"` で内部的に呼ばれる関数が `scatterplot`/`lineplot` に切り替わり、使える引数の組み合わせも変わる。`ax=` は使えない(`FacetGrid` が軸を管理するため)。

---

## 分布の可視化

### `sns.histplot(...)`

**用途**: ヒストグラムを描く。`kde=True` で密度曲線を重ねられる。

**シグネチャ**: `sns.histplot(data=None, *, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None, discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, color=None, log_scale=None, legend=True, ax=None, **kwargs)`

**使用例**:
```python
sns.histplot(data=tips, x="total_bill", hue="time", kde=True, bins=20)
plt.savefig("histplot.png")
```
実行結果:
`total_bill` を20ビンに分けた積み重ね(`multiple="layer"`)ヒストグラムが Lunch/Dinner の2色で重なって描かれ、それぞれにKDE曲線が重畳表示される。y軸は `stat="count"`(件数)。

**注意点・落とし穴**:
- デフォルトの `multiple="layer"` は複数グループを半透明で重ねるだけなので、`hue` のグループ数が多いと見づらい。積み上げたい場合は `multiple="stack"` を使う。

### `sns.kdeplot(...)`

**用途**: カーネル密度推定(KDE)による滑らかな分布曲線を描く。1変数・2変数(等高線)どちらにも対応。

**シグネチャ**: `sns.kdeplot(data=None, *, x=None, y=None, hue=None, weights=None, palette=None, hue_order=None, hue_norm=None, color=None, fill=None, multiple='layer', common_norm=True, common_grid=False, cumulative=False, bw_method='scott', bw_adjust=1, warn_singular=True, log_scale=None, levels=10, thresh=0.05, gridsize=200, cut=3, clip=None, legend=True, cbar=False, cbar_ax=None, cbar_kws=None, ax=None, **kwargs)`

**使用例**:
```python
sns.kdeplot(data=tips, x="total_bill", hue="time", fill=True, common_norm=False)
plt.savefig("kdeplot_1d.png")

penguins = sns.load_dataset("penguins")
sns.kdeplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", fill=True)
plt.savefig("kdeplot_2d.png")
```
実行結果:
1変数版は Lunch/Dinner それぞれの `total_bill` 分布が塗りつぶし曲線として重なって描かれる(`common_norm=False` のため各曲線は独立に面積1に正規化される)。2変数版は `bill_length_mm` × `bill_depth_mm` の同時分布が種ごとに色分けされた塗りつぶし等高線として描かれ、3種のペンギンのクラスタがおおむね分離して見える。

**注意点・落とし穴**:
- `common_norm=True`(デフォルト)だとグループ間でサンプル数の違いが曲線の高さに反映される(サンプル数が多いグループの山が高くなる)。グループごとの「形」だけを比較したい場合は `common_norm=False` にする。

### `sns.displot(...)`

**用途**: `histplot`/`kdeplot`/`ecdfplot` を `FacetGrid` でラップした、分布可視化の関数レベルインターフェース。`kind=` で描画方式を切り替える。

**シグネチャ**: `sns.displot(data=None, *, x=None, y=None, hue=None, row=None, col=None, weights=None, kind='hist', rug=False, rug_kws=None, log_scale=None, legend=True, palette=None, hue_order=None, hue_norm=None, color=None, col_wrap=None, row_order=None, col_order=None, height=5, aspect=1, facet_kws=None, **kwargs)`

**使用例**:
```python
g = sns.displot(data=penguins, x="flipper_length_mm", hue="species", kind="kde", multiple="stack")
g.savefig("displot.png")
```
実行結果:
`kind="kde"` と `multiple="stack"` の組み合わせにより、3種のペンギンの `flipper_length_mm` のKDE曲線が積み上げ式(下から Adelie/Chinstrap/Gentoo)で塗りつぶされ、合計の分布形状も同時に確認できる図になる。

**注意点・落とし穴**:
- デフォルトの `kind="hist"` を明示せず使うと `displot` は常にヒストグラムになる。KDEにしたい場合は `kind="kde"` の指定を忘れやすい。

### `sns.ecdfplot(...)`

**用途**: 経験累積分布関数(ECDF)を描く。ビン幅に依存せず分布全体を1本の階段状の線で表現できる。

**シグネチャ**: `sns.ecdfplot(data=None, *, x=None, y=None, hue=None, weights=None, stat='proportion', complementary=False, palette=None, hue_order=None, hue_norm=None, log_scale=None, legend=True, ax=None, **kwargs)`

**使用例**:
```python
sns.ecdfplot(data=tips, x="total_bill", hue="time")
plt.savefig("ecdfplot.png")
```
実行結果:
`total_bill` に対する累積比率(0〜1)が階段状の線で描かれ、Lunch/Dinner の2本のECDF曲線を重ねて比較できる(Dinnerの方が高額帯に裾が伸びている様子が見える)。

**注意点・落とし穴**:
- ヒストグラムと違い、ビンの取り方による見た目の違いが生じないため、複数グループの分布を厳密に比較したい場合に向く。

### `sns.rugplot(...)`

**用途**: 個々のデータ点を軸に沿った短い縦線(ラグ)として表示する。他のプロットに重ねて実データの位置を補助的に示すのに使う。

**シグネチャ**: `sns.rugplot(data=None, *, x=None, y=None, hue=None, height=0.025, expand_margins=True, palette=None, hue_order=None, hue_norm=None, legend=True, ax=None, **kwargs)`

**使用例**:
```python
ax = sns.kdeplot(data=tips, x="total_bill")
sns.rugplot(data=tips, x="total_bill", ax=ax)
plt.savefig("rugplot.png")
```
実行結果:
KDE曲線の下(x軸付近)に、実際の `total_bill` の各観測値の位置を示す短い縦線群が密に描かれ、データが密集している価格帯が視覚的にわかる。

---

## カテゴリカルデータ

### `sns.boxplot(...)`

**用途**: カテゴリ別の箱ひげ図(四分位範囲・中央値・外れ値)を描く。

**シグネチャ**: `sns.boxplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, dodge='auto', width=0.8, gap=0, whis=1.5, linecolor='auto', linewidth=None, fliersize=None, hue_norm=None, native_scale=False, log_scale=None, formatter=None, legend='auto', ax=None, **kwargs)`

**使用例**:
```python
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker")
plt.savefig("boxplot.png")
```
実行結果:
曜日(Thur/Fri/Sat/Sun)ごとに、喫煙者/非喫煙者の箱ひげ図が横に並べて(`dodge`)描かれ、土日の方が総額の中央値・ばらつきがやや大きいことが見て取れる。外れ値は箱の外に個別の点として表示される。

**注意点・落とし穴**:
- `whis=1.5`(デフォルト)はIQR(四分位範囲)の1.5倍を超える点を外れ値として点表示する基準。分布によっては外れ値が非常に多く見えることがある。

### `sns.violinplot(...)`

**用途**: 箱ひげ図とKDEを組み合わせ、分布の形状(バイモーダルかどうか等)まで可視化する。

**シグネチャ**: `sns.violinplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, inner='box', split=False, width=0.8, dodge='auto', gap=0, linewidth=None, linecolor='auto', cut=2, gridsize=100, bw_method='scott', bw_adjust=1, density_norm='area', common_norm=False, hue_norm=None, formatter=None, log_scale=None, native_scale=False, legend='auto', ax=None, **kwargs)`(このほか `scale`/`scale_hue`/`bw` は非推奨引数として残っている)

**使用例**:
```python
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker", split=True)
plt.savefig("violinplot.png")
```
実行結果:
各曜日について、`split=True` により喫煙者(左半分)・非喫煙者(右半分)のバイオリン形状が1本にまとめて描かれ、中央に箱ひげ(`inner="box"`)が重ねて表示される。左右の分布形状の非対称性を直接比較できる。

**注意点・落とし穴**:
- `split=True` は `hue` の水準がちょうど2つのときのみ使える。
- `scale`(旧引数、面積/幅/カウントの正規化方法)は非推奨になり `density_norm` に置き換わっている。

### `sns.barplot(...)`

**用途**: カテゴリ別の集計値(デフォルトは平均)を棒グラフで表示し、エラーバーで信頼区間・標準偏差などを重ねる。

**シグネチャ**: `sns.barplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, seed=None, units=None, weights=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, hue_norm=None, width=0.8, dodge='auto', gap=0, log_scale=None, native_scale=False, formatter=None, legend='auto', capsize=0, err_kws=None, ax=None, **kwargs)`(`ci`/`errcolor`/`errwidth` は非推奨引数)

**使用例**:
```python
sns.barplot(data=tips, x="day", y="total_bill", hue="sex", errorbar="sd")
plt.savefig("barplot.png")
```
実行結果:
曜日×性別で `total_bill` の平均値を高さとする棒グラフが描かれ、各棒の上に `errorbar="sd"` による標準偏差のエラーバーが表示される。

**注意点・落とし穴**:
- pandasの `groupby().mean()` のような単純集計ではなく、デフォルトでは1000回のブートストラップによる95%信頼区間(`errorbar=('ci', 95)`)を計算するため、データ量が大きいとやや低速になる。旧来の `ci=` 引数は非推奨で `errorbar=` に統合された。

### `sns.stripplot(...)` / `sns.swarmplot(...)`

**用途**: カテゴリごとに個々の観測値を点として表示する(`stripplot` はジッターでランダムに散らす、`swarmplot` は重ならないよう整列させる)。

**シグネチャ**: `sns.stripplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, jitter=True, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor=<default>, linewidth=0, hue_norm=None, log_scale=None, native_scale=False, formatter=None, legend='auto', ax=None, **kwargs)` / `sns.swarmplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor=None, linewidth=0, hue_norm=None, log_scale=None, native_scale=False, formatter=None, legend='auto', warn_thresh=0.05, ax=None, **kwargs)`

**使用例**:
```python
sns.stripplot(data=tips, x="day", y="total_bill", hue="sex", dodge=True)
plt.savefig("stripplot.png")

sns.swarmplot(data=tips, x="day", y="total_bill", hue="sex")
plt.savefig("swarmplot.png")
```
実行結果:
どちらも曜日ごとに男女別の点群を横に並べて描くが、`stripplot` は点がx方向にランダムにジッターして重なりが多いのに対し、`swarmplot` は点同士が重ならないよう自動的に横に広げられ、密度の高さがビーズ状の幅の広がりとして視覚化される。

**注意点・落とし穴**:
- `swarmplot` は全点を非重複配置するアルゴリズムのため、データ数が非常に多い(数千件超)と描画が遅くなったり点が収まりきらず警告(`warn_thresh`)が出ることがある。

### `sns.countplot(...)`

**用途**: カテゴリごとの出現回数(度数)を棒グラフで表示する。`y=` に集計対象の値を渡す必要はない。

**シグネチャ**: `sns.countplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, hue_norm=None, stat='count', width=0.8, dodge='auto', gap=0, log_scale=None, native_scale=False, formatter=None, legend='auto', ax=None, **kwargs)`

**使用例**:
```python
sns.countplot(data=tips, x="day", hue="sex")
plt.savefig("countplot.png")
```
実行結果:
曜日ごとの来店件数(行数)が男女別に色分けされた棒として描かれ、Sat/Sunの来店件数が多いことがわかる。

**注意点・落とし穴**:
- `stat="count"`(デフォルト)以外に `stat="percent"` を指定すると、全体に対する割合(%)で棒の高さを表示できる。

### `sns.pointplot(...)`

**用途**: カテゴリごとの集計値(平均など)を点で示し、点同士を線で結んで水準間の変化を強調する。

**シグネチャ**: `sns.pointplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, seed=None, units=None, weights=None, color=None, palette=None, hue_norm=None, markers=<default>, linestyles=<default>, dodge=False, log_scale=None, native_scale=False, orient=None, capsize=0, formatter=None, legend='auto', err_kws=None, ax=None, **kwargs)`

**使用例**:
```python
sns.pointplot(data=tips, x="day", y="total_bill", hue="sex", dodge=True)
plt.savefig("pointplot.png")
```
実行結果:
曜日ごとの平均 `total_bill` が男女別に点と信頼区間の縦線で示され、点同士が折れ線でつながることで曜日間の傾向(週末に向けて上昇するなど)が `barplot` より追いやすい形で表示される。

**注意点・落とし穴**:
- 見た目は `lineplot` に似ているが、x軸は連続値ではなく離散カテゴリとして扱われる(`native_scale=True` にすると数値カテゴリを実際の数直線上の位置に配置できる)。

### `sns.boxenplot(...)`

**用途**: 箱ひげ図を拡張し、分位点を段階的に細分化した「letter-value plot」を描く。データ数が多い場合に裾の情報量を保ったまま可視化できる。

**シグネチャ**: `sns.boxenplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, dodge='auto', width=0.8, gap=0, linewidth=None, linecolor=None, width_method='exponential', k_depth='tukey', outlier_prop=0.007, trust_alpha=0.05, showfliers=True, hue_norm=None, log_scale=None, native_scale=False, formatter=None, legend='auto', ax=None, **kwargs)`

**使用例**:
```python
sns.boxenplot(data=tips, x="day", y="total_bill")
plt.savefig("boxenplot.png")
```
実行結果:
曜日ごとに中央の太い箱から外側に向かって段階的に細くなる複数の箱が積み重なった図が描かれ、`boxplot` よりも分布の裾(テール)の広がり方が細かい階層で表現される。

**注意点・落とし穴**:
- `boxplot` と似た見た目だが分位点の刻み方(`k_depth`)がデータ数に応じて自動決定されるため、サンプルサイズが小さいと `boxplot` とほぼ同じ見た目になる。

### `sns.catplot(...)`

**用途**: `boxplot`/`violinplot`/`barplot`/`stripplot` 等を `FacetGrid` でラップした、カテゴリカルデータ可視化の関数レベルインターフェース。

**シグネチャ**: `sns.catplot(data=None, *, x=None, y=None, hue=None, row=None, col=None, kind='strip', estimator='mean', errorbar=('ci', 95), n_boot=1000, seed=None, units=None, weights=None, order=None, hue_order=None, row_order=None, col_order=None, col_wrap=None, height=5, aspect=1, log_scale=None, native_scale=False, formatter=None, orient=None, color=None, palette=None, hue_norm=None, legend='auto', legend_out=True, sharex=True, sharey=True, margin_titles=False, facet_kws=None, **kwargs)`

**使用例**:
```python
g = sns.catplot(data=tips, x="day", y="total_bill", col="time", kind="box", hue="smoker")
g.savefig("catplot.png")
```
実行結果:
`time`(Lunch/Dinner)ごとに2パネルへ分割された箱ひげ図(`kind="box"`)が描かれ、各パネル内で `smoker` による箱の色分けがされる。

**注意点・落とし穴**:
- `kind` のデフォルトは `"strip"`。`boxplot` 相当が欲しい場合は `kind="box"` を明示しないと想定と異なる図(点群)になる。

---

## 回帰・線形モデル

### `sns.regplot(...)`

**用途**: 散布図に回帰直線(または多項式・ロバスト回帰など)と信頼区間を重ねて描く。

**シグネチャ**: `sns.regplot(data=None, *, x=None, y=None, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, seed=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=True, dropna=True, x_jitter=None, y_jitter=None, label=None, color=None, marker='o', scatter_kws=None, line_kws=None, ax=None)`

**使用例**:
```python
sns.regplot(data=tips, x="total_bill", y="tip")
plt.savefig("regplot.png")
```
実行結果:
`total_bill` と `tip` の散布図に、右肩上がりの線形回帰直線と95%信頼区間の帯が重ねて描かれる。

**注意点・落とし穴**:
- `regplot` は単一の `Axes` にしか描けず `hue` によるグループ分けができない(グループごとに色分けした回帰直線が欲しい場合は `lmplot` を使う)。

### `sns.lmplot(...)`

**用途**: `regplot` を `FacetGrid` でラップし、`hue`/`col`/`row` によるグループ分け・ファセット分割付きの回帰直線を描ける。

**シグネチャ**: `sns.lmplot(data, *, x=None, y=None, hue=None, col=None, row=None, palette=None, col_wrap=None, height=5, aspect=1, markers='o', sharex=None, sharey=None, hue_order=None, col_order=None, row_order=None, legend=True, legend_out=None, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, seed=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=True, x_jitter=None, y_jitter=None, scatter_kws=None, line_kws=None, facet_kws=None)`

**使用例**:
```python
g = sns.lmplot(data=tips, x="total_bill", y="tip", hue="smoker", col="time")
g.savefig("lmplot.png")
```
実行結果:
`time`(Lunch/Dinner)ごとに2パネルへ分割され、各パネル内で喫煙者/非喫煙者それぞれの散布図と回帰直線・信頼区間帯が別の色で重ねて描かれる。

**注意点・落とし穴**:
- `regplot` とほぼ同じ回帰系引数(`order`/`logistic`/`lowess`など)を持つが、`data` は第1位置引数として必須(キーワード省略不可)であり、`ax=` は使えない点が `regplot` と異なる。

### `sns.residplot(...)`

**用途**: 単回帰の残差(実測値−予測値)をプロットし、残差にパターンが残っていないか(線形性の仮定が妥当か)を確認する。

**シグネチャ**: `sns.residplot(data=None, *, x=None, y=None, x_partial=None, y_partial=None, lowess=False, order=1, robust=False, dropna=True, label=None, color=None, scatter_kws=None, line_kws=None, ax=None)`

**使用例**:
```python
sns.residplot(data=tips, x="total_bill", y="tip")
plt.savefig("residplot.png")
```
実行結果:
x軸に `total_bill`、y軸に線形回帰の残差を取った散布図が描かれ、y=0の水平線を中心に残差がやや扇状に広がっている(高額帯ほどばらつきが大きい)様子が確認できる。

**注意点・落とし穴**:
- 残差に明確な曲線パターンが見える場合は、線形回帰(`order=1`)ではなく高次の関係を疑うサインになる。

---

## ヒートマップ・行列

### `sns.heatmap(...)`

**用途**: 2次元の数値行列(DataFrameなど)を色の濃淡で可視化する。相関行列やクロス集計表の表示に使われる。

**シグネチャ**: `sns.heatmap(data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)`

**使用例**:
```python
flights = sns.load_dataset("flights")
flights_wide = flights.pivot(index="month", columns="year", values="passengers")
sns.heatmap(flights_wide, annot=True, fmt="d", cmap="YlGnBu")
plt.savefig("heatmap.png")
```
実行結果:
月(行)×年(列)の乗客数を表す12行×12列の行列が、値が大きいほど濃い青緑になるヒートマップとして描かれ、各セルには `fmt="d"` により整数値が注記(`annot=True`)される。夏場(6〜9月)や後年ほど色が濃い。

**注意点・落とし穴**:
- `heatmap` はワイド形式(行×列がそのまま軸に対応する)のデータを渡す必要がある。ロング形式のDataFrameはあらかじめ `pivot`/`pivot_table` でワイド形式に変換してから渡す。

### `sns.clustermap(...)`

**用途**: `heatmap` に階層的クラスタリングによる行・列の並べ替えとデンドログラム(樹形図)を組み合わせる。

**シグネチャ**: `sns.clustermap(data, *, pivot_kws=None, method='average', metric='euclidean', z_score=None, standard_scale=None, figsize=(10, 10), cbar_kws=None, row_cluster=True, col_cluster=True, row_linkage=None, col_linkage=None, row_colors=None, col_colors=None, mask=None, dendrogram_ratio=0.2, colors_ratio=0.03, cbar_pos=(0.02, 0.8, 0.05, 0.18), tree_kws=None, **kwargs)`

**使用例**:
```python
g = sns.clustermap(flights_wide, cmap="mako", standard_scale=1)
g.savefig("clustermap.png")
```
実行結果:
`standard_scale=1` により列(年)ごとに0〜1へ正規化されたうえで、行(月)・列(年)ともに類似度に基づいて並べ替えられたヒートマップが描かれ、上端・左端にデンドログラムが付く。冬季(Nov/Dec/Jan/Feb)の月同士、夏季(Jun〜Sep)の月同士がそれぞれ近いクラスタとしてまとまることが視覚的にわかる。

**注意点・落とし穴**:
- `heatmap` と異なり `ax=` を受け付けない(専用の `ClusterGrid` を内部で新規作成するため、既存の `Axes` に重ねて描画できない)。
- 戻り値は `ClusterGrid` オブジェクトで、`g.savefig(...)` のように呼ぶ(`plt.savefig` でも図全体は保存できるが `Axes` へのアクセスは `g.ax_heatmap` 等を使う)。

---

## ペアプロット・グリッド

### `sns.pairplot(...)`

**用途**: DataFrameの全数値列同士の組み合わせについて、散布図行列(対角はヒストグラム/KDE)を一括で描く。

**シグネチャ**: `sns.pairplot(data, *, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, corner=False, dropna=False, plot_kws=None, diag_kws=None, grid_kws=None, size=None)`

**使用例**:
```python
iris = sns.load_dataset("iris")
g = sns.pairplot(iris, hue="species", diag_kind="kde")
g.savefig("pairplot.png")
```
実行結果:
4つの数値列(sepal/petal の長さ・幅)の全組み合わせについて4×4の散布図行列が描かれ、対角には各列のKDE曲線、非対角には散布図が種(species)ごとに色分けされて表示される。setosaが他の2種から明確に分離している様子が確認できる。

**注意点・落とし穴**:
- 数値列の数が多いDataFrameにそのまま適用すると図が巨大かつ低速になるため、`vars=` で対象列を絞るのが実務的。

### `sns.jointplot(...)`

**用途**: 2変数の散布図(中央)と、それぞれの周辺分布(上端・右端)を1つの図にまとめる。

**シグネチャ**: `sns.jointplot(data=None, *, x=None, y=None, hue=None, kind='scatter', height=6, ratio=5, space=0.2, dropna=False, xlim=None, ylim=None, color=None, palette=None, hue_order=None, hue_norm=None, marginal_ticks=False, joint_kws=None, marginal_kws=None, **kwargs)`

**使用例**:
```python
g = sns.jointplot(data=tips, x="total_bill", y="tip", hue="time", kind="scatter")
g.savefig("jointplot.png")
```
実行結果:
中央に `total_bill` × `tip` の散布図、上端と右端にそれぞれの変数の周辺KDE曲線(塗りつぶし)が配置された図が描かれ、Lunch/Dinnerで色分けされる。

**注意点・落とし穴**:
- `kind="hex"` や `kind="reg"` など `kind` を変えると中央のプロットの種類(六角ビン集計、回帰直線付き散布図など)が切り替わるが、`kind="reg"` と `hue` は同時使用できない。

### `sns.PairGrid(...)` / `sns.FacetGrid(...)` / `sns.JointGrid(...)`

**用途**: `pairplot`/`relplot`等/`jointplot` の内部で使われている、より柔軟にカスタマイズ可能な低レベルのグリッドクラス。`.map()`/`.map_diag()`/`.map_offdiag()`/`.plot()` などに任意の描画関数を渡してグリッドを組み立てる。

**シグネチャ**: `sns.PairGrid(data, *, hue=None, vars=None, x_vars=None, y_vars=None, hue_order=None, palette=None, hue_kws=None, corner=False, diag_sharey=True, height=2.5, aspect=1, layout_pad=0.5, despine=True, dropna=False)` / `sns.FacetGrid(data, *, row=None, col=None, hue=None, col_wrap=None, sharex=True, sharey=True, height=3, aspect=1, palette=None, row_order=None, col_order=None, hue_order=None, hue_kws=None, dropna=False, legend_out=True, despine=True, margin_titles=False, xlim=None, ylim=None, subplot_kws=None, gridspec_kws=None)` / `sns.JointGrid(data=None, *, x=None, y=None, hue=None, height=6, ratio=5, space=0.2, palette=None, hue_order=None, hue_norm=None, dropna=False, xlim=None, ylim=None, marginal_ticks=False)`

**使用例**:
```python
g3 = sns.PairGrid(iris, hue="species")
g3.map_diag(sns.histplot)
g3.map_offdiag(sns.scatterplot)
g3.add_legend()
g3.savefig("pairgrid.png")

g4 = sns.FacetGrid(tips, col="time", row="smoker")
g4.map(sns.scatterplot, "total_bill", "tip")
g4.savefig("facetgrid.png")

g5 = sns.JointGrid(data=tips, x="total_bill", y="tip")
g5.plot(sns.scatterplot, sns.histplot)
g5.savefig("jointgrid.png")
```
実行結果:
`PairGrid` の例は `pairplot(kind="scatter", diag_kind="hist")` とほぼ同等の散布図行列になる。`FacetGrid` の例は `smoker`(行)× `time`(列)の2×2パネルにそれぞれ散布図が描かれるファセット図になる。`JointGrid` の例は `jointplot` と同様、中央に散布図・上端右端にヒストグラムが配置された図になる。

**注意点・落とし穴**:
- `FacetGrid.map()` に渡す描画関数は、位置引数としてx(・y)の列名文字列を受け取れる関数である必要がある(`data=`引数を要求する関数は `map_dataframe()` を使う)。
- これら3クラスは `relplot`/`catplot`/`displot`/`pairplot`/`jointplot` の内部実装であり、既存の関数レベルAPIで対応できない独自の組み合わせをしたいときに直接使う。

---

## スタイル・テーマ・カラーパレット

### `sns.set_theme(...)`

**用途**: 全体の見た目(背景スタイル・配色パレット・フォントサイズ)を一括設定する。多くの場合スクリプト冒頭で一度だけ呼ぶ。

**シグネチャ**: `sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)`

**使用例**:
```python
sns.set_theme(style="whitegrid", palette="pastel")
```
実行結果:
以降に描画するすべての図の背景が白地+グレーの格子線(`whitegrid`)になり、カテゴリカルな配色がパステルカラーパレットに切り替わる(戻り値はなし、副作用としてmatplotlibのrcParamsを書き換える)。

**注意点・落とし穴**:
- 内部的にmatplotlibの `rcParams` をグローバルに書き換えるため、同一プロセス内の以降の全プロットに影響する(1つの図だけ変えたい場合は `with sns.axes_style(...):` のようにコンテキストマネージャとして使う)。

### `sns.set_style(...)` / `sns.axes_style(...)`

**用途**: 背景の見た目(`darkgrid`/`whitegrid`/`dark`/`white`/`ticks`)だけを設定する(`set_theme`のスタイル部分のみに相当)。`axes_style`は同じ引数でスタイル辞書を取得・一時適用するのに使う。

**シグネチャ**: `sns.set_style(style=None, rc=None)` / `sns.axes_style(style=None, rc=None)`

**使用例**:
```python
with sns.axes_style("darkgrid"):
    sns.boxplot(data=tips, x="day", y="total_bill")
```
実行結果:
`with` ブロック内でのみ `darkgrid` スタイル(グレー背景+白グリッド線)が適用され、ブロックを抜けると元のスタイルに戻る。`sns.axes_style("darkgrid")` はスタイル名からrcParams辞書(`figure.facecolor`等のキーを含む)を返す関数としても使える。

### `sns.set_context(...)` / `sns.plotting_context(...)`

**用途**: 図の用途(論文/発表/ポスターなど)に応じて、線の太さ・文字サイズなど全体のスケールを調整する。

**シグネチャ**: `sns.set_context(context=None, font_scale=1, rc=None)` / `sns.plotting_context(context=None, font_scale=1, rc=None)`

**使用例**:
```python
sns.set_context("talk", font_scale=1.1)
```
実行結果:
`"talk"`(プレゼン向けにやや大きめ)のコンテキストが適用され、以降の図の軸ラベル・目盛りフォントや線の太さが `"notebook"`(デフォルト)より大きくなる。指定できる文字列は `"paper"`/`"notebook"`/`"talk"`/`"poster"` の4段階。

### `sns.color_palette(...)`

**用途**: カラーパレット(色のリスト)を生成・取得する。seabornの配色に関わるほぼ全ての関数の基盤。

**シグネチャ**: `sns.color_palette(palette=None, n_colors=None, desat=None, as_cmap=False)`

**使用例**:
```python
pal = sns.color_palette("viridis", 8)
print(len(pal), pal[0])
```
実行結果:
```
8 (0.281412, 0.155834, 0.469201)
```
`matplotlib`のカラーマップ名 `"viridis"` から8色を均等に抜き出したRGBタプルのリスト(`seaborn.palettes._ColorPalette`、リストのように振る舞う)が得られる。Jupyter上では自動的に色見本として表示される。

**注意点・落とし穴**:
- 引数なしで呼ぶと現在アクティブな(`set_theme`/`set_palette`で設定した)デフォルトパレットが返る。

### `sns.cubehelix_palette(...)` / `sns.husl_palette(...)`

**用途**: 知覚的に均等な明度変化を持つ連続的なカラーパレットを生成する(ヒートマップやKDEの `fill` に向く)。

**シグネチャ**: `sns.cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0, hue=0.8, light=0.85, dark=0.15, reverse=False, as_cmap=False)` / `sns.husl_palette(n_colors=6, h=0.01, s=0.9, l=0.65, as_cmap=False)`

**使用例**:
```python
pal = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
print(pal.as_hex())
```
実行結果:
```
['#ecf7ee', '#b9dec1', '#8cc297', '#65a271', '#447f50', '#275831', '#112d16', '#000000']
```
明るい緑系から暗い緑系へなめらかに変化する8色のパレットが生成される(`.as_hex()` で16進カラーコードのリストに変換できる)。`husl_palette` は色相環上で均等に離れた色を選ぶため、カテゴリ数が多いカテゴリカル変数の色分けに向く。

### `sns.light_palette(...)` / `sns.dark_palette(...)` / `sns.diverging_palette(...)`

**用途**: 単色から明→暗(またはその逆)へのグラデーションパレット(`light_palette`/`dark_palette`)、および中心色から2方向に発散する2色系パレット(`diverging_palette`、相関行列など正負を色分けしたい場合向き)を生成する。

**シグネチャ**: `sns.light_palette(color, n_colors=6, reverse=False, as_cmap=False, input='rgb')` / `sns.dark_palette(color, n_colors=6, reverse=False, as_cmap=False, input='rgb')` / `sns.diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, n=6, center='light', as_cmap=False)`

**使用例**:
```python
pal_light = sns.light_palette("seagreen", n_colors=6)
pal_div = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(flights_wide.corr(), cmap=pal_div, center=0)
```
実行結果:
`light_palette("seagreen")` は白に近い薄緑から `"seagreen"` そのものへの6段階グラデーション。`diverging_palette(220, 20, as_cmap=True)` は青(色相220)↔赤(色相20)の発散カラーマップとして `heatmap` の `cmap=` に直接渡せる(`center=0` と組み合わせることで0を中心とした正負の可視化になる)。

**注意点・落とし穴**:
- `as_cmap=True` を指定すると戻り値がmatplotlibの `Colormap` オブジェクトになり、`heatmap`/`kdeplot` の `cmap=` にそのまま渡せる(`as_cmap=False`のリストは `cmap=` には渡せない)。

### `sns.despine(...)`

**用途**: 図の上・右(デフォルト)の枠線(スパイン)を取り除き、すっきりした見た目にする。

**シグネチャ**: `sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)`

**使用例**:
```python
sns.boxplot(data=tips, x="day", y="total_bill")
sns.despine(left=True)
plt.savefig("despine.png")
```
実行結果:
デフォルトで消える上・右の枠線に加えて、`left=True` により左の枠線(y軸の縦線)も消え、下と左目盛りのみが残ったすっきりした箱ひげ図になる。

**注意点・落とし穴**:
- 図やAxesを作成した**後**に呼ぶ必要がある(`ax=`/`fig=`省略時は「現在アクティブな」図・Axesすべてに適用される)。

### `sns.move_legend(...)`

**用途**: 描画後に凡例の位置・体裁をまとめて変更する(matplotlibの `legend()` を再構成するより簡潔)。

**シグネチャ**: `sns.move_legend(obj, loc, **kwargs)`

**使用例**:
```python
ax = sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.savefig("move_legend.png", bbox_inches="tight")
```
実行結果:
デフォルトでプロット内右上に重なって表示されていた `day` の凡例が、`bbox_to_anchor=(1, 1)` によりプロット枠の外(右側)へ移動して表示される。

**注意点・落とし穴**:
- `obj` には `Axes` だけでなく `FacetGrid`/`PairGrid` などのグリッドオブジェクトも渡せる(`relplot`等が返すグリッドの凡例位置調整によく使う)。
- 凡例を枠外に出す場合、`plt.savefig(..., bbox_inches="tight")` を付けないと凡例が画像の外に切れて保存されることがある。

---

## データセット・ユーティリティ

### `sns.load_dataset(...)` / `sns.get_dataset_names()`

**用途**: seaborn公式リポジトリで配布されているサンプルCSVデータセット(`tips`/`iris`/`flights`/`penguins`等)をDataFrameとして取得する(動作確認・チュートリアル用途)。

**シグネチャ**: `sns.load_dataset(name, cache=True, data_home=None, **kws)` / `sns.get_dataset_names()`

**使用例**:
```python
names = sns.get_dataset_names()
print(len(names), names[:5])

tips = sns.load_dataset("tips")
print(tips.shape)
```
実行結果:
```
22 ['anagrams', 'anscombe', 'attention', 'brain_networks', 'car_crashes']
(244, 7)
```
`get_dataset_names()` は利用可能な22個のデータセット名のリストを返す。`load_dataset("tips")` は244行7列のDataFrameを返す。

**注意点・落とし穴**:
- 実体はGitHub上のCSVをその都度(初回)ダウンロードする実装のため、**インターネット接続が必須**。オフライン環境や社内プロキシ配下では失敗する(`cache=True`がデフォルトなので、一度成功すればローカルにキャッシュされ2回目以降はオフラインでも動く)。
- 本番データ分析用のデータセットではなく、あくまでドキュメント・チュートリアル・本ドキュメントのような検証用途に限定すべき。

---

## 応用・発展

seaborn 0.13系で整備された、より発展的・ニッチなAPI群。`seaborn.objects`(新しいオブジェクト指向インターフェース、通例 `import seaborn.objects as so` としてインポートする)、既存グリッドクラスの高度な使い方、統計的な注釈の応用、カラーパレットを自作するための低レベルユーティリティを扱う。

### 新しいオブジェクト指向インターフェース(seaborn.objects)

#### `so.Plot(...)` / `.add(...)`

**用途**: `seaborn.objects`(略称 `so`)の中心となるクラス。`Plot(data, x=..., y=..., ...)` でデータとデフォルトの変数マッピングを指定し、`.add(mark)` で描画要素(`so.Dot`/`so.Line`/`so.Bar`等の「マーク」)を追加していく、レイヤーを積み重ねる形の宣言的インターフェース。`relplot`等の関数レベルAPIより低レベルだが、`Axes`ベースの`matplotlib`オブジェクトより高レベルに位置する。

**シグネチャ**: `so.Plot(*args, data=None, x=None, y=None, color=None, alpha=None, fill=None, marker=None, pointsize=None, stroke=None, linewidth=None, linestyle=None, fillcolor=None, fillalpha=None, edgewidth=None, edgestyle=None, edgecolor=None, edgealpha=None, text=None, halign=None, valign=None, offset=None, fontsize=None, xmin=None, xmax=None, ymin=None, ymax=None, group=None)`
`Plot.add(mark, *transforms, orient=None, legend=True, label=None, data=None, **variables)`

**使用例**:
```python
import seaborn as sns
import seaborn.objects as so

sns.set_theme()
tips = sns.load_dataset("tips")

p = so.Plot(tips, x="total_bill", y="tip", color="time").add(so.Dot())
p.save("plot_dot.png", bbox_inches="tight")
```
実行結果:
`scatterplot(hue="time")` とほぼ同じ、Lunch/Dinnerで2色に塗り分けられた散布図が描かれる。`so.Dot()` が「マーク」(matplotlibでいう`Artist`に相当)で、`color="time"` は`Plot`コンストラクタで指定した変数マッピングがそのまま`add`したマークに継承される。

**注意点・落とし穴**:
- `Plot`オブジェクトはJupyter上では自動的に描画されるが、スクリプトから画像として保存するには`.save(loc, **kwargs)`(内部で`figure.savefig`を呼ぶ)か`.plot()`で`so.Plot`を`Plotter`に変換してから`.figure`を扱う必要があり、`plt.savefig()`をそのまま使うことはできない。
- `.save()`は`bbox_inches="tight"`を自動では付与しない。`color=`等で凡例が生成される図をそのまま`.save("x.png")`すると、凡例が図の右端で見切れて保存されることを実際に確認した(`bbox_inches="tight"`を明示的に渡すと解消する)。

#### `Plot.facet(...)`

**用途**: `so.Plot`に`row`/`col`によるファセット分割(`FacetGrid`相当の小さい図の格子表示)を追加する。

**シグネチャ**: `Plot.facet(col=None, row=None, order=None, wrap=None)`

**使用例**:
```python
p2 = (
    so.Plot(tips, x="total_bill", y="tip")
    .facet(col="time", row="smoker")
    .add(so.Dot())
)
p2.save("plot_facet.png", bbox_inches="tight")
```
実行結果:
`smoker`(行)×`time`(列)の2×2パネルに分割された散布図が描かれ、各パネルの上部に`"Yes | Lunch"`のように行・列の値を`|`で連結したタイトルが自動で付く。`relplot(row=..., col=...)`とほぼ同じ結果だが、`Plot`はメソッドチェーンで組み立てる点が異なる。

#### `so.Agg()` / `so.Est()`(Stat)

**用途**: マークを追加する`.add()`にマークと一緒に渡す「Stat(統計変換)」オブジェクト。`so.Agg()`は集計関数(平均など)を適用し、`so.Est()`はさらにブートストラップ等による誤差区間も計算する(`barplot`/`pointplot`が内部で行っている処理に相当)。

**シグネチャ**: `so.Agg(func='mean')` / `so.Est(func='mean', errorbar=('ci', 95), n_boot=1000, seed=None)`

**使用例**:
```python
flights = sns.load_dataset("flights")
p3 = so.Plot(flights, x="year", y="passengers").add(so.Line(), so.Agg())
p3.save("plot_agg.png", bbox_inches="tight")

p3b = (
    so.Plot(tips, x="day", y="total_bill", color="sex")
    .add(so.Dot(), so.Jitter())
    .add(so.Range(), so.Est(errorbar="sd"))
)
p3b.save("plot_est.png", bbox_inches="tight")
```
実行結果:
`p3`は年ごとの`passengers`の平均値(月別の重複を`so.Agg()`が平均に集約)を結んだ右肩上がりの1本の折れ線になる。`p3b`は曜日ごとに男女別の点群(ジッターあり)の上に、`so.Est(errorbar="sd")`による標準偏差の範囲を示す縦線(`so.Range`)が重ねて描かれる。

**注意点・落とし穴**:
- `so.Agg`/`so.Est`はあくまで「Stat」であり、それ単体では描画されない。`.add(mark, stat)`のように必ずマークとセットで`.add()`に渡す必要がある。

#### `so.Dodge()` / `so.Stack()`(Move)

**用途**: `.add()`にマーク・Statと一緒に渡す「Move(位置調整)」オブジェクト。`so.Dodge()`は重なる要素を左右にずらして並べ(`barplot`の`dodge`相当)、`so.Stack()`は積み上げる(棒グラフを積み上げ式にする)。

**シグネチャ**: `so.Dodge(empty='keep', gap=0, by=None)` / `so.Stack()`

**使用例**:
```python
p4 = (
    so.Plot(tips, x="day", y="total_bill", color="sex")
    .add(so.Bar(), so.Agg(), so.Dodge())
)
p4.save("plot_dodge.png", bbox_inches="tight")

p4b = (
    so.Plot(tips, x="day", color="smoker")
    .add(so.Bar(), so.Count(), so.Stack())
)
p4b.save("plot_stack.png", bbox_inches="tight")
```
実行結果:
`p4`は曜日ごとに男女別の平均`total_bill`の棒が左右に並んで(dodge)描かれる。`p4b`は曜日ごとの来店件数(`so.Count()`)が`smoker`の有無で下から積み上げられた棒グラフになる。

**注意点・落とし穴**:
- `.add()`に渡すTransform(Stat/Move)は左から右へ順番に適用されるパイプラインなので、`so.Dodge()`を先に書くか後に書くかで結果が変わりうる(集計してから並べ替えるのが基本)。

#### `Plot.pair(...)`

**用途**: 複数のx変数・y変数の組み合わせについて、`pairplot`のようなサブプロット群を`so.Plot`のメソッドチェーンで組み立てる。

**シグネチャ**: `Plot.pair(x=None, y=None, wrap=None, cross=True)`

**使用例**:
```python
penguins = sns.load_dataset("penguins")
p5 = (
    so.Plot(penguins, color="species")
    .pair(x=["bill_length_mm", "bill_depth_mm"], y=["flipper_length_mm"])
    .add(so.Dot())
)
p5.save("plot_pair.png", bbox_inches="tight")
```
実行結果:
`bill_length_mm`/`bill_depth_mm`(x軸候補)× `flipper_length_mm`(y軸)の組み合わせで横に2枚並んだ散布図が描かれ、それぞれ`species`ごとに色分けされる。`pairplot`と違い、x側とy側の変数リストを個別に指定できるため全組み合わせ(正方行列)にならない図を作れる。

---

### FacetGridの高度なカスタマイズ

#### `FacetGrid.map_dataframe(...)`

**用途**: `FacetGrid.map()`は列名を位置引数(ベクトル)として受け取る関数しか使えないが、`map_dataframe()`はファセットごとの**部分DataFrameそのもの**を`data=`引数として渡す。行相関やサンプル数など、複数列にまたがる統計量をファセットごとに計算して注釈したい場合に使う。

**シグネチャ**: `FacetGrid.map_dataframe(func, *args, **kwargs)`

**使用例**:
```python
import matplotlib.pyplot as plt

def annotate_corr(data, **kwargs):
    r = data["total_bill"].corr(data["tip"])
    plt.gca().text(0.05, 0.9, f"r={r:.2f}", transform=plt.gca().transAxes)

g6 = sns.FacetGrid(tips, col="time")
g6.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g6.map_dataframe(annotate_corr)
g6.savefig("map_dataframe.png")
```
実行結果:
Lunch/Dinnerの2パネルに散布図が描かれたうえで、各パネル左上に`annotate_corr`が計算した`total_bill`と`tip`の相関係数(`r=0.81`/`r=0.63`)がテキストとして注釈される。`annotate_corr`は`data`(そのファセットの部分DataFrame全体)を受け取れるため、単一列のベクトルだけでは計算できない列間の統計量を扱える。

**注意点・落とし穴**:
- `map_dataframe`に渡す関数は`data`という名前のキーワード引数を受け取れる必要がある(`sns.scatterplot`のように`data=`対応の関数はそのまま渡せるが、独自関数を書く場合は`def f(data, **kwargs):`のシグネチャにする)。

#### `FacetGrid.refline(...)`

**用途**: 各ファセットに水平線・垂直線の参照線(基準線)をまとめて引く。全体平均や閾値をすべてのパネルに一括で重ねたいときに便利。

**シグネチャ**: `FacetGrid.refline(*, x=None, y=None, color='.5', linestyle='--', **line_kws)`

**使用例**:
```python
g7 = sns.FacetGrid(tips, col="day", col_wrap=2)
g7.map_dataframe(sns.histplot, x="total_bill")
g7.refline(x=tips["total_bill"].mean(), color="red", linestyle="--")
g7.savefig("refline.png")
```
実行結果:
曜日ごと(2×2に折り返し)のヒストグラムそれぞれに、`total_bill`の全体平均位置を示す赤い破線の垂直線が共通して重ねて描かれ、各曜日の分布が全体平均よりどちらに偏っているかが一目でわかる。

#### `FacetGrid.set_titles(...)` / `set_axis_labels(...)` / `tight_layout(...)`

**用途**: ファセットごとのタイトル文字列のテンプレートを変更したり(`set_titles`)、全パネル共通の軸ラベルをまとめて設定したり(`set_axis_labels`)、パネル間の余白を自動調整したり(`tight_layout`)する、仕上げ用のメソッド群。

**シグネチャ**: `FacetGrid.set_titles(template=None, row_template=None, col_template=None, **kwargs)`

**使用例**:
```python
g8 = sns.FacetGrid(tips, col="time", row="smoker")
g8.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g8.set_titles(row_template="smoker={row_name}", col_template="{col_name}")
g8.set_axis_labels("total bill (USD)", "tip (USD)")
g8.tight_layout()
g8.savefig("set_titles.png")
```
実行結果:
デフォルトの`"smoker = Yes | time = Lunch"`のような冗長なタイトルが、`row_template`/`col_template`の指定により`"smoker=Yes | Lunch"`という簡潔な表記に変わり、x軸・y軸ラベルも全パネル共通で`"total bill (USD)"`/`"tip (USD)"`に置き換わる。

---

### 統計的注釈の応用

#### `so.PolyFit()`

**用途**: `so.Plot`の`.add()`に渡すStatの1つ。散布データに指定次数の多項式回帰を当てはめた曲線を描く(`regplot(order=2)`相当の処理を`objects`インターフェースで行う)。

**シグネチャ**: `so.PolyFit(order=2, gridsize=100)`

**使用例**:
```python
healthexp = sns.load_dataset("healthexp")
p9 = (
    so.Plot(healthexp, x="Year", y="Life_Expectancy", color="Country")
    .add(so.Dots())
    .add(so.Line(), so.PolyFit(order=2))
)
p9.save("polyfit.png", bbox_inches="tight")
```
実行結果:
国別(5カ国)に色分けされた「年×平均寿命」の実測点(`so.Dots()`)の上に、国ごとに2次多項式でフィットした滑らかな曲線が重ねて描かれ、どの国も右肩上がりだが2000年代以降に伸びが鈍化する曲線カーブの違いが比較できる。

#### `so.Range()` + `so.Est()`

**用途**: 集計した代表値(`so.Agg`/`so.Dot`等)に、`so.Est()`が計算した誤差区間を`so.Range()`という専用マークで別レイヤーとして重ねる。`barplot`のエラーバーを`objects`インターフェースで自前に組み立てる形。

**シグネチャ**: `so.Range(color='C0', alpha=1, linewidth=<rc:lines.linewidth>, linestyle=<rc:lines.linestyle>)`

**使用例**:
```python
p10 = (
    so.Plot(tips, x="day", y="total_bill", color="sex")
    .add(so.Dot(), so.Agg(), so.Dodge())
    .add(so.Range(), so.Est(errorbar="ci"), so.Dodge())
)
p10.save("range_est.png", bbox_inches="tight")
```
実行結果:
曜日×性別ごとに平均`total_bill`を示す点(`so.Dot`)と、その95%信頼区間を示す縦線(`so.Range`)が同じ位置に重なって描かれる。点用と誤差区間用の2つの`.add()`にそれぞれ`so.Dodge()`を付ける必要があり、片方にだけ付けると点と誤差線の横位置がずれる。

**注意点・落とし穴**:
- 点(`so.Dot`+`so.Agg`)と誤差区間(`so.Range`+`so.Est`)は別々の`.add()`呼び出しであるため、`so.Dodge()`のような位置調整(Move)も**両方に**個別に指定しないと、dodgeされた点の真下に誤差線が来ず横方向にずれる。

#### `so.Text()`

**用途**: 集計値などをマーク(棒・点)の上にテキストとして直接注記する。`barplot`に値ラベルを付けたい場合などに使う。

**シグネチャ**: `so.Text(text='', color='k', alpha=1, fontsize=<rc:font.size>, halign='center', valign='center_baseline', offset=4)`

**使用例**:
```python
agg = tips.groupby("day", observed=True)["total_bill"].mean().round(1).reset_index()
p11 = (
    so.Plot(agg, x="day", y="total_bill", text="total_bill")
    .add(so.Bar())
    .add(so.Text(color="w", valign="top"))
)
p11.save("text.png", bbox_inches="tight")
```
実行結果:
曜日ごとの平均`total_bill`を表す棒グラフの上部(`valign="top"`)に、白色で`17.7`/`17.2`/`20.4`/`21.4`という集計値そのものがテキストラベルとして各棒に1つずつ表示される。

**注意点・落とし穴**:
- 先に`tips`を直接`so.Plot(tips, ...).add(so.Bar(), so.Agg()).add(so.Text(...), so.Agg(), text="total_bill")`のように**集計前の生データ**に`so.Agg()`を付けて試したところ、棒の高さ(y)は正しく集計されるのに、テキストラベルは集計されず元の全行の`total_bill`の値がすべて同じx位置に重ねて描画され、文字が大量に重なり判読不能になることを実際に確認した。`so.Text`に渡す`text=`変数は`so.Agg`等のStatによる集計の対象にならないため、テキストで見せたい値は`groupby`等で**事前に集計したDataFrーム**を`so.Plot`に渡すのが安全。

#### `so.Area()` + `so.KDE()`

**用途**: `so.Area()`マークと`so.KDE()`(Stat)を組み合わせて、`kdeplot(fill=True)`相当の塗りつぶしカーネル密度推定を`objects`インターフェースで描く。

**シグネチャ**: `so.Area(color='C0', alpha=0.2, fill=True, edgecolor=<depend:color>, edgealpha=1, edgewidth=<rc:patch.linewidth>, edgestyle='-', baseline=0)` / `so.KDE(bw_adjust=1, bw_method='scott', common_norm=True, common_grid=True, gridsize=200, cut=3, cumulative=False)`

**使用例**:
```python
p12 = so.Plot(tips, x="total_bill", color="time").add(so.Area(), so.KDE())
p12.save("area_kde.png", bbox_inches="tight")
```
実行結果:
`kdeplot(data=tips, x="total_bill", hue="time", fill=True)`とほぼ同じ、Lunch/Dinner2色の塗りつぶしKDE曲線が重なって描かれる。`common_norm=True`(デフォルト)のため、サンプル数の多いDinnerの山がLunchより高く描かれる。

---

### カラーパレットの高度な作成

#### `sns.blend_palette(...)`

**用途**: 任意の色のリストを指定し、それらを滑らかに補間したグラデーションパレット(または連続カラーマップ)を作る。2色発散パレット(`diverging_palette`)や単色グラデーション(`light_palette`)では表現できない、3色以上の任意配色のカラーマップを自作したいときに使う。

**シグネチャ**: `sns.blend_palette(colors, n_colors=6, as_cmap=False, input='rgb')`

**使用例**:
```python
pal = sns.blend_palette(["midnightblue", "gold", "crimson"], n_colors=9)
print(pal.as_hex())

cmap = sns.blend_palette(["midnightblue", "gold", "crimson"], as_cmap=True)
flights_wide = sns.load_dataset("flights").pivot(index="month", columns="year", values="passengers")
sns.heatmap(flights_wide, cmap=cmap)
plt.savefig("blend_cmap_heatmap.png")
```
実行結果:
```
['#191970', '#534954', '#8c7838', '#c6a81c', '#ffd600', '#f6a50f', '#ed741e', '#e5432d', '#dc143c']
```
`blend_palette`は紺(`midnightblue`)→金(`gold`)→深紅(`crimson`)へ滑らかに変化する9色のパレットを返す。`as_cmap=True`にした`cmap`を`heatmap`の`cmap=`にそのまま渡すと、乗客数が少ないセルほど紺、多いセルほど赤みがかった配色のヒートマップになる。

**注意点・落とし穴**:
- `light_palette`/`dark_palette`/`diverging_palette`は色の指定が1〜2色に限定されるが、`blend_palette`は3色以上のリストを渡せる唯一の組み込みパレット生成関数で、ブランドカラーなど任意の配色を再現したい場合に選ぶ。

#### `sns.mpl_palette(...)` / `sns.xkcd_palette(...)`

**用途**: `mpl_palette`はmatplotlib組み込みのカラーマップ名(`"plasma"`等、`color_palette`にも同名指定が効くが引数構成がシンプル)からパレットを取得する。`xkcd_palette`はxkcdの色名調査に基づく954色の俗称カラー名(`"windows blue"`等)からパレットを作る。

**シグネチャ**: `sns.mpl_palette(name, n_colors=6, as_cmap=False)` / `sns.xkcd_palette(colors)`

**使用例**:
```python
pal_mpl = sns.mpl_palette("plasma", 8)
pal_xkcd = sns.xkcd_palette(["windows blue", "amber", "faded green", "dusty purple"])
print(pal_mpl.as_hex())
print(pal_xkcd.as_hex())
```
実行結果:
```
['#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26']
['#3778bf', '#feb308', '#7bb274', '#825f87']
```
`mpl_palette("plasma", 8)`は紫→ピンク→オレンジ→黄へ変化する8色。`xkcd_palette`は指定した4つの色名(青・琥珀色・淡緑・くすんだ紫)にそれぞれ対応する単色4色のパレットになる。

**注意点・落とし穴**:
- `xkcd_palette`に渡せる色名はxkcdの命名規則に従う必要があり、存在しない名前を渡すと`ValueError`になる(利用可能な全色名は`matplotlib.colors.XKCD_COLORS`で確認できる)。

#### `sns.desaturate(...)` / `sns.saturate(...)` / `sns.set_hls_values(...)`

**用途**: パレット全体ではなく、単一の色そのものを加工するユーティリティ。`desaturate`は彩度を指定割合まで下げ、`saturate`は彩度を最大(100%)にし、`set_hls_values`はHLS(色相・輝度・彩度)の任意の成分だけを書き換える。

**シグネチャ**: `sns.desaturate(color, prop)` / `sns.saturate(color)` / `sns.set_hls_values(color, h=None, l=None, s=None)`

**使用例**:
```python
base = "seagreen"
colors = [
    sns.desaturate(base, 0.3),
    base,
    sns.saturate(base),
    sns.set_hls_values(base, l=0.8),
]
print(colors)
sns.palplot(colors)
plt.savefig("desaturate.png")
```
実行結果:
```
[(0.308, 0.417, 0.356), 'seagreen', (0.0, 0.725, 0.320), (0.699, 0.901, 0.788)]
```
左から、元の`"seagreen"`より彩度を70%落としてくすませた灰緑、元の`seagreen`そのもの、彩度を100%まで上げた鮮やかな緑、輝度を0.8まで上げた淡いパステル緑の4色が横に並んで表示される。

**注意点・落とし穴**:
- いずれも1色ずつを変換する関数であり、`color_palette`のようなパレット(色のリスト)ではなく単一のRGBタプル/色名を受け取る。パレット全体に適用したい場合はリスト内包表記で1色ずつ回す必要がある。
