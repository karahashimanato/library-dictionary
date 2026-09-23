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
