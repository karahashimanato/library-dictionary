# PyTorch vs JAX(flax/optax)比較 — 設計思想・自動微分・JIT・実装対応

対象: PyTorch 2.13.0+cpu と JAX 0.11.1(flax 0.12.9 / optax 0.2.8)。両方とも `/home/manaty/library-practicing/.venv` に同居する形でインストールされており、検証はすべて `/home/manaty/library-practicing/.venv/bin/python` 上で実行した。検証日: 2026-09-24。CPU: AMD Ryzen AI 7 PRO 350(16論理コア)、**GPUなし・CUDA対応jaxlib未インストール**(`jax.devices()` は `[CpuDevice(id=0)]` のみ)。

**このドキュメントの性質**: [pytorch/README.md](../pytorch/README.md) や [jax/README.md](../jax/README.md) がライブラリ単体のAPIリファレンス(関数1つずつのエントリー集)であるのに対し、本ドキュメントは**2つのフレームワークを横断した比較**であり、構成も判断ガイド+ベンチマーク+対応表という形式にしている。

**品質方針**: ここに書く速度比較の数値・コードの等価性の主張はすべて、本ドキュメント作成時に実際に上記venvでコードを実行して得た結果である。記憶からの推測は一切含まない。ベンチマークはCPU専用環境での結果であり、GPU/TPU上での挙動(特にtorch.compileのTritonバックエンドやXLAのGPUコンパイルの効果)は一切検証していない点に注意。

## 目次

1. [設計思想の違い:命令型(PyTorch)vs 関数型(JAX)](#1-設計思想の違い命令型pytorchvs-関数型jax)
2. [自動微分の書き方対応表](#2-自動微分の書き方対応表)
3. [JITコンパイルの効果の実測](#3-jitコンパイルの効果の実測)
4. [ニューラルネット構築の対応表](#4-ニューラルネット構築の対応表)
5. [どちらを使うべきかの判断基準](#5-どちらを使うべきかの判断基準)

---

## 1. 設計思想の違い:命令型(PyTorch)vs 関数型(JAX)

PyTorchの`nn.Module`はパラメータをオブジェクトの内部状態として保持し、`optimizer.step()`がその状態を**インプレースで書き換える**。一方JAXは関数型を徹底しており、パラメータは単なるPython pytree(dict/list/tupleのネスト)として明示的に受け渡しされ、`jax.Array`自体は不変(immutable)なので、更新のたびに**新しいパラメータの木を作って返す**。

### PyTorch: 状態を持つモデル

```python
import torch, torch.nn as nn

model = nn.Linear(1, 1)                       # パラメータはmodel.weight/model.biasとしてモデル内部に保持される
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad()
loss = loss_fn(model(X), y)
loss.backward()                                # 勾配は各パラメータの .grad に蓄積される(副作用)
optimizer.step()                               # model.weight / model.bias がインプレースで更新される
```

### JAX: 純粋関数+明示的なパラメータ渡し

```python
import jax, jax.numpy as jnp

params = {"w": jnp.array([[0.0]]), "b": jnp.array([0.0])}

def predict(params, x):
    return x @ params["w"] + params["b"]        # 副作用なし。paramsは引数として渡されるだけ

def loss_fn(params, x, y):
    return jnp.mean((predict(params, x) - y) ** 2)

grads = jax.grad(loss_fn)(params, X, y)          # 勾配は新しいpytreeとして返る(paramsは変更されない)
params = {k: params[k] - lr * grads[k] for k in params}  # 新しいparams木を作って束縛し直す
```

この違いは表面的なAPIの差ではなく、以降のすべての節(自動微分・JIT・NN構築)に波及する。PyTorchでは「モデルというオブジェクトに状態が閉じ込められている」ため、`model.train()`/`model.eval()`のようなモード切り替えや`.grad`のクリア(`zero_grad()`)が必要になる。JAXでは状態は常に呼び出し側が持ち歩く必要があるため、学習ループは「`params`, `opt_state`をタプル/dictとして引き回す」形になり、`jax.jit`や`jax.vmap`のような変換と非常に相性が良い(副作用がないため関数全体を安全に変換できる)反面、素朴に書くと`for`ループの外側でも自分で状態を管理するコードが増える。

検証コード: `01_linreg_torch.py` / `02_linreg_jax.py`(下記2節で使用したものと同じ)。

---

## 2. 自動微分の書き方対応表

同一問題(線形回帰: `y = 3x + 2 + noise`、64サンプル、200ステップ、学習率0.1のSGD)を両フレームワークで実装し、挙動を確認した。乱数の生成方法自体が異なる(`torch.manual_seed` vs `jax.random.PRNGKey`)ため、ノイズの実現値は一致せず損失の絶対値は完全一致しないが、**同じ学習率・同じステップ数で同様に収束すること**を確認できた。

| 概念 | PyTorch | JAX |
|---|---|---|
| パラメータの持ち方 | `model.parameters()`(`nn.Parameter`、モデル内部状態) | `params`という明示的なpytree(dict等) |
| 勾配の計算 | `loss.backward()`(逆伝播、`.grad`に副作用として蓄積) | `jax.grad(loss_fn)(params, ...)`(勾配を新しい値として返す) |
| 値と勾配を同時に欲しい場合 | `loss.backward()`後に`loss.item()`を別途読む | `jax.value_and_grad(loss_fn)(params, ...)`で1回のトレースで両方取得 |
| 勾配のクリア | `optimizer.zero_grad()`が毎ステップ必須(`.grad`は蓄積されるため) | 不要(`jax.grad`は毎回新しい勾配を返すだけで蓄積されない) |
| パラメータ更新 | `optimizer.step()`(インプレース) | `params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)`のような明示的な木の変換 |
| 勾配停止 | `with torch.no_grad():` / `tensor.detach()` | `jax.lax.stop_gradient(x)` |

### 実測損失の推移

`01_linreg_torch.py` / `02_linreg_jax.py` を実行した結果(loss@ステップ0, 50, 100, 150, 199):

| フレームワーク | step0 | step50 | step100 | step150 | step199 | 学習後の w, b(真値 w=3.0, b=2.0) |
|---|---|---|---|---|---|---|
| PyTorch (`nn.Linear`+SGD) | 6.634965 | 0.004317 | 0.002637 | 0.002636 | 0.002636 | w=3.0152, b=2.0046 |
| JAX (pure func+`jax.grad`) | 7.122336 | 0.004548 | 0.002062 | 0.002060 | 0.002060 | w=3.0005, b=2.0060 |

両者とも50ステップ程度でほぼ収束し、200ステップ後には真値(w=3.0, b=2.0)にごく近い値に到達している。初期損失・最終損失の絶対値が異なるのはノイズの実現値がフレームワーク間で異なるためであり(同じ乱数アルゴリズムではない)、収束の「形」自体は同一である。

---

## 3. JITコンパイルの効果の実測

`jax.jit`と`torch.compile`はどちらも「計算グラフを一度コンパイルしてから繰り返し実行することで高速化する」という点で似た役割を持つが、実測すると**CPU環境ではその効果はモデルサイズに強く依存し、必ずしも劇的な高速化にはならない**ことが分かった。以下は3層MLPのforward+MSE lossを1000回(小モデル)/200回(大モデル)呼び出したときの合計時間。いずれも「ウォームアップ1回実行→破棄→2回目を計測」という手順で、初回コンパイルのオーバーヘッドを含まない定常状態の時間を測っている。初回コンパイル込みの時間は別途単独で計測した。

### 3.1 小モデル(64→256→256→1、バッチ256、1000回呼び出し)

検証コード: `03_jit_jax.py` / `04_compile_torch.py`

| | 初回呼び出し(コンパイル込み) | ウォームアップ後の1000回合計 | 1回あたり | eagerとの比 |
|---|---|---|---|---|
| JAX eager(jitなし) | - | 1.2861秒 | 1286.07 us/call | 1.00x(基準) |
| JAX + `jax.jit` | 0.1371秒 | 0.8492秒 | 849.21 us/call | **1.51x高速** |
| PyTorch eager | - | 0.4699秒 | 469.92 us/call | 1.00x(基準) |
| PyTorch + `torch.compile` | 6.2837秒 | 0.4704秒 | 470.45 us/call | **1.00x(速度差なし)** |

このサイズでは`torch.compile`は定常状態でも全く速くなっていない(むしろ誤差範囲でわずかに遅い)。加えて初回コンパイルに6.28秒もかかっており、この規模の計算をこの程度の呼び出し回数しか行わないなら`torch.compile`のコストは回収できない。一方`jax.jit`は初回コンパイルが0.14秒と軽く、定常状態でも1.51倍速くなっている。これは、JAXのeagerモードが「Python側のディスパッチ+XLA呼び出し」のオーバーヘッドを毎回払っているのに対し、`jit`はそのオーバーヘッドをコンパイル時に大きく削減できるため、かつPyTorchのeager実行(Cレベルの最適化された実装が使われる)はそもそも同規模の処理に対してすでに十分速く、`torch.compile`(TorchDynamo+Inductor)によるカーネル融合の恩恵が小さいためと考えられる。

### 3.2 大モデル(256→1024→1024→1024→1、バッチ1024、200回呼び出し)

検証コード: `05_jit_jax_large.py` / `06_compile_torch_large.py`

| | 初回呼び出し(コンパイル込み) | ウォームアップ後の200回合計 | 1回あたり | eagerとの比 |
|---|---|---|---|---|
| JAX eager(jitなし) | - | 2.8366秒 | 14182.94 us/call | 1.00x(基準) |
| JAX + `jax.jit` | 0.0844秒 | 2.4410秒 | 12204.84 us/call | **1.16x高速** |
| PyTorch eager | - | 19.6396秒 | 98198.17 us/call | 1.00x(基準) |
| PyTorch + `torch.compile` | 6.0143秒 | 17.2799秒 | 86399.47 us/call | **1.14x高速** |

計算量を増やす(行列サイズを4倍程度にする)と、`torch.compile`にも1.14倍の速度向上が実測できた。JAXの`jit`も同程度(1.16倍)。つまりこのCPU環境では、**JIT/コンパイルによる高速化効果はモデルが大きく計算のカーネル実行時間が支配的になるほど出やすく、小さいモデルではPythonディスパッチオーバーヘッドの削減効果が(特にPyTorchでは)相対的に小さい**という結果になった。なお`torch.compile`の初回コンパイル時間は常に`jax.jit`より1桁以上長い(6秒前後 vs 0.1秒前後)。これはTorchDynamoによるグラフキャプチャ+Inductorによるコード生成のコストが、JAXのXLA(CPUバックエンド)コンパイルより重いためと考えられるが、内部実装の詳細までは本検証では踏み込んでいない。

**注意**: この結果はCPU・この2つのモデルサイズに限った実測であり、一般に「JAXの方がJITが速い」「torch.compileは無意味」と結論づけるものではない。GPU上ではtorch.compileのTriton生成カーネルが大きな効果を発揮するケースが広く知られているが、本環境にはGPUがなく検証できていない。

---

## 4. ニューラルネット構築の対応表

同じアーキテクチャ(1→32→32→1のMLP、ReLU、非線形回帰 `y = sin(3x) + noise`)を`nn.Sequential`+`nn.Linear`+`torch.optim.Adam`と、`flax.linen.Dense`+`optax.adam`の双方で実装し、学習させた。検証コード: `07_nn_torch_adam.py` / `08_nn_flax_optax.py`。

### 4.1 API対応表

| 役割 | PyTorch | Flax/optax |
|---|---|---|
| 全結合層 | `nn.Linear(in_features, out_features)` | `flax.linen.Dense(features)`(入力次元は`.init()`時に自動推論) |
| 層の連結 | `nn.Sequential(layer1, layer2, ...)` | `@nn.compact`を付けた`__call__`内で逐次呼び出し(独自の`nn.Module`サブクラスを書く) |
| 活性化関数 | `nn.ReLU()`(層として)/`torch.relu`(関数として) | `flax.linen.relu`(関数のみ。層オブジェクトは無い) |
| パラメータ初期化 | `nn.Linear`のコンストラクタ内で自動実行(モデルが状態を持つ) | `params = model.init(key, dummy_input)`を明示的に呼ぶ(モデル自体は状態を持たない) |
| 順伝播の実行 | `model(x)`(パラメータはモデル内部から暗黙に参照) | `model.apply(params, x)`(パラメータを毎回明示的に渡す) |
| モデル構造の確認 | `print(model)` | `model.tabulate(key, x)`(パラメータ数・shapeまで含めた表を出力) |
| Optimizer | `torch.optim.Adam(model.parameters(), lr=0.01)` | `optax.adam(learning_rate=0.01)` |
| Optimizerの状態 | `optimizer`オブジェクト内部(モーメント等はモデルパラメータに紐づけて保持) | `opt_state = optimizer.init(params)`として明示的に保持 |
| 1ステップの更新 | `optimizer.zero_grad(); loss.backward(); optimizer.step()` | `grads = grad_fn(params, x, y); updates, opt_state = optimizer.update(grads, opt_state, params); params = optax.apply_updates(params, updates)` |

### 4.2 実装が一致していることの確認(パラメータ数)

`nn.Sequential(nn.Linear(1,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32,1))`と、`Dense(32)→relu→Dense(32)→relu→Dense(1)`のflax版は、どちらもパラメータ総数**1,153個**で一致した(実行結果より。内訳: 層1が1×32+32=64、層2が32×32+32=1,056、層3が32×1+1=33、合計1,153)。これはPyTorchの`sum(p.numel() for p in model.parameters())`とFlaxの`model.tabulate()`(下記出力の`Total Parameters: 1,153`)の両方で確認した。

`model.tabulate()`の実行結果(抜粋):
```
                                  MLP Summary
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ path    ┃ module ┃ inputs          ┃ outputs         ┃ params                ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Dense_0 │ Dense  │ float32[128,1]  │ float32[128,32] │ bias: float32[32]     │
│         │        │                 │                 │ kernel: float32[1,32] │
│         │        │                 │                 │ 64 (256 B)            │
├─────────┼────────┼─────────────────┼─────────────────┼───────────────────────┤
│ Dense_1 │ Dense  │ float32[128,32] │ float32[128,32] │ 1,056 (4.2 KB)        │
├─────────┼────────┼─────────────────┼─────────────────┼───────────────────────┤
│ Dense_2 │ Dense  │ float32[128,32] │ float32[128,1]  │ 33 (132 B)            │
└─────────┴────────┴─────────────────┴─────────────────┴───────────────────────┘
                        Total Parameters: 1,153 (4.6 KB)
```

### 4.3 Adamのデフォルトハイパーパラメータの比較

`inspect.signature()`で両者のデフォルト値を確認したところ、`torch.optim.Adam`と`optax.adam`は`betas`/`(b1, b2)` = `(0.9, 0.999)`、`eps` = `1e-8`とデフォルト値が完全に一致していた(数式上も同じAdamアルゴリズム)。相違点は、`torch.optim.Adam`は`lr`のデフォルト値`0.001`を持つのに対し、`optax.adam`は`learning_rate`が必須の位置引数でデフォルト値を持たない点。

### 4.4 学習の挙動(300ステップ、lr=0.01)

| フレームワーク | loss@step0 | step100 | step200 | step299 |
|---|---|---|---|---|
| PyTorch(`nn.Sequential`+Adam) | 0.464314 | 0.018398 | 0.004833 | 0.003314 |
| Flax(`flax.linen`+optax.adam) | 0.563060 | 0.089952 | 0.049289 | 0.009609 |

両者とも単調に損失が減少し学習が進んでいることを確認した。初期値・収束速度に差があるのは、重み初期化の方式(PyTorchの`nn.Linear`はKaiming-uniform系、Flaxの`Dense`はデフォルトでLeCun normal系)と乱数の実現値が異なるためであり、Adamの実装そのものの違いによるものではない(4.3節のとおりハイパーパラメータのデフォルト値自体は一致している)。

---

## 5. どちらを使うべきかの判断基準

以下は本検証で確認できた事実に基づく所見と、一般に言われている傾向(本検証では検証していない一般論であることを明記する)を分けて述べる。

### 本検証で確認できたこと(実測に基づく)

- **コードの記述量**: 同じ「線形回帰1本」でも、JAXは`params`辞書の初期化・`grad_fn`・パラメータ更新の3行を自分で書く必要があり、PyTorchは`nn.Linear`+`optim.SGD`+`backward()`/`step()`で完結する。同じ非線形回帰でも、Flax版はモデルクラス定義後に`model.init()`と`optax`の`opt_state`管理を明示的に書く必要があり、行数はPyTorch版より多くなった(`wc -l`実測: `07_nn_torch_adam.py`が35行、`08_nn_flax_optax.py`が52行。コメント・print文・空行を含む)。定型的な学習ループを素早く書きたいだけなら、この検証範囲ではPyTorchの方が記述コストが低い。
- **JIT/コンパイルの効果**: 3節の実測のとおり、CPU環境では計算量が小さいうちは`jax.jit`の方が速度向上・コンパイルコストの両面で有利だった(1.51倍高速・コンパイル0.14秒)のに対し、`torch.compile`は同条件で速度向上なし・コンパイル6.28秒だった。計算量を増やすと両者とも1.1〜1.2倍程度の同等な高速化が得られた。「小さい関数を大量に呼ぶ」ような使い方(例えば多数の独立したシミュレーションやMCMCのような用途)では、この検証結果から見る限りJAXの`jit`の方がオーバーヘッドが小さく有利な可能性がある。
- **パラメータの取り回し**: JAXは`params`が単なるpytreeなので、`jax.vmap`でバッチ化したり、複数モデルのパラメータをまとめて`jax.tree_util.tree_map`で操作したりする処理を書きやすい(1節参照)。PyTorchは`nn.Module`が状態を持つため、そうした操作をしようとすると`torch.func`(旧`functorch`)のような別APIが必要になる(本検証では未検証)。

### 一般に言われている傾向(本検証では未検証・参考情報)

- 研究目的でカスタムの微分可能アルゴリズム(独自の最適化手法、物理シミュレーション、確率的プログラミングなど)を書く場合、JAXの関数型・pure functionの設計が`grad`/`vmap`/`jit`を自由に組み合わせやすく好まれることが多いと言われる。
- プロダクション用途や、既存のモデル・学習済み重み・周辺ツール(torchvision、Hugging Face Transformers、各種デプロイツールなど)のエコシステムの広さを重視する場合はPyTorchが選ばれることが多いと言われる。
- これらはこの環境で実際に確かめた事実ではなく、一般的に言われている傾向として参考までに記載するものであり、断定はしない。実際の選択は、対象タスクでの記述コストとJIT/コンパイルの効果(本ドキュメントの2〜4節のように実測して確かめること)を踏まえて判断するのが望ましい。

### まとめ表

| 観点 | 本検証での実測結果 |
|---|---|
| 定型的な学習ループの記述量 | PyTorchの方が少ない行数で書けた(同一問題で比較) |
| 小さい計算のJIT効果(CPU) | `jax.jit`: 1.51倍高速・コンパイル0.14秒 / `torch.compile`: 速度向上なし・コンパイル6.28秒 |
| 大きい計算のJIT効果(CPU) | `jax.jit`: 1.16倍高速 / `torch.compile`: 1.14倍高速(ほぼ同等) |
| Adamのアルゴリズム自体の一致度 | デフォルトハイパーパラメータ(betas/eps)は完全一致。学習の収束の「形」も一致 |
| パラメータの操作性(vmap/tree_map的な処理) | JAXの方がpytreeベースで統一的に扱いやすい(設計上の帰結、1節参照) |

---

## 付録: 検証に使用したスクリプト一覧

本ドキュメントの数値・出力はすべて以下のスクリプトを `/home/manaty/library-practicing/.venv/bin/python` で実行して得たもの(スクリプト自体はこのドキュメントの検証用に作成した一時ファイルであり、リポジトリには含めていない)。

| ファイル | 内容 | 対応節 |
|---|---|---|
| `01_linreg_torch.py` | 線形回帰(PyTorch, `nn.Linear`+SGD) | 1, 2 |
| `02_linreg_jax.py` | 線形回帰(JAX, pure func+`jax.grad`) | 1, 2 |
| `03_jit_jax.py` | 小MLPでの`jax.jit`速度比較 | 3.1 |
| `04_compile_torch.py` | 小MLPでの`torch.compile`速度比較 | 3.1 |
| `05_jit_jax_large.py` | 大MLPでの`jax.jit`速度比較 | 3.2 |
| `06_compile_torch_large.py` | 大MLPでの`torch.compile`速度比較 | 3.2 |
| `07_nn_torch_adam.py` | MLP回帰(PyTorch, `nn.Sequential`+Adam) | 4 |
| `08_nn_flax_optax.py` | MLP回帰(Flax, `flax.linen`+optax.adam) | 4 |
