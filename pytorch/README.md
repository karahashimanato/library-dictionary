# PyTorch 逆引き辞書

PyTorch 2.13.0 (CPU) で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(torch 2.13.0+cpu, CPUのみ)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。C拡張で実装された関数(`torch.tensor`や`torch.zeros`など)は`inspect.signature()`が使えないため、インストール済みの`__doc__`(先頭のシグネチャ行)と実行結果の両方で確認している。

## 目次

1. [テンソル生成・基礎操作](#1-テンソル生成基礎操作)
2. [形状操作](#2-形状操作)
3. [インデックス・スライシング](#3-インデックススライシング)
4. [演算・ブロードキャスト](#4-演算ブロードキャスト)
5. [自動微分(autograd)](#5-自動微分autograd)
6. [ニューラルネット構築](#6-ニューラルネット構築)
7. [損失関数](#7-損失関数)
8. [最適化](#8-最適化)
9. [データローディング](#9-データローディング)
10. [デバイス・型変換](#10-デバイス型変換)
11. [保存・読み込み](#11-保存読み込み)

---

## 1. テンソル生成・基礎操作

### `torch.tensor(...)`

**用途**: Pythonのリストやスカラーからテンソルを作成する(最も基本的なテンソル生成方法)。

**シグネチャ**: `torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor`(`__doc__`の先頭行より。C拡張実装のため`inspect.signature()`は使えない)

**使用例**:
```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
print(x)
print(x.dtype, x.shape)

xf = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(xf, xf.dtype)
```
実行結果:
```
tensor([[1, 2],
        [3, 4]])
torch.int64 torch.Size([2, 2])
tensor([1., 2., 3.]) torch.float32
```

**注意点・落とし穴**:
- 整数のPythonリストからは`torch.int64`、浮動小数のリストからは`torch.float32`が自動推論される。学習で使うテンソルは`dtype=torch.float32`を明示することが多い。
- `torch.tensor(data)`は常に`data`をコピーする。既存のテンソルを再利用したい場合は`.clone().detach()`や`torch.as_tensor()`を検討する。

### `torch.zeros(...)` / `torch.ones(...)`

**用途**: 指定した形状の、全要素が0または1のテンソルを作成する。

**シグネチャ**: `torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`(`ones`も同型)

**使用例**:
```python
print(torch.zeros(2, 3))
print(torch.ones(2, 3))
```
実行結果:
```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

**注意点・落とし穴**:
- デフォルトの`dtype`は`torch.float32`(整数を渡す`torch.arange`とは既定dtypeが異なる)。

### `torch.arange(...)`

**用途**: 等差数列のテンソルを作成する(Python標準の`range`のテンソル版)。

**シグネチャ**: `torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

**使用例**:
```python
print(torch.arange(0, 10, 2))
```
実行結果:
```
tensor([0, 2, 4, 6, 8])
```

**注意点・落とし穴**:
- `end`は含まれない(半開区間`[start, end)`)。整数のみを渡すと`dtype`は`torch.int64`になる。

### `torch.linspace(...)`

**用途**: 開始値から終了値まで(両端を含む)を等間隔に分割したテンソルを作成する。

**シグネチャ**: `torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

**使用例**:
```python
print(torch.linspace(0, 1, 5))
```
実行結果:
```
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

**注意点・落とし穴**:
- `arange`と違い`end`(ここでは`1`)を**含む**。`steps`は要素数そのもの(間隔の数ではない)。

### `torch.rand(...)` / `torch.randn(...)`

**用途**: 乱数テンソルを生成する。`rand`は`[0, 1)`の一様分布、`randn`は標準正規分布(平均0・分散1)。

**シグネチャ**: `torch.rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor`(`randn`も同型)

**使用例**:
```python
torch.manual_seed(0)
print(torch.rand(2, 2))
torch.manual_seed(0)
print(torch.randn(2, 2))
```
実行結果:
```
tensor([[0.4963, 0.7682],
        [0.0885, 0.1320]])
tensor([[ 1.5410, -0.2934],
        [-2.1788,  0.5684]])
```

**注意点・落とし穴**:
- `torch.manual_seed()`で毎回同じシード値から生成し始めれば再現可能になる(上の例は同じシード0から`rand`と`randn`をそれぞれ呼んでいるため、両者の値に直接の関係はない)。

### `torch.from_numpy(...)` / `.numpy()`

**用途**: NumPy配列とPyTorchテンソルを相互変換する。

**シグネチャ**: `torch.from_numpy(ndarray) -> Tensor`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
t = torch.from_numpy(a)
print(t)

t2 = torch.tensor([1, 2, 3])
print(t2.numpy())
```
実行結果:
```
tensor([1, 2, 3])
[1 2 3]
```

**注意点・落とし穴**:
- `from_numpy()`はメモリを共有する(コピーしない)。どちらか一方を書き換えると、もう片方にも影響する。
- `.numpy()`はCPU上のテンソルにしか使えない(GPU上のテンソルは先に`.cpu()`が必要)。また`requires_grad=True`のテンソルには使えず、`.detach().numpy()`が必要。

### `torch.manual_seed(...)`

**用途**: 乱数生成のシードを固定し、実行結果を再現可能にする。

**シグネチャ**: `torch.manual_seed(seed) -> Generator`(`__doc__`より: 「Sets the seed for generating random numbers on all devices.」)

**使用例**:
```python
torch.manual_seed(42)
print(torch.rand(2))
torch.manual_seed(42)
print(torch.rand(2))
```
実行結果:
```
tensor([0.8823, 0.9150])
tensor([0.8823, 0.9150])
```

**注意点・落とし穴**:
- CPU・GPU(CUDA)両方の乱数生成器のシードをまとめて設定する。CUDA専用に別シードを設定したい場合は`torch.cuda.manual_seed()`を使う。
- `DataLoader(shuffle=True)`のシャッフル順を固定したい場合は、`generator=torch.Generator().manual_seed(...)`をDataLoaderに渡す方が確実(9章参照)。

---

## 2. 形状操作

### `.view(...)` / `.reshape(...)`

**用途**: テンソルの形状(shape)を変更する。要素数は変わらない。

**シグネチャ**: `Tensor.view(*shape) -> Tensor` / `Tensor.reshape(*shape) -> Tensor`

**使用例**:
```python
x = torch.arange(6)
print(x.view(2, 3))
print(x.reshape(3, 2))

y = torch.arange(6).view(2, 3)
yt = y.t()  # 転置してメモリ上は非連続(non-contiguous)になる
try:
    print(yt.view(6))
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:80])
print(yt.reshape(6))
```
実行結果:
```
tensor([[0, 1, 2],
        [3, 4, 5]])
tensor([[0, 1],
        [2, 3],
        [4, 5]])
ERROR: RuntimeError view size is not compatible with input tensor's size and str
tensor([0, 3, 1, 4, 2, 5])
```

**注意点・落とし穴**:
- `.view()`は元のメモリを共有する(コピーしない)ため高速だが、メモリ上で連続(contiguous)でないテンソル(例: `.t()`や`.transpose()`後)には使えず`RuntimeError`になる。
- `.reshape()`は可能なら`.view()`と同様にメモリを共有し、連続でない場合は自動的にコピーして形状変更する。どちらを使うか迷ったら`.reshape()`の方が安全。

### `.squeeze(...)` / `.unsqueeze(...)`

**用途**: サイズ1の次元を削除(`squeeze`)、または指定位置に追加(`unsqueeze`)する。

**シグネチャ**: `torch.squeeze(input, dim=None) -> Tensor` / `torch.unsqueeze(input, dim) -> Tensor`

**使用例**:
```python
a = torch.zeros(1, 3, 1)
print(a.shape)
print(a.squeeze().shape)
print(a.squeeze(0).shape)

b = torch.zeros(3)
print(b.unsqueeze(0).shape, b.unsqueeze(1).shape)
```
実行結果:
```
torch.Size([1, 3, 1])
torch.Size([3])
torch.Size([3, 1])
torch.Size([1, 3]) torch.Size([3, 1])
```

**注意点・落とし穴**:
- `dim`を省略した`squeeze()`はサイズ1の次元を**すべて**削除する。バッチサイズがたまたま1の場合など、意図しない次元まで消えることがあるので、通常は`dim`を明示する方が安全。
- `unsqueeze`は`dim`が必須引数(省略不可)。

### `torch.cat(...)`

**用途**: 複数のテンソルを既存の次元に沿って連結する(次元数は変わらない)。

**シグネチャ**: `torch.cat(tensors, dim=0, *, out=None) -> Tensor`

**使用例**:
```python
p = torch.tensor([[1, 2], [3, 4]])
q = torch.tensor([[5, 6], [7, 8]])
print(torch.cat([p, q], dim=0))
print(torch.cat([p, q], dim=1))
```
実行結果:
```
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
```

**注意点・落とし穴**:
- 連結する次元(`dim`)以外のサイズは全テンソルで一致していないと`RuntimeError`になる。

### `torch.stack(...)`

**用途**: 複数のテンソルを**新しい次元**を作って積み重ねる(`cat`と違い次元数が1つ増える)。

**シグネチャ**: `torch.stack(tensors, dim=0, *, out=None) -> Tensor`

**使用例**:
```python
print(torch.stack([p, q], dim=0).shape)
```
実行結果:
```
torch.Size([2, 2, 2])
```

**注意点・落とし穴**:
- `stack`は全テンソルの形状が完全に一致している必要がある(`cat`のように一部次元だけ異なるのは不可)。

### `.permute(...)` / `.transpose(...)`

**用途**: 次元の順序を並べ替える。`transpose`は2つの次元のみ、`permute`は全次元を任意順に指定できる。

**シグネチャ**: `Tensor.permute(*dims) -> Tensor` / `torch.transpose(input, dim0, dim1) -> Tensor`

**使用例**:
```python
t = torch.arange(24).view(2, 3, 4)
print(t.shape)
print(t.permute(2, 0, 1).shape)
print(t.transpose(0, 1).shape)
```
実行結果:
```
torch.Size([2, 3, 4])
torch.Size([4, 2, 3])
torch.Size([3, 2, 4])
```

**注意点・落とし穴**:
- どちらもメモリを共有するビュー(view)を返し、結果は非連続(non-contiguous)になることが多い。その後`.view()`したい場合は先に`.contiguous()`が必要(前述の`RuntimeError`の例を参照)。

### `.flatten(...)`

**用途**: 指定範囲の次元をまとめて1次元に平坦化する。

**シグネチャ**: `Tensor.flatten(start_dim=0, end_dim=-1) -> Tensor`

**使用例**:
```python
t = torch.arange(24).view(2, 3, 4)
print(t.flatten().shape)
print(t.flatten(1).shape)
```
実行結果:
```
torch.Size([24])
torch.Size([2, 12])
```

**注意点・落とし穴**:
- `start_dim=1`とすると先頭(バッチ)次元は残したまま、それ以降だけ平坦化できる(CNNの出力を全結合層に渡す前によく使われる)。

### `.repeat(...)` / `.expand(...)`

**用途**: テンソルを繰り返して拡張する。`repeat`はデータを実際にコピーし、`expand`はメモリを共有した(コピーしない)ビューを作る。

**シグネチャ**: `Tensor.repeat(*repeats) -> Tensor` / `Tensor.expand(*size) -> Tensor`

**使用例**:
```python
r = torch.tensor([1, 2, 3])
print(r.repeat(2))
print(r.repeat(2, 1))

e = torch.tensor([[1], [2], [3]])
print(e.expand(3, 4))
```
実行結果:
```
tensor([1, 2, 3, 1, 2, 3])
tensor([[1, 2, 3],
        [1, 2, 3]])
tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]])
```

**注意点・落とし穴**:
- `expand`はサイズ1の次元しか拡張できず、拡張先のテンソルへの書き込みは(メモリ共有のため)全ての複製箇所に影響する。実データが必要なら`expand(...).clone()`や`repeat`を使う。

---

## 3. インデックス・スライシング

### 基本スライシング `[...]`

**用途**: NumPyと同様のインデックス指定・スライスで部分テンソルを取り出す。

**シグネチャ**: `Tensor.__getitem__(index)`(Python標準の`[]`構文)

**使用例**:
```python
m = torch.arange(12).view(3, 4)
print(m)
print(m[1])
print(m[:, 1])
print(m[1, 2])
print(m[0:2, 1:3])
```
実行結果:
```
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
tensor([4, 5, 6, 7])
tensor([1, 5, 9])
tensor(6)
tensor([[1, 2],
        [5, 6]])
```

**注意点・落とし穴**:
- 単純なスライス(`m[0:2]`など)はビュー(元のメモリを共有)を返すが、整数配列やブールマスクによる高度な(fancy)インデックスはコピーを返す。

### ブールマスクによるインデックス

**用途**: 条件を満たす要素だけを抽出する。

**シグネチャ**: `Tensor.__getitem__(mask: BoolTensor) -> Tensor`

**使用例**:
```python
v = torch.tensor([1, -2, 3, -4, 5])
mask = v > 0
print(mask)
print(v[mask])
```
実行結果:
```
tensor([ True, False,  True, False,  True])
tensor([1, 3, 5])
```

**注意点・落とし穴**:
- 結果は常に1次元になり、元の形状は失われる(2次元以上のテンソルにブールマスクを使っても平坦化された1次元テンソルが返る)。

### `torch.where(...)`

**用途**: 条件に応じて2つのテンソルから要素ごとに値を選択する(NumPyの`np.where`相当)。

**シグネチャ**: `torch.where(condition, input, other, *, out=None) -> Tensor`

**使用例**:
```python
print(torch.where(v > 0, v, torch.zeros_like(v)))
```
実行結果:
```
tensor([1, 0, 3, 0, 5])
```

**注意点・落とし穴**:
- `input`/`other`のどちらもテンソルである必要がある(片方だけスカラーにしたい場合は`torch.zeros_like(v)`のように同じ形状のテンソルを用意する)。

### `torch.gather(...)`

**用途**: 指定した次元に沿って、インデックステンソルで指定した位置の値を集める。

**シグネチャ**: `torch.gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor`

**使用例**:
```python
src = torch.tensor([[10, 20, 30], [40, 50, 60]])
idx = torch.tensor([[0, 0], [2, 1]])
print(torch.gather(src, 1, idx))
```
実行結果:
```
tensor([[10, 10],
        [60, 50]])
```

**注意点・落とし穴**:
- `index`の形状は、集める次元(`dim`)以外は`input`と一致している必要がある。分類問題で正解クラスの確率だけを取り出す(`gather(probs, 1, target.unsqueeze(1))`)用途でよく使われる。

### `torch.index_select(...)`

**用途**: 指定した次元に沿って、1次元のインデックステンソルで指定した行・列だけを抽出する。

**シグネチャ**: `torch.index_select(input, dim, index, *, out=None) -> Tensor`

**使用例**:
```python
print(torch.index_select(src, 0, torch.tensor([1, 0])))
print(torch.index_select(src, 1, torch.tensor([2, 0])))
```
実行結果:
```
tensor([[40, 50, 60],
        [10, 20, 30]])
tensor([[30, 10],
        [60, 40]])
```

**注意点・落とし穴**:
- `index`は必ず1次元のテンソルでなければならない(`gather`と違い、多次元のインデックス指定はできない)。同じ添字を重複指定すれば同じ行・列を複数回抽出できる。

---

## 4. 演算・ブロードキャスト

### ブロードキャスト演算(`+`, `*` など)

**用途**: 形状の異なるテンソル同士を、NumPyと同じルール(ブロードキャスト)で要素ごとに演算する。

**シグネチャ**: `Tensor.__add__(other)` など各種演算子オーバーロード

**使用例**:
```python
A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
Bv = torch.tensor([10., 20., 30.])
print(A + Bv)
print(A * 2)
```
実行結果:
```
tensor([[11., 22., 33.],
        [14., 25., 36.]])
tensor([[ 2.,  4.,  6.],
        [ 8., 10., 12.]])
```

**注意点・落とし穴**:
- ブロードキャストのルールはNumPyと同じで、末尾の次元から比較して「サイズが同じ」か「どちらかが1」であれば拡張される。形状が全く噛み合わないと`RuntimeError`になる。

### `torch.matmul(...)` / `@`

**用途**: 行列積(内積)を計算する。要素ごとの積(`*`)とは異なる。

**シグネチャ**: `torch.matmul(input, other, *, out=None) -> Tensor`

**使用例**:
```python
M1 = torch.tensor([[1., 2.], [3., 4.]])
M2 = torch.tensor([[5., 6.], [7., 8.]])
print(M1 @ M2)
print(torch.matmul(M1, M2))
```
実行結果:
```
tensor([[19., 22.],
        [43., 50.]])
tensor([[19., 22.],
        [43., 50.]])
```

**注意点・落とし穴**:
- `@`演算子は`torch.matmul`のシンタックスシュガー。要素ごとの積がほしい場合は`*`(`torch.mul`)を使う(`matmul`と混同しやすい)。
- 3次元以上のテンソル同士では、先頭次元をバッチとみなしたバッチ行列積として扱われる。

### `.sum(...)` / `.mean(...)` / `.max(...)`(`dim`引数)

**用途**: 合計・平均・最大値などを計算する。`dim`を指定するとその次元に沿って集約する。

**シグネチャ**: `torch.sum(input, *, dtype=None) -> Tensor`(全要素版)、`torch.sum(input, dim, keepdim=False, *, dtype=None) -> Tensor`(`dim`指定版。`mean`も同型)。`torch.max(input) -> Tensor` / `torch.max(input, dim, keepdim=False) -> (values, indices)`(`__doc__`内の複数シグネチャより)

**使用例**:
```python
X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(X.sum())
print(X.sum(dim=0))
print(X.sum(dim=1))
print(X.mean(dim=1))
print(X.max(dim=1))
```
実行結果:
```
tensor(21.)
tensor([5., 7., 9.])
tensor([ 6., 15.])
tensor([2., 5.])
torch.return_types.max(
values=tensor([3., 6.]),
indices=tensor([2, 2]))
```

**注意点・落とし穴**:
- `dim`を省略すると全要素のスカラーが返るが、`dim`を指定するとその次元が(デフォルトでは)潰れて次元数が1つ減る。次元数を保ちたい場合は`keepdim=True`を指定する。
- `dim`付きの`max`/`min`は値だけでなく**インデックス**も含む名前付きタプル(`torch.return_types.max`)を返す(NumPyの`np.max`とは戻り値の構造が異なる)。

### `torch.einsum(...)`

**用途**: アインシュタインの縮約記法で、内積・行列積・転置などを一つの記法で汎用的に表現する。

**シグネチャ**: `torch.einsum(equation, *operands) -> Tensor`

**使用例**:
```python
u = torch.tensor([1., 2., 3.])
w = torch.tensor([4., 5., 6.])
print(torch.einsum('i,i->', u, w))  # 内積

Mm = torch.arange(6.).view(2, 3)
Nn = torch.arange(12.).view(3, 4)
print(torch.einsum('ik,kj->ij', Mm, Nn))  # 行列積
```
実行結果:
```
tensor(32.)
tensor([[20., 23., 26., 29.],
        [56., 68., 80., 92.]])
```

**注意点・落とし穴**:
- `'ik,kj->ij'`は`torch.matmul(Mm, Nn)`と等価。複雑なテンソル演算を1行で書けるが、記法に慣れるまで可読性は低い。

### `torch.clamp(...)`

**用途**: テンソルの値を指定した範囲(`min`〜`max`)に切り詰める。

**シグネチャ**: `torch.clamp(input, min=None, max=None, *, out=None) -> Tensor`

**使用例**:
```python
c = torch.tensor([-2., 0.5, 3.0, 10.0])
print(torch.clamp(c, min=0.0, max=5.0))
```
実行結果:
```
tensor([0.0000, 0.5000, 3.0000, 5.0000])
```

**注意点・落とし穴**:
- `min`/`max`はどちらか片方だけ指定することもできる(例: `clamp(min=0.0)`はReLUと同じ挙動になる)。勾配クリッピング(`torch.nn.utils.clip_grad_norm_`)とは別物なので混同しないこと。

---

## 5. 自動微分(autograd)

### `requires_grad=True`

**用途**: テンソルに対する演算の計算グラフを記録し、後で微分(勾配計算)できるようにする。

**シグネチャ**: `torch.tensor(data, *, requires_grad=False, ...) -> Tensor` / `Tensor.requires_grad_(requires_grad=True) -> Tensor`

**使用例**:
```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(x)
y = (x ** 2).sum()
print(y)
```
実行結果:
```
tensor([2., 3.], requires_grad=True)
tensor(13., grad_fn=<SumBackward0>)
```

**注意点・落とし穴**:
- `requires_grad=True`のテンソルから生まれた計算結果には`grad_fn`(逆伝播用の関数)が自動的に付与される。整数型テンソルには`requires_grad=True`を設定できない(浮動小数点のみ)。

### `.backward()`

**用途**: スカラー値のテンソルを起点に誤差逆伝播を行い、計算グラフ上の各リーフテンソルの`.grad`に勾配を蓄積する。

**シグネチャ**: `Tensor.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None) -> None`

**使用例**:
```python
y.backward()
print(x.grad)
```
実行結果:
```
tensor([4., 6.])
```

**注意点・落とし穴**:
- **勾配は蓄積(加算)される**。同じパラメータで`backward()`を複数回呼ぶと`.grad`が加算され続けるため、学習ループでは毎回`optimizer.zero_grad()`(または`param.grad = None`)でリセットする必要がある(実際に検証: 2回目の`backward()`で`4.0`→`8.0`に増加し、`.zero_()`後に`4.0`に戻ることを確認済み)。
- デフォルトでは`backward()`を呼んだ後、計算グラフは解放される。同じグラフで再度`backward()`したい場合は`retain_graph=True`が必要。

### `torch.no_grad()`

**用途**: `with`ブロック内で計算グラフの記録を無効化する(推論時やパラメータ更新時に使う)。

**シグネチャ**: `torch.no_grad()`(コンテキストマネージャ)

**使用例**:
```python
with torch.no_grad():
    z = x * 2
    print(z.requires_grad)
print((x * 2).requires_grad)
```
実行結果:
```
False
True
```

**注意点・落とし穴**:
- ブロックを抜けると通常のグラフ記録に戻る。推論(評価)時にこれを使わないと、不要な計算グラフがメモリに溜まり続けてメモリ使用量が増える。

### `.detach()`

**用途**: 現在のテンソルと値(メモリ)は共有しつつ、計算グラフから切り離した新しいテンソルを作る。

**シグネチャ**: `Tensor.detach() -> Tensor`

**使用例**:
```python
w = x.detach()
print(w.requires_grad, w)
```
実行結果:
```
False tensor([2., 3.])
```

**注意点・落とし穴**:
- `.detach()`はメモリを共有するため、値を書き換えると元のテンソルにも影響する(独立したコピーが欲しい場合は`.clone().detach()`)。

### `torch.autograd.grad(...)`

**用途**: `.grad`に蓄積せず、指定した入力に対する勾配だけをその場で(タプルとして)受け取る。

**シグネチャ**: `torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, ...) -> Tuple[Tensor, ...]`

**使用例**:
```python
a = torch.tensor(2.0, requires_grad=True)
b = a ** 3
g = torch.autograd.grad(b, a)
print(g)
```
実行結果:
```
(tensor(12.),)
```
(`d(a^3)/da = 3a^2 = 3*4 = 12` で一致)

**注意点・落とし穴**:
- `.backward()`と違い`a.grad`は更新されず、戻り値のタプルとして勾配が返る。高階微分やメタ学習など、勾配自体を計算グラフに組み込みたい場合に使う。

---

## 6. ニューラルネット構築

### `nn.Module` を継承したモデル定義

**用途**: レイヤーとforward計算をまとめ、パラメータ管理・保存/読み込み・GPU転送などを一括で扱えるモデルクラスを定義する。

**シグネチャ**: `class MyModel(nn.Module): def __init__(self): super().__init__(); ... ; def forward(self, x): ...`

**使用例**:
```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

torch.manual_seed(0)
model = MLP()
inp = torch.randn(2, 4)
out = model(inp)
print(out.shape)
print(model)
```
実行結果:
```
torch.Size([2, 3])
MLP(
  (fc1): Linear(in_features=4, out_features=8, bias=True)
  (fc2): Linear(in_features=8, out_features=3, bias=True)
)
```

**注意点・落とし穴**:
- モデルの呼び出しは`model.forward(inp)`ではなく`model(inp)`(`__call__`)を使う。`__call__`はフックの実行なども行うため、直接`forward`を呼ぶと一部機能が動作しない。
- `__init__`内で`super().__init__()`を呼び忘れると、`self.fc1 = ...`などの代入時にエラーになる(`nn.Module`が内部でサブモジュール登録の仕組みを持つため)。

### `nn.Linear(...)`

**用途**: 全結合層(線形変換 `y = xW^T + b`)を作る。

**シグネチャ**: `nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`

**使用例**:
```python
torch.manual_seed(0)
lin = nn.Linear(3, 2)
print(lin.weight.shape, lin.bias.shape)
xin = torch.randn(5, 3)
print(lin(xin).shape)
```
実行結果:
```
torch.Size([2, 3]) torch.Size([2])
torch.Size([5, 2])
```

**注意点・落とし穴**:
- `weight`の形状は`(out_features, in_features)`であり、`(in_features, out_features)`ではない点に注意(数式上`y = x @ weight.T + bias`)。
- 入力の最後の次元が`in_features`と一致していれば、それ以前の次元(バッチサイズなど)は任意でよい。

### `nn.Sequential(...)`

**用途**: 複数の層を順番に並べて1つのモデルとして扱う(単純な直列構造のとき`nn.Module`を自作するより簡潔)。

**シグネチャ**: `nn.Sequential(*args)`

**使用例**:
```python
torch.manual_seed(0)
seq = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
print(seq(torch.randn(1, 4)).shape)
print(seq)
```
実行結果:
```
torch.Size([1, 2])
Sequential(
  (0): Linear(in_features=4, out_features=8, bias=True)
  (1): ReLU()
  (2): Linear(in_features=8, out_features=2, bias=True)
)
```

**注意点・落とし穴**:
- 各層は登録順にそのまま`forward`に渡されるだけなので、分岐や複数入力があるモデルには使えない(その場合は`nn.Module`を自作する)。
- `seq[0]`のようにインデックスでレイヤーにアクセスできる。

### `F.relu(...)` / `nn.ReLU()`

**用途**: ReLU活性化関数(`max(0, x)`)を適用する。`nn.functional`版は関数、`nn`版はモジュール(`nn.Sequential`内で使える)。

**シグネチャ**: `torch.nn.functional.relu(input, inplace=False) -> Tensor` / `nn.ReLU(inplace=False)`

**使用例**:
```python
r = torch.tensor([-1.0, 0.0, 2.0])
print(F.relu(r))
print(nn.ReLU()(r))
```
実行結果:
```
tensor([0., 0., 2.])
tensor([0., 0., 2.])
```

**注意点・落とし穴**:
- 両者は結果が同じで、`nn.ReLU()`は内部で`F.relu()`を呼ぶラッパー。`nn.Sequential`の中で使うには状態を持つモジュールである`nn.ReLU()`が必要、`forward`メソッド内で直接使うなら`F.relu()`で十分。

---

## 7. 損失関数

### `nn.CrossEntropyLoss(...)`

**用途**: 多クラス分類の損失関数。内部でSoftmax(正確には`log_softmax`)とNLL損失を組み合わせて計算する。

**シグネチャ**: `nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)`

**使用例**:
```python
torch.manual_seed(0)
logits = torch.randn(3, 5)  # (バッチ3, クラス5) の生スコア
target = torch.tensor([1, 0, 4])  # 各サンプルの正解クラス番号
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, target)
print(loss)
```
実行結果:
```
tensor(2.7184)
```

**注意点・落とし穴**:
- **入力にSoftmaxを適用してはいけない**(`CrossEntropyLoss`が内部で`log_softmax`を行うため、事前にSoftmaxを通すと二重適用になり学習がおかしくなる)。渡すのは正規化前の生スコア(logits)。
- `target`はone-hotベクトルではなく、クラス番号を表す整数テンソル(`torch.long`)。

### `nn.MSELoss(...)`

**用途**: 平均二乗誤差(回帰問題の損失関数)。

**シグネチャ**: `nn.MSELoss(size_average=None, reduce=None, reduction='mean')`

**使用例**:
```python
pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
true = torch.tensor([3.0, -0.5, 2.0, 7.0])
mse = nn.MSELoss()
print(mse(pred, true))
```
実行結果:
```
tensor(0.3750)
```

**注意点・落とし穴**:
- `reduction='mean'`がデフォルト(バッチ内の平均)。合計値がほしい場合は`reduction='sum'`を指定する。

### `F.softmax(...)`

**用途**: 生スコアを合計1の確率分布に変換する。

**シグネチャ**: `torch.nn.functional.softmax(input, dim=None, ...) -> Tensor`(`__doc__`より「Apply a softmax function.」)

**使用例**:
```python
s = torch.tensor([1.0, 2.0, 3.0])
print(F.softmax(s, dim=0))
```
実行結果:
```
tensor([0.0900, 0.2447, 0.6652])
```

**注意点・落とし穴**:
- `dim`を省略すると動作はするが、実際に`UserWarning`(`Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.`)が出ることを確認済み。必ず`dim`を明示する。
- 分類の損失計算に使う場合、前述の通り`CrossEntropyLoss`は内部でSoftmax相当の処理を行うため、学習時に`F.softmax`を別途適用する必要はない(推論時に確率を見たい場合にのみ使う)。

---

## 8. 最適化

### `torch.optim.SGD(...)`

**用途**: 確率的勾配降下法でモデルパラメータを更新するオプティマイザ。

**シグネチャ**: `torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None)`

**使用例**:
```python
torch.manual_seed(0)
lin2 = nn.Linear(2, 1)
opt = torch.optim.SGD(lin2.parameters(), lr=0.1)
x2 = torch.tensor([[1.0, 2.0]])
target2 = torch.tensor([[1.0]])
print("before:", lin2.weight.data.clone())
for step in range(3):
    opt.zero_grad()
    pred2 = lin2(x2)
    l = F.mse_loss(pred2, target2)
    l.backward()
    opt.step()
print("after:", lin2.weight.data.clone())
print("loss:", l.item())
```
実行結果:
```
before: tensor([[-0.0053,  0.3793]])
after: tensor([[0.1339, 0.6577]])
loss: 0.0010985996341332793
```

**注意点・落とし穴**:
- 学習ループの定型パターンは「`zero_grad()` → `forward` → `loss.backward()` → `step()`」の順。`zero_grad()`を忘れると前述の勾配蓄積の仕様により、パラメータ更新が意図せず大きくなる。
- `lr`にはデフォルト値`0.001`が付いている(実際に`torch.optim.SGD(params)`を`lr`省略で呼んでもエラーにならないことを確認済み)が、SGDでは学習率を明示的に指定するのが一般的。

### `torch.optim.Adam(...)`

**用途**: 勾配の1次・2次モーメントを利用して学習率を自動調整するオプティマイザ(実務でのデフォルト選択肢の一つ)。

**シグネチャ**: `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None, decoupled_weight_decay=False)`

**使用例**:
```python
torch.manual_seed(0)
lin3 = nn.Linear(2, 1)
opt3 = torch.optim.Adam(lin3.parameters(), lr=0.01)
print(opt3)
```
実行結果:
```
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.01
    maximize: False
    weight_decay: 0
)
```

**注意点・落とし穴**:
- `SGD`と使い方(`zero_grad`/`step`)は同じ。パラメータごとに実効学習率が自動調整されるため、多くの場合`SGD`よりチューニングが楽で収束も速い。

### `torch.optim.lr_scheduler.StepLR(...)`

**用途**: 一定エポックごとに学習率を`gamma`倍して減衰させるスケジューラ。

**シグネチャ**: `torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)`

**使用例**:
```python
torch.manual_seed(0)
lin4 = nn.Linear(2, 1)
opt4 = torch.optim.SGD(lin4.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(opt4, step_size=2, gamma=0.5)
lrs = []
for epoch in range(5):
    lrs.append(opt4.param_groups[0]['lr'])
    opt4.step()
    sched.step()
print(lrs)
```
実行結果:
```
[0.1, 0.1, 0.05, 0.05, 0.025]
```

**注意点・落とし穴**:
- `scheduler.step()`は`optimizer.step()`とは別に、通常は1エポックの終わりに1回呼ぶ(学習率の更新タイミングを混同しないこと)。
- 現在の学習率は`optimizer.param_groups[0]['lr']`で確認できる(`scheduler.lr`のような属性はない)。

---

## 9. データローディング

### `torch.utils.data.TensorDataset(...)`

**用途**: 複数のテンソル(特徴量とラベルなど)を1つのデータセットとしてまとめる。

**シグネチャ**: `TensorDataset(*tensors: Tensor)`

**使用例**:
```python
from torch.utils.data import TensorDataset, DataLoader
feats = torch.arange(20.).view(10, 2)
labels = torch.arange(10)
ds = TensorDataset(feats, labels)
print(len(ds))
print(ds[0])
```
実行結果:
```
10
(tensor([0., 1.]), tensor(0))
```

**注意点・落とし穴**:
- 渡す全テンソルの先頭次元(サンプル数)が一致していないと作成時にエラーになる。

### `torch.utils.data.DataLoader(...)`

**用途**: `Dataset`からミニバッチを取り出すイテレータを作る(シャッフル・並列読み込みなどを担当)。

**シグネチャ**: `DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, ...)`

**使用例**:
```python
dl = DataLoader(ds, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
for bx, by in dl:
    print(bx.shape, by)
```
実行結果:
```
torch.Size([4, 2]) tensor([3, 7, 5, 2])
torch.Size([4, 2]) tensor([0, 8, 1, 6])
torch.Size([2, 2]) tensor([9, 4])
```

**注意点・落とし穴**:
- サンプル数(10)が`batch_size`(4)で割り切れないため、最後のバッチだけサイズが2になる。バッチサイズを厳密に揃えたい場合は`drop_last=True`で端数を捨てる。
- `shuffle=True`時の乱数を固定したい場合は、`torch.manual_seed()`のグローバル設定だけでなく、上記のように`generator`引数に専用の`torch.Generator`を渡す方が確実。
- 独自データセットを使う場合は`torch.utils.data.Dataset`を継承し、`__len__`と`__getitem__`を実装すれば`DataLoader`にそのまま渡せる。

---

## 10. デバイス・型変換

### `.to(...)`

**用途**: テンソルのデバイス(CPU/GPU)やdtypeを変更する。

**シグネチャ**: `Tensor.to(*args, **kwargs) -> Tensor`

**使用例**:
```python
t = torch.tensor([1, 2, 3])
print(t.to("cpu"))
print(t.to(torch.float32))

t2 = t.to("cpu")
print(t2 is t)  # 変更が不要なら同一オブジェクトを返す
```
実行結果:
```
tensor([1, 2, 3])
tensor([1., 2., 3.])
True
```

**注意点・落とし穴**:
- 変更後の状態(デバイス・dtype)が現在と同じ場合、`.to()`は新しいテンソルを作らず**元のオブジェクトをそのまま返す**(実行結果の`True`で確認済み)。異なる場合は新しいテンソルを返す(元のテンソルは変更されない)。
- 今回の検証環境はCPU版PyTorch(`torch.cuda.is_available()`は`False`)のため、`.cuda()`や`.to("cuda")`は実行できない。GPU環境では`model.to(device)`と`data.to(device)`の両方を同じデバイスに揃える必要がある(片方だけだと演算時に`RuntimeError`になる)。

### `.float()` / `.double()` / `.long()`

**用途**: テンソルのdtypeを変換するショートカットメソッド。

**シグネチャ**: `Tensor.float() -> Tensor`(`torch.float32`) / `Tensor.double() -> Tensor`(`torch.float64`) / `Tensor.long() -> Tensor`(`torch.int64`)

**使用例**:
```python
t2 = torch.tensor([1, 2, 3])
print(t2.float().dtype)
print(t2.double().dtype)
print(t2.long().dtype)
```
実行結果:
```
torch.float32
torch.float64
torch.int64
```

**注意点・落とし穴**:
- `nn.CrossEntropyLoss`の`target`など整数ラベルを要求するAPIに浮動小数点テンソルを渡すと型エラーになるため、`.long()`での変換が必要になる場面が多い。

### `.item()`

**用途**: 要素数1のテンソルからPythonのスカラー値(int/float)を取り出す。

**シグネチャ**: `Tensor.item() -> number`

**使用例**:
```python
scalar_t = torch.tensor(3.14)
print(scalar_t.item(), type(scalar_t.item()))

big = torch.tensor([1.0, 2.0])
try:
    big.item()
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))
```
実行結果:
```
3.140000104904175 <class 'float'>
ERROR: RuntimeError a Tensor with 2 elements cannot be converted to Scalar
```

**注意点・落とし穴**:
- 要素数が2つ以上のテンソルに`.item()`を呼ぶと`RuntimeError: a Tensor with 2 elements cannot be converted to Scalar`になる(実行して確認済み)。ログ出力などで`loss.item()`のように使うのが典型パターン。

### `.clone()` / `.contiguous()`

**用途**: `.clone()`はデータをコピーした新しいテンソルを作る(計算グラフは維持される)。`.contiguous()`はメモリ上の並びを連続にしたコピーを作る(すでに連続なら何もしない)。

**シグネチャ**: `Tensor.clone(*, memory_format=torch.preserve_format) -> Tensor` / `Tensor.contiguous(memory_format=torch.contiguous_format) -> Tensor`

**使用例**:
```python
orig = torch.tensor([1, 2, 3])
cl = orig.clone()
cl[0] = 99
print(orig, cl)

tct = torch.arange(6).view(2, 3).t()
print(tct.is_contiguous())
print(tct.contiguous().is_contiguous())
```
実行結果:
```
tensor([1, 2, 3]) tensor([99,  2,  3])
False
True
```

**注意点・落とし穴**:
- `.clone()`は`requires_grad`や計算グラフへの接続を保持する(微分可能な複製)。計算グラフからも切り離したい場合は`.clone().detach()`とする。
- すでに連続なテンソルに`.contiguous()`を呼んでもコピーは発生せず、同一オブジェクトが返る(実行して確認済み)。

---

## 11. 保存・読み込み

### `torch.save(...)` / `torch.load(...)`

**用途**: テンソルやPythonオブジェクトをファイル(または任意のファイルライクオブジェクト)にシリアライズ/デシリアライズする。

**シグネチャ**: `torch.save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)` / `torch.load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)`

**使用例**:
```python
import io
buf = io.BytesIO()
sv = torch.tensor([1., 2., 3.])
torch.save(sv, buf)
buf.seek(0)
loaded = torch.load(buf)
print(loaded)
```
実行結果:
```
tensor([1., 2., 3.])
```

**注意点・落とし穴**:
- PyTorch 2.13時点で`torch.load`の`weights_only`はデフォルト`True`(セキュリティ上の理由でpickleの任意コード実行を防ぐ制限モード)。テンソル/state_dict以外の任意のPythonオブジェクトを保存していた場合は`weights_only=False`を明示しないと読み込めないことがある。

### `state_dict()` / `load_state_dict(...)`

**用途**: モデルの学習可能パラメータ(重み・バイアス)だけを辞書として取得・復元する。モデル全体を保存するより推奨される方法。

**シグネチャ**: `nn.Module.state_dict(...) -> OrderedDict` / `nn.Module.load_state_dict(state_dict, strict=True) -> _IncompatibleKeys`

**使用例**:
```python
torch.manual_seed(0)
m1 = nn.Linear(2, 2)
sd = m1.state_dict()
print(list(sd.keys()))

buf2 = io.BytesIO()
torch.save(m1.state_dict(), buf2)
buf2.seek(0)
m2 = nn.Linear(2, 2)
m2.load_state_dict(torch.load(buf2))
print(torch.equal(m1.weight, m2.weight))
```
実行結果:
```
['weight', 'bias']
True
```

**注意点・落とし穴**:
- `load_state_dict`を使う前に、読み込み先のモデル(`m2`)を保存元と**同じアーキテクチャ**で先にインスタンス化しておく必要がある(構造そのものは保存されず、パラメータの値だけが保存される)。
- `strict=True`(デフォルト)では、キーが一部でも一致しないと`RuntimeError`になる。部分的な読み込みを許容したい場合は`strict=False`を指定する。
