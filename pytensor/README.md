# pytensor 逆引き辞書

PyTensor 3.3.0 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認)。

PyTensor は PyMC の内部で使われているシンボリック計算グラフ・自動微分エンジン。numpy ライクな API で計算グラフを組み立て、コンパイルしてから実行する「遅延評価」モデルを採る点が numpy と大きく異なる。

## 目次

1. [シンボリック基礎](#シンボリック基礎)
2. [関数定義・コンパイル](#関数定義コンパイル)
3. [自動微分](#自動微分)
4. [演算子・数学関数](#演算子数学関数)
5. [制御フロー](#制御フロー)
6. [形状操作](#形状操作)
7. [共有変数](#共有変数)
8. [グラフの可視化・デバッグ](#グラフの可視化デバッグ)
9. [応用・発展](#応用発展)
   - [グラフ最適化(rewrite/optimizer)](#グラフ最適化rewriteoptimizer)
   - [カスタムOpの自作](#カスタムopの自作)
   - [勾配の応用](#勾配の応用)
   - [別バックエンドへのコンパイル](#別バックエンドへのコンパイル)

---

## シンボリック基礎

### `pt.scalar()` / `pt.vector()` / `pt.matrix()`

**用途**: 値を持たない「シンボリック変数」(プレースホルダ)を生成する、PyTensor の最も基本的な操作。実際の値は後で `.eval()` や `pytensor.function` を通して与える。

**シグネチャ**:
- `pt.scalar(name: str | None = None, *, dtype=None) -> TensorVariable`
- `pt.vector(name: str | None = None, *, dtype=None, shape=(None,)) -> TensorVariable`
- `pt.matrix(name: str | None = None, *, dtype=None, shape=(None, None)) -> TensorVariable`

**使用例**:
```python
import pytensor.tensor as pt

x = pt.scalar('x')
v = pt.vector('v')
M = pt.matrix('M')
print(type(x), x.type)
print(type(v), v.type)
print(type(M), M.type)
```
実行結果:
```
<class 'pytensor.tensor.variable.TensorVariable'> Scalar(float64, shape=())
<class 'pytensor.tensor.variable.TensorVariable'> Vector(float64, shape=(?,))
<class 'pytensor.tensor.variable.TensorVariable'> Matrix(float64, shape=(?, ?))
```

**注意点・落とし穴**:
- `pt.scalar('x')` の時点では `x` は値を持たない。`print(x)` しても値は出力されず、変数名(`x`)や `.type` の情報が見えるだけ。numpy の配列のように即座に値を確認することはできない。
- `dtype` を省略すると `pytensor.config.floatX`(既定 `float64`)が使われる。

---

### `pt.tensor(name, *, dtype, shape)`

**用途**: 任意次元数のシンボリックテンソルを生成する汎用関数。各軸のサイズを固定値・`None`(可変)で指定できる。

**シグネチャ**: `pt.tensor(name: str | None = None, *, dtype=None, shape: tuple[int | None, ...] | None = None, **kwargs) -> TensorVariable`

**使用例**:
```python
import pytensor.tensor as pt

t3 = pt.tensor('t3', shape=(2, 3, None), dtype='float64')
print(t3.type)
```
実行結果:
```
Tensor3(float64, shape=(2, 3, ?))
```

**注意点・落とし穴**:
- `shape` に整数を指定した軸はコンパイル時に厳密にチェックされ、異なるサイズの配列を渡すと実行時エラーになる。`None` を指定した軸のみ可変長になる。

---

### `dtype` 引数による型指定

**用途**: シンボリック変数の要素の型(`float64`, `int32` など)を明示する。

**使用例**:
```python
import pytensor.tensor as pt

y = pt.vector('y', dtype='int32')
print(y.dtype, y.name)
```
実行結果:
```
int32 y
```

**注意点・落とし穴**:
- `pytensor.config.floatX`(既定 `float64`)と異なる dtype の変数同士を混ぜると、`function`/`eval` 時に `TypeError: Cannot convert Type ... into Type ...` が出ることがある。特に `pt.constant(1.0)` は既定で `float32` になる(下記 `pt.constant` の注意点を参照)ため、`float64` の変数と組み合わせるときは `dtype` を明示する必要がある。

---

### `pt.constant(x, name=None, ndim=None, dtype=None)`

**用途**: 計算グラフに埋め込む定数(値が変わらないノード)を作る。

**シグネチャ**: `pt.constant(x, name=None, ndim=None, dtype=None) -> TensorConstant`

**使用例**:
```python
import pytensor.tensor as pt

c = pt.constant(5.0, dtype='float64', name='five')
print(c, c.eval())
```
実行結果:
```
five{5.0} 5.0
```

**注意点・落とし穴**:
- `pt.constant(10.0)` のように `dtype` を省略すると、`pytensor.config.floatX` が `float64` であっても実際には `float32` になる(実機で確認済み)。`float64` の変数を使うグラフに定数として混ぜたい場合は `dtype='float64'` を明示しないと型不一致エラーになる。

---

## 関数定義・コンパイル

### `Variable.eval(inputs_to_values)`

**用途**: シンボリック変数のグラフを、その場でコンパイル・実行して具体的な値を得る簡易手段。プロトタイピングやデバッグに向く。

**シグネチャ**: `Variable.eval(inputs_to_values: dict | None = None, **kwargs)`

**使用例**:
```python
import pytensor.tensor as pt

x = pt.scalar('x')
y = x ** 2 + 1
print(y.eval({x: 3.0}))
```
実行結果:
```
10.0
```

**注意点・落とし穴**:
- `eval()` は呼び出すたびに内部で `pytensor.function` を(キャッシュしつつ)呼ぶため、同じグラフを何度も評価する場合は `pytensor.function` で明示的にコンパイルしておいた方が高速。
- PyTensor の変数は「シンボリック」であり、`.eval()` や `function()` を通さない限り実際の数値にはならない。numpy のようにその場で計算されるわけではない、という遅延評価モデルの理解が最初の関門になる。

---

### `pytensor.function(inputs, outputs, ...)`

**用途**: シンボリックな入出力の対応関係から、実行可能な関数(コンパイル済み計算グラフ)を作る。PyTensor の中心的な API。

**シグネチャ**: `pytensor.function(inputs, outputs=None, mode=None, updates=None, givens=None, no_default_updates=None, accept_inplace=False, name=None, rebuild_strict=True, allow_input_downcast=None, profile=None, on_unused_input=None, trust_input=False)`

**使用例**:
```python
import pytensor
import pytensor.tensor as pt

a = pt.scalar('a')
b = pt.scalar('b')
z = a + b
f = pytensor.function([a, b], z)
print(f(2.0, 3.0))
print(type(f(2.0, 3.0)))
```
実行結果:
```
5.0
<class 'numpy.ndarray'>
```

**注意点・落とし穴**:
- 戻り値は Python の `float` ではなく `numpy.ndarray`(スカラーの場合は 0 次元配列)になる。
- `inputs` に列挙した変数だけがコンパイル済み関数の引数になる。グラフ中に使われているが `inputs` に含まれていない自由変数があるとコンパイル時にエラーになる(`givens` や `shared` で埋めない限り)。

---

### `givens` 引数によるグラフ内変数の置換

**用途**: グラフ中の特定の変数を、コンパイル時に別の式・定数に置き換える。入力を増やさずに一部の変数を固定したい場合に使う。

**使用例**:
```python
import pytensor
import pytensor.tensor as pt

a = pt.scalar('a')
c = pt.scalar('c')
expr = a * c
f2 = pytensor.function([a], expr, givens={c: pt.constant(10.0, dtype='float64')})
print(f2(5.0))
```
実行結果:
```
50.0
```

**注意点・落とし穴**:
- `givens` で置き換える値の dtype は元の変数の dtype と一致している必要がある(前述の `pt.constant` の float32/float64 の罠に注意)。

---

### `pytensor.config.floatX`

**用途**: `dtype` を省略したときに使われる既定の浮動小数点型を保持する設定値。

**使用例**:
```python
import pytensor
print(pytensor.config.floatX)
```
実行結果:
```
float64
```

**注意点・落とし穴**:
- CPU 環境ではこの検証環境のように既定で `float64` だが、GPU 利用を想定した設定では `float32` にすることが一般的(PyTensor/Theano 系譜の伝統)。`pt.constant` など一部の API はこの設定に関わらず `float32` を返すことがあるため、`floatX` を過信せず実際の `dtype` 属性を確認する習慣が重要。

---

## 自動微分

### `pytensor.grad(cost, wrt, ...)`

**用途**: シンボリックな式(コスト関数)を、指定した変数について微分した式(勾配)を計算グラフとして返す。

**シグネチャ**: `pytensor.grad(cost, wrt, consider_constant=None, disconnected_inputs='raise', add_names=True, known_grads=None, return_disconnected='zero', null_gradients='raise')`

**使用例**:
```python
import pytensor
import pytensor.tensor as pt
import numpy as np

x = pt.scalar('x')
y = x ** 3
gy = pytensor.grad(y, x)
print(gy.eval({x: 2.0}))  # dy/dx = 3x^2

v = pt.vector('v')
s = (v ** 2).sum()
gv = pytensor.grad(s, v)
print(gv.eval({v: np.array([1.0, 2.0, 3.0])}))
```
実行結果:
```
12.0
[2. 4. 6.]
```

**注意点・落とし穴**:
- `grad` が返すのは微分「値」ではなく微分「式」(シンボリックグラフ)。数値がほしい場合は `.eval()` や `pytensor.function` を通す必要がある。
- `wrt` の変数が `cost` の計算に(グラフ上)全く関与していない場合、既定 `disconnected_inputs='raise'` によりエラーになる。意図的に 0 勾配を許容したい場合は `disconnected_inputs='ignore'` を指定する。

---

### `pytensor.gradient.jacobian(expression, wrt)`

**用途**: ベクトル値関数のヤコビ行列(各出力要素の各入力要素に対する偏微分)を計算する。

**シグネチャ**: `pytensor.gradient.jacobian(expression, wrt, consider_constant=None, disconnected_inputs='raise', vectorize=False)`

**使用例**:
```python
import pytensor
import pytensor.tensor as pt
import numpy as np

w = pt.vector('w')
out = w ** 2
j = pytensor.gradient.jacobian(out, w)
print(j.eval({w: np.array([1.0, 2.0, 3.0])}))
```
実行結果:
```
[[2. 0. 0.]
 [0. 4. 0.]
 [0. 0. 6.]]
```

**注意点・落とし穴**:
- `w ** 2` は要素ごとの演算なので、ヤコビ行列は対角行列になる(各出力は対応する入力にしか依存しない)。出力とパラメータの依存関係が密な場合は計算コストが `出力数 × 入力数` で増える点に注意。

---

### `clone_replace(output, replace)`

**用途**: 既存の計算グラフをコピーしつつ、一部の変数だけを別の式・定数に差し替えた新しいグラフを作る。

**シグネチャ**: `pytensor.graph.replace.clone_replace(output, replace=None, **rebuild_kwds)`

**使用例**:
```python
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace

x = pt.scalar('x')
y = x * 2
z = y + 1
new_z = clone_replace(z, replace={x: pt.constant(5.0, dtype='float64')})
print(new_z.eval())
```
実行結果:
```
11.0
```

**注意点・落とし穴**:
- 元のグラフ(`z`)自体は変更されない。`clone_replace` は新しいグラフを返すだけなので、置き換え後の変数を使いたい場合は戻り値を使う必要がある。

---

## 演算子・数学関数

### 比較演算子(`==`, `!=`, `<`, `>` など)

**用途**: シンボリック変数同士を比較する。ただし `==`/`!=` と `<`/`>`/`<=`/`>=` で挙動が全く異なる点に要注意。

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

a = pt.vector('a')
b = pt.vector('b')

r_eq = a == b
print(type(r_eq), r_eq)          # 要素ごとの比較にならない

r_eq2 = pt.eq(a, b)
print(type(r_eq2))
print(r_eq2.eval({a: np.array([1., 2., 3.]), b: np.array([1., 5., 3.])}))

r_gt = a > b
print(type(r_gt))
print(r_gt.eval({a: np.array([1., 5., 3.]), b: np.array([2., 2., 2.])}))
```
実行結果:
```
<class 'bool'> False
<class 'pytensor.tensor.variable.TensorVariable'>
[ True False  True]
<class 'pytensor.tensor.variable.TensorVariable'>
[False  True  True]
```

**注意点・落とし穴**:
- `a == b` は要素ごとの比較にはならず、Python の `TensorVariable.__eq__` がオブジェクト同一性(identity)ベースの比較にオーバーライドされているため、単なる `bool`(この例では `False`)が返る。これは numpy と全く異なる挙動であり、最も踏みやすい罠の一つ。
- 要素ごとの等価比較をしたい場合は `pt.eq(a, b)` / 不等価は `pt.neq(a, b)` を使う。一方 `<`, `>`, `<=`, `>=` は演算子オーバーロードにより正しく要素ごとの `TensorVariable` を返す。

---

### `pt.exp()` / `pt.sqrt()` / `pt.log()`

**用途**: 指数関数・平方根・自然対数などを要素ごとに計算する numpy ライクな数学関数群。

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

a = pt.vector('a')
print(pt.exp(a).eval({a: np.array([0., 1., 2.])}))
print(pt.sqrt(a).eval({a: np.array([1., 4., 9.])}))
```
実行結果:
```
[1.         2.71828183 7.3890561 ]
[1. 2. 3.]
```

**注意点・落とし穴**:
- numpy の ufunc と名前・挙動はほぼ同じだが、戻り値はシンボリックな `TensorVariable` であり `.eval()` を経由しないと数値は得られない。

---

### `pt.dot(l, r)`

**用途**: 行列積・ベクトル内積を計算する。

**シグネチャ**: `pt.dot(l, r)`

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

M = pt.matrix('M')
N = pt.matrix('N')
print(pt.dot(M, N).eval({M: np.array([[1., 2.], [3., 4.]]), N: np.array([[5., 6.], [7., 8.]])}))
```
実行結果:
```
[[19. 22.]
 [43. 50.]]
```

**注意点・落とし穴**:
- numpy と同様、`@` 演算子も同じ意味で使える(`M @ N`)。

---

### `pt.sum()` / `pt.mean()`

**用途**: 配列全体または指定軸方向の合計・平均を計算する。

**シグネチャ**: `pt.sum(input, axis=None, dtype=None, keepdims=False, acc_dtype=None)`(`pt.mean` もほぼ同形)

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

M = pt.matrix('M')
print(pt.sum(M).eval({M: np.array([[1., 2.], [3., 4.]])}))
print(pt.sum(M, axis=0).eval({M: np.array([[1., 2.], [3., 4.]])}))
print(pt.mean(M, axis=1).eval({M: np.array([[1., 2.], [3., 4.]])}))
```
実行結果:
```
10.0
[4. 6.]
[1.5 3.5]
```

---

### `pt.switch(cond, ift, iff)`

**用途**: 条件に応じて要素ごとに2つの式のどちらかを選択する、numpy の `np.where` に相当する演算。

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

cond = pt.vector('cond')
a = pt.vector('a')
b = pt.vector('b')
sw = pt.switch(cond > 0, a, b)
print(sw.eval({cond: np.array([1., -1., 0.]), a: np.array([10., 20., 30.]), b: np.array([-10., -20., -30.])}))
```
実行結果:
```
[ 10. -20. -30.]
```

**注意点・落とし穴**:
- `pt.switch` は `ift`/`iff` の両方の式をグラフ上に構築し、実行時に要素ごとに選択する(Python の `if`/`else` のように片方だけを計算してスキップするわけではない)。両分岐で高コストな計算やエラーになりうる演算(0除算など)を書くと、選ばれない側でも評価されて問題になることがある。

---

### `pt.arange(start, stop=None, step=1, dtype=None)`

**用途**: 等間隔のシンボリックな整数列を生成する、`np.arange` のシンボリック版。

**使用例**:
```python
import pytensor.tensor as pt

print(pt.arange(0, 10, 2).eval())
```
実行結果:
```
[0 2 4 6 8]
```

---

## 制御フロー

### `pytensor.scan(fn, sequences, outputs_info, ...)`

**用途**: ループ(再帰的な計算)をシンボリックな計算グラフとして表現する。PyMC の時系列モデルなどで多用される、PyTensor で最も重要な制御フロー構造。

**シグネチャ**: `pytensor.scan(fn: Callable, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, truncate_gradient=-1, go_backwards=False, mode=None, name=None, profile=False, allow_gc=None, strict=False, return_list=False, return_updates: bool = True)`

**使用例**:
```python
import pytensor
import pytensor.tensor as pt
import numpy as np
from pytensor import scan

def step(x_t, acc_tm1):
    return acc_tm1 + x_t

xs = pt.vector('xs')
result = scan(
    fn=step,
    sequences=xs,
    outputs_info=pt.constant(0.0, dtype='float64'),
    return_updates=False,
)
f = pytensor.function([xs], result)
print(f(np.array([1.0, 2.0, 3.0, 4.0])))
```
実行結果:
```
[ 1.  3.  6. 10.]
```

**注意点・落とし穴**:
- 検証バージョン(3.3.0)では `scan()` の戻り値仕様が変更中で、既定(`return_updates` 省略)では `(outputs, updates)` のタプルを返すが `DeprecationWarning: Scan return signature will change` が出る。将来のバージョンでは `outputs` のみを返す形に変わる予定のため、`return_updates=False` を明示して警告を避けるのが安全。
- `outputs_info` に渡す初期値の dtype は `fn` の戻り値の dtype と一致していなければならない(`pt.constant(0.0)` は既定で `float32` になるため、`float64` の `xs` と組み合わせると `ValueError` になる。上記例のように `dtype='float64'` を明示する必要がある)。
- `sequences` に渡した配列は各ステップで先頭の軸に沿って1要素ずつ `fn` に渡される。

---

### `pytensor.ifelse.ifelse(condition, then_branch, else_branch)`

**用途**: スカラーの条件に応じて2つの分岐(サブグラフ)のどちらか一方だけを実行する。`pt.switch` と異なり、選ばれなかった分岐は計算されない(lazy)。

**シグネチャ**: `ifelse(condition, then_branch, else_branch, name=None)`

**使用例**:
```python
import pytensor
import pytensor.tensor as pt
from pytensor.ifelse import ifelse

cond = pt.scalar('cond', dtype='bool')
a = pt.scalar('a')
b = pt.scalar('b')
out = ifelse(cond, a, b)
f = pytensor.function([cond, a, b], out)
print(f(True, 1.0, 2.0))
print(f(False, 1.0, 2.0))
```
実行結果:
```
1.0
2.0
```

**注意点・落とし穴**:
- `pt.switch` は「値」を要素ごとに選ぶ(両方評価される)のに対し、`ifelse` は「計算グラフの分岐」を選ぶ(選ばれなかった側は評価されない)。片方の分岐だけが高コスト・副作用がある場合は `switch` ではなく `ifelse` を使うべき。
- `condition` はスカラーである必要がある(配列ごとに分岐を変えたい場合は `switch` を使う)。

---

## 形状操作

### `pt.reshape(x, newshape)` / `Variable.reshape(*shape)`

**用途**: 要素数を変えずに shape を変更する。

**シグネチャ**: `pt.reshape(x, newshape, *, ndim=None)` / `Variable.reshape(shape, *, ndim=None)`

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

v = pt.vector('v')
r = pt.reshape(v, (2, 3))
print(r.eval({v: np.arange(6.0)}))
```
実行結果:
```
[[0. 1. 2.]
 [3. 4. 5.]]
```

---

### `Variable.dimshuffle(*pattern)`

**用途**: 軸の並び替え・追加・削除を行う。numpy の `transpose` や `np.newaxis` に相当する PyTensor 独自の操作。

**シグネチャ**: `Variable.dimshuffle(*pattern)`

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

x = pt.vector('x')
xd = x.dimshuffle('x', 0)   # サイズ1の軸を先頭に追加
print(xd.eval({x: np.array([1., 2., 3.])}).shape)

y = pt.matrix('y')
yt = y.dimshuffle(1, 0)     # 転置
print(yt.eval({y: np.array([[1., 2., 3.], [4., 5., 6.]])}))
```
実行結果:
```
(1, 3)
[[1. 4.]
 [2. 5.]
 [3. 6.]]
```

**注意点・落とし穴**:
- パターン中の文字列 `'x'` は「サイズ1の新しい軸」を意味する(numpy の `np.newaxis` に相当)。整数はどの既存軸をその位置に持ってくるかを指定する。

---

### `pt.join(axis, *tensors)` / `pt.concatenate(tensor_list, axis=0)`

**用途**: 複数の配列を既存の軸に沿って連結する。`pt.concatenate` は numpy と同じ引数順(リストが先)、`pt.join` は軸が先という違いがある。

**シグネチャ**: `pt.join(axis, *tensors_list)` / `pt.concatenate(tensor_list, axis=0)`

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

c1 = pt.vector('c1')
c2 = pt.vector('c2')
print(pt.join(0, c1, c2).eval({c1: np.array([1., 2.]), c2: np.array([3., 4.])}))

a = pt.matrix('a')
b = pt.matrix('b')
print(pt.concatenate([a, b], axis=0).eval({a: np.array([[1., 2.]]), b: np.array([[3., 4.]])}))
```
実行結果:
```
[1. 2. 3. 4.]
[[1. 2.]
 [3. 4.]]
```

---

### `pt.stack(tensors, axis=0)`

**用途**: 複数の配列を「新しい軸」を作って積み重ねる。

**シグネチャ**: `pt.stack(tensors, axis=0)`

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

c1 = pt.vector('c1')
c2 = pt.vector('c2')
s = pt.stack([c1, c2])
print(s.eval({c1: np.array([1., 2.]), c2: np.array([3., 4.])}))
```
実行結果:
```
[[1. 2.]
 [3. 4.]]
```

---

### `pt.flatten(x, ndim=1)`

**用途**: 多次元配列を指定した次元数まで平坦化する。

**シグネチャ**: `pt.flatten(x, ndim=1)`

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

m = pt.matrix('m')
print(pt.flatten(m).eval({m: np.array([[1., 2.], [3., 4.]])}))
```
実行結果:
```
[1. 2. 3. 4.]
```

---

### スライシングと `pt.set_subtensor(subtensor, new_value)`

**用途**: numpy と同じ構文で部分配列を取り出す。書き換えたい場合はビューへの直接代入ではなく `pt.set_subtensor` を使い、更新後の「新しいグラフ」を得る。

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

v = pt.vector('v')
sub = v[1:3]
print(sub.eval({v: np.array([1., 2., 3., 4., 5.])}))

upd = pt.set_subtensor(v[1:3], np.array([100., 200.]))
print(upd.eval({v: np.array([1., 2., 3., 4., 5.])}))
```
実行結果:
```
[2. 3.]
[  1. 100. 200.   4.   5.]
```

**注意点・落とし穴**:
- numpy の `a[1:3] = [100, 200]` のような破壊的代入はできない(シンボリック変数は不変)。`pt.set_subtensor` は元の変数を書き換えるのではなく、「その部分だけ値が変わった新しいグラフ」を返す点が numpy と根本的に異なる。

---

### `Variable.shape`

**用途**: シンボリック変数の shape を、それ自体シンボリックな整数ベクトルとして取得する。

**使用例**:
```python
import pytensor.tensor as pt
import numpy as np

m = pt.matrix('m')
print(m.shape.eval({m: np.zeros((3, 4))}))
```
実行結果:
```
[3 4]
```

**注意点・落とし穴**:
- `m.shape` はただの Python タプルではなく `TensorVariable` であり、コンパイル時点では中身が確定していない(実行時に決まる)。`m.shape[0]` のように取り出した要素もシンボリックな値になる。

---

## 共有変数

### `pytensor.shared(value, name=None, ...)`

**用途**: グラフをまたいで状態を保持できる変数(shared variable)を作る。ニューラルネットの重みや PyMC の内部状態のように、関数呼び出しの間で値を保持・更新したいときに使う。

**シグネチャ**: `pytensor.shared(value, name=None, strict=False, allow_downcast=None, **kwargs)`

**使用例**:
```python
import pytensor
import pytensor.tensor as pt

state = pytensor.shared(0.0, name='state')
print(state.get_value())

inc = pt.scalar('inc')
accumulate = pytensor.function([inc], state, updates=[(state, state + inc)])
print(accumulate(1.0))
print(accumulate(2.0))
print(state.get_value())
```
実行結果:
```
0.0
0.0
1.0
3.0
```

**注意点・落とし穴**:
- `accumulate(1.0)` の戻り値 `0.0` は「更新前」の `state` の値。`updates` は関数の戻り値を計算した後に適用されるため、戻り値と更新後の値が食い違う(このズレは shared 変数を使う上で頻出の混乱ポイント)。

---

### `shared.get_value()` / `shared.set_value(new_value)`

**用途**: 共有変数の現在の値を numpy 配列として取得・上書きする。

**シグネチャ**: `get_value(borrow=False, return_internal_type=False)` / `set_value(new_value, borrow=False)`

**使用例**:
```python
import pytensor

state = pytensor.shared(0.0, name='state')
state.set_value(100.0)
print(state.get_value())
```
実行結果:
```
100.0
```

**注意点・落とし穴**:
- `pytensor.function` の `updates` を介さず、Python 側から直接 `set_value` で書き換えることもできる。学習ループの外からハイパーパラメータやバッチデータを差し替える際によく使われる。

---

### `pytensor.function` の `updates` 引数

**用途**: 関数を呼ぶたびに共有変数を自動更新する仕組み。`(共有変数, 更新後の式)` のペアのリスト/辞書を渡す。

**使用例**: 上記 `pytensor.shared` の例を参照(`updates=[(state, state + inc)]` により、呼び出すたびに `state` が `inc` 分だけ増加する)。

**注意点・落とし穴**:
- `updates` に指定できるのは shared 変数のみ。通常のシンボリック変数(`pt.scalar` などで作ったもの)を対象にはできない。

---

## グラフの可視化・デバッグ

### `pytensor.dprint(graph_like, ...)`

**用途**: 計算グラフをツリー形式でテキスト出力する、最も基本的なデバッグ手段。シンボリック変数・コンパイル済み関数のどちらにも使える。

**シグネチャ**: `pytensor.dprint(graph_like, depth=-1, print_type=False, print_shape=False, file=None, id_type='CHAR', ...)`(他にも多数のオプション引数がある)

**使用例**:
```python
import pytensor
import pytensor.tensor as pt

x = pt.matrix('x')
y = pt.matrix('y')
z = pt.dot(x, y).sum()
pytensor.dprint(z)
print('--- print_type=True ---')
pytensor.dprint(z, print_type=True)
```
実行結果:
```
Sum{axes=None} [id A]
 └─ Dot [id B]
    ├─ x [id C]
    └─ y [id D]
--- print_type=True ---
Sum{axes=None} [id A] <Scalar(float64, shape=())>
 └─ Dot [id B] <Matrix(float64, shape=(?, ?))>
    ├─ x [id C] <Matrix(float64, shape=(?, ?))>
    └─ y [id D] <Matrix(float64, shape=(?, ?))>
```

**注意点・落とし穴**:
- `pytensor.function` でコンパイル済みの関数を渡すと、最適化(グラフ書き換え)後の実際に実行される演算グラフが表示される。最適化前のグラフ(`z` のようなシンボリック変数そのもの)を渡した場合とは表示されるノードが異なることがある。
- グラフを画像として可視化する `pytensor.printing.pydotprint` も存在するが、追加で `pydot` パッケージ(および Graphviz)のインストールが必要(この検証環境には `pydot` が入っておらず未検証)。

---

### シンボリック計算の遅延評価モデル(まとめ)

**用途**: 本辞書全体を通した注意点として、PyTensor と numpy の根本的な違いを整理する。

**注意点・落とし穴**:
- PyTensor の変数(`pt.scalar`, `pt.vector` などで作ったもの)は値を持たない「式」であり、`print(x)` しても数値は出ない。数値が必要なときは必ず `.eval({...})` するか `pytensor.function` でコンパイルした関数を呼び出す。
- `a == b` のような Python 標準の等価演算子は要素ごとの比較にならない(オブジェクト同一性の `bool` を返す)。要素ごとの比較には `pt.eq`/`pt.neq` を使う(詳細は上記「比較演算子」の項を参照)。
- 配列への破壊的な要素代入(`a[i] = x`)はできない。書き換えたい場合は `pt.set_subtensor` で「新しいグラフ」を作る。
- `pt.constant(1.0)` のように `dtype` を省略すると `float32` になることがあり、`float64` の変数と混ぜると型エラーになりやすい。迷ったら `dtype` を明示する。

---

## 応用・発展

### グラフ最適化(rewrite/optimizer)

#### `pytensor.function` の `mode` 引数と最適化レベル

**用途**: コンパイル時にどのレベルのグラフ書き換え(rewrite/最適化)を適用するかを切り替える。`FAST_RUN` は積極的に最適化してから実行し、`FAST_COMPILE` は最適化をほぼ省略してコンパイル自体を高速化する。

**使用例**:
```python
import pytensor
import pytensor.tensor as pt

x = pt.scalar('x')
y = x ** 2
f_run = pytensor.function([x], y, mode='FAST_RUN')
f_compile = pytensor.function([x], y, mode='FAST_COMPILE')
print('--- FAST_RUN ---')
pytensor.dprint(f_run)
print('--- FAST_COMPILE ---')
pytensor.dprint(f_compile)
print(f_run(4.0), f_compile(4.0))
```
実行結果:
```
--- FAST_RUN ---
Sqr [id A] 0
 └─ x [id B]
--- FAST_COMPILE ---
Pow [id A] 0
 ├─ x [id B]
 └─ 2 [id C]
16.0 16.0
```

**注意点・落とし穴**:
- `FAST_RUN` では `x ** 2` という `Pow` 演算が専用の `Sqr`(2乗専用)Opに書き換えられており、実行されるグラフの構造そのものが変わっている。最終的な計算結果は同じだが、`dprint` で見えるノードは最適化レベルによって異なる。
- デバッグ時に「グラフがどう実行されるか」を確認したい場合、`mode='FAST_COMPILE'` の方が元のコード(`x ** 2`)に近い素直なグラフになり読みやすい。

---

#### `pytensor.graph.rewrite_graph(graph, include=(...))`

**用途**: `pytensor.function` を経由せず、シンボリックグラフ単体に対して明示的にグラフ書き換え(rewrite)を適用し、最適化後のグラフを取得する。グラフ最適化の効果を単体で確認したいときに使う。

**シグネチャ**: `pytensor.graph.rewrite_graph(graph, include=('canonicalize',), custom_rewrite=None, clone=False, **kwargs)`

**使用例**:
```python
import pytensor
import pytensor.tensor as pt
from pytensor.graph import rewrite_graph

x = pt.scalar('x')
y = (x + 0.0) * 1.0
print('最適化前:')
pytensor.dprint(y)
y_opt = rewrite_graph(y, include=('canonicalize',))
print('最適化後:')
pytensor.dprint(y_opt)
```
実行結果:
```
最適化前:
Mul [id A]
 ├─ Add [id B]
 │  ├─ x [id C]
 │  └─ 0.0 [id D]
 └─ 1.0 [id E]
最適化後:
x [id A]
```

**注意点・落とし穴**:
- `+0.0` や `*1.0` のような恒等演算が完全に消え、グラフが `x` そのものに簡約される。「グラフ最適化」が単なる高速化ではなく、代数的に等価なグラフへの書き換えであることが視覚的にわかる例。
- `include` に渡す文字列(`'canonicalize'` など)は `pytensor.compile.mode` 内部の rewrite データベースの登録名に対応する。どの名前がどの書き換え群を指すかは PyTensor 内部の実装に依存し、公開ドキュメントが薄いため、実際に `dprint` で前後を比較しながら使うのが実用的。

---

### カスタムOpの自作

#### `Op` 基底クラスの最小実装(`make_node` / `perform`)

**用途**: PyTensor に組み込まれていない演算を、独自の `Op` として計算グラフに組み込む。`make_node` で入出力の型(`Apply` ノード)を定義し、`perform` で実際の数値計算(numpy レベル)を書く。

**使用例**:
```python
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply

class DoubleOp(Op):
    __props__ = ()  # このOpはパラメータを持たない

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])  # 入力1つ、同じ型の出力1つ

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = np.asarray(x * 2)

double_op = DoubleOp()
x = pt.vector('x')
f = pytensor.function([x], double_op(x))
print(f(np.array([1.0, 2.0, 3.0])))
```
実行結果:
```
[2. 4. 6.]
```

**注意点・落とし穴**:
- `__props__ = ()` は必須に近い。これを定義しないと `Op` 同士の等価性判定(グラフのマージ最適化などで使われる)が正しく動かないことがある。
- `perform` の `output_storage` は「1要素のリストを1個含むリスト」という独特の形(`output_storage[0][0] = 値`)で書き込む。numpy 配列をそのまま `return` するわけではない点が初見でつまずきやすい。
- この最小実装では `grad`(下記参照)を定義していないため、この `DoubleOp` を含むグラフに対して `pytensor.grad` を呼ぶとエラーになる。

---

#### `Op.grad(inputs, output_grads)` による独自勾配の定義

**用途**: カスタム `Op` に対して `pytensor.grad` が使えるように、逆伝播時の勾配計算式を定義する。

**使用例**:
```python
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply

class DoubleOp(Op):
    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = np.asarray(x * 2)

    def grad(self, inputs, output_grads):
        (gz,) = output_grads
        return [gz * 2]  # d(2x)/dx = 2

double_op = DoubleOp()
x = pt.vector('x')
g = pytensor.grad(double_op(x).sum(), x)
print(g.eval({x: np.array([1.0, 2.0, 3.0])}))
```
実行結果:
```
[2. 2. 2.]
```

**注意点・落とし穴**:
- `grad` は「入力ごとの勾配式のリスト」を返す必要がある(`inputs` と同じ長さ)。多入力Opで一部の入力に勾配が定義できない場合は、後述の `grad_not_implemented` / `grad_undefined` を該当位置に入れる。
- 検証環境(3.3.0)では `grad`/`L_op` を実装すると `FutureWarning: <OpName> should implement \`pullback\` instead of \`L_op\`/\`grad\`. Direct \`L_op\`/\`grad\` implementations are deprecated and will stop being called in a future version.` という警告が出る。現時点では `grad` の実装でも動作するが、PyTensor は内部的に新しい `pullback` フックへの移行を進めている。

---

#### `pytensor.gradient.verify_grad(fun, pt, ...)`

**用途**: 数値微分(有限差分)と解析的勾配(`Op.grad` で定義した式)を比較し、カスタム `Op` の勾配実装が正しいかを自動検証する。

**シグネチャ**: `verify_grad(fun, pt, n_tests=2, rng=None, eps=None, out_type=None, abs_tol=None, rel_tol=None, mode=None, cast_to_output_type=False, no_debug_ref=True)`

**使用例**:
```python
import numpy as np
from pytensor.gradient import verify_grad
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply

class WrongDoubleOp(Op):
    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = np.asarray(x * 2)

    def grad(self, inputs, output_grads):
        (gz,) = output_grads
        return [gz * 3]  # わざと間違った勾配(正しくは *2)

wrong_op = WrongDoubleOp()
rng = np.random.default_rng(0)
verify_grad(wrong_op, [np.array([1.0, 2.0, 3.0])], rng=rng)
```
実行結果:
```
pytensor.gradient.GradientError: GradientError: numeric gradient and analytic gradient exceed tolerance:
        At position 2 of argument 0 with shape (3,),
            val1 = 1.622921      ,  val2 = 1.081947
            abs. error = 0.540974,  abs. tolerance = 0.000100
            rel. error = 0.200000,  rel. tolerance = 0.000100
```

**注意点・落とし穴**:
- `pt` 引数(第2引数)は `pytensor.tensor` モジュールのエイリアスとは別物で、検証したい入力値の numpy 配列のリスト。慣習的な引数名が `pt` になっているため `import pytensor.tensor as pt` と名前が衝突する点に注意(この辞書の他の例と同様に `pt` を tensor モジュールのエイリアスとして使っている場合、`verify_grad` 呼び出し時は位置引数で渡すか変数名を変えるとよい)。
- 上記の例では `grad` をわざと `gz * 3`(正しくは `gz * 2`)にしており、`verify_grad` が数値微分との差(誤差)を検出して `GradientError` を送出することを実機で確認した。正しい勾配(`gz * 2`)に直すと例外は発生しない。

---

### 勾配の応用

#### `pytensor.gradient.disconnected_grad(x)`

**用途**: グラフの一部を「勾配計算の対象外」として明示的に切り離す。ある変数がコスト関数の計算には使われるが、その経路については逆伝播させたくない場合に使う(stop-gradient に相当)。

**シグネチャ**: `disconnected_grad(x)`

**使用例**:
```python
import pytensor
import pytensor.tensor as pt
from pytensor.gradient import disconnected_grad

x = pt.scalar('x')
y = x ** 2
y_blocked = disconnected_grad(y)
z = y_blocked + x   # z = stop_grad(x**2) + x
g = pytensor.grad(z, x)
print(g.eval({x: 3.0}))
```
実行結果:
```
1.0
```
**注意点・落とし穴**:
- `z = x**2 + x` であれば `dz/dx = 2x + 1 = 7.0`(`x=3` のとき)になるはずだが、`disconnected_grad` で `x**2` の経路を切り離しているため、実際の勾配は `x` の項(傾き `1`)のみが伝播し `1.0` になる。この差分が `disconnected_grad` の効果そのもの。

---

#### `pytensor.gradient.grad_not_implemented` / `grad_undefined`

**用途**: カスタム `Op.grad` の中で、特定の入力に対する勾配が「未実装」(`grad_not_implemented`)なのか「数学的に定義できない」(`grad_undefined`)のかを区別しつつ、その入力の勾配計算を試みた時点でエラーを発生させる。

**シグネチャ**: `grad_not_implemented(op, x_pos, x, comment='')` / `grad_undefined(op, x_pos, x, comment='')`

**使用例**:
```python
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from pytensor.gradient import grad_undefined

class RoundOp(Op):
    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = np.asarray(np.round(x))

    def grad(self, inputs, output_grads):
        return [grad_undefined(self, 0, inputs[0], 'round() の勾配はほぼ至る所で0だが整数点で未定義')]

op = RoundOp()
x = pt.vector('x')
y = op(x).sum()
try:
    pytensor.grad(y, x)
except Exception as e:
    print(type(e).__name__, ':', str(e))
```
実行結果:
```
NullTypeGradError : `grad` encountered a NaN. This variable is Null because the grad method for input 0 (x) of the RoundOp op is undefined. round() の勾配はほぼ至る所で0だが整数点で未定義
```

**注意点・落とし穴**:
- `grad_not_implemented`/`grad_undefined` の docstring 上は、それぞれ `NotImplementedError`/`GradUndefinedError` が送出されると説明されているが、検証環境(3.3.0)で実際に `pytensor.grad` を呼んで確認したところ、どちらも送出される例外の型は共通して `pytensor.gradient.NullTypeGradError` だった。エラーメッセージの文面(「not implemented」か「undefined」か、および `comment` で渡した文字列)は使い分けに応じて変わるが、`except` 節で型を分けて捕捉することはできない点に注意。

---

#### `pytensor.gradient.hessian(cost, wrt)`

**用途**: スカラーコスト関数の2階微分(ヘッシアン行列)を計算する。

**シグネチャ**: `hessian(cost, wrt, consider_constant=None, disconnected_inputs='raise')`

**使用例**:
```python
import numpy as np
import pytensor.tensor as pt
from pytensor.gradient import hessian

v = pt.vector('v')
cost = (v ** 2).sum() + v[0] * v[1]
H = hessian(cost, v)
print(H.eval({v: np.array([1.0, 2.0])}))
```
実行結果:
```
[[2. 1.]
 [1. 2.]]
```

**注意点・落とし穴**:
- `cost = v0^2 + v1^2 + v0*v1` の解析的ヘッシアンは `[[2, 1], [1, 2]]` で、実行結果はこれと一致する。要素数が多い `wrt` に対して計算するとコストが `O(次元数^2)` で増えるため、大規模モデルでは PyMC 側でも多用は避けられる。

---

#### `pytensor.gradient.Rop` / `Lop`(前進・後退モード微分)

**用途**: ヤコビアンとベクトルの積(JVP: `Rop`, 前進モード)、およびヤコビアンの転置とベクトルの積(VJP: `Lop`, 後退モード)を、フルのヤコビ行列を陽に作らずに計算する。

**シグネチャ**: `Rop(f, wrt, eval_points, disconnected_outputs='raise', return_disconnected='zero', use_op_rop_implementation=False)` / `Lop(f, wrt, eval_points, consider_constant=None, disconnected_inputs='raise', return_disconnected='zero')`

**使用例**:
```python
import numpy as np
import pytensor.tensor as pt
from pytensor.gradient import Rop, Lop

w = pt.vector('w')
f_expr = w ** 2

ev = pt.vector('ev')
rop = Rop(f_expr, w, ev)
print('Rop ->', rop.eval({w: np.array([1., 2., 3.]), ev: np.array([1., 1., 1.])}))

og = pt.vector('og')
lop = Lop(f_expr, w, og)
print('Lop ->', lop.eval({w: np.array([1., 2., 3.]), og: np.array([1., 1., 1.])}))
```
実行結果:
```
Rop -> [2. 4. 6.]
Lop -> [2. 4. 6.]
```

**注意点・落とし穴**:
- `f = w**2` はヤコビアンが対角行列 `diag(2w)` になるため、`eval_points` がすべて `1` の場合 `Rop`/`Lop` の結果はどちらも `2w`(`[2, 4, 6]`)に一致する。一般の非対称なヤコビアンを持つ関数では `Rop` と `Lop` の結果は一致しない。
- 検証環境(3.3.0)では `Rop`/`Lop` を呼ぶと `FutureWarning: Rop is deprecated, use pushforward instead.` / `FutureWarning: Lop is deprecated, use pullback instead.` が出る。実体は動作するが、PyTensor は新しい名前 `pytensor.gradient.pushforward` / `pullback` への移行を進めており、新規コードではそちらの使用が推奨される。

---

### 別バックエンドへのコンパイル

#### `pytensor.function(..., mode="NUMBA")`

**用途**: 計算グラフを Numba の JIT コンパイラ経由でネイティブコードにコンパイルして実行する。

**使用例**:
```python
import pytensor
import pytensor.tensor as pt
import numpy as np

x = pt.vector('x')
y = pt.exp(x).sum()
f_numba = pytensor.function([x], y, mode='NUMBA')
out = f_numba(np.array([1.0, 2.0, 3.0]))
print(out, type(out))
```
実行結果:
```
30.19287485057736 <class 'numpy.ndarray'>
```

**注意点・落とし穴**:
- 検証環境には `numba`(0.66.0)がインストール済みで、追加設定なしに `mode='NUMBA'` が使えた。戻り値の型は通常の(C/Pythonバックエンドの)`function` と同じ `numpy.ndarray`。
- 下記「既定のリンカ」の項で述べる通り、この検証環境では `mode` を省略した場合の既定コンパイル先が実質的に Numba になっている。

---

#### `pytensor.function(..., mode="JAX")`

**用途**: 計算グラフを JAX の `jit` 経由でコンパイルして実行する。

**使用例**:
```python
import pytensor
import pytensor.tensor as pt
import numpy as np

x = pt.vector('x')
y = pt.exp(x).sum()
f_jax = pytensor.function([x], y, mode='JAX')
out = f_jax(np.array([1.0, 2.0, 3.0]))
print(out, type(out))
```
実行結果:
```
30.192874850577365 <class 'jaxlib._jax.ArrayImpl'>
```

**注意点・落とし穴**:
- 検証環境には `jax`(0.11.1)がインストール済みで動作した。ただし CUDA 対応の `jaxlib` が無い環境のため、実行時に `An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.` という警告が標準エラーに出力される(計算結果には影響しない)。
- `mode='NUMBA'` の場合と異なり、戻り値は `numpy.ndarray` ではなく `jaxlib._jax.ArrayImpl`。さらに実行結果の値も `30.19287485057736`(NUMBA/通常)と `30.192874850577365`(JAX)で最後の桁がわずかに異なった(浮動小数点演算の順序・実装差による丸め誤差)。バックエンド間で bit-exact な一致は保証されない点に注意。

---

#### 既定のリンカ(`pytensor.config.linker`)の実体

**用途**: `pytensor.function` に `mode` を明示しなかった場合に、実際にはどのバックエンドでコンパイルされるのかを確認する。

**使用例**:
```python
import pytensor
from pytensor.compile.mode import get_default_mode

print(pytensor.config.linker)
print(get_default_mode().linker)
```
実行結果:
```
auto
NumbaLinker()
```

**注意点・落とし穴**:
- `pytensor.config.linker` の値は `'auto'` だが、PyTensor 内部の実装(`pytensor/compile/mode.py`)では `linker == "auto"` の場合に `"numba"` へ解決するようハードコードされている。つまり検証環境(3.3.0)では、`mode` を指定せずに `pytensor.function` を呼んだ場合、伝統的な C/Python バックエンドではなく **Numba バックエンドが既定で使われる**。
- この辞書の他の項目(例: `pytensor.function` の基本例)で `mode` を指定せずにコンパイルした関数も、実際には内部で Numba 経由の JIT コンパイルが行われている。初回呼び出し時に Numba のコンパイルコストがかかる、`perform` しか実装していないカスタムOpでは「Numba will use object mode to run ... 's perform method」という警告が出る、といった実務上の影響がある(上記カスタムOpの例で実際に観測した)。
