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
