# numpy 逆引き辞書

numpy 2.4.6 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認)。

## 目次

1. [配列生成・基礎操作](#配列生成基礎操作)
2. [形状操作・結合・分割](#形状操作結合分割)
3. [インデックス・スライシング・検索](#インデックススライシング検索)
4. [ブロードキャスト・ユニバーサル関数](#ブロードキャストユニバーサル関数)
5. [集計・統計](#集計統計)
6. [線形代数](#線形代数)
7. [乱数](#乱数)
8. [数学関数](#数学関数)
9. [dtype・型変換](#dtype型変換)
10. [論理演算・マスク処理](#論理演算マスク処理)
11. [ファイルI/O](#ファイルio)
12. [その他便利関数](#その他便利関数)

---

## 配列生成・基礎操作

### `np.array(...)`

**用途**: Python のリストなどから ndarray を生成する、numpy の最も基本的な関数。

**シグネチャ**: `np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, ndmax=0, like=None)`

**使用例**:
```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(a.dtype, a.shape)
```
実行結果:
```
[[1 2 3]
 [4 5 6]]
int64 (2, 3)
```

**注意点・落とし穴**:
- デフォルトで `copy=True` のため、元のリスト/配列とはメモリを共有しない(明示的にコピーを避けたい場合は `np.asarray` を使う)。
- 整数のみの環境依存で `int64` になるかは OS/プラットフォームに依存する(Windows では `int32` になることがある)。

---

### `np.zeros(shape, ...)` / `np.ones(shape, ...)` / `np.full(shape, fill_value, ...)`

**用途**: 指定した shape をすべて 0・1・任意の値で埋めた配列を作る。

**シグネチャ**:
- `np.zeros(shape, dtype=None, order='C', *, device=None, like=None)`
- `np.ones(shape, dtype=None, order='C', *, device=None, like=None)`
- `np.full(shape, fill_value, dtype=None, order='C', *, device=None, like=None)`

**使用例**:
```python
import numpy as np
print(np.zeros((2, 3)))
print(np.ones((2, 3), dtype=int))
print(np.full((2, 2), 7))
```
実行結果:
```
[[0. 0. 0.]
 [0. 0. 0.]]
[[1 1 1]
 [1 1 1]]
[[7 7]
 [7 7]]
```

**注意点・落とし穴**:
- `dtype` を指定しないと `zeros`/`ones` は既定で `float64` になる(`int` 配列がほしいときは明示的に `dtype=int` を指定する必要がある)。

---

### `np.arange(start, stop, step, ...)`

**用途**: Python 標準の `range` の numpy 版。等間隔の数列を生成する。

**シグネチャ**: `np.arange(start_or_stop, /, stop=None, step=1, *, dtype=None, device=None, like=None)`

**使用例**:
```python
import numpy as np
print(np.arange(0, 10, 2))
print(np.arange(0, 1, 0.25))
```
実行結果:
```
[0 2 4 6 8]
[0.   0.25 0.5  0.75]
```

**注意点・落とし穴**:
- 浮動小数点の `step` を使うと丸め誤差で要素数が意図とずれることがある。連続値を等分したい場合は `np.linspace` の方が安全。

---

### `np.linspace(start, stop, num=50, ...)`

**用途**: 開始値と終了値の間を指定した個数で等分割する(既定で終了値を含む)。

**シグネチャ**: `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None)`

**使用例**:
```python
import numpy as np
print(np.linspace(0, 1, 5))
```
実行結果:
```
[0.   0.25 0.5  0.75 1.  ]
```

**注意点・落とし穴**:
- `np.arange` と違い `num`(要素数)を指定する点に注意。`endpoint=False` にすると終了値を含まなくなる。

---

### `np.eye(N, M=None, k=0, ...)`

**用途**: 単位行列(または対角に1を持つ行列)を生成する。

**シグネチャ**: `np.eye(N, M=None, k=0, dtype=<class 'float'>, order='C', *, device=None, like=None)`

**使用例**:
```python
import numpy as np
print(np.eye(3))
```
実行結果:
```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

---

### `np.zeros_like(a, ...)`

**用途**: 既存配列と同じ shape・dtype を持つゼロ配列を作る。

**シグネチャ**: `np.zeros_like(a, dtype=None, order='K', subok=True, shape=None, *, device=None)`

**使用例**:
```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
print(np.zeros_like(a))
```
実行結果:
```
[[0 0]
 [0 0]]
```

**注意点・落とし穴**:
- 元配列の dtype を引き継ぐため、`a` が int なら結果も int になる(`np.zeros` のようにデフォルトで float にはならない)。同様に `ones_like`, `full_like`, `empty_like` がある。

---

### `ndarray.copy(order='C')`

**用途**: 配列の完全な複製(ディープコピー)を作る。スライスや代入と違い元配列への参照を切る。

**シグネチャ**: `ndarray.copy(order='C')`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
b = a.copy()
b[0] = 99
print(a, b)
```
実行結果:
```
[1 2 3] [99  2  3]
```

**注意点・落とし穴**:
- スライス(`a[1:3]`)はビューを返すことが多く、要素を書き換えると元配列も変わる。独立させたい場合は必ず `.copy()` する。

---

## 形状操作・結合・分割

### `np.reshape(a, shape, ...)` / `ndarray.reshape(*shape)`

**用途**: 配列の要素数を変えずに shape を変更する。

**シグネチャ**: `np.reshape(a, /, shape, order='C', *, copy=None)` / `ndarray.reshape(*shape, order='C', copy=None)`

**使用例**:
```python
import numpy as np
a = np.arange(6)
print(a.reshape(2, 3))
print(a.reshape(2, -1))
```
実行結果:
```
[[0 1 2]
 [3 4 5]]
[[0 1 2]
 [3 4 5]]
```

**注意点・落とし穴**:
- `-1` を1箇所だけ指定すると自動で次元数を計算してくれる。
- 可能な限りビュー(元データを共有)を返すが、メモリレイアウト上不可能な場合はコピーになる。

---

### `ndarray.ravel(order='C')` / `ndarray.flatten(order='C')`

**用途**: 多次元配列を1次元に平坦化する。

**シグネチャ**: `np.ravel(a, order='C')` / `ndarray.flatten(order='C')`

**使用例**:
```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
print(a.ravel())

f = a.flatten()
f[0] = 99
print(a)   # 元配列は変わらない
print(f)
```
実行結果:
```
[1 2 3 4]
[[1 2]
 [3 4]]
[99  2  3  4]
```

**注意点・落とし穴**:
- `ravel` は可能な限りビューを返す(元配列を変更してしまう可能性がある)のに対し、`flatten` は常にコピーを返す。安全性重視なら `flatten`、パフォーマンス重視なら `ravel`。

---

### `ndarray.T` / `np.transpose(a, axes=None)`

**用途**: 行列・配列の軸を入れ替える(2次元なら転置)。

**シグネチャ**: `np.transpose(a, axes=None)`

**使用例**:
```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.T)
print(np.transpose(a))
```
実行結果:
```
[[1 4]
 [2 5]
 [3 6]]
[[1 4]
 [2 5]
 [3 6]]
```

**注意点・落とし穴**:
- 結果はビュー。3次元以上では `axes` 引数で並べ替える軸順を明示できる。

---

### `np.concatenate(arrays, axis=0, ...)`

**用途**: 複数の配列を既存の軸に沿って連結する。

**シグネチャ**: `np.concatenate(arrays, /, axis=0, out=None, *, dtype=None, casting='same_kind')`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.concatenate([a, b]))

c = np.array([[1, 2]])
d = np.array([[3, 4]])
print(np.concatenate([c, d], axis=0))
```
実行結果:
```
[1 2 3 4 5 6]
[[1 2]
 [3 4]]
```

**注意点・落とし穴**:
- 連結する軸以外の次元数は一致していないとエラーになる。新しい軸を作りたい場合は `concatenate` ではなく `np.stack` を使う。

---

### `np.stack(arrays, axis=0, ...)`

**用途**: 複数の配列を「新しい軸」を作って積み重ねる(`concatenate` と違い次元が1つ増える)。

**シグネチャ**: `np.stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.stack([a, b]))
print(np.stack([a, b], axis=1))
```
実行結果:
```
[[1 2 3]
 [4 5 6]]
[[1 4]
 [2 5]
 [3 6]]
```

---

### `np.hstack(tup)` / `np.vstack(tup)`

**用途**: 配列を水平方向(列を増やす)・垂直方向(行を増やす)に結合するショートカット。

**シグネチャ**: `np.hstack(tup, *, dtype=None, casting='same_kind')` / `np.vstack(tup, *, dtype=None, casting='same_kind')`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.hstack([a, b]))
print(np.vstack([a, b]))
```
実行結果:
```
[1 2 3 4 5 6]
[[1 2 3]
 [4 5 6]]
```

**注意点・落とし穴**:
- 1次元配列に対して `hstack` は単純連結、`vstack` は2次元化して縦に積む、という挙動の違いを混同しやすい。

---

### `np.split(ary, indices_or_sections, axis=0)` / `np.array_split(...)`

**用途**: 配列を複数個に分割する。

**シグネチャ**: `np.split(ary, indices_or_sections, axis=0)` / `np.array_split(ary, indices_or_sections, axis=0)`

**使用例**:
```python
import numpy as np
a = np.arange(9)
print(np.split(a, 3))

b = np.arange(8)
print(np.array_split(b, 3))
```
実行結果:
```
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]
```

**注意点・落とし穴**:
- `np.split` は等分できない場合(9要素を4分割など)は `ValueError` になる。等分できるか分からない場合は `np.array_split` を使うと余りを自動調整してくれる。

---

### `np.tile(A, reps)` / `np.repeat(a, repeats, axis=None)`

**用途**: `tile` は配列全体を繰り返しタイル状に並べる、`repeat` は各要素をその場で繰り返す。

**シグネチャ**: `np.tile(A, reps)` / `np.repeat(a, repeats, axis=None)`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
print(np.tile(a, 2))
print(np.tile(a, (2, 1)))
print(np.repeat(a, 2))
print(np.repeat(a, [1, 2, 3]))
```
実行結果:
```
[1 2 3 1 2 3]
[[1 2 3]
 [1 2 3]]
[1 1 2 2 3 3]
[1 2 2 3 3 3]
```

**注意点・落とし穴**:
- `tile([1,2,3], 2)` → `[1 2 3 1 2 3]`(全体を繰り返す)だが `repeat([1,2,3], 2)` → `[1 1 2 2 3 3]`(各要素を繰り返す)。名前が紛らわしいので挙動の違いを意識する。
- `repeat` は `repeats` にリストを渡すと要素ごとに異なる回数を指定できる。

---

### `np.expand_dims(a, axis)` / `np.squeeze(a, axis=None)`

**用途**: `expand_dims` はサイズ1の新しい軸を追加、`squeeze` はサイズ1の軸を削除する。

**シグネチャ**: `np.expand_dims(a, axis)` / `np.squeeze(a, axis=None)`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
print(np.expand_dims(a, axis=0).shape)
print(np.expand_dims(a, axis=1).shape)

d = np.array([[[1, 2, 3]]])
print(d.shape, np.squeeze(d).shape)
```
実行結果:
```
(1, 3)
(3, 1)
(1, 1, 3) (3,)
```

**注意点・落とし穴**:
- `axis` を指定しない `squeeze()` はサイズ1の軸を全て削除する。特定の軸だけ潰したい場合は `axis` を指定する(サイズが1でない軸を指定すると `ValueError`)。

---

### `np.newaxis`

**用途**: スライシング内で新しい軸(サイズ1)を追加するための定数(実体は `None`)。

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
print(a[:, np.newaxis].shape)
print(a[np.newaxis, :].shape)
```
実行結果:
```
(3, 1)
(1, 3)
```

**注意点・落とし穴**:
- `np.newaxis is None` は `True`。`expand_dims` と等価だが、スライス構文中で直感的に使えるためブロードキャスト時によく使われる。

---

## インデックス・スライシング・検索

### `np.where(condition, x=None, y=None)`

**用途**: 条件を満たす位置に応じて値を選択する、または条件を満たすインデックスを取得する。

**シグネチャ**: `np.where(condition, x=None, y=None, /)`

**使用例**:
```python
import numpy as np
a = np.array([1, -2, 3, -4, 5])
print(np.where(a > 0, a, 0))
print(np.where(a > 0))
```
実行結果:
```
[1 0 3 0 5]
(array([0, 2, 4]),)
```

**注意点・落とし穴**:
- `x`, `y` を省略すると `np.nonzero(condition)` と同じ、条件を満たすインデックスのタプルを返す(配列そのものではない)点に注意。

---

### `np.nonzero(a)`

**用途**: 0以外の要素のインデックスを軸ごとのタプルで返す。

**シグネチャ**: `np.nonzero(a)`

**使用例**:
```python
import numpy as np
a = np.array([0, 3, 0, 5])
print(np.nonzero(a))
```
実行結果:
```
(array([1, 3]),)
```

---

### `np.argmax(a, axis=None, ...)` / `np.argmin(a, axis=None, ...)`

**用途**: 最大値・最小値を持つ要素のインデックスを取得する。

**シグネチャ**: `np.argmax(a, axis=None, out=None, *, keepdims=<no value>)` / `np.argmin(...)` も同形。

**使用例**:
```python
import numpy as np
a = np.array([[1, 5, 3], [7, 2, 9]])
print(np.argmax(a))
print(np.argmax(a, axis=1))
print(np.argmin(a, axis=0))
```
実行結果:
```
5
[1 2]
[0 1 0]
```

**注意点・落とし穴**:
- `axis` を指定しない場合、`ravel` した後の1次元インデックス(フラットインデックス)を返す。

---

### `np.sort(a, axis=-1, ...)` / `ndarray.sort(...)`

**用途**: 配列をソートする。`np.sort` はコピーを返し、`ndarray.sort()` はその場でソートする。

**シグネチャ**: `np.sort(a, axis=-1, kind=None, order=None, *, stable=None)` / `ndarray.sort(axis=-1, kind=None, order=None, *, stable=None)`

**使用例**:
```python
import numpy as np
a = np.array([3, 1, 4, 1, 5, 9, 2])
print(np.sort(a))
print(a)      # 元配列は変わらない
a.sort()
print(a)      # in-place でソートされる

b = np.array([[3, 1], [2, 4]])
print(np.sort(b, axis=0))
```
実行結果:
```
[1 1 2 3 4 5 9]
[3 1 4 1 5 9 2]
[1 1 2 3 4 5 9]
[[2 1]
 [3 4]]
```

**注意点・落とし穴**:
- 多次元配列では既定 `axis=-1`(最後の軸)でソートされる。全体を1次元としてソートしたい場合は `axis=None` を指定するか `ravel()` してから使う。

---

### `np.argsort(a, ...)`

**用途**: ソート後の並び順になるインデックス配列を返す(値そのものは並べ替えない)。

**シグネチャ**: `np.argsort(a, axis=-1, kind=None, order=None, *, stable=None)`

**使用例**:
```python
import numpy as np
a = np.array([3, 1, 4, 1, 5])
idx = np.argsort(a)
print(idx)
print(a[idx])
```
実行結果:
```
[1 3 0 2 4]
[1 1 3 4 5]
```

---

### `np.unique(ar, ...)`

**用途**: 重複を除いたユニークな値を昇順で返す。件数や出現位置も同時に取得できる。

**シグネチャ**: `np.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True, sorted=True)`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 2, 3, 1, 4])
u, counts = np.unique(a, return_counts=True)
print(u)
print(counts)
```
実行結果:
```
[1 2 3 4]
[2 2 1 1]
```

**注意点・落とし穴**:
- 戻り値は既定でソート済み。元の出現順を知りたい場合は `return_index=True` で最初に出現したインデックスを取得する。

---

### `np.searchsorted(a, v, side='left', ...)`

**用途**: ソート済み配列に対し、値を挿入すべき位置(インデックス)を二分探索で求める。

**シグネチャ**: `np.searchsorted(a, v, side='left', sorter=None)`

**使用例**:
```python
import numpy as np
a = np.array([1, 3, 5, 7, 9])
print(np.searchsorted(a, 6))
print(np.searchsorted(a, [0, 4, 10]))
```
実行結果:
```
3
[0 2 5]
```

**注意点・落とし穴**:
- `a` が事前にソートされていることが前提。ソートされていないと結果は不定(エラーにはならない)。

---

### `np.take(a, indices, axis=None, ...)`

**用途**: 指定したインデックス群の要素を取り出す(ファンシーインデックスの関数版)。

**シグネチャ**: `np.take(a, indices, axis=None, out=None, mode='raise')`

**使用例**:
```python
import numpy as np
a = np.array([10, 20, 30, 40])
print(np.take(a, [0, 2, 3]))
print(a[[0, 2, 3]])  # 等価な書き方
```
実行結果:
```
[10 30 40]
[10 30 40]
```

---

### `np.isin(element, test_elements, ...)`

**用途**: 各要素が指定した集合に含まれるかを真偽値配列で返す。

**シグネチャ**: `np.isin(element, test_elements, assume_unique=False, invert=False, *, kind=None)`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3, 4, 5])
print(np.isin(a, [2, 4]))
```
実行結果:
```
[False  True False  True False]
```

---

### ブールインデックス(boolean indexing)

**用途**: 条件式から得られる真偽値配列を使い、条件を満たす要素の抽出・書き換えを行う。

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
mask = a % 2 == 0
print(a[mask])
a[mask] = 0
print(a)
```
実行結果:
```
[2 4 6]
[1 0 3 0 5 0]
```

**注意点・落とし穴**:
- ブールインデックスによる抽出結果はコピー(ビューではない)。一方で代入 `a[mask] = 0` は元配列を直接書き換える。

---

## ブロードキャスト・ユニバーサル関数

### ブロードキャスト(broadcasting)

**用途**: shape の異なる配列同士を、暗黙にサイズを合わせて要素ごとの演算を行う numpy の中核ルール。

**使用例**:
```python
import numpy as np
a = np.array([[1], [2], [3]])   # shape (3, 1)
b = np.array([10, 20, 30])       # shape (3,)
print(a + b)
print((a + b).shape)
```
実行結果:
```
[[11 21 31]
 [12 22 32]
 [13 23 33]]
(3, 3)
```

**注意点・落とし穴**:
- ブロードキャスト規則: 末尾の軸から比較し、サイズが同じか、どちらかが1であれば揃えられる。それ以外は `ValueError: operands could not be broadcast together` になる。

---

### `np.clip(a, a_min, a_max, ...)`

**用途**: 配列の値を指定範囲内に収める(範囲外の値は境界値に置き換える)。

**シグネチャ**: `np.clip(a, a_min=<no value>, a_max=<no value>, out=None, *, min=<no value>, max=<no value>, **kwargs)`

**使用例**:
```python
import numpy as np
a = np.array([1, 5, 10, 15, 20])
print(np.clip(a, 5, 15))
```
実行結果:
```
[ 5  5 10 15 15]
```

**注意点・落とし穴**:
- numpy 2.x では `a_min`/`a_max` に加えて `min`/`max` キーワードも使える(将来的に `a_min`/`a_max` は非推奨方向)。

---

### `np.vectorize(pyfunc, ...)`

**用途**: 通常の Python 関数を要素ごとに適用できる ufunc 風の関数に変換する。

**シグネチャ**: `np.vectorize(pyfunc=<no value>, otypes=None, doc=None, excluded=None, cache=False, signature=None)`

**使用例**:
```python
import numpy as np

def f(x):
    return x**2 if x % 2 == 0 else -x

vf = np.vectorize(f)
print(vf(np.array([1, 2, 3, 4, 5])))
```
実行結果:
```
[-1  4 -3 16 -5]
```

**注意点・落とし穴**:
- 内部的には単なる Python ループのラッパーであり、真の ufunc のような高速化は行われない。あくまで「書きやすさ」のための道具であり、可能なら本物のベクトル演算(ufunc・ブロードキャスト)に書き換えた方が高速。

---

## 集計・統計

### `np.sum(a, axis=None, ...)` / `np.mean(a, axis=None, ...)`

**用途**: 合計・平均を計算する。`axis` を指定すると特定の軸方向に集約する。

**シグネチャ**: `np.sum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)` / `np.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)`

**使用例**:
```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
print(np.mean(a))
print(np.mean(a, axis=1))
```
実行結果:
```
21
[5 7 9]
[ 6 15]
3.5
[2. 5.]
```

**注意点・落とし穴**:
- `axis=0` は「行方向に潰す(列ごとに集約)」、`axis=1` は「列方向に潰す(行ごとに集約)」であり直感と逆に感じやすいので注意。

---

### `np.std(a, ...)` / `np.var(a, ...)`

**用途**: 標準偏差・分散を計算する。

**シグネチャ**: `np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>, mean=<no value>, correction=<no value>)`(`np.var` も同形)

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3, 4, 5])
print(np.std(a))
print(np.var(a))
```
実行結果:
```
1.4142135623730951
2.0
```

**注意点・落とし穴**:
- 既定 `ddof=0`(母集団の標準偏差/分散)。標本標準偏差(不偏推定量)がほしい場合は `ddof=1` を指定する必要がある(pandas の `.std()` は逆に既定 `ddof=1` なので混同注意)。

---

### `np.median(a, ...)` / `np.percentile(a, q, ...)`

**用途**: 中央値・パーセンタイル値を計算する。

**シグネチャ**: `np.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)` / `np.percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, weights=None)`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
print(np.median(a))
print(np.percentile(a, 25))
print(np.percentile(a, [25, 50, 75]))
```
実行結果:
```
3.5
2.25
[2.25 3.5  4.75]
```

**注意点・落とし穴**:
- `q` はリストで複数同時に渡せる。要素数が偶数の場合、中央値は中央2値の平均になる(上記例で3と4の間なので3.5)。

---

### `np.min(a, ...)` / `np.max(a, ...)`

**用途**: 最小値・最大値を求める。

**シグネチャ**: `np.min(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)`(`np.max` も同形)

**使用例**:
```python
import numpy as np
a = np.array([[1, 5, 3], [7, 2, 9]])
print(np.min(a), np.max(a))
print(np.min(a, axis=0))
print(np.max(a, axis=1))
```
実行結果:
```
1 9
[1 2 3]
[5 9]
```

**注意点・落とし穴**:
- 配列に `NaN` が含まれると `np.min`/`np.max` は `NaN` を返してしまう。NaN を無視したい場合は `np.nanmin`/`np.nanmax` を使う。

---

### `np.cumsum(a, ...)` / `np.cumprod(a, ...)`

**用途**: 累積和・累積積を計算する。

**シグネチャ**: `np.cumsum(a, axis=None, dtype=None, out=None)` / `np.cumprod(a, axis=None, dtype=None, out=None)`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3, 4])
print(np.cumsum(a))
print(np.cumprod(a))
```
実行結果:
```
[ 1  3  6 10]
[ 1  2  6 24]
```

---

### `np.corrcoef(x, y=None, ...)`

**用途**: ピアソンの相関係数行列を計算する。

**シグネチャ**: `np.corrcoef(x, y=None, rowvar=True, *, dtype=None)`

**使用例**:
```python
import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
print(np.corrcoef(x, y))
```
実行結果:
```
[[1.         0.77459667]
 [0.77459667 1.        ]]
```

**注意点・落とし穴**:
- 既定 `rowvar=True` は「各行を1つの変数」として扱う。列方向に変数を並べたデータフレーム的な配列を渡す場合は `rowvar=False` を指定しないと誤った相関行列になる。

---

### `np.diff(a, n=1, ...)`

**用途**: 隣接要素の差分を計算する。

**シグネチャ**: `np.diff(a, n=1, axis=-1, prepend=<no value>, append=<no value>)`

**使用例**:
```python
import numpy as np
a = np.array([1, 3, 6, 10])
print(np.diff(a))
print(np.diff(a, n=2))
```
実行結果:
```
[2 3 4]
[1 1]
```

**注意点・落とし穴**:
- `n` 回差分を繰り返す。出力の長さは入力より `n` だけ短くなる。

---

## 線形代数

### `np.dot(a, b)` / `a @ b` / `np.matmul(x1, x2)`

**用途**: 行列積・ベクトル内積を計算する。

**シグネチャ**: `np.dot(a, b, out=None)` / `np.matmul(x1, x2, /, out=None, *, axes=<no value>, axis=<no value>, keepdims=False, casting='same_kind', order='K', dtype=None, subok=True, signature=None)`

**使用例**:
```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.dot(a, b))
print(a @ b)

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(np.dot(v1, v2))
```
実行結果:
```
[[19 22]
 [43 50]]
[[19 22]
 [43 50]]
32
```

**注意点・落とし穴**:
- 2次元配列同士では `np.dot` と `@`(`np.matmul`)は同じ結果になるが、3次元以上のバッチ処理では挙動が異なる(`matmul` はブロードキャストしてバッチ行列積を行うが `dot` は異なる規則になる)。新しいコードでは `@`/`matmul` を使うのが推奨。

---

### `np.linalg.inv(a)`

**用途**: 正方行列の逆行列を計算する。

**シグネチャ**: `np.linalg.inv(a)`

**使用例**:
```python
import numpy as np
a = np.array([[1., 2.], [3., 4.]])
inv = np.linalg.inv(a)
print(inv)
print(a @ inv)
```
実行結果:
```
[[-2.   1. ]
 [ 1.5 -0.5]]
[[1.0000000e+00 0.0000000e+00]
 [8.8817842e-16 1.0000000e+00]]
```

**注意点・落とし穴**:
- 浮動小数点演算のため `a @ inv` は厳密な単位行列にはならず、`8.88e-16` のような微小な誤差が残る。
- 特異行列(行列式が0)に対しては `LinAlgError: Singular matrix` が発生する。

---

### `np.linalg.solve(a, b)`

**用途**: 連立一次方程式 `a @ x = b` を解く(逆行列を経由するより高速・安定)。

**シグネチャ**: `np.linalg.solve(a, b)`

**使用例**:
```python
import numpy as np
A = np.array([[3, 1], [1, 2]], dtype=float)
b = np.array([9, 8], dtype=float)
x = np.linalg.solve(A, b)
print(x)
```
実行結果:
```
[2. 3.]
```

**注意点・落とし穴**:
- 連立方程式を解く目的なら `np.linalg.inv(A) @ b` より `np.linalg.solve(A, b)` の方が数値的に安定かつ高速。

---

### `np.linalg.norm(x, ord=None, axis=None, ...)`

**用途**: ベクトル・行列のノルム(既定でユークリッドノルム/フロベニウスノルム)を計算する。

**シグネチャ**: `np.linalg.norm(x, ord=None, axis=None, keepdims=False)`

**使用例**:
```python
import numpy as np
v = np.array([3, 4])
print(np.linalg.norm(v))

m = np.array([[1, 2], [3, 4]])
print(np.linalg.norm(m))
print(np.linalg.norm(m, axis=1))
```
実行結果:
```
5.0
5.477225575051661
[2.23606798 5.        ]
```

**注意点・落とし穴**:
- `axis` を指定すると行/列ごとのベクトルノルムを一括計算できる(上記は各行のユークリッドノルム)。

---

### `np.linalg.eig(a)`

**用途**: 正方行列の固有値・固有ベクトルを計算する。

**シグネチャ**: `np.linalg.eig(a)`

**使用例**:
```python
import numpy as np
a = np.array([[2, 0], [0, 3]])
w, v = np.linalg.eig(a)
print(w)
print(v)
```
実行結果:
```
[2. 3.]
[[1. 0.]
 [0. 1.]]
```

**注意点・落とし穴**:
- 戻り値 `v` は「列ベクトル」が固有ベクトル(`v[:, i]` が `w[i]` に対応する固有ベクトル)。行と間違えやすい。
- 対称行列であることが分かっている場合は、より安定・高速な `np.linalg.eigh` を使う方がよい。

---

### `np.linalg.det(a)`

**用途**: 正方行列の行列式を計算する。

**シグネチャ**: `np.linalg.det(a)`

**使用例**:
```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
print(np.linalg.det(a))
```
実行結果:
```
-2.0000000000000004
```

**注意点・落とし穴**:
- 理論値は `-2` だが浮動小数点演算により `-2.0000000000000004` のような誤差が生じる。厳密な整数判定をしたい場合は丸め処理や許容誤差比較が必要。

---

## 乱数

### `np.random.default_rng(seed=None)`

**用途**: 新しい乱数生成器(Generator)を作る、numpy 推奨の乱数 API のエントリポイント。

**シグネチャ**: `np.random.default_rng(seed=None)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(42)
print(rng.random(3))
print(rng.integers(0, 10, size=5))
```
実行結果:
```
[0.77395605 0.43887844 0.85859792]
[0 6 2 0 5]
```

**注意点・落とし穴**:
- 古い `np.random.seed()` / `np.random.rand()` 等のグローバル状態 API は legacy 扱いであり、新しいコードでは `default_rng` による `Generator` インスタンスの使用が推奨される。
- 同じ `seed` を渡せば再現性のある乱数列が得られる。

---

### `rng.random(size=None)`

**用途**: `[0, 1)` の一様乱数を生成する。

**シグネチャ**: `rng.random(size=None, dtype=<class 'numpy.float64'>, out=None)`

**使用例**: 上記 `default_rng` の例を参照(`rng.random(3)` → `[0.77395605 0.43887844 0.85859792]`)。

---

### `rng.integers(low, high=None, size=None, ...)`

**用途**: 整数の乱数を生成する。

**シグネチャ**: `rng.integers(low, high=None, size=None, dtype=<class 'numpy.int64'>, endpoint=False)`

**使用例**: 上記参照(`rng.integers(0, 10, size=5)` → `[0 6 2 0 5]`)。

**注意点・落とし穴**:
- 既定 `endpoint=False` のため `high` は含まれない(`[low, high)`)。`endpoint=True` にすると `high` を含む閉区間になる。
- 旧 API の `np.random.randint` とほぼ同じ役割だが、`Generator` 系列では `integers` という名前になっている。

---

### `rng.normal(loc=0.0, scale=1.0, size=None)`

**用途**: 正規分布(ガウス分布)に従う乱数を生成する。

**シグネチャ**: `rng.normal(loc=0.0, scale=1.0, size=None)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(0)
print(rng.normal(loc=0, scale=1, size=5))
```
実行結果:
```
[ 0.12573022 -0.13210486  0.64042265  0.10490012 -0.53566937]
```

---

### `rng.choice(a, size=None, replace=True, ...)`

**用途**: 配列から要素をランダムに選択する。

**シグネチャ**: `rng.choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(1)
a = np.array(['a', 'b', 'c', 'd'])
print(rng.choice(a, size=3))
print(rng.choice(a, size=3, replace=False))
```
実行結果:
```
['b' 'c' 'd']
['b' 'a' 'd']
```

**注意点・落とし穴**:
- 既定 `replace=True`(重複ありの復元抽出)。重複なしで抽出したい場合は `replace=False` を明示する。`p` 引数で各要素の選択確率(重み)を指定できる。

---

### `rng.shuffle(x, axis=0)`

**用途**: 配列をその場でシャッフルする(戻り値はなく、破壊的に変更する)。

**シグネチャ**: `rng.shuffle(x, axis=0)`

**使用例**:
```python
import numpy as np
rng = np.random.default_rng(2)
a = np.arange(5)
rng.shuffle(a)
print(a)
```
実行結果:
```
[2 4 3 0 1]
```

**注意点・落とし穴**:
- `np.random.permutation` と違い、`shuffle` は元の配列を直接書き換える(戻り値は `None`)。コピーが欲しい場合は `rng.permutation(a)` を使う。

---

## 数学関数

### `np.sqrt(x)` / `np.exp(x)` / `np.log(x)` / `np.abs(x)`

**用途**: 平方根・指数関数・自然対数・絶対値を要素ごとに計算する ufunc。

**シグネチャ**(すべて共通形): `np.sqrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)`(`exp`, `log`, `abs` も同形)

**使用例**:
```python
import numpy as np
a = np.array([1, 4, 9, 16])
print(np.sqrt(a))
print(np.exp(np.array([0, 1, 2])))
print(np.log(np.array([1, np.e, np.e**2])))
print(np.abs(np.array([-1, -2, 3])))
```
実行結果:
```
[1. 2. 3. 4.]
[1.         2.71828183 7.3890561 ]
[0. 1. 2.]
[1 2 3]
```

**注意点・落とし穴**:
- `np.log` は自然対数(底 `e`)。常用対数は `np.log10`、2進対数は `np.log2` を使う。
- 負の数に `np.sqrt` を適用すると `RuntimeWarning: invalid value encountered in sqrt` が出て `nan` になる(複素数がほしい場合は `np.emath.sqrt` や複素数配列を使う)。

---

### `np.round(a, decimals=0)`

**用途**: 指定した小数桁数に丸める。

**シグネチャ**: `np.round(a, decimals=0, out=None)`

**使用例**:
```python
import numpy as np
a = np.array([1.234, 2.567, -1.5, 2.5])
print(np.round(a, 2))
print(np.round(np.array([0.5, 1.5, 2.5])))
```
実行結果:
```
[ 1.23  2.57 -1.5   2.5 ]
[0. 2. 2.]
```

**注意点・落とし穴**:
- numpy の丸めは「銀行丸め(round half to even)」を採用しており、`0.5→0`、`1.5→2`、`2.5→2` のように .5 は最も近い偶数に丸められる。Python 標準の `round()` と同じ方式だが、単純な四捨五入を期待していると意外な結果になりやすい。

---

### `np.sin(x)` / `np.cos(x)`

**用途**: 三角関数を要素ごとに計算する(ラジアン単位)。

**使用例**:
```python
import numpy as np
a = np.array([0, np.pi / 2, np.pi])
print(np.sin(a))
print(np.cos(a))
```
実行結果:
```
[0.0000000e+00 1.0000000e+00 1.2246468e-16]
[ 1.000000e+00  6.123234e-17 -1.000000e+00]
```

**注意点・落とし穴**:
- `sin(π)` や `cos(π/2)` の理論値は 0 だが、`π` 自体が浮動小数点近似のため `1.22e-16` のような極小な誤差が残る。

---

### `np.power(x1, x2)` / `np.mod(x1, x2)`

**用途**: べき乗・剰余を要素ごとに計算する。

**シグネチャ**: `np.power(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)`(`np.mod` も同形)

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3, 4])
print(np.power(a, 2))
print(np.mod(a, 2))
```
実行結果:
```
[ 1  4  9 16]
[1 0 1 0]
```

**注意点・落とし穴**:
- `np.mod` は `%` 演算子と等価。負の数に対する剰余の符号は被除数ではなく除数の符号に従う(Python の `%` と同じ挙動)。

---

## dtype・型変換

### `ndarray.astype(dtype, ...)`

**用途**: 配列の dtype を変換したコピーを作る。

**シグネチャ**: `ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)`

**使用例**:
```python
import numpy as np
a = np.array([1.7, 2.3, 3.9])
b = a.astype(int)
print(b, b.dtype)
```
実行結果:
```
[1 2 3] int64
```

**注意点・落とし穴**:
- float から int への変換は四捨五入ではなく「切り捨て(0方向への切り捨て)」になる(`1.7→1`, `3.9→3`、四捨五入ではない)。
- 既定 `copy=True` のため常に新しい配列が作られる(元と同じ dtype でも明示的にコピーしたくない場合は `copy=False` を指定できるが、変換が不要なとき以外は無視されることがある)。

---

### `dtype` 引数によるデータ型の明示指定

**用途**: 配列生成時にビット幅・符号の有無などを明示的に指定する。

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3], dtype=np.float32)
print(a.dtype)
b = np.array([1, 2, 3], dtype='int8')
print(b.dtype)
```
実行結果:
```
float32
int8
```

**注意点・落とし穴**:
- `dtype='int8'` のように文字列でも指定できる。桁あふれ(オーバーフロー)に注意(例: `int8` は -128〜127 の範囲しか表現できず、超えるとラップアラウンドする)。

---

### `np.asarray(a, dtype=None, ...)`

**用途**: 入力をndarrayに変換する。すでにndarrayであれば(dtype等が一致する限り)コピーせずそのまま返す。

**シグネチャ**: `np.asarray(a, dtype=None, order=None, *, device=None, copy=None, like=None)`

**使用例**:
```python
import numpy as np
lst = [1, 2, 3]
a = np.asarray(lst)
print(a, type(a))

arr = np.array([1, 2, 3])
print(np.asarray(arr) is arr)
```
実行結果:
```
[1 2 3] <class 'numpy.ndarray'>
True
```

**注意点・落とし穴**:
- `np.array` は既定でコピーする(`copy=True`)のに対し、`np.asarray` は既に ndarray かつ dtype が一致していればコピーしない。不要なコピーを避けたい関数の入力受け取り部でよく使われる。

---

## 論理演算・マスク処理

### `np.all(a, ...)` / `np.any(a, ...)`

**用途**: 配列の全要素/いずれかの要素が真であるかを判定する。

**シグネチャ**: `np.all(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>)`(`np.any` も同形)

**使用例**:
```python
import numpy as np
a = np.array([True, True, False])
print(np.all(a), np.any(a))

b = np.array([[1, 0], [1, 1]])
print(np.all(b, axis=1))
```
実行結果:
```
False True
[False  True]
```

---

### `np.logical_and` / `np.logical_or` / `np.logical_not`

**用途**: 要素ごとの論理演算。

**使用例**:
```python
import numpy as np
a = np.array([True, False, True])
b = np.array([True, True, False])
print(np.logical_and(a, b))
print(np.logical_or(a, b))
print(np.logical_not(a))
```
実行結果:
```
[ True False False]
[ True  True  True]
[False  True False]
```

**注意点・落とし穴**:
- ブールインデックスの条件を組み合わせるとき、Python の `and`/`or` は使えない(単一の bool しか扱えないため)。`&`/`|`(ビット演算子、要優先順位の括弧)または `np.logical_and`/`np.logical_or` を使う必要がある。

---

### `np.isnan(x)` / `np.isinf(x)` / `np.isfinite(x)`

**用途**: 各要素が NaN・無限大・有限値かどうかを判定する。

**使用例**:
```python
import numpy as np
a = np.array([1, np.nan, np.inf, -np.inf, 3])
print(np.isnan(a))
print(np.isinf(a))
print(np.isfinite(a))
```
実行結果:
```
[False  True False False False]
[False False  True  True False]
[ True False False False  True]
```

**注意点・落とし穴**:
- `nan == nan` は `False` になるため、NaN の判定には `==` ではなく必ず `np.isnan` を使う。

---

## ファイルI/O

### `np.save(file, arr)` / `np.load(file)`

**用途**: 単一の ndarray をバイナリ形式(`.npy`)で保存・読み込みする。

**シグネチャ**: `np.save(file, arr, allow_pickle=True)` / `np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=10000)`

**使用例**:
```python
import numpy as np
import tempfile, os

a = np.array([1, 2, 3])
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'a.npy')
    np.save(path, a)
    b = np.load(path)
    print(b)
```
実行結果:
```
[1 2 3]
```

**注意点・落とし穴**:
- `np.save` は既定 `allow_pickle=True` だが `np.load` は既定 `allow_pickle=False`(セキュリティ上の理由)。オブジェクト配列を保存した場合、読み込み時に `allow_pickle=True` を明示しないとエラーになる。

---

### `np.savez(file, *args, **kwds)`

**用途**: 複数の ndarray を1つの `.npz` ファイル(名前付き)にまとめて保存する。

**シグネチャ**: `np.savez(file, *args, allow_pickle=True, **kwds)`

**使用例**:
```python
import numpy as np
import tempfile, os

a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'data.npz')
    np.savez(path, x=a, y=b)
    loaded = np.load(path)
    print(list(loaded.keys()))
    print(loaded['x'])
    print(loaded['y'])
```
実行結果:
```
['x', 'y']
[1 2 3]
[[1 2]
 [3 4]]
```

**注意点・落とし穴**:
- `np.load` が返すのは `NpzFile` オブジェクト(遅延読み込みの辞書的オブジェクト)であり、直接 ndarray ではない。キー名でアクセスする必要がある。圧縮したい場合は `np.savez_compressed` を使う。

---

### `np.savetxt(fname, X, ...)` / `np.loadtxt(fname, ...)`

**用途**: テキスト形式(CSV等)で配列を保存・読み込みする。

**シグネチャ**: `np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)` / `np.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, ...)`

**使用例**:
```python
import numpy as np
import tempfile, os

a = np.array([[1.0, 2.0], [3.0, 4.0]])
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'a.csv')
    np.savetxt(path, a, delimiter=',')
    b = np.loadtxt(path, delimiter=',')
    print(b)
```
実行結果:
```
[[1. 2.]
 [3. 4.]]
```

**注意点・落とし穴**:
- バイナリ形式(`.npy`/`.npz`)より遅く、ファイルサイズも大きくなりがちだが、他ツールとの互換性(Excel等でも開ける)が利点。既定の `dtype=float` のため整数として読み込みたい場合は明示的に指定する。

---

## その他便利関数

### `np.meshgrid(*xi, indexing='xy')`

**用途**: 複数の1次元座標配列から格子点(グリッド)座標の配列を生成する。

**シグネチャ**: `np.meshgrid(*xi, copy=True, sparse=False, indexing='xy')`

**使用例**:
```python
import numpy as np
x = np.array([1, 2, 3])
y = np.array([4, 5])
X, Y = np.meshgrid(x, y)
print(X)
print(Y)
```
実行結果:
```
[[1 2 3]
 [1 2 3]]
[[4 4 4]
 [5 5 5]]
```

**注意点・落とし穴**:
- 既定 `indexing='xy'`(デカルト座標系、行列的には転置された配置)。行列計算的な `(行, 列)` の並びで扱いたい場合は `indexing='ij'` を指定する。

---

### `np.pad(array, pad_width, mode='constant', ...)`

**用途**: 配列の周囲に指定した幅・方法でパディング(埋め合わせ)を追加する。

**シグネチャ**: `np.pad(array, pad_width, mode='constant', **kwargs)`

**使用例**:
```python
import numpy as np
a = np.array([1, 2, 3])
print(np.pad(a, (2, 1), mode='constant', constant_values=0))
```
実行結果:
```
[0 0 1 2 3 0]
```

**注意点・落とし穴**:
- `pad_width=(2, 1)` は「前に2個、後ろに1個」を意味する。`mode` には `'constant'` の他に `'edge'`(端の値で埋める)、`'reflect'`(反転コピー)などがある。

---

### `np.apply_along_axis(func1d, axis, arr)`

**用途**: 指定した軸に沿って任意の1次元関数を適用する。

**シグネチャ**: `np.apply_along_axis(func1d, axis, arr, *args, **kwargs)`

**使用例**:
```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.apply_along_axis(np.sum, 0, a))
print(np.apply_along_axis(np.sum, 1, a))
```
実行結果:
```
[5 7 9]
[ 6 15]
```

**注意点・落とし穴**:
- 内部的には Python レベルのループであり、`np.sum(a, axis=...)` のような組み込み集約関数が使える場合はそちらの方が高速。`apply_along_axis` は組み込みでは表現できない独自の1次元関数を適用したい場合の最終手段として使う。

---

### `ndarray.ndim` / `ndarray.size` / `ndarray.shape`

**用途**: 配列の次元数・全要素数・各軸のサイズを取得する属性。

**使用例**:
```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.ndim, a.size, a.shape)
```
実行結果:
```
2 6 (2, 3)
```

---

### `order` 引数によるメモリレイアウト(`'C'` / `'F'`)

**用途**: 配列を行優先(C言語順)か列優先(Fortran順)で並べ替える。

**使用例**:
```python
import numpy as np
a = np.arange(6)
print(a.reshape(2, 3, order='C'))
print(a.reshape(2, 3, order='F'))
```
実行結果:
```
[[0 1 2]
 [3 4 5]]
[[0 2 4]
 [1 3 5]]
```

**注意点・落とし穴**:
- 既定は `'C'`(行優先、C言語と同じ)。MATLAB や Fortran 由来のコードを移植する際は `'F'`(列優先)を意識する必要がある場合がある。
