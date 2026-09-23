# jax 逆引き辞書

jax 0.11.1 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認。付随するニューラルネット構築ライブラリ flax(0.12.9)・最適化ライブラリ optax(0.2.8)も同環境で検証)。

## 目次

1. [配列生成・基礎操作(jnp基礎)](#配列生成基礎操作jnp基礎)
2. [自動微分(grad/value_and_grad/jacobian)](#自動微分gradvalue_and_gradjacobian)
3. [JIT・コンパイル(jit)](#jitコンパイルjit)
4. [ベクトル化・並列化(vmap/pmap)](#ベクトル化並列化vmappmap)
5. [乱数(jax.random)](#乱数jaxrandom)
6. [制御フロー(lax.cond/scan/while_loop/fori_loop)](#制御フローlaxcondscanwhile_loopfori_loop)
7. [Pytree操作(tree_util)](#pytree操作tree_util)
8. [Flaxニューラルネット構築(flax.linen基礎)](#flaxニューラルネット構築flaxlinen基礎)
9. [最適化(optax)](#最適化optax)
10. [その他(デバッグ・numpy相互運用)](#その他デバッグnumpy相互運用)

---

## 配列生成・基礎操作(jnp基礎)

### `jnp.array(object, ...)`

**用途**: Python のリストなどから jax の配列(`jax.Array`)を生成する、numpy の `np.array` に相当する基本関数。

**シグネチャ**: `jax.numpy.array(object, dtype=None, copy=True, order='K', ndmin=0, *, device=None)`

**使用例**:
```python
import jax.numpy as jnp
a = jnp.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(a.dtype, a.shape)
```
実行結果:
```
[[1 2 3]
 [4 5 6]]
int32 (2, 3)
```

**注意点・落とし穴**:
- numpy は整数配列を既定で `int64` にするが、jax は既定で `int32`(浮動小数点は `float32`)になる。これは後述の x64 無効化がデフォルトのため。
- API は numpy とほぼ同じだが、返る型は `numpy.ndarray` ではなく `jaxlib._jax.ArrayImpl`(`jax.Array`)であり、後述のとおり不変(immutable)。

---

### `jnp.zeros(shape, ...)` / `jnp.ones(shape, ...)` / `jnp.full(shape, fill_value, ...)`

**用途**: 指定した shape をすべて 0・1・任意の値で埋めた配列を作る。numpy と同じインターフェース。

**使用例**:
```python
import jax.numpy as jnp
print(jnp.zeros((2, 3)))
print(jnp.ones((2, 3), dtype=int))
print(jnp.full((2, 2), 7))
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

---

### `jnp.arange(...)` / `jnp.linspace(...)`

**用途**: 等間隔の数列を生成する。numpy の `np.arange`/`np.linspace` と同じ役割。

**使用例**:
```python
import jax.numpy as jnp
print(jnp.arange(0, 10, 2))
print(jnp.linspace(0, 1, 5))
```
実行結果:
```
[0 2 4 6 8]
[0.   0.25 0.5  0.75 1.  ]
```

---

### 配列の不変性(immutability)と `.at[].set()`

**用途**: jax の配列は生成後に要素を書き換えられない(関数型プログラミングの前提)。代わりに `x.at[idx].set(value)` などの `.at[]` API が「更新後のコピー」を返す。

**使用例**:
```python
import jax.numpy as jnp
b = jnp.array([1, 2, 3])
try:
    b[0] = 99
except TypeError as e:
    print("TypeError:", e)

c = b.at[0].set(99)
print("b (unchanged):", b)
print("c:", c)
```
実行結果:
```
TypeError: JAX arrays are immutable and do not support in-place item assignment. Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html
b (unchanged): [1 2 3]
c: [99  2  3]
```

**注意点・落とし穴**:
- `x[idx] = y` は `TypeError` になる。numpy から移行する際に最もハマりやすい違い。
- `.at[idx].set/add/multiply/max/min(...)` はいずれも新しい配列を返す(元の `b` は変わらない)。ループ内で毎回コピーが発生するため、大量の要素を逐次更新する処理は `lax.scan` などに書き換えた方が効率的。

---

### dtype のデフォルト(`int32`/`float32`)と x64 無効化

**用途**: jax はデフォルトで64bit浮動小数点(`float64`)・64bit整数(`int64`)を無効化しており、明示的に `dtype=jnp.float64` を指定しても `float32` に丸められる。

**使用例**:
```python
import jax.numpy as jnp
x = jnp.array([1.0], dtype=jnp.float64)
print(x.dtype)
```
実行結果:
```
float32
```
(実行時に以下の `UserWarning` が出る)
```
UserWarning: Explicitly requested dtype float64 requested in array is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable.
```

**注意点・落とし穴**:
- 64bit精度が必要な場合は、プログラム冒頭で `jax.config.update("jax_enable_x64", True)` を呼ぶか、環境変数 `JAX_ENABLE_X64=1` を設定する必要がある(numpy から移植したコードで精度が足りず結果がズレる原因の定番)。

---

### `jax.device_put(x, device=None)` / `jax.devices()`

**用途**: 配列を明示的に特定のデバイス(CPU/GPU/TPU)に配置する、および利用可能なデバイス一覧を取得する。

**使用例**:
```python
import jax
import jax.numpy as jnp
d = jax.device_put(jnp.array([1, 2, 3]))
print(d, d.devices())
print(jax.devices())
```
実行結果(検証環境はCPUのみ):
```
[1 2 3] {CpuDevice(id=0)}
[CpuDevice(id=0)]
```

**注意点・落とし穴**:
- 検証環境は GPU 未搭載(CUDA 版 jaxlib 未インストール)のため `jax.devices()` は CPU のみを返す。`import jax` 時に「An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.」という警告が出る。

---

## 自動微分(grad/value_and_grad/jacobian)

### `jax.grad(fun, argnums=0, has_aux=False, ...)`

**用途**: スカラーを返す関数の勾配(導関数)を計算する関数を返す、jax の中核機能。

**シグネチャ**: `jax.grad(fun, argnums=0, has_aux=False, holomorphic=False, allow_int=False, reduce_axes=())`

**使用例**:
```python
import jax

def f(x):
    return x ** 2 + 3 * x + 1

g = jax.grad(f)
print(g(2.0))
```
実行結果:
```
7.0
```

**複数引数・`has_aux` の例**:
```python
import jax
import jax.numpy as jnp

def loss(w, b, x):
    return jnp.sum((w * x + b) ** 2)

gw, gb = jax.grad(loss, argnums=(0, 1))(2.0, 1.0, jnp.array([1.0, 2.0]))
print(gw, gb)

def f_aux(x):
    y = x ** 2
    return y, {"x": x}

g3, aux = jax.grad(f_aux, has_aux=True)(jnp.array(3.0))
print(g3, aux)
```
実行結果:
```
26.0 16.0
6.0 {'x': Array(3., dtype=float32, weak_type=True)}
```

**注意点・落とし穴**:
- `fun` は必ずスカラー(0次元)を返す必要がある。ベクトル/行列を返す関数に `grad` を使うと `TypeError` になる(その場合は後述の `jacfwd`/`jacrev` を使う)。
- `has_aux=True` にすると `(勾配, 補助出力)` のタプルを返す。補助情報(ロス以外のメトリクスなど)を一緒に取り出したいときに使う。
- 既定 `argnums=0` は第1引数についての勾配のみを計算する。複数引数の勾配が欲しい場合はタプルで指定する。

---

### `jax.value_and_grad(fun, argnums=0, has_aux=False, ...)`

**用途**: 関数の値と勾配を1回のトレースでまとめて計算する(`grad` だけだと値を再計算する必要があるため効率的)。

**シグネチャ**: `jax.value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False, allow_int=False, reduce_axes=())`

**使用例**:
```python
import jax

def f(x):
    return x ** 2 + 3 * x + 1

v, g = jax.value_and_grad(f)(2.0)
print(v, g)
```
実行結果:
```
11.0 7.0
```

---

### `jax.jacfwd(fun, ...)` / `jax.jacrev(fun, ...)`

**用途**: ベクトル値関数のヤコビ行列を計算する。`jacfwd` は前進モード、`jacrev` は後退モードの自動微分を使う(入力次元が出力次元より小さいなら `jacfwd`、大きいなら `jacrev` が効率的)。

**シグネチャ**: `jax.jacfwd(fun, argnums=0, has_aux=False, holomorphic=False)` / `jax.jacrev(fun, argnums=0, has_aux=False, holomorphic=False, allow_int=False)`

**使用例**:
```python
import jax
import jax.numpy as jnp

def vec_fn(x):
    return jnp.array([x[0]**2, x[0]*x[1], x[1]**2])

x = jnp.array([1.0, 2.0])
print(jax.jacfwd(vec_fn)(x))
print(jax.jacrev(vec_fn)(x))
```
実行結果:
```
[[2. 0.]
 [2. 1.]
 [0. 4.]]
[[2. 0.]
 [2. 1.]
 [0. 4.]]
```

**注意点・落とし穴**:
- 数学的には同じヤコビ行列を返すが、内部計算方式が違うだけで結果は一致する。どちらを使うかは入出力の次元比によるパフォーマンスの問題。

---

### `jax.hessian(fun, ...)`

**用途**: スカラー関数のヘッセ行列(2階微分)を計算する。内部的には `jacfwd(jacrev(fun))` に相当する。

**シグネチャ**: `jax.hessian(fun, argnums=0, has_aux=False, holomorphic=False)`

**使用例**:
```python
import jax
import jax.numpy as jnp

def scalar_fn(x):
    return jnp.sum(x ** 3)

print(jax.hessian(scalar_fn)(jnp.array([1.0, 2.0])))
```
実行結果:
```
[[ 6.  0.]
 [ 0. 12.]]
```

---

### `jax.lax.stop_gradient(x)`

**用途**: 計算グラフ上でその値を「定数」として扱い、そこから先へ勾配を流さないようにする。

**使用例**:
```python
import jax

def f_stop(x):
    return x * jax.lax.stop_gradient(x ** 2)

print(jax.grad(f_stop)(3.0))
```
実行結果:
```
9.0
```

**注意点・落とし穴**:
- `f(x) = x * stop_gradient(x**2)` は数値的には `x**3` と同じ値になるが、勾配計算では `x**2` の部分を定数(9.0)として扱うため `grad` の結果は `3*x**2`(=27)ではなく `x**2`(=9.0)になる。ターゲットネットワークの固定や、一部の勾配だけ止めたいカスタム損失でよく使われる。

---

## JIT・コンパイル(jit)

### `jax.jit(fun, static_argnums=None, donate_argnums=None, ...)`

**用途**: 関数を XLA でコンパイルし、繰り返し呼び出す際の実行を高速化する。jax の性能を引き出す最重要機能。

**シグネチャ**: `jax.jit(fun, /, *, in_shardings=UnspecifiedValue, out_shardings=UnspecifiedValue, static_argnums=None, static_argnames=None, donate_argnums=None, donate_argnames=None, keep_unused=False, device=None, backend=None, inline=False, compiler_options=None)`

**使用例**:
```python
import jax
import jax.numpy as jnp

def slow_f(x):
    return jnp.sum(x ** 2 + jnp.sin(x))

fast_f = jax.jit(slow_f)
x = jnp.arange(5.0)
print(slow_f(x))
print(fast_f(x))
```
実行結果:
```
31.135086
31.135086
```

**注意点・落とし穴**:
- `jit` された関数の中で、トレース対象の値(トレーサ)に対して Python の `if x > 0:` のような通常の分岐を書くと `TracerBoolConversionError` になる(下記 `jax.make_jaxpr` の項も参照)。動的な条件分岐が必要な場合は後述の `lax.cond`/`lax.while_loop` を使う。
- 初回呼び出し時にトレース・コンパイルが走るためオーバーヘッドがあり、2回目以降のみ高速化の恩恵がある。引数の shape/dtype が変わるたびに再コンパイルされる点にも注意。

---

### `jax.jit` の `static_argnums`

**用途**: 実行時に値が変わらない(かつ Python 側の分岐などに使う)引数を「静的引数」として指定し、その値ごとに個別にコンパイルさせる。

**使用例**:
```python
import jax
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def power(x, n):
    return x ** n

print(power(2.0, 3))
print(power(2.0, 4))
```
実行結果:
```
8.0
16.0
```

**注意点・落とし穴**:
- 静的引数はハッシュ可能である必要があり、値ごとに別々のコンパイル済みバージョンがキャッシュされる(`n` の値が変わるたびに再コンパイルが走る)。配列のような可変・非ハッシュ可能な値は静的引数にできない。

---

### `jax.make_jaxpr(fun, ...)`

**用途**: 関数をトレースし、jax の中間表現(jaxpr)を人間が読める形で出力する。`jit` が実際に何をコンパイルしているかを確認するデバッグ用ツール。

**シグネチャ**: `jax.make_jaxpr(fun, static_argnums=(), axis_env=None, return_shape=False)`

**使用例**:
```python
import jax
import jax.numpy as jnp

def f(x, y):
    return jnp.sin(x) * y

print(jax.make_jaxpr(f)(1.0, 2.0))
```
実行結果:
```
{ lambda ; a:f32[] b:f32[]. let c:f32[] = sin a; d:f32[] = mul c b in (d,) }
```

**Python の動的分岐が `jit` 内でエラーになる例**:
```python
import jax
import jax.numpy as jnp

@jax.jit
def bad(x):
    if x > 0:
        return x
    return -x

try:
    bad(jnp.array(1.0))
except Exception as e:
    print(type(e).__name__, str(e)[:60])
```
実行結果:
```
TracerBoolConversionError Attempted boolean conversion of traced array with shape bool[].
```

---

### `jit` の `donate_argnums`(バッファドネーション)

**用途**: 入力バッファのメモリを再利用してよいことをコンパイラに伝え、メモリコピーを省略して高速化・省メモリ化する。

**使用例**:
```python
import jax
import jax.numpy as jnp

@jax.jit
def add_one(x):
    return x + 1

donated = jax.jit(add_one, donate_argnums=0)
x = jnp.array([1.0, 2.0, 3.0])
y = donated(x)
print(y)
print(x)
```
実行結果:
```
[2. 3. 4.]
RuntimeError: Array has been deleted with shape=float32[3].
```

**注意点・落とし穴**:
- `donate_argnums` を指定した引数は、呼び出し後に元の配列(`x`)へアクセスすると `RuntimeError: Array has been deleted` になる。ドネーションした入力は二度と使わないことが確実な場合のみ使う。

---

## ベクトル化・並列化(vmap/pmap)

### `jax.vmap(fun, in_axes=0, out_axes=0, ...)`

**用途**: 明示的なバッチループを書かずに、関数を「バッチ次元に沿って自動でベクトル化」する。

**シグネチャ**: `jax.vmap(fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None, spmd_axis_name=None, sum_match=False)`

**使用例**:
```python
import jax
import jax.numpy as jnp

def dot(a, b):
    return jnp.dot(a, b)

batched_a = jnp.arange(6.0).reshape(2, 3)
batched_b = jnp.arange(6.0).reshape(2, 3)
print(jax.vmap(dot)(batched_a, batched_b))

mat = jnp.arange(6.0).reshape(2, 3)
vec = jnp.array([1.0, 2.0, 3.0])

def mv(m, v):
    return m @ v

print(jax.vmap(mv, in_axes=(0, None))(mat, vec))
```
実行結果:
```
[ 5. 50.]
[ 8. 26.]
```

**注意点・落とし穴**:
- `in_axes` はどの引数のどの軸をバッチ軸として扱うかを指定する。`None` を指定した引数はバッチ処理せず、そのまま(共通の値として)全バッチに使い回される(上記例では `vec` をブロードキャスト的に使い回している)。
- 単なる Python の `for` ループより高速な上、`jit` と組み合わせられる。

---

### `jax.pmap(fun, axis_name=None, in_axes=0, ...)`

**用途**: 複数デバイス(複数GPU/TPUコアなど)に処理を分散して並列実行する。`vmap` の「複数デバイス版」。

**シグネチャ**: `jax.pmap(fun, axis_name=None, *, in_axes=0, out_axes=0, static_broadcasted_argnums=(), devices=None, backend=None, axis_size=None, donate_argnums=())`

**使用例**(検証環境はCPU1個のみのため、環境変数 `XLA_FLAGS=--xla_force_host_platform_device_count=4` でCPUを4デバイスとして扱わせてデモ):
```python
import jax
import jax.numpy as jnp

print(jax.devices())
xs = jnp.arange(4.0)
f = jax.pmap(lambda x: x ** 2)
print(f(xs))
```
実行結果:
```
[CpuDevice(id=0), CpuDevice(id=1), CpuDevice(id=2), CpuDevice(id=3)]
[0. 1. 4. 9.]
```

**注意点・落とし穴**:
- 通常のマシンでは物理デバイス数(この検証環境ではCPU1個)しか見えないため、`pmap` に渡す配列の先頭次元はデバイス数と一致していなければならない(`ValueError` になる)。GPU/TPUが複数枚ある環境で真価を発揮する機能で、CPU一個の環境で試す場合は上記のように `XLA_FLAGS` でデバイス数を疑似的に増やす必要がある。
- 新しいコードでは、より柔軟な `jax.experimental.shard_map` や `jit` + `Sharding` の組み合わせが推奨されつつあるが、`pmap` は依然広く使われているシンプルなデータ並列APIとして現役。

---

## 乱数(jax.random)

### `jax.random.key(seed)` / `jax.random.PRNGKey(seed)`

**用途**: jax の乱数生成器の「鍵」を作る。jax の乱数はグローバル状態を持たず、鍵を明示的に受け渡す関数型スタイル。

**シグネチャ**: `jax.random.key(seed, *, impl=None, dtype=None)` / `jax.random.PRNGKey(seed, *, impl=None)`

**使用例**:
```python
import jax
key = jax.random.key(0)
print(key)

legacy_key = jax.random.PRNGKey(0)
print(legacy_key, legacy_key.dtype)
```
実行結果:
```
Array((), dtype=key<fry>) overlaying:
[0 0]
[0 0] uint32
```

**注意点・落とし穴**:
- `jax.random.key()` が現行の推奨API。戻り値は中身が見えない「不透明な鍵」型(`dtype=key<fry>`)であり、誤って算術演算に使ってしまう事故を防げる。
- `jax.random.PRNGKey()` は後方互換のための従来API で、生の `uint32` 配列(shape `(2,)`)を返す。この検証環境(jax 0.11.1)では両者とも非推奨警告(`DeprecationWarning`)は出ないが、公式ドキュメントは新規コードで `key()` を使うことを推奨している。
- 同じ `seed` からは常に同じ鍵(＝同じ乱数列)が得られる(再現性のため)。

---

### `jax.random.split(key, num=2)`

**用途**: 1つの鍵から、互いに独立な複数の新しい鍵を生成する。乱数を使うたびに鍵を分岐させるのが jax 流の作法。

**シグネチャ**: `jax.random.split(key, num=2)`

**使用例**:
```python
import jax
key = jax.random.key(42)
k1, k2 = jax.random.split(key)
print(k1)
print(k2)

k3, *subkeys = jax.random.split(key, 4)
print(len(subkeys))
```
実行結果:
```
Array((), dtype=key<fry>) overlaying:
[1832780943  270669613]
Array((), dtype=key<fry>) overlaying:
[  64467757 2916123636]
3
```

**注意点・落とし穴**:
- `num` を指定すると `num` 個の鍵を含む配列が返る(上記では4個に分割し、先頭を新しい親鍵、残り3個をサブ鍵として使う典型パターン)。

---

### `jax.random.normal(key, shape=(), ...)` / `jax.random.uniform(key, shape=(), ...)`

**用途**: 正規分布・一様分布に従う乱数を生成する。

**シグネチャ**: `jax.random.normal(key, shape=(), dtype=None, *, out_sharding=None)` / `jax.random.uniform(key, shape=(), dtype=None, minval=0.0, maxval=1.0, *, out_sharding=None)`

**使用例**:
```python
import jax
key = jax.random.key(0)
print(jax.random.normal(key, shape=(3,)))

key2 = jax.random.key(0)
print(jax.random.uniform(key2, shape=(3,)))
```
実行結果:
```
[ 1.6226422   2.0252647  -0.43359444]
[0.947667   0.9785799  0.33229148]
```

---

### `jax.random.randint(key, shape, minval, maxval, ...)`

**用途**: 指定範囲の整数乱数を生成する。

**シグネチャ**: `jax.random.randint(key, shape, minval, maxval, dtype=None, *, out_sharding=None)`

**使用例**:
```python
import jax
key = jax.random.key(1)
print(jax.random.randint(key, shape=(5,), minval=0, maxval=10))
```
実行結果:
```
[6 7 0 3 8]
```

**注意点・落とし穴**:
- numpy の `rng.integers` と同様、`maxval` は含まれない半開区間 `[minval, maxval)`。

---

### `jax.random.choice(key, a, shape=(), replace=True, ...)` / `jax.random.permutation(key, x, ...)`

**用途**: `choice` は配列から要素をランダムに選択、`permutation` は配列をランダムな順序に並べ替える(numpyの `rng.choice`/`rng.permutation` に相当)。

**シグネチャ**: `jax.random.choice(key, a, shape=(), replace=True, p=None, axis=0, mode=None)` / `jax.random.permutation(key, x, axis=0, independent=False, *, out_sharding=None)`

**使用例**:
```python
import jax
import jax.numpy as jnp

key = jax.random.key(2)
print(jax.random.choice(key, jnp.array([10, 20, 30, 40]), shape=(3,)))

key2 = jax.random.key(3)
print(jax.random.permutation(key2, jnp.arange(5)))
```
実行結果:
```
[40 10 20]
[0 4 2 1 3]
```

**注意点・落とし穴**:
- `permutation` は numpy の `shuffle` と違い、元配列を書き換えず新しい配列を返す(jax の配列は不変なので当然だが、numpy との対比で覚えておくとよい)。

---

### 鍵の使い回しの注意(乱数の再現・非再現)

**用途**: 同じ鍵を使い回すと「毎回同じ乱数」が返る、というjax乱数の重要な性質を確認する。

**使用例**:
```python
import jax
k = jax.random.key(0)
print(jax.random.normal(k, shape=(2,)))
print(jax.random.normal(k, shape=(2,)))
```
実行結果:
```
[1.6226422 2.0252647]
[1.6226422 2.0252647]
```

**注意点・落とし穴**:
- numpy の `Generator` はメソッド呼び出しのたびに内部状態が進むため呼ぶたびに違う乱数が出るが、jax は同じ鍵を渡せば必ず同じ結果になる(副作用がない関数型設計のため)。ループ内で複数回別々の乱数が必要な場合は、必ず `jax.random.split` で毎回新しい鍵を作って渡す必要がある。

---

## 制御フロー(lax.cond/scan/while_loop/fori_loop)

### `jax.lax.cond(pred, true_fun, false_fun, *operands)`

**用途**: `jit`/`grad` の中でも使える、条件分岐の関数版。Python の `if` と違い両方の分岐がトレース可能な形で扱われる。

**使用例**:
```python
from jax import lax

def f(x):
    return lax.cond(x > 0, lambda x: x * 2, lambda x: x * -1, x)

print(f(3.0))
print(f(-3.0))
```
実行結果:
```
6.0
3.0
```

**注意点・落とし穴**:
- `true_fun`/`false_fun` は同じ引数を受け取り、同じ shape/dtype の値を返す必要がある(型が食い違うと `TypeError`)。`jit` 内で `if x > 0:` のような通常の分岐が書けない(トレーサの真偽値評価ができない)ことの直接の解決策がこの `lax.cond`。

---

### `jax.lax.scan(f, init, xs=None, length=None, ...)`

**用途**: 「キャリー(carry)を引き継ぎながら配列に沿ってループする」処理を、Pythonの `for` ループより高速にコンパイルできる形で書く。RNN やキャリー付き累積計算の定番。

**シグネチャ**: `jax.lax.scan(f, init, xs=None, length=None, reverse=False, unroll=1, _split_transpose=False)`

**使用例**:
```python
from jax import lax
import jax.numpy as jnp

def cumsum_step(carry, x):
    new_carry = carry + x
    return new_carry, new_carry

final, ys = lax.scan(cumsum_step, 0.0, jnp.array([1.0, 2.0, 3.0, 4.0]))
print(final)
print(ys)
```
実行結果:
```
10.0
[ 1.  3.  6. 10.]
```

**注意点・落とし穴**:
- `f` は `(carry, x) -> (new_carry, y)` という形でなければならない。最終的な `carry` と、各ステップの `y` を積み重ねた配列の両方が返る。
- Python の `for` ループを書いてから `jit` するよりも、`scan` を使う方がコンパイル時間・実行効率ともに有利(`for` ループは展開されて巨大なグラフになりがち)。

---

### `jax.lax.while_loop(cond_fun, body_fun, init_val)`

**用途**: 条件式が真である間ループを続ける、`while` の関数版。ループ回数が実行時まで決まらない場合に使う。

**使用例**:
```python
from jax import lax

def cond_fn(state):
    i, x = state
    return i < 5

def body_fn(state):
    i, x = state
    return i + 1, x * 2

print(lax.while_loop(cond_fn, body_fn, (0, 1.0)))
```
実行結果:
```
(Array(5, dtype=int32, weak_type=True), Array(32., dtype=float32, weak_type=True))
```

**注意点・落とし穴**:
- ループ回数が動的(実行時のデータに依存)なため、`while_loop` は `jax.grad` で微分できない(勾配が必要な反復処理には固定回数の `lax.fori_loop` や `lax.scan` を使う)。

---

### `jax.lax.fori_loop(lower, upper, body_fun, init_val)`

**用途**: 固定回数(`lower` から `upper` まで)のループ。`for i in range(lower, upper): val = body_fun(i, val)` に相当する。

**使用例**:
```python
from jax import lax

def body(i, acc):
    return acc + i

print(lax.fori_loop(0, 5, body, 0))
```
実行結果:
```
10
```

**注意点・落とし穴**:
- `body_fun` は `(i, val) -> val` の形。回数が静的に決まっている場合は `while_loop` より `fori_loop` の方が意図が明確で、内部的には `scan` に近い形で最適化されうる。

---

## Pytree操作(tree_util)

### `jax.tree_util.tree_map(f, tree, *rest, is_leaf=None)`

**用途**: 辞書やリストなどネストした構造(pytree)の「葉(leaf)」全部に関数を一括適用する。パラメータ辞書全体への演算(更新・スケーリングなど)に多用される。

**シグネチャ**: `jax.tree_util.tree_map(f, tree, *rest, is_leaf=None)`

**使用例**:
```python
import jax
import jax.numpy as jnp

params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.5)}
doubled = jax.tree_util.tree_map(lambda x: x * 2, params)
print(doubled)

a = {"x": 1, "y": 2}
b = {"x": 10, "y": 20}
print(jax.tree_util.tree_map(lambda u, v: u + v, a, b))
```
実行結果:
```
{'b': Array(1., dtype=float32, weak_type=True), 'w': Array([2., 4.], dtype=float32)}
{'x': 11, 'y': 22}
```

**注意点・落とし穴**:
- 辞書を pytree として扱う際、jax はキーをソートして処理する(出力の `{'b': ..., 'w': ...}` の順序が入力の `{"w": ..., "b": ...}` と変わっている)。順序に依存したコードを書かないよう注意。
- 複数の pytree(`*rest`)を渡すと、対応する葉同士に `f` を適用する(構造が一致していないと `ValueError`)。

---

### `jax.tree_util.tree_leaves(tree)` / `jax.tree_util.tree_structure(tree)`

**用途**: pytree から葉のリストだけを取り出す、または構造(骨組み)だけを取り出す。

**シグネチャ**: `jax.tree_util.tree_leaves(tree, is_leaf=None)` / `jax.tree_util.tree_structure(tree, is_leaf=None)`

**使用例**:
```python
import jax
import jax.numpy as jnp

params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.5)}
print(jax.tree_util.tree_leaves(params))
print(jax.tree_util.tree_structure(params))
```
実行結果:
```
[Array(0.5, dtype=float32, weak_type=True), Array([1., 2.], dtype=float32)]
PyTreeDef({'b': *, 'w': *})
```

---

### `jax.tree_util.tree_flatten(tree)` / `jax.tree_util.tree_unflatten(treedef, leaves)`

**用途**: pytree を「葉のリスト」と「構造情報(treedef)」に分解し、後で同じ構造に組み立て直す。`vmap`/`jit` などの内部実装でも使われる基礎API。

**シグネチャ**: `jax.tree_util.tree_flatten(tree, is_leaf=None)` / `jax.tree_util.tree_unflatten(treedef, leaves)`

**使用例**:
```python
import jax
import jax.numpy as jnp

params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.5)}
flat, treedef = jax.tree_util.tree_flatten(params)
print(flat)
print(treedef)

rebuilt = jax.tree_util.tree_unflatten(treedef, [x + 1 for x in flat])
print(rebuilt)
```
実行結果:
```
[Array(0.5, dtype=float32, weak_type=True), Array([1., 2.], dtype=float32)]
PyTreeDef({'b': *, 'w': *})
{'b': Array(1.5, dtype=float32, weak_type=True), 'w': Array([2., 3.], dtype=float32)}
```

**注意点・落とし穴**:
- `tree_unflatten` に渡す葉のリストは `tree_flatten` が返した順序(＝ソート済みキー順)と同じ順序・個数でなければならない。

---

## Flaxニューラルネット構築(flax.linen基礎)

flax は jax 上に構築されたニューラルネットワーク構築ライブラリで、`flax.linen`(通常 `nn` としてimport)モジュールが標準的なレイヤー定義APIを提供する。

### `flax.linen.Module` と `@nn.compact`

**用途**: レイヤーを組み合わせたモデルをクラスとして定義する。`@nn.compact` を付けたメソッド内でサブレイヤーをインラインに宣言できる。

**使用例**:
```python
import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        return x

model = MLP(features=2)
key = jax.random.key(0)
x = jnp.ones((1, 4))
params = model.init(key, x)
print(jax.tree_util.tree_map(lambda p: p.shape, params))

out = model.apply(params, x)
print(out)
```
実行結果:
```
{'params': {'Dense_0': {'bias': (8,), 'kernel': (4, 8)}, 'Dense_1': {'bias': (2,), 'kernel': (8, 2)}}}
[[0.24190402 0.22841509]]
```

**注意点・落とし穴**:
- flax の Module はパラメータを自分で保持しない(pytorchの `nn.Module` と違う設計)。パラメータは `model.init(key, x)` が返す pytree(辞書)として外部で管理し、推論・学習時は毎回 `model.apply(params, x)` に明示的に渡す。この「モデル(処理ロジック)とパラメータ(状態)の分離」が jax/flax 流の関数型設計。
- レイヤー名(`Dense_0`, `Dense_1`)は `@nn.compact` メソッド内で宣言された順に自動採番される。

---

### `flax.linen.Dense(features, ...)`

**用途**: 全結合層(線形層)。`features` は出力次元数。

**シグネチャ**(主要引数を抜粋。実際の `inspect.signature` にはデフォルト値の詳細な関数オブジェクト表記が含まれ長大なため簡略化): `nn.Dense(features, use_bias=True, dtype=None, param_dtype=jnp.float32, precision=None, kernel_init=<variance_scaling初期化>, bias_init=<zeros>, name=None)`

**使用例**:
```python
import jax
import jax.numpy as jnp
import flax.linen as nn

dense = nn.Dense(features=3)
key = jax.random.key(1)
x = jnp.ones((2, 4))
params = dense.init(key, x)
print(jax.tree_util.tree_map(lambda p: p.shape, params))
print(dense.apply(params, x))
```
実行結果:
```
{'params': {'bias': (3,), 'kernel': (4, 3)}}
[[ 1.3005596  -0.50973934  0.66124046]
 [ 1.3005596  -0.50973934  0.66124046]]
```

---

### 活性化関数(`flax.linen.relu` など)

**用途**: flax.linen は `jax.nn` の活性化関数をそのまま `nn.relu` などの名前で再エクスポートしている。

**使用例**:
```python
import jax.numpy as jnp
import flax.linen as nn
print(nn.relu(jnp.array([-1.0, 0.0, 2.0])))
```
実行結果:
```
[0. 0. 2.]
```

---

## 最適化(optax)

### `optax.adam(learning_rate, ...)` / `optax.sgd(learning_rate, ...)`

**用途**: 勾配降下系の最適化アルゴリズム(Adam・SGD)を表す「勾配変換(GradientTransformation)」オブジェクトを作る。flax などのパラメータ更新に使う。

**シグネチャ**(型ヒントが `jax.Array | numpy.ndarray | ... | float | complex` という長い Union のため主要部分のみ抜粋): `optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None, *, nesterov=False)` / `optax.sgd(learning_rate, momentum=None, nesterov=False, accumulator_dtype=None)`

**使用例**:
```python
import jax
import jax.numpy as jnp
import optax

params = {"w": jnp.array([1.0, 2.0])}
tx = optax.adam(learning_rate=0.1)
opt_state = tx.init(params)

def loss_fn(params):
    return jnp.sum((params["w"] - 3.0) ** 2)

for step in range(3):
    grads = jax.grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    print(step, params["w"], loss_fn(params))
```
実行結果:
```
0 [1.0999993 2.0999994] 4.4200034
1 [1.1998318 2.1995862] 3.8812675
2 [1.2993746 2.2984118] 3.3843527
```

**注意点・落とし穴**:
- optax は「オプティマイザの状態」と「パラメータ更新」を分離した設計(`tx.init` → `tx.update` → `optax.apply_updates` の3段階)。pytorchの `optimizer.step()` のような単一メソッドではない点に注意。

---

### `optax.apply_updates(params, updates)`

**用途**: `tx.update()` が返した更新量(updates)を実際のパラメータに加算し、新しいパラメータ pytree を返す。

**シグネチャ**(型ヒントは `optax.apply_updates` も同様に長い Union のため抜粋): `optax.apply_updates(params, updates)`

**使用例**: 上記 `optax.adam` の例を参照(`params = optax.apply_updates(params, updates)`)。

**注意点・落とし穴**:
- 単純に `params + updates`(pytreeの各葉を加算)を行うだけの関数。学習率のスケーリングなどは `tx.update` 側(オプティマイザの勾配変換)で既に適用済みという設計。

---

### flax `TrainState` と optax の組み合わせ

**用途**: `flax.training.train_state.TrainState` は「モデルのapply関数・パラメータ・オプティマイザ」をひとつにまとめて学習ループを簡潔に書けるようにするヘルパークラス。

**シグネチャ**: `TrainState.create(*, apply_fn, params, tx, **kwargs)`

**使用例**:
```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=1)(x)

model = Model()
key = jax.random.key(0)
x = jnp.ones((1, 2))
params = model.init(key, x)

state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optax.adam(0.1),
)

xs = jnp.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
ys = jnp.array([[5.0], [4.0], [12.0]])

def loss_fn(params):
    preds = state.apply_fn(params, xs)
    return jnp.mean((preds - ys) ** 2)

for step in range(3):
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    print(step, loss)
```
実行結果:
```
0 84.33026
1 75.20452
2 66.63527
```

**注意点・落とし穴**:
- `TrainState` 自体は不変(dataclass的なpytree)。`state.apply_gradients(...)` は「更新後の新しい `TrainState`」を返すのであって、`state` を直接書き換えるわけではない(戻り値を必ず変数に代入し直す必要がある)。

---

## その他(デバッグ・numpy相互運用)

### `jax.debug.print(fmt, *args, ...)`

**用途**: `jit`/`vmap`/`lax.scan` などでトレースされたコード内から値を実際に出力する。通常の `print()` はトレース時(コンパイル時)にしか呼ばれず値を表示できないため専用APIが必要。

**シグネチャ**: `jax.debug.print(fmt=None, *args, ordered=False, partitioned=False, skip_format_check=False, **kwargs)`

**使用例**:
```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    jax.debug.print("x = {}", x)
    return x * 2

print(f(jnp.array(3.0)))
```
実行結果:
```
x = 3.0
6.0
```

**通常の `print()` との対比**:
```python
import jax
import jax.numpy as jnp

@jax.jit
def f2(x):
    print("python print:", x)
    return x * 2

print(f2(jnp.array(3.0)))
```
実行結果:
```
python print: JitTracer(~float32[])
6.0
```

**注意点・落とし穴**:
- 通常の `print()` は `jit` のトレース(コンパイル準備)が走る最初の1回だけ呼ばれ、しかも表示されるのは実際の値ではなくトレーサ(`JitTracer(~float32[])`)であり、以降の呼び出しでは一切出力されない。実行時の値を毎回確認したい場合は `jax.debug.print` を使う。

---

### `jax.disable_jit(disable=True)`

**用途**: `jit` によるコンパイル・トレースを一時的に無効化し、通常のPythonループ同様に「実際の値」で1行ずつ実行させる、デバッグ用のコンテキストマネージャ。

**シグネチャ**: `jax.disable_jit(disable=True)`

**使用例**:
```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    print("trace time print:", type(x))
    return x * 2

with jax.disable_jit():
    print(f(jnp.array(3.0)))
```
実行結果:
```
trace time print: <class 'jaxlib._jax.ArrayImpl'>
6.0
```

**注意点・落とし穴**:
- `disable_jit()` 中は `x` が抽象的なトレーサ(`Tracer`)ではなく具体的な `ArrayImpl` になるため、通常の `print(x)` でも実際の値を確認できる。`jit` 内で原因不明のエラーが出たときに、まず `disable_jit()` で普通の実行に戻して問題箇所を切り分けるのが定石のデバッグ手順。

---

### numpy 配列との相互変換(`jnp.asarray` / `np.asarray`)

**用途**: numpy 配列 (`np.ndarray`) と jax 配列 (`jax.Array`) を相互に変換する。

**使用例**:
```python
import numpy as np
import jax.numpy as jnp

a = jnp.array([1, 2, 3])
b = np.asarray(a)
print(type(b), b)

c = np.array([1.0, 2.0])
d = jnp.asarray(c)
print(type(d), d)
```
実行結果:
```
<class 'numpy.ndarray'> [1 2 3]
<class 'jaxlib._jax.ArrayImpl'> [1. 2.]
```

**注意点・落とし穴**:
- `np.asarray(jax配列)` は jax 配列の中身をCPUメモリへコピーして通常の numpy 配列にする(GPU/TPU上の配列であれば暗黙にデバイス間転送が発生する)。大きな配列を頻繁に変換するとパフォーマンスに影響するため、可能な限り `jnp` の API だけで完結させるのが望ましい。
