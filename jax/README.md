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
11. [応用・発展](#応用発展)
    - [低レベル自動微分・カスタム微分ルール(jvp/vjp/custom_jvp/custom_vjp)](#低レベル自動微分カスタム微分ルールjvpvjpcustom_jvpcustom_vjp)
    - [メモリ最適化・実行時チェック(checkpoint/eval_shape/checkify)](#メモリ最適化実行時チェックcheckpointeval_shapecheckify)
    - [高度なPytree操作(register_pytree_node/is_leaf)](#高度なpytree操作register_pytree_nodeis_leaf)
    - [jax.debug応用(callback/breakpoint)](#jaxdebug応用callbackbreakpoint)
    - [Flax/optaxの高度な機能](#flaxoptaxの高度な機能)

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

---

## 応用・発展

ここから先は、基礎的な使い方を一通り押さえた上で扱う、より高度・niche な jax / flax / optax の API を扱う。

### 低レベル自動微分・カスタム微分ルール(jvp/vjp/custom_jvp/custom_vjp)

#### `jax.jvp(fun, primals, tangents, has_aux=False)`

**用途**: 前進モード自動微分の低レベルAPI。`grad`/`jacfwd` は内部でこれを使って実装されている。評価点(`primals`)と入力側の方向(`tangents`)を渡すと、出力値とその方向への微分(方向微分)を同時に返す。

**シグネチャ**: `jax.jvp(fun: 'Callable', primals, tangents, has_aux: 'bool' = False) -> 'tuple[Any, ...]'`

**使用例**:
```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(x) * x

y, y_dot = jax.jvp(f, (2.0,), (1.0,))
print(y, y_dot)
print(jax.grad(f)(2.0))
```
実行結果:
```
1.8185948 0.07700372
0.07700372
```

**注意点・落とし穴**:
- `jax.grad` はスカラー関数の勾配だけを返す高レベルAPIだが、内部的には `jvp`(前進モード)と `vjp`(後退モード)という2つの低レベルAPIの合成で自動微分全体が構成されている。1入力1出力のスカラー関数では、`tangents=(1.0,)` で呼んだ `jvp` の第2戻り値は `grad` の結果と一致する。

---

#### `jax.vjp(fun, *primals, has_aux=False)`

**用途**: 後退モード自動微分の低レベルAPI。関数を評価しつつ、余接ベクトル(コタンジェント)を渡すと勾配を返す関数(vjp関数)を作る。`jax.grad(f)(x)` は本質的に `jax.vjp(f, x)[1](1.0)` と同じ計算をしている。

**シグネチャ**: `jax.vjp(fun: 'Callable', *primals, has_aux: 'bool' = False, reduce_axes=(), saveable_args: 'Any' = True, in_nzs: 'Any' = None) -> 'tuple[Any, Callable] | tuple[Any, Callable, Any]'`

**使用例**:
```python
import jax

def f(x):
    return jax.numpy.sin(x) * x

y, vjp_fn = jax.vjp(f, 2.0)
print(y)
print(vjp_fn(1.0))
```
実行結果:
```
1.8185948
(Array(0.07700372, dtype=float32, weak_type=True),)
```

**注意点・落とし穴**:
- `vjp_fn` は渡した `primals` の数だけの勾配を必ずタプルで返す(単一引数でも `(Array(...),)` のようにタプル)。`custom_vjp` で独自の逆伝播ルールを書く際は、この「入力の数だけの戻り値」という形式に合わせる必要がある。

---

#### `jax.custom_jvp` / `.defjvp(jvp, symbolic_zeros=False)`

**用途**: 関数の前進モード微分規則を手動で定義する。数値的に不安定な微分(0付近での発散など)を避けたい場合や、独自の勾配挙動を実装したい場合に使う。

**シグネチャ**: `jax.custom_jvp(fun=None, nondiff_argnums=(), nondiff_argnames=())`

**使用例**:
```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def f(x):
    return jnp.sin(x)

@f.defjvp
def f_jvp(primals, tangents):
    x, = primals
    t, = tangents
    primal_out = jnp.sin(x)
    tangent_out = jnp.cos(x) * t
    return primal_out, tangent_out

print(f(1.0))
print(jax.grad(f)(1.0))
print(jax.grad(jnp.sin)(1.0))
```
実行結果:
```
0.84147096
0.5403023
0.5403023
```

**注意点・落とし穴**:
- `defjvp` に渡す関数は `(primals, tangents) -> (primal_out, tangent_out)` という決まった形。今回は標準の `sin` と同じ微分規則を手書きしただけなので `jax.grad(jnp.sin)` と一致するが、実運用では数値安定化した近似式などをここに書く。

---

#### `jax.custom_vjp` / `.defvjp(fwd, bwd, symbolic_zeros=False, optimize_remat=False)`

**用途**: 後退モードの微分規則を手動で定義する。勾配クリッピングや straight-through estimator のように、順伝播の値はそのまま通しつつ逆伝播だけを別ルールにしたい場合に使う。

**シグネチャ**: `jax.custom_vjp(fun=None, nondiff_argnums=(), nondiff_argnames=())`

**使用例**(順伝播では値をそのまま通しつつ、逆伝播の勾配だけ `[-0.5, 0.5]` にクリップする):
```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def clip_grad(x, lo, hi):
    return x  # フォワードは恒等関数

def clip_grad_fwd(x, lo, hi):
    return x, (lo, hi)

def clip_grad_bwd(res, g):
    lo, hi = res
    return (jnp.clip(g, lo, hi), None, None)

clip_grad.defvjp(clip_grad_fwd, clip_grad_bwd)

def loss(x):
    return clip_grad(x, -0.5, 0.5) ** 2 * 10

print(jax.grad(loss)(3.0))
print(jax.grad(lambda x: x ** 2 * 10)(3.0))
```
実行結果:
```
0.5
60.0
```

**注意点・落とし穴**:
- `clip_grad` 自体はフォワードでは恒等関数(`x` をそのまま返す)なので、通常なら `x**2*10` の勾配(=`60.0`)になるはずだが、`defvjp` で逆伝播ルールを上書きしているため実際の勾配は `[-0.5, 0.5]` にクリップされた `0.5` になる。フォワードの計算結果とバックワードの勾配計算が完全に独立して定義できることを示す例。
- `bwd` 関数は「微分しない引数」(`lo`, `hi`)に対しても `None` を含めてタプルの要素数を `fwd` への入力引数の数と揃える必要がある。

---

### メモリ最適化・実行時チェック(checkpoint/eval_shape/checkify)

#### `jax.checkpoint(fun, ...)`(`jax.remat` と同一)

**用途**: 逆伝播(勾配計算)の際に中間の活性化値を保存せず、必要になった時点で順伝播を再計算することでメモリ使用量を削減する「勾配チェックポイント」。深いネットワークでメモリが不足する場合の定番のテクニック。`jax.remat` は同じ関数への別名。

**シグネチャ**: `jax.checkpoint(fun: 'Callable', *, prevent_cse: 'bool | Sequence[bool]' = True, policy: 'Callable[..., bool] | None' = None, static_argnums: 'int | tuple[int, ...]' = (), static_argnames: 'str | Iterable[str]' = ()) -> 'Callable'`

**使用例**(`jax.debug.print` でレイヤーが実際に何回「実行」されるかを可視化):
```python
import jax
import jax.numpy as jnp

def layer(x):
    jax.debug.print("layer called with x={}", x)
    return jnp.sin(x)

def f_plain(x):
    return layer(layer(x))

def f_ckpt(x):
    l = jax.checkpoint(layer)
    return l(l(x))

print("--- plain ---")
jax.grad(f_plain)(1.0)
print("--- checkpoint ---")
jax.grad(f_ckpt)(1.0)
```
実行結果:
```
--- plain ---
layer called with x=1.0
layer called with x=0.8414709568023682
--- checkpoint ---
layer called with x=1.0
layer called with x=0.8414709568023682
layer called with x=0.8414709568023682
layer called with x=1.0
```

**注意点・落とし穴**:
- `plain` 版は順伝播で各レイヤーが1回ずつしか呼ばれない(逆伝播は保存済みの中間値を使う)のに対し、`checkpoint` 版は逆伝播の際に順伝播をもう一度re-runしている(呼び出し回数が2倍になっている)ことが分かる。これは「メモリを節約する代わりに計算時間が増える」というトレードオフを実際の挙動として確認できる例。
- メモリが逼迫していないモデルに無闇に使うと、純粋に計算時間だけが増えて損をする。メモリボトルネックが実際にある箇所(深いResNet/Transformerの層など)に限定して使うのが定石。

---

#### `jax.eval_shape(fun, *args, **kwargs)`

**用途**: 関数を実際には実行せず(データも確保せず)、出力の shape/dtype だけを推論する。巨大な配列を扱う前に、メモリを確保せず出力形状だけ先に知りたい場合に使う。

**シグネチャ**: `jax.eval_shape(fun: 'Callable', *args, **kwargs)`

**使用例**:
```python
import jax
import jax.numpy as jnp

def f(x, y):
    return jnp.dot(x, y)

out = jax.eval_shape(f, jnp.zeros((3, 4)), jnp.zeros((4, 5)))
print(out, type(out))

big = jax.eval_shape(
    f,
    jax.ShapeDtypeStruct((1000, 1000), jnp.float32),
    jax.ShapeDtypeStruct((1000, 1000), jnp.float32),
)
print(big)
```
実行結果:
```
ShapeDtypeStruct(shape=(3, 5), dtype=float32) <class 'jax.ShapeDtypeStruct'>
ShapeDtypeStruct(shape=(1000, 1000), dtype=float32)
```

**注意点・落とし穴**:
- 戻り値は実データを持たない `jax.ShapeDtypeStruct`(shape/dtype情報のみ)。引数として実配列の代わりに `jax.ShapeDtypeStruct` をそのまま渡すこともでき、実配列を1つも確保せずに巨大な計算の出力形状だけを一瞬で調べられる。

---

#### `jax.experimental.checkify.checkify(f, errors=...)`

**用途**: NaN の発生やインデックス範囲外アクセスなど、通常は「静かに」処理されてしまうランタイムエラーを明示的にチェックし、検出できるようにするユーティリティ。

**シグネチャ**: `checkify.checkify(f: 'Callable[..., Out]', errors: 'frozenset[ErrorCategory]' = frozenset({<class 'jax._src.checkify.FailedCheckError'>})) -> 'Callable[..., tuple[Error, Out]]'`

**使用例**:
```python
import jax.numpy as jnp
from jax.experimental import checkify

def f(x):
    return jnp.log(x)

checked_f = checkify.checkify(f, errors=checkify.float_checks)
err, out = checked_f(jnp.array(-1.0))
print(out)
print(err.get())
```
実行結果:
```
nan
nan generated by primitive: log.
```

**注意点・落とし穴**:
- `checkify.checkify` で包んだ関数は、戻り値が `(Error, 元の出力)` というタプルに変わる。呼び出し側のコードもそれに合わせて書き換える必要がある。
- `errors` 引数で検出対象の種類を切り替えられる(既定はユーザー定義の `checkify.check` によるアサート失敗のみ。`checkify.float_checks` を渡すと今回のようなNaN/Inf発生も検出できる)。

---

### 高度なPytree操作(register_pytree_node/is_leaf)

#### `jax.tree_util.register_pytree_node(nodetype, flatten_func, unflatten_func, ...)`

**用途**: 独自に定義したPythonクラスをpytreeとして扱えるように登録する。登録すると `jit`/`grad`/`vmap`/`tree_map` など、すべてのpytree対応APIで自作クラスのインスタンスを辞書やリストと同じように(内部の属性ごとに)扱えるようになる。

**シグネチャ**: `jax.tree_util.register_pytree_node(nodetype: 'type[T]', flatten_func: 'Callable[[T], tuple[_Children, _AuxData]]', unflatten_func: 'Callable[[_AuxData, _Children], T]', flatten_with_keys_func=None) -> 'None'`

**使用例**:
```python
import jax
import jax.numpy as jnp

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

def point_flatten(p):
    return (p.x, p.y), None

def point_unflatten(aux_data, children):
    return Point(*children)

jax.tree_util.register_pytree_node(Point, point_flatten, point_unflatten)

p = Point(jnp.array(1.0), jnp.array(2.0))
print(jax.tree_util.tree_leaves(p))
print(jax.tree_util.tree_map(lambda v: v * 2, p))
```
実行結果:
```
[Array(1., dtype=float32, weak_type=True), Array(2., dtype=float32, weak_type=True)]
Point(x=2.0, y=4.0)
```

**注意点・落とし穴**:
- 登録しない場合、自作クラスは「1つの不透明な葉」として扱われ、中の属性(`x`, `y`)には分解されない。
- `flatten_func` は `(children, aux_data)` のタプルを返す必要がある。`aux_data` は微分・vmap の対象にならない「静的な補助情報」で、ハッシュ可能である必要がある(今回は使わないので `None`)。

---

#### `jax.tree_util.register_pytree_node_class`

**用途**: `register_pytree_node` のクラスデコレータ版。クラス自身に `tree_flatten`/`tree_unflatten` メソッドを定義し、デコレータを1行付けるだけでpytree登録できる。

**シグネチャ**: `jax.tree_util.register_pytree_node_class(cls: 'Typ') -> 'Typ'`

**使用例**(登録した自作クラスをそのまま `jax.jit` された関数の引数・戻り値として使う):
```python
import jax
import jax.numpy as jnp

@jax.tree_util.register_pytree_node_class
class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def tree_flatten(self):
        return (self.x, self.y), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"

@jax.jit
def add(a, b):
    return Vec2(a.x + b.x, a.y + b.y)

v1 = Vec2(jnp.array(1.0), jnp.array(2.0))
v2 = Vec2(jnp.array(10.0), jnp.array(20.0))
print(add(v1, v2))
```
実行結果:
```
Vec2(11.0, 22.0)
```

**注意点・落とし穴**:
- `tree_unflatten` は慣例として `classmethod` で定義する。これにより自作クラスを、辞書やリストと全く同じ感覚で `jit`/`grad`/`vmap` に直接渡せるようになる(equinox など、この仕組みの上にモデル定義APIを構築しているライブラリもある)。

---

#### `jax.tree_util.tree_map` の `is_leaf` 引数

**用途**: 通常は「葉」とみなされないコンテナ(リストなど)を、`is_leaf` に渡した述語がTrueを返した時点で「その位置で葉として扱う」ようにする。

**使用例**:
```python
import jax

tree = {"a": [1, 2], "b": [3, 4]}
# 既定: リストの中身(数値)が葉として扱われる
print(jax.tree_util.tree_map(lambda x: x * 10, tree))
# is_leaf でリスト自体を葉として扱わせる
print(jax.tree_util.tree_map(lambda x: sum(x), tree, is_leaf=lambda x: isinstance(x, list)))
```
実行結果:
```
{'a': [10, 20], 'b': [30, 40]}
{'a': 3, 'b': 7}
```

**注意点・落とし穴**:
- `is_leaf` がTrueを返した部分木は、それ以上再帰的に分解されない。`None` を「葉ではなく空のpytree」として無視したくない場合(`is_leaf=lambda x: x is None`)など、既定の分解ルールを部分的に上書きしたいときに使う。

---

### jax.debug応用(callback/breakpoint)

#### `jax.debug.callback(callback, *args, ordered=False, ...)`

**用途**: `jit`/`vmap`/`scan` でトレースされたコードの中から、任意のPython関数(ファイルへのログ書き込み、可視化ライブラリの呼び出しなど、純粋なPython側の副作用)をランタイムで呼び出す。`jax.debug.print` より自由度が高い汎用版。

**シグネチャ**: `jax.debug.callback(callback: 'Callable[..., None] | None' = None, *args: 'Any', ordered: 'bool' = False, partitioned: 'bool' = False, **kwargs: 'Any') -> 'Callable[..., None] | None'`

**使用例**:
```python
import jax
import jax.numpy as jnp

def host_side_effect(x):
    print("host received:", x, type(x))

@jax.jit
def f(x):
    jax.debug.callback(host_side_effect, x)
    return x * 2

print(f(jnp.array([1.0, 2.0, 3.0])))
```
実行結果:
```
host received: [1. 2. 3.] <class 'jaxlib._jax.ArrayImpl'>
[2. 4. 6.]
```

**注意点・落とし穴**:
- `callback` が受け取る引数は具体的な値(`ArrayImpl`)であり、トレーサではない。ただし `callback` の戻り値は計算グラフに戻せず(戻り値なしの副作用専用)、`callback` 内の処理は `jax.grad` の微分対象にもならない。

---

#### `jax.debug.breakpoint(...)`

**用途**: `jit` 化されたコードの実行を一時停止し、対話的なデバッガ(`jdb`)を起動して、その時点でのランタイムの実際の値を確認できる。

**シグネチャ**: `jax.debug.breakpoint(*, backend: 'str | None' = None, filter_frames: 'bool' = True, num_frames: 'int | None' = None, ordered: 'bool' = False, token=None, **kwargs)`

**使用例**(標準入力から `p y`(変数`y`を表示)、続けて `c`(続行)を与えて実行):
```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    y = x * 2
    jax.debug.breakpoint()
    return y + 1

print(f(jnp.array(3.0)))
```
実行結果(`printf 'p y\nc\n' | python script.py` として実行):
```
Entering jdb:
(jdb) Array(6., dtype=float32)
(jdb) 7.0
```

**注意点・落とし穴**:
- 通常の `pdb` と違い、ブレークポイントで見える変数はトレーサではなく実際のランタイムの値(`Array(6., dtype=float32)`)。`jit` 内部の計算をステップ実行しながら実データを確認できる。
- 対話端末がない環境(パイプ実行やCIなど)では標準入力からコマンドを与えないと停止したまま入力待ちになる点に注意。

---

### Flax/optaxの高度な機能

#### `flax.linen.initializers`(カスタムパラメータ初期化)

**用途**: `Dense`/`Conv` などの `kernel_init`/`bias_init` に渡す初期化関数群。numpyの乱数生成器と違い、`(key, shape, dtype)` を受け取って配列を返す関数として定義されている。

**シグネチャ**(代表例): `nn.initializers.constant(value: 'ArrayLike', dtype=None) -> 'Initializer'` / `nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(), dtype=None) -> 'Initializer'`

**使用例**:
```python
import jax
import jax.numpy as jnp
import flax.linen as nn

key = jax.random.key(0)
init_fn = nn.initializers.constant(0.5)
print(init_fn(key, (2, 3), jnp.float32))

dense = nn.Dense(features=3, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.constant(1.0))
params = dense.init(key, jnp.ones((1, 4)))
print(params)
```
実行結果:
```
[[0.5 0.5 0.5]
 [0.5 0.5 0.5]]
{'params': {'bias': Array([1., 1., 1.], dtype=float32), 'kernel': Array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]], dtype=float32)}}
```

**注意点・落とし穴**:
- `Dense` の既定 `kernel_init` は `lecun_normal`(前述の `flax.linen.Dense` の項を参照)。学習が不安定なときや再現実験のために、ゼロ初期化や定数初期化を明示的に指定したい場面で `kernel_init`/`bias_init` を差し替える。

---

#### Flaxの可変コレクション(mutable state, 例: `BatchNorm`)

**用途**: 勾配降下で更新される `params` とは別に、学習中に統計量として更新される値(`BatchNorm` の running mean/var など)を「別のコレクション」として管理する仕組み。`model.apply(variables, x, mutable=['batch_stats'])` で更新後の状態を明示的に取得する。

**使用例**:
```python
import jax
import jax.numpy as jnp
import flax.linen as nn

class Net(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(features=4)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x

model = Net()
key = jax.random.key(0)
x = jnp.ones((2, 3))
variables = model.init(key, x, train=True)
print(list(variables.keys()))

out, updated_state = model.apply(variables, x, train=True, mutable=["batch_stats"])
print("before:", variables["batch_stats"])
print("after :", updated_state["batch_stats"])
```
実行結果:
```
['params', 'batch_stats']
before: {'BatchNorm_0': {'mean': Array([0., 0., 0., 0.], dtype=float32), 'var': Array([1., 1., 1., 1.], dtype=float32)}}
after : {'BatchNorm_0': {'mean': Array([-0.00178755,  0.01625545, -0.01243106, -0.0002554 ], dtype=float32), 'var': Array([0.99, 0.99, 0.99, 0.99], dtype=float32)}}
```

**注意点・落とし穴**:
- `mutable=[...]` を指定しないと `apply` はモデルの出力だけを返し、更新後の統計量は得られない(指定すると `(出力, 更新後variables)` のタプルになる)。
- `params` は `optax` の勾配変換で更新されるのに対し、`batch_stats` は勾配とは無関係にフォワードパスのたびに更新される値なので、学習ループでは両者を別々に(例えば `TrainState` を拡張して)管理する必要がある。

---

#### `flax.training.train_state.TrainState` の保存・復元(`orbax.checkpoint`)

**用途**: 学習途中の `TrainState`(パラメータ+オプティマイザ状態)をディスクに保存し、後で復元する。flax は保存・復元のバックエンドとして `orbax.checkpoint` を使う。

**使用例**:
```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import orbax.checkpoint as ocp

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=1)(x)

model = Model()
key = jax.random.key(0)
x = jnp.ones((1, 2))
params = model.init(key, x)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.1))

ckptr = ocp.PyTreeCheckpointer()
path = "/tmp/orbax_ckpt_test/state"
ckptr.save(path, state)
print("saved   :", state.params["params"]["Dense_0"]["kernel"])

restored = ckptr.restore(path, item=state)
print("restored:", restored.params["params"]["Dense_0"]["kernel"])
print(type(restored))
```
実行結果:
```
saved   : [[-1.1679986]
 [ 0.5335484]]
restored: [[-1.1679986]
 [ 0.5335484]]
<class 'flax.training.train_state.TrainState'>
```

**注意点・落とし穴**:
- `restore` 時に `UserWarning: Sharding info not provided when restoring. Populating sharding info from sharding file. ...` という警告が出た。これはこの検証環境がCPU1個の単一デバイスであることに起因するもので、保存時と異なるデバイス構成で復元する際に関わる注意書きであり、今回の単一デバイスでの復元結果自体には影響していない。
- `ckptr.restore(path, item=state)` のように `item` に既存の(構造だけ合わせた)`TrainState` を渡すことで、復元後も正しい型(`TrainState`)・pytree構造で返ってくる。

---

#### `flax.linen.remat(target, ...)`(FlaxモジュールへのcheckpointingRemat適用)

**用途**: 前述の `jax.checkpoint` をFlaxの `Module` に直接適用するためのラッパー。`target`(Moduleクラス)を `nn.remat()` で包むだけで、そのモジュールの順伝播がチェックポイント対象になる。

**使用例**:
```python
import jax
import jax.numpy as jnp
import flax.linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x):
        jax.debug.print("block forward, x={}", x)
        return nn.Dense(features=4)(x)

RematBlock = nn.remat(Block)

class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = RematBlock()(x)
        x = RematBlock()(x)
        return jnp.sum(x)

model = Net()
key = jax.random.key(0)
x = jnp.ones((1, 4))
params = model.init(key, x)
print("=== jax.grad(model.apply) ===")
grads = jax.grad(model.apply)(params, x)
print(list(params["params"].keys()))
```
実行結果:
```
block forward, x=[[1. 1. 1. 1.]]
block forward, x=[[-0.01332378 -1.2157038  -0.5753304   1.633639  ]]
=== jax.grad(model.apply) ===
block forward, x=[[1. 1. 1. 1.]]
block forward, x=[[-0.01332378 -1.2157038  -0.5753304   1.633639  ]]
block forward, x=[[-0.01332378 -1.2157038  -0.5753304   1.633639  ]]
block forward, x=[[1. 1. 1. 1.]]
```

**注意点・落とし穴**:
- `init` の1回の順伝播では各 `Block` は1回ずつしか呼ばれないが、`jax.grad` による逆伝播計算では順伝播が再度実行され、呼び出し回数が倍になっている(`jax.checkpoint` と同じ再計算の挙動)。
- `nn.remat()` で包んだモジュールは、自動的に `CheckpointBlock_0`、`CheckpointBlock_1` のような名前でパラメータツリーに現れる(`print(list(params["params"].keys()))` の結果より)。

---

#### `optax.chain(...)` + `optax.clip_by_global_norm(max_norm)`

**用途**: 複数の勾配変換(`GradientTransformation`)を1つに合成する。勾配爆発を防ぐノルムクリッピングを、SGD/Adamなどの更新則の前段に挟むのが定番の組み合わせ。

**シグネチャ**: `optax.chain(*args: GradientTransformation) -> GradientTransformationExtraArgs` / `optax.clip_by_global_norm(max_norm) -> GradientTransformation`

**使用例**:
```python
import jax.numpy as jnp
import optax

params = {"w": jnp.array([1.0, 2.0])}
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.sgd(learning_rate=1.0),
)
opt_state = tx.init(params)
huge_grad = {"w": jnp.array([100.0, 100.0])}
updates, opt_state = tx.update(huge_grad, opt_state, params)
print("clipped  :", updates)

no_clip_tx = optax.sgd(learning_rate=1.0)
no_clip_state = no_clip_tx.init(params)
updates2, _ = no_clip_tx.update(huge_grad, no_clip_state, params)
print("unclipped:", updates2)
```
実行結果:
```
clipped  : {'w': Array([-0.7071068, -0.7071068], dtype=float32)}
unclipped: {'w': Array([-100., -100.], dtype=float32)}
```

**注意点・落とし穴**:
- `chain` に渡した順に変換が適用される(この例ではまずノルムを1.0にクリップしてから、SGDの更新量=`-learning_rate * grad` を計算している)。極端に大きな勾配(`100.0`)が、ノルム1.0の範囲(`-0.707...`、すなわち `[100,100]` 方向を保ったまま長さ1に正規化した値)に収まっていることが分かる。

---

#### 学習率スケジュール(`optax.exponential_decay` など)

**用途**: 学習率を定数ではなく、ステップ数に応じて変化する関数として指定する。`optax.adam(learning_rate=schedule)` のように、コール可能なスケジュール関数をそのまま渡せる。

**シグネチャ**: `optax.exponential_decay(init_value, transition_steps: int, decay_rate: float, transition_begin: int = 0, staircase: bool = False, end_value=None) -> Callable[[step], value]`

**使用例**:
```python
import jax.numpy as jnp
import optax

schedule = optax.exponential_decay(init_value=0.1, transition_steps=2, decay_rate=0.5)
print([float(schedule(s)) for s in range(5)])

tx = optax.adam(learning_rate=schedule)
params = {"w": jnp.array(1.0)}
opt_state = tx.init(params)
for step in range(3):
    grads = {"w": jnp.array(1.0)}
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    print(step, params["w"])
```
実行結果:
```
[0.10000000149011612, 0.0707106813788414, 0.05000000074505806, 0.0353553406894207, 0.02500000037252903]
0 0.9000007
1 0.82929075
2 0.7792909
```

**注意点・落とし穴**:
- `schedule` はステップ数(整数)を受け取り学習率(float)を返すだけの普通の関数で、`optax` の各種オプティマイザの `learning_rate` 引数にそのまま渡せる(内部でオプティマイザの状態からステップ数を追跡し、毎回 `schedule(step)` を呼んでいる)。
- 出力の学習率の列(`0.1, 0.0707..., 0.05, ...`)は `transition_steps=2` ごとに `decay_rate=0.5` 倍されていく(指数関数的減衰)ことに対応している。
