# numba 逆引き辞書

Numba 0.66.0 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python`(Python 3.12.3 / numpy 2.4.6 / llvmlite 0.48.0 / LLVM 22.1.0)上でスクリプトファイルとして実際に実行して確認)。性能の数値は WSL2 上の 16 スレッド環境(CPU: znver5)での単一マシン・単一実行の測定値で、環境によって大きく変わる。CUDA は本環境で GPU が使えないため実行検証しておらず、末尾で触れるに留める。

## 目次

1. [JIT基礎(@njit/@jit・初回コンパイル)](#jit基礎njitjit初回コンパイル)
2. [型・シグネチャ指定](#型シグネチャ指定)
3. [並列化(parallel=True/prange)](#並列化paralleltrueprange)
4. [ベクトル化(@vectorize/@guvectorize/@stencil)](#ベクトル化vectorizeguvectorizestencil)
5. [キャッシュ(cache=True)](#キャッシュcachetrue)
6. [対応・非対応機能とエラー(TypingErrorの実例)](#対応非対応機能とエラーtypingerrorの実例)
7. [構造化データ(typed.List/typed.Dict/tuple/レコード配列/jitclass)](#構造化データtypedlisttypeddicttupleレコード配列jitclass)
8. [検査・デバッグ](#検査デバッグ)
9. [その他(cfunc/overload/register_jitable・CUDA)](#その他cfuncoverloadregister_jitablecuda)

---

## JIT基礎(@njit/@jit・初回コンパイル)

### `@njit` / `numba.njit(*args, **kws)`

**用途**: Python 関数を、初回呼び出し時に LLVM で機械語へコンパイル(nopython モード)し、以降の呼び出しを高速化する。Numba の最も基本的なデコレータで、`@jit(nopython=True)` と同じ。

**シグネチャ**: `numba.njit(*args, **kws)`(`nogil` / `cache` / `parallel` / `fastmath` / `error_model` / `boundscheck` / `inline` などのオプションをキーワード引数で受け取る。一覧は `numba.jit.__doc__`)

**使用例**:
```python
import numpy as np
from numba import njit

@njit
def sum_sq(a):
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * a[i]
    return s

x = np.arange(5.0)
print(type(sum_sq).__name__)
print(sum_sq(x))
print(sum_sq.signatures)
```
実行結果:
```
CPUDispatcher
30.0
[(Array(float64, 1, 'C', False, aligned=True),)]
```

**注意点・落とし穴**:
- 関数はデコレートした時点ではコンパイルされず、最初に呼び出したときの引数の型で特殊化してコンパイルされる(上の `signatures` は 1 回呼んだあとに初めて埋まる)。
- デコレート後のオブジェクトは元の関数ではなく `CPUDispatcher`。元の Python 関数は `.py_func` で取り出せる(次項)。

---

### `@jit` / `numba.jit(signature_or_function=None, locals={}, cache=False, pipeline_class=None, boundscheck=None, **options)`

**用途**: `njit` の元になっているデコレータ。0.66.0 では `@jit` だけでも nopython モードが既定で、コンパイルできない関数は(古い版のようにオブジェクトモードへ暗黙にフォールバックせず)エラーになる。

**シグネチャ**: `numba.jit(signature_or_function=None, locals=mappingproxy({}), cache=False, pipeline_class=None, boundscheck=None, **options)`

**使用例**:
```python
import numpy as np
from numba import jit

@jit
def f_default(x):
    return np.fft.fft(x)          # nopython では未対応

@jit(nopython=False)
def f_nopython_false(x):
    return np.fft.fft(x)

@jit(forceobj=True)
def f_forceobj(x):
    return np.fft.fft(x)

x = np.arange(4.0)
for f in (f_default, f_nopython_false):
    try:
        f(x)
    except Exception as e:
        print(f.__name__, "->", type(e).__name__)
print(f_forceobj(x))
```
実行結果:
```
f_default -> TypingError
f_nopython_false -> TypingError
[ 6.+0.j -2.+2.j -2.+0.j -2.-2.j]
```

**注意点・落とし穴**:
- `@jit` と `@jit(nopython=False)` はどちらも `TypingError` になる(古い版にあった「失敗したら自動でオブジェクトモード」の挙動は 0.66.0 では起きない)。
- オブジェクトモードを使いたい場合は `forceobj=True` を明示する。ただし Python のオブジェクトを介するので高速化はほとんど得られない。特別な理由がなければ `@njit` を使う。

---

### 引数の型ごとの特殊化と `.signatures` / `.py_func`

**用途**: Numba は呼び出された引数の型ごとに別々の機械語を生成して `.signatures` に登録する。型が変わるたびに再コンパイルが走るので、どの型で何回コンパイルされたかは `.signatures` で確認できる。

**シグネチャ**: `dispatcher.signatures`(コンパイル済みの引数型タプルのリスト) / `dispatcher.py_func`(元の Python 関数) / `dispatcher.nopython_signatures`

**使用例**:
```python
import numpy as np
from numba import njit

@njit
def f(x):
    return x + 1

print(f.signatures)
print(f(1), f(1.5), f(np.float32(1)), f(True), f(1j))
print(f.signatures)
print(f.py_func(1), f.py_func.__name__)
```
実行結果:
```
[]
2 2.5 2.0 2 (1+1j)
[(int64,), (float64,), (float32,), (bool,), (complex128,)]
2 f
```

**注意点・落とし穴**:
- `True`(bool)を渡すと `int64` ではなく `bool` 型として別にコンパイルされ、戻り値は `2`(int)。整数・浮動小数点数・配列の dtype や次元、メモリレイアウトが変わるたびに再コンパイルされるため、型がバラバラな引数を大量に流す使い方では初回コスト(数十〜数百 ms/型)が積み上がる。
- `f.py_func` は元の Python 関数で、テスト時に純 Python の結果と突き合わせるのに使える。

---

### 初回コンパイルと定常状態の実行時間(性能の測り方)

**用途**: `@njit` の効果は、初回呼び出し(コンパイル時間を含む)と 2 回目以降(定常状態)を分けて測らないと正しく評価できない。ここでは逐次的な再帰計算(指数移動平均)を純 Python ループと `njit` で比べる。

**使用例**:
```python
import time
import numpy as np
from numba import njit

def ema_py(x, alpha):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out

ema_nb = njit(ema_py)
x = np.random.default_rng(0).standard_normal(1_000_000)

t = time.perf_counter(); r_nb = ema_nb(x, 0.1); first = time.perf_counter() - t
steady = []
for _ in range(5):
    t = time.perf_counter(); ema_nb(x, 0.1); steady.append(time.perf_counter() - t)
t = time.perf_counter(); r_py = ema_py(x, 0.1); py = time.perf_counter() - t

print(np.allclose(r_nb, r_py))
print(f"numba 1st call (compile+run): {first*1000:8.1f} ms")
print(f"numba steady (min of 5):      {min(steady)*1000:8.2f} ms")
print(f"pure Python:                  {py*1000:8.1f} ms")
print(f"speedup (steady):             {py/min(steady):8.0f} x")
```
(以下は単一マシン・単一実行の実測値。再実行すると数値は変わる)
実行結果:
```
True
numba 1st call (compile+run):    300.4 ms
numba steady (min of 5):          1.25 ms
pure Python:                     175.6 ms
speedup (steady):                  140 x
```

**注意点・落とし穴**:
- 初回呼び出しはコンパイル込みで約 0.3 秒かかっている。定常状態の速さで純 Python の 100 倍超になるが、呼び出し回数が少ない小さな処理ではコンパイル時間の方が支配的で、`njit` を付けると逆に遅くなる。
- ベンチマークのウォームアップを忘れて 1 回目の時間を測ると `njit` の効果を過小評価する。逆に `for i in range(n): s += i * i` のような単純な式は LLVM が閉形式に最適化して 0 秒に近い時間になり、「Numba が桁違いに速い」という誤った印象になることがある。実データに近い配列アクセスを含む処理で測るとよい。

---

### numpy 相当の処理との比較(ループを書く方が速い場合)

**用途**: numpy のブロードキャストは大きな中間配列を作るため、要素ごとのループを `njit` で書いた方が速いことがある。逆に numpy が 1 関数で完結する処理では差は出ない。

**使用例**:
```python
import time
import numpy as np
from numba import njit

def pdist_np(X):
    d = X[:, None, :] - X[None, :, :]
    return np.sqrt((d ** 2).sum(-1))

@njit
def pdist_nb(X):
    n, m = X.shape
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                d = X[i, k] - X[j, k]
                s += d * d
            out[i, j] = np.sqrt(s)
    return out

X = np.random.default_rng(1).random((2000, 3))
t = time.perf_counter(); r1 = pdist_nb(X); first = time.perf_counter() - t
ts = []
for _ in range(5):
    t = time.perf_counter(); pdist_nb(X); ts.append(time.perf_counter() - t)
tn = []
for _ in range(5):
    t = time.perf_counter(); r2 = pdist_np(X); tn.append(time.perf_counter() - t)
print(np.allclose(r1, r2))
print(f"numba 1st: {first*1000:.1f} ms, steady(min of 5): {min(ts)*1000:.1f} ms")
print(f"numpy broadcast steady(min of 5): {min(tn)*1000:.1f} ms")
```
(単一マシン・単一実行の実測値)
実行結果:
```
True
numba 1st: 334.4 ms, steady(min of 5): 7.3 ms
numpy broadcast steady(min of 5): 73.2 ms
```

**注意点・落とし穴**:
- 2000×2000×3 の中間配列(約 96 MB)を作る numpy 版に対し、ループ版 `njit` は中間配列を作らないため、定常状態で約 10 倍速かった(この環境・この 1 回の測定)。numpy が既に最適化された 1 回の呼び出し(`np.sum` など)で完結する処理では、こうした差は出ない。

---

### `nogil=True`

**用途**: コンパイル済み関数の実行中は Python の GIL を解放する。`threading.Thread` から呼ぶと複数スレッドで同時に CPU 計算ができる(`parallel=True` を使わない、手動スレッド並列向け)。

**使用例**:
```python
import math, threading, time
from numba import njit

@njit(nogil=True)
def busy_nogil(n):
    s = 0.0
    for i in range(n):
        s += math.sin(i)
    return s

@njit
def busy_gil(n):
    s = 0.0
    for i in range(n):
        s += math.sin(i)
    return s

N = 30_000_000
busy_nogil(10); busy_gil(10)

def run_threads(f):
    ths = [threading.Thread(target=f, args=(N,)) for _ in range(2)]
    t = time.perf_counter()
    for th in ths: th.start()
    for th in ths: th.join()
    return (time.perf_counter() - t) * 1000

t = time.perf_counter(); busy_nogil(N); busy_nogil(N)
print(f"serial x2:        {(time.perf_counter() - t) * 1000:.0f} ms")
print(f"2 threads nogil:  {run_threads(busy_nogil):.0f} ms")
print(f"2 threads (GIL):  {run_threads(busy_gil):.0f} ms")
```
(単一マシン・単一実行の実測値)
実行結果:
```
serial x2:        426 ms
2 threads nogil:  219 ms
2 threads (GIL):  433 ms
```

**注意点・落とし穴**:
- `nogil` なしだと 2 スレッドで走らせても逐次(422 ms)以上の時間(536 ms)がかかり、GIL のせいで実質並列にならない。`nogil=True` では約 228 ms とほぼ半分になった。
- 関数内で Python オブジェクトに触らない nopython モードだからこそ GIL を手放せる。スレッド間で同じ配列に書き込む場合の競合は自分で避ける必要がある。

---

### `fastmath=True`

**用途**: IEEE 754 の厳密性(演算順序の保持、NaN/Inf の扱いなど)を緩める最適化を許可し、SIMD 化などを促す。浮動小数点の総和などで結果の下位桁がわずかに変わる。

**使用例**:
```python
import math
import numpy as np
from numba import njit

@njit
def ssum(a):
    s = 0.0
    for x in a:
        s += x
    return s

@njit(fastmath=True)
def ssum_fast(a):
    s = 0.0
    for x in a:
        s += x
    return s

a = np.random.default_rng(0).random(50_000_000)
print("plain    :", repr(ssum(a)))
print("fastmath :", repr(ssum_fast(a)))
print("np.sum   :", repr(float(np.sum(a))))
print("fsum     :", repr(math.fsum(a)))
```
実行結果:
```
plain    : 24997430.602987777
fastmath : 24997430.60299362
np.sum   : 24997430.602993794
fsum     : 24997430.602993794
```

**注意点・落とし穴**:
- この測定では、厳密な総和(`math.fsum`)と一致したのは `np.sum`(ペアワイズ加算)で、逐次加算の `plain` と `fastmath` はどちらも下位桁がずれ、互いにも異なる値になった。`fastmath` は浮動小数点の加算順序の入れ替えを許す指定なので、結果の下位桁は環境(SIMD 幅など)にも依存する。
- `fastmath` にはフラグの集合も渡せる(`@njit(fastmath={'nnan', 'ninf'})` がこの環境で動作することを確認)。個別のフラグの意味は `numba.jit.__doc__` を参照。

---

### `error_model='numpy'`

**用途**: ゼロ除算時の挙動を切り替える。既定の `'python'` は `ZeroDivisionError` を送出し、`'numpy'` は `inf`/`nan` を返す(例外チェックを省くので速くもなる)。

**使用例**:
```python
from numba import njit

@njit
def div_py(a, b):
    return a / b

@njit(error_model="numpy")
def div_np(a, b):
    return a / b

try:
    div_py(1.0, 0.0)
except ZeroDivisionError as e:
    print("python model:", type(e).__name__, e)
print("numpy  model:", div_np(1.0, 0.0), div_np(-1.0, 0.0), div_np(0.0, 0.0))
```
実行結果:
```
python model: ZeroDivisionError division by zero
numpy  model: inf -inf nan
```

**注意点・落とし穴**:
- `'numpy'` では `1.0/0.0` が `inf`、`-1.0/0.0` が `-inf`、`0.0/0.0` が `nan` になる(上の出力)。

---

## 型・シグネチャ指定

### 明示シグネチャ(`@njit("float64(float64, float64)")`)

**用途**: 関数デコレータに型シグネチャを渡すと、呼び出しを待たずにデコレート時点でコンパイルする(eager compilation)。呼び出し時の型チェックも厳密になり、シグネチャに合わない型は拒否される。

**シグネチャ**: `@njit("戻り値型(引数型, ...)")` または `@njit(numba.float64(numba.float64, numba.float64))`(複数なら `[sig1, sig2]` のリスト)

**使用例**:
```python
from numba import njit

@njit("float64(float64, float64)")
def add(a, b):
    return a + b

print(add.signatures)          # 呼ぶ前からコンパイル済み
print(add(1, 2))               # int は float64 に変換される
print(add(1.5, 2))
try:
    add("x", 1)
except TypeError as e:
    print(type(e).__name__, e)
```
実行結果:
```
[(float64, float64)]
3.0
3.5
TypeError No matching definition for argument type(s) unicode_type, int64
```

**注意点・落とし穴**:
- 明示シグネチャを付けるとデコレート時にコンパイルされるので、モジュールの import 時にコンパイル時間がかかる。多数の関数に付けると import が遅くなる(その代わり初回呼び出しの遅延は無い)。
- シグネチャ外の型は自動再コンパイルされずに `TypeError: No matching definition for argument type(s) ...` になる(暗黙の型変換で通るものは通る)。

---

### 配列型の指定(`float64[:]` と `float64[::1]`)とレイアウト

**用途**: 配列引数は要素型・次元・メモリレイアウト(`C`=C 連続 / `F`=F 連続 / `A`=任意ストライド)で型が決まる。`[:]` は `A`(任意)、`[::1]` は `C` 連続を表す。

**シグネチャ**: `numba.float64[:]`(1次元・A) / `numba.float64[:, :]`(2次元・A) / `numba.float64[::1]`(1次元・C) / `numba.float64[:, ::1]`(2次元・C) / `numba.types.Array(dtype, ndim, layout)`

**使用例**:
```python
import numpy as np
from numba import njit, float64

@njit(float64(float64[:]))
def total(a):
    s = 0.0
    for v in a:
        s += v
    return s

@njit(float64(float64[::1]))
def total_c(a):
    return a.sum()

print(total.signatures)
print(total(np.arange(4.0)))
print(total(np.arange(8.0)[::2]))      # 非連続でも A レイアウトなので OK
print(total_c(np.arange(4.0)))
for f, arr in [(total, np.arange(4)), (total_c, np.arange(8.0)[::2])]:
    try:
        f(arr)
    except TypeError as e:
        print(type(e).__name__, e)
```
実行結果:
```
[(Array(float64, 1, 'A', False, aligned=True),)]
6.0
12.0
6.0
TypeError No matching definition for argument type(s) array(int64, 1d, C)
TypeError No matching definition for argument type(s) array(float64, 1d, A)
```

**注意点・落とし穴**:
- dtype が違う(`int64` 配列を `float64[:]` に渡す)と暗黙変換されずに `TypeError`。同様に非連続配列を `[::1]` に渡すのも `TypeError` になる。
- 型を指定せずに呼び出し時推論に任せる場合は、実際に渡した配列のレイアウト(通常は `C`)で特殊化される。

---

### 複数シグネチャの登録と `.nopython_signatures`

**用途**: シグネチャのリストを渡すと複数の型組み合わせをあらかじめコンパイルできる。呼び出し時は、引数型に最も合うものが選ばれる。

**シグネチャ**: `dispatcher.nopython_signatures`(`Signature` オブジェクトのリスト。戻り値型込み)

**使用例**:
```python
from numba import njit, int64, float64

@njit([int64(int64, int64), float64(float64, float64)])
def mul(a, b):
    return a * b

print(mul.signatures)
print(mul.nopython_signatures)
print(mul(2, 3), mul(2.5, 2))
```
実行結果:
```
[(int64, int64), (float64, float64)]
[(int64, int64) -> int64, (float64, float64) -> float64]
6 5.0
```

**注意点・落とし穴**:
- `.signatures` は引数型のタプルだけ、`.nopython_signatures` は戻り値型も含む `Signature`(`(int64, int64) -> int64` の形)。

---

### `numba.types` の型オブジェクト

**用途**: Numba の型を表すオブジェクト。シグネチャ指定・`typed.Dict`/`typed.List` の型パラメータ・`jitclass` の spec などで使う。`numba.float64` などは `numba.types.float64` のエイリアスとして直接 import できる。

**シグネチャ**: `numba.types.float64` / `int32` / `boolean` / `void` / `unicode_type` / `UniTuple(dtype, n)` / `Tuple((t1, t2, ...))` / `Array(dtype, ndim, layout)` / `ListType(item)` / `DictType(k, v)`

**使用例**:
```python
from numba import types

print(types.float64, types.int32, types.void, types.boolean)
print(types.float64[:], types.float64[:, :], types.float64[::1], types.float64[:, ::1])
print(types.Array(types.float64, 2, "C"))
print(types.UniTuple(types.int64, 3), types.Tuple((types.int64, types.float64)))
print(repr(types.float64[:]), types.float64[:].ndim, types.float64[:].layout)
```
実行結果:
```
float64 int32 none bool
array(float64, 1d, A) array(float64, 2d, A) array(float64, 1d, C) array(float64, 2d, C)
array(float64, 2d, C)
UniTuple(int64 x 3) Tuple(int64, float64)
Array(float64, 1, 'A', False, aligned=True) 1 A
```

---

### `numba.typeof(value)` / `numba.from_dtype(dtype)`

**用途**: Python の値が Numba にどの型として認識されるか調べる。「なぜその型でコンパイルされたか」「明示シグネチャに何を書けば合うか」を確認するのに使う。

**シグネチャ**:
- `numba.typeof(val)`
- `numba.from_dtype(dtype)`(NumPy の dtype から Numba の型を返す)

**使用例**:
```python
import numpy as np
import numba

print(numba.typeof(1), numba.typeof(1.0), numba.typeof(True), numba.typeof(1j), numba.typeof("a"))
print(numba.typeof(np.float32(1)), numba.typeof((1, 2.0)), numba.typeof([1, 2]), numba.typeof(None))
a = np.zeros((2, 3))
print(numba.typeof(a), numba.typeof(a.T), numba.typeof(np.zeros(3)[::2]))
print(numba.typeof(np.zeros(3)) == numba.types.float64[::1])
print(numba.from_dtype(np.dtype("float32")))
```
実行結果:
```
int64 float64 bool complex128 unicode_type
float32 Tuple(int64, float64) reflected list(int64)<iv=None> none
array(float64, 2d, C) array(float64, 2d, F) array(float64, 1d, A)
True
float32
```

**注意点・落とし穴**:
- Python の `int` は `int64`、`float` は `float64` になる。`a.T` のように転置した配列は `F` レイアウト、スライスの `[::2]` は `A` になり、C 連続の配列とは別の型としてコンパイルされる。
- `list` は `reflected list(int64)` として認識される。これは廃止予定の型(後述の `typed.List` の項を参照)。

---

### `dispatcher.compile(sig)`(事前コンパイル)

**用途**: デコレート後の関数に対して、シグネチャを追加してコンパイルする。遅延コンパイルの関数に、必要な型だけをあらかじめコンパイルしておきたいときに使う。

**シグネチャ**: `dispatcher.compile(sig)`(`"int64(int64)"` のような文字列、または引数型タプル)

**使用例**:
```python
from numba import njit, types

@njit
def lazy(x):
    return x * 2

print(lazy.signatures)
lazy.compile("int64(int64)")
print(lazy.signatures)
lazy.compile((types.float64,))
print(lazy.signatures)
print(lazy.nopython_signatures)
```
実行結果:
```
[]
[(int64,)]
[(int64,), (float64,)]
[(int64,) -> int64, (float64,) -> float64]
```

---

### `locals=` による変数の型指定

**用途**: 関数内のローカル変数の型を推論任せにせず明示する。累積用の変数を `float32` で持たせる、といった用途に使う。

**シグネチャ**: `@njit(locals={"変数名": 型, ...})`

**使用例**:
```python
from numba import njit, float32

@njit(locals={"s": float32})
def acc32(n):
    s = 0
    for i in range(n):
        s += 0.1
    return s

@njit
def acc64(n):
    s = 0
    for i in range(n):
        s += 0.1
    return s

print(repr(acc32(10)), repr(acc64(10)))
```
実行結果:
```
1.0000001192092896 0.9999999999999999
```

**注意点・落とし穴**:
- `s = 0`(整数リテラル)から始めても、`s += 0.1` で浮動小数点になるので推論は `float64` に統一される。`locals` で `float32` を指定すると累積が単精度で行われ、倍精度の `acc64`(`0.9999999999999999`)との差が出る(戻り値の Python オブジェクトはどちらも `float`)。

---

### 整数はオーバーフローする(型推論の落とし穴)

**用途**: Numba の整数は固定幅(既定 `int64`)で、Python の多倍長整数のようには振る舞わない。演算がオーバーフローしても例外は出ず、値がラップアラウンドする。

**使用例**:
```python
from numba import njit

@njit
def square(x):
    return x * x

print(square(3_000_000_000))       # int64 に収まる
print(square(2 ** 62))             # 2**124 は int64 に収まらず 0 に折り返す
print((2 ** 62) ** 2)              # 純 Python なら正しい値
try:
    square(2 ** 64)                # int64 の範囲外の入力は変換できない
except OverflowError as e:
    print(type(e).__name__, e)
```
実行結果:
```
9000000000000000000
0
21267647932558653966460912964485513216
OverflowError int too big to convert
```

**注意点・落とし穴**:
- 階乗・フィボナッチ・ハッシュ計算・大きな累乗など、`int64` に収まらない可能性がある計算では、オーバーフローが無警告で起きる。`float64` を使うか、途中で剰余を取る設計にするか、`np.int64` の範囲を意識して書く。

---

## 並列化(parallel=True/prange)

### `@njit(parallel=True)` と `numba.prange`

**用途**: `prange` で書いたループを複数スレッドに分割して自動並列実行する。ループの各反復が互いに独立(または単純なスカラーのリダクション)である場合に使う。

**シグネチャ**:
- `numba.prange(*args)`(`range` と同じ引数)
- `@njit(parallel=True)`

**使用例**:
```python
import time
import numpy as np
from numba import njit, prange

@njit
def serial(x):
    s = 0.0
    for i in range(x.shape[0]):
        s += np.sqrt(x[i]) * np.sin(x[i])
    return s

@njit(parallel=True)
def par(x):
    s = 0.0
    for i in prange(x.shape[0]):
        s += np.sqrt(x[i]) * np.sin(x[i])   # s は自動でリダクション扱い
    return s

def bench(f, x, n=5):
    t = time.perf_counter(); r = f(x); first = time.perf_counter() - t
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(x); ts.append(time.perf_counter() - t)
    return r, first, min(ts)

x = np.random.default_rng(0).random(20_000_000)
for name, f in [("serial", serial), ("parallel", par)]:
    r, first, steady = bench(f, x)
    print(f"{name:8s} result={r!r} 1st={first*1000:.1f} ms steady={steady*1000:.1f} ms")
ts = []
for _ in range(5):
    t = time.perf_counter(); (np.sqrt(x) * np.sin(x)).sum(); ts.append(time.perf_counter() - t)
print(f"numpy    steady={min(ts)*1000:.1f} ms")
```
(単一マシン・単一実行の実測値、16 スレッド環境)
実行結果:
```
serial   result=7283791.3926995555 1st=381.2 ms steady=135.3 ms
parallel result=7283791.392698994 1st=320.5 ms steady=23.9 ms
numpy    steady=191.7 ms
```

**注意点・落とし穴**:
- `prange` 内の `s += ...` のような形は Numba がリダクションと認識し、スレッドごとの部分和を最後に合成する。合成順序が逐次と異なるため、浮動小数点の結果が下位桁でわずかにずれる(上の `result` の末尾を比較)。
- `parallel=True` を付けないと `prange` は普通の `range` として動く(エラーにはならず、並列化されない)。
- 最初の呼び出しはスレッド起動を含めて 0.3 秒程度のコンパイル時間がかかる。ループが短い(数千要素など)場合はスレッド生成・同期のオーバーヘッドの方が大きく、逐次より遅くなることもある。

---

### `prange` 内の配列要素への `+=` は競合する(結果が壊れる)

**用途**: `prange` 内でスカラー変数に対する `s += ...` は安全なリダクションだが、配列要素への `out[j] += ...` はリダクションとして認識されず、複数スレッドが同じ要素を同時に更新して結果が不定になる。並列化するループの軸を選び直して各スレッドが別々の要素だけを書くようにする。

**使用例**:
```python
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def colsum_bad(A):
    out = np.zeros(A.shape[1])
    for i in prange(A.shape[0]):            # 行を並列化
        for j in range(A.shape[1]):
            out[j] += A[i, j]               # 全スレッドが同じ out[j] に書く
    return out

@njit(parallel=True)
def colsum_ok(A):
    out = np.zeros(A.shape[1])
    for j in prange(A.shape[1]):            # 列を並列化
        s = 0.0
        for i in range(A.shape[0]):
            s += A[i, j]
        out[j] = s                          # 各スレッドは別々の out[j] にだけ書く
    return out

A = np.ones((1000, 4))                      # 正解は各列 1000.0
print("bad:", colsum_bad(A), colsum_bad(A))
print("ok :", colsum_ok(A))

@njit(parallel=True)
def count_bad(n):
    cnt = np.zeros(1, dtype=np.int64)
    for i in prange(n):
        cnt[0] += 1
    return cnt[0]
print("count_bad:", [count_bad(1_000_000) for _ in range(3)])   # 正解は 1000000
```
(競合が起きる関数の出力は実行ごとに変わる。以下は 1 回の実測値で、`colsum_bad`/`count_bad` は必ずしも 1000.0・1000000 にならない)
実行結果:
```
bad: [124. 124. 124. 124.] [179. 179. 179. 179.]
ok : [1000. 1000. 1000. 1000.]
count_bad: [375000, 812500, 1000000]
```

**注意点・落とし穴**:
- エラーも警告も出ずに静かに誤った結果を返す。`NUMBA_NUM_THREADS=1` で実行すると偶然正しい値になってしまい、バグに気づきにくい(この環境では 1 スレッドで `colsum_bad` も 1000.0 になることを確認)。
- スカラー変数の累積(`s += x[i]`、`min`/`max`)は安全。配列要素・リストへの書き込みが競合するかを常に意識する。並列版と逐次版の結果を `np.allclose` で突き合わせるテストを書くのが確実。

---

### `numba.get_num_threads()` / `numba.set_num_threads(n)` / `numba.threading_layer()`

**用途**: 並列実行で使うスレッド数の確認・変更と、実際に使われているスレッディングレイヤ(`omp`/`tbb`/`workqueue`)の確認。

**シグネチャ**:
- `numba.get_num_threads()`
- `numba.set_num_threads(n)`(1 以上、起動時に決まる上限 `NUMBA_NUM_THREADS` 以下)
- `numba.threading_layer()`(並列関数を最初に実行するまでは未初期化)

**使用例**:
```python
import time
import numpy as np
import numba
from numba import njit, prange

try:
    print(numba.threading_layer())
except ValueError as e:
    print("before:", e)

@njit(parallel=True)
def par(x):
    s = 0.0
    for i in prange(x.shape[0]):
        s += np.sqrt(x[i]) * np.sin(x[i])
    return s

x = np.random.default_rng(0).random(20_000_000)
par(x[:10])
print("layer:", numba.threading_layer())
print("default threads:", numba.get_num_threads())

def steady(n):
    ts = []
    for _ in range(5):
        t = time.perf_counter(); par(x); ts.append(time.perf_counter() - t)
    return min(ts) * 1000

for n in (1, 4, 16):
    numba.set_num_threads(n)
    print(f"{n:2d} threads: {steady(n):.1f} ms")
try:
    numba.set_num_threads(100)
except ValueError as e:
    print(type(e).__name__, e)
```
(スレッド数ごとの時間は単一マシン・単一実行の実測値)
実行結果:
```
before: Threading layer is not initialized.
layer: omp
default threads: 16
 1 threads: 144.8 ms
 4 threads: 36.7 ms
16 threads: 21.4 ms
ValueError The number of threads must be between 1 and 16
```

**注意点・落とし穴**:
- スレッド数の上限は環境変数 `NUMBA_NUM_THREADS`(既定は CPU 数)。`set_num_threads` はその範囲内でしか増やせず、超えると `ValueError`。起動時に少ないスレッド数へ抑えたい場合は環境変数で指定する。
- 利用するレイヤは環境変数 `NUMBA_THREADING_LAYER`(`omp`/`tbb`/`workqueue`)で切り替えられる。この環境では `omp` が使われた。

---

### `func.parallel_diagnostics(level=...)`

**用途**: `parallel=True` の関数について、どのループが並列化・融合されたか(あるいは並列化されなかったか)を診断出力する。並列化が効いていない原因調査に使う。

**シグネチャ**: `dispatcher.parallel_diagnostics(signature=None, level=1)`(`level` は 1〜4、大きいほど詳細)

**使用例**:
```python
import contextlib
import io
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def f(x):
    out = np.empty_like(x)
    for i in prange(x.shape[0]):
        out[i] = x[i] * 2
    return out

f(np.ones(4))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    f.parallel_diagnostics(level=1)
lines = buf.getvalue().splitlines()
start = [i for i, l in enumerate(lines) if l.startswith("@njit")][0]
print("\n".join(l.rstrip() for l in lines[start:start + 8]))
```
実行結果:
```
@njit(parallel=True)                |
def f(x):                           |
    out = np.empty_like(x)          |
    for i in prange(x.shape[0]):----| #0
        out[i] = x[i] * 2           |
    return out                      |
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
```

**注意点・落とし穴**:
- 本来の出力にはソースファイルのパスを含むヘッダも付く(上ではその部分を省略している)。`level` は 1(既定・最小限)から 4 まであり、大きいほど詳細な情報が出る(`parallel_diagnostics.__doc__` の記載)。

---

## ベクトル化(@vectorize/@guvectorize/@stencil)

### `@vectorize`(明示シグネチャ) / `numba.vectorize(ftylist_or_function=(), **kws)`

**用途**: スカラーを受け取ってスカラーを返す関数から、NumPy の ufunc(要素ごと演算・ブロードキャスト・`reduce` などのメソッド対応)を作る。

**シグネチャ**: `numba.vectorize(ftylist_or_function=(), **kws)`(docstring 上のキーワードは `target='cpu'`(`'cpu'`/`'parallel'`)・`identity`・`cache`)

**使用例**:
```python
import numpy as np
from numba import vectorize

@vectorize(["float64(float64, float64)"])
def hyp(a, b):
    return np.sqrt(a * a + b * b)

print(hyp(3.0, 4.0))
print(hyp(np.array([3.0, 5.0]), np.array([4.0, 12.0])))
print(hyp(np.array([[3.0], [5.0]]), np.array([4.0, 12.0])))     # ブロードキャスト
print(hyp(np.array([3, 5]), np.array([4, 12])))                 # int 入力は float64 へ安全に変換
print(hyp.types, hyp.nin, hyp.nout)

@vectorize(["int64(int64, int64)", "float64(float64, float64)"])
def addv(a, b):
    return a + b

print(addv.types)
print(addv(np.array([1, 2]), np.array([3, 4])), addv(np.array([1.5]), 2))
print(addv.reduce(np.array([1, 2, 3, 4])), addv.accumulate(np.array([1, 2, 3, 4])))
print(addv.outer(np.array([1, 2]), np.array([10, 20])))
```
実行結果:
```
5.0
[ 5. 13.]
[[ 5.         12.36931688]
 [ 6.40312424 13.        ]]
[ 5. 13.]
['dd->d'] 2 1
['ll->l', 'dd->d']
[4 6] [3.5]
10 [ 1  3  6 10]
[[11 21]
 [12 22]]
```

**注意点・落とし穴**:
- 型指定の文字列は NumPy の ufunc `types` 表記(`'dd->d'` = float64×2→float64、`'ll->l'` = int64×2→int64)で `.types` に並ぶ。ufunc なので `reduce` / `accumulate` / `outer` などの標準メソッドがそのまま使える。
- 関数の中で `if` などの分岐を使ってもよい(スカラー処理なので `np.where` に書き換える必要はない)。

---

### `@vectorize`(遅延コンパイル) と `target="parallel"`

**用途**: シグネチャを省略すると、呼び出された型ごとに遅延コンパイルされる(`DUFunc`)。`target="parallel"` を付けると要素ごとの演算を複数スレッドに分けて実行する(明示シグネチャが必要)。

**使用例**:
```python
import time
import numpy as np
from numba import vectorize

@vectorize
def sq(x):
    return x * x

print(sq(np.arange(3)), sq(np.arange(3.0)))
print(type(sq).__name__, sq.types)

@vectorize
def relu(x):
    return x if x > 0 else 0.0
print(relu(np.array([-1.0, 2.0])))

@vectorize(["float64(float64)"])
def f_cpu(x):
    return np.sin(x) ** 2 + np.cos(x) ** 2 * x

@vectorize(["float64(float64)"], target="parallel")
def f_par(x):
    return np.sin(x) ** 2 + np.cos(x) ** 2 * x

def f_np(x):
    return np.sin(x) ** 2 + np.cos(x) ** 2 * x

x = np.random.default_rng(0).random(10_000_000)
def steady(f, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(x); ts.append(time.perf_counter() - t)
    return min(ts) * 1000
f_cpu(x[:10]); f_par(x[:10])
print(f"numpy: {steady(f_np):.1f} ms, vectorize cpu: {steady(f_cpu):.1f} ms, vectorize parallel: {steady(f_par):.1f} ms")
```
(最後の行は単一マシン・単一実行の実測値)
実行結果:
```
[0 1 4] [0. 1. 4.]
DUFunc ['l->l', 'd->d']
[0. 2.]
numpy: 161.6 ms, vectorize cpu: 83.1 ms, vectorize parallel: 23.6 ms
```

**注意点・落とし穴**:
- この測定では、numpy 式(中間配列を複数作る)より `@vectorize` が約 2 倍、`target="parallel"` が約 6 倍速かった。
- 遅延コンパイルの `DUFunc` は、使った型ごとに `.types` が増えていく(上では `int64` と `float64` の両方が登録された)。

---

### `@guvectorize`

**用途**: 配列(部分配列)を受け取って配列を返す一般化ユニバーサル関数(gufunc)を作る。「最後の軸をまるごと処理して、それ以外の軸(バッチ)はブロードキャスト」という NumPy の gufunc 規則に従う。出力は戻り値でなく引数に書き込む。

**シグネチャ**: `numba.guvectorize(ftylist, signature, *, target='cpu', identity=None, **kws)`(inspect.signature 上は `(*args, **kwargs)`。例: `"(n),(n)->()"` のレイアウト文字列を第2引数に取る)

**使用例**:
```python
import numpy as np
from numba import guvectorize

@guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()")
def dot(a, b, out):
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    out[0] = s

A = np.arange(6.0).reshape(2, 3)
B = np.ones((2, 3))
print(dot(A, B))                    # 行ごとの内積
print(dot(A, np.ones(3)))           # 片方をブロードキャスト
try:
    dot(np.ones(3), np.ones(4))
except ValueError as e:
    print(type(e).__name__, e)

@guvectorize(["void(float64[:,:], float64[:,:], float64[:,:])"], "(m,n),(n,p)->(m,p)")
def matmul(A, B, C):
    m, n = A.shape
    p = B.shape[1]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s

X = np.random.default_rng(0).random((4, 2, 3))
Y = np.random.default_rng(1).random((4, 3, 5))
print(np.allclose(matmul(X, Y), X @ Y), matmul(X, Y).shape)
```
実行結果:
```
[ 3. 12.]
[ 3. 12.]
ValueError dot: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n),(n)->() (size 4 is different from 3)
True (4, 2, 5)
```

**注意点・落とし穴**:
- 出力は戻り値で返すのではなく、最後の引数(`out`)に書き込む。スカラー出力は `"()"` で、`out[0] = ...` と書く。
- 最後の軸(コア次元 `n`)以外の軸はバッチとして自動でループ・ブロードキャストされる(上の `dot(A, np.ones(3))` は片方の引数だけ 1 次元)。コア次元のサイズが引数間で合わないと `ValueError` になる。

---

### `@stencil`

**用途**: 配列の近傍(前後の要素)を参照して各要素を計算するカーネルを、1 つの相対インデックス式で書く。畳み込み・移動平均・ラプラシアンなど。

**シグネチャ**: `numba.stencil(func_or_mode='constant', **options)`(`neighborhood=` で参照範囲、`cval=` で境界値を指定)

**使用例**:
```python
import numpy as np
from numba import njit, stencil

@stencil
def kernel(a):
    return 0.25 * (a[-1] + 2 * a[0] + a[1])

@njit
def smooth(a):
    return kernel(a)

print(smooth(np.arange(6.0)))

@stencil(neighborhood=((-2, 2),))
def k5(a):
    return (a[-2] + a[-1] + a[0] + a[1] + a[2]) / 5

@njit
def smooth5(a):
    return k5(a)

print(smooth5(np.arange(8.0)))

@stencil
def lap(a):
    return a[-1, 0] + a[1, 0] + a[0, -1] + a[0, 1] - 4 * a[0, 0]

@njit
def laplacian(a):
    return lap(a)

print(laplacian(np.arange(16.0).reshape(4, 4) ** 2))

@stencil(cval=-1.0)
def k_cval(a):
    return a[-1] + a[1]

@njit
def with_cval(a):
    return k_cval(a)

print(with_cval(np.arange(5.0)))
```
実行結果:
```
[0. 1. 2. 3. 4. 0.]
[0. 0. 2. 3. 4. 5. 0. 0.]
[[ 0.  0.  0.  0.]
 [ 0. 34. 34.  0.]
 [ 0. 34. 34.  0.]
 [ 0.  0.  0.  0.]]
[-1.  2.  4.  6. -1.]
```

**注意点・落とし穴**:
- `a[0]` が現在の要素、`a[-1]`/`a[1]` が前後の要素。カーネルが届かない端の要素は既定で `0`(`cval` で変更可能)になる。上の `smooth` の両端(`0.` と `0.`)が端の値。
- `neighborhood=((-2, 2),)` で参照する範囲を明示できる(上の `k5` は前後 2 要素ずつを参照するので、端の 2 要素ずつが `0.` になっている)。`cval=-1.0` を指定すると端の値が `-1.` になる。ステンシルのカーネルは他の `@njit` 関数から呼ぶ。

---

## キャッシュ(cache=True)

### `@njit(cache=True)`

**用途**: コンパイル結果をディスク(`__pycache__/*.nbi`/`*.nbc`)に保存し、次のプロセス起動時に再利用する。スクリプトを何度も起動する用途で、初回コンパイル時間を省ける。

**シグネチャ**: `@njit(cache=True)`(`dispatcher.stats` でキャッシュのヒット/ミスと保存先を確認できる)

**使用例**:
```python
# heavy.py
import time, sys
import numpy as np
from numba import njit
cache = sys.argv[1] == "1"

@njit(cache=cache)
def inner(a, b):
    return np.sqrt(a * a + b * b)

@njit(cache=cache)
def heavy(A, B):
    n, m = A.shape
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = inner(A[i, j], B[i, j])
    C = np.dot(A, B.T)
    return out.sum() + C.sum() + np.linalg.norm(A) + np.sort(A.ravel())[0]

A = np.random.default_rng(0).random((50, 50))
t = time.perf_counter()
heavy(A, A)
print(f"cache={cache}: first call {(time.perf_counter()-t)*1000:.0f} ms",
      "hits", sum(heavy.stats.cache_hits.values()),
      "misses", sum(heavy.stats.cache_misses.values()))
```
(以下は `python heavy.py 0` を 2 回、`python heavy.py 1` を 3 回続けて実行した単一マシン・単一実行の結果。`__pycache__` は事前に削除)
実行結果:
```
cache=False: first call 1770 ms hits 0 misses 1
cache=False: first call 1837 ms hits 0 misses 1
cache=True: first call 1918 ms hits 0 misses 1
cache=True: first call 232 ms hits 1 misses 0
cache=True: first call 218 ms hits 1 misses 0
```

**注意点・落とし穴**:
- `cache=True` の最初の実行(キャッシュミス)はコンパイル込みで遅く、2 回目以降(キャッシュヒット)は約 10 分の 1 の時間で初回呼び出しが終わる。`cache=False` では毎回コンパイルされる。
- キャッシュはソースファイルの更新時刻・内容に紐づく。ソースを編集(あるいは保存し直して更新日時が変わる)と該当関数のキャッシュは無効になり、再びコンパイルされる。
- `stats.cache_hits`/`cache_misses` の合計で、キャッシュが効いたか(上の `hits 1`)を確認できる。

---

### キャッシュの保存先・無効になる条件・使えないケース

**用途**: キャッシュは既定でソースファイルと同じディレクトリの `__pycache__/` に保存される(書き込めなければユーザー用のキャッシュディレクトリ)。環境変数 `NUMBA_CACHE_DIR` で保存先を変更できる。ファイルとして存在しないコード(`python -c`、標準入力、REPL)では `cache=True` は使えない。

**使用例**:
```python
import os
import numpy as np
from numba import njit

@njit(cache=True)
def f(x):
    s = 0.0
    for i in range(x.shape[0]):
        s += np.sqrt(x[i])
    return s

f(np.arange(100.0))
print(os.path.basename(f.stats.cache_path))
print(sorted(os.listdir(f.stats.cache_path)))
```
実行結果:
```
__pycache__
['cache_demo.f-6.py312.1.nbc', 'cache_demo.f-6.py312.nbi']
```

`NUMBA_CACHE_DIR` を指定して実行すると、保存先が変わる:
```python
# 実行: NUMBA_CACHE_DIR=./mycache python cache_dir_demo.py
import os
import numpy as np
from numba import njit

@njit(cache=True)
def f(x):
    return x.sum()

f(np.arange(3.0))
print(os.path.relpath(f.stats.cache_path))
```
実行結果:
```
mycache/cache2_62b01d6e9d0ea0c7b561ed73019860679fe21435
```

`python -c` や標準入力から実行した場合(ファイルがないため):
```python
# 実行: python -c "..." (標準入力も同様)
from numba import njit

@njit(cache=True)
def g(x):
    return x + 1
```
実行結果:
```
RuntimeError: cannot cache function 'g': no locator available for file '<string>'
```

**注意点・落とし穴**:
- 上の 2 つ目のコードは `NUMBA_CACHE_DIR=./mycache` を付けて実行した結果、3 つ目は `python -c` で実行した結果(最終行のエラーメッセージ)を示す。
- 保存先ディレクトリ名は元のスクリプトのパスに由来するハッシュ付きの名前になる(上の `cache2_62b0...`)。

---

## 対応・非対応機能とエラー(TypingErrorの実例)

### nopython モードで使える NumPy 機能の例

**用途**: 配列の生成・reshape・集計(`sum(axis=)`)・累積・条件分岐・ソート・線形代数・`np.random` など、頻出の NumPy 機能は `@njit` 内でそのまま使える。

**使用例**:
```python
import numpy as np
from numba import njit

@njit
def feats(a):
    b = a.reshape(2, 3)
    return (b.sum(axis=0), np.dot(b, b.T), np.cumsum(a), np.where(a > 2, a, 0.0),
            np.argsort(-a), a.mean(), np.std(a), np.linspace(0, 1, 3), np.arange(3))

for r in feats(np.arange(6.0)):
    print(r)

@njit
def more(a):
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    return (np.linalg.solve(A, np.array([9.0, 8.0])), np.sort(a), np.unique(a),
            np.concatenate((a, a)), np.maximum(a, 2.0), np.clip(a, 1, 3),
            np.median(a), np.percentile(a, 50))

for r in more(np.array([3.0, 1.0, 2.0, 1.0])):
    print(r)
```
実行結果:
```
[3. 5. 7.]
[[ 5. 14.]
 [14. 50.]]
[ 0.  1.  3.  6. 10. 15.]
[0. 0. 0. 3. 4. 5.]
[5 4 3 2 1 0]
2.5
1.707825127659933
[0.  0.5 1. ]
[0 1 2]
[2. 3.]
[1. 1. 2. 3.]
[1. 2. 3.]
[3. 1. 2. 1. 3. 1. 2. 1.]
[3. 2. 2. 2.]
[3. 1. 2. 1.]
1.5
1.5
```

**注意点・落とし穴**:
- 対応範囲は関数だけでなく引数にも及ぶ(`sum(axis=0)` は通るが `mean(axis=0)` は通らない、など。次項参照)。`np.dot` / `np.linalg.solve` は SciPy をインストールしたこの環境で動作を確認した。
- 未対応の関数・引数は、次項のような `TypingError` になる。

---

### 非対応の NumPy 関数・引数のエラー(`axis=`・`return_counts=`・`np.fft`・`default_rng`)

**用途**: `@njit` 内では、通常の NumPy で使える関数・引数の一部が使えない。エラーメッセージの `>>> 関数名(引数型...)` と `got an unexpected keyword argument ...` の行から何が未対応かを読み取る。

**使用例**:
```python
import re
import numpy as np
from numba import njit

@njit
def mean_ax(a):
    return a.mean(axis=0)

@njit
def max_ax(a):
    return np.max(a, axis=0)

@njit
def std_ax(a):
    return np.std(a, axis=1)

@njit
def uniq_counts(a):
    return np.unique(a, return_counts=True)

@njit
def fft(a):
    return np.fft.fft(a)

@njit
def rng():
    return np.random.default_rng(0).random(2)

M = np.arange(6.0).reshape(2, 3)
for f, arg in [(mean_ax, M), (max_ax, M), (std_ax, M), (uniq_counts, np.array([1, 1, 2])), (fft, np.arange(4.0)), (rng, None)]:
    try:
        f(arg) if arg is not None else f()
    except Exception as e:
        msg = str(e)
        key = [re.sub(r" from '[^']*'", " from '...'", l.strip()) for l in msg.splitlines() if l.strip().startswith(">>>") or "unexpected keyword" in l or "Unknown attribute" in l]
        print(f"{f.__name__}: {type(e).__name__}: {key[:2]}")
```
実行結果:
```
mean_ax: TypingError: ['>>> array_mean(array(float64, 2d, C), axis=Literal[int](0))', "TypingError: got an unexpected keyword argument 'axis'"]
max_ax: TypingError: ['>>> max(array(float64, 2d, C), axis=Literal[int](0))', "TypingError: got an unexpected keyword argument 'axis'"]
std_ax: TypingError: ['>>> std(array(float64, 2d, C), axis=Literal[int](1))', "TypingError: got an unexpected keyword argument 'axis'"]
uniq_counts: TypingError: ['>>> unique(array(int64, 1d, C), return_counts=Literal[bool](True))', "TypingError: got an unexpected keyword argument 'return_counts'"]
fft: TypingError: ["Unknown attribute 'fft' of type Module(<module 'numpy.fft' from '...'>)"]
rng: TypingError: ["Unknown attribute 'default_rng' of type Module(<module 'numpy.random' from '...'>)"]
```

**注意点・落とし穴**:
- `a.sum(axis=0)` は通るのに `a.mean(axis=0)` / `np.max(a, axis=0)` / `np.std(a, axis=1)` は `TypingError: got an unexpected keyword argument 'axis'` になる。対応状況は関数ごとに違うので、`axis` が必要な集計は自分で `for` ループに展開する(または `prange` で書く)。
- `np.random.default_rng()`(Generator API)は 0.66.0 でも `Unknown attribute 'default_rng'` で使えない。`np.random.rand`/`randint`/`seed`(旧 API)が使える。
- `np.fft.*` も使えない(`Unknown attribute 'fft'`)。FFT は njit の外で numpy/scipy を呼ぶ。

---

### 型統一エラー・空リスト・未定義名(型推論の失敗)

**用途**: Numba は変数ごとに 1 つの型に決める。分岐で型が食い違う、中身の型が決められない空リスト、未定義名などは `TypingError` になる。

**使用例**:
```python
from numba import njit

@njit
def unify(flag):
    if flag:
        y = 1
    else:
        y = "a"
    return y

@njit
def empty_list():
    a = []
    return a

@njit
def undefined(x):
    return undefined_name(x)

@njit
def str_method(x):
    return x.upper()

for f, arg in [(unify, True), (empty_list, None), (undefined, 1), (str_method, 3)]:
    try:
        f(arg) if arg is not None else f()
    except Exception as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        print(f"{f.__name__}: {type(e).__name__}: {lines[1][:120]}")
```
実行結果:
```
unify: TypingError: Can't unify return type from the following types: Literal[int](1), Literal[str](a)
empty_list: TypingError: Cannot infer the type of variable 'a', have imprecise type: list(undefined)<iv=None>.
undefined: TypingError: NameError: name 'undefined_name' is not defined
str_method: TypingError: Unknown attribute 'upper' of type int64
```

**注意点・落とし穴**:
- エラー本文の 1 行目は共通で `Failed in nopython mode pipeline (step: nopython frontend)`。原因は 2 行目以降に出る(上の出力はその 2 行目)。
- `unify` のように同じ変数に `int` と `str` を入れる書き方は、Python では動くが Numba では不可。戻り値の型は分岐間で揃える。

---

### `try/except`・f-string 書式指定・pandas 引数・Python 関数呼び出しの制限

**用途**: 例外処理は `except Exception` の形までしか対応しない。f-string の書式指定(`:.2f`)、pandas オブジェクト、`@njit` でない Python 関数の呼び出しも使えない。

**使用例**:
```python
import pandas as pd
from numba import njit

@njit
def zero_div_except(x):
    try:
        return 1 / x
    except ZeroDivisionError:      # 具体的な例外クラスは指定できない
        return -1

@njit
def fmt(x):
    return f"{x:.2f}"

@njit
def df_arg(df):
    return df["a"].sum()

def pyf(x):
    return x * 2

@njit
def call_py(x):
    return pyf(x)

cases = [(zero_div_except, 0), (fmt, 1.5), (df_arg, pd.DataFrame({"a": [1, 2]})), (call_py, 3)]
for f, arg in cases:
    try:
        f(arg)
    except Exception as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        pick = [l for l in lines if "limited to" in l or "format spec" in l or "Cannot determine" in l or "Untyped global" in l]
        print(f"{f.__name__}: {type(e).__name__}: {(pick or lines)[0].split(' Raised from')[0][:150]}")
```
実行結果:
```
zero_div_except: TypingError: UnsupportedError: Exception matching is limited to <class 'Exception'>
fmt: UnsupportedBytecodeError: format spec in f-strings not supported yet.
df_arg: TypingError: - argument 0: Cannot determine Numba type of <class 'pandas.DataFrame'>
call_py: TypingError: Untyped global name 'pyf': Cannot determine Numba type of <class 'function'>
```

`except Exception` での捕捉と、njit 内からの `raise` は使える:
```python
from numba import njit

@njit
def catch_all(x):
    try:
        return 1 / x
    except Exception:
        return -1

@njit
def check(x):
    if x < 0:
        raise ValueError("neg")
    return x

print(catch_all(0), catch_all(2))
try:
    check(-1)
except ValueError as e:
    print("ValueError", e)
```
実行結果:
```
-1.0 0.5
ValueError neg
```

**注意点・落とし穴**:
- 例外の送出(`raise ValueError(...)`、ゼロ除算の `ZeroDivisionError` など)には対応しており、`except Exception:` での捕捉も使える(上の例)。対応しないのは、`except ZeroDivisionError:` のように具体的な例外クラスで捕捉する形。
- pandas の `DataFrame`/`Series` は引数にできない。`df["a"].to_numpy()` の NumPy 配列を渡す。
- njit から呼べる関数は、`@njit` 済みの関数、または `numba.extending.register_jitable`/`overload` で登録した関数(後述)に限られる。

---

### グローバル変数はコンパイル時に定数として凍結される

**用途**: `@njit` 関数が参照するグローバル変数(スカラー・配列)は、コンパイル時点の値が定数として埋め込まれる。後からグローバルを書き換えても、コンパイル済みの関数には反映されない。

**使用例**:
```python
import numpy as np
from numba import njit

G = 10
ARR = np.array([1, 2, 3])

@njit
def use_g(x):
    return x + G

@njit
def use_arr():
    return ARR[0]

print(use_g(1), use_arr())
G = 100
ARR[0] = 99
print(use_g(1), use_arr())         # 変わらない
use_g.recompile()                  # 全シグネチャを再コンパイル
print(use_g(1))
```
実行結果:
```
11 1
11 1
101
```

**注意点・落とし穴**:
- 設定値は関数の引数で渡すのが安全。グローバルを更新して反映させたいなら `.recompile()`(上の最終行)が必要。

---

### `np.random` は NumPy 本体と独立した乱数状態を持つ

**用途**: `@njit` 内の `np.random.*` は、Numba 自身が持つ乱数生成器を使う。njit の外の `np.random.seed()` は njit 内の乱数に影響しない(逆も同様)。再現性が必要なら njit 内で `np.random.seed()` を呼ぶ。

**使用例**:
```python
import numpy as np
from numba import njit

@njit
def seed(s):
    np.random.seed(s)

@njit
def rnd():
    return np.random.rand(2)

np.random.seed(0)
print("numpy(seed=0):        ", np.random.rand(2))
seed(0)
print("numba(seed=0):        ", rnd())
seed(0)
print("numba(seed=0) 再現:   ", rnd())
np.random.seed(0)
print("numpy seed だけでは:  ", rnd())    # njit 側は 0 で再初期化されていない
```
実行結果:
```
numpy(seed=0):         [0.5488135  0.71518937]
numba(seed=0):         [0.5488135  0.71518937]
numba(seed=0) 再現:    [0.5488135  0.71518937]
numpy seed だけでは:   [0.60276338 0.54488318]
```

**注意点・落とし穴**:
- njit 内で `np.random.seed(0)` を呼ぶと、NumPy の旧 API と同じ乱数列(上の 1〜3 行目は同一)になる。一方、njit の外で `np.random.seed(0)` を呼んでも njit 側の乱数状態は再初期化されない(4 行目は njit 側の乱数列の続き `0.6027..., 0.5448...` になっている)。

---

### `numba.objmode`(nopython 関数内で Python 処理を一部だけ実行)

**用途**: `@njit` 関数の途中で、どうしても Python のオブジェクト・関数を使う部分だけを `with objmode(...)` ブロックで実行する。戻り値の変数名と型を `objmode(z="float64")` のように宣言する。

**シグネチャ**: `numba.objmode(*args, **kwargs)`(`with objmode(var="型"):` の形で使う)

**使用例**:
```python
import numpy as np
from numba import njit, objmode

@njit
def with_obj(x):
    y = x * 2
    with objmode(z="float64"):
        z = float(np.linalg.norm([3.0, 4.0]) + len(str(y)))    # Python の世界
    return z + y

print(with_obj(5))
```
実行結果:
```
17.0
```

**注意点・落とし穴**:
- objmode ブロックの中は通常の Python(オブジェクトモード)として実行される。ブロックに入るたびに Python との値のやり取りが必要になるため、ホットループの中に置くと高速化の効果が薄れる。

---

## 構造化データ(typed.List/typed.Dict/tuple/レコード配列/jitclass)

### `numba.typed.List`

**用途**: Numba ネイティブなリスト(型付きリスト)。njit 関数の中で作成・変更でき、関数の引数・戻り値として Python とやり取りできる。要素の型は 1 種類に固定される。

**シグネチャ**: `numba.typed.List(*args, lsttype=None, meminfo=None, allocated=0, **kwargs)` / `List.empty_list(item_type)`

**使用例**:
```python
import numpy as np
import numba
from numba import njit, types
from numba.typed import List

l = List()
l.append(1)
l.append(2)
print(l, type(l).__name__, numba.typeof(l))

l3 = List.empty_list(types.float64)
l3.append(1)                        # int を渡しても float64 として保持
print(l3)

@njit
def make(n):
    r = List()
    for i in range(n):
        r.append(i * i)
    return r

out = make(5)
print(out, type(out).__name__, out[2], len(out))

@njit
def total(lst):
    s = 0
    for v in lst:
        s += v
    return s
print(total(out))

la = List()
la.append(np.arange(3.0))
la.append(np.arange(5.0))               # 配列のリストも作れる(要素の型は揃える)

@njit
def total_arrays(lst):
    s = 0.0
    for a in lst:
        s += a.sum()
    return s
print(total_arrays(la))

try:
    bad = List()
    bad.append(1)
    bad.append("a")
except Exception as e:
    print(type(e).__name__)

@njit
def empty():
    return List()
try:
    empty()
except Exception as e:
    print(type(e).__name__, [x.strip() for x in str(e).splitlines() if "imprecise" in x])
```
実行結果:
```
[1, 2] List ListType[int64]
[1.0]
[0, 1, 4, 9, 16] List 4 5
30
13.0
TypingError
TypingError ["Cannot infer the type of variable '$14call.2' (temporary variable), have imprecise type: ListType[undefined]."]
```

**注意点・落とし穴**:
- njit 内で空の `List()` を作って返すだけだと型が決まらず `TypingError`(`Cannot infer the type of variable ...`)。何か要素を追加するか、`List.empty_list(types.int64)` で型を明示する。
- 要素の型は最初に追加した値で決まり、異なる型の値の追加(`bad.append("a")`)は `TypingError` になる。

---

### `list`(reflected list)は廃止予定 — `typed.List` を使う

**用途**: Python の通常の `list` を njit 関数の引数にするとコンパイルは通るが、`NumbaPendingDeprecationWarning`(廃止予定の型)が出る。Python 側のリストへの変更が呼び出し後に反映される「reflected list」という古い仕組み。

**使用例**:
```python
import warnings
from numba import njit

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    @njit
    def refl(x):
        x.append(99)
        return len(x)

    py = [1, 2, 3]
    print(refl(py), py)
    for i in w:
        print(i.category.__name__, "|", " ".join(str(i.message).split())[:130])
```
実行結果:
```
4 [1, 2, 3, 99]
NumbaPendingDeprecationWarning | Encountered the use of a type that is scheduled for deprecation: type 'reflected list' found for argument 'x' of function 'refl'.
```

**注意点・落とし穴**:
- njit の中での変更(`x.append(99)`)が呼び出し元の Python リストに反映されている(`[1, 2, 3, 99]`)。これが「reflected(反映される)list」の名前の由来で、廃止予定なので新規コードでは `typed.List` を使う。
- njit 内で作った `[1, 2, 3]` や list 内包表記は普通に使える(その関数の中で完結し、戻り値にするときにも通常の `list` に変換される)。

---

### `numba.typed.Dict`

**用途**: Numba ネイティブな辞書(型付き辞書)。キーと値の型を固定した辞書で、njit 内外で共有できる。

**シグネチャ**: `numba.typed.Dict(dcttype=None, meminfo=None, n_keys=0)` / `Dict.empty(key_type, value_type)`

**使用例**:
```python
from numba import njit, types
from numba.typed import Dict

d = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
d["a"] = 1.0
d["b"] = 2.5
print(d, len(d), d["a"], list(d.keys()), list(d.values()))

@njit
def use_d(d):
    d["c"] = 3.0
    s = 0.0
    for k, v in d.items():
        s += v
    return s
print(use_d(d), dict(d))

@njit
def build(n):
    d = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(n):
        d[i] = i * i
    return d
print(build(4))

@njit
def literal_dict():
    d = {}                     # njit 内では {} と書いても型付き辞書になる
    d[1] = 2.0
    d[2] = 4.0
    return d
r = literal_dict()
print(r, type(r).__name__)

@njit
def missing(d):
    return d[99]
try:
    missing(build(3))
except KeyError as e:
    print("KeyError", e)

@njit
def get_default(d):
    return d.get(99, -1)
print(get_default(build(3)))

dt = Dict.empty(types.UniTuple(types.int64, 2), types.float64)
dt[(0, 1)] = 1.0
print(dt, (0, 1) in dt)

try:
    use_d({"a": 1.0})
except Exception as e:
    print(type(e).__name__, "|", [l.strip() for l in str(e).splitlines() if "Cannot determine" in l or "non-precise" in l][:1])
```
実行結果:
```
{a: 1.0, b: 2.5} 2 1.0 ['a', 'b'] [1.0, 2.5]
6.5 {'a': 1.0, 'b': 2.5, 'c': 3.0}
{0: 0, 1: 1, 2: 4, 3: 9}
{1: 2.0, 2: 4.0} Dict
KeyError 99
-1
{(0, 1): 1.0} True
TypingError | ['non-precise type pyobject']
```

**注意点・落とし穴**:
- 通常の Python `dict` は njit 関数の引数に渡せない(`TypingError`)。`Dict.empty(...)` で作って詰め直す。
- キーにはタプルも使える(上の `UniTuple(int64, 2)`)。存在しないキーの参照は通常の `dict` と同じく `KeyError`。

---

### tuple / `collections.namedtuple` / `literal_unroll`

**用途**: tuple と namedtuple は njit の引数・戻り値・内部変数としてそのまま使える(値は不変)。型の異なる要素を持つ tuple(異種 tuple)をループするには `numba.literal_unroll` が必要。

**シグネチャ**: `numba.literal_unroll(container)`

**使用例**:
```python
import collections
import numpy as np
import numba
from numba import njit, literal_unroll

Point = collections.namedtuple("Point", ["x", "y"])

@njit
def norm(p):
    return (p.x ** 2 + p.y ** 2) ** 0.5

@njit
def make():
    return Point(1, 2.5)

print(norm(Point(3.0, 4.0)), make(), numba.typeof(Point(1, 2.0)))

@njit
def tp(t):
    return t[0] + t[1], len(t)
print(tp((1, 2.5)))

@njit
def het_sum(t):
    s = 0.0
    for v in literal_unroll(t):        # int・float・float32 を順に処理
        s += v
    return s
print(het_sum((1, 2.5, np.float32(3))))

@njit
def het_bad(t):
    s = 0.0
    for v in t:                        # 異種 tuple を普通に for すると型を決められない
        s += v
    return s
try:
    het_bad((1, 2.5))
except Exception as e:
    print(type(e).__name__, "|", [l.strip() for l in str(e).splitlines() if "getiter" in l][:1])
```
実行結果:
```
5.0 Point(x=1, y=2.5) Point(int64, float64)
(3.5, 2)
6.5
TypingError | ['Invalid use of getiter with parameters (Tuple(int64, float64))']
```

**注意点・落とし穴**:
- `for v in t:` は、要素の型が全て同じ tuple(`UniTuple`)でのみ書ける。型が混在する tuple は `literal_unroll(t)` で反復を静的に展開する必要がある。

---

### レコード配列(構造化配列)

**用途**: `np.dtype([("id", np.int32), ("val", np.float64)])` のような構造化 dtype の配列を njit に渡し、要素ごとにフィールドを属性として読み書きできる。

**使用例**:
```python
import numpy as np
from numba import njit

rec_t = np.dtype([("id", np.int32), ("val", np.float64)])
recs = np.zeros(3, dtype=rec_t)
recs["id"] = [1, 2, 3]
recs["val"] = [0.5, 1.5, 2.5]

@njit
def rsum(r):
    s = 0.0
    for i in range(r.shape[0]):
        s += r[i].val * r[i].id       # 属性アクセス
    r[0].val = 100.0                  # 書き込みも可能(呼び出し元の配列に反映)
    return s

@njit
def col_sum(r):
    return r["val"].sum()             # フィールド名での列アクセス

print(rsum(recs))
print(recs)
print(col_sum(recs))
```
実行結果:
```
11.0
[(1, 100. ) (2,   1.5) (3,   2.5)]
104.0
```

**注意点・落とし穴**:
- `r[i].val` のような属性アクセスと、`r["val"]`(フィールド名で列全体を取り出す)のどちらも njit 内で使える。

---

### `@jitclass`(numba.experimental)

**用途**: njit の中で使える(フィールド型が固定された)クラス。状態を持つ処理を njit 関数の中で扱いたいときに使う。フィールドの型は `spec` のリスト、または型アノテーションで宣言する。

**シグネチャ**: `numba.experimental.jitclass(cls_or_spec=None, spec=None)`

**使用例**:
```python
import numpy as np
from numba import njit, float64, int64
from numba.experimental import jitclass

spec = [("value", float64), ("count", int64), ("data", float64[:])]

@jitclass(spec)
class Acc:
    def __init__(self, n):
        self.value = 0.0
        self.count = 0
        self.data = np.zeros(n)

    def add(self, x):
        self.value += x
        self.data[self.count % self.data.shape[0]] = x
        self.count += 1

    def mean(self):
        return self.value / self.count

    @property
    def total(self):
        return self.value

a = Acc(3)
for v in [1.0, 2.0, 6.0, 7.0]:
    a.add(v)
print(a.mean(), a.count, a.total, a.data)

@njit
def use(n):
    acc = Acc(n)                       # njit の中でもインスタンス化できる
    for i in range(10):
        acc.add(float(i))
    return acc.mean()
print(use(4))

@jitclass
class P:                               # 型アノテーションでも宣言できる
    x: float
    y: int

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def s(self):
        return self.x + self.y
print(P(1.5, 2).s())

try:
    a.nonexistent = 3
except AttributeError as e:
    print(type(e).__name__, e)

try:
    @jitclass([("v", float64)])
    class W:
        def __init__(self, v):
            self.v = v

        def __repr__(self):
            return "W"
    W(1.0)
except TypeError as e:
    print(type(e).__name__, e)
```
実行結果:
```
4.0 4 16.0 [7. 2. 6.]
4.5
3.5
AttributeError 'Acc' object has no attribute 'nonexistent'
TypeError Method '__repr__' is not supported.
```

Python 側から呼ぶ場合と njit 内から呼ぶ場合のメソッド呼び出しコスト(単一マシン・単一実行の実測値):
```python
import time
import timeit
from numba import njit, float64
from numba.experimental import jitclass

@jitclass([("value", float64)])
class Counter:
    def __init__(self):
        self.value = 0.0

    def add(self, x):
        self.value += x

c = Counter()
c.add(1.0)
per_call = timeit.timeit(lambda: c.add(1.0), number=100000) / 100000
print(f"Python -> jitclass method: {per_call * 1e6:.2f} us/call")

@njit
def loop(obj, n):
    for i in range(n):
        obj.add(1.0)

loop(c, 1)
t = time.perf_counter()
loop(c, 10_000_000)
print(f"njit loop, 1e7 calls: {(time.perf_counter() - t) * 1000:.1f} ms")
```
実行結果:
```
Python -> jitclass method: 0.60 us/call
njit loop, 1e7 calls: 25.3 ms
```

**注意点・落とし穴**:
- `spec` にないフィールドを設定すると `AttributeError`、`__repr__` のような一部の特殊メソッドは jitclass で未対応で、インスタンス化時に `TypeError: Method '__repr__' is not supported.`(算術演算子の `__add__` などは使える)。
- フィールドを `__init__` で代入し忘れると、ゼロ初期化された値(`float64` なら `0.0`)がそのまま見える(この環境で確認)。
- Python 側からメソッドを 1 回呼ぶごとに約 0.4 マイクロ秒かかるのに対し、njit 関数の中から呼ぶと 1 回あたり約 2.5 ナノ秒(1000 万回で約 25 ms)だった(上の測定)。Python の `for` ループから jitclass のメソッドを何度も呼ぶ使い方は速くならず、ループごと njit 関数にするのが基本。

---

## 検査・デバッグ

### `func.inspect_types()`

**用途**: Numba が関数の各行・各変数にどの型を推論したかを、ソースに注釈を付けて表示する。「意図せず `float64` ではなく `int64` になっている」「オブジェクト型が混じっている」などを確認する最初の手段。

**シグネチャ**: `dispatcher.inspect_types(file=None, signature=None, pretty=False, style='default', **kwargs)`

**使用例**:
```python
import io
from numba import njit

@njit
def g(x):
    return x + 1

g(1)
g(1.5)
buf = io.StringIO()
g.inspect_types(file=buf)
for l in buf.getvalue().splitlines():
    if "::" in l or l.startswith("g ("):
        print(l.strip())
```
実行結果:
```
g (int64,)
#   x = arg(0, name=x)  :: int64
#   $const6.1.1 = const(int, 1)  :: Literal[int](1)
#   $binop_add8.2 = x + $const6.1.1  :: int64
#   $12return_value.3 = cast(value=$binop_add8.2)  :: int64
g (float64,)
#   x = arg(0, name=x)  :: float64
#   $const6.1.1 = const(int, 1)  :: Literal[int](1)
#   $binop_add8.2 = x + $const6.1.1  :: float64
#   $12return_value.3 = cast(value=$binop_add8.2)  :: float64
```

**注意点・落とし穴**:
- 通常は `f.inspect_types()` とだけ書けば標準出力に注釈付きソースが出る(ソースファイルのパスや IR が含まれる長い出力になる)。上では `::`(型注釈)を含む行だけを抽出している。
- `::` の右に `pyobject` と出る変数があれば、その部分は高速化されていない(オブジェクト扱い)。

---

### `inspect_llvm()` / `inspect_asm()`

**用途**: コンパイルされた LLVM IR とネイティブのアセンブリを、シグネチャごとに文字列で取得する。SIMD 化されたか(`vmulpd`/`vaddpd` などの AVX 命令が出ているか)を確認したいときなどに使う。

**シグネチャ**:
- `dispatcher.inspect_llvm(signature=None)`(引数を省略すると全シグネチャの辞書)
- `dispatcher.inspect_asm(signature=None)`
- `dispatcher.inspect_cfg(signature=None, ...)`(制御フローグラフ)

**使用例**:
```python
import numpy as np
from numba import njit

@njit
def f(x, n):
    s = 0.0
    for i in range(n):
        s += x[i] * 2
    return s

f(np.arange(4.0), 3)
sig = f.signatures[0]
llvm = f.inspect_llvm(sig)
asm = f.inspect_asm(sig)
print(type(llvm).__name__, llvm.splitlines()[0])
print(list(f.inspect_llvm().keys()))
print("asm has SIMD (ymm/zmm) registers:", any(("ymm" in l or "zmm" in l) for l in asm.splitlines()))
```
実行結果:
```
str ; ModuleID = 'f'
[(Array(float64, 1, 'C', False, aligned=True), int64)]
asm has SIMD (ymm/zmm) registers: True
```

---

### `NUMBA_DISABLE_JIT=1`(JIT を無効にしてデバッグ)

**用途**: 環境変数 `NUMBA_DISABLE_JIT=1` で JIT を無効にすると、`@njit` 関数がただの Python 関数として動く。`print`/`pdb`/例外のトレースバックで中身を調べるときに使う。

**使用例**:
```python
# debug_demo.py
import numpy as np
import numba
from numba import njit

print("DISABLE_JIT =", numba.config.DISABLE_JIT)

@njit
def f(x):
    s = 0
    for i in range(3):
        s += x
    return s

print(type(f).__name__, f(2))

@njit
def mixed(a):
    b = [a, "x"]                  # nopython では型を決められない書き方
    return b[0]

try:
    print(mixed(1))
except Exception as e:
    print(type(e).__name__)

@njit
def oob(x):
    return x[10]

try:
    print(oob(np.zeros(3)))
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
DISABLE_JIT = 1
function 6
1
IndexError index 10 is out of bounds for axis 0 with size 3
```

**注意点・落とし穴**:
- 通常実行(JIT 有効)では `mixed` は `TypingError`、`oob` は範囲外を読んだゴミ値を返す(次項参照)が、JIT 無効では純 Python として `1` や `IndexError: index 10 is out of bounds for axis 0 with size 3` になる。バグが Numba の変換に起因するのか、アルゴリズム自体なのかの切り分けができる。
- `numba.config.DISABLE_JIT` が `1` かは実行時に確認できる。

---

### `boundscheck=True` / `NUMBA_BOUNDSCHECK=1`(範囲外アクセスの検出)

**用途**: Numba の既定では配列の範囲外アクセスをチェックせず、範囲外の値(未定義のメモリ内容)を返したりクラッシュしたりする。`boundscheck=True`(または環境変数 `NUMBA_BOUNDSCHECK=1`)で `IndexError` を送出させる。

**使用例**:
```python
import numpy as np
from numba import njit

@njit(boundscheck=True)
def bad_checked(x):
    return x[10]

@njit
def bad_unchecked(x):
    return x[10]

try:
    print(bad_checked(np.zeros(3)))
except IndexError as e:
    print("IndexError:", e)
print(type(bad_unchecked(np.zeros(3))).__name__, "(値は不定)")
```
実行結果:
```
IndexError: index is out of bounds
float (値は不定)
```

**注意点・落とし穴**:
- 既定の `bad_unchecked(np.zeros(3))` はエラーにならず、範囲外のメモリを読んだ不定な値が返る。デバッグ時だけ `NUMBA_BOUNDSCHECK=1` を付けて実行し、本番では外す(チェックの分だけ遅くなる)。
- 環境変数 `NUMBA_BOUNDSCHECK=1` は、関数の `boundscheck=` 指定より優先されて全体に効く。

---

### `python -m numba -s` / `numba.config`

**用途**: Numba のバージョン・CPU の機能・LLVM のバージョン・スレッディングレイヤ・CUDA の状態を一覧表示する。不具合報告や環境差異の調査に使う。`numba.config` は設定変数(環境変数から読まれる)を保持する。

**シグネチャ**: `python -m numba -s`(コマンドライン)

**使用例**:
```python
import numba
print(numba.__version__)
print(numba.config.NUMBA_NUM_THREADS, repr(numba.config.THREADING_LAYER), numba.config.OPT, numba.config.DEBUG, numba.config.DISABLE_JIT)
```
実行結果:
```
0.66.0
16 'default' _OptLevel(3) 0 0
```

**注意点・落とし穴**:
- `python -m numba -s` の出力は、CPU 名(`znver5` など)、CPU 機能(`avx512f` など)、OS・Python・numba/llvmlite/LLVM のバージョン、CUDA ドライバの有無など多岐にわたる。この環境では `Numba Version : 0.66.0`、`llvmlite Version : 0.48.0`、`LLVM Version : 22.1.0`、`CUDA Device Initialized : False`。
- 主な環境変数: `NUMBA_DISABLE_JIT` / `NUMBA_NUM_THREADS` / `NUMBA_CACHE_DIR` / `NUMBA_BOUNDSCHECK` / `NUMBA_THREADING_LAYER` / `NUMBA_OPT`(最適化レベル、既定 3)。

---

## その他(cfunc/overload/register_jitable・CUDA)

### njit 関数から別の njit 関数を呼ぶ・`register_jitable`

**用途**: `@njit` 関数の中から別の `@njit` 関数を呼べる(呼び出し先も機械語になり、インライン展開されることもある)。`@njit` を付けない普通の Python 関数は呼べないが、`numba.extending.register_jitable` で登録すると、Python からも njit からも呼べる関数になる。

**シグネチャ**: `numba.extending.register_jitable(*args, **kwargs)`

**使用例**:
```python
from numba import njit
from numba.extending import register_jitable

@njit
def inner(x):
    return x * 2

@njit
def outer(x):
    return inner(x) + 1

def pyf(x):
    return x * 2

@njit
def bad(x):
    return pyf(x)

@register_jitable
def rj(x):
    return x + 1

@njit
def use_rj(x):
    return rj(x)

print(outer(3))
print(rj(1), use_rj(2))                # Python からも njit からも呼べる
try:
    bad(3)
except Exception as e:
    print(type(e).__name__, "|", [l.strip() for l in str(e).splitlines() if "Untyped global" in l][0][:100])
```
実行結果:
```
7
2 3
TypingError | Untyped global name 'pyf': Cannot determine Numba type of <class 'function'>
```

**注意点・落とし穴**:
- `@njit` 関数を共通部品として作り、他の njit 関数から呼び出す構成が基本。呼び出し関係が深いと初回コンパイル時間が伸びる(各関数がそれぞれコンパイルされる)。

---

### `@cfunc`(C コールバック関数の作成)

**用途**: Python の関数を C の関数ポインタとして呼べる形にコンパイルする。`scipy.integrate.quad` や `LowLevelCallable`、ctypes を介した C ライブラリのコールバックに渡すときに使う。

**シグネチャ**: `numba.cfunc(sig, locals={}, cache=False, pipeline_class=None, **options)`

**使用例**:
```python
from numba import cfunc

@cfunc("float64(float64)")
def cf(x):
    return x * 2

print(cf.address > 0)                         # 関数ポインタのアドレス
print(cf.ctypes(3.0))                         # ctypes 経由で C 関数として呼び出せる
print(cf.ctypes.argtypes, cf.ctypes.restype)
```
実行結果:
```
True
6.0
(<class 'ctypes.c_double'>,) <class 'ctypes.c_double'>
```

**注意点・落とし穴**:
- `cfunc` は明示シグネチャが必須(`@njit` のように呼び出し時の型では決まらない)。Python から直接呼ぶ用途ではなく、C 側にポインタを渡す用途。

---

### `numba.extending.overload`(njit からの独自関数の型別実装)

**用途**: njit 内から呼べる関数を、引数の型に応じた実装で登録する。標準ライブラリや NumPy の関数のうち Numba が未対応のものを、自分で njit 対応にするときの標準手段。

**シグネチャ**: `numba.extending.overload(func, jit_options=mappingproxy({}), strict=True, inline='never', prefer_literal=False, **kwargs)`

**使用例**:
```python
import numpy as np
from numba import njit, types
from numba.extending import overload

def my_len(x):
    pass                                     # Python 側の名前(実装は下の overload)

@overload(my_len)
def _ov_my_len(x):
    if isinstance(x, types.Array):
        return lambda x: x.size
    if isinstance(x, types.Integer):
        return lambda x: 1

@njit
def use_my_len(x):
    return my_len(x)

print(use_my_len(np.zeros((2, 3))), use_my_len(5))
```
実行結果:
```
6 1
```

**注意点・落とし穴**:
- `overload` の関数は「型を受け取って、その型用の実装(関数)を返す」形で書く。`if isinstance(x, types.Array)` のように型でディスパッチする。どの型にも合わないときは `None` を返す(そのままなら `TypingError` になる)。

---

### CUDA(`numba.cuda`)について

**用途**: `numba.cuda` は NVIDIA GPU 上で Python 関数を実行するための機能。本環境では CUDA ドライバが検出されず GPU が使えないため、実行検証はできていない。

**使用例**:
```python
import numba.cuda

print(numba.cuda.is_available())
try:
    numba.cuda.gpus
except Exception as e:
    print(type(e).__name__, "|", str(e).splitlines()[0][:80])
```
実行結果:
```
False
```

**注意点・落とし穴**:
- GPU が使える環境では `@numba.cuda.jit` でカーネルを定義して `kernel[blocks, threads](args)` で起動する形になる。ここに挙げた動作は本環境では確認していない。GPU 環境が必要な場合は `python -m numba -s` の `__CUDA Information__` で有効か確認する。
