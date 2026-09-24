# joblib 逆引き辞書

joblib 1.5.3 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(joblib 1.5.3 / Python 3.12.3 / Linux(WSL2)、論理16コア・物理8コア)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。実行時間は「単一マシン・単一実行」の実測値で、環境や負荷によって変動する。プロセスを使う並列コードはスクリプトファイルとして実行して検証した(`__main__` ガード付き)。

## 目次

1. [並列実行の基本](#1-並列実行の基本)
2. [並列設定](#2-並列設定)
3. [永続化](#3-永続化)
4. [キャッシュ](#4-キャッシュ)
5. [メモリマップ](#5-メモリマップ)
6. [ユーティリティ](#6-ユーティリティ)
7. [落とし穴・実践](#7-落とし穴実践)

---

## 1. 並列実行の基本

### `Parallel(...)` / `delayed(...)`

**用途**: 関数呼び出しのリストを複数のワーカーで並列実行し、結果を投入順のリストで受け取る。`delayed(f)(引数)` は「呼び出しを今は実行せず、`(関数, args, kwargs)` のタプルとして保留する」ためのラッパー。

**シグネチャ**:
- `joblib.Parallel(n_jobs=default(None), backend=default(None), return_as='list', verbose=default(0), timeout=None, pre_dispatch='2 * n_jobs', batch_size='auto', temp_folder=default(None), max_nbytes=default('1M'), mmap_mode=default('r'), prefer=default(None), require=default(None), **backend_kwargs)`
- `joblib.delayed(function)`

(`default(...)` は「引数未指定」を表す内部の目印で、`parallel_config` の設定があればそちらが使われる。)

**使用例**:
```python
from joblib import Parallel, delayed

def square(x):
    return x * x

print(Parallel(n_jobs=2)(delayed(square)(i) for i in range(8)))
print(delayed(square)(3))

def add(a, b=1):
    return a + b
print(Parallel(n_jobs=2)(delayed(add)(i, b=10) for i in range(4)))
```
実行結果:
```
[0, 1, 4, 9, 16, 25, 36, 49]
(<function square at 0x7be8c6c1d580>, (3,), {})
[10, 11, 12, 13]
```

**注意点・落とし穴**:
- 結果は完了順ではなく**投入順**に並ぶ(完了順で受け取りたいときは `return_as="generator_unordered"`)。
- `Parallel(...)(...)` の 2 つ目の括弧に渡すのは「`delayed(...)(...)` を yield するイテラブル」。`delayed` を付け忘れて `Parallel(n_jobs=2)(square(i) for i in range(3))` とすると、`TypeError: cannot unpack non-iterable int object` になる。
- ローカル関数(関数内で `def` したもの)やクロージャも、デフォルトの loky バックエンドでは cloudpickle 経由で動く(§7「pickle できないオブジェクト」参照)。

### `n_jobs`

**用途**: 並列ワーカー数を指定する。`1`=逐次実行(並列化なし)、`-1`=全CPU、`-2`=全CPU-1、`None`=未指定(`parallel_config` の設定があればそれ、なければ1)。

**シグネチャ**: `Parallel(n_jobs=...)`。実際の解決結果は `joblib.effective_n_jobs(n_jobs)` で確認できる。

**使用例**:
```python
from joblib import Parallel, delayed, cpu_count, effective_n_jobs

print("cpu_count:", cpu_count())
for n in [1, 2, -1, -2, None]:
    print(f"effective_n_jobs({n}) =", effective_n_jobs(n))
try:
    Parallel(n_jobs=0)(delayed(square)(i) for i in range(2))
except ValueError as e:
    print(type(e).__name__, e)
```
実行結果:
```
cpu_count: 16
effective_n_jobs(1) = 1
effective_n_jobs(2) = 2
effective_n_jobs(-1) = 16
effective_n_jobs(-2) = 15
effective_n_jobs(None) = 1
ValueError n_jobs == 0 in Parallel has no meaning
```

**注意点・落とし穴**:
- `n_jobs=None` は「全CPU」ではなく `1`(逐次)。`-1` で初めて全CPU。
- `n_jobs=-1` の「全CPU」は `cpu_count()`(論理コア数)で、物理コア数ではない。§6 `cpu_count` の環境変数・CPUアフィニティの扱いも参照。
- `n_jobs=0` は `ValueError`。

### `backend`

**用途**: 並列実行の基盤を選ぶ。`"loky"`(デフォルト、別プロセス)、`"multiprocessing"`(別プロセス、標準ライブラリの `multiprocessing.Pool`)、`"threading"`(スレッド)、`"sequential"`(逐次)。

**シグネチャ**: `Parallel(backend="loky" | "multiprocessing" | "threading" | "sequential" | 自作バックエンド名)`

**使用例**:
```python
import os
from joblib import Parallel, delayed

def work_pid(_):
    return os.getpid()

for b in ["loky", "threading", "multiprocessing", "sequential"]:
    print(b, Parallel(n_jobs=2, backend=b)(delayed(square)(i) for i in range(4)))
pids = Parallel(n_jobs=2, backend="loky")(delayed(work_pid)(i) for i in range(20))
print("loky: main pid in worker pids?", os.getpid() in pids)
pids = Parallel(n_jobs=2, backend="threading")(delayed(work_pid)(i) for i in range(20))
print("threading: all pids == main pid?", set(pids) == {os.getpid()})
try:
    Parallel(n_jobs=2, backend="foo")(delayed(square)(i) for i in range(4))
except ValueError as e:
    print(type(e).__name__, e)
```
実行結果:
```
loky [0, 1, 4, 9]
threading [0, 1, 4, 9]
multiprocessing [0, 1, 4, 9]
sequential [0, 1, 4, 9]
loky: main pid in worker pids? False
threading: all pids == main pid? True
ValueError Invalid backend: foo, expected one of ['loky', 'multiprocessing', 'sequential', 'threading']
```

**注意点・落とし穴**:
- 純Pythonの CPU バウンド処理は GIL のため `threading` では速くならない。`loky`/`multiprocessing` を使う。逆に I/O 待ち・GIL を解放する numpy 処理は `threading` が軽くて速い(実測は §7「どのバックエンドを選ぶか」)。
- `loky` と `multiprocessing` の違い: `loky` はローカル関数・ラムダ・クロージャを送れるが、`multiprocessing` は `AttributeError`(pickle不可)になる(§7)。

### `prefer` / `require`

**用途**: バックエンドを直接指名せず「性質」で希望を出す。`prefer="threads"|"processes"` はヒント、`require="sharedmem"` は必須条件(共有メモリ=スレッド系のみ)。

**シグネチャ**: `Parallel(prefer=None | "processes" | "threads", require=None | "sharedmem")`

**使用例**:
```python
print("prefer=threads    ->", type(Parallel(n_jobs=2, prefer="threads")._backend).__name__)
print("prefer=processes  ->", type(Parallel(n_jobs=2, prefer="processes")._backend).__name__)
print("require=sharedmem ->", type(Parallel(n_jobs=2, require="sharedmem")._backend).__name__)
print("backend=loky, prefer=threads ->", type(Parallel(n_jobs=2, backend="loky", prefer="threads")._backend).__name__)
try:
    Parallel(n_jobs=2, backend="loky", require="sharedmem")
except ValueError as e:
    print(type(e).__name__, str(e).split(" at 0x")[0] + " ...")
```
実行結果:
```
prefer=threads    -> ThreadingBackend
prefer=processes  -> LokyBackend
require=sharedmem -> ThreadingBackend
backend=loky, prefer=threads -> LokyBackend
ValueError Backend <joblib._parallel_backends.LokyBackend object ...
```

**注意点・落とし穴**:
- `backend` を明示すると `prefer` は無視される(`backend="loky", prefer="threads"` は loky のまま)。一方 `backend` と `require="sharedmem"` の矛盾は `ValueError`(メッセージ末尾は "does not support shared memory")。
- ライブラリ側のコードで「呼び出し元の `parallel_config` を尊重しつつスレッド向きであることだけ伝えたい」ときに `prefer="threads"` が向く。

### `batch_size`

**用途**: 1回のワーカー呼び出しにまとめる関数呼び出し数。デフォルト `"auto"` は所要時間を見て動的に調整する。

**シグネチャ**: `Parallel(batch_size="auto" | int)`

**使用例**(loky、n_jobs=4、`tiny(x)=x+1` を 20000 回。単一マシン・単一実行):
```python
import time

def tiny(x):
    return x + 1

Parallel(n_jobs=4)(delayed(tiny)(i) for i in range(8))  # プール起動のウォームアップ
N = 20000
for bs in [1, 10, 100, 1000, "auto"]:
    t = time.perf_counter()
    Parallel(n_jobs=4, batch_size=bs)(delayed(tiny)(i) for i in range(N))
    print(f"batch_size={bs!r:7} {time.perf_counter() - t:.3f}s")
```
実行結果:
```
batch_size=1       8.608s
batch_size=10      0.950s
batch_size=100     0.171s
batch_size=1000    0.073s
batch_size='auto'  0.357s
```

**注意点・落とし穴**:
- 極小タスクを大量に投げるとき、1件ずつ送る(`batch_size=1`)とプロセス間通信のオーバーヘッドで極端に遅くなる。
- `"auto"` は最適とは限らない(同じ条件で別の実行では 0.096s だった。ウォームアップや自動調整のタイミングでぶれる)。数十ms 以上かかる重いタスクなら `"auto"` のままでよい。

### `pre_dispatch`

**用途**: イテラブル(ジェネレータ)から先読みしてワーカーへ投入しておくタスク数の上限。デフォルト `'2 * n_jobs'`。巨大なジェネレータを一気にメモリへ展開しないための調整。

**シグネチャ**: `Parallel(pre_dispatch='2 * n_jobs' | 'all' | int | 式文字列)`

**使用例**(各タスク 0.3 秒、n_jobs=2、`batch_size=1`。`consumed` は「ジェネレータから i 番目が取り出された経過秒」):
```python
def sleepy(t):
    time.sleep(t); return t

t0 = time.perf_counter()
consumed = []
def gen():
    for i in range(8):
        consumed.append((i, round(time.perf_counter() - t0, 1)))
        yield delayed(sleepy)(0.3)

Parallel(n_jobs=2, pre_dispatch="all", batch_size=1)(gen())
print("all :", consumed)
consumed.clear(); t0 = time.perf_counter()
Parallel(n_jobs=2, pre_dispatch=2, batch_size=1)(gen())
print("2   :", consumed)
```
実行結果:
```
all : [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0)]
2   : [(0, 0.0), (1, 0.0), (2, 0.3), (3, 0.3), (4, 0.6), (5, 0.6), (6, 0.9), (7, 0.9)]
```

**注意点・落とし穴**:
- `"all"` はジェネレータを最初に全部消費する。データを遅延生成して巨大な入力を逐次流したいときは小さめの数値/デフォルトにする。

### `return_as`

**用途**: 結果の受け取り方。`"list"`(デフォルト、全完了後にリスト)、`"generator"`(投入順に逐次取り出し)、`"generator_unordered"`(完了した順に取り出し)。

**シグネチャ**: `Parallel(return_as="list" | "generator" | "generator_unordered")`

**使用例**(`slow(i)` は i=0 だけ 0.5 秒、他は 0.05 秒):
```python
g = Parallel(n_jobs=3, return_as="generator")(delayed(square)(i) for i in range(5))
print(type(g).__name__, list(g))

def slow(i):
    time.sleep(0.5 if i == 0 else 0.05)
    return i
print("ordered  :", list(Parallel(n_jobs=3, return_as="generator", batch_size=1)(delayed(slow)(i) for i in range(3))))
print("unordered:", list(Parallel(n_jobs=3, return_as="generator_unordered", batch_size=1)(delayed(slow)(i) for i in range(3))))
```
実行結果:
```
generator [0, 1, 4, 9, 16]
ordered  : [0, 1, 2]
unordered: [1, 2, 0]
```

**注意点・落とし穴**:
- 結果を1件ずつ処理したいとき(全結果を最後までリストとして溜めたくないとき、進捗を見ながら処理したいとき)に使う。
- `generator_unordered` は入力との対応が崩れるので、必要なら関数の戻り値に入力のIDを含める。

### `verbose`

**用途**: 進捗ログを標準エラー(stderr)に出す。数値が大きいほど詳細。

**シグネチャ**: `Parallel(verbose=0)`

**使用例**:
```python
Parallel(n_jobs=2, verbose=5)(delayed(tiny)(i) for i in range(6))
```
実行結果(stderr):
```
[Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done   6 out of   6 | elapsed:    0.3s finished
```

**注意点・落とし穴**:
- 検証時、`cross_val_score` を `with parallel_config(n_jobs=2, verbose=5):` の中で呼んでも上のようなログは出なかった(scikit-learn 側が `verbose` を明示指定しているためと推測されるが、未確認)。

### `timeout`

**用途**: 各結果を取得するときの待ち時間(秒)の上限。超えると例外。

**シグネチャ**: `Parallel(timeout=None)`

**使用例**:
```python
def sleepy(t):
    time.sleep(t); return t
try:
    Parallel(n_jobs=2, timeout=0.5)(delayed(sleepy)(t) for t in [0.1, 3])
except Exception as e:
    print(type(e).__name__, repr(str(e)))
```
実行結果:
```
TimeoutError ''
```

**注意点・落とし穴**:
- 例外メッセージは空文字。本環境では loky でのみ確認した(他のバックエンドは未検証)。
- 超過後にワーカー側の処理が止まるかは未確認。

### `Parallel` をコンテキストマネージャとして使う

**用途**: `with Parallel(...) as par:` でワーカープールを保持し、複数回 `par(...)` を呼ぶ(プール起動コストを1回に抑える)。

**シグネチャ**: `Parallel.__call__(self, iterable)`、`with Parallel(...) as par:`

**使用例**:
```python
with Parallel(n_jobs=2) as par:
    a = par(delayed(square)(i) for i in range(3))
    b = par(delayed(square)(i) for i in range(3, 6))
print(a, b)
```
実行結果:
```
[0, 1, 4] [9, 16, 25]
```

**注意点・落とし穴**:
- loky バックエンドではワーカーはプロセス間で再利用されるので、コンテキストマネージャなしでも 2 回目以降の呼び出しは速い(実測: 1回目 0.264s、直後の2回目 0.012s。単一マシン・単一実行)。

### ワーカー内で発生した例外

**用途**: ワーカーで発生した例外は、同じ型で親側に再送出される。

**使用例**:
```python
def bad(x):
    return 1 / x

try:
    Parallel(n_jobs=2)(delayed(bad)(x) for x in [1, 0])
except ZeroDivisionError as e:
    print(type(e).__name__, e, "| cause:", type(e.__cause__).__name__)
```
実行結果:
```
ZeroDivisionError division by zero | cause: _RemoteTraceback
```

**注意点・落とし穴**:
- `__cause__` にワーカー側のトレースバック文字列(`_RemoteTraceback`)が入る。デバッグ時はスタックトレース全文で確認できる。
- 失敗が1つでも起きると `Parallel` の呼び出し全体が例外で終わり、成功済みの結果は捨てられる。失敗を許容するなら関数内で `try/except` して戻り値で表現する。

---

## 2. 並列設定

### `parallel_config(...)`

**用途**: `Parallel` のデフォルト設定(バックエンド・`n_jobs`・`verbose` 等)を、`with` ブロック内、またはグローバルに設定する。`Parallel` を直接呼べない箇所(scikit-learn の内部など)の並列設定にも効く。

**シグネチャ**: `joblib.parallel_config(backend=default(None), *, n_jobs=default(None), verbose=default(0), temp_folder=default(None), max_nbytes=default('1M'), mmap_mode=default('r'), prefer=default(None), require=default(None), inner_max_num_threads=None, **backend_params)`

**使用例**:
```python
from joblib import Parallel, parallel_config

def info():
    p = Parallel()
    return type(p._backend).__name__, p.n_jobs

print("before:", info())
with parallel_config(backend="threading", n_jobs=3):
    print("inside:", info())
    p = Parallel(n_jobs=2, backend="loky")
    print("explicit args win:", type(p._backend).__name__, p.n_jobs)
print("after :", info())

parallel_config(backend="threading", n_jobs=2)      # with を使わない=グローバル設定
print("set   :", info())
parallel_config(backend="loky", n_jobs=1)           # 元に戻す
print("reset :", info())
try:
    parallel_config(backend=None)
except AttributeError as e:
    print("backend=None ->", type(e).__name__, e)
```
実行結果:
```
before: ('LokyBackend', 1)
inside: ('ThreadingBackend', 3)
explicit args win: LokyBackend 2
after : ('LokyBackend', 1)
set   : ('ThreadingBackend', 2)
reset : ('LokyBackend', 1)
backend=None -> AttributeError 'NoneType' object has no attribute 'nesting_level'
```

**注意点・落とし穴**:
- `Parallel(...)` の引数で明示した値が、`parallel_config` の設定より優先される。
- `with` を使わずに呼ぶとグローバルに設定が残る。引数なしの `parallel_config()` や `backend=None` では**リセットされない**(前者は設定が残ったまま、後者は `AttributeError`)。戻すには `parallel_config(backend="loky", n_jobs=1)` のように明示的に上書きする(上の例)。

### `parallel_backend(...)`

**用途**: `parallel_config` とほぼ同じ使い方のコンテキストマネージャ(バックエンドの切り替え)。

**シグネチャ**: `joblib.parallel_backend(backend, n_jobs=-1, inner_max_num_threads=None, **backend_params)`

**使用例**:
```python
from joblib import parallel_backend
with parallel_backend("threading", n_jobs=2):
    print(info())
```
実行結果:
```
('ThreadingBackend', 2)
```

**注意点・落とし穴**:
- `parallel_backend` は `n_jobs` のデフォルトが `-1`(`parallel_config` は未指定)。実際に `with parallel_backend("threading"):` の中で `Parallel().n_jobs` を確認すると `-1` だった。`n_jobs` を書かないと全CPUになる。

### `inner_max_num_threads`

**用途**: loky ワーカー内の BLAS/OpenMP 等のスレッド数の上限を決める(プロセス並列 × スレッド並列の「CPU過剰契約」を避ける)。

**シグネチャ**: `parallel_config(..., inner_max_num_threads=None)` / `parallel_backend(..., inner_max_num_threads=None)`

**使用例**:
```python
import os
def env_threads(_):
    return os.environ.get("OMP_NUM_THREADS"), os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS")

with parallel_config("loky", n_jobs=2, inner_max_num_threads=1):
    print(Parallel()(delayed(env_threads)(i) for i in range(1)))
```
実行結果:
```
[('1', '1', '1')]
```

**注意点・落とし穴**:
- ワーカー側の環境変数(`OMP_NUM_THREADS` など)が設定される。ワーカーが既に起動している場合の効果までは検証していない。

### scikit-learn の `n_jobs` との優先関係

**用途**: `parallel_config` の `n_jobs` は、scikit-learn のように `n_jobs=None`(デフォルト)で内部 `Parallel` を呼ぶ関数にも効く。どちらの指定が優先されるかを実測。

**使用例**(独自バックエンド `Counting`(`ThreadingBackend` を継承して `submit` 回数を数える。§6 `register_parallel_backend` 参照)を使い、`cross_val_score(cv=3)` が何回ディスパッチされたかを数える):
```python
run("config(n_jobs=2), sklearn n_jobs=None", parallel_config(backend="counting", n_jobs=2))
run("config(n_jobs=2), sklearn n_jobs=1", parallel_config(backend="counting", n_jobs=2), n_jobs=1)
run("config(n_jobs=2), sklearn n_jobs=3", parallel_config(backend="counting", n_jobs=2), n_jobs=3)
run("config(backend only), sklearn n_jobs=None", parallel_config(backend="counting"))
```
実行結果:
```
config(n_jobs=2), sklearn n_jobs=None                -> batches via backend: 3
config(n_jobs=2), sklearn n_jobs=1                   -> batches via backend: 0
config(n_jobs=2), sklearn n_jobs=3                   -> batches via backend: 3
config(backend only), sklearn n_jobs=None            -> batches via backend: 0
```

**注意点・落とし穴**:
- scikit-learn 側で `n_jobs=1` を明示すると `parallel_config` の `n_jobs` は無視され、並列化されない(逐次)。
- `parallel_config(backend=...)` だけで `n_jobs` を指定しないと、`n_jobs=None` のままなので並列化されない。`n_jobs` も併せて指定する。

---

## 3. 永続化

### `dump(...)`

**用途**: Pythonオブジェクト(特に大きな numpy 配列を含むもの)をファイルに保存する。pickle をベースに、numpy 配列を効率的に(別バッファとして)扱う。

**シグネチャ**: `joblib.dump(value, filename, compress=0, protocol=None)`

**使用例**:
```python
import numpy as np
from joblib import dump, load
from pathlib import Path

obj = {"a": 1, "arr": np.arange(5), "s": "text"}
ret = dump(obj, "obj.joblib")
print("dump returns:", ret)
print(load("obj.joblib"))
dump(obj, Path("pl.joblib"))            # pathlib.Path もOK
with open("fo.joblib", "wb") as f:      # ファイルオブジェクトもOK
    dump(obj, f)
with open("fo.joblib", "rb") as f:
    print(load(f)["s"])
```
実行結果(`dump returns` は、実行時のファイル名がリストで返る):
```
dump returns: ['obj.joblib']
{'a': 1, 'arr': array([0, 1, 2, 3, 4]), 's': 'text'}
text
```

**注意点・落とし穴**:
- 戻り値は保存したファイル名のリスト(1要素)。
- joblib が書いたファイルは、numpy 配列部分が独自形式のため、標準の `pickle.load` では読めない(`UnpicklingError: invalid load key, '\x03'.` のようになる)。読むのは必ず `joblib.load`。
- ラムダ式などpickle できないオブジェクトは保存できない(`PicklingError: Can't pickle <function ...<lambda> ...`)。

### `load(...)`

**用途**: `dump` で保存したファイルを読み込む。

**シグネチャ**: `joblib.load(filename, mmap_mode=None, ensure_native_byte_order='auto')`

**使用例**: 上の `dump` の例、および §3 `mmap_mode`・scikit-learn の例を参照。

**注意点・落とし穴**:
- **信頼できないファイルを `load` してはいけない**。pickle と同様、読み込み時に任意コードが実行される。実際に `__reduce__` で `print` を仕込んだオブジェクトが、`load` の時点で実行される(下の「scikit-learn モデルの保存」で確認)。

### `compress`

**用途**: 保存ファイルを圧縮してサイズを減らす。整数(0〜9、zlib)、圧縮方式名、`(方式名, レベル)` のタプルが使える。

**シグネチャ**: `dump(value, filename, compress=0)`。方式名は `'zlib'`, `'gzip'`, `'bz2'`, `'lzma'`, `'xz'`, `'lz4'`。

**使用例**(`zeros`(圧縮しやすい)・標準正規乱数(圧縮しにくい)・連番の 1,000,000 要素3配列=生で 24,000,000 B。単一マシン・単一実行):
```python
rng = np.random.default_rng(0)
big = {"zeros": np.zeros(1_000_000), "randn": rng.standard_normal(1_000_000), "ints": np.arange(1_000_000)}
for comp in [0, 3, ("zlib", 3), ("gzip", 3), ("bz2", 3), ("xz", 3), ("lzma", 3), "lz4"]:
    try:
        t = time.perf_counter(); dump(big, "c.joblib", compress=comp); td = time.perf_counter() - t
        t = time.perf_counter(); load("c.joblib"); tl = time.perf_counter() - t
        print(f"compress={comp!r:14} size={os.path.getsize('c.joblib'):>9,} B  dump {td:.3f}s  load {tl:.3f}s")
    except ValueError as e:
        print(f"compress={comp!r:14} ValueError: {e}")
```
実行結果:
```
compress=0              size=24,000,402 B  dump 0.028s  load 0.013s
compress=3              size=9,257,937 B  dump 0.358s  load 0.082s
compress=('zlib', 3)    size=9,257,937 B  dump 0.374s  load 0.070s
compress=('gzip', 3)    size=9,257,949 B  dump 0.363s  load 0.074s
compress=('bz2', 3)     size=8,175,085 B  dump 1.198s  load 0.604s
compress=('xz', 3)      size=7,818,944 B  dump 3.879s  load 0.608s
compress=('lzma', 3)    size=7,817,656 B  dump 4.167s  load 0.604s
compress='lz4'          ValueError: LZ4 is not installed. Install it with pip: https://python-lz4.readthedocs.io/
```

**注意点・落とし穴**:
- 圧縮すると保存・読み込みとも大幅に遅くなり(この例では dump が約13倍〜150倍、load が約5倍〜46倍)、`mmap_mode` も使えなくなる。ディスク容量やネットワーク転送を優先するときだけ使う。
- 整数の `compress=3` は `('zlib', 3)` と同じ結果(同一サイズ)。
- `lz4` は別途 `pip install lz4` が必要(未インストールだと `ValueError`)。
- 未対応の方式名は `ValueError: Non valid compression method given: "bogus". Possible values are {...}`。

### 拡張子による自動圧縮

**用途**: ファイル名の拡張子が `.z` `.gz` `.bz2` `.xz` `.lzma` なら、`compress` 引数を省略しても対応する方式で圧縮される。

**使用例**(全要素ゼロの 1,000,000 要素配列 = 8,000,000 B を各拡張子で保存):
```python
z = np.zeros(1_000_000)
for ext in [".joblib", ".gz", ".bz2", ".xz", ".lzma", ".z"]:
    dump(z, "e" + ext)
    with open("e" + ext, "rb") as f: head = f.read(3)
    print(f"{ext:8} {os.path.getsize('e' + ext):>9,} B  magic={head!r}")
```
実行結果:
```
.joblib  8,000,241 B  magic=b'\x80\x04\x95'
.gz         35,147 B  magic=b'\x1f\x8b\x08'
.bz2           263 B  magic=b'BZh'
.xz          1,480 B  magic=b'\xfd7z'
.lzma        1,406 B  magic=b']\x00\x00'
.z          35,135 B  magic=b'x^\xec'
```

**注意点・落とし穴**:
- `.joblib` 拡張子では圧縮されない(サイズが生データとほぼ同じ 8,000,241 B)。圧縮の有無がファイル名で暗黙に決まるので、意図は `compress=` で明示するとよい。

### `mmap_mode`(load)

**用途**: 保存した numpy 配列を、全体をメモリに読み込まず `numpy.memmap`(メモリマップ)として開く。巨大な配列の一部だけ触る、複数プロセスで同じデータを共有する、といった用途向け。

**シグネチャ**: `load(filename, mmap_mode=None)`。`mmap_mode` は `'r'`(読み取り専用)、`'r+'`(読み書き・ファイルに反映)、`'c'`(copy-on-write、変更はメモリ内のみ)。

**使用例**(160 MB の float64 配列。RSS はプロセスの常駐メモリ増加量。単一マシン・単一実行):
```python
arr = np.arange(20_000_000, dtype=np.float64)   # 160 MB
dump(arr, "big.joblib")
def rss_mb():
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024

r0 = rss_mb(); t = time.perf_counter(); a = load("big.joblib"); dt = time.perf_counter() - t
print(f"mmap_mode=None: {type(a).__name__:8} {dt:.4f}s  RSS +{rss_mb() - r0:.0f} MB"); del a
r0 = rss_mb(); t = time.perf_counter(); m = load("big.joblib", mmap_mode="r"); dt = time.perf_counter() - t
print(f"mmap_mode='r' : {type(m).__name__:8} {dt:.4f}s  RSS +{rss_mb() - r0:.0f} MB, writeable={m.flags.writeable}")
try: m[0] = 1
except ValueError as e: print("write:", e)
del m
mc = load("big.joblib", mmap_mode="c"); mc[0] = -1.0
print("mode 'c':", mc[0], "| file:", load("big.joblib")[0]); del mc
mr = load("big.joblib", mmap_mode="r+"); mr[0] = -5.0; mr.flush(); del mr
print("mode 'r+' -> file:", load("big.joblib")[0])
```
実行結果:
```
mmap_mode=None: ndarray  0.1196s  RSS +153 MB
mmap_mode='r' : memmap   0.0004s  RSS +0 MB, writeable=False
write: assignment destination is read-only
mode 'c': -1.0 | file: 0.0
mode 'r+' -> file: -5.0
```

**注意点・落とし穴**:
- `mmap_mode` を指定しても、メモリに乗るのは実際にアクセスした部分だけ(ロード直後の常駐増加は 0 MB)。
- dict の中の numpy 配列は memmap になるが、list などその他の要素は普通に読み込まれる(`{'x': 'memmap', 'y': 'memmap', 'z': 'list'}`)。
- **圧縮ファイルには効かない**。警告 `mmap_mode "r" is not compatible with compressed file <path>. "r" flag will be ignored.` が出て、通常の `ndarray` として読み込まれる。
- 圧縮なしで `dump` したファイルに対してのみ使う。

### scikit-learn モデルの保存

**用途**: 学習済みの scikit-learn モデルや `Pipeline` を `dump`/`load` で保存・復元する。

**使用例**:
```python
import warnings, sklearn
import sklearn.base
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, random_state=0)).fit(X, y)
dump(pipe, "m.joblib")
m = load("m.joblib")
print("sklearn", sklearn.__version__, "| same predictions:", (m.predict(X) == pipe.predict(X)).all(),
      "| size KB:", round(os.path.getsize("m.joblib") / 1e3))
dump(pipe, "mc.joblib", compress=3)
print("compress=3 size KB:", round(os.path.getsize("mc.joblib") / 1e3))

# 古いバージョンで保存したモデルを模擬(保存時だけ sklearn のバージョン文字列を書き換える)
real = sklearn.base.__version__
sklearn.base.__version__ = "0.0.1"
dump(RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y), "old.joblib")
sklearn.base.__version__ = real
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    load("old.joblib")
print(type(w[-1].message).__name__, "|", str(w[-1].message).split("\n")[0])

class Evil:
    def __reduce__(self):
        return (print, ("!!! code executed during joblib.load",))
dump(Evil(), "evil.joblib")
load("evil.joblib")
```
実行結果:
```
sklearn 1.9.0 | same predictions: True | size KB: 97
compress=3 size KB: 15
InconsistentVersionWarning | Trying to unpickle estimator RandomForestClassifier from version 0.0.1 when using version 1.9.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
!!! code executed during joblib.load
```

**注意点・落とし穴**:
- 保存したモデルは pickle であり、**ライブラリのバージョン(scikit-learn・numpy・Python)が違う環境で読むと壊れる/結果が変わる可能性がある**。scikit-learn は、保存時と読み込み時のバージョンが違うと `InconsistentVersionWarning` を出す(上記。この例では推定器(内部の各決定木やフォレスト本体)ごとに警告が出て、複数回出力された)。本番運用ではバージョンを固定し、再現可能な環境で保存・読み込みする。
- 信頼できない出所のファイルは `load` しない(上の `Evil` の例のとおり、`load` しただけでコードが実行される)。
- 木の多いランダムフォレストなどは巨大になりやすい。この例では圧縮(`compress=3`)で 97 KB → 15 KB になった。読み込みが遅くなるのとの兼ね合いで選ぶ。
- 詳しくは scikit-learn ドキュメントの「Model persistence」参照(本辞書では検証していない)。

---

## 4. キャッシュ

### `Memory(...)`

**用途**: 関数の戻り値をディスクにキャッシュする(引数のハッシュをキーにして保存)ためのオブジェクト。同じ引数で再度呼ぶと、関数を実行せずディスクから結果を読み出す。

**シグネチャ**: `joblib.Memory(location=None, backend='local', mmap_mode=None, compress=False, verbose=1, backend_options=None)`

**使用例**: 下の `Memory.cache` を参照。

**注意点・落とし穴**:
- `location=None` の `Memory` の `.cache(func)` は何もキャッシュしない(後述)。
- `verbose=1` がデフォルトで、キャッシュミス時(関数を実際に呼ぶとき)に標準出力へログが出る。静かにしたいときは `verbose=0`。

### `Memory.cache(...)`

**用途**: 関数をキャッシュ対象にする(デコレータまたは関数を渡して使う)。戻り値は `MemorizedFunc` オブジェクト。

**シグネチャ**: `Memory.cache(self, func=None, ignore=None, verbose=None, mmap_mode=False, cache_validation_callback=None)`

**使用例**(単一マシン・単一実行):
```python
import os, time
from joblib import Memory

mem = Memory(location="cache", verbose=0)
calls = []
@mem.cache
def slow_square(x):
    calls.append(x)
    time.sleep(0.5)
    return x * x

t = time.perf_counter(); r1 = slow_square(4); t1 = time.perf_counter() - t
t = time.perf_counter(); r2 = slow_square(4); t2 = time.perf_counter() - t
print(f"miss {t1:.3f}s / hit {t2:.4f}s  results {r1},{r2}  real calls {calls}")
print(type(slow_square).__name__)
for root, dirs, files in os.walk("cache"):
    depth = os.path.relpath(root, "cache").count(os.sep)
    name = os.path.basename(root)
    if len(name) > 40: name = name[:12] + "..."
    print("  " * depth + name + "/", sorted(files))
```
実行結果:
```
miss 0.516s / hit 0.0005s  results 16,16  real calls [4]
MemorizedFunc
cache/ ['.gitignore']
joblib/ []
  __main__--tm.../ []
    slow_square/ ['func_code.py']
      a68ee4c43ce0704fec19bc7fc7993e48/ ['metadata.json', 'output.pkl']
```

**注意点・落とし穴**:
- 保存先は `<location>/joblib/<モジュール名>/<関数名>/<引数ハッシュ>/output.pkl`。関数のソースコードは `func_code.py` として保存される。別途、関数のソースを `return x + 1` から `return x + 100` に書き換えて再ロードし、同じ引数 `1` で呼ぶと、結果が `2` から `101` になった(古いキャッシュは使われず再計算された)。ただし関数の外側の変更は検知されない(下の「キャッシュの盲点」)。
- 引数は位置引数・キーワード引数・デフォルト値を正規化した上でハッシュ化される。`k(1, 2)`, `k(1, b=2)`, `k(a=1, b=2)`, `k(1)`(デフォルト `b=2`)は同一の呼び出し扱い。

**使用例(引数の正規化と numpy 引数)**:
```python
mem = Memory("cache2", verbose=0)
cnt = []
@mem.cache
def k(a, b=2):
    cnt.append(1); return a + b
k(1, 2); k(1, b=2); k(a=1, b=2); k(1)
print("4 equivalent calls -> real calls:", len(cnt))

n3 = []
@mem.cache
def total(a):
    n3.append(1); return float(a.sum())
a = np.arange(1000); total(a); total(a.copy()); total(a + 1)
print("3 calls (2 unique contents) -> real calls:", len(n3))
```
実行結果:
```
4 equivalent calls -> real calls: 1
3 calls (2 unique contents) -> real calls: 2
```

### `Memory(verbose=1)` のログ

**用途**: キャッシュミス(関数の実行)をログで確認する。

**使用例**(関数は別モジュール `mymod.py` の `def add(a, b): return a + b`):
```python
import mymod
mem1 = Memory("cache1", verbose=1)
cached_add = mem1.cache(mymod.add)
cached_add(1, 2)          # 1回目=ミス
print(cached_add(1, 2))   # 2回目=ヒット(ログなし)
```
実行結果:
```
________________________________________________________________________________
[Memory] Calling mymod.add...
add(1, 2)
______________________________________________________________add - 0.0s, 0.0min
3
```

**注意点・落とし穴**:
- ログに出るのは**ミス時(実際に関数を呼んだとき)だけ**で、ヒット時は何も出ない。ログの出力先は標準出力(stdout)だった。
- スクリプトを直接実行しているとき、関数のモジュール名は `__main__--<スクリプトのパス>` になり、保存フォルダ名が長くなる(上のディレクトリ表示の `__main__--tm...`)。

### `ignore`

**用途**: キャッシュキー(ハッシュ)から除外する引数名を指定する(結果に影響しない `verbose` や `n_jobs` など)。

**シグネチャ**: `Memory.cache(ignore=["引数名", ...])`

**使用例**:
```python
n1, n2 = [], []
@mem.cache(ignore=["verbose"])
def with_ignore(x, verbose=0):
    n1.append(1); return x + 1
@mem.cache
def without_ignore(x, verbose=0):
    n2.append(1); return x + 1
with_ignore(1, verbose=0); with_ignore(1, verbose=5)
without_ignore(1, verbose=0); without_ignore(1, verbose=5)
print("ignore=['verbose']:", len(n1), "real calls | no ignore:", len(n2), "real calls")
```
実行結果:
```
ignore=['verbose']: 1 real calls | no ignore: 2 real calls
```

**注意点・落とし穴**:
- `ignore` は名前で指定する。指定していない引数は全てハッシュの対象。

### `MemorizedFunc.check_call_in_cache(...)` / `clear(...)` / `call(...)`

**用途**: 特定の引数の結果がキャッシュ済みか確認する(`check_call_in_cache`)、その関数のキャッシュを全部消す(`clear`)、キャッシュを無視して強制的に再計算する(`call`、戻り値は `(結果, メタデータ)` のタプル)。

**シグネチャ**:
- `MemorizedFunc.check_call_in_cache(self, *args, **kwargs)`
- `MemorizedFunc.clear(self, warn=True)`
- `MemorizedFunc.call(self, *args, **kwargs)`

**使用例**:
```python
print(cached_add.check_call_in_cache(1, 2), cached_add.check_call_in_cache(9, 9))
cached_add.clear(warn=False)
print("after clear:", cached_add.check_call_in_cache(1, 2))
```
実行結果:
```
True False
after clear: False
```

**注意点・落とし穴**:
- `call()` は常に関数を実行し、結果をキャッシュに書き戻す。戻り値が `(結果, {'duration': ..., 'input_args': {...}, 'time': ...})` のタプルである点に注意(結果だけ欲しいときは通常の呼び出し `f(...)` を使う)。
- `clear(warn=True)`(デフォルト)は警告ログを出す。

### `Memory(location=None)`

**用途**: キャッシュを無効化した `Memory`。コード上は `cache` を通したまま、設定だけでキャッシュをオフにできる。

**使用例**:
```python
f = Memory(location=None).cache(mymod.add)
print(type(f).__name__, f(1, 2))
```
実行結果:
```
NotMemorizedFunc 3
```

**注意点・落とし穴**:
- `MemorizedFunc` ではなく `NotMemorizedFunc` が返る(元の関数と同じ結果を返す)。`MemorizedFunc` 用のメソッド(`clear` など)を呼ぶコードは、この型では動かない可能性がある(未検証)。

### 例外はキャッシュされない

**用途**: キャッシュ対象の関数が例外を投げた場合、その結果は保存されず、次回も再実行される。

**使用例**:
```python
cnt = []
@mem.cache
def fail(x):
    cnt.append(1); raise RuntimeError("x")
for _ in range(2):
    try: fail(1)
    except RuntimeError: pass
print("real calls:", len(cnt))
```
実行結果:
```
real calls: 2
```

**注意点・落とし穴**:
- 失敗する処理を連打するとその都度フルで再実行される。

### `expires_after(...)`

**用途**: キャッシュの有効期限を設定する。`cache_validation_callback` に渡す。

**シグネチャ**: `joblib.expires_after(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)`

**使用例**:
```python
from joblib import expires_after
calls = []
@mem.cache(cache_validation_callback=expires_after(seconds=1))
def g(x):
    calls.append(x); return x
g(1); g(1); time.sleep(1.2); g(1)
print("real calls (call, call, sleep 1.2s, call):", len(calls))
```
実行結果:
```
real calls (call, call, sleep 1.2s, call): 2
```

**注意点・落とし穴**:
- 期限内(1 秒以内)の 2 回目はキャッシュヒットし、1.2 秒待った後の 3 回目は再計算された(実関数の呼び出し 2 回)。

### `Memory(mmap_mode=..., compress=...)`

**用途**: キャッシュ読み出し時に `numpy.memmap` で開く(`mmap_mode`)、キャッシュを圧縮して保存する(`compress`)。

**使用例**:
```python
memm = Memory("cache4", verbose=0, mmap_mode="r")
@memm.cache
def mk(n): return np.arange(n)
mk(10); print(type(mk(10)).__name__)

memz = Memory("cache6", verbose=0, compress=True)
@memz.cache
def zeros(): return np.zeros(1_000_000)
zeros()
sz = sum(os.path.getsize(os.path.join(r, f)) for r, d, fs in os.walk("cache6") for f in fs if f == "output.pkl")
print("compress=True output.pkl bytes:", sz, "(raw 8,000,000)")
```
実行結果:
```
memmap
compress=True output.pkl bytes: 35135 (raw 8,000,000)
```

**注意点・落とし穴**:
- 圧縮すると `load(mmap_mode=...)` で見たとおりメモリマップは効かなくなる。`Memory` でも同様と考えられるが、未検証。

### `MemorizedFunc.call_and_shelve(...)`

**用途**: 結果を返さず、結果への参照(`MemorizedResult`)だけを返す。大きな結果を今は読まずに、必要な時に `.get()` で取り出す。

**シグネチャ**: `MemorizedFunc.call_and_shelve(self, *args, **kwargs)`

**使用例**:
```python
memc = Memory("cache5", verbose=0)
@memc.cache
def mk2(n): return np.arange(n)
res = mk2.call_and_shelve(5)
print(type(res).__name__, res.get())
```
実行結果:
```
MemorizedResult [0 1 2 3 4]
```

**注意点・落とし穴**:
- `call_and_shelve` は関数を実行して結果を保存し、`MemorizedResult` を返す。結果を実際に使う時点で `.get()` を呼ぶ。

### `Memory.reduce_size(...)` / `Memory.clear(...)`

**用途**: キャッシュ全体のサイズを、バイト数・件数・古さで制限して削除する(`reduce_size`)、キャッシュを全削除する(`clear`)。

**シグネチャ**:
- `Memory.reduce_size(self, bytes_limit=None, items_limit=None, age_limit=None)`
- `Memory.clear(self, warn=True)`

**使用例**:
```python
mem7 = Memory("cache7", verbose=0)
@mem7.cache
def big(i): return np.arange(100_000) + i
for i in range(5): big(i); time.sleep(0.05)
count = lambda: sum(f == "output.pkl" for r, d, fs in os.walk("cache7") for f in fs)
print("entries:", count())
mem7.reduce_size(items_limit=2); print("after items_limit=2:", count())
mem7.reduce_size(bytes_limit="10K"); print("after bytes_limit='10K':", count())
mem7.clear(warn=False); print("after clear:", count())
```
実行結果:
```
entries: 5
after items_limit=2: 2
after bytes_limit='10K': 0
after clear: 0
```

**注意点・落とし穴**:
- `bytes_limit` は `'10K'` のような単位付き文字列も使える。この例では1件が約800KBあるため、10K以下に収まらず全件が消える。
- `Memory.clear()` 後もキャッシュのルートディレクトリ自体は残る(`.gitignore` と `joblib/` が残っていた)。

### `Pipeline(memory=...)`(scikit-learn 連携)

**用途**: scikit-learn の `Pipeline` に `Memory` を渡すと、途中のトランスフォーマの `fit_transform` 結果がキャッシュされ、ハイパーパラメータ探索で「後ろの段階だけ変更」したときに前段の再計算を省ける。

**使用例**(20000×200 のデータ、`StandardScaler → PCA(50) → LogisticRegression`。単一マシン・単一実行):
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=20000, n_features=200, random_state=0)

steps = lambda C: [("sc", StandardScaler()), ("pca", PCA(n_components=50, svd_solver="full")), ("clf", LogisticRegression(max_iter=300, C=C))]
pipe = Pipeline(steps(1.0), memory=Memory("pipe_cache", verbose=0))
t = time.perf_counter(); pipe.fit(X, y); print(f"cached pipeline fit 1 (cold cache)     {time.perf_counter() - t:.3f}s")
pipe.set_params(clf__C=0.1)
t = time.perf_counter(); pipe.fit(X, y); print(f"cached pipeline fit 2 (only clf__C)    {time.perf_counter() - t:.3f}s")
t = time.perf_counter(); Pipeline(steps(0.1)).fit(X, y); print(f"no-memory pipeline fit                 {time.perf_counter() - t:.3f}s")
```
実行結果:
```
cached pipeline fit 1 (cold cache)     1.061s
cached pipeline fit 2 (only clf__C)    0.273s
no-memory pipeline fit                 0.785s
```

**注意点・落とし穴**:
- 1回目はキャッシュ書き込みの分だけ、キャッシュなしより遅い(1.061s 対 0.785s)。2回目以降に「前段が同じ」場合だけ得をする。

### キャッシュの盲点(グローバル変数・呼び出し先の関数)

**用途**: `Memory.cache` がキャッシュの妥当性を判定するのは、**キャッシュ対象の関数自身のソースコードと引数**だけ。関数が読むグローバル変数や、呼び出している別関数の中身が変わっても検知されない。

**使用例**:
```python
FACTOR = 2
@mem.cache
def scale(x):
    return x * FACTOR
print(scale(10)); FACTOR = 100; print(scale(10), "<- stale, FACTOR is now", FACTOR)

def helper(x): return x + 1
@mem.cache
def use_helper(x):
    return helper(x)
print(use_helper(1))
def helper(x): return x + 1000
print(use_helper(1), "<- stale, helper() changed")
```
実行結果:
```
20
20 <- stale, FACTOR is now 100
2
2 <- stale, helper() changed
```

**注意点・落とし穴**:
- 副作用のない純粋関数(引数だけで結果が決まる)にだけ使う。設定値などは引数として渡す。
- 呼び出し先の関数を変えたら `mem.clear()` する、といった運用が必要。
- 関数の**中身を書き換えた場合**は自動で無効になる(`func_code.py` との比較による。本辞書ではこの動作を検証していない)。

---

## 5. メモリマップ

### `max_nbytes`

**用途**: `Parallel` が loky/multiprocessing のワーカーに大きな numpy 配列を渡すとき、pickle でコピーせず共有メモリ上のファイル(`numpy.memmap`)経由にするかどうかの閾値。この値以上の配列が自動でメモリマップ化される。

**シグネチャ**: `Parallel(max_nbytes='1M')`(`None` で無効化)

**使用例**(ワーカー内で受け取った配列の型と書き込み可否を見る):
```python
def kind(a):
    return type(a).__name__, bool(a.flags.writeable)
def try_write(a):
    try:
        a[0] = 1.0
        return "wrote"
    except Exception as e:
        return f"{type(e).__name__}: {e}"

small = np.zeros(1000)         # 8 KB
big = np.zeros(1_000_000)      # 8 MB
print("small (8KB) :", Parallel(n_jobs=2)(delayed(kind)(small) for _ in range(1)))
print("big (8MB)   :", Parallel(n_jobs=2)(delayed(kind)(big) for _ in range(1)))
print("write in worker:", Parallel(n_jobs=2)(delayed(try_write)(big) for _ in range(1)))
print("max_nbytes=None:", Parallel(n_jobs=2, max_nbytes=None)(delayed(kind)(big) for _ in range(1)))
print("max_nbytes=100 , small:", Parallel(n_jobs=2, max_nbytes=100)(delayed(kind)(small) for _ in range(1)))
```
実行結果:
```
small (8KB) : [('ndarray', True)]
big (8MB)   : [('memmap', False)]
write in worker: ['ValueError: assignment destination is read-only']
max_nbytes=None: [('ndarray', True)]
max_nbytes=100 , small: [('memmap', False)]
```

**注意点・落とし穴**:
- デフォルト(`'1M'`)では 1MB 未満の配列はそのまま pickle され、それ以上はワーカー側で**読み取り専用**の `memmap` として見える。ワーカー内で入力配列を書き換える実装は `ValueError: assignment destination is read-only` になる(コピーしてから書く)。
- `max_nbytes` は `1M`, `100` などバイト数・単位付き文字列。

### `mmap_mode`(Parallel)

**用途**: 自動メモリマップ化された配列をワーカーがどのモードで開くかを指定する(デフォルト `'r'`)。

**シグネチャ**: `Parallel(mmap_mode='r')`(`'r+'`, `'w+'`, `'c'` も指定可)

**使用例**(ワーカーが配列の先頭を書き換えたとき、親の配列が変わるか):
```python
for mode in ["r", "r+", "w+", "c"]:
    out = np.zeros(2_000_000)
    r = Parallel(n_jobs=2, mmap_mode=mode)(delayed(try_write)(out) for _ in range(1))
    print(f"mmap_mode={mode!r:5} worker: {r[0]:45} parent out[0] = {out[0]}")
```
実行結果:
```
mmap_mode='r'   worker: ValueError: assignment destination is read-only parent out[0] = 0.0
mmap_mode='r+'  worker: wrote                                         parent out[0] = 0.0
mmap_mode='w+'  worker: wrote                                         parent out[0] = 0.0
mmap_mode='c'   worker: wrote                                         parent out[0] = 0.0
```

**注意点・落とし穴**:
- `'r+'`/`'w+'`/`'c'` にするとワーカーは書き込めるが、**親の配列(`out`)には反映されなかった**(自動メモリマップは、親の配列を一時ファイルに書き出したものを開くため)。ワーカーの出力を親と共有したいなら、次項のように明示的に `np.memmap` を作る。

### `np.memmap` を使ってワーカーの出力を共有する

**用途**: 各ワーカーが結果を配列の異なる位置に書き込み、親で受け取る(戻り値としてコピーを返さない)。

**使用例**:
```python
import tempfile
def fill(out, i):
    out[i] = i * 10
    return i

tmp = tempfile.mkdtemp()
mm = np.memmap(os.path.join(tmp, "out.dat"), dtype=np.float64, shape=(8,), mode="w+")
Parallel(n_jobs=2)(delayed(fill)(mm, i) for i in range(8))
mm.flush()
print(np.asarray(mm))
```
実行結果:
```
[ 0. 10. 20. 30. 40. 50. 60. 70.]
```

**注意点・落とし穴**:
- 明示的に作った `np.memmap`(ファイルに裏付けられた書き込み可能なもの)は、そのままワーカーに渡され、書き込みが親にも見える。異なるワーカーが**同じ要素を同時に書かない**よう設計する必要がある(排他制御はされない)。
- 使い終わったら一時ファイルを消す。

### `temp_folder`

**用途**: 自動メモリマップ用の一時ファイルの置き場所を指定する。デフォルトは(この環境では)`/dev/shm`(共有メモリ上のファイルシステム)。

**シグネチャ**: `Parallel(temp_folder=None)`

**使用例**(プロセスの最初の `Parallel` 呼び出しとして実行):
```python
big = np.zeros(1_000_000)
tmp = tempfile.mkdtemp(dir=os.getcwd())
r = Parallel(n_jobs=2, temp_folder=tmp)(delayed(where)(big) for _ in range(1))   # where(a)=getattr(a, "filename", None)
print("under temp_folder:", r[0].startswith(tmp), "|", os.path.relpath(r[0], tmp).split(os.sep)[0][:16] + "...")
print("files left after return:", sum(len(f) for _, _, f in os.walk(tmp)))
```
実行結果:
```
under temp_folder: True | joblib_memmappin...
files left after return: 0
```

**注意点・落とし穴**:
- 一時ファイルは `Parallel` の呼び出しが終わると自動で消える(上の例で 0 個)。
- 観測した挙動: **同じ `n_jobs` の loky プールが既に起動済みのプロセスで `temp_folder` だけを変えて 2 回目の `Parallel` を呼ぶと、指定が反映されず前回の場所(`/dev/shm`)のままだった**。`n_jobs` を変えると反映された。原因の詳細は未調査。確実に効かせたいなら `parallel_config(temp_folder=...)` をプログラムの最初に設定するか、最初の呼び出しで指定する。
- `/dev/shm` が小さい環境では、巨大な配列で容量不足になりうる。その場合 `temp_folder` を別の場所にする。

### メモリマップ化の効果(ベンチマーク)

**用途**: 大きな配列を各タスクに渡す処理で、自動メモリマップ(デフォルト)と `max_nbytes=None`(通常の pickle コピー)を比較する。

**使用例**(100 MB の配列 `X` を 8 タスクに渡し、各タスクは1列の合計を返すだけ。n_jobs=4、ウォームアップ後に測定。単一マシン・単一実行):
```python
def col_sum(a, i):
    return float(a[:, i].sum())

X = np.random.default_rng(0).standard_normal((1_562_500, 8))
print("array MB:", X.nbytes / 1e6)
for rep in (1, 2):
    for label, kw in [("max_nbytes=None (pickle copy)", dict(max_nbytes=None)), ("default (memmap)", dict())]:
        t = time.perf_counter()
        Parallel(n_jobs=4, **kw)(delayed(col_sum)(X, i) for i in range(8))
        print(f"run {rep} {label:30} {time.perf_counter() - t:.3f}s")
t = time.perf_counter(); [col_sum(X, i) for i in range(8)]
print(f"sequential (no Parallel)       {time.perf_counter() - t:.3f}s")
```
実行結果:
```
array MB: 100.0
run 1 max_nbytes=None (pickle copy)  2.785s
run 1 default (memmap)               0.491s
run 2 max_nbytes=None (pickle copy)  2.592s
run 2 default (memmap)               0.475s
sequential (no Parallel)             0.038s
```

**注意点・落とし穴**:
- メモリマップ化で約 5〜6 倍高速になった(2.6〜2.8s → 0.48〜0.49s)が、**そもそもこのタスク(列の合計)は軽すぎて、並列化しない逐次実行(0.038s)が圧倒的に速い**。データ転送コスト(メモリマップでも 0.48s)が計算コストを上回る処理は、`Parallel` 自体をやめる。

---

## 6. ユーティリティ

### `cpu_count(...)`

**用途**: 利用可能なCPU数を返す(`n_jobs=-1` の解決に使われる)。

**シグネチャ**: `joblib.cpu_count(only_physical_cores=False)`

**使用例**:
```python
from joblib import cpu_count
print(cpu_count(), cpu_count(only_physical_cores=True))
```
```bash
LOKY_MAX_CPU_COUNT=4 python -c "from joblib import cpu_count, effective_n_jobs; print(cpu_count(), effective_n_jobs(-1))"
taskset -c 0-2 python -c "from joblib import cpu_count, effective_n_jobs; print(cpu_count(), effective_n_jobs(-1))"
```
実行結果:
```
16 8
4 4
3 3
```

**注意点・落とし穴**:
- 論理コア数を返すのがデフォルト(この環境は16)。物理コア数(8)は `only_physical_cores=True`。
- 環境変数 `LOKY_MAX_CPU_COUNT` と、`taskset` で制限した CPU アフィニティが反映される(上の 2 行目・3 行目)。

### `effective_n_jobs(...)`

**用途**: `n_jobs` の指定値(`-1` など)が、現在の設定・環境で実際に何ワーカーになるかを返す。

**シグネチャ**: `joblib.effective_n_jobs(n_jobs=-1)`

**使用例**: §1「`n_jobs`」の例を参照(`-1`→16、`-2`→15、`None`→1、`2`→2。バックエンドが `sequential` なら常に1になる、といった挙動は本辞書では確認していない)。

**注意点・落とし穴**:
- 引数のデフォルトは `-1`。`n_jobs=None` を渡すと `1`(またはアクティブな `parallel_config` の値)になる。

### `hash(...)`

**用途**: 任意のPythonオブジェクトの内容から決定的なハッシュ文字列(MD5/SHA1の16進)を作る。`Memory` の引数ハッシュに使われているのと同じ仕組み。

**シグネチャ**: `joblib.hash(obj, hash_name='md5', coerce_mmap=False)`

**使用例**:
```python
import numpy as np
from joblib import hash as jhash

print(jhash({"a": 1, "b": [1, 2, 3]}))
print("dict order ignored:", jhash({"b": [1, 2, 3], "a": 1}) == jhash({"a": 1, "b": [1, 2, 3]}))
print("set order ignored :", jhash({3, 1, 2}) == jhash({1, 2, 3}))
print("same ndarray:", jhash(np.arange(5)) == jhash(np.arange(5)), "| int vs float dtype:", jhash(np.arange(5)) == jhash(np.arange(5, dtype=float)))
print("1 vs 1.0:", jhash(1) == jhash(1.0), "| 'a' vs b'a':", jhash("a") == jhash(b"a"))
print("md5 :", jhash("x"), len(jhash("x")))
print("sha1:", jhash("x", hash_name="sha1"), len(jhash("x", hash_name="sha1")))
arr = np.arange(10); np.save("a.npy", arr)
mm = np.load("a.npy", mmap_mode="r")
print("memmap == ndarray:", jhash(mm) == jhash(arr), "| coerce_mmap=True:", jhash(mm, coerce_mmap=True) == jhash(arr, coerce_mmap=True))
try:
    jhash(lambda x: x)
except Exception as e:
    print("lambda:", type(e).__name__)
```
実行結果:
```
ad14f6c5f2d8c223a3276236e463a7df
dict order ignored: True
set order ignored : True
same ndarray: True | int vs float dtype: False
1 vs 1.0: False | 'a' vs b'a': False
md5 : 04b30cee3fc135be1c63d122b5404b19 32
sha1: 0694bef85fd9932962dedb4cd16fbcf1af3c7a47 40
memmap == ndarray: False | coerce_mmap=True: True
lambda: PicklingError
```

**注意点・落とし穴**:
- dict のキー順・set の順序は無視されるが、型は区別される(`1` と `1.0`、`str` と `bytes`、`int64` 配列と `float64` 配列は別ハッシュ)。
- `numpy.memmap` は同じ内容の `ndarray` と別ハッシュになる。同一視したいときは `coerce_mmap=True`。
- ラムダなどpickle できないオブジェクトは `PicklingError`。

### `wrap_non_picklable_objects(...)`

**用途**: 標準の pickle で送れないオブジェクト(ラムダ・ローカル関数等)を、cloudpickle でシリアライズされるようラップする。`multiprocessing` バックエンドで、引数としてラムダを渡したいときなどに使う。

**シグネチャ**: `joblib.wrap_non_picklable_objects(obj, keep_wrapper=True)`

**使用例**:
```python
from joblib import wrap_non_picklable_objects

def apply_f(f, x):
    return f(x)

fn = lambda x: x + 1
try:
    Parallel(n_jobs=2, backend="multiprocessing")(delayed(apply_f)(fn, i) for i in range(3))
except Exception as e:
    print("lambda as arg, multiprocessing:", type(e).__name__, e)
print("wrapped:", Parallel(n_jobs=2, backend="multiprocessing")(delayed(apply_f)(wrap_non_picklable_objects(fn), i) for i in range(3)))
```
実行結果:
```
lambda as arg, multiprocessing: AttributeError Can't pickle local object 'main.<locals>.<lambda>'
wrapped: [1, 2, 3]
```

**注意点・落とし穴**:
- デフォルトの loky バックエンドでは、ラップなしでも関数・ラムダ・クロージャは動いた(loky が cloudpickle を使うため)。`multiprocessing` バックエンドで必要になる。

### `register_parallel_backend(...)`

**用途**: 自作/サードパーティのバックエンドに名前を付けて登録し、`backend="名前"` や `parallel_config(backend="名前")` で使えるようにする。

**シグネチャ**: `joblib.register_parallel_backend(name, factory, make_default=False)`

**使用例**(`ThreadingBackend` を継承し、ディスパッチされたバッチ数を数えるバックエンド):
```python
from joblib import register_parallel_backend, parallel_config
from joblib._parallel_backends import ThreadingBackend

class CountingThreads(ThreadingBackend):
    n = 0
    def submit(self, func, callback=None):
        CountingThreads.n += 1
        return super().submit(func, callback)

register_parallel_backend("counting", CountingThreads)
print(Parallel(n_jobs=2, backend="counting", batch_size=1)(delayed(square)(i) for i in range(4)), "batches:", CountingThreads.n)
with parallel_config(backend="counting", n_jobs=2):
    print(Parallel(batch_size=2)(delayed(square)(i) for i in range(4)), "batches:", CountingThreads.n)
```
実行結果:
```
[0, 1, 4, 9] batches: 4
[0, 1, 4, 9] batches: 6
```

**注意点・落とし穴**:
- 公開APIは `joblib.ParallelBackendBase`(継承元)。この例は動作確認用に `joblib._parallel_backends`(非公開モジュール)の `ThreadingBackend` を継承している。
- バッチ実行の入口は 1.5.3 では `submit(self, func, callback=None)`(`apply_async` の名前を上書きしても呼ばれなかった)。
- 想定される用途は Dask 等の外部バックエンドの登録だが、本辞書では検証していない。

---

## 7. 落とし穴・実践

### どのバックエンドを選ぶか(ベンチマーク)

**用途**: 処理の種類ごとに、逐次・loky・threading の速度を実測する。

**使用例**(16コアマシン、`Parallel` のプールをウォームアップ後に測定。単一マシン・単一実行):
```python
def heavy(n):
    s = 0
    for i in range(n):
        s += i * i
    return s
def sleepy(t):
    time.sleep(t); return t
def sort_arr(seed):
    a = np.random.default_rng(seed).standard_normal(3_000_000)
    a.sort()
    return float(a[0])
def tm(f):
    t = time.perf_counter(); f(); return time.perf_counter() - t

Parallel(n_jobs=4)(delayed(abs)(-i) for i in range(4))   # warm up loky
print(f"sequential      {tm(lambda: [heavy(3_000_000) for _ in range(16)]):.3f}s")
print(f"loky n_jobs=4   {tm(lambda: Parallel(n_jobs=4)(delayed(heavy)(3_000_000) for _ in range(16))):.3f}s")
print(f"loky n_jobs=-1  {tm(lambda: Parallel(n_jobs=-1)(delayed(heavy)(3_000_000) for _ in range(16))):.3f}s")
print(f"threading n=4   {tm(lambda: Parallel(n_jobs=4, backend='threading')(delayed(heavy)(3_000_000) for _ in range(16))):.3f}s")
# (sleepy: 8 タスク × 0.2s、sort_arr: 3M要素の配列を16個ソート も同様に測定)
```
実行結果:
```
## CPU-bound pure Python: 16 tasks x 3e6 loop iterations
sequential      2.491s
loky n_jobs=4   0.714s
loky n_jobs=-1  0.852s
threading n=4   2.279s
## sleep (I/O-like): 8 tasks x 0.2 s
sequential      1.613s
threading n=8   0.220s
## numpy sort (GIL released): 16 arrays x 3M
sequential      1.121s
threading n=4   0.366s
loky n_jobs=4   0.750s
```

**注意点・落とし穴**:
- 純Python CPU処理: loky(プロセス並列)は約3.5倍速(n_jobs=4)、`threading` は GIL のためほぼ逐次と同じ。
- 待ち時間中心(sleep/I/O): スレッドで 0.2 秒台まで短縮できる(逐次の 1.6 秒に対し約7倍速)。
- numpy の `sort`(GIL を解放): `threading`(0.366s)が loky(0.750s)より速い(プロセス起動・データ転送を含まないため)。
- この環境では `n_jobs=-1`(16ワーカー)は `n_jobs=4` より少し遅かった(0.852s 対 0.714s)。ワーカー数は多いほど良いとは限らない。

### 小さなタスクの並列化は遅くなる

**用途**: 1件が極めて軽い処理(`x + 1`)を10000回、逐次・loky・threading で実行して比較する。

**使用例**(ウォームアップ後に測定。単一マシン・単一実行):
```python
def tiny(x):
    return x + 1

Parallel(n_jobs=4)(delayed(tiny)(i) for i in range(4))
N = 10000
print(f"sequential      {tm(lambda: [tiny(i) for i in range(N)]):.4f}s")
print(f"loky n_jobs=4   {tm(lambda: Parallel(n_jobs=4)(delayed(tiny)(i) for i in range(N))):.4f}s")
print(f"threading n=4   {tm(lambda: Parallel(n_jobs=4, backend='threading')(delayed(tiny)(i) for i in range(N))):.4f}s")
```
実行結果:
```
sequential      0.0016s
loky n_jobs=4   0.0842s
threading n=4   0.7567s
```

**注意点・落とし穴**:
- 逐次(0.0016s)が最速で、loky は約50倍、threading は約470倍遅い。1タスクが数ms未満なら並列化の管理コストが上回るので、並列化しない(またはまとめて1タスクを重くする)。
- この実験では `threading`(0.7567s)が loky(0.0842s)より約9倍遅かった。原因は未調査。極小タスクを大量に投げるなら `batch_size` を大きくするか、逐次実行のままにする。

### グローバル変数の変更は親に反映されない

**用途**: ワーカー内でのグローバル変数(リスト・辞書・カウンタなど)の変更は、プロセスベース(loky/multiprocessing)では親プロセスに反映されない。`threading` なら同一プロセスなので反映される。

**使用例**:
```python
results = []
def append_global(i):
    results.append(i)
    return len(results)

r = Parallel(n_jobs=2)(delayed(append_global)(i) for i in range(4))
print("loky     : worker return", r, "| parent results:", results)
Parallel(n_jobs=2, backend="threading")(delayed(append_global)(i) for i in range(4))
print("threading: parent results:", sorted(results))
```
実行結果:
```
loky     : worker return [1, 1, 1, 1] | parent results: []
threading: parent results: [0, 1, 2, 3]
```

**注意点・落とし穴**:
- 結果は必ず**戻り値**で受け取る。ワーカーの副作用(ログ・カウンタ・ファイル書き込み以外の状態変更)に頼らない。
- スレッドバックエンドで共有状態を書き換えるときはレースコンディション(排他制御)に注意。

### 乱数(`np.random`)は再現性がない

**用途**: ワーカー内で `np.random.rand()` のようなグローバル乱数を使うと、親側の `np.random.seed` の影響を受けず、実行ごとに結果が変わる。再現性が要るときは、シードをタスクごとに明示的に渡す。

**使用例**:
```python
def rand_np(_):
    return round(float(np.random.rand()), 6)
def rand_seeded(seed):
    return round(float(np.random.default_rng(seed).random()), 6)

np.random.seed(0)
print("parent draws:", [round(float(np.random.rand()), 4) for _ in range(2)])
print("loky workers:", Parallel(n_jobs=2, batch_size=1)(delayed(rand_np)(i) for i in range(3)))
print("loky workers:", Parallel(n_jobs=2, batch_size=1)(delayed(rand_np)(i) for i in range(3)))
print("explicit seed:", Parallel(n_jobs=2)(delayed(rand_seeded)(s) for s in range(3)))
print("explicit seed:", Parallel(n_jobs=2)(delayed(rand_seeded)(s) for s in range(3)))
```
実行結果:
```
parent draws: [0.5488, 0.7152]
loky workers: [0.51382, 0.640108, 0.600298]
loky workers: [0.272017, 0.248466, 0.840026]
explicit seed: [0.636962, 0.511822, 0.261612]
explicit seed: [0.636962, 0.511822, 0.261612]
```

**注意点・落とし穴**:
- 実行するたびに値が変わる(再現不能)。`seed` を関数の引数にして `np.random.default_rng(seed)` を使えば、何度実行しても同じ結果になる(上の 2 行)。
- (親で `np.random.seed(0)` を実行しても、ワーカーには伝わらない。)

### ネストした並列

**用途**: `Parallel` の中(ワーカー内)でさらに `Parallel` を呼んだ場合の挙動。

**使用例**:
```python
def inner_info(_):
    return os.getpid()
def outer(i):
    p = Parallel(n_jobs=2)
    b = type(p._backend).__name__
    pids = p(delayed(inner_info)(j) for j in range(2))
    return b, all(x == os.getpid() for x in pids)

print(Parallel(n_jobs=2)(delayed(outer)(i) for i in range(2)))
```
実行結果:
```
[('ThreadingBackend', True), ('ThreadingBackend', True)]
```

**注意点・落とし穴**:
- 内側の `Parallel(n_jobs=2)` は、loky のワーカー内ではプロセスを新たに作らず、`ThreadingBackend`(ワーカー自身のプロセス内のスレッド)になった(内側のタスクは外側と同じ PID)。プロセス爆発は起きない。
- (バージョンによって挙動が異なる可能性がある。この結果は 1.5.3 での観測。)
- 内側でCPUバウンドな純Python処理をしていると、スレッド化で並列の効果が消える。並列化は外側か内側のどちらか一方に集約する設計が安全。

### pickle できないオブジェクト

**用途**: プロセスベースのバックエンドでは、関数・引数・戻り値を pickle してワーカーに送る。ロックなど pickle できないオブジェクトを渡すとエラー。`threading` なら pickle が不要なので動く。

**使用例**:
```python
import threading
def use(lock):
    return "ok"

lock = threading.Lock()
try:
    Parallel(n_jobs=2)(delayed(use)(lock) for _ in range(2))
except Exception as e:
    print(type(e).__name__, e)
print("threading:", Parallel(n_jobs=2, backend="threading")(delayed(use)(lock) for _ in range(2)))

k = 5
print("closure with loky:", Parallel(n_jobs=2)(delayed(lambda x: x + k)(i) for i in range(3)))
try:
    Parallel(n_jobs=2, backend="multiprocessing")(delayed(lambda x: x + k)(i) for i in range(3))
except Exception as e:
    print("closure with multiprocessing:", type(e).__name__, e)
```
実行結果:
```
PicklingError Could not pickle the task to send it to the workers.
threading: ['ok', 'ok']
closure with loky: [5, 6, 7]
closure with multiprocessing: AttributeError Can't pickle local object 'main.<locals>.<genexpr>.<lambda>'
```

**注意点・落とし穴**:
- ロック・ジェネレータ・DB接続・ファイルハンドル等は loky でも送れない(`PicklingError`)。ワーカー内で生成し直す。
- ラムダ・クロージャは loky なら動くが、`multiprocessing` バックエンドでは `AttributeError`。`wrap_non_picklable_objects`(§6)か loky を使う。

### `__main__` ガード

**用途**: プロセスベースの並列処理を行うスクリプトでは、慣習として `if __name__ == "__main__":` で保護する。

**使用例**(ガードなしのスクリプトを Linux で実行):
```python
# g1.py
from joblib import Parallel, delayed
def sq(x): return x*x
print("module-level run, __name__ =", __name__)
print(Parallel(n_jobs=2)(delayed(sq)(i) for i in range(3)))
```
実行結果:
```
module-level run, __name__ = __main__
[0, 1, 4]
```

**注意点・落とし穴**:
- この Linux(WSL2)環境では、ガードなしでも loky で問題なく動いた(標準入力経由のスクリプトでも動作)。ただし Windows/macOS のように「spawn」でプロセスを起こす環境ではガードがないと再帰的にプロセスが生成される、というのが一般的な注意であり、本環境では検証していない。移植性のため常にガードを書く方が安全。

### `n_jobs=-1` の意味と注意

**用途**: `-1` は「全CPU」であり、多ければ良いとは限らない。

**注意点・落とし穴**:
- `-1` は `cpu_count()` の値(この環境は16。論理コア数)。`taskset` や `LOKY_MAX_CPU_COUNT` で制限するとその値になる(§6)。
- 全CPUを使うので、共有マシンでは `-2`(全CPU-1)や具体的な数を指定した方が無難。
- 上のベンチのとおり、`n_jobs=-1`(16)は `n_jobs=4` より遅かった(0.852s 対 0.714s)。
- ワーカー内で BLAS/OpenMP が独自にスレッドを使う処理(行列演算など)を並列化するなら、`inner_max_num_threads`(§2)でCPUの過剰契約を防ぐ。
