# h5py 逆引き辞書

h5py 3.16.0(+ safetensors 0.8.0)で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認)。

本ドキュメントの実行結果は、`h5py 3.16.0`(同梱 HDF5 2.0.0)/ `numpy 2.4.6` / `pandas 3.0.5` / `safetensors 0.8.0` / `torch 2.13.0+cpu` の環境で得たものです。**PyTables(`tables`)は未導入**のため `pandas.to_hdf` / `read_hdf` は実行例を載せず、`ImportError` になることの確認結果のみ記載しています。`dask` も未導入です。

各コード例は独立して実行できるようにしてあり(作業用のファイルはカレントディレクトリに作られます)、冒頭で以下を import 済みとします(コード例内で追加の import があるものはそのブロック内に書いています)。

```python
import os
import numpy as np
import h5py
```

h5py 3.16.0 で特に注意すべき挙動(いずれも本文で実行して確認)は以下のとおりです。

- **`File` の既定モードは `"r"`(読み取り専用)**。書き込むには `"w"` / `"a"` / `"r+"` を明示する。
- **文字列 dataset の読み出しは `bytes`**。`str` が欲しければ `d.asstr()[...]`。
- **`create_dataset(name, shape=...)` で `dtype` を省くと `H5pyDeprecationWarning`**。`dtype` は常に明示する。
- **fancy indexing は昇順・重複なし・1 軸のみ**(numpy と違い任意の整数配列は使えない)。
- **`del` してもファイルは縮まない**(コンパクト化は生きているオブジェクトを別ファイルへ `copy`)。
- **`safetensors.numpy.save_file` は非連続配列を黙って壊す**(`np.ascontiguousarray` が必須。torch 版は例外で拒否する)。
- **仮想データセットは元ファイルが無くてもエラーにならず `fillvalue` で埋まる**。

## 目次

1. [ファイル操作](#ファイル操作)
2. [グループと階層](#グループと階層)
3. [データセット作成・読み書き](#データセット作成・読み書き)
4. [チャンク・圧縮・リサイズ](#チャンク・圧縮・リサイズ)
5. [属性(attrs)](#属性attrs)
6. [複合型・文字列・可変長型](#複合型・文字列・可変長型)
7. [高度なインデックス・仮想データセット・並行アクセス](#高度なインデックス・仮想データセット・並行アクセス)
8. [pandas/numpy連携](#pandasnumpy連携)
9. [safetensors(姉妹フォーマット)](#safetensors姉妹フォーマット)

---

## ファイル操作

HDF5 ファイルを開く・閉じる・状態を調べる基本操作。

### `h5py.File(name, mode='r', ...)`

**用途**: HDF5 ファイルを開く(または作る)。返る `File` オブジェクトはルートグループでもあり、`f["path"]` や `f.create_dataset` などのグループ操作がそのまま使える。

**シグネチャ**: `h5py.File(name, mode='r', driver=None, libver=None, userblock_size=None, swmr=False, rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None, fs_strategy=None, fs_persist=False, fs_threshold=1, fs_page_size=None, page_buf_size=None, min_meta_keep=0, min_raw_keep=0, locking=None, alignment_threshold=1, alignment_interval=1, meta_block_size=None, *, track_times=False, **kwds)`

**使用例**:
```python
with h5py.File("demo.h5", "w") as f:      # 新規作成(既存なら中身ごと作り直す)
    f["x"] = np.arange(3)
    print(f.filename, f.mode, f.driver, f.libver)
print(bool(f))                             # with を抜けたら閉じている

for mode in ["r", "r+", "a", "x", "w-"]:
    try:
        with h5py.File("demo.h5", mode) as g:
            print(mode, "OK", g.mode, list(g))
    except Exception as e:
        print(mode, type(e).__name__)

try:
    h5py.File("nonexist.h5", "r")
except Exception as e:
    print(type(e).__name__)

with h5py.File("demo.h5", "r") as g:
    try:
        g.create_dataset("y", data=1)
    except Exception as e:
        print(type(e).__name__, e)

with h5py.File("new_a.h5", "a") as f:      # "a" は存在しなければ作成する
    print(f.mode, list(f))
with h5py.File("demo.h5", "w") as f:       # "w" は既存ファイルを空にする
    print(list(f))
with h5py.File("demo.h5") as f:            # mode を省略すると "r"
    print(f.mode)
```
実行結果:
```
demo.h5 r+ sec2 ('earliest', 'v200')
False
r OK r ['x']
r+ OK r+ ['x']
a OK r+ ['x']
x FileExistsError
w- FileExistsError
FileNotFoundError
ValueError Unable to synchronously create dataset (no write intent on file)
r+ []
[]
r
```

**注意点・落とし穴**:
- モードの意味(実行結果で確認): `r` 読み取り専用(存在必須)、`r+` 読み書き(存在必須)、`a` 読み書き(無ければ作成。既存ファイルは `mode` が `r+` として開く)、`w` 作成・既存なら**全削除して作り直し**、`x` / `w-` 新規作成のみ(既存なら `FileExistsError`)。
- `mode` 省略時は `"r"`(読み取り専用)。`h5py.get_config().default_file_mode` は非推奨で、警告文は `The default mode is now always 'r' (read-only).`(「バージョン情報」の項で実行確認)。
- `"r"` で開いたファイルに書き込むと `ValueError: ... (no write intent on file)`。
- HDF5 2.0.0 を同梱した本環境では、既定の `libver` は `('earliest', 'v200')`。`libver="latest"` を指定すると `('v200', 'v200')` になる(SWMR でも必要)。

---

### `f.close() / with 文 / 閉じた後のオブジェクト`

**用途**: ファイルを閉じてバッファを書き出す。`with` を使えば例外時にも確実に閉じる。

**シグネチャ**:
- `f.close()`
- `f.flush()`

**使用例**:
```python
f = h5py.File("demo.h5", "w")
d = f.create_dataset("x", data=np.arange(3))
print(f)
f.close()
print(f, d)               # ファイルを閉じると、そこから取得した dataset も閉じる
try:
    d[:]
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
<HDF5 file "demo.h5" (mode r+)>
<Closed HDF5 file> <Closed HDF5 dataset>
RuntimeError Unable to synchronously get dataspace (identifier is not of specified type)
```

**注意点・落とし穴**:
- 基本は `with` で開く。書き込み中に途中経過をディスクへ書き出したいときは `f.flush()`。
- `with` を抜けた後に、中で取り出した `Dataset` から読もうとすると `RuntimeError`。必要なデータは `d[()]` などで `numpy.ndarray` にコピーしてから抜ける。

---

### `h5py.is_hdf5(fname)`

**用途**: ファイルが HDF5 形式かどうかを判定する(存在しなければ `False`)。

**シグネチャ**: `h5py.is_hdf5(fname)`

**使用例**:
```python
with h5py.File("demo.h5", "w") as f:
    f["x"] = 1
open("plain.txt", "w").write("hi")
print(h5py.is_hdf5("demo.h5"), h5py.is_hdf5("plain.txt"), h5py.is_hdf5("nonexist.h5"))
```
実行結果:
```
True False False
```

---

### `h5py.__version__ / h5py.version / h5py.get_config()`

**用途**: h5py 本体・同梱 HDF5・依存 numpy のバージョンを確認する。バグ報告や再現性の記録に使う。

**シグネチャ**: `h5py.get_config()`

**使用例**:
```python
print(h5py.__version__)
print(h5py.version.hdf5_version)
print(h5py.version.info)

import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    h5py.get_config().default_file_mode
    print(w[0].category.__name__, w[0].message)
```
実行結果:
```
3.16.0
2.0.0
Summary of the h5py configuration
---------------------------------

h5py    3.16.0
HDF5    2.0.0
Python  3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]
sys.platform    linux
sys.maxsize     9223372036854775807
numpy   2.4.6
cython (built with) 3.2.4
numpy (built against) 2.4.2
HDF5 (built against) 2.0.0

H5pyDeprecationWarning h5py.get_config().default_file_mode is deprecated. The default mode is now always 'r' (read-only).
```

**注意点・落とし穴**:
- `h5py.get_config().default_file_mode` は非推奨(アクセスすると上のように `H5pyDeprecationWarning`)。
- `h5py.version.info` は複数行の文字列で、HDF5 のビルド情報・Python・numpy のバージョンも含む。

---

### `driver="core" / ファイルオブジェクト(メモリ上の HDF5)`

**用途**: ディスクに書かずメモリ上だけで HDF5 を扱う。テストや、バイト列としての受け渡し(`BytesIO`)に使う。

**シグネチャ**: `h5py.File(name, mode, driver=..., rdcc_nbytes=..., ...)`(全引数のシグネチャは `h5py.File` の項)

**使用例**:
```python
import io
# (1) core ドライバ: backing_store=False ならディスクには何も書かれない
f = h5py.File("mem.h5", "w", driver="core", backing_store=False)
f["a"] = np.arange(5)
print(f.driver, list(f))
f.close()
print(os.path.exists("mem.h5"))

# (2) BytesIO: HDF5 をバイト列として取り出して、読み直す
bio = io.BytesIO()
with h5py.File(bio, "w") as f:
    f["a"] = np.arange(5)
data = bio.getvalue()
print(data[:8])                             # HDF5 のシグネチャ
with h5py.File(io.BytesIO(data), "r") as f:
    print(f["a"][:])
```
実行結果:
```
core ['a']
False
b'\x89HDF\r\n\x1a\n'
[0 1 2 3 4]
```

**注意点・落とし穴**:
- `driver="core"` でも `name`(非空のファイル名)が必要。`backing_store=False` ならその名前のファイルはディスクに作られない(上の例で `os.path.exists` が `False`)。`backing_store` を指定しない既定では閉じるときにファイルへ書き出される(別途確認: `os.path.exists` が `True`)。

---

## グループと階層

HDF5 はファイルシステムのような階層構造(グループ=ディレクトリ、データセット=ファイル)。パスは `/` 区切りで指定する。

### `f.create_group(name) / f.require_group(name)`

**用途**: グループ(フォルダ相当)を作る。`create_group` は既存だとエラー、`require_group` は既存ならそれを返し、無ければ作る。

**シグネチャ**:
- `f.create_group(name, track_order=None, *, track_times=False)`
- `f.require_group(name)`

**使用例**:
```python
with h5py.File("g.h5", "w") as f:
    g = f.create_group("exp/run1")            # 途中の "exp" も自動で作られる
    print(g, g.name, g.parent.name)
    g2 = f.require_group("exp/run1")          # 既存なので同じグループが返る
    print(g2 == g)
    try:
        f.create_group("exp")
    except Exception as e:
        print(type(e).__name__, e)
    print(list(f), list(f["exp"]))
```
実行結果:
```
<HDF5 group "/exp/run1" (0 members)> /exp/run1 /exp
True
ValueError Unable to synchronously create group (name already exists)
['exp'] ['run1']
```

**注意点・落とし穴**:
- `create_group` で既に存在する名前を指定すると `ValueError: Unable to synchronously create group (name already exists)`(`KeyError` ではない)。冪等にしたいときは `require_group`。
- グループ内の要素の順序は既定では**名前順**(作成順ではない)。`h5py.File(..., track_order=True)` または `create_group(..., track_order=True)` で作成順を保持できる(下の項目で実行確認)。

---

### `f[path] / keys() / values() / items() / in / len() / get()`

**用途**: グループの中身を辞書のように列挙・参照する。

**シグネチャ**: `f.get(name, default=None, getclass=False, getlink=False, elink_mode=None, elink_locking=None, elink_swmr=None)`

**使用例**:
```python
with h5py.File("g.h5", "w") as f:
    f["exp/run1/loss"] = np.arange(3.0)
    f["exp/run2/loss"] = np.arange(5.0)
    f["exp/run2/acc"] = np.arange(5.0) / 5
    print(list(f.keys()), list(f["exp"].keys()))
    print(len(f), "exp" in f, "exp/run1/loss" in f, "nope" in f)
    print(list(f["exp"].items()))
    print(f["/exp/run1/loss"], f["exp"]["run1"]["loss"][:])
    for k, v in f["exp/run2"].items():
        print(k, v)
    try:
        f["nope"]
    except KeyError as e:
        print("KeyError", e)
    print(f.get("nope"), f.get("exp"))
    print(f.get("exp", getclass=True), f.get("exp/run1/loss", getclass=True))
```
実行結果:
```
['exp'] ['run1', 'run2']
1 True True False
[('run1', <HDF5 group "/exp/run1" (1 members)>), ('run2', <HDF5 group "/exp/run2" (2 members)>)]
<HDF5 dataset "loss": shape (3,), type "<f8"> [0. 1. 2.]
acc <HDF5 dataset "acc": shape (5,), type "<f8">
loss <HDF5 dataset "loss": shape (5,), type "<f8">
KeyError "Unable to synchronously open object (object 'nope' doesn't exist)"
None <HDF5 group "/exp" (2 members)>
<class 'h5py._hl.group.Group'> <class 'h5py._hl.dataset.Dataset'>
```

**注意点・落とし穴**:
- 存在しないパスは `KeyError`。`f.get(path)` は `None`(または `default`)を返すので存在確認つきの取得に便利。
- `len(f)` と `keys()` は**直下の要素だけ**(再帰しない)。全階層を辿るには `visit` / `visititems`。
- `f.get(path, getclass=True)` で中身を開かずに型(`Group` / `Dataset`)だけ調べられる。

---

### `f.visit(func) / f.visititems(func) / f.visit_links(func)`

**用途**: グループ配下を**再帰的に**全走査する。`func` は相対パス名(`visititems` は名前とオブジェクト)で呼ばれる。

**シグネチャ**:
- `f.visit(func)`
- `f.visititems(func)`
- `f.visit_links(func)`

**使用例**:
```python
with h5py.File("g.h5", "w") as f:
    f["exp/run1/loss"] = np.arange(3.0)
    f["exp/run2/loss"] = np.arange(5.0)
    f["exp/run2/acc"] = np.arange(5.0) / 5
    f["alias"] = h5py.SoftLink("/exp/run1")

    f.visit(lambda name: print("visit", name))

    def show(name, obj):
        print(name, type(obj).__name__, getattr(obj, "shape", ""))
    f.visititems(show)

    # コールバックが None 以外を返すとそこで走査が止まり、その値が返る
    print(f.visit(lambda n: n if n.endswith("acc") else None))

    f.visit_links(lambda name: print("visit_links", name))
```
実行結果:
```
visit exp
visit exp/run1
visit exp/run1/loss
visit exp/run2
visit exp/run2/acc
visit exp/run2/loss
exp Group 
exp/run1 Group 
exp/run1/loss Dataset (3,)
exp/run2 Group 
exp/run2/acc Dataset (5,)
exp/run2/loss Dataset (5,)
exp/run2/acc
visit_links alias
visit_links exp
visit_links exp/run1
visit_links exp/run1/loss
visit_links exp/run2
visit_links exp/run2/acc
visit_links exp/run2/loss
```

**注意点・落とし穴**:
- コールバックは**相対パス**(呼び出したグループからの)で呼ばれる。全データセットのうち条件に合うものを集めたいときは、`visititems` の中で `isinstance(obj, h5py.Dataset)` で絞る。
- `visit` / `visititems` は**ソフトリンク・外部リンクを辿らない**(上の例で `alias` は `visit` に出ない)。`visit_links` はリンクそのものも列挙する。

---

### `del f[path] / f.move(source, dest) / f.copy(source, dest, ...)`

**用途**: オブジェクトの削除・改名(移動)・コピー。`copy` は別ファイルへもコピーできる。

**シグネチャ**:
- `f.move(source, dest)`
- `f.copy(source, dest, name=None, shallow=False, expand_soft=False, expand_external=False, expand_refs=False, without_attrs=False)`

**使用例**:
```python
with h5py.File("g.h5", "w") as f:
    f["exp/run1/loss"] = np.arange(3.0)
    f["exp/run2/acc"] = np.arange(5.0)
    f.move("exp/run2/acc", "exp/run2/accuracy")
    print(list(f["exp/run2"]))
    f.copy("exp/run1", "exp/run3")             # グループごと(中身・属性も)コピー
    print(list(f["exp"]))
    del f["exp/run3"]
    print(list(f["exp"]))

# 別ファイルへのコピー
with h5py.File("src.h5", "w") as s:
    s["g/a"] = np.arange(4)
    s["g"].attrs["k"] = 1
with h5py.File("src.h5", "r") as s, h5py.File("dst.h5", "w") as d:
    s.copy("g", d, name="copied")
    s.copy(s["g/a"], d, name="solo")
    print(list(d), list(d["copied"]), dict(d["copied"].attrs))
```
実行結果:
```
['accuracy']
['run1', 'run2', 'run3']
['run1', 'run2']
['copied', 'solo'] ['a'] {'k': np.int64(1)}
```

**注意点・落とし穴**:
- **`del` してもファイルサイズは縮まない**(下の「削除してもファイルは縮まない」の項で実測)。コンパクトにするには、生きているオブジェクトを新しいファイルへ `copy` する。
- `copy` は既定でグループを再帰的にコピーし、属性も付く(`without_attrs=True` で除外可能)。`copy(..., expand_soft=True)` を付けるとソフトリンクが実体(ハードリンク)としてコピーされる(付けないと `<SoftLink to "/g/a">` のままコピーされることを確認)。

---

### `削除してもファイルは縮まない / コンパクト化`

**用途**: `del f[name]` で dataset を消しても、ファイルサイズは自動では小さくならない。小さくしたいときの手順。

**シグネチャ**: `del f[name]` / `f.copy(source, dest)`

**使用例**:
```python
def mb(p):
    return "%.3f MB" % (os.path.getsize(p) / 1e6)

with h5py.File("del.h5", "w") as f:
    f["a"] = np.zeros(1_000_000)             # 8MB
    f["b"] = np.arange(3)
print("作成直後      ", mb("del.h5"))
with h5py.File("del.h5", "a") as f:
    del f["a"]
print("a を削除後    ", mb("del.h5"))
with h5py.File("del.h5", "a") as f:
    f["c"] = np.zeros(500_000)               # 4MB を追加
print("4MB 追加後    ", mb("del.h5"))

# 生きているオブジェクトだけを新しいファイルへコピーして作り直す
with h5py.File("del.h5", "r") as s, h5py.File("del2.h5", "w") as d:
    for k in s:
        s.copy(k, d)
print("コピーで再構築", mb("del2.h5"))
```
実行結果:
```
作成直後       8.002 MB
a を削除後     8.002 MB
4MB 追加後     12.004 MB
コピーで再構築 4.002 MB
```

**注意点・落とし穴**:
- 別セッションで開き直して追加しても、削除した領域は再利用されず、ファイルが 8MB + 4MB に伸びた(上の出力)。
- `h5repack` などの CLI ツールは本環境では PATH 上に無かった(`shutil.which("h5repack")` が `None`)ため、ここでは `copy` による再構築を示した。

---

### `h5py.SoftLink / h5py.HardLink / h5py.ExternalLink`

**用途**: リンクを作る。ハードリンクは同一オブジェクトへの別名、ソフトリンクはパス文字列、外部リンクは別ファイル内のオブジェクトを指す。

**シグネチャ**:
- `h5py.SoftLink(path)`
- `h5py.HardLink()`
- `h5py.ExternalLink(filename, path)`

**使用例**:
```python
with h5py.File("l1.h5", "w") as f:
    f["data/a"] = np.arange(4)
    f["soft"] = h5py.SoftLink("/data/a")
    f["hard"] = f["data/a"]                     # ハードリンク(同じオブジェクトへの別名)
    f["dangling"] = h5py.SoftLink("/nowhere")
    print(f.get("soft", getlink=True), f.get("soft", getlink=True).path)
    print(type(f.get("hard", getlink=True)).__name__)
    print(f["soft"][:], f["soft"].name, f["hard"] == f["data/a"])
    print("dangling" in f)                       # リンク切れでも in は True
    try:
        f["dangling"]
    except KeyError as e:
        print("KeyError", e)
    f["data/a"][0] = 99
    print(f["hard"][:])                          # 実体が共有されている
    del f["data/a"]
    print(f["hard"][:])                          # ハードリンク側が残っていれば実体は残る

with h5py.File("l2.h5", "w") as f:
    f["ext"] = h5py.ExternalLink("l1.h5", "/hard")
    link = f.get("ext", getlink=True)
    print(link.filename, link.path)
    print(f["ext"][:], type(f["ext"]).__name__, os.path.basename(f["ext"].file.filename))
```
実行結果:
```
<SoftLink to "/data/a"> /data/a
HardLink
[0 1 2 3] /soft True
True
KeyError 'Unable to synchronously open object (component not found)'
[99  1  2  3]
[99  1  2  3]
l1.h5 /hard
[99  1  2  3] Dataset l1.h5
```

**注意点・落とし穴**:
- ソフトリンクは実体が消えても壊れたまま残り、`"dangling" in f` は `True` を返すが、実際に開くと `KeyError`。存在確認としては `f.get(path)`(`None` が返る)の方が確実。
- 外部リンクのファイル名は、リンクを持つファイルのあるディレクトリからの相対パスとして解決される(別ディレクトリのリンク元から相対名で辿れることを確認)。参照先ファイルを移動して名前が解決できなくなると `KeyError: ... (can't open file)`(`FileNotFoundError` ではない)。
- ソフトリンク経由で開いた dataset の `.name` は、実体のパスではなく**開いたときのリンク名**(上の例では `/soft`)になる。

---

### `track_order=True(作成順を保持)`

**用途**: グループ内の要素を、名前順ではなく作成順に列挙したいときに指定する。

**シグネチャ**: `f.create_group(name, track_order=None, *, track_times=False)`

**使用例**:
```python
with h5py.File("g2.h5", "w") as f:
    for n in ["b", "a", "c"]:
        f.create_group(n)
    print(list(f))
with h5py.File("g3.h5", "w", track_order=True) as f:
    for n in ["b", "a", "c"]:
        f.create_group(n)
    print(list(f))
    d = f.create_dataset("x", data=1, track_order=True)
    for k in ["z", "a", "m"]:
        d.attrs[k] = 1
    print(list(d.attrs))
    d = f.create_dataset("y", data=1)
    for k in ["z", "a", "m"]:
        d.attrs[k] = 1
    print(list(d.attrs))
```
実行結果:
```
['a', 'b', 'c']
['b', 'a', 'c']
['z', 'a', 'm']
['a', 'm', 'z']
```

**注意点・落とし穴**:
- `create_dataset(..., track_order=True)` を指定すると、そのデータセットの属性(`attrs`)も作成順で返る(`z, a, m` の順に付けて `['z', 'a', 'm']`)。指定しない場合の属性は名前順(`['a', 'm', 'z']`)。

---

## データセット作成・読み書き

`Dataset` は numpy 配列のようにスライスで読み書きできるが、中身はディスク上にある。`d[...]` で読んだ時点で初めて `numpy.ndarray` になる。

### `f.create_dataset(name, shape, dtype, data, ...)`

**用途**: データセットを作る。`data=` に配列(list でも可)を渡すか、`shape` と `dtype` を渡して領域だけ確保する。

**シグネチャ**: `f.create_dataset(name, shape=None, dtype=None, data=None, **kwds)`

**使用例**:
```python
with h5py.File("d.h5", "w") as f:
    a = np.arange(12).reshape(3, 4)
    d = f.create_dataset("a", data=a)
    print(d)
    print(d.shape, d.dtype, d.size, d.ndim, d.nbytes, d.chunks, d.compression, d.maxshape, d.fillvalue)

    e = f.create_dataset("empty", shape=(2, 3), dtype="f4")        # 未書き込みは 0
    print(e[:], e.fillvalue)
    fv = f.create_dataset("fv", shape=(2, 3), dtype="i4", fillvalue=-1)
    print(fv[:], fv.fillvalue)

    print(f.create_dataset("lst", data=[1, 2, 3]).dtype)            # list からは dtype を推論
    f.create_dataset("scalar", data=3.14)
    print(f["scalar"].shape, f["scalar"][()], type(f["scalar"][()]))
    f["auto"] = np.linspace(0, 1, 5)                                # 代入でも作れる
    print(f["auto"].dtype)
    f.create_dataset("cast", data=[1.5, 2.5], dtype="i2")           # data を dtype へ変換して保存
    print(f["cast"][:], f["cast"].dtype)
    for name, arr in [("bool", np.array([True, False])), ("complex", np.array([1 + 2j])), ("f2", np.array([1.5], dtype="f2"))]:
        f[name] = arr
        print(name, f[name].dtype, f[name][:])

    import warnings
    with warnings.catch_warnings(record=True) as w:                  # dtype も data も渡さない場合
        warnings.simplefilter("always")
        f.create_dataset("nodtype", shape=(2,))
        print(w[0].category.__name__, w[0].message)

    try:
        f.create_dataset("a", data=1)
    except Exception as ex:
        print(type(ex).__name__, ex)
    for label, arr in [("文字列<U", np.array(["a", "b"])), ("datetime64", np.array(["2020-01-01"], dtype="datetime64[D]"))]:
        try:
            f[label] = arr
        except Exception as ex:
            print(label, type(ex).__name__, ex)
```
実行結果:
```
<HDF5 dataset "a": shape (3, 4), type "<i8">
(3, 4) int64 12 2 96 None None (3, 4) 0
[[0. 0. 0.]
 [0. 0. 0.]] 0.0
[[-1 -1 -1]
 [-1 -1 -1]] -1
int64
() 3.14 <class 'numpy.float64'>
float64
[1 2] int16
bool bool [ True False]
complex complex128 [1.+2.j]
f2 float16 [1.5]
H5pyDeprecationWarning Creating a dataset without passing data or dtype is deprecated. Pass an explicit dtype. Using dtype='f4' will keep the current default behaviour.
ValueError Unable to synchronously create dataset (name already exists)
文字列<U TypeError No conversion path for dtype: dtype('<U1')
datetime64 TypeError No conversion path for dtype: dtype('<M8[D]')
```

**注意点・落とし穴**:
- 既存の名前に対して `create_dataset` すると `ValueError: ... (name already exists)`。上書きしたければ先に `del f[name]`、または `require_dataset`。
- `data=` を渡さず `shape` だけを渡して `dtype` を省くと、h5py 3.16.0 では `H5pyDeprecationWarning: Creating a dataset without passing data or dtype is deprecated. Pass an explicit dtype. Using dtype='f4' will keep the current default behaviour.` が出る(実行確認)。常に `dtype` を明示する。
- numpy の Unicode 配列(`<U`)や `datetime64` は `TypeError: No conversion path for dtype: ...`。文字列は `h5py.string_dtype()`、日時は整数/文字列に直して保存する。
- `bool` / `complex128` / `float16` は保存できる(上の例で dtype を保ったまま読み戻せた)。
- その他の主な引数: `chunks` / `compression` / `shuffle` / `maxshape` / `fillvalue`(次カテゴリ参照)、`track_order`。

---

### `d[()] / d[:] / d[...] / スライス / 部分読み込み`

**用途**: `Dataset` から値を読む。スカラーは `d[()]`、全体は `d[...]` か `d[:]`、一部だけならスライスで(ディスクからは必要な範囲だけ読む)。

**シグネチャ**: `d[index]`(`Dataset.__getitem__`。シグネチャなし、numpy 風インデックス)

**使用例**:
```python
with h5py.File("d.h5", "w") as f:
    d = f.create_dataset("a", data=np.arange(12).reshape(3, 4))
    print(type(d[()]), d[()].shape, d[:].shape, d[...].shape)
    print(d[1], d[1, 2], d[:, 1], d[-1, -1])
    print(d[1:, ::2])
    print(d[0:0].shape, d[10:].shape)
    try:
        d[5]
    except Exception as e:
        print(type(e).__name__, e)
    f["s"] = 3.0
    try:
        f["s"][:]
    except Exception as e:
        print(type(e).__name__, e)
    print(len(d), d.len())
    print(np.asarray(d).shape, np.sum(d))
    print([r.tolist() for r in d][:1])
    try:
        d + 1
    except Exception as e:
        print(type(e).__name__, e)
    print((d[:] + 1).sum())
```
実行結果:
```
<class 'numpy.ndarray'> (3, 4) (3, 4) (3, 4)
[4 5 6 7] 6 [1 5 9] 11
[[ 4  6]
 [ 8 10]]
(0, 4) (0, 4)
IndexError Index (5) out of range for (0-2)
ValueError Illegal slicing argument for scalar dataspace
3 3
(3, 4) 66
[[0, 1, 2, 3]]
TypeError unsupported operand type(s) for +: 'Dataset' and 'int'
78
```

**注意点・落とし穴**:
- `Dataset` は `numpy.ndarray` ではない。`d + 1` のような演算は `TypeError`(実行確認)。`d[:]` などで読み出してから計算する。ただし `np.asarray(d)` や `np.sum(d)` は内部で全体を読み込んで動く。
- 範囲外の整数インデックスは `IndexError: Index (5) out of range for (0-2)`。スライスの範囲外は numpy 同様に空配列(`(0, 4)`)。
- `for row in d` で行単位に反復できる。
- スカラーの Dataset は `d[()]` で読む(shape `()` に `d[:]` は `ValueError: Illegal slicing argument for scalar dataspace`)。大きな配列を `d[:]` で丸ごと読むとメモリに載り切らない場合があるので、`d[i:j]` で分割して処理する(「pandas/numpy連携」参照)。

---

### `d[...] = value(書き込み・ブロードキャスト)`

**用途**: スライスへの代入で値を書き込む。numpy 同様にブロードキャストされる。

**シグネチャ**: `d[index] = value`(`Dataset.__setitem__`)

**使用例**:
```python
with h5py.File("d.h5", "w") as f:
    d = f.create_dataset("a", data=np.arange(12).reshape(3, 4))
    d[0, :] = 100
    d[1:, 1] = [7, 8]
    d[...] = d[...] + 1
    print(d[:])
    try:
        d[0, :] = [1, 2]
    except Exception as e:
        print(type(e).__name__, e)
    d[2] = 5
    print(d[2])
```
実行結果:
```
[[101 101 101 101]
 [  5   8   7   8]
 [  9   9  11  12]]
TypeError Can't broadcast (2,) -> (4,)
[5 5 5 5]
```

**注意点・落とし穴**:
- 形状の合わない代入は `TypeError: Can't broadcast (2,) -> (4,)`。
- `"r"` で開いたファイルへの代入は失敗する(File の項参照)。

---

### `f.require_dataset(name, shape, dtype, exact=False) / f.create_dataset_like(name, other)`

**用途**: `require_dataset`: 既にあればそれを返し(shape・dtype が合っているか検証する)、無ければ作る。`create_dataset_like`: 既存データセットと同じ shape/dtype/チャンク設定などで新規作成する。

**シグネチャ**:
- `f.require_dataset(name, shape, dtype, exact=False, **kwds)`
- `f.create_dataset_like(name, other, **kwupdate)`

**使用例**:
```python
with h5py.File("d.h5", "w") as f:
    d = f.create_dataset("a", data=np.arange(12).reshape(3, 4))
    r = f.require_dataset("a", shape=(3, 4), dtype="i8")
    print(r == d)
    try:
        f.require_dataset("a", shape=(4, 4), dtype="i8")
    except Exception as ex:
        print(type(ex).__name__, ex)
    r = f.require_dataset("a", shape=(3, 4), dtype="i4")      # exact=False なら安全に変換できる dtype を許容
    print(r.dtype)
    try:
        f.require_dataset("a", shape=(3, 4), dtype="i4", exact=True)
    except Exception as ex:
        print(type(ex).__name__, ex)
    c = f.create_dataset_like("like", d)
    print(c.shape, c.dtype, c[:].sum())
```
実行結果:
```
True
TypeError Shapes do not match (existing (3, 4) vs new (4, 4))
int64
TypeError Datatypes do not exactly match (existing int64 vs new i4)
(3, 4) int64 0
```

**注意点・落とし穴**:
- `require_dataset` は既存があっても**中身は変更しない**。`exact=False` の場合、要求した `dtype`(`i4`)ではなく既存の dtype(`int64`)がそのまま返る。
- `create_dataset_like` はデータ値はコピーしない(`c[:].sum()` が 0)。値も欲しければ `f.copy` を使う。

---

### `d.read_direct(dest, source_sel, dest_sel) / d.write_direct(source, ...)`

**用途**: 既存の numpy 配列バッファへ直接読み込む/書き出す。一時配列を作らないので、繰り返し読み込むときにメモリ確保を減らせる。

**シグネチャ**:
- `d.read_direct(dest, source_sel=None, dest_sel=None)`
- `d.write_direct(source, source_sel=None, dest_sel=None)`

**使用例**:
```python
with h5py.File("d.h5", "w") as f:
    d = f.create_dataset("a", data=np.arange(12).reshape(3, 4))
    buf = np.zeros((3, 4), dtype=int)
    d.read_direct(buf, np.s_[0:2, :], np.s_[1:3, :])      # d[0:2,:] -> buf[1:3,:]
    print(buf)
    d.write_direct(np.full((2, 2), -5), np.s_[:, :], np.s_[0:2, 0:2])   # src[:, :] -> d[0:2, 0:2]
    print(d[:])
```
実行結果:
```
[[0 0 0 0]
 [0 1 2 3]
 [4 5 6 7]]
[[-5 -5  2  3]
 [-5 -5  6  7]
 [ 8  9 10 11]]
```

**注意点・落とし穴**:
- `read_direct` の書き込み先は C-contiguous かつ書き込み可能な配列である必要がある(Fortran 順の配列を渡すと `TypeError: Array must be C-contiguous and writable`)。選択範囲は `np.s_[...]` の形で渡す。`source_sel` はデータセット側、`dest_sel` はバッファ側の範囲。

---

### `d.astype(dtype)`

**用途**: 読み出し時に dtype を変換するビューを返す(ディスク上の dtype は変えない)。スライスで読むときに変換される。

**シグネチャ**: `d.astype(dtype)`

**使用例**:
```python
with h5py.File("d.h5", "w") as f:
    d = f.create_dataset("a", data=np.arange(6).reshape(2, 3))
    print(d.dtype, d.astype("f4")[0], d.astype("f4")[0].dtype)
```
実行結果:
```
int64 [0. 1. 2.] float32
```

**注意点・落とし穴**:
- 戻り値は `h5py._hl.dataset.AsTypeView`(`Dataset` そのものではない)。`[...]` でスライスして初めて読み込まれ、結果は変換後 dtype の `ndarray`。

---

### `h5py.Empty(dtype) / スカラー dataset`

**用途**: 値を持たない(shape 無しの)データセットや属性を表す。`shape == ()` のスカラーとは別物。

**シグネチャ**: `h5py.Empty(dtype)`

**使用例**:
```python
with h5py.File("d.h5", "w") as f:
    f.create_dataset("emp", data=h5py.Empty("f"))
    print(f["emp"].shape, f["emp"][()])
    f["sc"] = 3.14
    print(f["sc"].shape, f["sc"][()])
```
実行結果:
```
None Empty(dtype=dtype('<f4'))
() 3.14
```

**注意点・落とし穴**:
- `Empty` のデータセットは `shape` が `None`。`[()]` で読むと `Empty(dtype=dtype('<f4'))` が返る。

---

## チャンク・圧縮・リサイズ

圧縮や `resize` を使うにはデータセットが**チャンク化**されている必要がある(`compression` / `shuffle` / `maxshape` / `chunks` のいずれかを指定すると自動でチャンク化される)。

本環境で使える圧縮フィルタは `h5py.h5z.filter_avail` で確認した。

```python
import h5py
for n, i in [("gzip", h5py.h5z.FILTER_DEFLATE), ("lzf", h5py.h5z.FILTER_LZF), ("szip", h5py.h5z.FILTER_SZIP),
             ("shuffle", h5py.h5z.FILTER_SHUFFLE), ("fletcher32", h5py.h5z.FILTER_FLETCHER32), ("scaleoffset", h5py.h5z.FILTER_SCALEOFFSET)]:
    print(n, h5py.h5z.filter_avail(i))
```
実行結果:
```
gzip True
lzf True
szip True
shuffle True
fletcher32 True
scaleoffset True
```

**すべて利用可能**だったので、以下では gzip / lzf / szip をすべて実測している。`lzf` は h5py 独自、`szip` はビルドによって使えないことがあるので、ファイルを他人・他環境に渡す場合は相手側で同じフィルタが使えるか確認が必要。

### `chunks=(...)(チャンク形状)`

**用途**: データセットを固定サイズのブロック(チャンク)に分けて保存する。チャンク単位で読み書き・圧縮されるので、**アクセスパターンに合ったチャンク形状**を選ぶことが性能に直結する。

**シグネチャ**: `f.create_dataset(name, ..., chunks=True | tuple)`

**使用例**:
```python
import time
x = np.random.default_rng(0).normal(size=(4000, 4000)).astype("f4")   # 64MB

def best(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3

with h5py.File("ch.h5", "w") as f:
    f.create_dataset("contig", data=x)                          # チャンクなし(連続配置)
    f.create_dataset("row_chunk", data=x, chunks=(1, 4000))
    f.create_dataset("col_chunk", data=x, chunks=(4000, 1))
    f.create_dataset("sq_chunk", data=x, chunks=(100, 100))
    print(f.create_dataset("auto", data=x[:2000, :500], chunks=True).chunks)   # chunks=True は自動決定

with h5py.File("ch.h5", "r") as f:
    for n in ["contig", "row_chunk", "col_chunk", "sq_chunk"]:
        d = f[n]
        print(f"{n:10s} chunks={str(d.chunks):12s} 1行 {best(lambda: d[2000, :]):8.3f} ms  1列 {best(lambda: d[:, 2000]):8.3f} ms  全体 {best(lambda: d[:]):8.1f} ms")
```
実行結果:
```
(125, 63)
contig     chunks=None         1行    0.007 ms  1列    3.548 ms  全体      8.3 ms
row_chunk  chunks=(1, 4000)    1行    0.007 ms  1列   23.382 ms  全体     29.1 ms
col_chunk  chunks=(4000, 1)    1行   22.883 ms  1列    0.013 ms  全体     78.3 ms
sq_chunk   chunks=(100, 100)   1行    0.032 ms  1列    0.044 ms  全体     20.0 ms
```

**注意点・落とし穴**:
- 実測(このマシン、best of 3。値は環境依存だが傾向を見る用): 行方向チャンク `(1, 4000)` は1列読みが遅く、列方向チャンク `(4000, 1)` は1行読みが遅い。正方形に近いチャンク `(100, 100)` は行・列とも中庸。チャンクなし(連続)は行読み・全体読みが最速で、列読みは数 ms(行方向チャンクの 20〜30ms 程度よりは速いが、正方形チャンクの 0.1ms 未満よりは遅い)。同じ位置を繰り返し読んでいるので、極端に小さい値(0.005ms など)はチャンクキャッシュの影響を含む可能性がある(未検証)。
- 小さすぎるチャンクは圧縮率を悪くしオーバーヘッドも増える(下の「gzip」項に実測)。目安はチャンク1個あたり 10KB〜1MB 程度、最終的には実際のアクセスパターンでベンチマークする。
- `chunks` の各次元がデータの shape を超えると `ValueError: Chunk shape must not be greater than data shape in any dimension`。ただし `maxshape=(None,)` のように拡張可能にしておけば `shape=(2,)` でも `chunks=(4,)` を指定できる(実行確認)。スカラーには `chunks` も圧縮も指定できない(`TypeError: Scalar datasets don't support chunk/filter options`)。

---

### `compression="gzip" / "lzf" / "szip", compression_opts`

**用途**: チャンクごとに圧縮して保存する。読み出し時は自動で展開されるので、コード側は変更不要。

**シグネチャ**: `f.create_dataset(name, ..., compression="gzip" | "lzf" | "szip" | 0-9, compression_opts=..., shuffle=False)`

**使用例**:
```python
import time
rng = np.random.default_rng(0)
A = np.cumsum(rng.normal(size=(2000, 500)), axis=1).astype("f4")   # ランダムウォーク(float32)
B = rng.poisson(0.3, size=(2000, 500)).astype("i4")                 # 疎なカウントデータ(int32)

def run(x, **kw):
    tw, tr = [], []
    for _ in range(3):
        with h5py.File("z.h5", "w") as f:
            t = time.perf_counter(); f.create_dataset("x", data=x, **kw); f.flush(); tw.append(time.perf_counter() - t)
        t = time.perf_counter()
        with h5py.File("z.h5", "r") as f:
            y = f["x"][:]
        tr.append(time.perf_counter() - t)
    assert np.array_equal(x, y)
    return os.path.getsize("z.h5") / 1e6, min(tw) * 1e3, min(tr) * 1e3

cfgs = [("なし", {}), ("gzip(4)", dict(compression="gzip")), ("gzip(9)", dict(compression="gzip", compression_opts=9)),
        ("gzip(4)+shuffle", dict(compression="gzip", shuffle=True)), ("lzf", dict(compression="lzf")),
        ("lzf+shuffle", dict(compression="lzf", shuffle=True)), ("szip", dict(compression="szip"))]
for label, x in [("A float32 ランダムウォーク", A), ("B int32 疎なカウント", B)]:
    print(label, x.nbytes / 1e6, "MB")
    for n, kw in cfgs:
        mb, w, r = run(x, **kw)
        print(f"  {n:16s} {mb:6.2f} MB ({mb / (x.nbytes / 1e6) * 100:5.1f}%)  write {w:7.1f} ms  read {r:6.1f} ms")

with h5py.File("z.h5", "w") as f:
    d = f.create_dataset("x", data=B, compression="gzip", shuffle=True)
    print(d.compression, d.compression_opts, d.shuffle, d.chunks, d.filter_names)
    d2 = f.create_dataset("s", data=B, compression="szip")
    print(d2.compression, d2.compression_opts)
    print(f.create_dataset("y", data=B, compression=9).compression)       # 整数だけ指定すると gzip のレベル
    for kw in [dict(compression="gzip", compression_opts=10), dict(compression="bogus")]:
        try:
            f.create_dataset("bad", data=B, **kw)
        except Exception as e:
            print(type(e).__name__, e)
```
実行結果:
```
A float32 ランダムウォーク 4.0 MB
  なし                 4.00 MB (100.1%)  write     2.1 ms  read    1.3 ms
  gzip(4)            3.73 MB ( 93.2%)  write    97.7 ms  read   21.3 ms
  gzip(9)            3.73 MB ( 93.1%)  write   101.3 ms  read   19.7 ms
  gzip(4)+shuffle    3.24 MB ( 81.0%)  write    61.2 ms  read   12.6 ms
  lzf                4.03 MB (100.8%)  write    36.4 ms  read    3.3 ms
  lzf+shuffle        3.38 MB ( 84.6%)  write    36.0 ms  read   12.2 ms
  szip               3.27 MB ( 81.8%)  write    25.4 ms  read   37.6 ms
B int32 疎なカウント 4.0 MB
  なし                 4.00 MB (100.1%)  write     2.3 ms  read    1.7 ms
  gzip(4)            0.33 MB (  8.2%)  write    26.1 ms  read   10.2 ms
  gzip(9)            0.24 MB (  6.0%)  write  1092.0 ms  read    7.7 ms
  gzip(4)+shuffle    0.23 MB (  5.7%)  write    28.0 ms  read   14.6 ms
  lzf                0.84 MB ( 21.0%)  write    11.6 ms  read   11.5 ms
  lzf+shuffle        0.49 MB ( 12.3%)  write    11.2 ms  read   13.1 ms
  szip               0.35 MB (  8.7%)  write    16.2 ms  read   15.5 ms
gzip 4 True (125, 63) ('shuffle', 'deflate')
szip ('nn', 8)
gzip
ValueError GZIP setting must be an integer from 0-9, not 10
ValueError Compression filter "bogus" is unavailable
```

**注意点・落とし穴**:
- **圧縮の効果はデータの性質で決まる**(上の実測): ランダムウォークの float32 は gzip だけだと 93% 程度しか縮まず(lzf は 100.8% と逆に膨らんだ)、shuffle を併用して 81% 前後。疎な int32 カウントは gzip で 8% 程度、gzip+shuffle で 6% 程度まで縮んだ(lzf は 21%)。まず実データで試すこと。
- gzip はレベル 0〜9(既定は 4)。レベル 9 は圧縮率がわずかに良いが、この疎データでは書き込みが 1 秒前後(上の出力)と極端に遅くなった(gzip(4) は数十 ms)。通常はレベル 4 前後で十分。範囲外(例: 10)は `ValueError: GZIP setting must be an integer from 0-9, not 10`。
- `lzf` は h5py が同梱する独自フィルタ。ほかの HDF5 ツールで読めるかは本書では検証していない(相手環境にフィルタが無いと読めない可能性がある)。
- `szip` は本環境では使えた(`compression_opts` は `('nn', 8)` と表示)が、環境によっては利用不可。`f.create_dataset(..., compression="bogus")` のように利用できない/未知のフィルタ名は `ValueError: Compression filter "bogus" is unavailable`。
- `compression` に整数だけ渡すと gzip のレベルとして扱われる(`compression=9` → `d.compression == 'gzip'`)。
- 圧縮すると書き込み・読み出しとも遅くなる傾向がある(データ A では、無圧縮の読み出しが 1〜3ms のところ gzip は約 20ms)。なお、実行によっては B の「なし」の読み出しが 10ms 以上と出るなど測定のばらつきがあるので、数 ms の差は当てにしない方がよい。

---

### `shuffle=True / fletcher32=True / scaleoffset=n`

**用途**: 圧縮の前処理・整合性チェック・非可逆の桁落とし。`shuffle` はバイト配置を並べ替えて圧縮率を上げ、`fletcher32` はチェックサムを付ける。

**シグネチャ**: `f.create_dataset(name, ..., shuffle=False, fletcher32=False, scaleoffset=None)`

**使用例**:
```python
rng = np.random.default_rng(0)
A = np.cumsum(rng.normal(size=(2000, 500)), axis=1).astype("f4")
with h5py.File("z.h5", "w") as f:
    d = f.create_dataset("x", data=A, compression="gzip", shuffle=True, fletcher32=True, chunks=(125, 63))
    print(d.filter_names, d.shuffle, d.fletcher32)

for kw in [dict(), dict(scaleoffset=2), dict(scaleoffset=2, compression="gzip")]:
    with h5py.File("z.h5", "w") as f:
        d = f.create_dataset("x", data=A, **kw)
    with h5py.File("z.h5", "r") as f:
        y = f["x"][:]
    print(kw, "%.2f MB" % (os.path.getsize("z.h5") / 1e6), "最大誤差 %.4f" % np.abs(y - A).max())

with h5py.File("z.h5", "w") as f:
    d = f.create_dataset("s", data=A, shuffle=True)
    print(d.chunks is not None, d.filter_names)
```
実行結果:
```
('shuffle', 'deflate', 'fletcher32') True True
{} 4.00 MB 最大誤差 0.0000
{'scaleoffset': 2} 1.73 MB 最大誤差 0.0100
{'scaleoffset': 2, 'compression': 'gzip'} 1.71 MB 最大誤差 0.0100
True ('shuffle',)
```

**注意点・落とし穴**:
- `shuffle` は単独で指定しても受理される(圧縮なしでもチャンク化されフィルタが付く)が、圧縮と組み合わせて初めて意味がある。
- `scaleoffset=n` は**非可逆**の桁落とし圧縮(float では小数点以下の桁数を指定する)。実測では `scaleoffset=2` で最大誤差 0.0100(float32 データ)、サイズは 4.00MB → 1.73MB。誤差が許容できる場合(センサーデータなど)にのみ使う。
- `fletcher32` はチェックサムを付けるフィルタ(付いたことを `d.fletcher32` で確認できる)。フィルタの適用順は `d.filter_names` で確認できる(`shuffle` → `deflate` → `fletcher32`)。

---

### `maxshape=(None, ...) / d.resize(size, axis)`

**用途**: サイズ可変(追記可能)なデータセットを作り、後から拡張・縮小する。ログや時系列を逐次追記するときの基本パターン。

**シグネチャ**:
- `d.resize(size, axis=None)`
- `h5py.UNLIMITED`(= `None` と同じ意味で `maxshape` に使う)

**使用例**:
```python
with h5py.File("r.h5", "w") as f:
    d = f.create_dataset("log", shape=(0, 3), maxshape=(None, 3), dtype="f8", chunks=(4, 3))
    print(d.shape, d.maxshape, d.chunks)
    for i in range(3):
        n = d.shape[0]
        d.resize(n + 2, axis=0)            # 行方向に 2 行拡張
        d[n:] = np.full((2, 3), i)
    print(d.shape)
    print(d[:])
    d.resize((3, 3))                       # 縮小: 範囲外のデータは失われる
    d.resize(5, axis=0)                    # 再拡張: 新しい行は fillvalue(0)
    print(d[:])

    try:
        d.resize(4, axis=1)
    except Exception as e:
        print(type(e).__name__, e)
    try:
        f.create_dataset("bad", shape=(3,), maxshape=(2,), dtype="f8")
    except Exception as e:
        print(type(e).__name__, e)
    u = f.create_dataset("u", shape=(2,), maxshape=(None,), dtype="f8")
    print(u.chunks, u.maxshape)            # maxshape を指定すると自動でチャンク化
    n = f.create_dataset("nores", data=np.arange(3))
    try:
        n.resize(5, axis=0)
    except Exception as e:
        print(type(e).__name__, e)
    print(h5py.UNLIMITED)
```
実行結果:
```
(0, 3) (None, 3) (4, 3)
(6, 3)
[[0. 0. 0.]
 [0. 0. 0.]
 [1. 1. 1.]
 [1. 1. 1.]
 [2. 2. 2.]
 [2. 2. 2.]]
[[0. 0. 0.]
 [0. 0. 0.]
 [1. 1. 1.]
 [0. 0. 0.]
 [0. 0. 0.]]
RuntimeError Unable to synchronously change a dataset's dimensions (dimension cannot exceed the existing maximal size (new: 4 max: 3))
ValueError Maxdims is smaller than dims (maxdims is smaller than dims)
(2,) (None,)
TypeError Only chunked datasets can be resized
18446744073709551615
```

**注意点・落とし穴**:
- `maxshape` に `None` を指定した軸が無制限(内部では `h5py.UNLIMITED = 18446744073709551615`)。その軸だけ `resize` で増やせる(上限を超える拡張は `RuntimeError: ... dimension cannot exceed the existing maximal size`)。
- `maxshape` を指定する(あるいは圧縮する)とチャンク化され、チャンクなしのデータセットは `TypeError: Only chunked datasets can be resized`。
- 縮小してから再拡張しても、元のデータは戻らない(fillvalue で埋まる)。

---

### `d.iter_chunks(sel=None)`

**用途**: データセットをチャンク境界に沿って走査するスライスを列挙する。チャンク単位で読むと効率が良く、メモリにも載せやすい。

**シグネチャ**: `d.iter_chunks(sel=None)`

**使用例**:
```python
with h5py.File("r.h5", "w") as f:
    c = f.create_dataset("c", data=np.arange(20).reshape(4, 5), chunks=(2, 3))
    for s in c.iter_chunks():
        print(s, c[s].shape)
```
実行結果:
```
(slice(0, 2, 1), slice(0, 3, 1)) (2, 3)
(slice(0, 2, 1), slice(3, 5, 1)) (2, 2)
(slice(2, 4, 1), slice(0, 3, 1)) (2, 3)
(slice(2, 4, 1), slice(3, 5, 1)) (2, 2)
```

**注意点・落とし穴**:
- チャンクされていないデータセットでは `TypeError: Chunked dataset required`(実行確認)。端のチャンクは shape がデータの端で切られる(上の例の `(2, 2)`)。

---

### `fillvalue と疎な(未書き込みの)チャンク`

**用途**: チャンク化した巨大データセットでは、実際に書き込んだチャンクだけがディスクを消費する。未書き込み領域は `fillvalue` として読める。

**シグネチャ**: `f.create_dataset(name, ..., chunks=..., fillvalue=...)`

**使用例**:
```python
with h5py.File("sp.h5", "w") as f:
    d = f.create_dataset("big", shape=(100000, 100000), dtype="f8", chunks=(1000, 1000), fillvalue=np.nan)
    d[0:1000, 0:1000] = 1.0
    print(d.shape, d.nbytes / 1e9, "GB(論理サイズ)", d[5000, 5000], d[0, 0])
print(os.path.getsize("sp.h5") / 1e6, "MB(ファイル)")
```
実行結果:
```
(100000, 100000) 80.0 GB(論理サイズ) nan 1.0
8.004016 MB(ファイル)
```

**注意点・落とし穴**:
- 80GB(論理)のデータセットでも、書き込んだ 1000×1000 の float64 = 8MB 分だけがファイルサイズになる(実測 8.0MB)。未書き込みの領域を `d[:]` で全部読むと 80GB のメモリが必要になるので注意。

---

### `rdcc_nbytes / rdcc_nslots(チャンクキャッシュ)`

**用途**: `h5py.File` の引数で、ファイルごとのチャンクキャッシュのサイズを調整する。大きなチャンクを繰り返しランダム読みするときに効く。

**シグネチャ**: `h5py.File(name, mode, driver=..., rdcc_nbytes=..., ...)`(全引数のシグネチャは `h5py.File` の項)

**使用例**:
```python
with h5py.File("cc.h5", "w") as f:
    f["a"] = np.zeros((10, 10))
with h5py.File("cc.h5", "r", rdcc_nbytes=64 * 1024**2) as f:
    print(f.id.get_access_plist().get_cache())
with h5py.File("cc.h5", "r") as f:
    print(f.id.get_access_plist().get_cache())
```
実行結果:
```
(0, 8191, 67108864, 0.75)
(0, 8191, 8388608, 0.75)
```

**注意点・落とし穴**:
- `get_cache()` は 4 要素のタプルで、3 番目がバイト数(`rdcc_nbytes=64*1024**2` を指定すると `67108864` になることを確認)。何も指定しない場合の既定値は本環境(HDF5 2.0.0)で `8388608`(8MB)、2 番目は `8191`、4 番目は `0.75`。
- この実測ではキャッシュサイズの変更による速度差は測っていない。

---

## 属性(attrs)

グループ・データセット・ファイルに付けるメタデータ(単位、説明、パラメータなど)。`obj.attrs` は辞書風のインターフェース。

### `obj.attrs[key] = value / obj.attrs[key] / items() / in / del`

**用途**: メタデータの読み書き。スカラー・文字列・数値配列を保存でき、読むと numpy スカラー/配列(文字列は `str`)で返る。

**シグネチャ**: `obj.attrs`(`AttributeManager`。`dict` 風の `[]` / `keys` / `items` / `get` / `update` / `del` / `in` / `len`)

**使用例**:
```python
with h5py.File("at.h5", "w") as f:
    d = f.create_dataset("temp", data=np.arange(4.0))
    f.attrs["title"] = "実験1"
    d.attrs["units"] = "degC"
    d.attrs["scale"] = 0.5
    d.attrs["calib"] = np.array([1.0, 2.5])
    d.attrs["n"] = 3
    d.attrs["tags"] = ["a", "bb"]
    d.attrs["flag"] = True
    print(type(d.attrs).__name__, len(d.attrs), list(d.attrs.keys()))
    for k, v in d.attrs.items():
        print(k, repr(v), type(v).__name__)
    print(d.attrs["units"], "units" in d.attrs, d.attrs.get("nope", "default"), f.attrs["title"])
    try:
        d.attrs["nope"]
    except KeyError as e:
        print("KeyError", e)
    d.attrs["scale"] = "now string"          # 上書き(型も変えられる)
    print(repr(d.attrs["scale"]))
    d.attrs.update({"x": 1, "y": 2})
    del d.attrs["x"]
    print(sorted(d.attrs), "x" in d.attrs)
    g = f.create_group("grp"); g.attrs["k"] = 1
    print(dict(g.attrs))
    d.attrs["lst"] = [1, 2, 3]
    print(repr(d.attrs["lst"]))
    for label, v in [("dict", {"a": 1}), ("None", None)]:
        try:
            d.attrs[label] = v
        except Exception as e:
            print(label, type(e).__name__, e)

with h5py.File("at.h5", "r") as f:
    print(f["temp"].attrs["units"], type(f["temp"].attrs["units"]))
```
実行結果:
```
AttributeManager 6 ['calib', 'flag', 'n', 'scale', 'tags', 'units']
calib array([1. , 2.5]) ndarray
flag np.True_ bool
n np.int64(3) int64
scale np.float64(0.5) float64
tags array(['a', 'bb'], dtype=object) ndarray
units 'degC' str
degC True default 実験1
KeyError "Unable to synchronously open attribute (can't locate attribute: 'nope')"
'now string'
['calib', 'flag', 'n', 'scale', 'tags', 'units', 'y'] False
{'k': np.int64(1)}
array([1, 2, 3])
dict TypeError Object dtype dtype('O') has no native HDF5 equivalent
None TypeError Object dtype dtype('O') has no native HDF5 equivalent
degC <class 'str'>
```

**注意点・落とし穴**:
- スカラーは `np.int64(3)` / `np.float64(0.5)` / `np.True_` のような **numpy スカラー**で返る(Python の `int` ではない)。list は `numpy.ndarray` として返る。文字列の list は `dtype=object` の配列になる。
- `dict` や `None` は保存できない(`TypeError: Object dtype dtype('O') has no native HDF5 equivalent`)。dict はキーごとに属性へ分けるか、`json.dumps` した文字列で保存する。
- 属性は名前順で返る(挿入順にしたければ `track_order=True`。「グループと階層」参照)。

---

### `obj.attrs.create(name, data, shape, dtype) / attrs.modify(name, value) / attrs.get_id(name)`

**用途**: `create` は dtype/shape を明示して属性を作る。`modify` は既存の属性の値を更新する(既存の dtype・shape を保つ)。`get_id` は低レベルの属性オブジェクトを返し dtype などを確認できる。

**シグネチャ**:
- `obj.attrs.create(name, data, shape=None, dtype=None)`
- `obj.attrs.modify(name, value)`

**使用例**:
```python
with h5py.File("at.h5", "w") as f:
    d = f.create_dataset("temp", data=np.arange(4.0))
    d.attrs["n"] = 3
    d.attrs["units"] = "degC"
    d.attrs["calib"] = np.array([1.0, 2.5])
    d.attrs.modify("n", 10)
    print(d.attrs["n"])
    print(d.attrs.get_id("n").dtype, d.attrs.get_id("units").dtype, d.attrs.get_id("calib").shape)
    d.attrs.create("small", data=7, dtype="i1")
    print(d.attrs.get_id("small").dtype, repr(d.attrs["small"]))
    d.attrs["empty"] = h5py.Empty("f")
    print(repr(d.attrs["empty"]))
```
実行結果:
```
10
int64 object (2,)
int8 np.int8(7)
Empty(dtype=dtype('<f4'))
```

**注意点・落とし穴**:
- `attrs["n"] = 3` の dtype は `int64`(numpy 既定)になる。ディスク容量や他言語との互換で型を固定したいときに `create(..., dtype=...)` を使う。

---

### `属性サイズの上限(64KB)`

**用途**: 1 つの属性は約 64KB までしか保存できない。大きな配列は属性ではなく dataset に保存する。

**シグネチャ**: `obj.attrs[key] = ndarray`

**使用例**:
```python
with h5py.File("at.h5", "w") as f:
    for n in [8100, 8192]:               # float64 で 64800 バイト / 65536 バイト
        try:
            f.attrs[f"a{n}"] = np.zeros(n)
            print(n, "OK")
        except Exception as e:
            print(n, type(e).__name__, e)
```
実行結果:
```
8100 OK
8192 OSError Unable to synchronously create attribute (object header message is too large)
```

**注意点・落とし穴**:
- 境界は本環境で `float64` を 8100 要素(64800B)で成功、8192 要素(65536B)で失敗と確認した。エラーは `OSError: Unable to synchronously create attribute (object header message is too large)`。

---

## 複合型・文字列・可変長型

numpy の構造化 dtype、可変長文字列、ragged 配列、enum、オブジェクト参照。

### `複合型(compound dtype)/ d.fields(names)`

**用途**: 構造化 dtype(レコード配列)をそのまま保存する。列名でフィールド単位に読み書きできる。

**シグネチャ**: `d.fields(names, *, _prior_dtype=None)`

**使用例**:
```python
dt = np.dtype([("id", "i4"), ("x", "f8"), ("flag", "?")])
arr = np.array([(1, 0.5, True), (2, 1.5, False), (3, 2.5, True)], dtype=dt)
with h5py.File("ty.h5", "w") as f:
    d = f.create_dataset("rec", data=arr)
    print(d.dtype, d.shape, d.dtype.names)
    print(d[:])
    print(d["x"])                                    # 1 フィールドだけ読む
    try:
        d[["id", "x"]]                               # 複数フィールドの list 指定は不可
    except Exception as e:
        print(type(e).__name__, e)
    print(d.fields(["id", "x"])[:])                  # 複数フィールドは fields() で
    print(d.fields("x")[:], d.fields(["id"])[:])
    print(d[1], d[1]["x"], d[0:2]["id"])
    d[1] = (20, 9.9, True)                           # レコード単位の書き込み
    d["x"] = [0, 0, 0]                               # フィールド単位の書き込み
    print(d[:])
```
実行結果:
```
[('id', '<i4'), ('x', '<f8'), ('flag', '?')] (3,) ('id', 'x', 'flag')
[(1, 0.5,  True) (2, 1.5, False) (3, 2.5,  True)]
[0.5 1.5 2.5]
TypeError Indexing arrays must have integer dtypes
[(1, 0.5) (2, 1.5) (3, 2.5)]
[0.5 1.5 2.5] [(1,) (2,) (3,)]
(2, 1.5, False) 1.5 [1 2]
[( 1, 0.,  True) (20, 0.,  True) ( 3, 0.,  True)]
```

**注意点・落とし穴**:
- `d[["id", "x"]]` のような list でのフィールド指定は `TypeError: Indexing arrays must have integer dtypes` になる(fancy index と解釈される)。複数フィールドは `d.fields([...])`。
- `Dataset` の repr は複合型を `type "|V13"` のように生のバイト長で表示するが、`d.dtype` は正しい構造化 dtype。
- pandas の `DataFrame.to_records(index=False)` で作った record 配列もそのまま保存できる(「pandas/numpy連携」参照)。

---

### `h5py.string_dtype(encoding, length) / d.asstr() / h5py.check_string_dtype(dt)`

**用途**: 文字列の保存と読み出し。可変長 UTF-8 文字列は `string_dtype()`(既定)、固定長は `length=` を指定する。

**シグネチャ**:
- `h5py.string_dtype(encoding='utf-8', length=None)`
- `d.asstr(encoding=None, errors='strict')`
- `h5py.check_string_dtype(dt)`

**使用例**:
```python
print(h5py.string_dtype(), h5py.string_dtype("ascii"), h5py.string_dtype("utf-8", 10))
with h5py.File("ty.h5", "w") as f:
    s = f.create_dataset("names", data=["りんご", "banana"], dtype=h5py.string_dtype())
    print(s.dtype, h5py.check_string_dtype(s.dtype))
    print(s[:], type(s[0]))                          # 素の読み出しは bytes
    print(s.asstr()[:], s.asstr()[0], type(s.asstr()[0]))   # asstr() で str
    f["names2"] = ["a", "bc"]                        # list[str] は自動で可変長 str
    print(f["names2"].dtype, f["names2"][:])
    fx = f.create_dataset("fixed", data=np.array([b"ab", b"cde"], dtype="S3"))
    print(fx.dtype, fx[:])
    f["scalar_str"] = "hello"
    print(f["scalar_str"][()], f["scalar_str"].asstr()[()])
```
実行結果:
```
object object |S10
object string_info(encoding='utf-8', length=None)
[b'\xe3\x82\x8a\xe3\x82\x93\xe3\x81\x94' b'banana'] <class 'bytes'>
['りんご' 'banana'] りんご <class 'str'>
object [b'a' b'bc']
|S3 [b'ab' b'cde']
b'hello' hello
```

**注意点・落とし穴**:
- **h5py 3.x では、文字列 dataset を読むと `bytes` が返る**(`str` ではない)。`str` が欲しければ `d.asstr()[...]`(または `d.asstr(encoding, errors)`)を通す。属性(`attrs`)は `str` で返る。
- `dtype` は `object` として表示される(`h5py.check_string_dtype(dt)` が `string_info(encoding='utf-8', length=None)` を返す)。`string_dtype("utf-8", 10)` は固定長 `|S10`。
- numpy の `<U` 配列は保存できない(`TypeError: No conversion path for dtype: dtype('<U1')`。`.astype(object)` してから `dtype=h5py.string_dtype()` を指定するか、list で渡す)。

---

### `h5py.vlen_dtype(basetype)(可変長配列)`

**用途**: 要素ごとに長さの異なる配列(ragged array)を 1 つの dataset に保存する。

**シグネチャ**:
- `h5py.vlen_dtype(basetype)`
- `h5py.check_vlen_dtype(dt)`

**使用例**:
```python
vt = h5py.vlen_dtype(np.dtype("i4"))
print(vt, h5py.check_vlen_dtype(vt))
with h5py.File("ty.h5", "w") as f:
    v = f.create_dataset("ragged", shape=(3,), dtype=vt)
    v[0] = [1, 2, 3]
    v[1] = [4]
    v[2] = np.arange(5, dtype="i4")
    print(v[:])
    print(v[1], v[2].dtype)
```
実行結果:
```
object int32
[array([1, 2, 3], dtype=int32) array([4], dtype=int32)
 array([0, 1, 2, 3, 4], dtype=int32)]
[4] int32
```

**注意点・落とし穴**:
- 読み出すと `dtype=object` の numpy 配列で、要素が各 `ndarray`。上の例のように `v[i] = ...` と要素単位で代入した。

---

### `h5py.enum_dtype(values_dict, basetype) / h5py.check_enum_dtype(dt)`

**用途**: 名前付き整数(列挙型)を保存する。数値ラベルとの対応がファイルに残る。

**シグネチャ**:
- `h5py.enum_dtype(values_dict, basetype=<class 'numpy.uint8'>)`
- `h5py.check_enum_dtype(dt)`

**使用例**:
```python
et = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 2}, basetype="i1")
with h5py.File("ty.h5", "w") as f:
    e = f.create_dataset("color", data=np.array([0, 2, 1], dtype="i1"), dtype=et)
    print(e.dtype, h5py.check_enum_dtype(e.dtype), e[:])
```
実行結果:
```
int8 {'BLUE': 2, 'GREEN': 1, 'RED': 0} [0 2 1]
```

**注意点・落とし穴**:
- 読み出し値は基底型の整数(`int8`)で、名前への変換は `check_enum_dtype` で得た辞書を自分で引く必要がある。

---

### `h5py.ref_dtype / obj.ref(オブジェクト参照)`

**用途**: 別のグループ/データセットへの参照をデータとして保存する。参照から元オブジェクトを取り出せる。

**シグネチャ**:
- `h5py.ref_dtype`(`Reference` 用 dtype)
- `obj.ref`

**使用例**:
```python
with h5py.File("ty.h5", "w") as f:
    f["target"] = np.arange(3)
    r = f.create_dataset("refs", shape=(1,), dtype=h5py.ref_dtype)
    r[0] = f["target"].ref
    print(r[0])
    print(f[r[0]].name, f[r[0]][:])
    f.attrs["myref"] = f["target"].ref               # 属性にも保存できる
    print(f[f.attrs["myref"]].name)
    print(h5py.check_dtype(ref=h5py.ref_dtype))
```
実行結果:
```
<HDF5 object reference>
/target [0 1 2]
/target
<class 'h5py.h5r.Reference'>
```

**注意点・落とし穴**:
- 参照は `f[ref]` で開く。属性にも参照を保存できる(上の例)。

---

## 高度なインデックス・仮想データセット・並行アクセス

`Dataset` のインデックスは numpy と似ているが**制限がある**(fancy indexing)。複数ファイルをつなぐ仮想データセット、他プロセスが読みながら書く SWMR、multiprocessing との相性もここにまとめる。

### `fancy indexing の制限(list / bool マスク)`

**用途**: `d[[0, 2, 4]]` のように整数リストで行を選ぶ。numpy より制限が多い。

**シグネチャ**: `d[[i, j, ...]]` / `d[bool_mask]`

**使用例**:
```python
a = np.arange(30).reshape(5, 6)
with h5py.File("fi.h5", "w") as f:
    d = f.create_dataset("a", data=a)
    print(d[[0, 2, 4]].tolist())
    print(d[:, [1, 3]].tolist())
    for label, fn in [("降順", lambda: d[[3, 1]]),
                      ("重複", lambda: d[[1, 1]]),
                      ("2軸に list", lambda: d[[0, 1], [1, 2]]),
                      ("負の step", lambda: d[::-1]),
                      ("None(newaxis)", lambda: d[np.newaxis]),
                      ("bool マスク(1軸)", lambda: d[np.array([True, False, True, False, True])].shape),
                      ("bool マスク(2D)", lambda: d[a > 10].shape)]:
        try:
            print(label, "->", fn())
        except Exception as e:
            print(label, "->", type(e).__name__, e)

    # 回避策: 昇順・重複なしにして読み、numpy 側で並べ替える
    idx = np.array([3, 1, 1])
    u, inv = np.unique(idx, return_inverse=True)
    print(d[u][inv].tolist())
```
実行結果:
```
[[0, 1, 2, 3, 4, 5], [12, 13, 14, 15, 16, 17], [24, 25, 26, 27, 28, 29]]
[[1, 3], [7, 9], [13, 15], [19, 21], [25, 27]]
降順 -> TypeError Indexing elements must be in increasing order
重複 -> TypeError Indexing elements must be in increasing order
2軸に list -> TypeError Only one indexing vector or array is currently allowed for fancy indexing
負の step -> ValueError Step must be >= 1 (got -1)
None(newaxis) -> TypeError Indexing with None (or np.newaxis) is not supported
bool マスク(1軸) -> (3, 6)
bool マスク(2D) -> (19,)
[[18, 19, 20, 21, 22, 23], [6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11]]
```

**注意点・落とし穴**:
- リストのインデックスは**昇順・重複なし**、かつ**1 軸だけ**に使える(それぞれ `TypeError: Indexing elements must be in increasing order` / `Only one indexing vector or array is currently allowed for fancy indexing`)。負の step は `ValueError: Step must be >= 1`、`np.newaxis`(None)も `TypeError: Indexing with None (or np.newaxis) is not supported`。
- 順序が任意・重複ありのインデックスがほしいときは、`np.unique(idx, return_inverse=True)` で昇順の一意な index を作って読み、numpy 側で `[inv]` として並べ替える(上の例)。大量の離れた点を fancy indexing で読むのは遅くなりやすい(本書では速度は未計測)。
- bool マスクは 1 軸のマスク(`(5,)` の `np.array([...])`)にも、データと同じ形の 2D マスクにも使える(上の例)。

---

### `h5py.MultiBlockSlice(start, stride, count, block)`

**用途**: ブロック単位で飛び飛びに選択する(HDF5 の hyperslab)。numpy のスライスでは書けない「2 個読んで 1 個飛ばす」型の選択が 1 回の I/O でできる。

**シグネチャ**: `h5py.MultiBlockSlice(start=0, stride=1, count=None, block=1)`

**使用例**:
```python
with h5py.File("fi.h5", "w") as f:
    d = f.create_dataset("v", data=np.arange(12))
    mb = h5py.MultiBlockSlice(start=0, stride=3, count=2, block=2)
    print(d[mb])
```
実行結果:
```
[0 1 3 4]
```

**注意点・落とし穴**:
- `start=0, stride=3, count=2, block=2` は「0 から 2 要素、3 進んで 2 要素」= 要素 `[0, 1, 3, 4]`(実行結果)。

---

### `h5py.VirtualLayout / h5py.VirtualSource / f.create_virtual_dataset(name, layout, fillvalue)`

**用途**: 複数の HDF5 ファイルにまたがるデータを、コピーせずに 1 つの dataset に見せる(仮想データセット)。

**シグネチャ**:
- `h5py.VirtualLayout(shape, dtype, maxshape=None, filename=None)`
- `h5py.VirtualSource(path_or_dataset, name=None, shape=None, dtype=None, maxshape=None)`
- `f.create_virtual_dataset(name, layout, fillvalue=None)`

**使用例**:
```python
for i in range(3):
    with h5py.File(f"part{i}.h5", "w") as f:
        f["data"] = np.full(4, i, dtype="i4") + np.arange(4)

layout = h5py.VirtualLayout(shape=(3, 4), dtype="i4")
for i in range(3):
    layout[i] = h5py.VirtualSource(f"part{i}.h5", "data", shape=(4,))

with h5py.File("vds.h5", "w") as f:
    v = f.create_virtual_dataset("all", layout, fillvalue=-1)
    print(v.is_virtual, v.shape, v.dtype)
    print(v[:])
    print([(s.file_name, s.dset_name) for s in v.virtual_sources()])

os.rename("part1.h5", "part1_moved.h5")               # 元ファイルが見つからないと…
with h5py.File("vds.h5", "r") as f:
    print(f["all"][:])                               # …その部分は fillvalue になる(エラーにならない)
```
実行結果:
```
True (3, 4) int32
[[0 1 2 3]
 [1 2 3 4]
 [2 3 4 5]]
[('part0.h5', 'data'), ('part1.h5', 'data'), ('part2.h5', 'data')]
[[ 0  1  2  3]
 [-1 -1 -1 -1]
 [ 2  3  4  5]]
```

**注意点・落とし穴**:
- 元ファイルが見つからない・読めない場合、**例外にならず `fillvalue`(未指定なら 0)で埋められる**(実行確認: 行 1 が `-1`)。データ欠損に気付きにくいので、`fillvalue` に NaN や -1 など判別しやすい値を選ぶ。
- ここでは元ファイルをカレントディレクトリに置いたケースのみ確認しており、別ディレクトリからの解決規則は検証していない。

---

### `d.dims / make_scale / attach_scale(次元スケール)`

**用途**: 座標軸のような「次元に付随する別データセット」を関連付ける。netCDF(xarray の `h5netcdf`)はこの仕組みの上に作られている。

**シグネチャ**: `d.make_scale(name='')`

**使用例**:
```python
with h5py.File("ds.h5", "w") as f:
    d = f.create_dataset("a", data=np.arange(30.0).reshape(5, 6))
    t = f.create_dataset("time", data=np.arange(5.0))
    t.make_scale("time")
    d.dims[0].attach_scale(t)
    d.dims[0].label = "t"
    print(d.dims[0].label, list(d.dims[0].keys()), d.dims[0]["time"][:], t.is_scale)
```
実行結果:
```
t ['time'] [0. 1. 2. 3. 4.] True
```

**注意点・落とし穴**:
- スケールは `d.dims[i]` 経由で参照する。座標ラベル付きのデータ操作を行うなら xarray を使う(「pandas/numpy連携」の netCDF 項も参照)。

---

### `external=[(filename, offset, size)](外部バイナリへの保存)`

**用途**: データセットの本体を HDF5 の外部の生バイナリファイルに置き、HDF5 側はメタデータだけを持つ。

**シグネチャ**: `f.create_dataset(name, ..., external=[(filename, offset, size), ...])`

**使用例**:
```python
with h5py.File("ext.h5", "w") as f:
    e = f.create_dataset("ext", shape=(4,), dtype="i4", external=[("ext.bin", 0, 16)])
    e[:] = [1, 2, 3, 4]
    print(e.external)
print(open("ext.bin", "rb").read(16) == np.array([1, 2, 3, 4], dtype="i4").tobytes())
```
実行結果:
```
[('ext.bin', 0, 16)]
True
```

**注意点・落とし穴**:
- 外部ファイルに書かれるのは生のバイト列(`np.array([1, 2, 3, 4], dtype='i4').tobytes()` と一致することを実行確認)。

---

### `SWMR(単一書き手・複数読み手): libver="latest" / f.swmr_mode / h5py.File(..., swmr=True) / d.refresh()`

**用途**: 1 プロセスが書き込み中のファイルを、別プロセスが読み続けられるようにする(ログ/計測の追記など)。

**シグネチャ**:
- `h5py.File(name, 'r', swmr=True)`(読み手)
- `f.swmr_mode = True`(書き手)
- `d.refresh()`

**使用例**:
```python
import subprocess, sys
reader = ("import h5py\n"
          "with h5py.File('sw.h5', 'r', swmr=True) as f:\n"
          "    d = f['x']; print('reader:', d.shape, d[:].tolist())")
with h5py.File("sw.h5", "w", libver="latest") as f:
    d = f.create_dataset("x", shape=(0,), maxshape=(None,), dtype="i4", chunks=(4,))
    f.swmr_mode = True                                  # dataset を作り終えてから SWMR モードへ
    print(f.swmr_mode)
    for i in range(3):
        d.resize(d.shape[0] + 2, axis=0)
        d[-2:] = i
        d.flush()                                       # 読み手から見えるようにする
        print(subprocess.run([sys.executable, "-c", reader], capture_output=True, text=True).stdout.strip())

try:
    with h5py.File("sw2.h5", "w") as f:                # libver="latest" なし
        f.swmr_mode = True
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
True
reader: (2,) [0, 0]
reader: (4,) [0, 0, 1, 1]
reader: (6,) [0, 0, 1, 1, 2, 2]
RuntimeError Unable to start swmr writing (file superblock version - should be at least 3)
```

**注意点・落とし穴**:
- 書き手は **`libver="latest"` で作成**し、dataset は事前に作って `f.swmr_mode = True` にする。`libver="latest"` なしで SWMR を有効にしようとすると `RuntimeError: Unable to start swmr writing (file superblock version - should be at least 3)`。
- 書き手は変更のたびに `d.flush()` する。読み手が同じ `Dataset` オブジェクトを保持し続けて追記を確認したいときは `d.refresh()`(または開き直し)。同一プロセス内の別ハンドルで書き手・読み手を動かして `refresh()` 後に新しい shape が見えることも確認した。
- SWMR 中に新しい group/dataset を作る挙動は本書では検証していない(上の例は dataset を事前に作ってから SWMR モードにしている)。

---

### `並列・並行アクセス: pickle 不可 / multiprocessing / ファイルロック(locking)`

**用途**: 複数プロセスで読むときの作法と、他プロセスが書き込み中のファイルを開いたときのロックの挙動。

**シグネチャ**: `h5py.File(name, mode, locking=None, ...)`

**使用例**:
```python
import pickle, subprocess, sys
import multiprocessing as mp

def work(i):
    with h5py.File("mp.h5", "r") as f:                 # ワーカー内で開く
        return int(f["x"][i])

if __name__ == "__main__":
    with h5py.File("mp.h5", "w") as f:
        f["x"] = np.arange(10)
    f = h5py.File("mp.h5", "r")
    for obj in (f, f["x"]):
        try:
            pickle.dumps(obj)
        except Exception as e:
            print(type(e).__name__, e)
    f.close()

    with mp.get_context("spawn").Pool(2) as p:
        print(p.map(work, range(4)))

    w = h5py.File("mp.h5", "r+")                        # 書き込み用に開いたまま
    child = "import h5py\ntry:\n    h5py.File('mp.h5', 'r')\nexcept Exception as e:\n    print(type(e).__name__, str(e)[:80])\nelse:\n    print('opened')"
    print(subprocess.run([sys.executable, "-c", child], capture_output=True, text=True).stdout.strip())
    child2 = "import h5py\nwith h5py.File('mp.h5', 'r', locking=False) as f: print('locking=False:', f['x'][:3])"
    print(subprocess.run([sys.executable, "-c", child2], capture_output=True, text=True).stdout.strip())
    w.close()
```
実行結果:
```
TypeError h5py objects cannot be pickled
TypeError h5py objects cannot be pickled
[0, 1, 2, 3]
BlockingIOError [Errno 11] Unable to synchronously open file (unable to lock file, errno = 11, e
locking=False: [0 1 2]
```

**注意点・落とし穴**:
- `File` / `Dataset` は **pickle できない**(`TypeError: h5py objects cannot be pickled`)ので、`multiprocessing` にそのまま渡せない。**ファイル名を渡して、ワーカー内で開く**。上の例は `spawn` 方式で 4 要素を正しく並列に読めた。
- 書き込み用(`r+`)に開いている別プロセスのファイルを開こうとすると `BlockingIOError: [Errno 11] Unable to synchronously open file (unable to lock file, ...)`。`locking=False`(または環境変数 `HDF5_USE_FILE_LOCKING=FALSE`)でロックを無効にすれば開けるが、書き手が更新中のファイルを読んだときの整合性は本書では検証していない(必要なら SWMR を使う)。
- **同一プロセス内**では、`r+` で開いたファイルに `r` で開き直してもロックエラーにならなかった(別プロセスと挙動が違う)。
- `fork` 方式で開いたファイルハンドルを子プロセスへ引き継ぐ使い方は本書では検証していない。子プロセスで開き直す方針が無難。

---

## pandas/numpy連携

NumPy 配列との往復はほぼ自然に行える。pandas の `to_hdf` は PyTables(`tables`)が必要で、本環境には未導入のため、h5py だけで DataFrame を保存する方法を示す。

### `numpy との連携: np.asarray(d) / 分割読み込みで巨大データを処理`

**用途**: `Dataset` は numpy 関数に渡せるが全体を読み込む。メモリに載らないサイズはチャンク/ブロック単位で集計する。

**シグネチャ**: `np.asarray(d)` / `d.iter_chunks()` / `d[i:j]`

**使用例**:
```python
big = np.random.default_rng(0).normal(size=(100000, 10))
with h5py.File("p.h5", "w") as f:
    d = f.create_dataset("big", data=big, chunks=(10000, 10))

    s = np.zeros(10)                                   # チャンク単位の集計
    for sl in d.iter_chunks():
        s += d[sl].sum(axis=0)
    print(np.allclose(s / len(d), big.mean(axis=0)))

    n, s2 = 0, np.zeros(10)                            # 任意のブロック単位の集計
    for i in range(0, len(d), 25000):
        blk = d[i:i + 25000]
        s2 += blk.sum(axis=0)
        n += len(blk)
    print(np.allclose(s2 / n, big.mean(axis=0)))

    print(type(np.asarray(d)).__name__, np.asarray(d).shape)
```
実行結果:
```
True
True
ndarray (100000, 10)
```

**注意点・落とし穴**:
- `np.asarray(d)` / `np.mean(d)` は動くが内部で全体を読むので、メモリに載らないデータには向かない。dask が使える環境なら `dask.array.from_array(d, chunks=d.chunks)` という手もあるが、本環境には dask がなく未検証。

---

### `pandas: DataFrame を h5py で保存 / DataFrame.to_records → 複合型`

**用途**: `DataFrame.to_hdf`(PyTables)が使えない環境で、DataFrame を HDF5 に保存・復元する。

**シグネチャ**: `pd.DataFrame.to_hdf(path_or_buf, key, ...)`(要 `tables`。本環境は未導入)

**使用例**:
```python
import pandas as pd
df = pd.DataFrame({"id": [1, 2, 3], "x": [0.5, 1.5, 2.5], "name": ["a", "bb", "ccc"]})
try:
    df.to_hdf("p.h5", key="df")
except Exception as e:
    print(type(e).__name__, e)

# (1) 列ごとに dataset として保存
with h5py.File("p.h5", "w") as f:
    g = f.create_group("df")
    for c in df.columns:
        col = df[c]
        if col.dtype == object or str(col.dtype).startswith("str"):
            g.create_dataset(c, data=col.to_numpy(dtype=object), dtype=h5py.string_dtype())
        else:
            g.create_dataset(c, data=col.to_numpy())
    g.attrs["columns"] = list(df.columns)
with h5py.File("p.h5", "r") as f:
    g = f["df"]
    back = pd.DataFrame({c: (g[c].asstr()[:] if h5py.check_string_dtype(g[c].dtype) else g[c][:])
                         for c in g.attrs["columns"]})
print(back)
print(back.equals(df))

# (2) 数値だけなら to_records で複合型 dataset 1 個に
rec = df[["id", "x"]].to_records(index=False)
print(rec.dtype)
with h5py.File("p.h5", "w") as f:
    f["rec"] = rec
    print(pd.DataFrame(f["rec"][:]))
```
実行結果:
```
ImportError `Import pytables` failed.  Use pip or conda to install the pytables package.
   id    x name
0   1  0.5    a
1   2  1.5   bb
2   3  2.5  ccc
True
(numpy.record, [('id', '<i8'), ('x', '<f8')])
   id    x
0   1  0.5
1   2  1.5
2   3  2.5
```

**注意点・落とし穴**:
- `df.to_hdf()` は PyTables 未導入だと `ImportError: `Import pytables` failed.  Use pip or conda to install the pytables package.`。`pd.read_hdf` も同様。本環境には PyTables が無いので `to_hdf` の出力形式そのものは検証していない。
- 文字列列は h5py では `bytes` で返るので `d.asstr()[:]`(`str`)を通す。復元結果は pandas 3.0.5 の既定の文字列型(`StringDtype`)で、`back.equals(df)` が `True` になった。

---

### `xarray(h5netcdf)で書いた netCDF を h5py で覗く`

**用途**: netCDF4 形式は HDF5 なので、`xarray.to_netcdf(engine="h5netcdf")` で書いたファイルは h5py でそのまま読める。構造の確認やデバッグに便利。

**シグネチャ**: `h5py.File(name, mode='r', ...)`

**使用例**:
```python
import xarray as xr
ds = xr.Dataset({"t": (("x",), np.arange(3.0), {"units": "K"})}, coords={"x": [10, 20, 30]})
ds.to_netcdf("nc.nc", engine="h5netcdf")
with h5py.File("nc.nc", "r") as f:
    print(list(f), f["t"][:], f["x"][:])
    print(f["t"].attrs["units"], f["x"].is_scale, f["t"].dims[0].label)
    print(list(f["t"].attrs.keys()))
    print(dict(f.attrs))
```
実行結果:
```
['t', 'x'] [0. 1. 2.] [10 20 30]
K True 
['_Netcdf4Coordinates', '_Netcdf4Dimid', '_FillValue', 'units', 'DIMENSION_LIST']
{'_NCProperties': np.bytes_(b'version=2,h5netcdf=1.8.1,hdf5=2.0.0,h5py=3.16.0')}
```

**注意点・落とし穴**:
- 座標 `x` は次元スケール(`is_scale`)として保存され、データ変数 `t` は `DIMENSION_LIST` 属性(オブジェクト参照)で `x` に紐づく。ファイルの属性 `_NCProperties` に、書き込みに使った h5netcdf / hdf5 / h5py のバージョンが残る。

---

## safetensors(姉妹フォーマット)

safetensors 0.8.0 は、ML の重み(テンソルの辞書)を保存する「配列オンディスク」形式。h5py(HDF5)と同じく数値配列をファイルに置くが、設計の目的が違う。

- **pickle を使わない**設計(安全に配布するための形式。`torch.save` / `torch.load` が pickle ベースなのと対照的。この設計上の特徴自体は公式の説明に基づくもので、本書では悪意あるファイルでの検証はしていない)。
- ファイルは「8 バイトのヘッダ長 + JSON ヘッダ + 生データ」だけの単純な構造。**フラットな `{名前: テンソル}`** で、本書で扱った API には HDF5 のような階層・属性・圧縮・追記の機能は無い(文字列のメタデータ `metadata` は付けられる)。
- `safe_open` + `get_slice` で必要な部分だけ読める(実測は末尾の比較の項)。

各コード例は独立して実行できるようにしてあり、冒頭で以下を import 済みとします(このカテゴリのみ)。

```python
import numpy as np
import torch                                   # 2.13.0+cpu
from safetensors import safe_open
from safetensors.numpy import save_file, load_file
import safetensors.torch as stt
```

`safetensors.__version__` は `0.8.0`、`safetensors` 直下は `SafetensorError` / `TensorSpec` / `deserialize` / `numpy` / `safe_open` / `serialize` / `serialize_file` / `torch` を公開している(`dir` で確認)。

### `safetensors.numpy.save_file / load_file`

**用途**: numpy 配列の辞書を `.safetensors` に保存・読み込みする。

**シグネチャ**:
- `safetensors.numpy.save_file(tensor_dict: Dict[str, numpy.ndarray], filename: Union[str, os.PathLike], metadata: Optional[Dict[str, str]] = None) -> None`
- `safetensors.numpy.load_file(filename: Union[str, os.PathLike], *, backend: str = 'mmap') -> Dict[str, numpy.ndarray]`

**使用例**:
```python
t = {"w": np.arange(6, dtype="f4").reshape(2, 3), "b": np.zeros(3, dtype="i8")}
save_file(t, "m.safetensors", metadata={"format": "np", "note": "日本語OK"})
r = load_file("m.safetensors")
print({k: (v.shape, v.dtype) for k, v in r.items()})
print(np.array_equal(r["w"], t["w"]), type(r["w"]).__name__)
```
実行結果:
```
{'b': ((3,), dtype('int64')), 'w': ((2, 3), dtype('float32'))}
True ndarray
```

**注意点・落とし穴**:
- キーは `str` のみ(`int` キーは `TypeError: argument 'tensor_dict': 'int' object is not an instance of 'str'`)。`metadata` の値も `str` のみ(`{"n": 1}` は `TypeError`)。
- ファイルに保存されるテンソルの順序は名前のソート順(`b`, `w`)。`object` dtype は保存できない(`SafetensorError: Unknown dtype "object". Supported dtypes: bool, int8, uint8, ...`)。

---

### `safe_open(filename, framework, device) / get_tensor / get_slice / keys / metadata`

**用途**: ファイルを**遅延的に**開き、必要なテンソル(またはその一部)だけ読む。巨大モデルから一部の重みだけ取るときに使う。

**シグネチャ**: `safe_open(filename, framework, device=Ellipsis, *, backend=Ellipsis)`

**使用例**:
```python
save_file({"w": np.arange(6, dtype="f4").reshape(2, 3), "b": np.zeros(3, dtype="i8")},
          "m.safetensors", metadata={"format": "np", "note": "日本語OK"})
with safe_open("m.safetensors", framework="np") as f:
    print(list(f.keys()), f.metadata())
    print(f.get_tensor("w"))
    sl = f.get_slice("w")
    print(type(sl).__name__, sl.get_shape(), sl.get_dtype())
    print(sl[1:, :2], sl[:, 1])                      # 一部だけ読む
    print(list(f.get_tensors().keys()))
    try:
        f.get_tensor("nope")
    except Exception as e:
        print(type(e).__name__, e)
```
実行結果:
```
['b', 'w'] {'note': '日本語OK', 'format': 'np'}
[[0. 1. 2.]
 [3. 4. 5.]]
PySafeSlice [2, 3] F32
[[3. 4.]] [1. 4.]
['b', 'w']
SafetensorError File does not contain tensor nope
```

**注意点・落とし穴**:
- `framework` は `"np"` / `"pt"`(docstring には `pt`, `tf`, `flax`, `numpy` とある)。`device` の既定は `"cpu"`(docstring による。`inspect.signature` は既定値を `Ellipsis` と表示する)。
- `backend` は `"mmap"`(既定)か `"pread"`。`"read"` などは `SafetensorError: backend "read" is invalid (expected one of: "mmap", "pread")`(`load_file` で確認)。
- 存在しない名前は `SafetensorError: File does not contain tensor nope`。

---

### `safetensors.torch.save_file / load_file(bfloat16 と torch)`

**用途**: PyTorch テンソルの辞書の保存・読み込み。`bfloat16` など numpy に無い dtype も扱える。

**シグネチャ**:
- `safetensors.torch.save_file(tensors: Dict[str, torch.Tensor], filename: Union[str, os.PathLike], metadata: Optional[Dict[str, str]] = None)`
- `safetensors.torch.load_file(filename: Union[str, os.PathLike], device: Union[str, int] = 'cpu', *, backend: str = 'mmap') -> Dict[str, torch.Tensor]`

**使用例**:
```python
tt = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3), "bf": torch.ones(2, dtype=torch.bfloat16)}
stt.save_file(tt, "t.safetensors")
r = stt.load_file("t.safetensors")
print({k: (tuple(v.shape), v.dtype) for k, v in r.items()}, r["w"].device)
with safe_open("t.safetensors", framework="pt", device="cpu") as f:
    print(f.get_tensor("bf").dtype, f.get_slice("w")[0])

# 同じファイルを numpy 側で読む
with safe_open("t.safetensors", framework="np") as f:
    print(f.get_tensor("w").dtype)                  # float32 は numpy でも読める
    try:
        f.get_tensor("bf")
    except Exception as e:
        print(type(e).__name__, e)
try:
    load_file("t.safetensors")
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
{'w': ((2, 3), torch.float32), 'bf': ((2,), torch.bfloat16)} cpu
torch.bfloat16 tensor([0., 1., 2.])
float32
TypeError data type 'bfloat16' not understood
TypeError data type 'bfloat16' not understood
```

**注意点・落とし穴**:
- torch で保存したファイルは、numpy に存在する dtype(ここでは `float32`)なら numpy 側でも読める(フレームワーク間で共有できる)。`bfloat16` は numpy 側では `TypeError: data type 'bfloat16' not understood`。
- `load_file` の `device` 引数(既定 `'cpu'`)で GPU などへ直接読み込める(本環境は CPU 版のみで、GPU への読み込みは未検証)。

---

### `safetensors.torch.save_model / load_model / state_dict の保存`

**用途**: `torch.nn.Module` の重みを safetensors で保存・復元する。`save_model` は共有(tied)重みを考慮してくれる。

**シグネチャ**:
- `safetensors.torch.save_model(model: torch.nn.modules.module.Module, filename: str, metadata: Optional[Dict[str, str]] = None, force_contiguous: bool = True)`
- `safetensors.torch.load_model(model: torch.nn.modules.module.Module, filename: Union[str, os.PathLike], strict: bool = True, device: Union[str, int] = 'cpu', *, backend: str = 'mmap') -> Tuple[List[str], List[str]]`

**使用例**:
```python
m = torch.nn.Linear(3, 2)
stt.save_model(m, "lin.safetensors")
m2 = torch.nn.Linear(3, 2)
print(stt.load_model(m2, "lin.safetensors"), torch.equal(m.weight, m2.weight))
print(list(stt.load_file("lin.safetensors")))

# state_dict を直接保存・読み込みする方法
stt.save_file(m.state_dict(), "sd.safetensors")
m3 = torch.nn.Linear(3, 2)
print(m3.load_state_dict(stt.load_file("sd.safetensors")))
```
実行結果:
```
(set(), []) True
['bias', 'weight']
<All keys matched successfully>
```

**注意点・落とし穴**:
- `load_model` の戻り値は 2 要素のタプル(型注釈は `Tuple[List[str], List[str]]`。ここでは `(set(), [])`)。
- `stt.save_file(m.state_dict(), ...)` は、重みが共有(同じメモリ)されている場合にエラーになる(下の「制約・落とし穴」参照)。共有重みを持つモデルは `save_model` を使う。

---

### `safetensors.numpy.save / load(バイト列との相互変換)`

**用途**: ファイルを介さず `bytes` で受け渡す(ネットワーク送信など)。

**シグネチャ**:
- `safetensors.numpy.save(tensor_dict: Dict[str, numpy.ndarray], metadata: Optional[Dict[str, str]] = None) -> bytes`
- `safetensors.numpy.load(data: bytes) -> Dict[str, numpy.ndarray]`

**使用例**:
```python
t = {"w": np.arange(6, dtype="f4").reshape(2, 3)}
from safetensors.numpy import save, load
buf = save(t)
print(type(buf).__name__, len(buf), load(buf)["w"].shape)
```
実行結果:
```
bytes 96 (2, 3)
```

**注意点・落とし穴**:
- `safetensors.torch` にも同名の `save` / `load` がある。

---

### `ファイル構造(ヘッダ JSON)を直接読む`

**用途**: `.safetensors` の中身が単純な形式であることの確認・自前ローダの参考に。先頭 8 バイトが JSON ヘッダのバイト長(リトルエンディアン uint64)。

**シグネチャ**: `struct.unpack("<Q", data[:8])`

**使用例**:
```python
import json, struct
save_file({"w": np.arange(6, dtype="f4").reshape(2, 3), "b": np.zeros(3, dtype="i8")},
          "m.safetensors", metadata={"format": "np"})
b = open("m.safetensors", "rb").read()
n = struct.unpack("<Q", b[:8])[0]
print(n, json.loads(b[8:8 + n]))
print(len(b), 8 + n + 48)
```
実行結果:
```
144 {'__metadata__': {'format': 'np'}, 'b': {'dtype': 'I64', 'shape': [3], 'data_offsets': [0, 24]}, 'w': {'dtype': 'F32', 'shape': [2, 3], 'data_offsets': [24, 48]}}
200 200
```

**注意点・落とし穴**:
- `data_offsets` はヘッダの直後(8 + n バイト目)からの相対位置。ファイル全体のバイト数は `8 + ヘッダ長 + データ長`(最終行で一致を確認)。

---

### `制約・落とし穴(非連続配列・共有テンソル)`

**用途**: safetensors に保存できないもの/黙って壊れるものの整理。

**シグネチャ**: `safetensors.numpy.save_file` / `safetensors.torch.save_file`

**使用例**:
```python
a = np.arange(6.).reshape(2, 3).T                  # 転置 = 非 C-contiguous
print(a.flags.c_contiguous)
print(a)
save_file({"a": a}, "nc.safetensors")
print(load_file("nc.safetensors")["a"])            # 値が転置前の並びのまま(!)
save_file({"a": np.ascontiguousarray(a)}, "nc.safetensors")
print(np.array_equal(load_file("nc.safetensors")["a"], a))

f_arr = np.asfortranarray(np.arange(6.).reshape(2, 3))
save_file({"a": f_arr}, "nc.safetensors")
print(np.array_equal(load_file("nc.safetensors")["a"], f_arr))
s = np.arange(10.)[::2]                            # 飛び飛びのスライス
save_file({"a": s}, "nc.safetensors")
print(load_file("nc.safetensors")["a"], s)

tt = torch.arange(6.).reshape(2, 3).T
try:
    stt.save_file({"a": tt}, "e.safetensors")
except Exception as e:
    print(type(e).__name__, str(e)[:70])
stt.save_file({"a": tt.contiguous()}, "e.safetensors")
print(torch.equal(stt.load_file("e.safetensors")["a"], tt))

x = torch.zeros(3)
try:
    stt.save_file({"a": x, "b": x}, "e.safetensors")     # 同じメモリを共有
except Exception as e:
    print(type(e).__name__, " ".join(str(e).split())[:100])
```
実行結果:
```
False
[[0. 3.]
 [1. 4.]
 [2. 5.]]
[[0. 1.]
 [2. 3.]
 [4. 5.]]
True
False
[0. 1. 2. 3. 4.] [0. 2. 4. 6. 8.]
ValueError You are trying to save a non contiguous tensor: `a` which is not allow
True
RuntimeError Some tensors share memory, this will lead to duplicate memory on disk and potential differences when
```

**注意点・落とし穴**:
- **numpy 版は非 C-contiguous な配列を黙って壊して保存する**(実行確認): 転置・Fortran 順・飛び飛びスライスのいずれも、エラーにならず値が変わってしまう(転置した `a` は転置前の並びで読み戻された)。保存前に `np.ascontiguousarray(a)` を通す。
- torch 版は非連続テンソルを `ValueError: You are trying to save a non contiguous tensor ...` で拒否する。`.contiguous()` してから保存する。
- torch 版は、メモリを共有するテンソルを別名で保存しようとすると `RuntimeError: Some tensors share memory ...` になる(tied weights など)。`save_model`、または共有側を `.clone()` する。

---

### `h5py・.npy・safetensors の比較(サイズ・速度)`

**用途**: 同じ float32 配列 16MB を 3 形式で保存したときのファイルサイズ・全体書き込み/読み込み・部分読み込みの実測。

**シグネチャ**: `h5py.File` / `np.save` / `safetensors.numpy.save_file`

**使用例**:
```python
x = np.random.default_rng(0).normal(size=(4000, 1000)).astype("f4")   # 16MB

def best(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3

def write_h5():
    with h5py.File("b.h5", "w") as f:
        f["x"] = x
def read_h5():
    with h5py.File("b.h5", "r") as f:
        f["x"][:]
def part_st():
    with safe_open("b.safetensors", "np") as f:
        f.get_slice("x")[100:110]
def part_h5():
    with h5py.File("b.h5", "r") as f:
        f["x"][100:110]

print("write ms  safetensors %.1f  npy %.1f  h5py %.1f" % (
    best(lambda: save_file({"x": x}, "b.safetensors")), best(lambda: np.save("b.npy", x)), best(write_h5)))
print("size MB   safetensors %.3f  npy %.3f  h5py %.3f" % tuple(os.path.getsize(p) / 1e6 for p in ["b.safetensors", "b.npy", "b.h5"]))
print("read  ms  safetensors %.1f  npy %.1f  h5py %.1f" % (
    best(lambda: load_file("b.safetensors")), best(lambda: np.load("b.npy")), best(read_h5)))
print("rows 100:110 ms  safetensors %.3f  h5py %.3f" % (best(part_st), best(part_h5)))
print(os.path.getsize("b.safetensors") - x.nbytes, "バイト = safetensors のオーバーヘッド")
```
実行結果:
```
write ms  safetensors 9.5  npy 8.9  h5py 11.4
size MB   safetensors 16.000  npy 16.000  h5py 16.002
read  ms  safetensors 1.8  npy 1.6  h5py 2.2
rows 100:110 ms  safetensors 0.022  h5py 0.265
80 バイト = safetensors のオーバーヘッド
```

**注意点・落とし穴**:
- **実測(このマシン、best of 3。傾向のみ参考)**: サイズは 3 形式ともほぼ同じ(safetensors は生データ + 80 バイト、h5py は約 2KB のオーバーヘッド)。全体の書き込み・読み込みは 3 形式とも同じオーダーで、大差なし。10 行だけの部分読み込みは safetensors の方が h5py より速かった(上の出力)。
- h5py は圧縮・チャンク・追記・階層・属性・フィールド選択など多機能な汎用フォーマット。safetensors は「テンソルの辞書を配る」用途に絞った単純なフォーマットで、追記・圧縮に相当する API は無い。

---
