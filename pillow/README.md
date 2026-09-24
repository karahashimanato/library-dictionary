# Pillow 逆引き辞書

Pillow 12.3.0 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認。画像の実行結果は、コードで生成・保存した PNG を目視確認した内容を記述している)。`import PIL` / `from PIL import Image` で使う(パッケージ名は `Pillow`、import 名は `PIL`)。

検証環境で有効だった機能(`PIL.features.check`): JPEG(libjpeg-turbo)・zlib・WebP・AVIF・JPEG 2000・libtiff・FreeType(raqm)・littlecms2。ここに無い機能は使えない可能性があるため、形式・フォント関連のコードは自分の環境で確認すること。

## 目次

1. [画像の生成・読み込み・保存](#画像の生成読み込み保存)
2. [基本属性とモード](#基本属性とモード)
3. [幾何変換](#幾何変換)
4. [色・チャンネル操作](#色チャンネル操作)
5. [フィルタ(ImageFilter)](#フィルタimagefilter)
6. [画像強調(ImageEnhance / ImageOps)](#画像強調imageenhance--imageops)
7. [描画(ImageDraw / ImageFont)](#描画imagedraw--imagefont)
8. [numpy連携](#numpy連携)
9. [統計・メタデータ](#統計メタデータ)
10. [アニメーションGIF・複数フレーム](#アニメーションgif複数フレーム)
11. [その他](#その他)

---

## 画像の生成・読み込み・保存

### `Image.new(...)`

**用途**: 指定したモード・サイズ・色で新しい画像を作る。テスト画像や描画用キャンバス、マスクの作成の出発点になる。

**シグネチャ**: `Image.new(mode, size, color=0)`

**使用例**:
```python
from PIL import Image

im = Image.new("RGB", (200, 100), (255, 0, 0))
print(im.size, im.mode, im.format, im.getpixel((0, 0)))
print(Image.new("RGB", (4, 4), "skyblue").getpixel((0, 0)))
print(Image.new("RGB", (4, 4), "#00ff00").getpixel((0, 0)))
print(Image.new("L", (4, 4)).getpixel((0, 0)))
print(Image.new("RGBA", (4, 4)).getpixel((0, 0)))
im.save("01_new_red.png")
```
実行結果:
```
(200, 100) RGB None (255, 0, 0)
(135, 206, 235)
(0, 255, 0)
0
(0, 0, 0, 0)
```

**注意点・落とし穴**:
- `size` は `(幅, 高さ)` の順。numpy 配列の `shape` は `(高さ, 幅, ...)` なので逆になる。
- `color` を省略すると `0`。`RGBA` の場合は `(0, 0, 0, 0)` で完全な透明になる(黒の不透明ではない)。色は `(R, G, B)` のタプルのほか、色名(`"skyblue"`)や `"#rrggbb"` 文字列でも指定できる。
- `Image.new` で作った画像の `format` は `None`(ファイルから読み込んだ画像だけが `PNG` などの値を持つ)。

---

### `Image.open(...)`

**用途**: 画像ファイル(パスまたはファイルオブジェクト)を開く。形式は拡張子ではなく先頭バイトから自動判定される。

**シグネチャ**: `Image.open(fp, mode='r', formats=None)`

**使用例**:
```python
import io
from PIL import Image, UnidentifiedImageError

Image.new("RGB", (200, 100), (255, 0, 0)).save("02_src.png")

with Image.open("02_src.png") as im:
    print(type(im).__name__, im.format, im.size, im.mode)
    print(im.fp is not None)   # load() 前はファイルを開いたまま
    im.load()                  # ピクセルデータを実際に読み込む
    print(im.fp)               # 単一フレームなら読み込み後にファイルは閉じられる

try:
    Image.open(io.BytesIO(b"hello"))
except UnidentifiedImageError as e:
    print("UnidentifiedImageError")
try:
    Image.open("nonexistent.png")
except FileNotFoundError:
    print("FileNotFoundError")
try:
    Image.open("02_src.png", formats=["JPEG"])   # 許可する形式を限定
except UnidentifiedImageError:
    print("formats で PNG を除外 -> UnidentifiedImageError")
```
実行結果:
```
PngImageFile PNG (200, 100) RGB
True
None
UnidentifiedImageError
FileNotFoundError
formats で PNG を除外 -> UnidentifiedImageError
```

**注意点・落とし穴**:
- 戻り値は `Image.Image` のサブクラス(`PngImageFile` など)。ヘッダだけを読む遅延ロードで、実際のピクセルデータは `load()` や最初のピクセル操作(`getpixel`, `resize` 等)で初めて読まれる。そのため壊れたファイルでも `open` は成功し、後の処理で `OSError` になることがある(下の `verify` も参照)。
- 開いたままにするとファイルハンドルが残る(警告フィルタを有効にすると `ResourceWarning` が出る)。`with Image.open(...) as im:` で使うか、`im.load()` して `im.close()` する。
- 画像でないデータには `PIL.UnidentifiedImageError`、存在しないパスには `FileNotFoundError` が出る。

---

### `im.save(...)`

**用途**: 画像をファイル(またはファイルオブジェクト)へ保存する。形式は拡張子から判定されるが、`format=` で明示もできる。

**シグネチャ**: `Image.save(self, fp, format=None, **params)`

**使用例**:
```python
import io
from PIL import Image

im = Image.new("RGB", (200, 100), (255, 0, 0))
im.save("03_a.png")
im.save("03_a.jpg", quality=85)
im.save("03_noext", format="PNG")        # 拡張子が無いなら format 必須
print(Image.open("03_noext").format)

buf = io.BytesIO()
im.save(buf, format="PNG")               # メモリ上へ保存
print(len(buf.getvalue()))

for name, mode in [("x.unknown", "RGB"), ("a.jpg", "RGBA")]:
    try:
        Image.new(mode, (4, 4)).save(name)
    except Exception as e:
        print(type(e).__name__, e)
try:
    im.save(io.BytesIO())                # format も拡張子も無い
except Exception as e:
    print(type(e).__name__, e)
```
実行結果:
```
PNG
336
ValueError unknown file extension: .unknown
OSError cannot write mode RGBA as JPEG
ValueError unknown file extension: 
```

**注意点・落とし穴**:
- ファイルオブジェクト(`BytesIO` など)へ保存するときは拡張子が無いので `format=` が必須(省略すると `ValueError`)。
- `RGBA` や `P` など JPEG が扱えないモードの画像は `.jpg` で保存できず `OSError: cannot write mode RGBA as JPEG` になる。`im.convert("RGB")` してから保存する。
- `quality` などのフォーマット別オプションは `**params` で渡す(次項参照)。戻り値は `None`。

---

### `im.save` のフォーマット別オプション(JPEG / PNG / WebP / AVIF)

**用途**: 品質・圧縮レベル・可逆圧縮などを保存時に指定する。ファイルサイズと画質のトレードオフを調整する。

**シグネチャ**: `im.save(fp, format=None, **params)` の `params` の例: JPEG は `quality`(1-95)・`optimize`・`subsampling`、PNG は `compress_level`(0-9)・`optimize`、WebP は `quality`・`lossless`、AVIF は `quality`

**使用例**:
```python
import io
import numpy as np
from PIL import Image

rng = np.random.default_rng(0)
x = np.linspace(0, 255, 256)[None, :, None] * np.ones((256, 1, 3))
x[..., 1] = np.linspace(0, 255, 256)[:, None]
x[..., 2] = 128
photo = Image.fromarray(np.clip(x + rng.normal(0, 6, x.shape), 0, 255).astype("uint8"))

def size(**kw):
    b = io.BytesIO(); photo.save(b, **kw); return len(b.getvalue())

for q in (10, 50, 75, 95):
    print("JPEG quality", q, size(format="JPEG", quality=q))
print("JPEG optimize", size(format="JPEG", quality=75, optimize=True))
print("JPEG subsampling=0 (4:4:4)", size(format="JPEG", quality=75, subsampling=0))
for c in (0, 1, 9):
    print("PNG compress_level", c, size(format="PNG", compress_level=c))
print("WEBP quality=80", size(format="WEBP", quality=80))
print("WEBP lossless", size(format="WEBP", lossless=True))
print("AVIF quality=60", size(format="AVIF", quality=60))

b = io.BytesIO(); photo.save(b, format="PNG"); b.seek(0)
print("PNG lossless:", np.array_equal(np.asarray(photo), np.asarray(Image.open(b))))
b = io.BytesIO(); photo.save(b, format="JPEG", quality=50); b.seek(0)
print("JPEG lossless:", np.array_equal(np.asarray(photo), np.asarray(Image.open(b))))
```
実行結果:
```
JPEG quality 10 1764
JPEG quality 50 3836
JPEG quality 75 6915
JPEG quality 95 25903
JPEG optimize 5619
JPEG subsampling=0 (4:4:4) 9611
PNG compress_level 0 196993
PNG compress_level 1 138984
PNG compress_level 9 127214
WEBP quality=80 5148
WEBP lossless 149832
AVIF quality=60 3729
PNG lossless: True
JPEG lossless: False
```

**注意点・落とし穴**:
- JPEG は非可逆なので保存・再読み込みでピクセル値が変わる(上の `JPEG lossless: False`)。PNG は可逆。加工を繰り返す中間ファイルは PNG にし、最後だけ JPEG にするのが安全。
- サイズはノイズ入りのグラデーションを使った参考値で、libjpeg-turbo / libwebp などのバージョンや画像内容によって変わる。傾向(quality が高いほど大きい、`optimize` で小さくなる、`compress_level` を上げると小さくなる)だけを見る。
- このバージョン(12.3.0)では WebP・AVIF・JPEG 2000 などが利用できたが、これはビルド依存。使う前に `PIL.features.check("webp")` などで確認するのが確実(次項)。

---

### 対応フォーマットの確認(`Image.registered_extensions` / `Image.OPEN` / `Image.SAVE` / `features.check`)

**用途**: その環境の Pillow が読み書きできる形式と、コーデックの有無を調べる。

**シグネチャ**:
- `Image.registered_extensions()`
- `PIL.features.check(feature)`

**使用例**:
```python
import warnings
warnings.simplefilter("ignore")
from PIL import Image, features

ext = Image.registered_extensions()
print(len(ext), ext[".jpg"], ext[".jpeg"], ext[".webp"], ext[".tif"])
print("読める:", sorted(Image.OPEN))
print("書ける:", sorted(Image.SAVE))
for f in ["jpg", "zlib", "webp", "avif", "jpg_2000", "libtiff", "freetype2", "raqm", "littlecms2"]:
    print(f, features.check(f))
print(features.version("freetype2"))

Image.new("RGB", (64, 64), (10, 200, 30)).save("04_t.pdf")
try:
    Image.open("04_t.pdf")
except Exception as e:
    print("PDF は書き込み専用:", type(e).__name__)
```
実行結果:
```
70 JPEG JPEG WEBP TIFF
読める: ['AVIF', 'BLP', 'BMP', 'BUFR', 'CUR', 'DCX', 'DDS', 'DIB', 'EPS', 'FITS', 'FLI', 'FTEX', 'GBR', 'GIF', 'GRIB', 'HDF5', 'ICNS', 'ICO', 'IM', 'IMT', 'IPTC', 'JPEG', 'JPEG2000', 'MCIDAS', 'MPEG', 'MSP', 'PCD', 'PCX', 'PIXAR', 'PNG', 'PPM', 'PSD', 'QOI', 'SGI', 'SPIDER', 'SUN', 'TGA', 'TIFF', 'WEBP', 'WMF', 'XBM', 'XPM', 'XVTHUMB']
書ける: ['AVIF', 'BLP', 'BMP', 'BUFR', 'DDS', 'DIB', 'EPS', 'GIF', 'GRIB', 'HDF5', 'ICNS', 'ICO', 'IM', 'JPEG', 'JPEG2000', 'MPO', 'MSP', 'PALM', 'PCX', 'PDF', 'PNG', 'PPM', 'QOI', 'SGI', 'SPIDER', 'TGA', 'TIFF', 'WEBP', 'WMF', 'XBM']
jpg True
zlib True
webp True
avif True
jpg_2000 True
libtiff True
freetype2 True
raqm True
littlecms2 True
2.14.3
PDF は書き込み専用: UnidentifiedImageError
```

**注意点・落とし穴**:
- `Image.OPEN`(読める形式)と `Image.SAVE`(書ける形式)は一致しない。例えば PDF は保存のみ可能で、`Image.open` では開けない。
- `features.check` に未知の名前を渡すと `UserWarning: Unknown feature` が出て `False` 扱いになる(例: 12.3.0 では `webp_anim` は未登録名)。名前は `features.pilinfo()` や `features.get_supported()` で確認できる。

---

### `Image.frombytes(...)` / `im.tobytes()`

**用途**: 生のピクセルバイト列から画像を作る(`frombytes`)/ 画像を生のバイト列に書き出す(`tobytes`)。

**シグネチャ**:
- `Image.frombytes(mode, size, data, decoder_name='raw', *args)`
- `Image.tobytes(self, encoder_name='raw', *args)`

**使用例**:
```python
from PIL import Image

data = bytes([255, 0, 0,   0, 255, 0,
              0, 0, 255,   255, 255, 255])   # 2x2 の RGB (行優先)
im = Image.frombytes("RGB", (2, 2), data)
print(im.size, [im.getpixel((x, y)) for y in range(2) for x in range(2)])
print(im.tobytes() == data, len(im.tobytes()))
try:
    Image.frombytes("RGB", (2, 2), b"abc")
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
(2, 2) [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
True 12
ValueError: not enough image data
```

**注意点・落とし穴**:
- バイト列の長さは `幅 x 高さ x バンド数`(`RGB` なら 3 倍)ちょうど必要。短いと `ValueError: not enough image data`。並びは行優先で、各画素のチャンネルが連続する(`RGBRGB...`)。
- numpy 配列から作るなら `Image.fromarray` の方が形状・dtype が検査されて安全(numpy 連携の項を参照)。

---

## 基本属性とモード

### `im.size` / `im.width` / `im.height` / `im.mode` / `im.format` / `im.info` / `im.getbands()`

**用途**: 画像の基本属性を読む。

**シグネチャ**: `Image.size`(`(幅, 高さ)` のタプル)、`Image.width`、`Image.height`、`Image.mode`、`Image.format`、`Image.info`(dict)、`Image.getbands()`(バンド名のタプルを返すメソッド)

**使用例**:
```python
from PIL import Image

im = Image.new("RGB", (200, 100), (255, 128, 0))
print(im.size, im.width, im.height, im.mode, im.format, im.info, im.getbands())

im.save("05_attr.png")
o = Image.open("05_attr.png")
print(o.format, o.format_description, o.info, o.getbands())
print(Image.new("RGBA", (1, 1)).getbands(), Image.new("L", (1, 1)).getbands())
```
実行結果:
```
(200, 100) 200 100 RGB None {} ('R', 'G', 'B')
PNG Portable network graphics {} ('R', 'G', 'B')
('R', 'G', 'B', 'A') ('L',)
```

**注意点・落とし穴**:
- `size` は `(width, height)`。numpy に変換すると `shape == (height, width, channels)` と順序が逆になる。
- `format` は `Image.open` で開いた画像のみ値を持ち、`new` や `resize` の結果などでは `None` になる。
- `info` はファイル形式ごとのメタデータ(dpi、PNGテキスト、GIFの `duration` など)を持つ dict(詳細は「統計・メタデータ」の項)。

---

### モード一覧(`1` / `L` / `LA` / `P` / `RGB` / `RGBA` / `CMYK` / `HSV` / `I` / `F` / `I;16`)

**用途**: 画像のピクセル表現を決める `mode` の種類と、それぞれのバンド構成・値の型を確認する。

**シグネチャ**: `PIL.ImageMode.getmode(mode)`(`bands`, `basemode`, `basetype`, `typestr` を持つ)

**使用例**:
```python
from PIL import Image, ImageMode

for m in ["1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "HSV", "LAB", "I", "F", "I;16"]:
    d = ImageMode.getmode(m)
    print(f"{m:6s} bands={d.bands!s:24s} type={d.basetype} typestr={d.typestr}")

print(Image.new("1", (1, 1), 1).getpixel((0, 0)))
print(Image.new("L", (1, 1), 200).getpixel((0, 0)))
print(Image.new("LA", (1, 1), (10, 20)).getpixel((0, 0)))
print(Image.new("I", (1, 1), 100000).getpixel((0, 0)))
print(Image.new("F", (1, 1), 1.5).getpixel((0, 0)))
print(Image.new("CMYK", (1, 1), (0, 255, 255, 0)).getpixel((0, 0)))
```
実行結果:
```
1      bands=('1',)                   type=L typestr=|b1
L      bands=('L',)                   type=L typestr=|u1
LA     bands=('L', 'A')               type=L typestr=|u1
P      bands=('P',)                   type=L typestr=|u1
RGB    bands=('R', 'G', 'B')          type=L typestr=|u1
RGBA   bands=('R', 'G', 'B', 'A')     type=L typestr=|u1
CMYK   bands=('C', 'M', 'Y', 'K')     type=L typestr=|u1
YCbCr  bands=('Y', 'Cb', 'Cr')        type=L typestr=|u1
HSV    bands=('H', 'S', 'V')          type=L typestr=|u1
LAB    bands=('L', 'A', 'B')          type=L typestr=|u1
I      bands=('I',)                   type=I typestr=<i4
F      bands=('F',)                   type=F typestr=<f4
I;16   bands=('I',)                   type=L typestr=<u2
1
200
(10, 20)
100000
1.5
(0, 255, 255, 0)
```

**注意点・落とし穴**:
- `1` は2値画像(`convert("1")` の結果は 0 / 255 で読み出される)、`L` は 8bit グレースケール、`P` はパレット(値は色そのものではなくパレットのインデックス)、`I` は 32bit 整数、`F` は 32bit 浮動小数点。
- ピクセル値は `L` なら int、`RGB` なら int のタプル、`F` なら float で返る。`getpixel`/`putpixel` に渡す型もモードに合わせる必要がある。
- `I;16`(16bit グレースケール)は 16bit の PNG/TIFF を読むときなどに現れる。numpy の `uint16` 配列から `Image.fromarray` で作るとこのモードになる(numpy 連携の項を参照)。

---

### `im.convert(...)`

**用途**: 画像のモードを変換する(`RGB`→`L`、`RGBA`→`RGB` など)。元の画像は変更せず新しい画像を返す。

**シグネチャ**: `Image.convert(self, mode=None, matrix=None, dither=None, palette=Palette.WEB, colors=256)`

**使用例**:
```python
from PIL import Image

rgb = Image.new("RGB", (2, 1))
rgb.putpixel((0, 0), (255, 0, 0))
rgb.putpixel((1, 0), (0, 255, 0))
for m in ["L", "1", "RGBA", "CMYK", "HSV", "YCbCr", "I", "F", "LA"]:
    c = rgb.convert(m)
    print(m, c.mode, c.getpixel((0, 0)), c.getpixel((1, 0)))

# RGBA -> RGB: アルファは単に捨てられる(背景に合成はされない)
print(Image.new("RGBA", (1, 1), (255, 0, 0, 0)).convert("RGB").getpixel((0, 0)))
# 同じモードへの変換でもコピーが返る
print(rgb.convert("RGB") is rgb)

# matrix: RGB -> セピア調(各行が出力の R,G,B、各行4要素で末尾はオフセット)
px = Image.new("RGB", (1, 1), (100, 150, 200))
sepia = (0.393, 0.769, 0.189, 0,  0.349, 0.686, 0.168, 0,  0.272, 0.534, 0.131, 0)
print(px.convert("RGB", sepia).getpixel((0, 0)))
# 4要素の matrix を "L" に渡すと1チャンネルの線形結合
print(px.convert("L", (1, 0, 0, 0)).getpixel((0, 0)))

# 2値化: dither の有無
g = Image.linear_gradient("L").resize((256, 32))
print("dither(既定 FLOYDSTEINBERG):", g.convert("1").getcolors())
print("dither=NONE:              ", g.convert("1", dither=Image.Dither.NONE).getcolors())
```
実行結果:
```
L L 76 150
1 1 0 255
RGBA RGBA (255, 0, 0, 255) (0, 255, 0, 255)
CMYK CMYK (0, 255, 255, 0) (255, 0, 255, 0)
HSV HSV (0, 255, 255) (85, 255, 255)
YCbCr YCbCr (76, 84, 255) (149, 43, 21)
I I 76 150
F F 76.24500274658203 149.68499755859375
LA LA (76, 255) (150, 255)
(255, 0, 0)
False
(192, 171, 134)
100
dither(既定 FLOYDSTEINBERG): [(4035, 0), (4157, 255)]
dither=NONE:               [(4096, 0), (4096, 255)]
```

**注意点・落とし穴**:
- `RGB`→`L` は ITU-R 601 の輝度式 `L = R*299/1000 + G*587/1000 + B*114/1000` で計算される(赤 (255,0,0) → 76, 緑 (0,255,0) → 150)。
- `RGBA`→`RGB` は透明度を合成せず捨てるだけ。透明部分が黒(または元のRGB値)のまま出てくる。背景色に合成したいなら `Image.alpha_composite` や `paste(..., mask=alpha)` を使う。
- `convert("1")` は既定で Floyd-Steinberg ディザリングがかかる(ノイズ状の白黒点で階調を表現)。単純な閾値2値化にしたいなら `dither=Image.Dither.NONE`、または `point` を使う。
- 未対応の変換先(例: 存在しないモード名)は `ValueError: image has wrong mode` になる。

---

## 幾何変換

### `im.resize(...)` と `Image.Resampling`

**用途**: 画像を指定サイズに拡大・縮小する。補間方法は `resample` で選ぶ。

**シグネチャ**: `Image.resize(self, size, resample=None, box=None, reducing_gap=None)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

r = im.resize((60, 40))
print(r.size, r.getpixel((0, 0)), r.getpixel((59, 39)))

# 補間方法による違い(緑と黄の境界付近 = 縮小後の (30,20) の値)
for name in ["NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"]:
    res = getattr(Image.Resampling, name)
    print(f"{name:9s}", int(res), im.resize((60, 40), resample=res).getpixel((30, 20)))

# アスペクト比を保って幅300へ
w, h = im.size
print(im.resize((300, int(h * 300 / w))).size)

# box: 元画像の一部だけを切り出しつつリサイズ
rb = im.resize((60, 40), box=(60, 0, 120, 40))
print(rb.size, rb.getpixel((5, 5)))

# 拡大: NEAREST と BILINEAR の違い(2画素 -> 4画素)
strip = Image.new("L", (2, 1)); strip.putpixel((1, 0), 255)
print([strip.resize((4, 1), Image.Resampling.NEAREST).getpixel((x, 0)) for x in range(4)])
print([strip.resize((4, 1), Image.Resampling.BILINEAR).getpixel((x, 0)) for x in range(4)])
```
実行結果:
```
(60, 40) (255, 0, 0) (0, 160, 0)
NEAREST   0 (0, 160, 0)
BOX       4 (0, 160, 0)
BILINEAR  2 (32, 123, 28)
HAMMING   5 (10, 148, 9)
BICUBIC   3 (17, 139, 16)
LANCZOS   1 (14, 143, 13)
(300, 200)
(60, 40) (0, 0, 255)
[0, 0, 255, 255]
[0, 64, 191, 255]
```

**注意点・落とし穴**:
- `size` は `(幅, 高さ)`。片方だけ指定してアスペクト比を保つ機能は無いので、自分で計算する(比率を保ったまま収めたいだけなら `thumbnail` / `ImageOps.contain` が楽)。
- `resample` を省略した場合は `BICUBIC` が使われる(ただし `1` / `P` モードは `NEAREST`)。ドット絵・マスク・ラベル画像を拡大するときは `Resampling.NEAREST` を明示しないと色が混ざる。
- 縮小には `LANCZOS`(最高品質・低速)/ `BICUBIC` / `BOX`(高速)などが向く。`NEAREST` で縮小するとエイリアシングが出やすい。定数は `Image.Resampling.*` を使う(`Image.LANCZOS` のような旧来の整数定数も 12.3.0 に残っているが、`Image.LANCZOS` は単なる整数 `1` で、新規コードでは `Resampling` の列挙型を推奨)。

---

### `im.reduce(...)`

**用途**: 整数倍で縮小する(各ブロックの平均)。`resize` より高速に、ブロック平均で縮める。

**シグネチャ**: `Image.reduce(self, factor, box=None)`

**使用例**:
```python
from PIL import Image

im = Image.new("RGB", (120, 80), (255, 0, 0))
print(im.reduce(2).size, im.reduce((2, 4)).size)

g = Image.new("L", (4, 2)); 
for x, v in enumerate([0, 100, 200, 255]): g.putpixel((x, 0), v); g.putpixel((x, 1), v)
print([g.reduce(2).getpixel((x, 0)) for x in range(2)])
```
実行結果:
```
(60, 40) (60, 20)
[50, 228]
```

**注意点・落とし穴**:
- `factor` は整数(または `(x方向, y方向)` のタプル)で、割り切れない端数は切り上げたサイズになる。任意の縮小率が必要なときは `resize` を使う。

---

### `im.crop(...)`

**用途**: 矩形領域を切り出す。`box` は `(left, upper, right, lower)`。

**シグネチャ**: `Image.crop(self, box=None)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

c = im.crop((60, 0, 120, 40))
print(c.size, c.getpixel((0, 0)))

# 元画像の外側を含む範囲: はみ出た部分は黒(RGB)で埋まる
c2 = im.crop((100, 60, 140, 100))
print(c2.size, c2.getpixel((0, 0)), c2.getpixel((30, 30)))
c3 = im.crop((-10, -10, 10, 10))
print(c3.size, c3.getpixel((0, 0)), c3.getpixel((15, 15)))

# 切り出し結果は独立したコピー
c4 = im.crop((0, 0, 10, 10)); c4.putpixel((0, 0), (1, 2, 3))
print(im.getpixel((0, 0)))
```
実行結果:
```
(60, 40) (0, 0, 255)
(40, 40) (255, 255, 0) (0, 0, 0)
(20, 20) (0, 0, 0) (255, 0, 0)
(255, 0, 0)
```

**注意点・落とし穴**:
- `box` の右・下は「含まない」端(幅 = right - left)。`(60, 0, 120, 40)` は幅60・高さ40。
- `(x, y, w, h)` 形式ではない点に注意(`right`/`lower` は座標)。幅と高さで指定したいときは `crop((x, y, x + w, y + h))` と書く。
- 範囲外を含めてもエラーにならず、はみ出した部分は 0(`RGB` なら黒、`RGBA` なら透明黒)で埋められる。

---

### `im.rotate(...)`

**用途**: 画像を反時計回りに回転する(角度は度)。

**シグネチャ**: `Image.rotate(self, angle, resample=Resampling.NEAREST, expand=False, center=None, translate=None, fillcolor=None)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

for ang, ex in [(90, False), (90, True), (45, False), (45, True)]:
    print(ang, "expand=", ex, im.rotate(ang, expand=ex).size)

x = im.rotate(45, expand=True, fillcolor=(200, 200, 200))
print(x.getpixel((0, 0)))                      # 空いた角は fillcolor
print(im.rotate(45).getpixel((0, 0)))          # 既定は 0 (黒)
print(im.rotate(90).getpixel((0, 0)), im.rotate(90).getpixel((119, 0)))
print(im.rotate(30, resample=Image.Resampling.BICUBIC, expand=True).size)
print(im.rotate(0, translate=(10, 5), fillcolor=(0, 0, 0)).getpixel((10, 5)))
x.save("13_rotate45.png")
```
実行結果:
```
90 expand= False (120, 80)
90 expand= True (80, 120)
45 expand= False (120, 80)
45 expand= True (142, 142)
(200, 200, 200)
(0, 0, 0)
(0, 0, 0) (0, 0, 0)
(144, 130)
(255, 0, 0)
```
`13_rotate45.png` は 142x142。中央に元の画像(左=赤、右上=青、右下=緑+黄の楕円)が反時計回りに 45 度回った状態で置かれ、外側の空き領域は薄いグレー `(200,200,200)` で塗られる。

**注意点・落とし穴**:
- 既定は反時計回り。時計回りにしたいなら負の角度を渡す。
- `expand=False`(既定)は出力サイズが元と同じなので、はみ出した部分が切り落とされる。非正方形を 90 度回すと `(0, 0)` も `(119, 0)` も `(0, 0, 0)`(黒)になっているように、内容が大きく欠ける。全体を残すなら `expand=True`(出力サイズが変わる)。
- 90度単位の回転・反転なら `transpose` の方が補間が無いのでピクセルが劣化しない(`rotate(-90, expand=True)` と `transpose(ROTATE_270)` は同じ結果になる)。`resample` の既定は `NEAREST` でギザギザが出るので、滑らかにしたいときは `BILINEAR` / `BICUBIC` を指定する。
- 空いた領域の色は `fillcolor`(既定は黒、`RGBA` なら透明黒)で指定できる。

---

### `im.transpose(...)`

**用途**: 90度単位の回転と左右・上下反転を、補間なしで(ピクセルを劣化させずに)行う。

**シグネチャ**: `Image.transpose(self, method)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

for t in Image.Transpose:
    x = im.transpose(t)
    print(f"{t.name:16s}", x.size, x.getpixel((0, 0)))

pics = [im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        im.transpose(Image.Transpose.ROTATE_90), im.transpose(Image.Transpose.TRANSPOSE)]
canvas = Image.new("RGB", (sum(p.width for p in pics) + 10 * 5, 140), (230, 230, 230))
x = 10
for p in pics:
    canvas.paste(p, (x, 10)); x += p.width + 10
canvas.save("14_transpose.png")
```
実行結果:
```
FLIP_LEFT_RIGHT  (120, 80) (0, 0, 255)
FLIP_TOP_BOTTOM  (120, 80) (255, 0, 0)
ROTATE_90        (80, 120) (0, 0, 255)
ROTATE_180       (120, 80) (0, 160, 0)
ROTATE_270       (80, 120) (255, 0, 0)
TRANSPOSE        (80, 120) (255, 0, 0)
TRANSVERSE       (80, 120) (0, 160, 0)
```
`14_transpose.png`(450x140)は元画像・左右反転・90度反時計回り・TRANSPOSE を横に並べた図。2番目は左右が入れ替わって右半分が赤に、3番目は赤が下・青が左上・緑が右上に、4番目(TRANSPOSE)は赤が上・青が左下・緑が右下になっている。

**注意点・落とし穴**:
- `ROTATE_90` は反時計回り(`rotate(90, expand=True)` と同じ向き)。`ROTATE_270` は時計回りの90度。`TRANSPOSE` は主対角線(左上-右下)に関する転置、`TRANSVERSE` は反対角線に関する転置。
- 回転系(90/270/TRANSPOSE/TRANSVERSE)は幅と高さが入れ替わる。

---

### `im.thumbnail(...)`

**用途**: アスペクト比を保ったまま、指定サイズ内に収まるよう**その場で**縮小する。拡大はしない。

**シグネチャ**: `Image.thumbnail(self, size, resample=Resampling.BICUBIC, reducing_gap=2.0)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

t = im.copy()
ret = t.thumbnail((50, 50))
print(ret, t.size)                      # 戻り値は None, t が縮小される

t = im.copy(); t.thumbnail((500, 500)); print(t.size)   # 元より大きい枠 -> 変化なし
t = im.copy(); t.thumbnail((100, 20));  print(t.size)   # 高さ20に合わせて縮小
t = Image.new("RGB", (1000, 1)); t.thumbnail((100, 100)); print(t.size)  # 高さは最小1

# JPEG は draft で読み込み時に縮小デコードでき、大きい写真のサムネイル生成が速くなる
im.save("15_src.jpg")
j = Image.open("15_src.jpg")
print(j.draft("RGB", (60, 40)), j.size)
```
実行結果:
```
None (50, 33)
(120, 80)
(30, 20)
(100, 1)
('RGB', (0, 0, 60.0, 40.0)) (60, 40)
```

**注意点・落とし穴**:
- 元の画像オブジェクトを書き換え、戻り値は `None`。元画像を残したいなら先に `copy()` する(`im2 = im.thumbnail(...)` は `None` になる典型的な間違い)。
- 枠より小さい画像は拡大されない。拡大もしたい場合は `resize` か `ImageOps.contain` を使う。
- JPEG では `im.draft(mode, size)` を `load` 前に呼ぶと、1/2, 1/4, 1/8 単位で縮小しながらデコードできる(上の出力では `(120, 80)` が `(60, 40)` になった)。`thumbnail` は内部で自動的に `draft` を使う。

---

### `ImageOps.fit` / `ImageOps.contain` / `ImageOps.pad` / `ImageOps.scale` / `ImageOps.expand` / `ImageOps.crop`

**用途**: 固定サイズへのリサイズ(切り抜き・収める・余白付き)や、余白の追加・削除をまとめて行う。

**シグネチャ**:
- `ImageOps.fit(image, size, method=Resampling.BICUBIC, bleed=0.0, centering=(0.5, 0.5))`
- `ImageOps.contain(image, size, method=Resampling.BICUBIC)`
- `ImageOps.pad(image, size, method=Resampling.BICUBIC, color=None, centering=(0.5, 0.5))`
- `ImageOps.scale(image, factor, resample=Resampling.BICUBIC)`
- `ImageOps.expand(image, border=0, fill=0)`
- `ImageOps.crop(image, border=0)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

from PIL import ImageOps

print("fit    ", ImageOps.fit(im, (50, 50)).size)         # 中央を切り抜いて 50x50 ちょうど
print("contain", ImageOps.contain(im, (50, 50)).size)     # 比率を保って枠内に収める
p = ImageOps.pad(im, (50, 50), color=(0, 0, 0))            # 比率を保ち、余りを color で埋める
print("pad    ", p.size, p.getpixel((0, 0)), p.getpixel((0, 25)))
f = ImageOps.fit(im, (50, 50))
print("fit の左端/右上:", f.getpixel((0, 25)), f.getpixel((49, 10)))
print("scale  ", ImageOps.scale(im, 0.5).size)
print("expand ", ImageOps.expand(im, border=5, fill=(0, 0, 0)).size)
print("crop   ", ImageOps.crop(im, 10).size)
```
実行結果:
```
fit     (50, 50)
contain (50, 33)
pad     (50, 50) (0, 0, 0) (255, 0, 0)
fit の左端/右上: (255, 0, 0) (0, 0, 255)
scale   (60, 40)
expand  (130, 90)
crop    (100, 60)
```

**注意点・落とし穴**:
- `fit` は比率を保つが、はみ出す部分は切り落とす(SNSのアイコンなど正方形サムネイル向き)。`contain` は切り落とさず縮小するのでサイズは枠ぴったりとは限らない。`pad` は比率を保って枠ぴったりの画像を作る(余白を `color` で埋める)。
- `ImageOps.expand` は四辺に同じ幅の枠(`border=(横, 縦)` のタプルも可)を足し、`ImageOps.crop` は四辺から同じ幅を削る。
- `ImageOps.contain` は `thumbnail` と違い、元画像を変更せず新しい画像を返し、枠より小さい場合は拡大する。

---

### `im.transform(...)`(アフィン変換・遠近変換など)

**用途**: アフィン変換・矩形の切り出し・遠近変換などの幾何変換を行う。`data` の値は「出力座標 → 入力座標」の逆写像で指定する。

**シグネチャ**: `Image.transform(self, size, method, data=None, resample=Resampling.NEAREST, fill=1, fillcolor=None)`

**使用例**:
```python
from PIL import Image, ImageDraw

im = Image.new("RGB", (100, 60), (255, 0, 0))
ImageDraw.Draw(im).rectangle((50, 0, 99, 59), fill=(0, 0, 255))

# AFFINE: (a, b, c, d, e, f) で 入力座標 = (a*x + b*y + c, d*x + e*y + f)
# c=-20 -> 出力の x=20 が入力の x=0 に対応 = 画像が右へ20ピクセル移動
af = im.transform((100, 60), Image.Transform.AFFINE, (1, 0, -20, 0, 1, 0), fillcolor=(0, 0, 0))
print(af.getpixel((10, 10)), af.getpixel((30, 10)), af.getpixel((80, 10)))

# EXTENT: 入力の (x0,y0,x1,y1) を出力サイズへ拡縮
ex = im.transform((50, 30), Image.Transform.EXTENT, (0, 0, 100, 60))
print(ex.size)

# PERSPECTIVE: 8個の係数
per = im.transform((100, 60), Image.Transform.PERSPECTIVE, (1, 0.2, 0, 0, 1, 0, 0.002, 0, 1), Image.Resampling.BILINEAR)
print(per.size)
per.save("17_perspective.png")
```
実行結果:
```
(0, 0, 0) (255, 0, 0) (0, 0, 255)
(50, 30)
(100, 60)
```
`17_perspective.png` は 100x60。元は縦の境界線で左が赤・右が青の画像だが、遠近変換により境界線が垂直でなくなり、上端が右・下端が左へ傾いた斜めの境界になる。

**注意点・落とし穴**:
- `data` は「出力画素 (x, y) が入力のどこから来るか」の逆写像の係数。移動量の符号が直感と逆になりやすい(上の例では `c=-20` で右へ 20px 動く)。
- `resample` の既定は `NEAREST` でギザギザになる。滑らかにしたいなら `BILINEAR` / `BICUBIC` を指定する。単純な回転・縮小拡大には `rotate` / `resize` の方が読みやすい。

---

## 色・チャンネル操作

### `im.getpixel(...)` / `im.putpixel(...)` / `im.load()`

**用途**: 1ピクセルの値の読み書き。大量のピクセルを処理するなら `load()` で得たアクセサ、さらに速くしたいなら numpy を使う。

**シグネチャ**:
- `Image.getpixel(self, xy)`
- `Image.putpixel(self, xy, value)`
- `Image.load(self)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

print(im.getpixel((0, 0)), im.getpixel((119, 79)), im.getpixel((100, 60)))
print(im.convert("L").getpixel((0, 0)), im.convert("RGBA").getpixel((0, 0)), im.convert("P").getpixel((0, 0)))

t = im.copy()
t.putpixel((0, 0), (1, 2, 3))
print(t.getpixel((0, 0)), im.getpixel((0, 0)))     # 元画像は変わらない

pa = t.load()            # PixelAccess: pa[x, y] で読み書き
pa[1, 1] = (9, 9, 9)
print(pa[1, 1], t.getpixel((1, 1)))

try:
    im.getpixel((200, 200))
except IndexError as e:
    print("IndexError:", e)
row = Image.new("RGB", (3, 1)); row.putpixel((2, 0), (0, 0, 9))
print("負の座標は末尾から:", row.getpixel((-1, 0)))
try:
    t.putpixel((0, 0), (1, 2))          # 2要素は不可
except TypeError as e:
    print("TypeError:", e)
z = Image.new("RGB", (2, 2)); z.putpixel((0, 0), 5)   # int を渡してもエラーにならない
print(z.getpixel((0, 0)))
```
実行結果:
```
(255, 0, 0) (0, 160, 0) (255, 255, 0)
76 (255, 0, 0, 255) 15
(1, 2, 3) (255, 0, 0)
(9, 9, 9) (9, 9, 9)
IndexError: image index out of range
負の座標は末尾から: (0, 0, 9)
TypeError: color must be int, or tuple of one, three or four elements
(5, 0, 0)
```

**注意点・落とし穴**:
- 座標は `(x, y)` = `(列, 行)`。numpy の `a[y, x]` とは順序が逆。
- 範囲外の正の座標は `IndexError` だが、**負の座標はエラーにならず末尾から数えた位置**(`getpixel((-1, 0))` が最右列)になる。バグに気付きにくい。
- `RGB` 画像に整数 1 つを `putpixel` してもエラーにならず、その値が R チャンネルに入り `(5, 0, 0)` になる。値の型はモードに合わせる(`RGB` なら3要素のタプル)。
- ループで全画素を `getpixel`/`putpixel` するのは非常に遅い。`point`・`ImageChops`・numpy などのベクトル化された処理に置き換える。

---

### `im.split()` / `Image.merge(...)` / `im.getchannel(...)`

**用途**: チャンネルごとに分解する(`split`, `getchannel`)/ 単チャンネル画像を束ねて多チャンネル画像にする(`merge`)。

**シグネチャ**:
- `Image.split(self)`
- `Image.merge(mode, bands)`
- `Image.getchannel(self, channel)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

r, g, b = im.split()
print(r.mode, r.size, r.getpixel((0, 0)), g.getpixel((0, 0)), b.getpixel((100, 10)))
print(im.getchannel("R").getpixel((0, 0)), im.getchannel(2).getpixel((100, 10)))

swapped = Image.merge("RGB", (b, g, r))            # R と B を入れ替え
print(swapped.getpixel((0, 0)), swapped.getpixel((100, 10)))
print(Image.merge("RGBA", (r, g, b, Image.new("L", im.size, 128))).getpixel((0, 0)))

try:
    Image.merge("RGB", (r, g))
except ValueError as e:
    print("ValueError:", e)
try:
    Image.merge("RGB", (r, g, Image.new("L", (5, 5))))
except ValueError as e:
    print("ValueError:", e)
print(len(im.convert("L").split()), len(im.convert("RGBA").split()))
```
実行結果:
```
L (120, 80) 255 0 255
255 255
(0, 0, 255) (255, 0, 0)
(255, 0, 0, 128)
ValueError: wrong number of bands
ValueError: size mismatch
1 4
```

**注意点・落とし穴**:
- `split()` の戻り値は `L` モードの画像のタプル(`RGBA` なら4つ)。`L` 画像に対しては要素1つのタプル。
- `merge` に渡す画像は、モードのバンド数と同数で、すべて同じサイズ・`L` モードである必要がある(数やサイズが違うと `ValueError`)。
- 1チャンネルだけを取り出したいときは `getchannel` が `split()[i]` より手軽(`"R"` などの名前でも番号でも可)。

---

### `im.point(...)`

**用途**: 各ピクセル値をルックアップテーブル(関数または配列)で一括変換する。二値化・ガンマ補正・トーンカーブなどに使う。

**シグネチャ**: `Image.point(self, lut, mode=None)`

**使用例**:
```python
from PIL import Image, ImageDraw

# 120x80 のテスト画像: 左半分=赤, 右上=青, 右下=緑(+黄色い楕円)
im = Image.new("RGB", (120, 80), (255, 255, 255))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 59, 79), fill=(255, 0, 0))
d.rectangle((60, 0, 119, 39), fill=(0, 0, 255))
d.rectangle((60, 40, 119, 79), fill=(0, 160, 0))
d.ellipse((85, 55, 110, 75), fill=(255, 255, 0))

inv = im.point(lambda v: 255 - v)                       # 全チャンネルに同じ変換
print(inv.getpixel((0, 0)), inv.getpixel((100, 10)))

gray = im.convert("L")
th = gray.point(lambda v: 255 if v > 100 else 0)        # 閾値で二値化
print(th.mode, th.getcolors())
th1 = gray.point(lambda v: v > 100 and 255, mode="1")   # mode="1" で1bit画像に
print(th1.mode, th1.getcolors())

half = im.point(lambda v: v * 0.5)                       # 浮動小数点になっても丸められる
print(half.getpixel((0, 0)), half.mode)

# LUT を直接渡す: RGB は 256*3 要素、L は 256 要素
lut = [min(255, i * 2) for i in range(256)]
print(im.point(lut * 3).getpixel((60, 60)), gray.point(lut).getpixel((0, 0)))

# 1チャンネルにだけ適用
r, g, b = im.split()
print(Image.merge("RGB", (r.point(lambda v: v // 2), g, b)).getpixel((0, 0)))

try:
    Image.new("RGB", (2, 2)).point([0] * 256 * 2)      # 長さが合わない
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
(0, 255, 255) (255, 255, 0)
L [(9178, 0), (422, 255)]
1 [(9178, 0), (422, 255)]
(128, 0, 0) RGB
(0, 255, 0) 152
(127, 0, 0)
ValueError: wrong number of lut entries
```

**注意点・落とし穴**:
- 関数は 0-255 の各値に対して**256回だけ**呼ばれてテーブル化されるので、画素数に関わらず高速(全画素を Python ループで処理するより桁違いに速い)。そのため `lambda v: v + x_offset(座標)` のような座標依存の処理は書けない。
- LUT を直接渡す場合、`RGB` は `256 * 3` 要素、`L` は 256 要素が必要で、長さが違うと `ValueError: wrong number of lut entries`。
- `mode="1"` を付けると 1bit の二値画像として返る。`L` 画像に対して使う(`RGB` に付けると LUT の要素数が合わず `ValueError`)。閾値処理でマスクを作る用途に便利。
- `I`/`F` モードでは `v * scale + offset` の形の1次式しか使えない(非線形な式は `TypeError`)。

---

### `im.paste(...)`

**用途**: 画像・単色を別の画像の指定位置へ貼り付ける(**その場で**書き換える)。`mask` を渡すと、マスクの濃さに応じて合成される。

**シグネチャ**: `Image.paste(self, im, box=None, mask=None)`

**使用例**:
```python
from PIL import Image, ImageDraw

base = Image.new("RGB", (100, 60), (240, 240, 240))
patch = Image.new("RGB", (40, 30), (255, 0, 0))

b1 = base.copy(); b1.paste(patch, (10, 10))
print("貼り付け:", b1.getpixel((10, 10)), b1.getpixel((9, 9)))

b2 = base.copy(); b2.paste((0, 128, 255), (50, 10, 90, 50))     # 単色で矩形を塗る
print("単色:", b2.getpixel((60, 20)), b2.getpixel((49, 20)))

mask = Image.new("L", (40, 30), 0)
ImageDraw.Draw(mask).ellipse((0, 0, 39, 29), fill=255)          # 楕円=255 の L マスク
b3 = base.copy(); b3.paste(patch, (10, 10), mask)
print("楕円マスク:", b3.getpixel((30, 25)), b3.getpixel((10, 10)))

b4 = base.copy(); b4.paste(patch, (10, 10), Image.new("L", (40, 30), 128))
print("半透明マスク(128):", b4.getpixel((30, 25)))

# RGBA 画像は自分自身をマスクに渡すとアルファを使って貼れる
rgba = Image.new("RGBA", (40, 30), (255, 0, 0, 0))
ImageDraw.Draw(rgba).rectangle((10, 10, 29, 19), fill=(255, 0, 0, 255))
b5 = base.copy(); b5.paste(rgba, (0, 0), rgba)
print("RGBA+mask:", b5.getpixel((15, 15)), b5.getpixel((0, 0)))
b6 = base.copy(); b6.paste(rgba, (0, 0))
print("RGBA(maskなし):", b6.getpixel((0, 0)))                   # 透明部分も (255,0,0) になる

b8 = base.copy(); b8.paste(patch, (-20, -15))                    # はみ出す位置も可(範囲外は無視)
print("はみ出し:", b8.getpixel((0, 0)), b8.getpixel((30, 0)))
print("戻り値:", base.copy().paste(patch, (0, 0)))
b5.save("24_paste_mask.png")
```
実行結果:
```
貼り付け: (255, 0, 0) (240, 240, 240)
単色: (0, 128, 255) (240, 240, 240)
楕円マスク: (255, 0, 0) (240, 240, 240)
半透明マスク(128): (248, 120, 120)
RGBA+mask: (255, 0, 0) (240, 240, 240)
RGBA(maskなし): (255, 0, 0)
はみ出し: (255, 0, 0) (240, 240, 240)
戻り値: None
```
`24_paste_mask.png` は 100x60 の薄いグレー背景に、`rgba` の不透明部分(赤い 20x10 の矩形)だけが (10, 10)〜(29, 19) に貼られた画像。

**注意点・落とし穴**:
- 元画像を書き換え、戻り値は `None`。元を残したければ `copy()` してから貼る。
- `mask` は `L`(0-255)、`1`、`RGBA`(アルファが使われる)のいずれか。マスクの大きさは貼る画像と同じにする。値 255 が完全に貼り付け、0 で変化なし、中間は線形にブレンド(上の `128` で `(248, 120, 120)`)。
- `RGBA` 画像を `mask` なしで `RGB` へ貼ると、アルファが無視されて RGB 値がそのまま入る(上の `RGBA(maskなし)` が `(255, 0, 0)` になる)。透明度を活かすには `paste(im, pos, im)` か `Image.alpha_composite` を使う。
- `box` に `(左, 上)` の2要素を渡すと貼る画像のサイズ分、4要素 `(左, 上, 右, 下)` を渡すと領域サイズが貼る画像(または単色の塗り)のサイズと一致する必要がある。

---

### `im.putalpha(...)`

**用途**: 画像にアルファチャンネルを追加・置換する(`RGB`→`RGBA` になる)。

**シグネチャ**: `Image.putalpha(self, alpha)`

**使用例**:
```python
from PIL import Image, ImageDraw

rgba = Image.new("RGBA", (4, 4), (1, 2, 3, 255))
rgba.putalpha(100)                                       # 整数なら一様なアルファ
print(rgba.getpixel((0, 0)))

im = Image.new("RGB", (60, 40), (255, 0, 0))
mask = Image.new("L", im.size, 0)
ImageDraw.Draw(mask).ellipse((5, 5, 54, 34), fill=255)   # 楕円の内側だけ不透明
im.putalpha(mask)                                        # 画像そのものを書き換える
print(im.mode, im.getpixel((30, 20)), im.getpixel((0, 0)))
im.save("25_putalpha.png")
```
実行結果:
```
(1, 2, 3, 100)
RGBA (255, 0, 0, 255) (255, 0, 0, 0)
```
`25_putalpha.png` は 60x40 の RGBA PNG。赤い楕円だけが不透明で、その外側は透明(ビューアの背景色が透けて見える)。

**注意点・落とし穴**:
- その場で書き換え(戻り値 `None`)。`RGB` 画像に対して呼ぶと `RGBA` に変わる。アルファ付きで保存できるのは PNG など(JPEG は不可)。

---

### `Image.blend(...)`

**用途**: 2枚の画像を一定の割合でクロスフェード合成する(`im1 * (1 - alpha) + im2 * alpha`)。

**シグネチャ**: `Image.blend(im1, im2, alpha)`

**使用例**:
```python
from PIL import Image

A = Image.new("RGB", (4, 4), (255, 0, 0))
B = Image.new("RGB", (4, 4), (0, 0, 255))
for a in [0.0, 0.25, 0.5, 1.0, 1.5]:
    print(a, Image.blend(A, B, a).getpixel((0, 0)))
try:
    Image.blend(A, Image.new("RGB", (5, 5)), 0.5)
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
0.0 (255, 0, 0)
0.25 (191, 0, 63)
0.5 (127, 0, 127)
1.0 (0, 0, 255)
1.5 (0, 0, 255)
ValueError: images do not match
```

**注意点・落とし穴**:
- `alpha=0.0` で `im1`、`1.0` で `im2`。範囲外(例 1.5)はエラーにならず外挿され、0-255 にクリップされる。2枚のモード・サイズは一致している必要がある(不一致は `ValueError: images do not match`)。小数部は切り捨てられる(`0.25` で `(191, 0, 63)`)。

---

### `Image.composite(...)`

**用途**: マスク画像に従って、2枚の画像をピクセルごとに切り替えて合成する。

**シグネチャ**: `Image.composite(image1, image2, mask)`

**使用例**:
```python
from PIL import Image

A = Image.new("RGB", (4, 4), (255, 0, 0))
B = Image.new("RGB", (4, 4), (0, 0, 255))
mask = Image.new("L", (4, 4), 0)
mask.paste(255, (0, 0, 2, 4))                       # 左半分だけ 255
c = Image.composite(A, B, mask)                     # mask=255 -> A, 0 -> B
print(c.getpixel((0, 0)), c.getpixel((3, 0)))
```
実行結果:
```
(255, 0, 0) (0, 0, 255)
```

**注意点・落とし穴**:
- `mask` の値が 255 の場所は `image1`、0 の場所は `image2`、中間値はその比率でブレンドされる(`paste` の mask と同じ)。3枚ともサイズが同じである必要がある。新しい画像を返すので元画像は変更されない(`paste` は変更する)。

---

### `Image.alpha_composite(...)`

**用途**: アルファ付き(`RGBA`)の前景を背景の上に正しくアルファ合成する。

**シグネチャ**: `Image.alpha_composite(im1, im2)`

**使用例**:
```python
from PIL import Image

bg = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
fg = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
print(Image.alpha_composite(bg, fg).getpixel((0, 0)))
try:
    Image.alpha_composite(Image.new("RGB", (4, 4)), fg)
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
(255, 127, 127, 255)
ValueError: image has wrong mode
```

**注意点・落とし穴**:
- 両方とも `RGBA` で同じサイズであることが必須(`RGB` を渡すと `ValueError: image has wrong mode`)。`RGBA` 画像を `RGB` の背景に合成して保存したいときは、背景も `convert("RGBA")` してから合成し、最後に `convert("RGB")` する。

---

### `ImageChops`(`difference` / `multiply` / `add` / `lighter` など)

**用途**: 画像同士のピクセル単位の演算(差・乗算・加算・明るい方/暗い方の選択など)。画像の差分検出などに便利。

**シグネチャ**:
- `ImageChops.difference(image1, image2)`
- `ImageChops.multiply(image1, image2)`

**使用例**:
```python
from PIL import Image, ImageChops

A = Image.new("RGB", (4, 4), (255, 0, 0))
B = Image.new("RGB", (4, 4), (0, 0, 255))
print("difference", ImageChops.difference(A, B).getpixel((0, 0)))
print("multiply  ", ImageChops.multiply(A, B).getpixel((0, 0)))
print("add       ", ImageChops.add(A, B).getpixel((0, 0)))
print("lighter   ", ImageChops.lighter(A, B).getpixel((0, 0)))

# 画像の差分検出: 同じなら getbbox() は None
C = A.copy(); C.putpixel((2, 1), (0, 255, 0))
print(ImageChops.difference(A, A.copy()).getbbox())
print(ImageChops.difference(A, C).getbbox())
```
実行結果:
```
difference (255, 0, 255)
multiply   (0, 0, 0)
add        (255, 0, 255)
lighter    (255, 0, 255)
None
(2, 1, 3, 2)
```

**注意点・落とし穴**:
- `difference` の `getbbox()` が `None` なら2枚は完全一致、そうでなければ差のあるピクセルを含む最小の矩形が得られる(`(2, 1, 3, 2)` は (2,1) の1画素)。`add` は 0-255 にクリップされる(`add` の結果は `(255, 0, 255)`)。2枚のモード・サイズは一致させる。

---

## フィルタ(ImageFilter)

### `im.filter(...)` と組み込みフィルタ(`BLUR` / `SHARPEN` / `FIND_EDGES` / `EMBOSS` など)

**用途**: 畳み込み・平滑化などのフィルタを画像全体に適用する。

**シグネチャ**: `Image.filter(self, filter)`

**使用例**:
```python
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

# 左=0, 右=255 の垂直エッジ(10x10)で、y=5 行の各フィルタ応答を見る
step = Image.new("L", (10, 10), 0); step.paste(255, (5, 0, 10, 10))
row = lambda im, y=5: [im.getpixel((x, y)) for x in range(10)]
print("元         ", row(step))
for n in ["BLUR", "SMOOTH", "SMOOTH_MORE", "SHARPEN", "EDGE_ENHANCE", "DETAIL", "FIND_EDGES", "CONTOUR", "EMBOSS"]:
    print(f"{n:12s}", row(step.filter(getattr(ImageFilter, n))))
print("BLUR の y=0 行(縁)", row(step.filter(ImageFilter.BLUR), 0))

# 図で確認
src = Image.new("RGB", (160, 120), (255, 255, 255))
d = ImageDraw.Draw(src)
d.rectangle((15, 15, 80, 70), fill=(220, 40, 40)); d.ellipse((70, 40, 140, 105), fill=(30, 90, 220))
d.line((0, 110, 159, 10), fill=(0, 0, 0), width=2)
cells = [("orig", src), ("BLUR", src.filter(ImageFilter.BLUR)), ("FIND_EDGES", src.filter(ImageFilter.FIND_EDGES)),
         ("EMBOSS", src.filter(ImageFilter.EMBOSS)), ("CONTOUR", src.filter(ImageFilter.CONTOUR))]
cv = Image.new("RGB", (5 * 170 + 10, 145), (200, 200, 200)); f = ImageFont.load_default(11)
for i, (n, im_) in enumerate(cells):
    cv.paste(im_, (10 + i * 170, 10)); ImageDraw.Draw(cv).text((12 + i * 170, 132), n, fill=(0, 0, 0), font=f)
cv.save("31_filters.png")
```
実行結果:
```
元          [0, 0, 0, 0, 0, 255, 255, 255, 255, 255]
BLUR         [0, 0, 0, 80, 112, 143, 175, 255, 255, 255]
SMOOTH       [0, 0, 0, 0, 59, 196, 255, 255, 255, 255]
SMOOTH_MORE  [0, 0, 0, 13, 56, 199, 242, 255, 255, 255]
SHARPEN      [0, 0, 0, 0, 0, 255, 255, 255, 255, 255]
EDGE_ENHANCE [0, 0, 0, 0, 0, 255, 255, 255, 255, 255]
DETAIL       [0, 0, 0, 0, 0, 255, 255, 255, 255, 255]
FIND_EDGES   [0, 0, 0, 0, 0, 255, 0, 0, 0, 255]
CONTOUR      [0, 255, 255, 255, 0, 255, 255, 255, 255, 255]
EMBOSS       [0, 128, 128, 128, 128, 255, 128, 128, 128, 255]
BLUR の y=0 行(縁) [0, 0, 0, 0, 0, 255, 255, 255, 255, 255]
```
`31_filters.png` は元画像(赤い四角・青い円・黒い斜線)に BLUR / FIND_EDGES / EMBOSS / CONTOUR をかけた結果を横に並べた図。BLUR は全体がぼやけ、FIND_EDGES は黒地に輪郭だけが(色付きで、斜線は白い二重線として)残り、EMBOSS は灰色地に輪郭の凹凸だけが浮き出し、CONTOUR は白地に細い輪郭線だけが残る。

**注意点・落とし穴**:
- 組み込みフィルタは `BLUR`, `CONTOUR`, `DETAIL`, `EDGE_ENHANCE`, `EDGE_ENHANCE_MORE`, `EMBOSS`, `FIND_EDGES`, `SHARPEN`, `SMOOTH`, `SMOOTH_MORE`(すべて `ImageFilter.` 直下の定数)。任意の半径のぼかしは `GaussianBlur` / `BoxBlur`、任意の畳み込みは `Kernel` を使う。
- **画像の縁のピクセルはフィルタされずそのままコピーされる**(3x3 カーネルなら外周1ピクセル、5x5 の `BLUR` なら外周2ピクセル。上の `BLUR の y=0 行` が元のまま)。極端に小さい画像ではほとんど変化しない。
- `FIND_EDGES` などは負の値が 0 に切り捨てられるため、エッジの片側(明るい側)にしか応答が出ない(上の `FIND_EDGES` は x=5 だけ 255)。両側を出したいなら `Kernel` の `offset=128` を使う。
- `SHARPEN` / `EDGE_ENHANCE` / `DETAIL` は、0 と 255 だけの理想的な階段エッジでは行き過ぎた分がクリップされるため出力が変わらない(上の3行)。階調のある実画像では効果が出る。
- `P`(パレット)モードには使えず `ValueError: cannot filter palette images`。先に `convert("RGB")` などにする。

---

### `ImageFilter.GaussianBlur(...)` / `ImageFilter.BoxBlur(...)`

**用途**: 半径を指定できるぼかし。`GaussianBlur` はガウシアン、`BoxBlur` は単純平均。

**シグネチャ**:
- `ImageFilter.GaussianBlur(radius=2)`
- `ImageFilter.BoxBlur(radius)`

**使用例**:
```python
from PIL import Image, ImageFilter

step = Image.new("L", (10, 10), 0); step.paste(255, (5, 0, 10, 10))
row = lambda im: [im.getpixel((x, 5)) for x in range(10)]
print("元       ", row(step))
print("Gauss r=1", row(step.filter(ImageFilter.GaussianBlur(1))))
print("Gauss r=2", row(step.filter(ImageFilter.GaussianBlur(2))))
print("Box r=1  ", row(step.filter(ImageFilter.BoxBlur(1))))

rgb = Image.new("RGB", (20, 20), (10, 20, 30)).filter(ImageFilter.GaussianBlur(3))
print(rgb.size, rgb.mode, rgb.getpixel((10, 10)))
try:
    Image.new("I", (4, 4)).filter(ImageFilter.GaussianBlur(1))
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
元        [0, 0, 0, 0, 0, 255, 255, 255, 255, 255]
Gauss r=1 [0, 0, 1, 15, 75, 179, 240, 254, 255, 255]
Gauss r=2 [2, 10, 28, 59, 103, 152, 196, 227, 245, 253]
Box r=1   [0, 0, 0, 0, 85, 170, 255, 255, 255, 255]
(20, 20) RGB (10, 20, 30)
ValueError: image has wrong mode
```

**注意点・落とし穴**:
- `radius` は標準偏差(`GaussianBlur`)/ 窓の半径(`BoxBlur`)で、大きいほど強くぼける。`GaussianBlur` は画像の縁も処理される(組み込みの 3x3/5x5 フィルタと違い、上の r=2 では端 `x=0` も変化している)。
- `I` / `F` モードには使えない(`ValueError: image has wrong mode`)。`L`・`RGB`・`RGBA` で使う。
- `GaussianBlur` は元画像の `mode`/`size` を保ったまま返す。`radius` にタプル `(横, 縦)` を渡すと方向別にぼかせる(例: `(2, 0)` は横方向だけ)。

---

### `ImageFilter.UnsharpMask(...)`

**用途**: アンシャープマスクによる輪郭強調(シャープ化)。`SHARPEN` より強さを細かく調整できる。

**シグネチャ**: `ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)`

**使用例**:
```python
from PIL import Image, ImageDraw, ImageFilter

step = Image.new("L", (10, 10), 0); step.paste(255, (5, 0, 10, 10))
soft = step.filter(ImageFilter.GaussianBlur(1.5))
sharp = soft.filter(ImageFilter.UnsharpMask(radius=2, percent=300, threshold=0))
row = lambda im: [im.getpixel((x, 5)) for x in range(10)]
print("ぼかし後       ", row(soft))
print("UnsharpMask後  ", row(sharp))

img = Image.new("RGB", (120, 60), (255, 255, 255))
ImageDraw.Draw(img).rectangle((10, 10, 50, 50), fill=(220, 40, 40))
ImageDraw.Draw(img).ellipse((65, 10, 110, 50), fill=(30, 90, 220))
blur = img.filter(ImageFilter.GaussianBlur(2.5))
un = blur.filter(ImageFilter.UnsharpMask(radius=3, percent=300, threshold=0))
cv = Image.new("RGB", (3 * 130 + 10, 70), (200, 200, 200))
for i, im_ in enumerate([img, blur, un]): cv.paste(im_, (10 + i * 130, 5))
cv.save("33_unsharp.png")
```
実行結果:
```
ぼかし後        [0, 1, 12, 41, 96, 159, 214, 243, 254, 255]
UnsharpMask後   [0, 0, 0, 0, 60, 195, 255, 255, 255, 255]
```
`33_unsharp.png` は左から元画像(赤い四角と青い円)・ぼかした画像・ぼかしにアンシャープマスクを適用した画像。3枚目は輪郭がくっきり戻る一方、図形の縁に濃い縁取り(オーバーシュート)が出て不自然に見える。

**注意点・落とし穴**:
- `radius` = ぼかし半径、`percent` = 強調の強さ(%)、`threshold` = この値未満の差は強調しない(ノイズ抑制)。強くしすぎると縁に白/黒のハロー(輪)が出る。ぼけた画像を「復元」できるわけではなく、輪郭のコントラストを上げるだけ。

---

### `ImageFilter.Kernel(...)`

**用途**: 3x3 または 5x5 の任意の畳み込みカーネルを定義して `filter` に渡す。

**シグネチャ**: `ImageFilter.Kernel(size, kernel, scale=None, offset=0)`

**使用例**:
```python
from PIL import Image, ImageFilter

step = Image.new("L", (10, 10), 0); step.paste(255, (5, 0, 10, 10))
row = lambda im: [im.getpixel((x, 5)) for x in range(10)]

box = ImageFilter.Kernel((3, 3), [1] * 9)                 # scale 省略 = 係数の総和(9)
print("平均(3x3)    ", row(step.filter(box)), box.filterargs[1])
print("scale=1      ", row(step.filter(ImageFilter.Kernel((3, 3), [1] * 9, scale=1))))

sharpen = ImageFilter.Kernel((3, 3), [0, -1, 0, -1, 5, -1, 0, -1, 0], scale=1)
print("シャープ     ", row(step.filter(sharpen)))

sobel_x = ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1, offset=128)
print("Sobel(offset=128)", row(step.filter(sobel_x)))

try:
    ImageFilter.Kernel((3, 3), [1] * 8)
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
平均(3x3)     [0, 0, 0, 0, 85, 170, 255, 255, 255, 255] 9
scale=1       [0, 0, 0, 0, 255, 255, 255, 255, 255, 255]
シャープ      [0, 0, 0, 0, 0, 255, 255, 255, 255, 255]
Sobel(offset=128) [0, 128, 128, 128, 255, 255, 128, 128, 128, 255]
ValueError: not enough coefficients in kernel
```

**注意点・落とし穴**:
- `kernel` は行優先の平坦なリスト(3x3 なら9要素、5x5 なら25要素)。要素数が足りないと `ValueError: not enough coefficients in kernel`。
- `scale` を省略すると係数の総和(0 の場合は 1)で割られる。明るさを変えたくない平滑化は総和=1 になるよう正規化されるが、`scale=1` にすると 255 を超えて飽和する(上の `scale=1` は白飛び)。
- `offset` は結果に足す値。エッジ検出で負の応答も見たいときに `offset=128` を足して灰色を中心にする。3x3/5x5 以外のサイズは `filter` 時に `ValueError: bad kernel size` になり、縁のピクセルはフィルタされない。

---

### `ImageFilter.MedianFilter` / `RankFilter` / `MinFilter` / `MaxFilter` / `ModeFilter`

**用途**: 順序統計フィルタ。`MedianFilter` は塩コショウノイズ除去、`MinFilter`/`MaxFilter` は収縮・膨張(モルフォロジー)に使える。

**シグネチャ**:
- `ImageFilter.MedianFilter(size=3)`
- `ImageFilter.RankFilter(size, rank)`
- `ImageFilter.ModeFilter(size=3)`

**使用例**:
```python
from PIL import Image, ImageFilter

sp = Image.new("L", (7, 7), 100)
sp.putpixel((3, 3), 255)                                 # 1点だけの「塩」ノイズ
print("元          ", sp.getpixel((3, 3)))
print("Median(3)   ", sp.filter(ImageFilter.MedianFilter(3)).getpixel((3, 3)))
print("Min(3)      ", sp.filter(ImageFilter.MinFilter(3)).getpixel((3, 3)))
print("Max(3) 近傍 ", sp.filter(ImageFilter.MaxFilter(3)).getpixel((2, 2)))
print("Rank(3,0)   ", sp.filter(ImageFilter.RankFilter(3, 0)).getpixel((3, 3)))
```
実行結果:
```
元           255
Median(3)    100
Min(3)       100
Max(3) 近傍  255
Rank(3,0)    100
```

**注意点・落とし穴**:
- `size` は窓の一辺(奇数)。`MedianFilter(3)` は 3x3 の中央値、`RankFilter(size, rank)` は窓内を小さい順に並べて `rank` 番目の値(`rank=0` が最小、`MinFilter` と同じ、`rank=size*size//2` が中央値)、`ModeFilter(size)` は窓内の最頻値。窓が大きいほど遅い。

---

## 画像強調(ImageEnhance / ImageOps)

### `ImageEnhance.Brightness` / `Contrast` / `Color` / `Sharpness`

**用途**: 明るさ・コントラスト・彩度・シャープさを係数で調整する。`enhance(factor)` の `1.0` が元画像、`0.0` が最小。

**シグネチャ**: `ImageEnhance.Brightness(image)`

**使用例**:
```python
from PIL import Image, ImageEnhance

base = Image.new("RGB", (4, 1))
for x, v in enumerate([(100, 50, 50), (120, 120, 120), (140, 60, 200), (160, 160, 60)]):
    base.putpixel((x, 0), v)
px = lambda im: [im.getpixel((x, 0)) for x in range(im.width)]
print("元", px(base))
for f in (0.0, 0.5, 1.5):
    print("Brightness", f, px(ImageEnhance.Brightness(base).enhance(f)))
for f in (0.0, 0.5, 2.0):
    print("Contrast  ", f, px(ImageEnhance.Contrast(base).enhance(f)))
for f in (0.0, 0.5, 2.0):
    print("Color     ", f, px(ImageEnhance.Color(base).enhance(f)))
print(ImageEnhance.Sharpness(base).enhance(2.0).size)
```
実行結果:
```
元 [(100, 50, 50), (120, 120, 120), (140, 60, 200), (160, 160, 60)]
Brightness 0.0 [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
Brightness 0.5 [(50, 25, 25), (60, 60, 60), (70, 30, 100), (80, 80, 30)]
Brightness 1.5 [(150, 75, 75), (180, 180, 180), (210, 90, 255), (240, 240, 90)]
Contrast   0.0 [(109, 109, 109), (109, 109, 109), (109, 109, 109), (109, 109, 109)]
Contrast   0.5 [(104, 79, 79), (114, 114, 114), (124, 84, 154), (134, 134, 84)]
Contrast   2.0 [(91, 0, 0), (131, 131, 131), (171, 11, 255), (211, 211, 11)]
Color      0.0 [(65, 65, 65), (120, 120, 120), (100, 100, 100), (149, 149, 149)]
Color      0.5 [(82, 57, 57), (120, 120, 120), (120, 80, 150), (154, 154, 104)]
Color      2.0 [(135, 35, 35), (120, 120, 120), (180, 20, 255), (171, 171, 0)]
(4, 1)
```

**注意点・落とし穴**:
- 使い方は `ImageEnhance.Brightness(im).enhance(factor)` の2段階(クラスの生成→`enhance`)。`Contrast` / `Color` / `Sharpness` も同じ形で、それぞれ 0.0=平均グレー一色 / 白黒 / ぼかし、1.0=元、1 より大で強調。
- `Brightness` は全画素に係数を掛けるだけ(`0.0` で真っ黒、`1.5` で 140→210、200→255 とクリップされる)。`Contrast` は画像の平均輝度(この例では 109)を中心に広げる/縮める。`Color` は 0.0 でグレースケール、1 より大で彩度アップ。
- コントラストの強調はクリップで白飛び・黒つぶれが起きやすい(`Contrast 2.0` で R=0 になる画素がある)。自動で調整したいなら `ImageOps.autocontrast` が向く。

---

### `ImageOps.autocontrast(...)`

**用途**: ヒストグラムを引き伸ばして、最暗を 0・最明を 255 に合わせる(コントラスト自動調整)。

**シグネチャ**: `ImageOps.autocontrast(image, cutoff=0, ignore=None, mask=None, preserve_tone=False)`

**使用例**:
```python
from PIL import Image, ImageOps

low = Image.new("L", (4, 1))
for x, v in enumerate([100, 120, 140, 160]):
    low.putpixel((x, 0), v)
print("L:", [ImageOps.autocontrast(low).getpixel((x, 0)) for x in range(4)])

base = Image.new("RGB", (4, 1))
for x, v in enumerate([(100, 50, 50), (120, 120, 120), (140, 60, 200), (160, 160, 60)]):
    base.putpixel((x, 0), v)
px = lambda im: [im.getpixel((x, 0)) for x in range(im.width)]
print("RGB(チャンネル別)     ", px(ImageOps.autocontrast(base)))
print("RGB(preserve_tone)    ", px(ImageOps.autocontrast(base, preserve_tone=True)))

# cutoff: 両端の外れ値 N% を無視して引き伸ばす
g = Image.linear_gradient("L").resize((256, 4)).point(lambda v: 100 + v // 4)
print(g.getextrema(), "->", ImageOps.autocontrast(g).getextrema())
```
実行結果:
```
L: [0, 85, 170, 255]
RGB(チャンネル別)      [(0, 0, 0), (85, 162, 119), (170, 23, 255), (255, 255, 17)]
RGB(preserve_tone)     [(106, 0, 0), (166, 166, 166), (227, 0, 255), (255, 255, 0)]
(108, 155) -> (0, 255)
```

**注意点・落とし穴**:
- 既定ではチャンネルごとに独立に引き伸ばすので、`RGB` では色味が変わる(上の `RGB(チャンネル別)` は赤み・青みが強調される)。色味を保ちたいときは `preserve_tone=True`(輝度のヒストグラムで伸ばす)。
- 外れ値の影響を避けたいときは `cutoff`(%、`(下側, 上側)` のタプルも可)を指定して両端の極端な画素を無視する。背景の黒などを除外したいときは `ignore`(除外する値)を使う(例: 値 0 を含む `L` 画像で `ignore=0` にすると、0 を除いた最小値が 0 に引き伸ばされる)。

---

### `ImageOps.equalize(...)`

**用途**: ヒストグラム平坦化。輝度の分布を均一化して、暗い画像などの階調を広げる。

**シグネチャ**: `ImageOps.equalize(image, mask=None)`

**使用例**:
```python
import numpy as np
from PIL import Image, ImageOps, ImageStat

rng = np.random.default_rng(1)
dark = Image.fromarray(rng.integers(10, 60, (64, 64)).astype("uint8"))
eq = ImageOps.equalize(dark)
print("extrema:", dark.getextrema(), "->", eq.getextrema())
print("mean   :", round(ImageStat.Stat(dark).mean[0], 1), "->", round(ImageStat.Stat(eq).mean[0], 1))
```
実行結果:
```
extrema: (10, 59) -> (0, 255)
mean   : 34.7 -> 133.4
```

**注意点・落とし穴**:
- `autocontrast` が単純な線形の引き伸ばしなのに対し、`equalize` は非線形で、ヒストグラムを平坦化する(一様に分布させる)。ノイズ成分も強調されやすい。`RGB` ではチャンネル別に平坦化するので色がずれることがある(輝度だけ平坦化したい場合は `L` に変換して使う)。

---

### `ImageOps.invert` / `grayscale` / `posterize` / `solarize` / `mirror` / `flip`

**用途**: 階調反転・グレースケール化・階調数削減・ソラリゼーション・左右/上下反転。

**シグネチャ**:
- `ImageOps.invert(image)`
- `ImageOps.grayscale(image)`
- `ImageOps.posterize(image, bits)`
- `ImageOps.solarize(image, threshold=128)`

**使用例**:
```python
from PIL import Image, ImageOps

base = Image.new("RGB", (4, 1))
for x, v in enumerate([(100, 50, 50), (120, 120, 120), (140, 60, 200), (160, 160, 60)]):
    base.putpixel((x, 0), v)
px = lambda im: [im.getpixel((x, 0)) for x in range(im.width)]
print("invert   ", px(ImageOps.invert(base)))
g = ImageOps.grayscale(base)
print("grayscale", g.mode, px(g))
print("posterize", px(ImageOps.posterize(base, 2)))
print("solarize ", px(ImageOps.solarize(base, threshold=130)))
print("mirror   ", px(ImageOps.mirror(base)))
try:
    ImageOps.invert(Image.new("RGBA", (1, 1)))
except OSError as e:
    print("OSError:", e)
```
実行結果:
```
invert    [(155, 205, 205), (135, 135, 135), (115, 195, 55), (95, 95, 195)]
grayscale L [65, 120, 100, 149]
posterize [(64, 0, 0), (64, 64, 64), (128, 0, 192), (128, 128, 0)]
solarize  [(100, 50, 50), (120, 120, 120), (115, 60, 55), (95, 95, 60)]
mirror    [(160, 160, 60), (140, 60, 200), (120, 120, 120), (100, 50, 50)]
OSError: not supported for mode RGBA
```

**注意点・落とし穴**:
- `invert` は `RGBA` には使えず `OSError: not supported for mode RGBA`。`RGB` と `L` に対して使う(アルファも反転したくないなら `split` して RGB だけ反転して `merge`)。
- `posterize(image, bits)` は各チャンネルの上位 `bits` ビットだけ残す(`bits=2` なら 4 段階: 0/64/128/192)。`solarize` は閾値以上の値だけ反転する。`grayscale` は `convert("L")` と同等の `L` 画像を返す。
- `mirror` は左右反転、`flip` は上下反転(`transpose(FLIP_LEFT_RIGHT/FLIP_TOP_BOTTOM)` と同じ)。

---

### `ImageOps.colorize(...)`

**用途**: グレースケール画像を、黒に対応する色・白に対応する色(+中間色)へ線形に着色する(疑似カラー化)。

**シグネチャ**: `ImageOps.colorize(image, black, white, mid=None, blackpoint=0, whitepoint=255, midpoint=127)`

**使用例**:
```python
from PIL import Image, ImageOps

gray = Image.new("L", (4, 1))
for x, v in enumerate([0, 85, 170, 255]):
    gray.putpixel((x, 0), v)
px = lambda im: [im.getpixel((x, 0)) for x in range(im.width)]
print(px(ImageOps.colorize(gray, black="navy", white="yellow")))
print(px(ImageOps.colorize(gray, black="black", white="white", mid="red")))

grad = Image.linear_gradient("L")               # 256x256, 上=黒 下=白
ImageOps.colorize(grad, black="navy", white="yellow", mid="red").save("39_colorize.png")
```
実行結果:
```
[(0, 0, 128), (85, 85, 85), (170, 170, 42), (255, 255, 0)]
[(0, 0, 0), (170, 0, 0), (255, 85, 85), (255, 255, 255)]
```
`39_colorize.png`(256x256)は、上から下へ濃い青(navy)→赤→黄色へ連続的に変化するグラデーション。

**注意点・落とし穴**:
- 入力は `L` 画像が前提(`RGB` は先に `grayscale`/`convert("L")` する)。色は色名文字列でも `(R, G, B)` でも指定できる。`mid` を渡すと黒→中間→白の3点を補間する(`blackpoint`/`midpoint`/`whitepoint` で対応する入力値も変えられる)。

---

### `ImageOps.exif_transpose(...)`

**用途**: EXIF の Orientation タグに従って画像を回転・反転し、タグを削除して「見た目通りの向き」の画像にする。

**シグネチャ**: `ImageOps.exif_transpose(image, *, in_place=False)`

**使用例**:
```python
from PIL import Image, ImageOps

im = Image.new("RGB", (30, 20), (255, 0, 0))
ex = im.getexif()
ex[0x0112] = 6                       # Orientation=6: 90度回転して表示すべき画像
im.save("40_exif.jpg", exif=ex)

o = Image.open("40_exif.jpg")
print(o.size, o.getexif().get(0x0112))
t = ImageOps.exif_transpose(o)
print(t.size, t.getexif().get(0x0112))

o2 = Image.open("40_exif.jpg")
print(ImageOps.exif_transpose(o2, in_place=True), o2.size)   # in_place なら戻り値は None
```
実行結果:
```
(30, 20) 6
(20, 30) None
None (20, 30)
```

**注意点・落とし穴**:
- スマートフォンで撮った写真は、ピクセルデータは横向きのままで Orientation タグだけで向きを表していることが多い。`Image.open` は自動では回転しないので、そのまま処理・表示すると横倒しになる。読み込み直後に `exif_transpose` を通すのが定番。Orientation が無い/1 の画像はそのまま(コピー)返る。

---

## 描画(ImageDraw / ImageFont)

### `ImageDraw.Draw(...)`

**用途**: 画像に図形・文字を描くための描画オブジェクトを作る。以降の `line`/`rectangle`/`text` などはこのオブジェクトのメソッドで、元の画像を直接書き換える。

**シグネチャ**: `ImageDraw.Draw(im, mode=None)`

**使用例**:
```python
from PIL import Image, ImageDraw

im = Image.new("RGB", (100, 60), "white")
d = ImageDraw.Draw(im)
d.line((0, 0, 99, 59), fill="red", width=3)
print(im.getpixel((50, 30)), im.getpixel((99, 0)))   # im が直接書き換わる

# RGB 画像へ半透明で描く: mode="RGBA" を指定すると下地とブレンドされる
im2 = Image.new("RGB", (2, 1), (255, 255, 255))
ImageDraw.Draw(im2, "RGBA").point((0, 0), fill=(255, 0, 0, 64))
print(im2.getpixel((0, 0)))
im3 = Image.new("RGB", (2, 1), (255, 255, 255))
ImageDraw.Draw(im3).point((0, 0), fill=(255, 0, 0, 64))   # 通常の Draw はアルファを無視
print(im3.getpixel((0, 0)))
# RGBA 画像へ描くとアルファ値ごと上書きされる(ブレンドされない)
im4 = Image.new("RGBA", (2, 1), (255, 255, 255, 255))
ImageDraw.Draw(im4, "RGBA").point((0, 0), fill=(255, 0, 0, 64))
print(im4.getpixel((0, 0)))
```
実行結果:
```
(255, 0, 0) (255, 255, 255)
(255, 191, 191)
(255, 0, 0)
(255, 0, 0, 64)
```

**注意点・落とし穴**:
- 図形は `Draw` オブジェクトに対するメソッドで描き、結果は元の `Image` に反映される(`im` 自体が更新され、戻り値は `None`)。
- 半透明の図形を下地と混ぜたいなら **`RGB` 画像に `Draw(im, "RGBA")`** を使う(上の `(255, 191, 191)`)。`RGBA` 画像に描くとアルファも含めて**上書き**されるだけで合成されない(`(255, 0, 0, 64)`)。透明な下地に半透明で重ねたいときは、別の `RGBA` レイヤに描いて `Image.alpha_composite` で重ねる。

---

### `d.line` / `d.rectangle` / `d.ellipse` / `d.polygon` / `d.arc` / `d.pieslice` / `d.point` など

**用途**: 基本図形の描画。`fill` が塗り色、`outline` が枠線の色、`width` が線幅。

**シグネチャ**:
- `ImageDraw.line(self, xy, fill=None, width=1, joint=None)`
- `ImageDraw.rectangle(self, xy, fill=None, outline=None, width=1)`
- `ImageDraw.rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1, *, corners=None)`
- `ImageDraw.ellipse(self, xy, fill=None, outline=None, width=1)`
- `ImageDraw.polygon(self, xy, fill=None, outline=None, width=1)`
- `ImageDraw.arc(self, xy, start, end, fill=None, width=1)`
- `ImageDraw.pieslice(self, xy, start, end, fill=None, outline=None, width=1)`

**使用例**:
```python
from PIL import Image, ImageDraw

im = Image.new("RGB", (200, 120), "white")
d = ImageDraw.Draw(im)
d.line((10, 10, 190, 10), fill=(255, 0, 0), width=3)
d.line([(10, 20), (60, 50), (110, 20), (160, 50)], fill="blue", width=4, joint="curve")
d.rectangle((10, 60, 60, 100), outline="black", width=2, fill=(200, 255, 200))
d.rounded_rectangle((70, 60, 120, 100), radius=10, outline="purple", fill="lavender")
d.ellipse((130, 60, 190, 110), fill="orange", outline="black")
d.polygon([(170, 20), (190, 40), (150, 40)], fill="green")
d.arc((10, 60, 60, 100), start=0, end=90, fill="red", width=2)
d.pieslice((130, 60, 190, 110), start=-90, end=0, fill="red")
d.regular_polygon((100, 110, 8), n_sides=6, fill="teal")
im.save("43_shapes.png")
px = im.getpixel
print(px((50, 10)), px((35, 80)), px((160, 100)), px((180, 35)))

# 座標の意味: rectangle / ellipse の (x0, y0, x1, y1) は右下端も「含む」
t = Image.new("L", (10, 10), 0)
ImageDraw.Draw(t).rectangle((2, 2, 5, 5), fill=255)
print(t.getbbox(), t.getcolors())
t = Image.new("L", (10, 10), 0)
ImageDraw.Draw(t).ellipse((2, 2, 5, 5), fill=255)
print(t.getbbox())

# floodfill: 境界で囲まれた領域を塗る
ff = Image.new("RGB", (10, 10), "white")
ImageDraw.Draw(ff).rectangle((3, 3, 6, 6), outline="black")
ImageDraw.floodfill(ff, (4, 4), (255, 0, 0))
print(ff.getpixel((4, 4)), ff.getpixel((0, 0)), ff.getpixel((3, 3)))
```
実行結果:
```
(255, 0, 0) (200, 255, 200) (255, 165, 0) (0, 128, 0)
(2, 2, 6, 6) [(84, 0), (16, 255)]
(2, 2, 6, 6)
(255, 0, 0) (255, 255, 255) (0, 0, 0)
```
`43_shapes.png` は 200x120 の白背景。上端に赤い太線、その下に青いジグザグの折れ線(4点を結ぶ)、右上に緑の三角形。下段には左から薄緑の四角形(黒枠と、右下の角に赤い弧)、薄紫の角丸四角形(紫の枠)、オレンジの楕円(右上の 1/4 が赤い扇形)、そして中央下にティール色の六角形が描かれる。

**注意点・落とし穴**:
- `rectangle` / `ellipse` の座標 `(x0, y0, x1, y1)` は**右下端の座標も含めて**塗られる(上の `(2, 2, 5, 5)` は 4x4 = 16 画素で、`getbbox` は `(2, 2, 6, 6)`)。`crop` の `box` の感覚(右・下は含まない)とは異なる。
- `fill` を省略して `outline` だけ指定すると枠線だけが描かれ、内部はそのまま。`outline` を省略して `fill` だけなら枠なしで塗りつぶされる。両方省略すると既定色(`RGB` では白)の枠線になる。`width` を太くすると、枠は矩形の内側に向けて太くなる。
- `arc` / `pieslice` の角度は度で、3時方向が 0 度、時計回りに増える(上の `pieslice` の -90〜0 は「12時〜3時」の右上 1/4)。
- `line` の `joint="curve"` を付けると折れ線の継ぎ目が丸くなる(太い線でのつなぎ目のギザギザ防止)。

---

### `ImageFont.truetype(...)` / `ImageFont.load_default(...)`

**用途**: 文字描画用のフォントを読み込む。`truetype` は TTF/OTF ファイル、`load_default` は Pillow 同梱フォント。

**シグネチャ**:
- `ImageFont.truetype(font, size=10, index=0, encoding='', layout_engine=None)`
- `ImageFont.load_default(size=None)`

**使用例**:
```python
import os
from PIL import ImageFont

f0 = ImageFont.load_default()          # 既定サイズ(10)
f1 = ImageFont.load_default(24)        # size 指定(FreeType が使える環境のみ)
print(type(f0).__name__, type(f1).__name__, f1.size)
print(f1.getbbox("Hello"), f1.getlength("Hello"))

ttf = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ft = ImageFont.truetype(ttf, 32)
print(ft.getname(), ft.size, ft.getbbox("Hello"), ft.getlength("Hello"))

try:
    ImageFont.truetype("nonexistent.ttf", 20)
except OSError as e:
    print("OSError:", e)

ja = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"     # 日本語フォント(環境依存)
print(os.path.exists(ja))
print("DejaVu の '日本' マスク:", ft.getmask("日本").getbbox())
```
実行結果:
```
FreeTypeFont FreeTypeFont 24
(0, 7, 57, 24) 57.0
('DejaVu Sans', 'Book') 32 (0, 6, 81, 30) 81.109375
OSError: cannot open resource
True
DejaVu の '日本' マスク: (1, 0, 37, 28)
```

**注意点・落とし穴**:
- `truetype` にはフォントファイルのパスが必要で、置き場所は OS ごとに違う(この Linux 環境では `/usr/share/fonts/...` 配下)。見つからないと `OSError: cannot open resource`。
- `load_default(size)` で大きさを指定できる。この環境(FreeType 有効)では `size` を省略しても `FreeTypeFont`(サイズ 10)が返る。FreeType が無効なビルドでの挙動は、この環境では確認できていない。
- **日本語などの CJK 文字は、フォント側にグリフが無いと豆腐(□)や空白になる**。日本語を描くには IPAGothic / Noto Sans CJK JP など日本語対応フォントを `truetype` で指定する。`ft.getmask("日本")` が `(1, 0, 37, 28)` のような小さな箱になるのは DejaVu にグリフが無いため。
- 旧来の `font.getsize()` / `draw.textsize()` は削除済み。文字のサイズは `getbbox` / `getlength`(次項の `textbbox` / `textlength`)で取得する。

---

### `d.text(...)` / `d.textbbox(...)` / `d.textlength(...)`

**用途**: 文字列を描画する(`text`)/ 描画したときの範囲(`textbbox`)や幅(`textlength`)を測る。

**シグネチャ**:
- `ImageDraw.text(self, xy, text, fill=None, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, stroke_fill=None, embedded_color=False, *args, **kwargs)`
- `ImageDraw.textbbox(self, xy, text, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, embedded_color=False, *, font_size=None)`
- `ImageDraw.textlength(self, text, font=None, direction=None, features=None, language=None, embedded_color=False, *, font_size=None)`

**使用例**:
```python
import os
from PIL import Image, ImageDraw, ImageFont

ttf = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ja = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
font = ImageFont.truetype(ttf, 32)

im = Image.new("RGB", (260, 130), "white")
d = ImageDraw.Draw(im)
d.text((10, 10), "Hello", fill="black", font=font)
print(d.textbbox((10, 10), "Hello", font=font), d.textlength("Hello", font=font))

# anchor: 描画位置(xy)が文字のどこに対応するか(la=左上, mm=中央, rs=右・ベースライン)
for a in ["la", "mm", "rs"]:
    print(a, d.textbbox((100, 40), "Hello", font=font, anchor=a))

# 縁取り(stroke)と複数行
d.text((10, 60), "Edge", fill="white", stroke_width=2, stroke_fill="black", font=font)
d.multiline_text((130, 55), "Line1\nLine2", fill="navy", font=ImageFont.truetype(ttf, 20), spacing=6)
if os.path.exists(ja):
    d.text((10, 100), "日本語テキスト", fill="crimson", font=ImageFont.truetype(ja, 24))
im.save("45_text.png")
```
実行結果:
```
(10, 16, 91, 40) 81.109375
la (100, 46, 181, 70)
mm (59, 27, 140, 51)
rs (19, 16, 100, 40)
```
`45_text.png` は 260x130 の白背景。上段に黒い大きな "Hello"、中段左に白文字に黒い縁取りの "Edge"、その右に紺色の2行 "Line1" / "Line2"、下段に赤系の日本語 "日本語テキスト" が表示される(日本語フォントが見つかった場合)。

**注意点・落とし穴**:
- `xy` が指すのは既定(`anchor="la"`)では文字の**左上(left-ascender)**。`"mm"` にすると中央揃え、`"ms"` なら中央+ベースライン基準。中央寄せしたいだけなら `anchor="mm"` が簡単(上の `mm` で bbox が `(100, 40)` を中心に広がる)。
- `textbbox` は描画範囲の見積もりで、グリフの左右の余白(bearing)を含むため、実際にインクがある範囲(`getbbox()`)とは数ピクセルずれることがある。
- `fill` は色。`RGB` 画像に描くなら `(R, G, B)`、`L` 画像なら整数。`font` を省略すると同梱の既定フォント(小さい)が使われるので、大きさを制御したいときは必ず `font=` を渡す。
- `text` の `\n` は複数行だが、行間や揃え(`align`)を細かく調整するなら `multiline_text` を使う。

---

## numpy連携

### `np.asarray(im)` / `np.array(im)`

**用途**: 画像を numpy 配列に変換する。配列の形状は `(高さ, 幅, チャンネル)`。

**シグネチャ**: `np.asarray(Image)` / `np.array(Image)`(Pillow の `__array_interface__` 経由)

**使用例**:
```python
import numpy as np
from PIL import Image

im = Image.new("RGB", (4, 3), (10, 20, 30))
im.putpixel((3, 0), (200, 100, 50))

a = np.asarray(im)
print(a.shape, a.dtype, a.flags.writeable, a[0, 3], a[0, 0])   # a[y, x]
try:
    a[0, 0] = 1
except ValueError as e:
    print("ValueError:", e)

b = np.array(im)                    # コピーなので書き込み可
b[0, 0] = (1, 2, 3)
print(b.flags.writeable, im.getpixel((0, 0)))   # 元画像は変わらない

print(np.asarray(im.convert("L")).shape, np.asarray(im.convert("RGBA")).shape)
print(np.asarray(Image.new("1", (4, 3), 1)).dtype, np.asarray(Image.new("I", (4, 3), 70000)).dtype, np.asarray(Image.new("F", (4, 3), 1.5)).dtype)
print(np.asarray(Image.new("P", (4, 3))).dtype)
```
実行結果:
```
(3, 4, 3) uint8 False [200 100  50] [10 20 30]
ValueError: assignment destination is read-only
True (10, 20, 30)
(3, 4) (3, 4, 4)
bool int32 float32
uint8
```

**注意点・落とし穴**:
- 配列は `(height, width, channels)`(`L` は `(height, width)`)。`Image.size` の `(width, height)` と逆順。ピクセル `(x, y)` は `a[y, x]`。
- `np.asarray(im)` は書き込み不可(`writeable=False`)。加工するなら `np.array(im)`(コピー)か `np.asarray(im).copy()` を使う。
- モード別の dtype: `RGB`/`L`/`RGBA`/`P` は `uint8`、`1` は `bool`、`I` は `int32`、`F` は `float32`。`P` は色ではなくパレットのインデックス配列になるので、色として扱うなら先に `convert("RGB")` する。

---

### `Image.fromarray(...)`

**用途**: numpy 配列から画像を作る。dtype と形状からモードが推定される。

**シグネチャ**: `Image.fromarray(obj, mode=None)`

**使用例**:
```python
import numpy as np
from PIL import Image

arr = np.zeros((3, 4, 3), dtype=np.uint8)     # (高さ3, 幅4, RGB)
arr[..., 0] = 255
f = Image.fromarray(arr)
print(f.size, f.mode, f.getpixel((0, 0)))     # size は (幅, 高さ)

print(Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).mode,
      Image.fromarray(np.zeros((3, 4, 4), dtype=np.uint8)).mode,
      Image.fromarray(np.zeros((3, 4), dtype=np.uint16)).mode,
      Image.fromarray(np.zeros((3, 4), dtype=np.int32)).mode,
      Image.fromarray(np.zeros((3, 4), dtype=np.float32)).mode,
      Image.fromarray(np.zeros((3, 4), dtype=bool)).mode)

# サポート外の dtype / 形状
for shape, dt in [((3, 4, 3), np.float64), ((3, 4), np.int64)]:
    try:
        Image.fromarray(np.zeros(shape, dtype=dt))
    except TypeError as e:
        print("TypeError:", e)

# float [0,1] の画像は uint8 に変換してから渡す
fl = np.random.default_rng(0).random((3, 4, 3))
u = Image.fromarray((fl * 255).round().astype(np.uint8))
print(u.mode, u.getpixel((0, 0)), (fl[0, 0] * 255).round())

# 明示的なモード指定
print(Image.fromarray(np.array([[0, 255], [128, 64]], dtype=np.uint8), mode="L").getpixel((1, 0)))
```
実行結果:
```
(4, 3) RGB (255, 0, 0)
L RGBA I;16 I F 1
TypeError: Cannot handle this data type: (1, 1, 3), <f8
TypeError: Cannot handle this data type: (1, 1), <i8
RGB (162, 69, 10) [162.  69.  10.]
255
```

**注意点・落とし穴**:
- `uint8` の `(H, W)` は `L`、`(H, W, 3)` は `RGB`、`(H, W, 4)` は `RGBA` になる。`uint16` は `I;16`、`int32` は `I`、`float32` は `F`、`bool` は `1` になる。
- **`float64` や `int64` の配列は `TypeError: Cannot handle this data type`**。`np.random.random()` や `astype(float)` の結果をそのまま渡しがち。0-1 の float は `(x * 255).astype(np.uint8)` へ変換してから渡す。
- 0-255 の範囲を超える値を `astype(np.uint8)` するとオーバーフローで折り返す(例: 300 → 44、-5 → 251)。必ず `np.clip(x, 0, 255)` してから変換する。
- `fromarray` は `(height, width)` の順を前提にするので、`(width, height)` 順で持つ配列は `transpose` してから渡す。

---

### numpy 連携でよくある落とし穴(軸順・BGR・非連続配列)

**用途**: OpenCV など他ライブラリとの往復や、配列を加工して画像に戻すときの注意をまとめて確認する。

**シグネチャ**: (単独の関数ではなく `np.asarray` / `Image.fromarray` の使い方に関する補足)

**使用例**:
```python
import numpy as np
from PIL import Image

im = Image.new("RGB", (4, 3), (10, 20, 30)); im.putpixel((3, 0), (200, 100, 50))
w, h = im.size
a = np.asarray(im)
print("im.size =", (w, h), " array.shape =", a.shape)

# チャンネル別の加工(R を反転)して画像へ戻す
b = np.array(im); b[..., 0] = 255 - b[..., 0]
print(Image.fromarray(b).getpixel((0, 0)))

# 左右反転はスライスで
print(Image.fromarray(b[:, ::-1]).getpixel((0, 0)))

# RGB <-> BGR (OpenCV は BGR): 最後の軸を反転
bgr = np.asarray(im)[..., ::-1]
print("BGR 化の C_CONTIGUOUS:", bgr.flags["C_CONTIGUOUS"])
print("BGR のまま Image 化: ", Image.fromarray(bgr).getpixel((3, 0)))   # R と B が入れ替わる
print("元の RGB:            ", im.getpixel((3, 0)))

# uint8 のまま足し算すると 255 を超えた分は折り返す
print(a[0, 3], (a + 100)[0, 3])
# 平均・統計は float に変換してから
print(a.astype(np.float32).mean(axis=(0, 1)))
```
実行結果:
```
im.size = (4, 3)  array.shape = (3, 4, 3)
(245, 20, 30)
(55, 100, 50)
BGR 化の C_CONTIGUOUS: False
BGR のまま Image 化:  (50, 100, 200)
元の RGB:             (200, 100, 50)
[200 100  50] [ 44 200 150]
[25.833334 26.666666 31.666666]
```

**注意点・落とし穴**:
- `(H, W, C)` の順序と `(x, y)` 座標の対応、および `im.size == (W, H)` を意識しないと、縦横を取り違えて転置した画像になる。
- `arr[..., ::-1]` や `transpose` の結果のようなメモリ上で連続していない配列(`C_CONTIGUOUS: False`)も `Image.fromarray` はそのまま受け付ける(上の `BGR のまま Image 化` で確認)。
- OpenCV(`cv2.imread`)は BGR 順で読む。Pillow との受け渡しでは `arr[..., ::-1]` で軸を反転する(上の `BGR のまま Image 化` は R と B が入れ替わった画像になる)。
- uint8 のまま四則演算するとオーバーフローで折り返す(上の `a + 100` は `[200 100 50]` が `[44 200 150]` になる)。加工や統計は `astype(np.float32)` してから行い、画像へ戻す前に `np.clip(..., 0, 255).astype(np.uint8)` する。

---

## 統計・メタデータ

### `im.histogram()`

**用途**: 各チャンネルの画素値(0-255)ごとの度数を返す。

**シグネチャ**: `Image.histogram(self, mask=None, extrema=None)`

**使用例**:
```python
from PIL import Image, ImageDraw

im = Image.new("RGB", (10, 10), (255, 0, 0))
ImageDraw.Draw(im).rectangle((0, 0, 4, 9), fill=(0, 0, 255))     # 左半分を青に
h = im.histogram()
print(len(h), "R=255:", h[255], "R=0:", h[0], "G=0:", h[256], "B=255:", h[256 * 2 + 255], "total:", sum(h))

L = im.convert("L")
print(len(L.histogram()), [(i, c) for i, c in enumerate(L.histogram()) if c])

mask = Image.new("L", (10, 10), 0); mask.paste(255, (0, 0, 5, 10))   # 左半分だけ数える
print([(i, c) for i, c in enumerate(im.histogram(mask)) if c and i < 256])
print(len(Image.new("RGBA", (4, 4)).histogram()), len(Image.new("P", (4, 4)).histogram()))
```
実行結果:
```
768 R=255: 50 R=0: 50 G=0: 100 B=255: 50 total: 300
256 [(29, 50), (76, 50)]
[(0, 50)]
1024 256
```

**注意点・落とし穴**:
- 戻り値は「バンド数 x 256」の長さのリスト(`RGB` は 768 要素で、先頭 256 が R、次が G、最後が B)。`mask`(`L`/`1`)を渡すと、その領域の画素だけを数える。`matplotlib` でヒストグラムを描くときは `np.array(h).reshape(3, 256)` で分けるとよい。

---

### `im.getbbox()`

**用途**: 画像内の「ゼロでない領域」を囲む最小の矩形 `(left, upper, right, lower)` を返す(余白の自動トリミングに使える)。

**シグネチャ**: `Image.getbbox(self, *, alpha_only=True)`

**使用例**:
```python
from PIL import Image, ImageChops, ImageDraw

c = Image.new("RGB", (100, 80), (0, 0, 0))
ImageDraw.Draw(c).rectangle((20, 10, 49, 39), fill=(255, 255, 255))
print(c.getbbox(), Image.new("RGB", (5, 5)).getbbox())

rgba = Image.new("RGBA", (100, 80), (255, 255, 255, 0))
ImageDraw.Draw(rgba).ellipse((30, 20, 60, 50), fill=(255, 0, 0, 255))
print(rgba.getbbox(), rgba.getbbox(alpha_only=False))
print(rgba.crop(rgba.getbbox()).size)

# 白背景の余白トリミング: 背景色との差分の bbox を使う
wb = Image.new("RGB", (100, 80), (255, 255, 255))
ImageDraw.Draw(wb).rectangle((20, 10, 49, 39), fill=(0, 0, 0))
diff = ImageChops.difference(wb, Image.new("RGB", wb.size, (255, 255, 255)))
print(diff.getbbox(), wb.crop(diff.getbbox()).size)
```
実行結果:
```
(20, 10, 50, 40) None
(30, 20, 61, 51) (0, 0, 100, 80)
(31, 31)
(20, 10, 50, 40) (30, 30)
```

**注意点・落とし穴**:
- 「ゼロでない」つまり黒 (0) が背景として扱われる。全画素が 0 なら `None` を返すので `crop(None)` にならないよう注意。白背景の画像では上のように `ImageChops.difference` で背景色との差を取ってから `getbbox` する。
- `RGBA` の既定(`alpha_only=True`)はアルファが 0 でない領域だけを見る。`alpha_only=False` にすると RGB 側も含めて判断する(透明部分の RGB が 0 でない場合は全域になる)。
- 戻り値の `(right, lower)` は含まない端(`crop` にそのまま渡せる)。

---

### `im.getcolors(...)` / `im.getextrema()`

**用途**: 使われている色とその出現数の一覧(`getcolors`)/ チャンネルごとの最小・最大値(`getextrema`)。

**シグネチャ**:
- `Image.getcolors(self, maxcolors=256)`
- `Image.getextrema(self)`

**使用例**:
```python
from PIL import Image, ImageDraw

im = Image.new("RGB", (10, 10), (255, 0, 0))
ImageDraw.Draw(im).rectangle((0, 0, 4, 9), fill=(0, 0, 255))
print(im.getcolors())
print(sorted(im.getcolors(), reverse=True)[0])
print(im.getextrema(), im.convert("L").getextrema(), Image.new("F", (2, 2), 1.5).getextrema())

import numpy as np
rgb = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
many = Image.fromarray(rgb)
print(many.getcolors())                              # 256 色を超えると None
print(len(many.getcolors(maxcolors=32 * 32)))       # maxcolors を増やせば取得できる
print(Image.new("L", (3, 3), 7).getcolors())
```
実行結果:
```
[(50, (255, 0, 0)), (50, (0, 0, 255))]
(50, (255, 0, 0))
((0, 255), (0, 0), (0, 255)) (29, 76) (1.5, 1.5)
None
1024
[(9, 7)]
```

**注意点・落とし穴**:
- `getcolors()` は `[(個数, 色), ...]` のリストで、**色数が `maxcolors`(既定 256)を超えると `None`** を返す(エラーではない)。写真などでは `getcolors(maxcolors=w*h)` で全色を取得するか、`quantize` で減色してから使う。
- リストの順序は保証されないので、多い順に並べたければ `sorted(..., reverse=True)` する。
- `getextrema` は `RGB` なら `((Rmin, Rmax), (Gmin, Gmax), (Bmin, Bmax))`、`L` なら `(min, max)`。

---

### `ImageStat.Stat(...)` / `im.entropy()`

**用途**: 画像の統計量(平均・中央値・標準偏差・合計・RMS など)をチャンネルごとに計算する。

**シグネチャ**: `ImageStat.Stat(image_or_list, mask=None)`

**使用例**:
```python
from PIL import Image, ImageDraw, ImageStat

im = Image.new("RGB", (10, 10), (255, 0, 0))
ImageDraw.Draw(im).rectangle((0, 0, 4, 9), fill=(0, 0, 255))
s = ImageStat.Stat(im)
print("count :", s.count)
print("sum   :", s.sum)
print("mean  :", s.mean)
print("median:", s.median)
print("extrema:", s.extrema)
print("stddev:", s.stddev)

mask = Image.new("L", (10, 10), 0); mask.paste(255, (0, 0, 5, 10))
print("マスク領域の輝度平均:", ImageStat.Stat(im.convert("L"), mask).mean)

gray = Image.linear_gradient("L")
print("entropy:", round(gray.entropy(), 4))
```
実行結果:
```
count : [100, 100, 100]
sum   : [12750.0, 0.0, 12750.0]
mean  : [127.5, 0.0, 127.5]
median: [255, 0, 255]
extrema: [(0, 255), (0, 0), (0, 255)]
stddev: [127.5, 0.0, 127.5]
マスク領域の輝度平均: [29.0]
entropy: 8.0
```

**注意点・落とし穴**:
- 属性はすべてチャンネルごとのリスト(`L` なら要素1つ)。`mask` を渡すとその領域だけで計算する。`entropy()` はヒストグラムのシャノンエントロピー(ビット単位、`L` 全階調が均等なら 8.0)で、画像の情報量や単調さの目安になる。

---

### `im.getexif()` と `ExifTags`

**用途**: 画像のメタデータ(EXIF)を読み書きする。タグ番号→名前の対応は `PIL.ExifTags` にある。

**シグネチャ**: `Image.getexif(self)`

**使用例**:
```python
from PIL import Image, ExifTags

ex = Image.Exif()
ex[ExifTags.Base.Make] = "TestCam"
ex[ExifTags.Base.Orientation] = 6
ex[ExifTags.Base.Software] = "pil-doc"
Image.new("RGB", (30, 20), (255, 0, 0)).save("53_exif.jpg", exif=ex)

o = Image.open("53_exif.jpg")
e2 = o.getexif()
print(len(e2), dict(e2))
for k, v in e2.items():
    print(k, ExifTags.TAGS.get(k, k), v)
print(e2.get(ExifTags.Base.Make), e2.get(0x0112), e2.get(0x9999, "none"))
print(int(ExifTags.Base.Orientation), ExifTags.TAGS[274])
print(len(Image.new("RGB", (2, 2)).getexif()))     # EXIF が無い画像は空

# 一部を書き換えて再保存
e2[0x0112] = 1
e2.pop(ExifTags.Base.Software)
o.save("53_exif2.jpg", exif=e2)
print(dict(Image.open("53_exif2.jpg").getexif()))

# save に exif を渡さなければ EXIF は出力されない(コピーでも同様)
Image.open("53_exif.jpg").save("53_noexif.jpg")
print(dict(Image.open("53_noexif.jpg").getexif()))
```
実行結果:
```
3 {305: 'pil-doc', 274: 6, 271: 'TestCam'}
305 Software pil-doc
274 Orientation 6
271 Make TestCam
TestCam 6 none
274 Orientation
0
{274: 1, 271: 'TestCam'}
{}
```

**注意点・落とし穴**:
- `getexif()` は dict 風の `Image.Exif` を返す(キーはタグ番号)。名前で参照したい場合は `ExifTags.TAGS[番号]`、キーとして使うなら `ExifTags.Base.Orientation`(整数と同等)。
- 撮影日時・カメラ情報などは `getexif().get_ifd(ExifTags.IFD.Exif)` の中(サブ IFD)にある。GPS 情報は `ExifTags.IFD.GPSInfo`。
- EXIF を保持したまま保存するには `save(..., exif=exif)` を明示する。逆に、`exif` を渡さない `save` は EXIF を出力しないので、位置情報などを取り除いて公開したい場合は `Image.open(...).save(...)` で足りる(ただし PNG の `pnginfo`、ICC プロファイルなど別のメタデータには別途注意)。
- PNG にも `save(..., exif=...)` で EXIF を保存でき、読み直せる(この環境で確認)。ただしカメラが書く EXIF は通常 JPEG/TIFF に入っている。

---

### `im.info` と PNG テキスト・dpi(`PngInfo`)

**用途**: 形式ごとの付随情報(dpi・テキストチャンク・GIF の duration など)を `info` dict から読む。

**シグネチャ**: `Image.save(self, fp, format=None, **params)`

**使用例**:
```python
from PIL import Image
from PIL.PngImagePlugin import PngInfo

im = Image.new("RGB", (10, 10), (255, 0, 0))
pi = PngInfo()
pi.add_text("Author", "Ann")
pi.add_text("Comment", "hello")
im.save("54_meta.png", pnginfo=pi, dpi=(300, 300))
o = Image.open("54_meta.png")
print(o.info)
print(o.text)                                    # PNG のテキストチャンクだけ

im.save("54_dpi.jpg", dpi=(72, 72), quality=90)
print(Image.open("54_dpi.jpg").info)

g = Image.new("P", (4, 4))
g.save("54_i.gif", transparency=0, comment=b"note")
print(Image.open("54_i.gif").info)
```
実行結果:
```
{'Author': 'Ann', 'Comment': 'hello', 'dpi': (299.9994, 299.9994)}
{'Author': 'Ann', 'Comment': 'hello'}
{'jfif': 257, 'jfif_version': (1, 1), 'dpi': (72, 72), 'jfif_unit': 1, 'jfif_density': (72, 72)}
{'version': b'GIF89a', 'background': 0, 'transparency': 0, 'comment': b'note', 'duration': 0}
```

**注意点・落とし穴**:
- `info` は開いたファイル由来の値だけを持つので、`convert`/`resize` などで作った新しい画像ではほぼ空になる(必要なら引き継ぎ処理が必要)。
- PNG の dpi は「ピクセル/メートル」で保存されるため、`dpi=(300, 300)` で保存して読み直すと `299.9994` のようにわずかにずれる。JPEG は整数のまま。
- PNG のテキストは `o.text`(`PngImageFile` のみ)でも読める。JPEG の JFIF 情報(`jfif`, `jfif_version`, `dpi`)も `info` に入る。

---

## アニメーションGIF・複数フレーム

### `im.n_frames` / `im.is_animated` / `im.seek(...)` / `im.tell()` / `ImageSequence.Iterator`

**用途**: 複数フレームを持つ画像(GIF・APNG・WebP・TIFF)のフレーム数を調べ、任意のフレームへ移動して読む。

**シグネチャ**:
- `Image.seek(self, frame)`
- `ImageSequence.Iterator(im)`

**使用例**:
```python
from PIL import Image, ImageDraw, ImageSequence

# 5フレームのGIFを作る(円が右へ動く)
frames = []
for i in range(5):
    f = Image.new("RGB", (60, 40), (255, 255, 255))
    ImageDraw.Draw(f).ellipse((5 + i * 10, 10, 25 + i * 10, 30), fill=(255 - i * 50, 0, i * 50))
    frames.append(f)
frames[0].save("55_anim.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

g = Image.open("55_anim.gif")
print(g.format, g.mode, g.size, g.n_frames, g.is_animated, g.info.get("duration"), g.info.get("loop"), g.tell())

g.seek(2)
print(g.tell(), g.mode, g.getpixel((0, 0)))
try:
    g.seek(5)
except EOFError as e:
    print("EOFError:", e)

for i, fr in enumerate(ImageSequence.Iterator(g)):
    print(i, fr.tell(), fr.info.get("duration"), fr.mode, fr.size)

s = Image.new("RGB", (2, 2))                     # 単一フレームの画像
print(getattr(s, "n_frames", None), getattr(s, "is_animated", None))

copies = [fr.copy() for fr in ImageSequence.Iterator(g)]
print(len(copies), copies[0].mode)
lst = [fr for fr in ImageSequence.Iterator(g)]
print("コピーせず溜めた場合のユニークなオブジェクト数:", len({id(x) for x in lst}))
```
実行結果:
```
GIF P (60, 40) 5 True 100 0 0
2 RGB (255, 255, 255)
EOFError: attempt to seek outside sequence
0 0 100 P (60, 40)
1 1 100 RGB (60, 40)
2 2 100 RGB (60, 40)
3 3 100 RGB (60, 40)
4 4 100 RGB (60, 40)
None None
5 P
コピーせず溜めた場合のユニークなオブジェクト数: 1
```

**注意点・落とし穴**:
- `n_frames` / `is_animated` は複数フレーム対応の形式で開いた画像にのみある属性。単一フレームの画像(`new` で作ったもの、通常の PNG)には存在しないので、`getattr(im, "n_frames", 1)` のように使うと安全。
- 範囲外へ `seek` すると `EOFError`(負のフレーム番号も同様)。`tell()` は現在のフレーム番号。
- **GIF は 2フレーム目以降を読むと `mode` が `P` → `RGB` に変わる**(上の出力で先頭 `P`、以降 `RGB`)。差分フレームが合成済みの完全な画像として返る。
- `ImageSequence.Iterator(im)` は**同じ画像オブジェクトを `seek` しながら返す**ので、`[f for f in Iterator(g)]` のように溜めると全要素が同一オブジェクト(最後のフレーム)になる。保存したいフレームは `f.copy()` する(`ImageSequence.all_frames(im)` は各フレームのコピーのリストを返す)。

---

### `im.save(..., save_all=True, append_images=[...], duration=..., loop=...)`(アニメーション保存)

**用途**: 複数の画像を1つのアニメーション GIF(または APNG / WebP)/ 複数ページの TIFF・PDF として保存する。

**シグネチャ**: `Image.save(self, fp, format=None, **params)`

**使用例**:
```python
import os
from PIL import Image, ImageDraw

frames = []
for i in range(5):
    f = Image.new("RGB", (60, 40), (255, 255, 255))
    ImageDraw.Draw(f).ellipse((5 + i * 10, 10, 25 + i * 10, 30), fill=(255 - i * 50, 0, i * 50))
    frames.append(f)

frames[0].save("56_a.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
g = Image.open("56_a.gif"); print("GIF   ", g.n_frames, g.info["duration"], g.info["loop"])

# フレームごとに違う duration(ms)
frames[0].save("56_b.gif", save_all=True, append_images=frames[1:], duration=[100, 200, 300, 400, 500], loop=2)
g = Image.open("56_b.gif")
print("GIF   ", [(g.seek(i), g.info["duration"])[1] for i in range(g.n_frames)], "loop=", g.info.get("loop"))

# loop を指定しないとループ情報なし(再生は1回だけ)
frames[0].save("56_c.gif", save_all=True, append_images=frames[1:], duration=50)
print("loop 指定なし: 'loop' in info ->", "loop" in Image.open("56_c.gif").info)

# 同じ内容のフレームは1つに統合され、duration が合算される
same = [Image.new("RGB", (10, 10), "red")] * 3
same[0].save("56_same.gif", save_all=True, append_images=same[1:], duration=100)
s = Image.open("56_same.gif"); print("同一フレーム3枚 ->", s.n_frames, s.info["duration"])

frames[0].save("56_a.png", save_all=True, append_images=frames[1:], duration=100, loop=0)
p = Image.open("56_a.png"); print("APNG  ", p.format, p.n_frames, p.is_animated, p.info.get("duration"))
frames[0].save("56_a.webp", save_all=True, append_images=frames[1:], duration=100, loop=0)
w = Image.open("56_a.webp"); print("WEBP  ", w.n_frames, w.is_animated)
frames[0].save("56_m.tiff", save_all=True, append_images=frames[1:]); print("TIFF  ", Image.open("56_m.tiff").n_frames)
frames[0].save("56_m.pdf", save_all=True, append_images=frames[1:]); print("PDF   ", os.path.getsize("56_m.pdf") > 0)
```
実行結果:
```
GIF    5 100 0
GIF    [100, 200, 300, 400, 500] loop= 2
loop 指定なし: 'loop' in info -> False
同一フレーム3枚 -> 1 300
APNG   PNG 5 True 100.0
WEBP   5 True
TIFF   5
PDF    True
```

**注意点・落とし穴**:
- 最初のフレームで `save` を呼び、残りを `append_images` に渡す。`duration` はミリ秒(リストでフレーム別にも指定可)。`loop=0` は無限ループで、**省略するとループ情報が入らず1回だけ再生**される GIF になる。
- GIF は 256 色パレットなので、`RGB` フレームは減色される(グラデーションや写真では色数が落ちて見た目が変わる)。`optimize=True` や `disposal=2`(フレームごとに背景に戻す)などのオプションもある。
- 連続する同一内容のフレームは自動的に1フレームに統合される(上の `same.gif`)。アニメーションの間を意図的に空けたいなら `duration` を長くする。
- WebP・APNG(拡張子 `.png` + `save_all`)でもアニメーション保存できる。WebP・AVIF の利用可否はビルド依存(`features.check`)。PDF は複数ページの書き出し用で、読み込みはできない。

---

## その他

### `im.quantize(...)`

**用途**: 画像の色数を減らす(減色)。結果は `P`(パレット)モードで、`convert("RGB")` で色付き画像に戻せる。

**シグネチャ**: `Image.quantize(self, colors=256, method=None, kmeans=0, palette=None, dither=Dither.FLOYDSTEINBERG)`

**使用例**:
```python
import numpy as np
from PIL import Image

rng = np.random.default_rng(0)
x = np.linspace(0, 255, 256)[None, :, None] * np.ones((256, 1, 3))
x[..., 1] = np.linspace(0, 255, 256)[:, None]
x[..., 2] = 128
photo = Image.fromarray(np.clip(x + rng.normal(0, 6, x.shape), 0, 255).astype("uint8"))

q = photo.quantize(colors=8)
print(q.mode, len(q.getcolors()), q.getpalette()[:6])
for m in (Image.Quantize.MEDIANCUT, Image.Quantize.MAXCOVERAGE, Image.Quantize.FASTOCTREE):
    print(m.name, len(photo.quantize(16, method=m).getcolors()))
try:
    photo.quantize(16, method=Image.Quantize.LIBIMAGEQUANT)
except ValueError as e:
    print("LIBIMAGEQUANT:", e)

q2 = photo.quantize(4, dither=Image.Dither.NONE)
print(len(q2.getcolors()), q2.convert("RGB").getpixel((0, 0)))

# 自前パレット
pp = Image.new("P", (2, 1)); pp.putpalette([255, 0, 0, 0, 255, 0]); pp.putpixel((1, 0), 1)
print(pp.convert("RGB").getpixel((0, 0)), pp.convert("RGB").getpixel((1, 0)))
q.convert("RGB").save("57_quantized.png")
```
実行結果:
```
P 8 [191, 223, 128, 191, 158, 127]
MEDIANCUT 16
MAXCOVERAGE 16
FASTOCTREE 16
LIBIMAGEQUANT: dependency required by this method was not enabled at compile time
4 (63, 63, 128)
(255, 0, 0) (0, 255, 0)
```
`57_quantized.png`(256x256)は 8 色に減色された画像。元は横方向に赤、縦方向に緑が変化するグラデーションだったが、左右2列x上下4段の色のブロックになり、ブロックの境界付近には細かい点(ディザ)が散る。

**注意点・落とし穴**:
- 既定は Floyd-Steinberg ディザリング(`dither=Image.Dither.FLOYDSTEINBERG`)で、ディザを切りたいなら `Image.Dither.NONE`。`method` は既定で `MEDIANCUT`(`RGBA` では `FASTOCTREE`)。
- `LIBIMAGEQUANT` は libimagequant を組み込んだビルドでのみ使える(この環境では `ValueError: dependency required by this method was not enabled at compile time`)。
- `getcolors()` はパレット色の出現数を返すので、減色後は色数が `colors` 以下になる。GIF 保存やドット絵風の加工に使える。

---

### `ImageColor.getrgb(...)`

**用途**: 色名・16進・`rgb()`/`hsl()` 形式の文字列を `(R, G, B)` タプルへ変換する。

**シグネチャ**: `ImageColor.getrgb(color)`

**使用例**:
```python
from PIL import ImageColor

print(ImageColor.getrgb("red"), ImageColor.getrgb("#ff8000"), ImageColor.getrgb("#f80"))
print(ImageColor.getrgb("rgb(10,20,30)"), ImageColor.getrgb("hsl(120,100%,50%)"))
print(ImageColor.getrgb("#ff800080"))
print(ImageColor.getcolor("red", "L"), ImageColor.getcolor("red", "RGBA"))
try:
    ImageColor.getrgb("notacolor")
except ValueError as e:
    print("ValueError:", e)
```
実行結果:
```
(255, 0, 0) (255, 128, 0) (255, 136, 0)
(10, 20, 30) (0, 255, 0)
(255, 128, 0, 128)
76 (255, 0, 0, 255)
ValueError: unknown color specifier: 'notacolor'
```

**注意点・落とし穴**:
- `Image.new`/`paste`/`ImageDraw` の色引数は色名の文字列を直接受け取れるので、`getrgb` が必要なのは値そのものを取り出したいとき。`getcolor(color, mode)` はモードに合わせた値を返す(`"L"` なら輝度の整数)。8桁の16進(`#rrggbbaa`)は RGBA の4要素を返す。

---

### `Image.linear_gradient(...)` / `Image.radial_gradient(...)` / `Image.effect_noise(...)`

**用途**: テスト用・マスク用の画像(256x256 のグラデーション、ガウシアンノイズ)を生成する。

**シグネチャ**:
- `Image.linear_gradient(mode)`
- `Image.radial_gradient(mode)`
- `Image.effect_noise(size, sigma)`

**使用例**:
```python
import numpy as np
from PIL import Image

lg = Image.linear_gradient("L")
print(lg.size, lg.getpixel((0, 0)), lg.getpixel((0, 255)), lg.getpixel((100, 128)))
rg = Image.radial_gradient("L")
print(rg.size, rg.getpixel((128, 128)), rg.getpixel((0, 0)))
n = Image.effect_noise((64, 64), 30)
print(n.mode, n.size, 100 < np.asarray(n).mean() < 156)

# 円形グラデーションをマスクに使う(中心ほど不透明)
base = Image.new("RGB", (256, 256), (255, 0, 0))
mask = Image.radial_gradient("L").point(lambda v: 255 - v)
out = Image.composite(base, Image.new("RGB", (256, 256), (0, 0, 255)), mask)
out.save("59_radial_mask.png")
```
実行結果:
```
(256, 256) 0 255 128
(256, 256) 0 255
L (64, 64) True
```
`59_radial_mask.png`(256x256)は、中心が赤・周辺が青で、その間が滑らかに変化する放射状のグラデーション。

**注意点・落とし穴**:
- `linear_gradient` は上が黒(0)・下が白(255)の縦方向グラデーション、`radial_gradient` は中心が黒(0)で外側に向かって明るくなる(角では 255)。モード引数は通常 `"L"` を指定する。`effect_noise((w, h), sigma)` は平均 128 付近・標準偏差 `sigma` のノイズ画像(`L`)。

---

### `Image.MAX_IMAGE_PIXELS` と `im.verify()`(巨大画像・壊れた画像への対処)

**用途**: デコンプレッションボム(極端に大きな画像で DoS を狙う攻撃)対策の上限値と、ファイルが壊れていないかの簡易チェック。

**シグネチャ**: `Image.MAX_IMAGE_PIXELS`(モジュール変数)、`Image.verify(self)`

**使用例**:
```python
import warnings
from PIL import Image

print(Image.MAX_IMAGE_PIXELS)
Image.new("RGB", (20, 20)).save("60_bomb.png")
import numpy as np
rng = np.random.default_rng(0)
Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save("60_noise.png")

old = Image.MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS = 300                # 上限 300 画素に(テスト用に極端に小さくする)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    im = Image.open("60_bomb.png")
    print(im.size, [(x.category.__name__) for x in w])
Image.MAX_IMAGE_PIXELS = 100                # 上限の2倍(200)を超えるとエラー
try:
    Image.open("60_bomb.png")
except Image.DecompressionBombError as e:
    print("DecompressionBombError:", e)
Image.MAX_IMAGE_PIXELS = old

# verify: 壊れたファイルの検出(verify 後の画像は使い回せない)
ok = Image.open("60_noise.png"); print(ok.verify())
with open("60_noise.png", "rb") as f: data = f.read()
with open("60_broken.png", "wb") as f: f.write(data[:len(data) // 2])   # 半分で切る
try:
    Image.open("60_broken.png").load()
except OSError as e:
    print("load:", type(e).__name__, e)
try:
    Image.open("60_broken.png").verify()
except OSError as e:
    print("verify:", type(e).__name__, e)
```
実行結果:
```
89478485
(20, 20) ['DecompressionBombWarning']
DecompressionBombError: Image size (400 pixels) exceeds limit of 200 pixels, could be decompression bomb DOS attack.
None
load: OSError image file is truncated
verify: OSError Truncated File Read
```

**注意点・落とし穴**:
- `MAX_IMAGE_PIXELS`(既定 89,478,485 画素)を超える画像を開くと `DecompressionBombWarning`、**上限の2倍を超えると `DecompressionBombError`**(例外)になる。ユーザーがアップロードした画像を処理するサービスでは、この保護を外さず(`None` にしない)、必要なら上限を自分で決める。
- `verify()` は完全にデコードはせず、ファイルの整合性だけを検査して問題があれば例外を出す(正常なら `None`)。`verify()` 後の画像オブジェクトは使えなくなる(この環境では `load()`・`getpixel` が `AssertionError` になった)ので、検査後は開き直す。
- 途中で切れたファイルは `Image.open` 自体は成功し、`load()`(やその後の処理)で `OSError: image file is truncated` になる。切れた画像でも読める分だけ読みたいなら `ImageFile.LOAD_TRUNCATED_IMAGES = True` を設定する(黙って欠けた画像を通すので注意)。
