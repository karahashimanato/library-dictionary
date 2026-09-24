# SymPy 逆引き辞書

SymPy 1.14.0 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認)。

## 目次

1. [シンボルと式の基礎](#シンボルと式の基礎)
2. [式の操作](#式の操作)
3. [代入と評価](#代入と評価)
4. [微積分](#微積分)
5. [方程式](#方程式)
6. [線形代数](#線形代数)
7. [集合・論理](#集合論理)
8. [数論・組合せ](#数論組合せ)
9. [確率(sympy.stats)](#確率sympystats)
10. [コード生成・出力](#コード生成出力)
11. [その他](#その他)

---

## シンボルと式の基礎

### `symbols(names, ...)` / `Symbol(name, **assumptions)`

**用途**: 数式の変数(シンボル)を作る。SymPy のすべての計算の出発点。`symbols` は複数を一括生成でき、`'a:3'` のような範囲記法も使える。

**シグネチャ**:
- `symbols(names, *, cls=<class 'sympy.core.symbol.Symbol'>, **args) -> 'Any'`
- `Symbol(name, **assumptions)`

**使用例**:
```python
from sympy import symbols, Symbol, Function, sqrt
x, y, z = symbols('x y z')
print(symbols('a b c'))
print(symbols('a:3'))
print(symbols('x1:4'))
print(symbols('f g', cls=Function))
p = symbols('p', positive=True)
print(sqrt(p**2), sqrt(x**2))
print(Symbol('x') == Symbol('x', positive=True))
print(symbols('xyz'))
```
実行結果:
```
(a, b, c)
(a0, a1, a2)
(x1, x2, x3)
(f, g)
p sqrt(x**2)
False
xyz
```

**注意点・落とし穴**:
- `Symbol('x')` と `Symbol('x', positive=True)` は**別のシンボル**として扱われ、`==` は `False` になる。同じ名前でも仮定(assumptions)が違えば別物なので、混在させると `subs` や `solve` が意図通りに動かなくなる。
- 仮定なしのシンボルは複素数として扱われるため、`sqrt(x**2)` は `x` に簡約されない。`positive=True` などを付けると `p` に簡約される。
- `symbols('x y z')` のようにスペース(またはカンマ)区切りで書く。1 文字ずつ分割されるわけではなく、`symbols('xyz')` は `xyz` という名前の 1 つのシンボルを返す。

---

### `Integer(i)` / `Rational(p, q)` / `Float(num, dps)`

**用途**: SymPy の厳密な整数・有理数・任意精度の浮動小数点数を作る。Python の `int` / `float` との違いが重要で、`Rational` は誤差なしの分数演算ができる。

**シグネチャ**:
- `Integer(i)`
- `Rational(p, q=None, gcd=None)`
- `Float(num, dps=None, precision=None)`

**使用例**:
```python
from sympy import Integer, Rational, Float
print(Rational(1, 3) + Rational(1, 6))
print(1/3)
print(Integer(1)/3)
print(Float(1)/3)
print(Float('0.1', 30))
print(Rational('0.1'))
print(Rational(0.1))
```
実行結果:
```
1/2
0.3333333333333333
1/3
0.333333333333333
0.100000000000000000000000000000
1/10
3602879701896397/36028797018963968
```

**注意点・落とし穴**:
- Python の `1/3` は SymPy に渡す前に `float`(0.3333333333333333)になってしまう。厳密に扱いたいなら `Rational(1, 3)` か `Integer(1)/3` と書く。
- `Rational(0.1)` は 0.1 が 2 進 float で持つ誤差込みの値(`3602879701896397/36028797018963968`)になる。`Rational('0.1')` のように文字列で渡せば `1/10` になる。
- `Float(1)/3` の表示は既定で 15 桁だが、精度は `Float('0.1', 30)` のように桁数(`dps`)で指定できる。

---

### 定数 `pi` / `E` / `oo` / `I` / `zoo` / `nan`

**用途**: 円周率・ネイピア数・無限大・虚数単位などの数学定数(関数ではなくシングルトンオブジェクト)。厳密値のまま演算でき、`evalf` で数値化できる。

**シグネチャ**: なし(関数ではなくオブジェクト/属性として使う)

**使用例**:
```python
from sympy import pi, E, oo, I, zoo, nan, S, exp
print(pi.evalf(20))
print(E**2)
print(oo + 1)
print(I**2)
print(oo - oo)
print(pi.is_irrational)
print(type(pi), type(oo))
print(float(pi), int(pi), complex(1 + 2*I))
print(zoo, zoo + 1)
print(S(1)/0)
print(exp(1))
```
実行結果:
```
3.1415926535897932385
exp(2)
oo
-1
nan
True
<class 'sympy.core.numbers.Pi'> <class 'sympy.core.numbers.Infinity'>
3.141592653589793 3 (1+2j)
zoo zoo
zoo
E
```

**注意点・落とし穴**:
- `E` は大文字(ネイピア数)。`exp(1)` も同じ `E` に評価される(上の例)。小文字の `e` という名前は `sympy` には存在しない(`hasattr(sympy, 'e')` は `False`)。
- `oo - oo` は `nan` になる。`zoo` は複素無限大で、`S(1)/0` のようなゼロ除算の結果として現れる(Python の `1/0` は例外だが、SymPy の整数を使った `S(1)/0` は `zoo` を返す)。
- `from sympy import *` すると `E`・`I`・`S`・`N` などの 1 文字名がグローバルに入る。自分の変数名(例: 電流 `I`、誤差 `E`)と衝突して上書きしないよう注意。

---

### `sympify(a, ...)` / `S(...)`

**用途**: 文字列や Python オブジェクトを SymPy の式に変換する。`S('...')` は `sympify` の短縮形で、`S.Half` `S.Reals` のような特殊オブジェクトの入れ物でもある。

**シグネチャ**: `sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None)`

**使用例**:
```python
from sympy import sympify, S, Rational
print(sympify('x**2 + 2*x'))
e = sympify('1/3')
print(e, type(e))
print(sympify('2^3'))
print(S('x + 1'), S(3), S.Half, S.One)
print(S(1)/3)
```
実行結果:
```
x**2 + 2*x
1/3 <class 'sympy.core.numbers.Rational'>
8
x + 1 3 1/2 1
1/3
```

**注意点・落とし穴**:
- `sympify('1/3')` は `Rational(1, 3)` になる(Python の float `0.333...` にはならない)。`sympify('2^3')` は既定の `convert_xor=True` により `2**3 = 8` として評価される。
- `sympify` は内部で文字列を eval 相当で評価するため、docstring にもある通り**信頼できない(サニタイズされていない)入力には使ってはいけない**。

---

### `nsimplify(expr, constants=(), tolerance=None, ...)`

**用途**: 浮動小数点数を、近い有理数や指定した定数(`sqrt(2)` など)を使った厳密な式に変換する。

**シグネチャ**: `nsimplify(expr, constants=(), tolerance=None, full=False, rational=None, rational_conversion='base10')`

**使用例**:
```python
from sympy import nsimplify, sqrt
print(nsimplify(0.1))
print(nsimplify(0.1 + 0.2))
print(nsimplify(0.333333333333, tolerance=1e-9))
print(nsimplify(1.4142135623731, [sqrt(2)]))
print(0.1 + 0.2 == 0.3)
```
実行結果:
```
1/10
3/10
1/3
sqrt(2)
False
```

**注意点・落とし穴**:
- `0.1 + 0.2` は Python の float では `0.3` と等しくないが、`nsimplify` を通すと `3/10` に戻せる。ただし近似の推定なので、元の値が有限桁の誤差を持つ場合は `tolerance` で許容誤差を明示した方が安全。

---

### `Function(name)` / `Lambda(signature, expr)`

**用途**: 未定義の関数記号 `f(x)`(微分方程式や抽象的な式変形用)と、名前なしの無名関数を作る。

**シグネチャ**:
- `Function(*args)`
- `Lambda(signature, expr) -> 'Lambda'`

**使用例**:
```python
from sympy import Function, Lambda, symbols
x, y = symbols('x y')
f = Function('f')
print(f(x).diff(x))
print(f(x, y))
g = Lambda((x, y), x**2 + y)
print(g)
print(g(1, 2), g(x, x))
print(f(x).subs(f(x), x**2))
```
実行結果:
```
Derivative(f(x), x)
f(x, y)
Lambda((x, y), x**2 + y)
3 x**2 + x
x**2
```

**注意点・落とし穴**:
- `Function('f')` は関数クラスを返す(`f` 自体は式ではない)。`f(x)` として呼び出して初めて式になる。`dsolve` の第 2 引数などには `f(x)` の形で渡す。
- `Function` を継承して `eval` クラスメソッドを定義すれば、独自の記号関数を作ることもできる(この辞書の範囲外)。

---

### `refine(expr, assumption)` / `ask(query, assumption)` と `.is_*` 属性

**用途**: シンボルに付けた仮定、あるいは外から与えた仮定を使って式を簡約(`refine`)したり、性質を真偽で問い合わせたり(`ask`)する。

**シグネチャ**:
- `refine(expr, assumptions=True)`
- `ask(proposition, assumptions=True, context=AssumptionsContext())`

**使用例**:
```python
from sympy import symbols, Symbol, Q, refine, ask, sqrt, Abs
x = symbols('x')
q = Symbol('q', positive=True)
n = Symbol('n', integer=True)
print(q.is_positive, q.is_real, x.is_positive, x.is_real)
print((q + 1).is_positive, (x**2).is_nonnegative)
print(refine(sqrt(x**2), Q.positive(x)))
print(refine(Abs(x), Q.negative(x)))
print(ask(Q.positive(x**2 + 1)), ask(Q.positive(x), Q.positive(x)))
print(ask(Q.even(n*(n + 1)), Q.integer(n)))
print(ask(Q.positive(x)))
```
実行結果:
```
True True None None
True None
x
-x
None True
True
None
```

**注意点・落とし穴**:
- `.is_positive` などは `True` / `False` / `None` の 3 値を返す。`None` は「わからない」の意味で `False` ではない点に注意(`x.is_positive` は `None`)。
- `ask` も判定不能なら `None` を返す。`Q.positive(x**2 + 1)` のように仮定なしでは判定できない例が `None` になる。

---

## 式の操作

### `simplify(expr, ...)`

**用途**: 式を「より簡単な形」にヒューリスティックに変形する汎用関数。三角関数・ガンマ関数・有理式などを幅広く扱う。

**シグネチャ**: `simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False, doit=True, **kwargs)`

**使用例**:
```python
from sympy import symbols, simplify, sin, cos, gamma
x = symbols('x')
print(simplify((x**2 - 1)/(x - 1)))
print(simplify(sin(x)**2 + cos(x)**2))
print(simplify(gamma(x + 1)/gamma(x)))
```
実行結果:
```
x + 1
1
x
```

**注意点・落とし穴**:
- 「最も簡単」の基準は `measure`(既定は `count_ops`)で決まり、結果が人間の期待する形になるとは限らない。目的が決まっているなら `factor` / `cancel` / `trigsimp` などの専用関数の方が高速で結果も予測しやすい。
- 式によっては時間がかかる。ループ内で大量の式に呼ぶのは避けたい。

---

### `expand(e, ...)`

**用途**: 積・べき乗を展開して和の形にする。ヒント引数(`trig=True`、`log=True` など)で三角関数・対数の展開も行える。

**シグネチャ**: `expand(e, deep=True, modulus=None, power_base=True, power_exp=True, mul=True, log=True, multinomial=True, basic=True, **hints)`

**使用例**:
```python
from sympy import symbols, expand, sin, log, exp
x, y = symbols('x y')
print(expand((x + 1)**3))
print(expand((x + y)**2))
print(expand(sin(x + y), trig=True))
print(expand(log(x*y)))
print(expand(log(x*y), force=True))
print(expand(exp(x + y)))
```
実行結果:
```
x**3 + 3*x**2 + 3*x + 1
x**2 + 2*x*y + y**2
sin(x)*cos(y) + sin(y)*cos(x)
log(x*y)
log(x) + log(y)
exp(x)*exp(y)
```

**注意点・落とし穴**:
- `expand(log(x*y))` は `x`, `y` が正であるという仮定がないと展開されない(複素数では `log(x*y) != log(x) + log(y)` のため)。`force=True` で仮定を無視して展開できるが、数学的に不正確になり得る。
- `expand(exp(x + y))` は `exp(x)*exp(y)` になる(`power_exp=True` の効果)。`power_exp=False` を渡すと展開されない。

---

### `factor(f, *gens, ...)` / `factor_list(f, ...)`

**用途**: 多項式を既約因数の積に分解する。`factor_list` は係数と (因子, 重複度) のリストとして返す。

**シグネチャ**:
- `factor(f, *gens, deep=False, **args)`
- `factor_list(f, *gens, **args)`

**使用例**:
```python
from sympy import symbols, factor, factor_list, I
x = symbols('x')
print(factor(x**3 - 1))
print(factor(x**2 + 1))
print(factor(x**2 + 1, extension=I))
print(factor_list(x**3 - x))
```
実行結果:
```
(x - 1)*(x**2 + x + 1)
x**2 + 1
(x - I)*(x + I)
(1, [(x - 1, 1), (x, 1), (x + 1, 1)])
```

**注意点・落とし穴**:
- 既定では有理数体 ℚ 上で既約分解するため、`x**2 + 1` は分解されない。`extension=I` や `extension=sqrt(2)` などで拡大体上の分解ができる。

---

### `collect(expr, syms, ...)` / `Expr.coeff(x, n=1)`

**用途**: 式を指定したシンボルのべきでまとめる(`collect`)。特定の項の係数を取り出す(`coeff`)。

**シグネチャ**:
- `collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_term=True)`
- `Expr.coeff(x: 'Expr', n=1, right=False, _first=True)`

**使用例**:
```python
from sympy import symbols, collect
x, y, z = symbols('x y z')
e = x*y + x*z + x**2*y
print(collect(e, x))
print(collect(e, x, evaluate=False))
print((3*x**2 + 2*x).coeff(x, 2))
print((3*x**2 + 2*x + 5).coeff(x, 0))
print((x*y + x).coeff(x))
```
実行結果:
```
x**2*y + x*(y + z)
{x: y + z, x**2: y}
3
5
y + 1
```

**注意点・落とし穴**:
- `evaluate=False` を付けると、式ではなく `{べき: 係数}` の辞書で返る。係数の取り出しに便利。
- `coeff(x, 0)` は「`x` を含まない項」を返す。多項式の係数を全部取り出したいときは `Poly(...).all_coeffs()` の方が確実。

---

### `cancel(f, *gens, ...)`

**用途**: 有理式を「分子/分母が互いに素」な標準形に約分する。

**シグネチャ**: `cancel(f, *gens, _signsimp=True, **args)`

**使用例**:
```python
from sympy import symbols, cancel
x, y = symbols('x y')
print(cancel((x**2 - 1)/(x - 1)))
print(cancel((x**2 + 2*x + 1)/(x**2 - 1)))
print(cancel(1/x + 1/y))
```
実行結果:
```
x + 1
(x + 1)/(x - 1)
(x + y)/(x*y)
```

**注意点・落とし穴**:
- 通分と約分を同時にやってくれる。`1/x + 1/y` のような和も 1 つの分数にまとめられる。

---

### `apart(f, x=None, full=False, ...)`

**用途**: 有理式を部分分数分解する。積分やラプラス変換の前処理でよく使う。

**シグネチャ**: `apart(f, x=None, full=False, **options)`

**使用例**:
```python
from sympy import symbols, apart
x, y = symbols('x y')
print(apart(1/(x**2 - 1)))
print(apart((x**3 + 1)/(x**2 - 1)))
print(apart(1/((x + y)*(x + 2)), x))
```
実行結果:
```
-1/(2*(x + 1)) + 1/(2*(x - 1))
x + 1/(x - 1)
-1/((x + y)*(y - 2)) + 1/((x + 2)*(y - 2))
```

**注意点・落とし穴**:
- 複数のシンボルを含む式では、どれについて分解するかを第 2 引数で明示する(上の 3 例目)。

---

### `together(expr, deep=False, fraction=True)`

**用途**: 分数の和を通分して 1 つの分数にまとめる。約分はしない点が `cancel` と異なる。

**シグネチャ**: `together(expr, deep=False, fraction=True)`

**使用例**:
```python
from sympy import symbols, together
x, y = symbols('x y')
print(together(1/x + 1/y))
print(together(1/x + 1/(x + 1)))
```
実行結果:
```
(x + y)/(x*y)
(2*x + 1)/(x*(x + 1))
```

---

### `trigsimp(expr, ...)` / `expand_trig(expr)`

**用途**: 三角関数の恒等式を使って式を簡約する(`trigsimp`)/ 加法定理や倍角公式で展開する(`expand_trig`)。

**シグネチャ**:
- `trigsimp(expr, inverse=False, **opts)`
- `expand_trig(expr, deep=True)`

**使用例**:
```python
from sympy import symbols, trigsimp, expand_trig, sin, cos
x, y = symbols('x y')
print(trigsimp(sin(x)**2 + cos(x)**2))
print(trigsimp(sin(x)*cos(y) + cos(x)*sin(y)))
print(trigsimp(cos(x)**2 - sin(x)**2))
print(expand_trig(sin(2*x)))
print(expand_trig(cos(x + y)))
```
実行結果:
```
1
sin(x + y)
cos(2*x)
2*sin(x)*cos(x)
-sin(x)*sin(y) + cos(x)*cos(y)
```

---

### `powsimp(expr, ...)` / `radsimp(expr, ...)`

**用途**: べき乗の底・指数を結合する(`powsimp`)/ 分母の根号を有理化する(`radsimp`)。

**シグネチャ**:
- `powsimp(expr, deep=False, combine='all', force=False, measure=count_ops)`
- `radsimp(expr, symbolic=True, max_terms=4)`

**使用例**:
```python
from sympy import symbols, powsimp, radsimp, sqrt
x, y, z = symbols('x y z')
print(powsimp(2**x * 2**y))
print(powsimp(x**y * x**z))
print(powsimp(sqrt(x) * sqrt(y)))
print(powsimp(sqrt(x) * sqrt(y), force=True))
print(radsimp(1/(sqrt(2) + 1)))
```
実行結果:
```
2**(x + y)
x**(y + z)
sqrt(x)*sqrt(y)
sqrt(x*y)
-1 + sqrt(2)
```

**注意点・落とし穴**:
- `sqrt(x)*sqrt(y)` を `sqrt(x*y)` にまとめるには `x`, `y` が非負であるという仮定が必要。仮定なしでは変形されない(`force=True` で強制できるが数学的に危険)。

---

### `Expr.rewrite(*args, deep=True, **hints)`

**用途**: 式を別の関数(`exp`, `sin`, `cos`, `sqrt` など)で書き直す。オイラーの公式の変換によく使う。

**シグネチャ**: `Expr.rewrite(*args, deep=True, **hints)`

**使用例**:
```python
from sympy import symbols, exp, sin, cos, I
x = symbols('x')
print(exp(I*x).rewrite(cos))
print(sin(x).rewrite(exp))
print(cos(x).rewrite(exp))
```
実行結果:
```
I*sin(x) + cos(x)
-I*(exp(I*x) - exp(-I*x))/2
exp(I*x)/2 + exp(-I*x)/2
```

---

## 代入と評価

### `Basic.subs(arg1, arg2=None)` / `xreplace`

**用途**: 式中のシンボル(または部分式)を別の値・式で置き換える。辞書やタプルのリストで複数同時に指定できる。

**シグネチャ**: `Basic.subs(arg1: 'Mapping[Basic | complex, Basic | complex] | Iterable[tuple[Basic | complex, Basic | complex]] | Basic | complex', arg2: 'Basic | complex | None' = None, **kwargs: 'Any') -> 'Basic'`

**使用例**:
```python
from sympy import symbols, sin, pi
x, y = symbols('x y')
e = x**2 + x*y
print(e.subs(x, 2))
print(e.subs({x: 1, y: 2}))
print(e.subs([(x, y), (y, 3)]))
print(e.subs({x: y, y: x}))
print(e.subs({x: y, y: x}, simultaneous=True))
print(sin(x + 1).subs(x + 1, 5))
print((x**3).subs(x**2, y))
print((1/x).subs(x, 0))
print((x*y).xreplace({x: 2}))
```
実行結果:
```
2*y + 4
3
18
2*x**2
x*y + y**2
sin(5)
x**3
zoo
2*y
```

**注意点・落とし穴**:
- 辞書やリストによる複数置換は**順番に(逐次)**適用される。`{x: y, y: x}` で入れ替えを意図しても `2*x**2` になってしまう。真の同時置換には `simultaneous=True` を使う。
- `(x**3).subs(x**2, y)` は `x**3` のまま(`x**2` とは構造が一致しないため)。`subs` は数学的な同値ではなく式の構造(木)に基づく置換。
- `xreplace` は構造が完全一致するノードだけを高速に置換する低レベル版で、`subs` のような賢い(数学的な)パターン照合はしない。

---

### `Expr.evalf(n=15, subs=None, ...)` / `N(x, n=15)`

**用途**: 式を任意精度の浮動小数点数に数値評価する。`N(expr, n)` は `expr.evalf(n)` と同じ。

**シグネチャ**:
- `Expr.evalf(n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False)`
- `N(x, n=15, **options)`

**使用例**:
```python
from sympy import sqrt, pi, sin, symbols, N, Rational, exp, Float
x = symbols('x')
print(sqrt(2).evalf())
print(sqrt(2).evalf(30))
print(N(pi, 50))
print((sin(x) + 1).evalf(subs={x: 1}))
print((sin(x) + 1).evalf(5, subs={x: 1}))
print(Rational(1, 3).evalf(50))
print(N(exp(100)))
print(N(exp(100), 5))
print(1 + Float('1e-20', 30))
```
実行結果:
```
1.41421356237310
1.41421356237309504880168872421
3.1415926535897932384626433832795028841971693993751
1.84147098480790
1.8415
0.33333333333333333333333333333333333333333333333333
2.68811714181614e+43
2.6881e+43
1.00000000000000000001000000000
```

**注意点・落とし穴**:
- `n` は有効桁数(10 進)。既定は 15 桁で、それ以上を要求すれば任意精度で計算する。
- シンボルを含む式は数値にならず、`subs` で値を与える必要がある。`evalf(subs={...})` は `subs` してから評価する近道。

---

### `lambdify(args, expr, modules=None, ...)`

**用途**: SymPy の式を、NumPy / math / SciPy を使う高速な Python 関数に変換する。大量の点で評価したいときの標準手段。

**シグネチャ**: `lambdify(args, expr, modules=None, printer=None, use_imps=True, dummify=False, cse=False, docstring_limit=1000)`

**使用例**:
```python
import numpy as np
from sympy import symbols, lambdify, sin, exp, sqrt, Matrix, gamma, Piecewise
x, y = symbols('x y')
f = lambdify(x, x**2 + 1)
print(f(3), f(np.array([1, 2, 3])))
g = lambdify((x, y), sin(x)*exp(y), 'numpy')
print(g(np.array([0, np.pi/2]), 0.0))
print(lambdify(x, sin(x), 'math')(0.5))
print(lambdify(x, gamma(x), 'scipy')(5))
print(lambdify(x, Matrix([x, x**2]))(2))
print(lambdify(x, Piecewise((x, x > 0), (0, True)))(np.array([-1.0, 2.0])))
print(lambdify(x, 5, 'numpy')(np.array([1, 2, 3])))
print(lambdify(x, sqrt(x), 'numpy')(-1 + 0j), lambdify(x, sqrt(x), 'math')(4.0))
try:
    lambdify(x, sqrt(x), 'math')(-1.0)
except Exception as ex:
    print(type(ex).__name__, ex)
```
実行結果:
```
10 [ 2  5 10]
[0. 1.]
0.479425538604203
24.0
[[2]
 [4]]
[0. 2.]
5
1j 2.0
ValueError math domain error
```

**注意点・落とし穴**:
- 引数は `lambdify(x, ...)` または `lambdify((x, y), ...)` のように、式に現れるシンボルをタプルで渡す。
- `modules` により生成される関数が変わる。`'numpy'` は配列を要素ごとに評価でき、`'math'` はスカラー専用で定義域外(`sqrt(-1.0)` など)では例外になる。
- 速度の目安: `sin(x) + x**2` を 1 万点で評価した実測では、NumPy 版の lambdify が `evalf(subs=...)` の Python ループより約 3 桁(1000 倍超)速かった(環境依存の実測値)。
- 定数式(`lambdify(x, 5)`)は配列を渡しても**スカラー 5 を返す**(配列にはならない)ので、ブロードキャストが必要なら注意する。
- `Matrix` を渡すと NumPy 配列(2 次元)を返す。`Piecewise` も NumPy 配列に対応する。

---

## 微積分

### `diff(f, *symbols, **kwargs)` / `Derivative(expr, *variables)`

**用途**: 式を微分する(`diff`)。`Derivative` は評価せずに「微分する」という式のまま保持し、`.doit()` で評価する。

**シグネチャ**:
- `diff(f, *symbols, **kwargs)`
- `Derivative(expr, *variables, **kwargs)`

**使用例**:
```python
from sympy import symbols, diff, Derivative, sin, exp, Function
x, y = symbols('x y')
print(diff(sin(x)*x**2, x))
print(diff(x**4, x, 2))
print(diff(x**2*y**3, x, y))
print(diff(exp(x*y), x, 2, y, 1))
print(diff(sin(x), x, 0))
d = Derivative(sin(x), x)
print(d, d.doit())
print(d.subs(x, 0))
print(d.doit().subs(x, 0))
f = Function('f')
print(diff(f(x)**2, x))
```
実行結果:
```
x**2*cos(x) + 2*x*sin(x)
12*x**2
6*x*y**2
y*(x*y + 2)*exp(x*y)
sin(x)
Derivative(sin(x), x) cos(x)
Subs(Derivative(sin(x), x), x, 0)
1
2*f(x)*Derivative(f(x), x)
```

**注意点・落とし穴**:
- `diff(x**4, x, 2)` は 2 階微分。`diff(f, x, y)` は x で微分してから y で微分する偏微分。`diff(f, x, 0)`(0 階)は元の式を返す。
- `Derivative(sin(x), x).subs(x, 0)` は `Subs(Derivative(...), x, 0)` という未評価の形になる。値が欲しいときは先に `.doit()` する。

---

### `integrate(f, *symbols, ...)` / `Integral(function, *symbols)`

**用途**: 不定積分・定積分を計算する。`(x, a, b)` のタプルで区間を指定する。`Integral` は評価前の積分式を保持し、`.doit()` で評価する。

**シグネチャ**:
- `integrate(*args, meijerg=None, conds='piecewise', risch=None, heurisch=None, manual=None, **kwargs)`
- `Integral(function, *symbols, **assumptions) -> 'Integral'`

**使用例**:
```python
from sympy import symbols, integrate, Integral, sin, exp, oo, pi, symbols
x, y, a = symbols('x y a')
print(integrate(x**2, x))
print(integrate(sin(x), (x, 0, pi)))
print(integrate(exp(-x**2), (x, -oo, oo)))
print(integrate(1/x, x))
print(integrate(x*y, x, y))
print(integrate(x*y, (x, 0, 1), (y, 0, 2)))
print(integrate(exp(-x**2), x))
print(integrate(x**x, x))
ap = symbols('ap', positive=True)
print(integrate(exp(-a*x), (x, 0, oo)))
print(integrate(exp(-ap*x), (x, 0, oo)))
I1 = Integral(x**2, (x, 0, 1))
print(I1, I1.doit(), I1.evalf())
```
実行結果:
```
x**3/3
2
sqrt(pi)
log(x)
x**2*y**2/4
1
sqrt(pi)*erf(x)/2
Integral(x**x, x)
Piecewise((1/a, Abs(arg(a)) < pi/2), (Integral(exp(-a*x), (x, 0, oo)), True))
1/ap
Integral(x**2, (x, 0, 1)) 1/3 0.333333333333333
```

**注意点・落とし穴**:
- 不定積分の結果には積分定数 `C` は付かない。
- `integrate(exp(-x**2), x)` は `erf` で表される。原始関数が見つからない(`x**x` など)場合は式を `Integral(...)` のまま返す。エラーにはならないので、戻り値が `Integral` のままでないか確認すること。
- パラメータ `a` に仮定がないと、定積分の結果が `Piecewise`(条件付き)になる。`positive=True` を付けたシンボルなら `1/a` と単純に返る。
- `integrate(f, x, y)` は x で積分してから y で積分する(累次積分)。

---

### `limit(e, z, z0, dir='+')`

**用途**: `z` を `z0` に近づけたときの極限を求める。`oo` を指定すれば無限遠での極限も扱える。

**シグネチャ**: `limit(e, z, z0, dir='+')`

**使用例**:
```python
from sympy import symbols, limit, sin, oo, Abs
x = symbols('x')
print(limit(sin(x)/x, x, 0))
print(limit(1/x, x, 0))
print(limit(1/x, x, 0, '-'))
print(limit((1 + 1/x)**x, x, oo))
print(limit(1/x, x, oo))
print(limit(Abs(x)/x, x, 0, '+'))
print(limit(Abs(x)/x, x, 0, '-'))
print(limit(Abs(x)/x, x, 0))
```
実行結果:
```
1
oo
-oo
E
0
1
-1
1
```

**注意点・落とし穴**:
- `dir` の既定は `'+'`(右極限)。左右で値が異なる場合(`Abs(x)/x`)、`limit(..., x, 0)` は右極限の `1` を返すだけで「極限が存在しない」とは教えてくれない。左右の極限を両方確認すること。

---

### `series(expr, x=None, x0=0, n=6, dir='+')`

**用途**: 式を `x0` のまわりでべき級数(テイラー/ローラン展開)に展開する。

**シグネチャ**: `series(expr, x=None, x0=0, n=6, dir='+')`

**使用例**:
```python
from sympy import symbols, series, sin, cos, exp, log, pi
x = symbols('x')
print(series(sin(x), x))
print(series(exp(x), x, 0, 4))
print(series(cos(x), x, pi, 3))
print(series(log(x), x, 1, 3))
s = series(exp(x), x, 0, 3)
print(s, type(s))
print(s.removeO())
print(series(sin(x)/x, x, 0, 6))
```
実行結果:
```
x - x**3/6 + x**5/120 + O(x**6)
1 + x + x**2/2 + x**3/6 + O(x**4)
-1 + (x - pi)**2/2 + O((x - pi)**3, (x, pi))
-1 - (x - 1)**2/2 + x + O((x - 1)**3, (x, 1))
1 + x + x**2/2 + O(x**3) <class 'sympy.core.add.Add'>
x**2/2 + x + 1
1 - x**2/6 + x**4/120 + O(x**6)
```

**注意点・落とし穴**:
- `n` は「次数 n の手前まで」の項数の目安で、結果末尾に `O(x**n)`(剰余項)が付く。既定は `O(x**6)`。
- 剰余項 `O(...)` を含んだ式のまま演算すると扱いにくいので、多項式として使いたいときは `.removeO()` で取り除く。

---

### `summation(f, *symbols)` / `Sum(function, *symbols)` / `product(...)`

**用途**: 総和・総乗を記号的に評価する。`(k, 下限, 上限)` のタプルで範囲を指定し、上限に `oo` も使える。`Sum` は評価前の式で `.doit()` で評価する。

**シグネチャ**:
- `summation(f, *symbols, **kwargs)`
- `Sum(function, *symbols, **assumptions)`
- `product(*args, **kwargs)`

**使用例**:
```python
from sympy import symbols, summation, Sum, product, oo, Symbol
k = symbols('k')
n = Symbol('n', integer=True, positive=True)
print(summation(k, (k, 1, n)))
print(summation(1/k**2, (k, 1, oo)))
print(Sum(k**2, (k, 1, 10)).doit())
print(Sum(k, (k, 1, n)))
print(product(k, (k, 1, 5)))
print(summation(2**-k, (k, 0, oo)))
print(Sum(1/k**2, (k, 1, oo)).evalf())
```
実行結果:
```
n**2/2 + n/2
pi**2/6
385
Sum(k, (k, 1, n))
120
2
1.64493406684823
```

**注意点・落とし穴**:
- `Sum(...)` を `print` しても値は計算されない(式のまま)。`.doit()` か `.evalf()` を呼ぶ。`.evalf()` は無限和でも数値近似を返す。

---

## 方程式

### `Eq(lhs, rhs)`

**用途**: 等式(方程式)オブジェクトを作る。`solve` や `dsolve` に渡すための方程式の表現。

**シグネチャ**: `Eq(lhs, rhs, **options)`

**使用例**:
```python
from sympy import symbols, Eq, simplify, expand
x, y = symbols('x y')
e = Eq(x**2, 4)
print(e, e.lhs, e.rhs, type(e))
print(Eq(x, x), type(Eq(x, x)))
print(Eq(x, x + 1))
print(Eq(x + y, 2).subs(x, 1))
print(x == y)
print((x + 1)**2 == x**2 + 2*x + 1)
print(simplify((x + 1)**2 - (x**2 + 2*x + 1)) == 0)
print(expand((x + 1)**2 - (x**2 + 2*x + 1)))
```
実行結果:
```
Eq(x**2, 4) x**2 4 <class 'sympy.core.relational.Equality'>
True <class 'sympy.logic.boolalg.BooleanTrue'>
False
Eq(y + 1, 2)
False
False
True
0
```

**注意点・落とし穴**:
- Python の `==` は「構造が同一か」を判定するだけで、数学的な等式 `Eq` とは別物。`(x+1)**2 == x**2 + 2*x + 1` は `False` になる。式の同値確認は差を `simplify`/`expand` して `0` になるかを見る。
- 自明に成り立つ/成り立たない等式(`Eq(x, x)`、`Eq(x, x + 1)`)は、`Eq` オブジェクトではなく `True` / `False`(`BooleanTrue` など)を返す。
- `solve(x**2 - 4, x)` のように右辺 0 の式を直接渡す場合は `Eq` は不要。

---

### `solve(f, *symbols, **flags)`

**用途**: 方程式・連立方程式・不等式を解いて解のリスト(または辞書)を返す。最もよく使う方程式ソルバ。

**シグネチャ**: `solve(f, *symbols, **flags)`

**使用例**:
```python
from sympy import symbols, solve, Eq, sin, exp, Symbol
x, y = symbols('x y')
print(solve(x**2 - 4, x))
print(solve(x**2 + 1, x))
print(solve(Eq(x**2, 4), x, dict=True))
print(solve([x + y - 3, x - y - 1], [x, y]))
print(solve([x + y - 3, x - y - 1], [x, y], dict=True))
print(solve(x**2 - y, x))
print(solve(x**5 - x - 1, x))
print(solve(sin(x), x))
print(solve(x**2 > 4, x))
print(solve(exp(x) - 2, x))
r = Symbol('r', real=True)
print(solve(r**2 + 1, r))
```
実行結果:
```
[-2, 2]
[-I, I]
[{x: -2}, {x: 2}]
{x: 2, y: 1}
[{x: 2, y: 1}]
[-sqrt(y), sqrt(y)]
[CRootOf(x**5 - x - 1, 0), CRootOf(x**5 - x - 1, 1), CRootOf(x**5 - x - 1, 2), CRootOf(x**5 - x - 1, 3), CRootOf(x**5 - x - 1, 4)]
[0, pi]
((-oo < x) & (x < -2)) | ((2 < x) & (x < oo))
[log(2)]
[]
```

**注意点・落とし穴**:
- 戻り値の型は入力によって変わる。単一方程式は値のリスト、解が一意な連立方程式は辞書 `{x: 2, y: 1}`、解が複数ある連立方程式はタプルのリスト(例: `[(-sqrt(2)/2, -sqrt(2)/2), (sqrt(2)/2, sqrt(2)/2)]`)になる。後続処理を安定させるには `dict=True` を付けて「辞書のリスト」に統一するのが安全。
- `solve(sin(x), x)` は `[0, pi]` のみを返し、周期的な全解は返さない。全解が欲しいときは `solveset` を使う。
- 5 次方程式など代数的に解けないものは `CRootOf(...)`(根の記号表現)で返る。数値化は `.evalf()`。
- シンボルに `real=True` を付けると、`r**2 + 1 = 0` のように実数解がない場合は空リストになる(仮定なしなら複素数解 `[-I, I]`)。

---

### `solveset(f, symbol=None, domain=Complexes)`

**用途**: 解の**集合**を返すソルバ。定義域(`S.Reals` など)を指定でき、無限個の解や区間も集合として厳密に表現できる。

**シグネチャ**: `solveset(f, symbol=None, domain=Complexes)`

**使用例**:
```python
from sympy import symbols, solveset, sin, exp, S, Interval, oo, pi
x = symbols('x')
print(solveset(x**2 - 4, x))
print(solveset(x**2 + 1, x))
print(solveset(x**2 + 1, x, S.Reals))
print(solveset(sin(x), x))
print(solveset(x**2 > 4, x, S.Reals))
print(solveset(sin(x), x, Interval(0, 2*pi)))
print(solveset(exp(x), x))
print(solveset(x**2 - 4, x, Interval(0, oo)))
```
実行結果:
```
{-2, 2}
{-I, I}
EmptySet
Union(ImageSet(Lambda(_n, 2*_n*pi), Integers), ImageSet(Lambda(_n, 2*_n*pi + pi), Integers))
Union(Interval.open(-oo, -2), Interval.open(2, oo))
{0, pi, 2*pi}
EmptySet
{2}
```

**注意点・落とし穴**:
- `domain` の既定は複素数全体(`Complexes`)。実数解だけがほしいなら `S.Reals` を明示する(`x**2 + 1` は既定だと `{-I, I}`、`S.Reals` だと `EmptySet`)。
- 不等式は `domain=S.Reals` を指定しないと解けず、`ConditionSet(x, x**2 > 4, Complexes)`(未解決を表す集合)が返る。

---

### `linsolve(system, *symbols)` / `nonlinsolve(system, *symbols)`

**用途**: 連立方程式のソルバ。`linsolve` は連立一次方程式(式のリスト / 行列 / 拡大係数行列)、`nonlinsolve` は連立非線形方程式を解いて解の集合を返す。

**シグネチャ**:
- `linsolve(system, *symbols)`
- `nonlinsolve(system, *symbols)`

**使用例**:
```python
from sympy import symbols, linsolve, nonlinsolve, Matrix
x, y = symbols('x y')
print(linsolve([x + y - 3, x - y - 1], [x, y]))
print(linsolve(Matrix([[1, 1, 3], [1, -1, 1]]), [x, y]))
A = Matrix([[1, 1], [1, -1]]); b = Matrix([3, 1])
print(linsolve((A, b), x, y))
print(linsolve([x + y - 3, 2*x + 2*y - 6], [x, y]))
print(linsolve([x + y - 3, x + y - 4], [x, y]))
print(nonlinsolve([x**2 + y**2 - 1, x - y], [x, y]))
```
実行結果:
```
{(2, 1)}
{(2, 1)}
{(2, 1)}
{(3 - y, y)}
EmptySet
{(-sqrt(2)/2, -sqrt(2)/2), (sqrt(2)/2, sqrt(2)/2)}
```

**注意点・落とし穴**:
- 解は集合(タプルの集合)で返る。不定(解が無限にある)場合は `{(3 - y, y)}` のようにパラメータ付きで、解なしの場合は `EmptySet` を返す。
- 拡大係数行列を渡す場合は、最後の列が右辺(`Matrix([[1, 1, 3], ...])`)。

---

### `nsolve(f, x0, ...)`

**用途**: 初期値 `x0` から始めて方程式を数値的に(ニュートン法系で)解く。解析解が出ない超越方程式で使う。

**シグネチャ**: `nsolve(*args, dict=False, **kwargs)`

**使用例**:
```python
from sympy import symbols, nsolve, cos
x, y = symbols('x y')
print(nsolve(cos(x) - x, x, 1))
print(nsolve(x**2 - 2, x, 1.5))
print(nsolve(x**2 - 2, x, 1.5, prec=30))
print(nsolve([x**2 + y**2 - 1, x - y], [x, y], [1, 1]))
try:
    nsolve(x**2 + 1, x, 1)
except Exception as ex:
    print(type(ex).__name__, str(ex).splitlines()[0])
```
実行結果:
```
0.739085133215161
1.41421356237310
1.41421356237309504880168872421
Matrix([[0.707106781186548], [0.707106781186548]])
ValueError Could not find root within given tolerance. (5.22912334436213473229 > 2.16840434497100886801e-19)
```

**注意点・落とし穴**:
- **初期値を必ず与える**(第 2 / 第 3 引数)。どの解に収束するかは初期値に依存する。
- 解が見つからない場合(例: 実数の初期値 1 から実数解のない `x**2 + 1 = 0` を解こうとした場合)は `ValueError` になる。
- 連立方程式の場合は `Matrix`(列ベクトル)で解が返る。

---

### `dsolve(eq, func=None, hint='default', ics=None, ...)`

**用途**: 常微分方程式を解析的に解く。`Function('f')` で未知関数を作って `f(x).diff(x)` のように方程式を組み立て、初期条件は `ics` に辞書で渡す。

**シグネチャ**: `dsolve(eq, func=None, hint='default', simplify=True, ics=None, xi=None, eta=None, x0=0, n=6, **kwargs)`

**使用例**:
```python
from sympy import symbols, Function, dsolve, Eq
x = symbols('x')
f = Function('f')
print(dsolve(f(x).diff(x) - f(x), f(x)))
print(dsolve(f(x).diff(x, 2) + f(x), f(x)))
print(dsolve(f(x).diff(x) - f(x), f(x), ics={f(0): 1}))
print(dsolve(Eq(f(x).diff(x), x*f(x)), f(x)))
print(dsolve(f(x).diff(x) + f(x) - x, f(x)))
sol = dsolve(f(x).diff(x, 2) - f(x), f(x), ics={f(0): 0, f(x).diff(x).subs(x, 0): 1})
print(sol)
print(sol.rhs, sol.rhs.subs(x, 1))
```
実行結果:
```
Eq(f(x), C1*exp(x))
Eq(f(x), C1*sin(x) + C2*cos(x))
Eq(f(x), exp(x))
Eq(f(x), C1*exp(x**2/2))
Eq(f(x), C1*exp(-x) + x - 1)
Eq(f(x), exp(x)/2 - exp(-x)/2)
exp(x)/2 - exp(-x)/2 -exp(-1)/2 + E/2
```

**注意点・落とし穴**:
- 戻り値は `Eq(f(x), ...)`(等式)なので、右辺だけ欲しいときは `.rhs`。任意定数は `C1`, `C2`, ... で表される。
- 2 階の場合、微分の初期条件は `f(x).diff(x).subs(x, 0): 1` のように `subs` で評価点を指定した形で書く。

---

### `roots(f, *gens, ...)`

**用途**: 多項式の根を「{根: 重複度}」の辞書として返す。

**シグネチャ**: `roots(f, *gens, auto=True, cubics=True, trig=False, quartics=True, quintics=False, multiple=False, filter=None, predicate=None, strict=False, **flags)`

**使用例**:
```python
from sympy import symbols, roots
x = symbols('x')
print(roots(x**3 - 6*x**2 + 11*x - 6))
print(roots(x**2 - 2*x + 1))
print(roots(x**2 + 1))
print(roots(x**5 - x - 1))
```
実行結果:
```
{3: 1, 2: 1, 1: 1}
{1: 2}
{-I: 1, I: 1}
{}
```

**注意点・落とし穴**:
- 重複度が分かるのが `solve` との違い。代数的に表現できない根は辞書に含まれない(`roots(x**5 - x - 1)` は空辞書 `{}` を返す)。全部の根を数値で得たいときは `Poly(f).nroots()` を使う。

---

## 線形代数

### `Matrix(*args)`

**用途**: 厳密な要素(整数・分数・シンボル)を持つ行列/列ベクトルを作る。リストのリストから作り、`+` `*` `@` `**` などの演算や転置 `.T` が使える。

**シグネチャ**: `Matrix(*args, **kwargs)`

**使用例**:
```python
from sympy import Matrix, symbols
x, y = symbols('x y')
A = Matrix([[2, 1], [1, 2]])
B = Matrix([[1, 2, 3], [4, 5, 6]])
print(A * A)
print(A @ A)
print(A**2, A**-1)
print(B.T, B.shape)
print(B[0, 1], B[:, 1], B.row(0))
print(A.tolist())
v = Matrix([1, 2, 3])
print(v, v.shape)
print(Matrix(2, 2, [1, 2, 3, 4]))
print(Matrix(2, 2, lambda i, j: i + j))
print(Matrix([[x, y], [y, x]]).subs(x, 1))
```
実行結果:
```
Matrix([[5, 4], [4, 5]])
Matrix([[5, 4], [4, 5]])
Matrix([[5, 4], [4, 5]]) Matrix([[2/3, -1/3], [-1/3, 2/3]])
Matrix([[1, 4], [2, 5], [3, 6]]) (2, 3)
2 Matrix([[2], [5]]) Matrix([[1, 2, 3]])
[[2, 1], [1, 2]]
Matrix([[1], [2], [3]]) (3, 1)
Matrix([[1, 2], [3, 4]])
Matrix([[0, 1], [1, 2]])
Matrix([[1, y], [y, 1]])
```

**注意点・落とし穴**:
- `Matrix([1, 2, 3])` は**列ベクトル**(3x1)になる。行ベクトルにしたいときは `Matrix([[1, 2, 3]])` と二重にする。
- `*` は行列積(NumPy の要素積とは違う)。`A**-1` は逆行列。
- 要素は厳密値なので `A**-1` の要素は `2/3` のような分数のまま。数値化は `.evalf()` / `.n(桁数)`。

---

### `eye(n)` / `zeros(r, c)` / `ones(r, c)` / `diag(*values)`

**用途**: 単位行列・零行列・全 1 行列・対角行列(ブロック対角にも対応)を作る。

**シグネチャ**:
- `eye(*args, **kwargs)`
- `zeros(*args, **kwargs)`
- `ones(*args, **kwargs)`
- `diag(*values, strict=True, unpack=False, **kwargs)`

**使用例**:
```python
from sympy import eye, zeros, ones, diag, Matrix
A = Matrix([[2, 1], [1, 2]])
print(eye(3))
print(zeros(2, 3))
print(ones(2, 2))
print(diag(1, 2, 3))
print(diag(A, 5))
```
実行結果:
```
Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Matrix([[0, 0, 0], [0, 0, 0]])
Matrix([[1, 1], [1, 1]])
Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
Matrix([[2, 1, 0], [1, 2, 0], [0, 0, 5]])
```

**注意点・落とし穴**:
- `diag` に行列を渡すとブロック対角行列になる(最後の例)。

---

### `Matrix.det(method='bareiss', iszerofunc=None)`

**用途**: 行列式を計算する。シンボルを含む行列にも使える。

**シグネチャ**: `Matrix.det(method='bareiss', iszerofunc=None)`

**使用例**:
```python
from sympy import Matrix, symbols
x, y, z, t = symbols('x y z t')
print(Matrix([[2, 1], [1, 2]]).det())
print(Matrix([[x, y], [z, t]]).det())
print(Matrix([[1, 2], [2, 4]]).det())
```
実行結果:
```
3
t*x - y*z
0
```

---

### `Matrix.inv(method=None, ...)`

**用途**: 逆行列を計算する。厳密な分数・シンボル式のまま返る。

**シグネチャ**: `Matrix.inv(method=None, iszerofunc=_iszero, try_block_diag=False)`

**使用例**:
```python
from sympy import Matrix, symbols
x, y, z, t = symbols('x y z t')
print(Matrix([[2, 1], [1, 2]]).inv())
print(Matrix([[x, y], [z, t]]).inv())
try:
    Matrix([[1, 2], [2, 4]]).inv()
except Exception as ex:
    print(type(ex).__name__, ex)
```
実行結果:
```
Matrix([[2/3, -1/3], [-1/3, 2/3]])
Matrix([[t/(t*x - y*z), -y/(t*x - y*z)], [-z/(t*x - y*z), x/(t*x - y*z)]])
NonInvertibleMatrixError Matrix det == 0; not invertible.
```

**注意点・落とし穴**:
- 特異行列(行列式が 0)では `NonInvertibleMatrixError` が発生する。連立方程式を解くだけなら `inv()` を使わず `A.solve(b)` の方が効率的。

---

### `Matrix.rank()` / `Matrix.rref()` / `Matrix.nullspace()`

**用途**: 階数の計算(`rank`)、簡約階段形(`rref`: 行簡約された行列と主成分の列位置のタプルを返す)、零空間の基底(`nullspace`)。

**シグネチャ**:
- `Matrix.rank(iszerofunc=_iszero, simplify=False)`
- `Matrix.rref(iszerofunc=_iszero, simplify=False, pivots=True, normalize_last=True)`
- `Matrix.nullspace(simplify=False, iszerofunc=_iszero)`

**使用例**:
```python
from sympy import Matrix
S = Matrix([[1, 2], [2, 4]])
N = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(S.rank(), N.rank())
print(S.nullspace())
print(N.rref())
print(Matrix([[1, 2], [3, 4]]).rref())
```
実行結果:
```
1 2
[Matrix([
[-2],
[ 1]])]
(Matrix([
[1, 0, -1],
[0, 1,  2],
[0, 0,  0]]), (0, 1))
(Matrix([
[1, 0],
[0, 1]]), (0, 1))
```

**注意点・落とし穴**:
- `rref` の戻り値は `(行列, 主成分のある列の番号のタプル)` の 2 つ組。行列だけ欲しいときは `.rref()[0]`。

---

### `Matrix.eigenvals(error_when_incomplete=True, **flags)`

**用途**: 固有値を「{固有値: 代数的重複度}」の辞書で返す。

**シグネチャ**: `Matrix.eigenvals(error_when_incomplete=True, **flags)`

**使用例**:
```python
from sympy import Matrix
print(Matrix([[2, 1], [1, 2]]).eigenvals())
print(Matrix([[0, 1], [-1, 0]]).eigenvals())
print(Matrix([[2, 0], [0, 2]]).eigenvals())
```
実行結果:
```
{3: 1, 1: 1}
{-I: 1, I: 1}
{2: 2}
```

**注意点・落とし穴**:
- 値は重複度(同じ固有値が何回出るか)。`{2: 2}` は固有値 2 が 2 重解という意味。複素固有値もそのまま返る。

---

### `Matrix.eigenvects(...)` / `Matrix.diagonalize(...)`

**用途**: 固有ベクトルを `(固有値, 重複度, [固有ベクトル])` のリストで返す(`eigenvects`)/ 行列を `P`, `D` に対角化する(`diagonalize`、`A = P D P⁻¹`)。

**シグネチャ**:
- `Matrix.eigenvects(error_when_incomplete=True, iszerofunc=_iszero, **flags)`
- `Matrix.diagonalize(reals_only=False, sort=False, normalize=False)`

**使用例**:
```python
from sympy import Matrix
A = Matrix([[2, 1], [1, 2]])
for val, mult, vecs in A.eigenvects():
    print(val, mult, [list(v) for v in vecs])
P, D = A.diagonalize()
print(P, D)
print(P * D * P.inv() == A)
```
実行結果:
```
1 1 [[-1, 1]]
3 1 [[1, 1]]
Matrix([[-1, 1], [1, 1]]) Matrix([[1, 0], [0, 3]])
True
```

**注意点・落とし穴**:
- 固有ベクトルは正規化されない(単位ベクトルではない)。正規化が必要なら `v.normalized()` を呼ぶ(例: `Matrix([3, 4]).normalized()` は `Matrix([[3/5], [4/5]])`)。
- 対角化できない行列(例: `[[1, 1], [0, 1]]`)に `diagonalize` を呼ぶと `MatrixError: Matrix is not diagonalizable` になる。その場合は `jordan_form()` を検討する。

---

### `Matrix.solve(rhs, method='GJ')`

**用途**: 連立一次方程式 `A x = b` を解いて解ベクトルを返す。

**シグネチャ**: `Matrix.solve(rhs, method='GJ')`

**使用例**:
```python
from sympy import Matrix
A = Matrix([[2, 1], [1, 2]])
print(A.solve(Matrix([3, 3])))
print(A.LUsolve(Matrix([3, 3])))
print(Matrix([[1, 2], [3, 4]]).inv() * Matrix([5, 6]))
```
実行結果:
```
Matrix([[1], [1]])
Matrix([[1], [1]])
Matrix([[-4], [9/2]])
```

**注意点・落とし穴**:
- `LUsolve` は LU 分解を使う別解法。特異な行列(`[[1, 2], [2, 4]]` など)に `solve` を呼ぶと `NonInvertibleMatrixError` になるので、解が不定/なしになり得るときは `linsolve` を使う(パラメータ付きの解や `EmptySet` が得られる)。

---

### `Matrix.jacobian(X)` / `hessian(f, varlist)`

**用途**: ベクトル値関数のヤコビ行列(`jacobian`)と、スカラー関数のヘッセ行列(`hessian`)を返す。

**シグネチャ**:
- `Matrix.jacobian(X)`
- `hessian(f, varlist, constraints=())`

**使用例**:
```python
from sympy import Matrix, symbols, hessian
x, y = symbols('x y')
print(Matrix([x**2 + y, x*y]).jacobian([x, y]))
print(hessian(x**2*y, [x, y]))
```
実行結果:
```
Matrix([[2*x, 1], [y, x]])
Matrix([[2*y, 2*x], [2*x, 0]])
```

---

### NumPy との相互変換

**用途**: SymPy の `Matrix` と NumPy 配列を行き来する。大規模・数値計算は NumPy に渡した方が高速。

**シグネチャ**: なし(関数ではなくオブジェクト/属性として使う)

**使用例**:
```python
import numpy as np
from sympy import Matrix
A = Matrix([[2, 1], [1, 2]])
print(repr(np.array(A)))
print(np.array(A).dtype, np.array(A, dtype=float).dtype)
print(np.linalg.eigvals(np.array(A, dtype=float)))
print(Matrix(np.array([[1, 2], [3, 4]])))
print(type(A * np.array([1, 2])))
```
実行結果:
```
array([[2, 1],
       [1, 2]], dtype=object)
object float64
[3. 1.]
Matrix([[1, 2], [3, 4]])
<class 'numpy.ndarray'>
```

**注意点・落とし穴**:
- `np.array(A)` の dtype は `object` になり、そのまま `np.linalg.eigvals` などに渡すと `TypeError` になる。`dtype=float` を明示して変換する。
- `lambdify(x, Matrix(...))` を使えば、シンボルを含む行列を直接 NumPy 配列を返す関数にできる(「代入と評価」参照)。

---

## 集合・論理

### `Interval(start, end, left_open=False, right_open=False)`

**用途**: 実数区間を表す。`Interval.open` / `Interval.Lopen` / `Interval.Ropen` で開区間・半開区間、`oo` で無限区間を作る。

**シグネチャ**: `Interval(start, end, left_open=False, right_open=False)`

**使用例**:
```python
from sympy import Interval, oo, S
I1 = Interval(0, 5)
I2 = Interval.open(3, 8)
print(I1, I2)
print(3 in I1, I1.contains(6))
print(I1.start, I1.end, I1.measure)
print(I2.is_open, I1.left_open)
print(Interval(0, oo), Interval(-oo, oo) == S.Reals)
```
実行結果:
```
Interval(0, 5) Interval.open(3, 8)
True False
0 5 5
True False
Interval(0, oo) True
```

**注意点・落とし穴**:
- `Interval(0, 5)` は閉区間 [0, 5]。開区間は `Interval.open(0, 5)` か `left_open=True, right_open=True`。

---

### `FiniteSet(*args)` と集合演算(`Union` / `Intersection` / `Complement`)

**用途**: 有限集合を作る。`|` `&` `-` 演算子、または `Union` / `Intersection` / `Complement` で和・共通部分・差集合を計算できる。区間との混在も可。

**シグネチャ**:
- `FiniteSet(*args, **kwargs)`
- `Union(*args, **kwargs)`

**使用例**:
```python
from sympy import FiniteSet, Interval, Union, Intersection, Complement, S, pi, Rational
A = FiniteSet(1, 2, 3)
B = FiniteSet(3, 4)
print(A, A | B, A & B, A - B)
print(2 in A, A.is_subset(FiniteSet(1, 2, 3, 4)), len(A))
print(A.powerset())
print(Union(A, Interval(3, 4)))
print(FiniteSet(1, 1, 2))
I1 = Interval(0, 5); I2 = Interval.open(3, 8)
print(I1 | I2, I1 & I2, I1 - I2)
print(Union(I1, I2), Intersection(I1, I2), Complement(I1, I2))
print(S.EmptySet, S.Reals, S.Integers)
print(S.Reals.contains(pi), S.Integers.contains(Rational(1, 2)))
```
実行結果:
```
{1, 2, 3} {1, 2, 3, 4} {3} {1, 2}
True True 3
FiniteSet(EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3})
Union({1, 2}, Interval(3, 4))
{1, 2}
Interval.Ropen(0, 8) Interval.Lopen(3, 5) Interval(0, 3)
Interval.Ropen(0, 8) Interval.Lopen(3, 5) Interval(0, 3)
EmptySet Reals Integers
True False
```

**注意点・落とし穴**:
- 重複した要素は自動的に取り除かれる(`FiniteSet(1, 1, 2)` は `{1, 2}`)。
- 集合演算は可能な限り簡約されて返る(`Interval(0, 5) | Interval.open(3, 8)` は 1 つの半開区間 `Interval.Ropen(0, 8)` にまとまる)。

---

### `And` / `Or` / `Not` / `Implies` / `Xor` / `Equivalent`

**用途**: 論理式を組み立てる。`&` `|` `~` `>>` `^` の演算子でも書ける。関係式(`x > 1`)も論理式の一部として組み合わせられる。

**シグネチャ**:
- `And(*args)`
- `Or(*args)`
- `Not(arg)`
- `Implies(*args)`

**使用例**:
```python
from sympy import symbols, And, Or, Not, Implies, Xor, Equivalent
a, b, c, x = symbols('a b c x')
print(And(a, b), Or(a, b), Not(a), Implies(a, b))
print(Xor(a, b), Equivalent(a, b))
print(a & b | ~c)
print(And(a, True), Or(a, True), And(a, Not(a)))
print(Not(And(a, b)))
print(And(x > 1, x < 3))
print(And(x > 1, x < 3).subs(x, 2))
print(Or(a, b).subs(a, True))
print(Or(a, b).subs({a: False, b: False}))
```
実行結果:
```
a & b a | b ~a Implies(a, b)
a ^ b Equivalent(a, b)
~c | (a & b)
a True a & ~a
~(a & b)
(x > 1) & (x < 3)
True
True
False
```

**注意点・落とし穴**:
- `And(a, Not(a))` のような矛盾は**自動では `False` にならない**(`a & ~a` のまま)。簡約したいなら `simplify_logic`、充足可能性の判定なら `satisfiable` を使う。
- Python の `and` / `or` / `not` は SymPy の論理式に使えない(`&` `|` `~` を使う)。ただし `&` は比較演算子より優先度が高いため、`(x > 1) & (x < 3)` のように括弧が必要。

---

### `satisfiable(expr, algorithm=None, all_models=False, ...)`

**用途**: 論理式を真にする変数割り当て(モデル)が存在するかを SAT ソルバで判定する。あればその割り当ての辞書、なければ `False` を返す。

**シグネチャ**: `satisfiable(expr, algorithm=None, all_models=False, minimal=False, use_lra_theory=False)`

**使用例**:
```python
from sympy import symbols, Implies, Xor
from sympy.logic.inference import satisfiable
a, b, c = symbols('a b c')
print(satisfiable(a & b))
print(satisfiable(a & ~a))
print(list(satisfiable(a | b, all_models=True)))
print(satisfiable(Implies(a, b) & a & ~b))
print(satisfiable(Xor(a, b) & Xor(b, c) & Xor(a, c)))
```
実行結果:
```
{a: True, b: True}
False
[{a: True, b: True}, {a: True, b: False}, {b: True, a: False}]
False
False
```

**注意点・落とし穴**:
- `all_models=True` の戻り値はリストではなく**ジェネレータ**なので `list(...)` で展開する。
- `satisfiable` は `from sympy import satisfiable` でも `from sympy.logic.inference import satisfiable` でも使える。

---

### `simplify_logic(expr)` / `to_cnf(expr)` / `to_dnf(expr)`

**用途**: 論理式を最小化する(`simplify_logic`)/ 連言標準形(CNF)・選言標準形(DNF)に変換する。

**シグネチャ**:
- `simplify_logic(expr, form=None, deep=True, force=False, dontcare=None)`
- `to_cnf(expr, simplify=False, force=False)`
- `to_dnf(expr, simplify=False, force=False)`

**使用例**:
```python
from sympy import symbols, Implies
from sympy.logic.boolalg import simplify_logic, to_cnf, to_dnf
a, b, c = symbols('a b c')
print(simplify_logic((a & b) | (a & ~b)))
print(to_cnf((a & b) | c))
print(to_dnf(a & (b | c)))
print(simplify_logic(Implies(a, b)))
```
実行結果:
```
a
(a | c) & (b | c)
(a & b) | (a & c)
b | ~a
```

---

### 関係演算子 `Gt` / `Ge` / `Ne` と `x > 1`

**用途**: 大小・不等の関係式オブジェクト(`x > 1` は `StrictGreaterThan`)。`solve` / `solveset` / `Piecewise` の条件や論理式の構成要素になる。

**シグネチャ**: `Ne(lhs, rhs, **options)`

**使用例**:
```python
from sympy import symbols, Gt, Ge, Ne
x = symbols('x')
r = x > 1
print(r, type(r))
print(Gt(x, 1), Ge(x, 1), Ne(x, 1))
print(r.lhs, r.rhs, r.rel_op, r.reversed)
print(~r, r.subs(x, 3))
try:
    bool(x > 1)
except TypeError as ex:
    print(ex)
```
実行結果:
```
x > 1 <class 'sympy.core.relational.StrictGreaterThan'>
x > 1 x >= 1 Ne(x, 1)
x 1 > 1 < x
x <= 1 True
cannot determine truth value of Relational: x > 1
```

**注意点・落とし穴**:
- シンボルを含む関係式を `if` 文の条件に使うと `TypeError: cannot determine truth value of Relational` になる(真偽が決まらないため)。具体値を `subs` してから判定する。

---

## 数論・組合せ

### `isprime(n)`

**用途**: 整数 `n` が素数かを判定する。大きな数でも高速。

**シグネチャ**: `isprime(n)`

**使用例**:
```python
from sympy import isprime
print(isprime(97), isprime(91), isprime(2**61 - 1))
print(isprime(-7), isprime(1))
try:
    isprime(7.5)
except Exception as ex:
    print(type(ex).__name__, ex)
```
実行結果:
```
True False True
False False
ValueError 7.5 is not an integer
```

**注意点・落とし穴**:
- 負数と 1 は素数ではない(`False`)。整数以外を渡すと `ValueError`。

---

### `factorint(n, ...)`

**用途**: 整数を素因数分解し、`{素因数: 指数}` の辞書で返す。

**シグネチャ**: `factorint(n, limit=None, use_trial=True, use_rho=True, use_pm1=True, use_ecm=True, verbose=False, visual=None, multiple=False)`

**使用例**:
```python
from sympy import factorint
print(factorint(360))
print(factorint(2**64 + 1))
print(factorint(360, multiple=True))
print(factorint(360, visual=True))
print(factorint(1), factorint(97), factorint(-12))
```
実行結果:
```
{2: 3, 3: 2, 5: 1}
{274177: 1, 67280421310721: 1}
[2, 2, 2, 3, 3, 5]
2**3*3**2*5**1
{} {97: 1} {2: 2, 3: 1, -1: 1}
```

**注意点・落とし穴**:
- `multiple=True` で `[2, 2, 2, 3, 3, 5]` のようなフラットなリスト、`visual=True` で `2**3*3**2*5**1` という式にできる。
- `1` は空辞書 `{}`、負数は `-1: 1` を含む辞書になる。

---

### `primerange(a, b=None)` / `nextprime(n, ith=1)` / `prevprime(n)` / `prime(n)`

**用途**: 範囲内の素数の列挙(`primerange`)、次/前の素数、`n` 番目の素数(`prime`)を得る。

**シグネチャ**:
- `primerange(a, b=None)`
- `nextprime(n, ith=1)`
- `prevprime(n)`
- `prime(nth)`

**使用例**:
```python
from sympy import primerange, nextprime, prevprime, prime
print(list(primerange(10, 50)))
print(list(primerange(2, 7)), list(primerange(2, 5)))
print(list(primerange(7)))
print(nextprime(100), nextprime(100, 2), prevprime(100))
print(prime(10))
```
実行結果:
```
[11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
[2, 3, 5] [2, 3]
[2, 3, 5]
101 103 97
29
```

**注意点・落とし穴**:
- `primerange(a, b)` は `[a, b)` の**半開区間**(`b` 自身は含まない。`primerange(2, 5)` は `[2, 3]`)。`primerange(7)` のように 1 引数なら `[2, 7)` の意味。
- `prime(10)` は 10 番目の素数(29)。`nextprime(n, ith=2)` は `n` より大きい 2 番目の素数。

---

### `divisors(n, ...)` / `totient(n)`

**用途**: 約数の一覧(`divisors`)、オイラーのトーシェント関数 φ(n)(`totient`)。

**シグネチャ**:
- `divisors(n, generator=False, proper=False)`
- `totient(n)`

**使用例**:
```python
from sympy import divisors, totient, divisor_count, divisor_sigma, mobius
print(divisors(28))
print(divisors(28, proper=True))
print(divisor_count(28), divisor_sigma(28))
print(totient(36), mobius(30))
```
実行結果:
```
[1, 2, 4, 7, 14, 28]
[1, 2, 4, 7, 14]
6 56
12 -1
```

**注意点・落とし穴**:
- `totient` / `divisor_sigma` / `mobius` / `primepi` などは `from sympy import ...` で使う。`from sympy.ntheory import totient` のように旧来の場所から import した関数を呼ぶと `SymPyDeprecationWarning`(1.13 から非推奨、警告メッセージに移動先が表示される)が出る。
- `proper=True` で `n` 自身を除いた真の約数だけを返す。

---

### `gcd(f, g=None, *gens)` / `lcm(f, g=None, *gens)`

**用途**: 最大公約数・最小公倍数。整数だけでなく多項式にも使える。

**シグネチャ**:
- `gcd(f, g=None, *gens, **args)`
- `lcm(f, g=None, *gens, **args)`

**使用例**:
```python
from sympy import gcd, lcm, igcd, ilcm, symbols
x = symbols('x')
print(gcd(12, 18), lcm(4, 6))
print(gcd(x**2 - 1, x**2 - x))
print(lcm(x**2 - 1, x - 1))
print(igcd(12, 18, 27), ilcm(4, 6))
```
実行結果:
```
6 12
x - 1
x**2 - 1
3 12
```

**注意点・落とし穴**:
- `igcd` / `ilcm` は Python の整数専用版で 3 個以上の引数も取れる(結果は Python の `int`)。

---

### `binomial(n, k)`

**用途**: 二項係数 C(n, k)。`n` が記号でも扱え、`expand_func` で多項式に展開できる。

**シグネチャ**: `binomial(n, k)`

**使用例**:
```python
from sympy import binomial, symbols, expand_func
x = symbols('x')
print(binomial(5, 2), binomial(100, 50))
print(binomial(x, 2), expand_func(binomial(x, 2)))
print(binomial(5, 7), binomial(-2, 3))
```
実行結果:
```
10 100891344545564193334812497256
binomial(x, 2) x*(x - 1)/2
0 -4
```

**注意点・落とし穴**:
- `k > n` なら `0`。`n` が負でも定義され、`binomial(-2, 3)` は `-4`(一般化二項係数)。

---

### `factorial(n)` / `rf(x, k)` / `ff(x, k)`

**用途**: 階乗、上昇階乗(rising factorial: x(x+1)...(x+k-1))、下降階乗(falling factorial: x(x-1)...(x-k+1))。

**シグネチャ**:
- `factorial(n)`
- `rf(x, k)`
- `ff(x, k)`

**使用例**:
```python
from sympy import factorial, symbols, rf, ff
x = symbols('x')
print(factorial(5), factorial(0), factorial(20))
print(factorial(x), factorial(x).subs(x, 4))
print(factorial(2.5))
print(factorial(-1))
print(rf(x, 3), ff(x, 3))
```
実行結果:
```
120 1 2432902008176640000
factorial(x) 24
3.32335097044784
zoo
x*(x + 1)*(x + 2) x*(x - 2)*(x - 1)
```

**注意点・落とし穴**:
- 非整数には Γ 関数で拡張された値(`factorial(2.5)` は浮動小数点)、負の整数には `zoo`(複素無限大)を返す。
- 巨大な階乗も Python の `int` と同じ任意精度整数で正確に計算される。

---

### `mod_inverse(a, m)` / `Mod(p, q)` / `crt(m, v)`

**用途**: モジュラ逆元(`mod_inverse`)、剰余(`Mod`、記号的にも使える)、中国剰余定理(`crt`)。

**シグネチャ**:
- `mod_inverse(a, m)`
- `Mod(p, q)`
- `crt(m, v, symmetric=False, check=True)`

**使用例**:
```python
from sympy import mod_inverse, Mod, symbols
from sympy.ntheory.modular import crt
x = symbols('x')
print(mod_inverse(3, 7), pow(3, -1, 7))
print(Mod(-7, 3), -7 % 3)
print(Mod(x, 3), Mod(x + 3, 3))
print(crt([3, 5, 7], [2, 3, 2]))
```
実行結果:
```
5 5
2 2
Mod(x, 3) Mod(x, 3)
(23, 105)
```

**注意点・落とし穴**:
- `crt` は `sympy.ntheory.modular` から import する(`from sympy import crt` はできない)。戻り値は `(解, 法の積)`。
- `Mod(-7, 3)` は Python の `%` と同じく非負の `2`。`Mod(x + 3, 3)` のように記号でも剰余の性質を使って簡約される。

---

## 確率(sympy.stats)

### `Normal(name, mean, std)` などの確率変数(`Uniform` / `Exponential` / `Die` / `Binomial`)

**用途**: 確率変数を作る。名前を第 1 引数に取り、パラメータは数値でもシンボルでもよい。連続分布・離散分布とも `sympy.stats` の同じ API で扱える。

**シグネチャ**:
- `Normal(name, mean, std)`
- `Uniform(name, left, right)`
- `Exponential(name, rate)`
- `Die(name, sides=6)`
- `Binomial(name, n, p, succ=1, fail=0)`

**使用例**:
```python
from sympy import symbols, Rational
from sympy.stats import Normal, Uniform, Exponential, Die, Binomial, E, variance
mu, s = symbols('mu sigma', positive=True)
X = Normal('X', 0, 1)
Z = Normal('Z', mu, s)
U = Uniform('U', 0, 1)
W = Exponential('W', 2)
D = Die('D', 6)
B = Binomial('B', 10, Rational(1, 2))
print(E(X), variance(X))
print(E(Z), variance(Z))
print(E(U), variance(U))
print(E(W), variance(W))
print(E(D), variance(D))
print(E(B), variance(B))
```
実行結果:
```
0 1
mu sigma**2
1/2 1/12
1/2 1/4
7/2 35/12
5 5/2
```

**注意点・落とし穴**:
- `Normal` の第 3 引数は**標準偏差**(分散ではない)。`Exponential` の第 2 引数は**レート**(平均の逆数)で、`Exponential('W', 2)` の期待値は `1/2`。
- パラメータをシンボルにすると `E(Z)` や `variance(Z)` が解析的な式で返る。`positive=True` を付けないと結果が複雑になることがある。

---

### `E(expr, condition=None)` / `variance(X)` / `std(X)`

**用途**: 期待値・分散・標準偏差。`expr` は確率変数の関数(`X**2`、`2*X + 1`、`X + Y` など)でもよい。

**シグネチャ**:
- `E(expr, condition=None, numsamples=None, evaluate=True, **kwargs)`
- `variance(X, condition=None, **kwargs)`
- `std(X, condition=None, **kwargs)`

**使用例**:
```python
from sympy import symbols
from sympy.stats import Normal, E, variance, std
X = Normal('X', 0, 1)
Y = Normal('Y', 2, 3)
print(E(X**2), E(2*X + 1), variance(2*X + 1))
print(E(X + Y), variance(X + Y), std(Y))
mu, s = symbols('mu sigma', positive=True)
Z = Normal('Z', mu, s)
print(E(Z**2))
```
実行結果:
```
1 1 4
2 10 3
mu**2 + sigma**2
```

**注意点・落とし穴**:
- `variance(X + Y)` は `X` と `Y` が独立(別々の `Normal` として作られた)とみなした値(`1 + 9 = 10`)を返す。従属な変数の和を扱うときは注意する。

---

### `density(X)` / `cdf(X)`

**用途**: 確率密度関数(離散なら確率質量関数)と累積分布関数。戻り値は `x` を引数に取る関数的なオブジェクトで、`density(X)(x)` のように呼んで式を得る。

**シグネチャ**:
- `density(expr, condition=None, evaluate=True, numsamples=None, **kwargs)`
- `cdf(expr, condition=None, evaluate=True, **kwargs)`

**使用例**:
```python
from sympy import symbols, Rational
from sympy.stats import Normal, Uniform, Die, Binomial, density, cdf
x = symbols('x')
X = Normal('X', 0, 1)
print(density(X))
print(density(X)(x))
print(cdf(X)(x))
U = Uniform('U', 0, 1)
print(density(U)(x))
print(cdf(U)(x))
D = Die('D', 6)
print(density(D))
print(density(D).dict)
B = Binomial('B', 3, Rational(1, 2))
print(density(B).dict)
```
実行結果:
```
NormalDistribution(0, 1)
sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))
erf(sqrt(2)*x/2)/2 + 1/2
Piecewise((1, (x >= 0) & (x <= 1)), (0, True))
Piecewise((0, x < 0), (x, x <= 1), (1, True))
DieDistribution(6)
{1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
{0: 1/8, 1: 3/8, 2: 3/8, 3: 1/8}
```

**注意点・落とし穴**:
- `density(X)` そのものを `print` するとパラメータ表現(`NormalDistribution(0, 1)`)が出るだけ。式が欲しいときは `density(X)(x)` と呼ぶ。
- 離散分布では `density(D).dict` で `{値: 確率}` の辞書が得られる。

---

### `P(condition, given_condition=None)`

**用途**: 確率変数に関する条件(`X > 0`、`Eq(D, 3)` など)が成り立つ確率を計算する。

**シグネチャ**: `P(condition, given_condition=None, numsamples=None, evaluate=True, **kwargs)`

**使用例**:
```python
from sympy import Eq, Rational
from sympy.stats import Normal, Die, Binomial, P
X = Normal('X', 0, 1)
print(P(X > 0), P(X < 1))
print(P(X < 1).evalf(6))
D = Die('D', 6)
print(P(D > 4), P(Eq(D, 3)))
D1, D2 = Die('D1'), Die('D2')
print(P(Eq(D1 + D2, 7)), P(D1 + D2 > 10))
B = Binomial('B', 10, Rational(1, 2))
print(P(B >= 8), P(Eq(B, 5)))
```
実行結果:
```
1/2 1 - erfc(sqrt(2)/2)/2
0.841345
1/3 1/6
1/6 1/12
7/128 63/256
```

**注意点・落とし穴**:
- 連続分布の確率は誤差関数 `erfc` を含む厳密式(`1 - erfc(sqrt(2)/2)/2`)で返る。数値が欲しいときは `.evalf()`。
- 確率変数の**等号**の条件は `X == 3` ではなく `Eq(X, 3)` と書く(Python の `==` は構造の比較のため)。

---

### `sample(expr, condition=None, size=(), library='scipy', numsamples=1, seed=None)`

**用途**: 確率変数から乱数サンプルを生成する(モンテカルロ法)。内部で SciPy / NumPy を使うため、`seed` を指定すれば再現できる。

**シグネチャ**: `sample(expr, condition=None, size=(), library='scipy', numsamples=1, seed=None, **kwargs)`

**使用例**:
```python
from sympy.stats import Normal, sample
X = Normal('X', 0, 1)
s = sample(X, size=(3,), seed=0)
print(s)
print(type(s))
```
実行結果:
```
[ 0.12573022 -0.13210486  0.64042265]
<class 'numpy.ndarray'>
```

**注意点・落とし穴**:
- `sample` は SymPy 単体ではなく SciPy などの数値ライブラリに依存する(`library='scipy'` が既定)。戻り値は NumPy 配列。

---

## コード生成・出力

### `latex(expr, ...)`

**用途**: 式を LaTeX 文字列に変換する。論文・スライド・Jupyter での表示に使う。

**シグネチャ**: `latex(expr, *, full_prec=False, fold_frac_powers=False, fold_func_brackets=False, fold_short_frac=None, inv_trig_style='abbreviated', itex=False, ln_notation=False, long_frac_ratio=None, mat_delim='[', mat_str=None, mode='plain', mul_symbol=None, order=None, symbol_names={}, root_notation=True, mat_symbol_style='plain', imaginary_unit='i', gothic_re_im=False, decimal_separator='period', perm_cyclic=True, parenthesize_super=True, min=None, max=None, diff_operator='d', adjoint_style='dagger', disable_split_super_sub=False)`

**使用例**:
```python
from sympy import symbols, latex, Integral, sqrt, sin, exp, Matrix, Rational, Symbol, FiniteSet, Eq
x, y = symbols('x y')
ex = x**2/(1 + sqrt(y)) + sin(x)*exp(-y)
print(latex(ex))
print(latex(Integral(sqrt(x)/(1 + x**2), (x, 0, 1))))
print(latex(Matrix([[1, 2], [3, 4]])))
print(latex(Matrix([[1, 2], [3, 4]]), mat_delim='('))
print(latex(x**2, mode='inline'))
print(latex(x**2, mode='equation'))
print(latex(ex, mul_symbol='dot'))
print(latex(Symbol('alpha_1')))
print(latex(sin(x)**2))
print(latex(FiniteSet(1, 2)), latex(Eq(x, 1)))
```
実行結果:
```
\frac{x^{2}}{\sqrt{y} + 1} + e^{- y} \sin{\left(x \right)}
\int\limits_{0}^{1} \frac{\sqrt{x}}{x^{2} + 1}\, dx
\left[\begin{matrix}1 & 2\\3 & 4\end{matrix}\right]
\left(\begin{matrix}1 & 2\\3 & 4\end{matrix}\right)
$x^{2}$
\begin{equation}x^{2}\end{equation}
\frac{x^{2}}{\sqrt{y} + 1} + e^{- y} \cdot \sin{\left(x \right)}
\alpha_{1}
\sin^{2}{\left(x \right)}
\left\{1, 2\right\} x = 1
```

**注意点・落とし穴**:
- `mode='inline'` は `$...$`、`mode='equation'` は `\begin{equation}...\end{equation}` で囲む。既定は `plain`(囲みなし)。
- シンボル名 `alpha_1` は自動的に `\alpha_{1}`(ギリシャ文字 + 添字)に変換される。

---

### `pycode(expr, **settings)`

**用途**: 式を Python コード文字列に変換する。既定では `math.` 付きの関数名が出力される。

**シグネチャ**: `pycode(expr, **settings)`

**使用例**:
```python
from sympy import symbols, pycode, sqrt, sin, exp, gamma, erf, Piecewise
x, y = symbols('x y')
print(pycode(x**2/(1 + sqrt(y)) + sin(x)*exp(-y)))
print(pycode(sqrt(x)))
print(pycode(Piecewise((x, x > 0), (0, True))))
print(pycode(gamma(x)), pycode(erf(x)))
```
実行結果:
```
x**2/(math.sqrt(y) + 1) + math.exp(-y)*math.sin(x)
math.sqrt(x)
((x) if (x > 0) else (0))
math.gamma(x) math.erf(x)
```

**注意点・落とし穴**:
- 生成されるのは文字列。関数として実行できる形に直接したいときは `lambdify` を使う方が簡単(内部で同様のプリンタによるコード生成を行っている)。

---

### `ccode(expr, assign_to=None, ...)` / `fcode` / `jscode`

**用途**: 式を C / Fortran / JavaScript のコード文字列に変換する。`assign_to` で代入文の形にでき、`Piecewise` は `if` 文に展開される。

**シグネチャ**:
- `ccode(expr, assign_to=None, standard='c99', **settings)`
- `fcode(expr, assign_to=None, **settings)`
- `jscode(expr, assign_to=None, **settings)`

**使用例**:
```python
from sympy import symbols, ccode, fcode, jscode, sqrt, sin, exp, pi, Piecewise, Rational
x, y = symbols('x y')
ex = x**2/(1 + sqrt(y)) + sin(x)*exp(-y)
print(ccode(ex))
print(ccode(ex, assign_to='r'))
print(ccode(pi*x), ccode(Rational(1, 3)*x))
print(ccode(Piecewise((x, x > 0), (0, True)), assign_to='r'))
print(fcode(ex, source_format='free'))
print(jscode(ex))
```
実行結果:
```
pow(x, 2)/(sqrt(y) + 1) + exp(-y)*sin(x)
r = pow(x, 2)/(sqrt(y) + 1) + exp(-y)*sin(x);
M_PI*x (1.0/3.0)*x
if (x > 0) {
   r = x;
}
else {
   r = 0;
}
x**2/(sqrt(y) + 1) + exp(-y)*sin(x)
Math.pow(x, 2)/(Math.sqrt(y) + 1) + Math.exp(-y)*Math.sin(x)
```

**注意点・落とし穴**:
- `ccode` は `x**2` を `pow(x, 2)` と出力する(`x*x` には展開されない)。
- `Piecewise` は `assign_to` を指定しないと三項演算子、指定すると `if/else` 文になる。

---

### `srepr(expr)`

**用途**: 式を「SymPy の内部構造がそのまま分かる」文字列表現(再構築可能な Python コード)に変換する。式の木構造を確認するデバッグ用。

**シグネチャ**: `srepr(expr, *, order=None, perm_cyclic=True)`

**使用例**:
```python
from sympy import symbols, srepr, sin, Rational, Symbol, Float, Matrix, sympify
x, y = symbols('x y')
ex = x**2/(1 + y**Rational(1, 2)) + sin(x)
print(srepr(ex))
print(srepr(Rational(1, 3)), srepr(Symbol('x', positive=True)))
print(srepr(Matrix([[1, 2]])), srepr(Float(0.5)))
from sympy import *
print(eval(srepr(ex)) == ex, sympify(srepr(ex)) == ex)
print(ex.func, ex.args)
print(ex.free_symbols)
```
実行結果:
```
Add(Mul(Pow(Symbol('x'), Integer(2)), Pow(Add(Pow(Symbol('y'), Rational(1, 2)), Integer(1)), Integer(-1))), sin(Symbol('x')))
Rational(1, 3) Symbol('x', positive=True)
MutableDenseMatrix([[Integer(1), Integer(2)]]) Float('0.5', precision=53)
True True
<class 'sympy.core.add.Add'> (x**2/(sqrt(y) + 1), sin(x))
{x, y}
```

**注意点・落とし穴**:
- `srepr` の出力は `eval` で元の式に復元できる(`eval(srepr(ex)) == ex` は `True`)。ただし `from sympy import *` 済みの環境が必要。
- 内部構造を辿るには `expr.func`(最上位の型)、`expr.args`(引数の組)、`expr.free_symbols`(自由シンボル集合)も併用する。

---

### `str(expr)` / `pretty(expr)` / `pprint(expr)`

**用途**: 式を人間が読みやすい文字列に整形して出力する。`print(expr)` は 1 行の文字列、`pprint` は 2 次元(分数・べき乗を縦に並べた形)の表示。

**シグネチャ**:
- `pretty(expr, *, order=None, full_prec='auto', use_unicode=None, wrap_line=True, num_columns=None, use_unicode_sqrt_char=True, root_notation=True, mat_symbol_style='plain', imaginary_unit='i', perm_cyclic=True)`
- `pprint(expr, **kwargs)`

**使用例**:
```python
from sympy import symbols, pretty, pprint, sqrt, sin, exp, Matrix
x, y = symbols('x y')
ex = x**2/(1 + sqrt(y)) + exp(-y)*sin(x)
print(str(ex))
print(pretty(ex, use_unicode=False))
print(pretty(Matrix([[1, 2], [3, 4]]), use_unicode=False))
pprint(x**2/(1 + y), use_unicode=False)
print(pretty(x**2/(1 + y), use_unicode=True))
```
実行結果:
```
x**2/(sqrt(y) + 1) + exp(-y)*sin(x)
    2                 
   x         -y       
--------- + e  *sin(x)
  ___                 
\/ y  + 1             
[1  2]
[    ]
[3  4]
  2  
 x   
-----
y + 1
  2  
 x   
─────
y + 1
```

**注意点・落とし穴**:
- 既定では Unicode の罫線文字を使うため、出力先(ターミナルやファイル)によっては文字化けする。`use_unicode=False` で ASCII 表示にできる。
- Jupyter では `init_printing()` を呼ぶと、`print` しなくても式が LaTeX で自動整形される。

---

### `cse(exprs, ...)`

**用途**: 複数の式に共通する部分式(common subexpression)を抽出して、`(置換のリスト, 簡約された式のリスト)` を返す。数値計算コードの生成前に重複計算を削るために使う。

**シグネチャ**: `cse(exprs, symbols=None, optimizations=None, postprocess=None, order='canonical', ignore=(), list=True)`

**使用例**:
```python
from sympy import symbols, cse, sin, cos
x, y = symbols('x y')
print(cse([x**2 + sin(x**2), x**2*y + sin(x**2)]))
print(cse(sin(x**2) + cos(x**2) + sin(x**2)*x))
```
実行結果:
```
([(x0, x**2), (x1, sin(x0))], [x0 + x1, x0*y + x1])
([(x0, x**2), (x1, sin(x0))], [x*x1 + x1 + cos(x0)])
```

**注意点・落とし穴**:
- 戻り値の 1 要素目は `[(x0, x**2), (x1, sin(x0)), ...]` という一時変数の定義列、2 要素目はそれらを使って書き換えた式のリスト(常にリスト)。
- `lambdify(..., cse=True)` としても同じ最適化を適用できる。

---

## その他

### `Poly(rep, *gens, **args)`

**用途**: 多項式オブジェクトを作る。次数・係数・数値根・多項式除算など、通常の式(`Expr`)では面倒な操作を提供する。

**シグネチャ**: `Poly(rep, *gens, **args) -> 'Poly'`

**使用例**:
```python
from sympy import Poly, symbols, div, quo, rem
x, y = symbols('x y')
p = Poly(x**3 - 2*x**2 + x - 5, x)
print(p)
print(p.degree(), p.all_coeffs(), p.LC())
print(p.eval(2), p(2))
print(p.diff(x))
print(p.nroots())
print(p.div(Poly(x - 1, x)))
print(Poly(x**2*y + y, x, y).degree(y))
print(Poly(x**2 - 1) * Poly(x + 1))
print(div(x**3 - 1, x - 1), quo(x**3 - 1, x - 1), rem(x**3 - 1, x**2 + 1))
print(p.as_expr())
```
実行結果:
```
Poly(x**3 - 2*x**2 + x - 5, x, domain='ZZ')
3 [1, -2, 1, -5] 1
-3 -3
Poly(3*x**2 - 4*x + 1, x, domain='ZZ')
[2.43342766386382, -0.21671383193191 - 1.41695094572094*I, -0.21671383193191 + 1.41695094572094*I]
(Poly(x**2 - x, x, domain='ZZ'), Poly(-5, x, domain='ZZ'))
1
Poly(x**3 + x**2 - x - 1, x, domain='ZZ')
(x**2 + x + 1, 0) x**2 + x + 1 -x - 1
x**3 - 2*x**2 + x - 5
```

**注意点・落とし穴**:
- `all_coeffs()` は最高次から順に、欠けている次数は 0 で埋めて返す。`coeffs()` は非ゼロの係数だけ。
- `nroots()` は全根の数値解(複素数含む)を返す。`roots` / `solve` で解析的に解けない高次多項式に便利。
- `Poly.div` は `(商, 余り)` の `Poly` のタプル、関数版の `div(f, g)` は式のまま `(商, 余り)` を返す。

---

### `Piecewise((expr, cond), ...)` / `Max` / `Min` / `Abs`

**用途**: 区分定義関数(`Piecewise`)と、最大・最小・絶対値を記号的に扱う関数。

**シグネチャ**:
- `Piecewise(*_args)`
- `Max(*args)`
- `Min(*args)`

**使用例**:
```python
from sympy import symbols, Piecewise, Max, Min, Abs, integrate, floor, ceiling, sign
x = symbols('x')
f = Piecewise((x, x > 0), (0, True))
print(f)
print(f.subs(x, -3), f.subs(x, 3))
print(integrate(f, (x, -1, 1)))
print(Max(1, 3), Min(x, 3), Abs(-3), floor(2.5), ceiling(2.1), sign(-2))
print(Abs(x).diff(x))
r = symbols('r', real=True)
print(Abs(r).diff(r))
```
実行結果:
```
Piecewise((x, x > 0), (0, True))
0 3
1/2
3 Min(3, x) 3 2 3 -1
(re(x)*Derivative(re(x), x) + im(x)*Derivative(im(x), x))*sign(x)/x
sign(r)
```

**注意点・落とし穴**:
- `Piecewise` の最後の条件は `True`(それ以外)にしておく。どの条件にも当てはまらない値では `nan`(未定義)になるため(`Piecewise((x, x > 0)).subs(x, -1)` は `nan`)。
- 実数と分かっていない `x` について `Abs(x).diff(x)` は複素数の `re`/`im` を含む複雑な式になる。`Symbol('x', real=True)` として宣言すると単純になる。

---
