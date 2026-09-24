# networkx 逆引き辞書

NetworkX 3.6.1 で検証済み(すべてのシグネチャ・出力は `/home/manaty/library-practicing/.venv/bin/python` 上で実際に実行して確認。検証環境: numpy 2.4.6 / scipy 1.18.1 / pandas 3.0.5 / matplotlib 3.11.1。`pygraphviz` / `pydot` / `lxml` は未インストール)。

コード例はすべて `import networkx as nx` を前提とする。シグネチャは `inspect.signature` の結果で、末尾の dispatch 用引数 `*, backend=None, **backend_kwargs` は省略している。`nx.Graph` などクラスのシグネチャは `__init__` の内省結果、`Graph.degree` / `Graph.nodes` / `Graph.edges` のようなビュー(プロパティ)は内省できないためドキュメント上の呼び出し形を記載している。描画例は matplotlib の Agg バックエンドで PNG に保存し、画像を実際に開いて確認した。

## 目次

1. [グラフの生成と基本操作](#グラフの生成と基本操作)
2. [グラフ生成器・組み込みグラフ](#グラフ生成器組み込みグラフ)
3. [ノード・エッジの問い合わせ・変換・部分グラフ](#ノードエッジの問い合わせ変換部分グラフ)
4. [最短経路](#最短経路)
5. [連結性・成分・構造](#連結性成分構造)
6. [中心性・コミュニティ検出](#中心性コミュニティ検出)
7. [走査・順序・DAG](#走査順序dag)
8. [フロー・マッチング・スパニングツリー](#フローマッチングスパニングツリー)
9. [入出力(pandas・NumPy・SciPy連携)](#入出力pandasnumpyscipy連携)
10. [描画とレイアウト](#描画とレイアウト)
11. [その他(同型性・クリーク・彩色・リンク予測)](#その他同型性クリーク彩色リンク予測)

---

## グラフの生成と基本操作

### `networkx.Graph(incoming_graph_data=None, **attr)`

**用途**: 無向グラフ(自己ループは可、平行辺は不可)を作る、NetworkX の基本クラス。ノードは None 以外の任意のハッシュ可能オブジェクト。

**シグネチャ**: `networkx.Graph(incoming_graph_data=None, **attr)`

**使用例**:
```python
import networkx as nx
G = nx.Graph(name="demo")          # **attr はグラフ属性 (G.graph) になる
G.add_edges_from([(1, 2), (2, 3), (3, 1)])
print(G)
print(G.number_of_nodes(), G.number_of_edges())
print(G.graph)
print(nx.Graph([(1, 2), (2, 3)]).edges)   # 辺リストから直接作成
G.add_edge(1, 2)                          # 同じ辺の再追加は増えない
G.add_edge(2, 1)                          # 無向なので向きも区別しない
print(G.number_of_edges())
```
実行結果:
```
Graph named 'demo' with 3 nodes and 3 edges
3 3
{'name': 'demo'}
[(1, 2), (2, 3)]
3
```

**注意点・落とし穴**:
- `incoming_graph_data` にはエッジリスト・別の Graph・dict-of-dicts・NumPy/SciPy 行列などを渡せる。シグネチャは `Graph.__init__` の内省結果(`nx.Graph` 自体を `inspect.signature` すると dispatch 用の `(*args, backend=None, **kwargs)` が返る)。
- `G.add_edge(1, 2)` を繰り返しても辺は1本のまま。属性が異なる場合は後から渡した値で上書き(更新)される。同じノード対に複数の辺が必要なら `MultiGraph` を使う。

---

### `networkx.DiGraph(incoming_graph_data=None, **attr)`

**用途**: 有向グラフ。辺に向きがあり、`successors` / `predecessors` / `in_degree` / `out_degree` が使える。

**シグネチャ**: `networkx.DiGraph(incoming_graph_data=None, **attr)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph([(1, 2), (2, 3), (3, 1), (1, 3)])
print(D.has_edge(1, 2), D.has_edge(2, 1))   # 向きが区別される
print(D.number_of_edges())
print(list(D.successors(1)), list(D.predecessors(1)))
print(D.in_degree(3), D.out_degree(3))
print(D.to_undirected().number_of_edges())  # 3->1 と 1->3 は無向化で1本にまとまる
```
実行結果:
```
True False
4
[2, 3] [3]
2 1
3
```

**注意点・落とし穴**:
- `D.edges` の各要素は `(始点, 終点)`。無向グラフの `neighbors` に相当するものは有向では `successors`(`neighbors` は `successors` と同じ)。
- `DiGraph.degree` は入次数+出次数の合計を返す。

---

### `networkx.MultiGraph / MultiDiGraph`

**用途**: 同じノード対に複数の辺(平行辺)を持てるグラフ。各辺は `(u, v, key)` で識別される。`MultiDiGraph` は有向版。

**シグネチャ**:
- `networkx.MultiGraph(incoming_graph_data=None, multigraph_input=None, **attr)`
- `networkx.MultiGraph.add_edge(self, u_for_edge, v_for_edge, key=None, **attr)`

**使用例**:
```python
import networkx as nx
M = nx.MultiGraph()
k1 = M.add_edge("A", "B", weight=1)
k2 = M.add_edge("A", "B", weight=5)      # 平行辺: 別のキーで追加される
print(k1, k2, M.number_of_edges())
print(list(M.edges(keys=True, data=True)))
print(M["A"]["B"])                         # {key: 属性dict}
print(nx.Graph(M).edges(data=True))        # Graph に変換すると平行辺が1本にまとまる
```
実行結果:
```
0 1 2
[('A', 'B', 0, {'weight': 1}), ('A', 'B', 1, {'weight': 5})]
{0: {'weight': 1}, 1: {'weight': 5}}
[('A', 'B', {'weight': 5})]
```

**注意点・落とし穴**:
- `Graph` と違い `M[u][v]` は「キー→属性dict」の辞書になる(`G[u][v]` は属性dictそのもの)。辺の取得・削除は `M.edges[u, v, key]` / `M.remove_edge(u, v, key)` とキーが必要。
- `nx.Graph(M)` で単純グラフに変換すると、平行辺のうち最後に追加したものの属性が残り、情報が失われる(上の出力は weight=5 のみ)。`dijkstra_path_length` などは MultiGraph でも動き、平行辺のうち最小の重みが使われる(上の `M` なら 1)。

---

### `networkx.Graph.add_node / add_nodes_from`

**用途**: ノードを1つ、または複数まとめて追加する。ノード属性はキーワード引数で同時に付けられる。

**シグネチャ**:
- `networkx.Graph.add_node(self, node_for_adding, **attr)`
- `networkx.Graph.add_nodes_from(self, nodes_for_adding, **attr)`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_node("a", color="red")
G.add_nodes_from(["b", "c"], color="blue")            # 全ノードに同じ属性
G.add_nodes_from([("d", {"color": "green"}), ("e", {"size": 3})])  # (ノード, 属性dict) のタプル
print(G.nodes(data=True))
G.add_node("a", size=10)                              # 既存ノードには属性が追加・更新される
print(G.nodes["a"])
G.add_nodes_from(range(2))                            # 任意の iterable(整数・文字列・タプルも可)
print(list(G.nodes))
```
実行結果:
```
[('a', {'color': 'red'}), ('b', {'color': 'blue'}), ('c', {'color': 'blue'}), ('d', {'color': 'green'}), ('e', {'size': 3})]
{'color': 'red', 'size': 10}
['a', 'b', 'c', 'd', 'e', 0, 1]
```

**注意点・落とし穴**:
- 文字列を `add_nodes_from` に直接渡すと1文字ずつのノードとして追加される(`add_nodes_from('abc')` はノード `a`,`b`,`c`)。1つの文字列ノードなら `add_node` を使う。
- `add_node` を既存ノードに対して呼んでもノードは重複せず、属性だけが更新される。存在しないノードを `add_edge` に渡した場合は自動的にノードが作られる。

---

### `networkx.Graph.add_edge / add_edges_from`

**用途**: 辺を1本、または複数まとめて追加する。端点のノードが未登録なら自動で作られる。辺属性(重みなど)はキーワード引数で指定。

**シグネチャ**:
- `networkx.Graph.add_edge(self, u_of_edge, v_of_edge, **attr)`
- `networkx.Graph.add_edges_from(self, ebunch_to_add, **attr)`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_edge("A", "B", weight=2.5)
G.add_edges_from([("B", "C"), ("C", "D")], weight=1.0)     # 全辺に同じ属性
G.add_edges_from([("D", "E", {"weight": 4, "label": "x"})])  # (u, v, 属性dict)
print(list(G.edges(data=True)))
print(sorted(G.nodes))
```
実行結果:
```
[('A', 'B', {'weight': 2.5}), ('B', 'C', {'weight': 1.0}), ('C', 'D', {'weight': 1.0}), ('D', 'E', {'weight': 4, 'label': 'x'})]
['A', 'B', 'C', 'D', 'E']
```

**注意点・落とし穴**:
- 3要素タプルの3番目は「属性の dict」でなければならない。重みの数値をそのまま入れる `(u, v, 2.5)` は `add_edges_from` では `TypeError: 'float' object is not iterable`(重み付きタプルは `add_weighted_edges_from` を使う)。

---

### `networkx.Graph.add_weighted_edges_from`

**用途**: `(u, v, 重み)` のタプル列から、重み付きの辺をまとめて追加する。

**シグネチャ**: `networkx.Graph.add_weighted_edges_from(self, ebunch_to_add, weight='weight', **attr)`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([("A", "B", 4), ("B", "C", 2), ("A", "C", 9)])
print(list(G.edges(data=True)))
print(G["A"]["B"]["weight"])
H = nx.Graph()
H.add_weighted_edges_from([(1, 2, 0.5)], weight="cost")   # 属性名を変更
print(H[1][2])
```
実行結果:
```
[('A', 'B', {'weight': 4}), ('A', 'C', {'weight': 9}), ('B', 'C', {'weight': 2})]
4
{'cost': 0.5}
```

**注意点・落とし穴**:
- 属性名は既定で `'weight'`。最短経路(`dijkstra_path` など)・MST・PageRank などは既定でこの名前 `'weight'` を読む。別名(`'cost'` など)で持つ場合は、各関数の `weight='cost'` 引数で指定する必要がある。

---

### `networkx.Graph.remove_node / remove_edge / remove_nodes_from`

**用途**: ノード・辺を削除する。ノードを削除するとそのノードにつながる辺も一緒に消える。

**シグネチャ**:
- `networkx.Graph.remove_node(self, n)`
- `networkx.Graph.remove_edge(self, u, v)`
- `networkx.Graph.remove_nodes_from(self, nodes)`

**使用例**:
```python
import networkx as nx
G = nx.path_graph(5)              # 0-1-2-3-4
G.remove_node(2)                  # 2 につながる辺 (1,2),(2,3) も消える
print(list(G.edges), list(G.nodes))
G.remove_edge(3, 4)
print(list(G.edges))
G.remove_nodes_from([0, 99])      # 存在しないノードは黙って無視される
print(list(G.nodes))
try:
    G.remove_node(99)
except nx.NetworkXError as e:
    print("NetworkXError:", e)
try:
    G.remove_edge(0, 1)
except nx.NetworkXError as e:
    print("NetworkXError:", e)
```
実行結果:
```
[(0, 1), (3, 4)] [0, 1, 3, 4]
[(0, 1)]
[1, 3, 4]
NetworkXError: The node 99 is not in the graph.
NetworkXError: The edge 0-1 is not in the graph
```

**注意点・落とし穴**:
- `remove_node` / `remove_edge` は存在しない対象を指定すると `NetworkXError` を送出する。`remove_nodes_from` は存在しないノードを無視する。
- `for n in G.nodes: G.remove_node(n)` のようにノードを反復しながら削除すると `RuntimeError`(反復中の辞書サイズ変更)になる。`list(G.nodes)` のコピーを回すか、`remove_nodes_from` でまとめて削除する。

---

### `networkx.Graph.nodes[n] / networkx.Graph.edges[u, v]  (属性アクセス)`

**用途**: ノード・辺に付けた属性(dict)を読み書きする基本のアクセス方法。グラフ全体の属性は `G.graph`。

**シグネチャ**:
- `networkx.Graph.nodes  # NodeView(プロパティ)。n を指定して属性 dict を取得: G.nodes[n]`
- `networkx.Graph.edges  # EdgeView(プロパティ)。G.edges[u, v] で属性 dict を取得`

**使用例**:
```python
import networkx as nx
G = nx.Graph(title="road")
G.add_node("A", pop=100)
G.add_edge("A", "B", length=3.2)
G.nodes["A"]["pop"] += 50               # 属性の更新
G.edges["A", "B"]["length"] = 4.0       # G["A"]["B"]["length"] でも同じ
G.nodes["B"]["kind"] = "town"           # 後から属性を追加
print(dict(G.nodes(data=True)))
print(G["A"]["B"], G.graph)
print(G.nodes(data="pop"))              # data="属性名" で1属性だけ取り出す
print(list(G.nodes(data="pop", default=0)))   # 属性がないノードの既定値
print(list(G.edges(data="length")))
```
実行結果:
```
{'A': {'pop': 150}, 'B': {'kind': 'town'}}
{'length': 4.0} {'title': 'road'}
[('A', 150), ('B', None)]
[('A', 150), ('B', 0)]
[('A', 'B', 4.0)]
```

**注意点・落とし穴**:
- `G.nodes(data='pop')` は属性を持たないノードには `None`(または `default=` の値)を返す。
- 属性 dict は共有参照。`G.nodes['A']` の返り値を書き換えると元のグラフが変わる。

---

### `networkx.set_node_attributes / get_node_attributes`

**用途**: 辞書からノード属性を一括設定し、属性名を指定して `{ノード: 値}` の辞書として取り出す。

**シグネチャ**:
- `networkx.set_node_attributes(G, values, name=None)`
- `networkx.get_node_attributes(G, name, default=None)`

**使用例**:
```python
import networkx as nx
G = nx.path_graph(4)
nx.set_node_attributes(G, {0: "red", 1: "blue"}, name="color")  # 辞書の一部のノードだけ設定
nx.set_node_attributes(G, 0, name="score")                       # 全ノードに同じ値
nx.set_node_attributes(G, {2: {"color": "green", "x": 1}})       # 辞書のdict: name省略時は属性dictを更新
print(nx.get_node_attributes(G, "color"))
print(nx.get_node_attributes(G, "color", default="none"))
print(dict(G.nodes(data=True)))
```
実行結果:
```
{0: 'red', 1: 'blue', 2: 'green'}
{0: 'red', 1: 'blue', 2: 'green', 3: 'none'}
{0: {'color': 'red', 'score': 0}, 1: {'color': 'blue', 'score': 0}, 2: {'score': 0, 'color': 'green', 'x': 1}, 3: {'score': 0}}
```

**注意点・落とし穴**:
- `get_node_attributes` は、その属性を持つノードだけを返す(持たないノードはキーごと省略)。`default=` を指定するとその値で補完される。
- `name` を省略する場合、`values` は `{ノード: {属性名: 値}}` の形でなければならない。

---

### `networkx.set_edge_attributes / get_edge_attributes`

**用途**: 辞書から辺属性を一括設定し、属性名を指定して `{(u, v): 値}` の辞書として取り出す。

**シグネチャ**:
- `networkx.set_edge_attributes(G, values, name=None)`
- `networkx.get_edge_attributes(G, name, default=None)`

**使用例**:
```python
import networkx as nx
G = nx.path_graph(4)
nx.set_edge_attributes(G, {(0, 1): 5, (1, 2): 3}, name="weight")
nx.set_edge_attributes(G, "road", name="kind")                    # 全辺に同じ値
print(nx.get_edge_attributes(G, "weight"))
print(nx.get_edge_attributes(G, "weight", default=1))
print(list(G.edges(data=True)))
H = nx.Graph()
H.add_edge("b", "a", w=1)
print(nx.get_edge_attributes(H, "w"))     # キーは G.edges が返す向き
```
実行結果:
```
{(0, 1): 5, (1, 2): 3}
{(0, 1): 5, (1, 2): 3, (2, 3): 1}
[(0, 1, {'weight': 5, 'kind': 'road'}), (1, 2, {'weight': 3, 'kind': 'road'}), (2, 3, {'kind': 'road'})]
{('b', 'a'): 1}
```

**注意点・落とし穴**:
- `MultiGraph` では辺キーが `(u, v, key)` の3要素になる。
- 無向グラフでも `get_edge_attributes` が返すキーは `G.edges` が報告する向き(ノードの登録順に依存)になる。上の `H` は `('b', 'a')` で返り、`('a', 'b')` では引けない。

---

## グラフ生成器・組み込みグラフ

### `networkx.complete_graph(n)`

**用途**: n 個のノード全てが互いにつながった完全グラフ K_n を作る。

**シグネチャ**: `networkx.complete_graph(n, create_using=None)`

**使用例**:
```python
import networkx as nx
K4 = nx.complete_graph(4)
print(K4.number_of_nodes(), K4.number_of_edges())
print(list(K4.edges))
print(nx.complete_graph(["a", "b", "c"]).edges)       # n にノード列も渡せる
print(nx.complete_graph(3, create_using=nx.DiGraph).number_of_edges())  # 有向: 3*2=6
```
実行結果:
```
4 6
[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
[('a', 'b'), ('a', 'c'), ('b', 'c')]
6
```

**注意点・落とし穴**:
- `n` が整数なら、ノードは `0..n-1`。iterable を渡すとその要素がノードになる。
- `create_using=nx.DiGraph` のように「クラス」または「インスタンス」で型を指定する。他の生成器(`path_graph` など)でも共通の引数。

---

### `networkx.path_graph / cycle_graph / star_graph(n)`

**用途**: 直線状(パス)・環状(サイクル)・スター(中心+n個の葉)の基本形グラフを作る。

**シグネチャ**:
- `networkx.path_graph(n, create_using=None)`
- `networkx.cycle_graph(n, create_using=None)`
- `networkx.star_graph(n, create_using=None)`

**使用例**:
```python
import networkx as nx
print(list(nx.path_graph(4).edges))
print(list(nx.cycle_graph(4).edges))
S = nx.star_graph(4)                 # 中心 0 と葉 1..4 の計 5 ノード
print(S.number_of_nodes(), list(S.edges))
print(nx.path_graph(["x", "y", "z"]).edges)
```
実行結果:
```
[(0, 1), (1, 2), (2, 3)]
[(0, 1), (0, 3), (1, 2), (2, 3)]
5 [(0, 1), (0, 2), (0, 3), (0, 4)]
[('x', 'y'), ('y', 'z')]
```

**注意点・落とし穴**:
- `star_graph(n)` のノード数は `n+1`(中心1個 + 葉 n 個)。`path_graph(n)` / `cycle_graph(n)` のノード数は `n`。

---

### `networkx.grid_2d_graph(m, n)`

**用途**: m 行 n 列の格子グラフ。ノードは `(行, 列)` のタプル。`periodic=True` でトーラス(端がつながる)になる。

**シグネチャ**: `networkx.grid_2d_graph(m, n, periodic=False, create_using=None)`

**使用例**:
```python
import networkx as nx
G = nx.grid_2d_graph(2, 3)
print(list(G.nodes))
print(G.number_of_edges())
print(sorted(G.neighbors((0, 1))))
T = nx.grid_2d_graph(3, 3, periodic=True)
print(T.number_of_edges())
```
実行結果:
```
[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
7
[(0, 0), (0, 2), (1, 1)]
18
```

**注意点・落とし穴**:
- ノードが `(行, 列)` のタプルなので、整数ラベルにしたいときは `nx.convert_node_labels_to_integers(G)` で付け替える。
- `periodic=True` のとき辺数は m×n×2(3×3 で 18)。

---

### `networkx.erdos_renyi_graph / gnm_random_graph`

**用途**: ランダムグラフ。`erdos_renyi_graph(n, p)` は各ノード対を確率 p で結ぶ G(n,p) モデル、`gnm_random_graph(n, m)` は辺数がちょうど m 本の G(n,m) モデル。

**シグネチャ**:
- `networkx.erdos_renyi_graph(n, p, seed=None, directed=False, *, create_using=None)`
- `networkx.gnm_random_graph(n, m, seed=None, directed=False, *, create_using=None)`

**使用例**:
```python
import networkx as nx
G = nx.erdos_renyi_graph(20, 0.2, seed=42)
print(G.number_of_nodes(), G.number_of_edges())
H = nx.gnm_random_graph(20, 30, seed=42)
print(H.number_of_edges())
D = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=True)
print(D.is_directed(), D.number_of_edges())
print(nx.gnp_random_graph is nx.erdos_renyi_graph)   # 同一オブジェクト(別名)
```
実行結果:
```
20 37
30
True 26
True
```

**注意点・落とし穴**:
- 再現性のために `seed=` を必ず指定する(整数、または `random.Random` インスタンス)。省略すると実行のたびに結果が変わる。
- `gnp_random_graph` は `erdos_renyi_graph` の別名(`is` で比較すると同一オブジェクト、上の出力 True)。大きく疎なグラフには `fast_gnp_random_graph` が使える。

---

### `networkx.barabasi_albert_graph(n, m)`

**用途**: 優先的選択(次数の大きいノードほど新しい辺がつながりやすい)で成長する、スケールフリー(べき乗則)ネットワークを作る。

**シグネチャ**: `networkx.barabasi_albert_graph(n, m, seed=None, initial_graph=None, *, create_using=None)`

**使用例**:
```python
import networkx as nx
G = nx.barabasi_albert_graph(100, 2, seed=42)
print(G.number_of_nodes(), G.number_of_edges())   # 辺数は m*(n-m)
degs = sorted((d for _, d in G.degree), reverse=True)
print(degs[:5], min(degs))
```
実行結果:
```
100 196
[36, 25, 20, 13, 10] 2
```

**注意点・落とし穴**:
- 新ノードが m 本ずつ辺を持って加わるので最小次数は m。`m` は `1 <= m < n` を満たす必要がある。
- 辺数は `m*(n-m)`(この例では 2×98=196)。初期グラフは `initial_graph` で指定でき、未指定なら `star_graph(m)`(m+1 ノード)から成長する。

---

### `networkx.watts_strogatz_graph(n, k, p)`

**用途**: スモールワールドネットワーク。リング状に各ノードを近傍 k 個と結び、各辺を確率 p でランダムに付け替える。

**シグネチャ**: `networkx.watts_strogatz_graph(n, k, p, seed=None, *, create_using=None)`

**使用例**:
```python
import networkx as nx
G = nx.watts_strogatz_graph(20, 4, 0.1, seed=1)
print(G.number_of_nodes(), G.number_of_edges())     # 辺数は n*k/2
print(round(nx.average_clustering(G), 3), round(nx.average_shortest_path_length(G), 3))
R = nx.watts_strogatz_graph(20, 4, 0.0, seed=1)      # p=0 は純粋なリング格子
print(round(nx.average_clustering(R), 3), round(nx.average_shortest_path_length(R), 3))
```
実行結果:
```
20 40
0.398 2.547
0.5 2.895
```

**注意点・落とし穴**:
- `p=0` は純粋なリング格子(この例ではクラスタ係数 0.5・平均経路長 2.895)。`p` を上げると付け替えによって平均経路長が短くなる(上の例で 2.895 → 2.547)。
- 付け替え後にグラフが非連結になることがあり、その場合 `average_shortest_path_length` は `NetworkXError` になる。常に連結が保証される `connected_watts_strogatz_graph(n, k, p, tries=100)` もある。

---

### `networkx.karate_club_graph()`

**用途**: Zachary の空手クラブネットワーク(34ノード・78辺)。ノード属性 `club`('Mr. Hi' / 'Officer')付きで、コミュニティ検出や中心性の練習に使う定番の実データ。

**シグネチャ**: `networkx.karate_club_graph()`

**使用例**:
```python
import networkx as nx
G = nx.karate_club_graph()
print(G)
print(G.nodes[0], G.nodes[33])
print(G.edges[0, 1])
print(sorted(d for _, d in G.degree)[-3:])
```
実行結果:
```
Graph named "Zachary's Karate Club" with 34 nodes and 78 edges
{'club': 'Mr. Hi'} {'club': 'Officer'}
{'weight': 4}
[12, 16, 17]
```

**注意点・落とし穴**:
- ノードは整数 0..33。ほかに `les_miserables_graph`, `florentine_families_graph`, `davis_southern_women_graph` (二部グラフ)など組み込みの小さな実データ生成器がある。
- 辺には属性 `weight` が付いている(`G.edges[0, 1]` は `{'weight': 4}`)。重み付きで扱いたくない関数には `weight=None` を渡す。

---

## ノード・エッジの問い合わせ・変換・部分グラフ

### `networkx.Graph.degree / DiGraph.in_degree / out_degree`

**用途**: ノードの次数(つながっている辺の数)を返す。ノードを指定すれば整数、未指定なら `(ノード, 次数)` のビュー。`weight=` を指定すると重みの合計(強度)になる。

**シグネチャ**:
- `networkx.Graph.degree  # DegreeView。呼び出しシグネチャは DegreeView.__call__(self, nbunch=None, weight=None)`
- `networkx.DiGraph.in_degree / out_degree  # InDegreeView / OutDegreeView(同じ呼び出し形)`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([("A", "B", 2), ("A", "C", 3), ("B", "C", 5)])
print(G.degree("A"))
print(dict(G.degree))
print(dict(G.degree(weight="weight")))     # 重み付き次数
D = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
print(dict(D.in_degree), dict(D.out_degree))
print(sorted(G.degree, key=lambda x: x[1], reverse=True))
```
実行結果:
```
2
{'A': 2, 'B': 2, 'C': 2}
{'A': 5, 'B': 7, 'C': 8}
{1: 0, 2: 1, 3: 2} {1: 2, 2: 1, 3: 0}
[('A', 2), ('B', 2), ('C', 2)]
```

**注意点・落とし穴**:
- `G.degree` は括弧なしで `DegreeView`(反復すると `(ノード, 次数)`)。次数のみ欲しいときは `[d for _, d in G.degree]`、辞書は `dict(G.degree)`。
- 自己ループは次数に2として数えられる(`G.add_edge(1, 1)` → `G.degree(1)` は 2)。

---

### `networkx.Graph.nodes / Graph.edges (ビュー)`

**用途**: ノード一覧・辺一覧を取得する。`nbunch` で特定ノードに接続する辺だけ、`data=True` で属性付きで返す。返り値はグラフに連動する「ビュー」。

**シグネチャ**:
- `networkx.Graph.nodes  # NodeView。呼び出し: G.nodes(data=False, default=None)`
- `networkx.Graph.edges  # EdgeView。呼び出し: G.edges(nbunch=None, data=False, *, default=None)`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_edge("A", "B", weight=1)
G.add_edge("B", "C", weight=2)
print(list(G.nodes), len(G.nodes), "A" in G.nodes)
print(list(G.edges), len(G.edges))
print(list(G.edges("B")))                      # ノード B に接続する辺
print(list(G.edges(data="weight")))
nodes_view = G.nodes
G.add_node("Z")
print(list(nodes_view))                        # ビューはグラフの変更に追従する
```
実行結果:
```
['A', 'B', 'C'] 3 True
[('A', 'B'), ('B', 'C')] 2
[('B', 'A'), ('B', 'C')]
[('A', 'B', 1), ('B', 'C', 2)]
['A', 'B', 'C', 'Z']
```

**注意点・落とし穴**:
- `G.nodes` / `G.edges` は `set` ではなく専用のビュー。`list(G.nodes)` でリスト化する。辺はそれぞれ `(u, v)`(`data=True` なら `(u, v, dict)`)。
- 無向グラフの辺は列挙時に1本につき1回だけ、片方の向きで返される(上の例では `(2, 1) in G.edges` も真だが、列挙には `('B', 'A')` ではなく `('A', 'B')` が出る)。`G.edges('B')` のようにノードを指定すると、そのノードが先頭に来る向きで返る。

---

### `networkx.Graph.neighbors / DiGraph.successors / predecessors`

**用途**: 隣接ノード(つながっているノード)のイテレータを返す。有向グラフでは `successors`(出て行く先)と `predecessors`(入ってくる元)を区別する。

**シグネチャ**:
- `networkx.Graph.neighbors(self, n)`
- `networkx.DiGraph.successors(self, n)`
- `networkx.DiGraph.predecessors(self, n)`

**使用例**:
```python
import networkx as nx
G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4)])
print(list(G.neighbors(3)))
print(list(G[3]))                    # G[n] は隣接ノード→辺属性の辞書(AtlasView)
D = nx.DiGraph([(1, 2), (1, 3), (3, 1)])
print(list(D.successors(1)), list(D.predecessors(1)), list(D.neighbors(1)))
try:
    list(G.neighbors(99))
except nx.NetworkXError as e:
    print("NetworkXError:", e)
```
実行結果:
```
[1, 2, 4]
[1, 2, 4]
[2, 3] [3] [2, 3]
NetworkXError: The node 99 is not in the graph.
```

**注意点・落とし穴**:
- `neighbors` は list ではなく「イテレータ」。`len()` が必要なら `G.degree(n)` か `list()` で包む。
- 存在しないノードを渡すと `NetworkXError`。有向グラフで `predecessors` を使うには `DiGraph` が必要(`Graph` にはない)。

---

### `networkx.Graph.subgraph / Graph.edge_subgraph`

**用途**: 指定したノード集合(誘導部分グラフ)、または指定した辺集合から成る部分グラフを返す。返り値は元グラフの「読み取り専用ビュー」。

**シグネチャ**:
- `networkx.Graph.subgraph(self, nodes)`
- `networkx.Graph.edge_subgraph(self, edges)`

**使用例**:
```python
import networkx as nx
G = nx.karate_club_graph()
S = G.subgraph([0, 1, 2, 3, 7])          # ノード集合に対する誘導部分グラフ
print(S)
print(sorted(S.edges))
E = G.edge_subgraph([(0, 1), (0, 2)])   # 辺集合から作る
print(sorted(E.nodes), E.number_of_edges())
S2 = G.subgraph([0, 1, 2]).copy()        # 独立したグラフとして使うには copy()
S2.add_edge(0, 100)
print(G.has_edge(0, 100), S2.number_of_nodes())
try:
    S.add_node(999)
except nx.NetworkXError as e:
    print("NetworkXError:", e)
```
実行結果:
```
Graph named "Zachary's Karate Club" with 5 nodes and 10 edges
[(0, 1), (0, 2), (0, 3), (0, 7), (1, 2), (1, 3), (1, 7), (2, 3), (2, 7), (3, 7)]
[0, 1, 2] 2
False 4
NetworkXError: Frozen graph can't be modified
```

**注意点・落とし穴**:
- ビューなので、元グラフを変更すると部分グラフにも反映され、部分グラフ自体は変更できない(`add_node` 等は `NetworkXError: Frozen graph can't be modified`)。編集したいときは `.copy()`。
- `subgraph` に存在しないノードを含めても無視される(エラーにならない)。グラフ名(`name` 属性)は元グラフから引き継がれる(上の出力の `Graph named "Zachary's Karate Club" with 5 nodes`)。

---

### `networkx.ego_graph(G, n, radius=1)`

**用途**: ノード n を中心とし、距離 `radius` 以内のノードだけから成る部分グラフ(エゴネットワーク)を返す。

**シグネチャ**: `networkx.ego_graph(G, n, radius=1, center=True, undirected=False, distance=None)`

**使用例**:
```python
import networkx as nx
G = nx.karate_club_graph()
E1 = nx.ego_graph(G, 0, radius=1)
print(E1.number_of_nodes(), E1.number_of_edges())
E2 = nx.ego_graph(G, 0, radius=2)
print(E2.number_of_nodes())
E3 = nx.ego_graph(G, 0, radius=1, center=False)   # 中心ノードを除く
print(E3.number_of_nodes(), 0 in E3)
```
実行結果:
```
17 34
26
16 False
```

**注意点・落とし穴**:
- 返り値は新しいグラフ(コピー)で、元グラフのビューではない。有向グラフでは出ていく方向のみたどる(入ってくる方向も含めたいときは `undirected=True`)。

---

### `networkx.density(G) / number_of_nodes / number_of_edges`

**用途**: グラフの規模と密度(実際の辺数 / 取りうる最大辺数)を調べる。`len(G)` はノード数。

**シグネチャ**:
- `networkx.density(G)`
- `networkx.number_of_nodes(G)`
- `networkx.number_of_edges(G)`

**使用例**:
```python
import networkx as nx
G = nx.karate_club_graph()
print(len(G), G.number_of_nodes(), G.number_of_edges(), G.size())
print(round(nx.density(G), 4))
print(nx.density(nx.complete_graph(5)))
D = nx.DiGraph([(1, 2), (2, 1), (2, 3)])
print(round(nx.density(D), 4))         # 有向: m / (n*(n-1))
W = nx.Graph()
W.add_weighted_edges_from([(1, 2, 2.5), (2, 3, 4)])
print(W.size(), W.size(weight="weight"))   # size() は辺数、weight指定で重みの合計
```
実行結果:
```
34 34 78 78
0.139
1.0
0.5
2 6.5
```

**注意点・落とし穴**:
- 密度は 0(辺なし)〜1(完全グラフ)。`G.size()` は既定で辺数(`number_of_edges` と同じ)だが、`weight='weight'` を渡すと重みの総和になる点に注意。

---

### `networkx.Graph.copy / to_undirected / to_directed`

**用途**: グラフのコピー、および有向⇔無向の変換を行う。`as_view=True` ならコピーせず読み取り専用のビューを返す。

**シグネチャ**:
- `networkx.Graph.copy(self, as_view=False)`
- `networkx.Graph.to_undirected(self, as_view=False)`
- `networkx.Graph.to_directed(self, as_view=False)`

**使用例**:
```python
import networkx as nx
G = nx.Graph([(1, 2), (2, 3)])
G[1][2]["w"] = 5
H = G.copy()                       # 構造・属性dictともコピー(浅い属性コピー)
H.add_edge(3, 4)
print(G.number_of_edges(), H.number_of_edges())
D = G.to_directed()                # 無向辺1本 -> 双方向の有向辺2本
print(sorted(D.edges(data=True)))
U = nx.DiGraph([(1, 2), (2, 1), (2, 3)]).to_undirected()
print(sorted(U.edges))
R = nx.DiGraph([(1, 2), (2, 1)]).to_undirected(reciprocal=True)
print(sorted(R.edges))
```
実行結果:
```
2 3
[(1, 2, {'w': 5}), (2, 1, {'w': 5}), (2, 3, {}), (3, 2, {})]
[(1, 2), (2, 3)]
[(1, 2)]
```

**注意点・落とし穴**:
- `copy()` の属性値は「浅い」コピー。属性値がリストなどの可変オブジェクトの場合、コピー元と共有される。完全に独立させたければ `copy.deepcopy(G)`。
- 有向→無向の `to_undirected()` は、双方向の辺(u→v と v→u)を1本にまとめるとき、辺の列挙順で後になる方の属性が残る。相互辺のみ残すには `reciprocal=True`(上の例では `(1, 2)` のみ)。

---

### `networkx.relabel_nodes(G, mapping)`

**用途**: ノードのラベルを付け替える。`mapping` は辞書または関数。`convert_node_labels_to_integers` は 0(または `first_label`)からの連番に変換する。

**シグネチャ**:
- `networkx.relabel_nodes(G, mapping, copy=True)`
- `networkx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)`

**使用例**:
```python
import networkx as nx
G = nx.Graph([("a", "b"), ("b", "c")])
H = nx.relabel_nodes(G, {"a": "A", "b": "B"})        # 一部だけ変更(未指定のノードはそのまま)
print(sorted(H.nodes, key=str))
K = nx.relabel_nodes(G, str.upper)                    # 関数も可
print(sorted(K.nodes))
I = nx.convert_node_labels_to_integers(G, first_label=1, label_attribute="orig")
print(dict(I.nodes(data=True)))
nx.relabel_nodes(G, {"a": 0}, copy=False)             # in-place で変更
print(sorted(G.nodes, key=str))
```
実行結果:
```
['A', 'B', 'c']
['A', 'B', 'C']
{1: {'orig': 'a'}, 2: {'orig': 'b'}, 3: {'orig': 'c'}}
[0, 'b', 'c']
```

**注意点・落とし穴**:
- `copy=False`(in-place)でラベルの入れ替え(`{1: 2, 2: 1}` のような循環)を行うと `NetworkXUnfeasible: The node label sets are overlapping and no ordering can resolve the mapping. Use copy=True.` になる。`copy=True`(既定)なら入れ替えも可能。既存ノードと同じラベルへ付け替えると、そのノードと統合される。
- `label_attribute='orig'` を付けて整数化すると、元のラベルが各ノードの属性 `orig` に残るので、あとで元の名前に戻せる。

---

### `networkx.compose / union / disjoint_union`

**用途**: 2つのグラフを合成する。`compose` は共通ノードを同一視して統合、`union` はノード集合が互いに素であることが前提、`disjoint_union` は自動で 0.. の連番に振り直して連結する。

**シグネチャ**:
- `networkx.compose(G, H)`
- `networkx.union(G, H, rename=())`
- `networkx.disjoint_union(G, H)`

**使用例**:
```python
import networkx as nx
G = nx.Graph([(1, 2), (2, 3)])
H = nx.Graph([(3, 4), (4, 5)])
C = nx.compose(G, H)                       # ノード 3 が共通なので同一視される
print(sorted(C.edges))
G2 = nx.Graph([(1, 2)])
U = nx.union(G2, nx.Graph([(1, 2)]), rename=("g-", "h-"))   # 名前衝突を接頭辞で回避
print(sorted(U.nodes))
D = nx.disjoint_union(G2, nx.Graph([(1, 2)]))
print(sorted(D.nodes), sorted(D.edges))
try:
    nx.union(G2, G2)
except nx.NetworkXError as e:
    print("NetworkXError:", str(e).splitlines()[0])
```
実行結果:
```
[(1, 2), (2, 3), (3, 4), (4, 5)]
['g-1', 'g-2', 'h-1', 'h-2']
[0, 1, 2, 3] [(0, 1), (2, 3)]
NetworkXError: The node sets of the graphs are not disjoint.
```

**注意点・落とし穴**:
- `union` は重複ノードがあると `NetworkXError`(`rename=` か `disjoint_union` を使う)。属性が衝突する場合、`compose` は後の引数 `H` の属性が優先される。
- 複数グラフには `nx.compose_all([G1, G2, ...])` / `nx.union_all` / `nx.disjoint_union_all` がある。

---

## 最短経路

### `networkx.shortest_path(G, source, target, weight)`

**用途**: 最短経路(ノード列)と最短経路長を求める。`weight=None` なら辺数(ホップ数)、`weight='属性名'` なら重み付きの最短(既定の method は dijkstra)。source/target の省略で単一始点・全点対の一括計算になる。

**シグネチャ**:
- `networkx.shortest_path(G, source=None, target=None, weight=None, method='dijkstra')`
- `networkx.shortest_path_length(G, source=None, target=None, weight=None, method='dijkstra')`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([("A", "B", 1), ("B", "C", 1), ("A", "C", 5), ("C", "D", 2)])
print(nx.shortest_path(G, "A", "D"))                    # ホップ数最小(A-C-D)
print(nx.shortest_path(G, "A", "D", weight="weight"))   # 重み最小(A-B-C-D)
print(nx.shortest_path_length(G, "A", "D"), nx.shortest_path_length(G, "A", "D", weight="weight"))
print(nx.shortest_path(G, source="A"))                  # target省略: Aから全ノードへ
print(dict(nx.shortest_path_length(G, weight="weight")).get("D"))   # 全点対: 各始点の dict
```
実行結果:
```
['A', 'C', 'D']
['A', 'B', 'C', 'D']
2 4
{'A': ['A'], 'B': ['A', 'B'], 'C': ['A', 'C'], 'D': ['A', 'C', 'D']}
{'D': 0, 'C': 2, 'B': 3, 'A': 4}
```

**注意点・落とし穴**:
- `weight` の既定値は `None`(重みを無視してホップ数)。重み付きで求めたいときは必ず `weight='weight'` を明示する(`dijkstra_path` は既定で `weight='weight'`)。
- 始点から終点へ到達できないと `NetworkXNoPath`、存在しないノードは `NodeNotFound`。戻り値の型は引数で変わる: 両方指定=経路(list)/長さ、`source` のみ・`target` のみ=`dict`、両方省略=`(始点, dict)` を返すジェネレータ(`dict(...)` で辞書化できる)。
- 既定の dijkstra は負の重みを扱えない(誤った値を返すことがある。詳細は `bellman_ford_path` の項)。負の重みがあるときは `method='bellman-ford'` を指定する。

---

### `networkx.dijkstra_path / dijkstra_path_length / single_source_dijkstra`

**用途**: 重み付きグラフ(非負の重み)の最短経路・最短距離をダイクストラ法で求める。`single_source_dijkstra` は始点から全ノード(または target まで)の距離と経路を一度に返す。

**シグネチャ**:
- `networkx.dijkstra_path(G, source, target, weight='weight')`
- `networkx.dijkstra_path_length(G, source, target, weight='weight')`
- `networkx.single_source_dijkstra(G, source, target=None, cutoff=None, weight='weight')`
- `networkx.single_source_dijkstra_path_length(G, source, cutoff=None, weight='weight')`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([("A", "B", 4), ("A", "C", 1), ("C", "B", 2), ("B", "D", 5)])
print(nx.dijkstra_path(G, "A", "D"))
print(nx.dijkstra_path_length(G, "A", "D"))
dist, path = nx.single_source_dijkstra(G, "A")            # 全ノードへの距離と経路
print(dist)
print(path["D"])
d, p = nx.single_source_dijkstra(G, "A", target="D")      # (距離, 経路) のタプル
print(d, p)
print(nx.single_source_dijkstra_path_length(G, "A", cutoff=3))   # 距離3以内のノードのみ
```
実行結果:
```
['A', 'C', 'B', 'D']
8
{'A': 0, 'C': 1, 'B': 3, 'D': 8}
['A', 'C', 'B', 'D']
8 ['A', 'C', 'B', 'D']
{'A': 0, 'C': 1, 'B': 3}
```

**注意点・落とし穴**:
- `shortest_path` と違い、`weight` の既定値が `'weight'`。属性名が異なる場合は必ず指定する。属性が欠けた辺の重みは 1 として扱われる。
- `target` を指定した `single_source_dijkstra` は `(距離, 経路)` のタプル、省略すると `(距離のdict, 経路のdict)`。`weight` に「辺の属性 dict を受け取る関数」も渡せる(例 `weight=lambda u, v, d: d['w'] * 2`)。

---

### `networkx.bellman_ford_path / bellman_ford_path_length / negative_edge_cycle`

**用途**: 負の重みを持つ辺があってもよい最短経路(ベルマン・フォード法)。負の閉路(辺をたどるたびに長さが減り続ける閉路)があると最短経路が定義できず `NetworkXUnbounded`。`negative_edge_cycle` で事前検出できる。

**シグネチャ**:
- `networkx.bellman_ford_path(G, source, target, weight='weight')`
- `networkx.bellman_ford_path_length(G, source, target, weight='weight')`
- `networkx.negative_edge_cycle(G, weight='weight', heuristic=True)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph()
D.add_weighted_edges_from([(1, 2, 1), (1, 3, 2), (3, 2, -5)])
print(nx.dijkstra_path_length(D, 1, 2))         # ダイクストラは誤答(1)
print(nx.bellman_ford_path_length(D, 1, 2))     # 正しくは 2-5 = -3
print(nx.bellman_ford_path(D, 1, 2))
try:
    nx.single_source_dijkstra(D, 1)
except ValueError as e:
    print("ValueError:", e)
N = nx.DiGraph()
N.add_weighted_edges_from([(1, 2, 1), (2, 3, -3), (3, 1, 1)])
print(nx.negative_edge_cycle(N))
try:
    nx.bellman_ford_path(N, 1, 3)
except nx.NetworkXUnbounded as e:
    print("NetworkXUnbounded:", e)
```
実行結果:
```
1
-3
[1, 3, 2]
ValueError: ('Contradictory paths found:', 'negative weights?')
True
NetworkXUnbounded: Negative cycle detected.
```

**注意点・落とし穴**:
- ダイクストラ系(`dijkstra_*`)に負の重みを入れると、(1)矛盾を検出して `ValueError: ('Contradictory paths found:', 'negative weights?')` になる場合と、(2)上の例のように何もエラーが出ず誤った値(1)を返す場合がある。負の重みが混じる可能性があるなら `bellman_ford_*` を使う。
- **無向グラフ**の負の重みの辺は、往復するだけで負の閉路になる。負の重みは有向グラフで使うこと。

---

### `networkx.astar_path(G, source, target, heuristic)`

**用途**: A* 探索による最短経路。`heuristic(u, v)` に「u から target までの距離の見積もり」を与えると、探索範囲を目的地の方向に絞り込める。

**シグネチャ**:
- `networkx.astar_path(G, source, target, heuristic=None, weight='weight', *, cutoff=None)`
- `networkx.astar_path_length(G, source, target, heuristic=None, weight='weight', *, cutoff=None)`

**使用例**:
```python
import networkx as nx
G = nx.grid_2d_graph(5, 5)                          # ノードは (行, 列)
G.remove_nodes_from([(1, 1), (1, 2), (1, 3), (2, 1)])   # 障害物
def manhattan(u, v):
    return abs(u[0] - v[0]) + abs(u[1] - v[1])
path = nx.astar_path(G, (0, 0), (4, 4), heuristic=manhattan, weight=None)
print(path)
print(nx.astar_path_length(G, (0, 0), (4, 4), heuristic=manhattan, weight=None))
print(nx.shortest_path_length(G, (0, 0), (4, 4)))
```
実行結果:
```
[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
8
8
```

**注意点・落とし穴**:
- ヒューリスティックは真の距離を「過大評価しない」(admissible)ものでなければ最短性が保証されない。`heuristic=None` ならダイクストラと同じ動作になる。
- `weight` の既定は `'weight'` で、属性がない辺は重み 1 と見なされる。上の例のように `weight=None` を渡すと全辺を重み 1 として扱う(格子など重みなしグラフ向け)。

---

### `networkx.all_pairs_dijkstra / floyd_warshall / all_pairs_shortest_path_length`

**用途**: 全ノード対の最短距離を一度に求める。`all_pairs_dijkstra` は(始点, (距離dict, 経路dict))を返すジェネレータ、`floyd_warshall` は密なグラフ向け(距離のみ)、`johnson` は負の重みがあるスパースなグラフの全点対経路。

**シグネチャ**:
- `networkx.all_pairs_dijkstra(G, cutoff=None, weight='weight')`
- `networkx.floyd_warshall(G, weight='weight')`
- `networkx.all_pairs_shortest_path_length(G, cutoff=None)`
- `networkx.johnson(G, weight='weight')`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([(0, 1, 1), (1, 2, 2), (0, 2, 5)])
for src, (dist, paths) in nx.all_pairs_dijkstra(G):
    print(src, dist, paths[2])
fw = nx.floyd_warshall(G)
print(fw[0][2], dict(fw[0]))
print(dict(nx.all_pairs_shortest_path_length(G)))            # ホップ数
print(nx.floyd_warshall_numpy(G))                             # numpy 行列
D = nx.DiGraph()
D.add_weighted_edges_from([(1, 2, 1), (1, 3, 2), (3, 2, -5)])
print(nx.johnson(D)[1][2])                                    # 負の重みがあっても経路が正しい
```
実行結果:
```
0 {0: 0, 1: 1, 2: 3} [0, 1, 2]
1 {1: 0, 0: 1, 2: 2} [1, 2]
2 {2: 0, 1: 2, 0: 3} [2]
3 {0: 0, 1: 1, 2: 3}
{0: {0: 0, 1: 1, 2: 1}, 1: {1: 0, 0: 1, 2: 1}, 2: {2: 0, 1: 1, 0: 1}}
[[0. 1. 3.]
 [1. 0. 2.]
 [3. 2. 0.]]
[1, 3, 2]
```

**注意点・落とし穴**:
- `floyd_warshall` は O(n^3)。数千ノードを超えるグラフでは非現実的で、`all_pairs_dijkstra` を使う。返り値は「始点→`defaultdict`(終点→距離)」の dict で、到達できないノード対の値は `inf`。
- `all_pairs_shortest_path_length` は重みを見ない(ホップ数)。重み付きは `all_pairs_dijkstra` か `floyd_warshall`。

---

### `networkx.all_simple_paths / has_path / all_shortest_paths`

**用途**: 2ノード間の全ての単純経路(同じノードを2度通らない経路)や、最短経路を全て列挙する。`has_path` は到達可能かの判定。

**シグネチャ**:
- `networkx.all_simple_paths(G, source, target, cutoff=None)`
- `networkx.has_path(G, source, target)`
- `networkx.all_shortest_paths(G, source, target, weight=None, method='dijkstra')`

**使用例**:
```python
import networkx as nx
G = nx.Graph([(0, 1), (1, 3), (0, 2), (2, 3), (0, 3), (1, 2)])
print(sorted(nx.all_simple_paths(G, 0, 3)))
print(sorted(nx.all_simple_paths(G, 0, 3, cutoff=2)))    # 長さ(辺数)が2以下
print(sorted(nx.all_shortest_paths(G, 0, 3)))
H = nx.Graph([(0, 1)]); H.add_node(2)
print(nx.has_path(H, 0, 1), nx.has_path(H, 0, 2))
```
実行結果:
```
[[0, 1, 2, 3], [0, 1, 3], [0, 2, 1, 3], [0, 2, 3], [0, 3]]
[[0, 1, 3], [0, 2, 3], [0, 3]]
[[0, 3]]
True False
```

**注意点・落とし穴**:
- `all_simple_paths` はジェネレータ。パス数はグラフのサイズに対して指数的に増えるため、大きなグラフでは `cutoff=` で長さを制限するか、`itertools.islice` で先頭から一部だけ取り出す。
- `source` と `target` が同一のノードのときは、そのノード1つだけの経路 `[[source]]` が返る(`list(nx.all_simple_paths(G, 1, 1))` → `[[1]]`)。`cutoff` は経路の辺数の上限で、上の例では `[0, 1, 2, 3]` などの長い経路が除外される。

---

### `networkx.diameter / eccentricity / radius / average_shortest_path_length`

**用途**: グラフ全体の「大きさ」を測る距離指標。離心距離(そのノードから最も遠いノードまでの距離)、直径(離心距離の最大)、半径(同最小)、平均最短経路長。

**シグネチャ**:
- `networkx.diameter(G, e=None, usebounds=False, weight=None)`
- `networkx.eccentricity(G, v=None, sp=None, weight=None)`
- `networkx.radius(G, e=None, usebounds=False, weight=None)`
- `networkx.average_shortest_path_length(G, weight=None, method=None)`

**使用例**:
```python
import networkx as nx
G = nx.path_graph(5)              # 0-1-2-3-4
print(nx.eccentricity(G))
print(nx.diameter(G), nx.radius(G))
print(nx.average_shortest_path_length(G))
K = nx.karate_club_graph()
print(nx.diameter(K), round(nx.average_shortest_path_length(K), 4))
H = nx.Graph([(0, 1), (2, 3)])
try:
    nx.diameter(H)
except nx.NetworkXError as e:
    print("NetworkXError:", e)
```
実行結果:
```
{0: 4, 1: 3, 2: 2, 3: 3, 4: 4}
4 2
2.0
5 2.4082
NetworkXError: Found infinite path length because the graph is not connected
```

**注意点・落とし穴**:
- いずれも **連結グラフ**が前提で、非連結だと `NetworkXError`(有向グラフは強連結が必要で、そうでなければ `NetworkXError: Found infinite path length because the digraph is not strongly connected`)。非連結なら `connected_components` ごとに `G.subgraph(c)` を作って計算する。
- 全点対の最短経路を計算するので大きなグラフでは重い。近似には `nx.approximation.diameter(G)` などがある。

---

## 連結性・成分・構造

### `networkx.is_connected / connected_components / number_connected_components`

**用途**: 無向グラフの連結成分(互いに行き来できるノードの塊)を調べる。`connected_components` は各成分のノード集合を返すジェネレータ。

**シグネチャ**:
- `networkx.is_connected(G)`
- `networkx.connected_components(G)`
- `networkx.number_connected_components(G)`
- `networkx.node_connected_component(G, n)`

**使用例**:
```python
import networkx as nx
G = nx.Graph([(1, 2), (2, 3), (4, 5)])
G.add_node(6)
print(nx.is_connected(G), nx.number_connected_components(G))
comps = list(nx.connected_components(G))
print(comps)
largest = max(comps, key=len)
print(largest)
H = G.subgraph(largest).copy()          # 最大連結成分のグラフ
print(H)
print(nx.node_connected_component(G, 4))
print(sorted(map(len, nx.connected_components(G)), reverse=True))
```
実行結果:
```
False 3
[{1, 2, 3}, {4, 5}, {6}]
{1, 2, 3}
Graph with 3 nodes and 2 edges
{4, 5}
[3, 2, 1]
```

**注意点・落とし穴**:
- `connected_components` は「ノード集合(set)のジェネレータ」を返す(部分グラフではない)。成分ごとのグラフが欲しいときは `G.subgraph(c).copy()`。3.6.1 では `connected_component_subgraphs` は存在しない。
- 有向グラフに使うと `NetworkXNotImplemented`。有向グラフには `weakly_connected_components` / `strongly_connected_components` を使う。空グラフに `is_connected` を呼ぶと `NetworkXPointlessConcept`。

---

### `networkx.strongly_connected_components / weakly_connected_components`

**用途**: 有向グラフの連結成分。強連結成分(互いに有向路で行き来できる塊)と弱連結成分(向きを無視してつながる塊)の2種類がある。

**シグネチャ**:
- `networkx.strongly_connected_components(G)`
- `networkx.weakly_connected_components(G)`
- `networkx.is_strongly_connected(G)`
- `networkx.is_weakly_connected(G)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 4), (6, 5)])
print(sorted(map(sorted, nx.strongly_connected_components(D))))
print(sorted(map(sorted, nx.weakly_connected_components(D))))
print(nx.is_strongly_connected(D), nx.is_weakly_connected(D))
```
実行結果:
```
[[1, 2, 3], [4, 5], [6]]
[[1, 2, 3, 4, 5, 6]]
False True
```

**注意点・落とし穴**:
- 強連結成分は「サイクルの塊」で、1ノードだけの成分も含めて全ノードがいずれかの成分に属する(上の 6 は単独成分)。強連結成分を1ノードに縮約した DAG は `condensation` で得られる。

---

### `networkx.condensation(G)`

**用途**: 有向グラフの強連結成分を1ノードに縮約した DAG(縮約グラフ)を返す。循環を含む依存関係から、非循環の全体構造を取り出すときに使う。

**シグネチャ**: `networkx.condensation(G, scc=None)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 4)])
C = nx.condensation(D)
print(sorted(C.nodes))
print(sorted(C.edges))
print({n: sorted(m) for n, m in C.nodes(data="members")})
print(C.graph["mapping"])
print(nx.is_directed_acyclic_graph(C))
```
実行結果:
```
[0, 1]
[(1, 0)]
{0: [4, 5], 1: [1, 2, 3]}
{4: 0, 5: 0, 1: 1, 2: 1, 3: 1}
True
```

**注意点・落とし穴**:
- 縮約グラフのノードは 0.. の連番で、元のノード集合は属性 `members`、元ノード→成分番号の対応は `C.graph['mapping']` に入る。

---

### `networkx.bridges / articulation_points / biconnected_components`

**用途**: 無向グラフの弱点を見つける。橋(取り除くと連結成分が増える辺)、関節点(取り除くと連結成分が増えるノード)、二重連結成分(関節点を1つ外しても切れない塊)。

**シグネチャ**:
- `networkx.bridges(G, root=None)`
- `networkx.articulation_points(G)`
- `networkx.biconnected_components(G)`

**使用例**:
```python
import networkx as nx
# 三角形 0-1-2 と三角形 3-4-5 を辺 (2,3) で結ぶ
G = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
print(list(nx.bridges(G)))
print(sorted(nx.articulation_points(G)))
print(sorted(map(sorted, nx.biconnected_components(G))))
```
実行結果:
```
[(2, 3)]
[2, 3]
[[0, 1, 2], [2, 3], [3, 4, 5]]
```

**注意点・落とし穴**:
- 3種類とも無向グラフ専用(有向グラフだと `NetworkXNotImplemented`)。`bridges` の `root=` にノードを指定すると、そのノードを含む連結成分の橋だけを調べる(非連結グラフで `root` を省略すると全成分が対象)。
- 3つともジェネレータを返すので、`list(...)` / `sorted(...)` で確定する。

---

### `networkx.k_core / core_number`

**用途**: k-コア(次数が k 以上のノードだけを残す操作を繰り返して残る、密に結びついた部分グラフ)を取り出す。`core_number` は各ノードが属する最大の k。

**シグネチャ**:
- `networkx.k_core(G, k=None, core_number=None)`
- `networkx.core_number(G)`

**使用例**:
```python
import networkx as nx
G = nx.karate_club_graph()
cn = nx.core_number(G)
print(max(cn.values()))
print(sorted(n for n, k in cn.items() if k == 4))
core4 = nx.k_core(G, k=4)
print(core4.number_of_nodes(), core4.number_of_edges())
print(nx.k_core(G).number_of_nodes())     # k省略: 最大のコア
```
実行結果:
```
4
[0, 1, 2, 3, 7, 8, 13, 30, 32, 33]
10 25
10
```

**注意点・落とし穴**:
- 自己ループのあるグラフは `NetworkXNotImplemented`(`G.remove_edges_from(nx.selfloop_edges(G))` で先に取り除く)。返り値は元グラフの部分グラフのコピー(独立した `Graph`)。

---

### `networkx.clustering / transitivity / triangles / average_clustering`

**用途**: ノードの周辺がどれだけ三角形を作っているか(友達の友達も友達か)を測る。`clustering` はノードごとの係数、`transitivity` はグラフ全体での三角形の割合。

**シグネチャ**:
- `networkx.clustering(G, nodes=None, weight=None)`
- `networkx.average_clustering(G, nodes=None, weight=None, count_zeros=True)`
- `networkx.transitivity(G)`
- `networkx.triangles(G, nodes=None)`

**使用例**:
```python
import networkx as nx
G = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])       # 三角形 0-1-2 + 端 3
print(nx.triangles(G))
print(nx.clustering(G))
print(round(nx.average_clustering(G), 4), round(nx.transitivity(G), 4))
print(nx.clustering(G, 2))
K = nx.karate_club_graph()
print(round(nx.average_clustering(K), 4), round(nx.transitivity(K), 4))
```
実行結果:
```
{0: 1, 1: 1, 2: 1, 3: 0}
{0: 1.0, 1: 1.0, 2: 0.3333333333333333, 3: 0}
0.5833 0.6
0.3333333333333333
0.5706 0.2557
```

**注意点・落とし穴**:
- `average_clustering`(局所係数の平均)と `transitivity`(全体の3つ組に対する三角形の割合)は別の指標で、値が一致しない(上の例 0.5833 vs 0.6)。
- 次数1以下のノードの係数は 0 として扱われ(`count_zeros=True` の既定で平均に含まれる)、平均を下げる。`average_clustering(G, count_zeros=False)` で 0 のノードを平均から除外できる(上の例では 0.5833 → 0.7778)。`triangles` は各ノードが属する三角形の数(1つの三角形は3ノード全てで数えられる)。

---

## 中心性・コミュニティ検出

### `networkx.degree_centrality(G)`

**用途**: 次数中心性。各ノードの次数を `n-1` で割った値(0〜1)。「どれだけ多くのノードと直接つながっているか」の指標。有向グラフでは `in_degree_centrality` / `out_degree_centrality`。

**シグネチャ**:
- `networkx.degree_centrality(G)`
- `networkx.in_degree_centrality(G)`

**使用例**:
```python
import networkx as nx
G = nx.star_graph(4)                 # 中心 0 + 葉 1..4
print(nx.degree_centrality(G))
D = nx.DiGraph([(1, 2), (3, 2), (2, 4)])
print(nx.in_degree_centrality(D))
K = nx.karate_club_graph()
dc = nx.degree_centrality(K)
print(sorted(dc, key=dc.get, reverse=True)[:3], round(max(dc.values()), 4))
```
実行結果:
```
{0: 1.0, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
{1: 0.0, 2: 0.6666666666666666, 3: 0.0, 4: 0.3333333333333333}
[33, 0, 32] 0.5152
```

**注意点・落とし穴**:
- 結果は `{ノード: 値}` の辞書。上位を取り出すには `sorted(dc, key=dc.get, reverse=True)[:k]`。`degree_centrality` は重みを見ない(次数ベース)。
- 自己ループがあると値が 1 を超えることがある(自己ループは次数に2として数えられるため)。

---

### `networkx.betweenness_centrality(G)`

**用途**: 媒介中心性。「全ノード対の最短経路のうち、そのノードを通るものの割合」。情報の中継点・ボトルネックの発見に使う。辺版は `edge_betweenness_centrality`。

**シグネチャ**:
- `networkx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)`
- `networkx.edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None)`

**使用例**:
```python
import networkx as nx
G = nx.path_graph(5)
print(nx.betweenness_centrality(G))
print(nx.betweenness_centrality(G, normalized=False))
K = nx.karate_club_graph()
bc = nx.betweenness_centrality(K, seed=0)
print(sorted(bc, key=bc.get, reverse=True)[:3], round(max(bc.values()), 4))
ebc = nx.edge_betweenness_centrality(K)
print(max(ebc, key=ebc.get))
approx = nx.betweenness_centrality(K, k=10, seed=0)     # 10ノードだけ抽出して近似
print(round(approx[0], 3), round(bc[0], 3))
```
実行結果:
```
{0: 0.0, 1: 0.5, 2: 0.6666666666666666, 3: 0.5, 4: 0.0}
{0: 0.0, 1: 3.0, 2: 4.0, 3: 3.0, 4: 0.0}
[0, 33, 32] 0.4376
(0, 31)
0.405 0.438
```

**注意点・落とし穴**:
- Brandes 法で、ノード数 n・辺数 m に対し重みなしでおおよそ O(nm)。数千ノード以上では `k=` でサンプリングして近似する(その際 `seed=` で再現性を固定)。`k=None`(既定)は厳密計算で、`seed` は影響しない。
- 既定は `normalized=True`(無向グラフなら `2/((n-1)(n-2))` で割る)。`weight='weight'` を指定すると距離として辺の重みが使われる点に注意(重みが大きいほど遠い)。

---

### `networkx.closeness_centrality(G)`

**用途**: 近接中心性。他のノードへの平均最短距離の逆数(近いほど大きい)。全体に素早く情報を届けられるノードの指標。非連結グラフでは `harmonic_centrality` の方が安全。

**シグネチャ**:
- `networkx.closeness_centrality(G, u=None, distance=None, wf_improved=True)`
- `networkx.harmonic_centrality(G, nbunch=None, distance=None, sources=None)`

**使用例**:
```python
import networkx as nx
G = nx.path_graph(5)
print(nx.closeness_centrality(G))
print(nx.closeness_centrality(G, u=2))                 # 1ノードだけ
H = nx.Graph([(0, 1), (2, 3)])
print(nx.closeness_centrality(H))                      # 非連結: 各成分内の距離で補正される
print(nx.harmonic_centrality(H))                       # 距離の逆数の和
```
実行結果:
```
{0: 0.4, 1: 0.5714285714285714, 2: 0.6666666666666666, 3: 0.5714285714285714, 4: 0.4}
0.6666666666666666
{0: 0.3333333333333333, 1: 0.3333333333333333, 2: 0.3333333333333333, 3: 0.3333333333333333}
{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
```

**注意点・落とし穴**:
- 非連結グラフでは、到達できるノード数で補正される(`wf_improved=True` が既定、Wasserman-Faust の改善版)。`wf_improved=False` にすると到達できるノードだけの距離で計算する(上の `H` では全ノード 1.0 になる)。
- **重み付きグラフ**では `distance='weight'` を明示する必要がある(既定の `distance=None` は重みを見ない。辺 (0,1,重み1)・(1,2,重み10) のグラフで、ノード 0 の値は既定 0.667、`distance='weight'` で 0.167)。`weight=` ではなく `distance=` という引数名なので混同しやすい。

---

### `networkx.pagerank(G, alpha=0.85)`

**用途**: PageRank。「重要なノードからリンクされているノードは重要」という再帰的な定義で、有向グラフのノードの重要度を求める。`alpha` はランダムウォークを続ける確率(1-alpha はランダムジャンプ)。

**シグネチャ**: `networkx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A"), ("D", "C")])
pr = nx.pagerank(D)
print({k: round(v, 4) for k, v in pr.items()})
print(round(sum(pr.values()), 6))
pers = nx.pagerank(D, personalization={"A": 1})    # Aに偏らせたPageRank
print({k: round(v, 4) for k, v in pers.items()})
W = nx.DiGraph()
W.add_weighted_edges_from([("A", "B", 5), ("A", "C", 1), ("B", "A", 1), ("C", "A", 1)])
print({k: round(v, 4) for k, v in nx.pagerank(W).items()})
```
実行結果:
```
{'A': 0.3202, 'B': 0.3097, 'C': 0.3326, 'D': 0.0375}
1.0
{'A': 0.3887, 'B': 0.3304, 'C': 0.2809, 'D': 0.0}
{'A': 0.4865, 'B': 0.3946, 'C': 0.1189}
```

**注意点・落とし穴**:
- 値の総和は 1。既定で辺属性 `'weight'` を重みとして使うので、重みなしで計算したい場合は `weight=None`。無向グラフを渡すと各辺が双方向の有向辺として扱われる(`Graph([(1,2),(2,3)])` と両方向を持つ `DiGraph` の結果は同じ)。
- 反復が `max_iter`(既定100)以内に収束しないと `PowerIterationFailedConvergence`(`max_iter=1` で再現)。出次数0のノード(行き止まり)の確率は `dangling` の分布で全ノードに配り直される。`dangling` を省略すると `personalization`(それも省略なら一様分布)が使われる。

---

### `networkx.eigenvector_centrality(G) / katz_centrality / hits`

**用途**: 固有ベクトル中心性(重要なノードとつながるほど重要)。Katz 中心性はその減衰付きの拡張、`hits` は有向グラフの権威(authority)とハブ(hub)を分けて評価する。

**シグネチャ**:
- `networkx.eigenvector_centrality(G, max_iter=100, tol=1e-06, nstart=None, weight=None)`
- `networkx.katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight=None)`
- `networkx.hits(G, max_iter=100, tol=1e-08, nstart=None, normalized=True)`

**使用例**:
```python
import networkx as nx
K = nx.karate_club_graph()
ec = nx.eigenvector_centrality(K)
print(sorted(ec, key=ec.get, reverse=True)[:3], round(max(ec.values()), 4))
kc = nx.katz_centrality(K, alpha=0.05)
print(sorted(kc, key=kc.get, reverse=True)[:3])
D = nx.DiGraph([(1, 2), (1, 3), (2, 3), (4, 3), (3, 1)])
hubs, auth = nx.hits(D)
print({k: round(v, 3) + 0.0 for k, v in hubs.items()})     # +0.0 は -0.0 表示の回避
print({k: round(v, 3) + 0.0 for k, v in auth.items()})
```
実行結果:
```
[33, 0, 2] 0.3734
[33, 0, 32]
{1: 0.414, 2: 0.293, 3: 0.0, 4: 0.293}
{1: 0.0, 2: 0.293, 3: 0.707, 4: 0.0}
```

**注意点・落とし穴**:
- 固有ベクトル中心性は 2-ノルムで正規化される(値の総和が1ではなく、二乗和が1)。反復が `max_iter`(既定100)以内に収束しないと `PowerIterationFailedConvergence`(`max_iter=1` で再現)。既定では辺の重みを使わない(`weight=None`)。
- `katz_centrality` の `alpha` は「隣接行列の最大固有値の逆数」未満でなければならない。karate の隣接行列(重みなし)の最大固有値は約 6.7257(逆数 0.1487)で、`alpha=0.14` は収束、`alpha=0.16` は `PowerIterationFailedConvergence`。

---

### `networkx.community.louvain_communities(G)`

**用途**: ルーヴァン法でコミュニティ(内部で密・外部で疎に結びつくノードの塊)を検出する。モジュラリティ最大化を貪欲に繰り返す。返り値はノード集合(set)のリスト。`community.modularity` で分割の質を評価する。

**シグネチャ**:
- `networkx.community.louvain_communities(G, weight='weight', resolution=1, threshold=1e-07, max_level=None, seed=None)`
- `networkx.community.modularity(G, communities, weight='weight', resolution=1)`

**使用例**:
```python
import networkx as nx
from networkx.algorithms import community
G = nx.karate_club_graph()
comms = community.louvain_communities(G, seed=42)
print(len(comms), [len(c) for c in comms])
print(round(community.modularity(G, comms), 4))
comms2 = community.louvain_communities(G, resolution=2.0, seed=42)     # 大きいほど小さなコミュニティ
print(len(comms2), round(community.modularity(G, comms2), 4))
node2comm = {n: i for i, c in enumerate(comms) for n in c}
print(node2comm[0], node2comm[33])
```
実行結果:
```
4 [6, 10, 4, 14]
0.4266
7 0.344
1 3
```

**注意点・落とし穴**:
- Louvain は確率的なので、再現性のため `seed=` を固定する。実行間で分割が変わるので、固定しない場合は複数回実行して比較する。
- `resolution` を上げるとコミュニティが小さく・多くなる。返り値のコミュニティ番号(リストの順序)には意味がなく、分割そのものが結果。`weight='weight'` が既定で辺の重みを使う点にも注意(karate は重み付き)。
- `community.modularity(G, communities)` の `communities` は全ノードを重複なく分割している必要がある(でなければ `NotAPartition`、`NetworkXError` の派生クラス)。`modularity` も既定 `weight='weight'` で、karate では重みあり/なしで値が変わる(同じ分割でも重みあり 0.4110、`weight=None` で 0.3807)。

---

### `networkx.community.greedy_modularity_communities(G)`

**用途**: コミュニティ検出の代表的手法3つ。`greedy_modularity_communities`(Clauset-Newman-Moore法、決定的)、`label_propagation_communities`(高速)、`girvan_newman`(媒介中心性の高い辺を除去して階層的に分割)。

**シグネチャ**:
- `networkx.community.greedy_modularity_communities(G, weight=None, resolution=1, cutoff=1, best_n=None)`
- `networkx.community.label_propagation_communities(G)`
- `networkx.community.girvan_newman(G, most_valuable_edge=None)`

**使用例**:
```python
import networkx as nx
from networkx.algorithms import community
G = nx.karate_club_graph()
g = community.greedy_modularity_communities(G)
print([len(c) for c in g], round(community.modularity(G, g), 4))
g2 = community.greedy_modularity_communities(G, best_n=2)     # コミュニティ数を2に固定
print([len(c) for c in g2])
lp = list(community.label_propagation_communities(G))
print(len(lp))
gn = community.girvan_newman(G)                                # 分割の階層(ジェネレータ)
first = next(gn)                                               # 最初の分割(2分割)
print([sorted(c)[:4] for c in first], [len(c) for c in first])
```
実行結果:
```
[17, 9, 8] 0.411
[17, 17]
3
[[0, 1, 3, 4], [2, 8, 9, 14]] [15, 19]
```

**注意点・落とし穴**:
- `greedy_modularity_communities` の返り値は `frozenset` のリスト(サイズの大きい順)で、乱数を使わないので結果が決定的。
- `label_propagation_communities` は半同期(semi-synchronous)版で `seed` 引数を持たず、`PYTHONHASHSEED` を変えて実行しても結果は同じだった。返り値は list ではなく `dict_values` なので `list()` で包む。乱数を使う非同期版は `asyn_lpa_communities(G, seed=...)`。`girvan_newman` は各段階で辺の媒介中心性を再計算するため計算量が大きく、大きなグラフ向きではない。
- `greedy_modularity_communities` の `weight` 既定は `None`(重みを見ない)で、`louvain_communities` の既定 `'weight'` と違う。同じ karate でも重みなしだと [17, 9, 8]、`weight='weight'` を渡すと [18, 11, 5] になる。

---

## 走査・順序・DAG

### `networkx.bfs_tree / bfs_edges / bfs_layers / bfs_predecessors`

**用途**: 幅優先探索(始点に近いノードから順に訪れる)。訪問した辺、探索木、始点からの距離ごとの層、各ノードの親を取り出せる。

**シグネチャ**:
- `networkx.bfs_tree(G, source, reverse=False, depth_limit=None, sort_neighbors=None)`
- `networkx.bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None)`
- `networkx.bfs_layers(G, sources)`
- `networkx.bfs_predecessors(G, source, depth_limit=None, sort_neighbors=None)`

**使用例**:
```python
import networkx as nx
G = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (5, 6)])
print(list(nx.bfs_edges(G, 0)))
T = nx.bfs_tree(G, 0)
print(type(T).__name__, sorted(T.edges))
print(list(nx.bfs_layers(G, [0])))                # 距離ごとのノード群
print(dict(nx.bfs_predecessors(G, 0)))
print(list(nx.bfs_edges(G, 0, depth_limit=1)))
```
実行結果:
```
[(0, 1), (0, 2), (1, 3), (3, 4)]
DiGraph [(0, 1), (0, 2), (1, 3), (3, 4)]
[[0], [1, 2], [3], [4]]
{1: 0, 2: 0, 3: 1, 4: 3}
[(0, 1), (0, 2)]
```

**注意点・落とし穴**:
- 探索できるのは始点から到達可能な範囲だけ(上の例ではノード 5, 6 は出てこない)。`bfs_tree` は常に `DiGraph`(根から葉へ向かう有向木)を返す。
- 無向グラフの全域を対象にするには連結成分ごとに呼ぶ必要がある。`bfs_layers` の `sources` にはノードのリスト(複数始点)、または単一ノードを渡せる。

---

### `networkx.dfs_edges / dfs_preorder_nodes / dfs_postorder_nodes / dfs_tree`

**用途**: 深さ優先探索。行きがけ順(preorder)・帰りがけ順(postorder)のノード列、探索辺、探索木を得る。

**シグネチャ**:
- `networkx.dfs_edges(G, source=None, depth_limit=None, *, sort_neighbors=None)`
- `networkx.dfs_preorder_nodes(G, source=None, depth_limit=None, *, sort_neighbors=None)`
- `networkx.dfs_postorder_nodes(G, source=None, depth_limit=None, *, sort_neighbors=None)`
- `networkx.dfs_tree(G, source=None, depth_limit=None, *, sort_neighbors=None)`

**使用例**:
```python
import networkx as nx
G = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)])
print(list(nx.dfs_preorder_nodes(G, 0)))
print(list(nx.dfs_postorder_nodes(G, 0)))
print(list(nx.dfs_edges(G, 0)))
print(sorted(nx.dfs_tree(G, 0, depth_limit=1).edges))
print(list(nx.dfs_preorder_nodes(G, 0, sort_neighbors=lambda nb: sorted(nb, reverse=True))))
```
実行結果:
```
[0, 1, 3, 4, 2, 5]
[3, 4, 1, 5, 2, 0]
[(0, 1), (1, 3), (1, 4), (0, 2), (2, 5)]
[(0, 1), (0, 2)]
[0, 2, 5, 1, 4, 3]
```

**注意点・落とし穴**:
- `source=None` にすると全ノードを対象にする(非連結でも全成分を走査)。近傍の訪問順は、既定ではグラフへの登録順。順序を変えたいときは `sort_neighbors=` に「隣接ノードの iterable を受け取りソート済み iterable を返す関数」を渡す。
- 再帰ではなく反復実装なので、深いグラフでも Python の再帰上限に引っかからない。

---

### `networkx.topological_sort / topological_generations`

**用途**: 有向非巡回グラフ(DAG)の位相ソート(依存関係を満たす並び)。`topological_generations` は依存のない順に「同時に実行できる層」ごとに返す。

**シグネチャ**:
- `networkx.topological_sort(G)`
- `networkx.topological_generations(G)`
- `networkx.lexicographical_topological_sort(G, key=None)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph([("shirt", "tie"), ("tie", "jacket"), ("pants", "belt"),
                ("belt", "jacket"), ("pants", "shoes"), ("socks", "shoes")])
print(list(nx.topological_sort(D)))
for i, layer in enumerate(nx.topological_generations(D)):
    print(i, sorted(layer))
print(list(nx.lexicographical_topological_sort(D)))     # 同順位はノード名の辞書順
C = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
try:
    list(nx.topological_sort(C))
except nx.NetworkXUnfeasible as e:
    print("NetworkXUnfeasible:", e)
```
実行結果:
```
['shirt', 'pants', 'socks', 'tie', 'belt', 'shoes', 'jacket']
0 ['pants', 'shirt', 'socks']
1 ['belt', 'shoes', 'tie']
2 ['jacket']
['pants', 'belt', 'shirt', 'socks', 'shoes', 'tie', 'jacket']
NetworkXUnfeasible: Graph contains a cycle or graph changed during iteration
```

**注意点・落とし穴**:
- `topological_sort` はジェネレータ。閉路があると反復の途中(`list()` で確定した時点)で `NetworkXUnfeasible` になるため、`list()` で包んで例外を処理する。事前判定は `nx.is_directed_acyclic_graph(G)`。
- 位相ソートの結果は一意ではない。安定した順序が必要な場合は `lexicographical_topological_sort`(`key=` で優先順位関数を指定可)を使う。

---

### `networkx.is_directed_acyclic_graph / find_cycle / simple_cycles`

**用途**: 閉路(サイクル)の判定と検出。`is_directed_acyclic_graph` は DAG かの判定、`find_cycle` は閉路を1つ返し、`simple_cycles` は全ての単純閉路を列挙する。

**シグネチャ**:
- `networkx.is_directed_acyclic_graph(G)`
- `networkx.find_cycle(G, source=None, orientation=None)`
- `networkx.simple_cycles(G, length_bound=None)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4)])
print(nx.is_directed_acyclic_graph(D))
print(nx.find_cycle(D))
print(sorted(nx.simple_cycles(D)))
D.remove_edge(3, 1)
print(nx.is_directed_acyclic_graph(D))
try:
    nx.find_cycle(D)
except nx.NetworkXNoCycle as e:
    print("NetworkXNoCycle:", e)
G = nx.Graph([(1, 2), (2, 3), (3, 1)])
print(sorted(nx.cycle_basis(G)[0]), sorted(map(sorted, nx.simple_cycles(G))))
```
実行結果:
```
False
[(1, 2), (2, 3), (3, 1)]
[[1, 2, 3]]
True
NetworkXNoCycle: No cycle found.
[1, 2, 3] [[1, 2, 3]]
```

**注意点・落とし穴**:
- `find_cycle` は閉路がないと `NetworkXNoCycle` を送出する(空リストではない)。返り値は辺のリスト。
- 無向グラフの閉路の基底(独立な閉路の集合)は `cycle_basis`。3.6.1 の `simple_cycles` は無向グラフも扱える(上の例で三角形を1つ返す)。`length_bound=` で長さ上限を付けられる(閉路の数は指数的に増えうる)。

---

### `networkx.ancestors / descendants / dag_longest_path`

**用途**: 有向グラフで、あるノードから到達できるノード(子孫)・そのノードへ到達できるノード(祖先)を集合で返す。`dag_longest_path` は DAG の最長経路(タスクの依存関係でいうクリティカルパス)。

**シグネチャ**:
- `networkx.ancestors(G, source)`
- `networkx.descendants(G, source)`
- `networkx.dag_longest_path(G, weight='weight', default_weight=1, topo_order=None)`
- `networkx.dag_longest_path_length(G, weight='weight', default_weight=1)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph([("a", "b"), ("b", "c"), ("a", "d"), ("d", "c"), ("c", "e")])
print(sorted(nx.descendants(D, "b")))
print(sorted(nx.ancestors(D, "c")))
print(nx.dag_longest_path(D))
print(nx.dag_longest_path_length(D))
W = nx.DiGraph()
W.add_weighted_edges_from([("s", "a", 3), ("s", "b", 1), ("a", "t", 1), ("b", "t", 10)])
print(nx.dag_longest_path(W), nx.dag_longest_path_length(W))
```
実行結果:
```
['c', 'e']
['a', 'b', 'd']
['a', 'b', 'c', 'e']
3
['s', 'b', 't'] 11
```

**注意点・落とし穴**:
- `descendants` / `ancestors` は始点自身を含まない。無向グラフに使うと連結成分内の全ノードから自分を除いたものになる。
- `dag_longest_path` は重み(既定 `'weight'`、無い辺は `default_weight=1`)の合計が最大の経路を返す。閉路があると `NetworkXUnfeasible`。

---

## フロー・マッチング・スパニングツリー

### `networkx.maximum_flow / maximum_flow_value / minimum_cut`

**用途**: 最大流問題。辺の属性 `capacity`(容量)を持つ有向グラフで、始点 s から終点 t へ流せる最大流量とその流し方を求める。`minimum_cut` は最大流最小カット定理に基づく最小カットの値とその2分割。

**シグネチャ**:
- `networkx.maximum_flow(flowG, _s, _t, capacity='capacity', flow_func=None)`
- `networkx.maximum_flow_value(flowG, _s, _t, capacity='capacity', flow_func=None)`
- `networkx.minimum_cut(flowG, _s, _t, capacity='capacity', flow_func=None)`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph()
D.add_edge("s", "a", capacity=3)
D.add_edge("s", "b", capacity=2)
D.add_edge("a", "b", capacity=1)
D.add_edge("a", "t", capacity=2)
D.add_edge("b", "t", capacity=3)
value, flow = nx.maximum_flow(D, "s", "t")
print(value)
print(flow)
print(nx.maximum_flow_value(D, "s", "t"))
cut_value, (S, T) = nx.minimum_cut(D, "s", "t")
print(cut_value, sorted(S), sorted(T))
```
実行結果:
```
5
{'s': {'a': 3, 'b': 2}, 'a': {'b': 1, 't': 2}, 'b': {'t': 3}, 't': {}}
5
5 ['a', 'b', 's'] ['t']
```

**注意点・落とし穴**:
- 容量は属性名 `capacity` で持つ(`weight` ではない)。`capacity` が無い辺は容量無限として扱われ、そのような経路があると最大流が有限にならず `NetworkXUnbounded`。
- 無向グラフも渡せる(各辺は両方向に同じ容量を持つものとして扱われる)。返り値の `flow` は `flow[u][v]` の二重辞書。

---

### `networkx.min_cost_flow / min_cost_flow_cost`

**用途**: 最小費用流。各ノードの需給 `demand`(供給側が負、需要側が正)を満たしつつ、辺の単位費用 `weight`×流量の合計を最小にする流し方を求める。輸送・割当問題に使う。

**シグネチャ**:
- `networkx.min_cost_flow(G, demand='demand', capacity='capacity', weight='weight')`
- `networkx.min_cost_flow_cost(G, demand='demand', capacity='capacity', weight='weight')`

**使用例**:
```python
import networkx as nx
D = nx.DiGraph()
D.add_node("A", demand=-5)      # A から5単位供給
D.add_node("D", demand=5)       # D が5単位要求
D.add_edge("A", "B", weight=3, capacity=4)
D.add_edge("A", "C", weight=6, capacity=10)
D.add_edge("B", "D", weight=1, capacity=9)
D.add_edge("C", "D", weight=2, capacity=5)
flow = nx.min_cost_flow(D)
print(flow)
print(nx.min_cost_flow_cost(D))
```
実行結果:
```
{'A': {'B': 4, 'C': 1}, 'D': {}, 'B': {'D': 4}, 'C': {'D': 1}}
24
```

**注意点・落とし穴**:
- 総供給と総需要(`demand` の合計)が 0 でなければ `NetworkXUnfeasible`。流せない(容量不足)場合も `NetworkXUnfeasible`。**`demand` は整数**で与えるのが前提(浮動小数だと丸め誤差で失敗しうる。ドキュメントの注意)。
- `weight` は費用、`capacity` は容量。`capacity` が無い辺は容量無限。有向グラフ専用で無向グラフは `NetworkXNotImplemented`。

---

### `networkx.max_weight_matching / maximal_matching / min_weight_matching`

**用途**: 無向グラフのマッチング(ノードを重複なく2つ組にする辺の集合)。`max_weight_matching` は辺の重み合計が最大のもの(厳密解)、`maximal_matching` は「これ以上辺を追加できない」貪欲解(最大とは限らない)。

**シグネチャ**:
- `networkx.max_weight_matching(G, maxcardinality=False, weight='weight')`
- `networkx.maximal_matching(G)`
- `networkx.min_weight_matching(G, weight='weight')`
- `networkx.is_matching(G, matching)`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([(1, 2, 5), (2, 3, 11), (3, 4, 5)])   # 1-2-3-4 の直線
m = nx.max_weight_matching(G)
print(sorted(tuple(sorted(e)) for e in m))
m2 = nx.max_weight_matching(G, maxcardinality=True)         # 辺の数を優先
print(sorted(tuple(sorted(e)) for e in m2))
print(nx.is_matching(G, m), nx.is_matching(G, {(1, 2), (2, 3)}))
print(len(nx.maximal_matching(G)))
print(len(nx.maximal_matching(nx.Graph([(2, 3), (1, 2), (3, 4)]))))   # 同じグラフでも辺の順序で結果が変わる
mw = nx.min_weight_matching(G)                              # 本数最大のマッチングのうち重み最小
print(sorted(tuple(sorted(e)) for e in mw))
```
実行結果:
```
[(2, 3)]
[(1, 2), (3, 4)]
True False
2
1
[(1, 2), (3, 4)]
```

**注意点・落とし穴**:
- 返り値は辺(タプル)の集合で、各辺の端点の並び順は保証されない(`(1, 2)` か `(2, 1)` か)。比較・表示の前に `tuple(sorted(e))` で正規化する。
- `max_weight_matching` は既定では重み合計を最大化するだけで、辺の本数が最大とは限らない(上の例は重み 11 の辺 (2,3) の1本、`maxcardinality=True` なら (1,2)+(3,4) の2本)。`min_weight_matching` は内部で `maxcardinality=True` を使うため、本数が最大のマッチングの中で重み最小のものが返る。
- `maximal_matching` は「これ以上追加できない」貪欲解にすぎず、辺の順序に依存する(上の例で同じ 1-2-3-4 の直線でも 2本になる場合と 1本になる場合がある)。本数が最大のマッチングが必要なら `max_weight_matching(G, maxcardinality=True)`。

---

### `networkx.bipartite.sets / is_bipartite / maximum_matching`

**用途**: 二部グラフ(ノードを2グループに分け、辺が必ずグループ間をまたぐグラフ)を扱う。判定・2グループへの分割・最大マッチング(Hopcroft-Karp 法)・片側への射影。

**シグネチャ**:
- `networkx.bipartite.is_bipartite(G)`
- `networkx.bipartite.sets(G, top_nodes=None)`
- `networkx.bipartite.maximum_matching(G, top_nodes=None)`
- `networkx.bipartite.projected_graph(B, nodes, multigraph=False)`

**使用例**:
```python
import networkx as nx
from networkx.algorithms import bipartite
B = nx.Graph([("w1", "j1"), ("w1", "j2"), ("w2", "j1"), ("w3", "j2")])
print(nx.is_bipartite(B), nx.is_bipartite(nx.cycle_graph(3)))
X, Y = bipartite.sets(B)
print(sorted(X), sorted(Y))
m = bipartite.maximum_matching(B, top_nodes={"w1", "w2", "w3"})
print(len(m) // 2, sorted(k for k in m if k.startswith("j")))   # マッチ数と、マッチ済みの仕事
P = bipartite.projected_graph(B, ["w1", "w2", "w3"])       # 仕事を共有するワーカー同士
print(sorted(P.edges))
```
実行結果:
```
True False
['w1', 'w2', 'w3'] ['j1', 'j2']
2 ['j1', 'j2']
[('w1', 'w2'), ('w1', 'w3')]
```

**注意点・落とし穴**:
- `maximum_matching` の返り値は「両方向の辞書」(w1→j1 と j1→w1 の両方を持つ)なので、マッチ数は `len(m) // 2`。最大マッチングは一意とは限らず、どの組み合わせが返るかは実行ごとに変わりうる(上の例でも `w1-j1, w3-j2` と `w2-j1, w3-j2` のどちらも最大)。`top_nodes` は一方のグループを渡す(連結でない二部グラフでは必須で、省略すると `AmbiguousSolution`)。
- `bipartite.sets(B)` は連結グラフでのみ一意に決まり、非連結だと `AmbiguousSolution`。その場合は `top_nodes=` を指定するか、連結成分ごとに処理する。

---

### `networkx.minimum_spanning_tree / maximum_spanning_tree`

**用途**: 最小(最大)全域木。全ノードを連結し、辺の重み合計が最小(最大)になる辺の部分集合(閉路なし)を求める。非連結グラフでは各成分の全域木の集合(全域森)を返す。

**シグネチャ**:
- `networkx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal', ignore_nan=False)`
- `networkx.maximum_spanning_tree(G, weight='weight', algorithm='kruskal', ignore_nan=False)`
- `networkx.minimum_spanning_edges(G, algorithm='kruskal', weight='weight', keys=True, data=True, ignore_nan=False)`

**使用例**:
```python
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([("A", "B", 4), ("A", "C", 1), ("B", "C", 2), ("B", "D", 5), ("C", "D", 8)])
T = nx.minimum_spanning_tree(G)
print(sorted(T.edges(data="weight")))
print(T.size(weight="weight"))
Tmax = nx.maximum_spanning_tree(G)
print(sorted(Tmax.edges(data="weight")))
print(sorted(tuple(sorted(e)) for e in nx.minimum_spanning_edges(G, algorithm="prim", data=False)))
print(sorted(nx.minimum_spanning_tree(G, algorithm="boruvka").edges))
```
実行結果:
```
[('A', 'C', 1), ('B', 'C', 2), ('B', 'D', 5)]
8.0
[('A', 'B', 4), ('B', 'D', 5), ('C', 'D', 8)]
[('A', 'C'), ('B', 'C'), ('B', 'D')]
[('A', 'C'), ('B', 'C'), ('B', 'D')]
```

**注意点・落とし穴**:
- 返り値は新しい `Graph`(孤立ノードを含む元グラフと同じノード集合を持ち、辺の属性は引き継がれる)。辺の重みは既定で `'weight'` 属性。`algorithm` は `'kruskal'`(既定)/`'prim'`/`'boruvka'` で、辺の重みが全て異なれば結果(全域木)は同じ。
- 有向グラフには使えない(`NetworkXNotImplemented`)。有向グラフの最小全域木(最小arborescence)は `nx.minimum_spanning_arborescence`。

---

## 入出力(pandas・NumPy・SciPy連携)

### `networkx.from_pandas_edgelist / to_pandas_edgelist`

**用途**: pandas の DataFrame(1行=1辺)からグラフを作る、およびグラフの辺を DataFrame に書き出す。CSV の辺リストを読み込む基本の道具。

**シグネチャ**:
- `networkx.from_pandas_edgelist(df, source='source', target='target', edge_attr=None, create_using=None, edge_key=None)`
- `networkx.to_pandas_edgelist(G, source='source', target='target', nodelist=None, dtype=None, edge_key=None)`

**使用例**:
```python
import networkx as nx
import pandas as pd
df = pd.DataFrame({
    "src": ["A", "A", "B", "C"],
    "dst": ["B", "C", "C", "D"],
    "weight": [1.5, 2.0, 0.5, 3.0],
    "kind": ["x", "y", "x", "y"],
})
G = nx.from_pandas_edgelist(df, source="src", target="dst", edge_attr=["weight", "kind"])
print(G.edges(data=True))
D = nx.from_pandas_edgelist(df, "src", "dst", edge_attr=True, create_using=nx.DiGraph)   # 全列を属性に
print(D["A"]["B"])
df2 = nx.to_pandas_edgelist(G)
print(df2[["source", "target", "weight", "kind"]])      # 属性列の並び順は保証されないので明示
```
実行結果:
```
[('A', 'B', {'weight': 1.5, 'kind': 'x'}), ('A', 'C', {'weight': 2.0, 'kind': 'y'}), ('B', 'C', {'weight': 0.5, 'kind': 'x'}), ('C', 'D', {'weight': 3.0, 'kind': 'y'})]
{'weight': 1.5, 'kind': 'x'}
  source target  weight kind
0      A      B     1.5    x
1      A      C     2.0    y
2      B      C     0.5    x
3      C      D     3.0    y
```

**注意点・落とし穴**:
- `edge_attr` を省略(`None`)すると辺属性は付かない。`True` は source/target 以外の全列、リストなら指定列のみ。重みを使うアルゴリズム用には `edge_attr='weight'` のように列を明示する。
- `to_pandas_edgelist` は `source`, `target` 列のあとに辺属性の列が付くが、属性列の並び順は実行ごとに変わりうる(上の例のように列名で選び直すと安定する)。
- 既定の `create_using` は無向 `Graph`。有向にしたいときは `create_using=nx.DiGraph`。`source` / `target` の既定列名は `'source'` / `'target'`(列名が違うと `KeyError`)。

---

### `networkx.to_numpy_array / from_numpy_array / to_pandas_adjacency`

**用途**: グラフと隣接行列(NumPy 配列 / DataFrame)の相互変換。行列の (i, j) 成分が辺 i→j の重み(辺がなければ `nonedge`、既定 0)。

**シグネチャ**:
- `networkx.to_numpy_array(G, nodelist=None, dtype=None, order=None, multigraph_weight=<built-in function sum>, weight='weight', nonedge=0.0)`
- `networkx.from_numpy_array(A, parallel_edges=False, create_using=None, edge_attr='weight', *, nodelist=None)`
- `networkx.to_pandas_adjacency(G, nodelist=None, dtype=None, order=None, multigraph_weight=<built-in function sum>, weight='weight', nonedge=0.0)`
- `networkx.from_pandas_adjacency(df, create_using=None)`

**使用例**:
```python
import networkx as nx
import numpy as np
G = nx.Graph()
G.add_weighted_edges_from([("a", "b", 2.0), ("b", "c", 3.0)])
A = nx.to_numpy_array(G)                      # 行・列の順序は list(G.nodes)
print(list(G.nodes))
print(A)
print(nx.to_numpy_array(G, nodelist=["c", "b", "a"], weight=None))   # 順序指定・重みなし(0/1)
H = nx.from_numpy_array(np.array([[0, 2, 0], [2, 0, 5], [0, 5, 0]]))
print(H.edges(data=True))
Dg = nx.from_numpy_array(np.array([[0, 1], [0, 0]]), create_using=nx.DiGraph)
print(list(Dg.edges))
print(nx.to_pandas_adjacency(G))
```
実行結果:
```
['a', 'b', 'c']
[[0. 2. 0.]
 [2. 0. 3.]
 [0. 3. 0.]]
[[0. 1. 0.]
 [1. 0. 1.]
 [0. 1. 0.]]
[(0, 1, {'weight': 2}), (1, 2, {'weight': 5})]
[(0, 1)]
     a    b    c
a  0.0  2.0  0.0
b  2.0  0.0  3.0
c  0.0  3.0  0.0
```

**注意点・落とし穴**:
- `from_numpy_array` のノードは行番号 `0..n-1` になる。元の名前を付けたいときは `nodelist=[...]`(3.6.1 の引数)を渡すか `relabel_nodes` を使う。既定では無向グラフになるので、上三角と下三角が非対称な行列(例 `[[0,1],[2,0]]`)を渡しても向きは失われ、辺は1本で重みは 2(後から読まれた値)になる。**有向にしたいなら `create_using=nx.DiGraph`** を必ず付ける。
- `to_numpy_array` は `weight` 属性を持たない辺の値を 1 とする。重みなし(0/1)にしたいときは `weight=None`。多重グラフでは `multigraph_weight`(既定 `sum`)で重ねた辺の扱いを決める。ノード数の大きなグラフを密な配列にすると n×n のメモリが必要なので、疎行列(`to_scipy_sparse_array`)を使う。

---

### `networkx.to_scipy_sparse_array / from_scipy_sparse_array / adjacency_matrix`

**用途**: グラフを SciPy の疎行列(隣接行列・ラプラシアン行列)に変換する、またはその逆。大きなグラフを行列演算やスペクトル解析にかけるときに使う。

**シグネチャ**:
- `networkx.to_scipy_sparse_array(G, nodelist=None, dtype=None, weight='weight', format='csr')`
- `networkx.from_scipy_sparse_array(A, parallel_edges=False, create_using=None, edge_attribute='weight')`
- `networkx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')`
- `networkx.laplacian_matrix(G, nodelist=None, weight='weight')`

**使用例**:
```python
import networkx as nx
import numpy as np
from scipy import sparse
G = nx.path_graph(4)
A = nx.to_scipy_sparse_array(G)                 # 既定は CSR 形式
print(type(A).__name__, A.shape, A.nnz)
print(A.toarray())
print(type(nx.adjacency_matrix(G)).__name__)
L = nx.laplacian_matrix(G)
print(L.toarray())
S = sparse.csr_array(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
H = nx.from_scipy_sparse_array(S)
print(H.number_of_nodes(), list(H.edges(data=True)))
```
実行結果:
```
csr_array (4, 4) 6
[[0 1 0 0]
 [1 0 1 0]
 [0 1 0 1]
 [0 0 1 0]]
csr_array
[[ 1 -1  0  0]
 [-1  2 -1  0]
 [ 0 -1  2 -1]
 [ 0  0 -1  1]]
3 [(0, 1, {'weight': 1}), (1, 2, {'weight': 1})]
```

**注意点・落とし穴**:
- 3.6.1 では戻り値は疎な「array」(`csr_array`)。古い `scipy.sparse.csr_matrix`(matrix API)ではないため、`*` は要素積、行列積は `@` を使う(matrix API の `*` は行列積だった)。`nx.adjacency_matrix` も同じく `csr_array`。
- `nnz` は非ゼロ要素数で、無向グラフでは各辺が (i, j) と (j, i) の両方に入るので辺数の2倍(上の例 6)。ラプラシアンは `L = D - A`。ノード順は `list(G.nodes)`(`nodelist=` で指定可)。

---

### `networkx.write_edgelist / read_edgelist / write_adjlist / read_adjlist`

**用途**: テキストの辺リスト(1行=1辺)・隣接リスト(1行=1ノードとその隣接)の読み書き。シンプルなテキスト形式で、他のツールとのやり取りにも使える。

**シグネチャ**:
- `networkx.write_edgelist(G, path, comments='#', delimiter=' ', data=True, encoding='utf-8')`
- `networkx.read_edgelist(path, comments='#', delimiter=None, create_using=None, nodetype=None, data=True, edgetype=None, encoding='utf-8')`
- `networkx.write_adjlist(G, path, comments='#', delimiter=' ', encoding='utf-8')`
- `networkx.read_adjlist(path, comments='#', delimiter=None, create_using=None, nodetype=None, encoding='utf-8')`

**使用例**:
```python
from pathlib import Path
import networkx as nx
G = nx.Graph()
G.add_edge("a", "b", weight=1.5)
G.add_edge("b", "c", weight=2.0)
nx.write_edgelist(G, "g.edgelist", data=["weight"])       # 出力する属性を指定
print(Path("g.edgelist").read_text().strip())
H = nx.read_edgelist("g.edgelist", data=[("weight", float)])   # 型を指定して読む
print(H.edges(data=True))
try:
    nx.read_edgelist("g.edgelist")                          # 既定 data=True は辞書表記を期待する
except TypeError as e:
    print("TypeError:", e)
nx.write_edgelist(G, "g2.edgelist")                         # data=True(既定)で書くと辞書表記
print(Path("g2.edgelist").read_text().strip().splitlines()[0])
print(nx.read_edgelist("g2.edgelist").edges(data=True))
I = nx.read_edgelist("g.edgelist", nodetype=str, data=[("weight", float)], create_using=nx.DiGraph)
print(I.is_directed())
nx.write_adjlist(G, "g.adjlist")
print(Path("g.adjlist").read_text().strip().splitlines()[-2:])
```
実行結果:
```
a b 1.5
b c 2.0
[('a', 'b', {'weight': 1.5}), ('b', 'c', {'weight': 2.0})]
TypeError: Failed to convert edge data (['1.5']) to dictionary.
a b {'weight': 1.5}
[('a', 'b', {'weight': 1.5}), ('b', 'c', {'weight': 2.0})]
True
['b c', 'c']
```

**注意点・落とし穴**:
- `read_edgelist` は既定でノードを文字列として読み、数値ノードは `nodetype=int` を指定しないと `'1'` のような文字列になる。辺属性は既定 `data=True` だと「辞書表記(`{'weight': 1.5}`)」を期待するので、`a b 1.5` のような素の値の行は `TypeError`(上の出力)。素の値を読むには `data=[('weight', float)]` のように名前と型を渡し、属性を無視するなら `data=False`。
- `write_edgelist` の `data=True`(既定)は属性 dict を丸ごと `{'weight': 1.5}` の形で出力し、既定同士で `read_edgelist` すれば dict が復元される(上の最終行)。属性を1つずつの列で出したいときは `data=['weight']`。辺属性しか保存されず、ノード属性は保存されない。

---

### `networkx.write_gml / read_gml`

**用途**: GML(Graph Modelling Language)形式の読み書き。ノード・辺の属性(数値・文字列)を保持できるテキスト形式で、Gephi などの外部ツールとも互換。

**シグネチャ**:
- `networkx.write_gml(G, path, stringizer=None)`
- `networkx.read_gml(path, label='label', destringizer=None)`
- `networkx.parse_gml(lines, label='label', destringizer=None)`

**使用例**:
```python
from pathlib import Path
import networkx as nx
G = nx.Graph(name="demo")
G.add_node("a", weight=1.5)
G.add_node("b", color="red")
G.add_edge("a", "b", length=3)
nx.write_gml(G, "g.gml")
print(Path("g.gml").read_text())
H = nx.read_gml("g.gml")
print(H.nodes(data=True), H.edges(data=True))
I = nx.read_gml("g.gml", label="id")        # ノード名を 'id' (連番)にして、label は属性へ
print(list(I.nodes(data=True)))
nx.write_gml(nx.path_graph(3), "p.gml")     # 整数ノードの往復
print(list(nx.read_gml("p.gml").nodes), list(nx.read_gml("p.gml", destringizer=int).nodes))
```
実行結果:
```
graph [
  name "demo"
  node [
    id 0
    label "a"
    weight 1.5
  ]
  node [
    id 1
    label "b"
    color "red"
  ]
  edge [
    source 0
    target 1
    length 3
  ]
]

[('a', {'weight': 1.5}), ('b', {'color': 'red'})] [('a', 'b', {'length': 3})]
[(0, {'label': 'a', 'weight': 1.5}), (1, {'label': 'b', 'color': 'red'})]
['0', '1', '2'] [0, 1, 2]
```

**注意点・落とし穴**:
- `write_gml` は各ノードに `label`(元のノード名を文字列化したもの)と連番の `id` を出力する。**整数ノードで書いても `read_gml` すると文字列 `'0'` で戻る**(上の最終行)。整数に戻すには `destringizer=int`、または `label='id'`(連番 `id` が名前になり、元の名前は属性 `label` に残る)。
- 属性値として使えるのは数値・文字列・真偽値と、リスト・辞書まで(タプルはリストとして書かれ、読み戻すとリスト。`True` は `1` の整数で戻る)。`set` / `None` / 任意オブジェクトは `NetworkXError: ... is not a string`(`write_gml(G, path, stringizer=...)` に文字列化関数を渡せば書き出せる)。

---

### `networkx.node_link_data / node_link_graph`

**用途**: グラフを JSON 化しやすい dict(ノードのリスト+辺のリスト)に変換する、およびその逆。D3.js などの JavaScript 可視化との連携や、JSON 保存に使う。

**シグネチャ**:
- `networkx.node_link_data(G, *, source='source', target='target', name='id', key='key', edges='edges', nodes='nodes')`
- `networkx.node_link_graph(data, directed=False, multigraph=True, *, source='source', target='target', name='id', key='key', edges='edges', nodes='nodes')`

**使用例**:
```python
import json
import networkx as nx
G = nx.DiGraph()
G.add_node("a", size=3)
G.add_edge("a", "b", weight=2.5)
data = nx.node_link_data(G, edges="edges")
print(json.dumps(data))
H = nx.node_link_graph(json.loads(json.dumps(data)), edges="edges")
print(H.is_directed(), list(H.edges(data=True)), H.nodes["a"])
```
実行結果:
```
{"directed": true, "multigraph": false, "graph": {}, "nodes": [{"size": 3, "id": "a"}, {"id": "b"}], "edges": [{"weight": 2.5, "source": "a", "target": "b"}]}
True [('a', 'b', {'weight': 2.5})] {'size': 3}
```

**注意点・落とし穴**:
- 3.6.1 のシグネチャでは辺のリストのキー名は `edges='edges'` が既定。JSON 側が別名(たとえば D3.js 流の `'links'`)の場合は `edges='links'`、`nodes='vertices'` のように名前を合わせる。
- `node_link_data` の出力には `directed` / `multigraph` / `graph` キーが含まれ、`node_link_graph` はそれを読んで有向・多重の別を復元する(上の例では `is_directed` が True に戻る)。データにこれらのキーがない場合に限り、引数の既定値 `directed=False, multigraph=True` が使われるため、外部の JSON からは **`MultiGraph`** になる。単純な `Graph` にしたいなら `multigraph=False`。集合など JSON 化できない属性値があると `json.dumps` で失敗する。

---

### `networkx.write_graphml / read_graphml`

**用途**: GraphML(XML ベースのグラフ形式)の読み書き。属性の型(整数・浮動小数・文字列・真偽値)を保持でき、Gephi・Cytoscape などとやり取りできる。

**シグネチャ**:
- `networkx.write_graphml(G, path, encoding='utf-8', prettyprint=True, infer_numeric_types=False, named_key_ids=False, edge_id_from_attribute=None)`
- `networkx.read_graphml(path, node_type=<class 'str'>, edge_key_type=<class 'int'>, force_multigraph=False)`

**使用例**:
```python
import networkx as nx
G = nx.DiGraph()
G.add_node(1, label="x", size=2.5)
G.add_edge(1, 2, weight=3)
nx.write_graphml(G, "g.graphml")
H = nx.read_graphml("g.graphml")
print(H.is_directed(), list(H.nodes(data=True)), list(H.edges(data=True)))
K = nx.read_graphml("g.graphml", node_type=int)
print(list(K.nodes))
```
実行結果:
```
True [('1', {'label': 'x', 'size': 2.5}), ('2', {})] [('1', '2', {'weight': 3})]
[1, 2]
```

**注意点・落とし穴**:
- `read_graphml` は既定でノード名を **文字列** として読む(整数ノードで書いても `'1'` で戻る)ので、`node_type=int` を指定する。辺属性の型 `weight=3` は整数のまま戻る。
- 属性値に使えるのは数値・文字列・真偽値のみ。リストなどの属性があると書き出しで `NetworkXError: GraphML writer does not support <class 'list'> as data values.`。なお `lxml` が入っていない環境でも標準ライブラリの XML 実装にフォールバックして書き出せる(この環境は lxml なし)。

---

## 描画とレイアウト

### `networkx.draw(G, pos=None, ax=None, **kwds)`

**用途**: グラフを matplotlib で描画する最も簡単な方法。`draw` はラベルなしが既定で、`draw_networkx` はラベルありが既定(どちらも同じキーワード引数を受け付ける)。

**シグネチャ**:
- `networkx.draw(G, pos=None, ax=None, **kwds)`
- `networkx.draw_networkx(G, pos=None, arrows=None, with_labels=True, **kwds)`

**使用例**:
```python
import matplotlib
matplotlib.use("Agg")                      # 画面なし環境用(保存のみ)
import matplotlib.pyplot as plt
import networkx as nx
G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=7)          # 座標を固定して再現性を確保
colors = ["tab:red" if G.nodes[n]["club"] == "Mr. Hi" else "tab:blue" for n in G]
fig, ax = plt.subplots(figsize=(6, 6))
nx.draw(G, pos, ax=ax, node_color=colors, node_size=150, edge_color="gray",
        with_labels=True, font_size=7, width=0.8)
fig.savefig("karate.png", dpi=80)
plt.close(fig)
print(type(pos).__name__, len(pos), pos[0].shape)
```
実行結果:
```
dict 34 (2,)
```

**注意点・落とし穴**:
- `pos` を省略すると呼び出しのたびに `spring_layout`(乱数を使う)が実行され、描くたびに配置が変わる。複数の図で同じ配置を使う/再現するには、`pos = nx.spring_layout(G, seed=...)` を先に計算して渡す。
- `draw` の色・サイズ・ラベルの指定は `node_color`, `node_size`, `edge_color`, `width`, `with_labels`, `font_size` などのキーワード引数(内部で `draw_networkx_nodes/edges/labels` に振り分けられる)。ノードごとの色はノード順(`G.nodes` の順)のリストで渡す。
- Jupyter 以外(スクリプト・サーバ)では `plt.show()` の代わりに `fig.savefig(...)` する。GUI がない環境では `matplotlib.use('Agg')` が必要。有向グラフには自動で矢印が付く(`arrows=False` で消せる)。
- 出力画像(`karate.png`)を実際に開いて確認: 赤(`Mr. Hi`)のノードが図の左寄り、青(`Officer`)のノードが右寄りに固まり、ラベル付きの丸(直径約 150)が灰色の細い辺でつながる。ノード 0(赤側)と 33(青側)は両グループの中央付近に位置し、多くの辺が集まっている。

---

### `networkx.spring_layout / circular_layout / kamada_kawai_layout / shell_layout ほか`

**用途**: ノードの座標を計算する。返り値は `{ノード: 座標のnumpy配列}` の辞書で、`draw(G, pos)` に渡す。`spring_layout`(力学モデル)が汎用、`kamada_kawai_layout` は小〜中規模で見やすい、`circular_layout` は円周上、`shell_layout` は同心円、`bipartite_layout` は二部グラフを左右2列に並べる。

**シグネチャ**:
- `networkx.spring_layout(G, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=None, store_pos_as=None, *, method='auto', gravity=1.0)`
- `networkx.circular_layout(G, scale=1, center=None, dim=2, store_pos_as=None)`
- `networkx.kamada_kawai_layout(G, dist=None, pos=None, weight='weight', scale=1, center=None, dim=2, store_pos_as=None)`
- `networkx.shell_layout(G, nlist=None, rotate=None, scale=1, center=None, dim=2, store_pos_as=None)`
- `networkx.spectral_layout(G, weight='weight', scale=1, center=None, dim=2, store_pos_as=None)`
- `networkx.random_layout(G, center=None, dim=2, seed=None, store_pos_as=None)`
- `networkx.bipartite_layout(G, nodes=None, align='vertical', scale=1, center=None, aspect_ratio=1.3333333333333333, store_pos_as=None)`
- `networkx.planar_layout(G, scale=1, center=None, dim=2, store_pos_as=None)`

**使用例**:
```python
import networkx as nx
G = nx.cycle_graph(5)
pos = nx.circular_layout(G)
print({n: p.round(3).tolist() for n, p in pos.items()})
sp = nx.spring_layout(G, seed=42)
print({n: p.round(3).tolist() for n, p in list(sp.items())[:2]})
print(nx.spring_layout(G, seed=42)[0].round(3), nx.spring_layout(G, seed=43)[0].round(3))   # seed が違えば座標も違う
kk = nx.kamada_kawai_layout(G)
print(kk[0].round(3))
B = nx.complete_bipartite_graph(2, 3)
bp = nx.bipartite_layout(B, nodes=[0, 1])
print({n: p.round(2).tolist() for n, p in bp.items()})
print(nx.planar_layout(nx.path_graph(3))[1].round(3))
try:
    nx.planar_layout(nx.complete_graph(5))
except nx.NetworkXException as e:
    print(type(e).__name__, e)
```
実行結果:
```
{0: [1.0, 0.0], 1: [0.309, 0.951], 2: [-0.809, 0.588], 3: [-0.809, -0.588], 4: [0.309, -0.951]}
{0: [0.701, 0.725], 1: [0.908, -0.441]}
[0.701 0.725] [-1.    -0.055]
[1. 0.]
{0: [-1.0, -0.62], 1: [-1.0, 0.62], 2: [0.67, -0.62], 3: [0.67, 0.0], 4: [0.67, 0.62]}
[ 1.    -0.333]
NetworkXException G is not planar.
```

**注意点・落とし穴**:
- `spring_layout` / `random_layout` は乱数を使うので `seed=` を指定して再現させる(seed が違えば座標が変わる)。座標のスケールは既定で `[-1, 1]` 程度。`scale=` `center=` `dim=` で調整でき、`dim=3` なら3次元。
- `kamada_kawai_layout` は全ノード対の最短経路距離を使うため、ノード数の多いグラフでは重い(小〜中規模向け)。`planar_layout` は平面グラフ専用で、平面でないグラフ(K5 など)は `NetworkXException: G is not planar.`。
- `bipartite_layout(G, nodes=...)` の `nodes` は片側のグループを渡す必要がある。`shell_layout(G, nlist=[[...],[...]])` で層を指定する。`pos` の値は numpy 配列なので、`pos[n][0]` が x、`pos[n][1]` が y。

---

### `networkx.draw_networkx_nodes / draw_networkx_edges / draw_networkx_labels`

**用途**: 描画を「ノード」「辺」「ラベル」の部品に分けて、それぞれ個別に細かく指定する。ノードの大きさ・色を数値(次数や中心性)で連動させたい場合や、辺ごとに幅・色を変えたい場合に使う。

**シグネチャ**:
- `networkx.draw_networkx_nodes(G, pos, nodelist=None, node_size=300, node_color='#1f78b4', node_shape='o', alpha=None, cmap=None, vmin=None, vmax=None, ax=None, linewidths=None, edgecolors=None, label=None, margins=None, hide_ticks=True)`
- `networkx.draw_networkx_edges(G, pos, edgelist=None, width=1.0, edge_color='k', style='solid', alpha=None, arrowstyle=None, arrowsize=10, edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None, arrows=None, label=None, node_size=300, nodelist=None, node_shape='o', connectionstyle='arc3', min_source_margin=0, min_target_margin=0, hide_ticks=True)`
- `networkx.draw_networkx_labels(G, pos, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, clip_on=True, hide_ticks=True)`

**使用例**:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=7)
deg = dict(G.degree)
bc = nx.betweenness_centrality(G)
fig, ax = plt.subplots(figsize=(6, 6))
nodes = nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_size=[100 + 40 * deg[n] for n in G],   # 次数でサイズ
                               node_color=[bc[n] for n in G],              # 媒介中心性で色
                               cmap="viridis")
widths = [0.3 + 0.4 * G[u][v]["weight"] for u, v in G.edges]               # 辺の重みで太さ
nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.5)
nx.draw_networkx_labels(G, pos, labels={n: n for n in (0, 33)}, font_size=10, font_color="white", ax=ax)
fig.colorbar(nodes, ax=ax, shrink=0.7, label="betweenness")
ax.set_axis_off()
fig.savefig("karate_parts.png", dpi=80)
plt.close(fig)
print(type(nodes).__name__)
```
実行結果:
```
PathCollection
```

**注意点・落とし穴**:
- `draw_networkx_nodes` は matplotlib の `PathCollection` を返すので、そのまま `fig.colorbar(...)` に渡せる(数値で色を付けたときに使う)。`node_color` に数値列を渡すと `cmap`(カラーマップ)で着色される。
- 出力画像(`karate_parts.png`)を確認: ノード 0 が黄色(媒介中心性 約 0.44 で最大)、33 が緑(約 0.30)、その他は紫〜青の小さめの丸。ラベルは 0 と 33 だけ白文字で表示され、右側に `betweenness` のカラーバーが付く。辺は重みが大きいほど太い。
- `node_size` や `node_color` のリストは `G.nodes` の順に対応する。`nodelist=` を指定した場合はその順に対応させること。`draw_networkx_labels` の `labels={ノード: 文字列}` で一部のノードだけにラベルを付けられる。

---

### `networkx.draw_networkx_edge_labels / 有向グラフの描画`

**用途**: 辺の上にテキスト(重みなど)を描く。`edge_labels={(u, v): 表示文字}` を渡す。有向グラフの矢印は `draw_networkx_edges` の `arrows` / `connectionstyle` で制御する。

**シグネチャ**: `networkx.draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, rotate=True, clip_on=True, node_size=300, nodelist=None, connectionstyle='arc3', hide_ticks=True)`

**使用例**:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
D = nx.DiGraph()
D.add_weighted_edges_from([("A", "B", 3), ("B", "C", 1), ("A", "C", 7), ("C", "A", 2)])
pos = nx.circular_layout(D)
fig, ax = plt.subplots(figsize=(4, 4))
nx.draw_networkx_nodes(D, pos, node_size=600, node_color="lightyellow", edgecolors="black", ax=ax)
nx.draw_networkx_labels(D, pos, ax=ax)
nx.draw_networkx_edges(D, pos, arrows=True, arrowsize=18, node_size=600,
                       connectionstyle="arc3,rad=0.15", ax=ax)          # 往復の辺が重ならない
labels = nx.get_edge_attributes(D, "weight")
nx.draw_networkx_edge_labels(D, pos, edge_labels=labels, connectionstyle="arc3,rad=0.15", ax=ax)
ax.set_axis_off()
fig.savefig("digraph_labels.png", dpi=80)
plt.close(fig)
print(labels)
```
実行結果:
```
{('A', 'B'): 3, ('A', 'C'): 7, ('B', 'C'): 1, ('C', 'A'): 2}
```

**注意点・落とし穴**:
- 双方向の辺(A→C と C→A)は、既定の直線だと重なって片方が見えなくなる。`connectionstyle='arc3,rad=0.15'` で辺を弧にできる。`draw_networkx_edge_labels` も `connectionstyle` を受け取るので、同じ値を渡すとラベルが弧に沿った位置に描かれる。
- `draw_networkx_edges` の `node_size` は、矢印の先端をノードの円の縁に合わせる位置計算に使われる(ノード側の `node_size` と同じ値を渡す)。`edge_labels` は `{(u, v): 表示文字}` の辞書で、`nx.get_edge_attributes(D, 'weight')` の結果をそのまま渡せる。
- 出力画像(`digraph_labels.png`)を確認: A・B・C が三角形状に並び、A→B(ラベル 3)、B→C(1)、A→C(7)、C→A(2)が弧で描かれる。A と C の間の往復2本は別々の弧になって重ならず、矢印の先端が各ノードの円の縁に接している。

---

### `networkx.nx_agraph / nx_pydot(Graphviz 連携)`

**用途**: Graphviz(`dot`, `neato` 等)のレイアウトで座標を得る、または DOT 形式を出力する。階層構造(木・DAG)の見やすい配置に使われる。**この環境では pygraphviz / pydot が未インストール**なので、動作確認できていない。

**シグネチャ**:
- `networkx.nx_agraph.graphviz_layout(G, prog='neato', root=None, args='')  # pygraphviz が必要`
- `networkx.nx_pydot.graphviz_layout(G, prog='neato', root=None)  # pydot と Graphviz 本体が必要`

**使用例**:
```python
import networkx as nx
G = nx.path_graph(3)
try:
    nx.nx_agraph.graphviz_layout(G, prog="dot")
except ImportError as e:
    print("ImportError(nx_agraph):", e)
try:
    nx.nx_pydot.graphviz_layout(G, prog="dot")
except ImportError as e:
    print("ImportError(nx_pydot):", e)
```
実行結果:
```
ImportError(nx_agraph): requires pygraphviz http://pygraphviz.github.io/
ImportError(nx_pydot): No module named 'pydot'
```

**注意点・落とし穴**:
- Graphviz 連携には、別途 `pygraphviz`(または `pydot`)と、OS 側の Graphviz 本体のインストールが必要。上の出力は、これらが無い環境で呼び出したときの実際のエラーメッセージ。
- ツリーの階層レイアウトが必要だが Graphviz を入れたくない場合は、NetworkX 標準の `nx.multipartite_layout`(層のノード属性 `subset_key` で列を指定)や、`topological_generations` の結果をもとに自前で座標を作る。

---

## その他(同型性・クリーク・彩色・リンク予測)

### `networkx.is_isomorphic / isomorphism.GraphMatcher`

**用途**: グラフの同型判定(ノード名を付け替えれば同じ構造か)と、部分グラフの同型(パターンマッチ)。`node_match` / `edge_match` で属性も一致条件にできる。

**シグネチャ**:
- `networkx.is_isomorphic(G1, G2, node_match=None, edge_match=None)`
- `networkx.could_be_isomorphic(G1, G2, *, properties='dtc')`
- `networkx.isomorphism.GraphMatcher(G1, G2, node_match=None, edge_match=None)`

**使用例**:
```python
import networkx as nx
from networkx.algorithms import isomorphism as iso
G1 = nx.path_graph(4)
G2 = nx.relabel_nodes(G1, {0: "a", 1: "b", 2: "c", 3: "d"})
print(nx.is_isomorphic(G1, G2), nx.is_isomorphic(G1, nx.star_graph(3)))
gm = iso.GraphMatcher(G1, G2)
print(gm.is_isomorphic(), gm.mapping)                      # G1 -> G2 のノード対応
GM = iso.GraphMatcher(nx.cycle_graph(6), nx.path_graph(3))
print(GM.subgraph_is_isomorphic())                          # 大きいグラフ内に小さいパターンがあるか
A = nx.Graph(); A.add_nodes_from([(0, {"c": "r"}), (1, {"c": "b"})]); A.add_edge(0, 1)
B = nx.Graph(); B.add_nodes_from([(0, {"c": "b"}), (1, {"c": "b"})]); B.add_edge(0, 1)
print(nx.is_isomorphic(A, B))                               # 構造だけなら同型
print(nx.is_isomorphic(A, B, node_match=iso.categorical_node_match("c", None)))
print(nx.could_be_isomorphic(G1, nx.star_graph(3)))         # 高速な必要条件チェック
```
実行結果:
```
True False
True {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
True
True
False
False
```

**注意点・落とし穴**:
- 同型判定は一般に計算量が大きく、大きなグラフでは時間がかかる。事前に `could_be_isomorphic`(次数列・三角形数・クリーク数の必要条件、既定 `properties='dtc'`)で明らかな非同型を弾ける。`True` でも同型とは限らない(立方体グラフ `hypercube_graph(3)` と 8 頂点の Wagner グラフ `circulant_graph(8, [1, 4])` は、`could_be_isomorphic` が True だが `is_isomorphic` は False)。
- `subgraph_is_isomorphic` は「誘導部分グラフ」としての同型で、辺が余分にあってはいけない。辺が欠けていてもよい(モノモルフィズム)なら `subgraph_is_monomorphic`。

---

### `networkx.find_cliques / max_weight_clique`

**用途**: クリーク(全ノードが互いにつながった部分集合)を探す。`find_cliques` は極大クリークを全て列挙し、`max_weight_clique` は重み合計が最大(`weight=None` ならノード数最大)のクリークを返す。

**シグネチャ**:
- `networkx.find_cliques(G, nodes=None)`
- `networkx.max_weight_clique(G, weight='weight')`

**使用例**:
```python
import networkx as nx
G = nx.karate_club_graph()
cliques = list(nx.find_cliques(G))
print(len(cliques), max(len(c) for c in cliques))
print(sorted(max(cliques, key=len)))
clique, w = nx.max_weight_clique(G, weight=None)
print(sorted(clique), w)
print(sorted(map(sorted, nx.find_cliques(nx.Graph([(1, 2), (2, 3), (1, 3), (3, 4)])))))
```
実行結果:
```
36 5
[0, 1, 2, 3, 13]
[0, 1, 2, 3, 7] 5
[[1, 2, 3], [3, 4]]
```

**注意点・落とし穴**:
- `find_cliques` は極大クリーク(それ以上大きくできないもの)の列挙で、最大クリークだけとは限らず数が非常に多くなることがある(ジェネレータなので最大長は `max(map(len, ...))` で得る)。無向グラフ専用。
- `max_weight_clique` の `weight` は既定 `'weight'` だが、これは **ノード属性** の名前(辺ではない)。ノードにその属性がなければ `KeyError: 'Node 0 does not have the requested weight field.'`。ノード数最大なら `weight=None`。同じ大きさの最大クリークが複数あるとき、`find_cliques` から選んだもの(上の例 `[0, 1, 2, 3, 13]`)と `max_weight_clique` が返すもの(`[0, 1, 2, 3, 7]`)は一致するとは限らない。3.6.1 に `graph_clique_number` は存在しない。

---

### `networkx.greedy_color(G, strategy)`

**用途**: グラフ彩色(隣り合うノードが同じ色にならないよう、少ない色で塗る)を貪欲法で行う。返り値は `{ノード: 色番号}`。スケジューリングや周波数割当のヒューリスティックに使う。

**シグネチャ**: `networkx.greedy_color(G, strategy='largest_first', interchange=False)`

**使用例**:
```python
import networkx as nx
G = nx.karate_club_graph()
c = nx.greedy_color(G)                                    # 既定 strategy='largest_first'
print(max(c.values()) + 1)
print(all(c[u] != c[v] for u, v in G.edges))               # 隣同士は別の色
for st in ["largest_first", "smallest_last", "connected_sequential_bfs", "DSATUR"]:
    print(st, max(nx.greedy_color(G, strategy=st).values()) + 1)
# 二部グラフ(最適は2色)でも、ノードを処理する順序が悪いと色が増える
B = nx.Graph([(("a", i), ("b", j)) for i in range(5) for j in range(5) if i != j])
order = [(s, i) for i in range(5) for s in "ab"]          # a0,b0,a1,b1,...
print(max(nx.greedy_color(B, strategy="largest_first").values()) + 1,
      max(nx.greedy_color(B, strategy=lambda G, colors: order).values()) + 1)
```
実行結果:
```
5
True
largest_first 5
smallest_last 5
connected_sequential_bfs 6
DSATUR 5
2 5
```

**注意点・落とし穴**:
- 貪欲法なので、使う色数は最小(彩色数)より多くなりうる。karate では `connected_sequential_bfs` だけ 6 色になった(下限は 5 クリークの存在から 5)。`'DSATUR'` は `'saturation_largest_first'` の別名。
- `strategy` には文字列のほか、`(G, colors) -> ノードの順序` の関数も渡せる。上の二部グラフ例で、順序次第で 2 色で塗れるグラフが 5 色になることを確認できる。`interchange=True` で色の入れ替えによる改善を試せるが、`'independent_set'` と `'saturation_largest_first'`(DSATUR)とは併用できず `NetworkXPointlessConcept` になる。

---

### `networkx.jaccard_coefficient / adamic_adar_index(リンク予測)`

**用途**: リンク予測(まだ辺がないノード対が将来つながりやすいかのスコア)。共通の隣人の多さから、Jaccard 係数・Adamic-Adar 指標などで算出する。返り値は `(u, v, スコア)` のジェネレータ。

**シグネチャ**:
- `networkx.jaccard_coefficient(G, ebunch=None)`
- `networkx.adamic_adar_index(G, ebunch=None)`

**使用例**:
```python
import networkx as nx
G = nx.karate_club_graph()
top = sorted(nx.jaccard_coefficient(G), key=lambda t: -t[2])[:3]   # ebunch省略: 全ての非隣接ペア
print(top)
print(sorted(nx.adamic_adar_index(G, [(0, 9), (0, 4)])))
print(list(nx.jaccard_coefficient(nx.path_graph(3))))
```
実行結果:
```
[(14, 15, 1.0), (14, 18, 1.0), (14, 20, 1.0)]
[(0, 4, 1.631586747071319), (0, 9, 0.43429448190325176)]
[(0, 2, 1.0)]
```

**注意点・落とし穴**:
- `ebunch` を省略すると **辺のない全ノード対**(非隣接ペア)が対象になる。既存の辺のペアも計算したいなら `ebunch=G.edges`(例 `path_graph(3)` の辺 (0,1) は共通の隣人がなくスコア 0.0)を渡す。ノード数が大きいとペア数が O(n²) で重い。
- 有向・多重グラフ非対応(`NetworkXNotImplemented`)。他に `resource_allocation_index`, `preferential_attachment`, `common_neighbor_centrality` など同系統の関数がある。
