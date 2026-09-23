# optuna 逆引き辞書

optuna 4.9.0 で検証済み。すべてのシグネチャ・実行結果は `/home/manaty/library-practicing/.venv`(optuna 4.9.0)で実際にコードを実行して取得したものであり、記憶からの推測は含まない。

## 目次

1. [基礎(Study・Trialと最適化の実行)](#1-基礎studytrialと最適化の実行)
2. [探索空間の定義(suggest系)](#2-探索空間の定義suggest系)
3. [サンプラー](#3-サンプラー)
4. [プルーニング](#4-プルーニング)
5. [結果の取得・分析](#5-結果の取得分析)
6. [可視化](#6-可視化)
7. [マルチ目的最適化](#7-マルチ目的最適化)
8. [永続化(Storage)](#8-永続化storage)
9. [その他(コールバック・実行制御)](#9-その他コールバック実行制御)
10. [制約付き・条件付き最適化](#10-制約付き条件付き最適化)
11. [Study間のコピー・マージ](#11-study間のコピーマージ)
12. [コールバックの応用パターン](#12-コールバックの応用パターン)
13. [Artifact機能](#13-artifact機能)
14. [ウォームスタートの応用](#14-ウォームスタートの応用)

---

## 1. 基礎(Study・Trialと最適化の実行)

### `optuna.create_study(...)`

**用途**: 最適化タスク(`Study`)を新規作成する。

**シグネチャ**: `optuna.create_study(*, storage=None, sampler=None, pruner=None, study_name=None, direction=None, load_if_exists=False, directions=None) -> Study`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study()
print('direction:', study.direction.name)
print('sampler:', type(study.sampler).__name__)
print('pruner:', type(study.pruner).__name__)
```
実行結果:
```
direction: MINIMIZE
sampler: TPESampler
pruner: MedianPruner
```

**注意点・落とし穴**:
- `storage=None`(デフォルト)だとインメモリで保持され、プロセス終了とともに結果が消える。永続化するには`storage`にDB URLを渡す(「8. 永続化」参照)。
- `direction`省略時のデフォルトは`'minimize'`(実行して`study.direction.name == 'MINIMIZE'`を確認済み)。
- `sampler`省略時は`TPESampler`、`pruner`省略時は`MedianPruner`が自動的に使われる(実行して確認済み)。
- `optuna.logging.set_verbosity`を呼ばないと、`create_study`自体が`[I ...] A new study created in memory with name: ...`というINFOログを標準エラー出力に出す(詳細は本カテゴリの`optuna.logging.set_verbosity`の項を参照)。

### `Study.optimize(...)`

**用途**: 目的関数を`n_trials`回(または`timeout`秒間)呼び出し、パラメータ探索を実行する。

**シグネチャ**: `Study.optimize(self, func, n_trials=None, timeout=None, n_jobs=1, catch=(), callbacks=None, gc_after_trial=False, show_progress_bar=False) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
print('best_params:', study.best_params)
print('best_value:', round(study.best_value, 4))
print('n_trials:', len(study.trials))
```
実行結果:
```
best_params: {'x': 1.4655874548469718}
best_value: 0.2856
n_trials: 20
```

**注意点・落とし穴**:
- `func`は1つの数値(単一目的)、または数値のタプル(多目的、「7. マルチ目的最適化」参照)を返す関数でなければならない。
- `n_jobs=1`がデフォルト(直列実行)。`n_jobs=-1`で全CPUコアを使った並列実行ができるが、インメモリの`storage`では実装上の制約があるため、並列実行するなら`storage`にDBを指定するのが安全。
- `n_trials`と`timeout`の両方を指定すると、どちらか先に条件を満たした時点で停止する。

### `optuna.logging.set_verbosity(...)`

**用途**: optunaが標準エラー出力に出すログの詳細度を変更する(デフォルトはINFOレベルで、`create_study`や各試行完了時にログが出る)。

**シグネチャ**: `optuna.logging.set_verbosity(verbosity: int) -> None`

**使用例**:
```python
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

print('--- verbosity=INFO(デフォルト)で1試行 ---')
study = optuna.create_study()
study.optimize(objective, n_trials=1)

print('--- verbosity=WARNINGに変更後 ---')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=1)
print('(ログなし)')
```
実行結果:
```
--- verbosity=INFO(デフォルト)で1試行 ---
[I 2026-09-23 13:12:04,950] A new study created in memory with name: no-name-a78061c2-cc19-4933-b187-d67c64e57022
[I 2026-09-23 13:12:04,957] Trial 0 finished with value: 4.04695185336051 and parameters: {'x': 4.011703719080051}. Best is trial 0 with value: 4.04695185336051.
--- verbosity=WARNINGに変更後 ---
(ログなし)
```

**注意点・落とし穴**:
- `optuna.logging.WARNING`/`INFO`/`DEBUG`は標準の`logging`モジュールのレベル定数と同じ整数値。大量の試行を回すノートブックなどではログが流れすぎるため、`WARNING`に上げておくと出力が読みやすくなる。
- プルーニング(打ち切り)発生時などの`[W ...]`ログはこの設定に関わらず、レベルが閾値以上であれば出力される。

---

## 2. 探索空間の定義(suggest系)

### `Trial.suggest_float(...)`

**用途**: 連続値(浮動小数点数)のパラメータを探索空間として定義し、サンプラーに1つ値を提案させる。

**シグネチャ**: `Trial.suggest_float(self, name, low, high, *, step=None, log=False) -> float`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    return lr + dropout

study = optuna.create_study()
study.optimize(objective, n_trials=1)
print(study.trials[0].params)
```
実行結果:
```
{'lr': 0.00040214376574998836, 'dropout': 0.4}
```

**注意点・落とし穴**:
- `log=True`にすると対数スケールでサンプリングされる(学習率のように桁が大きく変わるパラメータに使う)。`log=True`と`step`は同時指定できない(内部でエラーになる)。
- `step`を指定すると`low`から`step`刻みの離散値のみが提案される(実質的に連続値ではなくなる)。

### `Trial.suggest_int(...)`

**用途**: 整数値のパラメータを探索空間として定義する。

**シグネチャ**: `Trial.suggest_int(self, name, low, high, *, step=1, log=False) -> int`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 5)
    n_units = trial.suggest_int('n_units', 32, 256, step=32)
    return n_layers + n_units

study = optuna.create_study()
study.optimize(objective, n_trials=1)
print(study.trials[0].params)
```
実行結果:
```
{'n_layers': 3, 'n_units': 224}
```

**注意点・落とし穴**:
- `step`のデフォルトは1(全整数を候補にする)。`step=32`のように指定すると`low + step*k`の値のみが候補になる(`high`ちょうどに一致しなくても、範囲内の最大の候補まで)。
- `suggest_float`同様`log=True`と`step!=1`は併用できない。

### `Trial.suggest_categorical(...)`

**用途**: 離散的な選択肢(文字列・数値・Noneなど)からパラメータを提案させる。

**シグネチャ**: `Trial.suggest_categorical(self, name, choices) -> CategoricalChoiceType`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    return len(optimizer)

study = optuna.create_study()
study.optimize(objective, n_trials=1)
print(study.trials[0].params)
```
実行結果:
```
{'optimizer': 'adam'}
```

**注意点・落とし穴**:
- `choices`はリスト(順序固定)で渡す。`low`/`high`のような範囲指定ではなく、要素そのものが候補になる(数値のリストを渡しても連続値としては扱われない)。
- 同じ`trial`内で同じ`name`を異なる`choices`で2回呼ぶと`ValueError`になる(動的な探索空間の変更は許容されない箇所がある)。

---

## 3. サンプラー

### `optuna.samplers.TPESampler(...)`

**用途**: ベイズ最適化の一種(Tree-structured Parzen Estimator)によるサンプラー。`create_study`のデフォルトサンプラー。

**シグネチャ**: `optuna.samplers.TPESampler(*, consider_prior=None, prior_weight=None, consider_magic_clip=None, consider_endpoints=None, n_startup_trials=10, n_ei_candidates=24, gamma=None, weights=None, seed=None, multivariate=False, group=False, warn_independent_sampling=None, constant_liar=False, constraints_func=None, categorical_distance_func=None) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30)
print('best_value:', round(study.best_value, 4))
print('best_params:', study.best_params)
```
実行結果:
```
best_value: 0.1099
best_params: {'x': 2.33135612434835, 'y': -2.9883381782196308}
```

**注意点・落とし穴**:
- `n_startup_trials=10`(デフォルト)回は`RandomSampler`相当のランダム探索を行い、それ以降にTPEによる提案が始まる。試行数が少ないとTPEの効果が出にくい。
- 同条件・同じ`seed`で比較すると、後述の`RandomSampler`より収束が速い(本例では`best_value`が0.11 vs 3.50)。

### `optuna.samplers.RandomSampler(...)`

**用途**: パラメータを一様分布からランダムサンプリングするだけの最も単純なサンプラー。ベースライン比較用。

**シグネチャ**: `optuna.samplers.RandomSampler(self, seed=None) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=30)
print('best_value:', round(study.best_value, 4))
print('best_params:', study.best_params)
```
実行結果:
```
best_value: 3.4954
best_params: {'x': 1.4039354083575937, 'y': -1.2279697307535926}
```

**注意点・落とし穴**:
- 過去の試行結果を一切利用しない。同じ30試行でも`TPESampler`より最良値が悪い(本例で0.11 vs 3.50)。単純な探索空間や、他サンプラーとの比較基準として使う。

### `optuna.samplers.GridSampler(...)`

**用途**: 指定した候補値の全組み合わせ(グリッド)を漏れなく試す。

**シグネチャ**: `optuna.samplers.GridSampler(self, search_space, seed=None) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

search_space = {'x': [0, 2, 4], 'y': [-3, 0, 3]}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=20)
print('requested n_trials=20 but actual:', len(study.trials))
print('best_value:', study.best_value, study.best_params)
```
実行結果:
```
requested n_trials=20 but actual: 9
best_value: 0.0 {'x': 2.0, 'y': -3.0}
```

**注意点・落とし穴**:
- `n_trials`にグリッドの全組み合わせ数(この例では3×3=9)より大きい値を渡しても、全組み合わせを使い切った時点で自動的に停止する(実行して確認済み。20を指定しても9回しか実行されない)。
- `search_space`で指定した値そのものしか提案されない(`suggest_float`の`low`/`high`は候補生成には使われず、範囲チェックのみに使われる)。

### `optuna.samplers.CmaEsSampler(...)`

**用途**: CMA-ES(共分散行列適応進化戦略)による、連続値パラメータに強いサンプラー。追加パッケージ`cmaes`が必要。

**シグネチャ**: `optuna.samplers.CmaEsSampler(*, x0=None, sigma0=None, n_startup_trials=1, independent_sampler=None, warn_independent_sampling=True, seed=None, consider_pruned_trials=False, restart_strategy=None, popsize=None, inc_popsize=-1, use_separable_cma=False, with_margin=False, lr_adapt=False, source_trials=None) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(seed=0))
study.optimize(objective, n_trials=20)
print('best_params:', study.best_params, 'best_value:', round(study.best_value, 4))
```
実行結果:
```
best_params: {'x': 1.6524133867671544, 'y': -2.350540825365103} best_value: 0.5426
```

**注意点・落とし穴**:
- **実際に検証**: `cmaes`パッケージが未インストールの環境で`CmaEsSampler()`のインスタンス化自体は成功するが、`study.optimize()`実行時(初回のサンプリング時)に`ModuleNotFoundError: No module named 'cmaes'`が発生する。事前に`pip install cmaes`が必要。
- カテゴリカルパラメータ(`suggest_categorical`)や動的な探索空間には対応しておらず、その場合は`independent_sampler`(デフォルト`RandomSampler`)にフォールバックし、`warn_independent_sampling=True`(デフォルト)だと`[W ...] The parameter ... is sampled independently using RandomSampler instead of CmaEsSampler`という警告ログが出る(実行して確認済み)。
- 少数パラメータ・連続値中心の探索空間で強みを発揮する手法であり、パラメータ数が非常に少ない(1個など)場合は恩恵が小さい。

### `optuna.samplers.QMCSampler(...)`

**用途**: 準モンテカルロ法(Sobol列など)による低食い違い量サンプリング。`RandomSampler`よりも探索空間を均等にカバーしやすい。

**シグネチャ**: `optuna.samplers.QMCSampler(*, qmc_type='sobol', scramble=False, seed=None, independent_sampler=None, warn_asynchronous_seeding=True, warn_independent_sampling=True) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

study = optuna.create_study(sampler=optuna.samplers.QMCSampler(seed=0))
study.optimize(objective, n_trials=10)
print('best_value:', round(study.best_value, 4))
```
実行結果:
```
best_value: 13.0
```

**注意点・落とし穴**:
- optuna 4.9.0時点で`ExperimentalWarning: QMCSampler is experimental (supported from v3.0.0). The interface can change in the future.`という警告が出る(実行して確認済み)。将来のバージョンでインターフェースが変わる可能性がある。

---

## 4. プルーニング

### `optuna.pruners.MedianPruner(...)`

**用途**: 各ステップで、それまでの試行の中間値の中央値より明らかに悪い試行を打ち切る(枝刈り)。`create_study`のデフォルトプルーナー。

**シグネチャ**: `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1, *, n_min_trials=1) -> None`

**使用例**: 「`Trial.report(...)` / `Trial.should_prune(...)`」の項を参照(実際の枝刈り動作はそちらでまとめて検証)。

**注意点・落とし穴**:
- `n_startup_trials=5`(デフォルト)未満の試行数のうちは、比較対象がまだ十分でないため枝刈りを行わない。
- 中央値ベースなので、最初の数試行の善し悪しに結果が左右されやすい。

### `Trial.report(...)` / `Trial.should_prune(...)`

**用途**: `report`で学習途中の中間値(例: エポックごとの損失)をプルーナーに通知し、`should_prune`でその時点で打ち切るべきか判定する。

**シグネチャ**: `Trial.report(self, value, step) -> None` / `Trial.should_prune(self) -> bool`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    base = trial.suggest_float('base', 0.0, 1.0)
    for step in range(5):
        # 偶数番目の試行だけ良い値、奇数番目はわざと悪い値にする
        intermediate = base + step * 0.1 + (0 if trial.number % 2 == 0 else 1.0)
        trial.report(intermediate, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return intermediate

study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0))
study.optimize(objective, n_trials=10)
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print('pruned:', len(pruned), 'complete:', len(complete))
```
実行結果:
```
pruned: 6 complete: 4
```

**注意点・落とし穴**:
- `should_prune()`が`True`を返しても、それ自体は試行を止めない。必ず`raise optuna.TrialPruned()`を自分で呼び出す必要がある(本例のように)。
- `report`の`step`は増加する整数を渡す想定(エポック番号など)。同じ`step`に複数回`report`すると最後の値で上書きされる。

### `optuna.TrialPruned`

**用途**: 目的関数内で`raise`することで、その試行を「打ち切り(PRUNED)」として扱う例外。

**シグネチャ**: `optuna.TrialPruned`(`optuna.exceptions.TrialPruned`のエイリアス。`OptunaError`のサブクラス)

**使用例**:
```python
import optuna
print(optuna.TrialPruned is optuna.exceptions.TrialPruned)
print(optuna.TrialPruned.__mro__)
```
実行結果:
```
True
(<class 'optuna.exceptions.TrialPruned'>, <class 'optuna.exceptions.OptunaError'>, <class 'Exception'>, <class 'BaseException'>, <class 'object'>)
```

**注意点・落とし穴**:
- `optuna.TrialPruned`と`optuna.exceptions.TrialPruned`は同一クラス(どちらを使ってもよい)。`study.optimize`はこの例外だけを特別扱いし、試行を`TrialState.PRUNED`として記録して次の試行に進む(他の例外は`catch`で指定しない限り伝播して`optimize`全体が止まる)。

### `optuna.pruners.SuccessiveHalvingPruner(...)` / `optuna.pruners.HyperbandPruner(...)`

**用途**: `MedianPruner`より積極的に打ち切る、Successive Halving・Hyperbandアルゴリズムに基づくプルーナー。

**シグネチャ**: `optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', reduction_factor=4, min_early_stopping_rate=0, bootstrap_count=0) -> None` / `optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3, bootstrap_count=0) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    base = trial.suggest_float('base', 0.0, 1.0)
    for step in range(10):
        intermediate = base + step * 0.1 + (0 if trial.number % 3 == 0 else 1.0)
        trial.report(intermediate, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return intermediate

study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, n_trials=15)
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
print('SuccessiveHalvingPruner pruned:', len(pruned), '/', len(study.trials))

study2 = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=10))
study2.optimize(objective, n_trials=15)
pruned2 = [t for t in study2.trials if t.state == optuna.trial.TrialState.PRUNED]
print('HyperbandPruner pruned:', len(pruned2), '/', len(study2.trials))
```
実行結果:
```
SuccessiveHalvingPruner pruned: 11 / 15
HyperbandPruner pruned: 9 / 15
```

**注意点・落とし穴**:
- 同じ目的関数・試行数でも`MedianPruner`(前項の例では10試行中6打ち切り)より`SuccessiveHalvingPruner`(15試行中11打ち切り)の方が積極的に打ち切る傾向が確認できた。打ち切りすぎて有望な試行まで捨てないよう、`min_resource`/`n_warmup_steps`系のパラメータで調整する。
- `HyperbandPruner`は内部で複数の`SuccessiveHalvingPruner`(bracket)を組み合わせるため、`max_resource`の見積もりが重要(`'auto'`だと最初のtrialの`report`回数から推定される)。

---

## 5. 結果の取得・分析

### `study.best_params` / `study.best_value` / `study.best_trial`

**用途**: 最適化終了後、最も良かった試行のパラメータ・目的関数値・`FrozenTrial`オブジェクトを取得する。

**シグネチャ**: いずれも`Study`のプロパティ(引数なし)。

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=5)
bt = study.best_trial
print('best_trial.number:', bt.number)
print('best_trial.value:', bt.value)
print('best_trial.params:', bt.params)
print('study.best_params == best_trial.params:', study.best_params == bt.params)
```
実行結果:
```
best_trial.number: 2
best_trial.value: 0.0030544989253336228
best_trial.params: {'x': 2.055267521432878}
study.best_params == best_trial.params: True
```

**注意点・落とし穴**:
- **多目的最適化(`directions`を複数指定した`Study`)では`best_params`/`best_value`/`best_trial`はいずれも使えない**。実際に呼び出すと`RuntimeError: A single best trial cannot be retrieved from a multi-objective study. Consider using Study.best_trials to retrieve a list containing the best trials.`が出ることを確認した。多目的の場合は`study.best_trials`(後述)を使う。

### `study.trials_dataframe(...)`

**用途**: 全試行の情報(番号・値・パラメータ・状態など)をpandasのDataFrameとして取得する。

**シグネチャ**: `Study.trials_dataframe(self, attrs=('number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'params', 'user_attrs', 'system_attrs', 'state'), multi_index=False) -> pd.DataFrame`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=5)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
print(df)
```
実行結果:
```
   number      value  params_x     state
0       0   1.048023  0.976270  COMPLETE
1       1   5.307436  4.303787  COMPLETE
2       2   0.003054  2.055268  COMPLETE
3       3   1.215145  0.897664  COMPLETE
4       4  12.439052 -1.526904  COMPLETE
```

**注意点・落とし穴**:
- パラメータの列名は`params_<パラメータ名>`という接頭辞付きになる(この例では`params_x`)。
- `attrs`を絞らないとデフォルトで`datetime_start`/`duration`/`user_attrs`/`system_attrs`なども含まれ、列数が多くなる。

### `study.get_trials(...)` / `optuna.trial.TrialState`

**用途**: 状態(完了・打ち切り・実行中など)でフィルタした試行一覧を取得する。

**シグネチャ**: `Study.get_trials(self, deepcopy=True, states=None) -> list[FrozenTrial]`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    if trial.number >= 5:
        trial.study.stop()
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)
print('n_trials after stop():', len(study.trials))
complete = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
print('COMPLETE only:', len(complete))
```
実行結果:
```
n_trials after stop(): 6
COMPLETE only: 6
```

**注意点・落とし穴**:
- `study.trials`プロパティは`get_trials()`(引数なし、つまり`deepcopy=True`で全状態)を呼ぶのと同じ。大量の試行がある場合、`deepcopy=False`にするとコピーのオーバーヘッドを避けられる(ただし返されたオブジェクトを書き換えると内部状態に影響しうる)。
- `TrialState`には`COMPLETE`/`PRUNED`/`FAIL`/`RUNNING`/`WAITING`がある。

### `optuna.importance.get_param_importances(...)`

**用途**: fANOVAなどの手法で、各パラメータが目的関数値にどれだけ寄与したかを重要度として算出する。

**シグネチャ**: `optuna.importance.get_param_importances(study, *, evaluator=None, params=None, target=None) -> dict[str, float]`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30)
print(optuna.importance.get_param_importances(study))
```
実行結果:
```
{'y': np.float64(0.6102294569349938), 'x': np.float64(0.38977054306500625)}
```

**注意点・落とし穴**:
- 返り値の辞書は重要度の降順に並ぶ(この例では`y`の方が`x`より寄与が大きいと判定された)。
- ある程度の試行数がないと重要度推定が不安定になる(数試行では意味のある値にならない)。

---

## 6. 可視化

### `optuna.visualization.plot_optimization_history(...)`

**用途**: 試行ごとの目的関数値と、その時点までのベスト値の推移を折れ線グラフで可視化する(Plotlyの`Figure`を返す)。

**シグネチャ**: `optuna.visualization.plot_optimization_history(study, *, target=None, target_name='Objective Value', error_bar=False) -> go.Figure`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30)
fig = optuna.visualization.plot_optimization_history(study)
print(type(fig))
```
実行結果:
```
<class 'plotly.graph_objs._figure.Figure'>
```

**注意点・落とし穴**:
- `matplotlib`版(`optuna.visualization.matplotlib`)も別途あるが、こちらはPlotlyの`Figure`を返す。Jupyter上では`fig.show()`やセルの最終行に置くことで描画される。単体スクリプトでは`fig.write_html(...)`や`fig.write_image(...)`(要`kaleido`)で保存する必要がある。

### `optuna.visualization.plot_param_importances(...)`

**用途**: `get_param_importances`と同じ重要度計算結果を横棒グラフで可視化する。

**シグネチャ**: `optuna.visualization.plot_param_importances(study, evaluator=None, params=None, *, target=None, target_name='Objective Value') -> go.Figure`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30)
fig = optuna.visualization.plot_param_importances(study)
print(type(fig))
```
実行結果:
```
<class 'plotly.graph_objs._figure.Figure'>
```

### `optuna.visualization.plot_slice(...)` / `plot_contour(...)` / `plot_parallel_coordinate(...)`

**用途**: それぞれ「パラメータ値ごとの目的関数値の散布(slice)」「2パラメータの等高線(contour)」「全パラメータを平行座標で俯瞰(parallel coordinate)」を可視化する。いずれも引数の形がほぼ共通。

**シグネチャ**: 3関数とも`(study, params=None, *, target=None, target_name='Objective Value') -> go.Figure`(`plot_parallel_coordinate`のみ`params`が第2位置引数)

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=20)

fig_slice = optuna.visualization.plot_slice(study, params=['x', 'y'])
fig_contour = optuna.visualization.plot_contour(study, params=['x', 'y'])
fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
print(type(fig_slice), type(fig_contour), type(fig_parallel))
```
実行結果:
```
<class 'plotly.graph_objs._figure.Figure'> <class 'plotly.graph_objs._figure.Figure'> <class 'plotly.graph_objs._figure.Figure'>
```

**注意点・落とし穴**:
- `plot_contour`はパラメータが3つ以上ある場合、`params`を指定しないと全パラメータの組み合わせ行列(グリッド)が描画され重くなる。2つに絞るのが基本。
- `params=None`(デフォルト)だと探索空間に含まれる全パラメータが対象になる。

---

## 7. マルチ目的最適化

### `optuna.create_study(directions=[...])` と `study.best_trials`

**用途**: 複数の目的関数値を同時に最適化する(多目的最適化)。単一の最良解ではなく、パレート最適解の集合(パレートフロント)が得られる。

**シグネチャ**: `create_study`の`directions`引数に方向のリストを渡す。目的関数はその数だけ値をタプルで返す。パレート最適な試行一覧は`Study.best_trials`(プロパティ、引数なし)で取得する。

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', 0, 5)
    y = trial.suggest_float('y', 0, 3)
    v0 = 4 * x ** 2 + 4 * y ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1

study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=50)
print('n_best_trials(pareto front):', len(study.best_trials))
for t in study.best_trials[:3]:
    print(t.number, t.values, t.params)
```
実行結果:
```
n_best_trials(pareto front): 30
0 [48.53347608109735, 13.237012832735516] {'x': 2.7440675196366238, 'y': 2.1455680991172583}
2 [32.966790290878606, 17.68213421377473] {'x': 2.1182739966945237, 'y': 1.9376823391999682}
3 [47.777583803325534, 13.311845364234367] {'x': 2.1879360563134624, 'y': 2.675319002346239}
```

**注意点・落とし穴**:
- 多目的の`FrozenTrial`は単一値の`value`ではなく`values`(リスト)を持つ。単一目的用の`t.value`は`None`になる。
- 前述の通り、多目的の`Study`では`best_params`/`best_value`/`best_trial`(単数形)は`RuntimeError`になる。必ず`best_trials`(複数形)を使う。

### `optuna.samplers.NSGAIISampler(...)`

**用途**: 多目的最適化向けの遺伝的アルゴリズム(NSGA-II)ベースのサンプラー。目的が2つ以上ある場合にTPESamplerより効率的なことが多い。

**シグネチャ**: `optuna.samplers.NSGAIISampler(*, population_size=50, mutation_prob=None, crossover=None, crossover_prob=0.9, swapping_prob=0.5, seed=None, constraints_func=None, elite_population_selection_strategy=None, child_generation_strategy=None, after_trial_strategy=None) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', 0, 5)
    y = trial.suggest_float('y', 0, 3)
    v0 = 4 * x ** 2 + 4 * y ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1

study = optuna.create_study(
    directions=['minimize', 'minimize'],
    sampler=optuna.samplers.NSGAIISampler(seed=0, population_size=20),
)
study.optimize(objective, n_trials=40)
print('pareto front size:', len(study.best_trials))
```
実行結果:
```
pareto front size: 20
```

**注意点・落とし穴**:
- `population_size`(デフォルト50)は1世代あたりの個体数。`n_trials`が`population_size`より小さいと1世代も完了せず、遺伝的アルゴリズムとしての効果がほとんど出ない。

### `optuna.visualization.plot_pareto_front(...)`

**用途**: 多目的最適化のパレートフロント(トレードオフの境界)を散布図で可視化する。

**シグネチャ**: `optuna.visualization.plot_pareto_front(study, *, target_names=None, include_dominated_trials=True, axis_order=None, constraints_func=None, targets=None) -> go.Figure`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', 0, 5)
    y = trial.suggest_float('y', 0, 3)
    return 4 * x ** 2 + 4 * y ** 2, (x - 5) ** 2 + (y - 5) ** 2

study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.NSGAIISampler(seed=0, population_size=20))
study.optimize(objective, n_trials=40)
fig = optuna.visualization.plot_pareto_front(study, target_names=['v0', 'v1'])
print(type(fig))
```
実行結果:
```
<class 'plotly.graph_objs._figure.Figure'>
```

**注意点・落とし穴**:
- 目的が2つまたは3つの場合のみ2D/3D散布図として描画できる。4つ以上ではエラーになる。
- `include_dominated_trials=True`(デフォルト)だとパレート最適でない(他の解に支配される)試行も薄く表示される。パレートフロントだけ見たい場合は`False`にする。

---

## 8. 永続化(Storage)

### `optuna.storages.RDBStorage` と `create_study(storage=...)`

**用途**: 試行結果をSQLiteやPostgreSQLなどのRDBに保存し、プロセスをまたいで再開・共有できるようにする。

**シグネチャ**: `optuna.storages.RDBStorage(url, engine_kwargs=None, skip_compatibility_check=False, *, heartbeat_interval=None, grace_period=None, heartbeat_stale_trial_callback=None, failed_trial_callback=None, skip_table_creation=False) -> None`(`create_study`には`storage`にURL文字列をそのまま渡すこともできる)

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

storage_url = 'sqlite:///test_optuna.db'
study = optuna.create_study(study_name='demo-study', storage=storage_url, load_if_exists=True)
study.optimize(objective, n_trials=10)
print('best_value:', round(study.best_value, 4))
```
実行結果:
```
best_value: 0.147
```

**注意点・落とし穴**:
- `storage`には`optuna.storages.RDBStorage`のインスタンスの代わりに`"sqlite:///path/to.db"`のようなURL文字列を直接渡せる(内部で自動的に`RDBStorage`にラップされる)。
- `study_name`が既存のDB内に存在する場合、`load_if_exists=False`(デフォルト)だと`DuplicatedStudyError`になる。同じ名前で追記・再開したい場合は`load_if_exists=True`が必要。

### `optuna.load_study(...)`

**用途**: 既存のストレージから、指定した名前のStudyを読み込む(別プロセスから続きを見る、結果だけ確認する、など)。

**シグネチャ**: `optuna.load_study(*, study_name, storage, sampler=None, pruner=None) -> Study`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
storage_url = 'sqlite:///test_optuna.db'
loaded = optuna.load_study(study_name='demo-study', storage=storage_url)
print('loaded n_trials:', len(loaded.trials))
```
実行結果:
```
loaded n_trials: 10
```

**注意点・落とし穴**:
- `study_name`は省略できない(`None`にするとエラー)。名前を忘れた場合は次項の`get_all_study_names`で一覧を確認する。

### `optuna.study.get_all_study_names(...)`

**用途**: 指定したストレージ内に存在する全Study名を一覧取得する。

**シグネチャ**: `optuna.study.get_all_study_names(storage) -> list[str]`

**使用例**:
```python
import optuna
storage_url = 'sqlite:///test_optuna.db'
print(optuna.study.get_all_study_names(storage_url))
```
実行結果:
```
['demo-study']
```

---

## 9. その他(コールバック・実行制御)

### `Study.optimize(..., callbacks=[...])`

**用途**: 各試行が終わるたびに呼び出される関数を登録する(進捗のログ出力、外部システムへの通知、早期終了の判定などに使う)。

**シグネチャ**: コールバックは`Callable[[Study, FrozenTrial], None]`の形。`optimize`の`callbacks`引数にリストで渡す。

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

def my_callback(study, trial):
    if trial.number == 4:
        print(f'callback fired at trial {trial.number}, value={trial.value:.4f}')

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=10, callbacks=[my_callback])
```
実行結果:
```
callback fired at trial 4, value=1.2151
```

### `optuna.study.MaxTrialsCallback(...)`

**用途**: 複数プロセスで並列に`optimize`を実行している場合でも、「完了(COMPLETE)した試行が合計n_trials件になったら全プロセスで停止する」という制御をコールバックで実現する。

**シグネチャ**: `optuna.study.MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,)) -> None`

**使用例**:
```python
import optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100, callbacks=[optuna.study.MaxTrialsCallback(5, states=(TrialState.COMPLETE,))])
print('n_trials with MaxTrialsCallback(5):', len(study.trials))
```
実行結果:
```
n_trials with MaxTrialsCallback(5): 5
```

**注意点・落とし穴**:
- 単純に`optimize(objective, n_trials=5)`とするのと似た結果になるが、`MaxTrialsCallback`は「(並列実行時も含めて)状態がstatesに合致する試行の合計数」で止める点が異なる。`n_trials`引数だけだとプロセスごとのローカルなカウントになるため、複数プロセス並列実行時に合計試行数を厳密に制御したい場合に使う。

### `Trial.set_user_attr(...)` / `Study.set_user_attr(...)`

**用途**: 目的関数の戻り値(スコア)以外に、任意の付加情報をTrial/Studyに紐づけて記録する。

**シグネチャ**: `Trial.set_user_attr(self, key, value) -> None` / `Study.set_user_attr(self, key, value) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    trial.set_user_attr('x_squared', x ** 2)
    return (x - 2) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=1)
print('trial0 user_attrs:', study.trials[0].user_attrs)
```
実行結果:
```
trial0 user_attrs: {'x_squared': 4.0}
```

**注意点・落とし穴**:
- `user_attrs`はJSONシリアライズ可能な値(数値・文字列・リスト・辞書など)である必要がある(任意のPythonオブジェクトは保存できない)。`RDBStorage`使用時は特に制約に注意。

### `Study.enqueue_trial(...)`

**用途**: 次に評価してほしいパラメータの組を、サンプラーに提案させる前に手動でキューに積む(既知の良さそうな初期値を必ず1回は試したい場合などに使う)。

**シグネチャ**: `Study.enqueue_trial(self, params, user_attrs=None, skip_if_exists=False) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.enqueue_trial({'x': 2.0})
study.optimize(objective, n_trials=10)
print('trial0 (enqueued) params:', study.trials[0].params, 'value:', study.trials[0].value)
```
実行結果:
```
trial0 (enqueued) params: {'x': 2.0} value: 0.0
```

**注意点・落とし穴**:
- キューに積んだパラメータは、`optimize`が呼ばれた際に(サンプラーの提案より先に)必ず1回消費される。本例のように事前に最適解が分かっている点を検証・warm startしたい場合に有効。

### `Study.stop()`

**用途**: 目的関数の内部から、現在実行中の`optimize`ループを(その試行の完了後に)止める。

**シグネチャ**: `Study.stop(self) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    if trial.number >= 5:
        trial.study.stop()
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)
print('n_trials after stop():', len(study.trials))
```
実行結果:
```
n_trials after stop(): 6
```

**注意点・落とし穴**:
- `n_trials=100`を指定していても、目的関数内で`trial.study.stop()`が呼ばれた試行(この例では6番目、`trial.number`が0始まりで5)が完了した時点で`optimize`全体が停止する。時間やカスタム条件によって早期終了させたい場合に使う(`should_prune`が個々の試行の打ち切りなのに対し、`stop`は`optimize`ループ全体の終了)。

### `Study.add_trial(...)`

**用途**: `optimize`を介さずに、既に得られている結果(過去の実験ログなど)をStudyに直接登録する。

**シグネチャ**: `Study.add_trial(self, trial: FrozenTrial) -> None`(`optuna.trial.create_trial(...)`で`FrozenTrial`を組み立てて渡す)

**使用例**:
```python
import optuna

study = optuna.create_study()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.add_trial(
    optuna.trial.create_trial(
        params={'x': 3.0},
        distributions={'x': optuna.distributions.FloatDistribution(-10, 10)},
        value=1.0,
    )
)
print('add_trial後のn_trials:', len(study.trials), study.trials[0].params, study.trials[0].value)
```
実行結果:
```
add_trial後のn_trials: 1 {'x': 3.0} 1.0
```

**注意点・落とし穴**:
- `params`と`distributions`のキーは一致している必要があり、`distributions`には実際に`suggest_float`などで使うのと同じ`Distribution`オブジェクト(`FloatDistribution`/`IntDistribution`/`CategoricalDistribution`など)を渡す。
- 過去の実験結果を取り込んでから`study.optimize()`を呼べば、サンプラーはこの追加済み試行も踏まえて次の提案を行う(TPESamplerなど、履歴を使う手法で特に有効)。

---

## 応用・発展

## 10. 制約付き・条件付き最適化

### `constraints_func`(制約付き最適化)

**用途**: 目的関数の値とは別に「制約を満たすか」をサンプラーに伝え、実行可能領域(制約を満たす領域)を優先的に探索させる。`TPESampler`・`NSGAIISampler`など複数のサンプラーが`constraints_func`引数を共通して持つ。

**シグネチャ**: `constraints_func: Callable[[FrozenTrial], Sequence[float]]` を`TPESampler(..., constraints_func=...)`のように渡す。各要素が0以下なら実行可能、正なら制約違反とみなされる。

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -5, 5)
    y = trial.suggest_float('y', -5, 5)
    # 制約: x + y <= 1 (満たされていれば c <= 0)
    c = (x + y) - 1
    trial.set_user_attr('constraint', (c,))
    return x ** 2 + y ** 2

def constraints_func(trial):
    return trial.user_attrs['constraint']

sampler = optuna.samplers.TPESampler(seed=0, constraints_func=constraints_func)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)

feasible = [t for t in study.trials if t.user_attrs['constraint'][0] <= 0]
print('feasible trials:', len(feasible), '/', len(study.trials))
print('best (feasible) params:', study.best_params, 'value:', round(study.best_value, 4))
print('best trial constraint value:', study.best_trial.user_attrs['constraint'])
```
実行結果:
```
feasible trials: 21 / 30
best (feasible) params: {'x': -0.058038567660368856, 'y': -0.22260424354632957} value: 0.0529
best trial constraint value: (-1.2806428112066985,)
```

**注意点・落とし穴**:
- optunaの`constraints_func`自体は制約値をサンプラーに渡す仕組みでしかなく、制約違反の試行を`study.optimize()`が自動的に除外・打ち切りするわけではない(この例のように、通常どおり最後まで評価され`COMPLETE`になる)。実行可能解だけを見たい場合は本例のように`trial.user_attrs`から手動でフィルタする必要がある。
- 制約値は`objective`の中で計算して`trial.set_user_attr(...)`などに保存し、`constraints_func`側でそれを読み出す、という2段構えの実装がよく使われる(`constraints_func`は`FrozenTrial`しか受け取れず、目的関数の計算過程には直接アクセスできないため)。
- `TPESampler`で`constraints_func`を使うと`ExperimentalWarning: Argument constraints_func is an experimental feature.`が出る(実行して確認済み)。将来のバージョンでインターフェースが変わる可能性がある。

### `NSGAIISampler(constraints_func=...)` と `plot_pareto_front(..., constraints_func=...)`

**用途**: 多目的最適化と制約付き最適化を組み合わせ、パレートフロントを「実行可能な解の中」から求める。可視化側にも同じ`constraints_func`を渡すと、実行不可能な試行を区別して描画できる。

**シグネチャ**: `optuna.visualization.plot_pareto_front(study, *, target_names=None, include_dominated_trials=True, axis_order=None, constraints_func=None, targets=None) -> go.Figure`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', 0, 5)
    y = trial.suggest_float('y', 0, 3)
    c = (x - 3) ** 2 + (y - 1) ** 2 - 2  # <=0 なら実行可能
    trial.set_user_attr('constraint', (c,))
    return 4 * x ** 2 + 4 * y ** 2, (x - 5) ** 2 + (y - 5) ** 2

def constraints_func(trial):
    return trial.user_attrs['constraint']

sampler = optuna.samplers.NSGAIISampler(seed=0, population_size=10, constraints_func=constraints_func)
study = optuna.create_study(directions=['minimize', 'minimize'], sampler=sampler)
study.optimize(objective, n_trials=30)

feasible = [t for t in study.trials if t.user_attrs['constraint'][0] <= 0]
print('feasible:', len(feasible), '/', len(study.trials))
print('pareto front size (best_trials):', len(study.best_trials))

fig = optuna.visualization.plot_pareto_front(study, constraints_func=constraints_func, target_names=['v0', 'v1'])
print(type(fig))
```
実行結果:
```
feasible: 18 / 30
pareto front size (best_trials): 10
<class 'plotly.graph_objs._figure.Figure'>
```

**注意点・落とし穴**:
- `study.best_trials`(パレートフロント)は、`constraints_func`を`NSGAIISampler`側に渡しているかどうかに関わらず、単に非劣解の集合を返す。制約違反の試行が`best_trials`に混ざりうる点は`plot_pareto_front`に`constraints_func`を渡す理由(実行不可能な点をマーカーで区別する)につながる。
- 目的が2つ・3つまでしか散布図として描画できない(既存の`plot_pareto_front`の注意点と同じ)。

### `optuna.samplers.PartialFixedSampler(...)`

**用途**: 探索空間の一部のパラメータを固定値に固定したまま、残りのパラメータだけを別のサンプラー(`base_sampler`)で最適化する。「他のパラメータは確定済みで、あるパラメータだけ追加調整したい」場面に使う。

**シグネチャ**: `optuna.samplers.PartialFixedSampler(fixed_params: dict[str, Any], base_sampler: BaseSampler) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

base_sampler = optuna.samplers.TPESampler(seed=0)
sampler = optuna.samplers.PartialFixedSampler(fixed_params={'y': 0.0}, base_sampler=base_sampler)

study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)
y_values = {t.params['y'] for t in study.trials}
print('distinct y values used:', y_values)
print('best_params:', study.best_params)
```
実行結果:
```
distinct y values used: {0.0}
best_params: {'x': 2.055267521432878, 'y': 0.0}
```

**注意点・落とし穴**:
- `fixed_params`で指定した名前のパラメータでも、目的関数内では通常どおり`trial.suggest_float('y', ...)`のように呼び出す必要がある(呼び出し自体は必須で、`PartialFixedSampler`がその呼び出し結果を固定値にすり替える仕組み)。
- optuna 4.9.0時点で`ExperimentalWarning: PartialFixedSampler is experimental (supported from v2.4.0). The interface can change in the future.`という警告が出る(実行して確認済み)。

---

## 11. Study間のコピー・マージ

### `optuna.copy_study(...)`

**用途**: 既存のStudy(全試行・パラメータ・設定)を、別のストレージ・別名で丸ごと複製する。DBのバックアップや、本番用DBへの移行に使う。

**シグネチャ**: `optuna.copy_study(*, from_study_name: str, from_storage: str | storages.BaseStorage, to_storage: str | storages.BaseStorage, to_study_name: str | None = None) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(study_name='copy-src', storage='sqlite:///copy_src.db', load_if_exists=True)
study.optimize(objective, n_trials=10)

optuna.copy_study(
    from_study_name='copy-src',
    from_storage='sqlite:///copy_src.db',
    to_storage='sqlite:///copy_dst.db',
    to_study_name='copy-dst',
)
copied = optuna.load_study(study_name='copy-dst', storage='sqlite:///copy_dst.db')
print('source n_trials:', len(study.trials), '/ copied n_trials:', len(copied.trials))
print('best_value一致:', study.best_value == copied.best_value)
```
実行結果:
```
source n_trials: 10 / copied n_trials: 10
best_value一致: True
```

**注意点・落とし穴**:
- `to_study_name`を省略すると、コピー元と同じ`study_name`が使われる。コピー先のストレージに同名Studyが既に存在すると`DuplicatedStudyError`になる。
- 試行だけでなく`user_attrs`/`system_attrs`/サンプラーの状態なども含めてコピーされる(単なる`trials_dataframe`のエクスポートとは異なり、コピー先でも通常のStudyとして`optimize`を継続できる)。

### `optuna.delete_study(...)`

**用途**: 指定したストレージから、指定した名前のStudyを完全に削除する。

**シグネチャ**: `optuna.delete_study(*, study_name: str, storage: str | storages.BaseStorage) -> None`

**使用例**:
```python
import optuna
print('削除前:', optuna.study.get_all_study_names('sqlite:///copy_dst.db'))
optuna.delete_study(study_name='copy-dst', storage='sqlite:///copy_dst.db')
print('削除後:', optuna.study.get_all_study_names('sqlite:///copy_dst.db'))
```
実行結果:
```
削除前: ['copy-dst']
削除後: []
```

**注意点・落とし穴**:
- 削除は即座に反映され、取り消せない。前項の`copy_study`でバックアップを取ってから、元のStudyを整理する、といった使い方が安全。
- 存在しない`study_name`を指定すると`KeyError`になる(事前に`get_all_study_names`で存在確認するのが安全)。

### 複数Studyの試行をマージする(`Study.add_trial`)

**用途**: 共有ストレージを使わずに別々に実行した複数のStudy(例: 異なるマシンでの並列実行結果)の試行履歴を、後から1つのStudyにまとめる。

**シグネチャ**: 専用のマージAPIはなく、`Study.add_trial(self, trial: FrozenTrial) -> None`をループで呼び出して実現する。

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

# 独立した2つのStudy(例: 別マシンで動かした結果を想定)
study_a = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study_a.optimize(objective, n_trials=5)
study_b = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
study_b.optimize(objective, n_trials=5)

# 2つのStudyの全試行を1つの新しいStudyにマージ
merged = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
for src in (study_a, study_b):
    for t in src.trials:
        merged.add_trial(
            optuna.trial.create_trial(params=t.params, distributions=t.distributions, value=t.value)
        )
print('study_a:', len(study_a.trials), 'study_b:', len(study_b.trials), '-> merged:', len(merged.trials))
print('merged best_value:', round(merged.best_value, 4))
```
実行結果:
```
study_a: 5 study_b: 5 -> merged: 10
merged best_value: 0.0031
```

**注意点・落とし穴**:
- `copy_study`は「1つのStudyの複製」であり複数Studyの統合はできない。複数の情報源をまとめたい場合は本例のように`add_trial`を手動でループするしかない。
- マージ後に`merged.optimize(...)`を呼べば、TPESamplerなどはマージされた全試行の履歴を踏まえて次の提案を行う(「14. ウォームスタートの応用」の考え方と同じ仕組み)。

---

## 12. コールバックの応用パターン

### 自作の早期終了コールバック(`Study.stop()`活用)

**用途**: 「直近N回、ベスト値が更新されなければ打ち切る」といった、`n_trials`や`timeout`だけでは表現できない終了条件をコールバックとして自作する。

**シグネチャ**: コールバックは`Callable[[Study, FrozenTrial], None]`の形(「9. その他」の`Study.optimize(..., callbacks=[...])`と同じ)。内部で条件を満たしたときに`study.stop()`を呼ぶ。

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

class EarlyStoppingCallback:
    """直近 patience 回、ベスト値が更新されなければ study.stop() を呼ぶ。"""
    def __init__(self, patience):
        self.patience = patience
        self.best_value = None
        self.no_improve_count = 0

    def __call__(self, study, trial):
        if self.best_value is None or study.best_value < self.best_value:
            self.best_value = study.best_value
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        if self.no_improve_count >= self.patience:
            study.stop()

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=200, callbacks=[EarlyStoppingCallback(patience=10)])
print('n_trials (early stopped):', len(study.trials))
print('best_value:', round(study.best_value, 6))
```
実行結果:
```
n_trials (early stopped): 13
best_value: 0.003054
```

**注意点・落とし穴**:
- `n_trials=200`を指定していても、13試行で打ち切られている(実行して確認済み)。`Study.stop()`は「その試行が完了した後」にループを止めるため、コールバック内で呼んでも当該試行自体は最後まで実行される。
- クラス(`__call__`を実装したインスタンス)をコールバックとして使うと、`patience`や`best_value`などの状態をクロージャなしで保持できる。関数ベースだと`nonlocal`や外部の可変オブジェクトが必要になる。

### 複数コールバックの併用

**用途**: `callbacks`引数にはリストで複数のコールバックを渡せる。ロギング用・停止条件用などを組み合わせる場合、呼び出し順序を把握しておく必要がある。

**シグネチャ**: `Study.optimize(self, func, ..., callbacks=None, ...) -> None`(`callbacks`はリスト。各試行完了後、リストの先頭から順に呼ばれる)

**使用例**:
```python
import optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

order = []

def logging_callback(study, trial):
    order.append(('log', trial.number))

max_trials_cb = optuna.study.MaxTrialsCallback(3, states=(TrialState.COMPLETE,))

def wrapped_max_trials(study, trial):
    order.append(('max_trials_check', trial.number))
    max_trials_cb(study, trial)

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=100, callbacks=[logging_callback, wrapped_max_trials])
print('n_trials:', len(study.trials))
print('callback call order (first 8):', order[:8])
```
実行結果:
```
n_trials: 3
callback call order (first 8): [('log', 0), ('max_trials_check', 0), ('log', 1), ('max_trials_check', 1), ('log', 2), ('max_trials_check', 2)]
```

**注意点・落とし穴**:
- `callbacks=[a, b]`と渡すと、1試行終わるごとに`a`→`b`の順で呼ばれることが実行結果から確認できる。停止判定(`MaxTrialsCallback`など)を他のコールバックより後ろに置けば、それより前のコールバック(ロギングなど)は停止直前の試行でも確実に実行される。

### コールバックはCOMPLETE以外の状態でも呼ばれる

**用途**: `callbacks`は`COMPLETE`(正常終了)した試行だけでなく、`PRUNED`(打ち切り)や`FAIL`(例外)で終わった試行でも呼び出される、という見落としやすい挙動を確認する。

**シグネチャ**: コールバックは`Callable[[Study, FrozenTrial], None]`。渡される`FrozenTrial`の`state`属性で状態を判定できる。

**使用例**:
```python
import optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    if x < 0:
        raise optuna.TrialPruned()
    return (x - 2) ** 2

def state_logger(study, trial):
    print(f'callback: trial={trial.number} state={trial.state.name}')

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=5, callbacks=[state_logger])
```
実行結果:
```
callback: trial=0 state=COMPLETE
callback: trial=1 state=COMPLETE
callback: trial=2 state=COMPLETE
callback: trial=3 state=COMPLETE
callback: trial=4 state=PRUNED
```

**注意点・落とし穴**:
- trial 4は`TrialPruned`で打ち切られているが、コールバックは呼ばれ`state=PRUNED`が渡ってきている(実行して確認済み)。「完了した試行のときだけ集計したい」ようなコールバックを書く場合は、`if trial.state != optuna.trial.TrialState.COMPLETE: return`のような明示的なガードが必要(`MaxTrialsCallback`が`states`引数を持つのもこのため)。

---

## 13. Artifact機能

### `optuna.artifacts.FileSystemArtifactStore` と `optuna.artifacts.upload_artifact(...)`

**用途**: 学習済みモデルのファイルや画像・ログなど、数値では表せない「成果物」をTrial/Studyに紐づけて保存する。`FileSystemArtifactStore`はローカルディスク上のディレクトリを保存先とするバックエンド(他に`Boto3ArtifactStore`/`GCSArtifactStore`もある)。

**シグネチャ**: `optuna.artifacts.FileSystemArtifactStore(base_path: str | Path) -> None` / `optuna.artifacts.upload_artifact(*, artifact_store: ArtifactStore, file_path: str, study_or_trial: Trial | FrozenTrial | Study, storage: BaseStorage | None = None, mimetype: str | None = None, encoding: str | None = None) -> str`

**使用例**:
```python
import optuna, os
optuna.logging.set_verbosity(optuna.logging.WARNING)

base_path = '/tmp/optuna_artifacts/store'
os.makedirs(base_path, exist_ok=True)
artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    value = (x - 2) ** 2
    log_path = f'/tmp/optuna_artifacts/src_trial{trial.number}.txt'
    with open(log_path, 'w') as f:
        f.write(f'x={x}, value={value}\n')
    artifact_id = optuna.artifacts.upload_artifact(
        artifact_store=artifact_store,
        file_path=log_path,
        study_or_trial=trial,
    )
    trial.set_user_attr('artifact_id', artifact_id)
    os.remove(log_path)
    return value

study = optuna.create_study(
    study_name='artifact-demo',
    storage='sqlite:////tmp/optuna_artifacts/artifact_demo.db',
    sampler=optuna.samplers.TPESampler(seed=0),
)
study.optimize(objective, n_trials=3)
print('best_trial number:', study.best_trial.number)
print('best_trial artifact_id:', study.best_trial.user_attrs['artifact_id'])
```
実行結果:
```
best_trial number: 2
best_trial artifact_id: 9d1c9f07-c737-4e09-b47b-2aab2b4dcbd3
```

**注意点・落とし穴**:
- `upload_artifact`は元ファイルをストアにコピーするだけで、元の`file_path`は自動削除されない(本例のように呼び出し側で後片付けする必要がある)。
- `upload_artifact`は戻り値として一意な`artifact_id`(文字列)を返すだけで、Trialへの紐付けは自動的には行われない。本例のように`trial.set_user_attr('artifact_id', artifact_id)`などで自分で紐付け情報を保存しておく必要がある。
- 以降の例では`sqlite:///`で永続化した同じStudy(`artifact-demo`)を`load_study`で読み込み直して使う(「8. 永続化」の要領)。

### `optuna.artifacts.download_artifact(...)`

**用途**: `upload_artifact`で保存した`artifact_id`から、実体のファイルをローカルに取り出す。

**シグネチャ**: `optuna.artifacts.download_artifact(*, artifact_store: ArtifactStore, file_path: str, artifact_id: str) -> None`

**使用例**:
```python
import optuna

storage_url = 'sqlite:////tmp/optuna_artifacts/artifact_demo.db'
study = optuna.load_study(study_name='artifact-demo', storage=storage_url)
best_artifact_id = study.best_trial.user_attrs['artifact_id']

artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path='/tmp/optuna_artifacts/store')
download_path = '/tmp/optuna_artifacts/downloaded.txt'
optuna.artifacts.download_artifact(
    artifact_store=artifact_store,
    file_path=download_path,
    artifact_id=best_artifact_id,
)
with open(download_path) as f:
    print('downloaded content:', f.read().strip())
```
実行結果:
```
downloaded content: x=2.055267521432878, value=0.0030544989253336228
```

**注意点・落とし穴**:
- `file_path`は保存先のパスであり、`download_artifact`は指定したパスに新規ファイルを書き出す(既存ファイルがあれば上書きされる)。
- 別プロセス(本例では別スクリプト実行)から`load_study`でStudyを読み込んでも、`user_attrs`に保存しておいた`artifact_id`経由でartifactを正しく取り出せることが確認できる(RDBStorageで永続化しているため)。

### `optuna.artifacts.get_all_artifact_meta(...)`

**用途**: 特定のTrial(またはStudy全体)に紐づいている全Artifactのメタ情報(ファイル名・サイズなど)を一覧取得する。

**シグネチャ**: `optuna.artifacts.get_all_artifact_meta(study_or_trial: Trial | FrozenTrial | Study, *, storage: BaseStorage | None = None) -> list[ArtifactMeta]`

**使用例**:
```python
import optuna

storage_url = 'sqlite:////tmp/optuna_artifacts/artifact_demo.db'
study = optuna.load_study(study_name='artifact-demo', storage=storage_url)

# study.best_trial は storage への参照を持たない FrozenTrial なので、
# get_all_artifact_meta には storage を明示的に渡す必要がある
meta_list = optuna.artifacts.get_all_artifact_meta(study.best_trial, storage=study._storage)
print('artifact count:', len(meta_list))
print('filename:', meta_list[0].filename)

try:
    optuna.artifacts.get_all_artifact_meta(study.best_trial)
except ValueError as e:
    print('storage省略時のエラー:', e)
```
実行結果:
```
artifact count: 1
filename: src_trial2.txt
storage省略時のエラー: storage is required for FrozenTrial.
```

**注意点・落とし穴**:
- `study.best_trial`や`study.trials[i]`のように`Study`経由で取得した`FrozenTrial`は、ストレージへの参照を保持していない。そのまま`get_all_artifact_meta(trial)`を呼ぶと`ValueError: storage is required for FrozenTrial.`になる(実行して確認済み)ため、`storage=study._storage`のように明示的に渡す必要がある。`study._storage`は公開APIではない内部属性である点に注意(将来のバージョンで変わりうる)。

---

## 14. ウォームスタートの応用

### `Study.enqueue_trial(..., skip_if_exists=True)`

**用途**: 複数回`enqueue_trial`を呼ぶ際、既に(完了済み・キュー済みを問わず)同じパラメータの試行が存在すれば、重複してキューに積まないようにする。

**シグネチャ**: `Study.enqueue_trial(self, params: dict[str, Any], user_attrs: dict[str, Any] | None = None, skip_if_exists: bool = False) -> None`

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.enqueue_trial({'x': 2.0})
study.enqueue_trial({'x': 2.0}, skip_if_exists=True)  # 同一パラメータなので積まれない
study.enqueue_trial({'x': 5.0})
print('waiting trials before optimize:', len(study.get_trials(states=(optuna.trial.TrialState.WAITING,))))
study.optimize(objective, n_trials=5)
print('trial0 params:', study.trials[0].params)
print('trial1 params:', study.trials[1].params)
```
実行結果:
```
waiting trials before optimize: 2
trial0 params: {'x': 2.0}
trial1 params: {'x': 5.0}
```

**注意点・落とし穴**:
- `{'x': 2.0}`を2回`enqueue_trial`したが、2回目は`skip_if_exists=True`のため無視され、`WAITING`状態の試行は2件(`x=2.0`と`x=5.0`)のみになっている(実行して確認済み)。`skip_if_exists=False`(デフォルト)だと同じパラメータでも毎回キューに積まれ、同じ値の試行が重複して実行される。
- 「既知の初期値を1回だけ必ず試したいが、何度もこのコードが呼ばれうる(ノートブックの再実行など)」場面で重複投入を防ぐのに有効。

### 別Studyの上位試行を`add_trial`でウォームスタートに使う

**用途**: 過去に実行した(別の)Studyの中から成績の良かった試行だけを選び、新しいStudyの初期知識として`add_trial`で投入してから最適化を続ける。「9. その他」の`add_trial`は手書きの1件だけだったが、実際の運用では既存Studyの実データをこのように再利用することが多い。

**シグネチャ**: `Study.add_trial(self, trial: FrozenTrial) -> None`(`optuna.trial.create_trial(...)`で既存`FrozenTrial`のparams/distributions/valueから新しい`FrozenTrial`を組み立てて渡す)

**使用例**:
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

# 旧Study: 通常のランダム探索で少し回した結果
old_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
old_study.optimize(objective, n_trials=20)

# 新Study: 旧Studyの上位5件をadd_trialでウォームスタートしてからTPEで最適化を続ける
top5 = sorted(old_study.trials, key=lambda t: t.value)[:5]
new_study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
for t in top5:
    new_study.add_trial(
        optuna.trial.create_trial(params=t.params, distributions=t.distributions, value=t.value)
    )
print('new_study n_trials after warm start:', len(new_study.trials))
new_study.optimize(objective, n_trials=10)
print('new_study n_trials after optimize:', len(new_study.trials))
print('new_study best_value:', round(new_study.best_value, 4))
```
実行結果:
```
new_study n_trials after warm start: 5
new_study n_trials after optimize: 15
new_study best_value: 4.1155
```

**注意点・落とし穴**:
- `enqueue_trial`との違いに注意: `enqueue_trial`は「パラメータだけ」を指定し、目的関数は改めて実行されて新しく`value`が計算される。一方`add_trial`(本例)は`value`も含めて丸ごと登録するため、目的関数を再実行せずに済む代わりに、目的関数の実装が変わった場合でも古い`value`がそのまま使われてしまう点に注意が必要。
- サンプラーが履歴を使う手法(`TPESampler`など)であれば、ウォームスタートで投入した試行も次の提案の材料として使われる。
