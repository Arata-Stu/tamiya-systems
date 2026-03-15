# magp_rl

`f1tenth_gym_jax` を使った強化学習プロジェクトです。  
現在は `PPO` / `SAC` を `Hydra` で切り替えて学習できます。  
`env.parallel.mode=independent`（デフォルト）で、独立環境を `jax.vmap` でGPU上に並列実行する構成です。
`env.reward.tal.enabled=true` で、Pure Pursuit教師との行動差分を使うTAL風報酬も利用できます。

## 1. セットアップ

```bash
cd /python_ws
sudo bash setup_python_env.sh
```

## 2. 構成

- 学習設定: `config/train.yaml`
- 評価設定: `config/eval.yaml`
- アルゴリズム設定:
  - `config/agent/ppo.yaml`
  - `config/agent/sac.yaml`

主要スクリプト:
- 学習: `train.py`
- 評価: `eval.py`

## 3. 学習

### PPO

```bash
python3 train.py agent=ppo
```

### SAC

```bash
python3 train.py agent=sac
```

例

```bash
python3 train.py agent=sac \
  env.track.name=BrandsHatch \
  train.total_timesteps=1000000 \
  train.num_envs=1 \
  train.max_episode_steps=5000 \
  env.reward.collision_penalty=20.0 \
  env.reward.speed_coef=0.005 \
  agent.actor_lr=1e-4 \
  agent.critic_lr=1e-4 \
  agent.alpha_lr=1e-4 \
  agent.batch_size=1000 \
  agent.start_steps=10000 \
  agent.update_after=10000 \
  agent.updates_per_step=1 \
  agent.print_every_steps=1000 \
  agent.tb_log_every_steps=1000 \
  agent.checkpoint_every_steps=5000
```

TAL報酬を有効化する例:

```bash
python3 train.py agent=sac \
  env.reward.tal.enabled=true \
  env.reward.tal.coef=0.1 \
  env.reward.tal.lookahead_distance=0.8
```

## 4. 評価

`eval.py` は `config/eval.yaml` を使います。

### PPO評価

```bash
python3 eval.py agent=ppo eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS
```

### SAC評価

```bash
python3 eval.py agent=sac eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS
```

評価出力:
- Average Return
- Average Length
- Average Progress (m)
- Average Progress (%)
- Average Speed (m/s)
- Completion Rate
- Collision Rate

動画出力（任意）:

```bash
python3 eval.py agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  eval.video.enabled=true \
  eval.video.output_dir=./records/eval
```

## 5. トラック指定

基本は `env.track` で指定します。

```yaml
env:
  track:
    root: '../f1tenth_racetracks'
    name: 'Austin'
    line_type: 'centerline' # centerline | raceline
```

内部で以下を自動解決します:
- `*_map.yaml`
- `*_{centerline|raceline}.csv`

必要なら `env.map_path` / `env.waypoints_path` を明示指定可能です。

## 6. ログ・保存先

- TensorBoard: `train.tensorboard.log_dir`
- チェックポイント: `train.checkpoint.dir`

デフォルト:
- `./logs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}`
- `./ckpts/train/${now:%Y-%m-%d}/${now:%H-%M-%S}`

## 7. 注意点

- `env.parallel.mode=independent`（推奨）:
  - `train.num_envs` が独立環境数です（1環境1車両）。
  - エージェント同士の衝突やLiDAR干渉は起きません。
- `env.parallel.mode=race`（従来挙動）:
  - `train.num_envs` は同一シミュレータ内の車両数として扱われます。
  - 車両同士の衝突やLiDAR干渉が発生します。
- `train.num_agents` / `eval.num_agents` は旧設定の互換用です。新規設定では `num_envs` を使ってください。
- SACは `start_steps` / `update_after` でwarmupを十分取るのが重要です。
- `train.quiet_absl=true` でOrbaxの冗長ログを抑制できます。
- 学習中のTensorBoardに `episode/progress_m` と `episode/progress_pct` を記録します。

## 8. よく使うコマンド

```bash
# GPU backend確認
python3 -c "import jax; print(jax.default_backend()); print(jax.devices())"

# TensorBoard
tensorboard --logdir ./logs
```
