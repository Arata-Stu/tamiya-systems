# magp_rl

`f1tenth_gym_jax` を使った強化学習プロジェクトです。

- アルゴリズム: `PPO` / `SAC` / `TD3`
- 設定管理: `Hydra`
- 並列実行: `env.parallel.mode=independent`（デフォルト）
- TAL風報酬: `env.reward.tal.enabled=true`
- 車両パラメータ切替: `vehicle=tamiya` / `vehicle=traxxas`

## 1. Quick Start

```bash
cd /python_ws
source env/bin/activate
cd magp_rl

# backend確認
python3 -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## 2. 構成

- 学習設定: `config/train.yaml`
- 評価設定: `config/eval.yaml`
- エージェント設定:
  - `config/agent/ppo.yaml`
  - `config/agent/sac.yaml`
  - `config/agent/td3.yaml`
- 実行スクリプト:
  - 学習: `train.py`
  - 評価: `eval.py`

## 3. Train コマンド

### 3.1 最小実行

```bash
python3 train.py agent=ppo
python3 train.py agent=sac
python3 train.py agent=td3
```

### 3.1.1 YDLiDAR crop (320点, angle ±2.343rad)

`config/train.yaml` のデフォルトは以下です。

- `env.obs_dim=320`
- `env.scan_beams=320`
- `env.scan_angle_min=-2.343`
- `env.scan_angle_max=2.343`（`fov=4.686`）

必要ならCLIで上書きできます。

```bash
python3 train.py agent=sac \
  env.obs_dim=320 \
  env.scan_beams=320 \
  env.scan_angle_min=-2.343 \
  env.scan_angle_max=2.343
```

### 3.2 SAC 本番例（単一環境）

```bash
python3 train.py agent=sac \
  env.parallel.mode=independent \
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

### 3.2.1 TD3 本番例（単一環境）

```bash
python3 train.py agent=td3 \
  env.parallel.mode=independent \
  env.track.name=BrandsHatch \
  train.total_timesteps=1000000 \
  train.num_envs=1 \
  train.max_episode_steps=5000 \
  env.reward.collision_penalty=20.0 \
  env.reward.speed_coef=0.005 \
  agent.actor_lr=1e-4 \
  agent.critic_lr=1e-4 \
  agent.batch_size=1000 \
  agent.start_steps=10000 \
  agent.update_after=10000 \
  agent.updates_per_step=1 \
  agent.policy_delay=2 \
  agent.target_policy_noise=0.2 \
  agent.target_noise_clip=0.5 \
  agent.print_every_steps=1000 \
  agent.tb_log_every_steps=1000 \
  agent.checkpoint_every_steps=5000
```

### 3.3 TAL報酬を有効化

```bash
python3 train.py agent=sac \
  env.parallel.mode=independent \
  env.track.name=BrandsHatch \
  env.reward.tal.enabled=true \
  env.reward.tal.coef=0.1 \
  env.reward.tal.lookahead_distance=0.8
```

### 3.3.1 TAL係数を徐々に弱める（スケジューラ）

```bash
python3 train.py agent=sac \
  env.parallel.mode=independent \
  env.track.name=BrandsHatch \
  env.reward.tal.enabled=true \
  env.reward.tal.coef=0.1 \
  env.reward.tal.schedule.enabled=true \
  env.reward.tal.schedule.mode=linear \
  env.reward.tal.schedule.start_step=2000000 \
  env.reward.tal.schedule.decay_steps=4000000 \
  env.reward.tal.schedule.coef_min=0.0
```

利用可能な `mode`:

- `linear`: 線形減衰
- `cosine`: コサイン減衰

### 3.4 Multi-Env 学習（推奨）

```bash
python3 train.py agent=sac \
  env.parallel.mode=independent \
  train.num_envs=128 \
  env.reward.tal.enabled=true \
  env.track.name=BrandsHatch \
  train.total_timesteps=10000000 \
  train.max_episode_steps=10000 \
  agent.start_steps=25000 \
  agent.update_after=25000 \
  agent.updates_per_step=1
```

### 3.5 車両パラメータを切り替えて学習

Hydraの階層で切替（推奨）:

```bash
python3 train.py agent=sac \
  vehicle=tamiya
```

ファイルパスを明示:

```bash
python3 train.py agent=sac \
  vehicle.path=./config/vehicle/traxxas.yaml
```

一部だけ上書き:

```bash
python3 train.py agent=sac \
  vehicle=tamiya \
  vehicle.v_max=15.0 \
  vehicle.a_max=8.0
```

## 4. Eval コマンド

`eval.py` は `config/eval.yaml` を使います。

### 4.1 基本評価

```bash
python3 eval.py agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS

python3 eval.py agent=td3 \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS
```

### 4.2 動画つき評価

```bash
python3 eval.py agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  eval.video.enabled=true \
  eval.video.output_dir=./records/eval \
  eval.video.filename_prefix=brandshatch_rollout \
  eval.video.sync_to_sim_time=true \
  eval.video.color_by_speed=true \
  eval.video.speed_min_mps=0.0 \
  eval.video.speed_max_mps=5.0
```

`sync_to_sim_time=true` の場合、`sim_dt`（デフォルト: simulatorの `time_step` = 0.01）を使って
動画フレームを間引き、再生時間がシミュレータ時間に近くなるように出力します。

`color_by_speed=true` の場合、軌跡は速度に応じた色（低速→高速）で描画されます。

### 4.3 比較評価向け（TAL報酬OFF）

```bash
python3 eval.py agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  env.parallel.mode=independent \
  eval.num_envs=1 \
  env.reward.tal.enabled=false
```

車両を合わせて評価（学習時と同じ指定にする）:

```bash
python3 eval.py agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  vehicle=tamiya
```

評価サマリ出力:

- `Average Return`
- `Average Length`
- `Average Progress (m)`
- `Average Progress (%)`
- `Average Speed (m/s)`
- `Completion Rate`
- `Collision Rate`

## 5. トラック指定

通常は `env.track` を指定します。

```yaml
env:
  track:
    root: '../f1tenth_racetracks'
    name: 'Austin'
    line_type: 'centerline' # centerline | raceline
```

内部で以下を自動解決します。

- `*_map.yaml`
- `*_{centerline|raceline}.csv`

必要なら明示指定も可能です。

- `env.map_path`
- `env.waypoints_path`

## 6. 車両パラメータ切替

`train.py` / `eval.py` の起動時に、使用された車両パラメータソースが表示されます。

- `Vehicle Params: default`
- `Vehicle Params: hydra.vehicle`
- `Vehicle Params: /abs/path/to/tamiya.yaml`
- `Vehicle Params: /abs/path/to/tamiya.yaml + inline.params`

優先順位:

1. `vehicle=<preset>`（Hydraグループ、デフォルトは `traxxas`）
2. `vehicle.path`（明示yaml）
3. `vehicle.<param>=...` のinline上書き

## 7. Parallel モードの違い

### 7.1 independent（デフォルト）

- `train.num_envs = 独立環境数`
- 1環境1車両
- 車両同士の LiDAR 干渉 / 衝突判定なし

### 7.2 race

- `train.num_envs = 同一環境内の車両数`
- 車両同士の LiDAR 干渉 / 衝突判定あり

## 8. Multi-Env スケーリングの目安（SAC）

`num_envs` を増やすと1ループで集まるサンプルが増えるため、`updates_per_step` も比例調整すると安定しやすいです。

- 目安式:
  - `new_updates_per_step = round(old_updates_per_step * new_envs / old_envs)`
  - `new_start_steps = old_start_steps * new_envs / old_envs`
  - `new_update_after = old_update_after * new_envs / old_envs`

倍々例（128基準）:

- `256`: `updates_per_step=2`, `start_steps=50000`, `update_after=50000`
- `512`: `updates_per_step=4`, `start_steps=100000`, `update_after=100000`
- `1024`: `updates_per_step=8`, `start_steps=200000`, `update_after=200000`

## 9. ログと保存先

- TensorBoard: `train.tensorboard.log_dir`
- Checkpoint: `train.checkpoint.dir`
- Hydra出力: `hydra.run.dir`
- Eval動画: `eval.video.output_dir`

デフォルト:

- `./logs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}`
- `./ckpts/train/${now:%Y-%m-%d}/${now:%H-%M-%S}`
- `./outputs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}`
- `./records/eval/${now:%Y-%m-%d}/${now:%H-%M-%S}`

最新チェックポイントrunを探す例:

```bash
ls -td ./ckpts/train/*/* | head
```

## 10. よく使うコマンド

```bash
# TensorBoard
tensorboard --logdir ./logs

# SAC学習を静かに実行（absl抑制）
python3 train.py agent=sac train.quiet_absl=true
```

## 11. 補足

- `train.total_timesteps` は「全環境合計」の遷移数です。
- `train.num_agents` / `eval.num_agents` は互換用。新規設定は `num_envs` を使ってください。
- `ep` は「完了したエピソード数の累積」です（multi-envでは同時完了分まとめて増えます）。

## 12. ONNX エクスポート

学習済みcheckpointから actor をONNXへ変換できます。

```bash
# SAC deterministic policy (tanh(mean)) を出力
python3 export_onnx.py \
  --agent sac \
  --checkpoint-dir ./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  --output ./ckpts/train/YYYY-MM-DD/HH-MM-SS/sac_actor.onnx \
  --obs-dim 1080 \
  --action-dim 2
```

```bash
# PPO policy を出力
python3 export_onnx.py \
  --agent ppo \
  --checkpoint-dir ./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  --output ./ckpts/train/YYYY-MM-DD/HH-MM-SS/ppo_actor.onnx \
  --obs-dim 1080 \
  --action-dim 2
```

SACは `--sac-output` で出力形式を選べます。

- `deterministic`（デフォルト）: `action`
- `mean_logstd`: `mean`, `log_std`
- `all`: `action`, `mean`, `log_std`

Isaac ROS向けに `scan_input` 形式（`[batch, 1, points]`）と入力正規化をONNX内で持たせる例:

```bash
python3 export_onnx.py \
  --agent sac \
  --checkpoint-dir ./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  --output ./ckpts/train/YYYY-MM-DD/HH-MM-SS/sac_actor_isaac.onnx \
  --lidar-profile hokuyo \
  --action-dim 2 \
  --input-layout scan \
  --normalize-input \
  --input-name scan_input \
  --output-name control_output
```

`obs_dim` とセンサ点数の運用:

- Hokuyo想定: `obs_dim=1080`, `scan_points=1080`, `fov_rad=4.7`, `max_range=30.0`
- T-mini / YDLiDAR crop想定: `obs_dim=320`, `scan_points=320`, `fov_rad=4.686`, `max_range=10.0`
- 学習モデルが `obs_dim=320` の場合のみ、`scan_points=1080` からONNX内でダウンサンプル可能（1080→320）

## 13. TensorRT / Triton デプロイ

`ONNX export -> TensorRT(.plan) -> Triton model repo 配置` を一括で実行するスクリプト:

- [deploy_isaac_triton.sh](/Users/at/project/competition/MAGP/tamiya-systems/python_ws/magp_rl/scripts/deploy_isaac_triton.sh)

デフォルトでは:

1. `./ckpts/train` から run を選択（`--yes` で最新を自動選択）
2. 最新 `checkpoint_*` から ONNX を export
3. `/workspaces/isaac_ros_assets/models/<model_name>/<version>/` に `model.onnx`, `model.plan` を配置
4. `/workspaces/isaac_ros_assets/models/<model_name>/config.pbtxt` を生成

### 13.1 Hokuyo (1080)

```bash
bash ./scripts/deploy_isaac_triton.sh \
  --model-name magp_sac_hokuyo \
  --agent sac \
  --lidar-profile hokuyo \
  --input-layout scan \
  --normalize-input \
  --input-name scan_input \
  --output-name control_output \
  --precision fp16 \
  --max-batch-size 1 \
  --yes
```

### 13.2 T-mini (320)

```bash
bash ./scripts/deploy_isaac_triton.sh \
  --model-name magp_sac_tmini \
  --agent sac \
  --lidar-profile t_mini_plus \
  --input-layout scan \
  --normalize-input \
  --input-name scan_input \
  --output-name control_output \
  --precision fp16 \
  --max-batch-size 1 \
  --yes
```

### 13.3 既存ONNXを直接デプロイ

```bash
bash ./scripts/deploy_isaac_triton.sh \
  --onnx /path/to/model.onnx \
  --model-name magp_sac_prod \
  --input-layout scan \
  --scan-points 1080 \
  --input-name scan_input \
  --output-name control_output
```
