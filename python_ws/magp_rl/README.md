# magp_rl

`f1tenth_gym_jax` ベースの強化学習プロジェクトです。  
この README は運用で迷わないことを目的に、学習モード切替と実行手順を中心にまとめています。

## 1. 前提（デフォルト）

- Agent: `sac`
- Vehicle: `tamiya`
- LiDAR: `T-mini Plus`（320点）
- 既定の並列学習: `train.num_envs=128`

注: Hokuyoへ切り替えたい場合は、学習/評価設定または ONNX/Deploy 時の LiDAR プロファイルを `hokuyo` に変更できます（詳細は各セクションの注釈参照）。

## 2. Quick Start

```bash
cd /python_ws
source env/bin/activate
cd magp_rl
python3 -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## 3. 重要パラメータ（まずここだけ把握）

- `env.parallel.mode`
- `independent`: 1環境=1台。GPUで多数環境を高速並列。
- `race`: 1ワールド内で複数台を同時シミュレート。
- `env.race.control_mode`（`race`時のみ有効）
- `selfplay`: 複数台すべてがポリシーで走行（共有1ポリシー）。
- `npc`: `ego` 1台のみ学習、他車は `PurePursuit`。
- `train.num_envs`
- `independent`: 環境数そのもの。
- `race + selfplay`: 学習対象台数（通常は `env.race.num_agents` と同値）。
- `race + npc`: `1` を使用（学習対象は ego のみ）。
- `env.race.num_agents`
- `race` ワールド内の総台数。`npc` モードでは `2` 以上必須。
- `env.race.ego_idx`
- `npc` モードで学習対象にする車両インデックス。
- `env.race.npc.speed_scale`
- `npc` モード時、ego以外のNPC速度倍率。追い越しシーンを作るなら `0.8 ~ 0.95` 推奨。

## 4. 学習モード切替（必須）

### 4.1 self policy - npcなし（単独走行）

```bash
python3 train.py \
  agent=sac \
  env.parallel.mode=independent \
  train.num_envs=1
```

- 学習対象: 1台
- 他車: なし
- 用途: まず単独でポリシーを安定化

### 4.2 self policy - npcあり（ego + NPC）

```bash
python3 train.py \
  agent=sac \
  env.parallel.mode=race \
  env.race.control_mode=npc \
  env.race.num_agents=8 \
  env.race.ego_idx=0 \
  env.race.npc.speed_scale=0.9 \
  train.num_envs=1
```

- 学習対象: ego 1台のみ
- 他車: `PurePursuit` で走行
- 速度調整: `env.race.npc.speed_scale` を NPC（ego以外）に適用
- リセット: 誰かが done で全体リセット
- 用途: 追従・回避・オーバーテイクの基礎学習

### 4.3 multi policy - 複数台（head-to-head）

```bash
python3 train.py \
  agent=sac \
  env.parallel.mode=race \
  env.race.control_mode=selfplay \
  env.race.num_agents=8 \
  train.num_envs=8
```

- 学習対象: 複数台（全台）
- 収集データ: 全台分
- 現在の実装: 共有1ポリシーを複数台に適用（車両ごと別ポリシーではない）
- 用途: 接触回避や head-to-head 特有の戦略学習

### 4.4 環境自体をたくさん（高速並列）

```bash
python3 train.py \
  agent=sac \
  env.parallel.mode=independent \
  train.num_envs=128
```

- 学習対象: 128環境ぶんの1台ずつ
- 他車: なし
- 用途: サンプル効率・スループット重視の大規模学習

## 5. よく使う学習差分

### 5.1 map変更

```bash
python3 train.py agent=sac train.num_envs=128 env.track.name=BrandsHatch
```

### 5.2 vehicle変更

```bash
python3 train.py agent=sac train.num_envs=128 vehicle=traxxas
```

### 5.3 map + vehicle同時変更

```bash
python3 train.py agent=sac train.num_envs=128 env.track.name=BrandsHatch vehicle=traxxas
```

### 5.4 事前学習重みを自動コピーして転移学習（元ckpt保護）

```bash
python3 train.py \
  agent=sac \
  train.checkpoint.resume=true \
  train.checkpoint.auto_fork_on_resume=true \
  train.checkpoint.dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  env.track.name=TARGET_MAP
```

- `resume=true` で既存重みを読み込み
- `auto_fork_on_resume=true` で指定 `ckpt_dir` を自動複製
- 学習は複製先で継続するため、元runは上書きされません

### 5.5 局所 trajectory + Pure Pursuit 追従

policy の出力を直接 `[steer, speed]` にせず、6次元の Bezier 制御点として扱い、env 内の Pure Pursuit で `[steer, speed]` に変換します。

```bash
python3 train.py \
  agent=sac \
  env.parallel.mode=independent \
  train.num_envs=128 \
  env.control_interface=local_trajectory_pp \
  env.action_dim=6 \
  env.reward.tal.enabled=false \
  env.trajectory.reward.centerline_coef=0.5
```

- action: `[p1_x, p1_y, p2_x, p2_y, p3_x, p3_y]` の正規化値
- trajectory: 原点 `base_link` から前方へ伸びる cubic Bezier
- control: 生成 trajectory を Pure Pursuit で追従
- 既存の `direct` mode は `env.control_interface=direct env.action_dim=2`

`env.trajectory.reward.smoothness_coef` と `env.trajectory.reward.lateral_coef` を上げると、曲線の暴れを報酬側でも抑制できます。
`env.trajectory.reward.centerline_coef` を上げると、ego frame に切り出した centerline と policy trajectory の誤差を罰します。最初は `0.2 ~ 1.0` くらいから試すのが扱いやすいです。
後半の経路がうねる場合は、後方点を強く見る設定を併用します。

```bash
python3 train.py \
  agent=sac \
  env.parallel.mode=independent \
  train.num_envs=128 \
  env.control_interface=local_trajectory_pp \
  env.action_dim=6 \
  env.reward.tal.enabled=false \
  env.trajectory.reward.centerline_coef=0.8 \
  env.trajectory.reward.centerline_tail_power=2.0 \
  env.trajectory.reward.tail_smoothness_coef=0.1 \
  env.trajectory.reward.terminal_lateral_coef=0.5 \
  env.trajectory.reward.terminal_heading_coef=0.2
```

## 6. 評価（Eval）

### 6.1 基本評価（単独）

```bash
python3 eval.py \
  agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  eval.num_envs=1 \
  eval.episodes=5
```

### 6.2 npcモード評価（ego + NPC）

```bash
python3 eval.py \
  agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  env.parallel.mode=race \
  env.race.control_mode=npc \
  env.race.num_agents=8 \
  env.race.npc.speed_scale=0.9 \
  eval.num_envs=1 \
  eval.episodes=5
```

### 6.3 selfplay評価（複数台）

```bash
python3 eval.py \
  agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  env.parallel.mode=race \
  env.race.control_mode=selfplay \
  env.race.num_agents=8 \
  eval.num_envs=8 \
  eval.episodes=5
```

### 6.4 動画つき評価

```bash
python3 eval.py \
  agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  eval.num_envs=1 \
  eval.episodes=3 \
  eval.video.enabled=true \
  eval.video.output_dir=./records/eval
```

## 7. ONNX Export（T-mini Plus 前提）

```bash
python3 export_onnx.py \
  --agent sac \
  --checkpoint-dir ./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  --output ./ckpts/train/YYYY-MM-DD/HH-MM-SS/sac_actor.onnx \
  --lidar-profile t_mini_plus \
  --input-layout scan \
  --normalize-input \
  --input-name scan_input \
  --output-name control_output
```

注: Hokuyo運用時は `--lidar-profile hokuyo` に切替可能です。

## 8. TensorRT / Triton Deploy

`deploy_isaac_triton.sh` は ONNX export 前に `obs_dim` の preflight 確認を行います。  
チェックポイントと指定次元が矛盾する場合は警告/停止します。

```bash
bash ./scripts/deploy_isaac_triton.sh \
  --checkpoint-dir ./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  --model-name magp_rl_policy \
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

checkpoint を読める環境では、deploy script が actor の action dimension を見て `direct` / `trajectory` を自動判定します。推定に失敗した場合は interactive に選択できます。

局所 trajectory policy を ROS2 の Pure Pursuit へつなぐ場合は、明示的に指定することもできます。

```bash
bash ./scripts/deploy_isaac_triton.sh \
  --policy-mode trajectory \
  --checkpoint-dir ./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  --lidar-profile t_mini_plus \
  --input-layout scan \
  --normalize-input \
  --yes
```

`--trajectory-policy` は `--policy-mode trajectory` の短縮形です。この preset は `model_name=magp_rl_trajectory`, `action_dim=6`, `output_name=trajectory_action` を使います。ROS2 側は `magp_rl_lidar_trajectory.launch.xml` が Triton 出力を `nav_msgs/Path` に戻し、`pure_pursuit_controller` へ接続します。

注: Hokuyoへ切替える場合は `--lidar-profile hokuyo` に変更します。
`--precision fp16|fp32` は TensorRT build の計算精度に反映されます。
I/O 互換性のため、生成される `config.pbtxt` の `input/output data_type` は
常に `TYPE_FP32` です。

## 9. トラブルシュート（最短）

### Gemm shape mismatch（`2176 vs 640` など）

- 原因: 学習時 `obs_dim` と export/deploy 時の次元が不一致
- 対応:
- 320学習モデルを使うなら `--obs-dim 320`
- 1080入力を使うなら `--scan-points 1080` を併用

## 10. 主な設定ファイル

- Train: `config/train.yaml`
- Eval: `config/eval.yaml`
- Agent: `config/agent/sac.yaml`
- Deploy script: `scripts/deploy_isaac_triton.sh`
