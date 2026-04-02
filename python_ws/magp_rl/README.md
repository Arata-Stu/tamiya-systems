# magp_rl

`f1tenth_gym_jax` を使った強化学習プロジェクトです。

この README は運用でよく使う内容に絞った短縮版です。
基準例は以下で統一しています。

- Agent: `sac`
- LiDAR profile: `T-mini Plus`（320点）
- Vehicle: `tamiya`
- `train.num_envs=128`

## 1. Quick Start

```bash
cd /python_ws
source env/bin/activate
cd magp_rl
python3 -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## 2. 基準コマンド（まずこれ）

```bash
python3 train.py agent=sac train.num_envs=128
```

- `tamiya` と `T-mini Plus(320)` はデフォルト設定を使用
- map を変えない限り、この1行で学習開始できます

## 3. よくある変更（差分だけ足す）

### 3.1 map だけ変える

```bash
python3 train.py agent=sac train.num_envs=128 env.track.name=BrandsHatch
```

### 3.2 vehicle だけ変える

```bash
python3 train.py agent=sac train.num_envs=128 vehicle=traxxas
```

### 3.3 map と vehicle を同時に変える

```bash
python3 train.py agent=sac train.num_envs=128 env.track.name=BrandsHatch vehicle=traxxas
```

### 3.4 単一環境でデバッグ（軽く回す）

```bash
python3 train.py agent=sac train.num_envs=1 train.max_episode_steps=5000
```

## 4. Eval

### 4.1 基本評価

```bash
python3 eval.py \
  agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  eval.num_envs=1 \
  eval.episodes=5
```

### 4.2 動画つき評価

```bash
python3 eval.py \
  agent=sac \
  eval.checkpoint_dir=./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  eval.num_envs=1 \
  eval.save_video=true \
  eval.video_path=./records/eval.mp4
```

## 5. ONNX Export

### 5.1 T-mini Plus (320) モデルをそのまま出力

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

### 5.2 Hokuyo入力(1080)で使う（学習モデルが320の場合）

```bash
python3 export_onnx.py \
  --agent sac \
  --checkpoint-dir ./ckpts/train/YYYY-MM-DD/HH-MM-SS \
  --output ./ckpts/train/YYYY-MM-DD/HH-MM-SS/sac_actor_hokuyo.onnx \
  --lidar-profile hokuyo \
  --obs-dim 320 \
  --scan-points 1080 \
  --input-layout scan \
  --normalize-input \
  --input-name scan_input \
  --output-name control_output
```

`obs_dim=320` のモデルを `scan-points=1080` 入力で受ける場合は、ONNX内で 1080->320 downsample されます。

## 6. TensorRT / Triton Deploy

`deploy_isaac_triton.sh` は ONNX export 前に `obs_dim` の preflight 確認を行います。  
checkpoint から推定した次元と `--obs-dim` がズレる場合、確認または停止します。

### 6.1 T-mini Plus (320)

```bash
bash ./scripts/deploy_isaac_triton.sh \
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

### 6.2 Hokuyo入力(1080)でDeploy（学習モデルが320の場合）

```bash
bash ./scripts/deploy_isaac_triton.sh \
  --model-name magp_rl_policy \
  --agent sac \
  --lidar-profile hokuyo \
  --obs-dim 320 \
  --scan-points 1080 \
  --input-layout scan \
  --normalize-input \
  --input-name scan_input \
  --output-name control_output \
  --precision fp16 \
  --max-batch-size 1 \
  --yes
```

## 7. エラー時の最短チェック

### Gemm shape mismatch (`2176 vs 640` など)

`obs_dim` と学習時モデル次元がズレています。

- 320学習モデルを使うなら: `--obs-dim 320`
- Hokuyo 1080入力を維持するなら: `--scan-points 1080` を併用

## 8. 主な設定ファイル

- Train設定: `config/train.yaml`
- Eval設定: `config/eval.yaml`
- Agent設定: `config/agent/sac.yaml`
- Deployスクリプト: `scripts/deploy_isaac_triton.sh`
