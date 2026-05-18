# Scan Image Classifier

`scan_image_projection_cropper` が publish した固定 `64x64` crop 画像を使って、
`rc_car / duct_tube / background` の3クラス分類モデルを学習・再学習するための環境です。

Triton に deploy する model 名は `scan_image_classifier` で統一しています。

## 1. Dataset 抽出

```bash
cd python_ws/scan_image_classifier
chmod +x 1_extract_dataset.sh

./1_extract_dataset.sh \
  --base_dir /path/to/rosbags \
  --outdir ./datasets \
  --image_topic /perception/crop/image
```

抽出結果は `./datasets/annotations.csv` に追記され、
画像本体は `./datasets/imports/<import_id>/raw/<sequence_id>/images/*.png` に保存されます。
extract を回すたびに
`import_id` 単位で蓄積されます。

```bash
./1_extract_dataset.sh \
  --base_dir /path/to/rosbags \
  --outdir ./datasets \
  --import_name head_to_head_may18
```

`./datasets/imports.csv` に import 一覧が残るので、あとで annotation 対象を import ごとに絞れます。
同じ `dataset_root` に対して何回 extract を回しても、`annotations.csv` と `imports.csv` が積み上がる運用です。

import 一覧を見たい場合:

```bash
python3 list_imports.py --dataset_root ./datasets
```

これで `import_id` ごとの件数、ラベル済み件数、reviewed 件数、train/val/test の内訳が見られます。

## 1.5 External Image Import

rosbag crop 以外の画像も同じ pipeline に取り込めます。
例えば他カメラ画像や、手元に保存した web 画像を class 別ディレクトリに置いて import できます。

```text
/tmp/external_images/
  car/
    img_001.jpg
    img_002.png
  duct_tube/
    img_101.jpg
  other/
    img_201.png
```

`car -> rc_car`、`other -> background` の alias は自動で吸収します。

```bash
python3 import_external_images.py \
  --dataset_root ./datasets \
  --source_dir /tmp/external_images \
  --import_name web_and_other_cameras
```

単一ラベルとして全部入れたい場合:

```bash
python3 import_external_images.py \
  --dataset_root ./datasets \
  --source_dir /tmp/rc_car_only \
  --label rc_car \
  --import_name rc_car_web_only
```

この import でも `annotations.csv` と `imports.csv` に統合されるので、
rosbag 由来データと外部画像を同じ annotation / split / train 流れで扱えます。

## 2. Annotation

```bash
python3 2_annotate_dataset.py --dataset_root ./datasets
```

特定 import だけ見たい場合:

```bash
python3 2_annotate_dataset.py \
  --dataset_root ./datasets \
  --import_ids 20260518_103000_head_to_head_may18
```

キーバインド:

- `1`: `rc_car`
- `2`: `duct_tube`
- `3`: `background`
- `a`: 前の画像
- `d`: 次の画像
- `u`: ラベル解除
- `j`: 次の未ラベルへジャンプ
- `s`: 保存
- `q`: 保存して終了

`annotations.csv` には `auto_label` と `auto_confidence` の列も持たせてあるので、
将来の自動 annotation 追加先として使えます。

## 3. Split

```bash
python3 3_assign_splits.py \
  --dataset_root ./datasets \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

既定では `bag_path` 単位で split を割り当て、すでに split が付いている既存データは保持します。
つまり、新しい import を追加したあとに再実行しても、過去データの split を壊さずに新規分だけ蓄積できます。

## 4. Version Export

annotation と split が終わったら、学習用の versioned dataset を切り出せます。
export 先は `./datasets/versions/<version_name>/train|val|test/<label>/*.png` です。

```bash
python3 export_dataset_version.py \
  --dataset_root ./datasets \
  --version_name ver1
```

特定 import だけで version を作る場合:

```bash
python3 export_dataset_version.py \
  --dataset_root ./datasets \
  --version_name ver2 \
  --import_ids 20260518_103000_head_to_head_may18 20260518_140000_duct_course
```

生成物:

- `./datasets/versions/<version_name>/manifest.csv`
- `./datasets/versions/<version_name>/metadata.csv`
- `./datasets/versions/<version_name>/train/rc_car/*.png`
- `./datasets/versions/<version_name>/train/duct_tube/*.png`
- `./datasets/versions/<version_name>/train/background/*.png`
- `./datasets/versions/<version_name>/val/...`
- `./datasets/versions/<version_name>/test/...`

この export は学習条件のスナップショット用途です。元データの正本は引き続き
`annotations.csv` と `imports/<import_id>/raw/...` 側に残します。

## 5. Train

```bash
python3 4_train.py data.dataset_root=./datasets
```

既定の backbone は `mobilenet_v3_small` です。比較用に
`model.architecture=shufflenet_v2_x0_5` や `tiny_cnn` も選べます。
`data.import_ids=[]` のままなら全 import をまとめて学習します。
特定 import だけ使う場合は `data.import_ids=[import_id_1,import_id_2]` を指定します。

学習時の前処理は以下です。

- 入力画像サイズがバラバラでも、必ず `64x64` に resize して学習
- `force_grayscale_3ch=true` が既定なので、color 画像も `3ch grayscale` に変換
- augment として `flip / rotate / translate+scale / brightness-contrast / blur / noise / cutout` を使用

つまり、duct tube のオレンジ色に依存しすぎず、mono 系カメラや gray 運用に寄せた学習ができます。

## 6. ONNX Export

```bash
python3 5_export_onnx.py \
  --checkpoint ./ckpts/train/YYYY-MM-DD/HH-MM-SS/best_model.pth \
  --input_normalization external
```

入力 tensor 名は `input_tensor`、出力 tensor 名は `output_logits` です。

## 7. Deploy

```bash
chmod +x 6_deploy_model.sh
./6_deploy_model.sh
```

既定では `/workspaces/isaac_ros_assets/models/scan_image_classifier/` に deploy します。

## 8. SCP Checkpoints

```bash
chmod +x scp_ckpts.sh
./scp_ckpts.sh
```

既定の転送先は
`/home/tamiya/workspaces/tamiya-systems/python_ws/ckpts/scan_image_classifier/`
です。
