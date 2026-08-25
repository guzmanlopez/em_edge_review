# Fish segmentation model

This directory trains, evaluates, and runs inference with a **YOLO26 instance segmentation model**. Unlike the detector in `fish_detector_model/`, which predicts bounding boxes, this model predicts a separate pixel mask for each fish.

The default `yolo26l-seg.pt` checkpoint follows the [Ultralytics YOLO26 segmentation model family](https://docs.ultralytics.com/models/yolo26). It is downloaded automatically by Ultralytics the first time it is used.

## Dataset

Use an Ultralytics segmentation dataset. Each image has a matching label text file whose rows contain a class ID followed by normalized polygon coordinates:

```text
class_id x1 y1 x2 y2 ... xn yn
```

The dataset YAML has the same structure as a detection dataset:

```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

names:
  0: tuna
  1: shark
```

Bounding-box-only labels cannot train a segmentation model. Export polygon or mask annotations from your annotation tool in Ultralytics YOLO segmentation format.

## Train

```bash
uv run fish_segmentation_model/train.py --data /path/to/data.yaml
```

Training uses `config/train_args.yaml`. Override it with `--config`, and optionally enable MLflow with `--mlflow` after setting `MLFLOW_TRACKING_URI`.

## Evaluate

```bash
uv run fish_segmentation_model/evaluate.py \
  --data /path/to/data.yaml \
  --model-path /path/to/best.pt
```

Evaluation uses the test split and reports both box and mask metrics.

## Predict

```bash
uv run fish_segmentation_model/predict.py \
  --source /path/to/image-or-video \
  --model-path /path/to/best.pt
```

Predictions and polygon labels are written under `Segmentation_prediction_results/`. Settings for each operation are in `fish_segmentation_model/config/`.