import argparse
from datetime import UTC, datetime

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict fish instance masks")
    parser.add_argument(
        "--config",
        type=str,
        default="fish_segmentation_model/config/predict_args.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="fish_segmentation_model/baseline_model/yolo26l-seg.pt",
        help="Path to a trained segmentation model",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, video, directory, URL, or stream to process",
    )
    return parser.parse_args()


def main() -> None:
    """Run prediction with a YOLO26 instance segmentation model."""
    args = parse_args()

    with open(args.config) as config_file:
        cfg = yaml.safe_load(config_file)

    model = YOLO(args.model_path)
    cfg["source"] = args.source
    cfg["name"] = datetime.now(UTC).strftime("date_%Y%m%d_%H%M%S")

    model.predict(**cfg)


if __name__ == "__main__":
    main()
