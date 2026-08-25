import argparse
from datetime import UTC, datetime

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate fish segmentation performance")
    parser.add_argument(
        "--config",
        type=str,
        default="fish_segmentation_model/config/eval_args.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="fish_segmentation_model/baseline_model/yolo26l-seg.pt",
        help="Path to a trained segmentation model",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml with your custom segmentation dataset",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate a YOLO26 instance segmentation model."""
    args = parse_args()

    with open(args.config) as config_file:
        cfg = yaml.safe_load(config_file)

    model = YOLO(args.model_path)
    cfg["data"] = args.data
    cfg["name"] = datetime.now(UTC).strftime("date_%Y%m%d_%H%M%S")

    model.val(**cfg)


if __name__ == "__main__":
    main()
