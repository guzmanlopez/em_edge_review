import argparse
from collections.abc import Mapping
from datetime import UTC, datetime

import yaml
from ultralytics import YOLO

from logger import get_logger

SEGMENTATION_METRIC_KEYS = (
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)


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
    logger = get_logger(__name__)
    args = parse_args()

    with open(args.config) as config_file:
        cfg = yaml.safe_load(config_file)

    logger.info("[fish] Loading segmentation model: %s", args.model_path)
    model = YOLO(args.model_path)
    cfg["data"] = args.data
    cfg["name"] = datetime.now(UTC).strftime("date_%Y%m%d_%H%M%S")

    logger.info("[chart] Evaluating segmentation model with dataset: %s", args.data)
    results = model.val(**cfg)
    metrics = getattr(results, "results_dict", {})
    if isinstance(metrics, Mapping):
        for metric_name in SEGMENTATION_METRIC_KEYS:
            if metric_name in metrics:
                logger.info("[metrics] %s: %.4f", metric_name, metrics[metric_name])
    logger.info("[check] Segmentation evaluation completed")


if __name__ == "__main__":
    main()
