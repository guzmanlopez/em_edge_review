import argparse
import os
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import yaml
from ultralytics import YOLO

from logger import get_logger

SEGMENTATION_METRIC_KEYS = (
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str) -> Path:
    """Resolve a path relative to the repository when it is not absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def model_run_name(model_path: Path) -> str:
    """Get the run directory name containing the model checkpoint."""
    return (
        model_path.parent.parent.name
        if model_path.parent.name == "weights"
        else model_path.parent.name
    )


def save_results_csv(metrics: Mapping, output_dir: Path) -> Path:
    """Save aggregate evaluation metrics in the evaluation run directory."""
    results_path = output_dir / "results_test_summary.csv"
    pl.DataFrame({"metric": list(metrics), "value": list(metrics.values())}).write_csv(results_path)
    return results_path


def save_class_results_csv(results, output_dir: Path) -> Path:
    """Save per-class segmentation metrics in the evaluation run directory."""
    class_results = []
    for class_index, class_name in results.names.items():
        class_metrics = {metric_name: None for metric_name in SEGMENTATION_METRIC_KEYS}
        try:
            class_metrics.update(
                zip(SEGMENTATION_METRIC_KEYS, results.class_result(class_index)[4:])
            )
        except IndexError:
            pass
        class_results.append({"class": class_name, **class_metrics})
    results_path = output_dir / "results_test.csv"
    pl.DataFrame(class_results).write_csv(results_path)
    return results_path


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

    with resolve_repo_path(args.config).open() as config_file:
        cfg = yaml.safe_load(config_file)
    if "project" in cfg and not os.path.isabs(cfg["project"]):
        cfg["project"] = str(REPO_ROOT / cfg["project"])

    model_path = resolve_repo_path(args.model_path)
    data_path = resolve_repo_path(args.data)
    logger.info("[fish] Loading segmentation model: %s", model_path)
    model = YOLO(str(model_path))
    cfg["data"] = str(data_path)
    cfg["name"] = model_run_name(model_path)

    logger.info("[chart] Evaluating segmentation model with dataset: %s", data_path)
    results = model.val(**cfg)
    metrics = getattr(results, "results_dict", {})
    if isinstance(metrics, Mapping):
        output_dir = Path(results.save_dir)
        class_results_path = save_class_results_csv(results, output_dir)
        summary_path = save_results_csv(metrics, output_dir)
        logger.info("[file] Class evaluation results saved to %s", class_results_path)
        logger.info("[file] Evaluation summary saved to %s", summary_path)
        for metric_name in SEGMENTATION_METRIC_KEYS:
            if metric_name in metrics:
                logger.info("[metrics] %s: %.4f", metric_name, metrics[metric_name])
    logger.info("[check] Segmentation evaluation completed")


if __name__ == "__main__":
    main()
