"""Run fish detection and segmentation model workflows."""

import argparse
import sys
from importlib import import_module

from rich.console import Console
from rich.panel import Panel

WORKFLOWS = {
    ("detection", "train"): "fish_detector_model.train",
    ("detection", "evaluate"): "fish_detector_model.evaluate",
    ("segmentation", "train"): "fish_segmentation_model.train",
    ("segmentation", "evaluate"): "fish_segmentation_model.evaluate",
    ("segmentation", "predict"): "fish_segmentation_model.predict",
}


def parse_args() -> argparse.Namespace:
    """Parse the workflow and pass remaining options to its entry point."""
    parser = argparse.ArgumentParser(description="Run fish model workflows")
    parser.add_argument("workflow", choices=("detection", "segmentation"))
    parser.add_argument("command", choices=("train", "evaluate", "predict"))
    parser.add_argument("workflow_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    """Dispatch a selected fish model workflow."""
    args = parse_args()
    module_name = WORKFLOWS.get((args.workflow, args.command))
    if module_name is None:
        raise SystemExit(f"{args.workflow} does not support {args.command}")

    console = Console()
    console.print(
        Panel.fit(
            f"[bold cyan]Fish {args.workflow.title()}[/bold cyan]\n"
            f"[green]▶[/green] {args.command.title()}",
            title="[bold]EM Edge Review[/bold]",
        )
    )
    console.log(f"[cyan]Launching[/cyan] {module_name}")

    previous_argv = sys.argv
    try:
        sys.argv = [module_name, *args.workflow_args]
        import_module(module_name).main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
