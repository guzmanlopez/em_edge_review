from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import yaml

from fish_segmentation_model import evaluate, predict, train


def write_config(tmp_path, config: dict) -> str:
    config_path = tmp_path / "args.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


def test_train_routes_segmentation_config_to_ultralytics(monkeypatch, tmp_path) -> None:
    config_path = write_config(
        tmp_path, {"model": "yolo26l-seg.pt", "epochs": 1, "project": "runs/segment/test"}
    )
    model = Mock()
    yolo = Mock(return_value=model)
    monkeypatch.setattr(train, "YOLO", yolo)
    monkeypatch.setattr(train, "load_dotenv", Mock())
    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: Namespace(config=config_path, data="fish.yaml", mlflow=False),
    )

    train.main()

    yolo.assert_called_once_with("yolo26l-seg.pt")
    model.train.assert_called_once()
    assert model.train.call_args.kwargs["data"] == "fish.yaml"
    assert model.train.call_args.kwargs["epochs"] == 1
    expected_project = Path(train.__file__).resolve().parents[1] / "runs/segment/test"
    assert model.train.call_args.kwargs["project"] == str(expected_project)


def test_train_handles_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    config_path = write_config(tmp_path, {"model": "yolo26l-seg.pt"})
    logger = Mock()
    model = Mock()
    model.train.side_effect = KeyboardInterrupt
    monkeypatch.setattr(train, "YOLO", Mock(return_value=model))
    monkeypatch.setattr(train, "get_logger", Mock(return_value=logger))
    monkeypatch.setattr(train, "load_dotenv", Mock())
    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: Namespace(config=config_path, data="fish.yaml", mlflow=False),
    )

    train.main()

    logger.info.assert_any_call("[stop] Segmentation training interrupted by user")


def test_evaluate_routes_data_to_ultralytics(monkeypatch, tmp_path) -> None:
    config_path = write_config(tmp_path, {"split": "test"})
    model = Mock()
    yolo = Mock(return_value=model)
    monkeypatch.setattr(evaluate, "YOLO", yolo)
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: Namespace(config=config_path, data="fish.yaml", model_path="best.pt"),
    )

    evaluate.main()

    yolo.assert_called_once_with(str(Path(evaluate.__file__).resolve().parents[1] / "best.pt"))
    model.val.assert_called_once()
    assert model.val.call_args.kwargs["data"] == str(
        Path(evaluate.__file__).resolve().parents[1] / "fish.yaml"
    )
    assert model.val.call_args.kwargs["split"] == "test"


def test_evaluate_logs_standard_mask_metrics(monkeypatch, tmp_path) -> None:
    config_path = write_config(tmp_path, {})
    logger = Mock()
    model = Mock()
    model.val.return_value.results_dict = {
        "metrics/precision(M)": 0.8,
        "metrics/recall(M)": 0.7,
        "metrics/mAP50(M)": 0.6,
        "metrics/mAP50-95(M)": 0.5,
    }
    model.val.return_value.names = {0: "fish"}
    model.val.return_value.class_result.return_value = [0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5]
    model.val.return_value.save_dir = tmp_path / "evaluate"
    model.val.return_value.save_dir.mkdir()
    monkeypatch.setattr(evaluate, "YOLO", Mock(return_value=model))
    monkeypatch.setattr(evaluate, "get_logger", Mock(return_value=logger))
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: Namespace(config=config_path, data="fish.yaml", model_path="best.pt"),
    )

    evaluate.main()

    for metric_name, value in model.val.return_value.results_dict.items():
        logger.info.assert_any_call("[metrics] %s: %.4f", metric_name, value)


def test_evaluate_saves_test_results_to_run_directory(monkeypatch, tmp_path) -> None:
    config_path = write_config(tmp_path, {"split": "test"})
    run_dir = tmp_path / "run"
    model_path = run_dir / "weights" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    model = Mock()
    model.val.return_value.results_dict = {"metrics/mAP50(M)": 0.6, "fitness": 0.5}
    model.val.return_value.save_dir = run_dir / "evaluate"
    model.val.return_value.save_dir.mkdir()
    model.val.return_value.names = {0: "fish"}
    model.val.return_value.class_result.return_value = [0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5]
    monkeypatch.setattr(evaluate, "YOLO", Mock(return_value=model))
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: Namespace(config=config_path, data="fish.yaml", model_path=str(model_path)),
    )

    evaluate.main()

    assert model.val.call_args.kwargs["name"] == "run"
    assert (run_dir / "evaluate" / "results_test_summary.csv").read_text() == (
        "metric,value\nmetrics/mAP50(M),0.6\nfitness,0.5\n"
    )
    assert (run_dir / "evaluate" / "results_test.csv").read_text() == (
        "class,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M)\n"
        "fish,0.8,0.7,0.6,0.5\n"
    )


def test_evaluate_saves_empty_metrics_for_unrepresented_class(monkeypatch, tmp_path) -> None:
    config_path = write_config(tmp_path, {})
    run_dir = tmp_path / "run"
    model_path = run_dir / "weights" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model = Mock()
    model.val.return_value.results_dict = {"fitness": 0.5}
    model.val.return_value.save_dir = run_dir / "evaluate"
    model.val.return_value.save_dir.mkdir()
    model.val.return_value.names = {0: "fish", 1: "shark"}
    model.val.return_value.class_result.side_effect = [
        [0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5],
        IndexError,
    ]
    monkeypatch.setattr(evaluate, "YOLO", Mock(return_value=model))
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: Namespace(config=config_path, data="fish.yaml", model_path=str(model_path)),
    )

    evaluate.main()

    assert "shark,,,," in (run_dir / "evaluate" / "results_test.csv").read_text()


def test_predict_routes_source_to_ultralytics(monkeypatch, tmp_path) -> None:
    config_path = write_config(tmp_path, {"conf": 0.5})
    model = Mock()
    yolo = Mock(return_value=model)
    monkeypatch.setattr(predict, "YOLO", yolo)
    monkeypatch.setattr(
        predict,
        "parse_args",
        lambda: Namespace(config=config_path, source="fish.mp4", model_path="best.pt"),
    )

    predict.main()

    yolo.assert_called_once_with("best.pt")
    model.predict.assert_called_once()
    assert model.predict.call_args.kwargs["source"] == "fish.mp4"
    assert model.predict.call_args.kwargs["conf"] == 0.5
