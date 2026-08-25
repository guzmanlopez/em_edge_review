from argparse import Namespace
from unittest.mock import Mock

import yaml

from fish_segmentation_model import evaluate, predict, train


def write_config(tmp_path, config: dict) -> str:
    config_path = tmp_path / "args.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


def test_train_routes_segmentation_config_to_ultralytics(monkeypatch, tmp_path) -> None:
    config_path = write_config(tmp_path, {"model": "yolo26l-seg.pt", "epochs": 1})
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

    yolo.assert_called_once_with("best.pt")
    model.val.assert_called_once()
    assert model.val.call_args.kwargs["data"] == "fish.yaml"
    assert model.val.call_args.kwargs["split"] == "test"


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
