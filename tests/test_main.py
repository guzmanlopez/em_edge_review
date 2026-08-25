import sys
from argparse import Namespace
from types import SimpleNamespace

import main


def test_main_dispatches_segmentation_prediction(monkeypatch) -> None:
    received_argv = []

    def workflow_main() -> None:
        received_argv.extend(sys.argv)

    monkeypatch.setattr(
        main,
        "parse_args",
        lambda: Namespace(
            workflow="segmentation",
            command="predict",
            workflow_args=["--source", "fish.mp4"],
        ),
    )
    monkeypatch.setattr(main, "import_module", lambda _: SimpleNamespace(main=workflow_main))

    main.main()

    assert received_argv == ["fish_segmentation_model.predict", "--source", "fish.mp4"]
