from logger import get_logger


def test_get_logger_returns_the_same_logger_for_a_path() -> None:
    assert get_logger(__file__) is get_logger(__file__)


def test_get_logger_adds_only_one_handler() -> None:
    logger = get_logger("unique_logger.py")

    get_logger("unique_logger.py")

    assert len(logger.handlers) == 1
