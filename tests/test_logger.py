from logger import get_logger


def test_get_logger_returns_the_same_logger_for_a_path() -> None:
    assert get_logger(__file__) is get_logger(__file__)
