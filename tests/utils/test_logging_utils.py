import logging

from coint2.utils.logging_utils import get_logger


def test_get_logger(capsys):
    """Logger should emit formatted messages to stdout."""
    logger = get_logger("test")
    logger.info("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert "test" in logger.name
    assert logger.level == logging.INFO

