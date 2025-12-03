from pathlib import Path

from causal_playground.core.logging_config import setup_logging


def test_setup_logging_creates_file(tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    logger = setup_logging(str(log_file))
    logger.info("test message")
    assert log_file.exists()
    assert logger.name == "causal_playground"
