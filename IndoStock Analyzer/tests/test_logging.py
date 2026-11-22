import json
import logging

from indostock_analyzer.logging_utils import JsonLogFormatter


def test_structured_log_formatter_emits_machine_readable_context():
    record = logging.LogRecord(
        name="indostock.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="source_retry_scheduled",
        args=(),
        exc_info=None,
    )
    record.event = "source_retry_scheduled"
    record.context = {"attempt": 1, "delay_seconds": 2.0}

    payload = json.loads(JsonLogFormatter().format(record))

    assert payload["level"] == "WARNING"
    assert payload["event"] == "source_retry_scheduled"
    assert payload["context"] == {"attempt": 1, "delay_seconds": 2.0}
