import logging
import os
from typing import Any, Callable
# from .main import main as search # pylint: disable=cyclic-import
from . import __version__
__version__ = __version__.get_versions()['version'] # type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO,
                    style="{",
                    format="{asctime} :: {levelname} :: {name} :: [{run}] -- {message}",
                    handlers=[logging.FileHandler("file.log", mode='a'),
                              logging.StreamHandler()])

old_factory = logging.getLogRecordFactory()

def logger_record_factory(run: int) -> Callable[..., logging.LogRecord]:
    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        record.run = run # type: ignore
        return record
    return record_factory
