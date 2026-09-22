"""Package logging that cooperates with tqdm progress bars."""

import logging
import sys

from tqdm.auto import tqdm


class _TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def get_logger(name="my_ddpm"):
    """Return a package logger; configure its shared console handler once.

    Use get_logger().setLevel(logging.DEBUG) to change the package log level.
    """
    if name != "my_ddpm" and not name.startswith("my_ddpm."):
        raise ValueError("Logger names must belong to the my_ddpm package.")
    package_logger = logging.getLogger("my_ddpm")
    if not package_logger.handlers:
        handler = _TqdmHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.INFO)
        package_logger.propagate = False
    return logging.getLogger(name)
