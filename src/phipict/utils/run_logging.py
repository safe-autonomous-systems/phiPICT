# Original work Copyright 2025 Aleksandra Franz, Nils Thuerey
# Modified work Copyright 2026 Jannis Becktepe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications:
# Loggers are created below the phipict namespace.
# Moved into the phipict package, formatted, linted and typed.

"""Setup of run directories and file/console logging for simulation runs."""

import datetime
import logging
import os
import socket
import sys
from typing import TextIO

from phipict.utils.profiling import DEFAULT_PROFILER

_LOG_INITIALZED = False
_LOG_PATH = ""
_LOG_ROOT: logging.Logger | None = None
_LOG: logging.Logger | None = None

_STDERR = sys.stderr


class StreamCapture:
    """Text stream that forwards writes to another stream and a log file.

    Parameters
    ----------
    file : str or os.PathLike
        Log file, opened in append mode.
    stream : TextIO, optional
        Stream to forward to. Default is ``sys.stdout``.
    """

    def __init__(
        self, file: str | os.PathLike[str], stream: TextIO = sys.stdout
    ) -> None:
        self.stream = stream
        self.log = open(file, "a")

    def write(self, msg: str) -> None:
        """Write a message to the stream and the log file.

        Parameters
        ----------
        msg : str
            Message to write.
        """
        self.stream.write(msg)
        self.log.write(msg)

    def flush(self) -> None:
        """Flush the stream and the log file."""
        self.stream.flush()
        self.log.flush()

    def close(self) -> TextIO:
        """Close the log file.

        Returns
        -------
        TextIO
            The wrapped stream, which is left open.
        """
        self.log.close()
        return self.stream

    def __del__(self) -> None:
        self.flush()
        self.close()


def setup_logging(log_path: str, console: bool = True, debug: bool = False) -> None:
    """Configure the root logger to write to files in ``log_path``.

    Creates ``logfile.log`` (INFO), ``error.log`` (WARNING) and, in debug mode,
    ``debug.log``. Additionally, stderr is captured to ``stderr.log``.

    Parameters
    ----------
    log_path : str
        Log directory, created if it does not exist.
    console : bool, optional
        Whether to also log to stdout. Default is True.
    debug : bool, optional
        Whether to enable DEBUG level logging. Default is False.
    """
    global _LOG_INITIALZED
    global _LOG_PATH
    global _LOG_ROOT
    global _LOG
    os.makedirs(log_path, exist_ok=True)

    # StreamCapture implements the parts of TextIO that are used in practice.
    sys.stderr = StreamCapture(  # type: ignore[assignment]
        os.path.join(log_path, "stderr.log"), _STDERR
    )

    # setup logging
    log_format = "[%(asctime)s][%(name)s:%(levelname)s] %(message)s"
    log_formatter = logging.Formatter(log_format)
    # logging.basicConfig(level=logging.INFO,
    # format=log_format,
    # #datefmt='%Y.%m.%d-%H:%M:%S'
    # filename=os.path.join(image_path, 'logfile.log'))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(log_path, "logfile.log"))
    logfile.setLevel(logging.INFO)
    logfile.setFormatter(log_formatter)
    root_logger.addHandler(logfile)
    errlog = logging.FileHandler(os.path.join(log_path, "error.log"))
    errlog.setLevel(logging.WARNING)
    errlog.setFormatter(log_formatter)
    root_logger.addHandler(errlog)
    if debug:
        debuglog = logging.FileHandler(os.path.join(log_path, "debug.log"))
        debuglog.setLevel(logging.DEBUG)
        debuglog.setFormatter(log_formatter)
        root_logger.addHandler(debuglog)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("[%(name)s:%(levelname)s] %(message)s")
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
    log = logging.getLogger("log setup")
    log.setLevel(logging.DEBUG)

    logging.captureWarnings(True)

    if debug:
        root_logger.setLevel(logging.DEBUG)
        log.info("Debug output active")

    _LOG_PATH = log_path
    _LOG_INITIALZED = True
    _LOG_ROOT = root_logger
    _LOG = log

    log.info("--- Log Start ---")
    log.info("host: %s, pid: %d", socket.gethostname(), os.getpid())
    log.info("Python: %s", sys.version)
    log.info("Log directory: %s", log_path)
    # log.info('TensorFlow version: %s', tf.__version__)


def get_now_string() -> str:
    """Get the current local time as a string.

    Returns
    -------
    str
        Current time formatted as ``%y%m%d-%H%M%S``.
    """
    now = datetime.datetime.now()
    now_str = now.strftime("%y%m%d-%H%M%S")
    return now_str


def setup_run(
    base_dir: str,
    name: str = "TEST",
    logging: bool = True,
    console: bool = True,
    debug: bool = False,
) -> str:
    """Create a time-stamped run directory and optionally set up logging.

    Parameters
    ----------
    base_dir : str
        Directory in which the run directory is created.
    name : str, optional
        Suffix of the run directory name. Default is ``"TEST"``.
    logging : bool, optional
        Whether to call :func:`setup_logging` for ``<run_dir>/log``.
        Default is True.
    console : bool, optional
        See :func:`setup_logging`. Default is True.
    debug : bool, optional
        See :func:`setup_logging`. Default is False.

    Returns
    -------
    str
        Path of the created run directory.
    """
    now_str = get_now_string()

    run_dir = os.path.join(base_dir, now_str + "_" + name)
    os.makedirs(run_dir)

    if logging:
        setup_logging(os.path.join(run_dir, "log"), console, debug)

    return run_dir


def get_logger(name: str) -> logging.Logger:
    """Get a logger in the ``phipict`` namespace.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        The logger.
    """
    # Keep every logger under the "phipict" namespace so that
    # phipict.logging.set_verbosity() controls the whole package at once.
    from phipict.logging import get_logger as _get_namespaced_logger

    return _get_namespaced_logger(name)


def close_logging() -> None:
    """Write profiling stats to the log directory and shut down logging.

    Does nothing if :func:`setup_logging` was not called.
    """
    if _LOG_INITIALZED:
        assert _LOG is not None
        with open(os.path.join(_LOG_PATH, "profiling.txt"), "w") as f:
            DEFAULT_PROFILER.stats(f)
        _LOG.info("DONE")
        logging.shutdown()
