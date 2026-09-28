# Copyright 2026 Jannis Becktepe
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

"""Logging setup for PhiPICT.

Every logger in the package lives below the ``phipict`` namespace, so that a
single call to :func:`set_verbosity` controls the log level of the whole
library. As a library, PhiPICT is quiet by default: the namespace is set to
``WARNING`` on import and a ``NullHandler`` is attached.
"""

import logging

LOGGER_NAMESPACE = "phipict"


def get_logger(name: str) -> logging.Logger:
    """Get a logger below the PhiPICT namespace.

    Parameters
    ----------
    name: str
        Logger name. Prefixed with ``phipict.`` unless it already is.

    Returns
    -------
    logging.Logger
        The namespaced logger.
    """
    if name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}."):
        return logging.getLogger(name)

    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")


def set_verbosity(level: int | str = logging.INFO) -> None:
    """Set the log level of every PhiPICT logger.

    PhiPICT is quiet by default (``WARNING``): as a library it does not emit
    progress output unless asked to. Call this once, e.g. at the top of a
    runscript, to turn on the solver, environment and AMG INFO logs.

    Parameters
    ----------
    level: int | str
        A :mod:`logging` level, either numeric or by name, e.g. ``"INFO"``.
    """
    logging.getLogger(LOGGER_NAMESPACE).setLevel(level)


# Library default: quiet, and never complain about a missing handler
logging.getLogger(LOGGER_NAMESPACE).addHandler(logging.NullHandler())
set_verbosity(logging.WARNING)
