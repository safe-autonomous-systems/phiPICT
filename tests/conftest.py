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

"""Shared test helpers."""

import importlib.util
import pathlib
from types import ModuleType

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "equivalence: compares phipict against the PICT solver bundled in fluidgym",
    )


def import_fluidgym_on_phipict(module: str) -> ModuleType:
    """Import a fluidgym module, skipping unless fluidgym is built on phipict.

    A fluidgym that still uses its own bundled PICT extension registers separate
    pybind types, so its domains cannot be passed to phipict (and loading both
    extensions in one process fails). The check therefore inspects the installed
    package without importing it.
    """
    spec = importlib.util.find_spec("fluidgym")
    if spec is None or spec.submodule_search_locations is None:
        pytest.skip("fluidgym is not installed")
    for location in spec.submodule_search_locations:
        init = pathlib.Path(location) / "simulation" / "__init__.py"
        if init.is_file() and "phipict" not in init.read_text():
            pytest.skip("installed fluidgym uses its own bundled PICT extension")
    return importlib.import_module(module)
