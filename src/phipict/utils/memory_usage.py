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
# Moved into the phipict package, formatted, linted and typed.

"""Tracking of peak CPU and GPU memory usage."""

import logging
import os

import psutil
import torch


class MemoryUsage:
    """Record the peak CPU and GPU memory usage over a series of checkpoints.

    Parameters
    ----------
    logger : logging.Logger or None, optional
        Logger used for reporting. If None, nothing is logged. Default is None.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.max_mem_GPU = 0
        self.max_mem_GPU_name = ""
        self.max_mem_CPU = 0
        self.max_mem_CPU_name = ""
        self.process = psutil.Process(os.getpid())
        self.LOG = logger

    def fmt_mem(self, value_bytes: float) -> str:
        """Format a memory size in MiB.

        Parameters
        ----------
        value_bytes : float
            Memory size in bytes.

        Returns
        -------
        str
            Formatted size, e.g. ``"12.34MiB"``.
        """
        return "%.02fMiB" % (value_bytes / (1024 * 1024),)

    def check_memory(self, name: str = "", verbose: bool = True) -> None:
        """Sample the current memory usage and update the recorded peaks.

        Parameters
        ----------
        name : str, optional
            Label of this checkpoint. Default is ``""``.
        verbose : bool, optional
            Whether to log the current usage. Default is True.
        """
        used_mem_GPU = torch.cuda.memory_allocated()
        if used_mem_GPU > self.max_mem_GPU:
            self.max_mem_GPU = used_mem_GPU
            self.max_mem_GPU_name = name
        used_mem_CPU = self.process.memory_info()[0]
        if used_mem_CPU > self.max_mem_CPU:
            self.max_mem_CPU = used_mem_CPU
            self.max_mem_CPU_name = name
        if verbose and (self.LOG is not None):
            self.LOG.info(
                "used memory '%s': CPU %s, GPU %s",
                name,
                self.fmt_mem(used_mem_CPU),
                self.fmt_mem(used_mem_GPU),
            )

    def print_max_memory(self) -> None:
        """Log the recorded peak CPU and GPU memory usage."""
        if self.LOG is not None:
            self.LOG.info(
                "Max used memory: CPU %s at %s, GPU %s at '%s'",
                self.fmt_mem(self.max_mem_CPU),
                self.max_mem_CPU_name,
                self.fmt_mem(self.max_mem_GPU),
                self.max_mem_GPU_name,
            )
