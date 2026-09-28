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
# Removed the unused TensorFlow gradient samples and JSON save/load.
# Moved into the phipict package, formatted, linted and typed.

"""Lightweight hierarchical wall-clock profiler."""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO

import numpy as np

DEFAULT_STATS_MODE = "WELFORD"


# https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
def format_time(t: float) -> str:
    """Format a duration as ``HH:MM:SS.sss``.

    Parameters
    ----------
    t : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration.
    """
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def time_unit(t: float, m: float = 1000.0) -> str:
    """Format a duration with the largest fitting unit.

    Parameters
    ----------
    t : float
        Duration in seconds.
    m : float, optional
        A unit is used if the value in that unit is below ``m``. Default is 1000.

    Returns
    -------
    str
        Formatted duration, e.g. ``"12.345 ms"``.
    """
    units = ["ns", "us", "ms", "s", "m", "h", "d", "y"]
    x = [1e-9, 1e-6, 1e-3, 1.0, 60.0, 3600.0, 3600.0 * 24.0, 3600.0 * 24.0 * 365.0]
    for d, u in zip(x, units, strict=False):
        if t / d < m or u == units[-1]:
            return f"{t / d:.3f} {u:<2}"
    raise AssertionError("unreachable")


class Profiler:
    """Hierarchical profiler measuring wall-clock time of nested samples.

    Parameters
    ----------
    verbose : bool, optional
        Whether to print the duration of every finished sample. Default is False.
    active : bool, optional
        Whether profiling is enabled. Default is True.
    stats_mode : {"LIST", "STREAMING", "WELFORD"}, optional
        How timings are accumulated: keep all samples, keep running sums, or use
        Welford's online algorithm for the variance. Default is ``"WELFORD"``.

    Raises
    ------
    ValueError
        If ``stats_mode`` is unknown.
    """

    class Sample:
        """Timing node that stores all measured durations.

        Parameters
        ----------
        name : str
            Sample name.
        parent : Profiler.Sample or None
            Parent node, None for the root.
        group : str or None, optional
            Group this sample is additionally accounted to. Default is None.
        """

        def __init__(
            self, name: str, parent: "Profiler.Sample | None", group: str | None = None
        ) -> None:
            self.name = name
            self.parent = parent
            self.group = group
            self.children: dict[str, Profiler.Sample] = {}
            self.samples: list[float] = []
            self.start = time.time()

        def add_sample(self, sample: float) -> None:
            """Add a measured duration.

            Parameters
            ----------
            sample : float
                Duration in seconds.
            """
            self.samples.append(sample)

        def begin(self) -> None:
            """Start a measurement."""
            self.start = time.time()

        def end(self) -> None:
            """Stop the current measurement and record it."""
            self.add_sample(time.time() - self.start)

        def get_child(self, name: str, group: str | None = None) -> "Profiler.Sample":
            """Get a child node, creating it if necessary.

            Parameters
            ----------
            name : str
                Child name.
            group : str or None, optional
                Group of a newly created child. Default is None.

            Returns
            -------
            Profiler.Sample
                The child node, of the same type as this node.
            """
            if name not in self.children:
                self.children[name] = self.__class__(name, self, group=group)
            return self.children[name]

        @property
        def num_samples(self) -> int:
            """Number of recorded durations.

            Returns
            -------
            int
                Number of durations recorded so far.
            """
            return len(self.samples)

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> float:
            return self.samples[idx]

        @property
        def min(self) -> float:
            """Minimum duration.

            Returns
            -------
            float
                Smallest recorded duration, in seconds.
            """
            return np.amin(self.samples)

        @property
        def max(self) -> float:
            """Maximum duration.

            Returns
            -------
            float
                Largest recorded duration, in seconds.
            """
            return np.amax(self.samples)

        @property
        def mean(self) -> float:
            """Mean duration.

            Returns
            -------
            float
                Mean of the recorded durations, in seconds.
            """
            return np.mean(self.samples)

        @property
        def var(self) -> float:
            """Variance of the durations.

            Returns
            -------
            float
                Variance of the recorded durations, in seconds squared.
            """
            return np.var(self.samples)

        @property
        def std(self) -> float:
            """Standard deviation of the durations.

            Returns
            -------
            float
                Standard deviation of the recorded durations, in seconds.
            """
            return np.std(self.samples)

        @property
        def sum(self) -> float:
            """Total duration.

            Returns
            -------
            float
                Sum of the recorded durations, in seconds.
            """
            return np.sum(self.samples)

    class StreamingSample(Sample):
        """Timing node that only keeps running statistics.

        Parameters
        ----------
        name : str
            Sample name.
        parent : Profiler.Sample or None
            Parent node, None for the root.
        group : str or None, optional
            Group this sample is additionally accounted to. Default is None.
        """

        def __init__(
            self, name: str, parent: "Profiler.Sample | None", group: str | None = None
        ) -> None:
            self._min = np.finfo(np.float64).max
            self._max = np.finfo(np.float64).min
            self._sum = np.float64(0)
            self._sum_sq = np.float64(0)
            self._num_samples = 0
            self._last_sample = np.float64(0)
            super().__init__(name, parent, group=group)
            del self.samples

        def add_sample(self, sample: float) -> None:
            """Add a measured duration.

            Parameters
            ----------
            sample : float
                Duration in seconds.
            """
            sample = np.float64(sample)
            self._min = np.minimum(self._min, sample)
            self._max = np.maximum(self._max, sample)
            self._sum += sample
            self._sum_sq += sample * sample
            self._num_samples += 1
            self._last_sample = sample

        def __getitem__(self, idx: int) -> float:
            if idx == -1:
                return self._last_sample
            else:
                raise IndexError(
                    "StreamingSample only keeps the last sample (idx = -1)."
                )

        @property
        def num_samples(self) -> int:
            """Number of recorded durations.

            Returns
            -------
            int
                Number of durations recorded so far.
            """
            return self._num_samples

        @property
        def min(self) -> float:
            """Minimum duration.

            Returns
            -------
            float
                Smallest recorded duration, in seconds.
            """
            return self._min

        @property
        def max(self) -> float:
            """Maximum duration.

            Returns
            -------
            float
                Largest recorded duration, in seconds.
            """
            return self._max

        @property
        def mean(self) -> float:
            """Mean duration.

            Returns
            -------
            float
                Mean of the recorded durations, in seconds.
            """
            return np.divide(self._sum, self._num_samples, dtype=np.float64)

        @property
        def var(self) -> float:
            """Variance of the durations.

            Returns
            -------
            float
                Variance of the recorded durations, in seconds squared.
            """
            mean = self.mean
            return (np.divide(self._sum_sq, self._num_samples, dtype=np.float64)) - (
                mean * mean
            )

        @property
        def std(self) -> float:
            """Standard deviation of the durations.

            Returns
            -------
            float
                Standard deviation of the recorded durations, in seconds.
            """
            return np.sqrt(self.var)

        @property
        def sum(self) -> float:
            """Total duration.

            Returns
            -------
            float
                Sum of the recorded durations, in seconds.
            """
            return self._sum

    class WelfordOnlineSample(StreamingSample):
        """Streaming timing node using Welford's online variance algorithm.

        See
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

        Parameters
        ----------
        name : str
            Sample name.
        parent : Profiler.Sample or None
            Parent node, None for the root.
        group : str or None, optional
            Group this sample is additionally accounted to. Default is None.
        """

        def __init__(
            self, name: str, parent: "Profiler.Sample | None", group: str | None = None
        ) -> None:
            super().__init__(name, parent, group=group)
            del self._sum_sq
            self._mean = np.float64(0)
            self._M2 = np.float64(0)

        def add_sample(self, sample: float) -> None:
            """Add a measured duration.

            Parameters
            ----------
            sample : float
                Duration in seconds.
            """
            sample = np.float64(sample)
            self._last_sample = sample
            self._min = np.minimum(self._min, sample)
            self._max = np.maximum(self._max, sample)
            self._sum += sample

            self._num_samples += 1
            delta = sample - self._mean
            self._mean += np.divide(delta, self._num_samples, dtype=np.float64)
            self._M2 += delta * (sample - self._mean)

        @property
        def mean(self) -> float:
            """Mean duration.

            Returns
            -------
            float
                Mean of the recorded durations, in seconds.
            """
            return self._mean

        @property
        def var(self) -> float:
            """Variance of the durations.

            Returns
            -------
            float
                Variance of the recorded durations, in seconds squared.
            """
            return np.divide(self._M2, self._num_samples, dtype=np.float64)

    def __init__(
        self,
        verbose: bool = False,
        active: bool = True,
        stats_mode: str = DEFAULT_STATS_MODE,
    ) -> None:
        stats_mode = stats_mode.upper()
        self._root: Profiler.Sample
        if stats_mode == "STREAMING":
            self._root = Profiler.StreamingSample("__root__", None)
        elif stats_mode == "LIST":
            self._root = Profiler.Sample("__root__", None)
        elif stats_mode == "WELFORD":
            self._root = Profiler.WelfordOnlineSample("__root__", None)
        else:
            raise ValueError("Unknown stats_mode '%s'" % stats_mode)
        self._current = self._root
        # group name -> [group sample, number of currently open samples]
        self._groups: dict[str, list[Any]] = {}
        self.verbose = verbose
        self._active = active

    @property
    def is_active(self) -> bool:
        """Whether profiling is enabled.

        Returns
        -------
        bool
            True if samples are being measured.
        """
        return self._active

    def current_sample_path(self) -> str:
        """Get the path of the currently open sample.

        Returns
        -------
        str
            Names of the open samples from the root, joined by ``/``.
        """
        names = []
        c = self._current
        while c != self._root:
            names.append(c.name)
            assert c.parent is not None
            c = c.parent
        return "/".join(names[::-1])

    def _get_group(self, group: str) -> list[Any]:
        """Get the bookkeeping entry for a group, creating it if necessary.

        Parameters
        ----------
        group : str
            Group name.

        Returns
        -------
        list of Any
            A two-element list ``[sample, n_open]`` holding the group's sample
            node and the number of currently open samples in the group.
        """
        if group not in self._groups:
            self._groups[group] = [self._root.get_child(group, group=None), 0]
        return self._groups[group]

    def _begin_group(self, group: str | None) -> None:
        """Start the group's measurement if it is not already running.

        Parameters
        ----------
        group : str or None
            Group name, or None for no group, in which case this is a no-op.
        """
        if group is None:
            return
        grp = self._get_group(group)
        if grp[1] <= 0:
            grp[1] = 0
            grp[0].begin()
        grp[1] += 1

    def _end_group(self, group: str | None) -> None:
        """Stop the group's measurement once its last open sample closes.

        Parameters
        ----------
        group : str or None
            Group name, or None for no group, in which case this is a no-op.
        """
        if group is None:
            return
        grp = self._get_group(group)
        grp[1] -= 1
        if grp[1] <= 0:
            grp[0].end()
            grp[1] = 0

    def _begin_sample(self, name: str, group: str | None = None) -> None:
        """Open a sample below the current one and start measuring it.

        Parameters
        ----------
        name : str
            Sample name.
        group : str or None, optional
            Group the duration is additionally accounted to. Default is None.
        """
        if not self._active:
            return
        self._current = self._current.get_child(name, group)
        self._begin_group(group)
        self._current.begin()

    def _end_sample(self, verbose: bool | None = None) -> None:
        """Stop the current sample, record it and return to its parent.

        Parameters
        ----------
        verbose : bool or None, optional
            Whether to print the duration. If None, the profiler's ``verbose``
            setting is used. Default is None.
        """
        if not self._active:
            return

        self._current.end()
        if verbose or (self.verbose and verbose is None):
            print(f"'{self._current.name}': {time_unit(self._current[-1])}")
        self._end_group(self._current.group)
        assert self._current.parent is not None
        self._current = self._current.parent

    @contextmanager
    def sample(
        self, name: str, verbose: bool | None = None, group: str | None = None
    ) -> Iterator[None]:
        """Measure the duration of a block of code.

        Parameters
        ----------
        name : str
            Sample name, nested below the currently open sample.
        verbose : bool or None, optional
            Whether to print the duration. If None, the profiler's ``verbose``
            setting is used. Default is None.
        group : str or None, optional
            Group the duration is additionally accounted to. Default is None.

        Yields
        ------
        None
            Control to the measured code.
        """
        if self._active:
            self._begin_sample(name, group=group)
        try:
            yield  # run code to measure
        finally:
            if self._active:
                self._end_sample(verbose)

    # for formatting
    def _get_max_depth(
        self, sample: "Profiler.Sample", level: int, level_indent: int
    ) -> int:
        """Get the width needed to print the sample names below ``sample``.

        Parameters
        ----------
        sample : Profiler.Sample
            Subtree root to measure.
        level : int
            Nesting level of ``sample``, used to account for its indentation.
        level_indent : int
            Number of characters each nesting level is indented by.

        Returns
        -------
        int
            Length of the longest indented name in the subtree.
        """
        depth = 0
        for name in sample.children:
            depth = max(depth, level_indent * level + len(name))
            depth = max(
                depth,
                self._get_max_depth(sample.children[name], level + 1, level_indent),
            )
        return depth

    # for formatting
    def _get_indent(self, level_indent: int, max_indent: int) -> int:
        """Get the name column width to use when printing the statistics.

        Parameters
        ----------
        level_indent : int
            Number of characters each nesting level is indented by.
        max_indent : int
            Upper bound on the width. Negative for no bound.

        Returns
        -------
        int
            The width to print names in.
        """
        indent = self._get_max_depth(self._root, 0, level_indent)
        if max_indent < 0:
            return indent
        return min(max_indent, indent)

    def _print_stats(
        self,
        sample: "Profiler.Sample",
        level: int,
        t_parent: float,
        t_root: float,
        file: TextIO,
        level_indent: int,
        max_indent: int,
    ) -> None:
        """Recursively write the statistics of a sample's children.

        Parameters
        ----------
        sample : Profiler.Sample
            Sample whose children are printed.
        level : int
            Nesting level of ``sample``, controlling the indentation.
        t_parent : float
            Total duration of ``sample``, used for the relative percentages.
        t_root : float
            Total duration of the root sample, used for the relative percentages.
        file : TextIO
            Stream to write to.
        level_indent : int
            Number of characters each nesting level is indented by.
        max_indent : int
            Upper bound on the name column width. Negative for no bound.
        """
        t_remaining = t_parent
        for name, current in sorted(
            sample.children.items(), key=lambda e: e[1].sum, reverse=True
        ):
            total_time = current.sum
            if level == 0:
                t_parent = total_time
                t_root = total_time
            t_remaining -= total_time
            s = " " * level_indent * level + f"'{name}'"
            s = ("{:<" + str(max_indent) + "}").format(s)
            file.write(
                f"{s}: {time_unit(current.mean):>10}, {current.num_samples:10d}, {format_time(total_time):>12}, {100.0 * total_time / t_parent: 10.05f}, {100.0 * total_time / t_root: 10.05f}, {time_unit(current.std):>10}, {time_unit(current.min):>10}, {time_unit(current.max):>10}\n"
            )
            self._print_stats(
                current, level + 1, total_time, t_root, file, level_indent, max_indent
            )
        if t_remaining > 0.0 and len(sample.children) > 0:
            s = " " * level_indent * level + "{}".format("-")
            s = ("{:<" + str(max_indent) + "}").format(s)
            file.write(
                "{}: {:>10}, {:10d}, {:>12}, {: 10.05f}, {: 10.05f}\n".format(
                    s,
                    "-",
                    0,
                    format_time(t_remaining),
                    100.0 * t_remaining / t_parent,
                    100.0 * t_remaining / t_root,
                )
            )

    def stats(
        self, file: TextIO = sys.stdout, level_indent: int = 4, max_indent: int = -1
    ) -> None:
        """Write a table of all timings.

        Parameters
        ----------
        file : TextIO, optional
            Output stream. Default is ``sys.stdout``.
        level_indent : int, optional
            Indentation per nesting level. Default is 4.
        max_indent : int, optional
            Maximum width of the name column, negative for unlimited.
            Default is -1.
        """
        if self._active:
            max_indent = self._get_indent(level_indent, max_indent) + 2
            file.write(
                (
                    "{:<"
                    + str(max_indent)
                    + "}: {:^10}| {:^10}| {:^12}| {:^10}| {:^10}| {:^10}| {:^10}| {:^10}|\n"
                ).format(
                    "Sample name",
                    "average",
                    "# samples",
                    "total",
                    "% parent",
                    "% root",
                    "std",
                    "min",
                    "max",
                )
            )
            self._print_stats(self._root, 0, 0, 0, file, level_indent, max_indent)
        else:
            file.write("\nProfiling disabled.\n")


if __name__ == "__main__":
    # test
    print("--- Variance stats tests ---")
    samples_low = [4e-3, 6e-4, 1e-3, 45e-3, 8e-4, 1.605e-3]
    samples_hight = [0.2, 0.00001, 12, 15983, 2.0400862, 3e-12]

    def _test_stats(stats_mode: str) -> None:
        """Print the statistics of two fixed sample sets for one stats mode.

        Parameters
        ----------
        stats_mode : str
            ``"LIST"``, ``"STREAMING"`` or ``"WELFORD"``.
        """
        p = Profiler(stats_mode=stats_mode)
        sample = p._root.get_child("low-var")
        for s in samples_low:
            sample.add_sample(s)

        sample = p._root.get_child("high-var")
        for s in samples_hight:
            sample.add_sample(s)
        p.stats()

    for mode in ["LIST", "STREAMING", "WELFORD"]:
        print(mode)
        _test_stats(mode)
        print()
    print()
    print()

    print("--- Profiler tests ---")

    def _test_profiler(p: Profiler) -> None:
        """Run a fixed set of nested sleeps through a profiler.

        Parameters
        ----------
        p : Profiler
            Profiler to exercise.
        """
        with p.sample("total (root)", True):
            for i in range(2):
                with p.sample("loop 4"):
                    with p.sample("test 1"):
                        time.sleep(0.3)
                        print(p.current_sample_path())
                    with p.sample("test 2"):
                        time.sleep(0.7)
                    time.sleep(0.12)
                # time.sleep(0.11)
            for i in range(65):
                with p.sample("loop 40"):
                    time.sleep(0.01)
            time.sleep(0.1)
            print(p.current_sample_path())
        with p.sample("test (root 2)"):
            time.sleep(0.1)

    p_samples = Profiler(stats_mode="LIST")
    p_stream = Profiler(stats_mode="STREAMING")
    p_welford = Profiler(stats_mode="WELFORD")

    try:
        _test_profiler(p_samples)
        _test_profiler(p_stream)
        _test_profiler(p_welford)
    except KeyboardInterrupt:
        pass
    print()
    print("sample-list profiler:")
    p_samples.stats()
    print()
    print("streaming profiler:")
    p_stream.stats()
    print()
    print("welford profiler:")
    p_welford.stats()
    print()

    def debug_mode(profiler: Profiler, sample_type: type) -> None:
        """Check that all samples of ``profiler`` are of ``sample_type``.

        Parameters
        ----------
        profiler : Profiler
            Profiler whose sample tree is checked.
        sample_type : type
            Expected type of every sample node.

        Raises
        ------
        AssertionError
            If a sample node is not an instance of ``sample_type``.
        """

        def debug_sample(sample: Profiler.Sample, sample_type: type) -> None:
            print("checking %s... " % (sample.name))
            assert isinstance(sample, sample_type), "is: %s, expected: %s" % (
                type(sample).__name__,
                sample_type.__name__,
            )
            print("OK")
            for name, child in sample.children.items():
                debug_sample(child, sample_type)

        debug_sample(profiler._root, sample_type)

    debug_mode(p_welford, Profiler.WelfordOnlineSample)
    # debug_mode(p_samples, Profiler.WelfordOnlineSample)

    sys.exit()
else:
    DEFAULT_PROFILER = Profiler()
    sample = DEFAULT_PROFILER.sample
    SAMPLE = sample
    stats = DEFAULT_PROFILER.stats
    STATS = stats
