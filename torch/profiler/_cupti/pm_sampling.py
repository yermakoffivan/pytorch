# mypy: allow-untyped-defs
"""CUPTI PM-sampling: continuous GPU performance-monitor sampling (SM-active %, DRAM-throughput
%) that runs concurrently with the activity monitor.

PM sampling reads dedicated on-chip performance-monitor units, so it has negligible GPU-side cost
and coexists with the activity subscriber, but it locks the GPU clocks while active (which can
shift absolute kernel durations) -- so it is opt-in via custom_profiler_config
{"enable_pm_sampling": true}, not always-on like the environment counters.

The HW units sample autonomously into a device-side buffer between start and decode, so there is
no background thread: the session starts at window open and the buffer is drained once at window
close. ``decode`` drains the entire buffer in a single call and caps at its ``max_samples`` (a
second call does not resume), so the host image is sized above the buffer's sample capacity.
Samples are HW-timestamped in the CUPTI clock domain, the same base the monitor's record
timestamps use, so running them through the monitor's clock conversion aligns them with the
activity records; they surface as GPU counter tracks (siblings of the environment counters)."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)

# (counter_id, display name, cupti metric). counter_ids continue after the environment counters
# (1-5, see monitor_trace._ENV_COUNTERS) so GpuCounterDescriptor ids stay unique across all GPU
# counter tracks. The full sm__throughput.* metric needs many passes (not single-pass, so not
# PM-sampleable); sm__cycles_active (SM-active %) is the single-pass SM-utilization metric.
PM_METRICS: tuple[tuple[int, str, str], ...] = (
    (6, "SM Active (%)", "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"),
    (7, "DRAM BW (%)", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
)

_SAMPLING_INTERVAL_NS = (
    1_000_000  # 1 ms HW sampling interval (GPU_TIME_INTERVAL units = ns)
)
# Sized so a bounded window's samples fit without overflowing before the close-time decode; the
# buffer is allocated only while PM sampling is enabled (window-scoped) and freed at disable.
_HW_BUFFER_SIZE = 256 * 1024 * 1024
# decode() drains the whole HW buffer in one call and caps at max_samples (it does not resume on a
# second call), so size the host image above the buffer's sample capacity. ~4 KiB/sample is a
# conservative lower bound (measured ~10 KiB), so a window that fit the buffer never caps.
_MAX_SAMPLES = _HW_BUFFER_SIZE // 4096


def is_available() -> bool:
    try:
        import cupti.pm_sampling  # noqa: F401  # pyrefly: ignore[missing-import]
    except Exception:
        return False
    return True


class PmSampler:
    """A single-device PM-sampling session, started at window open and drained at window close --
    both on the calling (foreground) thread, no background polling. Decoded samples are converted
    to unix-epoch ns and handed to ``sink`` as a named-column frame (``start_ns``/``device_id``/
    ``c<counter_id>`` per metric) for the observer to bucket into its window. Single-device by
    design (``Collector.disable`` tears down process-global profiler state, so the common
    one-process-per-GPU case is covered; the device is the current CUDA device resolved at
    :meth:`start`)."""

    def __init__(
        self,
        sink: Callable[[dict[str, Any]], None],
        convert_ns: Callable[[np.ndarray], np.ndarray],
        *,
        sampling_interval_ns: int = _SAMPLING_INTERVAL_NS,
    ) -> None:
        self._sink = sink
        self._convert_ns = convert_ns
        self._sampling_interval_ns = sampling_interval_ns
        self._metric_names = [m for _, _, m in PM_METRICS]
        self._col: Any = None
        self._device: int = -1

    def start(self) -> None:
        """Enable + configure + start the collector on the current device. No-op if already
        started or PM sampling is unavailable; a failure is logged and leaves it inactive."""
        if self._col is not None or not is_available():
            return
        col = None
        try:
            from cupti import pm_sampling as pm  # pyrefly: ignore[missing-import]

            import torch

            device = torch.cuda.current_device()
            col = pm.Collector(device_index=device)
            col.enable()
            col.configure(
                metrics=self._metric_names,
                hardware_buffer_size=_HW_BUFFER_SIZE,
                sampling_interval=self._sampling_interval_ns,
                trigger_mode=pm.TriggerMode.GPU_TIME_INTERVAL,
            )
            col.start()
            self._col = col
            self._device = device
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            if col is not None:
                try:
                    col.disable()
                except Exception:
                    pass

    def stop(self) -> None:
        """Stop the collector, drain the HW buffer once into ``sink``, and disable. Idempotent."""
        col = self._col
        if col is None:
            return
        self._col = None
        try:
            col.stop()
            self._drain(col)
        except Exception:
            logger.exception("PM sampling decode error")
        finally:
            try:
                col.disable()
            except Exception:
                pass

    def _drain(self, col: Any) -> None:
        # A single decode drains the whole buffer (see module docstring). An overflow during the
        # window is reported as a MemoryError (KEEP_OLDEST); warn rather than fail the export.
        try:
            cd = col.decode(max_samples=_MAX_SAMPLES)
        except MemoryError:
            logger.warning(
                "PM sampling HW buffer overflowed; samples were dropped. Use a shorter active "
                "window or a larger hardware buffer."
            )
            return
        n = cd.num_completed_samples
        if not n:
            return
        if n >= _MAX_SAMPLES:
            # The image filled before the buffer drained; the newest samples past the cap were
            # dropped. Sized not to happen for a window that fit the buffer, so this is a backstop.
            logger.warning(
                "PM sampling decoded the maximum %d samples; some were dropped.",
                _MAX_SAMPLES,
            )
        # Iterating CounterData evaluates the metrics host-side (the decode cost).
        samples = [cd[i] for i in range(n)]
        ts = np.fromiter((s.start_timestamp for s in samples), dtype=np.int64, count=n)
        # The very first sample of a session has an uninitialized (0) start timestamp; drop it.
        keep = ts > 0
        if not keep.any():
            return
        ts = ts[keep]
        vals = np.array([s.metric_values for s in samples], dtype=np.float64)[keep]
        frame: dict[str, Any] = {
            "start_ns": self._convert_ns(ts),
            "device_id": np.full(len(ts), self._device, dtype=np.int64),
        }
        for j, (cid, _, _) in enumerate(PM_METRICS):
            frame[f"c{cid}"] = vals[:, j]
        self._sink(frame)
