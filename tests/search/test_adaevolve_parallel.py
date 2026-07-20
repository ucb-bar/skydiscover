"""Tests for AdaEvolve parallel iteration support (deviation #5 fix).

Covers:
- AdaEvolveController: sequential vs parallel gating, end_iteration calls
- AdaEvolveDatabase.end_iteration: monotonic _iteration_count
- ChampSimEvaluator: ContextVar binary isolation under concurrency
- SerializableResult: sampling_mode / sampling_intensity fields
"""

import asyncio
import contextvars
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skydiscover.config import AdaEvolveDatabaseConfig
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program
from skydiscover.search.utils.discovery_utils import SerializableResult


# =========================================================================
# Helpers
# =========================================================================


def _scalar_db(num_islands=2, **extra):
    config = AdaEvolveDatabaseConfig(
        population_size=10,
        num_islands=num_islands,
        use_dynamic_islands=False,
        use_paradigm_breakthrough=False,
        **extra,
    )
    return AdaEvolveDatabase("test", config)


def _make_program(pid="p1", **metrics):
    return Program(id=pid, solution="int x;", metrics=metrics or {"score": 1.0})


# =========================================================================
# 1. AdaEvolveDatabase.end_iteration — monotonic _iteration_count
# =========================================================================


class TestEndIterationMonotonic:
    def test_in_order(self):
        db = _scalar_db()
        db.add(_make_program(), target_island=0)
        for i in range(5):
            db.end_iteration(i)
        assert db._iteration_count == 4

    def test_out_of_order_stays_monotonic(self):
        """Simulates parallel completions arriving out of order."""
        db = _scalar_db()
        db.add(_make_program(), target_island=0)
        db.end_iteration(3)
        assert db._iteration_count == 3
        db.end_iteration(1)
        assert db._iteration_count == 3  # must NOT regress
        db.end_iteration(5)
        assert db._iteration_count == 5

    def test_duplicate_iteration(self):
        db = _scalar_db()
        db.add(_make_program(), target_island=0)
        db.end_iteration(2)
        db.end_iteration(2)
        assert db._iteration_count == 2


# =========================================================================
# 2. SerializableResult carries sampling metadata
# =========================================================================


class TestSerializableResultSamplingFields:
    def test_defaults_are_none(self):
        r = SerializableResult()
        assert r.sampling_mode is None
        assert r.sampling_intensity is None

    def test_set_and_read(self):
        r = SerializableResult(sampling_mode="exploration", sampling_intensity=0.8)
        assert r.sampling_mode == "exploration"
        assert r.sampling_intensity == 0.8


# =========================================================================
# 3. AdaEvolveController — sequential vs parallel dispatch
# =========================================================================


def _make_mock_controller(max_parallel=1, num_iterations=5):
    """Build a minimal mock of AdaEvolveController for loop tests.

    We bypass __init__ entirely — the loop methods only need a few attrs.
    """
    from skydiscover.search.adaevolve.controller import AdaEvolveController

    ctrl = object.__new__(AdaEvolveController)

    # Config stub
    ctrl.config = MagicMock()
    ctrl.config.max_parallel_iterations = max_parallel

    # Database stub
    ctrl.database = MagicMock()
    ctrl.database.num_islands = 2
    ctrl.database.end_iteration = MagicMock()
    ctrl.database.log_status = MagicMock()
    ctrl.database.get_best_program = MagicMock(return_value=None)

    # Shutdown event
    import multiprocessing as mp
    ctrl.shutdown_event = mp.Event()

    # Track _run_iteration calls
    call_log = []

    async def fake_run_iteration(iteration, checkpoint_callback):
        call_log.append(("start", iteration))
        await asyncio.sleep(0.01)
        call_log.append(("end", iteration))

    ctrl._run_iteration = AsyncMock(side_effect=fake_run_iteration)
    ctrl._setup_iteration_stats_logging = MagicMock()
    ctrl._ensure_all_islands_seeded = MagicMock()
    ctrl._iteration_stats_log_path = None

    return ctrl, call_log


class TestAdaEvolveSequential:
    def test_sequential_calls_end_iteration(self):
        ctrl, _ = _make_mock_controller(max_parallel=1, num_iterations=4)
        asyncio.run(
            ctrl._run_discovery_sequential(0, 4)
        )
        assert ctrl._run_iteration.call_count == 4
        assert ctrl.database.end_iteration.call_count == 4
        # Called with iteration numbers 0-3
        called_iters = [c.args[0] for c in ctrl.database.end_iteration.call_args_list]
        assert called_iters == [0, 1, 2, 3]

    def test_sequential_end_iteration_on_error(self):
        ctrl, _ = _make_mock_controller(max_parallel=1)

        async def fail_iter(iteration, checkpoint_callback):
            raise RuntimeError("boom")

        ctrl._run_iteration = AsyncMock(side_effect=fail_iter)
        asyncio.run(
            ctrl._run_discovery_sequential(0, 3)
        )
        # end_iteration still called for every iteration despite errors
        assert ctrl.database.end_iteration.call_count == 3


class TestAdaEvolveParallel:
    def test_parallel_calls_end_iteration(self):
        ctrl, _ = _make_mock_controller(max_parallel=3)
        asyncio.run(
            ctrl._run_discovery_parallel(0, 6, max_parallel=3)
        )
        assert ctrl._run_iteration.call_count == 6
        assert ctrl.database.end_iteration.call_count == 6
        called_iters = sorted(c.args[0] for c in ctrl.database.end_iteration.call_args_list)
        assert called_iters == [0, 1, 2, 3, 4, 5]

    def test_parallel_concurrency(self):
        """Verify multiple iterations are actually in-flight simultaneously."""
        ctrl, call_log = _make_mock_controller(max_parallel=3)

        async def slow_iteration(iteration, checkpoint_callback):
            call_log.append(("start", iteration))
            await asyncio.sleep(0.05)
            call_log.append(("end", iteration))

        ctrl._run_iteration = AsyncMock(side_effect=slow_iteration)
        asyncio.run(
            ctrl._run_discovery_parallel(0, 3, max_parallel=3)
        )
        # All 3 should start before any ends (since they all sleep 50ms)
        starts = [i for tag, i in call_log if tag == "start"]
        ends = [i for tag, i in call_log if tag == "end"]
        assert len(starts) == 3
        assert len(ends) == 3
        # First end should come AFTER all starts (concurrency proof)
        first_end_idx = call_log.index(("end", ends[0]))
        start_indices = [call_log.index(("start", s)) for s in starts]
        assert all(si < first_end_idx for si in start_indices)

    def test_parallel_end_iteration_on_error(self):
        ctrl, _ = _make_mock_controller(max_parallel=2)

        async def sometimes_fail(iteration, checkpoint_callback):
            if iteration % 2 == 0:
                raise RuntimeError("boom")
            await asyncio.sleep(0.01)

        ctrl._run_iteration = AsyncMock(side_effect=sometimes_fail)
        asyncio.run(
            ctrl._run_discovery_parallel(0, 4, max_parallel=2)
        )
        # end_iteration still called for every iteration
        assert ctrl.database.end_iteration.call_count == 4

    def test_run_discovery_gates_on_config(self):
        """run_discovery delegates to sequential or parallel based on config."""
        ctrl, _ = _make_mock_controller(max_parallel=1)
        with patch.object(ctrl, "_run_discovery_sequential", new_callable=AsyncMock) as seq, \
             patch.object(ctrl, "_run_discovery_parallel", new_callable=AsyncMock) as par:
            asyncio.run(
                ctrl.run_discovery(0, 5)
            )
            seq.assert_called_once()
            par.assert_not_called()

        ctrl, _ = _make_mock_controller(max_parallel=3)
        with patch.object(ctrl, "_run_discovery_sequential", new_callable=AsyncMock) as seq, \
             patch.object(ctrl, "_run_discovery_parallel", new_callable=AsyncMock) as par:
            asyncio.run(
                ctrl.run_discovery(0, 5)
            )
            par.assert_called_once()
            seq.assert_not_called()

    def test_shutdown_event_stops_scheduling(self):
        ctrl, _ = _make_mock_controller(max_parallel=2)

        call_count = 0

        async def counting_iter(iteration, checkpoint_callback):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                ctrl.shutdown_event.set()
            await asyncio.sleep(0.01)

        ctrl._run_iteration = AsyncMock(side_effect=counting_iter)
        asyncio.run(
            ctrl._run_discovery_parallel(0, 100, max_parallel=2)
        )
        # Should stop well before 100
        assert call_count < 10


# =========================================================================
# 4. ContextVar binary isolation (ChampSimEvaluator pattern)
# =========================================================================


class TestContextVarBinaryIsolation:
    """Test the ContextVar pattern used in ChampSimEvaluator.

    We test the mechanism directly — ContextVar isolation across
    asyncio.create_task boundaries — without importing ChampSimEvaluator
    (which needs Ray/CHIA).
    """

    def test_tasks_see_own_contextvar(self):
        var: contextvars.ContextVar[str] = contextvars.ContextVar("test_var", default="unset")
        results = {}

        async def worker(name, value):
            var.set(value)
            await asyncio.sleep(0.02)
            results[name] = var.get()

        async def main():
            t1 = asyncio.create_task(worker("a", "binary_a"))
            t2 = asyncio.create_task(worker("b", "binary_b"))
            await asyncio.gather(t1, t2)

        asyncio.run(main())
        assert results["a"] == "binary_a"
        assert results["b"] == "binary_b"

    def test_set_then_read_within_task(self):
        """Mimics _dispatch_build setting, then _run reading, within one task."""
        var: contextvars.ContextVar[bytes] = contextvars.ContextVar("bin", default=None)
        read_values = []

        async def eval_program(binary_bytes):
            var.set(binary_bytes)
            await asyncio.sleep(0.01)  # simulates build await
            # Reads in _run (synchronous within the same task)
            read_values.append(var.get())

        async def main():
            t1 = asyncio.create_task(eval_program(b"AAA"))
            t2 = asyncio.create_task(eval_program(b"BBB"))
            await asyncio.gather(t1, t2)

        asyncio.run(main())
        assert set(read_values) == {b"AAA", b"BBB"}

    def test_interleaved_set_read_no_cross_talk(self):
        """Stress test: interleave set+read across many tasks."""
        var: contextvars.ContextVar[int] = contextvars.ContextVar("num", default=-1)
        mismatches = []

        async def worker(i):
            var.set(i)
            await asyncio.sleep(0.001 * (i % 5))
            got = var.get()
            if got != i:
                mismatches.append((i, got))

        async def main():
            tasks = [asyncio.create_task(worker(i)) for i in range(20)]
            await asyncio.gather(*tasks)

        asyncio.run(main())
        assert mismatches == [], f"Cross-talk detected: {mismatches}"
