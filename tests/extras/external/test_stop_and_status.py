"""
Tests for AlphaEvolve backend stop propagation and status callbacks.

These tests mock the AlphaEvolve SDK to avoid GCP dependencies.
They verify:
  - stop_check triggers max_programs_evaluated override
  - status_callback fires on each evaluation
  - monitor_callback is forwarded to the evaluator adapter
  - stop override logs a warning
"""

import asyncio
import logging
import os
import textwrap
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skydiscover.extras.external.alphaevolve_backend import (
    _make_alphaevolve_evaluator,
    run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED_SOURCE = "def solve(x):\n    return x + 1\n"

TRIVIAL_EVALUATOR = textwrap.dedent("""\
    def evaluate(program_path):
        return {"combined_score": 0.5}
""")


def _write_files(tmp_path):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED_SOURCE)
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(TRIVIAL_EVALUATOR)
    output = tmp_path / "output"
    output.mkdir()
    return str(seed), str(evaluator), str(output)


def _make_config():
    from skydiscover.config import Config

    config = object.__new__(Config)
    config.file_suffix = ".py"
    config.language = "python"
    config.max_iterations = 5
    return config


class FakeExperiment:
    """Minimal experiment stub that tracks max_programs_evaluated changes."""

    def __init__(self, max_programs: int = 5):
        self.max_programs_evaluated = max_programs
        self.parallel_evaluation = False
        self.stats: Dict[str, int] = {
            "num_programs_generated": 0,
            "num_programs_evaluated": 0,
        }

    def stopping_criteria_met(self):
        return self.stats["num_programs_evaluated"] >= self.max_programs_evaluated

    def create_experiment(self, config):
        pass

    def create_initial_program(self, program):
        pass

    def start_experiment(self):
        pass

    def list_programs(self, params=None):
        return {"alphaEvolvePrograms": []}


def _sdk_patches(experiment, fake_controller_loop):
    """Return a context manager that patches the AlphaEvolve SDK lazy imports.

    The SDK classes are imported inside ``run()`` via
    ``from alpha_evolve.client import AlphaEvolveClient`` etc., so we patch
    at the alpha_evolve module level.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with (
            patch(
                "skydiscover.extras.external.alphaevolve_backend._get_alphaevolve_config",
                return_value={"project_id": "test", "engine_id": "test"},
            ),
            patch(
                "skydiscover.extras.external.alphaevolve_backend._validate_gcp_credentials"
            ),
            patch(
                "alpha_evolve.client.AlphaEvolveClient",
                return_value=MagicMock(),
            ),
            patch(
                "alpha_evolve.experiment.AlphaEvolveExperiment",
                return_value=experiment,
            ),
            patch(
                "alpha_evolve.controller.run_controller_loop",
                side_effect=fake_controller_loop,
            ),
        ):
            yield

    return _ctx()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStopPropagation:
    """Verify stop_check triggers max_programs_evaluated override."""

    @pytest.mark.asyncio
    async def test_stop_overrides_max_programs(self, tmp_path):
        seed_path, eval_path, output_dir = _write_files(tmp_path)
        config = _make_config()

        experiment = FakeExperiment(max_programs=50)
        stop_after = 2

        async def fake_controller_loop(experiment, **kwargs):
            while not experiment.stopping_criteria_met():
                await asyncio.sleep(0.3)
                experiment.stats["num_programs_generated"] += 1
                experiment.stats["num_programs_evaluated"] += 1

        def stop_check():
            return experiment.stats["num_programs_evaluated"] >= stop_after

        with _sdk_patches(experiment, fake_controller_loop):
            result = await run(
                program_path=seed_path,
                evaluator_path=eval_path,
                config_obj=config,
                iterations=50,
                output_dir=output_dir,
                stop_check=stop_check,
            )

        # Should have stopped well before 50
        assert experiment.stats["num_programs_evaluated"] < 10

    @pytest.mark.asyncio
    async def test_no_stop_check_runs_to_completion(self, tmp_path):
        seed_path, eval_path, output_dir = _write_files(tmp_path)
        config = _make_config()

        experiment = FakeExperiment(max_programs=3)

        async def fake_controller_loop(experiment, **kwargs):
            for _ in range(3):
                await asyncio.sleep(0.01)
                experiment.stats["num_programs_generated"] += 1
                experiment.stats["num_programs_evaluated"] += 1

        with _sdk_patches(experiment, fake_controller_loop):
            result = await run(
                program_path=seed_path,
                evaluator_path=eval_path,
                config_obj=config,
                iterations=3,
                output_dir=output_dir,
                stop_check=None,
            )

        assert experiment.stats["num_programs_evaluated"] == 3
        assert experiment.max_programs_evaluated == 3


class TestStatusCallback:
    """Verify status_callback fires on evaluation progress."""

    @pytest.mark.asyncio
    async def test_status_callback_fires_on_progress(self, tmp_path):
        seed_path, eval_path, output_dir = _write_files(tmp_path)
        config = _make_config()

        experiment = FakeExperiment(max_programs=3)
        status_calls: List[int] = []

        async def fake_controller_loop(experiment, **kwargs):
            for i in range(1, 4):
                await asyncio.sleep(0.05)
                experiment.stats["num_programs_generated"] = i
                experiment.stats["num_programs_evaluated"] = i

        def status_callback(evaluated):
            status_calls.append(evaluated)

        with _sdk_patches(experiment, fake_controller_loop):
            result = await run(
                program_path=seed_path,
                evaluator_path=eval_path,
                config_obj=config,
                iterations=3,
                output_dir=output_dir,
                status_callback=status_callback,
            )

        assert len(status_calls) > 0
        assert all(isinstance(c, int) for c in status_calls)
        assert status_calls[-1] == 3

    @pytest.mark.asyncio
    async def test_status_callback_not_called_when_none(self, tmp_path):
        """No crash when status_callback is None."""
        seed_path, eval_path, output_dir = _write_files(tmp_path)
        config = _make_config()

        experiment = FakeExperiment(max_programs=1)

        async def fake_controller_loop(experiment, **kwargs):
            experiment.stats["num_programs_evaluated"] = 1

        with _sdk_patches(experiment, fake_controller_loop):
            result = await run(
                program_path=seed_path,
                evaluator_path=eval_path,
                config_obj=config,
                iterations=1,
                output_dir=output_dir,
                status_callback=None,
            )

        assert result is not None


class TestMonitorCallback:
    """Verify monitor_callback is forwarded to the evaluator adapter."""

    def test_monitor_callback_fires_on_evaluation(self, tmp_path):
        evaluator = tmp_path / "evaluator.py"
        evaluator.write_text(TRIVIAL_EVALUATOR)

        calls: List[Any] = []

        def monitor_cb(program, iteration):
            calls.append((program, iteration))

        ae_evaluator = _make_alphaevolve_evaluator(
            str(evaluator), monitor_callback=monitor_cb
        )

        candidate = {
            "content": {"files": [{"path": "prog.py", "content": "x = 1"}]}
        }
        result = ae_evaluator(candidate)

        assert len(calls) == 1
        prog, iteration = calls[0]
        assert prog.solution == "x = 1"
        assert iteration == 1

    def test_monitor_callback_none_does_not_crash(self, tmp_path):
        evaluator = tmp_path / "evaluator.py"
        evaluator.write_text(TRIVIAL_EVALUATOR)

        ae_evaluator = _make_alphaevolve_evaluator(str(evaluator), monitor_callback=None)
        candidate = {
            "content": {"files": [{"path": "prog.py", "content": "x = 1"}]}
        }
        result = ae_evaluator(candidate)
        assert "scores" in result


class TestStopWarningLogged:
    """Verify that the stop override logs a warning."""

    @pytest.mark.asyncio
    async def test_stop_logs_warning(self, tmp_path, caplog):
        seed_path, eval_path, output_dir = _write_files(tmp_path)
        config = _make_config()

        experiment = FakeExperiment(max_programs=50)

        async def fake_controller_loop(experiment, **kwargs):
            while not experiment.stopping_criteria_met():
                await asyncio.sleep(0.05)
                experiment.stats["num_programs_evaluated"] += 1

        def stop_check():
            return experiment.stats["num_programs_evaluated"] >= 1

        with _sdk_patches(experiment, fake_controller_loop):
            with caplog.at_level(logging.WARNING):
                await run(
                    program_path=seed_path,
                    evaluator_path=eval_path,
                    config_obj=config,
                    iterations=50,
                    output_dir=output_dir,
                    stop_check=stop_check,
                )

        assert any(
            "overriding max_programs_evaluated" in r.message for r in caplog.records
        )
