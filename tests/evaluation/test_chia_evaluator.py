"""Tests for ChiaEvaluator — all mock-based, no Ray or infrastructure needed.

Covers:
- CB-01: Evaluator contract conformance
- CB-02: Build-then-fan-out-runs dispatch flow
- CB-03: Async safety (ray.get via asyncio.to_thread)
- CB-04: Error handling (program vs transient errors, retry logic)
- D-13:  JSONL logging
- close() behavior
"""

import asyncio
import inspect
import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from skydiscover.evaluation.evaluation_result import EvaluationResult


# ------------------------------------------------------------------
# Mock exception classes (real classes so isinstance checks work)
# ------------------------------------------------------------------


class _MockRayTaskError(Exception):
    """Mock for ray.exceptions.RayTaskError."""

    def __init__(self, cause: Exception = None) -> None:
        self.cause = cause if cause is not None else Exception("remote error")
        super().__init__(str(self.cause))


class _MockWorkerCrashedError(Exception):
    """Mock for ray.exceptions.WorkerCrashedError."""

    pass


class _MockNodeDiedError(Exception):
    pass


class _MockObjectLostError(Exception):
    pass


class _MockGetTimeoutError(Exception):
    pass


class _MockRayActorError(Exception):
    pass


class _MockActorDiedError(Exception):
    pass


class _MockActorUnavailableError(Exception):
    pass


class _MockTaskPlacementGroupRemoved(Exception):
    pass


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def mock_ray():
    """Patch the ``ray`` module used by chia_evaluator with mock classes.

    Creates a fake ``ray`` module with ``ray.get`` as a regular MagicMock
    and ``ray.exceptions`` populated with real exception classes so
    ``isinstance`` checks work at runtime.
    """
    ray_mod = types.ModuleType("ray")
    ray_mod.get = MagicMock(name="ray.get")

    exc_mod = types.ModuleType("ray.exceptions")
    exc_mod.RayTaskError = _MockRayTaskError
    exc_mod.WorkerCrashedError = _MockWorkerCrashedError
    exc_mod.NodeDiedError = _MockNodeDiedError
    exc_mod.ObjectLostError = _MockObjectLostError
    exc_mod.GetTimeoutError = _MockGetTimeoutError
    exc_mod.RayActorError = _MockRayActorError
    exc_mod.ActorDiedError = _MockActorDiedError
    exc_mod.ActorUnavailableError = _MockActorUnavailableError
    exc_mod.TaskPlacementGroupRemoved = _MockTaskPlacementGroupRemoved

    ray_mod.exceptions = exc_mod

    # Inject into sys.modules so ``import ray`` inside chia_evaluator resolves
    old_ray = sys.modules.get("ray")
    old_ray_exc = sys.modules.get("ray.exceptions")
    sys.modules["ray"] = ray_mod
    sys.modules["ray.exceptions"] = exc_mod

    yield ray_mod

    # Restore
    if old_ray is not None:
        sys.modules["ray"] = old_ray
    else:
        sys.modules.pop("ray", None)
    if old_ray_exc is not None:
        sys.modules["ray.exceptions"] = old_ray_exc
    else:
        sys.modules.pop("ray.exceptions", None)


@pytest.fixture()
def tmp_output_dir(tmp_path):
    """Temporary output directory for JSONL logs."""
    d = str(tmp_path / "chia_out")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture()
def make_evaluator(mock_ray, tmp_output_dir):
    """Factory that creates a ``ChiaEvaluator`` with sensible defaults.

    Accepts keyword overrides for any constructor parameter.
    """
    from skydiscover.evaluation.chia_evaluator import ChiaEvaluator

    def _factory(**overrides):
        defaults = {
            "build_fn": MagicMock(name="build_fn", return_value="build_ref"),
            "run_fn": MagicMock(name="run_fn", side_effect=lambda workload: f"run_ref_{workload}"),
            "result_mapper_fn": lambda results: EvaluationResult(
                metrics={"combined_score": 0.95, "ipc": 1.23}
            ),
            "workloads": ["trace_a", "trace_b"],
            "output_dir": tmp_output_dir,
            "max_retries": 1,
        }
        defaults.update(overrides)
        return ChiaEvaluator(**defaults)

    return _factory


# ------------------------------------------------------------------
# Helper: synchronous asyncio.to_thread replacement
# ------------------------------------------------------------------


async def _fake_to_thread(fn, *args, **kwargs):
    """Call *fn* synchronously — replaces ``asyncio.to_thread`` in tests."""
    return fn(*args, **kwargs)


# ------------------------------------------------------------------
# CB-01: Evaluator Contract
# ------------------------------------------------------------------


class TestEvaluatorContract:
    """Verify ChiaEvaluator conforms to the evaluator contract."""

    @pytest.mark.asyncio
    async def test_evaluate_program_returns_evaluation_result(self, make_evaluator, mock_ray):
        mock_ray.get = MagicMock(side_effect=["build_ok", ["run_a", "run_b"]])
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            result = await ev.evaluate_program("int main(){}", "prog-1")

        assert isinstance(result, EvaluationResult)

    def test_evaluate_program_signature_matches_contract(self, make_evaluator):
        ev = make_evaluator()
        sig = inspect.signature(ev.evaluate_program)
        params = list(sig.parameters.keys())
        assert params == ["program_solution", "program_id"]
        assert sig.parameters["program_id"].default == ""

    @pytest.mark.asyncio
    async def test_evaluate_batch_returns_list(self, make_evaluator, mock_ray):
        mock_ray.get = MagicMock(side_effect=[
            "build_ok", ["run_a", "run_b"],
            "build_ok2", ["run_c", "run_d"],
        ])
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            results = await ev.evaluate_batch([("code1", "p1"), ("code2", "p2")])

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_init_validates_callables(self, mock_ray, tmp_output_dir):
        from skydiscover.evaluation.chia_evaluator import ChiaEvaluator

        with pytest.raises(ValueError, match="build_fn must be callable"):
            ChiaEvaluator(
                build_fn="not_callable",
                run_fn=MagicMock(),
                result_mapper_fn=MagicMock(),
                workloads=["t1"],
                output_dir=tmp_output_dir,
            )

    def test_init_validates_workloads_nonempty(self, mock_ray, tmp_output_dir):
        from skydiscover.evaluation.chia_evaluator import ChiaEvaluator

        with pytest.raises(ValueError, match="workloads must be a non-empty list"):
            ChiaEvaluator(
                build_fn=MagicMock(),
                run_fn=MagicMock(),
                result_mapper_fn=MagicMock(),
                workloads=[],
                output_dir=tmp_output_dir,
            )


# ------------------------------------------------------------------
# CB-02: Dispatch Flow
# ------------------------------------------------------------------


class TestDispatch:
    """Verify build-then-fan-out-runs dispatch flow with mock callables."""

    @pytest.mark.asyncio
    async def test_build_called_with_source_code(self, make_evaluator, mock_ray):
        mock_ray.get = MagicMock(side_effect=["build_ok", ["run_a", "run_b"]])
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            await ev.evaluate_program("source_code", "p1")

        ev.build_fn.assert_called_once_with("source_code")

    @pytest.mark.asyncio
    async def test_run_called_per_workload(self, make_evaluator, mock_ray):
        mock_ray.get = MagicMock(side_effect=["build_ok", ["run_a", "run_b"]])
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            await ev.evaluate_program("src", "p1")

        assert ev.run_fn.call_count == 2
        ev.run_fn.assert_any_call(workload="trace_a")
        ev.run_fn.assert_any_call(workload="trace_b")

    @pytest.mark.asyncio
    async def test_result_mapper_receives_run_results(self, make_evaluator, mock_ray):
        mock_ray.get = MagicMock(side_effect=["build_ok", ["run_a", "run_b"]])
        mapper = MagicMock(
            name="result_mapper_fn",
            return_value=EvaluationResult(metrics={"combined_score": 0.9}),
        )
        ev = make_evaluator(result_mapper_fn=mapper)

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            await ev.evaluate_program("src", "p1")

        mapper.assert_called_once_with(["run_a", "run_b"])

    @pytest.mark.asyncio
    async def test_build_failure_skips_runs(self, make_evaluator, mock_ray):
        """D-09: build failure -> skip all runs, return zero-score error result."""
        cause = RuntimeError("compilation failed")
        mock_ray.get = MagicMock(
            side_effect=_MockRayTaskError(cause=cause),
        )
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            result = await ev.evaluate_program("bad_code", "p1")

        # Runs never dispatched
        ev.run_fn.assert_not_called()
        # Error result
        assert result.metrics["combined_score"] == 0.0
        assert result.artifacts["failure_stage"] == "build"


# ------------------------------------------------------------------
# CB-03: Async Safety
# ------------------------------------------------------------------


class TestAsyncSafety:
    """Verify ray.get() is called via asyncio.to_thread, not directly."""

    @pytest.mark.asyncio
    async def test_ray_get_called_via_to_thread(self, make_evaluator, mock_ray):
        """asyncio.to_thread should be called for both build and run collection.

        We patch asyncio.to_thread to track calls, making it return mock
        results.  Then verify to_thread was called and that ray.get was
        NOT called directly (it should only be called inside to_thread).
        """
        to_thread_calls = []

        async def tracking_to_thread(fn, *args, **kwargs):
            to_thread_calls.append((fn, args, kwargs))
            # Return appropriate mock data based on position
            if len(to_thread_calls) == 1:
                # Build result
                return "build_ok"
            else:
                # Run results
                return ["run_a", "run_b"]

        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=tracking_to_thread):
            await ev.evaluate_program("src", "p1")

        # asyncio.to_thread was called at least twice (build + run collection)
        assert len(to_thread_calls) >= 2

        # Each call should be wrapping ray.get
        for fn, args, kwargs in to_thread_calls:
            assert fn is mock_ray.get, (
                f"asyncio.to_thread should wrap ray.get, got {fn}"
            )


# ------------------------------------------------------------------
# CB-04: Error Handling
# ------------------------------------------------------------------


class TestErrorHandling:
    """Verify error classification, retry logic, and error result details."""

    @pytest.mark.asyncio
    async def test_program_error_no_retry(self, make_evaluator, mock_ray):
        """RayTaskError -> no retry, immediate error result."""
        cause = RuntimeError("bad program")
        mock_ray.get = MagicMock(side_effect=_MockRayTaskError(cause=cause))
        ev = make_evaluator(max_retries=1)

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            result = await ev.evaluate_program("bad", "p1")

        # build_fn called exactly once (no retry)
        ev.build_fn.assert_called_once()
        assert result.metrics["error"] == 0.0
        assert result.metrics["combined_score"] == 0.0

    @pytest.mark.asyncio
    async def test_transient_error_retries_then_fails(self, make_evaluator, mock_ray):
        """WorkerCrashedError -> retry, then error result after exhaustion."""
        mock_ray.get = MagicMock(
            side_effect=_MockWorkerCrashedError("worker died"),
        )
        ev = make_evaluator(max_retries=1)

        with patch("asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await ev.evaluate_program("src", "p1")

        # build_fn called twice: original + 1 retry
        assert ev.build_fn.call_count == 2
        assert result.metrics["error"] == 0.0

    @pytest.mark.asyncio
    async def test_transient_error_succeeds_on_retry(self, make_evaluator, mock_ray):
        """WorkerCrashedError first time, success second time -> success result."""
        call_count = {"n": 0}

        def get_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _MockWorkerCrashedError("transient")
            elif call_count["n"] == 2:
                # Build succeeds on retry
                return "build_ok"
            else:
                # Run results
                return ["run_a", "run_b"]

        mock_ray.get = MagicMock(side_effect=get_side_effect)
        ev = make_evaluator(max_retries=1)

        with patch("asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await ev.evaluate_program("src", "p1")

        # Should succeed (not an error result)
        assert result.metrics.get("combined_score", 0) > 0
        assert "error" not in result.metrics or result.metrics.get("error") is None

    @pytest.mark.asyncio
    async def test_build_error_result_has_error_details(self, make_evaluator, mock_ray):
        """Program error result includes cause type and message in artifacts."""
        cause = ValueError("syntax error on line 42")
        mock_ray.get = MagicMock(side_effect=_MockRayTaskError(cause=cause))
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            result = await ev.evaluate_program("bad_code", "p1")

        assert result.artifacts["stderr"] == "syntax error on line 42"
        assert result.artifacts["error_type"] == "ValueError"
        assert result.artifacts["failure_stage"] == "build"


# ------------------------------------------------------------------
# D-13: JSONL Logging
# ------------------------------------------------------------------


class TestJSONLLogging:
    """Verify JSONL log file is written after each evaluation."""

    @pytest.mark.asyncio
    async def test_jsonl_written_after_evaluation(self, make_evaluator, mock_ray, tmp_output_dir):
        mock_ray.get = MagicMock(side_effect=["build_ok", ["run_a", "run_b"]])
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            await ev.evaluate_program("src", "p1")

        log_path = os.path.join(tmp_output_dir, "chia_eval_log.jsonl")
        assert os.path.exists(log_path)

        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "program_id" in record
        assert record["program_id"] == "p1"
        assert "timestamp" in record
        assert "mapped_result" in record

    @pytest.mark.asyncio
    async def test_jsonl_appends_multiple_evaluations(
        self, make_evaluator, mock_ray, tmp_output_dir
    ):
        mock_ray.get = MagicMock(side_effect=[
            "build_ok", ["run_a", "run_b"],
            "build_ok2", ["run_c", "run_d"],
        ])
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            await ev.evaluate_program("src1", "p1")
            await ev.evaluate_program("src2", "p2")

        log_path = os.path.join(tmp_output_dir, "chia_eval_log.jsonl")
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_jsonl_records_build_failure(self, make_evaluator, mock_ray, tmp_output_dir):
        """Build failures are still logged in JSONL."""
        cause = RuntimeError("compile error")
        mock_ray.get = MagicMock(side_effect=_MockRayTaskError(cause=cause))
        ev = make_evaluator()

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            await ev.evaluate_program("bad_code", "p1")

        log_path = os.path.join(tmp_output_dir, "chia_eval_log.jsonl")
        assert os.path.exists(log_path)

        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["program_id"] == "p1"
        # Build failure is logged with mapped_result showing error
        assert record["mapped_result"]["combined_score"] == 0.0


# ------------------------------------------------------------------
# close() behavior
# ------------------------------------------------------------------


class TestClose:
    """Verify close() is safe to call and does not raise."""

    def test_close_does_not_raise(self, make_evaluator):
        ev = make_evaluator()
        ev.close()  # Should not raise


# ------------------------------------------------------------------
# D-14: S3 soft-fail archival
# ------------------------------------------------------------------


class TestS3Upload:
    """Verify _maybe_upload_to_s3() behavior."""

    def test_returns_empty_string_when_s3_path_is_none(self, make_evaluator):
        ev = make_evaluator(s3_path=None)
        result = ev._maybe_upload_to_s3()
        assert result == ""

    def test_returns_empty_string_when_log_file_missing(self, make_evaluator, tmp_path):
        ev = make_evaluator(s3_path="s3://bucket/prefix", output_dir=str(tmp_path / "no_log"))
        # output_dir is created but _log_path file does not exist yet
        os.makedirs(str(tmp_path / "no_log"), exist_ok=True)
        ev._log_path = str(tmp_path / "no_log" / "nonexistent.jsonl")
        result = ev._maybe_upload_to_s3()
        assert result == ""

    def test_lazy_imports_boto3(self, make_evaluator, tmp_output_dir):
        ev = make_evaluator(s3_path="s3://mybucket/myprefix")
        # Create the log file so we get past the existence check
        with open(ev._log_path, "w") as f:
            f.write('{"test": true}\n')

        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            # Force re-import by patching builtins
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def fake_import(name, *args, **kwargs):
                if name == "boto3":
                    return mock_boto3
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                result = ev._maybe_upload_to_s3()

        assert result == "s3://mybucket/myprefix/chia_eval_log.jsonl"
        mock_boto3.client.assert_called_once_with("s3")
        mock_client.upload_file.assert_called_once_with(
            ev._log_path, "mybucket", "myprefix/chia_eval_log.jsonl"
        )

    def test_catches_exception_returns_empty_string(self, make_evaluator, tmp_output_dir):
        ev = make_evaluator(s3_path="s3://mybucket/prefix")
        with open(ev._log_path, "w") as f:
            f.write('{"test": true}\n')

        # Make boto3 import raise an exception
        def fake_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("no boto3")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = ev._maybe_upload_to_s3()

        assert result == ""

    def test_s3_path_without_prefix(self, make_evaluator, tmp_output_dir):
        ev = make_evaluator(s3_path="s3://mybucket")
        with open(ev._log_path, "w") as f:
            f.write('{"test": true}\n')

        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        def fake_import(name, *args, **kwargs):
            if name == "boto3":
                return mock_boto3
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = ev._maybe_upload_to_s3()

        assert result == "s3://mybucket/chia_eval_log.jsonl"
        mock_client.upload_file.assert_called_once_with(
            ev._log_path, "mybucket", "chia_eval_log.jsonl"
        )


# ------------------------------------------------------------------
# D-15: Lazy ray/boto3 imports
# ------------------------------------------------------------------


class TestLazyImports:
    """Verify ray and boto3 are NOT imported at module top level."""

    def test_ray_not_imported_at_module_top_level(self):
        """ray should only be imported inside __init__, not at module level."""
        import ast

        src_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "skydiscover", "evaluation", "chia_evaluator.py",
        )
        src_path = os.path.normpath(src_path)
        with open(src_path) as f:
            tree = ast.parse(f.read())

        # Check top-level imports (not inside class/function bodies)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("ray"), (
                        f"'import ray' found at module top level (line {node.lineno})"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("ray"):
                    raise AssertionError(
                        f"'from ray...' import at module top level (line {node.lineno})"
                    )

    def test_boto3_not_imported_at_module_top_level(self):
        """boto3 should only be imported inside _maybe_upload_to_s3, not at module level."""
        import ast

        src_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "skydiscover", "evaluation", "chia_evaluator.py",
        )
        src_path = os.path.normpath(src_path)
        with open(src_path) as f:
            tree = ast.parse(f.read())

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "boto3", (
                        f"'import boto3' found at module top level (line {node.lineno})"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "boto3":
                    raise AssertionError(
                        f"'from boto3...' import at module top level (line {node.lineno})"
                    )

    def test_module_importable_without_ray(self):
        """chia_evaluator module should be importable even without ray installed."""
        import importlib

        # Remove ray from sys.modules temporarily
        saved = {}
        for key in list(sys.modules.keys()):
            if key == "ray" or key.startswith("ray."):
                saved[key] = sys.modules.pop(key)

        # Also remove cached chia_evaluator to force re-import
        chia_key = "skydiscover.evaluation.chia_evaluator"
        saved_chia = sys.modules.pop(chia_key, None)

        try:
            # Module import should succeed (no top-level ray import)
            mod = importlib.import_module(chia_key)
            assert hasattr(mod, "ChiaEvaluator")
        finally:
            # Restore
            if saved_chia is not None:
                sys.modules[chia_key] = saved_chia
            for k, v in saved.items():
                sys.modules[k] = v
