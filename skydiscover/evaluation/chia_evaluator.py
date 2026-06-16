"""CHIA evaluator: dispatches program evaluation to CHIA simulator nodes over Ray.

Receives injectable callables (build_fn, run_fn, result_mapper_fn) at init time,
making it simulator-agnostic.  The flow (e.g. EvolverNode) binds these to specific
node methods before constructing the evaluator.

Build is called once per evaluation, then runs are fanned out in parallel across
all configured workloads.  Results are collected via ray.get() wrapped in
asyncio.to_thread() so the skydiscover event loop is never blocked.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from skydiscover.evaluation.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class ChiaEvaluator:
    """Evaluates programs by dispatching build/run to CHIA simulator nodes over Ray.

    The evaluator is constructed directly by the flow (not via ``create_evaluator``).
    Callables are pre-bound to specific node methods by the flow, e.g.::

        evaluator = ChiaEvaluator(
            build_fn=node.build_champsim.chia_remote,
            run_fn=node.run_champsim.chia_remote,
            result_mapper_fn=my_mapper,
            workloads=["trace1.xz", "trace2.xz"],
            output_dir="/tmp/chia_eval",
        )

    Error handling distinguishes program errors (``RayTaskError``, e.g. compilation
    failure) from transient infrastructure errors (worker crash, timeout).  Program
    errors are returned immediately with no retry; transient errors get up to
    ``max_retries`` retries with backoff.
    """

    def __init__(
        self,
        build_fn: Callable[..., Any],
        run_fn: Callable[..., Any],
        result_mapper_fn: Callable[..., EvaluationResult],
        workloads: List[str],
        output_dir: str,
        *,
        s3_path: Optional[str] = None,
        timeout: float = 3600.0,
        max_retries: int = 1,
    ) -> None:
        # Validate callables
        if not callable(build_fn):
            raise ValueError("build_fn must be callable")
        if not callable(run_fn):
            raise ValueError("run_fn must be callable")
        if not callable(result_mapper_fn):
            raise ValueError("result_mapper_fn must be callable")

        # Validate workloads
        if not workloads:
            raise ValueError("workloads must be a non-empty list")

        # Lazy-import ray (D-15) -- only when ChiaEvaluator is instantiated
        import ray
        import ray.exceptions

        self._ray = ray
        self._ray_exceptions = ray.exceptions

        # Build the transient error types tuple from ray.exceptions
        self._transient_error_types: Tuple[type, ...] = (
            ray.exceptions.WorkerCrashedError,
            ray.exceptions.NodeDiedError,
            ray.exceptions.ObjectLostError,
            ray.exceptions.GetTimeoutError,
            ray.exceptions.RayActorError,
            ray.exceptions.ActorDiedError,
            ray.exceptions.ActorUnavailableError,
            ray.exceptions.TaskPlacementGroupRemoved,
        )

        self.build_fn = build_fn
        self.run_fn = run_fn
        self.result_mapper_fn = result_mapper_fn
        self.workloads = list(workloads)
        self.output_dir = output_dir
        self.s3_path = s3_path
        self.timeout = timeout
        self.max_retries = max_retries

        os.makedirs(output_dir, exist_ok=True)
        self._log_path = os.path.join(output_dir, "chia_eval_log.jsonl")

        logger.info(
            f"ChiaEvaluator ready: {len(self.workloads)} workloads,"
            f" timeout={self.timeout}s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_program(
        self,
        program_solution: str,
        program_id: str = "",
    ) -> EvaluationResult:
        """Evaluate a program by dispatching build + fan-out runs to CHIA nodes.

        Args:
            program_solution: Source code of the candidate program.
            program_id: Optional identifier for logging.
        """
        label = f" {program_id}" if program_id else ""
        start_time = time.time()

        # ----------------------------------------------------------
        # Step 1: Build (D-07, D-08)
        # ----------------------------------------------------------
        build_result = await self._dispatch_build(program_solution, label)
        if build_result is None:
            logger.error(f"Build returned None (unexpected){label}")
            error_result = EvaluationResult(
                metrics={"error": 0.0, "combined_score": 0.0},
                artifacts={
                    "failure_stage": "build",
                    "error_type": "NoneResult",
                    "stderr": "build_fn returned None",
                },
            )
            self._log_evaluation(program_id, None, None, error_result)
            return error_result

        if isinstance(build_result, EvaluationResult):
            # Build failed -- log and return immediately (D-09)
            self._log_evaluation(program_id, None, None, build_result)
            return build_result

        # ----------------------------------------------------------
        # Step 2 + 3: Fan-out runs and collect (D-07, D-11)
        # ----------------------------------------------------------
        run_results = await self._dispatch_runs(label)
        if isinstance(run_results, EvaluationResult):
            # Run collection failed -- log and return
            self._log_evaluation(program_id, build_result, None, run_results)
            return run_results

        # ----------------------------------------------------------
        # Step 4: Map results (D-12)
        # ----------------------------------------------------------
        try:
            eval_result = self.result_mapper_fn(run_results)
        except Exception as e:
            logger.error(
                f"result_mapper_fn failed{label}: {type(e).__name__}: {e}"
            )
            eval_result = EvaluationResult(
                metrics={"error": 0.0, "combined_score": 0.0},
                artifacts={
                    "failure_stage": "result_mapping",
                    "error_type": type(e).__name__,
                    "stderr": str(e),
                },
            )

        # ----------------------------------------------------------
        # Step 5: Log (D-13)
        # ----------------------------------------------------------
        self._log_evaluation(program_id, build_result, run_results, eval_result)

        # ----------------------------------------------------------
        # Step 6: Optional S3 upload (D-14)
        # ----------------------------------------------------------
        self._maybe_upload_to_s3()

        elapsed = time.time() - start_time
        logger.info(f"Evaluated program{label} in {elapsed:.2f}s")
        return eval_result

    async def evaluate_batch(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[EvaluationResult]:
        """Evaluate multiple programs sequentially to prevent actor state races.

        When build_fn and run_fn are bound to the same stateful Ray actor,
        concurrent evaluations via asyncio.gather can interleave builds and
        runs, causing Run A to use Build B's artifacts.  Serializing prevents
        this by ensuring each build-then-run sequence completes before the next
        begins.

        Args:
            programs: List of ``(solution, program_id)`` tuples.

        Returns:
            Results in the same order as *programs*.
        """
        results: List[EvaluationResult] = []
        for solution, program_id in programs:
            results.append(await self.evaluate_program(solution, program_id))
        return results

    def close(self) -> None:
        """Perform final S3 upload (if configured) and log shutdown."""
        self._maybe_upload_to_s3()
        logger.info("ChiaEvaluator closed")

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    async def _dispatch_build(
        self,
        program_solution: str,
        label: str,
    ) -> Union[Any, EvaluationResult]:
        """Dispatch the build callable and return the build result.

        Returns the raw build result on success, or an ``EvaluationResult``
        on failure (program error or exhausted retries).
        """
        for attempt in range(self.max_retries + 1):
            try:
                build_ref = self.build_fn(program_solution)
                build_result = await asyncio.to_thread(
                    self._ray.get, build_ref, timeout=self.timeout
                )
                return build_result

            except self._ray_exceptions.RayTaskError as e:
                # Program error (D-03, D-09) -- no retry
                cause = getattr(e, "cause", e)
                logger.warning(
                    f"Build failed (program error){label}:"
                    f" {type(cause).__name__}: {cause}"
                )
                return EvaluationResult(
                    metrics={"error": 0.0, "combined_score": 0.0},
                    artifacts={
                        "failure_stage": "build",
                        "error_type": type(cause).__name__,
                        "stderr": str(cause),
                    },
                )

            except (*self._transient_error_types, asyncio.TimeoutError) as e:
                # Transient / timeout -- retry with backoff (D-03)
                logger.warning(
                    f"Build attempt {attempt + 1}/{self.max_retries + 1}"
                    f" failed (transient){label}: {type(e).__name__}: {e}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2.0)

            except Exception as e:
                # Unexpected error -- treat as transient, retry
                logger.error(
                    f"Build unexpected error{label}: {type(e).__name__}: {e}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2.0)

        # All retries exhausted
        logger.error(f"Build failed after {self.max_retries + 1} attempts{label}")
        return EvaluationResult(
            metrics={"error": 0.0, "combined_score": 0.0},
            artifacts={
                "failure_stage": "build",
                "error_type": "TransientError",
                "stderr": "All retry attempts exhausted",
            },
        )

    async def _dispatch_runs(
        self,
        label: str,
    ) -> Union[List[Any], EvaluationResult]:
        """Fan out run calls for all workloads and collect results.

        Returns the list of raw run results on success, or an
        ``EvaluationResult`` on failure.
        """
        run_refs = []
        for attempt in range(self.max_retries + 1):
            try:
                # Fan-out: dispatch one run per workload (D-11)
                run_refs = [
                    self.run_fn(workload=w) for w in self.workloads
                ]

                # Collect all results (D-10)
                run_results = await asyncio.to_thread(
                    self._ray.get, run_refs, timeout=self.timeout
                )
                return list(run_results)

            except self._ray_exceptions.RayTaskError as e:
                # Program error during run -- no retry
                cause = getattr(e, "cause", e)
                logger.warning(
                    f"Run failed (program error){label}:"
                    f" {type(cause).__name__}: {cause}"
                )
                return EvaluationResult(
                    metrics={"error": 0.0, "combined_score": 0.0},
                    artifacts={
                        "failure_stage": "run",
                        "error_type": type(cause).__name__,
                        "stderr": str(cause),
                    },
                )

            except (*self._transient_error_types, asyncio.TimeoutError) as e:
                # Cancel orphaned tasks before retry
                for ref in run_refs:
                    try:
                        self._ray.cancel(ref, force=True)
                    except Exception:
                        pass
                # Transient / timeout -- retry entire fan-out
                logger.warning(
                    f"Run attempt {attempt + 1}/{self.max_retries + 1}"
                    f" failed (transient){label}: {type(e).__name__}: {e}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2.0)

            except Exception as e:
                # Cancel orphaned tasks before retry
                for ref in run_refs:
                    try:
                        self._ray.cancel(ref, force=True)
                    except Exception:
                        pass
                # Unexpected error -- treat as transient, retry
                logger.error(
                    f"Run unexpected error{label}: {type(e).__name__}: {e}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2.0)

        # All retries exhausted
        logger.error(f"Runs failed after {self.max_retries + 1} attempts{label}")
        return EvaluationResult(
            metrics={"error": 0.0, "combined_score": 0.0},
            artifacts={
                "failure_stage": "run",
                "error_type": "TransientError",
                "stderr": "All retry attempts exhausted",
            },
        )

    # ------------------------------------------------------------------
    # Logging (D-13)
    # ------------------------------------------------------------------

    def _log_evaluation(
        self,
        program_id: str,
        build_result: Any,
        run_results: Any,
        eval_result: EvaluationResult,
    ) -> None:
        """Append a JSONL record for this evaluation."""
        try:
            def _summarize_build(b: Any) -> Any:
                if b is None:
                    return None
                return {
                    "success": getattr(b, "success", None),
                    "stdout_tail": getattr(b, "stdout_tail", "")[-500:],
                    "binary_size": len(getattr(b, "binary", b"")) if getattr(b, "success", False) else 0,
                }

            def _summarize_run(r: Any) -> Any:
                return {
                    "success": getattr(r, "success", None),
                    "ipc": getattr(r, "ipc", None),
                    "returncode": getattr(r, "returncode", None),
                    "timed_out": getattr(r, "timed_out", None),
                    "wall_s": getattr(r, "wall_s", None),
                    "stdout_tail": getattr(r, "stdout_tail", "")[-500:],
                }

            record: Dict[str, Any] = {
                "timestamp": time.time(),
                "program_id": program_id,
                "build": _summarize_build(build_result),
                "runs": (
                    [_summarize_run(r) for r in run_results]
                    if run_results is not None
                    else None
                ),
                "mapped_result": eval_result.to_dict(),
            }
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
        except Exception as e:
            logger.warning(f"Failed to write evaluation log: {e}")

    # ------------------------------------------------------------------
    # S3 archival (D-14)
    # ------------------------------------------------------------------

    def _maybe_upload_to_s3(self) -> str:
        """Upload the JSONL log to S3 if configured.  Soft-fail on error.

        Returns:
            S3 URI on success, empty string on skip or failure.
        """
        if self.s3_path is None:
            return ""
        if not os.path.exists(self._log_path):
            return ""

        try:
            import boto3  # noqa: lazy import (D-14)

            # Parse s3://bucket/key-prefix from self.s3_path
            path = self.s3_path
            if path.startswith("s3://"):
                path = path[5:]
            parts = path.split("/", 1)
            bucket = parts[0]
            key_prefix = parts[1] if len(parts) > 1 else ""
            key = (
                f"{key_prefix}/chia_eval_log.jsonl"
                if key_prefix
                else "chia_eval_log.jsonl"
            )

            boto3.client("s3").upload_file(self._log_path, bucket, key)
            uri = f"s3://{bucket}/{key}"
            logger.info(f"Uploaded evaluation log to {uri}")
            return uri

        except Exception as e:
            logger.warning(
                f"S3 upload failed: {e}; continuing without archival"
            )
            return ""
