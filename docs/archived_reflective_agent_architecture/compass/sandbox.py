from __future__ import annotations

import logging
import multiprocessing
import queue
import time


class SandboxProbe:
    """
    A lightweight sandbox for probing the 'Objective Resistance' of a task
    by attempting to execute code snippets.
    """

    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        self.logger = logging.getLogger("SandboxProbe")

    def measure_resistance(self, code_snippet: str) -> float:
        """
        Executes code to measure Objective Resistance (Ro).
        Returns: 0.0 (Easy) to 1.0 (Impossible/Crashes).
        """
        if not code_snippet or not code_snippet.strip():
            return 0.5  # Ambiguous

        # Create a queue to get the result
        result_queue: multiprocessing.Queue[tuple[bool, str | None]] = multiprocessing.Queue()

        # Create a process to run the code
        process = multiprocessing.Process(target=self._worker, args=(code_snippet, result_queue))

        try:
            start_time = time.time()
            process.start()

            # Wait for the process to finish or timeout
            process.join(self.timeout)

            if process.is_alive():
                # Timeout case: Infinite loop or too complex
                self.logger.warning("Sandbox Probe Timeout: Execution took too long.")
                process.terminate()
                process.join()
                return 1.0  # Max Resistance (Infinite Loop)

            # Check exit code
            if process.exitcode != 0:
                self.logger.warning(f"Sandbox Probe Failed: Exit code {process.exitcode}")
                return 1.0  # Max Resistance (Crash)

            # Check result from queue
            try:
                success, error = result_queue.get_nowait()
                elapsed_time = time.time() - start_time

                if success:
                    # Factor in execution time for successful runs
                    # Fast execution = low resistance, slow execution = higher resistance
                    time_ratio = elapsed_time / self.timeout
                    if time_ratio < 0.2:
                        return 0.1  # Very fast (Low Resistance)
                    elif time_ratio < 0.5:
                        return 0.2  # Moderately fast
                    elif time_ratio < 0.8:
                        return 0.3  # Slower
                    else:
                        return 0.5  # Barely completed (Medium Resistance)
                else:
                    self.logger.warning(f"Sandbox Probe Error: {error}")
                    return 0.8  # High Resistance (Runtime Error)
            except queue.Empty:
                return 1.0  # Process died without reporting

        except Exception as e:
            self.logger.error(f"Sandbox Infrastructure Error: {e}")
            return 0.5  # System error, unsure

    def _worker(
        self, code: str, result_queue: multiprocessing.Queue[tuple[bool, str | None]]
    ) -> None:
        """
        Worker function to execute the code.
        """
        try:
            # Define safe builtins (minimal)
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "range": range,
                    "len": len,
                    "int": int,
                    "float": float,
                    "str": str,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "abs": abs,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "enumerate": enumerate,
                    "zip": zip,
                    "bool": bool,
                }
            }

            # Execute the code
            # trunk-ignore(bandit/B102): Intentional for sandbox measurement
            exec(code, safe_globals)
            result_queue.put((True, None))

        except Exception as e:
            result_queue.put((False, str(e)))
