import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Mock the context needed for _search_codebase
class MockEntropyState:
    def __init__(self, value):
        self.value = value


class MockEntropyMonitor:
    def __init__(self, state_value="explore"):
        self.state = MockEntropyState(state_value)


class CognitiveWorkspaceMock:
    def __init__(self):
        self.entropy_monitor = MockEntropyMonitor()

    def _track_tool_usage(self, name):
        pass

    # Copying the method under test to verify logic in isolation
    def _search_codebase(self, query: str, path: str = ".") -> str:
        """Search for text pattern in codebase using grep, excluding common junk."""
        self._track_tool_usage("search_codebase")
        try:
            import subprocess

            # Validation: Ensure path is safe and exists
            search_path = Path(path).resolve()
            if not search_path.exists():
                return f"Error: Path not found: {path}"

            # Security: Prevent directory traversal outside of workspace (optional, but good practice)
            # For now, we just ensure it is a directory or file.

            # State-dependent behavior
            # FOCUS: Narrow search, fewer results (Convergence)
            # EXPLORE: Broad search, more results (Divergence)
            max_results = 100  # Default
            if self.entropy_monitor.state.value == "focus":
                max_results = 20
            elif self.entropy_monitor.state.value == "explore":
                max_results = 200

            # Exclude common non-code directories
            excludes = [
                "--exclude-dir=.git",
                "--exclude-dir=__pycache__",
                "--exclude-dir=chroma_data",
                "--exclude-dir=venv",
                "--exclude-dir=node_modules",
                "--exclude-dir=.pytest_cache",
                "--exclude-dir=.mypy_cache",
                "--exclude=*.pyc",
                "--exclude=*.rdb",
                "--exclude=*.log",
            ]

            # Construction: Use list format for subprocess to avoid shell injection.
            # Use '--' to delimit options from the query, protecting against queries starting with '-'
            cmd = ["grep", "-r", "-n"] + excludes + ["--", query, str(search_path)]

            # Add timeout to prevent hanging
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0 and result.returncode != 1:
                return f"Error searching: {result.stderr}"

            output = result.stdout
            if not output:
                return "No matches found."

            # Truncate if too long
            lines = output.splitlines()
            if len(lines) > max_results:
                return (
                    "\n".join(lines[:max_results])
                    + f"\n... and {len(lines)-max_results} more matches (truncated due to {self.entropy_monitor.state.value} state)."
                )
            return output

        except subprocess.TimeoutExpired:
            return "Error: Search timed out."
        except Exception as e:
            return f"Error searching codebase: {e}"


def verify_fix_isolated():
    print("Running isolated verification for _search_codebase...")
    ws = CognitiveWorkspaceMock()

    # Test 1: Valid Path
    print("\nTest 1: Valid Path")
    # We'll stick to listing this file itself to ensure a match
    query = "verify_fix_isolated"
    path = "."
    result = ws._search_codebase(query, path)
    print(f"Result length: {len(result)}")
    if query in result:
        print("SUCCESS: Found query in valid path.")
    else:
        print(f"FAILURE: Did not find query. Output: {result}")

    # Test 2: Invalid Path
    print("\nTest 2: Invalid Path")
    path = "./non_existent_folder_xyz"
    result = ws._search_codebase(query, path)
    print(f"Result: {result}")
    if "Error: Path not found" in result:
        print("SUCCESS: Correctly handled invalid path.")
    else:
        print("FAILURE: Did not handle invalid path correctly.")

    # Test 3: Injection Attempt
    print("\nTest 3: Injection Attempt in Path")
    # The new logic tries to resolve the path.
    # If the path string looks like an injection, Path(path).resolve() will treat it as a filename.
    # It definitely won't execute it.
    path = "; echo FAIL"
    result = ws._search_codebase(query, path)
    print(f"Result: {result}")
    # Likely "Path not found" or it treats it as a file named "; echo FAIL" which fails validation.
    if "Error: Path not found" in result:
        print("SUCCESS: Injection treated as invalid path.")
    else:
        print("FAILURE: Unexpected behavior.")


if __name__ == "__main__":
    verify_fix_isolated()
