import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.server import CognitiveWorkspace, CWDConfig


def verify_search():
    print("Initializing CognitiveWorkspace...")
    config = CWDConfig()
    # Mocking components that might fail without real services if needed,
    # but _search_codebase is mostly standalone.

    # However, CognitiveWorkspace init tries to connect to Neo4j and Chroma.
    # We might need to mock __init__ or use a mock class if we can't connect.
    # Let's try to verify _search_codebase by instantiating a stripped down version or just calling it if it was static?
    # It uses self.entropy_monitor.

    # Better approach might be to just test the logic or subprocess call,
    # but let's try to instantiate and catch errors if services are missing.
    try:
        ws = CognitiveWorkspace(config)
    except Exception as e:
        print(f"Full initialization failed: {e}")
        # If full init fails (e.g. Neo4j), we might need to mock just enough to test search
        # But _search_codebase relies on entropy_monitor being initialized.
        return

    print("Testing valid search...")
    # Search for something known in this file
    res = ws._search_codebase("verify_search", ".")
    print(f"Result (truncated): {res[:200]}")
    if "verify_search" in res:
        print("SUCCESS: Found search term.")
    else:
        print("FAILURE: Did not find search term.")

    print("\nTesting invalid path...")
    res_invalid = ws._search_codebase("foo", "./non_existent_folder")
    print(f"Result: {res_invalid}")
    if "Path not found" in res_invalid:
        print("SUCCESS: Handled invalid path.")
    else:
        print("FAILURE: Did not handle invalid path correctly.")

    print("\nTesting shell injection attempt in path...")
    res_inject = ws._search_codebase("foo", "; echo h4cked")
    print(f"Result: {res_inject}")
    # Should say path not found for the weird path, or filter it out.
    if "Path not found" in res_inject or "Error" in res_inject:
        print("SUCCESS: Injection attempt thwarted (likely clean exit or cached as invalid path).")


if __name__ == "__main__":
    verify_search()
