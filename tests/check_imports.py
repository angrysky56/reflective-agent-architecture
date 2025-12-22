import importlib
import os
import pkgutil
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_matplotlib_usage():
    print("Checking for matplotlib usage in src...")

    # Walk through all modules in src
    package_path = os.path.join(os.path.dirname(__file__), "..", "src")

    for importer, modname, ispkg in pkgutil.walk_packages([package_path], prefix="src."):
        # Check if matplotlib is already loaded
        if "matplotlib" in sys.modules:
            print(f"Matplotlib loaded BEFORE importing {modname}")
            del sys.modules["matplotlib"]  # Try to reset (might be tricky)

        try:
            importlib.import_module(modname)
            if "matplotlib" in sys.modules:
                print(f"MODULE IMPORTED MATPLOTLIB: {modname}")
                # Analyze who imported it?
                # Converting this script to just import one by one and exit on find might be better.
                # But let's just list all of them.
                # Once loaded, it stays loaded.

                # To really find the culprit, we can print sys.modules['matplotlib']'s parent?
                # or just flag it.
                return
        except Exception as e:
            # print(f"Failed to import {modname}: {e}")
            pass


if __name__ == "__main__":
    check_matplotlib_usage()
