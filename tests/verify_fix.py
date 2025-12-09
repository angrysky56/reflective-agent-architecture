import inspect
import os
import sys

# Add repo root to path
sys.path.append(os.getcwd())

try:
    from src.director.plasticity_modulator import PlasticityModulator

    print(
        f"Successfully imported PlasticityModulator from {sys.modules['src.director.plasticity_modulator'].__file__}"
    )

    print(f"Class file: {inspect.getfile(PlasticityModulator)}")
    sig = inspect.signature(PlasticityModulator)
    print(f"Signature: {sig}")

    # Try instantiation with new params
    try:
        pm = PlasticityModulator(min_p=0.1, max_p=1.0)
        print("Instantiation with min_p=0.1, max_p=1.0 SUCCEEDED")
    except TypeError as e:
        print(f"Instantiation with min_p/max_p FAILED: {e}")

    # Check that old params fail
    try:
        pm = PlasticityModulator(p_max=0.8)
        print("Instantiation with p_max=0.8 UNEXPECTEDLY SUCCEEDED")
    except TypeError as e:
        print(f"Instantiation with p_max=0.8 FAILED (Expected): {e}")

except Exception as e:
    print(f"Import failed: {e}")
