
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.compass.sandbox import SandboxProbe


def test_sandbox():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SandboxExplorer")

    probe = SandboxProbe(timeout=1.0)

    logger.info("--- Exploring SandboxProbe ---")

    # 1. Simple Code (Low Resistance)
    code_simple = "print('Hello, World!')"
    ro_simple = probe.measure_resistance(code_simple)
    logger.info(f"Code: Simple Print -> Ro: {ro_simple} (Expected: ~0.1)")

    # 2. Alien Physics (Medium/High Resistance?)
    # Use absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alien_path = os.path.join(script_dir, "alien_physics_generator.py")

    with open(alien_path, "r") as f:
        code_alien = f.read()

    # We need to modify it to actually run something, not just define functions
    # The file has an 'if __name__ == "__main__":' block, but exec() might not trigger it the same way
    # Let's append a call
    code_alien_exec = code_alien + "\n\ngenerate_sequence()"

    ro_alien = probe.measure_resistance(code_alien_exec)
    logger.info(f"Code: Alien Physics -> Ro: {ro_alien}")

    # 3. Infinite Loop (Max Resistance)
    code_loop = "while True: pass"
    ro_loop = probe.measure_resistance(code_loop)
    logger.info(f"Code: Infinite Loop -> Ro: {ro_loop} (Expected: 1.0)")

    # 4. Syntax Error (High Resistance)
    code_error = "print('Unclosed string"
    ro_error = probe.measure_resistance(code_error)
    logger.info(f"Code: Syntax Error -> Ro: {ro_error} (Expected: 0.8 or 1.0)")

if __name__ == "__main__":
    test_sandbox()
