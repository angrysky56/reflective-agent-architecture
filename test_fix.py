import os
import sys

# Set PYTHONPATH to include current directory
sys.path.insert(0, os.getcwd())

try:
    from src.server import RAAServerContext, server_context

    print("Successfully imported RAAServerContext")

    # Check if server_context is instantiated
    if isinstance(server_context, RAAServerContext):
        print("server_context instantiated successfully")

    # Verify device property exists (even if it raises RuntimeError context not initialized)
    try:
        print(f"Device property access: {server_context.device}")
    except RuntimeError as e:
        print(f"Caught expected RuntimeError accessing device: {e}")
    except AttributeError as e:
        print(f"FAILED: AttributeError accessing device: {e}")
        sys.exit(1)

    print("VERIFICATION SUCCESS")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
