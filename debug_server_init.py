import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)

async def test_init():
    try:
        print("Importing server...")
        from src.server import RAAServerContext

        print("Instantiating context...")
        ctx = RAAServerContext()

        print("Initializing...")
        # We need to mock the config loading if it relies on files I don't have or env vars
        # But let's try running it as is first.
        # initialize() is sync in the code I saw earlier?
        # Let's check if I changed it or if it calls async things.
        # The code I viewed earlier showed initialize() as sync.

        ctx.initialize()
        print("Initialization successful.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # If initialize is sync, we don't need asyncio run, but let's check.
    # The code I saw: def initialize(self): ...
    # But inside it calls _initialize_raa_components which I modified.

    # I'll run it directly.
    from src.server import RAAServerContext
    ctx = RAAServerContext()
    try:
        ctx.initialize()
        print("Success")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
