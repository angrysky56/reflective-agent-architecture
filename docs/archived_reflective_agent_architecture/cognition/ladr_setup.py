import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("raa.ladr_setup")

REPO_URL = "https://github.com/laitep/ladr.git"


def check_dependencies() -> bool:
    """Check if git and build tools are available."""
    missing = []
    if not shutil.which("git"):
        missing.append("git")
    if not shutil.which("gcc") and not shutil.which("clang"):
        missing.append("gcc/clang")
    if not shutil.which("make"):
        missing.append("make")

    if missing:
        logger.error(f"Missing dependencies for LADR installation: {', '.join(missing)}")
        return False
    return True


def ensure_ladr(base_dir: Path) -> bool:
    """
    Ensure LADR (Prover9) is installed in the given directory.
    Returns True if successful or already present.
    """
    ladr_dir = base_dir / "ladr"
    bin_dir = ladr_dir / "bin"
    prover9_exe = bin_dir / "prover9"

    if prover9_exe.exists() and os.access(prover9_exe, os.X_OK):
        logger.info(f"LADR binaries found at {bin_dir}")
        return True

    logger.info("LADR binaries not found. Initiating auto-installation...")

    if not check_dependencies():
        logger.error("Cannot install LADR due to missing system dependencies.")
        return False

    # Clean up incomplete install if exists
    if ladr_dir.exists():
        logger.warning(f"Removing incomplete LADR directory at {ladr_dir}")
        shutil.rmtree(ladr_dir)

    try:
        # Clone
        logger.info(f"Cloning {REPO_URL} into {ladr_dir}...")
        subprocess.run(
            ["git", "clone", REPO_URL, str(ladr_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Build
        logger.info("Building LADR...")
        # standard LADR build is usually just 'make all' in the root
        subprocess.run(
            ["make", "all"],
            cwd=ladr_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if prover9_exe.exists():
            logger.info("LADR installed successfully.")
            return True
        else:
            logger.error("Build completed but prover9 binary not found.")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"LADR installation failed: {e}")
        if e.stderr:
            logger.error(f"Build Error Output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error installing LADR: {e}")
        return False
