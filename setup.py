#!/usr/bin/env python3
"""
KMX Setup Script
Professional installation tool for KMX with comprehensive environment management.
"""

import os
import sys
import subprocess
import shutil
import warnings
import glob

# Show all warnings
warnings.filterwarnings('default')

# ================================
# Configuration
# ================================
ENV_NAME = "KMX-env"
PYTHON_VERSION = "3.11"  # RAPIDS requires Python 3.11
BOOST_VERSION = "1.77"
GCC_VERSION = "12"
CMAKE_MIN_VERSION = "3.13"
RAPIDS_VERSION = "25.12"
# CUDA_VERSION is auto-detected from the GPU driver at install time.
# See detect_cuda_version() below.

# Paths
GERBIL_SUBMODULE_PATH = "include/gerbil-DataFrame"
GERBIL_REPO_URL = "https://github.com/M-Serajian/gerbil-DataFrame.git"
GERBIL_BUILD_DIR = os.path.join(GERBIL_SUBMODULE_PATH, "build")
GERBIL_BINARY = os.path.join(GERBIL_BUILD_DIR, "gerbil")
GERBIL_WRAPPER = os.path.join(GERBIL_BUILD_DIR, "gerbil_wrapper.sh")

# ================================
# Colors
# ================================
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def print_colored(message, color=Colors.RESET, bold=False):
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{message}{Colors.RESET}")

def print_header(message):
    print_colored(f"\n{'='*70}", Colors.CYAN, bold=True)
    print_colored(f"  {message}", Colors.CYAN, bold=True)
    print_colored(f"{'='*70}", Colors.CYAN, bold=True)

def print_success(message):
    print_colored(f"✓ {message}", Colors.GREEN)

def print_error(message):
    print_colored(f"✗ {message}", Colors.RED, bold=True)

def print_warning(message):
    print_colored(f"⚠ {message}", Colors.YELLOW)

def print_info(message):
    print_colored(f"ℹ {message}", Colors.BLUE)

# ================================
# GPU / CUDA Driver Detection
# ================================

def detect_cuda_version():
    """Auto-detect the maximum CUDA version supported by the installed NVIDIA driver.
    
    Parses the output of nvidia-smi to get the driver's CUDA compatibility version.
    This is the MAXIMUM CUDA runtime version the driver can support.
    We use this to pin cuda-version in conda so that cuDF/cuPy install a
    compatible CUDA runtime (not a newer one the driver can't handle).
    
    Returns:
        tuple: (major, minor, full_string) e.g. (12, 8, "12.8") or None if no GPU found.
    """
    print_header("Detecting GPU and CUDA Driver Version")
    
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print_error("nvidia-smi not found!")
        print_info("NVIDIA GPU driver does not appear to be installed.")
        print_info("KMX requires an NVIDIA GPU with CUDA support for cuDF/RAPIDS.")
        print_info("If you're on an HPC cluster, you may need to load a module:")
        print_colored("  module load cuda", Colors.CYAN, bold=True)
        return None
    
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpu_info = result.stdout.strip()
        if gpu_info:
            print_success(f"GPU detected: {gpu_info}")
    except:
        pass
    
    # Parse CUDA version from nvidia-smi output
    try:
        result = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, check=True
        )
        output = result.stdout
        
        # Look for "CUDA Version: XX.Y" in the nvidia-smi output
        import re
        match = re.search(r'CUDA Version:\s+(\d+)\.(\d+)', output)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            full = f"{major}.{minor}"
            print_success(f"CUDA driver compatibility: {full}")
            print_info(f"  This means CUDA runtime up to {full} is supported by your driver.")
            return (major, minor, full)
        else:
            print_error("Could not parse CUDA version from nvidia-smi output")
            print_info("nvidia-smi output (first 5 lines):")
            for line in output.split('\n')[:5]:
                print_info(f"  {line}")
            return None
            
    except subprocess.CalledProcessError as e:
        print_error(f"nvidia-smi failed: {e}")
        return None
    except Exception as e:
        print_error(f"Error detecting CUDA version: {e}")
        return None

def get_cuda_pin_version(cuda_info):
    """Given the detected CUDA driver version, return the conda cuda-version pin.
    
    The driver's CUDA version is the MAXIMUM runtime it supports. We must pin
    cuda-version to be at most this version. For example:
      - Driver CUDA 12.8 -> pin "cuda-version>=12.0,<=12.8"
      - Driver CUDA 13.0 -> pin "cuda-version>=13.0,<=13.0"
    
    This prevents conda from installing CUDA 12.9 on a driver that only
    supports up to 12.8, which would cause cudaErrorInsufficientDriver.
    
    Args:
        cuda_info: tuple (major, minor, full_string) from detect_cuda_version()
    
    Returns:
        str: conda version spec like ">=12.0,<=12.8"
    """
    major = cuda_info[0]
    minor = cuda_info[1]
    # Pin to at most the driver's CUDA version
    return f">={major}.0,<={major}.{minor}"

# ================================
# Disk Space Estimation
# ================================

def estimate_disk_space():
    """Estimate disk space needed and check availability."""
    print_info("Estimated disk space needed:")
    print_info("  - cuDF + CUDA (RAPIDS, auto-resolved): ~5 GB")
    print_info("  - Python + dependencies: ~500 MB")
    print_info("  - Boost 1.77: ~200 MB")
    print_info(f"  - Build tools (CMake, Git, GCC {GCC_VERSION}): ~500 MB")
    print_info("  - Development libraries (zlib, bzip2): ~50 MB")
    print_info("  - gerbil source + build: ~200 MB")
    print_info("  - Package cache (temporary): ~2 GB")
    print_colored("\n  Total: ~8-9 GB\n", Colors.YELLOW, bold=True)

    check_path = os.path.expanduser("~")
    stat = shutil.disk_usage(check_path)
    free_gb = stat.free / (1024**3)
    print_info(f"Available disk space at {check_path}: {free_gb:.1f} GB")

    if free_gb < 12:
        print_warning("You have less than 12 GB free. Installation might fail!")
        response = input("Continue anyway? (y/N): ").strip().lower()
        return response == 'y'

    print_success("Sufficient disk space available")
    return True

# ================================
# Conda/Mamba Detection
# ================================
# On many HPC systems, conda and mamba are shell functions (not binaries
# on PATH). Python's shutil.which() and subprocess.run(["conda", ...])
# cannot find shell functions. We solve this by:
#   1. Checking environment variables (CONDA_EXE, MAMBA_EXE, CONDA_PREFIX)
#   2. Checking common binary locations derived from conda base
#   3. Falling back to running through "bash -l -c" (login shell) which
#      loads the user's shell profile where the functions are defined.
# All conda/mamba commands are run through _run_shell_cmd() which uses
# "bash -l -c" so shell functions always work.

def _run_shell_cmd(cmd_str, capture=True, check=False):
    """Run a command through a login shell so conda/mamba shell functions work.
    
    This is the ONLY way to reliably call conda/mamba on HPC systems where
    they are defined as shell functions in .bashrc / module loads.
    """
    shell_cmd = ["bash", "-l", "-c", cmd_str]
    try:
        if capture:
            result = subprocess.run(shell_cmd, capture_output=True, text=True, check=check)
            return result
        else:
            result = subprocess.run(shell_cmd, check=check)
            return result
    except subprocess.CalledProcessError as e:
        if capture:
            return e
        raise

def find_conda():
    """Find conda or mamba, handling both binary and shell-function installations."""
    print_header("Detecting Conda/Mamba")

    # Strategy 1: Check environment variables (most reliable)
    conda_exe_env = os.environ.get('CONDA_EXE')
    if conda_exe_env and os.path.isfile(conda_exe_env):
        # Determine if mamba is available alongside
        mamba_exe = os.path.join(os.path.dirname(conda_exe_env), "mamba")
        if os.path.isfile(mamba_exe):
            print_success(f"Found mamba via CONDA_EXE: {mamba_exe}")
            return mamba_exe, "mamba"
        print_success(f"Found conda via CONDA_EXE: {conda_exe_env}")
        return conda_exe_env, "conda"

    mamba_exe_env = os.environ.get('MAMBA_EXE')
    if mamba_exe_env and os.path.isfile(mamba_exe_env):
        print_success(f"Found mamba via MAMBA_EXE: {mamba_exe_env}")
        return mamba_exe_env, "mamba"

    # Strategy 2: shutil.which (works when conda is a real binary on PATH)
    mamba = shutil.which("mamba")
    if mamba:
        print_success(f"Found mamba on PATH: {mamba}")
        return mamba, "mamba"

    conda = shutil.which("conda")
    if conda:
        print_success(f"Found conda on PATH: {conda}")
        return conda, "conda"

    # Strategy 3: Try getting conda base through login shell (handles shell functions)
    conda_base = get_conda_base()
    if conda_base:
        # Try to find the actual binary from the base directory
        for candidate in [
            os.path.join(conda_base, "bin", "mamba"),
            os.path.join(conda_base, "bin", "conda"),
            os.path.join(conda_base, "condabin", "mamba"),
            os.path.join(conda_base, "condabin", "conda"),
        ]:
            if os.path.isfile(candidate):
                name = "mamba" if "mamba" in os.path.basename(candidate) else "conda"
                print_success(f"Found {name} via conda base: {candidate}")
                return candidate, name

        # Binary not found but shell function works - use shell mode
        # Test if mamba works as a shell function
        result = _run_shell_cmd("mamba --version")
        if result.returncode == 0:
            print_success(f"Found mamba as shell function (base: {conda_base})")
            return "mamba", "mamba"

        result = _run_shell_cmd("conda --version")
        if result.returncode == 0:
            print_success(f"Found conda as shell function (base: {conda_base})")
            return "conda", "conda"

    print_error("Neither conda nor mamba found!")
    print_info("Please install conda from: https://docs.conda.io/en/latest/miniconda.html")
    return None, None

def get_conda_base():
    """Get the conda base directory.
    
    Tries multiple strategies to handle HPC systems where conda is a shell function.
    """
    # Strategy 1: CONDA_EXE environment variable
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe and os.path.isfile(conda_exe):
        # CONDA_EXE is usually <base>/bin/conda
        base = os.path.dirname(os.path.dirname(conda_exe))
        if os.path.isdir(base):
            return base

    # Strategy 2: CONDA_PREFIX for the base environment
    # (When base env is active, CONDA_PREFIX == conda base)
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        # Check if this is the base env or a named env
        # Named envs are usually under <base>/envs/<name>
        parent = os.path.dirname(conda_prefix)
        grandparent = os.path.dirname(parent)
        if os.path.basename(parent) == 'envs' and os.path.isdir(grandparent):
            return grandparent
        # Could be base env itself
        if os.path.isdir(os.path.join(conda_prefix, "condabin")):
            return conda_prefix

    # Strategy 3: Run through login shell (picks up shell functions)
    try:
        result = _run_shell_cmd("conda info --base")
        if result.returncode == 0 and result.stdout.strip():
            base = result.stdout.strip()
            if os.path.isdir(base):
                return base
    except:
        pass

    return None

def _is_shell_function_mode(conda_exe):
    """Check if conda_exe is a bare name (shell function) vs an absolute path."""
    return '/' not in conda_exe

def run_conda_command_live(conda_exe, args):
    """Run conda command with DIRECT output - no buffering, no capturing.
    
    Handles both binary and shell-function conda/mamba.
    """
    cmd_str = f"{conda_exe} {' '.join(args)}"
    print_colored(f"\nRunning: {cmd_str}\n", Colors.BLUE)
    print_colored("=" * 70, Colors.CYAN)
    print()

    try:
        if _is_shell_function_mode(conda_exe):
            result = _run_shell_cmd(cmd_str, capture=False)
        else:
            result = subprocess.run([conda_exe] + args, check=False)
        print()
        print_colored("=" * 70, Colors.CYAN)
        return result.returncode == 0
    except Exception as e:
        print_colored("=" * 70, Colors.RED)
        print_error(f"Exception: {e}")
        return False

def run_conda_command(conda_exe, args, check=True):
    """Run a conda command (silent version for quick checks).
    
    Handles both binary and shell-function conda/mamba.
    """
    try:
        if _is_shell_function_mode(conda_exe):
            cmd_str = f"{conda_exe} {' '.join(args)}"
            result = _run_shell_cmd(cmd_str, capture=True)
            success = result.returncode == 0
            stdout = result.stdout if hasattr(result, 'stdout') else ""
            stderr = result.stderr if hasattr(result, 'stderr') else ""
            if check and not success:
                return False, stdout, stderr
            return success, stdout, stderr
        else:
            cmd = [conda_exe] + args
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)

# ================================
# Git Detection (system or conda)
# ================================

def find_git(conda_exe=None):
    """Find a working git executable - prefer system git, fall back to conda env git."""
    system_git = shutil.which("git")
    if system_git:
        try:
            result = subprocess.run([system_git, "--version"], capture_output=True, text=True, check=True)
            print_success(f"Found system git: {system_git} ({result.stdout.strip()})")
            return [system_git]
        except:
            pass

    if conda_exe and check_env_exists(conda_exe, ENV_NAME):
        try:
            result = subprocess.run(
                [conda_exe, "run", "-n", ENV_NAME, "git", "--version"],
                capture_output=True, text=True, check=True
            )
            print_success(f"Found git in conda env: {result.stdout.strip()}")
            return [conda_exe, "run", "-n", ENV_NAME, "git"]
        except:
            pass

    print_error("No working git found!")
    return None

# ================================
# Environment Management
# ================================

def check_env_exists(conda_exe, env_name):
    """Check if conda environment exists."""
    success, stdout, stderr = run_conda_command(conda_exe, ["env", "list"], check=False)
    if success:
        return env_name in stdout
    return False

def get_env_path(conda_exe, env_name):
    """Get the full path to the conda environment."""
    success, stdout, stderr = run_conda_command(conda_exe, ["env", "list"], check=False)
    if success:
        for line in stdout.split('\n'):
            if env_name in line and not line.strip().startswith('#'):
                parts = line.split()
                for part in parts:
                    if '/' in part and env_name in part:
                        return part
    return None

def create_environment(conda_exe, conda_type, cuda_info=None):
    """Create the conda environment with all dependencies.
    
    Args:
        conda_exe: path to conda/mamba executable
        conda_type: "conda" or "mamba"
        cuda_info: tuple from detect_cuda_version(), or None if no GPU detected
    """
    print_header(f"Creating Conda Environment: {ENV_NAME}")

    if check_env_exists(conda_exe, ENV_NAME):
        print_warning(f"Environment '{ENV_NAME}' already exists!")
        print_info("Choose an option:")
        print_info("  1. Keep and use existing environment")
        print_info("  2. Recreate environment (recommended if having issues)")
        response = input("Enter choice (1/2): ").strip()

        if response == '2':
            print_info(f"Recreating environment '{ENV_NAME}'...")
            delete_environment(conda_exe)
        else:
            print_info("Using existing environment")
            return True

    # Build cuda-version pin from detected driver
    cuda_pin = None
    if cuda_info:
        cuda_pin = get_cuda_pin_version(cuda_info)
        print_info(f"Pinning CUDA runtime to match your driver: cuda-version{cuda_pin}")
        print_info(f"  (Your driver supports up to CUDA {cuda_info[2]})")
    else:
        print_warning("No GPU detected - installing without CUDA version pin.")
        print_warning("cuDF may install a CUDA version incompatible with your system!")

    print_info("This may take 15-20 minutes...")
    print_info(f"Installing: cuDF (RAPIDS {RAPIDS_VERSION}), Python {PYTHON_VERSION}, "
               f"Boost {BOOST_VERSION}, GCC {GCC_VERSION}")
    if cuda_pin:
        print_info(f"CUDA runtime pinned to: cuda-version{cuda_pin}")
    print_warning("All warnings and messages will be shown (not suppressed)")

    # Pin cuda-version so cuDF installs a CUDA runtime compatible with the driver.
    # Without this, cuDF might pull CUDA 13 on a system whose driver only supports 12.x.
    packages = [
        f"python={PYTHON_VERSION}",
        f"cudf={RAPIDS_VERSION}",
        f"boost-cpp={BOOST_VERSION}",
        f"gcc={GCC_VERSION}",
        f"gxx={GCC_VERSION}",
        "cmake>=3.13",
        "git",
        "make",
        "zlib",
        "bzip2",
    ]
    
    # Add cuda-version pin if we detected the driver
    if cuda_pin:
        packages.append(f"cuda-version{cuda_pin}")

    args = [
        "create",
        "-n", ENV_NAME,
        "-c", "rapidsai",
        "-c", "conda-forge",
        "-c", "nvidia",
        "-y"
    ] + packages

    max_attempts = 2
    success = False
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print_warning(f"Attempt {attempt}/{max_attempts} - Cleaning conda cache and retrying...")
            subprocess.run([conda_exe, "clean", "--all", "-y"], check=False)

        success = run_conda_command_live(conda_exe, args)

        if success:
            break
        elif attempt < max_attempts:
            print_warning("Installation failed - will clean cache and retry")
        else:
            print_error("Installation failed after cleaning cache")

    if success:
        print_success(f"Environment '{ENV_NAME}' created successfully!")
        print_info("\nInstalled versions:")
        print_colored("=" * 70, Colors.CYAN)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "cudf"], check=False)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "cuda-toolkit"], check=False)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "boost-cpp"], check=False)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "cmake"], check=False)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "gcc"], check=False)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "gxx"], check=False)
        print_colored("=" * 70, Colors.CYAN)
        return True
    else:
        print_error("Failed to create environment!")
        print_info("\nManual fix: Run these commands then retry:")
        print_colored("  conda clean --all -y", Colors.CYAN, bold=True)
        print_colored("  rm -rf ~/.conda/pkgs/*", Colors.CYAN, bold=True)
        print_colored("  python setup.py install", Colors.CYAN, bold=True)
        return False

def delete_environment(conda_exe):
    """Delete the conda environment."""
    print_header(f"Deleting Conda Environment: {ENV_NAME}")

    if not check_env_exists(conda_exe, ENV_NAME):
        print_warning(f"Environment '{ENV_NAME}' does not exist!")
        return True

    args = ["env", "remove", "-n", ENV_NAME, "-y"]
    success, stdout, stderr = run_conda_command(conda_exe, args)

    if success:
        print_success(f"Environment '{ENV_NAME}' deleted successfully!")
        return True
    else:
        print_error("Failed to delete environment!")
        print_error(stderr)
        return False

# ================================
# Build Functions
# ================================

def clone_gerbil_submodule(conda_exe):
    """Initialize and update the gerbil-DataFrame git submodule."""
    print_header("Setting Up gerbil-DataFrame Submodule")

    include_dir = "include"
    os.makedirs(include_dir, exist_ok=True)

    if os.path.isdir(GERBIL_SUBMODULE_PATH):
        cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")
        if os.path.isfile(cmake_file):
            print_success(f"gerbil-DataFrame already exists and has source files at {GERBIL_SUBMODULE_PATH}")
            return True
        else:
            print_warning(f"gerbil-DataFrame directory exists but is EMPTY (no CMakeLists.txt)")
            print_info("Will clone fresh copy...")
            shutil.rmtree(GERBIL_SUBMODULE_PATH, ignore_errors=True)

    git_cmd = find_git(conda_exe)
    if not git_cmd:
        print_error("Cannot clone gerbil-DataFrame without git!")
        print_info("Install git with: sudo apt install git  OR  conda install git")
        return False

    print_info("Cloning gerbil-DataFrame repository...")

    # Try git submodule first if we're in a git repo
    try:
        result = subprocess.run(
            git_cmd + ["rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, check=False
        )

        if result.returncode == 0:
            print_info("Git repository detected - trying submodule commands first...")
            try:
                subprocess.run(git_cmd + ["submodule", "init"], check=True)
                subprocess.run(git_cmd + ["submodule", "update", "--recursive"], check=True)

                cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")
                if os.path.isfile(cmake_file):
                    print_success("gerbil-DataFrame submodule initialized successfully!")
                    return True
                else:
                    print_warning("Submodule init ran but directory is still empty - falling back to clone")
            except subprocess.CalledProcessError:
                print_warning("Submodule commands failed - falling back to direct clone")
    except:
        pass

    # Direct clone as fallback
    try:
        if os.path.exists(GERBIL_SUBMODULE_PATH):
            shutil.rmtree(GERBIL_SUBMODULE_PATH, ignore_errors=True)

        print_info(f"Cloning from {GERBIL_REPO_URL}...")
        clone_result = subprocess.run(
            git_cmd + ["clone", GERBIL_REPO_URL, GERBIL_SUBMODULE_PATH],
            check=True, text=True, capture_output=True
        )

        cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")
        if os.path.isfile(cmake_file):
            print_success("gerbil-DataFrame cloned successfully!")
            src_files = os.listdir(GERBIL_SUBMODULE_PATH)
            print_info(f"  Contents: {', '.join(sorted(src_files)[:10])}...")
            return True
        else:
            print_error("Clone appeared to succeed but CMakeLists.txt not found!")
            print_info(f"  Directory contents: {os.listdir(GERBIL_SUBMODULE_PATH) if os.path.isdir(GERBIL_SUBMODULE_PATH) else 'MISSING'}")
            return False

    except subprocess.CalledProcessError as e:
        print_error(f"Git clone failed!")
        print_error(f"  stdout: {e.stdout}")
        print_error(f"  stderr: {e.stderr}")
        print_info("\nManual fix:")
        print_colored(f"  git clone {GERBIL_REPO_URL} {GERBIL_SUBMODULE_PATH}", Colors.CYAN, bold=True)
        print_colored("  python setup.py install", Colors.CYAN, bold=True)
        return False

def patch_gerbil_cmake():
    """Patch gerbil-DataFrame CMakeLists.txt to fix CMake version requirement."""
    print_header("Patching gerbil-DataFrame CMakeLists.txt")

    cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")

    if not os.path.isfile(cmake_file):
        print_error(f"CMakeLists.txt not found at {cmake_file}")
        return False

    try:
        with open(cmake_file, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        if lines[0].startswith('cmake_minimum_required'):
            original_line = lines[0]
            print_info(f"Original: {original_line}")

            lines[0] = 'cmake_minimum_required(VERSION 3.13)'
            print_info(f"Patched:  {lines[0]}")

            with open(cmake_file, 'w') as f:
                f.write('\n'.join(lines))

            print_success("CMakeLists.txt patched successfully!")
            return True
        else:
            print_info("No patching needed")
            return True

    except Exception as e:
        print_error(f"Failed to patch CMakeLists.txt: {e}")
        return False

def patch_gerbil_cstdint():
    """Patch gerbil source files to add missing #include <cstdint>.

    GCC 12+ no longer transitively includes <cstdint> through other headers,
    so uint32_t, uint64_t, etc. are not available unless explicitly included.
    This patches all gerbil header and source files that use these types.
    """
    print_header("Patching gerbil Source Files for <cstdint> Compatibility")

    gerbil_source_abs = os.path.abspath(GERBIL_SUBMODULE_PATH)
    patched_count = 0
    skipped_count = 0

    # Types that require <cstdint>
    cstdint_types = ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
                     'int8_t', 'int16_t', 'int32_t', 'int64_t',
                     'size_t', 'uintptr_t', 'intptr_t']

    # Scan all .h, .hpp, .cpp, .cc files
    extensions = ('*.h', '*.hpp', '*.cpp', '*.cc', '*.cxx')
    source_files = []
    for ext in extensions:
        source_files.extend(glob.glob(os.path.join(gerbil_source_abs, '**', ext), recursive=True))

    print_info(f"Scanning {len(source_files)} source files...")

    for filepath in source_files:
        try:
            with open(filepath, 'r', errors='replace') as f:
                content = f.read()

            uses_cstdint_types = any(t in content for t in cstdint_types)
            if not uses_cstdint_types:
                continue

            if '#include <cstdint>' in content or '#include <stdint.h>' in content:
                skipped_count += 1
                continue

            lines = content.split('\n')
            insert_pos = 0
            found_include = False

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('#include'):
                    insert_pos = i + 1
                    found_include = True
                elif found_include and stripped and not stripped.startswith('//') and not stripped.startswith('#'):
                    break

            if insert_pos == 0:
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('#pragma once') or stripped.startswith('#ifndef') or stripped.startswith('#define'):
                        insert_pos = i + 1
                    elif stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*'):
                        break

            lines.insert(insert_pos, '#include <cstdint>')

            with open(filepath, 'w') as f:
                f.write('\n'.join(lines))

            rel_path = os.path.relpath(filepath, gerbil_source_abs)
            print_success(f"Patched: {rel_path} (added #include <cstdint> at line {insert_pos + 1})")
            patched_count += 1

        except Exception as e:
            rel_path = os.path.relpath(filepath, gerbil_source_abs)
            print_warning(f"Could not process {rel_path}: {e}")

    print_info(f"Summary: {patched_count} files patched, {skipped_count} already had <cstdint>")

    if patched_count > 0:
        print_success(f"Successfully patched {patched_count} files for <cstdint> compatibility")
    else:
        print_info("No files needed patching (all already include <cstdint> or don't use its types)")

    return True

def patch_run_gerbil_py():
    """Patch src/run_gerbil.py to use the wrapper script instead of the direct binary."""
    print_header("Patching run_gerbil.py to use wrapper script")

    run_gerbil_file = "src/run_gerbil.py"

    if not os.path.isfile(run_gerbil_file):
        print_error(f"{run_gerbil_file} not found")
        return False

    try:
        with open(run_gerbil_file, 'r') as f:
            content = f.read()

        if 'gerbil_wrapper.sh' in content:
            print_success("run_gerbil.py already patched to use wrapper script!")
            return True

        lines = content.split('\n')
        modified = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if 'GERBIL_EXECUTABLE' in line and '=' in line and 'gerbil' in line:
                if 'gerbil_wrapper.sh' not in line:
                    print_info(f"Original line {i+1}: {stripped}")

                    new_line = line.replace(
                        "'gerbil')", "'gerbil_wrapper.sh')"
                    ).replace(
                        '"gerbil")', '"gerbil_wrapper.sh")'
                    )

                    if new_line == line:
                        new_line = "GERBIL_EXECUTABLE = os.path.join(CURRENT_DIR, '..', 'include', 'gerbil-DataFrame', 'build', 'gerbil_wrapper.sh')"

                    lines[i] = new_line
                    print_info(f"Patched line {i+1}: {lines[i].strip()}")
                    modified = True

        if modified:
            with open(run_gerbil_file, 'w') as f:
                f.write('\n'.join(lines))

            with open(run_gerbil_file, 'r') as f:
                verify_content = f.read()
            if 'gerbil_wrapper.sh' in verify_content:
                print_success("Verified: run_gerbil.py now uses gerbil_wrapper.sh")
                return True
            else:
                print_error("Verification FAILED: wrapper reference not found after patching!")
                return False
        else:
            print_error("Could not find GERBIL_EXECUTABLE line to patch!")
            print_info("Current GERBIL_EXECUTABLE references in file:")
            for i, line in enumerate(content.split('\n'), 1):
                if 'GERBIL_EXECUTABLE' in line:
                    print_info(f"  Line {i}: {line.strip()}")
            return False

    except Exception as e:
        print_error(f"Failed to patch run_gerbil.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_gerbil_wrapper(conda_exe):
    """Create a wrapper script that runs gerbil with correct library paths."""
    print_header("Creating gerbil Wrapper Script")

    env_path = get_env_path(conda_exe, ENV_NAME)
    if not env_path:
        print_error(f"Could not find path for environment {ENV_NAME}")
        return False

    print_info(f"Environment path: {env_path}")

    gerbil_binary_abs = os.path.abspath(GERBIL_BINARY)
    wrapper_abs = os.path.abspath(GERBIL_WRAPPER)

    if not os.path.isfile(gerbil_binary_abs):
        print_error(f"Cannot create wrapper - gerbil binary does not exist at: {gerbil_binary_abs}")
        return False

    wrapper_content = f"""#!/bin/bash
# Wrapper script to run gerbil with conda environment libraries
# Generated by setup.py - do not edit manually

# Add conda lib path for runtime libraries (Boost, etc.)
export LD_LIBRARY_PATH="{env_path}/lib:$LD_LIBRARY_PATH"

# Verify gerbil binary exists
if [ ! -f "{gerbil_binary_abs}" ]; then
    echo "ERROR: gerbil binary not found at {gerbil_binary_abs}" >&2
    echo "Please run: python setup.py install" >&2
    exit 1
fi

# Run the actual gerbil binary with all arguments
exec "{gerbil_binary_abs}" "$@"
"""

    try:
        with open(wrapper_abs, 'w') as f:
            f.write(wrapper_content)

        os.chmod(wrapper_abs, 0o755)

        print_success(f"Wrapper script created: {wrapper_abs}")

        if os.access(wrapper_abs, os.X_OK):
            print_success("Wrapper is executable")
        else:
            print_error("Wrapper is not executable!")
            return False

        return True
    except Exception as e:
        print_error(f"Failed to create wrapper script: {e}")
        return False

def verify_gerbil_toolchain(env_path):
    """Verify that the conda environment has the correct versions of GCC and Boost
    needed to compile gerbil (CPU-only build).

    Returns True if all required tools are found, False otherwise.
    """
    print_header("Verifying Gerbil Build Toolchain (GCC, Boost)")

    all_ok = True

    # --- GCC (required for gerbil) ---
    gcc_path = os.path.join(env_path, "bin", "gcc")
    gxx_path = os.path.join(env_path, "bin", "g++")

    if os.path.isfile(gcc_path):
        try:
            result = subprocess.run([gcc_path, "--version"], capture_output=True, text=True, check=True)
            version_line = result.stdout.split('\n')[0]
            print_success(f"GCC: {version_line}")
            if f" {GCC_VERSION}." not in version_line and not version_line.endswith(f" {GCC_VERSION}"):
                print_warning(f"  Expected GCC {GCC_VERSION}.x but got: {version_line}")
                print_warning("  Build may use wrong GCC version!")
                all_ok = False
        except Exception as e:
            print_error(f"GCC found but failed to get version: {e}")
            all_ok = False
    else:
        print_error(f"GCC not found at {gcc_path}")
        print_info("  The gcc/gxx conda packages may not have installed properly.")
        print_info(f"  Fix: conda install -n {ENV_NAME} -c conda-forge gcc={GCC_VERSION} gxx={GCC_VERSION}")
        all_ok = False

    if os.path.isfile(gxx_path):
        print_success(f"G++ found: {gxx_path}")
    else:
        print_error(f"G++ not found at {gxx_path}")
        all_ok = False

    # --- Boost (required for gerbil) ---
    boost_include = os.path.join(env_path, "include", "boost")
    boost_lib = os.path.join(env_path, "lib")

    if os.path.isdir(boost_include):
        version_hpp = os.path.join(boost_include, "version.hpp")
        if os.path.isfile(version_hpp):
            try:
                with open(version_hpp, 'r') as f:
                    for line in f:
                        if 'BOOST_LIB_VERSION' in line and '"' in line:
                            ver = line.split('"')[1]
                            print_success(f"Boost headers found (version: {ver})")
                            break
            except:
                print_success(f"Boost headers found: {boost_include}")
        else:
            print_success(f"Boost headers found: {boost_include}")
    else:
        print_error(f"Boost headers NOT found at {boost_include}")
        print_info(f"  Fix: conda install -n {ENV_NAME} -c conda-forge boost-cpp={BOOST_VERSION}")
        all_ok = False

    boost_libs = glob.glob(os.path.join(boost_lib, "libboost_*"))
    if boost_libs:
        print_success(f"Boost libraries found: {len(boost_libs)} library files")
    else:
        print_warning("No Boost library files found - linking may fail")

    if all_ok:
        print_success("All gerbil build tools verified successfully!")
    else:
        print_error("Some build tools are missing or have wrong versions!")
        print_info("Run the install command again or fix manually with conda install.")

    return all_ok

def build_gerbil(conda_exe, force_rebuild=False):
    """Build gerbil (CPU-only) using conda environment's GCC and Boost."""
    print_header("Building gerbil-DataFrame (CPU-only)")

    cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")
    if not os.path.isfile(cmake_file):
        print_error(f"gerbil-DataFrame source not found! (no CMakeLists.txt at {cmake_file})")
        print_info("The gerbil-DataFrame submodule was not cloned properly.")
        print_info("Try running: python setup.py install")
        return False

    if os.path.isfile(GERBIL_BINARY) and not force_rebuild:
        print_success(f"gerbil binary already exists at {GERBIL_BINARY}")
        print_info("Choose an option:")
        print_info("  1. Skip build (use existing binary)")
        print_info("  2. Rebuild (recommended if compilation failed previously)")
        response = input("Enter choice (1/2): ").strip()

        if response != '2':
            print_info("Skipping build, using existing binary")
            if not create_gerbil_wrapper(conda_exe):
                print_error("Failed to create wrapper!")
                return False
            if not patch_run_gerbil_py():
                print_error("Failed to patch run_gerbil.py!")
                return False
            return True

        if os.path.isdir(GERBIL_BUILD_DIR):
            print_info("Cleaning previous build...")
            shutil.rmtree(GERBIL_BUILD_DIR)

    os.makedirs(GERBIL_BUILD_DIR, exist_ok=True)
    print_success(f"Build directory created: {GERBIL_BUILD_DIR}")

    gerbil_source_abs = os.path.abspath(GERBIL_SUBMODULE_PATH)
    build_dir_abs = os.path.abspath(GERBIL_BUILD_DIR)

    env_path = get_env_path(conda_exe, ENV_NAME)
    if not env_path:
        print_error(f"Could not find path for environment {ENV_NAME}")
        return False

    print_info(f"Using environment at: {env_path}")

    # Verify toolchain BEFORE building
    if not verify_gerbil_toolchain(env_path):
        print_error("Build toolchain verification failed!")
        print_info("Fix the issues above and retry.")
        return False

    # GCC paths from conda env (not system!)
    gcc_path = os.path.join(env_path, "bin", "gcc")
    gxx_path = os.path.join(env_path, "bin", "g++")

    # Boost paths from conda env
    boost_root = env_path
    boost_include = os.path.join(env_path, "include")
    boost_lib = os.path.join(env_path, "lib")

    try:
        print_info("Running CMake configuration (CPU-only, no CUDA)...")
        print_warning("All warnings and errors will be shown")
        print_info(f"GCC:          {gcc_path}")
        print_info(f"G++:          {gxx_path}")
        print_info(f"BOOST_ROOT:   {boost_root}")
        print_colored("=" * 70, Colors.CYAN)
        print()

        # Build environment with GCC and Boost paths from conda env.
        # All paths are absolute so we do NOT need "conda activate" here.
        # IMPORTANT: Remove system CUDA paths from PATH/env to prevent CMake
        # from finding system CUDA and trying to compile .cu files with
        # incomplete Thrust/CUDA headers.
        my_env = os.environ.copy()
        my_env['PATH'] = f"{env_path}/bin:{my_env.get('PATH', '')}"
        my_env['LD_LIBRARY_PATH'] = f"{boost_lib}:{my_env.get('LD_LIBRARY_PATH', '')}"
        my_env['BOOST_ROOT'] = boost_root
        my_env['BOOST_INCLUDEDIR'] = boost_include
        my_env['BOOST_LIBRARYDIR'] = boost_lib
        my_env['CC'] = gcc_path
        my_env['CXX'] = gxx_path
        # Prevent CMake from finding system CUDA
        my_env.pop('CUDA_HOME', None)
        my_env.pop('CUDA_ROOT', None)
        my_env.pop('CUDA_PATH', None)
        my_env.pop('CUDA_TOOLKIT_ROOT_DIR', None)

        # CMake command: GCC + Boost only, CUDA explicitly DISABLED.
        # Gerbil's CMakeLists.txt uses find_package(CUDA) which will find
        # system CUDA (e.g. /usr/local/cuda) on HPC clusters. If that CUDA
        # install lacks Thrust headers, compilation of .cu files fails.
        # We disable CUDA entirely so gerbil builds CPU-only.
        cmake_cmd = [
            "cmake",
            f"-DCMAKE_C_COMPILER={gcc_path}",
            f"-DCMAKE_CXX_COMPILER={gxx_path}",
            f"-DBOOST_ROOT={boost_root}",
            f"-DBoost_INCLUDE_DIR={boost_include}",
            f"-DBoost_LIBRARY_DIR={boost_lib}",
            "-DBoost_NO_SYSTEM_PATHS=ON",
            # Disable CUDA: prevents find_package(CUDA) from succeeding
            "-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON",
            "-DCUDA_TOOLKIT_ROOT_DIR=CUDA-NOTFOUND",
            "-S", gerbil_source_abs,
            "-B", build_dir_abs,
        ]

        # Use conda env's cmake if available, otherwise system cmake
        conda_cmake = os.path.join(env_path, "bin", "cmake")
        if os.path.isfile(conda_cmake):
            cmake_cmd[0] = conda_cmake

        subprocess.run(cmake_cmd, check=True, env=my_env)

        print()
        print_colored("=" * 70, Colors.CYAN)
        print_success("CMake configuration successful!")

        # Verify CMake picked up the right compilers
        cmake_cache = os.path.join(build_dir_abs, "CMakeCache.txt")
        if os.path.isfile(cmake_cache):
            print_info("Verifying CMake configuration...")
            with open(cmake_cache, 'r') as f:
                cache_content = f.read()

            for var_name in ['CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER',
                             'Boost_INCLUDE_DIR', 'BOOST_ROOT']:
                for line in cache_content.split('\n'):
                    if line.startswith(f'{var_name}:') or line.startswith(f'{var_name}='):
                        print_info(f"  CMakeCache: {line.strip()}")
                        break

        # Build
        cpu_count = os.cpu_count() or 4
        print_info(f"Building gerbil using {cpu_count} parallel jobs...")
        print_warning("All compilation warnings and errors will be shown")
        print_colored("=" * 70, Colors.CYAN)
        print()

        build_cmd_exe = cmake_cmd[0]  # same cmake binary
        build_cmd = [
            build_cmd_exe,
            "--build", build_dir_abs,
            f"-j{cpu_count}",
        ]

        subprocess.run(build_cmd, check=True, env=my_env)

        print()
        print_colored("=" * 70, Colors.CYAN)
        print_success("Build completed!")

        # CRITICAL VERIFICATION
        if not os.path.isfile(GERBIL_BINARY):
            print_error(f"BUILD FAILED: gerbil binary NOT found at {GERBIL_BINARY}")
            print_info("CMake/make returned success but no binary was produced.")
            print_info("Check the build output above for errors.")

            if os.path.isdir(GERBIL_BUILD_DIR):
                build_files = os.listdir(GERBIL_BUILD_DIR)
                print_info(f"  Build directory contains: {build_files}")
                for f in build_files:
                    full = os.path.join(GERBIL_BUILD_DIR, f)
                    if os.access(full, os.X_OK) and os.path.isfile(full):
                        print_warning(f"  Found executable: {f}")
            return False

        os.chmod(GERBIL_BINARY, 0o755)
        print_success(f"gerbil binary verified at: {os.path.abspath(GERBIL_BINARY)}")

        if not create_gerbil_wrapper(conda_exe):
            print_error("Failed to create wrapper script!")
            return False

        if not patch_run_gerbil_py():
            print_error("Failed to patch run_gerbil.py!")
            return False

        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Build failed with return code {e.returncode}")
        print_info("\nTo retry with a clean build:")
        print_colored("  rm -rf include/gerbil-DataFrame/build", Colors.CYAN, bold=True)
        print_colored("  python setup.py install", Colors.CYAN, bold=True)
        return False

# ================================
# Verify Installation
# ================================

def verify():
    """Verify that KMX installation is complete and all components are in place."""
    print_header("KMX Installation Verification")

    all_ok = True

    # 1. Check conda environment
    print_info("\n[1/7] Checking conda environment...")
    conda_exe, conda_type = find_conda()
    env_path = None
    if conda_exe and check_env_exists(conda_exe, ENV_NAME):
        env_path = get_env_path(conda_exe, ENV_NAME)
        print_success(f"Conda environment '{ENV_NAME}' exists at: {env_path}")
    else:
        print_error(f"Conda environment '{ENV_NAME}' NOT found!")
        print_info("  Fix: python setup.py install")
        all_ok = False

    # 2. Check gerbil build toolchain (GCC + Boost)
    print_info("\n[2/7] Checking gerbil build toolchain (GCC, Boost)...")
    if env_path:
        if not verify_gerbil_toolchain(env_path):
            all_ok = False
    else:
        print_error("Cannot check toolchain - environment not found")
        all_ok = False

    # 3. Check cuDF is installed
    print_info("\n[3/7] Checking cuDF (RAPIDS)...")
    if conda_exe and env_path:
        success, stdout, stderr = run_conda_command(conda_exe, ["list", "-n", ENV_NAME, "cudf"], check=False)
        if success and "cudf" in stdout:
            for line in stdout.split('\n'):
                if line.strip() and not line.startswith('#') and 'cudf' in line:
                    print_success(f"cuDF installed: {line.strip()}")
                    break
        else:
            print_error("cuDF NOT found in environment!")
            print_info("  Fix: python setup.py install")
            all_ok = False
    else:
        print_error("Cannot check cuDF - environment not found")
        all_ok = False

    # 4. Check gerbil source
    print_info("\n[4/7] Checking gerbil-DataFrame source...")
    cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")
    if os.path.isfile(cmake_file):
        print_success(f"gerbil-DataFrame source present")
    else:
        if os.path.isdir(GERBIL_SUBMODULE_PATH):
            contents = os.listdir(GERBIL_SUBMODULE_PATH)
            if not contents:
                print_error(f"gerbil-DataFrame directory exists but is EMPTY!")
                print_info("  The git submodule was not initialized properly.")
            else:
                print_error(f"gerbil-DataFrame directory exists but no CMakeLists.txt!")
                print_info(f"  Contents: {contents[:5]}")
        else:
            print_error(f"gerbil-DataFrame directory does not exist!")
        print_info("  Fix: python setup.py install")
        all_ok = False

    # 5. Check gerbil binary
    print_info("\n[5/7] Checking gerbil binary...")
    gerbil_binary_abs = os.path.abspath(GERBIL_BINARY)
    if os.path.isfile(gerbil_binary_abs):
        if os.access(gerbil_binary_abs, os.X_OK):
            print_success(f"gerbil binary exists and is executable: {gerbil_binary_abs}")

            cmake_cache = os.path.join(os.path.abspath(GERBIL_BUILD_DIR), "CMakeCache.txt")
            if os.path.isfile(cmake_cache):
                with open(cmake_cache, 'r') as f:
                    for line in f:
                        if line.startswith('CMAKE_CXX_COMPILER:'):
                            print_info(f"  Built with: {line.strip()}")
                            break
        else:
            print_warning(f"gerbil binary exists but is NOT executable: {gerbil_binary_abs}")
            print_info("  Fix: chmod +x " + gerbil_binary_abs)
            all_ok = False
    else:
        print_error(f"gerbil binary NOT found at: {gerbil_binary_abs}")
        if os.path.isdir(GERBIL_BUILD_DIR):
            build_files = os.listdir(GERBIL_BUILD_DIR)
            print_info(f"  Build directory exists with {len(build_files)} files")
            executables = [f for f in build_files if os.access(os.path.join(GERBIL_BUILD_DIR, f), os.X_OK) and os.path.isfile(os.path.join(GERBIL_BUILD_DIR, f))]
            if executables:
                print_info(f"  Executables found: {executables}")
        else:
            print_info("  Build directory does not exist - gerbil was never compiled")
        print_info("  Fix: rm -rf include/gerbil-DataFrame/build && python setup.py install")
        all_ok = False

    # 6. Check wrapper script and run_gerbil.py
    print_info("\n[6/7] Checking gerbil wrapper and run_gerbil.py...")
    wrapper_abs = os.path.abspath(GERBIL_WRAPPER)
    if os.path.isfile(wrapper_abs):
        if os.access(wrapper_abs, os.X_OK):
            print_success(f"Wrapper script exists and is executable")
            with open(wrapper_abs, 'r') as f:
                wrapper_content = f.read()
            if gerbil_binary_abs in wrapper_content or GERBIL_BINARY in wrapper_content:
                print_success("  Wrapper references correct gerbil binary path")
            else:
                print_warning("  Wrapper may reference wrong binary path")
        else:
            print_error(f"Wrapper script exists but is NOT executable: {wrapper_abs}")
            print_info("  Fix: chmod +x " + wrapper_abs)
            all_ok = False
    else:
        print_error(f"Wrapper script NOT found at: {wrapper_abs}")
        print_info("  Fix: python setup.py install")
        all_ok = False

    run_gerbil_file = "src/run_gerbil.py"
    if os.path.isfile(run_gerbil_file):
        with open(run_gerbil_file, 'r') as f:
            content = f.read()
        if 'gerbil_wrapper.sh' in content:
            print_success("run_gerbil.py is configured to use gerbil_wrapper.sh")
        else:
            print_error("run_gerbil.py is NOT patched - still points to raw gerbil binary!")
            print_info("  This WILL cause FileNotFoundError or GLIBC errors at runtime.")
            for i, line in enumerate(content.split('\n'), 1):
                if 'GERBIL_EXECUTABLE' in line and '=' in line:
                    print_info(f"  Line {i}: {line.strip()}")
            print_info("  Fix: python setup.py install")
            all_ok = False
    else:
        print_error(f"{run_gerbil_file} not found!")
        all_ok = False

    # 7. Check KMX.py
    print_info("\n[7/7] Checking KMX.py...")
    if os.path.isfile("KMX.py"):
        print_success("KMX.py exists")
    else:
        print_error("KMX.py not found!")
        all_ok = False

    # Summary
    print_header("Verification Summary")
    if all_ok:
        print_colored("\n  ✓ ALL CHECKS PASSED - KMX is ready to use!\n", Colors.GREEN, bold=True)
        print_info("  Usage:")
        print_colored("    conda activate KMX-env", Colors.CYAN, bold=True)
        print_colored("    python KMX.py -h", Colors.CYAN, bold=True)
        return 0
    else:
        print_colored("\n  ✗ SOME CHECKS FAILED - see above for fixes\n", Colors.RED, bold=True)
        print_info("  Most issues can be fixed by running:")
        print_colored("    python setup.py install", Colors.CYAN, bold=True)
        return 1

# ================================
# Main Functions
# ================================

def install():
    """Install: Create conda environment AND build gerbil in one command."""
    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  KMX Installation Tool", Colors.CYAN, bold=True)
    print_colored("  Professional Setup for Genomic K-mer Analysis", Colors.CYAN, bold=True)
    print_colored("="*70 + "\n", Colors.CYAN, bold=True)

    print_warning("Installation steps:")
    print_info("  1. Detect GPU and CUDA driver version")
    print_info("  2. Create conda environment (KMX-env)")
    print_info(f"  3. Install cuDF (RAPIDS {RAPIDS_VERSION}), Python {PYTHON_VERSION}, "
               f"Boost {BOOST_VERSION}, GCC {GCC_VERSION}")
    print_info("     (CUDA runtime pinned to match your GPU driver)")
    print_info("  4. Clone/initialize gerbil-DataFrame submodule")
    print_info("  5. Patch CMakeLists.txt for compatibility")
    print_info("  6. Patch source files for <cstdint> compatibility (GCC 12+)")
    print_info("  7. Verify build toolchain (GCC, Boost)")
    print_info("  8. Configure and build gerbil (CPU-only, using conda GCC + Boost)")
    print_info("  9. Create wrapper script and patch run_gerbil.py")
    print_warning("⏱  Total estimated time: 15-20 minutes")
    print()

    if not estimate_disk_space():
        print_warning("Installation cancelled.")
        return 1

    # Detect GPU and CUDA driver version FIRST
    cuda_info = detect_cuda_version()
    if not cuda_info:
        print_warning("No NVIDIA GPU or driver detected!")
        print_warning("KMX requires an NVIDIA GPU for cuDF/RAPIDS.")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print_warning("Installation cancelled.")
            return 1

    conda_exe, conda_type = find_conda()
    if not conda_exe:
        return 1

    if not create_environment(conda_exe, conda_type, cuda_info=cuda_info):
        print_error("Environment creation failed!")
        return 1

    if not clone_gerbil_submodule(conda_exe):
        print_error("Failed to setup gerbil-DataFrame!")
        print_error("This is a critical step - cannot continue without gerbil source code.")
        return 1

    if not patch_gerbil_cmake():
        print_error("Failed to patch CMakeLists.txt!")
        return 1

    if not patch_gerbil_cstdint():
        print_warning("Failed to patch <cstdint> includes - build may fail with GCC 12+")

    if not build_gerbil(conda_exe):
        print_error("Build failed!")
        return 1

    # FINAL VERIFICATION
    print_header("Final Verification")

    critical_files = {
        "gerbil binary": os.path.abspath(GERBIL_BINARY),
        "gerbil wrapper": os.path.abspath(GERBIL_WRAPPER),
    }

    all_ok = True
    for name, path in critical_files.items():
        if os.path.isfile(path):
            print_success(f"{name}: {path}")
        else:
            print_error(f"{name} MISSING: {path}")
            all_ok = False

    with open("src/run_gerbil.py", 'r') as f:
        if 'gerbil_wrapper.sh' in f.read():
            print_success("run_gerbil.py correctly references gerbil_wrapper.sh")
        else:
            print_error("run_gerbil.py does NOT reference gerbil_wrapper.sh!")
            all_ok = False

    cmake_cache = os.path.join(os.path.abspath(GERBIL_BUILD_DIR), "CMakeCache.txt")
    if os.path.isfile(cmake_cache):
        with open(cmake_cache, 'r') as f:
            cache_content = f.read()
        for var in ['CMAKE_CXX_COMPILER', 'BOOST_ROOT']:
            for line in cache_content.split('\n'):
                if line.startswith(f'{var}:') or line.startswith(f'{var}='):
                    print_info(f"  {line.strip()}")
                    break

    if not all_ok:
        print_error("\nInstallation completed but verification found issues!")
        print_info("Run: python setup.py verify   for detailed diagnostics")
        return 1

    # SUCCESS
    print_colored("\n" + "="*70, Colors.GREEN, bold=True)
    print_colored("  ✓ Installation Complete!", Colors.GREEN, bold=True)
    print_colored("="*70, Colors.GREEN, bold=True)

    print_colored("\n" + "="*70, Colors.YELLOW, bold=True)
    print_colored("  ⚠ IMPORTANT: SYSTEM REQUIREMENTS", Colors.YELLOW, bold=True)
    print_colored("="*70, Colors.YELLOW, bold=True)
    print_warning("   GPU: NVIDIA GPU with CUDA support (required by cuDF/RAPIDS)")
    print_warning("   RAM and VRAM: Depends on dataset size")
    print_warning("   Disk: At least 10 GB free for temporary files")

    print_colored("\n" + "="*70, Colors.YELLOW, bold=True)
    print_colored("  MAINTENANCE COMMANDS", Colors.YELLOW, bold=True)
    print_colored("="*70, Colors.YELLOW, bold=True)

    print_colored("\nVERIFY INSTALLATION:", Colors.CYAN, bold=True)
    print_colored("  python setup.py verify", Colors.GREEN)

    print_colored("\nREBUILD GERBIL (if compilation had errors):", Colors.CYAN, bold=True)
    print_colored("  rm -rf include/gerbil-DataFrame/build", Colors.GREEN)
    print_colored("  python setup.py install", Colors.GREEN)

    print_colored("\nRECREATE ENVIRONMENT (if dependency issues):", Colors.CYAN, bold=True)
    print_colored("  python setup.py uninstall", Colors.GREEN)
    print_colored("  python setup.py install", Colors.GREEN)

    print_colored("\nCOMPLETE UNINSTALLATION:", Colors.CYAN, bold=True)
    print_colored("  python setup.py uninstall", Colors.GREEN)

    print_colored("\nCLEAN CONDA CACHE (free disk space):", Colors.CYAN, bold=True)
    print_colored("  conda clean --all -y", Colors.GREEN)

    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  Installation Details", Colors.CYAN, bold=True)
    print_colored("="*70, Colors.CYAN, bold=True)
    print_colored(f"\nEnvironment name:   {ENV_NAME}", Colors.RESET)
    print_colored(f"gerbil binary:      {os.path.abspath(GERBIL_BINARY)}", Colors.RESET)
    print_colored(f"gerbil wrapper:     {os.path.abspath(GERBIL_WRAPPER)}", Colors.RESET)
    print_colored(f"gerbil built with:  GCC {GCC_VERSION}, Boost {BOOST_VERSION} (CPU-only)", Colors.RESET)
    cuda_str = f"CUDA {cuda_info[2]}" if cuda_info else "CUDA auto-resolved"
    print_colored(f"Python packages:    cuDF {RAPIDS_VERSION}, Python {PYTHON_VERSION} ({cuda_str})", Colors.RESET)

    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  HOW TO USE KMX", Colors.CYAN, bold=True)
    print_colored("="*70, Colors.CYAN, bold=True)

    print_colored("\nStep 1: ACTIVATE THE KMX ENVIRONMENT", Colors.GREEN, bold=True)
    print_colored("   conda activate KMX-env", Colors.CYAN, bold=True)
    print_warning("   ⚠ CRITICAL: You MUST activate KMX-env before running KMX!")

    print_colored("\nStep 2: VIEW USAGE INSTRUCTIONS", Colors.GREEN, bold=True)
    print_colored("   python KMX.py -h", Colors.CYAN, bold=True)

    print_colored("\nStep 3: RUN KMX", Colors.GREEN, bold=True)
    print_colored("   python KMX.py -l genomes.txt -k 31 -o output/ -t tmp/", Colors.CYAN, bold=True)

    print()
    return 0

def uninstall():
    """Uninstall: Delete conda environment and built files."""
    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  KMX Uninstallation", Colors.CYAN, bold=True)
    print_colored("="*70 + "\n", Colors.CYAN, bold=True)

    print_warning("This will remove:")
    print_info("  1. Conda environment (KMX-env)")
    print_info("  2. gerbil-DataFrame directory and compiled binary")
    print()

    response = input("Continue with uninstallation? (y/N): ").strip().lower()
    if response != 'y':
        print_info("Uninstallation cancelled")
        return 0

    conda_exe, conda_type = find_conda()
    if not conda_exe:
        return 1

    if not delete_environment(conda_exe):
        print_warning("Environment deletion had issues, continuing...")

    print_info("Cleaning up gerbil-DataFrame...")

    try:
        git_cmd = find_git(conda_exe)
        if git_cmd:
            result = subprocess.run(
                git_cmd + ["rev-parse", "--is-inside-work-tree"],
                capture_output=True, check=False
            )

            if result.returncode == 0:
                result = subprocess.run(
                    git_cmd + ["config", "--file", ".gitmodules", "--get-regexp", "path"],
                    capture_output=True, text=True, check=False
                )

                if GERBIL_SUBMODULE_PATH in result.stdout:
                    print_info("Removing git submodule properly...")
                    subprocess.run(git_cmd + ["submodule", "deinit", "-f", GERBIL_SUBMODULE_PATH], check=False)
                    subprocess.run(git_cmd + ["rm", "-f", GERBIL_SUBMODULE_PATH], check=False)

                    modules_path = os.path.join(".git", "modules", GERBIL_SUBMODULE_PATH)
                    if os.path.exists(modules_path):
                        shutil.rmtree(modules_path, ignore_errors=True)

                    subprocess.run(git_cmd + ["checkout", ".gitmodules"], check=False)
                    print_success("Git submodule removed")
    except:
        pass

    if os.path.isdir(GERBIL_SUBMODULE_PATH):
        print_info(f"Removing directory: {GERBIL_SUBMODULE_PATH}")
        try:
            shutil.rmtree(GERBIL_SUBMODULE_PATH)
            print_success("gerbil-DataFrame directory removed")
        except Exception as e:
            print_error(f"Failed to remove directory: {e}")
            print_info("You can manually remove it with:")
            print_colored(f"  rm -rf {GERBIL_SUBMODULE_PATH}", Colors.CYAN, bold=True)

    # Restore run_gerbil.py to original (unpatch)
    run_gerbil_file = "src/run_gerbil.py"
    if os.path.isfile(run_gerbil_file):
        try:
            with open(run_gerbil_file, 'r') as f:
                content = f.read()
            if 'gerbil_wrapper.sh' in content:
                content = content.replace('gerbil_wrapper.sh', 'gerbil')
                with open(run_gerbil_file, 'w') as f:
                    f.write(content)
                print_success("run_gerbil.py restored to original (unpatched)")
        except:
            pass

    print_colored("\n" + "="*70, Colors.GREEN, bold=True)
    print_colored("  ✓ Uninstallation Complete!", Colors.GREEN, bold=True)
    print_colored("="*70, Colors.GREEN, bold=True)

    print_colored("\nADDITIONAL CLEANUP (OPTIONAL):", Colors.YELLOW, bold=True)
    print_info("To free up disk space from conda cache:")
    print_colored("  conda clean --all -y", Colors.CYAN, bold=True)
    print()

    return 0

def show_usage():
    """Show usage information."""
    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  KMX Setup Tool - Professional Installation Manager", Colors.CYAN, bold=True)
    print_colored("="*70, Colors.CYAN, bold=True)

    print_colored("\nUSAGE:", Colors.YELLOW, bold=True)
    print_colored("  python setup.py install    ", Colors.GREEN, end="")
    print_colored("- Complete installation (environment + build)", Colors.RESET)
    print_colored("  python setup.py uninstall  ", Colors.GREEN, end="")
    print_colored("- Remove environment and files", Colors.RESET)
    print_colored("  python setup.py verify     ", Colors.GREEN, end="")
    print_colored("- Check installation health", Colors.RESET)
    print()

# ================================
# Main Entry Point
# ================================

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_usage()
        return 1

    command = sys.argv[1].lower()

    if command == "install":
        return install()
    elif command == "uninstall":
        return uninstall()
    elif command == "verify":
        return verify()
    else:
        print_error(f"Unknown command: {command}")
        show_usage()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_colored("\n\nInterrupted by user", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)