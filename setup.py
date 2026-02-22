#!/usr/bin/env python3
"""
KMX Setup Script
Professional installation tool for KMX conda environment management.
Creates the KMX-env with all required Python/CUDA packages.
"""

import os
import sys
import subprocess
import shutil
import warnings
import re

warnings.filterwarnings('default')

# ================================
# Configuration
# ================================
ENV_NAME            = "KMX-env"
PYTHON_VERSION      = "3.11.14"   # Exact version confirmed working
BOOST_VERSION       = "1.77.0"    # Exact version confirmed working
GCC_VERSION         = "12.2.0"    # Exact version confirmed working with CUDA 12.8
RAPIDS_VERSION      = "25.06"     # Latest RAPIDS confirmed working with CUDA 12.8
CUPY_VERSION        = "13.*"      # CuPy 14+ pulls CUDA 12.9 which breaks sm_89 kernels
CUDA_VERSION        = "12.8"      # Exact CUDA runtime version installed into conda env
CUDA_MIN_DRIVER     = (525, 60)   # Minimum NVIDIA driver version supporting CUDA 12.8
ZLIB_VERSION        = "1.3.1"     # Exact version confirmed working
BZIP2_VERSION       = "1.0.8"     # Exact version confirmed working
CMAKE_MIN_VERSION   = "3.5"       # Minimum version written into CMakeLists.txt patch
CMAKE_CONDA_VERSION = "4.2.3"     # Exact cmake version confirmed working

# ================================
# Colors
# ================================
class Colors:
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    RESET   = "\033[0m"
    BOLD    = "\033[1m"

def print_colored(message, color=Colors.RESET, bold=False):
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{message}{Colors.RESET}")

def print_header(message):
    print_colored(f"\n{'='*70}", Colors.CYAN, bold=True)
    print_colored(f"  {message}", Colors.CYAN, bold=True)
    print_colored(f"{'='*70}", Colors.CYAN, bold=True)

def print_success(message): print_colored(f"✓ {message}", Colors.GREEN)
def print_error(message):   print_colored(f"✗ {message}", Colors.RED, bold=True)
def print_warning(message): print_colored(f"⚠ {message}", Colors.YELLOW)
def print_info(message):    print_colored(f"ℹ {message}", Colors.BLUE)

# ================================
# NVIDIA Driver Detection
# ================================

def detect_nvidia_driver():
    """Verify the NVIDIA driver is installed and supports CUDA 12.8 or higher.

    This function checks ONLY the host NVIDIA driver version via nvidia-smi.
    The CUDA runtime (12.8) is installed automatically into the conda environment
    and does NOT need to be present on the host system.

    What this checks:
      - NVIDIA driver is installed and accessible
      - Driver version is >= 525.60 (minimum required to run CUDA 12.8 runtime)

    What this does NOT check (and does not need to):
      - System-level CUDA toolkit installation
      - CUDA_HOME / CUDA_PATH environment variables
      - nvcc availability

    NVIDIA drivers are backward compatible:
      - Driver 525.60 supports CUDA runtime <= 12.8
      - Driver 550.xx supports CUDA runtime <= 12.4  (still fine for 12.8 in env)
      - Any driver >= 525.60 can run the CUDA 12.8 conda runtime

    Returns:
        tuple: (major, minor, full_string) of the driver version e.g. (525, 60, "525.60")
               or None if the NVIDIA driver is not found or too old.
    """
    print_header("Detecting NVIDIA Driver")
    print_colored("\n  What is being checked:", Colors.CYAN)
    print_info("  • NVIDIA Driver  (host OS level) – must be >= 525.60")
    print_info(f"  • CUDA Runtime   (conda level)  – will be installed automatically as {CUDA_VERSION}")
    print_colored("  The driver must exist on the host; the runtime is handled by conda.\n", Colors.YELLOW)

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print_error("nvidia-smi not found!")
        print_error("The NVIDIA GPU driver does not appear to be installed.")
        print_colored("\n  To fix:", Colors.YELLOW, bold=True)
        print_info("  • On a workstation/server: install the NVIDIA driver from")
        print_info("    https://www.nvidia.com/Download/index.aspx")
        print_info(f"  • Minimum required driver version: {CUDA_MIN_DRIVER[0]}.{CUDA_MIN_DRIVER[1]}")
        print_info("  • On an HPC cluster: contact your sysadmin or check available GPU nodes")
        return None

    # Query GPU name
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpu_info = result.stdout.strip()
        if gpu_info:
            print_success(f"GPU detected: {gpu_info}")
    except Exception:
        pass

    # Parse driver version from nvidia-smi header
    try:
        result = subprocess.run([nvidia_smi], capture_output=True, text=True, check=True)

        # Parse driver version (e.g. "Driver Version: 525.105.17")
        driver_match = re.search(r'Driver Version:\s+(\d+)\.(\d+)', result.stdout)
        if not driver_match:
            print_error("Could not parse NVIDIA driver version from nvidia-smi output.")
            print_warning("Please verify your NVIDIA driver installation.")
            return None

        drv_major = int(driver_match.group(1))
        drv_minor = int(driver_match.group(2))
        drv_full  = f"{drv_major}.{drv_minor}"
        print_success(f"NVIDIA Driver version: {drv_full}")

        # Also show what CUDA version this driver supports (informational only)
        cuda_match = re.search(r'CUDA Version:\s+(\d+\.\d+)', result.stdout)
        if cuda_match:
            print_info(f"  This driver supports CUDA runtime up to: {cuda_match.group(1)}")
        print_info(f"  conda environment will use CUDA runtime: {CUDA_VERSION}")

        # Check minimum driver version
        min_major, min_minor = CUDA_MIN_DRIVER
        if (drv_major, drv_minor) < (min_major, min_minor):
            print_error(f"NVIDIA DRIVER TOO OLD: {drv_full}")
            print_error(f"KMX requires driver >= {min_major}.{min_minor} to run CUDA {CUDA_VERSION} runtime.")
            print_colored("\n  To fix:", Colors.YELLOW, bold=True)
            print_info(f"  • Update your NVIDIA driver to >= {min_major}.{min_minor}")
            print_info("  • https://www.nvidia.com/Download/index.aspx")
            raise RuntimeError(
                f"NVIDIA driver {drv_full} is below minimum required {min_major}.{min_minor}"
            )

        print_success(
            f"Driver {drv_full} >= {min_major}.{min_minor} – "
            f"compatible with CUDA {CUDA_VERSION} runtime ✓"
        )
        return (drv_major, drv_minor, drv_full)

    except subprocess.CalledProcessError as e:
        print_error(f"nvidia-smi failed: {e}")
        print_warning("Cannot determine driver version. Do you have GPU access on this node?")
        return None
    except RuntimeError:
        raise
    except Exception as e:
        print_warning(f"Error parsing nvidia-smi output: {e}")
        return None


def get_cuda_pin_version(driver_info):
    """Return the fixed conda cuda-version pin string.

    Always returns CUDA_VERSION (12.8) regardless of the driver version.
    This ensures the conda environment is always isolated to the tested
    CUDA version, even if the system driver supports a higher CUDA version.
    NVIDIA drivers are backward compatible — a newer driver can run
    an older CUDA runtime without any issues.
    """
    # Always pin to the tested/confirmed CUDA version, NOT derived from the driver
    major, minor = CUDA_VERSION.split(".")
    return f">={major}.0,<={major}.{minor}"

# ================================
# Disk Space Check
# ================================

def estimate_disk_space():
    """Estimate and verify available disk space."""
    print_info("Estimated disk space needed:")
    print_info("  - cuDF + CUDA runtime (RAPIDS, auto-resolved): ~5 GB")
    print_info("  - Python + dependencies:                       ~500 MB")
    print_info(f"  - Boost {BOOST_VERSION}:                              ~200 MB")
    print_info(f"  - Build tools (CMake, Git, GCC {GCC_VERSION}):       ~500 MB")
    print_info("  - Development libraries (zlib, bzip2):         ~50 MB")
    print_info("  - Package cache (temporary):                   ~2 GB")
    print_colored("\n  Total: ~8-9 GB\n", Colors.YELLOW, bold=True)

    check_path = os.path.expanduser("~")
    stat = shutil.disk_usage(check_path)
    free_gb = stat.free / (1024 ** 3)
    print_info(f"Available disk space at {check_path}: {free_gb:.1f} GB")

    if free_gb < 12:
        print_warning("You have less than 12 GB free. Installation might fail!")
        response = input("Continue anyway? (y/N): ").strip().lower()
        return response == 'y'

    print_success("Sufficient disk space available")
    return True

# ================================
# Conda / Mamba Detection
# ================================
# On HPC systems conda/mamba are often shell functions, not PATH binaries.
# We handle this by:
#   1. Checking CONDA_EXE / MAMBA_EXE environment variables
#   2. shutil.which()
#   3. Falling back to "bash -l -c" (login shell) which loads .bashrc

def _run_shell_cmd(cmd_str, capture=True, check=False):
    """Run a command through a login shell so conda/mamba shell functions work."""
    shell_cmd = ["bash", "-l", "-c", cmd_str]
    try:
        if capture:
            return subprocess.run(shell_cmd, capture_output=True, text=True, check=check)
        else:
            return subprocess.run(shell_cmd, check=check)
    except subprocess.CalledProcessError as e:
        if capture:
            return e
        raise


def find_conda():
    """Find conda or mamba, handling both binary and shell-function installations."""
    print_header("Detecting Conda/Mamba")

    # 1 – environment variables (most reliable on HPC)
    conda_exe_env = os.environ.get('CONDA_EXE')
    if conda_exe_env and os.path.isfile(conda_exe_env):
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

    # 2 – PATH
    for name in ("mamba", "conda"):
        path = shutil.which(name)
        if path:
            print_success(f"Found {name} on PATH: {path}")
            return path, name

    # 3 – derive from conda base directory
    conda_base = get_conda_base()
    if conda_base:
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

        # Last resort: shell function
        for name in ("mamba", "conda"):
            result = _run_shell_cmd(f"{name} --version")
            if result.returncode == 0:
                print_success(f"Found {name} as shell function (base: {conda_base})")
                return name, name

    print_error("Neither conda nor mamba found!")
    print_info("Please install conda: https://docs.conda.io/en/latest/miniconda.html")
    return None, None


def get_conda_base():
    """Determine the conda base directory via env vars or login shell."""
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe and os.path.isfile(conda_exe):
        base = os.path.dirname(os.path.dirname(conda_exe))
        if os.path.isdir(base):
            return base

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        parent = os.path.dirname(conda_prefix)
        grandparent = os.path.dirname(parent)
        if os.path.basename(parent) == 'envs' and os.path.isdir(grandparent):
            return grandparent
        if os.path.isdir(os.path.join(conda_prefix, "condabin")):
            return conda_prefix

    try:
        result = _run_shell_cmd("conda info --base")
        if result.returncode == 0 and result.stdout.strip():
            base = result.stdout.strip()
            if os.path.isdir(base):
                return base
    except Exception:
        pass

    return None


def _is_shell_function_mode(conda_exe):
    return '/' not in conda_exe


def run_conda_command_live(conda_exe, args):
    """Run a conda command with live (unbuffered) output."""
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
    """Run a conda command silently and return (success, stdout, stderr)."""
    try:
        if _is_shell_function_mode(conda_exe):
            cmd_str = f"{conda_exe} {' '.join(args)}"
            result = _run_shell_cmd(cmd_str, capture=True)
            success = result.returncode == 0
            stdout = getattr(result, 'stdout', "")
            stderr = getattr(result, 'stderr', "")
            return success, stdout, stderr
        else:
            result = subprocess.run([conda_exe] + args, check=check,
                                    capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)

# ================================
# Environment Management
# ================================

def check_env_exists(conda_exe, env_name):
    success, stdout, _ = run_conda_command(conda_exe, ["env", "list"], check=False)
    return success and env_name in stdout


def get_env_path(conda_exe, env_name):
    success, stdout, _ = run_conda_command(conda_exe, ["env", "list"], check=False)
    if success:
        for line in stdout.split('\n'):
            if env_name in line and not line.strip().startswith('#'):
                for part in line.split():
                    if '/' in part and env_name in part:
                        return part
    return None


def create_environment(conda_exe, conda_type, driver_info=None):
    """Create the KMX-env conda environment with all required packages."""
    print_header(f"Creating Conda Environment: {ENV_NAME}")

    if check_env_exists(conda_exe, ENV_NAME):
        print_warning(f"Environment '{ENV_NAME}' already exists!")
        print_info("  1. Keep and use existing environment")
        print_info("  2. Recreate environment (recommended if having issues)")
        response = input("Enter choice (1/2): ").strip()
        if response == '2':
            print_info(f"Recreating environment '{ENV_NAME}'...")
            delete_environment(conda_exe)
        else:
            print_info("Using existing environment")
            return True

    # Build CUDA version pin
    cuda_pin = None
    if driver_info:
        cuda_pin = get_cuda_pin_version(driver_info)
        print_info(f"Pinning CUDA runtime to: cuda-version{cuda_pin}")
        print_info(f"  (Driver {driver_info[2]} confirmed compatible)")
    else:
        print_warning("No GPU driver detected – installing without CUDA version pin.")
        print_warning("cuDF may install a CUDA runtime incompatible with your system!")

    print_info("This may take 15-20 minutes...")
    print_info(f"Installing: cuDF {RAPIDS_VERSION}, Python {PYTHON_VERSION}, "
               f"GCC {GCC_VERSION}, Boost {BOOST_VERSION}, CMake >={CMAKE_CONDA_VERSION}, "
               f"Git, Make, zlib, bzip2")
    print_info(f"  CUDA runtime {CUDA_VERSION} will be pulled in automatically as a dependency")
    if cuda_pin:
        print_info(f"  CUDA runtime pinned to: cuda-version{cuda_pin}")
    print_warning("All warnings and messages will be shown (not suppressed)")

    # ── Package list ──────────────────────────────────────────────────────────
    packages = [
        f"python={PYTHON_VERSION}",
        f"cudf={RAPIDS_VERSION}",
        f"gcc_linux-64={GCC_VERSION}",    # Exact version confirmed working
        f"gxx_linux-64={GCC_VERSION}",    # Matching G++
        f"boost-cpp={BOOST_VERSION}",
        f"cmake={CMAKE_CONDA_VERSION}",   # Exact cmake version confirmed working
        "git",
        "make",
        f"cupy={CUPY_VERSION}",
        f"zlib={ZLIB_VERSION}",
        f"bzip2={BZIP2_VERSION}",
    ]
    if cuda_pin:
        packages.append(f"cuda-version{cuda_pin}")
    # ─────────────────────────────────────────────────────────────────────────

    args = [
        "create", "-n", ENV_NAME,
        "-c", "rapidsai",
        "-c", "conda-forge",
        "-c", "nvidia",
        "-y",
    ] + packages

    max_attempts = 2
    success = False
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print_warning(f"Attempt {attempt}/{max_attempts} – cleaning conda cache and retrying...")
            subprocess.run([conda_exe, "clean", "--all", "-y"], check=False)
        success = run_conda_command_live(conda_exe, args)
        if success:
            break
        elif attempt < max_attempts:
            print_warning("Installation failed – will clean cache and retry")
        else:
            print_error("Installation failed after cleaning cache")

    if success:
        print_success(f"Environment '{ENV_NAME}' created successfully!")
        print_info("\nInstalled package versions:")
        print_colored("=" * 70, Colors.CYAN)
        for pkg in ["cudf", "cuda-toolkit", "boost-cpp", "cmake",
                    "gcc_linux-64", "gxx_linux-64"]:
            subprocess.run([conda_exe, "list", "-n", ENV_NAME, pkg], check=False)
        print_colored("=" * 70, Colors.CYAN)
        return True
    else:
        print_error("Failed to create environment!")
        print_info("\nManual fix:")
        print_colored("  conda clean --all -y", Colors.CYAN, bold=True)
        print_colored("  rm -rf ~/.conda/pkgs/*", Colors.CYAN, bold=True)
        print_colored("  python setup.py install", Colors.CYAN, bold=True)
        return False


def delete_environment(conda_exe):
    """Delete the KMX-env conda environment."""
    print_header(f"Deleting Conda Environment: {ENV_NAME}")

    if not check_env_exists(conda_exe, ENV_NAME):
        print_warning(f"Environment '{ENV_NAME}' does not exist – nothing to delete.")
        return True

    success, stdout, stderr = run_conda_command(
        conda_exe, ["env", "remove", "-n", ENV_NAME, "-y"]
    )
    if success:
        print_success(f"Environment '{ENV_NAME}' deleted successfully!")
        return True
    else:
        print_error("Failed to delete environment!")
        print_error(stderr)
        return False

# ================================
# Verify Installation
# ================================

def verify():
    """Verify that the KMX conda environment is correctly set up."""
    print_header("KMX Environment Verification")

    all_ok = True

    # [1/4] – Conda environment exists
    print_info("\n[1/4] Checking conda environment...")
    conda_exe, _ = find_conda()
    env_path = None
    if conda_exe and check_env_exists(conda_exe, ENV_NAME):
        env_path = get_env_path(conda_exe, ENV_NAME)
        print_success(f"Conda environment '{ENV_NAME}' exists at: {env_path}")
    else:
        print_error(f"Conda environment '{ENV_NAME}' NOT found!")
        print_info("  Fix: python setup.py install")
        all_ok = False

    # [2/4] – cuDF installed
    print_info("\n[2/4] Checking cuDF (RAPIDS)...")
    if conda_exe and env_path:
        success, stdout, _ = run_conda_command(
            conda_exe, ["list", "-n", ENV_NAME, "cudf"], check=False
        )
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
        print_error("Cannot check cuDF – environment not found")
        all_ok = False

    # [3/4] – GCC and Boost present
    print_info("\n[3/4] Checking GCC and Boost...")
    if env_path:
        # GCC
        gcc_path = os.path.join(env_path, "bin", "x86_64-conda-linux-gnu-gcc")
        if not os.path.isfile(gcc_path):
            gcc_path = os.path.join(env_path, "bin", "gcc")
        if os.path.isfile(gcc_path):
            try:
                result = subprocess.run([gcc_path, "--version"],
                                        capture_output=True, text=True, check=True)
                print_success(f"GCC: {result.stdout.split(chr(10))[0]}")
            except Exception as e:
                print_warning(f"GCC found but version check failed: {e}")
        else:
            print_error("GCC not found in conda environment!")
            all_ok = False

        # Boost
        boost_include = os.path.join(env_path, "include", "boost")
        if os.path.isdir(boost_include):
            version_hpp = os.path.join(boost_include, "version.hpp")
            if os.path.isfile(version_hpp):
                try:
                    with open(version_hpp) as f:
                        for line in f:
                            if 'BOOST_LIB_VERSION' in line and '"' in line:
                                ver = line.split('"')[1]
                                print_success(f"Boost headers found (version: {ver})")
                                break
                except Exception:
                    print_success("Boost headers found")
            else:
                print_success("Boost headers found")
        else:
            print_error("Boost headers NOT found!")
            all_ok = False
    else:
        print_error("Cannot check GCC/Boost – environment not found")
        all_ok = False

    # [4/4] – CMake version >= 4.2
    print_info("\n[4/4] Checking CMake version...")
    if env_path:
        cmake_path = os.path.join(env_path, "bin", "cmake")
        if os.path.isfile(cmake_path):
            try:
                result = subprocess.run([cmake_path, "--version"],
                                        capture_output=True, text=True, check=True)
                first_line = result.stdout.split('\n')[0]  # e.g. "cmake version 4.2.1"
                match = re.search(r'(\d+)\.(\d+)', first_line)
                if match:
                    maj, min_ = int(match.group(1)), int(match.group(2))
                    req_maj, req_min = int(CMAKE_CONDA_VERSION.split('.')[0]), \
                                       int(CMAKE_CONDA_VERSION.split('.')[1])
                    if (maj, min_) >= (req_maj, req_min):
                        print_success(f"CMake: {first_line}  (>= {CMAKE_CONDA_VERSION} ✓)")
                    else:
                        print_error(f"CMake too old: {first_line}")
                        print_info(f"  Required: cmake>={CMAKE_CONDA_VERSION}")
                        print_info(f"  Fix: conda install -n {ENV_NAME} -c conda-forge "
                                   f"cmake>={CMAKE_CONDA_VERSION}")
                        all_ok = False
                else:
                    print_success(f"CMake found: {first_line}")
            except Exception as e:
                print_warning(f"CMake found but version check failed: {e}")
        else:
            print_error("CMake not found in conda environment!")
            print_info(f"  Fix: conda install -n {ENV_NAME} -c conda-forge "
                       f"cmake>={CMAKE_CONDA_VERSION}")
            all_ok = False
    else:
        print_error("Cannot check CMake – environment not found")
        all_ok = False

    # Summary
    print_header("Verification Summary")
    if all_ok:
        print_colored("\n  ✓ ALL CHECKS PASSED – KMX environment is ready!\n",
                      Colors.GREEN, bold=True)
        print_info("  Usage:")
        print_colored("    conda activate KMX-env", Colors.CYAN, bold=True)
        print_colored("    python KMX.py -h",       Colors.CYAN, bold=True)
        return 0
    else:
        print_colored("\n  ✗ SOME CHECKS FAILED – see above for fixes\n",
                      Colors.RED, bold=True)
        print_info("  Most issues can be fixed by running:")
        print_colored("    python setup.py install", Colors.CYAN, bold=True)
        return 1

# ================================
# Install
# ================================

def install():
    """Create the KMX-env conda environment with all required packages."""
    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  KMX Environment Setup", Colors.CYAN, bold=True)
    print_colored("  Conda Environment Installer for Genomic K-mer Analysis", Colors.CYAN, bold=True)
    print_colored("="*70 + "\n", Colors.CYAN, bold=True)

    min_drv = f"{CUDA_MIN_DRIVER[0]}.{CUDA_MIN_DRIVER[1]}"
    print_colored("  REQUIREMENTS", Colors.YELLOW, bold=True)
    print_colored("  ─────────────────────────────────────────────────────", Colors.YELLOW)
    print_colored(f"  • NVIDIA Driver  >= {min_drv}  "
                  f"(host OS – must be installed before running this script)", Colors.YELLOW)
    print_colored(f"  • CUDA Runtime      {CUDA_VERSION}      "
                  f"(conda level – installed automatically by this script)", Colors.YELLOW)
    print_colored("  ─────────────────────────────────────────────────────\n", Colors.YELLOW)

    print_warning("Installation steps:")
    print_info(f"  1. Detect NVIDIA driver (REQUIRES driver >= {min_drv})")
    print_info("  2. Check available disk space")
    print_info("  3. Create conda environment (KMX-env)")
    print_info(f"  4. Install packages:")
    print_info(f"       cuDF {RAPIDS_VERSION}  |  Python {PYTHON_VERSION}  |  GCC {GCC_VERSION}  |  G++ {GCC_VERSION}")
    print_info(f"       Boost {BOOST_VERSION}  |  CMake {CMAKE_CONDA_VERSION}  |  Git  |  Make  |  zlib={ZLIB_VERSION}  |  bzip2={BZIP2_VERSION}")
    print_info(f"       CUDA runtime {CUDA_VERSION} (pulled in automatically as a dependency)")
    print_warning("⏱  Total estimated time: 15-20 minutes")
    print()

    if not estimate_disk_space():
        print_warning("Installation cancelled.")
        return 1

    driver_info = detect_nvidia_driver()
    if not driver_info:
        print_error(f"NVIDIA driver >= {min_drv} is REQUIRED for KMX!")
        print_error("Installation cannot continue without a compatible NVIDIA driver.")
        print_colored(f"\n  Install or update the NVIDIA driver (>= {min_drv}):", Colors.YELLOW, bold=True)
        print_info("  https://www.nvidia.com/Download/index.aspx")
        return 1

    conda_exe, conda_type = find_conda()
    if not conda_exe:
        return 1

    if not create_environment(conda_exe, conda_type, driver_info=driver_info):
        print_error("Environment creation failed!")
        return 1

    if not clone_gerbil():
        print_error("Failed to clone gerbil-DataFrame!")
        return 1

    if not patch_cmake():
        print_error("Failed to patch CMakeLists.txt!")
        return 1

    env_path = get_env_path(conda_exe, ENV_NAME)
    if not env_path:
        print_error("Cannot find conda environment path!")
        return 1

    gerbil_gpu = build_gerbil(env_path)
    if not gerbil_gpu:
        print_warning("gerbil build failed – KMX will continue but gerbil runs on CPU only!")

    # ── Success banner ────────────────────────────────────────────────────────
    print_colored("\n" + "="*70, Colors.GREEN, bold=True)
    print_colored("  ✓ Environment Setup Complete!", Colors.GREEN, bold=True)
    print_colored("="*70, Colors.GREEN, bold=True)

    if not gerbil_gpu:
        print_colored("\n" + "="*70, Colors.YELLOW, bold=True)
        print_colored("  ⚠  GERBIL GPU WARNING", Colors.YELLOW, bold=True)
        print_colored("="*70, Colors.YELLOW, bold=True)
        print_warning("  Gerbil was NOT compiled with GPU support.")
        print_warning("  Gerbil will run on CPU ONLY – this may be significantly slower.")
        print_warning("  Common causes:")
        print_info("    • CUDA toolkit not found during cmake configuration")
        print_info("    • cicc compiler not available in conda environment")
        print_info("    • Build error in CUDA kernel files")
        print_warning("  To fix, try recompiling manually:")
        print_colored("    conda activate KMX-env", Colors.CYAN, bold=True)
        print_colored("    cd include/gerbil-DataFrame", Colors.CYAN, bold=True)
        print_colored("    mkdir -p build && cd build", Colors.CYAN, bold=True)
        print_colored(f"    cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux", Colors.CYAN, bold=True)
        print_colored("    make -j", Colors.CYAN, bold=True)
        print_colored("="*70, Colors.YELLOW, bold=True)

    print_colored("\n" + "="*70, Colors.YELLOW, bold=True)
    print_colored("  ⚠  SYSTEM REQUIREMENTS", Colors.YELLOW, bold=True)
    print_colored("="*70, Colors.YELLOW, bold=True)
    print_warning("  GPU:  NVIDIA GPU with CUDA support (required by cuDF/RAPIDS)")
    print_warning("  RAM:  Depends on dataset size")
    print_warning("  Disk: At least 10 GB free for temporary files")

    print_colored("\n" + "="*70, Colors.YELLOW, bold=True)
    print_colored("  MAINTENANCE COMMANDS", Colors.YELLOW, bold=True)
    print_colored("="*70, Colors.YELLOW, bold=True)
    print_colored("\nVERIFY INSTALLATION:", Colors.CYAN, bold=True)
    print_colored("  python setup.py verify", Colors.GREEN)
    print_colored("\nRECREATE ENVIRONMENT (if dependency issues):", Colors.CYAN, bold=True)
    print_colored("  python setup.py uninstall", Colors.GREEN)
    print_colored("  python setup.py install",   Colors.GREEN)
    print_colored("\nCOMPLETE UNINSTALLATION:", Colors.CYAN, bold=True)
    print_colored("  python setup.py uninstall", Colors.GREEN)
    print_colored("\nCLEAN CONDA CACHE (free disk space):", Colors.CYAN, bold=True)
    print_colored("  conda clean --all -y", Colors.GREEN)

    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  Installation Details", Colors.CYAN, bold=True)
    print_colored("="*70, Colors.CYAN, bold=True)
    print_colored(f"\nEnvironment name:    {ENV_NAME}",            Colors.RESET)
    print_colored(f"Python version:      {PYTHON_VERSION}",        Colors.RESET)
    print_colored(f"cuDF version:        {RAPIDS_VERSION}",        Colors.RESET)
    print_colored(f"CUDA runtime:        {CUDA_VERSION}  (installed inside conda env)", Colors.RESET)
    print_colored(f"NVIDIA driver:       {driver_info[2]}  (host OS)", Colors.RESET)
    print_colored(f"GCC version:         {GCC_VERSION}",           Colors.RESET)
    print_colored(f"Boost version:       {BOOST_VERSION}",         Colors.RESET)
    print_colored(f"CMake version:       {CMAKE_CONDA_VERSION}",   Colors.RESET)

    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  HOW TO USE KMX", Colors.CYAN, bold=True)
    print_colored("="*70, Colors.CYAN, bold=True)
    print_colored("\nStep 1: ACTIVATE THE ENVIRONMENT", Colors.GREEN, bold=True)
    print_colored("   conda activate KMX-env", Colors.CYAN, bold=True)
    print_warning("   ⚠ You MUST activate KMX-env before running KMX!")
    print_colored("\nStep 2: VIEW USAGE INSTRUCTIONS", Colors.GREEN, bold=True)
    print_colored("   python KMX.py -h", Colors.CYAN, bold=True)
    print_colored("\nStep 3: RUN KMX", Colors.GREEN, bold=True)
    print_colored("   python KMX.py -l genomes.txt -k 31 -o output/ -t tmp/",
                  Colors.CYAN, bold=True)
    print()
    return 0

# ================================
# Uninstall
# ================================

def uninstall():
    """Remove the KMX-env conda environment and all compiled gerbil files."""
    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  KMX Uninstallation", Colors.CYAN, bold=True)
    print_colored("="*70 + "\n", Colors.CYAN, bold=True)

    # Show exactly what will be removed
    gerbil_exists = os.path.isdir(GERBIL_CLONE_PATH)
    print_warning("The following will be permanently removed:")
    print_info(f"  • Conda environment : {ENV_NAME}")
    if gerbil_exists:
        print_info(f"  • gerbil-DataFrame  : {os.path.abspath(GERBIL_CLONE_PATH)}")
        binary_abs = os.path.abspath(GERBIL_BINARY)
        if os.path.isfile(binary_abs):
            print_info(f"  • gerbil binary     : {binary_abs}")
    print()

    response = input("Are you sure you want to continue? (y/N): ").strip().lower()
    if response != 'y':
        print_info("Uninstallation cancelled – nothing was removed.")
        return 0

    all_ok = True

    # ── Step 1: Remove conda environment ─────────────────────────────────────
    print_header("Step 1/2 – Removing Conda Environment")
    conda_exe, _ = find_conda()
    if not conda_exe:
        print_error("Cannot find conda – skipping environment removal.")
        all_ok = False
    else:
        if not delete_environment(conda_exe):
            print_warning("Environment deletion had issues – continuing...")
            all_ok = False

    # ── Step 2: Remove gerbil-DataFrame directory ─────────────────────────────
    print_header("Step 2/2 – Removing gerbil-DataFrame")
    if gerbil_exists:
        try:
            shutil.rmtree(GERBIL_CLONE_PATH)
            print_success(f"Removed: {os.path.abspath(GERBIL_CLONE_PATH)}")
        except Exception as e:
            print_error(f"Failed to remove {GERBIL_CLONE_PATH}: {e}")
            print_info(f"  Manual fix: rm -rf {os.path.abspath(GERBIL_CLONE_PATH)}")
            all_ok = False
    else:
        print_info(f"gerbil-DataFrame directory not found – nothing to remove.")

    # ── Also remove empty include/ directory if nothing left ─────────────────
    include_dir = "include"
    if os.path.isdir(include_dir) and not os.listdir(include_dir):
        try:
            os.rmdir(include_dir)
            print_success(f"Removed empty directory: {os.path.abspath(include_dir)}")
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    print_colored("\n" + "="*70, Colors.GREEN if all_ok else Colors.YELLOW, bold=True)
    if all_ok:
        print_colored("  ✓ Uninstallation Complete!", Colors.GREEN, bold=True)
    else:
        print_colored("  ⚠ Uninstallation completed with some warnings.", Colors.YELLOW, bold=True)
    print_colored("="*70, Colors.GREEN if all_ok else Colors.YELLOW, bold=True)
    print_colored("\nOPTIONAL – free disk space from conda package cache:", Colors.YELLOW, bold=True)
    print_colored("  conda clean --all -y", Colors.CYAN, bold=True)
    print()
    return 0 if all_ok else 1

# ================================
# Gerbil Clone and Build
# ================================

GERBIL_REPO_URL    = "https://github.com/M-Serajian/gerbil-DataFrame.git"
GERBIL_CLONE_PATH  = "include/gerbil-DataFrame"
GERBIL_BUILD_DIR   = "include/gerbil-DataFrame/build"
GERBIL_BINARY      = "include/gerbil-DataFrame/build/gerbil"


def clone_gerbil():
    """Clone gerbil-DataFrame into include/."""
    print_header("Cloning gerbil-DataFrame")

    if os.path.isfile(os.path.join(GERBIL_CLONE_PATH, "CMakeLists.txt")):
        print_success(f"gerbil-DataFrame already cloned at {GERBIL_CLONE_PATH}")
        return True

    os.makedirs("include", exist_ok=True)

    if os.path.isdir(GERBIL_CLONE_PATH):
        print_warning("Directory exists but is empty – removing and re-cloning...")
        shutil.rmtree(GERBIL_CLONE_PATH, ignore_errors=True)

    print_info(f"Cloning from {GERBIL_REPO_URL}...")
    try:
        subprocess.run(
            ["git", "clone", GERBIL_REPO_URL, GERBIL_CLONE_PATH],
            check=True
        )
        print_success("gerbil-DataFrame cloned successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Git clone failed: {e}")
        return False


def patch_cmake():
    """Patch CMakeLists.txt line 1: VERSION 2.8 → VERSION 3.5 (required by CMake 4.2+)."""
    print_header("Patching gerbil CMakeLists.txt")

    cmake_file = os.path.join(GERBIL_CLONE_PATH, "CMakeLists.txt")
    if not os.path.isfile(cmake_file):
        print_error(f"CMakeLists.txt not found at {cmake_file}")
        return False

    with open(cmake_file, 'r') as f:
        lines = f.readlines()

    if lines and 'cmake_minimum_required' in lines[0]:
        original = lines[0].strip()
        lines[0] = 'cmake_minimum_required(VERSION 3.5)\n'
        with open(cmake_file, 'w') as f:
            f.writelines(lines)
        print_success(f"Patched: '{original}' → 'cmake_minimum_required(VERSION 3.5)'")
    else:
        print_info("cmake_minimum_required already patched or not on line 1")

    return True


def build_gerbil(env_path, force_rebuild=False):
    """Build gerbil inside the conda environment using $CONDA_PREFIX CUDA.

    If the binary already exists, asks the user whether to skip or recompile.
    Pass force_rebuild=True to always recompile without prompting.
    """
    print_header("Building gerbil-DataFrame")

    cmake_file = os.path.join(GERBIL_CLONE_PATH, "CMakeLists.txt")
    if not os.path.isfile(cmake_file):
        print_error("gerbil source not found! Run clone step first.")
        return False

    binary_abs = os.path.abspath(GERBIL_BINARY)

    # ── Already compiled? Ask the user ───────────────────────────────────────
    if os.path.isfile(binary_abs) and not force_rebuild:
        print_warning(f"gerbil binary already exists:")
        print_colored(f"  {binary_abs}", Colors.CYAN)
        file_size = os.path.getsize(binary_abs) / (1024 * 1024)
        import time
        mod_time  = time.strftime('%Y-%m-%d %H:%M:%S',
                                  time.localtime(os.path.getmtime(binary_abs)))
        print_info(f"  Size: {file_size:.1f} MB   |   Last built: {mod_time}")
        print()
        print_info("Options:")
        print_colored("  1. Skip recompilation  (use existing binary)", Colors.GREEN)
        print_colored("  2. Recompile from scratch  (clean build)", Colors.YELLOW)
        print()
        while True:
            response = input("Enter choice (1/2): ").strip()
            if response in ('1', '2'):
                break
            print_warning("  Please enter 1 or 2.")

        if response == '1':
            print_success("Skipping recompilation – using existing gerbil binary.")
            return True
        else:
            print_info("Cleaning previous build directory...")
            shutil.rmtree(os.path.abspath(GERBIL_BUILD_DIR), ignore_errors=True)
            print_success("Build directory cleaned.")

    # Create build directory
    os.makedirs(GERBIL_BUILD_DIR, exist_ok=True)

    gerbil_source_abs = os.path.abspath(GERBIL_CLONE_PATH)
    build_dir_abs     = os.path.abspath(GERBIL_BUILD_DIR)

    # Build env with conda paths and cicc on PATH
    my_env = os.environ.copy()
    my_env['PATH']             = f"{env_path}/nvvm/bin:{env_path}/bin:{my_env.get('PATH', '')}"
    my_env['LD_LIBRARY_PATH']  = f"{env_path}/lib:{my_env.get('LD_LIBRARY_PATH', '')}"
    my_env['CC']               = os.path.join(env_path, "bin", "x86_64-conda-linux-gnu-gcc")
    my_env['CXX']              = os.path.join(env_path, "bin", "x86_64-conda-linux-gnu-g++")
    cuda_root                  = os.path.join(env_path, "targets", "x86_64-linux")

    cmake_bin  = os.path.join(env_path, "bin", "cmake")
    cpu_count  = os.cpu_count() or 4

    # CMake configure
    print_info(f"Running cmake with CUDA root: {cuda_root}")
    print_info(f"cicc path: {env_path}/nvvm/bin/cicc")
    print_info(f"Forcing conda zlib/bzip2 (not system libraries)...")
    try:
        subprocess.run(
            [
                cmake_bin,
                f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_root}",
                f"-DZLIB_ROOT={env_path}",
                f"-DZLIB_LIBRARY={env_path}/lib/libz.so",
                f"-DZLIB_INCLUDE_DIR={env_path}/include",
                f"-DBZIP2_ROOT={env_path}",
                f"-DBZIP2_LIBRARIES={env_path}/lib/libbz2.so",
                f"-DBZIP2_INCLUDE_DIR={env_path}/include",
                "-S", gerbil_source_abs,
                "-B", build_dir_abs,
            ],
            check=True,
            env=my_env
        )
        # Check if CUDA was actually found in cmake output
        print_success("CMake configuration successful!")
    except subprocess.CalledProcessError as e:
        print_error(f"CMake failed with code {e.returncode}")
        print_colored("\n" + "="*70, Colors.YELLOW, bold=True)
        print_colored("  ⚠  GERBIL GPU WARNING", Colors.YELLOW, bold=True)
        print_colored("="*70, Colors.YELLOW, bold=True)
        print_warning("  CMake could not configure gerbil with CUDA.")
        print_warning("  Gerbil will NOT use the GPU – CPU only mode.")
        print_colored("="*70, Colors.YELLOW, bold=True)
        return False

    # Make
    print_info(f"Building with {cpu_count} parallel jobs...")
    try:
        subprocess.run(
            [cmake_bin, "--build", build_dir_abs, f"-j{cpu_count}"],
            check=True,
            env=my_env
        )
        print_success("Build completed!")
    except subprocess.CalledProcessError as e:
        print_error(f"Build failed with code {e.returncode}")
        return False

    # Verify binary
    binary_abs = os.path.abspath(GERBIL_BINARY)
    if os.path.isfile(binary_abs):
        os.chmod(binary_abs, 0o755)
        print_success(f"gerbil binary ready: {binary_abs}")
        return True
    else:
        print_error(f"Binary not found at {binary_abs} – build may have failed")
        return False


# ================================
# Usage / Entry Point
# ================================

def show_usage():
    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  KMX Setup Tool – Environment Installer", Colors.CYAN, bold=True)
    print_colored("="*70, Colors.CYAN, bold=True)
    print_colored("\nUSAGE:", Colors.YELLOW, bold=True)
    print_colored("  python setup.py install    ", Colors.GREEN, end="")
    print_colored("– Create KMX-env with all packages", Colors.RESET)
    print_colored("  python setup.py uninstall  ", Colors.GREEN, end="")
    print_colored("– Remove KMX-env", Colors.RESET)
    print_colored("  python setup.py verify     ", Colors.GREEN, end="")
    print_colored("– Check installation health", Colors.RESET)
    print()


def main():
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

