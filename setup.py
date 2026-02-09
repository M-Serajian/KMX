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
PYTHON_VERSION = "3.11"  # RAPIDS 25.06 requires Python 3.11
BOOST_VERSION = "1.77"
GCC_VERSION = "12"
CMAKE_MIN_VERSION = "3.13"
RAPIDS_VERSION = "25.06"
CUDA_VERSION = "12.6"  # CUDA toolkit version compatible with RAPIDS 25.06

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
# Disk Space Estimation
# ================================

def estimate_disk_space():
    """Estimate disk space needed and check availability."""
    print_info("Estimated disk space needed:")
    print_info("  - CUDA Toolkit 12.6: ~3 GB")
    print_info("  - cuDF (RAPIDS 25.06): ~2 GB")
    print_info("  - Python + dependencies: ~500 MB")
    print_info("  - Boost 1.77: ~200 MB")
    print_info("  - Build tools (CMake, Git, GCC 12): ~500 MB")
    print_info("  - Development libraries (zlib, bzip2): ~50 MB")
    print_info("  - gerbil source + build: ~200 MB")
    print_info("  - Package cache (temporary): ~2 GB")
    print_colored("\n  Total: ~8-9 GB\n", Colors.YELLOW, bold=True)
    
    # Check available space in home directory
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
# Conda Detection
# ================================

def find_conda():
    """Find conda or mamba executable."""
    print_header("Detecting Conda/Mamba")
    
    # Try mamba first (faster)
    mamba = shutil.which("mamba")
    if mamba:
        print_success(f"Found mamba: {mamba}")
        return mamba, "mamba"
    
    # Try conda
    conda = shutil.which("conda")
    if conda:
        print_success(f"Found conda: {conda}")
        return conda, "conda"
    
    print_error("Neither conda nor mamba found!")
    print_info("Please install conda from: https://docs.conda.io/en/latest/miniconda.html")
    return None, None

def get_conda_base():
    """Get the conda base directory."""
    try:
        result = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return None

def run_conda_command_live(conda_exe, args):
    """Run conda command with DIRECT output - no buffering, no capturing."""
    cmd = [conda_exe] + args
    print_colored(f"\nRunning: {' '.join(cmd)}\n", Colors.BLUE)
    print_colored("=" * 70, Colors.CYAN)
    print()
    
    try:
        # No capture - all output goes directly to terminal, including warnings
        result = subprocess.run(cmd, check=False)
        print()
        print_colored("=" * 70, Colors.CYAN)
        return result.returncode == 0
    except Exception as e:
        print_colored("=" * 70, Colors.RED)
        print_error(f"Exception: {e}")
        return False

def run_conda_command(conda_exe, args, check=True):
    """Run a conda command (silent version for quick checks)."""
    cmd = [conda_exe] + args
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

# ================================
# Git Detection (system or conda)
# ================================

def find_git(conda_exe=None):
    """Find a working git executable - prefer system git, fall back to conda env git."""
    # Try system git first
    system_git = shutil.which("git")
    if system_git:
        try:
            result = subprocess.run([system_git, "--version"], capture_output=True, text=True, check=True)
            print_success(f"Found system git: {system_git} ({result.stdout.strip()})")
            return [system_git]
        except:
            pass

    # Try conda env git
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

def create_environment(conda_exe, conda_type):
    """Create the conda environment with all dependencies."""
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
    
    print_info("This may take 15-20 minutes (CUDA toolkit is large)...")
    print_info(f"Installing: CUDA {CUDA_VERSION}, cuDF (RAPIDS {RAPIDS_VERSION}), Python {PYTHON_VERSION}, Boost {BOOST_VERSION}")
    print_warning("All warnings and messages will be shown (not suppressed)")
    
    packages = [
        f"python={PYTHON_VERSION}",
        f"cuda-toolkit={CUDA_VERSION}",
        f"cudf={RAPIDS_VERSION}",
        f"boost-cpp={BOOST_VERSION}",
        "cmake>=3.13",
        "git",
        "make",
        "zlib",
        "bzip2",
    ]
    
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
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "cuda-toolkit"], check=False)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "cudf"], check=False)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "boost-cpp"], check=False)
        subprocess.run([conda_exe, "list", "-n", ENV_NAME, "cmake"], check=False)
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
    
    # Create include directory if it doesn't exist
    include_dir = "include"
    os.makedirs(include_dir, exist_ok=True)
    
    # Check if submodule is already populated (has actual source files, not just empty dir)
    if os.path.isdir(GERBIL_SUBMODULE_PATH):
        cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")
        if os.path.isfile(cmake_file):
            print_success(f"gerbil-DataFrame already exists and has source files at {GERBIL_SUBMODULE_PATH}")
            return True
        else:
            print_warning(f"gerbil-DataFrame directory exists but is EMPTY (no CMakeLists.txt)")
            print_info("Will clone fresh copy...")
            # Remove the empty directory so git clone works
            shutil.rmtree(GERBIL_SUBMODULE_PATH, ignore_errors=True)

    # Find git
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
                
                # Verify it actually worked
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
        # Ensure target directory doesn't exist
        if os.path.exists(GERBIL_SUBMODULE_PATH):
            shutil.rmtree(GERBIL_SUBMODULE_PATH, ignore_errors=True)
        
        print_info(f"Cloning from {GERBIL_REPO_URL}...")
        clone_result = subprocess.run(
            git_cmd + ["clone", GERBIL_REPO_URL, GERBIL_SUBMODULE_PATH],
            check=True, text=True, capture_output=True
        )
        
        # VERIFY clone succeeded
        cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")
        if os.path.isfile(cmake_file):
            print_success("gerbil-DataFrame cloned successfully!")
            # List key files to confirm
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

def patch_cuda_scripts(env_path):
    """Patch generated CUDA cmake scripts to use absolute paths for cicc and other CUDA tools."""
    print_header("Patching Generated CUDA Build Scripts")
    
    build_dir_abs = os.path.abspath(GERBIL_BUILD_DIR)
    cuda_bin = os.path.join(env_path, "bin")
    
    # Find all generated cmake scripts
    cuda_scripts = glob.glob(os.path.join(build_dir_abs, "cuda_compile_*.cmake"))
    
    if not cuda_scripts:
        print_warning("No CUDA cmake scripts found to patch")
        return True
    
    print_info(f"Found {len(cuda_scripts)} CUDA cmake scripts to patch")
    
    # CUDA tools that need absolute paths
    cuda_tools = ["cicc", "nvcc", "ptxas", "fatbinary", "cudafe++"]
    
    patched_count = 0
    for script_path in cuda_scripts:
        try:
            with open(script_path, 'r') as f:
                content = f.read()
            
            modified = False
            for tool in cuda_tools:
                tool_path = os.path.join(cuda_bin, tool)
                if os.path.isfile(tool_path):
                    old_patterns = [
                        f'"{tool}"',
                        f"'{tool}'",
                        f' {tool} ',
                        f'({tool} ',
                        f' {tool})',
                    ]
                    
                    for pattern in old_patterns:
                        replacement = pattern.replace(tool, tool_path)
                        if pattern in content:
                            content = content.replace(pattern, replacement)
                            modified = True
            
            if modified:
                with open(script_path, 'w') as f:
                    f.write(content)
                print_success(f"Patched: {os.path.basename(script_path)}")
                patched_count += 1
        
        except Exception as e:
            print_warning(f"Could not patch {os.path.basename(script_path)}: {e}")
            continue
    
    print_success(f"Patched {patched_count} CUDA cmake scripts with absolute paths")
    return True

def patch_run_gerbil_py():
    """Patch src/run_gerbil.py to use the wrapper script instead of the direct binary.
    
    This is the CRITICAL patch that ensures gerbil runs with the correct library paths.
    The wrapper script sets LD_LIBRARY_PATH to include conda env libraries (CUDA, etc.)
    before executing the gerbil binary.
    """
    print_header("Patching run_gerbil.py to use wrapper script")
    
    run_gerbil_file = "src/run_gerbil.py"
    
    if not os.path.isfile(run_gerbil_file):
        print_error(f"{run_gerbil_file} not found")
        return False
    
    try:
        with open(run_gerbil_file, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'gerbil_wrapper.sh' in content:
            print_success("run_gerbil.py already patched to use wrapper script!")
            return True
        
        # Strategy: replace the GERBIL_EXECUTABLE line
        # The original line looks like:
        #   GERBIL_EXECUTABLE = os.path.join(CURRENT_DIR, '..', 'include', 'gerbil-DataFrame', 'build', 'gerbil')
        # We need to change 'gerbil') to 'gerbil_wrapper.sh')
        
        lines = content.split('\n')
        modified = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Match any line that sets GERBIL_EXECUTABLE and ends with 'gerbil')
            if 'GERBIL_EXECUTABLE' in line and '=' in line and 'gerbil' in line:
                # Check it's NOT already the wrapper
                if 'gerbil_wrapper.sh' not in line:
                    print_info(f"Original line {i+1}: {stripped}")
                    
                    # Replace: change the path to point to gerbil_wrapper.sh
                    # Handle both os.path.join(..., 'gerbil') and os.path.abspath(...) patterns
                    new_line = line.replace(
                        "'gerbil')", "'gerbil_wrapper.sh')"
                    ).replace(
                        '"gerbil")', '"gerbil_wrapper.sh")'
                    )
                    
                    # If the simple replacement didn't work, use a full replacement
                    if new_line == line:
                        new_line = "GERBIL_EXECUTABLE = os.path.join(CURRENT_DIR, '..', 'include', 'gerbil-DataFrame', 'build', 'gerbil_wrapper.sh')"
                    
                    lines[i] = new_line
                    print_info(f"Patched line {i+1}: {lines[i].strip()}")
                    modified = True
                    # Don't break - patch ALL GERBIL_EXECUTABLE assignments
        
        # Also handle the second assignment (os.path.abspath line) - skip it,
        # it just normalizes the path and will work fine with the wrapper
        
        if modified:
            with open(run_gerbil_file, 'w') as f:
                f.write('\n'.join(lines))
            
            # Verify
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
    
    # Get environment path
    env_path = get_env_path(conda_exe, ENV_NAME)
    if not env_path:
        print_error(f"Could not find path for environment {ENV_NAME}")
        return False
    
    print_info(f"Environment path: {env_path}")
    
    # Get absolute path to gerbil binary
    gerbil_binary_abs = os.path.abspath(GERBIL_BINARY)
    wrapper_abs = os.path.abspath(GERBIL_WRAPPER)
    
    # Verify the gerbil binary actually exists before creating wrapper
    if not os.path.isfile(gerbil_binary_abs):
        print_error(f"Cannot create wrapper - gerbil binary does not exist at: {gerbil_binary_abs}")
        return False
    
    # Create wrapper script
    wrapper_content = f"""#!/bin/bash
# Wrapper script to run gerbil with system GLIBC but conda CUDA libraries
# Generated by setup.py - do not edit manually

# Only add conda lib path for CUDA libraries, not for GLIBC
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
        
        # Make it executable
        os.chmod(wrapper_abs, 0o755)
        
        print_success(f"Wrapper script created: {wrapper_abs}")
        
        # Verify it's executable
        if os.access(wrapper_abs, os.X_OK):
            print_success("Wrapper is executable")
        else:
            print_error("Wrapper is not executable!")
            return False
            
        return True
    except Exception as e:
        print_error(f"Failed to create wrapper script: {e}")
        return False

def build_gerbil(conda_exe, force_rebuild=False):
    """Build gerbil using the conda environment with proper CUDA PATH setup."""
    print_header("Building gerbil-DataFrame")
    
    # FIRST: Verify source files exist
    cmake_file = os.path.join(GERBIL_SUBMODULE_PATH, "CMakeLists.txt")
    if not os.path.isfile(cmake_file):
        print_error(f"gerbil-DataFrame source not found! (no CMakeLists.txt at {cmake_file})")
        print_info("The gerbil-DataFrame submodule was not cloned properly.")
        print_info("Try running: python setup.py install")
        return False
    
    # Check if already built
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
        
        # Clean for rebuild
        if os.path.isdir(GERBIL_BUILD_DIR):
            print_info("Cleaning previous build...")
            shutil.rmtree(GERBIL_BUILD_DIR)
    
    # Create build directory
    os.makedirs(GERBIL_BUILD_DIR, exist_ok=True)
    print_success(f"Build directory created: {GERBIL_BUILD_DIR}")
    
    # Get absolute paths
    gerbil_source_abs = os.path.abspath(GERBIL_SUBMODULE_PATH)
    build_dir_abs = os.path.abspath(GERBIL_BUILD_DIR)
    
    # Get conda environment path
    env_path = get_env_path(conda_exe, ENV_NAME)
    if not env_path:
        print_error(f"Could not find path for environment {ENV_NAME}")
        return False
    
    print_info(f"Using environment at: {env_path}")
    
    # Get conda base for activation
    conda_base = get_conda_base()
    if not conda_base:
        print_error("Could not find conda base directory")
        return False
    
    # Construct paths for CUDA
    cuda_root = env_path
    cuda_bin = os.path.join(env_path, "bin")
    cuda_lib = os.path.join(env_path, "lib")
    
    try:
        # CMake configuration
        print_info("Running CMake configuration...")
        print_warning("All warnings and errors will be shown")
        print_info(f"Setting CUDA_TOOLKIT_ROOT_DIR={cuda_root}")
        print_colored("=" * 70, Colors.CYAN)
        print()
        
        # Build environment with CUDA paths
        my_env = os.environ.copy()
        my_env['PATH'] = f"{cuda_bin}:{my_env.get('PATH', '')}"
        my_env['LD_LIBRARY_PATH'] = f"{cuda_lib}:{my_env.get('LD_LIBRARY_PATH', '')}"
        my_env['CUDA_HOME'] = cuda_root
        my_env['CUDA_ROOT'] = cuda_root
        
        # Use SYSTEM GCC to avoid GLIBC version mismatch
        cmake_cmd = [
            "bash", "-c",
            f"source {conda_base}/etc/profile.d/conda.sh && conda activate {ENV_NAME} && "
            f"cmake -DCUDA_TOOLKIT_ROOT_DIR='{cuda_root}' "
            f"-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ "
            f"-S '{gerbil_source_abs}' -B '{build_dir_abs}'"
        ]
        
        subprocess.run(cmake_cmd, check=True, env=my_env)
        
        print()
        print_colored("=" * 70, Colors.CYAN)
        print_success("CMake configuration successful!")
        
        # Patch the generated CUDA cmake scripts to use absolute paths
        if not patch_cuda_scripts(env_path):
            print_warning("CUDA script patching had issues, build may fail")
        
        # Build with proper environment
        cpu_count = os.cpu_count() or 4
        print_info(f"Building gerbil using {cpu_count} parallel jobs...")
        print_warning("All compilation warnings and errors will be shown")
        print_colored("=" * 70, Colors.CYAN)
        print()
        
        build_cmd = [
            "bash", "-c",
            f"source {conda_base}/etc/profile.d/conda.sh && conda activate {ENV_NAME} && cmake --build '{build_dir_abs}' -j{cpu_count}"
        ]
        
        subprocess.run(build_cmd, check=True, env=my_env)
        
        print()
        print_colored("=" * 70, Colors.CYAN)
        print_success("Build completed!")
        
        # ===== CRITICAL VERIFICATION =====
        if not os.path.isfile(GERBIL_BINARY):
            print_error(f"BUILD FAILED: gerbil binary NOT found at {GERBIL_BINARY}")
            print_info("CMake/make returned success but no binary was produced.")
            print_info("Check the build output above for errors.")
            
            # List what IS in the build directory
            if os.path.isdir(GERBIL_BUILD_DIR):
                build_files = os.listdir(GERBIL_BUILD_DIR)
                print_info(f"  Build directory contains: {build_files}")
                # Check for binary with different name
                for f in build_files:
                    full = os.path.join(GERBIL_BUILD_DIR, f)
                    if os.access(full, os.X_OK) and os.path.isfile(full):
                        print_warning(f"  Found executable: {f}")
            return False
        
        os.chmod(GERBIL_BINARY, 0o755)
        print_success(f"gerbil binary verified at: {os.path.abspath(GERBIL_BINARY)}")
        
        # Create wrapper script
        if not create_gerbil_wrapper(conda_exe):
            print_error("Failed to create wrapper script!")
            return False
        
        # Patch run_gerbil.py to use wrapper
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
    print_info("\n[1/6] Checking conda environment...")
    conda_exe, conda_type = find_conda()
    if conda_exe and check_env_exists(conda_exe, ENV_NAME):
        env_path = get_env_path(conda_exe, ENV_NAME)
        print_success(f"Conda environment '{ENV_NAME}' exists at: {env_path}")
    else:
        print_error(f"Conda environment '{ENV_NAME}' NOT found!")
        print_info("  Fix: python setup.py install")
        all_ok = False
    
    # 2. Check gerbil source
    print_info("\n[2/6] Checking gerbil-DataFrame source...")
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
    
    # 3. Check gerbil binary
    print_info("\n[3/6] Checking gerbil binary...")
    gerbil_binary_abs = os.path.abspath(GERBIL_BINARY)
    if os.path.isfile(gerbil_binary_abs):
        if os.access(gerbil_binary_abs, os.X_OK):
            print_success(f"gerbil binary exists and is executable: {gerbil_binary_abs}")
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
    
    # 4. Check wrapper script
    print_info("\n[4/6] Checking gerbil wrapper script...")
    wrapper_abs = os.path.abspath(GERBIL_WRAPPER)
    if os.path.isfile(wrapper_abs):
        if os.access(wrapper_abs, os.X_OK):
            print_success(f"Wrapper script exists and is executable: {wrapper_abs}")
            # Check wrapper content points to valid binary
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
    
    # 5. Check run_gerbil.py patching
    print_info("\n[5/6] Checking run_gerbil.py configuration...")
    run_gerbil_file = "src/run_gerbil.py"
    if os.path.isfile(run_gerbil_file):
        with open(run_gerbil_file, 'r') as f:
            content = f.read()
        if 'gerbil_wrapper.sh' in content:
            print_success("run_gerbil.py is configured to use gerbil_wrapper.sh")
        else:
            print_error("run_gerbil.py is NOT patched - still points to raw gerbil binary!")
            print_info("  This WILL cause FileNotFoundError or GLIBC errors at runtime.")
            # Show what it currently points to
            for i, line in enumerate(content.split('\n'), 1):
                if 'GERBIL_EXECUTABLE' in line and '=' in line:
                    print_info(f"  Line {i}: {line.strip()}")
            print_info("  Fix: python setup.py install")
            all_ok = False
    else:
        print_error(f"{run_gerbil_file} not found!")
        all_ok = False
    
    # 6. Check KMX.py
    print_info("\n[6/6] Checking KMX.py...")
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
    print_info("  1. Create conda environment (KMX-env)")
    print_info("  2. Install CUDA Toolkit 12.6, cuDF (RAPIDS 25.06), Boost 1.77")
    print_info("  3. Clone/initialize gerbil-DataFrame submodule")
    print_info("  4. Patch CMakeLists.txt for compatibility")
    print_info("  5. Configure build with CMake")
    print_info("  6. Patch generated CUDA scripts with absolute paths")
    print_info("  7. Compile gerbil binary with CUDA support")
    print_info("  8. Create wrapper script for GLIBC compatibility")
    print_info("  9. Patch run_gerbil.py to use wrapper")
    print_warning("⏱  Total estimated time: 15-20 minutes")
    print()
    
    # Disk space check
    if not estimate_disk_space():
        print_warning("Installation cancelled.")
        return 1
    
    # Find conda
    conda_exe, conda_type = find_conda()
    if not conda_exe:
        return 1
    
    # Create environment
    if not create_environment(conda_exe, conda_type):
        print_error("Environment creation failed!")
        return 1
    
    # Clone submodule
    if not clone_gerbil_submodule(conda_exe):
        print_error("Failed to setup gerbil-DataFrame!")
        print_error("This is a critical step - cannot continue without gerbil source code.")
        return 1
    
    # Patch CMakeLists
    if not patch_gerbil_cmake():
        print_error("Failed to patch CMakeLists.txt!")
        return 1
    
    # Build gerbil
    if not build_gerbil(conda_exe):
        print_error("Build failed!")
        return 1
    
    # ================================
    # FINAL VERIFICATION
    # ================================
    print_header("Final Verification")
    
    # Verify all critical files exist
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
    
    # Verify run_gerbil.py patch
    with open("src/run_gerbil.py", 'r') as f:
        if 'gerbil_wrapper.sh' in f.read():
            print_success("run_gerbil.py correctly references gerbil_wrapper.sh")
        else:
            print_error("run_gerbil.py does NOT reference gerbil_wrapper.sh!")
            all_ok = False
    
    if not all_ok:
        print_error("\nInstallation completed but verification found issues!")
        print_info("Run: python setup.py verify   for detailed diagnostics")
        return 1
    
    # ================================
    # SUCCESS
    # ================================
    print_colored("\n" + "="*70, Colors.GREEN, bold=True)
    print_colored("  ✓ Installation Complete!", Colors.GREEN, bold=True)
    print_colored("="*70, Colors.GREEN, bold=True)
    
    print_colored("\n" + "="*70, Colors.YELLOW, bold=True)
    print_colored("  ⚠ IMPORTANT: SYSTEM REQUIREMENTS", Colors.YELLOW, bold=True)
    print_colored("="*70, Colors.YELLOW, bold=True)
    print_warning("   GPU: NVIDIA GPU with CUDA support (compatible with RAPIDS AI 25.06)")
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
    
    # Installation details
    print_colored("\n" + "="*70, Colors.CYAN, bold=True)
    print_colored("  Installation Details", Colors.CYAN, bold=True)
    print_colored("="*70, Colors.CYAN, bold=True)
    print_colored(f"\nEnvironment name:   KMX-env", Colors.RESET)
    print_colored(f"gerbil binary:      {os.path.abspath(GERBIL_BINARY)}", Colors.RESET)
    print_colored(f"gerbil wrapper:     {os.path.abspath(GERBIL_WRAPPER)}", Colors.RESET)
    print_colored(f"Configuration:      CUDA 12.6, cuDF 25.06, Python 3.11, Boost 1.77", Colors.RESET)
    
    # How to use
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
    
    # Clean up gerbil-DataFrame
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
