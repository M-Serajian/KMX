#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
import platform

# ================================
# Configuration
# ================================
GCC_VERSION_REQUIRED = "12.2"
PYTHON_VERSION_REQUIRED = (3, 8)
RAPIDS_VERSION_REQUIRED = "24.08"
SUBMODULE_PATH = "include/gerbil-DataFrame"
GIT_REPO = "https://github.com/M-Serajian/gerbil-DataFrame.git"

# ================================
# Utility Functions
# ================================

def run_command(cmd, check=True, capture_output=False, shell=False):
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)
            return result.stdout.decode().strip()
        else:
            subprocess.run(cmd, check=check, shell=shell)
            return None
    except subprocess.CalledProcessError as e:
        if capture_output:
            return e.stdout.decode().strip()
        raise e

def command_exists(cmd):
    return shutil.which(cmd) is not None

def print_color(msg, color_code):
    print(f"{color_code}{msg}\033[0m")

RED = "\033[0;31m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"

def is_windows():
    return platform.system() == "Windows"

# ================================
# Main Logic
# ================================

def main():
    force_install = "--force" in sys.argv

    if force_install:
        print_color("Forcing reinstallation... Removing existing submodule.", RED)
        run_command(["git", "submodule", "deinit", "-f", SUBMODULE_PATH], check=False)
        shutil.rmtree(".git/modules/" + SUBMODULE_PATH, ignore_errors=True)
        run_command(["git", "rm", "-f", SUBMODULE_PATH], check=False)
        shutil.rmtree(SUBMODULE_PATH, ignore_errors=True)
        run_command(["git", "add", ".gitmodules"], check=False)
        run_command(["git", "commit", "-m", "Removed gerbil-DataFrame submodule"], check=False)

    if os.path.isfile(os.path.join(SUBMODULE_PATH, "build", "gerbil")) and not force_install:
        print_color("The software is already installed.", BLUE)
        return

    # Python Version Check
    if sys.version_info < PYTHON_VERSION_REQUIRED:
        print_color(f"Python {PYTHON_VERSION_REQUIRED[0]}.{PYTHON_VERSION_REQUIRED[1]}+ is required.", RED)
        sys.exit(1)

    missing_tools = []

    # GCC or equivalent
    if not command_exists("gcc") and not is_windows():
        if command_exists("ml"):
            try:
                run_command("ml gcc", shell=True)
                print_color("Loaded GCC via 'ml gcc'.", GREEN)
            except:
                print_color("Error: Could not load GCC.", RED)
                missing_tools.append("gcc")
        else:
            print_color("GCC not found. Install GCC 12.2+", RED)
            missing_tools.append("gcc")

    # CMake
    if not command_exists("cmake"):
        print_color("CMake not found. Install CMake 3.13+", RED)
        missing_tools.append("cmake")

    # Boost (assumed via module or preinstalled)
    if command_exists("ml"):
        try:
            run_command("ml boost", shell=True)
            print_color("Loaded Boost via 'ml boost'.", GREEN)
        except:
            print_color("Could not load Boost via 'ml'.", RED)
            missing_tools.append("boost")
    else:
        print_color("Ensure Boost is available or installed.", BLUE)

    print_color("Note: Preferred RAPIDS AI version is 24.08 or above.", BLUE)

    if missing_tools:
        print_color("Missing dependencies: " + ", ".join(missing_tools), RED)
        sys.exit(1)

    # Add submodule
    os.makedirs("include", exist_ok=True)
    if not os.path.isdir(SUBMODULE_PATH):
        print_color("Adding submodule...", GREEN)
        run_command(["git", "submodule", "add", GIT_REPO, SUBMODULE_PATH])

    run_command(["git", "submodule", "init"])
    run_command(["git", "submodule", "update", "--recursive"])

    # Build
    build_dir = os.path.join(SUBMODULE_PATH, "build")
    os.makedirs(build_dir, exist_ok=True)
    os.chdir(build_dir)

    print("Running CMake...")
    try:
        run_command(["cmake", ".."])
        print_color("CMake configuration successful.", GREEN)
    except:
        print_color("CMake configuration failed.", RED)
        sys.exit(1)

    print("Building project...")
    try:
        if is_windows():
            run_command(["cmake", "--build", ".", "--config", "Release"])
        else:
            run_command(["cmake", "--build", ".", f"-j{os.cpu_count()}"])
        print_color("Build completed successfully.", GREEN)
    except:
        print_color("Build failed.", RED)
        sys.exit(1)

    os.chdir(os.path.abspath(os.path.join(build_dir, "../../")))

if __name__ == "__main__":
    main()
