import os
import subprocess

# Find the directory where run_gerbil.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the gerbil binary (src/ -> project root -> include/gerbil-DataFrame/build/gerbil)
GERBIL_EXECUTABLE = os.path.join(CURRENT_DIR, '..', 'include', 'gerbil-DataFrame', 'build', 'gerbil')
GERBIL_EXECUTABLE = os.path.abspath(GERBIL_EXECUTABLE)

# Marker file written by setup.py when gerbil was built with CUDA/GPU support.
# Sits next to the gerbil binary in include/gerbil-DataFrame/build/.gerbil_gpu
GERBIL_GPU_MARKER = os.path.join(os.path.dirname(GERBIL_EXECUTABLE), '.gerbil_gpu')

# Cache the "did we already warn the user?" state so we don't spam logs.
_GPU_FALLBACK_WARNED = False


def gerbil_built_with_gpu():
    """Return True if gerbil was compiled with CUDA/GPU support.

    Detected via the marker file written by setup.py at build time.
    """
    return os.path.isfile(GERBIL_GPU_MARKER)


def _resolve_gpu_flag(enable_gpu):
    """Decide whether to pass -g to gerbil based on user request and build mode.

    Returns True only when both the user wants GPU AND gerbil was compiled with
    GPU support. Logs a one-time warning if the user wants GPU but gerbil is
    CPU-only.
    """
    global _GPU_FALLBACK_WARNED
    if not enable_gpu:
        return False
    if gerbil_built_with_gpu():
        return True
    if not _GPU_FALLBACK_WARNED:
        print(
            "Warning: gerbil was compiled CPU-only; ignoring GPU request and "
            "running gerbil on CPU. To enable GPU, reinstall with: "
            "python setup.py install --gerbil-gpu",
            flush=True,
        )
        _GPU_FALLBACK_WARNED = True
    return False


def _get_gerbil_env():
    """Build an environment dict that includes conda's lib directory in LD_LIBRARY_PATH.

    This ensures gerbil can find CUDA shared libraries (libcudart.so, etc.)
    from the active conda environment without needing a wrapper script.
    Works on any system as long as the KMX-env conda environment is activated.
    """
    env = os.environ.copy()
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _check_gerbil_exists():
    """Verify gerbil binary exists and give a clear error if not."""
    if not os.path.isfile(GERBIL_EXECUTABLE):
        raise FileNotFoundError(
            f"gerbil binary not found at: {GERBIL_EXECUTABLE}\n"
            f"Please run 'python setup.py install' to build it, "
            f"then 'python setup.py verify' to confirm."
        )


def set_of_all_unique_kmers_extractor(genome_file, output_directory, kmer_length, min_threshold, max_threshold, temp_directory, disable_normalization=False, enable_gpu=True):
    """Run the gerbil-DataFrame tool to extract unique k-mers from the given genome file."""
    _check_gerbil_exists()

    command = [
        GERBIL_EXECUTABLE,
        "-k", str(kmer_length),
        "-o", "csv",
        "-l", str(min_threshold),
        "-z", str(max_threshold)]

    if _resolve_gpu_flag(enable_gpu):
        command.append("-g")

    if disable_normalization:
        command.append("-d")

    command.extend([genome_file, temp_directory, output_directory])

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, env=_get_gerbil_env())
        print(f"Unique k-mers successfully extracted and stored at: {output_directory}")
    except subprocess.CalledProcessError as error:
        print(f"Error: Extraction of unique k-mers failed with return code {error.returncode}.")
        print(f"Standard Output:\n{error.stdout}")
        print(f"Standard Error:\n{error.stderr}")


def single_genome_kmer_extractor(kmer_size, tmp_dir, output_file, genome_dir, disable_normalization=False, enable_gpu=True):
    """Extract k-mers from a single genome using the gerbil-DataFrame tool."""
    _check_gerbil_exists()

    command = [
        GERBIL_EXECUTABLE,
        "-k", str(kmer_size),
        "-o", "csv",
        "-l", str(1),
        "-z", str(10**9)]

    if _resolve_gpu_flag(enable_gpu):
        command.append("-g")

    if disable_normalization:
        command.append("-d")

    command.extend([genome_dir, tmp_dir, output_file])
    
    result = subprocess.run(command, check=True, text=True, capture_output=True, env=_get_gerbil_env())
    
    return output_file
