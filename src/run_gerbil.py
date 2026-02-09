import os
import subprocess

# Find the directory where run_gerbil.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the gerbil binary (src/ -> project root -> include/gerbil-DataFrame/build/gerbil)
GERBIL_EXECUTABLE = os.path.join(CURRENT_DIR, '..', 'include', 'gerbil-DataFrame', 'build', 'gerbil_wrapper.sh')
GERBIL_EXECUTABLE = os.path.abspath(GERBIL_EXECUTABLE)


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
    
    if enable_gpu:
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
    
    if enable_gpu:
        command.append("-g")

    if disable_normalization:
        command.append("-d")
    
    command.extend([genome_dir, tmp_dir, output_file])
    
    result = subprocess.run(command, check=True, text=True, capture_output=True, env=_get_gerbil_env())
    
    return output_file
