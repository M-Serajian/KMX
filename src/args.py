import argparse
import os
import sys

# Define color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ensure_trailing_slash(directory):
    """Ensure the directory path ends with '/'."""
    return directory if directory.endswith("/") else directory + "/"

def check_temp_space(directory):
    """Ensure the temporary directory exists and has at least 10GB of free space."""
    directory = ensure_trailing_slash(directory)  # Ensure '/' at the end

    if os.path.isdir(directory):
        print(f"{GREEN}Temporary directory exists and found: {directory}{RESET}", flush=True)
    else:
        print(f"{YELLOW}Creating temporary directory: {directory}{RESET}", flush=True)
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"{GREEN}Successfully created: {directory}{RESET}", flush=True)
        except Exception as e:
            print(f"{RED}Error: Could not create temporary directory '{directory}'. Reason: {e}{RESET}", file=sys.stderr, flush=True)
            sys.exit(1)

    # Check available disk space
    free_space = os.statvfs(directory).f_frsize * os.statvfs(directory).f_bavail / (1024 ** 3)
    if free_space < 10:
        print(f"{RED}Error: Temporary directory '{directory}' has insufficient space ({free_space:.2f}GB available). "
              f"At least 10GB is required.{RESET}", file=sys.stderr, flush=True)
        sys.exit(1)

    return directory

def check_kmer_size(value):
    """Ensure the k-mer size is between 8 and 136."""
    try:
        kmer_size = int(value)
        if not (8 <= kmer_size <= 136):
            raise ValueError
    except ValueError:
        print(f"{RED}Error: K-mer size must be an integer between 8 and 136. Given: {value}{RESET}", file=sys.stderr, flush=True)
        sys.exit(1)
    return kmer_size

def check_output_directory(directory):
    """Ensure the output directory exists or create it."""
    directory = ensure_trailing_slash(directory)  # Ensure '/' at the end

    if os.path.isdir(directory):
        print(f"{GREEN}Output directory exists and found: {directory}{RESET}", flush=True)
    else:
        print(f"{YELLOW}Creating output directory: {directory}{RESET}", flush=True)
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"{GREEN}Successfully created: {directory}{RESET}", flush=True)
        except Exception as e:
            print(f"{RED}Error: Could not create output directory '{directory}'. Reason: {e}{RESET}", file=sys.stderr, flush=True)
            sys.exit(1)

    return directory

def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Processes a list of genomes and extracts k-mer frequency data into a CSR matrix.",
        usage="python main.py -l PATH/genomes.txt -t PATH/tmp -k KMER_SIZE -o OUTPUT_DIR [-min X] [-max Y] [-d]"
    )

    parser.add_argument("-l", "--genome-list", type=str, required=True,
                        help="Path to a text file containing a list of genome file paths (required).")

    parser.add_argument("-min", type=int, default=1,
                        help="Minimum occurrence threshold for a k-mer to be retained (default: 1).")

    parser.add_argument("-max", type=int, default=None,
                        help="Maximum occurrence threshold for a k-mer to be retained. If not provided, no limit is applied.")

    parser.add_argument("-t", "--tmp", type=check_temp_space, required=True,
                        help="Path to a temporary directory with at least 10GB free space (required).")

    parser.add_argument("-k", "--kmer-size", type=check_kmer_size, required=True,
                        help="Size of the k-mers to be analyzed (must be between 8 and 136, inclusive).")

    parser.add_argument("-d", "--disable-normalization", action="store_true",
                        help="Disable normalization of k-mers. If normalization is disabled, a k-mer and its reverse complement "
                             "are considered as different k-mers. If normalization is enabled (default), both k-mer and "
                             "its reverse complement are mapped to the same k-mer.")

    parser.add_argument("-c", "--cpu", action="store_true", default=False,
                        help="Force CPU mode. If not specified, GPU acceleration is enabled by default.")

    parser.add_argument("-o", "--output", type=check_output_directory, required=True,
                        help="Path to the output directory where results will be stored (required).")

    args = parser.parse_args()

    if args.max is not None and args.max < args.min:
        print(f"{RED}Error: -max ({args.max}) must be equal to or greater than -min ({args.min}).{RESET}", file=sys.stderr, flush=True)
        sys.exit(1)

    return args
