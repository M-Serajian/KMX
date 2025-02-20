import argparse
import os
import sys

RED = "\033[91m"
RESET = "\033[0m"

def check_temp_space(directory):
    """Ensure the temporary directory has at least 10GB of free space."""
    if not os.path.isdir(directory):
        print(f"{RED}Error: Temporary directory not found - {directory}{RESET}", file=sys.stderr)
        sys.exit(1)

    free_space = os.statvfs(directory).f_frsize * os.statvfs(directory).f_bavail / (1024 ** 3)
    if free_space < 10:
        print(f"{RED}Error: Temporary directory '{directory}' has insufficient space ({free_space:.2f}GB available). "
              "At least 10GB is required.{RESET}", file=sys.stderr)
        sys.exit(1)

    return directory

def check_kmer_size(value):
    """Ensure the k-mer size is between 8 and 136."""
    try:
        kmer_size = int(value)
        if not (8 <= kmer_size <= 136):
            raise ValueError
    except ValueError:
        print(f"{RED}Error: K-mer size must be an integer between 8 and 136. Given: {value}{RESET}", file=sys.stderr)
        sys.exit(1)
    return kmer_size

def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Processes a list of genomes and extracts k-mer frequency data into a CSR matrix.",
        usage="python main.py -l PATH/genomes.txt -t PATH/tmp -o OUTPUT_FILE -k KMER_SIZE [--min X] [--max Y]"
    )

    parser.add_argument("-l", "--genome-list", type=str, required=True,
                        help="Path to a text file containing a list of genome file paths (required).")

    parser.add_argument("-min", type=int, default=1,
                        help="Minimum occurrence threshold for a k-mer to be retained (default: 1).")

    parser.add_argument("-max", type=int, default=None,
                        help="Maximum occurrence threshold for a k-mer to be retained. If not provided, no limit is applied.")

    parser.add_argument("-t", "--tmp", type=check_temp_space, required=True,
                        help="Path to a temporary directory with at least 10GB free space (required).")

    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output file where CSR matrix data (columns.npy, rows.npy, data.npy) will be stored (required).")

    parser.add_argument("-k", "--kmer-size", type=check_kmer_size, required=True,
                        help="Size of the k-mers to be analyzed (must be between 8 and 136, inclusive).")

    args = parser.parse_args()

    if args.max is not None and args.max <= args.min:
        print(f"{RED}Error: --max ({args.max}) must be greater than --min ({args.min}).{RESET}", file=sys.stderr)
        sys.exit(1)

    return args
