import argparse
import os
import sys

RED = "\033[91m"
RESET = "\033[0m"

def check_file_exists(file_path):
    """Ensure the specified file exists."""
    if not os.path.isfile(file_path):
        print(f"{RED}Error: File not found - {file_path}{RESET}", file=sys.stderr)
        sys.exit(1)
    return file_path

def check_directory_writable(directory):
    """Ensure the directory exists and is writable."""
    if not os.path.isdir(directory):
        print(f"{RED}Error: Directory not found - {directory}{RESET}", file=sys.stderr)
        sys.exit(1)
    if not os.access(directory, os.W_OK):
        print(f"{RED}Error: Directory is not writable - {directory}{RESET}", file=sys.stderr)
        sys.exit(1)
    return directory

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

def validate_genome_list(file_path):
    """Validate genome list file: ensure all referenced genome files exist and are readable."""
    if not os.path.isfile(file_path):
        print(f"{RED}Error: Genome list file not found - {file_path}{RESET}", file=sys.stderr)
        sys.exit(1)

    with open(file_path, "r") as f:
        genome_files = [line.strip() for line in f.readlines() if line.strip()]

    missing_files = []
    unreadable_files = []

    for genome in genome_files:
        if not os.path.isfile(genome):
            missing_files.append(genome)
        elif not os.access(genome, os.R_OK):
            unreadable_files.append(genome)

    if missing_files:
        print(f"{RED}Error: The following genome files were not found:{RESET}", file=sys.stderr)
        for file in missing_files:
            print(f"  {RED}- {file}{RESET}", file=sys.stderr)
        sys.exit(1)

    if unreadable_files:
        print(f"{RED}Error: The following genome files are not readable. Read permission is required:{RESET}", file=sys.stderr)
        for file in unreadable_files:
            print(f"  {RED}- {file}{RESET}", file=sys.stderr)
        sys.exit(1)

    return file_path

def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Processes a list of genomes and extracts k-mer frequency data into a CSR matrix.",
        usage="python main.py -l PATH/genomes.txt -t PATH/tmp -o PATH/output/ [--min X] [--max Y]"
    )

    parser.add_argument("-l", "--genome-list", type=validate_genome_list, required=True,
                        help="Path to a text file containing a list of genome file paths (required).")

    parser.add_argument("--min", type=int, default=1,
                        help="Minimum occurrence threshold for a k-mer to be retained (default: 1).")

    parser.add_argument("--max", type=int, default=None,
                        help="Maximum occurrence threshold for a k-mer to be retained. If not provided, no limit is applied.")

    parser.add_argument("-t", "--tmp", type=check_temp_space, required=True,
                        help="Path to a temporary directory with at least 10GB free space (required).")

    parser.add_argument("-o", "--output", type=check_directory_writable, required=True,
                        help="Output directory where CSR matrix files (columns.npy, rows.npy, data.npy) will be stored (required).")

    args = parser.parse_args()

    if args.max is not None and args.max <= args.min:
        print(f"{RED}Error: --max ({args.max}) must be greater than --min ({args.min}).{RESET}", file=sys.stderr)
        sys.exit(1)

    return args
