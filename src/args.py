#!/usr/bin/env python3
import argparse
import os
import sys

# Define color codes (only used for interactive terminals)
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _use_color() -> bool:
    """Return True if stdout is a TTY (safe to emit ANSI colors)."""
    return sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    """Colorize text if appropriate."""
    return f"{color}{text}{RESET}" if _use_color() else text


def check_temp_space(directory: str, min_free_gb: float = 10.0) -> str:
    """
    Ensure the temporary directory exists and has at least `min_free_gb` of free space.
    Returns an absolute path (no trailing-slash enforcement needed).
    """
    directory = os.path.abspath(os.path.expanduser(directory))

    if os.path.isdir(directory):
        print(_c(f"Temporary directory exists and found: {directory}", GREEN), flush=True)
    else:
        print(_c(f"Creating temporary directory: {directory}", YELLOW), flush=True)
        try:
            os.makedirs(directory, exist_ok=True)
            print(_c(f"Successfully created: {directory}", GREEN), flush=True)
        except Exception as e:
            print(_c(f"Error: Could not create temporary directory '{directory}'. Reason: {e}", RED),
                  file=sys.stderr, flush=True)
            sys.exit(1)

    # Check available disk space
    try:
        st = os.statvfs(directory)
        free_space_gb = (st.f_frsize * st.f_bavail) / (1024 ** 3)
    except Exception as e:
        print(_c(f"Error: Could not check free space for '{directory}'. Reason: {e}", RED),
              file=sys.stderr, flush=True)
        sys.exit(1)

    if free_space_gb < min_free_gb:
        print(
            _c(
                f"Error: Temporary directory '{directory}' has insufficient space "
                f"({free_space_gb:.2f} GB available). At least {min_free_gb:.0f} GB is required.",
                RED,
            ),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    return directory


def check_kmer_size(value: str) -> int:
    """Ensure the k-mer size is between 8 and 136."""
    try:
        kmer_size = int(value)
    except ValueError:
        print(_c(f"Error: K-mer size must be an integer between 8 and 136. Given: {value}", RED),
              file=sys.stderr, flush=True)
        sys.exit(1)

    if not (8 <= kmer_size <= 136):
        print(_c(f"Error: K-mer size must be between 8 and 136. Given: {kmer_size}", RED),
              file=sys.stderr, flush=True)
        sys.exit(1)

    return kmer_size


def check_output_directory(directory: str) -> str:
    """Ensure the output directory exists or create it. Returns an absolute path."""
    directory = os.path.abspath(os.path.expanduser(directory))

    if os.path.isdir(directory):
        print(_c(f"Output directory exists and found: {directory}", GREEN), flush=True)
    else:
        print(_c(f"Creating output directory: {directory}", YELLOW), flush=True)
        try:
            os.makedirs(directory, exist_ok=True)
            print(_c(f"Successfully created: {directory}", GREEN), flush=True)
        except Exception as e:
            print(_c(f"Error: Could not create output directory '{directory}'. Reason: {e}", RED),
                  file=sys.stderr, flush=True)
            sys.exit(1)

    return directory


def count_inputs_in_genome_list(genome_list_path: str) -> int:
    """
    Count the number of FASTA file paths in the genome list file.
    Ignores blank lines and lines starting with '#'.
    """
    n = 0
    try:
        with open(genome_list_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                n += 1
    except Exception as e:
        print(_c(f"Error: Could not read genome list file '{genome_list_path}'. Reason: {e}", RED),
              file=sys.stderr, flush=True)
        sys.exit(1)
    return n


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Processes a list of genomes and extracts k-mer frequency data into a CSR matrix."
    )

    parser.add_argument(
        "-l", "--genome-list", type=str, required=True,
        help="Path to a text file containing FASTA file paths (required)."
    )

    parser.add_argument(
        "-min", dest="min", type=int, default=5,
        help="Minimum occurrence threshold for a k-mer to be retained (default: 5)."
    )

    parser.add_argument(
        "-max", dest="max", type=int, default=None,
        help=("Maximum occurrence threshold for a k-mer to be retained. "
              "If omitted, defaults to N/2 where N is the number of FASTA paths listed in --genome-list "
              "(after ignoring blank lines and lines starting with '#').")
    )

    parser.add_argument(
        "-t", "--tmp", type=check_temp_space, required=True,
        help="Path to a temporary directory with at least 10GB free space (required)."
    )

    parser.add_argument(
        "-k", "--kmer-size", type=check_kmer_size, required=True,
        help="Size of the k-mers to be analyzed (must be between 8 and 136, inclusive)."
    )

    parser.add_argument(
        "-d", "--disable-normalization", action="store_true",
        help=("Disable normalization of k-mers. If normalization is disabled, a k-mer and its reverse complement "
              "are considered as different k-mers. If normalization is enabled (default), both k-mer and its reverse "
              "complement are mapped to the same k-mer.")
    )

    parser.add_argument(
        "-c", "--cpu", action="store_true", default=False,
        help="Force CPU mode. If not specified, GPU acceleration is enabled by default."
    )

    parser.add_argument(
        "-o", "--output", type=check_output_directory, required=True,
        help="Path to the output directory where results will be stored (required)."
    )

    args = parser.parse_args()

    # Normalize genome list path early
    args.genome_list = os.path.abspath(os.path.expanduser(args.genome_list))

    # Compute dynamic default for --max if not provided
    if args.max is None:
        n_inputs = count_inputs_in_genome_list(args.genome_list)
        args.max = max(1, n_inputs // 2)

    # Validate min/max relationship
    if args.max < args.min:
        parser.error(f"--max ({args.max}) must be equal to or greater than --min ({args.min}).")

    return args
