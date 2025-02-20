import os
import sys
import cudf
import cupy as cp

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from args import parse_arguments
from create_csr_matrix import create_csr_matrix

if __name__ == "__main__":
    args = parse_arguments()
    create_csr_matrix(
        genome_list=os.path.abspath(args.genome_list),
        kmer_size=args.kmer_size,
        min_val=args.min,
        max_val=args.max,
        tmp_dir=args.tmp
    )
