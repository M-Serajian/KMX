"""KMX — GPU-accelerated genome × k-mer CSR matrix builder.

Public API
----------
    import KMX

    # high-level: build straight from a manifest CSV (same as the `KMX` CLI)
    KMX.build("manifest.csv", kmer_size=21, tmp_dir="/scratch/tmp", output_dir="out/")
        -> writes the CSR matrix + metadata to out/, and returns
           (data, column, row, kmer_index_df, sparsity)

    # low-level: if you already have parsed-manifest inputs
    KMX.create_csr_matrix(...)

Run from the shell
------------------
    KMX -l manifest.csv -k 21 -t /scratch/tmp -o out/
    python -m KMX -l manifest.csv -k 21 -t /scratch/tmp -o out/

See help(KMX.build) for all arguments.
"""

__version__ = "2.0.0.dev0"

from .cli import build                              # high-level entry point
from .create_csr_matrix import create_csr_matrix    # low-level builder

__all__ = ["build", "create_csr_matrix", "__version__"]
