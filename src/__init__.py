"""KMX — GPU-accelerated genome × k-mer CSR matrix builder.

Public API
----------
    import KMX
    KMX.create_csr_matrix(...)   -> (data, column, row, kmer_index_df, sparsity)

Run-from-CLI
------------
    KMX -l genomes.txt -k 21 -t /scratch/tmp -o /results/out
    python -m KMX -l ...
"""

__version__ = "2.0.0.dev0"

from .create_csr_matrix import create_csr_matrix  # re-exported for library use

__all__ = ["create_csr_matrix", "__version__"]
