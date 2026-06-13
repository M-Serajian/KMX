"""KMX — GPU-accelerated genome × k-mer CSR matrix builder.

Public API
----------
    import KMX

    # high-level: build straight from a manifest CSV (same as the `KMX` CLI)
    KMX.build("manifest.csv", kmer_size=21, tmp_dir="/scratch/tmp", output_dir="out/")
        -> writes the CSR matrix + metadata to out/, and returns
           (data, column, row, kmer_index_df, sparsity)

    # cache the reference once (CPU only, no GPU), then reuse it (skips stage 1)
    KMX.build_reference("manifest.csv", kmer_size=21, tmp_dir="/scratch/tmp",
                        reference_dir="ref/")
    KMX.build("manifest.csv", kmer_size=21, tmp_dir="/scratch/tmp",
              output_dir="out/", reference="ref/")

    # low-level: if you already have parsed-manifest inputs
    KMX.create_csr_matrix(...)

    # huge matrices: split the output into device-sized column chunks (all rows,
    # a band of k-mers each) so a user can stream them on whatever device they
    # have. Returns a manifest; read one chunk with KMX.load_chunk(dir, band_id).
    KMX.chunk_output("out/", "k31_min5_max1000_d0", device_gb=16)

Run from the shell
------------------
    KMX -l manifest.csv -k 21 -t /scratch/tmp -o out/
    python -m KMX -l manifest.csv -k 21 -t /scratch/tmp -o out/

See help(KMX.build) for all arguments.
"""

__version__ = "2.0.0.dev0"

from .cli import build, build_reference             # high-level entry points
from .create_csr_matrix import create_csr_matrix    # low-level builder
from .create_csr_matrix import load_csr             # load output → scipy / cupyx CSR
from .chunked_output import (                        # device-sized 2-D chunk grid
    chunk_existing_output as chunk_output,
    load_chunk, load_chunk_arrays, read_manifest, reconstruct_csr, verify_chunks)

__all__ = ["build", "build_reference", "create_csr_matrix", "load_csr",
           "chunk_output", "load_chunk", "load_chunk_arrays", "read_manifest",
           "reconstruct_csr", "verify_chunks", "__version__"]
