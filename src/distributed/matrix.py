"""Stage 2 (matrix build) + Stage 3 (tiles) — by COUNT-ONCE JOIN, never a recount.

Stage 2: join each genome's already-counted table (from Stage 1) against the
         reference on the packed key -> (row, col, count). The matrix is the SAME
         counted entries, filtered + column-relabelled. No second count.
Stage 3: write per-node row-block tiles via the core chunked-output (the 2-D grid),
         tagged with a global row offset so a node's genomes occupy the right rows.
         Assembly across nodes is manifest aggregation (rows tile globally).

Validated end-to-end == single-node matrix: tmp/dist_test/validate_stage23.py
"""
import numpy as np
import KMX.chunked_output as ck
from .reference import key_cols


def build_matrix_csr(genome_tables, reference, *, key=None):
    """JOIN reused per-genome counts with the reference -> CSR arrays
    (column ids, counts, row pointer). One row per genome, in input order."""
    key = key or key_cols(reference)
    idx, dat, ip = [], [], [0]
    for t in genome_tables:                       # reuse the count-once tables
        j = t.merge(reference, on=key, how="inner")
        idx.append(j["col"].to_numpy())
        dat.append(j["count"].to_numpy())
        ip.append(ip[-1] + len(j))
    indices = np.concatenate(idx) if idx else np.empty(0, np.int64)
    data = np.concatenate(dat) if dat else np.empty(0)
    return indices, data, np.asarray(ip, np.int64)


def write_tiles(indices, data, indptr, n_cols, out_dir, *, device_gb=None,
                budget_bytes=None, kmers=None, data_dtype=None):
    """Write this node's matrix as device-sized 2-D tiles (reuses chunked_output).
    Pass device_gb (per the consumption device) or an explicit budget_bytes."""
    item = (data_dtype or data.dtype)
    item_bytes = int(np.dtype(item).itemsize)
    if budget_bytes is None:
        if device_gb is None:
            raise ValueError("pass device_gb or budget_bytes")
        budget_bytes = float(device_gb) * (1024 ** 3) * 0.8
    grid = ck.plan_grid(indices, data, indptr, n_cols,
                        budget_bytes=budget_bytes, item_bytes=item_bytes)
    return ck.write_chunks(indices=indices, data=data, indptr=indptr, n_cols=n_cols,
                           out_dir=out_dir, grid=grid, kmers=kmers,
                           data_dtype=(data_dtype or data.dtype))


def reconstruct(chunks_dir):
    """Rebuild the full CSR from tiles (for validation / whole-matrix consumers)."""
    return ck.reconstruct_csr(chunks_dir)
