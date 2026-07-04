"""Regime 2 — reference EXCEEDS one node's VRAM, so it is PARTITIONED by column
(k-mer) across nodes and the matrix-build JOIN is distributed.

Each node holds ONE column-partition of the reference (sized to fit its VRAM) and
streams ALL genomes' k-mers through it; that partition's local join yields one
column-band tile (all genomes x the band, local int32 cols). The full matrix is the
union of the column-band tiles.

Correctness: a join is PARTITION-INVARIANT — splitting the reference by column and
joining each part independently yields EXACTLY the single-node join. So Regime 2 ==
single-node == Regime 1. Validated by forcing many column-partitions on data that
would otherwise fit one node (tmp/dist_test/validate_regime2.py).

Distinction from Regime 1 (multinode.py):
  Regime 1 partitions ROWS (genomes across nodes), replicates the small reference.
  Regime 2 partitions COLUMNS (reference across nodes), streams all genomes.
Here the per-partition joins run in a loop (one process simulating N nodes); on a
cluster each node owns its column-partition and the genome k-mers are shuffled to the
owning partition (a framework join), but the RESULT is identical to this loop.
"""
import json
import os
import numpy as np
import KMX.chunked_output as ck
from .reference import key_cols
from .matrix import build_matrix_csr


def partition_reference(reference, n_parts):
    """Split the reference into n_parts contiguous COLUMN partitions (each must fit a
    node's VRAM). Returns list of (c0, c1) over global column ids (col is 0..n-1)."""
    n = len(reference)
    if n == 0:
        return []
    return ck.plan_bands(np.ones(n, np.int64), budget_nnz=max(1, -(-n // max(1, int(n_parts)))))


def distributed_join(genome_tables, reference, parts, out_dir, *, key=None,
                     data_dtype=None, kmers=None):
    """For each COLUMN partition, JOIN all genomes -> that partition's column-band tile
    (all genomes x the band, local int32 cols). Writes tiles + manifest; returns it."""
    key = key or key_cols(reference)
    n_rows, n_cols = len(genome_tables), len(reference)
    chunks, bands, DT = [], [], np.dtype(data_dtype or np.int64)
    for b, (c0, c1) in enumerate(parts):
        sub = reference.iloc[c0:c1]                              # this node's column slice
        col, dat, ip = build_matrix_csr(genome_tables, sub, key=key)   # the distributed JOIN
        if dat.size and data_dtype is None:
            DT = dat.dtype
        rel = f"band_{b:05d}/block_00000"
        d = os.path.join(out_dir, rel); os.makedirs(d, exist_ok=True)
        np.save(d + "/column.npy", np.ascontiguousarray((col - c0).astype(np.int32)))
        np.save(d + "/data.npy", np.ascontiguousarray(dat.astype(DT)))
        np.save(d + "/row.npy", np.ascontiguousarray(ip, np.int64))
        if kmers is not None:
            bd = os.path.join(out_dir, f"band_{b:05d}"); os.makedirs(bd, exist_ok=True)
            with open(bd + "/kmers.csv", "w") as fh:
                fh.write("index,K-mer\n"); fh.writelines(f"{j},{kmers[c0 + j]}\n" for j in range(c1 - c0))
        chunks.append(dict(id=b, band_id=b, block_id=0, dir=rel, c0=int(c0), c1=int(c1),
                           n_cols=int(c1 - c0), g0=0, g1=int(n_rows), n_rows=int(n_rows), nnz=int(ip[-1])))
        bands.append(dict(id=b, dir=f"band_{b:05d}", c0=int(c0), c1=int(c1),
                          n_cols=int(c1 - c0), n_row_blocks=1, chunk_ids=[b]))
    dt = str(np.dtype(DT))
    man = dict(format="kmx2-chunked-v2", n_rows=int(n_rows), n_cols=int(n_cols),
               n_bands=len(bands), n_chunks=len(chunks), total_nnz=sum(c["nnz"] for c in chunks),
               blocking="2d", regime="regime2", value_semantics="counts",
               index_dtype="int32", data_dtype=dt, dtypes=dict(index="int32", data=dt, indptr="int64"),
               bands=bands, chunks=chunks)
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as fh: json.dump(man, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))
    return man


def build_regime2(genome_tables, reference, out_dir, *, n_parts, key=None,
                  data_dtype=None, kmers=None):
    """DEPRECATED shim — superseded by mechanism_b.build_b (the out-of-core rounds
    engine). Kept for back-compat: builds all `n_parts` column-shards in a single round
    (shards_per_round = n_parts). Returns (manifest, parts) like before.

    Prefer: mechanism_b.build_b(genomes, reference, out_dir, shards_per_round=total_gpus,
            n_shards=plan(...)['n_shards'])  — which streams shards from disk in rounds.
    """
    from . import mechanism_b
    key = key or key_cols(reference)
    man, _ = mechanism_b.build_b(genome_tables, reference, out_dir,
                                 shards_per_round=n_parts, n_shards=n_parts, key=key,
                                 data_dtype=data_dtype, kmers=kmers)
    return man, partition_reference(reference, n_parts)
