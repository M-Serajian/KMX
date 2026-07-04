"""Mechanism B — sharded reference, column-parallel, OUT-OF-CORE rounds.

The reference is column-sharded to DISK; VRAM (here, RAM) holds only the CURRENT
round's shards (<= shards_per_round = total GPUs). Genomes are STREAMED from disk one
at a time and joined against each resident shard, producing that shard's column-band
tile (all genomes x the shard). Shards are processed in rounds:

    n_rounds = ceil(n_shards / shards_per_round)

Nothing larger than one round's shards + one genome + that round's tiles is ever
resident, so the distinct-k-mer count is bounded by DISK, not VRAM. Shards beyond the
current round simply wait on disk.

Correctness: a join is PARTITION-INVARIANT, so the union of shard tiles == the
single-node matrix regardless of n_shards / n_rounds. The join backend is pluggable
via `join_fn`: CPU pandas by default; a GPU (cuDF) backend plugs in unchanged.

INVARIANT preserved: each genome is COUNTED once (the streamed tables are pre-counted,
cached on disk); rounds RE-READ them, never re-count.
"""
import json
import os
import numpy as np
import pandas as pd
import KMX.chunked_output as ck
from .reference import key_cols


def _load(src):
    """A genome source is either an in-memory table or a parquet path on disk."""
    return pd.read_parquet(src) if isinstance(src, str) else src


def cpu_join(genome_table, subref, key):
    """Default join backend: inner-join one genome against one reference shard ->
    (global column ids, counts) for that genome within the shard."""
    j = genome_table.merge(subref, on=key, how="inner")
    return j["col"].to_numpy(), j["count"].to_numpy()


def shard_reference(reference, shard_dir, n_shards, *, key=None):
    """Write the reference as `n_shards` contiguous COLUMN shards (parquet) to disk;
    each shard sized to fit one GPU. Returns [(shard_id, c0, c1, path)]."""
    key = key or key_cols(reference)
    os.makedirs(shard_dir, exist_ok=True)
    n = len(reference)
    bnds = (ck.plan_bands(np.ones(n, np.int64), budget_nnz=max(1, -(-n // max(1, int(n_shards)))))
            if n else [])
    meta = []
    for sid, (c0, c1) in enumerate(bnds):
        path = os.path.join(shard_dir, f"shard_{sid:05d}.parquet")
        reference.iloc[c0:c1][list(key) + ["col"]].to_parquet(path)
        meta.append((sid, int(c0), int(c1), path))
    return meta


def build_b(genome_sources, reference, out_dir, *, shards_per_round, n_shards=None,
            key=None, join_fn=None, data_dtype=None, kmers=None, shard_dir=None):
    """Out-of-core rounds engine. Shards the reference to disk, then processes the shards
    in rounds of `shards_per_round` (= total GPUs), streaming genomes from disk each
    round. Writes column-band tiles + manifest. Returns (manifest, n_rounds).

    genome_sources : list of pre-counted tables OR parquet paths (streamed, one at a time)
    shards_per_round : how many shards are resident at once (one per GPU)
    n_shards : total column shards (>= shards_per_round); default = shards_per_round
    """
    os.makedirs(out_dir, exist_ok=True)
    key = list(key or key_cols(reference))
    join_fn = join_fn or cpu_join
    n_rows, n_cols = len(genome_sources), len(reference)
    n_shards = max(1, int(n_shards if n_shards is not None else shards_per_round))
    shard_dir = shard_dir or os.path.join(out_dir, "_ref_shards")
    meta = shard_reference(reference, shard_dir, n_shards, key=key)      # reference -> DISK
    DT = np.dtype(data_dtype or np.int64)
    spr = max(1, int(shards_per_round))
    n_rounds = -(-len(meta) // spr)
    chunks, bands, peak_resident = [], [], 0

    for r in range(n_rounds):
        round_meta = meta[r * spr:(r + 1) * spr]
        subrefs = {sid: pd.read_parquet(path) for (sid, c0, c1, path) in round_meta}  # LOAD shards
        peak_resident = max(peak_resident, len(subrefs))
        cols = {sid: [] for sid, *_ in round_meta}
        dats = {sid: [] for sid, *_ in round_meta}
        ips = {sid: [0] for sid, *_ in round_meta}
        for src in genome_sources:                                       # STREAM genomes
            g = _load(src)
            for (sid, c0, c1, _) in round_meta:
                gc, gd = join_fn(g, subrefs[sid], key)
                cols[sid].append(gc); dats[sid].append(gd)
                ips[sid].append(ips[sid][-1] + len(gc))
            del g
        for (sid, c0, c1, _) in round_meta:                             # flush this round's tiles
            col = np.concatenate(cols[sid]) if cols[sid] else np.empty(0, np.int64)
            dat = np.concatenate(dats[sid]) if dats[sid] else np.empty(0, DT)
            ip = np.asarray(ips[sid], np.int64)
            rel = f"band_{sid:05d}/block_00000"; d = os.path.join(out_dir, rel); os.makedirs(d, exist_ok=True)
            np.save(d + "/column.npy", np.ascontiguousarray((col - c0).astype(np.int32)))
            np.save(d + "/data.npy", np.ascontiguousarray(dat.astype(DT)))
            np.save(d + "/row.npy", np.ascontiguousarray(ip, np.int64))
            if kmers is not None:
                bd = os.path.join(out_dir, f"band_{sid:05d}"); os.makedirs(bd, exist_ok=True)
                with open(bd + "/kmers.csv", "w") as fh:
                    fh.write("index,K-mer\n"); fh.writelines(f"{j},{kmers[c0 + j]}\n" for j in range(c1 - c0))
            chunks.append(dict(id=sid, band_id=sid, block_id=0, dir=rel, c0=c0, c1=c1,
                               n_cols=c1 - c0, g0=0, g1=n_rows, n_rows=n_rows, nnz=int(ip[-1])))
            bands.append(dict(id=sid, dir=f"band_{sid:05d}", c0=c0, c1=c1,
                              n_cols=c1 - c0, n_row_blocks=1, chunk_ids=[sid]))
        del subrefs, cols, dats, ips                                    # FREE the round
    chunks.sort(key=lambda c: c["c0"]); bands.sort(key=lambda b: b["c0"])
    dt = str(DT)
    man = dict(format="kmx2-chunked-v2", mechanism="B", n_rows=n_rows, n_cols=n_cols, n_bands=len(bands),
               n_chunks=len(chunks), total_nnz=sum(c["nnz"] for c in chunks), blocking="2d",
               regime="mechanism-B", value_semantics="counts", index_dtype="int32",
               data_dtype=dt, dtypes=dict(index="int32", data=dt, indptr="int64"),
               bands=bands, chunks=chunks, n_rounds=int(n_rounds),
               peak_resident_shards=int(peak_resident))
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as fh: json.dump(man, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))
    return man, n_rounds
