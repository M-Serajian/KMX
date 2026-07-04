"""KMX distributed (Regime 1) — multi-node reference + matrix build.

An OPTIONAL layer on top of the single-node core. It REUSES the core per node and
adds only (a) the one cross-node reduce and (b) per-node tile orchestration.
Install the cross-node backend with the extra:  pip install kmx[spark].

The PLANNER picks one of two mechanisms by whether the filtered reference fits ONE
node's VRAM (both give byte-identical output; the choice is performance):
    A  replicated / row-parallel  — broadcast reference, partition genomes across nodes.
    B  sharded / column-parallel  — column-shard reference to disk, stream genomes
                                    through it in rounds (unbounded distinct k-mers).
Entry point: build(...) routes A or B via plan(). See README for the full map.

Stage map — one module per stage, each independently testable:
    planner.py    plan(): Mechanism A vs B + n_shards / n_rounds / genome groups
    reference.py  Stage 1: count-once -> per-node sum -> reduce -> filter -> reference
    reduce.py     the ONE cross-node op: group-by-kmer SUM ('pandas' | 'spark' backends)
    matrix.py     count-once JOIN -> 2-D tiles (reuses chunked_output)
    multinode.py  Mechanism A: shared bands + per-node row-blocks + concat-only assemble
    mechanism_b.py Mechanism B: disk-sharded reference + out-of-core ROUNDS engine

Invariants (the things that actually matter — keep these true):
    * each genome counted by kmcpy EXACTLY ONCE; the table is reused for the reference
      AND the matrix (matrix = a JOIN). Mechanism B rounds RE-READ tables, never recount.
    * ONE shard fits one GPU — the only hard rule. Past one node = more rounds, never a fail.
    * the per-genome merge is intra-node; only the reduce crosses nodes.

How to debug (start here):
    >>> import KMX.distributed as d
    >>> d.selfcheck(["g0.fna","g1.fna", ...])      # full pipeline on CPU, asserts == single-node
    Then isolate a stage with plain function calls (reduce backend='pandas' is
    deterministic and local). For Mechanism B: mechanism_b.build_b(..., join_fn=...).

Validated == single-node (CPU, incl. real MTB fastq): planner A/B, Mechanism A
(multinode), Mechanism B (out-of-core rounds, all shard/round combos + memory bound),
unified build(). Harnesses in tmp/dist_test/ (val_mechb, val_build, val_real_mechanisms).
Status: correctness complete. TODO = GPU join backend + Spark jar + hardware runs (README).
"""
import os
import tempfile
import numpy as np

from . import planner, reference, reduce as reduce_mod, matrix, multinode, regime2, mechanism_b, robust, runlog
from .planner import plan_regime, plan, layout
from .reference import (count_genome, node_partial, build_reference,
                        decode_reference, key_cols)
from .reduce import reduce_pandas, reduce_spark
from .matrix import build_matrix_csr, write_tiles, reconstruct

__all__ = ["planner", "reference", "reduce_mod", "matrix", "multinode", "regime2",
           "mechanism_b", "robust", "runlog", "plan_regime", "plan", "layout",
           "count_genome", "node_partial", "build_reference", "decode_reference",
           "key_cols", "reduce_pandas", "reduce_spark", "build_matrix_csr",
           "write_tiles", "reconstruct", "build_distributed", "build_multinode",
           "build", "assemble_final", "selfcheck"]


def build_distributed(genome_files, *, k, min_count, max_count, out_dir,
                      input_fmt="mfasta", canonical=True, device_gb=None,
                      budget_bytes=None):
    """Single-process Regime-1 orchestrator: count-once -> reference -> matrix tiles.
    Returns (reference_df, chunks_dir).

    Multi-node: run count_genome + node_partial PER NODE, write each node's partial
    to parquet, pass the parquet paths to reduce_spark for the cross-node reduce,
    broadcast the reference, then call build_matrix_csr + write_tiles per node with a
    global row offset; assembly is manifest aggregation (rows tile globally). See README.
    """
    os.makedirs(out_dir, exist_ok=True)
    tables = [count_genome(f, k, input_fmt=input_fmt, canonical=canonical)
              for f in genome_files]                       # COUNT ONCE
    key = key_cols(tables[0])
    ref = build_reference([node_partial(tables, key)], min_count=min_count,
                          max_count=max_count, key=key)     # Stage 1
    col, dat, ip = build_matrix_csr(tables, ref, key=key)   # Stage 2 (reuse tables)
    chunks = os.path.join(out_dir, "chunks")
    write_tiles(col, dat, ip, len(ref), chunks, device_gb=device_gb,
                budget_bytes=budget_bytes, kmers=decode_reference(ref, k))  # Stage 3
    return ref, chunks


def build_multinode(node_genomes, *, k, min_count, max_count, out_dir, n_bands=8,
                   input_fmt="mfasta", canonical=True, reduce_fn=None):
    """Regime-1 MULTI-NODE driver: ties the validated pieces into one call.

    node_genomes = list of per-node genome-file lists. Per node: count-once +
    node_partial; cross-node reduce (reduce_fn; pass reduce_spark on a cluster) ->
    reference; per node: count-once JOIN -> write row-block tiles in SHARED bands at
    a global offset; assemble -> one manifest. Returns (reference_df, out_dir).

    Run here in one process (simulating N nodes); on a cluster, run the per-node
    blocks on each node and use reduce_spark for the cross-node reduce.
    """
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    node_tables = [[count_genome(f, k, input_fmt=input_fmt, canonical=canonical) for f in gl]
                   for gl in node_genomes]
    key = key_cols(node_tables[0][0])
    partials = [node_partial(tabs, key) for tabs in node_tables]
    ref = build_reference(partials, min_count=min_count, max_count=max_count, key=key,
                          reduce_fn=(reduce_fn or reduce_pandas))
    kmers = decode_reference(ref, k)
    bands = multinode.plan_shared_bands(len(ref), n_bands=n_bands)
    node_recs, offset, DT = [], 0, np.int64
    for nid, (gl, tabs) in enumerate(zip(node_genomes, node_tables)):
        col, dat, ip = build_matrix_csr(tabs, ref, key=key)
        DT = dat.dtype if dat.size else np.int64
        node_recs.append(multinode.write_node(col, dat, ip, bands, out_dir, node_id=nid,
                         row_offset=offset, data_dtype=DT, kmers=(kmers if nid == 0 else None)))
        offset += len(gl)
    multinode.assemble(out_dir, n_rows=offset, n_cols=len(ref), bands=bands,
                       node_recs=node_recs, data_dtype=DT)
    return ref, out_dir


def assemble_final(out_dir):
    """THE single, method-agnostic final assembly. Works on ANY method's output —
    Mechanism A (row-block tiles), Mechanism B (column-band tiles), single-process, or
    multi-node — because every method writes the same self-describing `kmx2-chunked-v2`
    manifest. Verifies tile integrity, reconstructs the full CSR, and reassembles the
    global k-mer legend, returning ONE coherent dataset. Raises if integrity fails.

    Returns: {indices, data, indptr, n_rows, n_cols, kmers, mechanism, format}
      indices/data/indptr = the full CSR; kmers[col] = the k-mer string for each column.
    """
    import KMX.chunked_output as ck
    m = ck.read_manifest(out_dir)
    ok, issues = ck.verify_chunks(out_dir)
    if not ok:
        raise ValueError(f"chunk integrity check FAILED for {out_dir}: {issues[:3]}")
    col, data, indptr, n_cols = ck.reconstruct_csr(out_dir, manifest=m)
    return dict(indices=col, data=data, indptr=indptr, n_rows=int(m["n_rows"]),
                n_cols=int(n_cols), kmers=ck.read_kmers(out_dir, manifest=m),
                mechanism=m.get("mechanism", m.get("regime", "?")), format=m["format"])


def _split(items, n):
    """Partition a list into n contiguous near-equal groups (genome -> node assignment)."""
    n = max(1, int(n)); q, r = divmod(len(items), n); out, i = [], 0
    for g in range(n):
        sz = q + (1 if g < r else 0); out.append(items[i:i + sz]); i += sz
    return [grp for grp in out if grp]


def build(genome_files, *, k, min_count, max_count, out_dir,
          n_nodes=1, n_gpus_per_node=1, node_vram_gb=24, n_words=1,
          input_fmt="mfasta", canonical=True, data_dtype=None, cache_dir=None):
    """Unified distributed driver. Count-once -> reference -> the PLANNER picks:

      Mechanism A (reference fits one node): replicate the reference, partition genomes
        across nodes (row-parallel), shard the reference across each node's GPUs ->
        row-block tiles (reuses the validated multinode path).
      Mechanism B (reference exceeds one node): column-shard the reference to disk and
        stream genomes through the shards in ceil(n_shards/total_gpus) rounds
        (mechanism_b.build_b) -> column-band tiles.

    Both emit chunked_output tiles and reconstruct identically. Returns (reference, out_dir, plan).
    If cache_dir is given, counted tables are written there as parquet and STREAMED from
    disk in Mechanism B (true out-of-core)."""
    os.makedirs(out_dir, exist_ok=True)
    tables, sources = [], []
    for i, f in enumerate(genome_files):
        t = count_genome(f, k, input_fmt=input_fmt, canonical=canonical)   # COUNT ONCE
        tables.append(t)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            p = os.path.join(cache_dir, f"g{i:06d}.parquet"); t.to_parquet(p); sources.append(p)
        else:
            sources.append(t)
    key = key_cols(tables[0])
    ref = build_reference([node_partial(tables, key)], min_count=min_count,
                          max_count=max_count, key=key)
    kmers = decode_reference(ref, k)
    DT = data_dtype or (tables[0]["count"].dtype if "count" in tables[0] else np.int64)
    p = plan(len(ref), n_nodes=n_nodes, n_gpus_per_node=n_gpus_per_node,
             node_vram_gb=node_vram_gb, n_words=n_words)

    if p["mechanism"] == "A":
        groups = _split(list(range(len(tables))), p["n_genome_groups"])
        bands = multinode.plan_shared_bands(len(ref), n_bands=p["shards_per_node"])
        recs, off = [], 0
        for nid, idxs in enumerate(groups):
            tt = [tables[i] for i in idxs]
            c, da, ipp = build_matrix_csr(tt, ref, key=key)
            recs.append(multinode.write_node(c, da, ipp, bands, out_dir, node_id=nid,
                        row_offset=off, data_dtype=DT, kmers=(kmers if nid == 0 else None)))
            off += len(tt)
        multinode.assemble(out_dir, n_rows=len(tables), n_cols=len(ref), bands=bands,
                           node_recs=recs, data_dtype=DT)
    else:
        mechanism_b.build_b(sources, ref, out_dir, shards_per_round=p["total_gpus"],
                            n_shards=p["n_shards"], key=key, data_dtype=DT, kmers=kmers)
    return ref, out_dir, p


def selfcheck(genome_files, *, k=31, min_count=2, max_count=8, budget_bytes=20000,
              verbose=True):
    """Run the full distributed pipeline on CPU and assert it equals the single-node
    matrix (pooled reference + per-genome merge), by k-mer string. Returns True on PASS.
    This is the first thing to run when debugging."""
    import kmcpy
    out = tempfile.mkdtemp()
    ref, chunks = build_distributed(genome_files, k=k, min_count=min_count,
                                    max_count=max_count, out_dir=out, budget_bytes=budget_bytes)
    ref_str = decode_reference(ref, k)
    ri, rd, rp, _ = reconstruct(chunks)
    DIST = [dict(zip(ref_str[ri[int(rp[r]):int(rp[r + 1])]],
                     rd[int(rp[r]):int(rp[r + 1])].astype(int).tolist()))
            for r in range(len(genome_files))]
    # independent single-node ground truth
    ref_gt = kmcpy.count_kmers(list(genome_files), tmp_dir=tempfile.mkdtemp(), k=k,
                               input_fmt="mfasta", canonical=True, min_count=min_count,
                               max_count=max_count, decoded=False, drop_count=False)
    key = key_cols(ref_gt)
    GT = []
    for f in genome_files:
        t = count_genome(f, k)
        j = t.merge(ref_gt[key], on=key, how="inner")
        dec = kmcpy.decode_kmers(j[key], k=k)["kmer"].to_numpy()
        GT.append(dict(zip(dec, j["count"].astype(int).tolist())))
    mism = sum(1 for r in range(len(genome_files)) if GT[r] != DIST[r])
    ok = (mism == 0 and len(ref) == len(ref_gt))
    if verbose:
        print(f"[selfcheck] genomes={len(genome_files)} reference={len(ref):,} "
              f"(gt {len(ref_gt):,})  rows_differing={mism}  -> "
              f"{'PASS' if ok else 'FAIL'}")
    return ok
