"""KMX v2 — build the (genome × k-mer) CSR matrix using KMC-DataFrame (kmcpy).

Pipeline:
  Stage 1 (one call, all CPUs as threads): kmcpy.count_kmers on the
    flattened list of every input file across every genome -> packed-uint64
    keys for the global reference set, pre-filtered by KMC at the C++ layer
    (min_count, max_count). No counts are transferred (drop_count=True).

  Stage 2 (one process per *genome* via ProcessPoolExecutor with
    `threads=1`): each worker hands kmcpy the full file list belonging to
    one sample_id. KMC reads all of that genome's files in a single call
    and merges k-mer counts internally, so paired-end pairs and multi-run
    genomes collapse into one DataFrame per genome with no extra merge
    code on the Python side. The main process moves each result to GPU as
    a cuDF, merges with the global reference to recover the column index,
    and appends to the CSR (data, column) arrays in cupy.

  Stage 3: decode the global keys back to ACGT strings for the user-facing
    set_of_all_unique_kmers CSV.

The function is driven by a parsed Manifest (see src/manifest.py): the
caller is responsible for validating paths and detecting the format
family before calling in.

Returns: (data, column, row, kmer_index_df, sparsity).
"""

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd

import kmcpy


def _count_one_genome(args):
    """Worker for the stage-2 pool. CPU-only — never imports cupy/cudf.

    args: (sample_id, files, kmer_size, tmp_dir, canonical, max_count,
           max_ram_gb, input_fmt)
    Returns a pandas DataFrame with columns kmer_0..kmer_{n-1} (uint64) +
    count (uint32). When ``files`` has multiple paths, KMC merges their
    counts internally into a single DataFrame.
    """
    (sample_id, files, kmer_size, tmp_dir, canonical,
     max_count, max_ram_gb, input_fmt) = args
    return kmcpy.count_kmers(
        files,
        tmp_dir=tmp_dir,
        k=kmer_size,
        input_fmt=input_fmt,
        threads=1,
        max_ram_gb=max_ram_gb,
        canonical=canonical,
        min_count=1,
        max_count=max_count,
        decoded=False,
        drop_count=False,
    )


def create_csr_matrix(
    sample_ids: List[str],
    files_by_sample: Dict[str, List[str]],
    flat_files: List[str],
    input_fmt: str,
    kmer_size: int,
    tmp_dir: str,
    min_val: int = 1,
    max_val: int = 10**9,
    disable_normalization: bool = False,
    threads: int = 0,
    max_ram_gb_per_worker: int = 2,
    max_ram_gb_stage1: int = 8,
    max_ram_gb: float = 0.0,
):
    # GPU imports are *here*, not at module top, so spawn'd workers don't pull
    # cupy/cudf when they re-import this module.
    import cudf
    import cupy as cp

    canonical = not disable_normalization
    n_cpus    = threads if threads and threads > 0 else (os.cpu_count() or 4)
    n_genomes = len(sample_ids)
    n_files   = len(flat_files)

    print(f"[create_csr_matrix] genomes={n_genomes}  files={n_files}  "
          f"k={kmer_size}  min={min_val}  max={max_val}  "
          f"canonical={canonical}  input_fmt={input_fmt}  "
          f"n_cpus={n_cpus}", flush=True)

    # ── Stage 1: global reference k-mer set ──────────────────────────────────
    t0 = time.time()
    print(f"[stage 1] kmcpy on {n_files} combined inputs (across "
          f"{n_genomes} genomes)  threads={n_cpus}  "
          f"max_ram_gb={max_ram_gb_stage1}", flush=True)
    ref_pdf = kmcpy.count_kmers(
        flat_files,
        tmp_dir=tmp_dir,
        k=kmer_size,
        input_fmt=input_fmt,
        threads=n_cpus,
        max_ram_gb=max_ram_gb_stage1,
        canonical=canonical,
        min_count=min_val,
        max_count=max_val,
        decoded=False,
        drop_count=True,
    )
    n_unique = len(ref_pdf)
    n_words  = sum(1 for c in ref_pdf.columns if c.startswith("kmer_"))
    key_cols = [f"kmer_{i}" for i in range(n_words)]
    print(f"[stage 1] unique kmers={n_unique}  in {time.time()-t0:.1f}s",
          flush=True)

    if n_unique == 0:
        # Edge case: no k-mer survives the global filter.
        empty_idx_df = pd.DataFrame({"index": [], "K-mer": []})
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.uint32),
            np.zeros(n_genomes + 1, dtype=np.int64),
            empty_idx_df,
            100.0,
        )

    # Move reference keys to GPU and tag every row with its column index.
    # KMC radix-sorts lexicographically, so the row order in ref_pdf IS the
    # column ordering of the resulting matrix.
    ref_gdf = cudf.DataFrame.from_pandas(ref_pdf)
    ref_gdf["__col_idx__"] = cp.arange(n_unique, dtype=cp.uint32)

    # ── Stage 2: per-genome counts in parallel (each CPU = one genome) ───────
    #
    # Strategy:
    #   * Submit one job per genome. Each job receives the genome's full
    #     file list (1 file for assemblies / single-end runs, 2+ for
    #     paired-end, 4+ for multi-lane). KMC merges the per-file counts
    #     internally — we get one DataFrame per genome, no Python-side
    #     reduction.
    #   * As workers finish (in *arbitrary* order), pull the result and
    #     immediately send it through the GPU merge against the reference
    #     set. This keeps the GPU busy while remaining workers count.
    #   * Stash each genome's (col_idx, count) cupy arrays in a dict keyed
    #     by the *manifest* sample index.
    #   * After all genomes are done, assemble the CSR with rows in the
    #     SAME order as the manifest's first-appearance sample_id order,
    #     so the output row index matches the genome_index_<suffix>.csv
    #     mapping.
    t0 = time.time()
    print(f"[stage 2] pool max_workers={n_cpus}  threads-per-worker=1  "
          f"max_ram_gb-per-worker={max_ram_gb_per_worker}  "
          f"(out-of-order GPU merge)", flush=True)

    work = [
        (sid, files_by_sample[sid], kmer_size, tmp_dir, canonical,
         max_val, max_ram_gb_per_worker, input_fmt)
        for sid in sample_ids
    ]

    # per_genome[g_idx] = (col_idx_NUMPY_uint32, count_NUMPY_float32)
    # Tier-1 spill: each genome's merge result is moved off the GPU into
    # CPU RAM as numpy arrays immediately after the merge, so peak GPU
    # memory stays at ~(reference + one merge's scratch) regardless of
    # how many genomes are in flight.
    per_genome = {}

    # spawn (not fork) so the parent's CUDA context can't be inherited.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_cpus, mp_context=ctx) as pool:
        # Map each Future back to its *manifest* sample index so we can
        # reassemble in input order at the end.
        futures = {
            pool.submit(_count_one_genome, args): g_idx
            for g_idx, args in enumerate(work)
        }

        completed = 0
        for fut in as_completed(futures):
            g_idx = futures[fut]
            df_g  = fut.result()

            # Move to GPU and merge against the reference to recover
            # the column index for each surviving k-mer.
            gdf    = cudf.DataFrame.from_pandas(df_g)
            merged = gdf.merge(ref_gdf, on=key_cols, how="inner")
            # Spill to CPU RAM as numpy arrays, then drop the GPU-side
            # cudf/cupy objects so VRAM doesn't grow with genome count.
            idx_np = cp.asnumpy(
                merged["__col_idx__"].values.astype(cp.uint32))
            val_np = cp.asnumpy(
                merged["count"].values.astype(cp.float32))
            per_genome[g_idx] = (idx_np, val_np)

            del gdf, merged, df_g
            cp.get_default_memory_pool().free_all_blocks()

            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == n_genomes:
                print(f"[stage 2]   completed={completed}/{n_genomes}  "
                      f"(sample_id={sample_ids[g_idx]!r})", flush=True)

    # Assemble final CSR on the CPU (numpy) — no GPU memory spike here.
    total_nnz = sum(int(idx.size) for idx, _ in per_genome.values())
    col_buf   = np.empty(total_nnz, dtype=np.uint32)
    val_buf   = np.empty(total_nnz, dtype=np.float32)
    row_ptr   = np.empty(n_genomes + 1, dtype=np.int64)
    row_ptr[0] = 0
    cur = 0
    for g_idx in range(n_genomes):
        idx, val = per_genome[g_idx]
        sz       = int(idx.size)
        col_buf[cur:cur + sz] = idx
        val_buf[cur:cur + sz] = val
        cur += sz
        row_ptr[g_idx + 1] = cur
    del per_genome  # free the spilled numpy arrays

    print(f"[stage 2] total nnz={cur}  in {time.time()-t0:.1f}s  "
          f"(rows reordered to match manifest order; final CSR on CPU)",
          flush=True)

    # ── Stage 3: decode reference keys to ACGT for the user-facing CSV ──────
    decoded = kmcpy.decode_kmers(ref_pdf, k=kmer_size)
    decoded.insert(0, "index", np.arange(n_unique, dtype=np.int64))
    decoded.rename(columns={"kmer": "K-mer"}, inplace=True)

    sparsity = 100.0 * (1.0 - cur / float(n_genomes * n_unique))

    return (
        val_buf[:cur],   # numpy float32, on CPU
        col_buf[:cur],   # numpy uint32,  on CPU
        row_ptr,         # numpy int64,   on CPU
        decoded,
        sparsity,
    )
