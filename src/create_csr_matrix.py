"""KMX v2 — build the (genome × k-mer) CSR matrix using KMC-DataFrame (kmcpy).

Pipeline:
  Stage 1 (one call, all CPUs as threads): kmcpy.count_kmers on the
    flattened list of every input file across every genome -> packed-uint64
    keys for the global reference set, pre-filtered by KMC at the C++ layer
    (min_count, max_count). No counts are transferred (drop_count=True).
    Stage 1 runs alone, so the RAM budget hands it (almost) all available
    memory (see ram_budget.py).

  Stage 2 (one process per *genome* via ProcessPoolExecutor with
    `threads=1`): each worker hands kmcpy the full file list belonging to
    one sample_id. KMC reads all of that genome's files in a single call
    and merges k-mer counts internally, so paired-end pairs and multi-run
    genomes collapse into one DataFrame per genome with no extra merge
    code on the Python side. The main process moves each result to GPU as
    a cuDF, merges with the global reference to recover the column index,
    and appends to the CSR (data, column) arrays in cupy. Each worker's KMC
    RAM target is the available memory split across the concurrent workers.

  Stage 3: decode the global keys back to ACGT strings for the user-facing
    set_of_all_unique_kmers CSV.

Memory tiers:
  * Tier-1 (always on): every per-genome merge result is moved off the GPU
    into CPU RAM immediately, so peak VRAM stays at ~(reference + one
    merge's scratch) regardless of genome count.
  * Tier-2 (auto / --max-ram-gb): the host-resident CSR accumulator is
    flushed to disk chunk files under <tmp_dir> once it exceeds the budget's
    spill threshold, so the main process RAM stays bounded for large cohorts.

The function is driven by a parsed Manifest (see src/manifest.py): the
caller is responsible for validating paths and detecting the format
family before calling in.

Returns: (data, column, row, kmer_index_df, sparsity).
"""

import contextlib
import hashlib
import json
import os
import sys
import time
import uuid
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from typing import Dict, List

import numpy as np
import pandas as pd

import kmcpy

from .ram_budget import (plan_ram_budget, plan_vram_shards, plan_gpu_layout,
                         available_cpus)

# Set KMX_VERBOSE=1 to see KMC's raw per-genome "Stage 1/2: …%" output and the
# per-file EOF warnings (useful for debugging; very noisy at 1000 genomes).
_VERBOSE = bool(int(os.environ.get("KMX_VERBOSE", "0") or 0))

# Stage-2 watchdog. KMC's ProcessStage2() can intermittently deadlock (a worker
# never returns from count_kmers); a plain as_completed() would then hang the
# whole run forever. We instead wait with a per-round deadline: if no genome
# finishes within this many seconds while work remains, a worker is wedged — we
# kill the pool and resubmit the stuck genomes to a fresh pool. The hang is a
# race, so a retry almost always succeeds. A single genome counts in well under
# a second (tens of seconds even under heavy node contention), so a 90s window
# cleanly separates "wedged" from "slow". A false positive is cheap anyway — the
# genome is just recounted — so erring low costs at most a recount, never
# correctness. Tunable via KMX_STAGE2_DEADLINE_S.
_STAGE2_DEADLINE_S  = float(os.environ.get("KMX_STAGE2_DEADLINE_S", "45"))
_MAX_BLOCK_ATTEMPTS = int(os.environ.get("KMX_STAGE2_MAX_ATTEMPTS", "8"))
# How often to poll while waiting, so a stall shows a "still working" heartbeat
# instead of a frozen progress bar (the watchdog still fires at the full deadline).
_HEARTBEAT_S        = 5.0


@contextlib.contextmanager
def _quiet_kmc(active=True):
    """Mute KMC's in-process C++ progress by redirecting fds 1 & 2 to /dev/null.

    KMC prints "Stage 1/2: …%" and per-file EOF warnings straight to the
    process's file descriptors, so at 1000 genomes × 16 workers the terminal is
    flooded. This silences it at the fd level (Python logging is unaffected).
    A no-op under KMX_VERBOSE=1.
    """
    if not active or _VERBOSE:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


class _Progress:
    """Tiny dependency-free progress bar: a live bar on a TTY, tidy periodic
    lines when output is a file/pipe (no carriage-return spam)."""

    def __init__(self, total, label, stream=None, width=26):
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self.n = 0
        self.t0 = time.time()
        self._last = 0.0
        self.stream = stream if stream is not None else sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def _line(self):
        frac = self.n / self.total
        elapsed = time.time() - self.t0
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        fill = int(self.width * frac)
        bar = "█" * fill + "·" * (self.width - fill)
        w = len(str(self.total))
        return (f"  {self.label} |{bar}| {self.n:>{w}}/{self.total} "
                f"{frac * 100:3.0f}%  {rate:5.0f}/s  ETA {eta:4.0f}s")

    def update(self, k=1):
        self.n += k
        if self.tty:
            now = time.time()
            if now - self._last < 0.1 and self.n < self.total:
                return
            self._last = now
            self.stream.write("\r" + self._line())
            self.stream.flush()
        else:
            step = max(1, self.total // 20)
            if self.n == 1 or self.n == self.total or self.n % step == 0:
                self.stream.write(self._line() + "\n")
                self.stream.flush()

    def note(self, msg):
        """Emit a one-off message without leaving a half-drawn bar behind."""
        if self.tty:
            self.stream.write("\r\033[K")
        self.stream.write(msg + "\n")
        self.stream.flush()

    def close(self):
        if self.tty:
            self.stream.write("\r" + self._line() + "\n")
            self.stream.flush()


def _count_one_genome(args):
    """Worker for the stage-2 pool. CPU-only — never imports cupy/cudf.

    args: (sample_id, files, kmer_size, tmp_dir, canonical, max_count,
           max_ram_gb, input_fmt, spill_dir, spill_bytes, kmc_threads)

    Counts one genome's k-mers (uint64 packed) + count via KMC, then hands the
    result back to the parent. Hand-off is **in memory by default**; only when
    the result is large (>= spill_bytes) is it written to a unique Parquet file
    under spill_dir and a path returned instead — so many big per-genome tables
    (e.g. chr19 at high k, ~1 GB each) can't pile up in the parent's RAM at once.

    Returns ("mem", DataFrame)  or  ("spill", parquet_path).
    """
    (sample_id, files, kmer_size, tmp_dir, canonical,
     max_count, max_ram_gb, input_fmt, spill_dir, spill_bytes, kmc_threads) = args
    with _quiet_kmc():   # mute KMC's per-genome C++ progress (KMX_VERBOSE=1 to keep)
        df = kmcpy.count_kmers(
            files,
            tmp_dir=tmp_dir,
            k=kmer_size,
            input_fmt=input_fmt,
            threads=kmc_threads,   # leftover CPUs given to each worker (n_cpus//n_workers)
            max_ram_gb=max_ram_gb,
            canonical=canonical,
            min_count=1,
            max_count=max_count,
            decoded=False,
            drop_count=False,
        )
    if spill_bytes and int(df.memory_usage(index=False, deep=True).sum()) >= spill_bytes:
        # Unique name (pid + uuid4) -> never collides across workers, genomes,
        # or watchdog retries. Written fully before the path is returned, so the
        # parent only ever reads a complete file.
        path = os.path.join(spill_dir, f"handoff_{os.getpid()}_{uuid.uuid4().hex}.parquet")
        try:
            df.to_parquet(path, index=False)
            del df
            _reclaim_memory()                  # drop KMC arena + the spilled table
            return ("spill", path)
        except Exception:                       # disk full / IO error -> degrade
            try: os.remove(path)
            except OSError: pass               # to in-memory rather than failing
    # Hand back the table, but first return KMC's freed-but-retained sort/decode
    # arenas to the OS, so this pooled worker's RSS drops to ~the table size
    # before it picks up the next genome instead of lingering at the count-time
    # peak (glibc keeps freed arenas; that retention let pooled workers climb).
    _reclaim_memory()
    return ("mem", df)


def _materialize(result):
    """Resolve a worker hand-off (in-memory df, or a spilled Parquet path) into a
    DataFrame, deleting the spill file once loaded. Round-trip is exact for the
    integer k-mer/count columns, so the assembled matrix is identical either way."""
    kind, payload = result
    if kind == "mem":
        return payload
    df = pd.read_parquet(payload)
    try: os.remove(payload)
    except OSError: pass
    return df


def _buf_spill_one(spill_root: str, g_idx: int, idx_np, val_np) -> str:
    """Spill ONE genome's (idx, val) to disk for the streaming reorder buffer.

    Used only when an out-of-order genome must wait in the buffer and the buffer
    has grown past its cap — the farthest-from-the-write-frontier genome is
    parked on disk and read back (in input order) when its turn comes."""
    path = os.path.join(spill_root, f"buf_{g_idx}.npz")
    np.savez(path, idx=idx_np, val=val_np)
    return path


def _buf_load_one(path: str):
    """Read back a buffer-spilled genome and delete its file."""
    with np.load(path) as z:
        idx_np = z["idx"]; val_np = z["val"]
    try: os.remove(path)
    except OSError: pass
    return idx_np, val_np


def _spill_chunk(fragments, tmp_dir: str, chunk_no: int) -> str:
    """Flush a list of accumulator fragments to a packed .npz chunk; return path.

    ``fragments`` is a list of ``(g_idx, idx_int, val)``. A chunk
    stores, per fragment: its genome's manifest index (``gidxs``), the
    fragment's nnz (``sizes``), and the concatenated column indices (``idx``)
    and values (``val``). A genome may appear in several fragments (one per
    VRAM column-block), so ``gidxs`` may repeat — the cursor-based replay
    handles that. Concatenating keeps it to four arrays per chunk.
    """
    gidxs = np.fromiter((g for g, _, _ in fragments), dtype=np.int64,
                        count=len(fragments))
    sizes = np.fromiter((idx.size for _, idx, _ in fragments), dtype=np.int64,
                        count=len(fragments))
    big_idx = (np.concatenate([idx for _, idx, _ in fragments]) if fragments
               else np.empty(0, dtype=np.int32))
    big_val = (np.concatenate([val for _, _, val in fragments]) if fragments
               else np.empty(0, dtype=np.float32))
    path = os.path.join(tmp_dir, f"kmx_csr_chunk_{chunk_no:05d}.npz")
    np.savez(path, gidxs=gidxs, sizes=sizes, idx=big_idx, val=big_val)
    return path


def _scatter(col_buf, val_buf, cursor, g_idx, idx, val):
    """Append one fragment into a genome's CSR row at its running cursor.

    ``cursor[g_idx]`` is the next free slot inside genome ``g_idx``'s row range;
    it is advanced after each write. This lets fragments arrive in any order and
    lets a genome receive several fragments (column-sharded build), while the
    final row order still matches the manifest via the precomputed row pointer.
    """
    start = int(cursor[g_idx])
    sz = int(idx.size)
    col_buf[start:start + sz] = idx
    val_buf[start:start + sz] = val
    cursor[g_idx] = start + sz


def _max_genome_bytes(files_by_sample) -> int:
    """Largest genome's total input size in bytes (gzip counted ~5x its content).

    Used only to estimate the per-merge transient when planning VRAM shards;
    a rough upper bound is fine since the OOM path is a safety net.
    """
    best = 0
    for files in files_by_sample.values():
        tot = 0
        for f in files:
            try:
                sz = os.path.getsize(f)
            except OSError:
                continue
            if f.endswith(".gz"):
                sz *= 5
            tot += sz
        if tot > best:
            best = tot
    return best


def _proc_rss_gb() -> float:
    """This process's resident set size in GB (Linux /proc), or -1 if unknown."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except Exception:
        return -1.0


def _children_rss_gb() -> float:
    """Summed RSS (GB) of all descendant processes of this one (the stage-2
    workers). Used to attribute a memory climb to the parent vs. the workers —
    i.e. to tell a KMX2 (cuDF parent) leak apart from a kmcpy (worker) leak."""
    page = os.sysconf("SC_PAGE_SIZE")
    me = os.getpid()
    kids = {}
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat") as f:
                    data = f.read()
                ppid = int(data[data.rfind(")") + 2:].split()[1])
                with open(f"/proc/{name}/statm") as f:
                    rss = int(f.read().split()[1]) * page
            except (OSError, ValueError, IndexError):
                continue
            kids[int(name)] = (ppid, rss)
    except OSError:
        return -1.0
    # sum everything whose ancestry leads back to me
    children = {}
    for pid, (ppid, _) in kids.items():
        children.setdefault(ppid, []).append(pid)
    total, stack, seen = 0, list(children.get(me, ())), set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if pid in kids:
            total += kids[pid][1]
            stack.extend(children.get(pid, ()))
    return total / (1024 ** 3)


# ── Streaming .npy writer ────────────────────────────────────────────────────
# A standard .npy is header + raw data, and the header stores the array shape.
# To append rows to the output *before* the final nonzero count is known (the
# no-reload streaming assembly), we write a FIXED-LENGTH header up front with a
# placeholder shape, append the data sequentially, then rewrite the header
# in place once the true length is known. The header length is padded to a
# multiple of 64 (a .npy requirement) and large enough for any shape, so the
# final rewrite is exactly the same number of bytes — no data is ever moved.
_NPY_HDR_BYTES = 128   # magic(6)+ver(2)+hlen(2)+dict(118) ; 128 % 64 == 0


def _npy_write_header(fp, dtype, n: int) -> None:
    """(Re)write a fixed-length .npy v1.0 header for a 1-D array of length n."""
    import struct
    descr = np.lib.format.dtype_to_descr(np.dtype(dtype))
    body = "{'descr': %r, 'fortran_order': False, 'shape': (%d,), }" % (descr, n)
    hlen = _NPY_HDR_BYTES - 10                      # bytes after the 10-byte preamble
    pad = hlen - len(body) - 1                      # -1 for the trailing newline
    if pad < 0:
        raise ValueError("npy header reserve too small for shape")
    body = body + (" " * pad) + "\n"
    fp.seek(0)
    fp.write(b"\x93NUMPY\x01\x00")
    fp.write(struct.pack("<H", hlen))
    fp.write(body.encode("latin1"))


def _npy_open_stream(path: str, dtype):
    """Create a .npy with a placeholder header; return a file ready for append."""
    fp = open(path, "wb+")
    _npy_write_header(fp, dtype, 0)                 # leaves fp at the data offset
    return fp


def _npy_finalize_stream(fp, dtype, n: int) -> None:
    """Rewrite the header with the true length and close. Data is left intact."""
    fp.flush()
    _npy_write_header(fp, dtype, n)
    fp.flush()
    os.fsync(fp.fileno())
    fp.close()


def _flush_release(*arrs) -> None:
    """Flush a memmap's dirty pages to disk AND drop them from the page cache.

    Scattering a ~21 GB CSR into a memmap fills RAM with dirty pages (the page
    cache counts against the cgroup), which is what pushed assembly to the 32 GB
    edge. Flushing writes them out; MADV_DONTNEED then releases them, so the
    assembly's resident footprint stays at ~one chunk instead of the whole
    output. Re-access (rare, only if a row spans chunks) just re-faults from
    disk — no data loss, since flush ran first.
    """
    import mmap as _m
    for a in arrs:
        try:
            a.flush()
        except Exception:
            pass
        mm = getattr(a, "_mmap", None)
        if mm is not None:
            try:
                mm.madvise(_m.MADV_DONTNEED)
            except Exception:
                pass


def _reclaim_memory() -> None:
    """Return freed heap pages to the OS.

    Stage 1's KMC makes large C++/pybind11 allocations; when they're freed,
    glibc keeps the arenas (process RSS stays high), which then collides with
    stage 2's worker processes and OOMs a tight box. A gc + malloc_trim(0)
    hands the pages back so stage 2 starts from a clean baseline.
    """
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def estimate_worker_df_gb(files_by_sample, kmer_size: int) -> float:
    """Estimate the RAM one stage-2 worker holds for its genome's k-mer table.

    A worker counts ONE genome and returns its packed k-mer table; that table
    (not spillable while being produced) sets, together with KMC's sort buffer,
    the per-worker footprint the budget uses to choose how many workers run at
    once. n_words = ceil(k/32) (KMC packs 32 two-bit bases per uint64); a row is
    the packed key (8*n_words) plus a 4-byte column index. #rows is bounded by
    the genome's base count ~ its file bytes (one distinct k-mer per base, an
    upper bound). Single source of truth shared by the CLI log and the pipeline.
    """
    n_words = (kmer_size + 31) // 32
    bytes_per_row = 8 * n_words + 4
    return _max_genome_bytes(files_by_sample) * bytes_per_row / (1024 ** 3)


def _looks_like_oom(exc) -> bool:
    """Heuristic: did this exception come from running out of GPU memory?"""
    s = (str(exc) + " " + type(exc).__name__).lower()
    return any(k in s for k in (
        "out of memory", "outofmemory", "bad_alloc", "memoryerror",
        "rmm", "cudaerrormemoryallocation", "cuda_error_out_of_memory"))


# Per-genome GPU merge transient ≈ this many × the genome table bytes
# (the device DataFrame + the join hash table + the output). Used to decide,
# BEFORE the merge, whether it fits the *currently free* VRAM.
_MERGE_VRAM_MULT   = 3.0
_MERGE_VRAM_SAFETY = 0.80     # leave headroom for fragmentation/driver


def _col_index_dtype(n_unique):
    """Pick the column-index dtype dynamically from the reference size.

    The column index holds values in [0, n_unique). We use SIGNED integers so the
    arrays load straight into ``scipy.sparse`` and ``cupyx.scipy.sparse`` (RAPIDS)
    without dtype-conversion copies or errors (both expect int32/int64; unsigned
    types are not their native index dtype). int32 (≤ 2,147,483,647) covers the
    vast majority of references compactly; only when the reference genuinely
    exceeds that do we widen to int64. .npy files self-describe their dtype, so
    consumers load the right type with no metadata. ``KMX_COL_DTYPE`` forces it
    (e.g. to test the int64 path).
    """
    env = os.environ.get("KMX_COL_DTYPE")
    if env:
        return np.dtype(env)
    return (np.dtype(np.int32) if int(n_unique) <= np.iinfo(np.int32).max
            else np.dtype(np.int64))


def _cpu_merge_block(df_g, block_pdf, key_cols, binarize=False):
    """Exact CPU (pandas) inner join — identical result to the cuDF merge."""
    merged = df_g.merge(block_pdf, on=key_cols, how="inner")  # block_pdf has __col_idx__
    # Keep the column-index dtype the block was built with (int32 or int64) —
    # block_pdf["__col_idx__"] carries it, so the merge output already has it.
    idx = merged["__col_idx__"].to_numpy().astype(block_pdf["__col_idx__"].dtype)
    # binarize → presence/absence (uint8 1s); else the float32 count.
    val = (np.ones(idx.size, dtype=np.int8) if binarize
           else merged["count"].to_numpy().astype(np.float32))
    return idx, val


def _merge_genome_block(df_g, ref_gdf_b, block_pdf, key_cols, cudf, cp, oom_state,
                        binarize=False):
    """Merge one genome's counts against one reference column-block.

    Returns ``(col_idx_np, val_np)`` for the k-mers of this
    genome that fall in this block. Runs on whichever device is **safe**: the
    GPU (cuDF inner join, fast) when this genome's transient fits the currently
    free VRAM with margin, otherwise an exact CPU (pandas) merge. The GPU path
    also keeps a reactive fallback, so a misestimate still can't crash the run.
    Both devices produce identical ``(col_idx, count)`` — same inner join.
    """
    # ── Routing: KMX_CPU_MERGE forces CPU; else GPU only if it comfortably fits ─
    use_gpu = not os.environ.get("KMX_CPU_MERGE")
    if use_gpu:
        try:
            free_vram = int(cp.cuda.runtime.memGetInfo()[0])
            per_row = 8 * len(key_cols) + 4        # n_words uint64 keys + uint32 count
            need = int(len(df_g)) * per_row * _MERGE_VRAM_MULT
            if need > free_vram * _MERGE_VRAM_SAFETY:
                use_gpu = False                    # too big for VRAM → CPU (safe)
        except Exception:
            pass                                   # can't probe → just try GPU

    if use_gpu:
        try:
            gdf = cudf.DataFrame.from_pandas(df_g)
            merged = gdf.merge(ref_gdf_b, on=key_cols, how="inner")
            # Preserve the block's column-index dtype (int32 or int64).
            idx = cp.asnumpy(merged["__col_idx__"].values.astype(
                block_pdf["__col_idx__"].dtype))
            val = (np.ones(idx.size, dtype=np.int8) if binarize
                   else cp.asnumpy(merged["count"].values.astype(cp.float32)))
            del gdf, merged
            cp.get_default_memory_pool().free_all_blocks()
            return idx, val
        except Exception as exc:  # noqa: BLE001 — narrowed by _looks_like_oom
            if not _looks_like_oom(exc):
                raise
            cp.get_default_memory_pool().free_all_blocks()   # reactive fallback

    # CPU path: chosen proactively, or after a GPU-OOM fallback.
    oom_state["cpu_fallbacks"] += 1
    return _cpu_merge_block(df_g, block_pdf, key_cols, binarize)


def _force_kill_pool(pool):
    """Terminate every worker process of a (presumed-wedged) ProcessPoolExecutor.

    ProcessPoolExecutor cannot cancel a task already running inside a deadlocked
    worker, so the only way to recover is to kill the worker processes outright;
    the genomes they were counting are then resubmitted to a fresh pool.
    """
    procs = list(getattr(pool, "_processes", {}).values())
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(timeout=5)
        except Exception:
            pass
    for p in procs:
        try:
            if p.is_alive():
                p.kill()
        except Exception:
            pass


def _completed_genomes(work_items, indices, n_cpus, ctx, sample_ids,
                       deadline_s=_STAGE2_DEADLINE_S,
                       max_attempts=_MAX_BLOCK_ATTEMPTS,
                       worker_fn=None, progress=None):
    """Yield (genome_index, worker_result) for each genome, deadlock-proof.

    Runs ``worker_fn`` (default :func:`_count_one_genome`) over ``indices`` in a
    spawn pool and yields results as they complete — preserving the CPU-count /
    GPU-merge overlap, since the consumer does its GPU work between yields while
    the remaining workers keep counting.

    Watchdog: if no genome completes within ``deadline_s`` while work remains, a
    worker is wedged inside KMC (see module note). We kill that pool's workers
    and resubmit the unfinished genomes to a fresh pool. After ``max_attempts``
    rounds with no progress we give up loudly. Genomes that finished before a
    hang are never recounted.
    """
    if worker_fn is None:
        worker_fn = _count_one_genome
    remaining = list(indices)
    attempt = 0
    while remaining:
        attempt += 1
        if attempt > max_attempts:
            raise RuntimeError(
                f"[stage 2] {len(remaining)} genome(s) hung in KMC across "
                f"{max_attempts} retries (e.g. sample_id="
                f"{sample_ids[remaining[0]]!r}); likely a KMC ProcessStage2 "
                f"deadlock that reproduces deterministically on this input.")
        pool = ProcessPoolExecutor(max_workers=n_cpus, mp_context=ctx)
        fut_to_g = {pool.submit(worker_fn, work_items[i]): i for i in remaining}
        pending = set(fut_to_g)
        done_genomes = set()
        hung = False
        abort = False
        try:
            while pending:
                # Poll in short steps so a stall shows a heartbeat instead of a
                # frozen bar; only declare a hang after `deadline_s` of silence.
                done = set()
                stalled = 0.0
                while pending and not done and stalled < deadline_s:
                    step = min(_HEARTBEAT_S, deadline_s - stalled)
                    done, pending = wait(pending, timeout=step,
                                         return_when=FIRST_COMPLETED)
                    if not done:
                        stalled += step
                        if progress is not None and abs(stalled - _HEARTBEAT_S) < 1e-6:
                            progress.note(
                                f"      … still working — waiting on {len(pending)} "
                                f"genome(s); a hang is auto-recovered in "
                                f"{deadline_s - stalled:.0f}s")
                if not done:                 # nobody finished within the deadline
                    hung = True
                    break
                for fut in done:
                    g_idx = fut_to_g.pop(fut)   # stop retaining this future…
                    result = fut.result()
                    done_genomes.add(g_idx)
                    yield g_idx, result
                    # …and drop the result + future once the consumer has merged
                    # it. A Future caches its return value (fut._result), so left
                    # in fut_to_g every genome's ~1 GB table would stay pinned for
                    # the whole block — the parent's ~1 GB/genome leak.
                    del result, fut
                done.clear()
        except GeneratorExit:
            # The consumer abandoned us (e.g. the merge stage errored and broke
            # out of its loop, dropping this generator). Do NOT let the finally's
            # pool.shutdown(wait=True) block until every still-running count
            # finishes — on a large manifest that is the entire stage-2 counting
            # time, which read as a "hang" when aborting a failed run. Kill the
            # pool's workers and propagate immediately.
            abort = True
            raise
        finally:
            if hung or abort:
                _force_kill_pool(pool)
            pool.shutdown(wait=not (hung or abort))
        remaining = [i for i in remaining if i not in done_genomes]
        if hung:
            preview = ", ".join(str(sample_ids[i]) for i in remaining[:3])
            msg = (f"      ⚠ watchdog: no genome finished in {deadline_s:.0f}s "
                   f"with {len(remaining)} left (KMC deadlock) — killed workers, "
                   f"retrying (attempt {attempt + 1}); stuck e.g.: {preview}")
            if progress is not None:
                progress.note(msg)
            else:
                print(msg, flush=True)


def _shard_ranges(n, k):
    """Split ``[0, n)`` into exactly ``k`` contiguous ranges of near-equal size.

    Used to shard the reference across a group's GPUs (one shard per GPU). Sizes
    differ by at most one row, so the GPUs in a group are balanced.
    """
    base, extra = divmod(int(n), int(k))
    out, lo = [], 0
    for i in range(int(k)):
        sz = base + (1 if i < extra else 0)
        out.append((lo, lo + sz))
        lo += sz
    return out


def _gpu_merge_worker(gpu_id, ref_parquet_path, block_ranges, key_cols,
                      in_q, out_q, cpu_merge, col_dtype, binarize):
    """One merge worker pinned to a single GPU.

    Holds one or more reference column-BLOCKS — ``block_ranges`` is a list of
    ``(lo, hi)`` global-column slices this worker owns — resident on its GPU. For
    each genome pulled from ``in_q`` it merges the genome against EVERY block it
    owns and pushes one fragment per block, ``(g_idx, block_lo, col_idx, count)``,
    to ``out_q``. The collector reassembles a genome's row from its fragments once
    every block has reported (the completeness gather).

    Used two ways:
      * data-parallel      — each worker owns the single full reference block and a
        genome is sent to ONE worker (one fragment per genome).
      * reference-parallel — the reference is sharded; each worker owns a subset of
        blocks and a genome is BROADCAST to all workers (B fragments per genome).

    The payload is either a hand-off tuple (resolved by ``_materialize``) or an
    already-materialized DataFrame — broadcast pre-materializes once in the parent
    so a spilled hand-off file isn't read-and-deleted by several workers.

    With ``cpu_merge=True`` it runs the exact pandas merge and never imports
    cupy/cudf, so the whole orchestration is testable on a GPU-less box.
    """
    # Keep host thread pools to 1 in this worker (same rationale as the count
    # workers): avoid a per-core OpenBLAS/OMP pool in every merge process.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

    if cpu_merge:
        os.environ["KMX_CPU_MERGE"] = "1"     # force the exact pandas merge path
        cudf = cp = None
    else:
        # Pin THIS worker to its GPU BEFORE importing cupy, so it sees exactly one
        # device (its own) as cuda:0 — a fresh CUDA context (spawn, not inherited).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        import cudf
        import cupy as cp

    # Load only THIS worker's blocks. Read the full reference once, slice out the
    # owned blocks (tagged with GLOBAL column indices), upload each to the GPU,
    # then free the full host copy — VRAM holds only the shard, the whole point.
    full = pd.read_parquet(ref_parquet_path)
    blocks = []                                # (lo, hi, block_pdf, ref_gdf)
    for (lo, hi) in block_ranges:
        bp = full.iloc[lo:hi].copy()
        bp["__col_idx__"] = np.arange(lo, hi, dtype=col_dtype)
        gdf = None if cpu_merge else cudf.DataFrame.from_pandas(bp)
        blocks.append((lo, hi, bp, gdf))
    del full
    _reclaim_memory()
    oom_state = {"cpu_fallbacks": 0}
    # Fault-injection hook (testing only): KMX_TEST_KILL_AFTER=N makes the worker
    # pinned to device "0" die SILENTLY (no __error__/__done__) after N genomes.
    # Kept as a regression trigger for the full abort path: the collector's
    # dead-worker watchdog must surface it, the count pool must be force-killed
    # (not drained), surviving workers SIGKILL-reaped, and the process must EXIT
    # promptly — never hang at CUDA teardown. Inert unless the env var is set.
    _kill_after = int(os.environ.get("KMX_TEST_KILL_AFTER", "0") or 0)
    _processed = 0

    while True:
        item = in_q.get()
        if item is None:                      # sentinel → drain done, exit
            break
        if _kill_after and str(gpu_id) == "0" and _processed >= _kill_after:
            os._exit(137)                     # simulate SIGKILL: no error, no done
        _processed += 1
        g_idx, payload, exp = item            # exp = fragments the row needs total
        try:
            df_g = payload if isinstance(payload, pd.DataFrame) \
                else _materialize(payload)
            for (lo, hi, bp, gdf) in blocks:  # one fragment per owned block
                idx_np, val_np = _merge_genome_block(
                    df_g, gdf, bp, key_cols, cudf, cp, oom_state, binarize)
                out_q.put((g_idx, lo, idx_np, val_np, exp))
            del df_g
        except Exception as exc:              # surface to collector; never hang
            out_q.put(("__error__", g_idx, repr(exc)))
        if not cpu_merge:
            try:
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        _reclaim_memory()
    out_q.put(("__done__", gpu_id, oom_state["cpu_fallbacks"]))


def _build_streaming_multigpu(*, out_dir, out_suffix, ref_pdf, n_unique, n_words,
                              col_dtype, val_dtype, binarize,
                              key_cols, kmer_size, blocks, work, sample_ids,
                              n_genomes, budget, ctx, spill_root, handoff_dir,
                              spill_bytes, gpu_layout, oom_state, progress,
                              cpu_merge, t0):
    """Multi-GPU streaming assembly (input-order rows, no reload).

    Handles both multi-GPU modes off the same machinery:
      * ``data_parallel``      — reference replicated per GPU, genomes load-balanced
        across GPUs (one fragment per genome).
      * ``reference_parallel`` — reference column-blocks sharded across GPUs, each
        genome BROADCAST to all GPUs; a row is assembled from its B per-shard
        fragments. A row is released to the writer ONLY once all its fragments
        have arrived (completeness gather), so a row can never be truncated.

    Pipeline (one-directional, no cross-stage barriers):

        CPU count pool ─▶ in_q(s) ─▶ D GPU workers ─▶ out_q ─▶ collector ─▶ .npy
                                       (own block(s))          (gather+reorder, input order)

    Rows are written in input order via the reorder buffer, so the on-disk matrix
    is identical to the single-GPU build. Returns
    ``(None, None, row_ptr, decoded, sparsity)``.
    """
    import queue
    import shutil
    import threading

    os.makedirs(out_dir, exist_ok=True)
    D = gpu_layout.n_gpus
    grouped = (gpu_layout.mode == "reference_parallel")

    ref_path = os.path.join(spill_root, f"_ref_replica_{out_suffix}.parquet")
    ref_pdf.to_parquet(ref_path, index=False)

    # Bound the output queue so a transient collector stall can't let the GPU
    # workers pile up unbounded fragments in RAM. The collector drains it
    # continuously (and never blocks on the workers), so this is pure backpressure
    # — workers block on put when full and resume as it drains; no deadlock. A
    # dead collector is handled by the worker-join timeout + terminate below.
    out_q = ctx.Queue(maxsize=max(16, 4 * D))
    # Build the worker set as a device mesh. `group_qs[r]` are the per-worker queues
    # of replica group r — a genome is broadcast to all of group r's workers and
    # gathered from them. Data-parallel/single is the degenerate mesh: one shared
    # queue, every worker holds the full reference, genomes load-balanced by pull.
    if grouped:
        # R×S mesh: each group holds the FULL reference sharded across its own GPUs
        # (exactly 1 shard / GPU → no OOM). Genomes are data-parallel across groups
        # (round-robin) and reference-parallel within a group (broadcast + gather).
        group_qs, specs = [], []
        for grp in gpu_layout.groups:
            ranges = _shard_ranges(n_unique, len(grp))   # one shard per GPU in grp
            qs = []
            for j, gid in enumerate(grp):
                q = ctx.Queue(maxsize=3)
                qs.append(q)
                specs.append((gid, [ranges[j]], q))      # worker holds exactly 1 shard
            group_qs.append(qs)
        in_qs = [q for (_, _, q) in specs]
    else:
        shared = ctx.Queue(maxsize=max(4, 3 * D))        # load-balanced pull
        specs = [(gpu_layout.gpu_ids[d], list(blocks), shared) for d in range(D)]
        group_qs = None
        in_qs = [shared]

    workers = []
    for (gid, block_ranges, q) in specs:
        p = ctx.Process(target=_gpu_merge_worker,
                        args=(gid, ref_path, block_ranges, key_cols, q, out_q,
                              cpu_merge, col_dtype, binarize),
                        daemon=True)
        p.start()
        workers.append(p)

    # ── Streaming in-order writer (same policy as the single-GPU path) ────────
    col_path = os.path.join(out_dir, f"column_{out_suffix}.npy")
    val_path = os.path.join(out_dir, f"data_{out_suffix}.npy")
    col_fp = _npy_open_stream(col_path, col_dtype)
    val_fp = _npy_open_stream(val_path, val_dtype)
    sizes = np.zeros(n_genomes, dtype=np.int64)
    buf = {}            # g_idx -> (idx_np, val_np), full row waiting in RAM
    buf_spill = {}      # g_idx -> path, parked on disk (overflow)
    gather = {}         # g_idx -> [(block_lo, idx_np, val_np), …] partial fragments
    buf_cap = spill_bytes if spill_bytes > 0 else (1 << 30)
    state = {"next_write": 0, "total_nnz": 0, "written_since_sync": 0,
             "received": 0, "buf_bytes": 0, "error": None}

    def _emit(gi, idx_np, val_np):
        sz = int(idx_np.size)
        if sz:
            col_fp.write(memoryview(np.ascontiguousarray(idx_np, dtype=col_dtype)))
            val_fp.write(memoryview(np.ascontiguousarray(val_np, dtype=val_dtype)))
            sizes[gi] = sz
            state["total_nnz"] += sz
            state["written_since_sync"] += sz * 8

    def _drain():
        nw = state["next_write"]
        while nw < n_genomes and (nw in buf or nw in buf_spill):
            if nw in buf:
                idx_np, val_np = buf.pop(nw)
                state["buf_bytes"] -= (idx_np.nbytes + val_np.nbytes)
            else:
                idx_np, val_np = _buf_load_one(buf_spill.pop(nw))
            _emit(nw, idx_np, val_np)
            nw += 1
        state["next_write"] = nw

    def _row_ready(g_idx, idx_full, val_full):
        """A genome's full row is assembled → buffer it for in-order writing."""
        buf[g_idx] = (idx_full, val_full)
        state["buf_bytes"] += idx_full.nbytes + val_full.nbytes
        _drain()
        while state["buf_bytes"] > buf_cap and buf:        # park farthest genomes
            gmax = max(buf)
            i2, v2 = buf.pop(gmax)
            state["buf_bytes"] -= (i2.nbytes + v2.nbytes)
            buf_spill[gmax] = _buf_spill_one(spill_root, gmax, i2, v2)
        state["received"] += 1
        progress.update()
        if state["written_since_sync"] >= (1 << 30):       # bound page cache
            for _fp in (col_fp, val_fp):
                _fp.flush()
                try:
                    os.fsync(_fp.fileno())
                    os.posix_fadvise(_fp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
            state["written_since_sync"] = 0

    def _ingest(g_idx, block_lo, idx_np, val_np, exp):
        """Gather a genome's per-shard fragments; release the row only when ALL
        ``exp`` shards (its group's size) have reported — never write a partial row.
        ``exp`` rides on each fragment, so no shared state across threads."""
        lst = gather.get(g_idx)
        if lst is None:
            lst = []
            gather[g_idx] = lst
        lst.append((block_lo, idx_np, val_np))
        if len(lst) >= exp:                    # completeness: every shard in
            del gather[g_idx]
            if len(lst) == 1:
                idx_full, val_full = lst[0][1], lst[0][2]
            else:                              # stitch shards in column order
                lst.sort(key=lambda t: t[0])
                idx_full = np.concatenate([t[1] for t in lst])
                val_full = np.concatenate([t[2] for t in lst])
            _row_ready(g_idx, idx_full, val_full)

    def _collect():
        """Drain fragments; gather + stream rows to disk in input order.

        Watchdog for a SILENTLY dead worker (SIGKILL/host-OOM/driver crash — no
        ``__error__``): a clean worker sends exactly one ``__done__`` before
        exiting, so if more workers are dead than have signaled ``__done__`` while
        rows remain, one died without finishing. This catches the case where a
        single worker dies and the others are still alive — which the global
        all-alive check misses, and which would otherwise HANG ``reference_parallel``
        (per-worker queues: the dead worker's queue stops draining and dispatch
        blocks on it forever). Setting ``state["error"]`` here also unblocks the
        stuck dispatch, whose ``_put`` checks it each loop.
        """
        done = 0
        try:
            while state["received"] < n_genomes:
                try:
                    item = out_q.get(timeout=10.0)
                except queue.Empty:
                    n_dead = sum(1 for p in workers if not p.is_alive())
                    if n_dead > done:        # a worker died without signaling done
                        state["error"] = RuntimeError(
                            f"a GPU merge worker died before finishing "
                            f"({n_dead} dead, {done} clean exit(s)) — "
                            f"{n_genomes - state['received']} row(s) unmerged")
                        return
                    continue
                tag = item[0]
                if tag == "__error__":
                    state["error"] = RuntimeError(
                        f"GPU merge worker failed on genome "
                        f"{sample_ids[item[1]]!r}: {item[2]}")
                    return
                if tag == "__done__":
                    done += 1
                    continue
                g_idx, block_lo, idx_np, val_np, exp = item
                _ingest(g_idx, block_lo, idx_np, val_np, exp)
        except Exception as exc:               # noqa: BLE001
            state["error"] = exc

    collector = threading.Thread(target=_collect, daemon=True)
    collector.start()

    def _put(q, item):                         # backpressure put + liveness check
        while True:
            if state["error"] is not None:
                return
            try:
                q.put(item, timeout=5.0)
                return
            except queue.Full:
                if not any(p.is_alive() for p in workers):
                    state["error"] = RuntimeError(
                        "all GPU merge workers exited before the run finished")
                    return

    # ── Dispatch: drive the count pool, feed completed genomes to the GPUs ────
    if grouped:
        R = len(group_qs)
        rr = 0
        for g_idx, res in _completed_genomes(work, range(n_genomes),
                                             budget.n_workers, ctx, sample_ids,
                                             progress=progress):
            if state["error"] is not None:
                break
            grp_qs = group_qs[rr % R]          # round-robin genome → replica group
            rr += 1
            # Materialize ONCE, then broadcast a copy to that group's workers (a
            # spilled hand-off file isn't read+deleted by several workers).
            df_g = _materialize(res)
            exp = len(grp_qs)                  # this row needs `exp` shard fragments
            for q in grp_qs:
                _put(q, (g_idx, df_g, exp))
            del df_g
    else:
        for g_idx, res in _completed_genomes(work, range(n_genomes),
                                             budget.n_workers, ctx, sample_ids,
                                             progress=progress):
            if state["error"] is not None:
                break
            _put(in_qs[0], (g_idx, res, 1))    # shared queue, 1 fragment per genome

    if state["error"] is None:
        # Clean finish: send one stop sentinel per worker so each drains and exits.
        for q in in_qs:                        # one per dedicated queue…
            try:
                q.put(None, timeout=30)
            except queue.Full:
                pass
        if not grouped:                        # …the shared queue feeds D workers
            for _ in range(D - 1):
                try:
                    in_qs[0].put(None, timeout=30)
                except queue.Full:
                    pass
    else:
        # A worker died or errored: a graceful sentinel would block on the dead
        # worker's full queue. Terminate the workers hard instead so cleanup can't
        # hang — we're aborting and about to raise anyway.
        for p in workers:
            try:
                p.terminate()
            except Exception:
                pass

    collector.join()
    # Reap workers, escalating to SIGKILL. A worker wedged inside a CUDA driver
    # call ignores SIGTERM (p.terminate), so a plain terminate would leave it
    # alive — and because the workers are daemons, multiprocessing's atexit
    # handler then force-joins them with NO timeout, hanging the whole process
    # at interpreter shutdown (observed: the error is reported, then the run
    # hangs to the wall-clock limit). SIGKILL (p.kill) is the only reliable
    # escalation, so every worker is dead before we return and atexit finds
    # nothing to join. On the clean path the workers already exited on their
    # sentinel, so the first join returns immediately and nothing escalates.
    grace = 30 if state["error"] is None else 5
    for p in workers:
        p.join(timeout=grace)
        if p.is_alive():
            try:
                p.terminate()                  # SIGTERM
            except Exception:
                pass
            p.join(timeout=5)
        if p.is_alive():
            try:
                p.kill()                       # SIGKILL — survives a wedged CUDA call
            except Exception:
                pass
            p.join(timeout=5)
    # Don't let a parent-side queue feeder thread block process exit: if a worker
    # died with items still queued (error path), its queue's feeder is stuck
    # flushing to a pipe with no reader, and Python would join it forever at exit.
    # cancel_join_thread() drops that join. Harmless on the clean path (all
    # delivered → nothing to flush). out_q's feeders live in the workers (gone).
    for q in in_qs:
        try:
            q.cancel_join_thread()
        except Exception:
            pass

    try:
        os.remove(ref_path)
    except OSError:
        pass
    shutil.rmtree(handoff_dir, ignore_errors=True)

    if state["error"] is not None:
        # On failure, DELETE the partial column/data files rather than finalizing
        # them — a finalized-but-truncated .npy looks like a valid (shorter) matrix
        # and could be mistaken for a smaller-but-complete result. Removing them
        # leaves no readable artifact; the run also exits non-zero and never writes
        # row_*.npy, so a reload would fail loudly.
        for _fp, _path in ((col_fp, col_path), (val_fp, val_path)):
            try:
                _fp.close()
            except Exception:
                pass
            try:
                os.remove(_path)
            except OSError:
                pass
        raise state["error"]

    _drain()
    if state["next_write"] != n_genomes:
        raise RuntimeError(
            f"[stage 2] multi-GPU streaming wrote {state['next_write']} of "
            f"{n_genomes} rows; a genome never completed")
    if gather:
        raise RuntimeError(
            f"[stage 2] {len(gather)} genome(s) never received all of their "
            f"reference-shard fragment(s) — incomplete row(s), aborting")

    _npy_finalize_stream(col_fp, col_dtype, state["total_nnz"])
    _npy_finalize_stream(val_fp, val_dtype, state["total_nnz"])
    progress.close()

    total_nnz = state["total_nnz"]
    row_ptr = np.empty(n_genomes + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(sizes, out=row_ptr[1:])
    sparsity = (100.0 * (1.0 - total_nnz / float(n_genomes * n_unique))
                if n_unique else 100.0)
    _mode = (f"{D}× GPU reference-parallel ({len(gpu_layout.groups)} group(s) × "
             f"~{gpu_layout.n_blocks} shards)" if grouped
             else f"{D}× GPU data-parallel")
    print(f"      → {total_nnz:,} nonzeros · {sparsity:.1f}% sparse  "
          f"({time.time()-t0:.1f}s · {_mode}, streamed, input order)", flush=True)
    print("[3/3] decoding k-mers …", flush=True)
    with _quiet_kmc():
        decoded = kmcpy.decode_kmers(ref_pdf, k=kmer_size)
    # column j ↔ ref_pdf row j (the positional __col_idx__). kmcpy.decode_kmers
    # is fully vectorised and maps each row positionally — it does NOT sort or
    # reorder — so decoded row j IS column j's k-mer. Assert the 1:1 length so a
    # future decode change that drops/reorders rows fails loudly, not silently.
    if len(decoded) != n_unique:           # if/raise (survives python -O, unlike assert)
        raise RuntimeError(
            f"decode_kmers returned {len(decoded)} rows for {n_unique} reference "
            f"k-mers — column↔k-mer legend would be misaligned")
    decoded.insert(0, "index", np.arange(n_unique, dtype=np.int64))
    decoded.rename(columns={"kmer": "K-mer"}, inplace=True)
    return (None, None, row_ptr, decoded, sparsity)


# ── Reference set: build (stage 1) + on-disk cache ──────────────────────────
# The reference is the COLUMN space: the global set of unique k-mers, stored as
# KMC's packed 2-bit keys (columns kmer_0..kmer_{n_words-1}, uint64 — 32 two-bit
# bases per word). Stage 2 merges genomes against exactly these columns, so the
# cached form needs no decoding. Building it (stage 1) reads the whole dataset
# once and uses NO GPU, so it can run on a CPU node and be reused.
_REFERENCE_PARQUET = "reference.parquet"
_REFERENCE_META    = "reference_meta.json"
_REFERENCE_FORMAT  = "kmx2-reference-v1"


def _count_reference(flat_files, input_fmt, kmer_size, tmp_dir, min_val, max_val,
                     canonical, n_cpus, stage1_gb):
    """Stage 1: build the global packed-2bit k-mer reference set. CPU only.

    Returns ``ref_pdf`` — columns ``kmer_0..kmer_{n_words-1}`` of packed uint64
    keys (KMC's 2-bit-per-base encoding), pre-filtered by ``min_val``/``max_val``
    at the C++ layer, no counts (``drop_count``). Never imports cupy/cudf.
    """
    with _quiet_kmc():
        ref_pdf = kmcpy.count_kmers(
            flat_files, tmp_dir=tmp_dir, k=kmer_size, input_fmt=input_fmt,
            threads=n_cpus, max_ram_gb=int(stage1_gb), canonical=canonical,
            min_count=min_val, max_count=max_val, decoded=False, drop_count=True)
    return ref_pdf


def _file_fingerprint(path) -> str:
    """Content fingerprint of one file: size + sha256 of its first and last 64 KiB.

    Keyed on CONTENT, not path or mtime, so it is **stable under stage-in** (the
    same data copied to node-local scratch with fresh paths/mtimes fingerprints
    identically — no false cache rebuild), while still catching real content
    changes (different bytes → different hash) and added/removed files (the set of
    fingerprints changes). It does NOT read the whole file, so it stays cheap even
    for very large inputs. The only miss is a change confined to the middle of a
    file that preserves its exact size and both 64 KiB ends — vanishingly unlikely
    for real sequence files.
    """
    blk = 1 << 16
    try:
        sz = os.path.getsize(path)
    except OSError:
        return "missing"
    h = hashlib.sha256()
    h.update(str(sz).encode())
    try:
        with open(path, "rb") as fh:
            h.update(fh.read(blk))
            if sz > 2 * blk:
                fh.seek(-blk, os.SEEK_END)
                h.update(fh.read(blk))
    except OSError:
        return f"unreadable:{sz}"
    return h.hexdigest()


def _files_signature(flat_files) -> str:
    """Order-independent CONTENT signature of the input file set.

    A cached reference is valid only for the exact file *contents* it was built
    from. We hash the sorted per-file content fingerprints (see
    :func:`_file_fingerprint`) — not basenames, paths, or mtimes — so the cache
    survives restaging/moves of identical data (no false rebuild) but rejects any
    content or file-set change (which would silently merge genomes against the
    wrong column space).
    """
    parts = sorted(_file_fingerprint(f) for f in flat_files)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def save_reference(reference_dir, ref_pdf, kmer_size, min_val, max_val,
                   canonical, flat_files) -> str:
    """Write the packed-2bit reference + metadata sidecar to ``reference_dir``."""
    os.makedirs(reference_dir, exist_ok=True)
    ref_pdf.to_parquet(os.path.join(reference_dir, _REFERENCE_PARQUET), index=False)
    meta = {
        "format":          _REFERENCE_FORMAT,
        "encoding":        "packed-2bit-uint64",   # KMC 2-bit keys, mergeable by stage 2
        "kmer_size":       int(kmer_size),
        "min_count":       int(min_val),
        "max_count":       int(max_val),
        "canonical":       bool(canonical),
        "n_unique":        int(len(ref_pdf)),
        "n_words":         int(sum(1 for c in ref_pdf.columns if c.startswith("kmer_"))),
        "key_cols":        [c for c in ref_pdf.columns if c.startswith("kmer_")],
        "n_files":         int(len(flat_files)),
        "files_signature": _files_signature(flat_files),
    }
    with open(os.path.join(reference_dir, _REFERENCE_META), "w") as fh:
        json.dump(meta, fh, indent=2)
    return reference_dir


def load_reference(reference_dir):
    """Load a cached reference. Returns ``(ref_pdf, meta_dict)``."""
    meta_path = os.path.join(reference_dir, _REFERENCE_META)
    pq_path   = os.path.join(reference_dir, _REFERENCE_PARQUET)
    if not (os.path.exists(meta_path) and os.path.exists(pq_path)):
        raise FileNotFoundError(
            f"no cached reference in {reference_dir!r} (expected "
            f"{_REFERENCE_PARQUET} + {_REFERENCE_META}). Build one with "
            f"--build-reference / KMX.build_reference().")
    with open(meta_path) as fh:
        meta = json.load(fh)
    return pd.read_parquet(pq_path), meta


def validate_reference(meta, kmer_size, min_val, max_val, canonical, flat_files):
    """STRICT check that a cached reference matches this run. Raises on any
    mismatch — a silently-wrong reference would corrupt the whole matrix."""
    expected = {
        "kmer_size": int(kmer_size), "min_count": int(min_val),
        "max_count": int(max_val), "canonical": bool(canonical),
    }
    mism = [(k, meta.get(k), v) for k, v in expected.items() if meta.get(k) != v]
    sig = _files_signature(flat_files)
    if meta.get("files_signature") != sig:
        mism.append(("input file set", "(cached set)", "(current set differs)"))
    if mism:
        lines = "\n".join(f"    {k}: reference={a!r}  this run={b!r}"
                          for k, a, b in mism)
        raise ValueError(
            "cached reference does not match this run (strict validation):\n"
            f"{lines}\n"
            "  Rebuild it with --build-reference, or fix the parameters to match.")


def load_csr(output_dir, *, suffix=None, device="cpu", shape=None):
    """Load a KMX output as a ready-to-compute sparse CSR matrix.

    Reads the ``column``/``data``/``row`` ``.npy`` arrays from a ``build()`` /
    ``KMX -o`` run and returns a CSR matrix with the correct
    ``(n_genomes × n_unique)`` shape and native dtypes (int32/int64 indices,
    float32 values), so it plugs straight into downstream computation with NO
    conversion:

      * ``device="cpu"`` → ``scipy.sparse.csr_matrix`` (scipy / scikit-learn /
        statsmodels — e.g. ``sklearn.feature_selection.chi2``).
      * ``device="gpu"`` → ``cupyx.scipy.sparse.csr_matrix`` (RAPIDS — cuML / cuPy).

    ``suffix`` is auto-detected when the directory holds exactly one matrix;
    otherwise pass it (e.g. ``"k31_min5_max1000_d0"`` or ``..._presence``). The
    column count comes from ``set_of_all_unique_kmers_<suffix>.csv`` (the full
    reference width, so all-zero trailing columns aren't dropped); pass
    ``shape=(n_rows, n_cols)`` to override.
    """
    import glob
    if suffix is None:
        cols = glob.glob(os.path.join(output_dir, "column_*.npy"))
        if len(cols) != 1:
            raise ValueError(
                f"pass suffix=… — found {len(cols)} matrices in {output_dir!r}")
        suffix = os.path.basename(cols[0])[len("column_"):-len(".npy")]
    column = np.load(os.path.join(output_dir, f"column_{suffix}.npy"))
    data   = np.load(os.path.join(output_dir, f"data_{suffix}.npy"))
    row    = np.load(os.path.join(output_dir, f"row_{suffix}.npy"))
    if shape is None:
        kcsv = os.path.join(output_dir, f"set_of_all_unique_kmers_{suffix}.csv")
        with open(kcsv) as fh:
            n_cols = sum(1 for _ in fh) - 1            # minus the header row
        shape = (int(row.size - 1), int(n_cols))
    if device == "gpu":
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        return cusparse.csr_matrix(
            (cp.asarray(data), cp.asarray(column), cp.asarray(row)), shape=shape)
    import scipy.sparse
    return scipy.sparse.csr_matrix((data, column, row), shape=shape)


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
    max_ram_gb: float = 0.0,
    out_dir: str = "",
    out_suffix: str = "",
    max_gpus: int = 0,
    reference_in: str = "",
    binarize: bool = False,
):
    # GPU imports are *here*, not at module top, so spawn'd workers don't pull
    # cupy/cudf when they re-import this module. When KMX_CPU_MERGE is forced and
    # no usable GPU is present (CPU-only box / CI), fall back to running the whole
    # pipeline on the CPU — cudf/cupy stay None and every merge takes the exact
    # pandas path. With a GPU present this behaves exactly as before.
    cpu_merge_forced = bool(os.environ.get("KMX_CPU_MERGE"))
    cudf = cp = None
    try:
        import cudf as _cudf
        import cupy as _cp
        _cp.cuda.runtime.getDeviceCount()        # probe — raises without a usable GPU
        cudf, cp = _cudf, _cp
    except Exception:
        if not cpu_merge_forced:
            raise
        print("  [cpu-merge] no usable GPU detected — running fully on CPU",
              flush=True)

    canonical = not disable_normalization
    n_cpus    = threads if threads and threads > 0 else available_cpus()
    n_genomes = len(sample_ids)
    n_files   = len(flat_files)

    # Size KMC's per-stage RAM targets from the memory actually free at launch.
    # max_ram_gb (--max-ram-gb) only overrides the Tier-2 accumulator spill
    # threshold; 0 = auto-derive everything.
    # Size the concurrent worker count to the per-worker footprint (KMC buffer +
    # the genome's k-mer table), so a big genome at high k runs fewer workers and
    # stage 2 cannot OOM. See estimate_worker_df_gb().
    budget = plan_ram_budget(n_cpus, n_genomes, spill_override_gb=max_ram_gb,
                             worker_df_gb=estimate_worker_df_gb(files_by_sample,
                                                               kmer_size))

    print(f"KMX · {n_genomes:,} genomes · k={kmer_size} · min={min_val} · "
          f"max={max_val} · {'canonical' if canonical else 'non-canonical'} · "
          f"{n_cpus} workers · {input_fmt}", flush=True)
    print(f"  {budget.describe()}", flush=True)

    # ── Stage 1: global reference k-mer set (build, or load a cached one) ─────
    t0 = time.time()
    if reference_in:
        # Load the packed-2bit reference from disk and skip stage 1 entirely.
        # STRICT validation: the cache must match this run's k/min/max/canonical
        # and input file set, else we refuse (a wrong reference would corrupt the
        # matrix silently). The reference is already in the exact form stage 2
        # merges against, so no rebuild/decode is needed here.
        print(f"[1/3] loading cached reference from {reference_in} …", flush=True)
        ref_pdf, _ref_meta = load_reference(reference_in)
        validate_reference(_ref_meta, kmer_size, min_val, max_val, canonical,
                           flat_files)
        n_unique = len(ref_pdf)
        print(f"      → {n_unique:,} unique k-mers  (loaded, stage 1 skipped · "
              f"{time.time()-t0:.1f}s)", flush=True)
    else:
        print(f"[1/3] building the global k-mer set from {n_files:,} file(s) …",
              flush=True)
        ref_pdf = _count_reference(flat_files, input_fmt, kmer_size, tmp_dir,
                                   min_val, max_val, canonical, n_cpus,
                                   budget.stage1_gb)
        n_unique = len(ref_pdf)
        print(f"      → {n_unique:,} unique k-mers  ({time.time()-t0:.1f}s)",
              flush=True)
    n_words  = sum(1 for c in ref_pdf.columns if c.startswith("kmer_"))
    key_cols = [f"kmer_{i}" for i in range(n_words)]
    # Release stage 1's freed KMC arenas before stage 2 spawns workers, else its
    # ~stage1_gb residue stacks on top of the workers and OOMs a tight box.
    _rss_pre = _proc_rss_gb()
    _reclaim_memory()
    print(f"      (parent RSS after stage 1: {_rss_pre:.1f} GB → "
          f"{_proc_rss_gb():.1f} GB after reclaim)", flush=True)

    if n_unique == 0:
        # Edge case: no k-mer survives the global filter.
        empty_idx_df = pd.DataFrame({"index": [], "K-mer": []})
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.zeros(n_genomes + 1, dtype=np.int64),
            empty_idx_df,
            100.0,
        )

    # ── Plan VRAM shards for the reference ───────────────────────────────────
    # The reference is held on the GPU and every genome is merged against it.
    # If it (plus a genome's merge transient) would exceed VRAM, build the
    # matrix in B contiguous column-blocks, re-counting genomes per block.
    # B == 1 is the single-pass path (unchanged). KMX_FORCE_SHARDS overrides B.
    free_vram = int(cp.cuda.runtime.memGetInfo()[0]) if cp is not None else 0
    # Artificial VRAM cap for the *planning* math only (the real merge still probes
    # actual free VRAM at merge time). Lets you simulate a smaller GPU / leave
    # headroom so the reference is column-sharded on purpose. KMX_VRAM_BUDGET_GB.
    _vram_cap = os.environ.get("KMX_VRAM_BUDGET_GB")
    if _vram_cap:
        free_vram = max(1, int(float(_vram_cap) * (1024 ** 3)))
        print(f"  [vram-cap] planning VRAM artificially limited to "
              f"{float(_vram_cap):.4g} GB", flush=True)
    force = int(os.environ.get("KMX_FORCE_SHARDS", "0") or 0)
    blocks = plan_vram_shards(n_unique, n_words,
                              _max_genome_bytes(files_by_sample), free_vram,
                              force=force)
    B = len(blocks)

    # Column-index dtype, chosen from the reference size: signed int32 for ≤ 2.15 B
    # columns (the usual case; scipy/cupyx-native, compact), int64 above it so very
    # large references (huge-VRAM GPUs) can't silently overflow the index.
    col_dtype = _col_index_dtype(n_unique)
    if col_dtype != np.int32:
        why = ("forced via KMX_COL_DTYPE" if os.environ.get("KMX_COL_DTYPE")
               else f"reference has {n_unique:,} columns, exceeds the int32 limit "
                    f"{np.iinfo(np.int32).max:,}")
        print(f"  column index: {col_dtype} ({why})", flush=True)

    # Value dtype: presence → int8 0/1 (4x smaller than float32 — a real VRAM/IO
    # win at scale; safe for sums/chi-squared because scipy/numpy/cupy upcast the
    # accumulator to int64). Counts → float32.
    val_dtype = np.dtype(np.int8) if binarize else np.dtype(np.float32)
    if binarize:
        print("  values: presence/absence (int8 0/1, 4x smaller than float32)",
              flush=True)

    # ── Plan the multi-GPU layout from the block count B ─────────────────────
    # B == 1 (reference fits one GPU) + >1 GPU → data-parallel (replicate ref,
    # split genomes). B > 1 (reference too large) + >1 GPU → reference-parallel
    # (shard the B blocks across GPUs, broadcast genomes). Otherwise single-GPU.
    # Rows are emitted in input order in every mode. KMX_FORCE_GPUS/KMX_MAX_GPUS
    # (and KMX_FORCE_SHARDS, via B) override.
    try:
        n_visible = int(cp.cuda.runtime.getDeviceCount()) if cp is not None else 0
    except Exception:
        n_visible = 0
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = ([s for s in _vis.split(",") if s != ""] if _vis
                   else [str(i) for i in range(max(n_visible, 1))])
    _max_gpus = max_gpus if max_gpus and max_gpus > 0 else \
        int(os.environ.get("KMX_MAX_GPUS", "4") or 4)
    gpu_layout = plan_gpu_layout(
        n_visible, B, n_genomes,
        ram_available_gb=(budget.detected_gb or 0.0),
        ref_bytes=n_unique * (8 * n_words + 4),
        visible_ids=visible_ids, max_gpus=_max_gpus, min_count=min_val)
    print(f"  {gpu_layout.describe()}", flush=True)
    if gpu_layout.warning:
        # Surface the D < S_min resource shortfall loudly and early (before the
        # long stage-2 work), with the concrete fix, so the user can cancel and
        # re-request more GPUs if the slowdown isn't acceptable.
        print(f"  ⚠ WARNING: {gpu_layout.warning}", flush=True)

    # ── Stage 2: per-genome counts in parallel (each CPU = one genome) ───────
    #   * One job per genome; KMC merges a genome's own files internally.
    #   * As workers finish (arbitrary order), merge the result against the
    #     resident reference block on the GPU and spill the (col_idx, count)
    #     numpy arrays to host RAM (Tier-1). The accumulator flushes to disk
    #     (Tier-2) past the budget threshold.
    #   * With B > 1 the genomes are re-counted once per block, so a genome
    #     contributes one fragment per block; the fragments are concatenated
    #     into its CSR row at assembly. Rows keep manifest order via row_ptr.
    t0 = time.time()
    blocks_note = ("" if B == 1 else
                   f"  ·  reference too large for one GPU pass "
                   f"(free={free_vram / (1024**3):.1f} GB) → {B} column-blocks")
    print(f"[2/3] counting {n_genomes:,} genomes on {budget.n_workers} workers"
          f"{f' (capped from {n_cpus} to fit RAM)' if budget.n_workers < n_cpus else ''}"
          f"{blocks_note}", flush=True)
    progress = _Progress(n_genomes * B, "      counting")

    # Worker -> parent hand-off policy. In memory by default; a genome's k-mer
    # table is spilled to disk only when it is large, so up to n_cpus of them
    # can't accumulate in the parent's RAM and OOM it (the chr19/high-k case).
    # Default threshold keeps the aggregate in-memory hand-off to ~25% of the RAM
    # budget; KMX_WORKER_SPILL_MB overrides (0 = never spill / always in memory).
    _env_spill = os.environ.get("KMX_WORKER_SPILL_MB")
    if _env_spill is not None:
        handoff_spill_bytes = int(float(_env_spill) * 1024 ** 2)
    else:
        avail_gb = budget.detected_gb if budget.detected_gb else 8.0
        handoff_spill_bytes = max(64 * 1024 ** 2,
                                  int(0.25 * avail_gb * 1024 ** 3
                                      / max(1, budget.n_workers)))
    # Spill destination. By default everything spills under tmp_dir (-t), which
    # is normally fast node-local scratch. KMX_SPILL_DIR redirects ONLY the
    # disk-spill artefacts (worker hand-off parquet + Tier-2 CSR chunks) to a
    # separate location, so KMC's hot working files can stay on fast scratch
    # while large spills go to a roomier shared volume.
    spill_root = os.environ.get("KMX_SPILL_DIR") or tmp_dir
    os.makedirs(spill_root, exist_ok=True)
    handoff_dir = os.path.join(spill_root, "_handoff")
    os.makedirs(handoff_dir, exist_ok=True)
    print(f"[2/3] worker hand-off: in-memory, spill-to-disk above "
          f"{handoff_spill_bytes / 1024**2:.0f} MB/genome", flush=True)

    # Smart CPU distribution: the worker COUNT is memory-bound (few workers for big
    # genomes), which would otherwise leave most CPUs idle during stage 2. Hand the
    # leftover cores to each worker as KMC threads so all cores stay busy at the
    # SAME RAM (KMC's memory is bounded by max_ram_gb + the genome's table, which is
    # thread-independent; threads only add parallelism). KMX_KMC_THREADS overrides.
    #
    # Multi-GPU: each GPU worker is a host process doing cuDF.from_pandas + H2D/D2H
    # copies (~1 core), plus the dispatch/collect coordinator (~1 core). Reserve
    # D+1 cores for those FIRST, then split the rest among count workers — else the
    # count pool oversubscribes the cores, the GPU feeders starve, and the GPUs idle
    # waiting on host-side prep. Single-GPU keeps all cores for counting (the merge
    # runs in the main process there).
    _gpu_reserve = (gpu_layout.n_gpus + 1) if gpu_layout.n_gpus > 1 else 0
    _count_cores = max(budget.n_workers, n_cpus - _gpu_reserve)
    _env_thr = os.environ.get("KMX_KMC_THREADS")
    if _env_thr:
        kmc_threads = max(1, int(_env_thr))
    else:
        kmc_threads = max(1, _count_cores // max(1, budget.n_workers))
    print(f"[2/3] CPU split: {budget.n_workers} workers x {kmc_threads} KMC thread(s) "
          f"= {budget.n_workers * kmc_threads}/{n_cpus} cores", flush=True)

    work = [
        (sid, files_by_sample[sid], kmer_size, tmp_dir, canonical,
         max_val, int(budget.per_worker_gb), input_fmt,   # KMC max_ram_gb must be int
         handoff_dir, handoff_spill_bytes, kmc_threads)
        for sid in sample_ids
    ]

    # Total nnz per genome (summed across blocks); the row pointer rebuilds from it.
    sizes = np.zeros(n_genomes, dtype=np.int64)
    # Accumulator of (g_idx, idx_np, val_np) fragments not yet flushed to disk.
    resident = []
    resident_bytes = 0
    chunk_files: List[str] = []
    spill_bytes = int(budget.spill_threshold_gb * (1024 ** 3))
    oom_state = {"cpu_fallbacks": 0}

    # Cap host thread pools to 1 in the workers (parent env -> inherited by the
    # spawned children at import time). Each worker runs KMC single-threaded and
    # never needs numpy/OpenBLAS or KMC OpenMP parallelism, so without this every
    # worker would spin up a per-core OpenBLAS+OpenMP thread pool — hundreds of
    # threads total. That oversubscription both wastes resources and worsens the
    # scheduling timing that trips KMC's ProcessStage2 race.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"

    # spawn (not fork) so the parent's CUDA context can't be inherited.
    ctx = mp.get_context("spawn")

    # ── Streaming in-order assembly (no reload; fixed input-order rows) ───────
    # When the reference fits one GPU pass (B == 1) and an output location is
    # given, write each genome's CSR row straight to the output .npy files in
    # INPUT ORDER as soon as it — and all earlier genomes — are ready. Out-of-
    # order finishers wait in a small reorder buffer that spills per-genome to
    # disk only if it outgrows the budget. This removes the accumulator, the
    # Tier-2 chunk spill, the final reload, and the scatter: the matrix is
    # written once, sequentially, so peak memory is independent of its size and
    # row i always corresponds to genome G_i. (B > 1 keeps the scatter path.)
    # Multi-GPU: data-parallel (B==1, replicate ref + split genomes) OR
    # reference-parallel (B>1, shard the blocks across GPUs + broadcast genomes).
    # Both stream rows in input order — same on-disk result as the single-GPU
    # paths below. The reference-parallel branch also replaces the slow B>1
    # single-GPU loop (which re-counts genomes per block).
    if (gpu_layout.mode in ("data_parallel", "reference_parallel")
            and out_dir and out_suffix):
        return _build_streaming_multigpu(
            out_dir=out_dir, out_suffix=out_suffix, ref_pdf=ref_pdf,
            n_unique=n_unique, n_words=n_words, col_dtype=col_dtype,
            val_dtype=val_dtype, binarize=binarize, key_cols=key_cols,
            kmer_size=kmer_size, blocks=blocks, work=work, sample_ids=sample_ids,
            n_genomes=n_genomes, budget=budget, ctx=ctx, spill_root=spill_root,
            handoff_dir=handoff_dir, spill_bytes=spill_bytes,
            gpu_layout=gpu_layout, oom_state=oom_state, progress=progress,
            cpu_merge=cpu_merge_forced, t0=t0)

    if B == 1 and out_dir and out_suffix:
        os.makedirs(out_dir, exist_ok=True)
        lo, hi = blocks[0]
        block_pdf = ref_pdf.iloc[lo:hi].copy()
        block_pdf["__col_idx__"] = np.arange(lo, hi, dtype=col_dtype)
        ref_gdf_b = None if cudf is None else cudf.DataFrame.from_pandas(block_pdf)

        col_fp = _npy_open_stream(os.path.join(out_dir, f"column_{out_suffix}.npy"),
                                  col_dtype)
        val_fp = _npy_open_stream(os.path.join(out_dir, f"data_{out_suffix}.npy"),
                                  val_dtype)
        buf = {}            # g_idx -> (idx_np, val_np), waiting in RAM
        buf_spill = {}      # g_idx -> path, parked on disk (overflow)
        buf_bytes = 0
        buf_cap = spill_bytes if spill_bytes > 0 else (1 << 30)
        next_write = 0
        total_nnz = 0
        written_since_sync = 0
        print("[2/3] assembling rows in input order, streamed to disk (no reload)",
              flush=True)

        def _emit(gi, idx_np, val_np):
            nonlocal total_nnz, written_since_sync
            sz = int(idx_np.size)
            if sz:
                # Zero-copy write: hand the array's own buffer to the file (a
                # memoryview), instead of .tobytes() which copies it first.
                col_fp.write(memoryview(np.ascontiguousarray(idx_np, dtype=col_dtype)))
                val_fp.write(memoryview(np.ascontiguousarray(val_np, dtype=val_dtype)))
                sizes[gi] = sz
                total_nnz += sz
                written_since_sync += sz * 8

        def _drain():
            nonlocal next_write, buf_bytes
            while next_write < n_genomes and (next_write in buf or next_write in buf_spill):
                if next_write in buf:
                    idx_np, val_np = buf.pop(next_write)
                    buf_bytes -= (idx_np.nbytes + val_np.nbytes)
                else:
                    idx_np, val_np = _buf_load_one(buf_spill.pop(next_write))
                _emit(next_write, idx_np, val_np)
                next_write += 1

        _gcount = 0
        for g_idx, res in _completed_genomes(work, range(n_genomes),
                                             budget.n_workers, ctx, sample_ids,
                                             progress=progress):
            df_g = _materialize(res)
            idx_np, val_np = _merge_genome_block(
                df_g, ref_gdf_b, block_pdf, key_cols, cudf, cp, oom_state, binarize)
            del df_g
            buf[g_idx] = (idx_np, val_np)
            buf_bytes += idx_np.nbytes + val_np.nbytes
            _drain()
            while buf_bytes > buf_cap and buf:           # park farthest genomes
                gmax = max(buf)
                i2, v2 = buf.pop(gmax)
                buf_bytes -= (i2.nbytes + v2.nbytes)
                buf_spill[gmax] = _buf_spill_one(spill_root, gmax, i2, v2)
            progress.update()
            _gcount += 1
            try:
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            _reclaim_memory()
            if written_since_sync >= (1 << 30):          # bound output page cache
                for _fp in (col_fp, val_fp):
                    _fp.flush()
                    try:
                        os.fsync(_fp.fileno())
                        os.posix_fadvise(_fp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    except Exception:
                        pass
                written_since_sync = 0
            if _gcount % 10 == 0:
                _self, _kids = _proc_rss_gb(), _children_rss_gb()
                progress.note(f"      [mem@{_gcount}] parent={_self:.1f} GB  "
                              f"workers={_kids:.1f} GB  total={_self + _kids:.1f} GB")

        _drain()
        if next_write != n_genomes:
            raise RuntimeError(
                f"[stage 2] streaming assembly wrote {next_write} of "
                f"{n_genomes} rows; a genome never completed")
        del ref_gdf_b, block_pdf
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
        progress.close()
        import shutil
        shutil.rmtree(handoff_dir, ignore_errors=True)
        _npy_finalize_stream(col_fp, col_dtype, total_nnz)
        _npy_finalize_stream(val_fp, val_dtype, total_nnz)
        if oom_state["cpu_fallbacks"]:
            print(f"      note: {oom_state['cpu_fallbacks']} merge(s) ran on the "
                  f"CPU (didn't fit VRAM, or forced) — slower, still correct.",
                  flush=True)
        row_ptr = np.empty(n_genomes + 1, dtype=np.int64)
        row_ptr[0] = 0
        np.cumsum(sizes, out=row_ptr[1:])
        sparsity = (100.0 * (1.0 - total_nnz / float(n_genomes * n_unique))
                    if n_unique else 100.0)
        print(f"      → {total_nnz:,} nonzeros · {sparsity:.1f}% sparse  "
              f"({time.time()-t0:.1f}s · streamed, input order)", flush=True)
        print("[3/3] decoding k-mers …", flush=True)
        with _quiet_kmc():
            decoded = kmcpy.decode_kmers(ref_pdf, k=kmer_size)
        # decode_kmers maps rows positionally (no sort) → row j is column j; assert
        # the 1:1 length so a future change can't silently misalign the legend.
        if len(decoded) != n_unique:       # if/raise (survives python -O)
            raise RuntimeError(
                f"decode_kmers returned {len(decoded)} rows for {n_unique} reference "
                f"k-mers — column↔k-mer legend would be misaligned")
        decoded.insert(0, "index", np.arange(n_unique, dtype=np.int64))
        decoded.rename(columns={"kmer": "K-mer"}, inplace=True)
        return (None, None, row_ptr, decoded, sparsity)

    _gcount = 0  # genomes merged so far (across blocks), for the memory probe
    for b, (lo, hi) in enumerate(blocks):
        # Upload only this column-block of the reference; tag each row with its
        # GLOBAL column index (lo..hi). Keep the host copy for the OOM fallback.
        block_pdf = ref_pdf.iloc[lo:hi].copy()
        block_pdf["__col_idx__"] = np.arange(lo, hi, dtype=col_dtype)
        ref_gdf_b = None if cudf is None else cudf.DataFrame.from_pandas(block_pdf)

        # Watchdog-driven counting: the generator yields each genome's hand-off
        # as workers finish, resubmitting any genome whose worker deadlocks.
        for g_idx, res in _completed_genomes(work, range(n_genomes),
                                             budget.n_workers,
                                             ctx, sample_ids, progress=progress):
            df_g = _materialize(res)   # in-memory df, or read back a spilled one
            idx_np, val_np = _merge_genome_block(
                df_g, ref_gdf_b, block_pdf, key_cols, cudf, cp, oom_state, binarize)
            del df_g

            if idx_np.size:
                sizes[g_idx] += idx_np.size
                resident.append((g_idx, idx_np, val_np))
                resident_bytes += idx_np.nbytes + val_np.nbytes
                # Tier-2: flush the accumulator to disk once it exceeds the cap.
                if spill_bytes > 0 and resident_bytes >= spill_bytes:
                    path = _spill_chunk(resident, spill_root, len(chunk_files))
                    chunk_files.append(path)
                    progress.note(f"      ↳ spilled {len(resident)} fragment(s), "
                                  f"{resident_bytes / (1024**3):.2f} GB → "
                                  f"{os.path.basename(path)}")
                    resident = []
                    resident_bytes = 0

            progress.update()

            # Per genome the parent merges a large table on the GPU; the freed
            # HOST buffers (glibc arenas + cuDF/cupy PINNED host pool) are not
            # returned to the OS on their own, so the parent's RSS would climb
            # ~1 GB/genome and OOM any box. Reclaim every genome to stay flat.
            _gcount += 1
            try:
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            _reclaim_memory()
            if _gcount % 10 == 0:
                _self, _kids = _proc_rss_gb(), _children_rss_gb()
                progress.note(f"      [mem@{_gcount}] parent={_self:.1f} GB  "
                              f"workers={_kids:.1f} GB  total={_self + _kids:.1f} GB")

        del ref_gdf_b, block_pdf
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()

    progress.close()
    # Remove any orphaned hand-off files (a watchdog kill can leave a partial
    # Parquet from a worker that was mid-spill); successful ones were already
    # read-and-deleted by _materialize.
    import shutil
    shutil.rmtree(handoff_dir, ignore_errors=True)
    if oom_state["cpu_fallbacks"]:
        print(f"      note: {oom_state['cpu_fallbacks']} merge(s) ran on the CPU "
              f"(didn't fit VRAM, or forced) — slower, still correct.",
              flush=True)

    # ── Assemble final CSR on the CPU (cursor-based; supports multi-block) ────
    # row_ptr is built from `sizes` alone; a per-genome cursor lets fragments
    # (resident first, then replayed chunks) land in any order and lets one
    # genome receive several fragments (one per VRAM block).
    total_nnz = int(sizes.sum())
    # The final CSR is the single biggest allocation (e.g. chr19/k61/n1000 →
    # 2.6 B nnz ≈ 21 GB for column+data). Building it in RAM and then saving is
    # what OOMs a tight box at the finish line. When out_dir/out_suffix are given
    # we memory-map the column/data .npy files DIRECTLY and scatter into them, so
    # the OS pages them to disk and the assembly peak is ~one chunk, not 21 GB.
    # Same scatter ops, same total I/O (no separate save), so speed is unchanged.
    _memmap_out = bool(out_dir and out_suffix and total_nnz > 0)
    if _memmap_out:
        os.makedirs(out_dir, exist_ok=True)
        col_buf = np.lib.format.open_memmap(
            os.path.join(out_dir, f"column_{out_suffix}.npy"),
            mode="w+", dtype=col_dtype, shape=(total_nnz,))
        val_buf = np.lib.format.open_memmap(
            os.path.join(out_dir, f"data_{out_suffix}.npy"),
            mode="w+", dtype=val_dtype, shape=(total_nnz,))
    else:
        col_buf = np.empty(total_nnz, dtype=col_dtype)
        val_buf = np.empty(total_nnz, dtype=val_dtype)
    row_ptr   = np.empty(n_genomes + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(sizes, out=row_ptr[1:])
    cursor = row_ptr[:n_genomes].copy()

    for g_idx, idx, val in resident:
        _scatter(col_buf, val_buf, cursor, g_idx, idx, val)
    del resident
    if _memmap_out:
        _flush_release(col_buf, val_buf)   # keep the page cache from filling

    for path in chunk_files:
        with np.load(path) as chunk:
            gidxs   = chunk["gidxs"]
            csizes  = chunk["sizes"]
            big_idx = chunk["idx"]
            big_val = chunk["val"]
        off = 0
        for g_idx, sz in zip(gidxs.tolist(), csizes.tolist()):
            _scatter(col_buf, val_buf, cursor, g_idx,
                     big_idx[off:off + sz], big_val[off:off + sz])
            off += sz
        try:
            os.remove(path)
        except OSError:
            pass
        if _memmap_out:
            # Flush + release after every chunk so the assembly's resident pages
            # stay ~one chunk, not the whole 21 GB output (the 31.7 GB near-miss).
            _flush_release(col_buf, val_buf)

    shard_note = f"  ·  {B} VRAM shards" if B > 1 else ""
    spill_note = (f"  ·  {len(chunk_files)} disk chunk(s)" if chunk_files else "")
    sparsity = 100.0 * (1.0 - total_nnz / float(n_genomes * n_unique))
    print(f"      → {total_nnz:,} nonzeros · {sparsity:.1f}% sparse  "
          f"({time.time()-t0:.1f}s{shard_note}{spill_note})", flush=True)

    # ── Stage 3: decode reference keys to ACGT for the user-facing CSV ──────
    print("[3/3] assembling CSR matrix and decoding k-mers …", flush=True)
    with _quiet_kmc():
        decoded = kmcpy.decode_kmers(ref_pdf, k=kmer_size)
    # column j ↔ ref_pdf row j (the positional __col_idx__). kmcpy.decode_kmers
    # is fully vectorised and maps each row positionally — it does NOT sort or
    # reorder — so decoded row j IS column j's k-mer. Assert the 1:1 length so a
    # future decode change that drops/reorders rows fails loudly, not silently.
    if len(decoded) != n_unique:           # if/raise (survives python -O, unlike assert)
        raise RuntimeError(
            f"decode_kmers returned {len(decoded)} rows for {n_unique} reference "
            f"k-mers — column↔k-mer legend would be misaligned")
    decoded.insert(0, "index", np.arange(n_unique, dtype=np.int64))
    decoded.rename(columns={"kmer": "K-mer"}, inplace=True)

    if _memmap_out:
        # Flush column/data to disk and drop the memmaps — the .npy files are now
        # complete on disk, so signal the caller (cli) NOT to re-save them.
        col_buf.flush(); val_buf.flush()
        del col_buf, val_buf
        _reclaim_memory()
        return (None, None, row_ptr, decoded, sparsity)

    return (
        val_buf,         # numpy float32, on CPU
        col_buf,         # numpy int32/int64,  on CPU
        row_ptr,         # numpy int64,   on CPU
        decoded,
        sparsity,
    )
