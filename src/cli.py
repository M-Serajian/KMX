#!/usr/bin/env python3
"""KMX v2 — genome × k-mer CSR matrix builder.

Stage 1 + Stage 2 k-mer counting are both performed by KMC-DataFrame
(kmcpy) on CPU. Per-genome counts run in parallel across CPUs (one
genome per worker). The merge into the global reference set and the
final CSR assembly run on a single GPU via cuDF + CuPy.
"""
import os
import time
import datetime
import gc
import logging
import threading

from .args import parse_arguments
from .create_csr_matrix import create_csr_matrix
from .manifest import parse_manifest, write_genome_index, ManifestError
from .ram_budget import plan_ram_budget, available_cpus
from .create_csr_matrix import (
    estimate_worker_df_gb, _count_reference, save_reference)

log = logging.getLogger(__name__)


class GPUMemoryMonitor:
    def __init__(self, cp, interval: float = 1.0):
        self.cp = cp
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._peak_used_bytes = 0
        self._started = False

    def _monitor(self):
        while not self._stop_event.is_set():
            free, total = self.cp.cuda.runtime.memGetInfo()
            used = total - free
            if used > self._peak_used_bytes:
                self._peak_used_bytes = used
            time.sleep(self.interval)

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self):
        if not self._started:
            return
        self._stop_event.set()
        self._thread.join()

    def peak_gb(self) -> float:
        return self._peak_used_bytes / (1024 ** 3)


def write_stats(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def build_reference(
    manifest: str,
    kmer_size: int,
    tmp_dir: str,
    reference_dir: str,
    *,
    min_count: int = 5,
    max_count: "int | None" = None,
    disable_normalization: bool = False,
    threads: int = 0,
    max_ram_gb: float = 0.0,
):
    """Build ONLY the global k-mer reference set and cache it to disk — no GPU.

    Runs stage 1 (the column space) over every input file and writes the packed
    **2-bit** reference (``reference.parquet`` — KMC uint64 keys, exactly what
    stage 2 merges against) plus a ``reference_meta.json`` sidecar to
    ``reference_dir``, then stops. A later ``build(..., reference=reference_dir)``
    (or ``KMX -l … --reference reference_dir``) loads it and **skips stage 1**.

    Because stage 1 touches no GPU, run this on a CPU-only node so the (often
    multi-hour) reference build never sits on the GPU clock, and so it isn't
    repeated across downstream runs.

    Args:
        manifest: manifest CSV (``sample_id,file``).
        kmer_size: k-mer length, 8–136. ``min_count``/``max_count`` are baked into
            the reference and must match the later ``build`` call (strictly checked).
        tmp_dir: scratch directory (created if absent).
        reference_dir: directory to write the cached reference into (created).
        min_count/max_count: global k-mer count filter (``max_count=None`` → N/2).
        disable_normalization, threads, max_ram_gb: as in :func:`build`.

    Returns:
        ``reference_dir``.

    Raises:
        ManifestError, ValueError (``max_count < min_count``).
    """
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    m = parse_manifest(manifest)
    n_genomes = len(m.sample_ids)
    if max_count is None:
        max_count = max(1, n_genomes // 2)
    if max_count < min_count:
        raise ValueError(f"max_count ({max_count}) must be >= min_count ({min_count})")

    canonical   = not disable_normalization
    eff_threads = threads if threads and threads > 0 else available_cpus()
    budget = plan_ram_budget(
        eff_threads, n_genomes, max_ram_gb,
        worker_df_gb=estimate_worker_df_gb(m.files_by_sample, kmer_size))
    log.info("Building reference (CPU only): %d genome(s), %d file(s), k=%d, "
             "min=%d, max=%d, canonical=%s", n_genomes, len(m.flat_files),
             kmer_size, min_count, max_count, canonical)

    t0 = time.time()
    ref_pdf = _count_reference(m.flat_files, m.input_fmt, kmer_size, tmp_dir,
                               min_count, max_count, canonical, eff_threads,
                               budget.stage1_gb)
    save_reference(reference_dir, ref_pdf, kmer_size, min_count, max_count,
                   canonical, m.flat_files)
    log.info("Reference: %d unique k-mers (packed 2-bit) → %s  [%s]",
             len(ref_pdf), reference_dir,
             datetime.timedelta(seconds=(time.time() - t0)))
    return reference_dir


def build(
    manifest: str,
    kmer_size: int,
    tmp_dir: str,
    output_dir: str,
    *,
    min_count: int = 5,
    max_count: "int | None" = None,
    disable_normalization: bool = False,
    threads: int = 0,
    max_ram_gb: float = 0.0,
    max_gpus: int = 0,
    reference: str = "",
    presence: bool = False,
    chunk_device_gb: float = 0.0,
    write_output: bool = True,
):
    """Build a genome × k-mer CSR matrix from a manifest — the same pipeline as the ``KMX`` CLI.

    Args:
        manifest: Path to the manifest CSV with columns ``sample_id,file``.
            Multiple files for one genome = multiple rows sharing a ``sample_id``.
        kmer_size: K-mer length, 8–136.
        tmp_dir: Scratch directory for intermediate KMC / spill files (created if absent).
        output_dir: Where outputs are written (created if absent).
        min_count: Drop k-mers occurring fewer than this across the dataset (default 5).
        max_count: Drop k-mers occurring more than this. ``None`` → half the genome count.
        disable_normalization: If True, a k-mer and its reverse complement are distinct features.
        threads: CPU threads; ``0`` = all cores granted to the process.
        max_ram_gb: Host-RAM cap for the accumulator before spilling; ``0`` = auto (cgroup/SLURM).
        max_gpus: Max GPUs to use for the merge. ``0`` = auto (env ``KMX_MAX_GPUS`` or 4).
            When >1 GPU is available and the reference fits one GPU, the per-genome
            merge runs data-parallel across GPUs; rows stay in input order.
        reference: Path to a cached reference directory (from ``build_reference`` /
            ``--build-reference``). If set, stage 1 is **skipped** and the packed
            2-bit reference is loaded from there; it is strictly validated against
            this run's ``kmer_size``/``min``/``max``/normalization and input files.
            ``""`` = build the reference inline as usual.
        presence: If True, store presence/absence (``int8`` 0/1) instead of counts — a
            0/1 matrix ideal for chi-squared / association tests, 4x smaller than the
            ``float32`` counts (sum-safe: scipy/numpy/cupy upcast the accumulator to
            int64). Output files gain a ``_presence`` suffix.
        write_output: If True (default), write ``data``/``column``/``row`` ``.npy``, the k-mer CSV,
            the genome index, and the stats file to ``output_dir`` (CLI behavior). If False, skip
            the files and just return the arrays (note: this builds the matrix in RAM, so only use
            it when the matrix fits memory).

    Returns:
        ``(data, column, row, kmer_index_df, sparsity)``. When the matrix was streamed to disk,
        ``data`` and ``column`` are ``None`` (read them from the ``.npy`` files); ``row`` and
        ``kmer_index_df`` (column index → k-mer string) are always returned, plus the sparsity %.

    Tuning is automatic from the job's cores / RAM / VRAM. Env overrides:
    ``KMX_KMC_THREADS``, ``KMX_WORKER_SPILL_MB``, ``KMX_SPILL_DIR``.

    Raises:
        ManifestError: the manifest is malformed (bad header, missing file, mixed formats, …).
        ValueError: ``max_count < min_count``.
        ImportError: RAPIDS cuDF / CuPy is not available (activate the RAPIDS env).
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)          # kmcpy requires the scratch dir to exist

    # Parse + validate the manifest up front (raises ManifestError) so any error
    # surfaces before we touch the GPU or spawn KMC subprocesses.
    m = parse_manifest(manifest)
    n_genomes = len(m.sample_ids)
    log.info("Manifest: %d genome(s), %d file(s), family=%s (input_fmt=%s)",
             n_genomes, len(m.flat_files), m.family, m.input_fmt)

    if max_count is None:                       # same default the CLI uses: N/2
        max_count = max(1, n_genomes // 2)
    if max_count < min_count:
        raise ValueError(f"max_count ({max_count}) must be >= min_count ({min_count})")

    import cupy as cp                            # RAPIDS required (raises ImportError)
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    time.sleep(1)

    dev   = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    log.info("GPU: %s | total=%.2f GB | free=%.2f GB", props["name"].decode("utf-8"),
             props["totalGlobalMem"] / (1024 ** 3), dev.mem_info[0] / (1024 ** 3))

    eff_threads = threads if threads and threads > 0 else available_cpus()
    ram_budget = plan_ram_budget(
        eff_threads, n_genomes, max_ram_gb,
        worker_df_gb=estimate_worker_df_gb(m.files_by_sample, kmer_size))
    log.info("%s", ram_budget.describe())

    monitor = GPUMemoryMonitor(cp=cp, interval=1.0)
    monitor.start()
    t0 = time.time()
    d_status = 1 if disable_normalization else 0
    suffix   = (f"k{kmer_size}_min{min_count}_max{max_count}_d{d_status}"
                + ("_presence" if presence else ""))

    try:
        data, column, row, unique_kmers, sparsity = create_csr_matrix(
            sample_ids=m.sample_ids,
            files_by_sample=m.files_by_sample,
            flat_files=m.flat_files,
            input_fmt=m.input_fmt,
            kmer_size=kmer_size,
            tmp_dir=tmp_dir,
            min_val=min_count,
            max_val=max_count,
            disable_normalization=disable_normalization,
            threads=threads,
            max_ram_gb=max_ram_gb,
            max_gpus=max_gpus,
            reference_in=reference,
            binarize=presence,
            out_dir=output_dir if write_output else "",
            out_suffix=suffix if write_output else "",
        )
    finally:
        monitor.stop()

    elapsed   = datetime.timedelta(seconds=(time.time() - t0))
    peak_used = monitor.peak_gb()

    if write_output:
        stats_text = (
            f"Sparsity: {sparsity}%\n"
            f"K-mer size: {kmer_size}\n"
            f"Temporary directory: {tmp_dir}\n"
            f"Manifest: {os.path.abspath(manifest)}\n"
            f"Genomes (rows): {n_genomes}\n"
            f"Total files: {len(m.flat_files)}\n"
            f"Input format family: {m.family} (kmcpy input_fmt={m.input_fmt})\n"
            f"Min value: {min_count}\n"
            f"Max value: {max_count}\n"
            f"Normalization disabled: {disable_normalization}\n"
            f"Threads: {threads}\n"
            f"RAM budget [{ram_budget.source}]: available="
            f"{ram_budget.detected_gb if ram_budget.detected_gb else 'unknown'} GB, "
            f"stage1={ram_budget.stage1_gb} GB, "
            f"stage2={ram_budget.n_workers}x{ram_budget.per_worker_gb} GB/worker, "
            f"accumulator_spill={ram_budget.spill_threshold_gb} GB\n"
            f"Processing time: {elapsed}\n"
            f"Peak GPU memory used (cupy memGetInfo): {peak_used:.2f} GB\n"
        )
        write_stats(os.path.join(output_dir, f"feature_matrix_stats_{suffix}.txt"), stats_text)
        unique_kmers.to_csv(
            os.path.join(output_dir, f"set_of_all_unique_kmers_{suffix}.csv"), index=False)
        write_genome_index(output_dir, suffix, m.sample_ids)
        # data/column are None when create_csr_matrix already streamed them to
        # disk (the normal path); only save them here for the in-RAM path.
        if data is not None:
            cp.save(os.path.join(output_dir, f"data_{suffix}.npy"), data)
        if column is not None:
            cp.save(os.path.join(output_dir, f"column_{suffix}.npy"), column)
        cp.save(os.path.join(output_dir, f"row_{suffix}.npy"), row)
        log.info("Files saved successfully in %s", output_dir)

        if chunk_device_gb and chunk_device_gb > 0:
            # Split the just-written matrix into device-sized column chunks (all
            # rows, a band of k-mers each) for streaming on small devices.
            from .chunked_output import chunk_existing_output
            man = chunk_existing_output(output_dir, suffix, device_gb=chunk_device_gb)
            log.info("Chunked output: %d band(s) sized to %.3g GB devices → %s",
                     man["n_bands"], chunk_device_gb,
                     os.path.join(output_dir, f"chunks_{suffix}"))

    return data, column, row, unique_kmers, sparsity


def main() -> int:
    """CLI entry point: parse argv, run build(), map errors to exit codes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_arguments()
    try:
        if args.build_reference:
            # Stage 1 only: build + cache the reference (no GPU), then stop.
            build_reference(
                manifest=args.genome_list,
                kmer_size=args.kmer_size,
                tmp_dir=args.tmp,
                reference_dir=args.reference,
                min_count=args.min,
                max_count=args.max,
                disable_normalization=args.disable_normalization,
                threads=args.threads,
                max_ram_gb=args.max_ram_gb,
            )
        else:
            build(
                manifest=args.genome_list,
                kmer_size=args.kmer_size,
                tmp_dir=args.tmp,
                output_dir=args.output,
                min_count=args.min,
                max_count=args.max,
                disable_normalization=args.disable_normalization,
                threads=args.threads,
                max_ram_gb=args.max_ram_gb,
                max_gpus=args.max_gpus,
                reference=args.reference,
                presence=args.presence,
                chunk_device_gb=args.chunk_device_gb,
            )
    except ManifestError as e:
        log.error("Manifest error: %s", e)
        return 3
    except (ValueError, FileNotFoundError) as e:        # bad params / cache mismatch
        log.error("%s", e)
        return 4
    except ImportError as e:
        log.error("RAPIDS cuDF/CuPy import failed — is the RAPIDS env active? (%s)", e)
        return 2
    except Exception:                                   # noqa: BLE001
        # Any other failure (e.g. a GPU merge worker died mid-run) has already
        # cleaned up its partial output and re-raised. Report it with a full
        # traceback, then HARD-exit: tearing down the parent's CUDA/cupy context
        # at normal interpreter shutdown can itself wedge after an abnormal GPU
        # worker death, which would hang the process long after the error is
        # reported. os._exit bypasses atexit so a failed run always terminates
        # promptly with a non-zero status.
        import sys
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
