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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_arguments()
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Parse + validate the manifest CSV up front so any error surfaces
    # before we touch the GPU or spawn KMC subprocesses.
    try:
        manifest = parse_manifest(args.genome_list)
    except ManifestError as e:
        log.error("Manifest error: %s", e)
        return 3
    log.info(
        "Manifest: %d genome(s), %d file(s), family=%s (input_fmt=%s)",
        len(manifest.sample_ids), len(manifest.flat_files),
        manifest.family, manifest.input_fmt,
    )

    try:
        import cupy as cp
    except Exception as e:
        log.error("CuPy import failed: %s", e)
        return 2

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    time.sleep(1)

    dev          = cp.cuda.Device(0)
    props        = cp.cuda.runtime.getDeviceProperties(dev.id)
    gpu_name     = props["name"].decode("utf-8")
    total_mem_gb = props["totalGlobalMem"] / (1024 ** 3)
    free_mem_gb  = dev.mem_info[0] / (1024 ** 3)
    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    log.info("GPU: %s | total=%.2f GB | free=%.2f GB", gpu_name, total_mem_gb, free_mem_gb)

    monitor = GPUMemoryMonitor(cp=cp, interval=1.0)
    monitor.start()

    t0 = time.time()

    try:
        data, column, row, unique_kmers, sparsity = create_csr_matrix(
            sample_ids=manifest.sample_ids,
            files_by_sample=manifest.files_by_sample,
            flat_files=manifest.flat_files,
            input_fmt=manifest.input_fmt,
            kmer_size=args.kmer_size,
            tmp_dir=args.tmp,
            min_val=args.min,
            max_val=args.max,
            disable_normalization=args.disable_normalization,
            threads=args.threads,
            max_ram_gb=args.max_ram_gb,
        )
    finally:
        monitor.stop()

    elapsed   = datetime.timedelta(seconds=(time.time() - t0))
    peak_used = monitor.peak_gb()

    d_status   = 1 if args.disable_normalization else 0
    suffix     = f"k{args.kmer_size}_min{args.min}_max{args.max}_d{d_status}"
    stats_path = os.path.join(output_dir, f"feature_matrix_stats_{suffix}.txt")

    stats_text = (
        f"Sparsity: {sparsity}%\n"
        f"K-mer size: {args.kmer_size}\n"
        f"Temporary directory: {args.tmp}\n"
        f"Manifest: {os.path.abspath(args.genome_list)}\n"
        f"Genomes (rows): {len(manifest.sample_ids)}\n"
        f"Total files: {len(manifest.flat_files)}\n"
        f"Input format family: {manifest.family} (kmcpy input_fmt={manifest.input_fmt})\n"
        f"Min value: {args.min}\n"
        f"Max value: {args.max}\n"
        f"Normalization disabled: {args.disable_normalization}\n"
        f"Threads: {args.threads}\n"
        f"Processing time: {elapsed}\n"
        f"Peak GPU memory used (cupy memGetInfo): {peak_used:.2f} GB\n"
    )
    write_stats(stats_path, stats_text)

    unique_kmers.to_csv(
        os.path.join(output_dir, f"set_of_all_unique_kmers_{suffix}.csv"),
        index=False,
    )
    write_genome_index(output_dir, suffix, manifest.sample_ids)
    cp.save(os.path.join(output_dir, f"data_{suffix}.npy"),   data)
    cp.save(os.path.join(output_dir, f"row_{suffix}.npy"),    row)
    cp.save(os.path.join(output_dir, f"column_{suffix}.npy"), column)

    log.info("Files saved successfully in %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
