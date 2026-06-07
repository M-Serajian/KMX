"""Runtime RAM budgeting for the KMX v2 two-stage pipeline.

KMC's ``max_ram_gb`` is a *soft target* for its sort bins, not a hard cap:
more RAM means fewer disk passes (faster), too little means it spills to its
own scratch but still completes. We therefore size it from the RAM that is
actually free at launch instead of a hardcoded constant.

Two stages, two regimes:

  * Stage 1 (global reference k-mer count) runs ALONE with every thread, so
    it is handed (almost) all of the usable RAM.
  * Stage 2 runs ``n_workers`` single-threaded KMC processes concurrently
    while the main process holds the reference set, the GPU-merge scratch,
    and the growing CSR accumulator. We carve out a reserve for the main
    process and split the rest evenly across workers, clamped to a sane
    range.

The same main-process reserve doubles as the default Tier-2 spill threshold:
once the host-resident CSR accumulator exceeds it, stage 2 flushes to disk
(see create_csr_matrix.py). All values are *targets*; nothing here makes a
hard allocation.
"""

import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Tunables (GB / fractions). Conservative on purpose — KMC degrades
# gracefully when under-provisioned but the OS does not when over-provisioned.
_MARGIN_FLOOR_GB       = 4.0    # never hand out the last few GB (OS, page cache)
_MARGIN_FRAC           = 0.15
_MAIN_RESERVE_FLOOR_GB = 2.0    # main proc: reference + GPU staging + accumulator
_MAIN_RESERVE_FRAC     = 0.25
_PER_WORKER_MIN_GB     = 2.0    # KMC hard-errors below 2 GB ("min memory must be at least 2GB")
_PER_WORKER_CAP_GB     = 4.0    # stage-2 counts ONE genome per worker; a big sort
                                # buffer is wasted and, x16 workers, OOMs a big node
_WORKER_DF_PEAK_MULT   = 5.0    # MEASURED: a worker's peak ~ KMC budget + ~5x its
                                # k-mer table. For chr19/k61 (~1.15 GB table) one
                                # worker peaks ~7.7 GB — KMC's counting structures,
                                # its decode buffer, and the materialised pandas
                                # table all coexist, and KMC ignores the 2 GB soft
                                # cap for a single big genome. Plus the parent's
                                # merge of a returned table aligns with worker peaks
                                # transiently. Sizing the footprint to the measured
                                # peak (-> 2 chr19/k61 workers on 32 GB) holds.
_STAGE1_MIN_GB         = 2.0

# Used only when available RAM cannot be detected (non-Linux, odd cgroup, …).
_FALLBACK_STAGE1_GB     = 8     # int: feeds KMC's int-typed max_ram_gb
_FALLBACK_PER_WORKER_GB = 2     # int
_FALLBACK_SPILL_GB      = 8.0


@dataclass
class RamBudget:
    """Resolved per-stage RAM targets for one run."""
    stage1_gb: int              # KMC max_ram_gb for the stage-1 reference count (int: pybind11)
    per_worker_gb: int          # KMC max_ram_gb for each stage-2 worker (int: pybind11)
    spill_threshold_gb: float   # host RAM held by the accumulator before disk flush
    n_workers: int              # concurrent stage-2 workers this plan assumes
    detected_gb: Optional[float]  # MemAvailable at launch, or None if undetectable
    source: str                 # "auto" | "fallback"

    def describe(self) -> str:
        det = f"{self.detected_gb:.1f}" if self.detected_gb else "unknown"
        return (
            f"RAM plan [{self.source}]: available={det} GB  "
            f"stage1={self.stage1_gb:.1f} GB (all threads)  "
            f"stage2={self.n_workers}x{self.per_worker_gb:.1f} GB/worker  "
            f"accumulator_spill@{self.spill_threshold_gb:.1f} GB"
        )


def _meminfo_available_gb() -> Optional[float]:
    """Node-level available RAM (Linux ``/proc/meminfo`` ``MemAvailable``).

    ``MemAvailable`` is the kernel's own estimate of how much can be handed
    out without swapping. On a shared HPC node this reflects the *whole
    node*, not this job's share — see :func:`available_ram_gb`.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)  # KiB -> GiB
    except Exception:
        pass
    return None


def _read_first_int(path: str) -> Optional[int]:
    """Read a cgroup file holding a single value; None for missing / ``max``."""
    try:
        with open(path) as f:
            s = f.read().strip()
    except Exception:
        return None
    if not s or s == "max":
        return None
    try:
        return int(s.split()[0])
    except ValueError:
        return None


def _cgroup_available_gb() -> Optional[float]:
    """Headroom within this process's memory cgroup, or None if uncapped.

    This is the number that matters under SLURM / containers: a 1 TB node
    handed a 64 GB job still only has ~64 GB of headroom, and sizing KMC
    from the node total would get the job OOM-killed. Handles cgroup v2
    (unified ``memory.max`` / ``memory.current``) and v1
    (``memory.limit_in_bytes`` / ``memory.usage_in_bytes``), reading this
    process's own cgroup path first and falling back to the mount root.
    """
    path_v2 = None
    memb_v1 = None
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) != 3:
                    continue
                if parts[0] == "0" and parts[1] == "":
                    path_v2 = parts[2]
                if "memory" in parts[1].split(","):
                    memb_v1 = parts[2]
    except Exception:
        pass

    # cgroup v2 (unified hierarchy).
    base = "/sys/fs/cgroup"
    if os.path.exists(os.path.join(base, "cgroup.controllers")):
        for cdir in ([base + path_v2] if path_v2 else []) + [base]:
            limit = _read_first_int(os.path.join(cdir, "memory.max"))
            if limit is not None:
                cur = _read_first_int(os.path.join(cdir, "memory.current")) or 0
                return max(0.0, (limit - cur) / (1024 ** 3))
        return None  # uncapped at every level we can see

    # cgroup v1.
    if memb_v1 is not None:
        v1base = "/sys/fs/cgroup/memory"
        for cdir in (v1base + memb_v1, v1base):
            limit = _read_first_int(os.path.join(cdir, "memory.limit_in_bytes"))
            # v1 reports a huge sentinel (~2^63) when unlimited.
            if limit is not None and limit < (1 << 62):
                cur = _read_first_int(os.path.join(cdir, "memory.usage_in_bytes")) or 0
                return max(0.0, (limit - cur) / (1024 ** 3))
    return None


def _slurm_mem_gb() -> Optional[float]:
    """Memory granted to this SLURM job, in GB, or None if not under SLURM.

    SLURM exports the job's memory grant directly, which is the authoritative
    cap on an HPC node and avoids depending on how the scheduler wires up the
    memory cgroup (whose enforced limit can sit on a parent cgroup that simple
    path resolution misses). ``SLURM_MEM_PER_NODE`` is the whole-job total;
    ``SLURM_MEM_PER_CPU`` must be multiplied by the allocated CPU count. Both
    are in MB.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            return int(per_node) / 1024.0
        except ValueError:
            pass
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if per_cpu:
        cpus = (os.environ.get("SLURM_CPUS_PER_TASK")
                or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
                or os.environ.get("SLURM_CPUS_ON_NODE") or "")
        m = re.match(r"\d+", str(cpus))  # e.g. "16" or "16(x2)"
        if m:
            try:
                return int(per_cpu) * int(m.group()) / 1024.0
            except ValueError:
                pass
    return None


# ── VRAM shard planning ──────────────────────────────────────────────────────
# The reference k-mer vocabulary is held on the GPU and every genome is merged
# against it. When it (plus a genome's merge transient) would exceed VRAM, we
# split the reference into contiguous column-blocks and build the matrix in B
# passes. B is sized from free VRAM, the reference width, and the largest genome.

_VRAM_SAFETY_FRAC   = 0.80          # leave headroom for fragmentation/driver
_VRAM_TRANSIENT_MULT = 3            # genome df + join hash table + output + scratch
_VRAM_TRANSIENT_FLOOR = 1 << 30     # never reserve less than 1 GiB for the transient
_VRAM_BLOCK_FLOOR    = 64 << 20     # a block must be allowed at least 64 MiB


def plan_vram_shards(n_unique: int, n_words: int, max_genome_bytes: int,
                     free_vram_bytes: int, force: int = 0) -> List[Tuple[int, int]]:
    """Split the reference into contiguous column-blocks that fit in VRAM.

    Returns a list of ``(lo, hi)`` row ranges of the reference (row order ==
    column order). A single block ``[(0, n_unique)]`` means no sharding — the
    original single-pass behaviour.

    Sizing (adaptive, no hardcoded block count):
      * reference bytes  R = n_unique * (8*n_words + 4)   — exact for the packed
        uint64 keys + uint32 column index (fixed-width cuDF columns).
      * transient  T = max(1 GiB, 3 * largest-genome df bytes), estimated from
        the largest input (≈ one distinct k-mer per base). This is the part that
        does NOT shrink with more blocks, since each genome merges in full.
      * block budget = free_vram*0.80 − T ; B = ceil(R / block_budget).

    ``force`` > 0 overrides B (used by KMX_FORCE_SHARDS for testing sharding on
    data whose reference would otherwise fit in one pass).
    """
    if n_unique <= 0:
        return [(0, 0)]
    bytes_per_row   = 8 * n_words + 4
    R               = n_unique * bytes_per_row
    genome_df_bytes = max_genome_bytes * bytes_per_row          # ~1 distinct kmer / base
    transient       = max(_VRAM_TRANSIENT_FLOOR, _VRAM_TRANSIENT_MULT * genome_df_bytes)
    usable          = free_vram_bytes * _VRAM_SAFETY_FRAC
    block_budget    = max(usable - transient, _VRAM_BLOCK_FLOOR)

    if force and force > 0:
        B = int(force)
    else:
        B = max(1, math.ceil(R / block_budget)) if free_vram_bytes > 0 else 1
    B = max(1, min(B, n_unique))

    step = math.ceil(n_unique / B)
    return [(lo, min(lo + step, n_unique)) for lo in range(0, n_unique, step)]


def available_ram_gb() -> Optional[float]:
    """Best-effort RAM this process may actually use, in GB.

    Takes the *minimum* of the node-level ``MemAvailable``, the memory cgroup
    headroom, and any SLURM memory grant, so the budget is correct on a bare
    node, inside a container/cgroup limit, and under an HPC scheduler alike.
    Returns None only if nothing is readable (caller then falls back to fixed
    defaults).
    """
    cands = [v for v in (_meminfo_available_gb(),
                         _cgroup_available_gb(),
                         _slurm_mem_gb())
             if v is not None and v > 0]
    return min(cands) if cands else None


def plan_ram_budget(n_cpus: int, n_genomes: int,
                    spill_override_gb: float = 0.0,
                    worker_df_gb: float = 0.0) -> RamBudget:
    """Build a :class:`RamBudget` from the RAM available at launch.

    Args:
        n_cpus: total threads the run was given (already resolved, > 0).
        n_genomes: number of genomes (rows); caps the worker count, since
            stage 2 never runs more workers than there are genomes.
        spill_override_gb: explicit Tier-2 accumulator cap from ``--max-ram-gb``.
            0 = auto-derive from detected RAM; > 0 = use as-is (a small value
            forces the disk-spill path for testing).
        worker_df_gb: estimated RAM a single worker holds for its genome's k-mer
            table (peak, in GB). Added to the KMC budget to form the true
            per-worker footprint, which sets how many workers can run at once.
            A big genome (e.g. chr19 at high k ~ 1.2 GB) therefore runs FEWER
            concurrent workers so the stage cannot OOM. 0 = unknown (footprint
            falls back to the KMC budget alone, the old behaviour).
    """
    n_workers = max(1, min(int(n_cpus), int(n_genomes)))
    detected = available_ram_gb()

    # Could not detect RAM: fall back to fixed, conservative targets.
    if not detected or detected <= 0:
        spill = spill_override_gb if spill_override_gb and spill_override_gb > 0 \
            else _FALLBACK_SPILL_GB
        return RamBudget(
            stage1_gb=_FALLBACK_STAGE1_GB,
            per_worker_gb=_FALLBACK_PER_WORKER_GB,
            spill_threshold_gb=spill,
            n_workers=n_workers,
            detected_gb=None,
            source="fallback",
        )

    # Leave the OS / page cache a margin; never hand out everything.
    margin = max(_MARGIN_FLOOR_GB, _MARGIN_FRAC * detected)
    usable = max(detected - margin, _STAGE1_MIN_GB)

    # Stage 1 runs alone with every thread -> give it essentially all usable RAM.
    stage1 = max(usable, _STAGE1_MIN_GB)

    # Stage 2: reserve for the main process (reference set + GPU staging +
    # accumulator), then fit concurrent workers into the REST.
    #
    # Each worker holds two things at once: KMC's sort buffer (its max_ram
    # budget) AND the genome's resulting k-mer table (worker_df_gb), which is
    # NOT spillable while the worker is producing it. The per-worker FOOTPRINT
    # is the sum. We cannot run all n_cpus workers if their combined footprint
    # exceeds the pool: 16 chr19/k61 workers (~3 GB KMC + ~1.2 GB table each)
    # need ~67 GB and OOM a 32 GB node no matter how aggressively we spill the
    # accumulator/hand-off. So size the CONCURRENT worker count to the footprint
    # — fewer, well-fed workers on a tight box; full width on a roomy one.
    main_reserve = max(_MAIN_RESERVE_FLOOR_GB, _MAIN_RESERVE_FRAC * usable)
    worker_pool = max(usable - main_reserve, _PER_WORKER_MIN_GB)
    # KMC budget per worker: modest — one genome never needs a huge sort buffer.
    per_worker = min(_PER_WORKER_CAP_GB, max(_PER_WORKER_MIN_GB,
                                             worker_pool / n_workers))
    # Real per-worker PEAK is larger than (budget + table): while KMC counts the
    # genome it holds its sort/hash structures AND then materialises the k-mer
    # table, both ~the table's size, so they briefly coexist. Counting the table
    # ~2x (plus the KMC budget) matches measured chr19/k61 peaks (~4.5 GB for a
    # ~1.2 GB table). Under-counting this is what let 6 workers OOM a 32 GB box.
    footprint = per_worker + _WORKER_DF_PEAK_MULT * max(0.0, worker_df_gb)
    n_fit = max(1, int(worker_pool // footprint))
    n_workers = min(n_workers, n_fit)

    spill = spill_override_gb if spill_override_gb and spill_override_gb > 0 \
        else main_reserve

    # stage1_gb / per_worker_gb feed KMC's max_ram_gb, whose pybind11 binding
    # is typed `int` and rejects a Python float — keep them whole GB.
    return RamBudget(
        stage1_gb=max(2, int(stage1)),       # KMC requires max_ram_gb >= 2
        per_worker_gb=max(2, int(per_worker)),
        spill_threshold_gb=round(spill, 2),
        n_workers=n_workers,
        detected_gb=round(detected, 2),
        source="auto",
    )
