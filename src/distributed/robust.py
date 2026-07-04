"""Fault-tolerant per-genome counting — the shared robustness primitive for EVERY regime
(single/multi node, single/multi GPU; counting is CPU/KMC in all of them).

Deliberately SIMPLE and stateless (no cross-run cache/resume — that path invited
stale-result bugs). Two guarantees:
  1. ISOLATION + TIMEOUT + RETRY — each genome counts in its own process (fork); a hung
     KMC is killed at `timeout_s`, a crash/exception is retried up to `retries`.
  2. QUARANTINE + LOUD WARNING — a genome that fails every attempt is skipped, and an
     `on_warn` callback fires so it lands in the run report (a dropped genome is NEVER
     silent). The result is COUNT-ONCE within the run, spilled to the run's own dir.

Pair with runlog.RunLogger: pass `on_warn=log.warn` so quarantines appear in the
per-run report's WARNINGS section, and `spill_dir=log.path("counts")` for run isolation.
"""
import multiprocessing as mp
import os
import tempfile
from .reference import count_genome


def _count_child(q, genome, k, out_path, kw):
    try:
        count_genome(genome, k, **kw).to_parquet(out_path)
        q.put(("ok", out_path))
    except BaseException as e:                       # any failure -> reported, not fatal
        q.put(("err", f"{type(e).__name__}: {e}"))


def safe_count_genome(genome, k, out_path, *, retries=2, timeout_s=900, **kw):
    """Count ONE genome in isolation with timeout + retry. (out_path, None) on success,
    (None, reason) if every attempt failed. Hang killed at timeout_s; crash retried."""
    # fork (not spawn): a library must not require callers to guard __main__; the child
    # only execs KMC (a CPU subprocess) so forking after a cudf import is safe here.
    ctx = mp.get_context("fork")
    last = "unknown"
    for _ in range(int(retries) + 1):
        q = ctx.Queue()
        p = ctx.Process(target=_count_child, args=(q, genome, k, out_path, kw))
        p.start(); p.join(timeout_s)
        if p.is_alive():                             # HANG -> kill, retry
            p.terminate(); p.join(5)
            if p.is_alive():
                try: p.kill()
                except Exception: pass
                p.join(5)
            last = f"timeout>{timeout_s}s (killed)"
            continue
        try:
            status, payload = q.get_nowait()
        except Exception:                            # CRASH (segfault: child died mute)
            last = f"crash exitcode={p.exitcode}"; continue
        if status == "ok":
            return out_path, None
        last = payload                               # exception inside count -> retry
    return None, last


def count_all(genomes, k, *, spill_dir=None, retries=2, timeout_s=900,
              on_warn=None, on_event=None, **kw):
    """Count ALL genomes, fault-tolerantly. Returns (paths, kept_idx, quarantined):
      paths       : parquet path per KEPT genome (count-once, this run's spill dir)
      kept_idx    : original indices of the genomes that succeeded
      quarantined : [(idx, genome, reason)] for genomes skipped after exhausting retries
    A failed genome is SKIPPED (never fatal) and reported via on_warn so it appears in the
    run report — n_kept < n_genomes is always visible, never silent."""
    spill_dir = spill_dir or tempfile.mkdtemp()
    os.makedirs(spill_dir, exist_ok=True)
    paths, kept, quarantined = [], [], []
    for i, g in enumerate(genomes):
        out = os.path.join(spill_dir, f"g{i:06d}.parquet")
        path, reason = safe_count_genome(g, k, out, retries=retries, timeout_s=timeout_s, **kw)
        if path is None:
            quarantined.append((i, g, reason))
            msg = f"QUARANTINE genome {i} ({g}): {reason} — skipped after {retries} retr{'y' if retries==1 else 'ies'}"
            (on_warn or (lambda m: None))(msg)
        else:
            paths.append(path); kept.append(i)
            if on_event and (i % 100 == 0):
                on_event(f"counted {i + 1}/{len(genomes)}")
    return paths, kept, quarantined
