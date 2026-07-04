"""Per-run reporting + isolation.

Every run gets a UNIQUE id and its OWN directory, so multiple runs (concurrent or
sequential) never mix outputs, temps, or logs. The RunLogger samples peak host RAM and
peak VRAM in the background, times named stages, collects warnings (e.g. quarantined
genomes — so a dropped genome is NEVER silent), and at the end writes BOTH a
human-readable report (run_report.txt) and a machine-readable one (run_report.json)
with inputs, resources, timing, matrix dimensions, and sparsity.

    log = RunLogger(out_dir, params={...})
    with log.stage("counting"): ...
    log.warn("QUARANTINE genome 7 (...): KMC crash")
    log.finalize(n_genomes=100, n_kept=98, rows=98, cols=4_155_479, nnz=95_254_323)

The logger never raises into the pipeline: every sampler/IO path is guarded.
"""
import collections
import contextlib
import datetime
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid


def new_run_id():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    host = socket.gethostname().split(".")[0]
    return f"{ts}_{host}_{os.getpid()}_{uuid.uuid4().hex[:6]}"     # collision-proof across runs/hosts


class RunLogger:
    def __init__(self, out_dir, *, params=None, run_id=None, sample=True):
        self.run_id = run_id or new_run_id()
        self.run_dir = os.path.join(os.path.abspath(out_dir), f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.params = dict(params or {})
        self.started = datetime.datetime.now().isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        self.stages = collections.OrderedDict()
        self.warnings = []
        self.info = {}
        self.peak = {"rss_gb": 0.0, "vram_gb": 0.0}
        self.gpus = self._gpu_info()
        self._stop = threading.Event()
        self._threads = []
        if sample:
            self._threads.append(threading.Thread(target=self._ram_sampler, daemon=True))
            if self.gpus:
                self._threads.append(threading.Thread(target=self._vram_sampler, daemon=True))
            for t in self._threads:
                t.start()

    # ---- public ----
    def path(self, *p):
        return os.path.join(self.run_dir, *p)

    @contextlib.contextmanager
    def stage(self, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round(self.stages.get(name, 0.0) + (time.perf_counter() - t), 2)

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"[KMX][WARN] {msg}", file=sys.stderr, flush=True)

    def set(self, **kv):
        self.info.update(kv)

    def finalize(self, *, n_genomes=None, n_kept=None, rows=None, cols=None, nnz=None,
                 reference_kmers=None, total_kmers=None, output_dir=None):
        self._stop.set()
        time.sleep(0.3)
        wall = round(time.perf_counter() - self._t0, 1)
        dense = (rows or 0) * (cols or 0)
        sparsity = round(100.0 * (1 - nnz / dense), 4) if (dense and nnz is not None) else None
        try:
            cpus = len(os.sched_getaffinity(0))
        except Exception:
            cpus = os.cpu_count()
        rep = dict(
            run_id=self.run_id, started=self.started, finished=datetime.datetime.now().isoformat(timespec="seconds"),
            host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
            params=self.params, n_genomes=n_genomes, n_kept=n_kept,
            n_quarantined=(None if n_genomes is None or n_kept is None else n_genomes - n_kept),
            matrix_rows=rows, matrix_cols=cols, nnz=nnz, sparsity_pct=sparsity,
            reference_kmers=reference_kmers, total_kmers_counted=total_kmers,
            cpus=cpus, gpus=self.gpus, peak_host_ram_gb=round(self.peak["rss_gb"], 2),
            peak_vram_gb=round(self.peak["vram_gb"], 2), wall_s=wall,
            stages_s=dict(self.stages), info=self.info, warnings=self.warnings,
            output_dir=output_dir or self.run_dir)
        with open(self.path("run_report.json"), "w") as fh:
            json.dump(rep, fh, indent=2)
        text = self._render(rep)
        with open(self.path("run_report.txt"), "w") as fh:
            fh.write(text)
        print("\n" + text, flush=True)
        return rep

    # ---- internals (all guarded; never raise into the pipeline) ----
    def _gpu_info(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
            return [l.strip() for l in out.splitlines() if l.strip()]
        except Exception:
            return []

    def _cgroup_peak_gb(self):
        for p in ("/sys/fs/cgroup/memory.peak",
                  "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
            try:
                return int(open(p).read().strip()) / 1024 ** 3
            except Exception:
                pass
        return None

    def _tree_rss_gb(self):
        children = collections.defaultdict(list); rss = {}
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/stat") as f:
                        ppid = int(f.read().split()[3])
                    children[ppid].append(int(pid))
                    with open(f"/proc/{pid}/status") as f:
                        for ln in f:
                            if ln.startswith("VmRSS"):
                                rss[int(pid)] = int(ln.split()[1]); break
                except Exception:
                    pass
            tot, stack = 0, [os.getpid()]
            while stack:
                p = stack.pop(); tot += rss.get(p, 0); stack += children.get(p, [])
            return tot / 1024 ** 2
        except Exception:
            return 0.0

    def _ram_sampler(self):
        while not self._stop.is_set():
            g = self._cgroup_peak_gb()
            if g is None:
                g = self._tree_rss_gb()
            if g:
                self.peak["rss_gb"] = max(self.peak["rss_gb"], g)
            self._stop.wait(0.5)

    def _vram_sampler(self):
        try:
            proc = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits", "-lms", "300"],
                                    stdout=subprocess.PIPE, text=True)
            for line in proc.stdout:
                if self._stop.is_set():
                    break
                try:
                    self.peak["vram_gb"] = max(self.peak["vram_gb"], float(line.strip()) / 1024.0)
                except Exception:
                    pass
            proc.terminate()
        except Exception:
            pass

    def _render(self, r):
        L = []
        bar = "=" * 66
        L.append(bar); L.append(f"  KMX RUN REPORT   {r['run_id']}"); L.append(bar)
        L.append(f"  started   {r['started']}    host {r['host']}"
                 f"{('   SLURM ' + r['slurm_job']) if r['slurm_job'] else ''}")
        if r["params"]:
            L.append("  params    " + ", ".join(f"{k}={v}" for k, v in r["params"].items()))
        L.append("")
        L.append("  -- inputs ----------------------------------------------------")
        L.append(f"  genomes              {r['n_genomes']}"
                 + (f"   ({r['n_kept']} counted, {r['n_quarantined']} QUARANTINED — see warnings)"
                    if r['n_quarantined'] else ""))
        if r["total_kmers_counted"] is not None:
            L.append(f"  total k-mers counted {r['total_kmers_counted']:,}")
        L.append("")
        L.append("  -- output ----------------------------------------------------")
        if r["matrix_rows"] is not None:
            L.append(f"  matrix               {r['matrix_rows']:,} genomes x {r['matrix_cols']:,} k-mers")
            L.append(f"  nonzeros             {r['nnz']:,}")
            L.append(f"  sparsity             {r['sparsity_pct']} %")
        if r["reference_kmers"] is not None:
            L.append(f"  reference k-mers     {r['reference_kmers']:,}")
        L.append("")
        L.append("  -- resources -------------------------------------------------")
        L.append(f"  CPUs                 {r['cpus']}")
        L.append(f"  GPUs                 {len(r['gpus'])}" + (f"   ({'; '.join(r['gpus'])})" if r['gpus'] else ""))
        L.append(f"  peak host RAM        {r['peak_host_ram_gb']} GB")
        L.append(f"  peak VRAM            {r['peak_vram_gb']} GB")
        L.append("")
        L.append("  -- timing ----------------------------------------------------")
        L.append(f"  total wall           {r['wall_s']} s")
        for k, v in r["stages_s"].items():
            L.append(f"    {k:<18} {v} s")
        if r["warnings"]:
            L.append("")
            L.append(f"  -- WARNINGS ({len(r['warnings'])}) " + "-" * (48 - len(str(len(r['warnings'])))))
            for w in r["warnings"]:
                L.append(f"  ! {w}")
        L.append("")
        L.append(f"  output dir  {r['output_dir']}")
        L.append(bar)
        return "\n".join(L)
