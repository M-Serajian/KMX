# KMX — GPU-Accelerated K-mer Matrix Constructor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-blue.svg)]()
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![RAPIDS cuDF](https://img.shields.io/badge/RAPIDS-cuDF-orange.svg)]()

KMX turns a set of genomes into a **genome × k-mer count matrix** in sparse **CSR** format.
You give it a manifest that maps each genome to its sequence file(s); KMX counts k-mers
([KMC](https://github.com/refresh-bio/KMC) via [kmcpy](https://github.com/M-Serajian/KMC-DataFrame)),
merges them against the global k-mer set on the GPU ([cuDF](https://github.com/rapidsai/cudf)/[CuPy](https://cupy.dev/)),
and writes the matrix to disk.

**What makes it scale:** the matrix is built **out-of-core** — genomes are counted in parallel on the CPU,
streamed through the GPU, and written to disk **in input order** within an **automatic memory budget**.
Peak GPU and host memory stay bounded *regardless of dataset size*, so even datasets whose matrix is far larger
than RAM or VRAM run on a modest box. When several GPUs are available, the per-genome merge runs in parallel
across them automatically — replicating the reference and splitting the genomes when it fits one GPU, or sharding
the k-mer columns across GPUs when it doesn't (see [Multiple GPUs](#multiple-gpus)) — and rows still land in input order.

---

## Requirements

- **Linux** (RAPIDS cuDF is Linux-only) with an **NVIDIA GPU** (CUDA 12.x).
- **Python ≥ 3.10**, and **RAPIDS cuDF/CuPy** (installed via conda — not pip-installable on most clusters).
- Tested with **cuDF 25.06 / CUDA 12.8**.
- K-mer size **8–136**.

---

## Installation

KMX is a pip package; its only special dependency is **kmcpy** (a KMC C++ extension, built on install)
and **cuDF/CuPy** (from the RAPIDS conda channel).

```bash
# 1. Create an environment with RAPIDS + the tools kmcpy needs to compile
conda create -n KMX-env -c rapidsai -c conda-forge -c nvidia \
    python=3.11 cudf=25.06 cuda-version=12.8 \
    gxx_linux-64 make zlib bzip2 git -y
conda activate KMX-env

# 2. Install KMX (also builds + installs kmcpy / KMC-DataFrame)
pip install git+https://github.com/M-Serajian/KMX.git
```

For development, clone and install editable:

```bash
git clone https://github.com/M-Serajian/KMX.git
pip install -e KMX
```

> kmcpy build details (compilers, zlib/bzip2) are documented in the
> [KMC-DataFrame repo](https://github.com/M-Serajian/KMC-DataFrame). If `pip` can't build it,
> install kmcpy from there first, then `pip install` KMX.

Verify:

```bash
conda activate KMX-env
KMX --help
```

---

## Quick Start

```bash
KMX -l manifest.csv -k 31 -t /scratch/tmp -o results/ --min 5 --max 100
```

That's it — KMX sizes CPUs, GPUs, and RAM automatically. Outputs land in `results/`.

---

## Input: the manifest

A two-column CSV mapping each genome to its file(s):

```csv
sample_id,file
GENOME_A,/data/A_R1.fq.gz
GENOME_A,/data/A_R2.fq.gz     # same sample_id -> merged into one row
GENOME_B,/data/B.fasta
```

- **`sample_id`** — any label; rows with the same label become **one matrix row** (KMC merges their counts).
  So paired-end reads, multi-lane runs, and multi-contig assemblies are just multiple rows with the same `sample_id`.
- **`file`** — absolute, or relative to the manifest's directory; must exist.
- Header `sample_id,file` required. Blank lines and `#` comments are skipped; relative paths resolve next to the manifest.

**Supported types:** FASTA (`.fa/.fasta/.fna/.faa`) and FASTQ (`.fq/.fastq`), each optionally `.gz`.
**One family per manifest** — all FASTA *or* all FASTQ (KMX builds the global k-mer set with a single KMC call, which takes one format).

The manifest is fully validated before any counting starts (clear errors for a missing file, wrong header, mixed families, or an unsupported extension).

---

## Options

```
KMX -l <manifest.csv> -k <kmer_size> -t <tmp_dir> -o <output_dir> [options]
```

| Flag | Req | Default | Description |
|------|-----|---------|-------------|
| `-l`, `--genome-list` | yes | — | Manifest CSV (`sample_id,file`). |
| `-k`, `--kmer-size` | yes | — | K-mer length, **8–136**. |
| `-t`, `--tmp` | yes | — | Scratch dir for intermediate files (created if absent). |
| `-o`, `--output` | yes | — | Output dir (created if absent). |
| `--min` | no | `5` | Drop k-mers occurring fewer than this many times across the dataset. |
| `--max` | no | `N/2` | Drop k-mers occurring more than this (default: half the genome count). Must be ≥ `--min`. |
| `-p`, `--presence` | no | off | Store presence/absence (`int8` `0/1`) instead of counts — a 0/1 matrix for chi-squared / association tests, **4× smaller** than `float32` counts. Output files gain a `_presence` suffix. |
| `-d`, `--disable-normalization` | no | off | Treat a k-mer and its reverse complement as distinct (default: canonical). |
| `-T`, `--threads` | no | `0` | CPU threads; `0` = all cores. |
| `--max-ram-gb` | no | `0` (auto) | Cap on host RAM for the in-memory accumulator before it spills to disk. `0` = auto from the cgroup/SLURM limit. |
| `--max-gpus` | no | `0` (auto) | Max GPUs for the merge. With >1 GPU and a reference that fits one GPU, the merge runs data-parallel across them. `0` = auto (env `KMX_MAX_GPUS`, else up to 4). |
| `--reference` | no | — | Cached reference directory. With `--build-reference`, write it here; otherwise load it here and skip stage 1 (strictly validated). See [Caching the reference](#caching-the-reference). |
| `--build-reference` | no | off | Build **only** the reference set into `--reference` (no GPU), then stop. Run on a CPU node. |

### Automatic resource budgeting

KMX picks the worker count, threads-per-worker, spill thresholds, **and the multi-GPU layout** automatically from
the cores, host RAM (cgroup/SLURM), and the GPUs/VRAM it's given (see [Multiple GPUs](#multiple-gpus) for the layout
rules). Rows always come out in input order. No tuning needed. Optional env vars override the defaults:

| Variable | Effect |
|----------|--------|
| `KMX_MAX_GPUS` | Cap on GPUs used for the merge (default: up to 4 / all visible). |
| `KMX_KMC_THREADS` | KMC threads per worker (default: `cores / workers`). |
| `KMX_WORKER_SPILL_MB` | Per-genome table size above which a worker spills to disk (`0` = always in RAM). |
| `KMX_SPILL_DIR` | Where disk-spill files go (default: `<tmp_dir>`). Point at a roomy/fast volume. |

---

## Multiple GPUs

KMX chooses the layout automatically from two numbers: **`S`** = how many shards the reference needs
(`S = ⌈reference ÷ per-GPU VRAM⌉`, so `S = 1` means it fits one GPU) and **`D`** = the GPUs available. In every
case rows come out in **input order** and the reference is **counted once**. **The build is most efficient when the
whole reference fits one GPU (`S = 1`)** — keep it within one GPU's VRAM if you can.

| Scenario | Condition | Mode | What KMX does |
|----------|-----------|------|---------------|
| One GPU, reference fits | `D = 1`, `S = 1` | `single` | Builds on the one GPU, streaming rows to disk in input order. |
| One GPU, reference too big | `D = 1`, `S > 1` | `sharded_single` | Splits the reference into `S` column-blocks and merges them **sequentially** on the one GPU. Correct, slower. |
| Multi-GPU, reference fits one GPU | `D > 1`, `S = 1` | `data_parallel` | **Each GPU holds a full copy** of the reference; genomes load-balanced across GPUs. **The efficient case** — near-linear with GPU count. |
| Reference needs exactly all GPUs | `D > 1`, `D = S` | `reference_parallel` (1 group) | Reference **sharded across all `D` GPUs** (one shard each); every genome is broadcast to the shards and reassembled in column order. |
| More GPUs than shards | `D > 1`, `D > S` | `reference_parallel` (`R = ⌊D/S⌋` groups) | GPUs split into `R` even **replica groups** (e.g. 11 GPUs, `S = 4` → **6 + 5**). Each group holds the full reference sharded across its GPUs; **genomes data-parallel across groups**, reference-parallel within. All GPUs busy + genome parallelism. |
| Fewer GPUs than shards — **least efficient** | `D > 1`, `D < S` | ⚠ warning → `sharded_single` | Reference exceeds **all** GPUs' VRAM combined — the worst case: it can't be held even across every GPU, so KMX falls back to a **single-GPU sequential** build and the other GPUs sit idle. It **warns** with the fix and continues (still correct). |

**Avoiding the `D < S` case.** This is the configuration to design *out* of. The cheapest lever is usually **raising
`--min`**: it drops the rare/error k-mer tail (huge for FASTQ), which shrinks the reference and lowers `S` until it
fits your GPUs — turning the slow single-GPU fallback back into a parallel build. The other option is to allocate
more / higher-VRAM GPUs (`D ≥ S`). Raising `--min` *changes which k-mers are kept* (a feature-set change, not just
speed), so choose a threshold you're comfortable with. Use the [capacity table](#how-many-k-mers-fit-on-one-gpu)
to see how many k-mers each GPU holds, and HLL/ntCard to estimate your reference size before committing.

### How many k-mers fit on one GPU

A resident reference k-mer costs `8·⌈k/32⌉ + 4` bytes (the packed 2-bit key + its column index). For **k ≤ 32**
that's **12 bytes**, so a GPU holds roughly `VRAM × 0.80 ÷ 12` distinct k-mers — the `0.80` is KMX's default VRAM
headroom; a ~1 GB per-genome working reserve trims it a little more in practice:

| GPU | VRAM | ~max distinct k-mers (k ≤ 32) |
|-----|------|-------------------------------|
| NVIDIA T4    | 16 GB  | ~1.1 billion |
| NVIDIA L4    | 24 GB  | ~1.7 billion |
| V100         | 32 GB  | ~2.3 billion |
| A100         | 40 GB  | ~2.9 billion |
| L40S         | 48 GB  | ~3.4 billion |
| A100 / H100  | 80 GB  | ~5.7 billion |
| H200         | 141 GB | ~10 billion  |
| B200         | 192 GB | ~14 billion  |

**Longer k shrinks this in steps, not linearly.** The cost `8·⌈k/32⌉ + 4` jumps by 8 bytes every 32 bases, and is
**flat within a band** — so k = 31 is already in the top (cheapest) band, and *every* k from 1–32 fits the same
count. To convert the table for longer k, multiply by `12 / (8·⌈k/32⌉ + 4)`:

| k | bytes/k-mer | capacity vs table |
|---|-------------|-------------------|
| ≤ 32     | 12 | ×1.00 |
| 33–64    | 20 | ×0.60 |
| 65–96    | 28 | ×0.43 |
| 97–128   | 36 | ×0.33 |
| 129–136  | 44 | ×0.27 |

So if your reference is just over one GPU at, say, k = 35, dropping to k ≤ 32 alone gives ~1.7× the capacity and may
let it fit (changing k changes the feature set, though).

---

## Caching the reference

The first stage — building the **global k-mer set** (the matrix columns) — reads the whole dataset once and uses
**no GPU**. For big inputs (e.g. FASTQ) it's the slow part. You can build it **once on a CPU node**, cache it to
disk, and reuse it so later runs **skip stage 1** (and don't burn GPU time on it):

```bash
# 1) Build + cache the reference (CPU node, no GPU). Writes ref/.
KMX --build-reference -l manifest.csv -k 31 --min 5 --max 10000 -t /scratch/tmp --reference ref/

# 2) Build the matrix, reusing it (GPU node) — stage 1 is skipped.
KMX -l manifest.csv -k 31 --min 5 --max 10000 -t /scratch/tmp -o results/ --reference ref/
```

`ref/` holds `reference.parquet` (the **packed 2-bit** KMC keys — exactly what the merge consumes, no decoding)
plus `reference_meta.json`. The reference is **strictly validated** on load: `-k`, `--min`, `--max`, normalization,
and the input file set must all match what it was built from, or KMX refuses (a stale reference would silently
corrupt the matrix). Rebuild with `--build-reference` if any of those change.

---

## Output

Written to `-o`, with a suffix `k{K}_min{MIN}_max{MAX}_d{0|1}` (`d1` = normalization disabled):

| File | Description |
|------|-------------|
| `data_<suffix>.npy` | CSR values — `float32` counts (or `int8` `0/1` with `--presence`). |
| `column_<suffix>.npy` | CSR column indices — signed `int32` (`int64` above ~2.1 B columns). |
| `row_<suffix>.npy` | CSR row pointers (`int64`; row `i` = genome `i`, manifest order). |
| `set_of_all_unique_kmers_<suffix>.csv` | Column index → k-mer string. |
| `genome_index_<suffix>.csv` | Row index → `sample_id`. |
| `feature_matrix_stats_<suffix>.txt` | Sparsity, parameters, processing time, peak GPU memory. |

The arrays use **compute-native dtypes** so they load into `scipy.sparse` and `cupyx.scipy.sparse` (RAPIDS) with
no conversion: **signed `int32`/`int64`** column indices (int64 only when the reference exceeds ~2.1 B columns),
`int64` row pointers, and `float32` values (or `int8` `0/1` with `--presence`). `float32` counts are exact up to
**16,777,216** (2²⁴) — far above any realistic per-genome k-mer count; a single k-mer occurring more than that in one
genome would round, so use `--presence` (or treat such extreme counts with care) in that unusual case.

### Load the matrix

One call returns a ready CSR with the correct shape — CPU (scipy) or GPU (RAPIDS):

```python
import KMX
M_cpu = KMX.load_csr("results/")                 # scipy.sparse.csr_matrix → scikit-learn, statsmodels
M_gpu = KMX.load_csr("results/", device="gpu")   # cupyx.scipy.sparse.csr_matrix → cuML, CuPy (RAPIDS)

# straight into a chi-squared feature test (presence/absence with --presence):
from sklearn.feature_selection import chi2
chi2_scores, p_values = chi2(M_cpu, labels)       # labels: one per genome (manifest order)
```

Or build it yourself from the three `.npy` files (pass the shape so all-zero trailing columns aren't dropped):

```python
import numpy as np, scipy.sparse
s = "k31_min5_max100_d0"
n_cols = sum(1 for _ in open(f"set_of_all_unique_kmers_{s}.csv")) - 1
data, col, row = (np.load(f"{x}_{s}.npy") for x in ("data", "column", "row"))
M = scipy.sparse.csr_matrix((data, col, row), shape=(row.size - 1, n_cols))
```

For RAPIDS, swap `numpy`→`cupy` and `scipy.sparse`→`cupyx.scipy.sparse`.

**`--presence`** stores a 0/1 matrix (`int8` values) instead of counts — ideal for chi-squared / association tests.
Beyond being the right input for those tests, the `int8` `data` array is **4× smaller than `float32` counts**, which
meaningfully shrinks the VRAM and I/O footprint on large datasets (more genomes in flight per GPU, lighter spill/queue
traffic). It stays correct for the sums those tests perform: `scipy`/`numpy`/`cupy` upcast the `int8` accumulator to
`int64`, so column sums (e.g. inside `chi2`) don't overflow. (The only thing to avoid is an op that accumulates while
*staying* `int8`, e.g. an `int8 @ int8` Gram product — not something standard association pipelines do.)

---

## Python API

The whole pipeline is one function. **Call it inside `if __name__ == "__main__":`** — KMX uses
multiprocessing (spawn), so an *unguarded* top-level call would re-spawn itself.

```python
import KMX

if __name__ == "__main__":
    data, column, row, kmers, sparsity = KMX.build(
        "manifest.csv",
        kmer_size=31,
        tmp_dir="/scratch/tmp",
        output_dir="results/",
        min_count=5,        # default 5
        max_count=None,     # default: half the genome count
    )
    # outputs are written to results/ (same as the CLI); arrays are also returned.
```

`KMX.build(...)` takes the same options as the CLI — see `help(KMX.build)` for the full signature.
Defaults match the CLI (`threads=0` = all cores, `max_ram_gb=0` = auto, `max_gpus=0` = auto/all GPUs), and the
`KMX_*` env knobs apply.
Pass `write_output=False` to skip the files and only return the arrays (this builds the matrix in RAM,
so use it only when the matrix fits memory). For advanced use, `KMX.create_csr_matrix(...)` is the
low-level builder that takes already-parsed manifest inputs.

To cache the reference once (CPU only) and reuse it (see [Caching the reference](#caching-the-reference)):

```python
import KMX
KMX.build_reference("manifest.csv", kmer_size=31, tmp_dir="/scratch/tmp",
                    reference_dir="ref/", min_count=5, max_count=10000)   # no GPU; writes ref/

if __name__ == "__main__":
    KMX.build("manifest.csv", kmer_size=31, tmp_dir="/scratch/tmp",
              output_dir="results/", min_count=5, max_count=10000,
              reference="ref/")          # loads ref/, skips stage 1 (strictly validated)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: cudf` / `cupy` | Activate the env: `conda activate KMX-env` (RAPIDS must be installed). |
| `pip` fails building **kmcpy** | Ensure compilers + zlib/bzip2 are in the env; see the [KMC-DataFrame](https://github.com/M-Serajian/KMC-DataFrame) repo. |
| `--max (..) must be ≥ --min` | Raise `--max` or lower `--min`. |
| `cached reference does not match this run` | The `--reference` cache was built with different `-k`/`--min`/`--max`/normalization or input files. Rebuild it with `--build-reference`. |
| `manifest mixes FASTA and FASTQ` | Split into one manifest per format family. |
| Out-of-memory | KMX auto-adapts, but a very large reference at high `k` may need more RAM/VRAM — give the job more memory, or lower `--max-ram-gb` to spill harder. |
| Slow | The GPU merge is the throughput ceiling — give the job more GPUs (KMX uses them data-parallel automatically), or check `nvidia-smi` for contention. |

---

## Citation

> **Serajian M.**, *et al.* "KMX: GPU-Accelerated K-mer Matrix Constructor." (in preparation).

## License

MIT — © 2025 Mohammadali (Ali) Serajian.
Questions: **ma.serajian@gmail.com** or open an [issue](https://github.com/M-Serajian/KMX/issues).
