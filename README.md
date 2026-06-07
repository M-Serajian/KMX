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
streamed through the GPU one at a time, and written to disk **in input order** within an **automatic memory budget**.
Peak GPU and host memory stay bounded *regardless of dataset size*, so even datasets whose matrix is far larger
than RAM or VRAM run on a modest box.

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

That's it — KMX sizes CPUs, GPU, and RAM automatically. Outputs land in `results/`.

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
| `-d`, `--disable-normalization` | no | off | Treat a k-mer and its reverse complement as distinct (default: canonical). |
| `-T`, `--threads` | no | `0` | CPU threads; `0` = all cores. |
| `--max-ram-gb` | no | `0` (auto) | Cap on host RAM for the in-memory accumulator before it spills to disk. `0` = auto from the cgroup/SLURM limit. |

### Automatic resource budgeting

KMX picks the worker count, threads-per-worker, and spill thresholds **automatically** from the cores, host RAM
(cgroup/SLURM), and free VRAM it's given. No tuning needed. Three optional env vars override the defaults:

| Variable | Effect |
|----------|--------|
| `KMX_KMC_THREADS` | KMC threads per worker (default: `cores / workers`). |
| `KMX_WORKER_SPILL_MB` | Per-genome table size above which a worker spills to disk (`0` = always in RAM). |
| `KMX_SPILL_DIR` | Where disk-spill files go (default: `<tmp_dir>`). Point at a roomy/fast volume. |

---

## Output

Written to `-o`, with a suffix `k{K}_min{MIN}_max{MAX}_d{0|1}` (`d1` = normalization disabled):

| File | Description |
|------|-------------|
| `data_<suffix>.npy` | CSR non-zero values (counts). |
| `column_<suffix>.npy` | CSR column indices. |
| `row_<suffix>.npy` | CSR row pointers (row `i` = genome `i`, manifest order). |
| `set_of_all_unique_kmers_<suffix>.csv` | Column index → k-mer string. |
| `genome_index_<suffix>.csv` | Row index → `sample_id`. |
| `feature_matrix_stats_<suffix>.txt` | Sparsity, parameters, processing time, peak GPU memory. |

### Load the matrix

```python
import numpy as np, scipy.sparse          # or: import cupy as np, cupyx.scipy.sparse as scipy_sparse
s = "k31_min5_max100_d0"
M = scipy.sparse.csr_matrix((
        np.load(f"data_{s}.npy"),
        np.load(f"column_{s}.npy"),
        np.load(f"row_{s}.npy")))
```

Use `cupy` + `cupyx.scipy.sparse` for the GPU version.

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
Defaults match the CLI (`threads=0` = all cores, `max_ram_gb=0` = auto), and the `KMX_*` env knobs apply.
Pass `write_output=False` to skip the files and only return the arrays (this builds the matrix in RAM,
so use it only when the matrix fits memory). For advanced use, `KMX.create_csr_matrix(...)` is the
low-level builder that takes already-parsed manifest inputs.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: cudf` / `cupy` | Activate the env: `conda activate KMX-env` (RAPIDS must be installed). |
| `pip` fails building **kmcpy** | Ensure compilers + zlib/bzip2 are in the env; see the [KMC-DataFrame](https://github.com/M-Serajian/KMC-DataFrame) repo. |
| `--max (..) must be ≥ --min` | Raise `--max` or lower `--min`. |
| `manifest mixes FASTA and FASTQ` | Split into one manifest per format family. |
| Out-of-memory | KMX auto-adapts, but a very large reference at high `k` may need more RAM/VRAM — give the job more memory, or lower `--max-ram-gb` to spill harder. |
| Slow | The GPU merge is the throughput ceiling; check `nvidia-smi` for contention. |

---

## Citation

> **Serajian M.**, *et al.* "KMX: GPU-Accelerated K-mer Matrix Constructor." (in preparation).

## License

MIT — © 2025 Mohammadali (Ali) Serajian.
Questions: **ma.serajian@gmail.com** or open an [issue](https://github.com/M-Serajian/KMX/issues).
