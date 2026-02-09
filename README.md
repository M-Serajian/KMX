# KMX — GPU-Accelerated K-mer Matrix Constructor

## Overview

KMX is a high-performance, GPU-accelerated tool for extracting k-mers from large genomic datasets and constructing the corresponding feature matrix in Compressed Sparse Row (CSR) format. Given a list of genomes in FASTA format, KMX:

- Extracts k-mers from each genome using [gerbil-DataFrame](https://github.com/M-Serajian/gerbil-DataFrame), a CUDA-enabled k-mer counter forked from [Gerbil](https://github.com/uni-halle/gerbil).
- Constructs a CSR matrix where rows represent genomes and columns represent unique k-mers, with frequency values as entries.
- Leverages GPU acceleration through [RAPIDS cuDF](https://github.com/rapidsai/cudf) and [CuPy](https://cupy.dev/) for fast, memory-efficient DataFrame operations and sparse matrix assembly.
- Scales to large datasets with dynamic memory management and parallel processing.

The resulting CSR matrix integrates directly into machine learning workflows via `cupyx.scipy.sparse.csr_matrix` (GPU) or `scipy.sparse.csr_matrix` (CPU), making KMX well-suited for genomic feature extraction, clustering, classification, and other bioinformatics applications in high-performance computing (HPC) environments.

### K-mer Size

KMX supports k-mer sizes from **8 to 136** (inclusive). This range is inherited from the underlying [Gerbil](https://github.com/uni-halle/gerbil) k-mer counter, which encodes k-mers in a fixed-width binary representation with a compiled maximum of 136. See the [Gerbil documentation](https://github.com/uni-halle/gerbil) for details.

### Platform Support

**KMX runs on Linux only.** This is a hard requirement — [RAPIDS cuDF](https://rapids.ai/), which KMX depends on for GPU-accelerated DataFrame operations, is exclusively available on Linux. There is no Windows or macOS support.

### Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+ recommended) |
| **GPU** | NVIDIA GPU with CUDA support (compatible with RAPIDS 25.06) |
| **VRAM** | Depends on dataset size |
| **RAM** | Depends on dataset size |
| **Disk** | At least 10 GB free for temporary files; ~10 GB for installation |

---

## Citation

If you use KMX in your research, please cite:

> **Serajian M.**, *et al.* **"KMX: GPU-Accelerated K-mer Matrix Constructor."** *Journal Name*, **Volume** (Year): pages. DOI: xx.xxxx/xxxxxxxx

---

## Installation

### Prerequisites

Before installing KMX, ensure the following are available on your system:

| Prerequisite | Notes |
|-------------|-------|
| **Linux** | Required. RAPIDS cuDF is Linux-only. |
| **NVIDIA GPU + Driver** | A CUDA-capable GPU with an up-to-date NVIDIA driver. |
| **Conda or Mamba** | Required for the recommended installation method. Install from [Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). |
| **Git** | Required to clone the repository and the gerbil-DataFrame submodule. |
| **GCC ≤ 13** | gerbil-DataFrame does **not** compile with GCC 14+. GCC 12 is recommended and tested. Most Linux systems ship with a compatible version; if yours defaults to GCC 14+, install an older version (e.g., `sudo apt install gcc-12 g++-12`). |

### Option 1: Automated Installation with Conda (Recommended)

This method creates an isolated Conda environment with all dependencies (Python 3.11, CUDA Toolkit 12.6, cuDF, CuPy, Boost, CMake, etc.) and compiles the gerbil binary automatically.

```bash
# Clone the repository
git clone https://github.com/M-Serajian/KMX.git
cd KMX

# Run the setup script
python setup.py install
```

The setup script will:

1. Create a Conda environment named `KMX-env` with Python 3.11.
2. Install CUDA Toolkit 12.6, cuDF (RAPIDS 25.06), Boost 1.77, CMake, and build tools.
3. Clone the gerbil-DataFrame submodule.
4. Compile the gerbil binary with CUDA support.

**Estimated time:** 15–20 minutes (the CUDA Toolkit download is large).

After installation, verify everything is in place:

```bash
python setup.py verify
```

#### Maintenance Commands

```bash
# Verify installation health
python setup.py verify

# Rebuild gerbil (if compilation had errors)
rm -rf include/gerbil-DataFrame/build
python setup.py install

# Recreate the entire environment
python setup.py uninstall
python setup.py install

# Free disk space from conda package cache
conda clean --all -y
```

### Option 2: Manual Installation

Use this if you prefer to manage dependencies yourself or are working in an HPC environment with module systems.

#### Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11 | Runtime (must match RAPIDS version) |
| GCC | 12.x (≤ 13) | Compiling gerbil-DataFrame. **GCC 14+ is not compatible.** |
| CMake | ≥ 3.13 | Build system for gerbil-DataFrame |
| Boost | 1.77 | Required by gerbil-DataFrame |
| CUDA Toolkit | 12.x | GPU acceleration for gerbil and cuDF |
| RAPIDS cuDF | 25.06 | GPU-accelerated DataFrames |
| CuPy | (installed with cuDF) | GPU arrays and sparse matrix operations |
| Git | any | Cloning gerbil-DataFrame |
| zlib | any | Compression support for gerbil |
| libbz2 | any | Compression support for gerbil |

> ⚠️ **Important: Library Isolation in Manual Installations**
>
> When installing dependencies manually, be careful **not to mix libraries** from different sources. In particular:
>
> - **Boost:** gerbil-DataFrame is tested with **Boost 1.77**. Other versions may introduce build errors or ABI incompatibilities. If your system has a different Boost version installed, make sure the compiler picks up the correct one via `CMAKE_PREFIX_PATH` or `-DBOOST_ROOT`.
> - **GCC:** Use the **same GCC version** to compile gerbil-DataFrame and to link against your system libraries. Mixing GCC versions (e.g., compiling with GCC 12 but linking against libraries built with GCC 14) can cause `GLIBCXX` symbol errors at runtime.
> - **CUDA + RAPIDS:** RAPIDS cuDF ships its own CUDA runtime libraries. If you also have a system-wide CUDA installation, conflicting versions on `LD_LIBRARY_PATH` can cause crashes. Ensure the active CUDA version matches what cuDF expects.
>
> The automated Conda installation (Option 1) avoids these issues entirely by isolating all dependencies in a single environment.

##### HPC Environment (Module System)

```bash
ml gcc/12.2
ml cmake
ml boost/1.77
ml rapidsai/25.06   # or install via conda
ml python/3.11
```

##### Ubuntu/Debian

```bash
sudo apt-get update && sudo apt-get install -y \
    gcc-12 g++-12 \
    cmake \
    git \
    libboost-all-dev \
    zlib1g-dev \
    libbz2-dev
```

> ⚠️ RAPIDS cuDF must be installed separately via [Conda](https://rapids.ai/start.html) — it is not available through `apt`.

#### Build gerbil-DataFrame

```bash
git clone https://github.com/M-Serajian/KMX.git
cd KMX/include
git clone https://github.com/M-Serajian/gerbil-DataFrame.git
cd gerbil-DataFrame
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
make -j$(nproc)
cd ../../..
```

After building, verify the binary exists:

```bash
ls -la include/gerbil-DataFrame/build/gerbil
```

---

## Usage

**Before running KMX, activate the environment:**

```bash
conda activate KMX-env    # if using the automated Conda installation
```

### Command

```bash
python KMX.py -l <genome_list> -k <kmer_size> -t <tmp_dir> -o <output_dir> [options]
```

### Arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `-l`, `--genome-list` | str | **yes** | — | Path to a text file listing one FASTA file path per line. Blank lines and lines starting with `#` are ignored. |
| `-k`, `--kmer-size` | int | **yes** | — | K-mer length. Must be between **8** and **136** (inclusive), as [documented by Gerbil](https://github.com/uni-halle/gerbil). |
| `-t`, `--tmp` | path | **yes** | — | Temporary directory for intermediate files. Created if absent. Must have at least **10 GB** free space. |
| `-o`, `--output` | path | **yes** | — | Output directory for results. Created if absent. |
| `--min` | int | no | `5` | Minimum k-mer occurrence threshold. K-mers observed fewer than this many times across all genomes are discarded. |
| `--max` | int | no | `N/2` | Maximum k-mer occurrence threshold. K-mers observed more than this many times are discarded. If omitted, defaults to half the number of genomes in the genome list (`N/2`). Must satisfy `max ≥ min`. |
| `-d`, `--disable-normalization` | flag | no | off | Treat a k-mer and its reverse complement as distinct features. By default, both are mapped to the same canonical k-mer. |
| `-c`, `--cpu` | flag | no | off | Force CPU-only execution. By default, GPU acceleration is enabled. |

### Example

```bash
python KMX.py \
    -l genomes.txt \
    -k 31 \
    -t /scratch/tmp \
    -o /results/output \
    --min 5 \
    --max 100
```

---

## Output Files

All output files are written to the directory specified by `-o`. File names include a suffix encoding the run parameters: `k{K}_min{MIN}_max{MAX}_d{0|1}`, where `d0` means normalization is enabled and `d1` means it is disabled.

| File | Format | Description |
|------|--------|-------------|
| `data_<suffix>.npy` | NumPy `.npy` | Non-zero values of the CSR matrix (k-mer frequencies). |
| `row_<suffix>.npy` | NumPy `.npy` | Row pointer array of the CSR matrix. |
| `column_<suffix>.npy` | NumPy `.npy` | Column index array of the CSR matrix. |
| `set_of_all_unique_kmers_<suffix>.csv` | CSV | Mapping of column indices to canonical k-mer strings. |
| `feature_matrix_stats_<suffix>.txt` | Text | Run statistics: sparsity, parameters, timing, and peak GPU memory usage. |

### Reconstructing the CSR Matrix

```python
# GPU (CuPy)
import cupy as cp
import cupyx.scipy.sparse

data = cp.load("data_k31_min5_max100_d0.npy")
row = cp.load("row_k31_min5_max100_d0.npy")
column = cp.load("column_k31_min5_max100_d0.npy")
matrix = cupyx.scipy.sparse.csr_matrix((data, column, row))
```

```python
# CPU (SciPy)
import numpy as np
import scipy.sparse

data = np.load("data_k31_min5_max100_d0.npy")
row = np.load("row_k31_min5_max100_d0.npy")
column = np.load("column_k31_min5_max100_d0.npy")
matrix = scipy.sparse.csr_matrix((data, column, row))
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: .../gerbil` | The gerbil binary was not built. Run `python setup.py install` or build manually (see Option 2). |
| `GLIBC_2.XX not found` | Likely mixing system and conda libraries. Ensure `KMX-env` is activated before running KMX. |
| gerbil compilation fails with GCC errors | gerbil-DataFrame requires GCC ≤ 13. GCC 14+ is **not compatible**. Specify the compiler explicitly: `cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12`. |
| `GLIBCXX` version errors at runtime | The gerbil binary was compiled with a different GCC than the system libraries expect. Rebuild with the same GCC used by your system or use the Conda installation. |
| `ImportError: cudf` or `ImportError: cupy` | Ensure the Conda environment is activated: `conda activate KMX-env`. |
| Boost-related build errors | gerbil-DataFrame is tested with Boost 1.77. Other versions may cause issues. Set `-DBOOST_ROOT=/path/to/boost-1.77` in the CMake command if needed. |
| Insufficient disk space during installation | Free at least 12 GB. Run `conda clean --all -y` to clear cached packages. |
| Slow performance | Check GPU utilization with `nvidia-smi`. Ensure no other processes are using the GPU. |

---

## License

KMX is released under the [MIT License](LICENSE).

Copyright (c) 2025 Mohammadali (Ali) Serajian

## Contact

For questions or support, contact **ma.serajian@gmail.com** or open an issue on [GitHub](https://github.com/M-Serajian/KMX).
