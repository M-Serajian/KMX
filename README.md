# KMX — GPU-Accelerated K-mer Matrix Constructor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-blue.svg)]()
[![CUDA: 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)]()
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![RAPIDS: 25.06](https://img.shields.io/badge/RAPIDS-25.06-orange.svg)]()

---

## Overview

KMX is a high-performance, GPU-accelerated tool for extracting k-mers from large genomic datasets and constructing the corresponding feature matrix in **Compressed Sparse Row (CSR)** format. Given a list of genomes in FASTA format, KMX:

- Extracts k-mers from each genome using [gerbil-DataFrame](https://github.com/M-Serajian/gerbil-DataFrame), a CUDA-enabled k-mer counter forked from [Gerbil](https://github.com/uni-halle/gerbil).
- Constructs a CSR matrix where rows represent genomes and columns represent unique k-mers, with frequency values as entries.
- Leverages GPU acceleration through [RAPIDS cuDF](https://github.com/rapidsai/cudf) and [CuPy](https://cupy.dev/) for fast, memory-efficient DataFrame operations and sparse matrix assembly.
- Scales to large datasets with dynamic memory management and parallel processing.

The resulting CSR matrix integrates directly into machine learning workflows via `cupyx.scipy.sparse.csr_matrix` (GPU) or `scipy.sparse.csr_matrix` (CPU), making KMX well-suited for genomic feature extraction, clustering, classification, and other bioinformatics applications in high-performance computing (HPC) environments.

---

## K-mer Size

KMX supports k-mer sizes from **8 to 136** (inclusive). This range is inherited from the underlying [Gerbil](https://github.com/uni-halle/gerbil) k-mer counter, which encodes k-mers in a fixed-width binary representation with a compiled maximum of 136.

---

## Platform Support

**KMX runs on Linux only.** [RAPIDS cuDF](https://rapids.ai/), which KMX depends on for GPU-accelerated DataFrame operations, is exclusively available on Linux. There is no Windows or macOS support.

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+ recommended) |
| **GPU** | NVIDIA GPU with CUDA support |
| **CUDA Driver** | Compatible with CUDA 12.8.1 (see note below) |
| **VRAM** | Depends on dataset size |
| **RAM** | Depends on dataset size |
| **Disk** | ≥ 10 GB free for temporary files; ~10 GB for installation |

> ⚠️ **CUDA Version Notice**
>
> KMX has been fully tested and confirmed working with **CUDA 12.8.1**.
>
> Higher CUDA versions (12.9+) have been tested but are **not fully supported** at this time. Specifically, the gerbil-DataFrame component encounters compilation issues with CUDA versions beyond 12.8.1. Until this is resolved, **CUDA 12.8.1 is the recommended and supported version**.
>
> If you are on an HPC cluster with a module system, ensure you load the correct version:
> ```bash
> module load cuda/12.8.1
> ```

---

## Citation

If you use KMX in your research, please cite:

> **Serajian M.**, *et al.* **"KMX: GPU-Accelerated K-mer Matrix Constructor."** *Journal Name*, **Volume** (Year): pages. DOI: xx.xxxx/xxxxxxxx

---

## Installation

### Prerequisites

| Prerequisite | Version | Notes |
|-------------|---------|-------|
| **Linux** | any | Required. RAPIDS cuDF is Linux-only. |
| **NVIDIA GPU + Driver** | CUDA 12.8.1 | See CUDA version notice above. |
| **Conda or Mamba** | any | Required for automated installation. Install from [Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). |
| **Git** | any | Required to clone the repository. |
| **GCC** | ≤ 13 (12.2.0 recommended) | gerbil-DataFrame does **not** compile with GCC 14+. GCC 12.2.0 is tested and confirmed working. |

---

### Option 1: Automated Installation with Conda (Recommended)

This method creates a fully isolated Conda environment (`KMX-env`) with all dependencies and compiles gerbil-DataFrame automatically.

```bash
# Clone the repository
git clone https://github.com/M-Serajian/KMX.git
cd KMX

# Run the setup script
python setup.py install
```

The setup script will automatically:

1. Detect your GPU and CUDA driver version (requires CUDA 12.8.1+)
2. Check available disk space (~10 GB required)
3. Create a Conda environment named `KMX-env` with all exact versions:
   - Python 3.11.14
   - RAPIDS cuDF 25.06
   - CUDA Toolkit 12.8
   - GCC / G++ 12.2.0
   - Boost 1.77.0
   - CMake 4.2.3
   - zlib 1.3.1, bzip2 1.0.8
4. Clone [gerbil-DataFrame](https://github.com/M-Serajian/gerbil-DataFrame) into `include/`
5. Compile gerbil with full CUDA GPU support

**Estimated time:** 15–20 minutes (CUDA Toolkit download is large).

After installation, verify everything is working:

```bash
python setup.py verify
```

#### Maintenance Commands

```bash
# Verify installation health
python setup.py verify

# Recompile gerbil only (if build had errors)
python setup.py install        # will prompt: skip or recompile

# Recreate the entire environment from scratch
python setup.py uninstall
python setup.py install

# Remove KMX-env and gerbil-DataFrame completely
python setup.py uninstall

# Free disk space from conda package cache
conda clean --all -y
```

> ℹ️ `python setup.py uninstall` removes **both** the `KMX-env` conda environment **and** the `include/gerbil-DataFrame` directory.

---

### Option 2: Manual Installation

Use this if you prefer to manage dependencies yourself or are working in an HPC environment with a module system.

#### Dependency Versions

The versions listed below are **recommended and confirmed working**. Other versions may work under certain conditions, but violating the hard constraints listed below will cause build or runtime failures.

| Dependency | Recommended Version | Hard Constraint | Notes |
|-----------|---------------------|-----------------|-------|
| Python | 3.11.14 | Must match RAPIDS build | Other 3.11.x patch versions will likely work; 3.12+ is untested with RAPIDS 25.06 |
| GCC / G++ | 12.2.0 | **Must be ≤ GCC 13** | **GCC 14+ is incompatible with gerbil-DataFrame and will cause compilation errors.** GCC 11.x and 13.x may work but are untested. |
| CMake | 4.2.3 | Must be ≥ 3.5 | CMake 4.x requires patching gerbil's `CMakeLists.txt` line 1 to `VERSION 3.5` — the setup script does this automatically. CMake 3.x (≥ 3.5) will also work. |
| Boost | 1.77.0 | 1.77.x strongly recommended | Other Boost versions may introduce ABI incompatibilities or build errors in gerbil. If using a different version, pass `-DBOOST_ROOT` explicitly to CMake. |
| CUDA Toolkit | 12.8.1 | **Must be ≤ 12.8.x** | **CUDA 12.9+ has known incompatibilities with gerbil-DataFrame.** CUDA 12.8.1 is the highest tested and supported version. Lower 12.x versions may work but are untested. |
| RAPIDS cuDF | 25.06 | Must match CUDA version | cuDF 25.06 is the latest version confirmed working with CUDA 12.8. cuDF 25.12+ requires CUDA 13 which breaks gerbil. |
| CuPy | installed with cuDF | — | Installed automatically as a cuDF dependency. Do not install separately. |
| zlib | 1.3.1 | Must come from the same source (conda or system) | **Do not mix conda and system zlib.** If CMake picks up `/usr/lib64/libz.so` instead of conda's, you will get `cmath` errors in gerbil. Pass `-DZLIB_ROOT` explicitly. |
| libbz2 | 1.0.8 | Must come from the same source (conda or system) | Same mixing warning as zlib above. Pass `-DBZIP2_ROOT` explicitly if needed. |
| Git | any | — | Only needed to clone gerbil-DataFrame. |

> ⚠️ **Library Isolation Warning**
>
> When installing dependencies manually, be careful not to mix libraries from different sources:
>
> - **Boost:** Use exactly Boost 1.77.0. Other versions may introduce build errors or ABI incompatibilities. Pass `-DBOOST_ROOT=/path/to/boost-1.77` to CMake if needed.
> - **GCC:** Use the **same GCC version** throughout. Mixing compiler versions (e.g. compiling with GCC 12 but linking against GCC 14 libraries) causes `GLIBCXX` symbol errors at runtime.
> - **CUDA:** RAPIDS cuDF ships its own CUDA runtime libraries. Conflicting versions on `LD_LIBRARY_PATH` can cause crashes. Keep CUDA 12.8.1 active throughout.
> - **zlib / bzip2:** Use conda or system libraries — **do not mix them**. If cmake picks up system `/usr/lib64/libz.so` instead of the conda one, you will get `cmath` compilation errors in gerbil.
>
> The automated Conda installation (Option 1) avoids all of these issues by isolating every dependency in a single environment.

#### HPC Environment (Module System)

```bash
module load gcc/12.2
module load cuda/12.8.1
module load cmake/4.2
module load boost/1.77
module load python/3.11
```

> ⚠️ RAPIDS cuDF must be installed separately via Conda — it is not available as an HPC module.

#### Ubuntu / Debian

```bash
sudo apt-get update && sudo apt-get install -y \
    gcc-12 g++-12 \
    cmake \
    git \
    libboost-all-dev \
    zlib1g-dev \
    libbz2-dev
```

> ⚠️ RAPIDS cuDF must be installed separately via [Conda](https://rapids.ai/start.html).

#### Install RAPIDS cuDF via Conda

```bash
conda create -n KMX-env -c rapidsai -c conda-forge -c nvidia \
    python=3.11.14 \
    cudf=25.06 \
    gcc_linux-64=12.2.0 \
    gxx_linux-64=12.2.0 \
    boost-cpp=1.77.0 \
    cmake=4.2.3 \
    cuda-version=12.8 \
    zlib=1.3.1 \
    bzip2=1.0.8 \
    git make -y

conda activate KMX-env
```

#### Build gerbil-DataFrame

```bash
# Clone KMX and gerbil-DataFrame
git clone https://github.com/M-Serajian/KMX.git
cd KMX
mkdir -p include && cd include
git clone https://github.com/M-Serajian/gerbil-DataFrame.git
cd gerbil-DataFrame

# Patch cmake minimum version (required for CMake 4.x)
sed -i '1s/.*/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt

# Build
mkdir -p build && cd build

cmake .. \
  -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux \
  -DZLIB_ROOT=$CONDA_PREFIX \
  -DZLIB_LIBRARY=$CONDA_PREFIX/lib/libz.so \
  -DZLIB_INCLUDE_DIR=$CONDA_PREFIX/include \
  -DBZIP2_ROOT=$CONDA_PREFIX \
  -DBZIP2_LIBRARIES=$CONDA_PREFIX/lib/libbz2.so \
  -DBZIP2_INCLUDE_DIR=$CONDA_PREFIX/include

make -j$(nproc)

cd ../../..
```

> ℹ️ The `-DCUDA_TOOLKIT_ROOT_DIR` flag is critical. Without it, CMake may find an incomplete CUDA installation (missing `cicc`) and the build will fail with `cicc: command not found`.

Verify the binary was built successfully:

```bash
ls -lh include/gerbil-DataFrame/build/gerbil
```

---

## Usage

Activate the environment before running KMX:

```bash
conda activate KMX-env
```

### Command

```bash
python KMX.py -l <genome_list> -k <kmer_size> -t <tmp_dir> -o <output_dir> [options]
```

### Arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `-l`, `--genome-list` | str | **yes** | — | Path to a text file listing one FASTA file path per line. Blank lines and lines starting with `#` are ignored. |
| `-k`, `--kmer-size` | int | **yes** | — | K-mer length. Must be between **8** and **136** (inclusive). |
| `-t`, `--tmp` | path | **yes** | — | Temporary directory for intermediate files. Created if absent. Must have ≥ 10 GB free. |
| `-o`, `--output` | path | **yes** | — | Output directory for results. Created if absent. |
| `--min` | int | no | `5` | Minimum k-mer occurrence threshold. K-mers seen fewer times are discarded. |
| `--max` | int | no | `N/2` | Maximum k-mer occurrence threshold. Defaults to half the number of genomes. Must satisfy `max ≥ min`. |
| `-d`, `--disable-normalization` | flag | no | off | Treat a k-mer and its reverse complement as distinct features. By default both map to the same canonical k-mer. |
| `-c`, `--cpu` | flag | no | off | Force CPU-only execution. By default GPU acceleration is used. |

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

data   = cp.load("data_k31_min5_max100_d0.npy")
row    = cp.load("row_k31_min5_max100_d0.npy")
column = cp.load("column_k31_min5_max100_d0.npy")
matrix = cupyx.scipy.sparse.csr_matrix((data, column, row))
```

```python
# CPU (SciPy)
import numpy as np
import scipy.sparse

data   = np.load("data_k31_min5_max100_d0.npy")
row    = np.load("row_k31_min5_max100_d0.npy")
column = np.load("column_k31_min5_max100_d0.npy")
matrix = scipy.sparse.csr_matrix((data, column, row))
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: .../gerbil` | The gerbil binary was not compiled. Run `python setup.py install` or follow the manual build instructions above. |
| `cicc: command not found` during build | CMake found an incomplete CUDA installation. Pass `-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux` to cmake explicitly. |
| `cmath` errors during gerbil build | CMake picked up system zlib/bzip2 instead of conda's. Pass the `-DZLIB_*` and `-DBZIP2_*` flags shown in the manual build instructions. |
| gerbil build fails with `cmake_minimum_required` error | CMake 4.x requires VERSION ≥ 3.5. Patch line 1 of gerbil's `CMakeLists.txt`: `sed -i '1s/.*/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt` |
| gerbil compiles but **no GPU** (CPU only) | CUDA was not found during cmake. Confirm `$CONDA_PREFIX/nvvm/bin/cicc` exists and pass `-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux`. |
| gerbil fails with CUDA > 12.8.1 | Known issue. CUDA 12.8.1 is the tested and supported version. Higher versions have known incompatibilities with gerbil. |
| `GLIBC_2.XX not found` | Library mismatch between conda and system. Ensure `KMX-env` is activated before running. |
| gerbil compilation fails with GCC 14+ | gerbil-DataFrame requires GCC ≤ 13. Use GCC 12.2.0: specify `-DCMAKE_C_COMPILER=$(which x86_64-conda-linux-gnu-gcc)` in cmake. |
| `GLIBCXX` version errors at runtime | gerbil was compiled with a different GCC than expected. Rebuild inside the activated `KMX-env` conda environment. |
| `ImportError: cudf` or `ImportError: cupy` | Run `conda activate KMX-env` before launching KMX. |
| Boost-related build errors | Use exactly Boost 1.77.0. Pass `-DBOOST_ROOT=$CONDA_PREFIX` to cmake if needed. |
| Insufficient disk space | Free at least 12 GB. Run `conda clean --all -y` to clear cached packages. |
| Slow performance | Check GPU utilization: `nvidia-smi`. Ensure no other process is saturating GPU memory. |

---

## License

KMX is released under the [MIT License](LICENSE).

Copyright © 2025 Mohammadali (Ali) Serajian

---

## Contact

For questions or support, contact **ma.serajian@gmail.com** or open an issue on [GitHub](https://github.com/M-Serajian/KMX).
