
# KMX

## Introduction

KMX is a GPU-accelerated software designed for efficient k-mer extraction from large genomic datasets and construction of the corresponding matrix. Given a list of genomes in FASTA format, it:

- Extracts k-mers from each genome.
- Constructs a CSR (Compressed Sparse Row) matrix, where rows represent genomes and columns represent extracted k-mers.
- Optimizes time and memory usage, enabling scalability for large datasets.
- Integrates seamlessly into machine learning (ML) workflows, making it suitable for genomic feature extraction, clustering, classification tasks, and bioinformatics applications in high-performance computing (HPC) environments.

## Citation

If you use **KMX** in your research, please cite the paper below:

> **Serajian M.**, *et al.* **“KMX: GPU-Accelerated K-mer K-mer Matrix Constructor.”** *Journal Name*, **Volume** (Year): pages. DOI: xx.xxxx/xxxxxxxx


## Dependencies

### Required Software (Modules Recommended for HPC)

- `gcc` (module: `gcc/12.2`, recommended)
- `cmake` (module: `cmake`, latest)
- `boost` (module: `boost/1.77`, recommended)
- `rapidsai` (module: `rapidsai/24.08`, recommended)
- `rapidsai` (module: `rapidsai/24.08`, recommended)
- `python` (module: `python/3.8`, recommended)
- `git`
- `libboost-all-dev`
- `libz3-dev`
- `libbz2-dev`

> **For HPC environments**, load dependencies using environment modules:

```bash
ml gcc/12.2
ml cmake
ml python/3.8
ml boost/1.77
ml rapidsai/24.08
```

> **For Ubuntu/Debian systems**, use `apt` to install system-wide packages:

```bash
sudo apt-get update && sudo apt-get install -y \
    g++-12 \
    cmake \
    git \
    libboost-all-dev \
    libz3-dev \
    libbz2-dev
```

> ⚠️ Note: RAPIDS AI must be installed separately via [Conda](https://rapids.ai/start.html) or Docker.

## Installation Guide

### Manual Installation 

```bash
git clone https://github.com/M-Serajian/KMX.git
cd KMX/include
git clone https://github.com/M-Serajian/gerbil-DataFrame.git
cd gerbil-DataFrame
mkdir build 
cd build
cmake ..
make -j
cd ../../..
```


## Usage
`KMX.py` can be located at the KMX directory (the root on the cloned directory).


```bash
python KMX.py -l PATH/genomes.txt -t PATH/tmp -k KMER_SIZE -o OUTPUT_DIR \
    [ -min X ] [ -max Y ] [ -d ] [ -c ]
```

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `-l`, `--genome-list` | *str* | **yes** | – | Path to a plain‑text file with one genome file (FASTA/FASTQ, optionally gzipped) per line. |
| `-min` | *int* | no | `1` | Lower bound on k‑mer count across **all** genomes; k‑mers observed fewer than *min* times are discarded. |
| `-max` | *int* | no | *unlimited* | Upper bound on k‑mer count; set to remove highly conserved k‑mers that offer no discriminatory information. Must satisfy *max ≥ min*. |
| `-t`, `--tmp` | *path* | **yes** | – | Directory for temporary files; created if absent. Script aborts if free space < 10 GB. |
| `-k`, `--kmer-size` | *int* | **yes** | – | Length of k‑mers to extract; valid range **8 – 136**. |
| `-d`, `--disable-normalization` | flag | no | *normalisation enabled* | Treat reverse complements as distinct features. |
| `-c`, `--cpu` | flag | no | *GPU mode* | Force CPU execution; useful on hosts without CUDA‑capable devices. |
| `-o`, `--output` | *path* | **yes** | – | Destination directory for final CSR matrix (`row.npy`,`column.npy`,`data.npy`), k‑mer index (`kmers.csv`), and run log file. |

## Output Files

| File | Format | Contents |
|------|--------|----------|
| `row.npy` | NumPy NPY | a 1-d array representing row pointers of the CSR Matrix. |
| `column.npy` | NumPy NPY | a 1-d array representing column pointers of the CSR Matrix. |
| `data.npy` | NumPy NPY | a 1-d array representing data of the CSR Matrix. |
| `kmers.csv` | CSV | Tab‑separated list mapping column index → canonical k‑mer string. |
| `run.log` | plain text | Resource usage record of parameter settings, and warnings. |

After generating `row.npy`, `column.npy`, and `data.npy`, you can reconstruct the CSR matrix on the GPU with cupyx.scipy.sparse.csr_matrix((data, column, row)) or on the CPU with scipy.sparse.csr_matrix((data, (row, column))).

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions or support, contact at **ma.serajian@gmail.com**, or open an issue on GitHub.
