# GPU-CSR-KMER: GPU-Accelerated Compressed K-mer Matrix Generator

## Introduction

GPU-CSR-KMER is a GPU-accelerated software designed for efficient k-mer extraction from large genomic datasets. Given a list of genomes in FASTA or FASTQ format, it:

- Extracts k-mers from each genome.
- Constructs a CSR (Compressed Sparse Row) matrix, where rows represent genomes and columns represent extracted k-mers.
- Optimizes time and memory usage, enabling scalability for large datasets.
- Integrates seamlessly into machine learning (ML) workflows, making it suitable for genomic feature extraction, clustering, classification tasks, and bioinformatics applications in high-performance computing (HPC) environments.

## Dependencies

### Required Software

| Dependency | Purpose                                              | Notes                                              |

|------------|------------------------------------------------------|----------------------------------------------------|

| GCC        | Required for compilation                             | Available in most Linux distributions             |

| CMake      | Build system for generating Makefiles                | Version 3.10 or higher is recommended             |

| Boost      | Required for optimized performance                   | Must include system, thread, filesystem, and regex |

### HPC-Specific Requirements

If you are using Gerbil-DataFrame in an HPC environment, ensure that module loading is available (for example):

```

ml gcc

ml cmake

ml boost

```

## Installation Guide

### Automatic Installation

Gerbil-DataFrame provides an automated installation script that handles dependency verification, submodule initialization, and software compilation. This method is recommended for HPC environments or large-scale ML workflows.

1. Clone the repository:

   ```

   git clone https://github.com/M-Serajian/gerbil-DataFrame.git

   cd gerbil-DataFrame

   ```
2. Run the installation script:

   ```

   sh setup.sh

   ```

   This will:

   - Check for required dependencies (GCC, CMake, Boost).
   - Load dependencies via ml (for HPC environments).
   - Clone and initialize submodules.
   - Compile the software in the build/ directory.
3. Force reinstallation (if needed):

   ```

   sh setup.sh --force

   ```

   This removes any previous installation and reinitializes everything from scratch.

### Manual Installation

For systems where the automated script is not preferred, use the following steps:

1. Install dependencies.

   For Ubuntu/Debian:

   ```

   sudo apt update

   sudo apt install -y build-essential cmake libboost-all-dev

   ```

   For CentOS/RHEL:

   ```

   sudo yum groupinstall -y "Development Tools"

   sudo yum install -y cmake boost-devel

   ```
2. Clone the repository:

   ```

   git clone https://github.com/M-Serajian/gerbil-DataFrame.git

   cd gerbil-DataFrame

   ```
3. Initialize submodules:

   ```

   git submodule init

   git submodule update --recursive

   ```
4. Build the software:

   ```

   mkdir -p include/gerbil-DataFrame/build

   cd include/gerbil-DataFrame/build

   cmake ..

   make -j$(nproc)

   ```
5. Verify the installation:

   ```

   ls -l include/gerbil-DataFrame/build/

   ```

## Usage

1. Navigate to the build directory:

   ```

   cd include/gerbil-DataFrame/build

   ```
2. Run the program:

   ```

   ./gerbil [OPTIONS]

   ```

### Example

```

./gerbil --input genome.fasta --kmer-size 31 --output kmers.csr

```

### Command-Line Options

| Option       | Description                                          |

|------------- |------------------------------------------------------|

| --input      | Path to the input genome file (FASTA/FASTQ)          |

| --kmer-size  | Length of k-mers to extract                          |

| --output     | Path to save the generated CSR matrix               |

For more options, run:

```

./gerbil --help

```

## Troubleshooting

| Issue                                                   | Solution                                                                                |

|---------------------------------------------------------|-----------------------------------------------------------------------------------------|

| CMake Error: Could NOT find Boost                       | Ensure Boost is installed and correctly loaded (via ml boost for HPC or apt/yum for Linux). |

| fatal: 'include/gerbil-DataFrame' already exists in the index | Run `sh setup.sh --force` to fully remove and reinstall the software.                     |

| make: command not found                                 | Install build-essential (Ubuntu/Debian) or Development Tools (CentOS/RHEL).              |

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions or support, contact [Your Name] at [Your Email], or open an issue on GitHub.
