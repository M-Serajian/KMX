import cudf
import cupy as cp
import os
import subprocess
from src.run_gerbil import set_of_all_unique_kmers_extractor, single_genome_kmer_extractor

def expand_array(array, new_size):
    """Expand the given CuPy array to a new size while preserving existing data."""
    expanded_array = cp.empty(new_size, dtype=array.dtype)
    expanded_array[: len(array)] = array  # Copy existing data
    return expanded_array

def create_csr_matrix(genome_list, kmer_size, min_val=1, max_val=10**9, tmp_dir="./tmp"):
    """Processes genome data and generates a Compressed Sparse Row (CSR) matrix representation."""
    
    # Define the output directory for extracted k-mers
    set_of_all_unique_kmers_dir = os.path.join(
        os.path.abspath(tmp_dir),
        f"set_of_all_unique_kmers_min{min_val}_max{max_val}_kmer{kmer_size}.csv"
    )
    
    # Extract unique k-mers for the provided genome list
    set_of_all_unique_kmers_extractor(genome_list, set_of_all_unique_kmers_dir, kmer_size, min_val, max_val, tmp_dir)

    # Read the genome list file and extract directory paths
    with open(genome_list, "r", encoding="utf-8") as file:
        genome_dirs = [line.strip() for line in file if line.strip()]

    # Load extracted k-mers into a cuDF DataFrame
    set_of_all_unique_kmers_dataframe = cudf.read_csv(set_of_all_unique_kmers_dir)
    set_of_all_unique_kmers_dataframe = set_of_all_unique_kmers_dataframe[["K-mer"]]

    # Estimate memory requirements and initialize CSR matrix components
    density = 0.001  # Initial estimation of k-mer matrix density
    row = [0]
    estimated_size = int(density * len(set_of_all_unique_kmers_dataframe) * len(genome_dirs))

    column = cp.empty(estimated_size, dtype=cp.uint32)
    data = cp.zeros_like(column, dtype=cp.float32)
    current_position = 0

    # Iterate over each genome directory and extract k-mer frequency data
    for genome_num, genome_dir in enumerate(genome_dirs):
        tmp_dataframe_dir = single_genome_kmer_extractor(kmer_size, tmp_dir, genome_dir, genome_num)

        df_csv = cudf.read_csv(tmp_dataframe_dir)
        
        df_comp = set_of_all_unique_kmers_dataframe.reset_index()
        df_comp = df_comp.merge(df_csv, on="K-mer", how="right")
        df_comp.dropna(inplace=True)
        
        idx = df_comp["index"].values.astype(cp.uint32)
        val = df_comp["Frequency"].values

        size = idx.size
        new_size = current_position + size

        # Expand CSR matrix arrays dynamically if required
        if new_size > column.size:
            new_capacity = max(new_size, column.size * 2)
            column = expand_array(column, new_capacity)
            data = expand_array(data, new_capacity)

        column[current_position : current_position + size] = idx
        data[current_position : current_position + size] = val
        current_position += size
        row.append(current_position)

        if genome_num % 1000 == 0:
            print(f"Processed {genome_num} genomes", flush=True)
    
    last_value = row[-1]
    
    df_comp = set_of_all_unique_kmers_dataframe.reset_index()

    return data,column,row,df_comp