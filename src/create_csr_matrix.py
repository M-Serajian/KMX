import cudf
import cupy as cp
import os
import subprocess
#import cupyx
from src.run_gerbil import set_of_all_unique_kmers_extractor, single_genome_kmer_extractor



def expand_array(array, new_size):
    """Expand the given CuPy array to a new size while preserving existing data."""
    expanded_array = cp.empty(new_size, dtype=array.dtype)
    expanded_array[: len(array)] = array  # Copy existing data
    return expanded_array



# def expand_array(array, new_size, dtype):
#     """Expand the given CuPy array to a new size with the specified data type while preserving existing data."""
#     expanded_array = cp.empty(new_size, dtype=dtype)  # Create an empty array with the specified dtype
#     expanded_array[: len(array)] = array  # Copy existing data
#     return expanded_array


def create_csr_matrix(genome_list, kmer_size, tmp_dir, min_val=1, max_val=10**9, disable_normalization=False):

    """Processes genome data and generates a Compressed Sparse Row (CSR) matrix representation."""
    
    # Define the output directory for extracted k-mers
    set_of_all_unique_kmers_dir = os.path.join(
        os.path.abspath(tmp_dir),
        f"temporary_set_of_all_unique_kmers_min{min_val}_max{max_val}_kmer{kmer_size}_"
        f"{'normalization_disabled' if disable_normalization else 'normalization_enabled'}.csv"
    )
    
    # Extract unique k-mers for the provided genome list
    set_of_all_unique_kmers_extractor(genome_list, set_of_all_unique_kmers_dir, kmer_size, min_val, max_val, tmp_dir, disable_normalization)

    # Load extracted k-mers into a cuDF DataFrame
    set_of_all_unique_kmers_dataframe = cudf.read_csv(set_of_all_unique_kmers_dir)
    set_of_all_unique_kmers_dataframe = set_of_all_unique_kmers_dataframe[["K-mer"]]
    #Removing temporary file
    os.remove(set_of_all_unique_kmers_dir)

    # Read the genome list file and extract directory paths
    
    with open(genome_list, "r", encoding="utf-8") as file:
        genome_dirs = [line.strip() for line in file if line.strip()]

    # Estimate memory requirements and initialize CSR matrix components
    density = 0.001  # Initial estimation of k-mer matrix density
    row = [0]
    estimated_size = int(density * len(set_of_all_unique_kmers_dataframe) * len(genome_dirs))
    print(f"estimated size is: {estimated_size}")

    column = cp.empty(estimated_size, dtype=cp.uint32)
    data = cp.zeros_like(column, dtype=cp.float32)
    current_position = 0



    for genome_number, genome_dir in enumerate(genome_dirs):
        # Define the temporary DataFrame directory
        tmp_genome_output_dir = os.path.join(
            tmp_dir,
            f"temporary_output_genome_{genome_number}_"
            f"{'normalization_disabled' if disable_normalization else 'normalization_enabled'}.csv"
        )

        single_genome_kmer_extractor(kmer_size, tmp_dir, tmp_genome_output_dir, genome_dir, genome_number, disable_normalization)
        df_csv = cudf.read_csv(tmp_genome_output_dir)
        #removing the temporary file
        os.remove(tmp_genome_output_dir)

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

        if genome_number % 1000 == 0 and genome_number != 0:
            print(f"Processed {genome_number} genomes", flush=True)
            # density=round (100*current_position/(len(set_of_all_unique_kmers_dataframe)*genome_number+1),2)
            # print(f"Current density is : {density}%",flush=True)
            # print(f"Number of non-zero elements so far {current_position}",flush=True)
    
    last_value = row[-1]
    
    #Index for each K-mer
    df_comp = set_of_all_unique_kmers_dataframe.reset_index()

    sparsity=100-density
    #csr_matrix = cupyx.scipy.sparse.csr_matrix((data[:last_value],column[:last_value],cp.asarray(row)))

    return data[:last_value] , column[:last_value] , cp.asarray(row) , df_comp , sparsity





