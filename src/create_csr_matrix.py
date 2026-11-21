import cudf
import cupy as cp
import os
import subprocess
import time as _time
#import cupyx
from src.run_gerbil import set_of_all_unique_kmers_extractor, single_genome_kmer_extractor

# Env-gated stage profiler (KMX_PROFILE=1). Timing prints only — zero behavior change.
_PROF = os.environ.get("KMX_PROFILE", "") not in ("", "0")

# Per-genome gerbil runs on CPU by default. GPU gerbil allocates 0.8*free_GPU_mem (~143 GB on a
# B200) PER invocation (cuda_ds CountingHashTable), which is ruinous in the per-genome loop (one
# gerbil call per genome). GPU only pays off for the single reference pass over ALL genomes, where
# that 143 GB table is amortized. So `enable_gpu` (-g) applies to the reference extraction, while
# the per-genome counting stays on CPU. Set KMX_PERGENOME_GPU=1 to restore the old (slow) behavior.
_PERGENOME_GPU = os.environ.get("KMX_PERGENOME_GPU", "0") == "1"



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


def create_csr_matrix(genome_list, kmer_size, tmp_dir, min_val=1, max_val=10**9, disable_normalization=False, enable_gpu=False):

    """Processes genome data and generates a Compressed Sparse Row (CSR) matrix representation."""
    
    # Define the output directory for extracted k-mers
    set_of_all_unique_kmers_dir = os.path.join(
        os.path.abspath(tmp_dir),
        f"temporary_set_of_all_unique_kmers_min{min_val}_max{max_val}_kmer{kmer_size}_"
        f"{'normalization_disabled' if disable_normalization else 'normalization_enabled'}.csv"
    )
    
    # Extract unique k-mers for the provided genome list
    _t_p1 = _time.time()
    set_of_all_unique_kmers_extractor(genome_list, set_of_all_unique_kmers_dir, kmer_size, min_val, max_val, tmp_dir, disable_normalization,enable_gpu)
    _prof_pass1 = _time.time() - _t_p1
    _prof_count = 0.0   # pass-2 per-genome gerbil counting
    _prof_merge = 0.0   # pass-2 cuDF read + merge + CSR assembly

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

        _t_c = _time.time()
        single_genome_kmer_extractor(kmer_size, tmp_dir, tmp_genome_output_dir, genome_dir, disable_normalization, enable_gpu and _PERGENOME_GPU)
        _prof_count += _time.time() - _t_c
        _t_m = _time.time()
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
        _prof_merge += _time.time() - _t_m

        if genome_number % 1000 == 0 and genome_number != 0:
            print(f"Processed {genome_number} genomes", flush=True)
            # density=round (100*current_position/(len(set_of_all_unique_kmers_dataframe)*genome_number+1),2)
            # print(f"Current density is : {density}%",flush=True)
            # print(f"Number of non-zero elements so far {current_position}",flush=True)
    
    last_value = row[-1]

    if _PROF:
        print(f"[KMX1 PROFILE] pass1_count_all={_prof_pass1:.1f}s  "
              f"pass2_count={_prof_count:.1f}s  pass2_merge={_prof_merge:.1f}s  "
              f"(genomes={len(genome_dirs)}, cols={len(set_of_all_unique_kmers_dataframe):,})",
              flush=True)

    #Index for each K-mer
    df_comp = set_of_all_unique_kmers_dataframe.reset_index()

    total_cells = len(set_of_all_unique_kmers_dataframe) * len(genome_dirs)
    sparsity = (1 - last_value / total_cells) * 100 if total_cells > 0 else 0.0
    #csr_matrix = cupyx.scipy.sparse.csr_matrix((data[:last_value],column[:last_value],cp.asarray(row)))

    return data[:last_value] , column[:last_value] , cp.asarray(row) , df_comp , sparsity





