from src.args import parse_arguments
from src.run_gerbil import set_of_all_unique_kmers_extractor
from src.run_gerbil import single_genome_kmer_extractor
import os


import cudf
import cupy as cp


def create_csr_matrix():
    """Main function for processing genome data and generating a CSR matrix."""
    args = parse_arguments()

    set_of_all_unique_kmers_dir = os.path.join(
    os.path.abspath(args.tmp),
    f"set_of_all_unique_kmers_min{args.min}_max{args.max}_kmer{args.kmer_size}.csv")
    
    set_of_all_unique_kmers_extractor(args,set_of_all_unique_kmers_dir)

    # Read the genome list file and create a list of directories
    with open(args.genome_list, "r", encoding="utf-8") as file:
        genome_dirs = [line.strip() for line in file if line.strip()]




    set_of_all_unique_kmers_dataframe = cudf.read_csv(set_of_all_unique_kmers_dir)
    set_of_all_unique_kmers_dataframe = set_of_all_unique_kmers_dataframe[["K-mer"]]

    # Initial CSR matrix pointers
    density=0.001 # initial estimation of the density of the k-mer matrix representation
    row_pnt = [0]
    estimated_size = int(density * len(set_of_all_unique_kmers_dataframe) * len(genome_dirs))

    column_idx = cp.empty(estimated_size, dtype=cp.uint32)
    vals = cp.zeros_like(column_idx, dtype=cp.float32)
    current_position = 0

    def expand_array(arr, new_size):
        expanded = cp.empty(new_size, dtype=arr.dtype)
        expanded[: len(arr)] = arr  # Copy existing data
        return expanded




    for genome_num, genome_dir in enumerate(genome_dirs):

        tmp_dataframe_dir= single_genome_kmer_extractor(args,genome_dir,genome_num)
        df_csv = cudf.read_csv(tmp_dataframe_dir)
        
        df_comp = set_of_all_unique_kmers_dataframe.reset_index()


        df_comp = df_comp.merge(df_csv, on="K-mer", how="right")
        df_comp.dropna(inplace=True)


        idx = df_comp["index"].values.astype(cp.uint32)
        val = df_comp["Frequency"].values

        size = idx.size
        new_size = current_position + size

        # Expand arrays if needed
        if new_size > column_idx.size:
            new_capacity = max(new_size, column_idx.size * 2)  # Double the size
            column_idx = expand_array(column_idx, new_capacity)
            vals = expand_array(vals, new_capacity)

        column_idx[current_position : current_position + size] = idx
        vals[current_position : current_position + size] = val
        current_position += size
        row_pnt.append(current_position)

        if genome_num % 1000 == 0:
            print(genome_num, flush=True)

    last_value = row_pnt[-1]







    df_comp = set_of_all_unique_kmers_dataframe.reset_index()
    print(df_comp)




if __name__ == "__main__":
    create_csr_matrix()

