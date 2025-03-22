import os
import sys
import cudf
import cupy as cp
import time
import datetime
# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from args import parse_arguments
from src.create_csr_matrix import create_csr_matrix

# Cleaning memory
import gc

device = cp.cuda.Device(0)

# Get the total memory of the GPU
import rmm

# Get the assigned GPU (Slurm sets CUDA_VISIBLE_DEVICES)
gpu_id = cp.cuda.Device(0).id  # Get GPU ID
gpu_props = cp.cuda.runtime.getDeviceProperties(gpu_id)  # Get GPU properties
# Extract details
gpu_name = gpu_props["name"].decode("utf-8")  # GPU name (decode from bytes)
total_mem = gpu_props["totalGlobalMem"] / (1024**3)  # Total VRAM in GB
free_mem = cp.cuda.Device(0).mem_info[0] / (1024**3)  # Free VRAM in GB
print(gpu_name)
print(f"Total memorry: {total_mem}")
print(f"Free memorry: {free_mem}")


total_gpu_memory = device.mem_info[1]

pool_size = int(total_gpu_memory * 0.8) // 256 * 256

# Reinitialize RMM with the calculated pool size
rmm.reinitialize(
    pool_allocator=True,  # Enable the memory pool
    initial_pool_size=pool_size  # Set the pool size
)

cp.get_default_memory_pool().free_all_blocks()
gc.collect()

if __name__ == "__main__":
    args = parse_arguments()

    # Start the timer before executing the function

    t0=time.time()
    data , column, row, set_of_all_unique_kmers , sparsity = create_csr_matrix(
        genome_list=os.path.abspath(args.genome_list),
        kmer_size=args.kmer_size,
        tmp_dir=args.tmp,
        min_val=args.min,
        max_val=args.max,
        disable_normalization=args.disable_normalization 
    )
    t1=time.time()
    elapsed_time = t1 - t0
    formatted_time = str(datetime.timedelta(seconds=elapsed_time))

    output_text = (
        f"The sparsity of the genome k-mer matrix is: {sparsity}%\n"
        f"K-mer size: {args.kmer_size}\n"
        f"Temporary directory: {args.tmp}\n"
        f"Min value: {args.min}\n"
        f"Max value: {args.max}\n"
        f"-d flag activated: {'Yes! Normalization disabled.' if args.disable_normalization else 'No! Normalization enabled.'}\n"
        f"Processing time is: {formatted_time} .\n"

    )



    # Ensure the output directory exists
    output_dir = os.path.abspath(args.output)  # Assuming -o is stored in args.output
    os.makedirs(output_dir, exist_ok=True)

    # Determine if -d flag is enabled (1 for enabled, 0 for disabled)
    d_status = 1 if args.disable_normalization else 0



    output_stats_filename = f"feature_matrix_stats_k{args.kmer_size}_min{args.min}_max{args.max}_d{d_status}.txt"
    output_file_DIR= os.path.join(output_dir, output_stats_filename)

    # Write output to the specified file
    with open(output_file_DIR, "w") as f:
        f.write(output_text)



    file_suffix = f"k{args.kmer_size}_min{args.min}_max{args.max}_d{d_status}"

    # Save CuPy arrays as NumPy files
    cp.save(os.path.join(args.output, f"data_{file_suffix}.npy"), data)
    cp.save(os.path.join(args.output, f"row_{file_suffix}.npy"), row)
    cp.save(os.path.join(args.output, f"column_{file_suffix}.npy"), column)

    # Save the DataFrame as CSV
    set_of_all_unique_kmers.to_csv(os.path.join(args.output, f"set_of_all_unique_kmers_{file_suffix}.csv"), index=False)

    print(f"Files saved successfully in {args.output}",flush=True)