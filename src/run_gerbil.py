import subprocess
import os

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def set_of_all_unique_kmers_extractor(args,set_of_all_unique_kmers_dir):
    """Run the gerbil-DataFrame command with the parsed arguments."""
    

    command = [
        "./include/gerbil-DataFrame/build/gerbil",
        "-k", str(args.kmer_size),
        "-o", "csv",
        "-l", str(args.min),
        "-z", str(args.max),
        "-g",
        "-d",
        args.genome_list,
        args.tmp,
        set_of_all_unique_kmers_dir
    ]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"{GREEN} The list of all unique k-mers is stored here: {set_of_all_unique_kmers_dir}{RESET}")
    except subprocess.CalledProcessError as e:
        print(f"{RED}Error: set_of_all_unique_kmers_extractor failed with return code {e.returncode}.{RESET}")
        print(f"{RED}Standard Output:\n{e.stdout}{RESET}")
        print(f"{RED}Standard Error:\n{e.stderr}{RESET}")
    except FileNotFoundError:
        print(f"{RED}Error: The executable './include/gerbil-DataFrame/build/gerbil' was not found.{RESET}")






def single_genome_kmer_extractor(args,genome_dir,genome_number):
    """Run the gerbil-DataFrame command with the parsed arguments."""
    

    
    #output directory
    tmp_dir = os.path.join(args.tmp, f"temporary_output_genome_{genome_number}.csv")


    command = [
        "./include/gerbil-DataFrame/build/gerbil",
        "-k", str(args.kmer_size),
        "-o", "csv",
        "-l", str(1),
        "-z", str(10**9),           #infinity
        "-g",
        "-d",
        genome_dir,
        args.tmp,
        tmp_dir
    ]

    result = subprocess.run(command, check=True, text=True, capture_output=True)

    return(tmp_dir)
