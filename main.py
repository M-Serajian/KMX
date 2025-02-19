from src.args import parse_arguments

def main():
    """Main function for processing genome data and generating a CSR matrix."""
    args = parse_arguments()
    
    print(f"Genome list file: {args.genome_list}")
    print(f"Minimum k-mer occurrence threshold: {args.min}")
    print(f"Maximum k-mer occurrence threshold: {args.max if args.max else 'No upper limit'}")
    print(f"Temporary directory: {args.tmp}")
    print(f"Output directory: {args.output}")

    # The main computation logic would follow here...

if __name__ == "__main__":
    main()
