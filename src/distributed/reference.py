"""Stage 1 — distributed reference build.

INVARIANT: count each genome by kmcpy EXACTLY ONCE; the returned table is reused
by Stage 2 (it is *not* recounted). Per-k-mer counts are summed across genomes/
nodes, then filtered by min/max. Because counts are additive this is identical to
the single-node pooled KMC reference (validated: tmp/dist_test/validate_stage1.py).

Data flow:
  count_genome(file)  -> compact (kmer_0.., count) table        [count-once]
  node_partial(tabs)  -> per-node group-by-sum                  [local reduce]
  build_reference(partials, reduce_fn) -> reference DataFrame   [global reduce+filter]
     reference columns: kmer_0.. (packed key) + 'col' (0..n-1 column id)
"""
import tempfile
import numpy as np
import kmcpy
from .reduce import reduce_pandas


def key_cols(df):
    return [c for c in df.columns if c.startswith("kmer")]


def count_genome(genome_files, k, *, input_fmt="mfasta", canonical=True,
                 tmp_dir=None, counter_max=65535):
    """Count ONE genome's k-mers once, UNFILTERED (min=1, no max) -> compact table.
    This single table is reused for the reference AND the matrix (count-once)."""
    if isinstance(genome_files, str):
        genome_files = [genome_files]
    return kmcpy.count_kmers(genome_files, tmp_dir=tmp_dir or tempfile.mkdtemp(),
                             k=k, input_fmt=input_fmt, canonical=canonical,
                             min_count=1, max_count=10 ** 9, counter_max=counter_max,
                             decoded=False, drop_count=False)


def node_partial(genome_tables, key=None):
    """Sum a node's genome counts per k-mer -> compact node-level partial."""
    key = key or key_cols(genome_tables[0])
    return reduce_pandas(genome_tables, key)


def build_reference(partials, *, min_count, max_count, key=None, reduce_fn=reduce_pandas):
    """Global reduce (sum partials) + filter min/max + assign deterministic column
    ids (sorted by packed key). ``reduce_fn`` is the cross-node backend (pandas or
    spark wrapper). Returns the reference DataFrame (key cols + 'col')."""
    key = key or key_cols(partials[0])
    glob = reduce_fn(partials, key) if reduce_fn is not reduce_pandas else reduce_pandas(partials, key)
    keep = glob[(glob["count"] >= min_count) & (glob["count"] <= max_count)]
    ref = keep[key].sort_values(key).reset_index(drop=True)
    ref["col"] = np.arange(len(ref), dtype=np.int64)
    return ref


def decode_reference(reference, k):
    """Decode the packed reference keys -> k-mer strings (column id -> string)."""
    return kmcpy.decode_kmers(reference[key_cols(reference)], k=k)["kmer"].to_numpy()
