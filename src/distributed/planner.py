"""Planner — pick the partition strategy and compute the shard / round layout.

The single decision that shapes the whole run, in ONE place so it is easy to find/tune.

Two mechanisms (see README):
  A — replicated:  the filtered reference fits ONE node's VRAM. Broadcast it,
                   partition GENOMES across nodes (row-parallel, each genome joined
                   once), and shard the reference across that node's GPUs for
                   utilisation. Always 1 round.
  B — sharded:     the reference exceeds one node. Column-shard it across ALL GPUs
                   and stream genomes through the shards in ROUNDS. Unbounded in
                   distinct k-mers (disk-limited, not VRAM-limited):
                       n_rounds = ceil(n_shards / total_gpus).

The ONLY hard rule is: one shard fits one GPU (always satisfiable by finer sharding).
Everything past one node becomes "more rounds", never a failure. The A/B boundary is a
PERFORMANCE choice — both mechanisms produce byte-identical output.
"""
import math

PACKED_BYTES_PER_KMER = 8        # k<=32 -> one uint64 key word (n_words for larger k)
INDEX_BYTES = 8                  # global int64 column id (worst case)


def reference_bytes(n_kmers, n_words=1, index_bytes=INDEX_BYTES):
    """VRAM bytes to hold the reference (packed keys + column ids)."""
    return int(n_kmers) * (PACKED_BYTES_PER_KMER * int(n_words) + int(index_bytes))


def plan(n_kmers, *, n_nodes, n_gpus_per_node, node_vram_gb, n_words=1,
         merge_headroom=0.25, index_bytes=INDEX_BYTES):
    """Choose Mechanism A or B and the shard/round layout for `n_kmers` reference k-mers.

    Returns a dict:
      common: mechanism ('A'|'B'), total_gpus, reference_bytes, per_gpu_bytes,
              node_usable_bytes, fits_one_node (bool)
      A:      n_genome_groups (= n_nodes), shards_needed, shards_per_node, n_rounds(=1)
      B:      n_shards (multiple of total_gpus), shards_per_round(=total_gpus), n_rounds
    """
    n_nodes = max(1, int(n_nodes)); n_gpus_per_node = max(1, int(n_gpus_per_node))
    total_gpus = n_nodes * n_gpus_per_node
    usable = 1.0 - float(merge_headroom)
    per_gpu = (float(node_vram_gb) / n_gpus_per_node) * usable * (1024 ** 3)
    node_usable = float(node_vram_gb) * usable * (1024 ** 3)
    ref = reference_bytes(n_kmers, n_words, index_bytes)
    shards_needed = max(1, math.ceil(ref / per_gpu)) if per_gpu > 0 else 1

    common = dict(total_gpus=total_gpus, reference_bytes=ref,
                  per_gpu_bytes=int(per_gpu), node_usable_bytes=int(node_usable),
                  fits_one_node=bool(ref <= node_usable))

    if ref <= node_usable:
        # Mechanism A — replicate per node, partition genomes across nodes, and shard
        # the reference across ALL of a node's GPUs ("5 shards on 8 GPUs -> use 8").
        # shards_needed <= n_gpus_per_node here (ref <= node_usable). Round UP to
        # n_gpus_per_node for utilisation, but never more shards than k-mers.
        shards_per_node = max(1, min(n_gpus_per_node, int(n_kmers) if n_kmers else 1))
        return dict(mechanism="A", n_genome_groups=n_nodes,
                    shards_needed=shards_needed, shards_per_node=shards_per_node,
                    n_rounds=1, **common)

    # Mechanism B — column-shard across ALL gpus; iterate rounds. Round the shard count
    # UP to a whole multiple of total_gpus so every round is full (balanced).
    n_shards = math.ceil(shards_needed / total_gpus) * total_gpus
    n_rounds = n_shards // total_gpus
    return dict(mechanism="B", n_shards=int(n_shards), shards_per_round=total_gpus,
                n_rounds=int(n_rounds), **common)


def layout(n_kmers, n_genomes, *, n_nodes, n_gpus_per_node, node_vram_gb, n_words=1,
           merge_headroom=0.25, index_bytes=INDEX_BYTES):
    """Smart work distribution — turn plan() into an EXPLICIT device->work mesh so no
    GPU sits idle and the on-node vs multi-node split is optimal for the shard count.

    Mechanism A (reference fits one node) -> DATA-PARALLEL replica groups:
        pack as many reference replicas as fit per node
        (groups_per_node = n_gpus_per_node // shards_to_fit_ref), each group = one replica
        sharded across its gpus_per_group GPUs; split the GENOMES across ALL groups
        cluster-wide. Replicas stay intra-node (no cross-node sharding). Collapses to one
        whole-node group (model-parallel) only when a replica needs > half the node.
        e.g. 8 GPUs, ref fits 3 -> 2 groups of 4, genomes split 2 ways per node.
    Mechanism B (reference exceeds one node) -> MODEL-PARALLEL column shards across ALL
        GPUs, iterated in rounds (a replica can't fit). Genomes streamed to every shard;
        shard k -> round k//total_gpus, gpu k%total_gpus.

    Returns the plan() dict plus an `assignment` list (who does what) and a `parallelism`
    label. Genome slices are [lo,hi); gpu ids are GLOBAL (node*n_gpus_per_node + local)."""
    p = plan(n_kmers, n_nodes=n_nodes, n_gpus_per_node=n_gpus_per_node,
             node_vram_gb=node_vram_gb, n_words=n_words, merge_headroom=merge_headroom,
             index_bytes=index_bytes)
    n_nodes = max(1, int(n_nodes)); g = max(1, int(n_gpus_per_node))
    shards_to_fit = math.ceil(p["reference_bytes"] / p["per_gpu_bytes"]) if p["per_gpu_bytes"] else 1
    shards_to_fit = max(1, shards_to_fit)

    if p["mechanism"] == "A":
        groups_per_node = max(1, g // shards_to_fit)            # replicas that fit a node
        gpus_per_group = g // groups_per_node                   # GPUs each replica uses
        spare_per_node = g - groups_per_node * gpus_per_group   # idle GPUs (odd counts)
        replica_groups = groups_per_node * n_nodes
        q, r = divmod(int(n_genomes), replica_groups)
        assign, lo = [], 0
        for grp in range(replica_groups):
            node, local = divmod(grp, groups_per_node)
            base = node * g + local * gpus_per_group
            sz = q + (1 if grp < r else 0)
            assign.append(dict(group=grp, node=node, gpus=list(range(base, base + gpus_per_group)),
                               genomes=[lo, lo + sz]))
            lo += sz
        p.update(parallelism="data-parallel (replica groups)", shards_to_fit_ref=shards_to_fit,
                 groups_per_node=groups_per_node, gpus_per_group=gpus_per_group,
                 replica_groups=replica_groups, spare_gpus_per_node=spare_per_node,
                 assignment=assign)
        return p

    # Mechanism B
    total = p["total_gpus"]
    assign = [dict(shard=s, round=s // total, node=(s % total) // g, gpu=(s % total) % g)
              for s in range(p["n_shards"])]
    p.update(parallelism="model-parallel column shards + rounds",
             shards_to_fit_ref=shards_to_fit, assignment=assign)
    return p


def plan_regime(n_kmers, *, node_vram_gb, n_gpus_per_node, n_words=1,
                merge_headroom=0.25):
    """Back-compat shim over plan() for one node. ('regime1', shards_per_node) if the
    reference fits the node (Mechanism A), else ('regime2', n_shards) (Mechanism B)."""
    p = plan(n_kmers, n_nodes=1, n_gpus_per_node=n_gpus_per_node,
             node_vram_gb=node_vram_gb, n_words=n_words, merge_headroom=merge_headroom)
    if p["mechanism"] == "A":
        return ("regime1", p["shards_per_node"])
    return ("regime2", p["n_shards"])
