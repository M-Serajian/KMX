# KMX distributed (multi-node)

Optional layer over the single-node core. **Reuses** the core per node; adds only the
one cross-node reduce, the partition planner, and tile orchestration. Heavy deps are an
extra: `pip install kmx[spark]`.

## Two mechanisms (the planner picks one)

The whole run is shaped by one question: **does the filtered reference fit ONE node's
VRAM?** Both mechanisms produce byte-identical output — the choice is performance.

- **Mechanism A — replicated / row-parallel** (`reference ≤ one node`): broadcast the
  reference, partition **genomes** across nodes (each genome joined once on its home
  node), shard the reference across that node's GPUs. Output = row-block tiles. The fast
  common case. `n_rounds = 1`.
- **Mechanism B — sharded / column-parallel + rounds** (`reference > one node`):
  column-shard the reference to **disk**; VRAM holds only the current round's shards
  (one per GPU). Stream genomes through them in `n_rounds = ceil(n_shards / total_gpus)`
  rounds, flushing column-band tiles each round. **Unbounded in distinct k-mers**
  (disk-limited, not VRAM-limited). The only hard rule: *one shard fits one GPU*.

These map to textbook patterns: A = broadcast (map-side) join + data parallelism;
B = grace hash join with partition spilling + sharded/FSDP parallelism + out-of-core
parameter streaming (ZeRO-Infinity-style). The A/B boundary = a cost-based optimizer's
broadcast-threshold.

## Modules (one concern each)
1. **planner.py** — `plan(n_kmers, n_nodes, n_gpus_per_node, node_vram_gb)` → `A` (with
   `n_genome_groups`, `shards_per_node`) or `B` (with `n_shards`, `n_rounds`). Rounds the
   shard count up to the full GPU count (use every GPU). One place to tune the boundary.
2. **reference.py** — Stage 1: `count_genome` (once) → `node_partial` (local sum) →
   `build_reference` (cross-node reduce + filter + column ids).
3. **reduce.py** — the only cross-node op: `groupBy(kmer).sum`. `reduce_pandas`
   (local/debug) and `reduce_spark` (cluster). Proven identical.
4. **matrix.py** — `build_matrix_csr` (count-once JOIN, never a recount) + `write_tiles`.
5. **multinode.py** — Mechanism A multi-node: shared column bands + per-node row-blocks +
   `assemble` (concat-only manifest aggregation; no cross-node matrix movement).
6. **mechanism_b.py** — Mechanism B out-of-core rounds engine: `shard_reference` (→ disk)
   + `build_b` (round iteration, streams genomes from disk, pluggable `join_fn` =
   CPU pandas now / GPU cuDF later). Memory-bounded: `peak_resident_shards ==
   shards_per_round`, regardless of `n_shards`.
7. **`__init__.build()`** — unified driver: count-once → reference → `plan()` → route A or B.

(`regime2.py` is a thin back-compat shim over `mechanism_b`.)

## Invariants
- **Count once.** Each genome counted by kmcpy exactly once; the table feeds the
  reference AND the matrix (a join). Mechanism B's rounds *re-read* cached tables but
  **never re-count**.
- **One shard fits one GPU** — the only hard requirement. Everything past one node is
  *more rounds*, never a failure.
- **Only the reduce crosses nodes.**

## Debugging
- `KMX.distributed.selfcheck([genomes...])` → whole pipeline on CPU, asserts ==
  single-node. Run first.
- Each stage is a plain function — call standalone with `reduce_pandas`; swap to
  `reduce_spark` only for the cluster. For B, `mechanism_b.build_b(..., join_fn=...)`.
- Harnesses (`tmp/dist_test/`): `val_mechb.py` (B rounds == single-node + memory bound,
  all shard/round combos), `val_build.py` (build() routes A and B), `val_real_mechanisms.py`
  (real cached MTB, both mechanisms), `validate_stage1.py` (reference; CPU + real 2 nodes).

## Status
- **Correctness complete (CPU, incl. real MTB fastq):** planner A/B, Mechanism A
  (replicate + row-parallel, multi-node), Mechanism B (out-of-core rounds, unbounded
  k-mers), and the unified `build()` — all validated == single-node.
- **TODO (execution/scale only, correctness already proven):**
  - Plug the **GPU join backend** into `mechanism_b.build_b(join_fn=...)` and reuse the
    single-node pinned-GPU + RAM-budgeted counting engine per node (instead of the CPU
    pandas scaffold). Needs a GPU run to validate.
  - **GPU-accelerated Spark reduce** (needs the `rapids-4-spark` jar on the cluster).
  - 2-node hardware run of the GPU matrix merge.
  - Disk-shard `build_reference` itself for references that exceed aggregate **RAM**
    (Spark gives this natively; the pandas path materialises the reference in memory).
