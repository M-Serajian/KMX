"""Multi-node (Regime 1) tile orchestration + assembly.

Each node builds the matrix for ITS genomes against the SHARED reference, then writes
one row-block per SHARED column band (same band boundaries on every node) tagged with
a GLOBAL row offset. assemble() aggregates the per-node tiles into one
chunked_output manifest, so the rows tile globally per band and the result reconstructs
with the core reader. This is the concat-only assembly: no cross-node data movement of
the matrix itself — each node writes locally, assembly is metadata only.
"""
import json
import os
import numpy as np
import KMX.chunked_output as ck


def plan_shared_bands(n_cols, *, n_bands=None, budget_nnz=None, col_nnz=None):
    """Column bands shared by ALL nodes (identical boundaries -> tiles concat).
    By global col_nnz if given (balanced), else equal-width into n_bands."""
    if n_cols == 0:
        return []
    if col_nnz is None:
        col_nnz = np.ones(n_cols, dtype=np.int64)
    if budget_nnz is None:
        budget_nnz = max(1, int(np.asarray(col_nnz).sum() // max(1, (n_bands or 1))))
    return ck.plan_bands(col_nnz, budget_nnz=budget_nnz)


def write_node(indices, data, indptr, bands, out_dir, *, node_id, row_offset,
               data_dtype=None, kmers=None):
    """Write this node's matrix as one row-block per shared band (global rows).
    Returns chunk records for assembly."""
    indices = np.asarray(indices); data = np.asarray(data); indptr = np.asarray(indptr)
    n_rows = int(indptr.shape[0] - 1)
    dtype = np.dtype(data_dtype or (data.dtype if data.size else np.int64))
    recs = []
    for b, (c0, c1) in enumerate(bands):
        li, ld, lp = ck.slice_band(indices, data, indptr, c0, c1)
        rel = f"band_{b:05d}/node_{node_id:03d}"
        d = os.path.join(out_dir, rel); os.makedirs(d, exist_ok=True)
        np.save(d + "/column.npy", np.ascontiguousarray(li, np.int32))
        np.save(d + "/data.npy", np.ascontiguousarray(ld.astype(dtype)))
        np.save(d + "/row.npy", np.ascontiguousarray(lp, np.int64))
        if kmers is not None:                              # band legend, written once
            os.makedirs(os.path.join(out_dir, f"band_{b:05d}"), exist_ok=True)
            with open(os.path.join(out_dir, f"band_{b:05d}", "kmers.csv"), "w") as fh:
                fh.write("index,K-mer\n"); fh.writelines(f"{j},{kmers[c0 + j]}\n" for j in range(c1 - c0))
        recs.append(dict(band_id=b, dir=rel, c0=int(c0), c1=int(c1), n_cols=int(c1 - c0),
                         g0=int(row_offset), g1=int(row_offset + n_rows), n_rows=n_rows, nnz=int(lp[-1])))
    return recs


def assemble(out_dir, *, n_rows, n_cols, bands, node_recs, data_dtype):
    """Aggregate per-node tile records -> one chunked_output manifest (rows tile per band)."""
    flat = sorted([r for rs in node_recs for r in rs], key=lambda r: (r["c0"], r["g0"]))
    chunks, band_recs = [], {}
    for cid, r in enumerate(flat):
        rec = dict(r, id=cid, block_id=len([c for c in chunks if c["band_id"] == r["band_id"]]))
        chunks.append(rec)
        bd = band_recs.setdefault(r["band_id"], dict(id=r["band_id"], dir=f"band_{r['band_id']:05d}",
                                  c0=r["c0"], c1=r["c1"], n_cols=r["n_cols"], n_row_blocks=0, chunk_ids=[]))
        bd["n_row_blocks"] += 1; bd["chunk_ids"].append(cid)
    dt = str(np.dtype(data_dtype))
    man = dict(format="kmx2-chunked-v2", mechanism="A", n_rows=int(n_rows), n_cols=int(n_cols),
               n_bands=len(bands), n_chunks=len(chunks), total_nnz=sum(c["nnz"] for c in chunks),
               blocking="2d", value_semantics="counts", index_dtype="int32", data_dtype=dt,
               dtypes=dict(index="int32", data=dt, indptr="int64"),
               bands=[band_recs[b] for b in sorted(band_recs)], chunks=chunks)
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as fh: json.dump(man, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))
    return man
