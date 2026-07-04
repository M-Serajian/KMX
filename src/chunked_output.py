#!/usr/bin/env python3
"""Chunked CSR output for very large genome × k-mer matrices (2-D blocking).

At human-genome-project scale the full matrix is tens of TB to PB — it cannot
be materialised, loaded, or held resident on any one device. We partition it
into device-sized tiles. The primary cut is by COLUMN band (a contiguous range
of k-mers) holding ALL rows; when a single band's content (or even the row
pointer alone) still exceeds the device, that band is additionally split into
ROW blocks. So a chunk is a ``(row-block × column-band)`` tile, and the planner
**guarantees every chunk fits the target device** (it narrows columns and splits
rows until it does — in the limit, one column × a few rows always fits).

Layout under ``<out_dir>/chunks_<suffix>/``::

    manifest.json                       # grid + provenance + dtypes
    band_00000/
      kmers.csv                         # this band's k-mer legend (shared by its blocks)
      block_00000/column.npy            # LOCAL int32 column index (global − band c0)
      block_00000/data.npy              # values (same dtype as the matrix)
      block_00000/row.npy               # int64 CSR indptr for this block's rows
      block_00001/ …
    band_00001/ …

Robustness guarantees:
  * every chunk fits ``device_gb`` (column-narrow + row-split recursion);
  * columns tile ``[0, n_cols)`` and, within each band, rows tile ``[0, n_rows)`` —
    no gaps/overlap, verified by :func:`verify_chunks`;
  * the manifest is written last via an atomic temp→rename, so a crashed run
    leaves no manifest (an incomplete output is never mistaken for a complete one);
  * Σ chunk nnz == total nnz; every row appears in every band.

Pure-numpy core (no scipy/cupy on the write/verify path); scipy/cupy are imported
lazily only by the matrix-returning reader.
"""
import datetime
import json
import os
import re

import numpy as np

_INT32_MAX = int(np.iinfo(np.int32).max)
MANIFEST = "manifest.json"
_FORMAT = "kmx2-chunked-v2"


# ── per-column nonzero counts ────────────────────────────────────────────────
def col_nnz_from_indices(indices, n_cols):
    if n_cols == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(np.asarray(indices), minlength=n_cols).astype(np.int64)


# ── column-band planning (1-D) ───────────────────────────────────────────────
def plan_bands(col_nnz, *, budget_nnz):
    """Greedy contiguous column bands, each ≤ ``budget_nnz`` total nonzeros."""
    n_cols = int(np.asarray(col_nnz).shape[0])
    if budget_nnz < 1:
        raise ValueError(f"budget_nnz must be >= 1, got {budget_nnz}")
    if n_cols == 0:
        return []
    bands = []
    start = 0
    run = 0
    for c in range(n_cols):
        w = int(col_nnz[c])
        if run and run + w > budget_nnz:
            bands.append((start, c))
            start = c
            run = 0
        run += w
    bands.append((start, n_cols))
    return bands


def _coalesce_runts(bands, col_nnz, budget_nnz, *, min_frac=0.1, slack=1.1):
    """Fold tiny remainder bands into a neighbour so we never emit a sliver."""
    if len(bands) <= 1:
        return [tuple(b) for b in bands]
    cum = np.concatenate(([0], np.cumsum(np.asarray(col_nnz, dtype=np.int64))))
    def nnz(c0, c1):
        return int(cum[c1] - cum[c0])
    thr = max(1, int(min_frac * budget_nnz))
    cap = int(budget_nnz * slack)
    out = [list(bands[0])]
    for c0, c1 in bands[1:]:
        if nnz(c0, c1) < thr and nnz(out[-1][0], out[-1][1]) + nnz(c0, c1) <= cap:
            out[-1][1] = c1
        else:
            out.append([c0, c1])
    if len(out) > 1 and nnz(out[0][0], out[0][1]) < thr \
            and nnz(out[0][0], out[1][1]) <= cap:
        out[1][0] = out[0][0]
        out.pop(0)
    return [tuple(b) for b in out]


# ── slicing a CSR into one column band (pure numpy) ──────────────────────────
def slice_band(indices, data, indptr, c0, c1):
    """Band ``[c0, c1)`` → ``(local_idx_i32, data, indptr_i64)`` over ALL rows."""
    indices = np.asarray(indices)
    indptr = np.asarray(indptr)
    mask = (indices >= c0) & (indices < c1)
    local = indices[mask] - c0
    if local.size and int(local.max()) > _INT32_MAX:
        raise ValueError("band wider than int32 — split further")
    cum = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    new_indptr = cum[indptr].astype(np.int64)
    return local.astype(np.int32, copy=False), np.asarray(data)[mask], new_indptr


# ── row-block planning within a band (1-D over rows) ─────────────────────────
def _row_blocks(per_row_nnz, *, per_nnz_bytes, budget_bytes):
    """Split ``[0, n_rows)`` into contiguous row blocks, each fitting
    ``budget_bytes`` as ``nnz·per_nnz_bytes + 8·(rows+1)``. A single row that
    alone exceeds the budget becomes its own (over-budget) block — the caller
    narrows the band's columns first so this stays rare."""
    n = int(np.asarray(per_row_nnz).shape[0])
    if n == 0:
        return [(0, 0)]
    blocks = []
    g0 = 0
    run_nnz = 0
    run_rows = 0
    for g in range(n):
        w = int(per_row_nnz[g])
        if run_rows > 0:
            size = (run_nnz + w) * per_nnz_bytes + 8 * (run_rows + 2)
            if size > budget_bytes:
                blocks.append((g0, g))
                g0 = g
                run_nnz = 0
                run_rows = 0
        run_nnz += w
        run_rows += 1
    blocks.append((g0, n))
    return blocks


def _col_split_point(c0, c1, col_nnz):
    """Split a band's columns near the nnz-median so a too-wide band shrinks."""
    cum = np.cumsum(np.asarray(col_nnz[c0:c1], dtype=np.int64))
    if cum.size <= 1 or cum[-1] == 0:
        return (c0 + c1) // 2
    half = cum[-1] / 2.0
    k = int(np.searchsorted(cum, half)) + 1
    mid = c0 + max(1, min(k, (c1 - c0) - 1))       # keep both sides non-empty
    return mid


# ── 2-D grid planner: guarantees every tile fits the device ──────────────────
def plan_grid(indices, data, indptr, n_cols, *, budget_bytes, item_bytes):
    """Plan the ``(row-block × column-band)`` grid. Returns a list of band dicts:
    ``{c0, c1, local_idx, local_data, band_indptr, row_blocks:[(g0,g1),…]}``.

    Column bands are sized by nnz; each band is then row-split to fit
    ``budget_bytes``; if a single row still overflows a multi-column band, the
    band's columns are split and re-planned (recursion terminates at 1 column).
    """
    indices = np.asarray(indices)
    data = np.asarray(data)
    indptr = np.asarray(indptr)
    per_nnz = 4 + int(item_bytes)                  # local int32 index + value
    col_nnz = col_nnz_from_indices(indices, n_cols)
    col_budget_nnz = max(1, int(budget_bytes / per_nnz))
    bands0 = _coalesce_runts(plan_bands(col_nnz, budget_nnz=col_budget_nnz),
                             col_nnz, col_budget_nnz)
    work = list(bands0)
    out = []
    guard = 0
    while work:
        guard += 1
        if guard > 4 * (n_cols + 8):
            raise RuntimeError("plan_grid did not converge — column split loop")
        c0, c1 = work.pop(0)
        li, ld, bip = slice_band(indices, data, indptr, c0, c1)
        per_row = np.diff(bip)
        if per_row.size and int(per_row.max()) * per_nnz + 16 > budget_bytes \
                and (c1 - c0) > 1:
            mid = _col_split_point(c0, c1, col_nnz)
            work = [(c0, mid), (mid, c1)] + work    # keep column order
            continue
        rblocks = _row_blocks(per_row, per_nnz_bytes=per_nnz, budget_bytes=budget_bytes)
        out.append({"c0": int(c0), "c1": int(c1), "local_idx": li,
                    "local_data": ld, "band_indptr": bip, "row_blocks": rblocks})
    out.sort(key=lambda b: b["c0"])                 # column-tiling order
    return out


def grid_for_device(indices, data, indptr, n_cols, *, device_gb, item_bytes,
                    headroom=0.8):
    budget_bytes = float(device_gb) * (1024 ** 3) * float(headroom)
    if budget_bytes < (4 + item_bytes) + 16:
        raise ValueError(f"device_gb={device_gb} is impossibly small")
    return plan_grid(indices, data, indptr, n_cols,
                     budget_bytes=budget_bytes, item_bytes=item_bytes), budget_bytes


# ── index-pointer fix-up (kept for the streaming flush-assemble path) ─────────
def concat_row_blocks(blocks):
    """Concatenate vertically-stacked CSR row-blocks → one CSR (indptr fix-up)."""
    if not blocks:
        return (np.empty(0, np.int32), np.empty(0), np.array([0], np.int64))
    idx_parts, val_parts = [], []
    indptr = [np.array([0], dtype=np.int64)]
    running = np.int64(0)
    for indices, data, ip in blocks:
        ip = np.asarray(ip, dtype=np.int64)
        indices = np.asarray(indices)
        if ip[0] != 0:
            raise ValueError("each block's indptr must start at 0")
        if int(ip[-1]) != int(indices.shape[0]):
            raise ValueError("block indptr[-1] != len(indices) — corrupt block")
        idx_parts.append(indices)
        val_parts.append(np.asarray(data))
        indptr.append(ip[1:] + running)
        running += np.int64(ip[-1])
    return (np.concatenate(idx_parts), np.concatenate(val_parts),
            np.concatenate(indptr))


def assemble_band(parts, *, n_rows, c0, c1):
    """Assemble one band's full-height CSR from streamed contiguous row-range
    parts ``(g0, g1, local_idx, data, indptr)`` that tile ``[0, n_rows)``."""
    parts = sorted(parts, key=lambda p: p[0])
    expect = 0
    blocks = []
    for g0, g1, indices, data, indptr in parts:
        if g0 != expect:
            raise ValueError(f"genome range gap/overlap: expected {expect}, got {g0}")
        if (g1 - g0) != (np.asarray(indptr).shape[0] - 1):
            raise ValueError("part rows != indptr length-1")
        blocks.append((indices, data, indptr))
        expect = g1
    if expect != n_rows:
        raise ValueError(f"parts cover {expect} rows, expected {n_rows}")
    indices, data, indptr = concat_row_blocks(blocks)
    if indices.size and int(indices.max()) >= (c1 - c0):
        raise ValueError("local column index >= band width — bad offset")
    return indices.astype(np.int32, copy=False), data, indptr


# ── writing the grid (atomic manifest commit) ────────────────────────────────
def _save_arrays(chunk_dir, indices, data, indptr):
    os.makedirs(chunk_dir, exist_ok=True)
    np.save(os.path.join(chunk_dir, "column.npy"),
            np.ascontiguousarray(indices, dtype=np.int32))
    np.save(os.path.join(chunk_dir, "data.npy"), np.ascontiguousarray(data))
    np.save(os.path.join(chunk_dir, "row.npy"),
            np.ascontiguousarray(indptr, dtype=np.int64))


def write_chunks(*, indices, data, indptr, n_cols, out_dir, grid,
                 kmers=None, data_dtype=None, meta=None):
    """Write the planned ``grid`` as ``band/block`` chunks + an atomic manifest."""
    indices = np.asarray(indices)
    indptr = np.asarray(indptr)
    n_rows = int(indptr.shape[0] - 1)
    if data_dtype is None:
        data_dtype = np.asarray(data).dtype
    data_dtype = np.dtype(data_dtype)
    os.makedirs(out_dir, exist_ok=True)
    band_recs, chunk_recs = [], []
    total = 0
    chunk_id = 0
    any_rowsplit = False
    for b, band in enumerate(grid):
        c0, c1 = band["c0"], band["c1"]
        li, ld, bip = band["local_idx"], band["local_data"], band["band_indptr"]
        band_dir = os.path.join(out_dir, f"band_{b:05d}")
        os.makedirs(band_dir, exist_ok=True)
        if kmers is not None:                       # band legend, written once
            with open(os.path.join(band_dir, "kmers.csv"), "w") as fh:
                fh.write("index,K-mer\n")
                fh.writelines(f"{j},{kmers[c0 + j]}\n" for j in range(c1 - c0))
        rblocks = band["row_blocks"]
        any_rowsplit = any_rowsplit or len(rblocks) > 1
        cids = []
        for rb, (g0, g1) in enumerate(rblocks):
            s, e = int(bip[g0]), int(bip[g1])
            cidx = li[s:e]
            cdat = ld[s:e].astype(data_dtype)
            cip = (bip[g0:g1 + 1] - bip[g0]).astype(np.int64)
            rel = f"band_{b:05d}/block_{rb:05d}"
            _save_arrays(os.path.join(out_dir, rel), cidx, cdat, cip)
            nnz = int(cip[-1])
            total += nnz
            chunk_recs.append({"id": chunk_id, "band_id": b, "block_id": rb,
                               "dir": rel, "c0": int(c0), "c1": int(c1),
                               "g0": int(g0), "g1": int(g1),
                               "n_cols": int(c1 - c0), "n_rows": int(g1 - g0),
                               "nnz": nnz})
            cids.append(chunk_id)
            chunk_id += 1
        band_recs.append({"id": b, "dir": f"band_{b:05d}", "c0": int(c0),
                          "c1": int(c1), "n_cols": int(c1 - c0),
                          "n_row_blocks": len(rblocks), "chunk_ids": cids})
    manifest = {"format": _FORMAT}
    if meta:
        manifest.update(meta)
    manifest.update({
        "n_rows": n_rows, "n_cols": int(n_cols),
        "n_bands": len(grid), "n_chunks": chunk_id, "total_nnz": total,
        "blocking": "2d" if any_rowsplit else "1d",
        "sparsity": (1.0 - total / float(n_rows * n_cols)) if n_rows and n_cols else 1.0,
        "value_semantics": "presence" if data_dtype == np.int8 else "counts",
        "dtypes": {"index": "int32", "data": str(data_dtype), "indptr": "int64"},
        "chunk_files": {"index": "column.npy", "data": "data.npy", "indptr": "row.npy"},
        "band_legend": "kmers.csv (in each band dir; shared by the band's blocks)",
        "row_axis": "genomes in input-manifest order; a band's blocks tile all rows",
        "col_axis": "k-mers; bands partition [0,n_cols); chunk col j == global c0+j",
        "index_dtype": "int32", "data_dtype": str(data_dtype),   # legacy flat keys
        "bands": band_recs,
        "chunks": chunk_recs,
    })
    tmp = os.path.join(out_dir, MANIFEST + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, MANIFEST))   # atomic commit
    return manifest


def chunk_existing_output(output_dir, suffix, *, device_gb, out_dir=None):
    """Convert a finished KMX matrix into a device-sized 2-D chunk grid."""
    column = np.load(os.path.join(output_dir, f"column_{suffix}.npy"))
    data = np.load(os.path.join(output_dir, f"data_{suffix}.npy"))
    row = np.load(os.path.join(output_dir, f"row_{suffix}.npy"))
    kcsv = os.path.join(output_dir, f"set_of_all_unique_kmers_{suffix}.csv")
    kmers = None
    n_cols = None
    if os.path.exists(kcsv):
        kmers = []
        with open(kcsv) as fh:
            next(fh, None)
            for line in fh:
                kmers.append(line.rstrip("\n").split(",", 1)[1])
        n_cols = len(kmers)
    if n_cols is None:
        n_cols = (int(column.max()) + 1) if column.size else 0
    grid, _ = grid_for_device(column, data, row, n_cols, device_gb=device_gb,
                              item_bytes=data.dtype.itemsize)
    if out_dir is None:
        out_dir = os.path.join(output_dir, f"chunks_{suffix}")
    meta = _provenance(suffix, output_dir, device_gb)
    return write_chunks(indices=column, data=data, indptr=row, n_cols=n_cols,
                        out_dir=out_dir, grid=grid, kmers=kmers,
                        data_dtype=data.dtype, meta=meta)


def _provenance(suffix, source_dir, device_gb):
    try:
        from . import __version__ as _ver
    except Exception:
        _ver = "unknown"
    params = {}
    m = re.match(r"k(\d+)_min(\d+)_max(\d+)_d([01])(_presence)?$", suffix or "")
    if m:
        k, mn, mx, d, _pres = m.groups()
        params = {"kmer_size": int(k), "min_count": int(mn), "max_count": int(mx),
                  "canonical": d == "0"}
    return {"tool": "KMX", "tool_version": _ver,
            "created_utc": datetime.datetime.now(datetime.timezone.utc)
                                   .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "suffix": suffix, "params": params,
            "source_dir": os.path.abspath(source_dir),
            "target_device_gb": float(device_gb)}


# ── reading ──────────────────────────────────────────────────────────────────
def read_manifest(chunks_dir):
    with open(os.path.join(chunks_dir, MANIFEST)) as fh:
        return json.load(fh)


def read_kmers(chunks_dir, *, manifest=None):
    """Global ordered k-mer legend: column id -> k-mer string, reassembled from the
    per-band kmers.csv files (each holds LOCAL ids; global col = band.c0 + local).
    Returns a list of length n_cols (None for any column whose legend wasn't written)."""
    m = manifest or read_manifest(chunks_dir)
    out = [None] * int(m["n_cols"])
    for b in m["bands"]:
        path = os.path.join(chunks_dir, b["dir"], "kmers.csv")
        if not os.path.exists(path):
            continue
        c0 = int(b["c0"])
        with open(path) as fh:
            next(fh)
            for ln in fh:
                idx, km = ln.rstrip("\n").split(",", 1)
                out[c0 + int(idx)] = km
    return out


def _chunk_rec(manifest, chunk_id):
    for ch in manifest["chunks"]:
        if ch["id"] == chunk_id:
            return ch
    raise KeyError(f"chunk id {chunk_id} not in manifest")


def load_chunk_arrays(chunks_dir, chunk_id, *, manifest=None):
    """Raw arrays for one tile: ``(idx_i32, data, indptr_i64, shape, c0, g0)``."""
    if manifest is None:
        manifest = read_manifest(chunks_dir)
    rec = _chunk_rec(manifest, chunk_id)
    cdir = os.path.join(chunks_dir, rec["dir"])
    idx = np.load(os.path.join(cdir, "column.npy"))
    dat = np.load(os.path.join(cdir, "data.npy"))
    ip = np.load(os.path.join(cdir, "row.npy"))
    return idx, dat, ip, (rec["n_rows"], rec["n_cols"]), int(rec["c0"]), int(rec["g0"])


def load_chunk(chunks_dir, chunk_id, *, device="cpu", manifest=None):
    """One tile as a sparse matrix (``n_rows × n_cols`` of the tile) + ``(c0, g0)``
    global offsets (tile row i ↔ genome g0+i, col j ↔ k-mer c0+j)."""
    idx, dat, ip, shape, c0, g0 = load_chunk_arrays(chunks_dir, chunk_id,
                                                    manifest=manifest)
    if device == "gpu":
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        m = cusparse.csr_matrix(
            (cp.asarray(dat), cp.asarray(idx), cp.asarray(ip)), shape=shape)
    else:
        import scipy.sparse
        m = scipy.sparse.csr_matrix((dat, idx, ip), shape=shape)
    return m, c0, g0


def reconstruct_csr(chunks_dir, *, manifest=None):
    """Rebuild full CSR arrays ``(indices_global, data, indptr, n_cols)`` from the
    2-D grid (pure numpy). For validation / a user who wants the whole matrix."""
    if manifest is None:
        manifest = read_manifest(chunks_dir)
    n_rows, n_cols = int(manifest["n_rows"]), int(manifest["n_cols"])
    data_dtype = np.dtype(manifest["data_dtype"])
    # tiles in column order (band c0), then row order (g0) → per-row column-sorted
    recs = sorted(manifest["chunks"], key=lambda c: (c["c0"], c["g0"]))
    loaded = []
    counts = np.zeros(n_rows, dtype=np.int64)
    for rec in recs:
        idx, dat, ip, _shape, c0, g0 = load_chunk_arrays(chunks_dir, rec["id"],
                                                         manifest=manifest)
        c = np.diff(np.asarray(ip)).astype(np.int64)
        counts[g0:g0 + c.shape[0]] += c
        loaded.append((idx, dat, np.asarray(ip), c0, g0, c))
    full_indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    nnz = int(full_indptr[-1])
    full_idx = np.empty(nnz, dtype=np.int64)
    full_dat = np.empty(nnz, dtype=data_dtype)
    placed = full_indptr[:-1].copy()
    for idx, dat, ip, c0, g0, c in loaded:
        if idx.size:
            rows = np.repeat(np.arange(g0, g0 + c.shape[0]), c)
            within = np.arange(idx.size, dtype=np.int64) - ip[rows - g0]
            dest = placed[rows] + within
            full_idx[dest] = idx.astype(np.int64) + c0
            full_dat[dest] = dat
        placed[g0:g0 + c.shape[0]] += c
    return full_idx, full_dat, full_indptr, n_cols


# ── integrity verification ───────────────────────────────────────────────────
def verify_chunks(chunks_dir, *, manifest=None, deep=True):
    """Check the on-disk grid is complete and consistent. Returns ``(ok, issues)``.

    Asserts: column bands tile ``[0, n_cols)``; within each band the blocks tile
    ``[0, n_rows)``; Σ tile nnz == total; every tile loads with matching shape;
    local indices in range. ``deep=False`` skips loading the arrays (metadata only).
    """
    if manifest is None:
        manifest = read_manifest(chunks_dir)
    issues = []
    nr, ncl = int(manifest["n_rows"]), int(manifest["n_cols"])
    bands = sorted(manifest["bands"], key=lambda b: b["c0"])
    # 1) column bands tile [0, n_cols)
    expect = 0
    for b in bands:
        if b["c0"] != expect:
            issues.append(f"column gap/overlap at band {b['id']}: c0={b['c0']} expected {expect}")
        expect = b["c1"]
    if expect != ncl:
        issues.append(f"columns cover {expect}, expected {ncl}")
    # 2) within each band, row blocks tile [0, n_rows); collect tile nnz
    chunks_by_band = {}
    for ch in manifest["chunks"]:
        chunks_by_band.setdefault(ch["band_id"], []).append(ch)
    total = 0
    for b in bands:
        chs = sorted(chunks_by_band.get(b["id"], []), key=lambda c: c["g0"])
        rexp = 0
        for ch in chs:
            if ch["g0"] != rexp:
                issues.append(f"band {b['id']} row gap/overlap: g0={ch['g0']} expected {rexp}")
            if ch["c0"] != b["c0"] or ch["c1"] != b["c1"]:
                issues.append(f"chunk {ch['id']} col range != its band")
            rexp = ch["g1"]
            total += ch["nnz"]
        if rexp != nr:
            issues.append(f"band {b['id']} rows cover {rexp}, expected {nr}")
    if total != int(manifest["total_nnz"]):
        issues.append(f"Σ tile nnz {total} != total_nnz {manifest['total_nnz']}")
    # 3) deep: load every tile, check shapes / dtypes / index range / indptr
    if deep:
        idtype = np.dtype(manifest["dtypes"]["data"])
        for ch in manifest["chunks"]:
            try:
                idx, dat, ip, shape, c0, g0 = load_chunk_arrays(
                    chunks_dir, ch["id"], manifest=manifest)
            except Exception as exc:                 # noqa: BLE001
                issues.append(f"chunk {ch['id']} failed to load: {exc!r}")
                continue
            if idx.dtype != np.int32:
                issues.append(f"chunk {ch['id']} index dtype {idx.dtype} != int32")
            if dat.dtype != idtype:
                issues.append(f"chunk {ch['id']} data dtype {dat.dtype} != {idtype}")
            if ip.shape[0] != ch["n_rows"] + 1:
                issues.append(f"chunk {ch['id']} indptr len {ip.shape[0]} != n_rows+1")
            if int(ip[-1]) != ch["nnz"] or idx.shape[0] != ch["nnz"]:
                issues.append(f"chunk {ch['id']} nnz mismatch")
            if idx.size and int(idx.max()) >= ch["n_cols"]:
                issues.append(f"chunk {ch['id']} local col index >= band width")
    return (len(issues) == 0), issues
