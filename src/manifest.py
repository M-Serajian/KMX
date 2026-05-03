"""KMX v2 — input manifest parser.

The user-facing input is a long-format CSV with exactly two columns:

    sample_id,file
    GENOME_A,/data/A_R1.fq.gz
    GENOME_A,/data/A_R2.fq.gz
    GENOME_B,/data/B.fa

Rules:
  * Header row required: ``sample_id,file`` (case-insensitive, whitespace-stripped).
  * Blank lines and ``#``-prefixed comment lines are skipped.
  * Each row maps one file to one sample_id. Multiple rows may share a
    sample_id; their files are merged into one genome at count time.
  * Every file must exist and have a recognized extension.
  * Format family is auto-detected from the extension. The whole manifest
    must be one family — all FASTA-like or all FASTQ-like. ``.gz`` is
    transparent and may be mixed freely with plain files within a family.
  * Relative paths are resolved against the manifest's directory (the
    same convention nf-core uses), so manifests are portable when carried
    alongside the data.

This module performs ALL validation up front so failures are surfaced
before any KMC subprocess is spawned.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


_FASTA_EXTS = ("fa", "fasta", "fna", "faa")
_FASTQ_EXTS = ("fq", "fastq")

_REQUIRED_HEADER = ("sample_id", "file")


@dataclass(frozen=True)
class Manifest:
    """Parsed manifest, ready to drive stage 1 + stage 2."""

    sample_ids: List[str]                 # ordered by first appearance in CSV
    files_by_sample: Dict[str, List[str]] # sample_id -> list of resolved file paths
    flat_files: List[str]                 # all files in CSV order (for stage 1)
    input_fmt: str                        # "mfasta" or "fastq"
    family: str                           # "fasta" or "fastq" (human-readable label)


class ManifestError(Exception):
    """Raised when the manifest is malformed or references missing files."""


def _classify_extension(path: str) -> str:
    """Return 'fasta', 'fastq', or '' (unknown) for a file path.

    Strips a single trailing '.gz' before classifying so that '.fa.gz' and
    '.fa' resolve identically.
    """
    base = path.lower()
    if base.endswith(".gz"):
        base = base[:-3]
    _, ext = os.path.splitext(base)
    ext = ext.lstrip(".")
    if ext in _FASTA_EXTS:
        return "fasta"
    if ext in _FASTQ_EXTS:
        return "fastq"
    return ""


def _resolve_path(raw: str, manifest_dir: str) -> str:
    """Expand ~ and resolve relative paths against the manifest's directory."""
    p = os.path.expanduser(raw)
    if not os.path.isabs(p):
        p = os.path.join(manifest_dir, p)
    return os.path.abspath(p)


def parse_manifest(manifest_path: str) -> Manifest:
    """Parse and fully validate a manifest CSV.

    Raises ManifestError with a clear message on any structural problem,
    missing file, unknown extension, or mixed FASTA/FASTQ content.
    """
    manifest_path = os.path.abspath(os.path.expanduser(manifest_path))
    if not os.path.isfile(manifest_path):
        raise ManifestError(f"Manifest file not found: {manifest_path}")
    manifest_dir = os.path.dirname(manifest_path)

    sample_ids: List[str] = []
    seen_samples: Dict[str, int] = {}     # sample_id -> insertion index
    files_by_sample: Dict[str, List[str]] = {}
    flat_files: List[str] = []
    seen_pairs: set = set()               # (sample_id, abs_path) — dedup repeated rows

    families_seen: Dict[str, int] = {}    # family -> first line number for diagnostics
    header_seen = False

    with open(manifest_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for line_no, row in enumerate(reader, start=1):
            # Drop pure-blank rows and full-line comments. csv splits on
            # commas, so a line beginning with '#' shows up as row[0]=='#...'.
            if not row:
                continue
            stripped_first = row[0].strip()
            if not stripped_first and len(row) == 1:
                continue
            if stripped_first.startswith("#"):
                continue
            # All-empty cells (e.g. ",,,") — treat as blank.
            if all(not c.strip() for c in row):
                continue

            cells = [c.strip() for c in row]

            if not header_seen:
                if len(cells) < 2 or tuple(c.lower() for c in cells[:2]) != _REQUIRED_HEADER:
                    raise ManifestError(
                        f"{manifest_path}:{line_no}: expected header "
                        f"'sample_id,file' as the first non-comment row, got "
                        f"{','.join(cells) or '(empty)'!r}."
                    )
                header_seen = True
                continue

            if len(cells) < 2:
                raise ManifestError(
                    f"{manifest_path}:{line_no}: expected 2 columns "
                    f"(sample_id,file), got {len(cells)}."
                )
            # Tolerate trailing empty cells, reject extra non-empty ones.
            if len(cells) > 2 and any(c for c in cells[2:]):
                raise ManifestError(
                    f"{manifest_path}:{line_no}: only 2 columns are allowed "
                    f"(sample_id,file). Found extra non-empty cells: "
                    f"{cells[2:]!r}. Multiple files per genome are expressed "
                    f"as multiple rows with the same sample_id, not extra columns."
                )

            sample_id, raw_path = cells[0], cells[1]
            if not sample_id:
                raise ManifestError(
                    f"{manifest_path}:{line_no}: sample_id cell is empty."
                )
            if not raw_path:
                raise ManifestError(
                    f"{manifest_path}:{line_no}: file cell is empty for "
                    f"sample_id={sample_id!r}."
                )

            abs_path = _resolve_path(raw_path, manifest_dir)
            if not os.path.isfile(abs_path):
                raise ManifestError(
                    f"{manifest_path}:{line_no}: file does not exist: "
                    f"{abs_path}  (raw value: {raw_path!r})"
                )

            family = _classify_extension(abs_path)
            if not family:
                raise ManifestError(
                    f"{manifest_path}:{line_no}: unrecognized extension on "
                    f"{abs_path!r}. Accepted: .fa, .fasta, .fna, .faa, .fq, "
                    f".fastq (any with optional .gz)."
                )
            families_seen.setdefault(family, line_no)

            pair_key = (sample_id, abs_path)
            if pair_key in seen_pairs:
                # Same (sample_id, file) appearing twice is a no-op duplicate.
                # Skip silently rather than double-count or error.
                continue
            seen_pairs.add(pair_key)

            if sample_id not in seen_samples:
                seen_samples[sample_id] = len(sample_ids)
                sample_ids.append(sample_id)
                files_by_sample[sample_id] = []
            files_by_sample[sample_id].append(abs_path)
            flat_files.append(abs_path)

    if not header_seen:
        raise ManifestError(
            f"{manifest_path}: empty manifest (no header found). "
            f"First non-comment row must be 'sample_id,file'."
        )
    if not sample_ids:
        raise ManifestError(
            f"{manifest_path}: header present but no data rows found."
        )

    if len(families_seen) > 1:
        first_fasta = families_seen.get("fasta")
        first_fastq = families_seen.get("fastq")
        raise ManifestError(
            f"{manifest_path}: manifest mixes FASTA and FASTQ inputs "
            f"(first FASTA at line {first_fasta}, first FASTQ at line "
            f"{first_fastq}). KMX requires all files in one manifest to "
            f"share one format family because stage 1 runs a single KMC "
            f"call across every file. Split into two manifests or remove "
            f"the minority family."
        )

    family = next(iter(families_seen))
    input_fmt = "mfasta" if family == "fasta" else "fastq"

    return Manifest(
        sample_ids=sample_ids,
        files_by_sample=files_by_sample,
        flat_files=flat_files,
        input_fmt=input_fmt,
        family=family,
    )


def write_genome_index(out_dir: str, suffix: str, sample_ids: List[str]) -> str:
    """Emit a genome_index_<suffix>.csv mapping row index -> sample_id.

    The CSR matrix has one row per genome, and the row order matches
    sample_ids exactly. This file lets users recover which genome is
    which row after the matrix is loaded.
    """
    path = os.path.join(out_dir, f"genome_index_{suffix}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("index", "sample_id"))
        for i, sid in enumerate(sample_ids):
            w.writerow((i, sid))
    return path
