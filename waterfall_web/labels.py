from __future__ import annotations

import csv
from pathlib import Path


def _parse_label(value: str) -> int:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "signal", "with_signal", "with-signal"}:
        return 1
    if v in {"0", "false", "f", "no", "n", "no_signal", "no-signal", "without_signal", "without-signal"}:
        return 0
    raise ValueError(f"Unrecognized label value: {value!r}")


def load_ground_truth(labels_file: Path | None) -> dict[str, int]:
    """Loads ground-truth labels as {relative_filename: 0|1}.

    Expected CSV formats:
      - filename,label
      - filename,has_signal
    Header is optional.
    """

    if labels_file is None:
        return {}

    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    mapping: dict[str, int] = {}

    with labels_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return mapping

        # Detect header
        header_like = {c.strip().lower() for c in first_row}
        has_header = "filename" in header_like and ("label" in header_like or "has_signal" in header_like)

        if has_header:
            # Re-read as DictReader
            f.seek(0)
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                filename = (row.get("filename") or "").strip()
                label_raw = (row.get("label") or row.get("has_signal") or "").strip()
                if not filename:
                    continue
                mapping[filename] = _parse_label(label_raw)
            return mapping

        # No header: treat first row as data
        if len(first_row) >= 2:
            mapping[first_row[0].strip()] = _parse_label(first_row[1])

        for row in reader:
            if len(row) < 2:
                continue
            mapping[row[0].strip()] = _parse_label(row[1])

    return mapping
