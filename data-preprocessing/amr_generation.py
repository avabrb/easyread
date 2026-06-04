#!/usr/bin/env python3
"""
AMR Graph Generation Pipeline Step.
Parses Abstract Meaning Representation (AMR) graphs from normal sentences and easy segments
using amrlib's BART-based parser model.

Input: JSONL file containing "normal_text" and "easy_read_rewritten" (or "easy_text").
Output: JSONL file containing original fields plus "normal_amr" and "easy_amr" lists of AMR graphs in PENMAN format.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, List


def setup_amr_model():
    """Download and install the BART-based AMR parser model if not already present."""
    import amrlib

    amrlib_dir = Path(amrlib.__file__).resolve().parent
    data_dir = amrlib_dir / "data"
    data_dir.mkdir(exist_ok=True)

    url = "https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz"
    tar_path = data_dir / "model_parse_xfm_bart_large-v0_1_0.tar.gz"
    extracted_dir = data_dir / "model_parse_xfm_bart_large-v0_1_0"
    target_dir = data_dir / "model_stog"

    if not target_dir.exists():
        if not tar_path.exists():
            print("Downloading AMR parser model (this might take a few minutes)...")
            urllib.request.urlretrieve(url, tar_path)

        if not extracted_dir.exists():
            print("Extracting model tarball...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)

        extracted_dir.rename(target_dir)
        print("Installed parser model at:", target_dir)
    else:
        print("AMR model is already installed.")


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parse_amrs(sentences: List[str], stog_model, batch_size: int = 16) -> List[str]:
    if not sentences:
        return []

    outputs: List[str] = []
    for batch in batched(sentences, batch_size):
        graphs = stog_model.parse_sents(batch)
        outputs.extend(graphs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AMR parses for normal and simplified texts."
    )
    parser.add_argument("input", type=Path, help="Input JSONL file.")
    parser.add_argument("output", type=Path, help="Output JSONL file.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="AMR parsing batch size (default: 16).",
    )
    args = parser.parse_args()

    # Ensure model is installed
    setup_amr_model()

    # Load input records
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading records from {args.input}...")
    records = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No records found in input file. Writing empty output.")
        with args.output.open("w", encoding="utf-8") as f:
            pass
        return

    import amrlib

    print("Loading AMR model...")
    stog_model = amrlib.load_stog_model()

    # Gather sentences to parse
    normal_texts: List[str] = []
    flat_easy_texts: List[str] = []
    easy_lens: List[int] = []

    for r in records:
        # Collect normal text
        normal_texts.append(r.get("normal_text", ""))

        # Collect rewritten Easy Read sentences (or fallback to original segments)
        easy_sents = r.get("easy_read_rewritten")
        if not easy_sents:
            easy_sents = r.get("easy_text", [])
        flat_easy_texts.extend(easy_sents)
        easy_lens.append(len(easy_sents))

    # Parse in batches
    print(f"Parsing {len(normal_texts)} normal sentence(s)...")
    normal_amrs = parse_amrs(normal_texts, stog_model, batch_size=args.batch_size)

    print(f"Parsing {len(flat_easy_texts)} easy sentence(s)...")
    flat_easy_amrs = parse_amrs(
        flat_easy_texts, stog_model, batch_size=args.batch_size
    )

    # Reconstruct records
    easy_idx = 0
    for i, r in enumerate(records):
        # Store normal AMR as a list containing the AMR string
        r["normal_amr"] = [normal_amrs[i]] if i < len(normal_amrs) and normal_amrs[i] else []

        # Store easy AMR as a list of AMR strings corresponding to easy sentences
        n_easy = easy_lens[i]
        r["easy_amr"] = flat_easy_amrs[easy_idx : easy_idx + n_easy]
        easy_idx += n_easy

    # Write output JSONL
    print(f"Writing parsed records to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("AMR Generation completed successfully.")


if __name__ == "__main__":
    main()
