#!/usr/bin/env python3
"""
This file converse segmented Easy Read data into a JSONL file that has both the normal text and the corresponding segmented text (which is split up on each "_seg_" indicator).
Convert segmented Easy Read data into JSONL records with normal/easy text and AMRs.

Input expectations: a text file with the segmentation data. Normal sentences with "_seg_" indicators within them, where sentences should be split.

Processing rules:
    - Split the text on periods
    - Generate the easy read text by splitting on segs. 
    - Get normal_text and easy_text entries
    - Write to a JSONL file

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

SEG_TOKEN = "_seg_"
WHITESPACE_RE = re.compile(r"\s+")


def normalize_spaces(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()

def split_by_period_only(text: str) -> List[str]:
    """
    Split the text by the periods
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = text.split(".")
    sentences: List[str] = []

    for i, chunk in enumerate(chunks):
        cleaned = normalize_spaces(chunk)
        if not cleaned:
            continue

        if i < len(chunks) - 1:
            cleaned = f"{cleaned}."
        sentences.append(cleaned)

    return sentences

def make_normal_text(sentence: str) -> str:
    """removing seg markers """
    text = sentence.replace(SEG_TOKEN, " ")
    text = normalize_spaces(text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text

def make_easy_segments(sentence: str) -> List[str]:
    """Split on the _seg_ and return the fractured sentences"""
    parts = sentence.split(SEG_TOKEN)
    segments = []
    for part in parts:
        segment = normalize_spaces(part)
        if segment:
            segment = re.sub(r"\s+([.,!?;:])", r"\1", segment)
            segments.append(segment)
    return segments

def load_input_sentences(input_path: Path) -> List[str]:
    raw_text = input_path.read_text(encoding="utf-8")
    return split_by_period_only(raw_text)

def write_jsonl(records: List[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL with Easy Read")
    parser.add_argument("input", type=Path, help="path to seg file")
    parser.add_argument("output", type=Path, help="path to output JSONL")
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="",
        help="Prefix for record IDs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="AMR parsing batch size (default: 16).",
    )
    args = parser.parse_args()

    sentences = load_input_sentences(args.input)
    if not sentences:
        write_jsonl([], args.output)
        return

    records = []
    for i, sentence in enumerate(sentences):
        records.append({
            "id": f"{args.id_prefix}{i:05d}",
            "normal_text": make_normal_text(sentence),
            "easy_text": make_easy_segments(sentence),
        })

    write_jsonl(records, args.output)


if __name__ == "__main__":
    main()

#python3 segmentation.py en.dev.seg data1.jsonl
