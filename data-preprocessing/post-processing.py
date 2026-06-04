#!/usr/bin/env python3

import json
import re
from pathlib import Path


FILLER_PATTERNS = [
    r"\bexists?\b",
    r"\bsomething\b",
    r"\bthis happens\b",
    r"\bit happens\b",
    r"\bare involved\b",
    r"\bneed support\b",
    r"\bsaid something\b",
    r"\bsays something\b",
    r"\bacted\b",
]

INCOMPLETE_SENTENCE_PATTERNS = [
    r"\babout\.$",
    r"\bto\.$",
    r"\bfor\.$",
    r"\bwith\.$",
    r"\bthe\.$",
    r"\band\.$",
    r"\bby\.$",
    r"\bof\.$"
]


def is_flagged(sentence: str) -> tuple[bool, str]:
    s = sentence.strip()

    for pattern in FILLER_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return True, f"filler pattern: {pattern}"

    for pattern in INCOMPLETE_SENTENCE_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return True, f"incomplete sentence pattern: {pattern}"

    words = s.split()
    if len(words) <= 2 and s not in ("1.", "2.", "3."):
        return True, "too short"

    return False, ""


def flag_and_write_fixes(rewritten_path: Path) -> None:
    """
    Find all the elements with issues in rewritten data in the JSONL and write them to a seperate file so they can be manually checked
    """
    fixes_path = Path("segmentation_fixes_all.jsonl")
    flagged_count = 0

    with rewritten_path.open("r", encoding="utf-8") as f_in, \
         fixes_path.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            record = json.loads(line)
            rewritten = record.get("easy_read_rewritten", [])

            flags = []
            for j, sentence in enumerate(rewritten):
                flagged, reason = is_flagged(sentence)
                if flagged:
                    flags.append({
                        "index": j,
                        "sentence": sentence,
                        "reason": reason,
                        "original_segment": record["easy_text"][j] if j < len(record["easy_text"]) else None,
                    })

            if flags:
                record["flags"] = flags
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                flagged_count += 1

    print(f"Flagged {flagged_count} records -> segmentation_fixes_all.jsonl")
    print("Manually correct segmentation_fixes_all.jsonl, then run merge_fixes().")

def diagnose(rewritten_path: Path) -> None:
    from collections import Counter
    reasons = Counter()

    with rewritten_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for sentence in record.get("easy_read_rewritten", []):
                flagged, reason = is_flagged(sentence)
                if flagged:
                    reasons[reason] += 1

    for reason, count in reasons.most_common():
        print(f"{count:4d}  {reason}")

def merge_fixes(rewritten_path: Path) -> None:
    """
    Once segmentation_fixes_all.jsonl  has been manually correct, this function uses th efull cleaned up records and writes them to post_processed_segmentation.jsonl
    """
    fixes_path = Path("segmentation_fixes_all.jsonl")
    output_path = Path("post_processed_segmentation_all.jsonl")

    if not fixes_path.exists():
        print("segmentation_fixes_all.jsonl not found. Run flag_and_write_fixes() first.")
        return

    # We load all the corrected records keyed by id
    corrected = {}
    with fixes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            #strip the flags from issues before writing to the final output
            record.pop("flags", None)
            record.pop("rewrite_errors", None)
            corrected[record["id"]] = record

    clean_count = 0
    fixed_count = 0

    with rewritten_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["id"] in corrected:
                # using the version that was fixed
                f_out.write(json.dumps(corrected[record["id"]], ensure_ascii=False) + "\n")
                fixed_count += 1
            else:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                clean_count += 1

    print(f"Clean records:  {clean_count}")
    print(f"Fixed records:  {fixed_count}")
    print(f"Total:          {clean_count + fixed_count}")
    print(f"Output       -> {output_path}")


if __name__ == "__main__":
    """
    Its a two step process, as described below. Please comment out whichever step you arent running right now.
    Remember to manually correct/edit the data in segmentation_fixes_all.jsonl in between the two fixes.
    """
    rewritten_path = Path("rewritten_data_all.jsonl")

    # Step 1: run the follwoing and then just go manually edit segmentation_fixes_all.jsonl
    #flag_and_write_fixes(rewritten_path)
    #diagnose(rewritten_path)

   # Step 2: once  edits are done, comment out step 1 and run this to get the full final pre-processing data output
    merge_fixes(rewritten_path)