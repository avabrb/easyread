# pip install anthropic python-dotenv

#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

import anthropic
from dotenv import load_dotenv


load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Missing ANTHROPIC_API_KEY. Add ANTHROPIC_API_KEY to your .env file.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def clean_line(text: str) -> str:
    text = text.strip().strip('"').strip("'").strip()
    text = " ".join(text.split())
    if text and text[-1] not in ".!?":
        text += "."
    return text


def rewrite_segments(
    normal_text: str,
    segments: List[str],
    model_name: str,
    max_retries: int = 3,
) -> List[str]:
    numbered_fragments = "\n".join(f"{i+1}. {s}" for i, s in enumerate(segments))

    prompt = f"""You rewrite text into shorter easy read standalone sentences.

A sentence was split into numbered fragments for Easy Read formatting. The fragments together build one complete meaning. Rewrite EACH fragment as a short standalone sentence, such that the sentences read sequentially produce a coherent meaning identical to the original sentence, but simpler.

To do this well:
- Understand what the full sentence is saying overall.
- Understand what role each fragment plays in building that meaning.
- Understand what comes before and after each fragment.

Rules:
- Keep each fragment's original wording as much as possible.
- Only add the minimum words needed to make it a complete sentence.
- Use pronouns (they, she, he, it) instead of repeating full noun phrases — unless it is the first fragment or it would be unclear.
- Do the same for objects — do not repeat long noun phrases if a short reference works.
- Do not use vague placeholder words, especially when the information is contained in the original sentence.
- Never invent content. If a fragment has no predicate, borrow it from the normal sentence — do not make one up.
- Do NOT copy the full normal sentence into any fragment.
- Each output should be similar in length to its input fragment, just completed into a sentence.
- The segments should make sense when read in sequential order, and express the same meaning as the original full sentence, but in a simpler, more legible way.
- Simple language. Short sentences. Easy Read style.

Full sentence (context only):
{normal_text}

Fragments to rewrite:
{numbered_fragments}

Return ONLY a numbered list in the same format, one rewritten sentence per line, nothing else:""".strip()

    last_error = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = message.content[0].text.strip()
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            rewritten = []
            for line in lines:
                cleaned = re.sub(r"^\d+\.\s*", "", line)
                cleaned = clean_line(cleaned)
                if cleaned:
                    rewritten.append(cleaned)

            if len(rewritten) != len(segments):
                raise ValueError(
                    f"Expected {len(segments)} lines, got {len(rewritten)}. "
                    f"Raw output: {raw!r}"
                )
            return rewritten

        except anthropic.RateLimitError as e:
            last_error = e
            wait = 2 ** (attempt + 1)
            print(f"Rate limited. Retrying in {wait}s...")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code >= 500:
                wait = 2 ** (attempt + 1)
                print(f"Server error {e.status_code}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Claude rewrite failed after retries: {last_error}")


def process_jsonl(
    input_path: Path,
    output_path: Path,
    model_name: str,
    limit: Optional[int] = None,
    continue_on_error: bool = True,
) -> None:
    """
    Read the data from the original JSONL (with "id," "normal_text," and "easy_text) and generate a similar JSONL with the additional field: "easy_read_rewritten"
    """
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for i, line in enumerate(f_in):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            normal_text = record["normal_text"]
            easy_text: List[str] = record["easy_text"]
            try:
                rewritten_segments = rewrite_segments(
                    normal_text=normal_text,
                    segments=easy_text,
                    model_name=model_name,
                )
            except Exception as e:
                if not continue_on_error:
                    raise

                print(
                    f"Warning: failed to rewrite record "
                    f"{record.get('id', i)}: {e}"
                )

                rewritten_segments = easy_text
                record["rewrite_errors"] = str(e)

            record["easy_read_rewritten"] = rewritten_segments
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            print(f"Processed {record.get('id', i)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite Easy Read segments into standalone sentences using Claude."
    )

    parser.add_argument("input", type=Path, help="Input JSONL file.")
    parser.add_argument("output", type=Path, help="Output JSONL file.")

    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model name. Default: claude-haiku-4-5-20251001",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records to process for testing.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if Claude fails on a record.",
    )

    args = parser.parse_args()
    process_jsonl(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        limit=args.limit,
        continue_on_error=not args.stop_on_error,
    )


if __name__ == "__main__":
    main()


# python3 rewrite_easy_segments.py data1.jsonl rewritten_data.jsonl (--limit 5)