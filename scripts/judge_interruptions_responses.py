#!/usr/bin/env python3
"""Rate post-interruption response relevance with Gemma 4 31B IT.

For every existing ``response_score.json`` in one model/language interruptions
directory, this script applies the Full-Duplex-Bench user-interruption rubric to
``context_en``, ``interrupt_en``, and ``transcription_en``. It atomically adds
an integer ``rating`` from 0 through 5 and a string ``analysis`` to the score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


CATEGORY = "interruptions"
MODEL_ID = "google/gemma-4-31B-it"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = """
The scenario is that the user and AI are talking in the spoken conversation.
The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
Your task is to rate the quality of AI's response after the user interrupt the turn.


Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
- 0: The AI's response is totally unrelated to the user's interrupting turn.
- 1: The AI's response is not related to the user's interrupting turn.
- 2: The AI's response is slightly related to the user's interrupting turn.
- 3: The AI's response is related to the user's interrupting turn.
- 4: The AI's response is highly related to the user's interrupting turn.
- 5: The AI's response is perfectly related to the user's interrupting turn.


Firstly, briefly analyze the user's interrupting turn and the AI's response
Then, you must return the overall output as the following format:
Analysis: [Your analysis].
I would rate the AI's response as [Rating].
""".strip()

OUTPUT_PATTERN = re.compile(
    r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="model output root",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="benchmark input root",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="language directory to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Gemma generation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "maximum generated tokens per judgment "
            f"(default: {DEFAULT_MAX_NEW_TOKENS})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate existing valid rating and analysis fields",
    )
    return parser.parse_args()


def read_score(score_path: Path) -> dict[str, Any]:
    """Read and validate the three English fields used by the judge."""
    with score_path.open("r", encoding="utf-8") as score_file:
        score = json.load(score_file)
    if not isinstance(score, dict):
        raise ValueError(f"response score must be a JSON object: {score_path}")
    for field in ("context_en", "interrupt_en", "transcription_en"):
        if field not in score:
            raise KeyError(f"'{field}' is missing from {score_path}")
        if not isinstance(score[field], str):
            raise TypeError(f"'{field}' must be a string in {score_path}")
    return score


def has_valid_judgment(score: dict[str, Any]) -> bool:
    """Return whether a score already contains a complete valid judgment."""
    rating = score.get("rating")
    analysis = score.get("analysis")
    return (
        isinstance(rating, int)
        and not isinstance(rating, bool)
        and 0 <= rating <= 5
        and isinstance(analysis, str)
        and bool(analysis.strip())
    )


def user_prompt(score: dict[str, Any]) -> str:
    """Build the Full-Duplex-Bench labeled user message from English fields."""
    return (
        f"- Contextual user turn: {score['context_en']}\n"
        f"- User interrupting turn: {score['interrupt_en']}\n"
        f"- AI's response: {score['transcription_en']}"
    )


def parse_judgment(text: str) -> dict[str, Any]:
    """Parse the benchmark's required output format and validate its range."""
    parsed: dict[str, Any] = {}
    for match in OUTPUT_PATTERN.finditer(text + "\n"):
        parsed = {
            "analysis": match.group(1).strip(),
            "rating": int(match.group(2).strip()),
        }
    if not parsed:
        raise ValueError(f"could not parse judgment output: {text!r}")
    if not parsed["analysis"]:
        raise ValueError("judgment analysis is empty")
    if not 0 <= parsed["rating"] <= 5:
        raise ValueError(f"judgment rating is outside 0-5: {parsed['rating']}")
    return parsed


def write_score_atomic(score_path: Path, score: dict[str, Any]) -> None:
    """Atomically replace one response score with indented UTF-8 JSON."""
    temporary_path = score_path.with_name(
        f".{score_path.stem}.{os.getpid()}.json"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as score_file:
            json.dump(score, score_file, ensure_ascii=False, indent=2)
            score_file.write("\n")
        os.replace(temporary_path, score_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def extract_parsed_text(parsed: Any) -> str | None:
    """Extract text from the processor's version-dependent parsed response."""
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, (tuple, list)):
        for item in reversed(parsed):
            result = extract_parsed_text(item)
            if result:
                return result
    if isinstance(parsed, dict):
        for key in ("final", "answer", "content", "text", "response"):
            if key in parsed:
                result = extract_parsed_text(parsed[key])
                if result:
                    return result
    for attribute in ("final", "answer", "content", "text", "response"):
        if hasattr(parsed, attribute):
            result = extract_parsed_text(getattr(parsed, attribute))
            if result:
                return result
    return None


def decode_completion(processor: Any, token_ids: Any, prefix: Any) -> str:
    """Decode one Gemma completion using its structured response parser."""
    # ``generate`` right-pads shorter rows to the longest completion in a
    # batch. Gemma's structured parser can mistake those trailing pad tokens
    # for the response, so remove them before either decoding path.
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        token_ids = token_ids[token_ids != pad_token_id]
    if hasattr(processor, "parse_response"):
        raw = processor.decode(token_ids, skip_special_tokens=False)
        try:
            parsed = extract_parsed_text(
                processor.parse_response(raw, prefix=prefix)
            )
            if parsed:
                return parsed.strip()
        except Exception as error:
            print(
                f"Warning: processor.parse_response failed ({error}); "
                "using decoded text",
                flush=True,
            )
    return processor.decode(token_ids, skip_special_tokens=True).strip()


def tokenize_prompts(processor: Any, user_prompts: list[str]) -> Any:
    """Apply Gemma's chat template to benchmark system/user messages."""
    conversations = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        for prompt in user_prompts
    ]
    return processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )


def load_model() -> tuple[Any, Any]:
    """Load the cached Gemma 4 31B IT processor and model."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    model_dir = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    common = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    try:
        from transformers import AutoModelForMultimodalLM

        model_class = AutoModelForMultimodalLM
    except ImportError:
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    model = model_class.from_pretrained(model_dir, **common).eval()
    return processor, model


def report_progress(processed: int, total: int, interval: int) -> None:
    """Print progress at approximately five-percent intervals."""
    if processed % interval == 0 or processed == total:
        percentage = 100.0 * processed / total
        print(f"Progress: {processed}/{total} samples ({percentage:.0f}%)", flush=True)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("error: --batch-size must be at least 1", file=sys.stderr)
        return 2
    if args.max_new_tokens < 1:
        print("error: --max-new-tokens must be at least 1", file=sys.stderr)
        return 2

    input_category_dir = args.input_dir / args.language / CATEGORY
    output_category_dir = args.output_dir / args.language / CATEGORY
    missing = [
        path
        for path in (input_category_dir, output_category_dir)
        if not path.is_dir()
    ]
    if missing:
        print("Skipping: missing directory/directories: " + ", ".join(map(str, missing)))
        return 0

    all_sample_dirs = sorted(
        path for path in output_category_dir.iterdir() if path.is_dir()
    )
    score_paths = [
        sample_dir / "response_score.json"
        for sample_dir in all_sample_dirs
        if (sample_dir / "response_score.json").is_file()
    ]
    missing_scores = len(all_sample_dirs) - len(score_paths)
    if not score_paths:
        print(
            f"Skipping: no response_score.json files found in {output_category_dir}; "
            f"samples without scores: {missing_scores}"
        )
        return 0

    failures: list[tuple[Path, Exception]] = []
    pending: list[tuple[Path, dict[str, Any], str]] = []
    skipped_existing = 0
    for score_path in score_paths:
        try:
            score = read_score(score_path)
            if has_valid_judgment(score) and not args.overwrite:
                skipped_existing += 1
                continue
            pending.append((score_path, score, user_prompt(score)))
        except Exception as error:
            failures.append((score_path, error))
            print(
                f"Failed {score_path}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

    if not pending:
        print(
            f"Nothing to judge in {output_category_dir}; missing scores: "
            f"{missing_scores}; skipped existing: {skipped_existing}; "
            f"failures: {len(failures)}."
        )
        return 1 if failures else 0

    print(
        f"Found {len(score_paths)} response scores; pending: {len(pending)}; "
        f"missing scores: {missing_scores}; skipped existing: {skipped_existing}",
        flush=True,
    )

    try:
        import torch
    except ImportError as error:
        print(f"error: missing dependency: {error}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print(
            "error: CUDA is not available; run this script in a GPU allocation",
            file=sys.stderr,
        )
        return 2

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Loading {MODEL_ID} from the local Hugging Face cache", flush=True)
    try:
        processor, model = load_model()
    except Exception as error:
        print(f"error: failed to load model: {error}", file=sys.stderr)
        return 2

    completed = 0
    processed = 0
    progress_interval = max(1, len(pending) // 20)
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        try:
            inputs = tokenize_prompts(
                processor,
                [prompt for _, _, prompt in batch],
            ).to(model.device)
            input_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            completions = [
                decode_completion(
                    processor,
                    row[input_width:],
                    inputs["input_ids"][index],
                )
                for index, row in enumerate(generated)
            ]
            for completion, (score_path, score, _) in zip(completions, batch):
                try:
                    judgment = parse_judgment(completion)
                    score.update(judgment)
                    write_score_atomic(score_path, score)
                    completed += 1
                except Exception as error:
                    failures.append((score_path, error))
                    print(
                        f"Failed {score_path}: {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception as error:
            for score_path, _, _ in batch:
                failures.append((score_path, error))
                print(
                    f"Failed {score_path}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            if args.batch_size > 1:
                processed += len(batch)
                report_progress(processed, len(pending), progress_interval)
                break

        processed += len(batch)
        report_progress(processed, len(pending), progress_interval)

    print(
        f"Judged {completed}/{len(pending)} pending response scores in "
        f"{output_category_dir}; missing scores: {missing_scores}; "
        f"skipped existing: {skipped_existing}; failures: {len(failures)}.",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
