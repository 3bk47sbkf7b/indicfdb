#!/usr/bin/env python3
"""Summarize voice-agent evaluation scores by language and model.

The benchmark tree is treated as the authoritative sample set, so stale model
outputs that no longer have a corresponding benchmark sample are ignored.
Markdown and JSON summaries are written on every invocation.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


CATEGORIES = (
    "pausehandling",
    "turntaking",
    "backchanneling",
    "interruptions",
)
DEFAULT_LANGUAGE_ORDER = (
    "english",
    "hindi",
    "bengali",
    "punjabi",
    "gujarati",
    "marathi",
    "kannada",
    "telugu",
    "tamil",
    "malayalam",
)
DEFAULT_MODEL_ORDER = (
    "human",
    "gptlive",
    "xai",
    "gemini38live",
    "gemini31flash",
    "personaplex",
    "moshi",
    "hum1",
)
MODEL_DISPLAY_NAMES = {
    "human": "human",
    "gptlive": "gpt-live-1",
    "xai": "grok-voice-think-fast-2.0",
    "gemini38live": "gemini-3.8-live",
    "gemini31flash": "gemini-3.1-flash-live-preview",
    "personaplex": "personaplex",
    "moshi": "moshi",
    "hum1": "joshtalks-human-1",
}
CATEGORY_DISPLAY_NAMES = {
    "pausehandling": "PauseHandling",
    "turntaking": "TurnTaking",
    "backchanneling": "Backchanneling",
    "interruptions": "UserInterruptions",
}
METRIC_DISPLAY_NAMES = {
    "success_rate": "SR",
    "mean_latency_seconds": "Latency",
    "mean_jsd": "JSD",
    "mean_backchannel_frequency_per_second": "Freq",
    "mean_rating": "Rating",
}
SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="benchmark root organized as <language>/<category>/<sample>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="model-output root organized as <model>/<language>/<category>/<sample>",
    )
    parser.add_argument(
        "--human-output",
        type=Path,
        default=Path("evaluation_summary.md"),
        help="Markdown destination (default: evaluation_summary.md)",
    )
    parser.add_argument(
        "--machine-output",
        type=Path,
        default=Path("evaluation_summary.json"),
        help="JSON destination (default: evaluation_summary.json)",
    )
    parser.add_argument(
        "--model-order",
        nargs="+",
        default=list(DEFAULT_MODEL_ORDER),
        help="preferred model row order; unlisted model directories are appended",
    )
    return parser.parse_args()


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def aggregate(scores: Sequence[Dict[str, Any]], category: str) -> Dict[str, Any]:
    sample_count = len(scores)
    success_count = sum(bool(score["success"]) for score in scores)
    result: Dict[str, Any] = {
        "sample_count": sample_count,
        "success_count": success_count,
        "success_rate": success_count / sample_count,
    }

    if category in ("turntaking", "interruptions"):
        latencies = [
            float(score["latency"])
            for score in scores
            if score.get("latency") is not None
        ]
        result["latency_count"] = len(latencies)
        result["mean_latency_seconds"] = mean(latencies)

    if category == "interruptions":
        ratings = [
            int(score["response_rating"])
            for score in scores
            if score.get("response_rating") is not None
        ]
        result["rating_count"] = len(ratings)
        result["mean_rating"] = mean(ratings)
        result["rating_distribution"] = {
            str(rating): ratings.count(rating) for rating in range(6)
        }

    if category == "backchanneling":
        jsds = [float(score["jsd"]) for score in scores]
        frequencies = [float(score["backchannel_frequency"]) for score in scores]
        result["mean_jsd"] = mean(jsds)
        result["mean_backchannel_frequency_per_second"] = mean(frequencies)

    return result


def ordered_models(output_dir: Path, preferred: Iterable[str]) -> List[str]:
    available = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
    preferred_existing = [model for model in preferred if model in available]
    preferred_set = set(preferred_existing)
    return preferred_existing + [model for model in available if model not in preferred_set]


def ordered_languages(input_dir: Path) -> List[str]:
    available = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
    preferred_existing = [
        language for language in DEFAULT_LANGUAGE_ORDER if language in available
    ]
    preferred_set = set(preferred_existing)
    return preferred_existing + [
        language for language in available if language not in preferred_set
    ]


def benchmark_samples(input_dir: Path, language: str, category: str) -> List[str]:
    category_dir = input_dir / language / category
    if not category_dir.is_dir():
        return []
    return sorted(
        sample_dir.name
        for sample_dir in category_dir.iterdir()
        if sample_dir.is_dir() and (sample_dir / "input.wav").is_file()
    )


def load_scores(
    input_dir: Path,
    output_dir: Path,
    language: str,
    model: str,
    category: str,
) -> List[Dict[str, Any]]:
    scores = []
    for sample in benchmark_samples(input_dir, language, category):
        sample_dir = output_dir / model / language / category / sample
        output_path = sample_dir / "output.wav"
        if not output_path.is_file():
            continue
        score_path = sample_dir / "score.json"
        if not score_path.is_file():
            raise FileNotFoundError(f"output has no score.json: {output_path}")
        try:
            with score_path.open("r", encoding="utf-8") as score_file:
                score = json.load(score_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read {score_path}: {error}") from error
        if "success" not in score:
            raise ValueError(f"score is missing 'success': {score_path}")
        if category in ("turntaking", "interruptions") and "latency" not in score:
            raise ValueError(f"score is missing 'latency': {score_path}")
        if category == "backchanneling":
            for field in ("jsd", "backchannel_frequency"):
                if field not in score:
                    raise ValueError(f"score is missing '{field}': {score_path}")
        if category == "interruptions":
            response_score_path = sample_dir / "response_score.json"
            if response_score_path.is_file():
                try:
                    with response_score_path.open(
                        "r", encoding="utf-8"
                    ) as response_score_file:
                        response_score = json.load(response_score_file)
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"could not read {response_score_path}: {error}"
                    ) from error
                if not isinstance(response_score, dict):
                    raise ValueError(
                        f"response score is not a JSON object: "
                        f"{response_score_path}"
                    )
                rating = response_score.get("rating")
                if (
                    not isinstance(rating, int)
                    or isinstance(rating, bool)
                    or not 0 <= rating <= 5
                ):
                    raise ValueError(
                        f"response score has invalid 'rating': "
                        f"{response_score_path}"
                    )
                score = dict(score)
                score["response_rating"] = rating
        scores.append(score)
    return scores


def build_summary(input_dir: Path, output_dir: Path, model_order: Sequence[str]) -> Dict[str, Any]:
    languages = ordered_languages(input_dir)
    models = ordered_models(output_dir, model_order)
    language_results: Dict[str, Any] = {}

    for language in languages:
        model_results: Dict[str, Any] = {}
        for model in models:
            category_results: Dict[str, Any] = {}
            for category in CATEGORIES:
                scores = load_scores(input_dir, output_dir, language, model, category)
                if scores:
                    category_results[category] = aggregate(scores, category)
            if category_results:
                model_results[model] = category_results
        language_results[language] = {"models": model_results}

    average_success_rates: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model in models:
        language_averages: Dict[str, Dict[str, Any]] = {}
        for language, language_data in language_results.items():
            model_data = language_data["models"].get(model)
            if not model_data:
                continue
            category_rates = [
                float(category_data["success_rate"])
                for category_data in model_data.values()
            ]
            language_averages[language] = {
                "category_count": len(category_rates),
                "success_rate": mean(category_rates),
            }
        if language_averages:
            average_success_rates[model] = language_averages

    return {
        "schema_version": SCHEMA_VERSION,
        "metric_notes": {
            "success_rate": "success_count divided by sample_count",
            "mean_latency_seconds": (
                "mean over non-null latency values; evaluators set latency to null "
                "when success is false"
            ),
            "mean_jsd": "mean Jensen-Shannon distance over all backchannel samples",
            "mean_backchannel_frequency_per_second": (
                "mean backchannel frequency over all backchannel samples"
            ),
            "mean_rating": (
                "mean 0-5 semantic interruption-response rating over samples "
                "with response_score.json"
            ),
            "rating_distribution": (
                "counts of semantic interruption-response ratings from 0 through 5"
            ),
        },
        "category_order": list(CATEGORIES),
        "model_order": models,
        "languages": language_results,
        "average_success_rates": {
            "definition": (
                "unweighted mean of available category success rates for each "
                "model and language"
            ),
            "models": average_success_rates,
        },
    }


def percent(value: Optional[float]) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def seconds(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}s"


def decimal(value: Optional[float], places: int) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def category_metric(model: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
    value = model.get(category)
    return value if isinstance(value, dict) else None


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Evaluation summary",
        "",
        (
            "Latency is averaged over successful samples only. Backchanneling JSD "
            "and Freq include all evaluated Backchanneling samples. UserInterruptions "
            "Rating is averaged over samples with response_score.json. An em dash "
            "means that the model has no outputs for that category."
        ),
        "",
    ]
    header = (
        f"| Model | {CATEGORY_DISPLAY_NAMES['pausehandling']} "
        f"{METRIC_DISPLAY_NAMES['success_rate']} | "
        f"{CATEGORY_DISPLAY_NAMES['turntaking']} "
        f"{METRIC_DISPLAY_NAMES['success_rate']} | "
        f"{CATEGORY_DISPLAY_NAMES['turntaking']} "
        f"{METRIC_DISPLAY_NAMES['mean_latency_seconds']} | "
        f"{CATEGORY_DISPLAY_NAMES['backchanneling']} "
        f"{METRIC_DISPLAY_NAMES['success_rate']} | "
        f"{CATEGORY_DISPLAY_NAMES['backchanneling']} "
        f"{METRIC_DISPLAY_NAMES['mean_jsd']} | "
        f"{CATEGORY_DISPLAY_NAMES['backchanneling']} "
        f"{METRIC_DISPLAY_NAMES['mean_backchannel_frequency_per_second']} | "
        f"{CATEGORY_DISPLAY_NAMES['interruptions']} "
        f"{METRIC_DISPLAY_NAMES['success_rate']} | "
        f"{CATEGORY_DISPLAY_NAMES['interruptions']} "
        f"{METRIC_DISPLAY_NAMES['mean_latency_seconds']} | "
        f"{CATEGORY_DISPLAY_NAMES['interruptions']} "
        f"{METRIC_DISPLAY_NAMES['mean_rating']} |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    for language, language_data in summary["languages"].items():
        lines.extend((f"## {language.title()}", "", header, divider))
        models = language_data["models"]
        for model in summary["model_order"]:
            if model not in models:
                continue
            model_data = models[model]
            pause = category_metric(model_data, "pausehandling")
            turn = category_metric(model_data, "turntaking")
            interruption = category_metric(model_data, "interruptions")
            backchannel = category_metric(model_data, "backchanneling")
            values = (
                MODEL_DISPLAY_NAMES.get(model, model),
                percent(pause["success_rate"]) if pause else "—",
                percent(turn["success_rate"]) if turn else "—",
                seconds(turn["mean_latency_seconds"]) if turn else "—",
                percent(backchannel["success_rate"]) if backchannel else "—",
                decimal(backchannel["mean_jsd"], 4) if backchannel else "—",
                (
                    decimal(backchannel["mean_backchannel_frequency_per_second"], 5)
                    if backchannel
                    else "—"
                ),
                percent(interruption["success_rate"]) if interruption else "—",
                seconds(interruption["mean_latency_seconds"]) if interruption else "—",
                decimal(interruption["mean_rating"], 3) if interruption else "—",
            )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    languages = list(summary["languages"])
    lines.extend(
        (
            "## Average SR by model and language",
            "",
            (
                "This is the unweighted mean of the available category SR values. "
                "The human reference uses three categories because it has no "
                "UserInterruptions outputs; models with all categories use all four."
            ),
            "",
            "| Model | " + " | ".join(language.title() for language in languages) + " |",
            "|---|" + "---:|" * len(languages),
        )
    )
    averages = summary["average_success_rates"]["models"]
    for model in summary["model_order"]:
        if model not in averages:
            continue
        cells = [
            percent(averages[model][language]["success_rate"])
            if language in averages[model]
            else "—"
            for language in languages
        ]
        lines.append(
            "| "
            + " | ".join((MODEL_DISPLAY_NAMES.get(model, model), *cells))
            + " |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(contents)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")
    if not args.output_dir.is_dir():
        raise SystemExit(f"output directory does not exist: {args.output_dir}")
    if args.human_output.resolve() == args.machine_output.resolve():
        raise SystemExit("--human-output and --machine-output must be different files")

    summary = build_summary(args.input_dir, args.output_dir, args.model_order)
    atomic_write_text(args.human_output, render_markdown(summary))
    atomic_write_text(
        args.machine_output,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    print(f"Wrote Markdown summary to {args.human_output}")
    print(f"Wrote JSON summary to {args.machine_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
