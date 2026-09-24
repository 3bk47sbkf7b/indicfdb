# Evaluation scripts

These scripts evaluate full-duplex conversational agents using VAD (vocal activity detection) heuristics that scale across languages for four categories: Pause Handling, Turn Taking, Backchanneling and User Interruptions. For the User Interruptions category, agent responses to interrupting questions are transcribed and translated to English and then rated by an LLM judge on relevance and quality.

## Directory layout

The scripts expect benchmark inputs and model outputs to use the following
layout:

```text
benchmark/
  <language>/
    <category>/
      <sample>/
        input.wav
        metadata.json

outputs/
  <model>/
    <language>/
      <category>/
        <sample>/
          output.wav
```

The supported category directory names are `pausehandling`, `turntaking`,
`backchanneling`, and `interruptions`. Evaluation and enrichment files are
written beside each sample's `output.wav`.

Most scripts operate on one model and language at a time. For those scripts,
`--output-dir` is the model directory, such as `outputs/my-model`, while
`--input-dir` is the benchmark root. `summarize_evaluations.py` is the
exception: its `--output-dir` is the root containing all model directories,
such as `outputs`.

## Recommended order

1. Run the four category evaluators. They are independent and may be run in
   any order:
   - `eval_pausehandling.py`
   - `eval_turntaking.py`
   - `eval_backchanneling.py`
   - `eval_interruptions.py`
2. After `eval_interruptions.py`, run the interruption semantic pipeline in
   this order:
   1. `extract_interruptions_responses.py`
   2. `transcribe_interruptions_responses.py`
   3. `translate_interruptions_responses.py`
   4. `extract_interruptions_inputs.py`
   5. `judge_interruptions_responses.py`
3. After all desired model/language evaluations are complete, run
   `summarize_evaluations.py` once across the complete output root.

The interruption semantic pipeline is optional if only timing and VAD-based
metrics are needed. Without it, the summary still includes interruption
success and latency, but semantic ratings are unavailable.

## Per-model, per-language example

The following example evaluates one model in one language and then runs the
complete interruption pipeline:

```bash
INPUT_ROOT=benchmark
MODEL_OUTPUT=outputs/my-model
LANGUAGE=english

python scripts/eval_pausehandling.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"

python scripts/eval_turntaking.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"

python scripts/eval_backchanneling.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"

python scripts/eval_interruptions.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"

python scripts/extract_interruptions_responses.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"

python scripts/transcribe_interruptions_responses.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"

python scripts/translate_interruptions_responses.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"

python scripts/extract_interruptions_inputs.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"

python scripts/judge_interruptions_responses.py \
  --input-dir "$INPUT_ROOT" --output-dir "$MODEL_OUTPUT" \
  --language "$LANGUAGE"
```

Repeat these commands for each model/language pair that has outputs. Each
script skips a missing category directory, so a model does not need to cover
every language or category.

## Script reference

### `eval_pausehandling.py`

Evaluates whether the agent incorrectly takes over during a user pause. It
uses Silero VAD, joins speech regions separated by less than 0.2 seconds, and
marks any resulting region longer than 2.0 seconds as a takeover.
A sample succeeds only when it has no takeover.

It writes
`score.json` containing `takeover`, the detected `takeovers`, and `success`.


### `eval_turntaking.py`

Evaluates whether the agent responds after the end of the user's turn without
taking over early. Speech before `input_turn_end` is checked using the
pause-handling thresholds; speech after the boundary is joined across gaps
shorter than 1.5 seconds, and regions at least 1.0 second long count as
responses. A sample succeeds only when it has a response and no early
takeover.

The generated `score.json` includes response and early-takeover intervals,
`success`, and response `latency`. Latency is measured from `input_turn_end`
and is `null` for unsuccessful samples.

### `eval_backchanneling.py`

Evaluates short agent acknowledgments or backchannels during the user's turn. Speech regions
separated by less than 0.5 seconds are joined. Regions no longer than 1.2
seconds are classified as backchannels; longer regions are takeovers. The
script compares predicted timing with the reference `human_backchannels` from
`metadata.json` using a FullDuplexBench-compatible Jensen-Shannon distance. The script measures
backchannel frequency (backchannels per second) using the number of backchannels in a sample and sample duration.
A sample succeeds only when it has no takeover.

The generated `score.json` includes detected backchannels and takeovers,
backchannel frequency, JSD, and `success`.

### `eval_interruptions.py`

Evaluates whether the agent responds after the user interrupts. Only VAD
regions starting strictly after `interrupt_start` are eligible. Regions are
joined across gaps shorter than 1.5 seconds, and merged regions at least 1.0
second long count as responses.
A sample succeeds only when it has a response.

The generated `score.json` includes `response`, response intervals, `success`,
and latency measured from `interrupt_end`. This file is the input to the
interruption response-extraction stage.

### `extract_interruptions_responses.py`

Reads interruption `score.json` files and skips samples where no response was
detected. For each response sample, it extracts audio from the earliest scored
response start through the end of `output.wav`, applies loudness
normalization, and writes `response.wav`.

The script uses `--workers` for parallel extraction and overwrites an existing
`response.wav` for processed samples.

### `transcribe_interruptions_responses.py`

Transcribes each `response.wav` with
[`bodhan-ai/indic-transcribe-core`](https://huggingface.co/bodhan-ai/indic-transcribe-core),
using its long-form helper for responses that need chunking. It writes an
indented `response_score.json` containing:

```json
{
  "transcription": "..."
}
```

Supported language directories are `english, hindi, bengali, punjabi, gujarati, marathi, kannada, 
telugu, tamil, malayalam`.

### `translate_interruptions_responses.py`

Adds `transcription_en` to each existing `response_score.json`. English
transcriptions are copied directly; the other supported languages are
translated to English with `google/gemma-4-31B-it`. Empty transcriptions remain
empty.

The script supports `--batch-size` and `--max-new-tokens`.

### `extract_interruptions_inputs.py`

Adds the user-side text needed by the semantic judge. It copies
`context_text` and `interrupt_text` from the native-language benchmark
`metadata.json` into `context` and `interrupt`, and copies the corresponding
English sample text into `context_en` and `interrupt_en`. For English samples,
the native and English fields are identical.

The correspondence between a non-English sample and its English version is
based on the shared sample directory name. Only samples with an existing
`response_score.json` are processed.

### `judge_interruptions_responses.py`

Applies the Full-Duplex-Bench user-interruption rubric to `context_en`,
`interrupt_en`, and `transcription_en` using `google/gemma-4-31B-it`. It adds:

- `rating`: an integer from 0 to 5 measuring how well the response addresses
  the user's interruption;
- `analysis`: the judge's brief rationale.

The script supports `--batch-size` and `--max-new-tokens`.

### `summarize_evaluations.py`

Aggregates every model and language under the output root. It reads category
`score.json` files and, when present, semantic ratings from interruption
`response_score.json` files. The benchmark tree is authoritative, so stale
output samples without a corresponding benchmark sample are ignored.

The Markdown summary reports:

- success rate for every category;
- mean latency for TurnTaking and UserInterruptions;
- mean JSD and backchannel frequency for Backchanneling;
- mean semantic rating for UserInterruptions.

The JSON summary additionally includes counts and the complete 0–5 rating
distribution. Both destinations are overwritten atomically on each run.

```bash
python scripts/summarize_evaluations.py \
  --input-dir benchmark \
  --output-dir outputs \
  --human-output summary/evaluation_summary.md \
  --machine-output summary/evaluation_summary.json
```

Run any script with `--help` for its complete command-line interface.
