# Model output explorer

Open `index.html` in a browser. The dashboard is self-contained: it embeds its
case data and uses the MP3 files under `audio/`. It does not need a web server
or access to the original `benchmark/` and `outputs/` trees.

Navigation follows category → model → case. Each case has aligned user input,
human reference (where available), and agent output tracks. Use **Play all from start** to rewind and play every track together, or use an
individual audio control to listen to one track. Markers show reference events and detected agent events.
Interruption cases also show the source text, transcription, English
translation, rating, and judge analysis when available. The expandable section
shows the source metadata and scores.

Selections are made across all languages available for each model. Within a
selection slot, a sample is used only once. Extreme latency, JSD, pause count,
or rating determines the primary order. Languages are used as tie breakers and
to diversify equal-metric examples. For interruption ratings, “high” selects
the highest available scores and “low” selects the lowest available scores.
The sidebar reports a shortfall when fewer than two matching scored samples
exist. `selection_manifest.json` records every selected case and each slot's
available count.

This published dashboard is a static selection. Regeneration requires the full
benchmark and model-output trees used to build it.
