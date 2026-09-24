# IndicFDB

**IndicFDB: Benchmarking Full-Duplex Voice Agents across Indian Languages**

Project page: https://3bk47sbkf7b.github.io/indicfdb/
Repository: https://github.com/3bk47sbkf7b/indicfdb

IndicFDB evaluates pause handling, turn taking, backchanneling, and user
interruption behavior across ten languages spoken in India. The benchmark has
12,350 samples, and the project dashboard presents selected model outputs and
judgments.

## Project contents

- [`index.html`](index.html): GitHub Pages landing page.
- [`dashboard/`](dashboard/): portable model-output explorer and audio examples.
- [`scripts/`](scripts/): evaluation scripts and [usage guide](scripts/README.md).
- [`mining/`](mining/): VAD and corpus-mining scripts with [instructions](mining/README.txt).
- [`paper.pdf`](paper.pdf): paper draft.

The full benchmark and all model-output trees are not included in this project
page repository. The dashboard contains only its selected example audio and
per-case metadata. The scripts expect benchmark and output directories in the
layout documented in `scripts/README.md`.

## GitHub Pages

This repository is ready to serve from the root of the `main` branch. Enable
GitHub Pages in repository settings with **Deploy from a branch** → **main** →
**/(root)**. The root `.nojekyll` file allows all static files to be served as
written. Open `index.html` locally to review the landing page before release.
