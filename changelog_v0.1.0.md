# tokenscope v0.1.0

Initial public release of `tokenscope`, an offline terminal tokenizer explorer for local HuggingFace tokenizer files.

## Highlights

- Added local tokenizer loading from folders, `tokenizer.json`, WordPiece vocab files, and BPE vocab/merge files.
- Added interactive token spans, token IDs, offsets, bytes, statistics, selected-token inspection, decode round-trip checks, special-token views, prompt budgets, and BPE merge-tree inspection.
- Added side-by-side compare mode with aligned diff rows, compare metrics, corpus comparison, and tokenizer metadata inspection.
- Added corpus and batch prompt analyzers for local `.txt`, `.md`, `.jsonl`, `.json`, and `.csv` inputs.
- Added project save/load, tokenizer diffing, prompt packing simulation, regression suites, Unicode inspection, RAG chunk analysis, token-count distributions, cost estimates, and tokenizer repair suggestions.
- Added JSON, CSV, Markdown, and HTML exports.
- Added headless `analyze` mode for scripts and CI checks.
- Added PyInstaller packaging for Windows and macOS hosts, plus Docker-based Linux binary builds.

## Release Assets

- `tokenscope-linux-x86_64`
- `tokenscope-windows-x86_64.exe`

macOS binaries are supported through the macOS build workflow or a macOS host build, but are not part of the manually attached v0.1.0 assets.

## Validation

- `python -m compileall .`
- `python -m unittest discover -v`
- Windows binary `--help` and headless `analyze` smoke test.
- Linux binary `--help` and headless `analyze` smoke test in Docker.
