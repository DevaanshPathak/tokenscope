# tokenscope

`tokenscope` is an offline terminal tokenizer explorer for local HuggingFace tokenizer files. It loads a tokenizer from disk, lets you type text interactively, and shows colored token spans, token IDs, vocabulary stats, a token ID table, BPE merge reconstruction, and vocabulary search.

No runtime network calls are made. The app uses HuggingFace `tokenizers` directly and does not depend on `transformers`.

## Install

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Load a tokenizer at launch:

```bash
python main.py --tokenizer ./gpt2-local
```

You can pass either a tokenizer directory or a direct `tokenizer.json` path:

```bash
python main.py --tokenizer ./gpt2-local/tokenizer.json
```

If `--tokenizer` is omitted, `tokenscope` opens a path prompt inside the TUI.

Supported local layouts include:

- `tokenizer.json`
- `tokenizer_config.json` with a local `tokenizer_file`
- `vocab.txt` for WordPiece tokenizers
- `vocab.json` plus `merges.txt` for BPE tokenizers

## Keyboard Shortcuts

- `Ctrl+L`: clear input
- `Ctrl+O`: open a new tokenizer path
- `Tab`: cycle bottom panel tabs
- `Ctrl+S`: save current tokenization to `tokenscope_export.json`
- `Ctrl+C` or `q`: quit

## Panels

- Token view: colored token spans, visible token boundaries using `·`, and aligned token IDs
- Stats panel: vocab size, token count, character count, chars per token, compression ratio, unique token count, most frequent token ID, and average token length
- Token ID table: index, token string, token ID, and UTF-8 byte representation
- Merge tree: BPE-only best-effort ASCII merge tree reconstruction from merge ranks
- Vocab search: substring search over tokenizer vocabulary

## Build

The Linux binary build uses PyInstaller:

```bash
chmod +x build.sh
./build.sh
```

The output binary is written to:

```text
dist/tokenscope
```

Build the binary on Linux x86_64 when you need a Linux x86_64 artifact.
