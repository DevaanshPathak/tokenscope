from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, Label, LoadingIndicator, Static

from theme import APP_CSS
from tokenizer_engine import TokenizationResult, TokenizerEngine, TokenizerLoadError
from widgets.input_bar import DebouncedTextInput
from widgets.merge_tree import MergeTreeWidget
from widgets.stats_panel import StatsPanel
from widgets.token_view import TokenView


class TokenscopeApp(App[None]):
    CSS = APP_CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+l", "clear_input", "Clear input"),
        Binding("ctrl+o", "open_tokenizer", "Open tokenizer"),
        Binding("tab", "cycle_bottom_tab", "Cycle bottom tab"),
        Binding("ctrl+s", "save_export", "Save export"),
    ]

    def __init__(self, tokenizer_path: str | None = None) -> None:
        super().__init__()
        self.initial_tokenizer_path = tokenizer_path
        self.engine: TokenizerEngine | None = None
        self.current_result: TokenizationResult | None = None
        self._load_task: asyncio.Task[None] | None = None
        self._encode_task: asyncio.Task[None] | None = None
        self._encode_sequence = 0

    def compose(self) -> ComposeResult:
        yield Static("tokenscope v0.1 | no tokenizer loaded", id="header")

        with Container(id="path-screen"):
            yield Label("Load a local HuggingFace tokenizer", id="path-title")
            yield Static(
                "Enter a directory or tokenizer.json path. Runtime loading is local only.",
                id="path-help",
            )
            yield Input(placeholder=r"D:\models\gpt2-local or ./gpt2-local", id="path-input")
            yield Static("", id="path-status")
            yield LoadingIndicator(id="loading")

        with Vertical(id="main-layout"):
            yield DebouncedTextInput(
                placeholder="Type text here",
                id="text-input",
                debounce_seconds=0.08,
            )
            with Horizontal(id="content-row"):
                yield TokenView(id="token-view")
                yield StatsPanel(id="stats-panel")
            yield MergeTreeWidget(id="bottom-panel")

    def on_mount(self) -> None:
        self.query_one("#main-layout", Vertical).add_class("hidden")
        self.query_one("#loading", LoadingIndicator).add_class("hidden")
        if self.initial_tokenizer_path:
            self.query_one("#path-input", Input).value = self.initial_tokenizer_path
            self._request_tokenizer_load(self.initial_tokenizer_path)
        else:
            self.query_one("#path-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "path-input":
            return
        event.stop()
        self._request_tokenizer_load(event.value.strip())

    def on_debounced_text_input_debounced(self, event: DebouncedTextInput.Debounced) -> None:
        event.stop()
        self._schedule_encode(event.value)

    def _request_tokenizer_load(self, path_value: str) -> None:
        if not path_value:
            self.query_one("#path-status", Static).update("Enter a tokenizer path.")
            return
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
        self._show_loader(f"Loading {path_value}")
        self._load_task = asyncio.create_task(self._load_tokenizer(path_value))

    async def _load_tokenizer(self, path_value: str) -> None:
        try:
            engine = await asyncio.to_thread(TokenizerEngine.load, path_value)
        except asyncio.CancelledError:
            return
        except (TokenizerLoadError, OSError, ValueError) as exc:
            self._show_load_error(str(exc))
            return
        except Exception as exc:
            self._show_load_error(f"Unexpected tokenizer load error: {exc}")
            return

        self.engine = engine
        self.current_result = engine.encode("")
        self.query_one("#header", Static).update(engine.header_label)
        self.query_one("#path-screen", Container).add_class("hidden")
        self.query_one("#main-layout", Vertical).remove_class("hidden")
        self.query_one("#loading", LoadingIndicator).add_class("hidden")
        self.query_one("#path-status", Static).update("")
        self.query_one("#bottom-panel", MergeTreeWidget).set_engine(engine)
        self.query_one("#stats-panel", StatsPanel).update_stats(self.current_result)
        self.query_one("#token-view", TokenView).update_tokens(self.current_result)
        self.query_one("#text-input", DebouncedTextInput).focus()

    def _show_loader(self, message: str) -> None:
        self.query_one("#path-screen", Container).remove_class("hidden")
        self.query_one("#loading", LoadingIndicator).remove_class("hidden")
        self.query_one("#path-status", Static).update(message)

    def _show_load_error(self, message: str) -> None:
        self.query_one("#loading", LoadingIndicator).add_class("hidden")
        self.query_one("#path-screen", Container).remove_class("hidden")
        self.query_one("#path-status", Static).update(f"[red]Error:[/red] {message}")
        self.query_one("#path-input", Input).focus()

    def _schedule_encode(self, text: str) -> None:
        if self.engine is None:
            return
        self._encode_sequence += 1
        sequence = self._encode_sequence
        if self._encode_task and not self._encode_task.done():
            self._encode_task.cancel()
        self._encode_task = asyncio.create_task(self._encode_text(sequence, text))

    async def _encode_text(self, sequence: int, text: str) -> None:
        engine = self.engine
        if engine is None:
            return
        try:
            result = await asyncio.to_thread(engine.encode, text)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.notify(f"Tokenization failed: {exc}", severity="error")
            return
        if sequence != self._encode_sequence:
            return
        self.current_result = result
        self.query_one("#token-view", TokenView).update_tokens(result)
        self.query_one("#stats-panel", StatsPanel).update_stats(result)
        self.query_one("#bottom-panel", MergeTreeWidget).update_result(result)

    def action_clear_input(self) -> None:
        input_widget = self.query_one("#text-input", DebouncedTextInput)
        input_widget.value = ""
        input_widget.emit_now()

    def action_open_tokenizer(self) -> None:
        self.query_one("#path-screen", Container).remove_class("hidden")
        self.query_one("#main-layout", Vertical).add_class("hidden")
        self.query_one("#loading", LoadingIndicator).add_class("hidden")
        self.query_one("#path-status", Static).update("")
        self.query_one("#path-input", Input).focus()

    def action_cycle_bottom_tab(self) -> None:
        if self.query_one("#main-layout", Vertical).has_class("hidden"):
            return
        self.query_one("#bottom-panel", MergeTreeWidget).cycle_tab()

    def action_save_export(self) -> None:
        if self.current_result is None or self.engine is None:
            self.notify("Nothing to export yet.", severity="warning")
            return

        output = Path.cwd() / "tokenscope_export.json"
        payload = {
            "tokenizer": {
                "name": self.engine.name,
                "type": self.engine.tokenizer_type,
                "vocab_size": self.engine.vocab_size,
                "source_path": str(self.engine.source_path),
            },
            "tokenization": self.current_result.to_export_dict(),
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.notify(f"Saved {output.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline tokenizer explorer TUI.")
    parser.add_argument("--tokenizer", help="Local tokenizer directory or tokenizer.json path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TokenscopeApp(tokenizer_path=args.tokenizer).run()


if __name__ == "__main__":
    main()
