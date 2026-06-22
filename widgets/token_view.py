from __future__ import annotations

from rich.cells import cell_len, set_cell_size
from rich.console import Group
from rich.text import Text
from textual.widgets import Static

from theme import TOKEN_COLORS
from tokenizer_engine import TokenSpan, TokenizationResult


class TokenView(Static):
    def on_mount(self) -> None:
        self.show_empty()

    def show_empty(self) -> None:
        message = Text("Type text to see colored token spans.", style="dim")
        self.update(message)

    def update_tokens(self, result: TokenizationResult | None) -> None:
        if result is None or not result.spans:
            self.show_empty()
            return

        colored = Text()
        boundary = Text("boundaries ", style="dim")
        ids = Text("token ids  ", style="dim")

        cursor = 0
        for index, span in enumerate(result.spans):
            style = f"bold {TOKEN_COLORS[index % len(TOKEN_COLORS)]}"
            if span.offset_start > cursor:
                colored.append(self._visible(result.input_text[cursor : span.offset_start]), style="dim")

            display_text = self._display_text(span)
            colored.append(self._visible(display_text), style=style)
            cursor = max(cursor, span.offset_end)

            width = max(cell_len(self._visible(display_text)), cell_len(str(span.token_id)), 1)
            if index:
                boundary.append("·", style="dim")
                ids.append("·", style="dim")
            boundary.append(set_cell_size(self._visible(display_text), width), style=style)
            ids.append(set_cell_size(str(span.token_id), width), style=style)

        if cursor < len(result.input_text):
            colored.append(self._visible(result.input_text[cursor:]), style="dim")

        self.update(Group(colored, Text(), boundary, ids))

    @staticmethod
    def _display_text(span: TokenSpan) -> str:
        return span.text or span.token

    @staticmethod
    def _visible(value: str) -> str:
        return value.replace("\n", "\\n").replace("\t", "\\t")
