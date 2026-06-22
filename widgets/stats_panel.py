from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from tokenizer_engine import TokenStats, TokenizationResult


class StatsPanel(Static):
    def on_mount(self) -> None:
        self.update_stats(None)

    def update_stats(self, result: TokenizationResult | None) -> None:
        if result is None:
            self.update(Text("Load a tokenizer to see stats.", style="dim"))
            return
        self.update(self._table(result.stats))

    @staticmethod
    def _table(stats: TokenStats) -> Table:
        table = Table.grid(expand=True)
        table.add_column("metric", style="dim")
        table.add_column("value", justify="right", style="bold #f0f6fc")
        table.add_row("Vocab size", f"{stats.vocab_size:,}")
        table.add_row("Tokens", f"{stats.token_count:,}")
        table.add_row("Characters", f"{stats.character_count:,}")
        table.add_row("Chars/token", f"{stats.chars_per_token:.2f}")
        table.add_row("Compression", f"{stats.compression_ratio:.2f}")
        table.add_row("Unique IDs", f"{stats.unique_token_count:,}")
        table.add_row(
            "Most frequent ID",
            "n/a" if stats.most_frequent_token_id is None else str(stats.most_frequent_token_id),
        )
        table.add_row("Avg token length", f"{stats.avg_token_length:.2f}")
        return table
