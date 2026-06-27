from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from compare_engine import CompareResult
from tokenizer_engine import TokenStats, TokenizationResult


class StatsPanel(Static):
    def on_mount(self) -> None:
        self.update_stats(None)

    def update_stats(self, result: TokenizationResult | None) -> None:
        if result is None:
            self.update(Text("Load a tokenizer to see stats.", style="dim"))
            return
        self.update(self._table(result.stats))

    def update_comparison(self, comparison: CompareResult | None) -> None:
        if comparison is None:
            self.update(Text("Load two tokenizers to compare stats.", style="dim"))
            return

        summary = comparison.summary
        table = Table.grid(expand=True)
        table.add_column("metric", style="dim")
        table.add_column("primary", justify="right", style="bold #f0f6fc")
        table.add_column("compare", justify="right", style="bold #f2cc60")
        table.add_row(
            "Tokens",
            f"{summary.primary_token_count:,}",
            f"{summary.compare_token_count:,} ({summary.token_count_delta:+,})",
        )
        table.add_row(
            "Chars/token",
            f"{summary.primary_chars_per_token:.2f}",
            f"{summary.compare_chars_per_token:.2f} ({summary.chars_per_token_delta:+.2f})",
        )
        table.add_row(
            "Compression",
            f"{summary.primary_compression:.2f}",
            f"{summary.compare_compression:.2f} ({summary.compression_delta:+.2f})",
        )
        table.add_row("Matching spans", str(summary.matching_spans), "")
        table.add_row("Boundary mismatches", str(summary.boundary_mismatches), "")
        table.add_row("Token mismatches", str(summary.token_mismatches), "")
        table.add_row("ID mismatches", str(summary.id_mismatches), "")
        table.add_row("Missing ranges", str(summary.missing_ranges), "")
        self.update(table)

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
