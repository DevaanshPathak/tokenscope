from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import DataTable, Input, Static, TabPane, TabbedContent

from tokenizer_engine import TokenizationResult, TokenizerEngine


class MergeTreeWidget(Vertical):
    TAB_IDS = ("token-table-tab", "merge-tree-tab", "vocab-search-tab")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.engine: TokenizerEngine | None = None
        self.result: TokenizationResult | None = None
        self._active_index = 0

    def compose(self):
        with TabbedContent(initial=self.TAB_IDS[0], id="bottom-tabs"):
            with TabPane("Token ID Table", id=self.TAB_IDS[0]):
                yield DataTable(id="token-table")
            with TabPane("Merge Tree", id=self.TAB_IDS[1]):
                yield Static("Load a tokenizer to inspect BPE merges.", id="merge-tree")
            with TabPane("Vocab Search", id=self.TAB_IDS[2]):
                yield Input(placeholder="Search vocabulary substring", id="vocab-search")
                yield DataTable(id="vocab-table")

    def on_mount(self) -> None:
        token_table = self.query_one("#token-table", DataTable)
        token_table.cursor_type = "row"
        token_table.add_columns("index", "token_string", "token_id", "byte_repr")

        vocab_table = self.query_one("#vocab-table", DataTable)
        vocab_table.cursor_type = "row"
        vocab_table.add_columns("token_string", "token_id")

    def set_engine(self, engine: TokenizerEngine | None) -> None:
        self.engine = engine
        self.result = None
        self._update_vocab_table("")
        self.update_result(None)

    def update_result(self, result: TokenizationResult | None) -> None:
        self.result = result
        self._update_token_table()
        self._update_merge_tree()

    def cycle_tab(self) -> None:
        tabs = self.query_one("#bottom-tabs", TabbedContent)
        active = getattr(tabs, "active", self.TAB_IDS[self._active_index])
        try:
            self._active_index = self.TAB_IDS.index(active)
        except ValueError:
            self._active_index = 0
        self._active_index = (self._active_index + 1) % len(self.TAB_IDS)
        tabs.active = self.TAB_IDS[self._active_index]
        if self.TAB_IDS[self._active_index] == "vocab-search-tab":
            self.query_one("#vocab-search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "vocab-search":
            return
        event.stop()
        self._update_vocab_table(event.value)

    def _update_token_table(self) -> None:
        table = self.query_one("#token-table", DataTable)
        table.clear()
        if self.result is None:
            return
        for span in self.result.spans:
            table.add_row(
                str(span.index),
                span.token,
                str(span.token_id),
                span.byte_repr,
            )

    def _update_merge_tree(self) -> None:
        tree = self.query_one("#merge-tree", Static)
        if self.engine is None:
            tree.update("Load a tokenizer to inspect BPE merges.")
            return
        tree.update(self.engine.render_bpe_merge_tree(self.result))

    def _update_vocab_table(self, query: str) -> None:
        table = self.query_one("#vocab-table", DataTable)
        table.clear()
        if self.engine is None:
            return
        for token, token_id in self.engine.search_vocab(query):
            table.add_row(token, str(token_id))
