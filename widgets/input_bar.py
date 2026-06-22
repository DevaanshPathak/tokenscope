from __future__ import annotations

from typing import Any

from textual.message import Message
from textual.widgets import Input


class DebouncedTextInput(Input):
    class Debounced(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(self, *args: Any, debounce_seconds: float = 0.08, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.debounce_seconds = debounce_seconds
        self._debounce_timer: Any | None = None

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is not self:
            return
        event.stop()
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        self._debounce_timer = self.set_timer(
            self.debounce_seconds,
            lambda: self.post_message(self.Debounced(self.value)),
        )

    def emit_now(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
            self._debounce_timer = None
        self.post_message(self.Debounced(self.value))
