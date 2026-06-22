from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer, ByteLevelBPETokenizer


class TokenizerLoadError(ValueError):
    """Raised when a local tokenizer path cannot be loaded."""


@dataclass(frozen=True)
class TokenSpan:
    index: int
    token: str
    token_id: int
    offset_start: int
    offset_end: int
    text: str
    byte_repr: str


@dataclass(frozen=True)
class TokenStats:
    vocab_size: int
    token_count: int
    character_count: int
    chars_per_token: float
    compression_ratio: float
    unique_token_count: int
    most_frequent_token_id: int | None
    avg_token_length: float


@dataclass(frozen=True)
class TokenizationResult:
    input_text: str
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    spans: tuple[TokenSpan, ...]
    stats: TokenStats

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "tokens": list(self.tokens),
            "token_ids": list(self.token_ids),
            "spans": [asdict(span) for span in self.spans],
            "stats": asdict(self.stats),
        }


@dataclass
class MergeNode:
    text: str
    rank: int | None = None
    left: "MergeNode | None" = None
    right: "MergeNode | None" = None


class TokenizerEngine:
    def __init__(
        self,
        tokenizer: Tokenizer,
        source_path: Path,
        name: str,
        tokenizer_type: str,
        raw_config: dict[str, Any] | None = None,
        merge_ranks: dict[tuple[str, str], int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.source_path = source_path
        self.name = name
        self.tokenizer_type = tokenizer_type
        self.raw_config = raw_config or {}
        self.vocab = tokenizer.get_vocab()
        self.vocab_size = tokenizer.get_vocab_size()
        self.merge_ranks = merge_ranks or {}
        self._vocab_by_id = sorted(self.vocab.items(), key=lambda item: item[1])

    @classmethod
    def load(cls, path_value: str | Path) -> "TokenizerEngine":
        source = Path(path_value).expanduser()
        if not source.exists():
            raise TokenizerLoadError(f"Path does not exist: {source}")

        if source.is_file():
            tokenizer_file = source
            root = source.parent
        else:
            root = source
            tokenizer_file = cls._find_tokenizer_json(root)

        config = cls._read_optional_json(root / "tokenizer_config.json")

        if tokenizer_file is None and config:
            config_file = config.get("tokenizer_file")
            if isinstance(config_file, str):
                candidate = (root / config_file).resolve()
                if candidate.exists():
                    tokenizer_file = candidate

        raw_tokenizer = cls._read_optional_json(tokenizer_file) if tokenizer_file else None
        tokenizer = cls._load_tokenizer(root, tokenizer_file)
        tokenizer_type = cls._detect_type(tokenizer, raw_tokenizer)
        name = cls._detect_name(root, config, raw_tokenizer)
        merge_ranks = cls._extract_merge_ranks(raw_tokenizer, root)

        return cls(
            tokenizer=tokenizer,
            source_path=root.resolve(),
            name=name,
            tokenizer_type=tokenizer_type,
            raw_config=raw_tokenizer,
            merge_ranks=merge_ranks,
        )

    @staticmethod
    def _find_tokenizer_json(root: Path) -> Path | None:
        direct = root / "tokenizer.json"
        if direct.exists():
            return direct
        matches = sorted(root.glob("**/tokenizer.json"))
        return matches[0] if matches else None

    @staticmethod
    def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TokenizerLoadError(f"Invalid JSON in {path}: {exc}") from exc

    @staticmethod
    def _load_tokenizer(root: Path, tokenizer_file: Path | None) -> Tokenizer:
        if tokenizer_file is not None:
            try:
                return Tokenizer.from_file(str(tokenizer_file))
            except Exception as exc:  # tokenizers raises several Rust-backed exceptions.
                raise TokenizerLoadError(f"Could not load tokenizer file {tokenizer_file}: {exc}") from exc

        vocab_txt = root / "vocab.txt"
        if vocab_txt.exists():
            return BertWordPieceTokenizer(str(vocab_txt), lowercase=False)

        vocab_json = root / "vocab.json"
        merges_txt = root / "merges.txt"
        if vocab_json.exists() and merges_txt.exists():
            return ByteLevelBPETokenizer(str(vocab_json), str(merges_txt))

        config = root / "tokenizer_config.json"
        if config.exists():
            raise TokenizerLoadError(
                "Found tokenizer_config.json but no loadable tokenizer.json, "
                "tokenizer_file entry, vocab.txt, or vocab.json/merges.txt pair."
            )

        raise TokenizerLoadError(
            "Missing tokenizer files. Expected tokenizer.json, tokenizer_config.json, "
            "vocab.txt, or vocab.json plus merges.txt."
        )

    @staticmethod
    def _detect_type(tokenizer: Tokenizer, raw_tokenizer: dict[str, Any] | None) -> str:
        model = (raw_tokenizer or {}).get("model", {})
        model_type = model.get("type")
        if isinstance(model_type, str) and model_type:
            return model_type
        return type(tokenizer.model).__name__

    @staticmethod
    def _detect_name(
        root: Path,
        config: dict[str, Any] | None,
        raw_tokenizer: dict[str, Any] | None,
    ) -> str:
        for source in (config, raw_tokenizer):
            if not source:
                continue
            for key in ("name_or_path", "tokenizer_class", "name"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return Path(value).name
        return root.name or "local-tokenizer"

    @classmethod
    def _extract_merge_ranks(
        cls,
        raw_tokenizer: dict[str, Any] | None,
        root: Path,
    ) -> dict[tuple[str, str], int]:
        merges: Iterable[Any] | None = None
        model = (raw_tokenizer or {}).get("model", {})
        if model.get("type") == "BPE":
            merges = model.get("merges")

        if merges is None and (root / "merges.txt").exists():
            lines = (root / "merges.txt").read_text(encoding="utf-8").splitlines()
            merges = [line for line in lines if line and not line.startswith("#")]

        ranks: dict[tuple[str, str], int] = {}
        if not merges:
            return ranks

        for rank, item in enumerate(merges):
            pair: tuple[str, str] | None = None
            if isinstance(item, str):
                parts = item.split()
                if len(parts) >= 2:
                    pair = (parts[0], parts[1])
            elif isinstance(item, Sequence) and len(item) >= 2:
                pair = (str(item[0]), str(item[1]))
            if pair is not None and pair not in ranks:
                ranks[pair] = rank
        return ranks

    @property
    def is_bpe(self) -> bool:
        return self.tokenizer_type.lower() == "bpe" or bool(self.merge_ranks)

    @property
    def header_label(self) -> str:
        return (
            f"tokenscope v0.1 | {self.name} | "
            f"{self.tokenizer_type} | vocab: {self.vocab_size}"
        )

    def encode(self, text: str) -> TokenizationResult:
        if not text:
            return self._empty_result()

        encoding = self.tokenizer.encode(text)
        tokens = tuple(encoding.tokens)
        ids = tuple(int(token_id) for token_id in encoding.ids)
        offsets = tuple(encoding.offsets)

        spans: list[TokenSpan] = []
        for index, (token, token_id) in enumerate(zip(tokens, ids, strict=True)):
            start, end = offsets[index] if index < len(offsets) else (0, 0)
            span_text = self._safe_slice(text, start, end) or token
            spans.append(
                TokenSpan(
                    index=index,
                    token=token,
                    token_id=token_id,
                    offset_start=start,
                    offset_end=end,
                    text=span_text,
                    byte_repr=self._byte_repr(span_text),
                )
            )

        return TokenizationResult(
            input_text=text,
            tokens=tokens,
            token_ids=ids,
            spans=tuple(spans),
            stats=self._stats(text, tokens, ids),
        )

    def _empty_result(self) -> TokenizationResult:
        return TokenizationResult(
            input_text="",
            tokens=(),
            token_ids=(),
            spans=(),
            stats=TokenStats(
                vocab_size=self.vocab_size,
                token_count=0,
                character_count=0,
                chars_per_token=0.0,
                compression_ratio=0.0,
                unique_token_count=0,
                most_frequent_token_id=None,
                avg_token_length=0.0,
            ),
        )

    @staticmethod
    def _safe_slice(text: str, start: int, end: int) -> str:
        if start < 0 or end < start:
            return ""
        try:
            return text[start:end]
        except IndexError:
            return ""

    @staticmethod
    def _byte_repr(text: str) -> str:
        return " ".join(f"{byte:02x}" for byte in text.encode("utf-8", errors="replace"))

    def _stats(self, text: str, tokens: Sequence[str], ids: Sequence[int]) -> TokenStats:
        token_count = len(ids)
        character_count = len(text)
        token_lengths = np.array([len(token) for token in tokens], dtype=np.float64)
        counts = Counter(ids)
        most_frequent = counts.most_common(1)[0][0] if counts else None
        return TokenStats(
            vocab_size=self.vocab_size,
            token_count=token_count,
            character_count=character_count,
            chars_per_token=(character_count / token_count) if token_count else 0.0,
            compression_ratio=(token_count / character_count) if character_count else 0.0,
            unique_token_count=len(counts),
            most_frequent_token_id=most_frequent,
            avg_token_length=float(np.mean(token_lengths)) if token_lengths.size else 0.0,
        )

    def search_vocab(self, query: str, limit: int = 200) -> list[tuple[str, int]]:
        needle = query.casefold()
        if not needle:
            return self._vocab_by_id[: min(limit, len(self._vocab_by_id))]
        matches: list[tuple[str, int]] = []
        for token, token_id in self._vocab_by_id:
            if needle in token.casefold():
                matches.append((token, token_id))
                if len(matches) >= limit:
                    break
        return matches

    def render_bpe_merge_tree(self, result: TokenizationResult | None, max_tokens: int = 24) -> str:
        if not self.is_bpe:
            return "Merge tree only available for BPE tokenizers."
        if not result or not result.spans:
            return "Type text to inspect BPE merges."

        lines: list[str] = []
        for span in result.spans[:max_tokens]:
            lines.append(f"[{span.index}] {span.token!r} id={span.token_id}")
            node, exact = self._build_bpe_tree(span.token)
            if node is None:
                lines.append("  no merge history found")
                continue
            self._render_node(node, lines, prefix="  ")
            if not exact:
                lines.append("  note: token contains segments not present in merge ranks")

        if len(result.spans) > max_tokens:
            lines.append(f"... {len(result.spans) - max_tokens} more tokens omitted")
        return "\n".join(lines)

    def _build_bpe_tree(self, token: str) -> tuple[MergeNode | None, bool]:
        if not token:
            return None, False

        nodes = [MergeNode(symbol) for symbol in self._initial_symbols(token)]
        if len(nodes) == 1:
            return nodes[0], True

        exact = True
        while len(nodes) > 1:
            best_index: int | None = None
            best_rank: int | None = None
            for index in range(len(nodes) - 1):
                rank = self.merge_ranks.get((nodes[index].text, nodes[index + 1].text))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index

            if best_index is None:
                exact = False
                left = nodes.pop(0)
                right = nodes.pop(0)
                nodes.insert(0, MergeNode(left.text + right.text, None, left, right))
                continue

            left = nodes[best_index]
            right = nodes[best_index + 1]
            nodes[best_index : best_index + 2] = [
                MergeNode(left.text + right.text, best_rank, left, right)
            ]

        return nodes[0], exact

    @staticmethod
    def _initial_symbols(token: str) -> list[str]:
        if token.startswith("<") and token.endswith(">"):
            return [token]
        return list(token)

    @classmethod
    def _render_node(cls, node: MergeNode, lines: list[str], prefix: str) -> None:
        rank = f" rank={node.rank}" if node.rank is not None else ""
        lines.append(f"{prefix}+- {node.text!r}{rank}")
        if node.left is not None:
            cls._render_node(node.left, lines, prefix + "|  ")
        if node.right is not None:
            cls._render_node(node.right, lines, prefix + "   ")
