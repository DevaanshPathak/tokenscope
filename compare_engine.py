from __future__ import annotations

from dataclasses import asdict, dataclass

from tokenizer_engine import TokenSpan, TokenizationResult


@dataclass(frozen=True)
class CompareDiffRow:
    offset_start: int
    offset_end: int
    text: str
    primary_token: str | None
    primary_token_id: int | None
    compare_token: str | None
    compare_token_id: int | None
    status: str


@dataclass(frozen=True)
class CompareSummary:
    primary_token_count: int
    compare_token_count: int
    token_count_delta: int
    primary_chars_per_token: float
    compare_chars_per_token: float
    chars_per_token_delta: float
    primary_compression: float
    compare_compression: float
    compression_delta: float
    matching_spans: int
    boundary_mismatches: int
    token_mismatches: int
    id_mismatches: int
    missing_ranges: int


@dataclass(frozen=True)
class CompareResult:
    primary: TokenizationResult
    compare: TokenizationResult
    rows: tuple[CompareDiffRow, ...]
    summary: CompareSummary

    def to_export_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary.to_export_dict(),
            "compare": self.compare.to_export_dict(),
            "summary": asdict(self.summary),
            "rows": [asdict(row) for row in self.rows],
        }


def build_compare_result(
    primary: TokenizationResult,
    compare: TokenizationResult,
) -> CompareResult:
    if primary.input_text != compare.input_text:
        raise ValueError("Cannot compare tokenizations for different input text.")

    rows = tuple(_build_rows(primary, compare))
    counts = _count_rows(rows)
    summary = CompareSummary(
        primary_token_count=primary.stats.token_count,
        compare_token_count=compare.stats.token_count,
        token_count_delta=compare.stats.token_count - primary.stats.token_count,
        primary_chars_per_token=primary.stats.chars_per_token,
        compare_chars_per_token=compare.stats.chars_per_token,
        chars_per_token_delta=compare.stats.chars_per_token - primary.stats.chars_per_token,
        primary_compression=primary.stats.compression_ratio,
        compare_compression=compare.stats.compression_ratio,
        compression_delta=compare.stats.compression_ratio - primary.stats.compression_ratio,
        matching_spans=counts["same"],
        boundary_mismatches=counts["boundary"],
        token_mismatches=counts["token"],
        id_mismatches=counts["id"],
        missing_ranges=counts["missing"],
    )
    return CompareResult(primary=primary, compare=compare, rows=rows, summary=summary)


def _build_rows(
    primary: TokenizationResult,
    compare: TokenizationResult,
) -> list[CompareDiffRow]:
    boundaries = _collect_boundaries(primary, compare)
    if len(boundaries) < 2:
        return []

    rows: list[CompareDiffRow] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if start == end:
            continue
        primary_span = _find_covering_span(primary.spans, start, end)
        compare_span = _find_covering_span(compare.spans, start, end)
        if primary_span is None and compare_span is None:
            continue
        rows.append(
            CompareDiffRow(
                offset_start=start,
                offset_end=end,
                text=primary.input_text[start:end],
                primary_token=primary_span.token if primary_span else None,
                primary_token_id=primary_span.token_id if primary_span else None,
                compare_token=compare_span.token if compare_span else None,
                compare_token_id=compare_span.token_id if compare_span else None,
                status=_status_for(primary_span, compare_span),
            )
        )
    return rows


def _collect_boundaries(primary: TokenizationResult, compare: TokenizationResult) -> list[int]:
    text_length = len(primary.input_text)
    boundaries = {0, text_length}
    for result in (primary, compare):
        for span in result.spans:
            if 0 <= span.offset_start <= text_length:
                boundaries.add(span.offset_start)
            if 0 <= span.offset_end <= text_length:
                boundaries.add(span.offset_end)
    return sorted(boundaries)


def _find_covering_span(
    spans: tuple[TokenSpan, ...],
    start: int,
    end: int,
) -> TokenSpan | None:
    for span in spans:
        if span.offset_start <= start and span.offset_end >= end:
            return span
    return None


def _status_for(primary: TokenSpan | None, compare: TokenSpan | None) -> str:
    if primary is None or compare is None:
        return "missing"
    if (primary.offset_start, primary.offset_end) != (compare.offset_start, compare.offset_end):
        return "boundary"
    if primary.token != compare.token:
        return "token"
    if primary.token_id != compare.token_id:
        return "id"
    return "same"


def _count_rows(rows: tuple[CompareDiffRow, ...]) -> dict[str, int]:
    counts = {"same": 0, "id": 0, "token": 0, "boundary": 0, "missing": 0}
    for row in rows:
        if row.status == "same":
            counts["same"] += 1
        elif row.status == "boundary":
            counts["boundary"] += 1
        elif row.status == "missing":
            counts["missing"] += 1
        else:
            if row.primary_token != row.compare_token:
                counts["token"] += 1
            if row.primary_token_id != row.compare_token_id:
                counts["id"] += 1
    return counts
