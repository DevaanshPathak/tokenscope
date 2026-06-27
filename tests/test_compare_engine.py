from __future__ import annotations

import unittest

from compare_engine import build_compare_result
from tokenizer_engine import TokenSpan, TokenStats, TokenizationResult


def make_result(text: str, spans: list[tuple[int, int, str, int]]) -> TokenizationResult:
    token_spans = tuple(
        TokenSpan(
            index=index,
            token=token,
            token_id=token_id,
            offset_start=start,
            offset_end=end,
            text=text[start:end],
            byte_repr="",
        )
        for index, (start, end, token, token_id) in enumerate(spans)
    )
    token_count = len(token_spans)
    stats = TokenStats(
        vocab_size=100,
        token_count=token_count,
        character_count=len(text),
        chars_per_token=(len(text) / token_count) if token_count else 0.0,
        compression_ratio=(token_count / len(text)) if text else 0.0,
        unique_token_count=len({span.token_id for span in token_spans}),
        most_frequent_token_id=token_spans[0].token_id if token_spans else None,
        avg_token_length=0.0,
    )
    return TokenizationResult(
        input_text=text,
        tokens=tuple(span.token for span in token_spans),
        token_ids=tuple(span.token_id for span in token_spans),
        spans=token_spans,
        stats=stats,
    )


class CompareEngineTests(unittest.TestCase):
    def test_identical_tokenizations_are_same(self) -> None:
        primary = make_result("abc", [(0, 1, "a", 1), (1, 3, "bc", 2)])
        compare = make_result("abc", [(0, 1, "a", 1), (1, 3, "bc", 2)])

        result = build_compare_result(primary, compare)

        self.assertEqual([row.status for row in result.rows], ["same", "same"])
        self.assertEqual(result.summary.matching_spans, 2)

    def test_same_boundaries_different_id(self) -> None:
        primary = make_result("abc", [(0, 3, "abc", 1)])
        compare = make_result("abc", [(0, 3, "abc", 9)])

        result = build_compare_result(primary, compare)

        self.assertEqual(result.rows[0].status, "id")
        self.assertEqual(result.summary.id_mismatches, 1)

    def test_same_boundaries_different_token(self) -> None:
        primary = make_result("abc", [(0, 3, "abc", 1)])
        compare = make_result("abc", [(0, 3, "ABC", 1)])

        result = build_compare_result(primary, compare)

        self.assertEqual(result.rows[0].status, "token")
        self.assertEqual(result.summary.token_mismatches, 1)

    def test_same_boundaries_different_token_and_id_counts_both(self) -> None:
        primary = make_result("abc", [(0, 3, "abc", 1)])
        compare = make_result("abc", [(0, 3, "ABC", 9)])

        result = build_compare_result(primary, compare)

        self.assertEqual(result.rows[0].status, "token")
        self.assertEqual(result.summary.token_mismatches, 1)
        self.assertEqual(result.summary.id_mismatches, 1)

    def test_boundary_mismatch_splits_rows(self) -> None:
        primary = make_result("ab", [(0, 2, "ab", 1)])
        compare = make_result("ab", [(0, 1, "a", 2), (1, 2, "b", 3)])

        result = build_compare_result(primary, compare)

        self.assertEqual([row.status for row in result.rows], ["boundary", "boundary"])
        self.assertEqual(result.summary.boundary_mismatches, 2)

    def test_missing_side_is_reported(self) -> None:
        primary = make_result("a", [(0, 1, "a", 1)])
        compare = make_result("a", [])

        result = build_compare_result(primary, compare)

        self.assertEqual(result.rows[0].status, "missing")
        self.assertEqual(result.summary.missing_ranges, 1)

    def test_empty_input_has_no_rows(self) -> None:
        primary = make_result("", [])
        compare = make_result("", [])

        result = build_compare_result(primary, compare)

        self.assertEqual(result.rows, ())
        self.assertEqual(result.summary.primary_token_count, 0)
        self.assertEqual(result.summary.compare_token_count, 0)


if __name__ == "__main__":
    unittest.main()
