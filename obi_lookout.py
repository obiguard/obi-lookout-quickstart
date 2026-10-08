"""obi-lookout in one file: load the model, find sensitive spans, redact them. Works on text of any length.

    from obi_lookout import ObiLookout
    lookout = ObiLookout()
    for span in lookout.detect("Hubungi Aisha binti Rahman di aisha@example.test"):
        print(span.label, span.text, span.score)

Command line:  python obi_lookout.py --redact label < samples/malay.txt

Why this file is more than a call to the model: the model reads a few hundred words at a time. On a 33,000-character text one call took 351 seconds and
found 52% of the values, while 800-character windows took 54 seconds and found 91 to 93%. So long text is cut into overlapping windows here, and a
value cut in half by a window edge is distrusted. The window and tie-break logic is copied from Obiguard's serving code (obiguard-models, serve/chunking.py
and eval/baselines/run.py), which is what produced the numbers on the model card.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

MODEL = "obiguard/obi-lookout"
# Pinned so this guide cannot break when the model changes. Use "main" to follow the latest release.
REVISION = "bb0429a83618afa64e8fc0f99a98809398ed62fc"  # v0.6.0 (earlier: v0.4.0 = 92b43ba0, v0.1.0 = 90a4843f; tags "v0.4.0", "v0.1.0")

# The labels the model was trained and scored with (v0.4 no longer lists `medical_or_sensitive_category`, which v0.1 named but never had data for).
LABELS = (
    "person", "id_number", "phone", "email", "address", "date_of_birth", "financial_account", "organisation",
    "secret", "internal_host", "financial_figure", "client_or_project_name",
)
# When two labels claim the same text with the same confidence, the earlier one here wins (same rule as the scoring on the model card).
PRIORITY = (
    "secret", "internal_host", "id_number", "email", "phone", "date_of_birth", "address",
    "financial_account", "financial_figure", "client_or_project_name", "organisation", "person",
)
_RANK = {label: i for i, label in enumerate(PRIORITY)}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    score: float
    text: str


def windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) windows of at most `size` characters, each starting `overlap` before the previous one ends, cut at whitespace where possible."""
    n = len(text)
    if n <= size:
        return [(0, n)]

    def snap_back(end: int, floor: int) -> int:
        i = end
        while i > floor and not text[i - 1].isspace():
            i -= 1
        return i if i > floor else end

    def snap_forward(pos: int, limit: int) -> int:
        i = pos
        while 0 < i < limit and not text[i - 1].isspace():
            i += 1
        return i

    out: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + size, n)
        if end < n:
            end = snap_back(end, start + size // 2)
        out.append((start, end))
        if end >= n:
            return out
        nxt = snap_forward(max(end - overlap, start + 1), end)
        start = nxt if nxt > start else end


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Where spans overlap keep the most confident (ties: label order above, then the longer span). The result never overlaps."""
    ordered = sorted(spans, key=lambda s: (-s.score, _RANK.get(s.label, len(_RANK)), -(s.end - s.start), s.start))
    kept: list[Span] = []
    for span in ordered:
        if all(span.end <= k.start or span.start >= k.end for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def redact(text: str, spans: list[Span], style: str = "label") -> str:
    """`label` turns a span into [EMAIL]; `mask` into asterisks of the same length."""
    if style not in ("label", "mask"):
        raise ValueError("style must be 'label' or 'mask'")
    out, cursor = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        if s.start < cursor:
            continue
        out.append(text[cursor:s.start])
        out.append(f"[{s.label.upper()}]" if style == "label" else "*" * (s.end - s.start))
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)


class ObiLookout:
    def __init__(self, model=None, threshold: float = 0.5, size: int = 800, overlap: int = 300, margin: int = 100,
                 repo: str = MODEL, revision: str = REVISION):
        """`model` is only for tests (anything with extract_entities). Normally the weights are downloaded from Hugging Face once (about 1.2 GB) and cached."""
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("size must be positive and overlap smaller than size")
        if overlap < 2 * margin:
            raise ValueError("overlap must be at least twice the margin, or a value at a window edge could be seen from the inside by no window")
        self.threshold, self.size, self.overlap, self.margin = threshold, size, overlap, margin
        if model is None:
            from gliner2 import GLiNER2
            from huggingface_hub import snapshot_download

            model = GLiNER2.from_pretrained(snapshot_download(repo, revision=revision))
        self.model = model

    def _window(self, text: str) -> list[Span]:
        result = self.model.extract_entities(text, list(LABELS), threshold=self.threshold, include_confidence=True, include_spans=True)
        spans = []
        for label, items in (result.get("entities", result) or {}).items():
            for item in items or []:
                # The model can report an end one past the text when a value runs to the very end of the input, so clamp.
                start, end = max(0, int(item["start"])), min(len(text), int(item["end"]))
                while start < end and text[start].isspace():  # the labels never include a leading or trailing space
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                if end > start:
                    spans.append(Span(start, end, label, float(item.get("confidence", 1.0)), text[start:end]))
        return resolve_overlaps(spans)

    def _interior(self, span: Span, a: int, b: int, n: int) -> bool:
        return (a == 0 or span.start - a >= self.margin) and (b == n or b - span.end >= self.margin)

    def detect(self, text: str) -> list[Span]:
        """Non-overlapping spans, in order of position, with character offsets into `text`."""
        wins = windows(text, self.size, self.overlap)
        if len(wins) == 1:
            return self._window(text)
        n = len(text)
        kept: list[Span] = []
        edge: list[Span] = []
        for a, b in wins:
            for s in self._window(text[a:b]):
                shifted = Span(s.start + a, s.end + a, s.label, s.score, s.text)
                (kept if self._interior(shifted, a, b, n) else edge).append(shifted)
        for s in edge:  # near a window edge: keep it only if no window holds it from the inside
            if not any(a <= s.start and s.end <= b and self._interior(s, a, b, n) for a, b in wins):
                kept.append(s)
        best: dict[tuple[int, int, str], Span] = {}
        for s in kept:
            key = (s.start, s.end, s.label)
            if key not in best or s.score > best[key].score:
                best[key] = s
        return resolve_overlaps(list(best.values()))

    def redact(self, text: str, style: str = "label") -> str:
        return redact(text, self.detect(text), style)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find sensitive spans in text from a file or standard input.")
    parser.add_argument("file", nargs="?", help="text file to scan (standard input if omitted)")
    parser.add_argument("--redact", choices=["label", "mask"], help="print the text with spans replaced instead of listing them")
    parser.add_argument("--json", action="store_true", help="print spans as JSON")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    text = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    lookout = ObiLookout(threshold=args.threshold)
    spans = lookout.detect(text)
    if args.redact:
        print(redact(text, spans, args.redact))
    elif args.json:
        print(json.dumps([s.__dict__ for s in spans], ensure_ascii=False, indent=2))
    else:
        for s in spans:
            print(f"{s.start:>6}-{s.end:<6} {s.label:<24} {s.score:.2f}  {s.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
