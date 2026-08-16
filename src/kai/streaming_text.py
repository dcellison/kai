"""Publishable-prefix policy for streaming assistant text.

Live transports render assistant output progressively from raw accumulated
StreamEvent text. Backend chunks arrive at protocol boundaries (tokens,
model-defined messages) which rarely line up with human boundaries. The
helper below answers "is there a stable prefix of this accumulated text I
can publish right now?" and returns the longest such prefix or None.

Transport callers (the Telegram live-edit path and the Workshop browser
preview path) consult this helper for every streamed update; a None answer
means "withhold this update; the final response path still delivers
everything at completion". A non-None answer is the candidate text to
publish. Backends are untouched; this is purely a presentation policy.
"""

import re

# Triple-backtick fence delimiter, allowing up to three leading spaces
# (CommonMark allows 0-3) before the opening sequence.
_FENCE_LINE_RE = re.compile(r"^[ ]{0,3}```")
# Sentence terminators.
_SENTENCE_END_CHARS = frozenset(".?!")
# Closing punctuation that may follow a sentence terminator and stay part
# of the same sentence (quote-then-period style).
_CLOSE_PUNCT_CHARS = frozenset("\"')]`")
# Markdown list-item line shapes.
_LIST_LINE_WITH_CONTENT_RE = re.compile(r"^[ ]*([-*+]|\d+\.)\s+\S")
_LIST_LINE_RE = re.compile(r"^[ ]*([-*+]|\d+\.)(\s|$)")
# Ordered-list marker at the start of a line. Used in _sentence_cuts to
# advance past the marker period; without this, `1. Item one` would
# treat the `.` in `1.` as a sentence boundary and publish a bare
# numbered marker as if it were a completed sentence.
_ORDERED_LIST_MARKER_RE = re.compile(r"^[ ]*\d+\.(\s|$)")
# Looser pattern that also matches a forming next-item marker. The
# stream may have emitted just the digits of the next ordered-list
# marker (`3`) before the period and following text arrive; treating
# the partial marker as a list-item signal lets list_item cuts fire on
# the previous complete items even while the next marker is still
# mid-emission.
_LIST_LINE_FORMING_RE = re.compile(r"^[ ]*(\d+\.?|[-*+])(\s|$)")
# Minimum length for the long-span fallback to fire. Picked to be long
# enough that a coherent paragraph is likely visible, but short enough
# that streamed paragraphs reach it before the user gives up watching.
_LONG_SPAN_MIN_CHARS = 240
# Candidate kinds ordered by preference when two kinds resolve to the
# same cut position. Lower number = stronger boundary, evaluated first.
_KIND_PRIORITY = {
    "sentence": 1,
    "paragraph": 2,
    "list_item": 3,
    "closed_fence": 4,
    "long_span": 5,
    "full": 6,
}


def _is_fence_line(line: str) -> bool:
    return bool(_FENCE_LINE_RE.match(line))


def _has_open_fenced_code(text: str) -> bool:
    """True iff `text` has an odd number of triple-backtick fence lines.

    An odd count means the last fence opened is still unclosed; the helper
    must not publish through an open fenced block because the next chunk
    is going to land inside it.
    """
    fences = sum(1 for line in text.splitlines() if _is_fence_line(line))
    return (fences % 2) == 1


def _segments_outside_fences(text: str) -> str:
    """Return the lines of `text` that sit outside fenced code blocks.

    Used by the inline-Markdown checks: backticks and brackets inside a
    fenced code block are content, not Markdown delimiters, and must not
    affect the balance counts.
    """
    parts = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _is_fence_line(line.rstrip("\n")):
            in_fence = not in_fence
            continue
        if not in_fence:
            parts.append(line)
    return "".join(parts)


def _has_unbalanced_inline_markdown(text: str) -> bool:
    """True iff `text` has an unbalanced inline-code span or link.

    Checks (all candidate-wide per spec §6 D6 N-4 fix):
      - Odd count of unescaped single backticks outside fenced regions.
      - Unmatched `[` (no later `]`) anywhere outside fenced regions.
      - Unmatched `](` (no later closing `)`) anywhere outside fenced
        regions, which catches links whose target was cut mid-stream.
    """
    outside = _segments_outside_fences(text)
    n = len(outside)

    # Backtick parity, skipping backslash-escaped chars.
    backticks = 0
    i = 0
    while i < n:
        if outside[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if outside[i] == "`":
            backticks += 1
        i += 1
    if backticks % 2 == 1:
        return True

    # Bracket + link-target balance. A `[label]` reference link closes
    # cleanly; `[label](target)` opens a target on the matching `]` and
    # tracks paren depth until it closes. An unbalanced state at end of
    # text means the stream cut inside an open link.
    open_brackets = 0
    in_link_target = False
    paren_depth = 0
    i = 0
    while i < n:
        if outside[i] == "\\" and i + 1 < n:
            i += 2
            continue
        c = outside[i]
        if not in_link_target:
            if c == "[":
                open_brackets += 1
            elif c == "]" and open_brackets > 0:
                open_brackets -= 1
                if i + 1 < n and outside[i + 1] == "(":
                    in_link_target = True
                    paren_depth = 1
                    i += 2
                    continue
        else:
            if c == "(":
                paren_depth += 1
            elif c == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    in_link_target = False
        i += 1
    return open_brackets > 0 or in_link_target


def _line_ends_inside_unmatched_inline(line: str) -> bool:
    """True iff `line` has an odd backtick count or open `[`."""
    bt = 0
    open_brackets = 0
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "\\" and i + 1 < n:
            i += 2
            continue
        c = line[i]
        if c == "`":
            bt += 1
        elif c == "[":
            open_brackets += 1
        elif c == "]" and open_brackets > 0:
            open_brackets -= 1
        i += 1
    return (bt % 2 == 1) or open_brackets > 0


def _has_dangling_final_line(candidate: str, kind: str) -> bool:
    """True iff the final visible line of `candidate` is a dangling fragment.

    For non-`full` candidates the boundary collection in Phase 2 already
    validated the final line as a stable unit (a complete sentence,
    paragraph end, list item, etc.), so only the unmatched-inline-Markdown
    rule on the final line runs. For `full` candidates the cut sits at
    the stream's tail and the full §D5 guard applies: bare list/heading
    markers, short or single-word lines without sentence punctuation, and
    mid-line broken inline Markdown all reject.
    """
    stripped = candidate.rstrip()
    if not stripped:
        return False
    lines = stripped.splitlines()
    final_line = lines[-1] if lines else ""
    final_stripped = final_line.strip()
    if not final_stripped:
        return False

    # Look past trailing closing punctuation runs to find the true
    # terminator. `He said "Yes."` ends with `"` but the real terminator
    # is the `.`; that distinction lets compound endings like `?!` and
    # quote-wrapped sentences satisfy the "ends in sentence punctuation"
    # check below.
    last_real = final_stripped
    while last_real and last_real[-1] in _CLOSE_PUNCT_CHARS:
        last_real = last_real[:-1]
    ends_in_sentence = bool(last_real) and last_real[-1] in _SENTENCE_END_CHARS

    # The `full` candidate sits at the very tail of the accumulated
    # buffer; it has no preceding boundary marker to vouch for it. It
    # is safe to publish only when the final visible line ends in
    # sentence punctuation (with optional closing quotes/parens). Any
    # other tail shape (bare list/heading markers, mid-sentence prose
    # of any length, a dangling fragment after an earlier sentence on
    # the same line) risks shipping unstable text that the next stream
    # chunk will overwrite. Other stable boundary kinds (paragraph,
    # sentence, closed_fence, list_item) produce their own candidates
    # at lower or equal cut positions and win via the priority sort
    # when they coincide.
    if kind == "full" and not ends_in_sentence:
        return True

    # Rule 5 applies to every candidate kind: an open inline span on the
    # final line is broken Markdown regardless of cut origin.
    if _is_fence_line(final_line):
        # Triple-backtick fence delimiter; the trailing backticks are the
        # closing fence marker, not an unclosed inline code span. Without
        # this carve-out, closed-fence candidates get rejected for ending
        # with three "unmatched" backticks.
        return False
    return _line_ends_inside_unmatched_inline(final_line)


def _paragraph_cuts(working: str) -> list[int]:
    """Positions of `\\n\\n` separators that are followed by more content."""
    cuts: list[int] = []
    i = 0
    n = len(working)
    while True:
        j = working.find("\n\n", i)
        if j < 0:
            break
        if working[j + 2 : n].strip():
            cuts.append(j)
        i = j + 1
    return cuts


def _sentence_cuts(working: str) -> list[int]:
    """End-of-sentence positions outside any open fenced code block.

    A `.?!` run (optionally followed by closing punctuation like quotes
    or parens) qualifies as a sentence boundary only when the run ends
    at end-of-line or is followed by whitespace. Mid-token periods
    (decimals like `3.13`, version strings like `v1.2.3`, file paths
    like `src/bot.py`, domain names) are NOT sentence ends; cutting at
    them would publish a misleading prefix that splits the token in
    half (e.g. `Use Python 3.` while the stream still has `13` to come).
    The next stream chunk would then overwrite the visible message with
    the correctly-joined text, but the user has already seen the wrong
    prefix flash by.
    """
    cuts: list[int] = []
    in_fence = False
    pos = 0
    n = len(working)
    while pos < n:
        nl = working.find("\n", pos)
        line_end = nl if nl >= 0 else n
        line = working[pos:line_end]
        if _is_fence_line(line):
            in_fence = not in_fence
        elif not in_fence:
            line_len = len(line)
            # Advance past an ordered-list marker so its period isn't
            # treated as a sentence terminator. The marker `.` is
            # followed by whitespace and would otherwise satisfy the
            # sentence-end predicate, letting `1.` publish as if it
            # were a complete sentence while the list item is still
            # being typed.
            list_match = _ORDERED_LIST_MARKER_RE.match(line)
            j = list_match.end() if list_match else 0
            while j < line_len:
                if line[j] in _SENTENCE_END_CHARS:
                    # Extend the cut through any closing-punctuation or
                    # compound sentence-end run so `Yes."` and `?!` keep
                    # the closer as part of the published prefix.
                    k = j + 1
                    while k < line_len and (line[k] in _CLOSE_PUNCT_CHARS or line[k] in _SENTENCE_END_CHARS):
                        k += 1
                    # Sentence-boundary predicate: the run must land at
                    # end-of-line or be followed by a whitespace
                    # separator. Anything else means the punctuation is
                    # internal to a token and should not become a cut.
                    if k == line_len or line[k].isspace():
                        cuts.append(pos + k)
                    j = k
                else:
                    j += 1
        pos = line_end + 1 if nl >= 0 else line_end
    return cuts


def _closed_fence_cuts(working: str) -> list[int]:
    """Positions immediately after each closing fence line."""
    cuts: list[int] = []
    in_fence = False
    pos = 0
    n = len(working)
    while pos < n:
        nl = working.find("\n", pos)
        line_end = nl if nl >= 0 else n
        line = working[pos:line_end]
        if _is_fence_line(line):
            in_fence = not in_fence
            if not in_fence:
                cuts.append(line_end)
        pos = line_end + 1 if nl >= 0 else line_end
    return cuts


def _list_item_cuts(working: str) -> list[int]:
    """End-of-line positions where a complete list item closes."""
    cuts: list[int] = []
    lines = working.splitlines()
    if not lines:
        return cuts
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    for i, line in enumerate(lines):
        if not _LIST_LINE_WITH_CONTENT_RE.match(line):
            continue
        if i + 1 >= len(lines):
            continue
        next_line = lines[i + 1]
        # A list-item boundary fires when the next line is either
        # blank, a fully-formed list-item line, or a still-forming
        # marker (bare digits like `3` before the period arrives, a
        # lone bullet character). That keeps prose paragraphs after a
        # list intact (the paragraph boundary handles them), avoids
        # cutting a list short when the next line is still building,
        # and accepts the partial next-item marker as evidence that
        # the previous item is complete.
        if not next_line.strip() or _LIST_LINE_FORMING_RE.match(next_line):
            cuts.append(starts[i] + len(line))
    return cuts


def _long_span_cut(working: str) -> int | None:
    """Cut at a whitespace boundary in long unpunctuated prose.

    Long-form responses without paragraph or sentence breaks still need
    a streaming surface so the user sees progress. Two shapes are
    handled:

      1. Multi-line: when a newline exists and the prefix before the
         final newline already holds at least ``_LONG_SPAN_MIN_CHARS``
         of content, cut before the final newline. This preserves the
         dangling-line guard's protection on the final (possibly
         in-progress) line.
      2. Single-line: when no newline exists, cut at the rightmost
         whitespace whose position is at or beyond the threshold.
         Without this fallback, an inner-Claude monologue streamed as
         one long unpunctuated paragraph would have no stable prefix
         until a sentence terminator finally appears; the user sees a
         stalled message for the entire run.
    """
    last_nl = working.rfind("\n")
    if last_nl >= 0:
        prefix = working[:last_nl].rstrip()
        if len(prefix) >= _LONG_SPAN_MIN_CHARS:
            return last_nl
        return None
    # Single-line fallback. Scan right-to-left for a whitespace at
    # position ≥ threshold; the word immediately before such a
    # whitespace is guaranteed complete (it has a separator after it),
    # whereas the final word at the buffer tail may still be growing.
    n = len(working)
    if n <= _LONG_SPAN_MIN_CHARS:
        return None
    for i in range(n - 1, _LONG_SPAN_MIN_CHARS - 1, -1):
        if working[i].isspace():
            prefix = working[:i].rstrip()
            if len(prefix) >= _LONG_SPAN_MIN_CHARS:
                return i
    return None


def stream_publishable_prefix(text: str) -> str | None:
    """
    Return the longest stable prefix of `text` safe for a live transport
    update, or None when no stable prefix exists yet.

    Stable means the prefix ends at a coherent boundary (paragraph,
    sentence, closed fenced code block, list-item boundary, or a long
    whitespace-aligned span) and its final visible line is not a
    dangling fragment. The helper is pure and deterministic; transport
    callers may invoke it on every streamed update and either publish
    the returned prefix or wait for the next event.
    """
    if not text or not text.strip():
        return None
    working = text.rstrip()
    if not working:
        return None

    candidates: list[tuple[int, str]] = []
    candidates.extend((p, "paragraph") for p in _paragraph_cuts(working))
    candidates.extend((p, "sentence") for p in _sentence_cuts(working))
    candidates.extend((p, "closed_fence") for p in _closed_fence_cuts(working))
    candidates.extend((p, "list_item") for p in _list_item_cuts(working))
    ls = _long_span_cut(working)
    if ls is not None:
        candidates.append((ls, "long_span"))
    candidates.append((len(working), "full"))

    # Evaluate longest-first; when two kinds resolve to the same cut
    # position, prefer the stronger boundary so the dangling-line guard
    # runs in its lighter mode for the chosen candidate.
    candidates.sort(key=lambda pk: (-pk[0], _KIND_PRIORITY[pk[1]]))

    seen_positions: set[int] = set()
    for cut, kind in candidates:
        if cut in seen_positions:
            continue
        seen_positions.add(cut)
        candidate = working[:cut].rstrip()
        if not candidate:
            continue
        if _has_open_fenced_code(candidate):
            continue
        if _has_unbalanced_inline_markdown(candidate):
            continue
        if _has_dangling_final_line(candidate, kind):
            continue
        return candidate
    return None
