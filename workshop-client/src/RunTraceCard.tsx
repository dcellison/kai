import { memo, useEffect, useRef, useState } from "react";

import type { WorkshopRunTraceEntry } from "./types";

interface TraceRow {
  call: WorkshopRunTraceEntry | null;
  key: number;
  marker: WorkshopRunTraceEntry | null;
  result: WorkshopRunTraceEntry | null;
}

// Fold each tool_result into its paired tool_call row via tool_use_id.
// Unpaired results stay their own row: a crashed turn can drop the
// terminal update, and some backends emit already-terminal calls, so
// pairing is best-effort by design.
function buildRows(entries: WorkshopRunTraceEntry[]): TraceRow[] {
  const rows: TraceRow[] = [];
  const openCalls = new Map<string, TraceRow>();
  for (const entry of entries) {
    if (entry.kind === "tool_call") {
      const row: TraceRow = { call: entry, key: entry.seq, marker: null, result: null };
      rows.push(row);
      if (entry.toolUseId) {
        openCalls.set(entry.toolUseId, row);
      }
    } else if (entry.kind === "tool_result") {
      const paired = entry.toolUseId ? openCalls.get(entry.toolUseId) : undefined;
      if (paired && paired.result === null) {
        paired.result = entry;
      } else {
        rows.push({ call: null, key: entry.seq, marker: null, result: entry });
      }
    } else {
      rows.push({ call: null, key: entry.seq, marker: entry, result: null });
    }
  }
  return rows;
}

// Display-only transform: a non-diff detail that parses as JSON is
// re-serialized with two-space indentation. Falling back to the raw text
// on parse failure is load-bearing, not just defensive: the backend
// truncates details at the source, so a payload can arrive cut mid-JSON
// and must still render as stored.
function formatDetail(detail: string): string {
  try {
    return JSON.stringify(JSON.parse(detail), null, 2);
  } catch {
    return detail;
  }
}

function DetailBlock({ entry }: { entry: WorkshopRunTraceEntry }): React.JSX.Element | null {
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<number | null>(null);

  // Cancel a pending checkmark reset if the block unmounts (row collapsed,
  // run switched) before the swap-back fires.
  useEffect(() => {
    return () => {
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current);
      }
    };
  }, []);

  if (!entry.detail) {
    return null;
  }

  // The copy button copies exactly what is rendered: the formatted JSON
  // when formatting applied, the raw detail otherwise. Both are already
  // redacted and truncated at the source.
  const text = entry.isDiff ? entry.detail : formatDetail(entry.detail);

  // The async clipboard API only exists in secure contexts (localhost
  // qualifies); render no button at all rather than one that always fails.
  const clipboard = typeof navigator === "undefined" ? undefined : navigator.clipboard;
  const copy = (): void => {
    void clipboard?.writeText(text).then(
      () => {
        setCopied(true);
        if (copyTimerRef.current !== null) {
          window.clearTimeout(copyTimerRef.current);
        }
        copyTimerRef.current = window.setTimeout(() => setCopied(false), 1500);
      },
      () => {
        // A rejected write (permission revoked, window unfocused) leaves
        // the button in its idle state; there is no error surface here.
      },
    );
  };

  return (
    <div className="trace-detail-wrap">
      {entry.isDiff ? (
        <pre className="trace-detail">
          {/* The trace-lines wrapper, not each line, carries the max-content
              sizing: sizing lines individually would let short lines' tints
              stop at the container's client width while a longer sibling
              forces horizontal scroll past them. */}
          <span className="trace-lines">
            {entry.detail.split("\n").map((line, index) => (
              <span
                key={index}
                className={
                  line.startsWith("+")
                    ? "trace-line trace-diff-add"
                    : line.startsWith("-")
                      ? "trace-line trace-diff-del"
                      : "trace-line"
                }
              >
                {/* Block-level lines supply their own breaks, so the newline
                    separators are dropped; an empty line keeps an explicit
                    one so it still occupies a row. */}
                {line === "" ? "\n" : line}
              </span>
            ))}
          </span>
        </pre>
      ) : (
        <pre className="trace-detail">{text}</pre>
      )}
      {clipboard && (
        <button
          type="button"
          className="trace-copy"
          aria-label={
            copied
              ? "Copied"
              : entry.kind === "tool_call"
                ? "Copy call detail"
                : "Copy result detail"
          }
          onClick={copy}
        >
          {copied ? "✓" : "⧉"}
        </button>
      )}
    </div>
  );
}

// Memoized because the enclosing view re-renders on every keystroke and
// resize pointermove while the card's props (entries array identity and
// scalars) only change when trace pages land or the inspected run moves.
export const RunTraceCard = memo(function RunTraceCard({
  entries,
  failed,
  loaded,
  runId,
}: {
  entries: WorkshopRunTraceEntry[];
  failed: boolean;
  loaded: boolean;
  runId: string | null;
}): React.JSX.Element {
  // Rows the user explicitly toggled; untouched rows fall back to the
  // default, which expands diffs and collapses everything else.
  const [toggled, setToggled] = useState<Map<number, boolean>>(new Map());
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    setToggled(new Map());
    pinnedRef.current = true;
  }, [runId]);

  // Follow appended steps only while the reader is already at the
  // bottom, the same convention the timeline uses; a deliberate scroll
  // back stays put. This container is the card's own; the timeline-pin
  // effect is untouched.
  useEffect(() => {
    const container = scrollRef.current;
    if (container && pinnedRef.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [entries]);

  if (!runId) {
    return <p className="trace-empty">No runs yet in this channel.</p>;
  }
  if (failed && entries.length === 0) {
    return <p className="trace-empty">Trace unavailable for this run.</p>;
  }
  if (loaded && entries.length === 0) {
    return <p className="trace-empty">No steps recorded for this run.</p>;
  }

  const rows = buildRows(entries);
  const isExpanded = (row: TraceRow): boolean =>
    toggled.get(row.key) ?? row.call?.isDiff ?? false;
  const toggle = (row: TraceRow): void => {
    setToggled((current) => {
      const next = new Map(current);
      next.set(row.key, !isExpanded(row));
      return next;
    });
  };

  return (
    <div
      className="trace-card"
      ref={scrollRef}
      onScroll={() => {
        const container = scrollRef.current;
        if (container) {
          pinnedRef.current =
            container.scrollHeight - container.scrollTop - container.clientHeight < 16;
        }
      }}
    >
      <ol className="trace-steps">
        {rows.map((row) => {
          if (row.marker) {
            return (
              <li key={row.key} className="trace-truncated">
                {row.marker.summary}
              </li>
            );
          }
          const primary = row.call ?? row.result;
          if (!primary) {
            return null;
          }
          const expanded = isExpanded(row);
          const errored = row.result?.isError ?? primary.isError;
          return (
            <li key={row.key} className={`trace-row${errored ? " trace-error" : ""}`}>
              <button
                type="button"
                className="trace-step"
                aria-expanded={expanded}
                onClick={() => toggle(row)}
              >
                <span className="trace-icon" aria-hidden="true">
                  {row.call ? "⚙" : "↩"}
                </span>
                <span className="trace-tool">{primary.toolName ?? ""}</span>
                <span className="trace-summary">{primary.summary}</span>
                {row.result && row.call && (
                  <span className={`trace-chip${row.result.isError ? " trace-chip-error" : ""}`}>
                    {row.result.isError ? "error" : "done"}
                  </span>
                )}
              </button>
              {expanded && (
                <div className="trace-expansion">
                  {row.call && <DetailBlock entry={row.call} />}
                  {row.result && <DetailBlock entry={row.result} />}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
});
