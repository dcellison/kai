import { useEffect, useRef, useState } from "react";

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

function DetailBlock({ entry }: { entry: WorkshopRunTraceEntry }): React.JSX.Element | null {
  if (!entry.detail) {
    return null;
  }
  if (!entry.isDiff) {
    return <pre className="trace-detail">{entry.detail}</pre>;
  }
  return (
    <pre className="trace-detail">
      {entry.detail.split("\n").map((line, index) => (
        <span
          key={index}
          className={
            line.startsWith("+")
              ? "trace-diff-add"
              : line.startsWith("-")
                ? "trace-diff-del"
                : undefined
          }
        >
          {line}
          {"\n"}
        </span>
      ))}
    </pre>
  );
}

export function RunTraceCard({
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
}
