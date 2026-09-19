import {
  KeyboardEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { WorkshopCapabilityAvailability } from "./types";

export interface WorkshopActionPaletteEntry {
  capability: WorkshopCapabilityAvailability;
  key: string;
  targetAgentId: string | null;
  targetLabel: string | null;
}

export function ActionPalette({
  actions,
  error,
  loading,
  onClose,
  onInvoke,
}: {
  actions: WorkshopActionPaletteEntry[];
  error: string | null;
  loading: boolean;
  onClose: () => void;
  onInvoke: (action: WorkshopActionPaletteEntry) => Promise<string | null>;
}): React.JSX.Element {
  const inputRef = useRef<HTMLInputElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const [query, setQuery] = useState("");
  const [selection, setSelection] = useState(0);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const filtered = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    if (!normalized) return actions;
    return actions.filter(({ capability, targetLabel }) =>
      [capability.label, capability.description, targetLabel ?? ""]
        .join(" ")
        .toLocaleLowerCase()
        .includes(normalized)
    );
  }, [actions, query]);

  useLayoutEffect(() => {
    restoreFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    inputRef.current?.focus();
    return () => restoreFocusRef.current?.focus();
  }, []);

  useEffect(() => {
    setSelection((current) => Math.min(current, Math.max(filtered.length - 1, 0)));
  }, [filtered.length]);

  const invoke = async (action: WorkshopActionPaletteEntry): Promise<void> => {
    if (busyKey !== null) return;
    setBusyKey(action.key);
    setActionError(null);
    setResult(null);
    try {
      const nextResult = await onInvoke(action);
      if (nextResult === null) {
        onClose();
      } else {
        setResult(nextResult);
      }
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "Could not complete this action.");
    } finally {
      setBusyKey(null);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (filtered.length === 0) return;
      const direction = event.key === "ArrowDown" ? 1 : -1;
      setSelection((current) => (current + direction + filtered.length) % filtered.length);
      return;
    }
    if (event.key === "Enter" && filtered[selection]) {
      event.preventDefault();
      void invoke(filtered[selection]);
    }
  };

  return (
    <div className="action-palette-backdrop" onMouseDown={onClose}>
      <section
        aria-label="Help and actions"
        aria-modal="true"
        className="action-palette"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
      >
        <header>
          <div>
            <p className="overline">Kai Workshop</p>
            <h2>Help and actions</h2>
          </div>
          <button
            aria-label="Close help and actions"
            className="panel-icon-button"
            onClick={onClose}
            title="Close"
            type="button"
          >
            <span aria-hidden="true">×</span>
          </button>
        </header>
        <label className="action-palette-search">
          <span className="visually-hidden">Search actions</span>
          <input
            aria-activedescendant={filtered[selection] ? `action-${filtered[selection].key}` : undefined}
            aria-controls="action-palette-results"
            aria-expanded="true"
            aria-label="Search actions"
            autoComplete="off"
            onChange={(event) => {
              setQuery(event.target.value);
              setSelection(0);
              setResult(null);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Search actions and destinations…"
            ref={inputRef}
            role="combobox"
            value={query}
          />
          <kbd>Esc</kbd>
        </label>
        <div className="action-palette-results" id="action-palette-results" role="listbox">
          {loading ? (
            <p role="status">Loading available actions…</p>
          ) : error ? (
            <p className="settings-error" role="alert">{error}</p>
          ) : filtered.length === 0 ? (
            <p>No available actions match “{query}”.</p>
          ) : filtered.map((action, index) => (
            <button
              aria-selected={selection === index}
              className={selection === index ? "selected" : ""}
              disabled={busyKey !== null}
              id={`action-${action.key}`}
              key={action.key}
              onClick={() => void invoke(action)}
              onMouseMove={() => setSelection(index)}
              role="option"
              type="button"
            >
              <span>
                <strong>{action.capability.label}</strong>
                <small>{action.capability.description}</small>
              </span>
              {action.targetLabel && <em>{action.targetLabel}</em>}
            </button>
          ))}
        </div>
        {(result || actionError) && (
          <p className={actionError ? "settings-error action-palette-feedback" : "action-palette-feedback"}
            role={actionError ? "alert" : "status"}
          >
            {actionError ?? result}
          </p>
        )}
        <footer>
          <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
          <span><kbd>Enter</kbd> open</span>
          <span><kbd>Esc</kbd> close</span>
        </footer>
      </section>
    </div>
  );
}
