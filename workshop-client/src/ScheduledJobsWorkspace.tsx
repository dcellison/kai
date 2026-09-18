import {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  AuthenticationError,
  cancelScheduledJob,
  loadScheduledJobs,
} from "./api";
import type { WorkshopScheduledJob, WorkshopSession } from "./types";
import type { WorkshopPrincipalEvents } from "./usePrincipalEvents";
import { useConfirmation } from "./ConfirmationDialog";

interface ScheduledJobsDetailPanelLayout {
  width: number;
  minimumWidth: number;
  maximumWidth: number;
  onKeyDown: (event: ReactKeyboardEvent<HTMLDivElement>) => void;
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void;
}

function CancelIcon(): React.JSX.Element {
  return <svg aria-hidden="true" fill="none" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" /></svg>;
}

function ClockIcon(): React.JSX.Element {
  return <svg aria-hidden="true" fill="none" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5" stroke="currentColor" strokeWidth="1.7" /><path d="M12 7.5v5l3.2 2" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7" /></svg>;
}

function displayDate(value: string | null): string {
  if (!value) return "Not currently scheduled";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function scheduleLabel(job: WorkshopScheduledJob): string {
  let data: unknown;
  try { data = JSON.parse(job.scheduleData); } catch { return job.scheduleType; }
  if (!data || typeof data !== "object") return job.scheduleType;
  const record = data as Record<string, unknown>;
  if (job.scheduleType === "once") {
    return typeof record.run_at === "string" ? `Once · ${displayDate(record.run_at)}` : "Once";
  }
  if (job.scheduleType === "interval") {
    const seconds = record.seconds;
    if (typeof seconds !== "number") return "Recurring interval";
    if (seconds % 3600 === 0) return `Every ${seconds / 3600} hour${seconds === 3600 ? "" : "s"}`;
    if (seconds % 60 === 0) return `Every ${seconds / 60} minute${seconds === 60 ? "" : "s"}`;
    return `Every ${seconds} seconds`;
  }
  const times = Array.isArray(record.times) ? record.times.filter((item): item is string => typeof item === "string") : [];
  return times.length ? `Daily · ${times.join(", ")} UTC` : "Daily";
}

function transportLabel(value: string): string {
  if (value === "workshop_client") return "Workshop";
  return value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ");
}

export function ScheduledJobsWorkspace({
  detailPanelLayout,
  onAuthenticationFailure,
  onClose,
  principalEvents,
  session,
}: {
  detailPanelLayout?: ScheduledJobsDetailPanelLayout;
  onAuthenticationFailure: (message: string) => void;
  onClose: () => void;
  principalEvents: WorkshopPrincipalEvents;
  session: WorkshopSession;
}): React.JSX.Element {
  const confirm = useConfirmation();
  const [jobs, setJobs] = useState<WorkshopScheduledJob[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    setLoading(true);
    try {
      const next = await loadScheduledJobs(session);
      setJobs(next);
      setSelectedId((current) => next.some((job) => job.id === current) ? current : (next[0]?.id ?? null));
      setError(null);
    } catch (caught) {
      if (caught instanceof AuthenticationError) onAuthenticationFailure(caught.message);
      else setError(caught instanceof Error ? caught.message : "Could not load scheduled jobs.");
    } finally {
      setLoading(false);
    }
  }, [onAuthenticationFailure, session]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => principalEvents.subscribe((event) => {
    if (event.kind === "synchronize" || event.batch.changes.some((change) => change.jobChanges.length > 0)) {
      void refresh();
    }
  }), [principalEvents, refresh]);

  const selected = useMemo(
    () => jobs.find((job) => job.id === selectedId) ?? null,
    [jobs, selectedId],
  );
  const oneTimeJobs = jobs.filter((job) => job.scheduleType === "once");
  const recurringJobs = jobs.filter((job) => job.scheduleType !== "once");

  const cancel = async (): Promise<void> => {
    if (!selected || busy || !await confirm(
      `Cancel ${selected.name}? Its active schedule will be removed, while existing conversation and delivery history will remain.`,
    )) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await cancelScheduledJob(
        session,
        selected.id,
        `workshop-job-cancel-${selected.id}-${Date.now()}`,
      );
      setNotice(`${selected.name} was cancelled.`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not cancel scheduled job.");
    } finally {
      setBusy(false);
    }
  };

  const renderGroup = (heading: string, items: WorkshopScheduledJob[]): React.JSX.Element | null => items.length ? (
    <section className="scheduled-job-group">
      <h2>{heading}</h2>
      <ul>{items.map((job) => <li key={job.id}><button className={job.id === selectedId ? "selected" : ""} type="button" onClick={() => setSelectedId(job.id)}><span><strong>{job.name}</strong><span className="scheduled-job-status">Active</span></span><small>{scheduleLabel(job)}</small><small>{job.jobType === "reminder" ? "Reminder" : "Agent task"} · {job.agentName}</small></button></li>)}</ul>
    </section>
  ) : null;

  return (
    <main className="workspaces-workspace scheduled-jobs-workspace" aria-label="Scheduled jobs">
      <section className="workspaces-browser-pane">
        <header className="workspaces-header"><div><p className="breadcrumbs">Kai Workshop / Scheduled jobs</p><h1>Scheduled jobs</h1></div><div className="workspaces-header-actions"><button className="panel-icon-button" type="button" aria-label="Back to conversation" title="Back to conversation" onClick={onClose}><span aria-hidden="true">←</span></button></div></header>
        {(notice || error) && <div className="workspaces-notices" aria-live="polite">{notice && <p className="settings-notice" role="status">{notice}</p>}{error && <p className="settings-error" role="alert">{error}</p>}</div>}
        <section className="scheduled-job-catalogue" aria-label="Active scheduled jobs">
          {loading ? <p className="workspaces-empty">Loading scheduled jobs…</p> : jobs.length === 0 ? <div className="scheduled-jobs-empty"><ClockIcon /><h2>No active scheduled jobs</h2><p>Jobs created through Kai will appear here for inspection and cancellation.</p></div> : <>{renderGroup("Upcoming", oneTimeJobs)}{renderGroup("Recurring", recurringJobs)}</>}
        </section>
      </section>
      <aside className="context-pane workspace-detail scheduled-job-detail" aria-label="Scheduled job detail">
        {detailPanelLayout && <div className="context-resize-handle" role="separator" aria-label="Resize scheduled job detail" aria-orientation="vertical" aria-valuemin={detailPanelLayout.minimumWidth} aria-valuemax={detailPanelLayout.maximumWidth} aria-valuenow={detailPanelLayout.width} tabIndex={0} onKeyDown={detailPanelLayout.onKeyDown} onPointerDown={detailPanelLayout.onPointerDown} onPointerMove={detailPanelLayout.onPointerMove} />}
        <div className="workspace-detail-scroll" aria-live="polite">
          {selected ? <><div className="workspace-detail-heading"><div><p className="overline">Selected job</p><h2>{selected.name}</h2></div><button className="panel-icon-button danger-icon-button" type="button" aria-label="Cancel scheduled job" title="Cancel scheduled job" disabled={busy} onClick={() => void cancel()}><CancelIcon /></button></div><dl><div><dt>Status</dt><dd>Active</dd></div><div><dt>Schedule</dt><dd>{scheduleLabel(selected)}</dd></div><div><dt>Next run</dt><dd>{displayDate(selected.nextRunAt)}</dd></div><div><dt>Created</dt><dd>{displayDate(selected.createdAt)}</dd></div><div><dt>Type</dt><dd>{selected.jobType === "reminder" ? "Reminder" : "Agent task"}</dd></div><div><dt>Agent</dt><dd>{selected.agentName}</dd></div><div><dt>Delivery</dt><dd>{selected.channelName}{selected.deliveryTransports.length ? ` · ${selected.deliveryTransports.map(transportLabel).join(", ")}` : ""}</dd></div><div><dt>Auto-removal</dt><dd>{selected.autoRemove ? "Enabled" : "Disabled"}</dd></div></dl><section className="scheduled-job-prompt"><h3>Prompt</h3><p>{selected.prompt}</p></section></> : <p className="workspaces-empty">Select a scheduled job.</p>}
        </div>
      </aside>
    </main>
  );
}
