import { useEffect, useRef, useState } from "react";

import type {
  WorkshopRunTraceEntry,
  WorkshopRunTracePage,
  WorkshopRunTraceSignal,
} from "./types";

export function useRunTrace(
  runId: string | null,
  signal: WorkshopRunTraceSignal | null,
  fetchPage: (runId: string, afterSeq: number) => Promise<WorkshopRunTracePage>,
): { entries: WorkshopRunTraceEntry[]; loaded: boolean } {
  const [entries, setEntries] = useState<WorkshopRunTraceEntry[]>([]);
  const [loaded, setLoaded] = useState(false);
  // Highest seq already held, the after_seq cursor for every fetch; a
  // ref rather than state so a doorbell arriving mid-drain reads the
  // freshest position without re-rendering per page.
  const highestSeqRef = useRef(0);
  // Monotonic run-change marker: a drain started for a previous run (or
  // a previous mount) discards its pages instead of appending them to
  // the successor's entries.
  const generationRef = useRef(0);

  const drain = async (target: string, generation: number): Promise<void> => {
    let hasMore = true;
    while (hasMore) {
      let page: WorkshopRunTracePage;
      try {
        page = await fetchPage(target, highestSeqRef.current);
      } catch {
        return;
      }
      if (generationRef.current !== generation) {
        return;
      }
      if (page.entries.length > 0) {
        highestSeqRef.current = page.entries[page.entries.length - 1].seq;
        setEntries((current) => [...current, ...page.entries]);
      }
      hasMore = page.hasMore;
    }
    setLoaded(true);
  };

  useEffect(() => {
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    highestSeqRef.current = 0;
    setEntries([]);
    setLoaded(false);
    if (runId) {
      void drain(runId, generation);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, fetchPage]);

  useEffect(() => {
    if (!runId || !signal || signal.runId !== runId || signal.seq <= highestSeqRef.current) {
      return;
    }
    void drain(runId, generationRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, signal, fetchPage]);

  return { entries, loaded };
}
