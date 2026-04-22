import { useEffect, useRef, useState } from "react";
import type { KaboEvent } from "../types";

/**
 * Subscribe to the backend SSE stream and accumulate events.
 *
 * Features:
 *  - Auto-reconnect with a small backoff.
 *  - Ring-buffer retention: keep at most `maxBuffer` events in memory.
 *  - Exposes the raw `EventSource` state for status indicators.
 */
export function useEventStream(url = "/api/runs/current/events", maxBuffer = 2000) {
  const [events, setEvents] = useState<KaboEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const bufferRef = useRef<KaboEvent[]>([]);

  useEffect(() => {
    let cancelled = false;
    let retryTimer: number | null = null;

    const open = () => {
      if (cancelled) return;
      const es = new EventSource(url);
      esRef.current = es;

      es.onopen = () => setConnected(true);
      es.onerror = () => {
        setConnected(false);
        es.close();
        if (!cancelled) {
          retryTimer = window.setTimeout(open, 1500);
        }
      };
      es.onmessage = (ev) => {
        if (!ev.data) return;
        try {
          const parsed = JSON.parse(ev.data) as KaboEvent;
          bufferRef.current.push(parsed);
          if (bufferRef.current.length > maxBuffer) {
            bufferRef.current = bufferRef.current.slice(-maxBuffer);
          }
          // Rebuild a fresh array so React re-renders
          setEvents([...bufferRef.current]);
        } catch (err) {
          console.warn("Failed to parse SSE event", err, ev.data);
        }
      };
    };

    open();

    return () => {
      cancelled = true;
      if (retryTimer) window.clearTimeout(retryTimer);
      esRef.current?.close();
    };
  }, [url, maxBuffer]);

  const clear = () => {
    bufferRef.current = [];
    setEvents([]);
  };

  return { events, connected, clear };
}
