import { useCallback, useEffect, useRef, useState } from "react";

// Calls fn now and then every `ms` milliseconds (ms = 0 means once). Polling replaces WebSockets.
export function usePoll(fn, ms = 0, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true, at: 0 });
  const ref = useRef(fn);
  ref.current = fn;
  const load = useCallback(async () => {
    try {
      const data = await ref.current();
      setState({ data, error: null, loading: false, at: Date.now() });
    } catch (e) {
      setState((s) => ({ ...s, error: e.message, loading: false }));
    }
  }, []);
  useEffect(() => {
    load();
    if (!ms) return undefined;
    const t = setInterval(load, ms);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms, ...deps]);
  return { ...state, reload: load };
}

export function useNow(ms = 1000) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(t);
  }, [ms]);
  return now;
}
