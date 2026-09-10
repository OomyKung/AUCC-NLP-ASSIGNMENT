/** Shared hooks: data fetching, theme, debounce, toasts. */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../services/api'

/* -------------------------------------------------------------------------- */
/* Data fetching                                                              */
/* -------------------------------------------------------------------------- */

export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: string | null
  reload: () => void
}

/**
 * Run an async loader and track loading/error state.
 *
 * `deps` behaves like a useEffect dependency list. Results from a superseded
 * request are discarded, so fast filter changes cannot render stale data.
 */
export function useAsync<T>(
  loader: () => Promise<T>,
  deps: unknown[] = [],
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)

  // Identifies the most recent request so older ones can be ignored.
  const latest = useRef(0)

  useEffect(() => {
    const ticket = ++latest.current
    setLoading(true)
    setError(null)

    loader()
      .then((result) => {
        if (ticket !== latest.current) return
        setData(result)
        setError(null)
      })
      .catch((err: unknown) => {
        if (ticket !== latest.current) return
        setError(
          err instanceof ApiError || err instanceof Error
            ? err.message
            : 'Unexpected error',
        )
      })
      .finally(() => {
        if (ticket === latest.current) setLoading(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((value) => value + 1), [])
  return { data, loading, error, reload }
}

/* -------------------------------------------------------------------------- */
/* Theme                                                                      */
/* -------------------------------------------------------------------------- */

export type Theme = 'light' | 'dark'
const THEME_KEY = 'tni-theme'

/** Read the stored theme; light is the default per the design brief. */
function storedTheme(): Theme {
  try {
    return localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light'
  } catch {
    // Private browsing can throw on access.
    return 'light'
  }
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(storedTheme)

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    root.classList.toggle('light', theme === 'light')
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch {
      /* storage unavailable: the class is still applied for this session */
    }
  }, [theme])

  const toggle = useCallback(
    () => setTheme((value) => (value === 'dark' ? 'light' : 'dark')),
    [],
  )
  return { theme, toggle }
}

/**
 * Chart colours, read from the CSS custom properties so charts and badges use
 * exactly the same validated palette and re-read it when the theme flips.
 */
export function useChartColors(theme: Theme) {
  const [colors, setColors] = useState(() => readChartColors())
  useEffect(() => setColors(readChartColors()), [theme])
  return colors
}

function readChartColors() {
  const styles = getComputedStyle(document.documentElement)
  const read = (name: string, fallback: string) =>
    styles.getPropertyValue(name).trim() || fallback
  return {
    positive: read('--chart-positive', '#0d9488'),
    neutral: read('--chart-neutral', '#64748b'),
    negative: read('--chart-negative', '#dc2626'),
    brand: read('--chart-brand', '#4f46e5'),
    grid: read('--chart-grid', '#e2e8f0'),
    axis: read('--chart-axis', '#64748b'),
    surface: read('--chart-surface', '#ffffff'),
    ink: read('--chart-ink', '#1e293b'),
  }
}

/* -------------------------------------------------------------------------- */
/* Utilities                                                                  */
/* -------------------------------------------------------------------------- */

/** Delay a rapidly changing value, so typing does not fire a request per key. */
export function useDebounced<T>(value: T, delay = 350): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return debounced
}

/* -------------------------------------------------------------------------- */
/* Toasts                                                                     */
/* -------------------------------------------------------------------------- */

export interface Toast {
  id: number
  message: string
  tone: 'success' | 'error' | 'info'
}

let toastId = 0
type Listener = (toasts: Toast[]) => void

const listeners = new Set<Listener>()
let current: Toast[] = []

function publish() {
  for (const listener of listeners) listener([...current])
}

/** Show a toast. Callable from anywhere, including non-component code. */
export function notify(message: string, tone: Toast['tone'] = 'info') {
  const toast: Toast = { id: ++toastId, message, tone }
  current = [...current, toast]
  publish()
  setTimeout(() => {
    current = current.filter((item) => item.id !== toast.id)
    publish()
  }, 5000)
}

export function dismissToast(id: number) {
  current = current.filter((item) => item.id !== id)
  publish()
}

export function useToasts(): Toast[] {
  const [toasts, setToasts] = useState<Toast[]>(current)
  useEffect(() => {
    listeners.add(setToasts)
    return () => {
      listeners.delete(setToasts)
    }
  }, [])
  return toasts
}
