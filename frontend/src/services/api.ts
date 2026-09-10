/**
 * Typed API client.
 *
 * Every network call goes through `request`, so error handling is uniform: the
 * dev proxy answers 503 with a `detail` explaining how to start the backend,
 * and FastAPI answers 422 with validation detail. Both are surfaced verbatim
 * rather than replaced with a generic message.
 */

import type {
  AnalyzeResult,
  BackendStatus,
  Evaluation,
  ChatMessage,
  Health,
  IngestResult,
  KeywordCount,
  NewsDetail,
  NewsFilters,
  NewsPage,
  SnapshotInfo,
  Statistics,
  Stream,
  Taxonomy,
  TrendGranularity,
  TrendResponse,
} from '../types'

const BASE = '/api'

/**
 * Absolute URL of FastAPI's interactive docs.
 *
 * It cannot be a bare "/docs" link: the Vite dev server only proxies "/api", so
 * "/docs" would resolve to the SPA's catch-all route instead of the API docs. In
 * dev the backend is a different origin, in a production build it is the same
 * one, and this handles both.
 */
export const API_DOCS_URL: string = import.meta.env.DEV
  ? 'http://127.0.0.1:8000/docs'
  : '/docs'

export class ApiError extends Error {
  // Declared and assigned explicitly: TypeScript's `erasableSyntaxOnly` (on by
  // default in this Vite template) disallows constructor parameter properties.
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** Turn a FastAPI/proxy error body into a readable sentence. */
function describe(body: unknown, status: number, path: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    // 422 bodies carry a list of per-field errors.
    if (Array.isArray(detail)) {
      const parts = detail
        .map((item) => {
          if (item && typeof item === 'object') {
            const loc = Array.isArray((item as { loc?: unknown[] }).loc)
              ? (item as { loc: unknown[] }).loc.slice(1).join('.')
              : ''
            const msg = (item as { msg?: string }).msg ?? ''
            return loc ? `${loc}: ${msg}` : msg
          }
          return String(item)
        })
        .filter(Boolean)
      if (parts.length) return parts.join('; ')
    }
  }
  return `${path} returned ${status}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    throw new ApiError(
      'Cannot reach the API. Is the backend running? ' +
        'Start it with: cd backend && uvicorn app.main:app --reload',
      0,
    )
  }

  if (response.status === 204) return undefined as T

  const text = await response.text()
  let body: unknown = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = text
  }

  if (!response.ok) {
    throw new ApiError(describe(body, response.status, path), response.status)
  }
  return body as T
}

/** Build a query string, omitting empty values. */
function query(params: Record<string, unknown>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

export const api = {
  health: () => request<Health>('/health'),
  topics: () => request<Taxonomy[]>('/topics'),
  sentiments: () => request<Taxonomy[]>('/sentiments'),
  pipeline: () => request<BackendStatus[]>('/pipeline'),
  evaluation: () => request<Evaluation>('/evaluation'),

  statistics: () => request<Statistics>('/statistics'),
  trend: (granularity: TrendGranularity) =>
    request<TrendResponse>(`/statistics/trend${query({ granularity })}`),
  keywords: (limit = 60, topic?: string) =>
    request<KeywordCount[]>(`/statistics/keywords${query({ limit, topic })}`),

  news: (filters: NewsFilters = {}) =>
    request<NewsPage>(`/news${query(filters as Record<string, unknown>)}`),
  newsDetail: (id: number) => request<NewsDetail>(`/news/${id}`),
  newsMessages: (id: number, limit = 200) =>
    request<ChatMessage[]>(`/news/${id}/messages${query({ limit })}`),
  deleteNews: (id: number) => request<void>(`/news/${id}`, { method: 'DELETE' }),

  analyze: (
    payload: {
      title: string
      content: string
      source?: string
      url?: string
      published_at?: string
    },
    store = false,
  ) =>
    request<AnalyzeResult>(`/analyze${query({ store })}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  streams: () => request<Stream[]>('/streams'),
  snapshots: () => request<SnapshotInfo[]>('/ingest/snapshots'),
  ingest: (payload: {
    source: string
    collector?: string
    limit?: number
    save_snapshot?: boolean
    analyse?: boolean
  }) =>
    request<IngestResult>('/ingest/youtube', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  reanalyse: (stream_id?: number) =>
    request<Record<string, number>>('/ingest/reanalyse', {
      method: 'POST',
      body: JSON.stringify({ stream_id: stream_id ?? null }),
    }),
}
