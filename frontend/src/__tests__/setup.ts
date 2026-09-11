/**
 * Test environment setup.
 *
 * `fetch` is routed to fixtures captured from the *real* running API, so these
 * tests exercise the components against genuine response shapes rather than
 * hand-written mocks that can drift from the backend.
 */

import '@testing-library/dom'
import { afterEach, beforeEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

import statistics from './fixture-statistics.json'
import news from './fixture-news.json'
import topics from './fixture-topics.json'
import sentiments from './fixture-sentiments.json'
import trend from './fixture-trend.json'
import pipeline from './fixture-pipeline.json'
import detail from './fixture-detail.json'
import evaluation from './fixture-evaluation.json'
import programmes from './fixture-programmes.json'
import programme from './fixture-programme.json'

export const fixtures = {
  statistics,
  news,
  topics,
  sentiments,
  trend,
  pipeline,
  detail,
  evaluation,
  programmes,
  programme,
}

/** Map a request path to its fixture. */
function resolve(path: string): unknown {
  if (path.startsWith('/api/statistics/trend')) return trend
  if (path.startsWith('/api/statistics/keywords')) return statistics.top_keywords
  if (path.startsWith('/api/statistics')) return statistics
  if (path.startsWith('/api/topics')) return topics
  if (path.startsWith('/api/sentiments')) return sentiments
  if (path.startsWith('/api/pipeline')) return pipeline
  if (path.startsWith('/api/evaluation')) return evaluation
  if (/^\/api\/broadcast\/programmes\/.+/.test(path)) return programme
  if (path.startsWith('/api/broadcast/programmes')) return programmes
  if (path.startsWith('/api/broadcast/segments'))
    return { total: programme.segments.length, limit: 100, offset: 0, items: programme.segments }
  if (path.startsWith('/api/streams')) return []
  if (path.startsWith('/api/ingest/snapshots')) return []
  if (/^\/api\/news\/\d+\/messages/.test(path)) return []
  if (/^\/api\/news\/\d+/.test(path)) return detail
  if (path.startsWith('/api/news')) return news
  if (path.startsWith('/api/health')) {
    return {
      status: 'ok',
      app: 'Thai News Intelligence',
      version: '1.0.0',
      python: '3.14.6',
      nlp_backends: {},
      llm_summarizer_configured: false,
    }
  }
  return null
}

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path = typeof input === 'string' ? input : String(input)
      const body = resolve(path)
      if (body === null) {
        return new Response(JSON.stringify({ detail: `no fixture for ${path}` }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }),
  )

  // Recharts measures its container; jsdom reports zero, which would render
  // nothing. Give every element a real size.
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
    configurable: true,
    value: 800,
  })
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
    configurable: true,
    value: 400,
  })
  window.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})
