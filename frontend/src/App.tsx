import { useEffect, useState } from 'react'

type Health = {
  status: string
  app: string
  version: string
  python: string
  nlp_backends: Record<string, string>
  llm_summarizer_configured: boolean
}

type Taxonomy = { slug: string; thai: string; english: string; color: string }

/**
 * Phase 1 connectivity page: proves the Vite dev server, the API proxy,
 * FastAPI and Thai font rendering all work together before any feature
 * code is written.
 */
export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [topics, setTopics] = useState<Taxonomy[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        const [h, t] = await Promise.all([
          fetch('/api/health'),
          fetch('/api/topics'),
        ])
        if (!h.ok || !t.ok) throw new Error(`API returned ${h.status}/${t.status}`)
        setHealth(await h.json())
        setTopics(await t.json())
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      }
    }
    void load()
  }, [])

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900 dark:text-white">
          Thai News Intelligence
        </h1>
        <p className="mt-1 text-slate-500 dark:text-slate-400">
          NLP-powered Thai News Topic &amp; Sentiment Analysis
        </p>
      </header>

      {error && (
        <div className="card border-negative/30 bg-negative-soft/60 p-4 dark:bg-negative/10">
          <p className="font-medium text-negative">Cannot reach the API</p>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
            {error} &mdash; start the backend with{' '}
            <code className="rounded bg-slate-200 px-1.5 py-0.5 text-xs dark:bg-slate-700">
              uvicorn app.main:app --reload
            </code>
          </p>
        </div>
      )}

      {!error && !health && (
        <div className="card animate-pulse p-6">
          <div className="h-4 w-40 rounded bg-slate-200 dark:bg-slate-700" />
          <div className="mt-3 h-3 w-64 rounded bg-slate-200 dark:bg-slate-700" />
        </div>
      )}

      {health && (
        <div className="card p-6">
          <div className="flex items-center gap-2">
            <span className="inline-block size-2.5 rounded-full bg-positive" />
            <span className="font-medium text-slate-900 dark:text-white">
              Backend connected
            </span>
            <span className="text-sm text-slate-500">
              v{health.version} &middot; Python {health.python}
            </span>
          </div>

          <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
            {Object.entries(health.nlp_backends).map(([key, value]) => (
              <div key={key}>
                <dt className="text-xs uppercase tracking-wide text-slate-400">
                  {key}
                </dt>
                <dd className="font-medium text-slate-700 dark:text-slate-200">
                  {value}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {topics.length > 0 && (
        <section className="mt-8">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            {topics.length} categories from the API
          </h2>
          <div className="flex flex-wrap gap-2">
            {topics.map((t) => (
              <span
                key={t.slug}
                className="rounded-full px-3 py-1 text-sm font-medium"
                style={{ backgroundColor: `${t.color}1a`, color: t.color }}
                lang="th"
              >
                {t.thai}
                <span className="ml-1.5 text-xs opacity-60">{t.english}</span>
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
