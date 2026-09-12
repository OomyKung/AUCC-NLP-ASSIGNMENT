/**
 * News detail: the full document plus every explainable NLP output.
 *
 * This is where a reader can check the model's work: the probability tables,
 * the tokenisation, the entities, and -- for a chat window -- the individual
 * messages the verdict was aggregated from.
 */

import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  ConfidenceBar,
  ErrorState,
  LoadingState,
  SentimentBadge,
  TopicBadge,
  formatDateTime,
} from '../components/ui'
import { useAsync } from '../hooks'
import { isUnclear } from '../lib/sentiment'
import { api } from '../services/api'
import type { SentimentSlug, Taxonomy } from '../types'

export default function NewsDetail() {
  const { id } = useParams<{ id: string }>()
  const newsId = Number(id)

  const news = useAsync(() => api.newsDetail(newsId), [newsId])
  const topics = useAsync(() => api.topics(), [])
  const sentiments = useAsync(() => api.sentiments(), [])
  const [showMessages, setShowMessages] = useState(false)

  if (news.error) return <ErrorState message={news.error} onRetry={news.reload} />
  if (news.loading && !news.data) return <LoadingState rows={3} />

  const item = news.data
  if (!item) return null

  const analysis = item.analysis
  const topicLabel = (slug: string) =>
    topics.data?.find((t) => t.slug === slug)?.thai ?? slug
  const topicColor = (slug: string) => topics.data?.find((t) => t.slug === slug)?.color
  const sentimentLabel = (slug: string) =>
    sentiments.data?.find((s) => s.slug === slug)?.thai ?? slug

  const isWindow = item.source_type === 'chat_window'

  return (
    <div className="space-y-5">
      <Link
        to="/explorer"
        className="inline-flex items-center gap-1 text-sm text-slate-500 hover:text-brand-600 dark:text-slate-400 dark:hover:text-brand-400"
      >
        ← กลับไปหน้าสำรวจข่าว
      </Link>

      {/* ----------------------------------------------------------- header */}
      <article className="card p-6">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <TopicBadge label={topicLabel(item.topic)} color={topicColor(item.topic)} />
          <SentimentBadge
            sentiment={item.sentiment}
            label={sentimentLabel(item.sentiment)}
            confidence={item.sentiment_confidence}
          />
          <span className="text-xs text-slate-400">{formatDateTime(item.published_at)}</span>
        </div>

        <h2
          className="text-xl font-semibold leading-snug text-slate-900 dark:text-white"
          lang="th"
        >
          {item.title}
        </h2>

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
          <span lang="th">แหล่งที่มา: {item.source}</span>
          {item.message_count != null && (
            <span className="tabular-nums">{item.message_count} ข้อความ</span>
          )}
          {item.window_start && item.window_end && (
            <span>
              ช่วง {formatDateTime(item.window_start)} – {formatDateTime(item.window_end)}
            </span>
          )}
          {item.url && (
            <a
              href={item.url}
              target="_blank"
              rel="noreferrer noopener"
              className="text-brand-600 hover:underline dark:text-brand-400"
            >
              เปิดต้นฉบับ ↗
            </a>
          )}
        </div>

        {item.summary && (
          <div className="mt-5 rounded-xl border-l-4 border-brand-400 bg-brand-50/60 p-4 dark:bg-brand-500/10">
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-brand-700 dark:text-brand-300">
              สรุปโดยระบบ (Extractive Summary)
            </p>
            <p className="text-sm text-slate-700 dark:text-slate-200" lang="th">
              {item.summary}
            </p>
          </div>
        )}

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <ConfidenceBar
            value={item.topic_confidence}
            label={`ความมั่นใจหัวข้อ · ${topicLabel(item.topic)}`}
          />
          <ConfidenceBar
            value={item.sentiment_confidence}
            tone={item.sentiment}
            label={`ความมั่นใจความรู้สึก · ${sentimentLabel(item.sentiment)}`}
          />
        </div>
      </article>

      {/* ------------------------------------------------- probability tables */}
      {analysis && (
        <div className="grid gap-5 lg:grid-cols-2">
          <ProbabilityTable
            title="Topic Probability"
            subtitle="การกระจายความน่าจะเป็นทั้ง 15 หมวด"
            probabilities={analysis.topic_probabilities}
            labels={topics.data ?? []}
            limit={6}
          />
          <ProbabilityTable
            title="Sentiment Probability"
            subtitle={
              isWindow
                ? 'สัดส่วนความรู้สึกของข้อความในช่วงนี้'
                : 'การกระจายความน่าจะเป็นของความรู้สึก'
            }
            probabilities={analysis.sentiment_probabilities}
            labels={sentiments.data ?? []}
            tone
          />
        </div>
      )}

      {/* ---------------------------------------------------------- keywords */}
      {item.keywords.length > 0 && (
        <section className="card p-5">
          <h3 className="mb-3 font-semibold text-slate-900 dark:text-white">
            คำสำคัญ (Keywords)
          </h3>
          <ul className="flex flex-wrap gap-2">
            {item.keywords.map((keyword) => (
              <li
                key={keyword.word}
                className="flex items-baseline gap-1.5 rounded-lg bg-slate-100 px-2.5 py-1 text-sm dark:bg-slate-800"
              >
                <span className="text-slate-700 dark:text-slate-200" lang="th">
                  {keyword.word}
                </span>
                <span className="text-[10px] tabular-nums text-slate-400">
                  {keyword.score.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ---------------------------------------------------------- entities */}
      {analysis && analysis.entities.length > 0 && (
        <section className="card p-5">
          <h3 className="mb-3 font-semibold text-slate-900 dark:text-white">
            Named Entities
          </h3>
          <ul className="flex flex-wrap gap-2">
            {analysis.entities.map((entity, index) => (
              <li
                key={`${entity.text}-${index}`}
                className="rounded-lg border border-slate-200 px-2.5 py-1 text-sm dark:border-slate-700"
              >
                <span className="text-slate-700 dark:text-slate-200" lang="th">
                  {entity.text}
                </span>
                <span className="ml-2 rounded bg-brand-50 px-1.5 py-0.5 text-[10px] font-medium text-brand-700 dark:bg-brand-500/15 dark:text-brand-300">
                  {entity.label}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* --------------------------------------------------------- NLP stats */}
      {analysis && (
        <section className="card p-5">
          <h3 className="mb-4 font-semibold text-slate-900 dark:text-white">
            NLP Analysis
          </h3>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Metric label="Tokens" value={analysis.token_count} />
            <Metric label="Unique words" value={analysis.unique_token_count} />
            <Metric label="Stopwords removed" value={analysis.stopword_removed_count} />
            <Metric label="Processing" value={`${analysis.processing_ms.toFixed(0)} ms`} />
          </dl>

          <div className="mt-5">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Tokenized words ({analysis.tokens.length})
            </p>
            <div className="scroll-thin max-h-40 overflow-y-auto rounded-xl bg-slate-50 p-3 dark:bg-slate-900/50">
              <div className="flex flex-wrap gap-1">
                {analysis.tokens.slice(0, 300).map((token, index) => (
                  <span
                    key={`${token}-${index}`}
                    className="rounded bg-white px-1.5 py-0.5 text-xs text-slate-600 shadow-sm dark:bg-slate-800 dark:text-slate-300"
                    lang="th"
                  >
                    {token}
                  </span>
                ))}
              </div>
              {analysis.tokens.length > 300 && (
                <p className="mt-2 text-xs text-slate-400">
                  แสดง 300 จาก {analysis.tokens.length} โทเคน
                </p>
              )}
            </div>
          </div>

          {Object.keys(analysis.model_versions).length > 0 && (
            <div className="mt-4 border-t border-slate-100 pt-3 dark:border-slate-700/60">
              <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Models used
              </p>
              <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
                {Object.entries(analysis.model_versions).map(([stage, name]) => (
                  <li key={stage}>
                    <span className="text-slate-400">{stage}:</span>{' '}
                    <span className="font-medium text-slate-600 dark:text-slate-300">
                      {name}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {/* ------------------------------------------------------ full content */}
      <section className="card p-5">
        <h3 className="mb-3 font-semibold text-slate-900 dark:text-white">
          {isWindow ? 'ข้อความในช่วงนี้ (ต้นฉบับ)' : 'เนื้อหาต้นฉบับ'}
        </h3>
        <div className="scroll-thin max-h-96 overflow-y-auto whitespace-pre-wrap rounded-xl bg-slate-50 p-4 text-sm leading-relaxed text-slate-700 dark:bg-slate-900/50 dark:text-slate-300">
          <p lang="th">{item.content}</p>
        </div>
      </section>

      {/* -------------------------------------------- per-message drill-down */}
      {isWindow && (
        <section className="card p-5">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-slate-900 dark:text-white">
              ความรู้สึกรายข้อความ
            </h3>
            <button
              type="button"
              onClick={() => setShowMessages((value) => !value)}
              className="btn-ghost"
              aria-expanded={showMessages}
            >
              {showMessages ? 'ซ่อน' : 'แสดงข้อความ'}
            </button>
          </div>
          {showMessages && <MessageList newsId={newsId} />}
        </section>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */

function MessageList({ newsId }: { newsId: number }) {
  const messages = useAsync(() => api.newsMessages(newsId, 200), [newsId])

  if (messages.error) {
    return <ErrorState message={messages.error} onRetry={messages.reload} />
  }
  if (messages.loading && !messages.data) {
    return <p className="mt-3 text-sm text-slate-400">กำลังโหลด…</p>
  }
  if (!messages.data?.length) {
    return <p className="mt-3 text-sm text-slate-400">ไม่พบข้อความ</p>
  }

  return (
    <ul className="scroll-thin mt-3 max-h-96 divide-y divide-slate-100 overflow-y-auto dark:divide-slate-700/60">
      {messages.data.map((message) => (
        <li key={message.id} className="flex items-start gap-3 py-2">
          <span
            className={`mt-1.5 size-2 shrink-0 rounded-full ${
              message.sentiment === 'positive'
                ? 'bg-positive'
                : message.sentiment === 'negative'
                  ? 'bg-negative'
                  : 'bg-neutral'
            } ${isUnclear(message.sentiment_confidence) ? 'opacity-30' : ''}`}
            title={
              isUnclear(message.sentiment_confidence)
                ? 'ความมั่นใจต่ำเกินกว่าจะระบุ'
                : (message.sentiment ?? 'ไม่ระบุ')
            }
          />
          <div className="min-w-0 flex-1">
            <p className="text-sm text-slate-700 dark:text-slate-200" lang="th">
              {message.text}
            </p>
            <p className="mt-0.5 text-[11px] text-slate-400">
              {message.author ?? 'ไม่ทราบชื่อ'} · {formatDateTime(message.published_at)}
              {/* Under the measured threshold the classifier is right 47.5% of
                  the time, so the label is not presented as a finding. */}
              {message.sentiment &&
                (isUnclear(message.sentiment_confidence)
                  ? ' · ไม่ชัดเจน'
                  : ` · ${message.sentiment} ${Math.round(
                      (message.sentiment_confidence ?? 0) * 100,
                    )}%`)}
            </p>
          </div>
        </li>
      ))}
    </ul>
  )
}

function ProbabilityTable({
  title,
  subtitle,
  probabilities,
  labels,
  limit,
  tone = false,
}: {
  title: string
  subtitle?: string
  probabilities: Record<string, number>
  labels: Taxonomy[]
  limit?: number
  tone?: boolean
}) {
  const rows = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, limit ?? undefined)

  if (!rows.length) return null

  return (
    <section className="card p-5">
      <h3 className="font-semibold text-slate-900 dark:text-white">{title}</h3>
      {subtitle && (
        <p className="mb-3 mt-0.5 text-xs text-slate-500 dark:text-slate-400" lang="th">
          {subtitle}
        </p>
      )}
      <ul className="space-y-2.5">
        {rows.map(([slug, value]) => (
          <li key={slug}>
            <ConfidenceBar
              value={value}
              tone={tone ? (slug as SentimentSlug) : 'brand'}
              label={labels.find((item) => item.slug === slug)?.thai ?? slug}
            />
          </li>
        ))}
      </ul>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="mt-0.5 text-lg font-semibold tabular-nums text-slate-800 dark:text-slate-100">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </dd>
    </div>
  )
}
