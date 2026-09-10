/**
 * Small reusable presentation components.
 *
 * Sentiment and topic are never communicated by colour alone: every badge
 * carries its Thai label, and confidence bars carry a numeric value. That is
 * what keeps the dashboard readable for colour-blind viewers and in print.
 */

import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { Keyword, NewsListItem, SentimentSlug } from '../types'
import { dismissToast, useToasts } from '../hooks'

/* -------------------------------------------------------------------------- */
/* Badges                                                                     */
/* -------------------------------------------------------------------------- */

const SENTIMENT_STYLES: Record<SentimentSlug, string> = {
  positive:
    'bg-teal-50 text-teal-800 ring-teal-600/20 dark:bg-teal-500/10 dark:text-teal-300 dark:ring-teal-400/30',
  neutral:
    'bg-slate-100 text-slate-700 ring-slate-500/20 dark:bg-slate-500/10 dark:text-slate-300 dark:ring-slate-400/30',
  negative:
    'bg-red-50 text-red-800 ring-red-600/20 dark:bg-red-500/10 dark:text-red-300 dark:ring-red-400/30',
}

const SENTIMENT_DOT: Record<SentimentSlug, string> = {
  positive: 'bg-positive',
  neutral: 'bg-neutral',
  negative: 'bg-negative',
}

export function SentimentBadge({
  sentiment,
  label,
  confidence,
}: {
  sentiment: SentimentSlug
  label?: string
  confidence?: number
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${SENTIMENT_STYLES[sentiment]}`}
    >
      <span className={`size-1.5 rounded-full ${SENTIMENT_DOT[sentiment]}`} />
      <span lang="th">{label ?? sentiment}</span>
      {confidence !== undefined && (
        <span className="tabular-nums opacity-70">
          {Math.round(confidence * 100)}%
        </span>
      )}
    </span>
  )
}

export function TopicBadge({
  label,
  color,
  count,
}: {
  label: string
  color?: string
  count?: number
}) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium"
      style={
        color
          ? { backgroundColor: `${color}1a`, color, boxShadow: `inset 0 0 0 1px ${color}33` }
          : undefined
      }
    >
      <span lang="th">{label}</span>
      {count !== undefined && <span className="tabular-nums opacity-70">{count}</span>}
    </span>
  )
}

/* -------------------------------------------------------------------------- */
/* Confidence                                                                 */
/* -------------------------------------------------------------------------- */

export function ConfidenceBar({
  value,
  tone = 'brand',
  label,
}: {
  value: number
  tone?: 'brand' | SentimentSlug
  label?: string
}) {
  const percent = Math.max(0, Math.min(100, value * 100))
  const fill =
    tone === 'brand'
      ? 'bg-brand-500'
      : tone === 'positive'
        ? 'bg-positive'
        : tone === 'negative'
          ? 'bg-negative'
          : 'bg-neutral'

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
        <span lang="th">{label}</span>
        <span className="tabular-nums font-medium">{percent.toFixed(0)}%</span>
      </div>
      <div
        className="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-700/60"
        role="meter"
        aria-valuenow={Math.round(percent)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label ?? 'confidence'}
      >
        <div
          className={`h-full rounded-full ${fill} transition-[width] duration-500`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Cards                                                                      */
/* -------------------------------------------------------------------------- */

export function StatCard({
  label,
  value,
  hint,
  tone = 'brand',
  icon,
}: {
  label: string
  value: string | number
  hint?: string
  tone?: 'brand' | SentimentSlug
  icon?: ReactNode
}) {
  const accent =
    tone === 'brand'
      ? 'text-brand-600 dark:text-brand-400'
      : tone === 'positive'
        ? 'text-positive'
        : tone === 'negative'
          ? 'text-negative'
          : 'text-neutral dark:text-slate-400'

  return (
    <div className="card card-hover p-5">
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          {label}
        </p>
        {icon && <span className={accent}>{icon}</span>}
      </div>
      <p
        className={`mt-2 text-3xl font-semibold tabular-nums tracking-tight ${accent}`}
        lang="th"
      >
        {typeof value === 'number' ? value.toLocaleString() : value}
      </p>
      {hint && (
        <p className="mt-1 text-xs text-slate-400" lang="th">
          {hint}
        </p>
      )}
    </div>
  )
}

export function ChartCard({
  title,
  subtitle,
  actions,
  children,
  footer,
}: {
  title: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
  footer?: ReactNode
}) {
  return (
    <section className="card p-5">
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-semibold text-slate-900 dark:text-white">{title}</h2>
          {subtitle && (
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              {subtitle}
            </p>
          )}
        </div>
        {actions}
      </header>
      {children}
      {footer && <div className="mt-3">{footer}</div>}
    </section>
  )
}

/* -------------------------------------------------------------------------- */
/* News card                                                                  */
/* -------------------------------------------------------------------------- */

const SOURCE_TYPE_LABEL: Record<string, string> = {
  chat_window: 'ช่วงแชทสด',
  article: 'บทความ',
  dataset: 'ชุดข้อมูล',
}

export function NewsCard({
  item,
  topicLabel,
  topicColor,
  sentimentLabel,
}: {
  item: NewsListItem
  topicLabel: string
  topicColor?: string
  sentimentLabel: string
}) {
  return (
    <article className="card card-hover flex flex-col p-5">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <TopicBadge label={topicLabel} color={topicColor} />
        <SentimentBadge
          sentiment={item.sentiment}
          label={sentimentLabel}
          confidence={item.sentiment_confidence}
        />
        <span className="ml-auto text-[11px] text-slate-400" lang="th">
          {SOURCE_TYPE_LABEL[item.source_type] ?? item.source_type}
        </span>
      </div>

      <h3 className="font-medium leading-snug text-slate-900 dark:text-white" lang="th">
        <Link to={`/news/${item.id}`} className="hover:text-brand-600 dark:hover:text-brand-400">
          {item.title}
        </Link>
      </h3>

      {item.summary && (
        <p
          className="mt-2 line-clamp-3 text-sm text-slate-600 dark:text-slate-300"
          lang="th"
        >
          {item.summary}
        </p>
      )}

      {item.keywords.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-1.5">
          {item.keywords.slice(0, 6).map((keyword: Keyword) => (
            <li
              key={keyword.word}
              className="rounded-md bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
              lang="th"
            >
              {keyword.word}
            </li>
          ))}
        </ul>
      )}

      <footer className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-100 pt-3 text-xs text-slate-400 dark:border-slate-700/60">
        <span lang="th">{item.source}</span>
        <span>{formatDate(item.published_at)}</span>
        {item.message_count != null && (
          <span className="tabular-nums">{item.message_count} ข้อความ</span>
        )}
        {item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noreferrer noopener"
            className="ml-auto text-brand-600 hover:underline dark:text-brand-400"
          >
            ต้นฉบับ ↗
          </a>
        )}
      </footer>
    </article>
  )
}

/* -------------------------------------------------------------------------- */
/* States                                                                     */
/* -------------------------------------------------------------------------- */

export function LoadingState({ rows = 3, label }: { rows?: number; label?: string }) {
  return (
    <div className="space-y-3" role="status" aria-live="polite">
      <span className="sr-only">{label ?? 'กำลังโหลด'}</span>
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="card animate-pulse p-5">
          <div className="h-3 w-24 rounded bg-slate-200 dark:bg-slate-700" />
          <div className="mt-3 h-4 w-3/4 rounded bg-slate-200 dark:bg-slate-700" />
          <div className="mt-2 h-3 w-full rounded bg-slate-100 dark:bg-slate-700/60" />
          <div className="mt-1.5 h-3 w-5/6 rounded bg-slate-100 dark:bg-slate-700/60" />
        </div>
      ))}
    </div>
  )
}

export function ChartSkeleton({ height = 260 }: { height?: number }) {
  return (
    <div
      className="animate-pulse rounded-xl bg-slate-100 dark:bg-slate-800/60"
      style={{ height }}
      role="status"
      aria-label="กำลังโหลดกราฟ"
    />
  )
}

export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string
  hint?: string
  action?: ReactNode
}) {
  return (
    <div className="card flex flex-col items-center gap-2 px-6 py-14 text-center">
      <div className="mb-1 grid size-11 place-items-center rounded-full bg-slate-100 text-slate-400 dark:bg-slate-800">
        <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="1.8">
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-3.5-3.5" strokeLinecap="round" />
        </svg>
      </div>
      <p className="font-medium text-slate-700 dark:text-slate-200" lang="th">
        {title}
      </p>
      {hint && (
        <p className="max-w-md text-sm text-slate-500 dark:text-slate-400" lang="th">
          {hint}
        </p>
      )}
      {action}
    </div>
  )
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string
  onRetry?: () => void
}) {
  return (
    <div className="card border-negative/30 bg-negative-soft/50 p-5 dark:bg-negative/10">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full bg-negative/15 text-negative">
          !
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-medium text-negative">เกิดข้อผิดพลาด</p>
          <p className="mt-1 break-words text-sm text-slate-600 dark:text-slate-300">
            {message}
          </p>
          {onRetry && (
            <button type="button" onClick={onRetry} className="btn-ghost mt-3">
              ลองอีกครั้ง
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Pagination                                                                 */
/* -------------------------------------------------------------------------- */

export function Pagination({
  page,
  pages,
  total,
  onChange,
}: {
  page: number
  pages: number
  total: number
  onChange: (page: number) => void
}) {
  if (pages <= 1) {
    return (
      <p className="text-center text-xs text-slate-400 tabular-nums">
        {total.toLocaleString()} รายการ
      </p>
    )
  }

  // A compact window around the current page, so 39 pages does not wrap.
  const numbers: (number | '…')[] = []
  const push = (value: number | '…') => numbers.push(value)
  const window = 1
  for (let index = 1; index <= pages; index += 1) {
    if (
      index === 1 ||
      index === pages ||
      (index >= page - window && index <= page + window)
    ) {
      push(index)
    } else if (numbers[numbers.length - 1] !== '…') {
      push('…')
    }
  }

  return (
    <nav className="flex flex-wrap items-center justify-center gap-1.5" aria-label="แบ่งหน้า">
      <button
        type="button"
        className="btn-ghost px-2.5 py-1.5"
        onClick={() => onChange(page - 1)}
        disabled={page <= 1}
        aria-label="หน้าก่อนหน้า"
      >
        ←
      </button>
      {numbers.map((value, index) =>
        value === '…' ? (
          <span key={`gap-${index}`} className="px-1 text-slate-400">
            …
          </span>
        ) : (
          <button
            key={value}
            type="button"
            onClick={() => onChange(value)}
            aria-current={value === page ? 'page' : undefined}
            className={
              value === page
                ? 'rounded-xl bg-brand-600 px-3 py-1.5 text-sm font-medium text-white tabular-nums'
                : 'btn-ghost px-3 py-1.5 tabular-nums'
            }
          >
            {value}
          </button>
        ),
      )}
      <button
        type="button"
        className="btn-ghost px-2.5 py-1.5"
        onClick={() => onChange(page + 1)}
        disabled={page >= pages}
        aria-label="หน้าถัดไป"
      >
        →
      </button>
      <span className="ml-2 text-xs text-slate-400 tabular-nums">
        {total.toLocaleString()} รายการ
      </span>
    </nav>
  )
}

/* -------------------------------------------------------------------------- */
/* Keyword cloud                                                              */
/* -------------------------------------------------------------------------- */

export function KeywordCloud({
  keywords,
  onSelect,
}: {
  keywords: { word: string; count: number }[]
  onSelect?: (word: string) => void
}) {
  if (!keywords.length) {
    return <p className="text-sm text-slate-400">ยังไม่มีคำสำคัญ</p>
  }

  const counts = keywords.map((item) => item.count)
  const max = Math.max(...counts)
  const min = Math.min(...counts)
  const span = Math.max(max - min, 1)

  return (
    <ul className="flex flex-wrap items-baseline gap-x-3 gap-y-2">
      {keywords.map((item) => {
        // Size AND opacity both encode frequency, so the cloud is readable
        // without relying on size discrimination alone.
        const weight = (item.count - min) / span
        const size = 0.8 + weight * 1.0
        return (
          <li key={item.word}>
            <button
              type="button"
              onClick={onSelect ? () => onSelect(item.word) : undefined}
              disabled={!onSelect}
              title={`${item.word} · ${item.count} เอกสาร`}
              lang="th"
              className={`rounded-md leading-tight text-slate-700 tabular-nums dark:text-slate-200 ${
                onSelect ? 'hover:text-brand-600 dark:hover:text-brand-400' : 'cursor-default'
              }`}
              style={{ fontSize: `${size}rem`, opacity: 0.55 + weight * 0.45 }}
            >
              {item.word}
            </button>
          </li>
        )
      })}
    </ul>
  )
}

/* -------------------------------------------------------------------------- */
/* Toasts                                                                     */
/* -------------------------------------------------------------------------- */

export function ToastHost() {
  const toasts = useToasts()
  if (!toasts.length) return null

  const tones: Record<string, string> = {
    success: 'border-positive/30 bg-positive-soft text-teal-900 dark:bg-positive/15 dark:text-teal-100',
    error: 'border-negative/30 bg-negative-soft text-red-900 dark:bg-negative/15 dark:text-red-100',
    info: 'border-slate-200 bg-white text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100',
  }

  return (
    <div
      className="pointer-events-none fixed inset-x-3 bottom-3 z-50 flex flex-col items-center gap-2 sm:inset-x-auto sm:right-4 sm:items-end"
      aria-live="polite"
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`pointer-events-auto w-full max-w-sm rounded-xl border px-4 py-2.5 text-sm shadow-lg ${tones[toast.tone]}`}
        >
          <div className="flex items-start gap-3">
            <p className="min-w-0 flex-1 break-words" lang="th">
              {toast.message}
            </p>
            <button
              type="button"
              onClick={() => dismissToast(toast.id)}
              className="shrink-0 opacity-50 hover:opacity-100"
              aria-label="ปิด"
            >
              ✕
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

export function formatDate(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleDateString('th-TH', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  })
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString('th-TH', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  })
}
