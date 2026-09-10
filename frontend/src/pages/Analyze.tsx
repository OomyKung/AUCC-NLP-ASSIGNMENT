/**
 * Analyze News: paste Thai text and run the real pipeline, or import a
 * YouTube live chat.
 *
 * Both buttons do real work -- no placeholders. The analyse form calls
 * POST /api/analyze; the import form calls POST /api/ingest/youtube, which
 * collects chat, stores it, windows it and analyses it.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ConfidenceBar,
  ErrorState,
  SentimentBadge,
  TopicBadge,
} from '../components/ui'
import { notify, useAsync } from '../hooks'
import { ApiError, api } from '../services/api'
import type { AnalyzeResult, IngestResult, SentimentSlug } from '../types'

const SAMPLE = {
  title: 'รถกระบะเสียหลักชนเสาไฟฟ้า มีผู้ได้รับบาดเจ็บ 3 ราย',
  content:
    'รถกระบะเสียหลักพุ่งชนเสาไฟฟ้าริมถนนมิตรภาพ จังหวัดขอนแก่น ทำให้มีผู้ได้รับบาดเจ็บ 3 ราย ' +
    'เจ้าหน้าที่กู้ภัยนำส่งโรงพยาบาลขอนแก่นทันที นายสมชาย ผู้เห็นเหตุการณ์ ระบุว่ารถเสียหลักเมื่อเวลา 14.30 น. ' +
    'ตำรวจคาดว่าสาเหตุมาจากการหลับในขณะขับขี่ ความเสียหายประเมินไว้ 250,000 บาท',
  source: 'ตัวอย่างข่าว',
}

export default function Analyze() {
  const topics = useAsync(() => api.topics(), [])
  const sentiments = useAsync(() => api.sentiments(), [])

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
          Analyze News
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400" lang="th">
          วางข้อความข่าวภาษาไทย หรือนำเข้าแชทสดจาก YouTube เพื่อวิเคราะห์
        </p>
      </header>

      <div className="grid gap-6 xl:grid-cols-2">
        <TextAnalysisPanel
          topicLabel={(slug) => topics.data?.find((t) => t.slug === slug)?.thai ?? slug}
          topicColor={(slug) => topics.data?.find((t) => t.slug === slug)?.color}
          sentimentLabel={(slug) =>
            sentiments.data?.find((s) => s.slug === slug)?.thai ?? slug
          }
        />
        <YouTubeImportPanel />
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Paste-and-analyse                                                          */
/* -------------------------------------------------------------------------- */

function TextAnalysisPanel({
  topicLabel,
  topicColor,
  sentimentLabel,
}: {
  topicLabel: (slug: string) => string
  topicColor: (slug: string) => string | undefined
  sentimentLabel: (slug: string) => string
}) {
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [source, setSource] = useState('')
  const [url, setUrl] = useState('')
  const [publishedAt, setPublishedAt] = useState('')
  const [store, setStore] = useState(true)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResult | null>(null)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const response = await api.analyze(
        {
          title: title.trim(),
          content: content.trim(),
          source: source.trim() || undefined,
          url: url.trim() || undefined,
          published_at: publishedAt ? new Date(publishedAt).toISOString() : undefined,
        },
        store,
      )
      setResult(response)
      notify(
        store && response.news_id
          ? 'วิเคราะห์และบันทึกเรียบร้อย'
          : 'วิเคราะห์เรียบร้อย',
        'success',
      )
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'วิเคราะห์ไม่สำเร็จ'
      setError(message)
      notify(message, 'error')
    } finally {
      setBusy(false)
    }
  }

  const fillSample = () => {
    setTitle(SAMPLE.title)
    setContent(SAMPLE.content)
    setSource(SAMPLE.source)
  }

  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="card space-y-3 p-5">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-slate-900 dark:text-white">
            วิเคราะห์ข้อความข่าว
          </h3>
          <button type="button" onClick={fillSample} className="btn-ghost text-xs">
            ใส่ตัวอย่าง
          </button>
        </div>

        <label className="block">
          <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
            หัวข้อข่าว <span className="text-negative">*</span>
          </span>
          <input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            required
            maxLength={500}
            className="input"
            placeholder="พาดหัวข่าว"
            lang="th"
          />
        </label>

        <label className="block">
          <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
            เนื้อหาข่าว <span className="text-negative">*</span>
          </span>
          <textarea
            value={content}
            onChange={(event) => setContent(event.target.value)}
            required
            rows={7}
            className="input resize-y"
            placeholder="วางเนื้อหาข่าวภาษาไทยที่นี่"
            lang="th"
          />
          <span className="mt-1 block text-right text-[11px] tabular-nums text-slate-400">
            {content.length.toLocaleString()} ตัวอักษร
          </span>
        </label>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              แหล่งที่มา
            </span>
            <input
              value={source}
              onChange={(event) => setSource(event.target.value)}
              className="input"
              placeholder="เช่น ไทยรัฐ"
              lang="th"
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              วันที่เผยแพร่
            </span>
            <input
              type="date"
              value={publishedAt}
              onChange={(event) => setPublishedAt(event.target.value)}
              className="input"
            />
          </label>
        </div>

        <label className="block">
          <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
            URL ต้นฉบับ
          </span>
          <input
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            className="input"
            placeholder="https://…"
          />
        </label>

        <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
          <input
            type="checkbox"
            checked={store}
            onChange={(event) => setStore(event.target.checked)}
            className="size-4 rounded border-slate-300 text-brand-600 focus:ring-brand-400"
          />
          <span lang="th">บันทึกผลลงฐานข้อมูล (แสดงบนแดชบอร์ด)</span>
        </label>

        <button
          type="submit"
          disabled={busy || !title.trim() || !content.trim()}
          className="btn-primary w-full"
        >
          {busy ? 'กำลังวิเคราะห์…' : 'Analyze News'}
        </button>
      </form>

      {error && <ErrorState message={error} />}

      {result && (
        <AnalysisResultCard
          result={result}
          topicLabel={topicLabel}
          topicColor={topicColor}
          sentimentLabel={sentimentLabel}
        />
      )}
    </div>
  )
}

function AnalysisResultCard({
  result,
  topicLabel,
  topicColor,
  sentimentLabel,
}: {
  result: AnalyzeResult
  topicLabel: (slug: string) => string
  topicColor: (slug: string) => string | undefined
  sentimentLabel: (slug: string) => string
}) {
  const topTopics = Object.entries(result.topic_probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 4)

  return (
    <div className="card space-y-4 p-5">
      <div className="flex flex-wrap items-center gap-2">
        <TopicBadge label={topicLabel(result.topic)} color={topicColor(result.topic)} />
        <SentimentBadge
          sentiment={result.sentiment}
          label={sentimentLabel(result.sentiment)}
          confidence={result.sentiment_confidence}
        />
        <span className="ml-auto text-xs tabular-nums text-slate-400">
          {result.processing_ms.toFixed(0)} ms
        </span>
      </div>

      {result.summary && (
        <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/50">
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            สรุป
          </p>
          <p className="text-sm text-slate-700 dark:text-slate-200" lang="th">
            {result.summary}
          </p>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Topic probability
          </p>
          <ul className="space-y-2">
            {topTopics.map(([slug, value]) => (
              <li key={slug}>
                <ConfidenceBar value={value} label={topicLabel(slug)} />
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Sentiment probability
          </p>
          <ul className="space-y-2">
            {Object.entries(result.sentiment_probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([slug, value]) => (
                <li key={slug}>
                  <ConfidenceBar
                    value={value}
                    tone={slug as SentimentSlug}
                    label={sentimentLabel(slug)}
                  />
                </li>
              ))}
          </ul>
        </div>
      </div>

      {result.keywords.length > 0 && (
        <div>
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Keywords
          </p>
          <ul className="flex flex-wrap gap-1.5">
            {result.keywords.map((keyword) => (
              <li
                key={keyword.word}
                className="rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                lang="th"
              >
                {keyword.word}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.entities.length > 0 && (
        <div>
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Named entities
          </p>
          <ul className="flex flex-wrap gap-1.5">
            {result.entities.map((entity, index) => (
              <li
                key={`${entity.text}-${index}`}
                className="rounded-md border border-slate-200 px-2 py-0.5 text-xs dark:border-slate-700"
              >
                <span lang="th">{entity.text}</span>
                <span className="ml-1.5 text-[10px] text-brand-600 dark:text-brand-400">
                  {entity.label}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <dl className="grid grid-cols-3 gap-3 border-t border-slate-100 pt-3 text-center dark:border-slate-700/60">
        <div>
          <dt className="text-[10px] uppercase text-slate-400">Tokens</dt>
          <dd className="font-semibold tabular-nums">{result.token_count}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase text-slate-400">Unique</dt>
          <dd className="font-semibold tabular-nums">{result.unique_token_count}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase text-slate-400">Stopwords</dt>
          <dd className="font-semibold tabular-nums">
            {result.stopword_removed_count}
          </dd>
        </div>
      </dl>

      {result.news_id && (
        <Link to={`/news/${result.news_id}`} className="btn-ghost w-full">
          ดูรายละเอียดเอกสารที่บันทึก →
        </Link>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* YouTube import                                                             */
/* -------------------------------------------------------------------------- */

function YouTubeImportPanel() {
  const snapshots = useAsync(() => api.snapshots(), [])
  const streams = useAsync(() => api.streams(), [])

  const [source, setSource] = useState('')
  const [collector, setCollector] = useState<'ytdlp' | 'file'>('ytdlp')
  const [limit, setLimit] = useState('2000')
  const [saveSnapshot, setSaveSnapshot] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<IngestResult | null>(null)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const response = await api.ingest({
        source: source.trim(),
        collector,
        limit: limit ? Number(limit) : undefined,
        save_snapshot: collector === 'ytdlp' && saveSnapshot,
        analyse: true,
      })
      setResult(response)
      notify(
        `นำเข้า ${response.stored.toLocaleString()} ข้อความ · สร้าง ${response.windows_created} ช่วงแชท`,
        'success',
      )
      snapshots.reload()
      streams.reload()
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'นำเข้าไม่สำเร็จ'
      setError(message)
      notify(message, 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="card space-y-3 p-5">
        <h3 className="font-semibold text-slate-900 dark:text-white">
          นำเข้าแชทสดจาก YouTube
        </h3>
        <p className="text-xs text-slate-500 dark:text-slate-400" lang="th">
          ใช้ yt-dlp ดึงแชท (ทั้งสตรีมสดและย้อนหลัง) โดยไม่ต้องใช้ API key
          หรือเลือกเล่นซ้ำจากไฟล์ที่บันทึกไว้แบบออฟไลน์
        </p>

        <div
          className="flex rounded-xl border border-slate-200 p-0.5 dark:border-slate-700"
          role="group"
          aria-label="แหล่งข้อมูล"
        >
          {(
            [
              { value: 'ytdlp', label: 'YouTube (yt-dlp)' },
              { value: 'file', label: 'ไฟล์ออฟไลน์' },
            ] as const
          ).map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => setCollector(option.value)}
              aria-pressed={collector === option.value}
              className={`flex-1 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                collector === option.value
                  ? 'bg-brand-600 text-white'
                  : 'text-slate-500 hover:bg-slate-50 dark:text-slate-400 dark:hover:bg-slate-800'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>

        {collector === 'ytdlp' ? (
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              URL หรือ Video ID
            </span>
            <input
              value={source}
              onChange={(event) => setSource(event.target.value)}
              required
              className="input"
              placeholder="https://www.youtube.com/watch?v=…"
            />
          </label>
        ) : (
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              ไฟล์ snapshot
            </span>
            <select
              value={source}
              onChange={(event) => setSource(event.target.value)}
              required
              className="input"
            >
              <option value="">เลือกไฟล์…</option>
              {snapshots.data?.map((item) => (
                <option key={item.file} value={item.file}>
                  {item.file} · {item.message_count ?? '?'} ข้อความ
                </option>
              ))}
            </select>
          </label>
        )}

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              จำนวนข้อความสูงสุด
            </span>
            <input
              type="number"
              min={1}
              max={50000}
              value={limit}
              onChange={(event) => setLimit(event.target.value)}
              className="input"
            />
          </label>
          {collector === 'ytdlp' && (
            <label className="flex items-end gap-2 pb-2 text-sm text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={saveSnapshot}
                onChange={(event) => setSaveSnapshot(event.target.checked)}
                className="size-4 rounded border-slate-300 text-brand-600 focus:ring-brand-400"
              />
              <span lang="th">บันทึก snapshot</span>
            </label>
          )}
        </div>

        <button
          type="submit"
          disabled={busy || !source.trim()}
          className="btn-primary w-full"
        >
          {busy ? 'กำลังนำเข้าและวิเคราะห์…' : 'นำเข้าและวิเคราะห์'}
        </button>
        {busy && collector === 'ytdlp' && (
          <p className="text-center text-xs text-slate-400" lang="th">
            การดึงแชทจาก YouTube อาจใช้เวลาหลายสิบวินาที
          </p>
        )}
      </form>

      {error && <ErrorState message={error} />}

      {result && (
        <div className="card space-y-2 p-5 text-sm">
          <p className="font-medium text-slate-900 dark:text-white" lang="th">
            {result.title ?? result.video_id}
          </p>
          <p className="text-xs text-slate-400" lang="th">
            {result.channel} · {result.is_live ? 'กำลังถ่ายทอดสด' : 'ย้อนหลัง'}
          </p>
          <dl className="grid grid-cols-2 gap-2 pt-2 sm:grid-cols-3">
            <Stat label="เก็บได้" value={result.collected} />
            <Stat label="บันทึกใหม่" value={result.stored} />
            <Stat label="ซ้ำ (ข้าม)" value={result.duplicates} />
            <Stat label="ช่วงแชท" value={result.windows_created} />
            <Stat label="วิเคราะห์" value={result.messages_scored} />
            <Stat label="กรองสแปม" value={result.messages_skipped_noise} />
          </dl>
          {result.snapshot && (
            <p className="pt-1 text-xs text-slate-400">snapshot: {result.snapshot}</p>
          )}
          <Link to="/" className="btn-ghost mt-2 w-full">
            ดูผลบนแดชบอร์ด →
          </Link>
        </div>
      )}

      {streams.data && streams.data.length > 0 && (
        <div className="card p-5">
          <h4 className="mb-3 text-sm font-semibold text-slate-900 dark:text-white">
            สตรีมที่เก็บข้อมูลแล้ว ({streams.data.length})
          </h4>
          <ul className="scroll-thin max-h-64 divide-y divide-slate-100 overflow-y-auto text-sm dark:divide-slate-700/60">
            {streams.data.map((stream) => (
              <li key={stream.id} className="py-2">
                <p className="truncate text-slate-700 dark:text-slate-200" lang="th">
                  {stream.title ?? stream.video_id}
                </p>
                <p className="text-[11px] tabular-nums text-slate-400" lang="th">
                  {stream.channel} · {stream.message_count.toLocaleString()} ข้อความ ·{' '}
                  {stream.window_count} ช่วง · {stream.collector}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-wide text-slate-400" lang="th">
        {label}
      </dt>
      <dd className="font-semibold tabular-nums text-slate-800 dark:text-slate-100">
        {value.toLocaleString()}
      </dd>
    </div>
  )
}
