/**
 * News Timeline: the stories found inside a programme's audio.
 *
 * This page answers the question the chat pages cannot: *what was actually
 * said*, when, and where to click to hear it. Each card is one detected story
 * with the frame captured at that moment, and the whole card is a link into the
 * video at the right second — so nobody has to scrub a four-hour stream looking
 * for the bit they want.
 *
 * Two deliberate choices:
 *
 * 1. Every card shows *why* the segmenter cut there ("lexical-cohesion",
 *    "topic-change", "music-break"). Boundaries are inferred, not given, so the
 *    page says how confident it is rather than presenting a split as fact.
 * 2. The topic strip along the top is the whole programme at a glance, with each
 *    story coloured by topic. It is what makes a change of subject visible as a
 *    shape rather than as a list.
 */

import { useMemo, useState } from 'react'
import { ErrorState, LoadingState } from '../components/ui'
import { useAsync } from '../hooks'
import { api } from '../services/api'
import type { NewsSegment, Programme } from '../types'

/** How each boundary signal is explained to a reader, in Thai. */
const REASON_LABEL: Record<string, string> = {
  'lexical-cohesion': 'คำศัพท์เปลี่ยน',
  'topic-change': 'หัวข้อเปลี่ยน',
  'music-break': 'ดนตรีคั่น',
  'programme-start': 'เริ่มรายการ',
}

function minutes(ms: number): string {
  const total = Math.round(ms / 1000)
  const m = Math.floor(total / 60)
  const s = total % 60
  return m > 0 ? `${m} นาที ${s} วิ` : `${s} วิ`
}

export default function TimelinePage() {
  const programmes = useAsync(() => api.programmes(), [])
  const [videoId, setVideoId] = useState<string | null>(null)

  const selected = videoId ?? programmes.data?.[0]?.video_id ?? null
  const programme = useAsync(
    () => (selected ? api.programme(selected) : Promise.resolve(null)),
    [selected],
  )

  if (programmes.error) {
    return <ErrorState message={programmes.error} onRetry={programmes.reload} />
  }
  if (programmes.loading && !programmes.data) return <LoadingState rows={3} />

  const list = programmes.data ?? []
  if (list.length === 0) return <NothingAnalysed />

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
          ไทม์ไลน์ข่าว (News Timeline)
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400" lang="th">
          แบ่งช่วงข่าวจากเสียงพูดในคลิป ระบุหัวข้อของแต่ละช่วง
          และกดเพื่อข้ามไปยังวินาทีที่พูดถึงเรื่องนั้นได้ทันที
        </p>
      </header>

      {list.length > 1 && (
        <ProgrammePicker
          programmes={list}
          selected={selected}
          onSelect={setVideoId}
        />
      )}

      {programme.error ? (
        <ErrorState message={programme.error} onRetry={programme.reload} />
      ) : programme.loading && !programme.data ? (
        <LoadingState rows={4} />
      ) : programme.data ? (
        <ProgrammeTimeline programme={programme.data} />
      ) : null}
    </div>
  )
}

/* -------------------------------------------------------------------------- */

function NothingAnalysed() {
  return (
    <div className="card border-l-4 border-amber-400 p-5">
      <h3 className="font-semibold text-slate-900 dark:text-white">
        ยังไม่มีคลิปที่ถอดเสียงไว้
      </h3>
      <p className="mt-2 text-sm text-slate-600 dark:text-slate-300" lang="th">
        หน้านี้แสดงผลจากเสียงพูดในคลิปข่าว ต้องถอดเสียงและแบ่งช่วงก่อน
      </p>
      <pre className="mt-4 overflow-x-auto rounded-xl bg-slate-50 p-3 text-xs text-slate-600 dark:bg-slate-900/50 dark:text-slate-300">
        {'python analyse_video.py <YouTube URL>'}
      </pre>
    </div>
  )
}

function ProgrammePicker({
  programmes,
  selected,
  onSelect,
}: {
  programmes: Programme[]
  selected: string | null
  onSelect: (id: string) => void
}) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="เลือกรายการ">
      {programmes.map((p) => (
        <button
          key={p.video_id}
          type="button"
          onClick={() => onSelect(p.video_id)}
          aria-pressed={p.video_id === selected}
          className={`rounded-xl border px-3 py-2 text-left text-xs ${
            p.video_id === selected
              ? 'border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-200'
              : 'border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800'
          }`}
        >
          <span className="block max-w-[22rem] truncate font-medium">
            {p.title ?? p.video_id}
          </span>
          <span className="text-[11px] text-slate-400">
            {p.segment_count} ช่วงข่าว
          </span>
        </button>
      ))}
    </div>
  )
}

function ProgrammeTimeline({ programme }: { programme: Programme }) {
  const segments = programme.segments
  const total = useMemo(
    () => segments.reduce((max, s) => Math.max(max, s.end_ms), 0),
    [segments],
  )

  return (
    <>
      <section className="card p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="truncate font-semibold text-slate-900 dark:text-white">
              {programme.title ?? programme.video_id}
            </h3>
            {programme.channel && (
              <p className="text-xs text-slate-500">{programme.channel}</p>
            )}
          </div>
          {programme.transcript && (
            <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
              <Fact label="ความยาว" value={`${Math.round(programme.transcript.duration_ms / 60000)} นาที`} />
              <Fact label="ช่วงข่าว" value={String(programme.segment_count)} />
              <Fact label="ประโยคถอดเสียง" value={programme.transcript.cue_count.toLocaleString()} />
              <Fact label="ที่มาข้อความ" value={programme.transcript.source} />
            </dl>
          )}
        </div>

        <TopicStrip segments={segments} total={total} />
      </section>

      <ol className="space-y-3">
        {segments.map((segment) => (
          <li key={segment.id}>
            <SegmentCard segment={segment} />
          </li>
        ))}
      </ol>
    </>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="font-medium text-slate-700 dark:text-slate-200">{value}</dd>
    </div>
  )
}

/**
 * The whole programme as one bar, each story coloured by topic.
 *
 * This is the view that makes "the topic changed here" legible: a run of one
 * colour is a single subject, and a change of colour is a change of story.
 */
function TopicStrip({
  segments,
  total,
}: {
  segments: NewsSegment[]
  total: number
}) {
  if (!total) return null
  return (
    <div className="mt-4">
      <div
        className="flex h-6 w-full overflow-hidden rounded-lg"
        role="img"
        aria-label={`ลำดับหัวข้อข่าวตลอดรายการ ${segments.length} ช่วง`}
      >
        {segments.map((segment) => (
          <a
            key={segment.id}
            href={segment.youtube_url}
            target="_blank"
            rel="noopener noreferrer"
            title={`${segment.timecode} · ${segment.topic_label} · ${segment.headline}`}
            style={{
              width: `${(segment.duration_ms / total) * 100}%`,
              backgroundColor: segment.topic_color,
            }}
            className="block h-full transition-opacity hover:opacity-70"
          />
        ))}
      </div>
      <p className="mt-1 text-[11px] text-slate-400" lang="th">
        แต่ละแถบคือหนึ่งช่วงข่าว สีคือหัวข้อ — คลิกเพื่อไปยังวินาทีนั้นในคลิป
      </p>
    </div>
  )
}

function SegmentCard({ segment }: { segment: NewsSegment }) {
  const [open, setOpen] = useState(false)

  return (
    <article className="card overflow-hidden">
      <div className="flex flex-col gap-4 p-4 sm:flex-row">
        {/* The captured frame doubles as the deep link: clicking the picture of
            the moment takes you to the moment. */}
        <a
          href={segment.youtube_url}
          target="_blank"
          rel="noopener noreferrer"
          className="group relative shrink-0 overflow-hidden rounded-xl bg-slate-100 dark:bg-slate-800"
          style={{ width: 160 }}
          aria-label={`ดูคลิปที่ ${segment.timecode}`}
        >
          {segment.frame_url ? (
            <img
              src={segment.frame_url}
              alt={`ภาพจากคลิปที่ ${segment.timecode}`}
              width={160}
              height={90}
              loading="lazy"
              className="block h-[90px] w-[160px] object-cover transition-transform group-hover:scale-105"
            />
          ) : (
            <div className="flex h-[90px] w-[160px] items-center justify-center text-xs text-slate-400">
              ไม่มีภาพ
            </div>
          )}
          <span className="absolute bottom-1 right-1 rounded bg-black/75 px-1.5 py-0.5 font-mono text-[11px] text-white tabular-nums">
            {segment.timecode}
          </span>
        </a>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="rounded-full px-2 py-0.5 text-[11px] font-medium text-white"
              style={{ backgroundColor: segment.topic_color }}
            >
              {segment.topic_label}
            </span>
            <span className="text-[11px] text-slate-400 tabular-nums">
              {(segment.topic_confidence * 100).toFixed(0)}% · {minutes(segment.duration_ms)}
            </span>
            <BoundaryReasons segment={segment} />
          </div>

          <h4 className="mt-1.5 font-medium text-slate-900 dark:text-white" lang="th">
            {segment.headline}
          </h4>

          {segment.summary && (
            <p
              className="mt-1 line-clamp-2 text-sm text-slate-600 dark:text-slate-300"
              lang="th"
            >
              {segment.summary}
            </p>
          )}

          <div className="mt-2 flex flex-wrap items-center gap-3">
            <a
              href={segment.youtube_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm font-medium text-brand-600 hover:underline dark:text-brand-300"
            >
              ▶ ดูช่วงนี้ในคลิป
            </a>
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              aria-expanded={open}
              className="text-xs text-slate-500 hover:underline dark:text-slate-400"
            >
              {open ? 'ซ่อนข้อความถอดเสียง' : 'ดูข้อความถอดเสียง'}
            </button>
          </div>

          {open && (
            <p
              className="mt-3 max-h-56 overflow-y-auto rounded-xl bg-slate-50 p-3 text-xs leading-relaxed text-slate-600 dark:bg-slate-900/50 dark:text-slate-300"
              lang="th"
            >
              {segment.transcript_text_preview ?? segment.summary ?? ''}
              {segment.keywords.length > 0 && (
                <span className="mt-2 block text-slate-400">
                  คำสำคัญ: {segment.keywords.join(' · ')}
                </span>
              )}
            </p>
          )}
        </div>
      </div>
    </article>
  )
}

/** Says which signals produced this boundary, so a split is explainable. */
function BoundaryReasons({ segment }: { segment: NewsSegment }) {
  if (segment.boundary_reasons.length === 0) return null
  const labels = segment.boundary_reasons.map((r) => REASON_LABEL[r] ?? r)
  return (
    <span
      className="text-[11px] text-slate-400"
      title={`ตรวจพบขอบเขตจาก: ${labels.join(', ')} (ความมั่นใจ ${(segment.boundary_confidence * 100).toFixed(0)}%)`}
    >
      · แบ่งช่วงจาก {labels.join(' + ')}
    </span>
  )
}
