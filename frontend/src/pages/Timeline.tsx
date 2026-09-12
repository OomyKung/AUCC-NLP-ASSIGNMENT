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

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ErrorState, LoadingState } from '../components/ui'
import { notify, useAsync } from '../hooks'
import { isUnclear } from '../lib/sentiment'
import { ApiError, api } from '../services/api'
import type {
  EnrichmentJob,
  NewsSegment,
  Programme,
  Reaction,
  SegmentReaction,
  TimelineView,
} from '../types'

/** How often to ask a running enrichment job how it is getting on. */
const POLL_MS = 4000

/**
 * The two halves of a broadcast, and the switch between them.
 *
 * A news programme is two recordings of the same hour: what the newsreader
 * said, and what the audience said back. They share a clock, so every story can
 * show either — or both, which is where the interesting disagreements are.
 */
const VIEWS: { value: TimelineView; label: string; hint: string }[] = [
  { value: 'reporter', label: 'ผู้ประกาศ', hint: 'เนื้อข่าวจากเสียงพูดในคลิป' },
  { value: 'viewers', label: 'ผู้ชม', hint: 'ความเห็นในแชทช่วงเวลาเดียวกัน' },
  { value: 'both', label: 'ทั้งคู่', hint: 'เทียบข่าวกับความเห็นผู้ชม' },
]

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
  const [view, setView] = useState<TimelineView>('reporter')

  const selected = videoId ?? programmes.data?.[0]?.video_id ?? null
  const programme = useAsync(
    () => (selected ? api.programme(selected) : Promise.resolve(null)),
    [selected],
  )

  // Reloading both lists is what makes the timeline fill in while the job runs:
  // the cards get their new headlines and the picker's pending count drops.
  const refresh = useCallback(() => {
    programme.reload()
    programmes.reload()
  }, [programme, programmes])

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

      <ViewSwitch
        view={view}
        onChange={setView}
        chatSegments={
          list.find((item) => item.video_id === selected)?.chat_segments ?? 0
        }
      />

      {programme.error ? (
        <ErrorState message={programme.error} onRetry={programme.reload} />
      ) : programme.loading && !programme.data ? (
        <LoadingState rows={4} />
      ) : programme.data ? (
        <ProgrammeTimeline
          programme={programme.data}
          view={view}
          onEnriched={refresh}
        />
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

/**
 * Switches the timeline between the newsreader and the audience.
 *
 * Both are already stored and already aligned -- chat carries the offset from
 * the start of the stream, and so does every story -- so this is a filter, not
 * a second analysis. Disabled with an explanation when the programme has no
 * chat, which is most of them: news channels turn chat replay off after a
 * broadcast, and an enabled control that does nothing is worse than an honest
 * disabled one.
 */
function ViewSwitch({
  view,
  onChange,
  chatSegments,
}: {
  view: TimelineView
  onChange: (view: TimelineView) => void
  chatSegments: number
}) {
  const hasChat = chatSegments > 0
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div
        className="flex rounded-xl border border-slate-200 p-0.5 dark:border-slate-700"
        role="group"
        aria-label="เลือกมุมมอง"
      >
        {VIEWS.map((option) => {
          const disabled = !hasChat && option.value !== 'reporter'
          const active = view === option.value
          return (
            <button
              key={option.value}
              type="button"
              onClick={() => onChange(option.value)}
              disabled={disabled}
              title={disabled ? 'คลิปนี้ไม่มีแชท' : option.hint}
              aria-pressed={active}
              className={`rounded-lg px-3 py-1.5 text-sm transition ${
                active
                  ? 'bg-brand-600 font-medium text-white'
                  : disabled
                    ? 'cursor-not-allowed text-slate-300 dark:text-slate-600'
                    : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'
              }`}
            >
              {option.label}
            </button>
          )
        })}
      </div>
      <p className="text-xs text-slate-400" lang="th">
        {hasChat
          ? `${VIEWS.find((v) => v.value === view)?.hint} · มีแชท ${chatSegments} ช่วง`
          : 'คลิปนี้ไม่มีแชท จึงดูได้เฉพาะเนื้อข่าวจากเสียงพูด'}
      </p>
    </div>
  )
}

function ProgrammeTimeline({
  programme,
  view,
  onEnriched,
}: {
  programme: Programme
  view: TimelineView
  onEnriched: () => void
}) {
  const segments = programme.segments
  const total = useMemo(
    () => segments.reduce((max, s) => Math.max(max, s.end_ms), 0),
    [segments],
  )

  // One request for every story's chat counts, and only when the reader has
  // actually asked to see them.
  const wantsChat = view !== 'reporter'
  const reactions = useAsync(
    () =>
      wantsChat && programme.chat_segments > 0
        ? api.programmeReactions(programme.video_id)
        : Promise.resolve<SegmentReaction[]>([]),
    [programme.video_id, wantsChat, programme.pending_reactions],
  )
  const bySegment = useMemo(() => {
    const map = new Map<number, SegmentReaction>()
    for (const row of reactions.data ?? []) map.set(row.segment_id, row)
    return map
  }, [reactions.data])

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
        <EnrichmentRunner
          key={programme.video_id}
          programme={programme}
          onProgress={onEnriched}
        />
      </section>

      <ol className="space-y-3">
        {segments.map((segment) => (
          <li key={segment.id}>
            <SegmentCard
              segment={segment}
              view={view}
              reaction={bySegment.get(segment.id)}
            />
          </li>
        ))}
      </ol>
    </>
  )
}

/**
 * Runs the LLM work an import could not wait for: story headlines, and one line
 * of what the audience said during each story.
 *
 * A model spends 10-40 seconds on each, so a 56-story programme is most of an
 * hour. That is far too long to hold an HTTP request, but perfectly fine as a
 * job the page watches: the backend commits every unit as it lands, so this
 * reloads the timeline on each poll and the cards improve one by one while the
 * reader is looking at them.
 *
 * Nothing here is destructive. Stopping keeps everything already written, and
 * closing the page does not stop the job — reopening it picks the progress back
 * up.
 */
function EnrichmentRunner({
  programme,
  onProgress,
}: {
  programme: Programme
  onProgress: () => void
}) {
  const [job, setJob] = useState<EnrichmentJob | null>(null)
  const [starting, setStarting] = useState(false)
  const videoId = programme.video_id
  // Kept in a ref so the polling effect does not restart whenever the parent
  // hands down a new callback identity, which it does on every render.
  const progressRef = useRef(onProgress)
  useEffect(() => {
    progressRef.current = onProgress
  }, [onProgress])

  // Pick up a job that was already running when the page opened. No reset of
  // `job` is needed here: the parent keys this component on the video id, so
  // switching programmes mounts a fresh one rather than reusing this state.
  useEffect(() => {
    let cancelled = false
    api
      .enrichmentStatus(videoId)
      .then((status) => {
        if (!cancelled) setJob(status)
      })
      .catch(() => {
        /* 404 simply means nothing has been started for this programme. */
      })
    return () => {
      cancelled = true
    }
  }, [videoId])

  const running = job?.state === 'running' || job?.state === 'queued'

  useEffect(() => {
    if (!running) return
    let stopped = false
    const timer = window.setInterval(async () => {
      try {
        const status = await api.enrichmentStatus(videoId)
        if (stopped) return
        setJob((previous) => {
          // Only reload the list when a headline actually landed, so a poll
          // that changed nothing does not refetch the whole programme.
          if (previous && status.done > previous.done) progressRef.current()
          return status
        })
        if (status.state !== 'running' && status.state !== 'queued') {
          progressRef.current()
          notify(
            status.state === 'done'
              ? `โมเดลเขียนครบ ${status.total} รายการแล้ว`
              : status.note || `งานของโมเดล: ${status.state}`,
            status.state === 'done' ? 'success' : 'info',
          )
        }
      } catch {
        /* A failed poll is not worth surfacing; the next one will tell us. */
      }
    }, POLL_MS)
    return () => {
      stopped = true
      window.clearInterval(timer)
    }
  }, [running, videoId])

  const start = async () => {
    setStarting(true)
    try {
      setJob(await api.startEnrichment(videoId))
    } catch (err) {
      notify(
        err instanceof ApiError ? err.message : 'เริ่มงานของโมเดลไม่สำเร็จ',
        'error',
      )
    } finally {
      setStarting(false)
    }
  }

  const stop = async () => {
    try {
      setJob(await api.stopEnrichment(videoId))
    } catch {
      /* Already finished on its own. */
    }
  }

  const pendingHeadlines = programme.pending_headlines
  const pendingReactions = programme.pending_reactions
  const pending = pendingHeadlines + pendingReactions
  if (!running && pending === 0) {
    return (
      <p className="mt-4 text-xs text-slate-400" lang="th">
        ทุกช่วงข่าวมีพาดหัวและสรุปความเห็นผู้ชมที่เขียนโดยโมเดลภาษาแล้ว
      </p>
    )
  }

  if (!running) {
    return (
      <div className="mt-4 flex flex-wrap items-center gap-3 rounded-xl bg-slate-50 p-3 dark:bg-slate-900/40">
        <div className="min-w-0 flex-1">
          <p className="text-sm text-slate-700 dark:text-slate-200" lang="th">
            {[
              pendingHeadlines > 0 ? `พาดหัว ${pendingHeadlines} ช่วง` : null,
              pendingReactions > 0 ? `สรุปความเห็นผู้ชม ${pendingReactions} ช่วง` : null,
            ]
              .filter(Boolean)
              .join(' · ')}{' '}
            ยังไม่ได้ให้โมเดลเขียน
          </p>
          <p className="mt-0.5 text-xs text-slate-400" lang="th">
            โมเดลภาษาทำงานในเครื่อง ฟรี · ประมาณ 10-40 วินาทีต่อรายการ
            ทำงานเบื้องหลัง ปิดหน้านี้ได้
            {job?.note ? ` · ${job.note}` : ''}
          </p>
        </div>
        <button
          type="button"
          onClick={start}
          disabled={starting}
          className="btn-primary shrink-0"
        >
          {starting ? 'กำลังเริ่ม…' : 'ให้ AI เขียนพาดหัว + สรุปความเห็น'}
        </button>
      </div>
    )
  }

  const percent = job.total ? Math.round((job.done / job.total) * 100) : 0
  return (
    <div className="mt-4 space-y-2 rounded-xl bg-slate-50 p-3 dark:bg-slate-900/40">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-slate-700 dark:text-slate-200" lang="th">
          กำลังให้โมเดลเขียน {job.done}/{job.total} รายการ
          {job.headlines + job.reactions > 0 && (
            <span className="text-slate-400">
              {' '}
              (พาดหัว {job.headlines} · ความเห็นผู้ชม {job.reactions})
            </span>
          )}
          {job.eta_seconds != null && job.eta_seconds > 0 && (
            <span className="text-slate-400">
              {' '}
              · เหลืออีกประมาณ {Math.ceil(job.eta_seconds / 60)} นาที
            </span>
          )}
        </p>
        <button type="button" onClick={stop} className="btn-ghost shrink-0 text-xs">
          หยุด (เก็บที่เขียนแล้ว)
        </button>
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="h-full rounded-full bg-brand-500 transition-[width] duration-500"
          style={{ width: `${percent}%` }}
        />
      </div>
      {job.latest && (
        <p className="truncate text-xs text-slate-400" lang="th">
          ล่าสุด: {job.latest}
        </p>
      )}
    </div>
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

function SegmentCard({
  segment,
  view,
  reaction,
}: {
  segment: NewsSegment
  view: TimelineView
  reaction?: SegmentReaction
}) {
  const [open, setOpen] = useState(false)
  const showReporter = view !== 'viewers'
  const showViewers = view !== 'reporter'

  return (
    <article className="card overflow-hidden">
      <div className="flex flex-col gap-4 p-4 sm:flex-row">
        {/* The captured frame doubles as the deep link: clicking the picture of
            the moment takes you to the moment. */}
        {/* `self-start` matters: a flex item stretches to the row height by
            default, which left the 16:9 frame floating in a tall empty box.
            `aspect-video` matches the 1280x720 source exactly, so the image
            fills its container at every width instead of being pinned to
            hard-coded pixels. */}
        <a
          href={segment.youtube_url}
          target="_blank"
          rel="noopener noreferrer"
          className="group relative block w-full shrink-0 self-start overflow-hidden rounded-xl bg-slate-100 sm:w-48 dark:bg-slate-800"
          aria-label={`ดูคลิปที่ ${segment.timecode}`}
        >
          {segment.frame_url ? (
            <img
              src={segment.frame_url}
              alt={`ภาพจากคลิปที่ ${segment.timecode}`}
              loading="lazy"
              className="block aspect-video w-full object-cover transition-transform duration-200 group-hover:scale-105"
            />
          ) : (
            <div className="flex aspect-video w-full items-center justify-center text-xs text-slate-400">
              ไม่มีภาพ
            </div>
          )}
          <span className="pointer-events-none absolute bottom-1.5 right-1.5 rounded bg-black/75 px-1.5 py-0.5 font-mono text-[11px] text-white tabular-nums">
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
            <HeadlineSource segment={segment} />
          </h4>

          <NameCorrections segment={segment} />

          {showReporter && segment.summary && (
            <p
              className="mt-1 line-clamp-2 text-sm text-slate-600 dark:text-slate-300"
              lang="th"
            >
              {segment.summary}
            </p>
          )}

          {/* Keywords used to BE the headline. Now the headline is a real
              phrase, so they move here — still visible, no longer pretending
              to be a title. */}
          {showReporter && segment.keywords.length > 0 && (
            <ul className="mt-2 flex flex-wrap gap-1.5" aria-label="คำสำคัญ">
              {segment.keywords.slice(0, 6).map((word) => (
                <li
                  key={word}
                  className="rounded-md bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                  lang="th"
                >
                  {word}
                </li>
              ))}
            </ul>
          )}

          {showViewers && <ViewerReaction segment={segment} reaction={reaction} />}

          <div className="mt-2 flex flex-wrap items-center gap-3">
            <a
              href={segment.youtube_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm font-medium text-brand-600 hover:underline dark:text-brand-300"
            >
              ▶ ดูช่วงนี้ในคลิป
            </a>
            {showReporter && (
              <button
                type="button"
                onClick={() => setOpen((value) => !value)}
                aria-expanded={open}
                className="text-xs text-slate-500 hover:underline dark:text-slate-400"
              >
                {open ? 'ซ่อนข้อความถอดเสียง' : 'ดูข้อความถอดเสียง'}
              </button>
            )}
          </div>

          {showReporter && open && (
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

/**
 * Marks a headline a language model wrote.
 *
 * The extractive headline is a span lifted verbatim out of the transcript, so it
 * is guaranteed to be something that was said. A written one reads far better
 * and is a paraphrase, so it is not guaranteed in the same way. That difference
 * belongs on screen rather than in the README — the same reason a boundary shows
 * the signals that produced it instead of presenting itself as an oracle.
 */
function HeadlineSource({ segment }: { segment: NewsSegment }) {
  if (segment.enriched_by !== 'llm') return null
  return (
    <span
      className="ml-1.5 align-middle rounded bg-violet-100 px-1.5 py-0.5 text-[10px] font-medium text-violet-700 dark:bg-violet-500/15 dark:text-violet-300"
      title="พาดหัวนี้เขียนโดยโมเดลภาษา (สรุปความ) ไม่ใช่ข้อความที่ตัดมาจากคำถอดเสียงโดยตรง"
    >
      AI เขียนพาดหัว
    </span>
  )
}

/**
 * Shows names the model respelled, with the form the ASR produced.
 *
 * A correction is a claim about what was said, so the reader gets to see what it
 * replaced. Empty unless LLM_CORRECT_NAMES is on, which it is not by default —
 * a small local model invents Thai names rather than admitting it cannot tell.
 */
function NameCorrections({ segment }: { segment: NewsSegment }) {
  const pairs = (segment.name_corrections ?? []).filter((pair) => pair.length === 2)
  if (pairs.length === 0) return null
  return (
    <p className="mt-1 text-[11px] text-slate-400" lang="th">
      แก้การสะกดชื่อ:{' '}
      {pairs.map(([asr, fixed], index) => (
        <span key={`${asr}-${fixed}`}>
          {index > 0 && ', '}
          <span className="line-through">{asr}</span> → {fixed}
        </span>
      ))}
    </p>
  )
}

/** Sentiment colours, matching the badges used elsewhere in the dashboard. */
const MOOD_COLOR: Record<string, string> = {
  positive: 'bg-emerald-500',
  neutral: 'bg-slate-400',
  negative: 'bg-rose-500',
}

const MOOD_LABEL: Record<string, string> = {
  positive: 'เชิงบวก',
  neutral: 'กลาง ๆ',
  negative: 'เชิงลบ',
  // A tie has no majority, and saying so is truer than picking a side.
  mixed: 'ความเห็นแบ่งกัน',
}

/**
 * What the audience said while this story was on air.
 *
 * The counts and the mood bar come from data already stored, so they are always
 * there. The one-line summary is the model's, and is simply absent until
 * someone runs the enrichment — an absent line is honest; an invented one would
 * not be.
 *
 * Sample messages are fetched only when the reader opens them: 76 cards would
 * otherwise ship thousands of messages nobody asked to read.
 */
function ViewerReaction({
  segment,
  reaction,
}: {
  segment: NewsSegment
  reaction?: SegmentReaction
}) {
  const [detail, setDetail] = useState<Reaction | null>(null)
  const [loading, setLoading] = useState(false)

  if (!reaction || reaction.total === 0) {
    return (
      <p className="mt-2 text-xs text-slate-400" lang="th">
        ไม่มีข้อความแชทในช่วงเวลานี้
      </p>
    )
  }

  const counts = reaction.sentiment_counts
  const open = async () => {
    if (detail) {
      setDetail(null)
      return
    }
    setLoading(true)
    try {
      setDetail(await api.segmentChat(segment.id))
    } catch {
      /* Leaving the counts on screen is a better failure than an error box. */
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mt-2 rounded-xl bg-slate-50 p-3 dark:bg-slate-900/40">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
        <span className="font-medium text-slate-700 dark:text-slate-200" lang="th">
          ผู้ชม {reaction.total.toLocaleString()} ข้อความ
        </span>
        {reaction.mood && (
          <span lang="th">
            ·{' '}
            {reaction.mood === 'mixed'
              ? MOOD_LABEL.mixed
              : `ส่วนใหญ่${MOOD_LABEL[reaction.mood] ?? reaction.mood}`}
          </span>
        )}
      </div>

      {/* The mood mix as one bar: the shape of the reaction at a glance. */}
      <div
        className="mt-2 flex h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700"
        role="img"
        aria-label={Object.entries(counts)
          .map(([name, value]) => `${MOOD_LABEL[name] ?? name} ${value}`)
          .join(', ')}
      >
        {(['positive', 'neutral', 'negative'] as const).map((mood) =>
          counts[mood] ? (
            <div
              key={mood}
              className={MOOD_COLOR[mood]}
              style={{ width: `${(counts[mood] / reaction.total) * 100}%` }}
            />
          ) : null,
        )}
      </div>

      {reaction.summary ? (
        <p className="mt-2 text-sm text-slate-700 dark:text-slate-200" lang="th">
          {reaction.summary}
          <span
            className="ml-1.5 align-middle rounded bg-violet-100 px-1.5 py-0.5 text-[10px] font-medium text-violet-700 dark:bg-violet-500/15 dark:text-violet-300"
            title="สรุปความเห็นผู้ชมโดยโมเดลภาษา"
          >
            AI สรุป
          </span>
        </p>
      ) : (
        <p className="mt-2 text-xs text-slate-400" lang="th">
          ยังไม่มีสรุปความเห็นผู้ชมช่วงนี้ · กด “ให้ AI เขียนพาดหัว + สรุปความเห็น” ด้านบน
        </p>
      )}

      <button
        type="button"
        onClick={open}
        aria-expanded={detail !== null}
        className="mt-2 text-xs text-slate-500 hover:underline dark:text-slate-400"
      >
        {loading
          ? 'กำลังโหลด…'
          : detail
            ? 'ซ่อนข้อความจากผู้ชม'
            : 'ดูข้อความจากผู้ชม'}
      </button>

      {detail && (
        <div className="mt-2 space-y-2">
          {detail.keywords.length > 0 && (
            <ul className="flex flex-wrap gap-1.5" aria-label="คำที่ผู้ชมใช้บ่อย">
              {detail.keywords.map((word) => (
                <li
                  key={word}
                  className="rounded-md bg-white px-1.5 py-0.5 text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                  lang="th"
                >
                  {word}
                </li>
              ))}
            </ul>
          )}
          <ul className="max-h-56 space-y-1 overflow-y-auto text-xs">
            {detail.samples.map((message, index) => (
              <li
                key={`${message.offset_ms}-${index}`}
                className="flex gap-2 text-slate-600 dark:text-slate-300"
                lang="th"
              >
                {/* Faded when the classifier was not confident enough to be
                    presenting a verdict at all — see lib/sentiment. */}
                <span
                  className={`mt-1.5 size-1.5 shrink-0 rounded-full ${
                    MOOD_COLOR[message.sentiment] ?? MOOD_COLOR.neutral
                  } ${isUnclear(message.confidence) ? 'opacity-30' : ''}`}
                  aria-hidden
                />
                <span className="min-w-0">{message.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
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
