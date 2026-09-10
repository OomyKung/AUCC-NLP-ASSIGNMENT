/**
 * NLP Pipeline page.
 *
 * The stage diagram is not decorative: pick any stored document (or type your
 * own text) and each stage shows that document's *real* intermediate output.
 * That is what makes the pipeline inspectable rather than illustrative.
 */

import { useState } from 'react'
import { ErrorState, LoadingState } from '../components/ui'
import { useAsync } from '../hooks'
import { api } from '../services/api'
import type { NLPAnalysis, NewsDetail } from '../types'

const STAGES = [
  { key: 'raw', name: 'Raw Text', thai: 'ข้อความต้นฉบับ' },
  { key: 'clean', name: 'Text Cleaning', thai: 'ทำความสะอาดข้อความ' },
  { key: 'tokens', name: 'Thai Tokenization', thai: 'ตัดคำภาษาไทย' },
  { key: 'filtered', name: 'Stopword Removal', thai: 'ตัดคำหยุด' },
  { key: 'features', name: 'Feature Extraction', thai: 'สกัดคุณลักษณะ' },
  { key: 'topic', name: 'Topic Classification', thai: 'จำแนกหัวข้อ' },
  { key: 'sentiment', name: 'Sentiment Analysis', thai: 'วิเคราะห์ความรู้สึก' },
  { key: 'keywords', name: 'Keyword Extraction', thai: 'สกัดคำสำคัญ' },
  { key: 'summary', name: 'Summary Generation', thai: 'สร้างบทสรุป' },
  { key: 'dashboard', name: 'Dashboard', thai: 'แสดงผลบนแดชบอร์ด' },
] as const

export default function PipelinePage() {
  const backends = useAsync(() => api.pipeline(), [])
  const recent = useAsync(() => api.news({ page_size: 8, sort: 'newest' }), [])

  const [selectedId, setSelectedId] = useState<number | null>(null)
  const chosen = selectedId ?? recent.data?.items[0]?.id ?? null

  const detail = useAsync(
    () => (chosen ? api.newsDetail(chosen) : Promise.resolve(null)),
    [chosen],
  )

  // Hoisted to locals so TypeScript keeps the null-check narrowing inside JSX.
  const document = detail.data
  const analysis = document?.analysis ?? null

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
          NLP Pipeline
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400" lang="th">
          กระบวนการประมวลผลภาษาธรรมชาติทั้ง 10 ขั้นตอน พร้อมผลลัพธ์จริงของเอกสารที่เลือก
        </p>
      </header>

      {/* ------------------------------------------------- active backends */}
      <section className="card p-5">
        <h3 className="mb-1 font-semibold text-slate-900 dark:text-white">
          โมเดลที่กำลังใช้งาน
        </h3>
        <p className="mb-4 text-xs text-slate-500 dark:text-slate-400" lang="th">
          ทุกส่วนสามารถสลับได้ผ่านไฟล์ .env โดยไม่ต้องแก้โค้ด
        </p>

        {backends.error ? (
          <ErrorState message={backends.error} onRetry={backends.reload} />
        ) : !backends.data ? (
          <LoadingState rows={1} />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400 dark:border-slate-700">
                  <th className="pb-2 pr-4 font-medium">Stage</th>
                  <th className="pb-2 pr-4 font-medium">Requested</th>
                  <th className="pb-2 pr-4 font-medium">Active</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-700/60">
                {backends.data.map((row) => (
                  <tr key={row.stage}>
                    <td className="py-2 pr-4 font-medium text-slate-700 dark:text-slate-200">
                      {row.stage}
                    </td>
                    <td className="py-2 pr-4 text-slate-500 dark:text-slate-400">
                      {row.requested}
                    </td>
                    <td className="py-2 pr-4">
                      <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs dark:bg-slate-800">
                        {row.active}
                      </code>
                    </td>
                    <td className="py-2">
                      {row.trained ? (
                        <span className="rounded-full bg-teal-50 px-2 py-0.5 text-[11px] font-medium text-teal-800 dark:bg-teal-500/10 dark:text-teal-300">
                          พร้อมใช้งาน
                        </span>
                      ) : (
                        <span
                          className="rounded-full bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-800 dark:bg-amber-500/10 dark:text-amber-300"
                          title={row.note}
                        >
                          baseline
                        </span>
                      )}
                      {row.note && (
                        <span className="ml-2 text-[11px] text-slate-400">{row.note}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* ---------------------------------------------------- doc selector */}
      <section className="card p-5">
        <label className="block">
          <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
            เลือกเอกสารเพื่อดูผลลัพธ์แต่ละขั้นตอน
          </span>
          <select
            value={chosen ?? ''}
            onChange={(event) => setSelectedId(Number(event.target.value))}
            className="input"
            disabled={!recent.data?.items.length}
          >
            {recent.data?.items.map((item) => (
              <option key={item.id} value={item.id}>
                [{item.topic}/{item.sentiment}] {item.title.slice(0, 70)}
              </option>
            ))}
          </select>
        </label>
      </section>

      {/* -------------------------------------------------- stage pipeline */}
      {detail.error ? (
        <ErrorState message={detail.error} onRetry={detail.reload} />
      ) : detail.loading && !detail.data ? (
        <LoadingState rows={4} />
      ) : !analysis || !document ? (
        <p className="text-sm text-slate-400" lang="th">
          ยังไม่มีเอกสารสำหรับแสดงกระบวนการ
        </p>
      ) : (
        <ol className="space-y-3">
          {STAGES.map((stage, index) => (
            <li key={stage.key}>
              <div className="card p-5">
                <div className="mb-3 flex items-center gap-3">
                  <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-brand-600 text-xs font-semibold text-white tabular-nums">
                    {index + 1}
                  </span>
                  <div className="min-w-0">
                    <p className="font-medium text-slate-900 dark:text-white">
                      {stage.name}
                    </p>
                    <p className="text-[11px] text-slate-400" lang="th">
                      {stage.thai}
                    </p>
                  </div>
                </div>
                <StageOutput
                  stage={stage.key}
                  detail={document}
                  analysis={analysis}
                />
              </div>
              {index < STAGES.length - 1 && (
                <div className="flex justify-center py-1" aria-hidden="true">
                  <svg
                    viewBox="0 0 24 24"
                    className="size-4 text-slate-300 dark:text-slate-600"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M12 5v14m0 0-5-5m5 5 5-5" strokeLinecap="round" />
                  </svg>
                </div>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */

function StageOutput({
  stage,
  detail,
  analysis,
}: {
  stage: (typeof STAGES)[number]['key']
  detail: NewsDetail
  analysis: NLPAnalysis
}) {
  const box =
    'scroll-thin max-h-40 overflow-y-auto rounded-xl bg-slate-50 p-3 text-sm dark:bg-slate-900/50'

  switch (stage) {
    case 'raw':
      return (
        <div className={box}>
          <p className="whitespace-pre-wrap text-slate-600 dark:text-slate-300" lang="th">
            {detail.content.slice(0, 1200)}
            {detail.content.length > 1200 && '…'}
          </p>
          <p className="mt-2 text-[11px] tabular-nums text-slate-400">
            {detail.content.length.toLocaleString()} ตัวอักษร
          </p>
        </div>
      )

    case 'clean':
      return (
        <div className={box}>
          <p className="whitespace-pre-wrap text-slate-600 dark:text-slate-300" lang="th">
            {(analysis.cleaned_text ?? '').slice(0, 1200)}
          </p>
          <p className="mt-2 text-[11px] text-slate-400" lang="th">
            ลบ URL, HTML, อีโมจิซ้ำ, อักขระซ้ำ และปรับรูปอักขระไทยให้เป็นมาตรฐาน
          </p>
        </div>
      )

    case 'tokens':
      return (
        <div>
          <TokenList tokens={analysis.tokens} limit={160} />
          <p className="mt-2 text-[11px] tabular-nums text-slate-400">
            {analysis.token_count.toLocaleString()} โทเคน ·{' '}
            {analysis.unique_token_count.toLocaleString()} คำไม่ซ้ำ (PyThaiNLP newmm)
          </p>
        </div>
      )

    case 'filtered':
      return (
        <div>
          <TokenList tokens={analysis.filtered_tokens} limit={160} tone="brand" />
          <p className="mt-2 text-[11px] tabular-nums text-slate-400" lang="th">
            ตัดออก {analysis.stopword_removed_count.toLocaleString()} โทเคน · เหลือ{' '}
            {analysis.filtered_tokens.length.toLocaleString()}
          </p>
        </div>
      )

    case 'features':
      return (
        <div className={box}>
          <p className="text-slate-600 dark:text-slate-300" lang="th">
            แปลงคำที่เหลือเป็นเวกเตอร์ TF-IDF และจับคู่กับคลังศัพท์ (gazetteer /
            lexicon) แบบรองรับคำประสมภาษาไทย
          </p>
          <ul className="mt-2 space-y-1 text-[11px] tabular-nums text-slate-500 dark:text-slate-400">
            <li>คำที่ใช้เป็นคุณลักษณะ: {analysis.filtered_tokens.length.toLocaleString()}</li>
            <li>ประโยคที่ตัดได้: {analysis.sentences.length.toLocaleString()}</li>
            <li>เวลาประมวลผลทั้งกระบวนการ: {analysis.processing_ms.toFixed(0)} ms</li>
          </ul>
        </div>
      )

    case 'topic': {
      const rows = Object.entries(analysis.topic_probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
      return (
        <div className={box}>
          <ul className="space-y-1.5">
            {rows.map(([slug, value]) => (
              <li key={slug} className="flex items-center gap-2 text-sm">
                <span className="w-28 shrink-0 text-slate-600 dark:text-slate-300">
                  {slug}
                </span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                  <span
                    className="block h-full rounded-full bg-brand-500"
                    style={{ width: `${value * 100}%` }}
                  />
                </span>
                <span className="w-12 text-right tabular-nums text-slate-500">
                  {value.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[11px] text-slate-400">
            ผลลัพธ์: <strong>{detail.topic}</strong> (
            {Math.round(detail.topic_confidence * 100)}%)
          </p>
        </div>
      )
    }

    case 'sentiment': {
      const rows = Object.entries(analysis.sentiment_probabilities).sort(
        ([, a], [, b]) => b - a,
      )
      return (
        <div className={box}>
          <ul className="space-y-1.5">
            {rows.map(([slug, value]) => (
              <li key={slug} className="flex items-center gap-2 text-sm">
                <span className="w-28 shrink-0 text-slate-600 dark:text-slate-300">
                  {slug}
                </span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                  <span
                    className={`block h-full rounded-full ${
                      slug === 'positive'
                        ? 'bg-positive'
                        : slug === 'negative'
                          ? 'bg-negative'
                          : 'bg-neutral'
                    }`}
                    style={{ width: `${value * 100}%` }}
                  />
                </span>
                <span className="w-12 text-right tabular-nums text-slate-500">
                  {value.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[11px] text-slate-400">
            ผลลัพธ์: <strong>{detail.sentiment}</strong> (
            {Math.round(detail.sentiment_confidence * 100)}%)
          </p>
        </div>
      )
    }

    case 'keywords':
      return (
        <div>
          <ul className="flex flex-wrap gap-1.5">
            {analysis.keyword_scores.map((keyword) => (
              <li
                key={keyword.word}
                className="flex items-baseline gap-1 rounded-md bg-slate-100 px-2 py-0.5 text-xs dark:bg-slate-800"
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
          <p className="mt-2 text-[11px] text-slate-400" lang="th">
            คะแนน TF-IDF จากสถิติคลังข้อมูลจริง
          </p>
        </div>
      )

    case 'summary':
      return (
        <div className={box}>
          <p className="text-slate-700 dark:text-slate-200" lang="th">
            {detail.summary || '—'}
          </p>
          <p className="mt-2 text-[11px] text-slate-400" lang="th">
            สกัดประโยคเด่นด้วยความเป็นศูนย์กลาง TF-IDF ร่วมกับน้ำหนักประโยคต้น
          </p>
        </div>
      )

    case 'dashboard':
      return (
        <div className={box}>
          <p className="text-slate-600 dark:text-slate-300" lang="th">
            บันทึกผลลงฐานข้อมูล SQLite แล้วรวมเป็นสถิติ กราฟ และหน้าสำรวจข่าว
          </p>
          <ul className="mt-2 space-y-1 text-[11px] text-slate-500 dark:text-slate-400">
            {Object.entries(analysis.model_versions).map(([key, value]) => (
              <li key={key}>
                {key}: <code>{value}</code>
              </li>
            ))}
          </ul>
        </div>
      )

    default:
      return null
  }
}

function TokenList({
  tokens,
  limit,
  tone = 'plain',
}: {
  tokens: string[]
  limit: number
  tone?: 'plain' | 'brand'
}) {
  const shown = tokens.slice(0, limit)
  return (
    <div className="scroll-thin max-h-40 overflow-y-auto rounded-xl bg-slate-50 p-3 dark:bg-slate-900/50">
      <div className="flex flex-wrap gap-1">
        {shown.map((token, index) => (
          <span
            key={`${token}-${index}`}
            className={`rounded px-1.5 py-0.5 text-xs shadow-sm ${
              tone === 'brand'
                ? 'bg-brand-50 text-brand-800 dark:bg-brand-500/15 dark:text-brand-200'
                : 'bg-white text-slate-600 dark:bg-slate-800 dark:text-slate-300'
            }`}
            lang="th"
          >
            {token}
          </span>
        ))}
      </div>
      {tokens.length > limit && (
        <p className="mt-2 text-[11px] tabular-nums text-slate-400">
          แสดง {limit} จาก {tokens.length.toLocaleString()}
        </p>
      )}
    </div>
  )
}
