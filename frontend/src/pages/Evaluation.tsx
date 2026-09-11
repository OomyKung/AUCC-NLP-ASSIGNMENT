/**
 * Evaluation page: real metrics from a held-out split.
 *
 * Three things this page does deliberately:
 *
 * 1. It shows the rule-based baseline beside the trained model and their blend,
 *    all scored on the *same* held-out rows. A trained model's accuracy means
 *    little without knowing what a simple approach already achieved.
 * 2. It shows the random-guess floor, so a reader can judge whether the numbers
 *    are impressive for the number of classes involved.
 * 3. It marks which model is actually serving requests and opens on it. The
 *    dashboard's numbers come from one specific backend, and a metrics page
 *    that quietly described a different one would be worse than no page.
 */

import { useState } from 'react'
import { ErrorState, LoadingState } from '../components/ui'
import { useAsync } from '../hooks'
import type { ModelKey, ModelMetrics, TaskEvaluation } from '../types'
import { api } from '../services/api'

export default function EvaluationPage() {
  const evaluation = useAsync(() => api.evaluation(), [])

  if (evaluation.error) {
    return <ErrorState message={evaluation.error} onRetry={evaluation.reload} />
  }
  if (evaluation.loading && !evaluation.data) return <LoadingState rows={3} />

  const data = evaluation.data
  if (!data) return null

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
          ประเมินผลโมเดล (Model Evaluation)
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400" lang="th">
          Accuracy, Precision, Recall, F1-score และ Confusion Matrix จากชุดทดสอบที่โมเดลไม่เคยเห็น
        </p>
      </header>

      {!data.available ? (
        <NotAvailable reason={data.reason} howTo={data.how_to} />
      ) : (
        <>
          <section className="card p-5">
            <h3 className="mb-3 font-semibold text-slate-900 dark:text-white">
              วิธีการประเมิน
            </h3>
            <dl className="grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
              <Fact label="ชุดข้อมูล" value={`${data.dataset.documents.toLocaleString()} เอกสาร`} />
              <Fact
                label="สัดส่วนชุดทดสอบ"
                value={`${Math.round(data.split.test_size * 100)}%`}
              />
              <Fact label="Random seed" value={String(data.split.random_state)} />
              <Fact label="Stratified split" value={data.split.stratified ? 'ใช่' : 'ไม่'} />
            </dl>
            <p
              className="mt-4 rounded-xl bg-slate-50 p-3 text-xs text-slate-600 dark:bg-slate-900/50 dark:text-slate-300"
              lang="th"
            >
              โมเดลที่ฝึกแล้วและโมเดลฐาน (rule-based) ถูกวัดผลบนชุดทดสอบชุดเดียวกัน
              เนื่องจากคลังคำและกฎถูกเขียนขึ้นสำหรับโดเมนนี้โดยตรง
              การวัดผลบนข้อมูลที่ใช้ออกแบบกฎจะทำให้ผลดูดีเกินจริง
            </p>
          </section>

          {Object.entries(data.tasks).map(([task, entry]) => (
            <TaskSection key={task} task={task} entry={entry} />
          ))}

          <p className="text-center text-xs text-slate-400">
            สร้างเมื่อ {new Date(data.generated_at).toLocaleString('th-TH')}
          </p>
        </>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */

function NotAvailable({ reason, howTo }: { reason?: string; howTo?: string[] }) {
  return (
    <div className="card border-l-4 border-amber-400 p-5">
      <h3 className="font-semibold text-slate-900 dark:text-white">
        ยังไม่มีผลการประเมิน
      </h3>
      <p className="mt-2 text-sm text-slate-600 dark:text-slate-300" lang="th">
        {reason ??
          'ยังไม่ได้สร้างไฟล์ผลการประเมิน การแสดงตัวเลขในขั้นนี้จะเป็นการกุข้อมูล'}
      </p>
      {howTo && howTo.length > 0 && (
        <div className="mt-4 rounded-xl bg-slate-50 p-3 dark:bg-slate-900/50">
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            วิธีสร้างผลการประเมิน
          </p>
          <pre className="overflow-x-auto text-xs text-slate-600 dark:text-slate-300">
            {howTo.join('\n')}
          </pre>
        </div>
      )}
    </div>
  )
}

/** Display order and Thai labels for the models evaluate.py scores. */
const MODEL_LABELS: { key: ModelKey; thai: string }[] = [
  { key: 'blend', thai: 'โมเดลผสม' },
  { key: 'trained', thai: 'โมเดลที่ฝึกแล้ว' },
  { key: 'baseline', thai: 'โมเดลฐาน' },
]

function TaskSection({ task, entry }: { task: string; entry: TaskEvaluation }) {
  const baseline = entry.models.baseline
  // Only offer models that were actually scored, in a fixed order.
  const available = MODEL_LABELS.filter(({ key }) => entry.models[key])
  // Default to whichever model is serving requests, so the page opens on the
  // numbers that describe the live system.
  const active: ModelKey = entry.active_model ?? (entry.models.trained ? 'trained' : 'baseline')
  const [view, setView] = useState<ModelKey>(active)

  // `baseline` is the only model guaranteed present (there is no artefact until
  // train.py has run), so it is the fallback.
  const shown: ModelMetrics = entry.models[view] ?? baseline
  // Compare against the baseline, except when the baseline is what's shown --
  // then compare it against the serving model, so the arrow always contrasts
  // the rule-based floor with the real system.
  const reference: ModelMetrics | undefined =
    view === 'baseline' ? entry.models[active] : baseline

  const title = task === 'topic' ? 'การจำแนกหัวข้อ (Topic)' : 'การวิเคราะห์ความรู้สึก (Sentiment)'

  const name = (slug: string) => entry.label_names?.[slug]?.thai ?? slug

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
          {title}
        </h3>
        <div
          className="flex rounded-xl border border-slate-200 p-0.5 dark:border-slate-700"
          role="group"
          aria-label="เลือกโมเดล"
        >
          {available.map(({ key, thai }) => (
            <button
              key={key}
              type="button"
              onClick={() => setView(key)}
              aria-pressed={view === key}
              className={`rounded-lg px-3 py-1 text-xs font-medium ${
                view === key
                  ? 'bg-brand-600 text-white'
                  : 'text-slate-500 hover:bg-slate-50 dark:text-slate-400 dark:hover:bg-slate-800'
              }`}
            >
              {thai}
              {key === active && (
                <span
                  aria-hidden="true"
                  className={`ml-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle ${
                    view === key ? 'bg-white/90' : 'bg-teal-500'
                  }`}
                />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Says plainly which model answers API requests, so the tables on this
          page can never be mistaken for describing something else. */}
      <p className="text-xs text-slate-500 dark:text-slate-400" lang="th">
        <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-teal-500 align-middle" />
        โมเดลที่ใช้งานจริง:{' '}
        <strong className="text-slate-700 dark:text-slate-200">
          {MODEL_LABELS.find((m) => m.key === active)?.thai ?? active}
        </strong>
        {entry.models[active]?.algorithm && ` (${entry.models[active]?.algorithm})`}
      </p>

      {/* Headline metrics, the shown model beside its reference and the random floor. */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Accuracy"
          value={shown.accuracy}
          compare={reference?.accuracy}
        />
        <MetricCard
          label="Precision (macro)"
          value={shown.precision_macro}
          compare={reference?.precision_macro}
        />
        <MetricCard
          label="Recall (macro)"
          value={shown.recall_macro}
          compare={reference?.recall_macro}
        />
        <MetricCard
          label="F1-score (macro)"
          value={shown.f1_macro}
          compare={reference?.f1_macro}
          emphasis
        />
      </div>

      <div className="card p-4 text-sm">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <span className="text-slate-500 dark:text-slate-400" lang="th">
            อัลกอริทึม: <strong className="text-slate-700 dark:text-slate-200">{shown.algorithm}</strong>
          </span>
          <span className="text-slate-500 dark:text-slate-400" lang="th">
            ชุดทดสอบ: <strong className="tabular-nums text-slate-700 dark:text-slate-200">{shown.support}</strong> เอกสาร
          </span>
          <span className="text-slate-500 dark:text-slate-400" lang="th">
            เดาสุ่มได้: <strong className="tabular-nums text-slate-700 dark:text-slate-200">
              {(entry.random_baseline_accuracy * 100).toFixed(1)}%
            </strong> ({entry.labels.length} คลาส)
          </span>
          {shown.cross_validation && (
            <span className="text-slate-500 dark:text-slate-400" lang="th">
              Cross-validation ({shown.cross_validation.folds}-fold):{' '}
              <strong className="tabular-nums text-slate-700 dark:text-slate-200">
                {shown.cross_validation.mean.toFixed(3)} ± {shown.cross_validation.std.toFixed(3)}
              </strong>
            </span>
          )}
        </div>
      </div>

      {/* Per-class table: where a macro average hides the real weaknesses. */}
      <div className="card overflow-x-auto p-5">
        <h4 className="mb-3 font-semibold text-slate-900 dark:text-white">
          ผลแยกตามคลาส
        </h4>
        <table className="w-full min-w-[520px] text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400 dark:border-slate-700">
              <th className="pb-2 pr-4 font-medium">คลาส</th>
              <th className="pb-2 pr-4 text-right font-medium">Precision</th>
              <th className="pb-2 pr-4 text-right font-medium">Recall</th>
              <th className="pb-2 pr-4 text-right font-medium">F1</th>
              <th className="pb-2 text-right font-medium">Support</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-700/60">
            {entry.labels.map((slug) => {
              const row = shown.per_class[slug]
              if (!row) return null
              return (
                <tr key={slug}>
                  <td className="py-2 pr-4">
                    <span
                      className="mr-2 inline-block size-2 rounded-full align-middle"
                      style={{ backgroundColor: entry.label_names?.[slug]?.color }}
                    />
                    <span lang="th">{name(slug)}</span>
                  </td>
                  <td className="py-2 pr-4 text-right tabular-nums">
                    {row.precision.toFixed(3)}
                  </td>
                  <td className="py-2 pr-4 text-right tabular-nums">
                    {row.recall.toFixed(3)}
                  </td>
                  <td className="py-2 pr-4 text-right">
                    <F1Cell value={row.f1} />
                  </td>
                  <td className="py-2 text-right tabular-nums text-slate-400">
                    {row.support}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <ConfusionMatrix
        labels={shown.confusion_matrix.labels}
        matrix={shown.confusion_matrix.matrix}
        name={name}
      />
    </section>
  )
}

function MetricCard({
  label,
  value,
  compare,
  emphasis = false,
}: {
  label: string
  value: number
  compare?: number
  emphasis?: boolean
}) {
  const delta = compare === undefined ? null : value - compare
  return (
    <div className="card p-5">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p
        className={`mt-1 text-3xl font-semibold tabular-nums ${
          emphasis ? 'text-brand-600 dark:text-brand-400' : 'text-slate-800 dark:text-slate-100'
        }`}
      >
        {(value * 100).toFixed(1)}%
      </p>
      {delta !== null && (
        <p
          className={`mt-1 text-xs tabular-nums ${
            delta > 0.001
              ? 'text-positive'
              : delta < -0.001
                ? 'text-negative'
                : 'text-slate-400'
          }`}
        >
          {delta >= 0 ? '+' : ''}
          {(delta * 100).toFixed(1)} จุด เทียบอีกโมเดล
        </p>
      )}
    </div>
  )
}

function F1Cell({ value }: { value: number }) {
  // Bar plus number: the value is readable without relying on colour.
  return (
    <span className="inline-flex items-center gap-2">
      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
        <span
          className="block h-full rounded-full bg-brand-500"
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
      </span>
      <span className="w-11 text-right tabular-nums">{value.toFixed(3)}</span>
    </span>
  )
}

function ConfusionMatrix({
  labels,
  matrix,
  name,
}: {
  labels: string[]
  matrix: number[][]
  name: (slug: string) => string
}) {
  const max = Math.max(1, ...matrix.flat())

  return (
    <div className="card p-5">
      <h4 className="font-semibold text-slate-900 dark:text-white">Confusion Matrix</h4>
      <p className="mb-4 mt-0.5 text-xs text-slate-500 dark:text-slate-400" lang="th">
        แถว = คลาสจริง · คอลัมน์ = คลาสที่โมเดลทำนาย · แนวทแยงคือการทำนายถูก
      </p>

      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-surface p-1 dark:bg-surface-dark" />
              {labels.map((slug) => (
                <th
                  key={slug}
                  className="h-24 w-8 p-1 align-bottom font-medium text-slate-500 dark:text-slate-400"
                >
                  <span
                    className="inline-block whitespace-nowrap"
                    style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                    lang="th"
                  >
                    {name(slug)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => {
              const total = row.reduce((sum, value) => sum + value, 0)
              return (
                <tr key={labels[rowIndex]}>
                  <th
                    className="sticky left-0 z-10 whitespace-nowrap bg-surface pr-2 text-right font-medium text-slate-600 dark:bg-surface-dark dark:text-slate-300"
                    lang="th"
                  >
                    {name(labels[rowIndex])}
                    <span className="ml-1 text-slate-400">({total})</span>
                  </th>
                  {row.map((value, columnIndex) => {
                    const correct = rowIndex === columnIndex
                    // Sequential single-hue ramp by magnitude; the diagonal uses
                    // the positive hue so hits and misses are distinguishable
                    // by more than intensity alone.
                    const intensity = value === 0 ? 0 : 0.15 + (value / max) * 0.85
                    const background = correct
                      ? `color-mix(in oklab, var(--chart-positive) ${intensity * 100}%, transparent)`
                      : `color-mix(in oklab, var(--chart-negative) ${intensity * 100}%, transparent)`
                    return (
                      <td
                        key={columnIndex}
                        className="border border-slate-100 p-0 text-center dark:border-slate-800"
                        style={{ background: value ? background : undefined }}
                        title={`จริง: ${name(labels[rowIndex])} → ทำนาย: ${name(labels[columnIndex])} = ${value}`}
                      >
                        <span
                          className={`block px-1.5 py-1 tabular-nums ${
                            value === 0
                              ? 'text-slate-300 dark:text-slate-600'
                              : 'font-medium text-slate-800 dark:text-slate-100'
                          }`}
                        >
                          {value}
                        </span>
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-sm bg-positive" /> ทำนายถูก
        </span>
        <span className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-sm bg-negative" /> ทำนายผิด
        </span>
        <span>ความเข้มของสีแสดงจำนวนเอกสาร</span>
      </div>
    </div>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wide text-slate-400" lang="th">
        {label}
      </dt>
      <dd className="mt-0.5 font-semibold text-slate-800 dark:text-slate-100">{value}</dd>
    </div>
  )
}
