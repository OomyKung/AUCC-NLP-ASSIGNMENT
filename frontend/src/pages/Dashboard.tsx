/** Dashboard: headline figures, the four charts, and the keyword cloud. */

import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  SentimentDonut,
  TopicBarChart,
  TopicSentimentChart,
  TrendChart,
} from '../components/charts'
import {
  ChartCard,
  ChartSkeleton,
  EmptyState,
  ErrorState,
  KeywordCloud,
  LoadingState,
  NewsCard,
  StatCard,
  icons,
} from '../components/ui'
import { useAsync, useChartColors, useTheme } from '../hooks'
import { api } from '../services/api'
import type { TrendGranularity } from '../types'

export default function Dashboard() {
  const { theme } = useTheme()
  const colors = useChartColors(theme)
  const navigate = useNavigate()

  const [granularity, setGranularity] = useState<TrendGranularity>('daily')
  const [normalise, setNormalise] = useState(false)

  const stats = useAsync(() => api.statistics(), [])
  const trend = useAsync(() => api.trend(granularity), [granularity])
  const topics = useAsync(() => api.topics(), [])
  const recent = useAsync(() => api.news({ page_size: 3, sort: 'newest' }), [])

  if (stats.error) return <ErrorState message={stats.error} onRetry={stats.reload} />

  if (stats.loading && !stats.data) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          {Array.from({ length: 5 }).map((_, index) => (
            <div key={index} className="card h-28 animate-pulse" />
          ))}
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          <ChartSkeleton />
          <ChartSkeleton />
        </div>
      </div>
    )
  }

  const data = stats.data
  if (!data) return null

  if (data.total_news === 0) {
    return (
      <EmptyState
        title="ยังไม่มีข้อมูลในระบบ"
        hint="นำเข้าแชทสดจาก YouTube หรือวางข้อความข่าวเพื่อเริ่มวิเคราะห์"
        action={
          <Link to="/analyze" className="btn-primary mt-3">
            เริ่มวิเคราะห์ข่าว
          </Link>
        }
      />
    )
  }

  const topicColor = (slug: string) =>
    topics.data?.find((item) => item.slug === slug)?.color

  return (
    <div className="space-y-6">
      {/* ---------------------------------------------------------- header */}
      <section className="card animate-rise relative overflow-hidden p-6">
        {/* Decorative wash. Purely cosmetic: no text sits on the saturated end,
            so contrast is unchanged. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 opacity-[0.07] dark:opacity-[0.12]"
          style={{
            background:
              'radial-gradient(60rem 20rem at 12% -20%, var(--chart-brand), transparent 60%),' +
              'radial-gradient(40rem 18rem at 88% 120%, var(--chart-positive), transparent 60%)',
          }}
        />
        <div className="relative">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
            ภาพรวมการวิเคราะห์
          </h2>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300" lang="th">
            วิเคราะห์แล้ว{' '}
            <strong className="text-brand-700 dark:text-brand-300">
              {data.total_news.toLocaleString()}
            </strong>{' '}
            เอกสาร จากข้อความแชทจริง{' '}
            <strong className="text-brand-700 dark:text-brand-300">
              {data.total_messages.toLocaleString()}
            </strong>{' '}
            ข้อความ ใน {data.total_streams} สตรีม
          </p>
        </div>
      </section>

      {/* ------------------------------------------------------ stat cards */}
      <section
        className="stagger grid gap-4 sm:grid-cols-2 xl:grid-cols-5"
        aria-label="สรุปภาพรวม"
      >
        <StatCard
          label="Total News"
          value={data.total_news}
          icon={icons.documents}
          hint={`${data.chat_windows} ช่วงแชท · ${data.articles} บทความ`}
        />
        <StatCard
          label="Positive"
          value={data.positive}
          tone="positive"
          icon={icons.up}
          share={data.total_news ? data.positive / data.total_news : 0}
          hint={`เชิงบวก · ${pct(data.positive, data.total_news)}`}
        />
        <StatCard
          label="Neutral"
          value={data.neutral}
          tone="neutral"
          icon={icons.minus}
          share={data.total_news ? data.neutral / data.total_news : 0}
          hint={`เป็นกลาง · ${pct(data.neutral, data.total_news)}`}
        />
        <StatCard
          label="Negative"
          value={data.negative}
          tone="negative"
          icon={icons.down}
          share={data.total_news ? data.negative / data.total_news : 0}
          hint={`เชิงลบ · ${pct(data.negative, data.total_news)}`}
        />
        <StatCard
          label="Most Common"
          value={data.most_common_topic_label ?? '—'}
          icon={icons.tag}
          hint={`${data.most_common_topic_count.toLocaleString()} เอกสาร`}
        />
      </section>

      {/* Corpus context: makes the scale of the underlying data visible. */}
      <section className="card animate-rise flex flex-wrap items-center gap-x-8 gap-y-3 px-5 py-4 text-sm">
        <Figure label="ข้อความแชททั้งหมด" value={data.total_messages.toLocaleString()} />
        <Figure label="วิเคราะห์ความรู้สึกแล้ว" value={data.scored_messages.toLocaleString()} />
        <Figure label="กรองเป็นสแปม" value={data.noise_messages.toLocaleString()} />
        <Figure label="สตรีมที่เก็บข้อมูล" value={String(data.total_streams)} />
        <Figure
          label="ความมั่นใจเฉลี่ย (หัวข้อ)"
          value={`${Math.round(data.avg_topic_confidence * 100)}%`}
        />
        <Figure
          label="ความมั่นใจเฉลี่ย (ความรู้สึก)"
          value={`${Math.round(data.avg_sentiment_confidence * 100)}%`}
        />
      </section>

      {/* ---------------------------------------------------------- charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard
          title="News by Topic"
          subtitle="จำนวนเอกสารแยกตามหมวดหมู่ · คลิกแท่งเพื่อกรอง"
        >
          <TopicBarChart
            data={data.by_topic}
            colors={colors}
            onSelect={(topic) => navigate(`/explorer?topic=${topic}`)}
          />
        </ChartCard>

        <ChartCard title="Sentiment Distribution" subtitle="สัดส่วนความรู้สึกของทั้งคลัง">
          <SentimentDonut
            data={data.by_sentiment}
            colors={colors}
            total={data.total_news}
          />
        </ChartCard>

        <ChartCard
          title="News Trend"
          subtitle="ปริมาณเอกสารตามช่วงเวลา แยกตามความรู้สึก"
          actions={
            <div
              className="flex rounded-xl border border-slate-200 p-0.5 dark:border-slate-700"
              role="group"
              aria-label="ช่วงเวลา"
            >
              {(['daily', 'weekly', 'monthly'] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setGranularity(option)}
                  aria-pressed={granularity === option}
                  className={`rounded-lg px-2.5 py-1 text-xs font-medium capitalize transition-colors ${
                    granularity === option
                      ? 'bg-brand-600 text-white'
                      : 'text-slate-500 hover:bg-slate-50 dark:text-slate-400 dark:hover:bg-slate-800'
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          }
        >
          {trend.loading && !trend.data ? (
            <ChartSkeleton />
          ) : trend.error ? (
            <ErrorState message={trend.error} onRetry={trend.reload} />
          ) : (
            <TrendChart
              points={trend.data?.points ?? []}
              colors={colors}
              granularity={granularity}
            />
          )}
        </ChartCard>

        <ChartCard
          title="Topic × Sentiment"
          subtitle="ความสัมพันธ์ระหว่างการจำแนกหัวข้อและการวิเคราะห์ความรู้สึก"
          actions={
            <button
              type="button"
              onClick={() => setNormalise((value) => !value)}
              aria-pressed={normalise}
              className={`rounded-xl border px-2.5 py-1 text-xs font-medium transition-colors ${
                normalise
                  ? 'border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300'
                  : 'border-slate-200 text-slate-500 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800'
              }`}
            >
              แสดงเป็น %
            </button>
          }
          footer={
            <p className="text-xs text-slate-400" lang="th">
              หมวดที่มีเอกสารน้อยจะเห็นสัดส่วนชัดกว่าเมื่อเปิดโหมดเปอร์เซ็นต์
            </p>
          }
        >
          <TopicSentimentChart
            rows={data.topic_sentiment}
            colors={colors}
            normalise={normalise}
          />
        </ChartCard>
      </div>

      {/* ------------------------------------------------------- keywords */}
      <ChartCard
        title="Keyword Cloud"
        subtitle={`คำสำคัญที่พบมากที่สุด ${data.top_keywords.length} คำ · คลิกเพื่อค้นหา`}
      >
        <KeywordCloud
          keywords={data.top_keywords}
          onSelect={(word) => navigate(`/explorer?search=${encodeURIComponent(word)}`)}
        />
      </ChartCard>

      {/* --------------------------------------------------- recent items */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold text-slate-900 dark:text-white">
            เอกสารล่าสุด
          </h2>
          <Link
            to="/explorer"
            className="text-sm text-brand-600 hover:underline dark:text-brand-400"
          >
            ดูทั้งหมด →
          </Link>
        </div>

        {recent.loading && !recent.data ? (
          <LoadingState rows={2} />
        ) : recent.error ? (
          <ErrorState message={recent.error} onRetry={recent.reload} />
        ) : (
          <div className="grid gap-4 lg:grid-cols-3">
            {recent.data?.items.map((item) => (
              <NewsCard
                key={item.id}
                item={item}
                topicLabel={
                  topics.data?.find((t) => t.slug === item.topic)?.thai ?? item.topic
                }
                topicColor={topicColor(item.topic)}
                sentimentLabel={
                  data.by_sentiment.find((s) => s.key === item.sentiment)?.label ??
                  item.sentiment
                }
              />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

/** Percentage of a total, guarding against an empty corpus. */
function pct(part: number, total: number): string {
  if (!total) return '0%'
  return `${Math.round((part / total) * 100)}%`
}

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-slate-400" lang="th">
        {label}
      </p>
      <p className="font-semibold tabular-nums text-slate-800 dark:text-slate-100">
        {value}
      </p>
    </div>
  )
}
