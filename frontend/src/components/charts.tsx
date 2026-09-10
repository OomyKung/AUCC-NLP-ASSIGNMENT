/**
 * The four dashboard charts.
 *
 * Design decisions worth stating:
 *
 * - **Topic bars are coloured per category, but the axis label carries the
 *   meaning.** A 15-hue categorical set could never be colour-blind safe if the
 *   hue were the only way to tell series apart -- here every bar is named on the
 *   axis and repeated in the tooltip, so colour is decoration over a label that
 *   already identifies it. The same palette is reused for the card rails and
 *   badges, so a topic looks the same everywhere.
 * - **Sentiment uses a diverging scale**: teal-green and red poles either side
 *   of a neutral gray midpoint, validated for colour-vision deficiency.
 * - **No dual axes anywhere.** Where two measures matter they get two charts.
 * - Every series is labelled in a legend and in tooltips, so meaning never
 *   depends on colour alone.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ReactNode } from 'react'
import type {
  LabelCount,
  SentimentSlice,
  TopicSentimentRow,
  TrendGranularity,
  TrendPoint,
} from '../types'

type Colors = ReturnType<typeof import('../hooks').useChartColors>

/* -------------------------------------------------------------------------- */
/* Shared pieces                                                              */
/* -------------------------------------------------------------------------- */

/** Tooltip shell matching the card surface in both themes. */
function TooltipShell({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white/95 px-3 py-2 text-xs shadow-lg backdrop-blur dark:border-slate-700 dark:bg-slate-800/95">
      {children}
    </div>
  )
}

const axisStyle = (colors: Colors) => ({
  fontSize: 11,
  fill: colors.axis,
})

/** Shorten long Thai labels so the axis does not overlap. */
function truncate(value: string, max = 12): string {
  return value.length > max ? `${value.slice(0, max)}…` : value
}

/* -------------------------------------------------------------------------- */
/* Chart 1: documents by topic (bar)                                          */
/* -------------------------------------------------------------------------- */

export function TopicBarChart({
  data,
  colors,
  onSelect,
}: {
  data: LabelCount[]
  colors: Colors
  onSelect?: (topic: string) => void
}) {
  // Drop empty categories and sort by magnitude: a bar chart's job is comparison.
  const rows = data
    .filter((item) => item.count > 0)
    .sort((a, b) => b.count - a.count)
    .map((item) => ({ ...item, short: truncate(item.label) }))

  if (!rows.length) return <EmptyChart />

  return (
    <ResponsiveContainer width="100%" height={Math.max(240, rows.length * 30)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 40, bottom: 4, left: 8 }}>
        <CartesianGrid horizontal={false} stroke={colors.grid} strokeDasharray="2 4" />
        <XAxis type="number" tick={axisStyle(colors)} stroke={colors.grid} allowDecimals={false} />
        <YAxis
          type="category"
          dataKey="short"
          tick={axisStyle(colors)}
          stroke={colors.grid}
          width={92}
        />
        <Tooltip
          cursor={{ fill: colors.grid, opacity: 0.35 }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const row = payload[0].payload as LabelCount
            return (
              <TooltipShell>
                <p className="font-medium text-slate-900 dark:text-white" lang="th">
                  {row.label}
                </p>
                <p className="text-slate-500 dark:text-slate-400">{row.label_en}</p>
                <p className="mt-1 tabular-nums text-slate-700 dark:text-slate-200">
                  {row.count.toLocaleString()} เอกสาร
                </p>
              </TooltipShell>
            )
          }}
        />
        <Bar
          dataKey="count"
          radius={[0, 4, 4, 0]}
          maxBarSize={18}
          cursor={onSelect ? 'pointer' : undefined}
          onClick={
            onSelect
              ? (entry: unknown) => {
                  const row = entry as { payload?: LabelCount }
                  if (row?.payload) onSelect(row.payload.key)
                }
              : undefined
          }
        >
          {rows.map((row) => (
            <Cell key={row.key} fill={row.color || colors.brand} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

/* -------------------------------------------------------------------------- */
/* Chart 2: sentiment distribution (donut)                                    */
/* -------------------------------------------------------------------------- */

export function SentimentDonut({
  data,
  colors,
  total,
}: {
  data: SentimentSlice[]
  colors: Colors
  total: number
}) {
  const rows = data.filter((item) => item.count > 0)
  if (!rows.length) return <EmptyChart />

  const paint: Record<string, string> = {
    positive: colors.positive,
    neutral: colors.neutral,
    negative: colors.negative,
  }

  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={260}>
        <PieChart>
          <Pie
            data={rows}
            dataKey="count"
            nameKey="label"
            innerRadius="56%"
            outerRadius="84%"
            paddingAngle={2}
            animationDuration={600}
            // A 2px surface-coloured gap separates adjacent fills.
            stroke={colors.surface}
            strokeWidth={2}
          >
            {rows.map((row) => (
              <Cell key={row.key} fill={paint[row.key] ?? colors.neutral} />
            ))}
          </Pie>
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const row = payload[0].payload as SentimentSlice
              return (
                <TooltipShell>
                  <p className="font-medium text-slate-900 dark:text-white" lang="th">
                    {row.label}
                  </p>
                  <p className="mt-0.5 tabular-nums text-slate-700 dark:text-slate-200">
                    {row.count.toLocaleString()} · {row.percentage}%
                  </p>
                </TooltipShell>
              )
            }}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Hero number in the hole: the total the slices add up to. */}
      <div className="pointer-events-none absolute inset-0 grid place-items-center">
        <div className="text-center">
          <p className="text-2xl font-semibold tabular-nums text-slate-900 dark:text-white">
            {total.toLocaleString()}
          </p>
          <p className="text-[11px] text-slate-400" lang="th">
            เอกสาร
          </p>
        </div>
      </div>

      {/* Legend with values, so identity never depends on colour alone. */}
      <ul className="mt-3 flex flex-wrap justify-center gap-x-4 gap-y-1.5">
        {rows.map((row) => (
          <li key={row.key} className="flex items-center gap-1.5 text-xs">
            <span
              className="size-2.5 rounded-sm"
              style={{ backgroundColor: paint[row.key] }}
            />
            <span className="text-slate-600 dark:text-slate-300" lang="th">
              {row.label}
            </span>
            <span className="tabular-nums text-slate-400">{row.percentage}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Chart 3: volume over time (line)                                           */
/* -------------------------------------------------------------------------- */

export function TrendChart({
  points,
  colors,
  granularity,
}: {
  points: TrendPoint[]
  colors: Colors
  granularity: TrendGranularity
}) {
  if (!points.length) return <EmptyChart />

  // A single point cannot draw a line; show it as a bar instead so the panel is
  // never mysteriously blank.
  const single = points.length === 1

  const label = (period: string) => {
    if (granularity === 'monthly') return period
    if (granularity === 'weekly') return period.replace('-W', ' W')
    return period.slice(5) // MM-DD
  }

  const series = [
    { key: 'positive', name: 'เชิงบวก', color: colors.positive },
    { key: 'neutral', name: 'เป็นกลาง', color: colors.neutral },
    { key: 'negative', name: 'เชิงลบ', color: colors.negative },
  ] as const

  const rows = points.map((point) => ({ ...point, label: label(point.period) }))

  const tooltip = (
    <Tooltip
      cursor={{ stroke: colors.axis, strokeWidth: 1, strokeDasharray: '3 3' }}
      content={({ active, payload, label: axisLabel }) => {
        if (!active || !payload?.length) return null
        const row = payload[0].payload as TrendPoint
        return (
          <TooltipShell>
            <p className="font-medium text-slate-900 dark:text-white">{axisLabel}</p>
            <p className="mb-1 tabular-nums text-slate-500">
              รวม {row.count.toLocaleString()}
            </p>
            {series.map((item) => (
              <p key={item.key} className="flex items-center gap-1.5 tabular-nums">
                <span
                  className="size-2 rounded-sm"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-slate-600 dark:text-slate-300" lang="th">
                  {item.name}
                </span>
                <span className="ml-auto text-slate-500">{row[item.key]}</span>
              </p>
            ))}
          </TooltipShell>
        )
      }}
    />
  )

  if (single) {
    return (
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
          <CartesianGrid vertical={false} stroke={colors.grid} strokeDasharray="2 4" />
          <XAxis dataKey="label" tick={axisStyle(colors)} stroke={colors.grid} />
          <YAxis tick={axisStyle(colors)} stroke={colors.grid} allowDecimals={false} />
          {tooltip}
          <Legend
            wrapperStyle={{ fontSize: 11, color: colors.axis }}
            formatter={(value) => <span style={{ color: colors.axis }}>{value}</span>}
          />
          {series.map((item) => (
            <Bar
              key={item.key}
              dataKey={item.key}
              name={item.name}
              stackId="sentiment"
              fill={item.color}
              maxBarSize={64}
              stroke={colors.surface}
              strokeWidth={2}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
        <CartesianGrid vertical={false} stroke={colors.grid} strokeDasharray="2 4" />
        <XAxis dataKey="label" tick={axisStyle(colors)} stroke={colors.grid} />
        <YAxis tick={axisStyle(colors)} stroke={colors.grid} allowDecimals={false} />
        {tooltip}
        <Legend
          wrapperStyle={{ fontSize: 11 }}
          formatter={(value) => <span style={{ color: colors.axis }}>{value}</span>}
        />
        {series.map((item) => (
          <Line
            key={item.key}
            animationDuration={700}
            type="monotone"
            dataKey={item.key}
            name={item.name}
            stroke={item.color}
            strokeWidth={2}
            dot={{ r: 3, strokeWidth: 2, stroke: colors.surface }}
            activeDot={{ r: 5, strokeWidth: 2, stroke: colors.surface }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

/* -------------------------------------------------------------------------- */
/* Chart 4: topic x sentiment (stacked bar)                                   */
/* -------------------------------------------------------------------------- */

export function TopicSentimentChart({
  rows,
  colors,
  normalise = false,
}: {
  rows: TopicSentimentRow[]
  colors: Colors
  /** Show shares instead of counts, so small topics stay comparable. */
  normalise?: boolean
}) {
  if (!rows.length) return <EmptyChart />

  const data = rows.map((row) => {
    const total = row.total || 1
    const scale = (value: number) => (normalise ? (value / total) * 100 : value)
    return {
      ...row,
      short: truncate(row.label),
      p: scale(row.positive),
      u: scale(row.neutral),
      n: scale(row.negative),
    }
  })

  const series = [
    { key: 'p', name: 'เชิงบวก', color: colors.positive, raw: 'positive' as const },
    { key: 'u', name: 'เป็นกลาง', color: colors.neutral, raw: 'neutral' as const },
    { key: 'n', name: 'เชิงลบ', color: colors.negative, raw: 'negative' as const },
  ]

  return (
    <ResponsiveContainer width="100%" height={Math.max(260, data.length * 42)}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 16, bottom: 4, left: 8 }}
        stackOffset={normalise ? 'none' : undefined}
      >
        <CartesianGrid horizontal={false} stroke={colors.grid} strokeDasharray="2 4" />
        <XAxis
          type="number"
          tick={axisStyle(colors)}
          stroke={colors.grid}
          allowDecimals={false}
          domain={normalise ? [0, 100] : undefined}
          tickFormatter={normalise ? (value) => `${value}%` : undefined}
        />
        <YAxis
          type="category"
          dataKey="short"
          tick={axisStyle(colors)}
          stroke={colors.grid}
          width={92}
        />
        <Tooltip
          cursor={{ fill: colors.grid, opacity: 0.35 }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const row = payload[0].payload as TopicSentimentRow
            return (
              <TooltipShell>
                <p className="font-medium text-slate-900 dark:text-white" lang="th">
                  {row.label}
                </p>
                <p className="mb-1 tabular-nums text-slate-500">
                  รวม {row.total.toLocaleString()}
                </p>
                {series.map((item) => (
                  <p key={item.key} className="flex items-center gap-1.5 tabular-nums">
                    <span
                      className="size-2 rounded-sm"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-slate-600 dark:text-slate-300" lang="th">
                      {item.name}
                    </span>
                    <span className="ml-auto text-slate-500">
                      {row[item.raw]}
                      {row.total ? ` (${Math.round((row[item.raw] / row.total) * 100)}%)` : ''}
                    </span>
                  </p>
                ))}
              </TooltipShell>
            )
          }}
        />
        <Legend
          wrapperStyle={{ fontSize: 11 }}
          formatter={(value) => <span style={{ color: colors.axis }}>{value}</span>}
        />
        {series.map((item, index) => (
          <Bar
            key={item.key}
            dataKey={item.key}
            name={item.name}
            stackId="sentiment"
            fill={item.color}
            maxBarSize={20}
            // 2px surface gap between stacked segments.
            stroke={colors.surface}
            strokeWidth={2}
            radius={index === series.length - 1 ? [0, 4, 4, 0] : undefined}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

/* -------------------------------------------------------------------------- */

function EmptyChart() {
  return (
    <div className="grid h-[240px] place-items-center rounded-xl border border-dashed border-slate-200 text-sm text-slate-400 dark:border-slate-700">
      <p lang="th">ยังไม่มีข้อมูลเพียงพอสำหรับกราฟนี้</p>
    </div>
  )
}
