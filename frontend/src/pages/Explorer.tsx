/**
 * News Explorer: search, filter, sort and paginate.
 *
 * Filter state lives in the URL, so a filtered view is shareable and the
 * browser's back button behaves as users expect.
 */

import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  EmptyState,
  ErrorState,
  LoadingState,
  NewsCard,
  Pagination,
} from '../components/ui'
import { useAsync, useDebounced } from '../hooks'
import { api } from '../services/api'
import type { NewsFilters } from '../types'

const SORTS: { value: NonNullable<NewsFilters['sort']>; label: string }[] = [
  { value: 'newest', label: 'ใหม่สุด' },
  { value: 'oldest', label: 'เก่าสุด' },
  { value: 'confidence', label: 'ความมั่นใจสูงสุด' },
  { value: 'messages', label: 'ข้อความมากสุด' },
]

const SOURCE_TYPES = [
  { value: '', label: 'ทุกแหล่ง' },
  { value: 'chat_window', label: 'ช่วงแชทสด' },
  { value: 'article', label: 'บทความ' },
]

export default function Explorer() {
  const [params, setParams] = useSearchParams()

  // Free text is kept in local state and debounced, so typing does not fire a
  // request per keystroke; the URL updates once the value settles.
  const [searchInput, setSearchInput] = useState(params.get('search') ?? '')
  const debouncedSearch = useDebounced(searchInput, 350)

  const topic = params.get('topic') ?? ''
  const sentiment = params.get('sentiment') ?? ''
  const sourceType = params.get('source_type') ?? ''
  const sort = (params.get('sort') as NewsFilters['sort']) ?? 'newest'
  const dateFrom = params.get('date_from') ?? ''
  const dateTo = params.get('date_to') ?? ''
  const page = Number(params.get('page') ?? 1)

  // Keep the input in sync when navigation changes the URL (e.g. a keyword
  // click on the dashboard, or the back button).
  useEffect(() => {
    setSearchInput(params.get('search') ?? '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.get('search')])

  useEffect(() => {
    const current = params.get('search') ?? ''
    if (debouncedSearch === current) return
    const next = new URLSearchParams(params)
    if (debouncedSearch) next.set('search', debouncedSearch)
    else next.delete('search')
    next.delete('page')
    setParams(next, { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSearch])

  const update = (key: string, value: string) => {
    const next = new URLSearchParams(params)
    if (value) next.set(key, value)
    else next.delete(key)
    next.delete('page') // a changed filter invalidates the page number
    setParams(next)
  }

  const goToPage = (value: number) => {
    const next = new URLSearchParams(params)
    if (value <= 1) next.delete('page')
    else next.set('page', String(value))
    setParams(next)
  }

  const clearAll = () => {
    setSearchInput('')
    setParams(new URLSearchParams())
  }

  const filters: NewsFilters = {
    search: debouncedSearch || undefined,
    topic: topic || undefined,
    sentiment: sentiment || undefined,
    source_type: sourceType || undefined,
    date_from: dateFrom || undefined,
    date_to: dateTo || undefined,
    sort,
    page,
    page_size: 12,
  }

  const news = useAsync(() => api.news(filters), [
    filters.search,
    filters.topic,
    filters.sentiment,
    filters.source_type,
    filters.date_from,
    filters.date_to,
    filters.sort,
    filters.page,
  ])
  const topics = useAsync(() => api.topics(), [])
  const sentiments = useAsync(() => api.sentiments(), [])

  const activeCount = [topic, sentiment, sourceType, dateFrom, dateTo, debouncedSearch]
    .filter(Boolean).length

  return (
    <div className="space-y-5">
      <header>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
          News Explorer
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400" lang="th">
          ค้นหาและกรองเอกสารที่วิเคราะห์แล้ว
        </p>
      </header>

      {/* ---------------------------------------------------- filter panel */}
      <section className="card p-4" aria-label="ตัวกรอง">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              ค้นหา
            </span>
            <div className="relative">
              <input
                type="search"
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
                placeholder="พิมพ์คำค้น เช่น อุบัติเหตุ, ฟุตบอล"
                className="input pl-9"
                lang="th"
              />
              <svg
                viewBox="0 0 24 24"
                className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-slate-400"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              >
                <circle cx="11" cy="11" r="7" />
                <path d="m20 20-3.5-3.5" strokeLinecap="round" />
              </svg>
            </div>
          </label>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              หมวดหมู่
            </span>
            <select
              value={topic}
              onChange={(event) => update('topic', event.target.value)}
              className="input"
            >
              <option value="">ทุกหมวด</option>
              {topics.data?.map((item) => (
                <option key={item.slug} value={item.slug}>
                  {item.thai} ({item.english})
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              ความรู้สึก
            </span>
            <select
              value={sentiment}
              onChange={(event) => update('sentiment', event.target.value)}
              className="input"
            >
              <option value="">ทุกความรู้สึก</option>
              {sentiments.data?.map((item) => (
                <option key={item.slug} value={item.slug}>
                  {item.thai} ({item.english})
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              เรียงลำดับ
            </span>
            <select
              value={sort}
              onChange={(event) => update('sort', event.target.value)}
              className="input"
            >
              {SORTS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              ประเภทแหล่งข้อมูล
            </span>
            <select
              value={sourceType}
              onChange={(event) => update('source_type', event.target.value)}
              className="input"
            >
              {SOURCE_TYPES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              ตั้งแต่วันที่
            </span>
            <input
              type="date"
              value={dateFrom.slice(0, 10)}
              onChange={(event) => update('date_from', event.target.value)}
              className="input"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
              ถึงวันที่
            </span>
            <input
              type="date"
              value={dateTo.slice(0, 10)}
              onChange={(event) => update('date_to', event.target.value)}
              className="input"
            />
          </label>

          <div className="flex items-end">
            <button
              type="button"
              onClick={clearAll}
              disabled={!activeCount}
              className="btn-ghost w-full"
            >
              ล้างตัวกรอง{activeCount ? ` (${activeCount})` : ''}
            </button>
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------- results */}
      {news.error ? (
        <ErrorState message={news.error} onRetry={news.reload} />
      ) : news.loading && !news.data ? (
        <LoadingState rows={3} />
      ) : !news.data?.items.length ? (
        <EmptyState
          title="ไม่พบเอกสารที่ตรงกับเงื่อนไข"
          hint="ลองลดจำนวนตัวกรอง หรือเปลี่ยนคำค้นหา"
          action={
            activeCount ? (
              <button type="button" onClick={clearAll} className="btn-primary mt-3">
                ล้างตัวกรองทั้งหมด
              </button>
            ) : undefined
          }
        />
      ) : (
        <>
          <p className="text-sm text-slate-500 tabular-nums dark:text-slate-400">
            พบ {news.data.total.toLocaleString()} รายการ · หน้า {news.data.page}/
            {news.data.pages}
          </p>

          <div
            className={`grid gap-4 md:grid-cols-2 xl:grid-cols-3 ${
              news.loading ? 'opacity-60 transition-opacity' : ''
            }`}
          >
            {news.data.items.map((item) => (
              <NewsCard
                key={item.id}
                item={item}
                topicLabel={
                  topics.data?.find((t) => t.slug === item.topic)?.thai ?? item.topic
                }
                topicColor={topics.data?.find((t) => t.slug === item.topic)?.color}
                sentimentLabel={
                  sentiments.data?.find((s) => s.slug === item.sentiment)?.thai ??
                  item.sentiment
                }
              />
            ))}
          </div>

          <Pagination
            page={news.data.page}
            pages={news.data.pages}
            total={news.data.total}
            onChange={goToPage}
          />
        </>
      )}
    </div>
  )
}
