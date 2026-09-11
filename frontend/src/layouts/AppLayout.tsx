/**
 * Application shell: sidebar, topbar, theme toggle, toast host.
 *
 * The sidebar is a persistent rail on desktop and an off-canvas drawer below
 * `lg`, which is what "collapse on smaller screens" means in practice.
 */

import { useEffect, useState } from 'react'
import type { ReactElement } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { ToastHost } from '../components/ui'
import { API_DOCS_URL } from '../services/api'
import { useTheme } from '../hooks'

interface NavItem {
  to: string
  label: string
  labelEn: string
  icon: ReactElement
}

const icon = (path: string) => (
  <svg
    viewBox="0 0 24 24"
    className="size-[18px] shrink-0"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.7"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d={path} />
  </svg>
)

const NAV: NavItem[] = [
  {
    to: '/',
    label: 'แดชบอร์ด',
    labelEn: 'Dashboard',
    icon: icon('M4 13h6V4H4v9Zm0 7h6v-4H4v4Zm10 0h6V11h-6v9Zm0-13h6V4h-6v3Z'),
  },
  {
    to: '/explorer',
    label: 'สำรวจข่าว',
    labelEn: 'News Explorer',
    icon: icon('M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14Zm9 2-4.35-4.35'),
  },
  {
    to: '/timeline',
    label: 'ไทม์ไลน์ข่าว',
    labelEn: 'News Timeline',
    icon: icon('M4 6h16M4 12h10M4 18h7'),
  },
  {
    to: '/analyze',
    label: 'วิเคราะห์ข่าว',
    labelEn: 'Analyze News',
    icon: icon('M12 5v14m-7-7h14'),
  },
  {
    to: '/pipeline',
    label: 'กระบวนการ NLP',
    labelEn: 'NLP Pipeline',
    icon: icon('M6 3v6a3 3 0 0 0 3 3h6a3 3 0 0 1 3 3v6M6 3H4m2 0h2m10 18h-2m2 0h2'),
  },
  {
    to: '/evaluation',
    label: 'ประเมินผลโมเดล',
    labelEn: 'Evaluation',
    icon: icon('M4 20V10m5 10V4m5 16v-7m5 7V7'),
  },
  {
    to: '/about',
    label: 'เกี่ยวกับโครงงาน',
    labelEn: 'About Project',
    icon: icon('M12 16v-4m0-4h.01M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z'),
  },
]

export default function AppLayout() {
  const { theme, toggle } = useTheme()
  const [open, setOpen] = useState(false)
  const location = useLocation()

  // Close the drawer on navigation, and reset scroll between pages.
  useEffect(() => {
    setOpen(false)
    window.scrollTo({ top: 0 })
  }, [location.pathname])

  // Escape closes the drawer.
  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  return (
    <div className="min-h-screen lg:flex">
      {/* Backdrop for the mobile drawer */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-slate-900/40 lg:hidden"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 shrink-0 flex-col border-r border-slate-200/80 bg-surface transition-transform duration-200 dark:border-slate-700/60 dark:bg-surface-dark lg:sticky lg:top-0 lg:h-screen lg:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
        aria-label="เมนูหลัก"
      >
        <div className="flex h-16 items-center gap-2.5 px-5">
          <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 text-sm font-bold text-white shadow-sm shadow-brand-600/25">
            N
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-900 dark:text-white">
              Thai News Intelligence
            </p>
            <p className="truncate text-[11px] text-slate-400">NLP Dashboard</p>
          </div>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-2 scroll-thin">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? 'bg-gradient-to-r from-brand-50 to-transparent font-medium text-brand-700 shadow-[inset_2px_0_0_0_var(--chart-brand)] dark:from-brand-500/15 dark:text-brand-300'
                    : 'text-slate-600 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800/60'
                }`
              }
            >
              {item.icon}
              <span className="min-w-0 flex-1">
                <span className="block truncate" lang="th">
                  {item.label}
                </span>
                <span className="block truncate text-[10px] text-slate-400">
                  {item.labelEn}
                </span>
              </span>
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-slate-200/80 p-3 dark:border-slate-700/60">
          <a
            href={API_DOCS_URL}
            target="_blank"
            rel="noreferrer noopener"
            className="btn-ghost w-full"
          >
            API Docs ↗
          </a>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-slate-200/80 bg-canvas/85 px-4 backdrop-blur dark:border-slate-700/60 dark:bg-canvas-dark/85 sm:px-6">
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="btn-ghost px-2.5 py-2 lg:hidden"
            aria-label="เปิดเมนู"
            aria-expanded={open}
          >
            <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M4 7h16M4 12h16M4 17h16" strokeLinecap="round" />
            </svg>
          </button>

          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base font-semibold text-slate-900 dark:text-white sm:text-lg">
              Thai News Intelligence
            </h1>
            <p className="hidden truncate text-xs text-slate-500 dark:text-slate-400 sm:block">
              NLP-powered Thai News Topic &amp; Sentiment Analysis
            </p>
          </div>

          <button
            type="button"
            onClick={toggle}
            className="btn-ghost px-2.5 py-2"
            aria-label={theme === 'dark' ? 'สลับเป็นโหมดสว่าง' : 'สลับเป็นโหมดมืด'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? (
              <svg viewBox="0 0 24 24" className="size-[18px]" fill="none" stroke="currentColor" strokeWidth="1.7">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" strokeLinecap="round" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="size-[18px]" fill="none" stroke="currentColor" strokeWidth="1.7">
                <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a6.5 6.5 0 0 0 9.8 9.8Z" />
              </svg>
            )}
          </button>
        </header>

        <main className="flex-1 px-4 py-6 sm:px-6 lg:py-8">
          <Outlet />
        </main>

        <footer className="border-t border-slate-200/80 px-4 py-4 text-center text-xs text-slate-400 dark:border-slate-700/60 sm:px-6">
          <p lang="th">
            โครงงานรายวิชา NLP · วิเคราะห์หัวข้อและความรู้สึกจากแชทสดข่าวภาษาไทย
          </p>
        </footer>
      </div>

      <ToastHost />
    </div>
  )
}
