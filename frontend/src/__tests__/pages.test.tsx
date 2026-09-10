/**
 * Rendering tests for every page.
 *
 * These exist because a 200 from the dev server only proves index.html was
 * served -- it says nothing about whether React rendered or threw. Each test
 * mounts a real page against fixtures captured from the running API and asserts
 * that genuine values reach the DOM.
 */

import { describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import Dashboard from '../pages/Dashboard'
import Explorer from '../pages/Explorer'
import NewsDetail from '../pages/NewsDetail'
import Analyze from '../pages/Analyze'
import PipelinePage from '../pages/Pipeline'
import { AboutPage } from '../pages/About'
import EvaluationPage from '../pages/Evaluation'
import AppLayout from '../layouts/AppLayout'
import { fixtures } from './setup'

function mount(ui: React.ReactNode, path = '/') {
  return render(<MemoryRouter initialEntries={[path]}>{ui}</MemoryRouter>)
}

describe('Dashboard', () => {
  it('renders the five stat cards with real figures', async () => {
    mount(<Dashboard />)

    await waitFor(() =>
      expect(screen.getByText('Total News')).toBeTruthy(),
    )

    for (const label of ['Total News', 'Positive', 'Neutral', 'Negative', 'Most Common']) {
      expect(screen.getByText(label)).toBeTruthy()
    }

    // The total from the fixture must actually appear, not a placeholder.
    const total = fixtures.statistics.total_news.toLocaleString()
    expect(screen.getAllByText(total).length).toBeGreaterThan(0)
  })

  it('renders all four chart panels', async () => {
    mount(<Dashboard />)

    await waitFor(() => expect(screen.getByText('News by Topic')).toBeTruthy())
    expect(screen.getByText('Sentiment Distribution')).toBeTruthy()
    expect(screen.getByText('News Trend')).toBeTruthy()
    expect(screen.getByText('Topic × Sentiment')).toBeTruthy()
    expect(screen.getByText('Keyword Cloud')).toBeTruthy()
  })

  it('shows the most common topic in Thai', async () => {
    mount(<Dashboard />)
    const label = fixtures.statistics.most_common_topic_label
    await waitFor(() => expect(screen.getAllByText(label!).length).toBeGreaterThan(0))
  })

  it('renders keyword cloud entries from the corpus', async () => {
    mount(<Dashboard />)
    const first = fixtures.statistics.top_keywords[0].word
    await waitFor(() => expect(screen.getAllByText(first).length).toBeGreaterThan(0))
  })
})

describe('Explorer', () => {
  it('renders filters and result cards', async () => {
    mount(<Explorer />, '/explorer')

    await waitFor(() => expect(screen.getByText('News Explorer')).toBeTruthy())

    expect(screen.getByPlaceholderText(/พิมพ์คำค้น/)).toBeTruthy()
    expect(screen.getByText('หมวดหมู่')).toBeTruthy()
    expect(screen.getByText('ความรู้สึก')).toBeTruthy()
    expect(screen.getByText('เรียงลำดับ')).toBeTruthy()

    // A real document title from the fixture reaches the DOM.
    const title = fixtures.news.items[0].title
    await waitFor(() => expect(screen.getByText(title)).toBeTruthy())
  })

  it('populates the topic filter from the taxonomy', async () => {
    mount(<Explorer />, '/explorer')
    await waitFor(() =>
      expect(screen.getByText(/อุบัติเหตุ \(Accident\)/)).toBeTruthy(),
    )
  })
})

describe('NewsDetail', () => {
  it('renders the document with its NLP analysis', async () => {
    render(
      <MemoryRouter initialEntries={[`/news/${fixtures.detail.id}`]}>
        <Routes>
          <Route path="/news/:id" element={<NewsDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    await waitFor(() => expect(screen.getByText(fixtures.detail.title)).toBeTruthy())

    expect(screen.getByText('Topic Probability')).toBeTruthy()
    expect(screen.getByText('Sentiment Probability')).toBeTruthy()
    expect(screen.getByText('NLP Analysis')).toBeTruthy()

    // Token counts are real numbers from the analysis, not zeros.
    expect(
      screen.getAllByText(fixtures.detail.analysis.token_count.toLocaleString()).length,
    ).toBeGreaterThan(0)
  })
})

describe('Analyze', () => {
  it('renders both the text form and the YouTube import form', async () => {
    mount(<Analyze />, '/analyze')

    // "Analyze News" is both the page heading and the submit button, so the
    // heading is matched by role rather than by text.
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Analyze News' })).toBeTruthy(),
    )
    expect(screen.getByText('วิเคราะห์ข้อความข่าว')).toBeTruthy()
    expect(screen.getByText('นำเข้าแชทสดจาก YouTube')).toBeTruthy()

    // The submit button starts disabled: a blank form must not be submittable.
    const submit = screen.getByRole('button', { name: 'Analyze News' })
    expect((submit as HTMLButtonElement).disabled).toBe(true)
  })
})

describe('Pipeline', () => {
  it('renders all ten stages and the active backend table', async () => {
    mount(<PipelinePage />, '/pipeline')

    await waitFor(() => expect(screen.getByText('NLP Pipeline')).toBeTruthy())

    for (const stage of [
      'Raw Text',
      'Text Cleaning',
      'Thai Tokenization',
      'Stopword Removal',
      'Feature Extraction',
      'Topic Classification',
      'Sentiment Analysis',
      'Keyword Extraction',
      'Summary Generation',
      'Dashboard',
    ]) {
      await waitFor(() => expect(screen.getByText(stage)).toBeTruthy())
    }

    // Every stage's active backend is reported.
    for (const row of fixtures.pipeline) {
      expect(screen.getAllByText(row.active).length).toBeGreaterThan(0)
    }
  })
})

describe('About and Evaluation', () => {
  it('About renders every required project section', async () => {
    mount(<AboutPage />, '/about')

    await waitFor(() => expect(screen.getByText('เกี่ยวกับโครงงาน')).toBeTruthy())
    for (const heading of [
      'ปัญหา (Problem)',
      'วัตถุประสงค์ (Objective)',
      'เทคนิค NLP ที่ใช้ (NLP Techniques)',
      'สถาปัตยกรรมระบบ (System Architecture)',
      'ชุดข้อมูล (Dataset)',
      'โมเดล (Model)',
      'การประเมินผล (Evaluation)',
    ]) {
      expect(screen.getByText(heading)).toBeTruthy()
    }
  })

  it('Evaluation renders real metrics and the baseline comparison', async () => {
    mount(<EvaluationPage />, '/evaluation')

    await waitFor(() =>
      expect(screen.getByText('ประเมินผลโมเดล (Model Evaluation)')).toBeTruthy(),
    )

    // Both tasks, and the metrics that the specification requires.
    expect(screen.getByText('การจำแนกหัวข้อ (Topic)')).toBeTruthy()
    expect(screen.getByText('การวิเคราะห์ความรู้สึก (Sentiment)')).toBeTruthy()
    expect(screen.getAllByText('Accuracy').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Precision (macro)').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Recall (macro)').length).toBeGreaterThan(0)
    expect(screen.getAllByText('F1-score (macro)').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Confusion Matrix').length).toBe(2)

    // The real accuracy from the fixture must appear, not a placeholder.
    const accuracy = fixtures.evaluation.tasks.topic.models.trained.accuracy
    const rendered = `${(accuracy * 100).toFixed(1)}%`
    expect(screen.getAllByText(rendered).length).toBeGreaterThan(0)
  })
})

describe('AppLayout', () => {
  it('renders navigation and a working theme toggle', async () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<p>content</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    )

    expect(screen.getByText('แดชบอร์ด')).toBeTruthy()
    expect(screen.getByText('สำรวจข่าว')).toBeTruthy()
    expect(screen.getByText('ประเมินผลโมเดล')).toBeTruthy()

    // Light is the default; the toggle offers dark.
    const toggle = screen.getByRole('button', { name: 'สลับเป็นโหมดมืด' })
    expect(document.documentElement.classList.contains('dark')).toBe(false)
    toggle.click()
    await waitFor(() =>
      expect(document.documentElement.classList.contains('dark')).toBe(true),
    )
  })
})

describe('API error handling', () => {
  it('shows an actionable message when the backend is unreachable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )

    mount(<Dashboard />)

    await waitFor(() => expect(screen.getByText('เกิดข้อผิดพลาด')).toBeTruthy())
    expect(screen.getByText(/uvicorn app.main:app --reload/)).toBeTruthy()
  })
})
