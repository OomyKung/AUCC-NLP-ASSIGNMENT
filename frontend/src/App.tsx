import { BrowserRouter, Route, Routes } from 'react-router-dom'
import AppLayout from './layouts/AppLayout'
import Dashboard from './pages/Dashboard'
import Explorer from './pages/Explorer'
import NewsDetail from './pages/NewsDetail'
import Analyze from './pages/Analyze'
import PipelinePage from './pages/Pipeline'
import { AboutPage } from './pages/About'
import EvaluationPage from './pages/Evaluation'
import { EmptyState } from './components/ui'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="explorer" element={<Explorer />} />
          <Route path="news/:id" element={<NewsDetail />} />
          <Route path="analyze" element={<Analyze />} />
          <Route path="pipeline" element={<PipelinePage />} />
          <Route path="evaluation" element={<EvaluationPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route
            path="*"
            element={
              <EmptyState
                title="ไม่พบหน้าที่ต้องการ"
                hint="ลองเลือกเมนูจากแถบด้านซ้าย"
              />
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
