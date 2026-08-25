import { useState } from 'react'
import Overview from './components/Overview.jsx'
import TaxonomyGraph from './components/TaxonomyGraph.jsx'
import GenerateFidelity from './components/GenerateFidelity.jsx'
import LiveDetection from './components/LiveDetection.jsx'
import ClosedLoopResults from './components/ClosedLoopResults.jsx'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'taxonomy', label: 'Attack Taxonomy' },
  { id: 'generate', label: 'Generate & Fidelity' },
  { id: 'detect', label: 'Live Detection' },
  { id: 'loop', label: 'Closed-Loop Results' },
]

export default function App() {
  const [tab, setTab] = useState('overview')

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-5">
        <h1 className="text-2xl font-semibold tracking-tight">AI Defense Lab for Payment Security</h1>
        <p className="text-slate-400 text-sm mt-1">
          Mastercard Innovation Challenge 2026 — Identify → Generate → Defend closed loop
        </p>
      </header>

      <nav className="flex gap-1 px-6 pt-4 border-b border-slate-800 overflow-x-auto">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-sm rounded-t-lg border-b-2 transition-colors whitespace-nowrap shrink-0 ${
              tab === t.id
                ? 'border-emerald-400 text-emerald-300 bg-slate-900'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="p-6">
        {tab === 'overview' && <Overview onNavigate={setTab} />}
        {tab === 'taxonomy' && <TaxonomyGraph />}
        {tab === 'generate' && <GenerateFidelity />}
        {tab === 'detect' && <LiveDetection />}
        {tab === 'loop' && <ClosedLoopResults />}
      </main>
    </div>
  )
}
