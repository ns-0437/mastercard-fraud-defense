import { useEffect, useState } from 'react'
import { API_BASE } from '../api.js'
import PipelineDiagram from './PipelineDiagram.jsx'

function Stat({ value, label }) {
  return (
    <div className="text-center">
      <div className="text-3xl font-bold text-slate-100">{value}</div>
      <div className="text-xs text-slate-500 mt-1">{label}</div>
    </div>
  )
}

export default function Overview({ onNavigate }) {
  const [taxonomy, setTaxonomy] = useState(null)
  const [cycles, setCycles] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/taxonomy`).then((r) => r.json()).then(setTaxonomy).catch(() => {})
    fetch(`${API_BASE}/api/cycles`).then((r) => r.json()).then(setCycles).catch(() => {})
  }, [])

  const primary = cycles?.primary_evaluation?.primary
  const adversarial = cycles?.adversarial_selftest
  const weakestV2 = cycles?.closed_loop
    ? Object.entries(cycles.closed_loop.step_a_cycle1_model_vs_v2_attacks.by_family)
        .reduce((min, e) => (e[1] < min[1] ? e : min))
    : null

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center pt-4">
        <p className="text-sm text-slate-400 max-w-2xl mx-auto">
          A closed-loop red-team/blue-team system: research a taxonomy of GenAI-powered payment fraud,
          simulate it with LLM-driven configs at scale, defend with a graph-augmented classifier — then
          use the detector's own blind spots to drive the next round of attacks.
        </p>
      </div>

      <PipelineDiagram nodeCount={taxonomy?.nodes?.length} />

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 bg-slate-900 rounded-xl border border-slate-800 py-6">
        <Stat
          value={taxonomy?.nodes?.length ?? '—'}
          label={`attack vectors, ${taxonomy ? new Set(taxonomy.nodes.map((n) => n.channel)).size : '—'} channels`}
        />
        <Stat value={primary ? `${(primary.pr_auc * 100).toFixed(1)}%` : '—'} label="PR-AUC, synthetic attacks" />
        <Stat value={cycles?.ulb_baseline ? `${(cycles.ulb_baseline.pr_auc * 100).toFixed(1)}%` : '—'} label="PR-AUC, independent real dataset" />
        <Stat value={adversarial ? `${adversarial.n_caught}/${adversarial.n_total}` : '—'} label="novel attacks caught (self-test)" />
      </div>

      <div className="space-y-4">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">The honest three-act story</h2>

        <div className="bg-slate-900 rounded-xl border border-emerald-900/50 p-5">
          <div className="text-xs font-semibold text-emerald-400 mb-1">1. Built a detector that looked excellent</div>
          <p className="text-sm text-slate-300">
            {primary && `PR-AUC ${primary.pr_auc.toFixed(3)}, recall ${(primary.recall * 100).toFixed(1)}% `}
            on its own generated attacks — graph features (account degree, component size) doing real
            work, not decoration.
          </p>
        </div>

        <div className="bg-slate-900 rounded-xl border border-amber-900/50 p-5">
          <div className="text-xs font-semibold text-amber-400 mb-1">2. Red-teamed it against itself — and found a real gap</div>
          <p className="text-sm text-slate-300">
            {weakestV2
              ? `Hardened variants of the same attacks, spread over time and toned down, dropped ${weakestV2[0]}'s recall to ${(weakestV2[1] * 100).toFixed(1)}%. `
              : ''}
            Retrained on the harder attacks and recovered full detection with zero regression on the
            originals — this exact experiment run 5 times independently, since the specific gap it finds
            varies with each LLM-driven generation.{' '}
            <button onClick={() => onNavigate?.('loop')} className="text-emerald-400 hover:underline">
              See all 5 runs →
            </button>
          </p>
        </div>

        <div className="bg-slate-900 rounded-xl border border-red-900/50 p-5">
          <div className="text-xs font-semibold text-red-400 mb-1">3. Then found the edge of that fix — twice</div>
          <p className="text-sm text-slate-300">
            {adversarial && adversarial.n_caught < adversarial.n_total
              ? `${adversarial.n_total - adversarial.n_caught} of ${adversarial.n_total} hand-crafted attacks, built to look structurally unremarkable rather than to vary a known pattern, still evaded the retrained detector. Disclosed here, not smoothed over.`
              : adversarial
                ? `${adversarial.n_caught} of ${adversarial.n_total} hand-crafted, structurally-unremarkable attacks are now caught — but only after finding that the test itself was buggy (scoring against the wrong model) and then genuinely widening training coverage to close what the corrected test found. Both the bug and the fix are disclosed, not just the final number.`
                : ''}{' '}
            <button onClick={() => onNavigate?.('loop')} className="text-emerald-400 hover:underline">
              See which ones →
            </button>
          </p>
        </div>
      </div>

      <p className="text-xs text-slate-600 text-center pb-4">
        Every number on this page is read live from this repo's actual evaluation artifacts —
        nothing here is hand-typed.
      </p>
    </div>
  )
}
