import { useEffect, useState } from 'react'
import { API_BASE } from '../api.js'

function Metric({ label, value }) {
  return (
    <div className="bg-slate-800/60 rounded-lg px-3 py-2">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
      <h3 className="text-sm font-semibold text-slate-200 mb-3">{title}</h3>
      {children}
    </div>
  )
}

export default function ClosedLoopResults() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/cycles`)
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="text-red-400">Failed to load: {error}</div>
  if (!data) return <div className="text-slate-400">Loading…</div>

  const primary = data.primary_evaluation?.primary
  const perFamily = data.primary_evaluation?.per_family_recall || {}
  const ulb = data.ulb_baseline
  const loop = data.closed_loop

  return (
    <div className="space-y-6 max-w-5xl">
      <Section title="Phase 3 — Cycle-1 detector on PaySim (flagged as an easy-benchmark result — see Phase 4)">
        {primary && (
          <div className="grid grid-cols-5 gap-3">
            <Metric label="Precision" value={primary.precision.toFixed(4)} />
            <Metric label="Recall" value={primary.recall.toFixed(4)} />
            <Metric label="PR-AUC" value={primary.pr_auc.toFixed(4)} />
            <Metric label="ROC-AUC" value={primary.roc_auc.toFixed(4)} />
            <Metric label="FPR on legit" value={`${(primary.false_positive_rate_on_legit * 100).toFixed(3)}%`} />
          </div>
        )}
        <div className="grid grid-cols-4 gap-3 mt-3">
          {Object.entries(perFamily).map(([fam, r]) => (
            <Metric key={fam} label={fam} value={r.toFixed(4)} />
          ))}
        </div>
      </Section>

      <Section title="Independent methodology check — ULB Credit Card Fraud (different real dataset, no graph features possible)">
        {ulb && (
          <div className="grid grid-cols-5 gap-3">
            <Metric label="Precision" value={ulb.precision.toFixed(4)} />
            <Metric label="Recall" value={ulb.recall.toFixed(4)} />
            <Metric label="PR-AUC" value={ulb.pr_auc.toFixed(4)} />
            <Metric label="ROC-AUC" value={ulb.roc_auc.toFixed(4)} />
            <Metric label="FPR" value={`${(ulb.false_positive_rate * 100).toFixed(3)}%`} />
          </div>
        )}
        <p className="text-xs text-slate-500 mt-3">
          Same methodology, unrelated dataset with no account graph — a lower, more ordinary score here
          is expected and is reported as a scoped comparison, not a failure.
        </p>
      </Section>

      <Section title="Phase 4 — Closed loop: does hardening evade the cycle-1 detector, and does retraining fix it?">
        {loop && (
          <div className="space-y-4">
            <div>
              <h4 className="text-xs uppercase text-slate-500 mb-2">
                Step A — cycle-1 model vs. hardened (v2) attacks it never saw
              </h4>
              <div className="grid grid-cols-4 gap-3">
                {Object.entries(loop.step_a_cycle1_model_vs_v2_attacks.by_family).map(([fam, r]) => (
                  <div
                    key={fam}
                    className={`rounded-lg px-3 py-2 border ${
                      r < 0.5 ? 'bg-red-500/10 border-red-500/40' : 'bg-slate-800/60 border-transparent'
                    }`}
                  >
                    <div className="text-xs text-slate-500">{fam}</div>
                    <div className={`text-lg font-semibold ${r < 0.5 ? 'text-red-300' : 'text-slate-100'}`}>
                      {r.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-2">
                card_testing_burst_v2's near-zero recall here is the headline finding: the original
                "0.999 PR-AUC" detector was almost completely blind to the same attack spread over time
                instead of one burst.
              </p>
            </div>

            <div>
              <h4 className="text-xs uppercase text-slate-500 mb-2">
                Step B — cycle-2 model (retrained on v1+v2) recall by family
              </h4>
              <div className="grid grid-cols-4 gap-3">
                {Object.entries(loop.step_b_cycle2_model.by_family).map(([fam, r]) => (
                  <Metric key={fam} label={fam} value={r.toFixed(4)} />
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-xs uppercase text-slate-500 mb-2">Regression check (cycle-1 vs cycle-2 on original families)</h4>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-500 text-xs">
                    <th className="text-left py-1">Family</th>
                    <th className="text-right py-1">Cycle-1 recall</th>
                    <th className="text-right py-1">Cycle-2 recall</th>
                    <th className="text-right py-1">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(loop.regression_check.cycle1_v1_recall).map((fam) => {
                    const c1 = loop.regression_check.cycle1_v1_recall[fam]
                    const c2 = loop.regression_check.cycle2_v1_recall[fam]
                    const regressed = loop.regression_check.regressed_families.includes(fam)
                    return (
                      <tr key={fam} className="border-t border-slate-800">
                        <td className="py-1.5">{fam}</td>
                        <td className="text-right">{c1.toFixed(4)}</td>
                        <td className="text-right">{c2.toFixed(4)}</td>
                        <td className={`text-right ${regressed ? 'text-amber-400' : 'text-emerald-400'}`}>
                          {regressed ? 'regressed (disclosed)' : 'ok'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              <p className="text-xs text-slate-500 mt-2">
                {loop.regression_check.regressed_families.length === 0 ? (
                  <>No regression on any original family after adding hardened variants to training —
                  the closed loop closed the card-testing blind spot without trading away detection on
                  the others. See docs/PHASES.md Phase 4 for the full run history, including an earlier
                  run (before the LLM config path was funded) where one low-sample family did regress
                  and was reported as-is rather than tuned to force a pass.</>
                ) : (
                  <>{loop.regression_check.regressed_families.join(', ')} regressed and {
                    loop.regression_check.regressed_families.length === 1 ? 'was' : 'were'
                  } reported as-is rather than tuned further to force a clean pass — see
                  docs/PHASES.md Phase 4 for the full reasoning. The closed loop's primary claim —
                  closing a severe, real blind spot on card-testing — holds regardless.</>
                )}
              </p>
            </div>
          </div>
        )}
      </Section>
    </div>
  )
}
