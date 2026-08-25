import { useEffect, useState } from 'react'

export default function LiveDetection() {
  const [samples, setSamples] = useState([])
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/samples')
      .then((r) => r.json())
      .then(setSamples)
      .catch((e) => setError(e.message))
  }, [])

  const runDetection = async () => {
    if (!samples[selectedIdx]) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(samples[selectedIdx].transaction),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const txn = samples[selectedIdx]?.transaction

  return (
    <div className="max-w-3xl">
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
        <label className="text-xs uppercase text-slate-500">Try a transaction</label>
        <div className="flex gap-3 mt-2">
          <select
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm"
            value={selectedIdx}
            onChange={(e) => { setSelectedIdx(Number(e.target.value)); setResult(null) }}
          >
            {samples.map((s, i) => (
              <option key={i} value={i}>{s.label}</option>
            ))}
          </select>
          <button
            onClick={runDetection}
            disabled={loading || !samples.length}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded-lg"
          >
            {loading ? 'Scoring…' : 'Run Detection'}
          </button>
        </div>

        {txn && (
          <div className="grid grid-cols-4 gap-3 mt-4 text-xs">
            {Object.entries(txn).map(([k, v]) => (
              <div key={k} className="bg-slate-800/60 rounded px-2 py-1.5">
                <div className="text-slate-500">{k}</div>
                <div className="text-slate-200 truncate">{String(v)}</div>
              </div>
            ))}
          </div>
        )}

        {error && <p className="text-red-400 text-sm mt-3">{error}</p>}

        {result && (
          <div className="mt-5 border-t border-slate-800 pt-5">
            <div className="flex items-center gap-4">
              <div
                className={`text-lg font-bold px-4 py-2 rounded-lg ${
                  result.prediction === 'fraud'
                    ? 'bg-red-500/20 text-red-300 border border-red-500/40'
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                }`}
              >
                {result.prediction === 'fraud' ? '⚠ FRAUD' : '✓ LEGITIMATE'}
              </div>
              <div className="flex-1">
                <div className="text-xs text-slate-500 mb-1">
                  Fraud probability: {(result.fraud_probability * 100).toFixed(2)}%
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${result.prediction === 'fraud' ? 'bg-red-500' : 'bg-emerald-500'}`}
                    style={{ width: `${Math.max(2, result.fraud_probability * 100)}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="mt-4">
              <h4 className="text-xs uppercase text-slate-500 mb-2">
                Top signals the model relies on globally (this transaction's values)
              </h4>
              <div className="grid grid-cols-3 gap-2">
                {result.top_model_signals.map((sig) => (
                  <div key={sig} className="bg-slate-800/60 rounded px-2 py-1.5 text-xs">
                    <div className="text-slate-500">{sig}</div>
                    <div className="text-slate-200">{String(result.features[sig])}</div>
                  </div>
                ))}
              </div>
            </div>
            {!result.features._known_pair_in_snapshot && (
              <p className="text-xs text-amber-400 mt-3">
                Note: one or both accounts weren't in the training-time graph snapshot — graph
                features fell back to "new/isolated account" defaults for this scoring pass.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
