import { useEffect, useState } from 'react'

const CHANNEL_COLORS = {
  'card-not-present': '#f97316',
  'account-takeover': '#ef4444',
  'kyc-onboarding': '#eab308',
  'agentic-commerce': '#a855f7',
  'authorized-push-payment': '#ec4899',
  'mule-network': '#06b6d4',
  'ml-defense-attack': '#6366f1',
  'cross-cutting': '#64748b',
}

const W = 900
const H = 620
const PAD = 60

export default function TaxonomyGraph() {
  const [data, setData] = useState(null)
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/taxonomy')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="text-red-400">Failed to load taxonomy: {error}</div>
  if (!data) return <div className="text-slate-400">Loading taxonomy graph…</div>

  const xs = data.nodes.map((n) => n.x)
  const ys = data.nodes.map((n) => n.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const scaleX = (x) => PAD + ((x - minX) / (maxX - minX)) * (W - 2 * PAD)
  const scaleY = (y) => PAD + ((y - minY) / (maxY - minY)) * (H - 2 * PAD)
  const posById = Object.fromEntries(data.nodes.map((n) => [n.id, { x: scaleX(n.x), y: scaleY(n.y) }]))

  const selectedNode = data.nodes.find((n) => n.id === selected)

  return (
    <div className="flex gap-6">
      <div className="flex-1 bg-slate-900 rounded-xl border border-slate-800 p-4">
        <p className="text-sm text-slate-400 mb-2">
          {data.nodes.length} attack vectors · {data.edges.length} shared-technique links · click a node for detail
        </p>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[560px]">
          {data.edges.map((e, i) => {
            const a = posById[e.source], b = posById[e.target]
            if (!a || !b) return null
            return (
              <line
                key={i}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke="#334155" strokeWidth={1}
                opacity={selected && (e.source === selected || e.target === selected) ? 0.9 : 0.25}
              />
            )
          })}
          {data.nodes.map((n) => {
            const p = posById[n.id]
            const isSel = n.id === selected
            return (
              <g key={n.id} onClick={() => setSelected(n.id)} className="cursor-pointer">
                <circle
                  cx={p.x} cy={p.y} r={isSel ? 12 : 8}
                  fill={CHANNEL_COLORS[n.channel] || '#94a3b8'}
                  stroke={isSel ? '#fff' : 'none'} strokeWidth={2}
                />
                <text x={p.x + 12} y={p.y + 4} fontSize={10} fill="#cbd5e1">
                  {n.name.length > 28 ? n.name.slice(0, 26) + '…' : n.name}
                </text>
              </g>
            )
          })}
        </svg>
        <div className="flex flex-wrap gap-3 mt-2 text-xs">
          {Object.entries(CHANNEL_COLORS).map(([channel, color]) => (
            <span key={channel} className="flex items-center gap-1.5 text-slate-400">
              <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: color }} />
              {channel}
            </span>
          ))}
        </div>
      </div>

      <div className="w-96 bg-slate-900 rounded-xl border border-slate-800 p-5">
        {!selectedNode && (
          <p className="text-slate-400 text-sm">Select a node to see its mechanism, real-world grounding, and severity.</p>
        )}
        {selectedNode && (
          <div className="space-y-3">
            <div>
              <span
                className="text-xs px-2 py-0.5 rounded-full text-slate-950 font-medium"
                style={{ background: CHANNEL_COLORS[selectedNode.channel] }}
              >
                {selectedNode.channel}
              </span>
              <h3 className="text-lg font-semibold mt-2">{selectedNode.name}</h3>
            </div>
            <div className="flex gap-4 text-xs text-slate-400">
              <span>Severity: <b className="text-slate-200">{selectedNode.severity}</b></span>
              <span>Likelihood: <b className="text-slate-200">{selectedNode.likelihood}</b></span>
            </div>
            <div>
              <h4 className="text-xs uppercase text-slate-500 mb-1">Mechanism</h4>
              <p className="text-sm text-slate-300">{selectedNode.mechanism}</p>
            </div>
            <div>
              <h4 className="text-xs uppercase text-slate-500 mb-1">Real-world grounding</h4>
              <p className="text-sm text-slate-300">{selectedNode.grounding}</p>
            </div>
            <div>
              <h4 className="text-xs uppercase text-slate-500 mb-1">Shared techniques</h4>
              <div className="flex flex-wrap gap-1.5">
                {selectedNode.techniques.map((t) => (
                  <span key={t} className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded">{t}</span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
