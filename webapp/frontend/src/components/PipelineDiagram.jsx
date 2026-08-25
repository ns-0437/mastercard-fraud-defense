export default function PipelineDiagram({ nodeCount }) {
  const STAGES = [
    { id: 'identify', label: 'Identify', sub: `${nodeCount ?? '…'}-node attack taxonomy graph`, color: '#f97316', x: 40 },
    { id: 'generate', label: 'Generate', sub: 'LLM config + deterministic simulator', color: '#a855f7', x: 260 },
    { id: 'defend', label: 'Defend', sub: 'XGBoost + graph features', color: '#06b6d4', x: 480 },
  ]
  const y = 60
  const boxW = 180
  const boxH = 90

  return (
    <svg viewBox="0 0 780 260" className="w-full max-w-3xl mx-auto">
      {STAGES.map((s, i) => (
        <g key={s.id}>
          <rect
            x={s.x} y={y} width={boxW} height={boxH} rx={12}
            fill="#0f172a" stroke={s.color} strokeWidth={2}
          />
          <text x={s.x + boxW / 2} y={y + 34} textAnchor="middle" fill={s.color} fontSize={18} fontWeight="700">
            {s.label}
          </text>
          <text x={s.x + boxW / 2} y={y + 58} textAnchor="middle" fill="#94a3b8" fontSize={11}>
            <tspan x={s.x + boxW / 2}>{s.sub.split(' + ')[0]}</tspan>
            {s.sub.includes('+') && <tspan x={s.x + boxW / 2} dy={14}>+ {s.sub.split(' + ')[1]}</tspan>}
          </text>
          {i < STAGES.length - 1 && (
            <>
              <line x1={s.x + boxW} y1={y + boxH / 2} x2={STAGES[i + 1].x} y2={y + boxH / 2}
                stroke="#475569" strokeWidth={2} markerEnd="url(#arrow)" />
            </>
          )}
        </g>
      ))}

      {/* Feedback loop arrow: Defend -> Generate, labeled "misses drive harder attacks" */}
      <path
        d={`M ${480 + boxW / 2} ${y + boxH} C ${480 + boxW / 2} ${y + boxH + 60}, ${260 + boxW / 2} ${y + boxH + 60}, ${260 + boxW / 2} ${y + boxH}`}
        fill="none" stroke="#ef4444" strokeWidth={2} strokeDasharray="5,4" markerEnd="url(#arrow-red)"
      />
      <text x="435" y={y + boxH + 78} textAnchor="middle" fill="#ef4444" fontSize={12} fontWeight="600">
        detector's missed cases → harder variants
      </text>

      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#475569" />
        </marker>
        <marker id="arrow-red" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#ef4444" />
        </marker>
      </defs>
    </svg>
  )
}
