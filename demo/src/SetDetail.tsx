import { useEffect } from 'react'
import { prepare, useCanvasSize } from './useCanvas'
import type { Segment, UserAction } from './types'
import { colourFor, pretty } from './labels'

/**
 * The "show your working" panel. If we are going to ask someone to trust an
 * automatic log, the least we can do is show the trace the decision came from
 * and where we thought each rep was.
 */
export function SetDetail({
  seg, threshold, action, onAct,
}: {
  seg: Segment
  threshold: number
  action: UserAction
  onAct: (a: UserAction) => void
}) {
  const [ref, { w, h }] = useCanvasSize<HTMLCanvasElement>()

  useEffect(() => {
    const g = prepare(ref.current, w, h)
    if (!g) return

    const t = seg.trace
    if (!t.length) return
    const mx = Math.max(...t.map(Math.abs)) || 1
    const y = (v: number) => h / 2 - (v / mx) * (h / 2 - 6)
    const x = (i: number) => (i / (t.length - 1)) * w

    g.strokeStyle = 'rgba(255,255,255,0.14)'
    g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke()

    g.strokeStyle = colourFor(seg.exercise, 68)
    g.lineWidth = 1.5
    g.beginPath()
    t.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))))
    g.stroke()
    g.lineWidth = 1

    for (const p of seg.peaks) {
      if (p >= t.length) continue
      g.fillStyle = '#fff'
      g.beginPath(); g.arc(x(p), y(t[p]), 3, 0, Math.PI * 2); g.fill()
    }
  }, [seg, w, h, ref])

  const auto = seg.confidence >= threshold
  const correct = seg.truth ? seg.truth.exercise === seg.exercise : null

  return (
    <div className="detail">
      <div className="detail-head">
        <span className="dot" style={{ background: colourFor(seg.exercise) }} />
        <h3>{pretty(seg.exercise)}</h3>
        <span className={`pill ${auto ? 'ok' : 'warn'}`}>
          {auto ? 'auto-logged' : 'needs confirmation'}
        </span>
      </div>

      <canvas ref={ref} className="wave" />
      <p className="cap">
        {seg.channel === 'gyr' ? 'gyroscope' : 'accelerometer'} projected onto its
        principal axis of motion, band-passed 0.2–3 Hz · dots mark detected repetitions
        {seg.multiplier !== 1 && ` · ×${seg.multiplier} alternating-arm correction applied`}
      </p>

      <div className="grid">
        <Metric label="Reps" value={String(seg.reps)} />
        <Metric label="Tempo" value={seg.tempo_s ? `${seg.tempo_s.toFixed(2)} s` : '–'} />
        <Metric label="Consistency" value={`${Math.round(seg.consistency * 100)}%`} />
        <Metric label="Confidence" value={`${Math.round(seg.confidence * 100)}%`} />
        <Metric label="Duration" value={`${Math.round(seg.end_s - seg.start_s)} s`} />
        <Metric label="Start" value={fmt(seg.start_s)} />
      </div>

      <div className="truth">
        {seg.truth ? (
          <>
            <span className={correct ? 'ok-text' : 'bad-text'}>
              {correct ? '✓ matches ground truth' : '✗ ground truth says ' + pretty(seg.truth.exercise)}
            </span>
            <span className="muted"> · labelled {seg.truth.reps} reps</span>
          </>
        ) : (
          <span className="bad-text">✗ no ground-truth set overlaps this detection (false positive)</span>
        )}
      </div>

      <div className="actions">
        <button className={action === 'confirmed' ? 'on' : ''} onClick={() => onAct('confirmed')}>
          Confirm
        </button>
        <button className={action === 'corrected' ? 'on' : ''} onClick={() => onAct('corrected')}>
          Correct
        </button>
        {action && <button className="ghost" onClick={() => onAct(null)}>Undo</button>}
      </div>
    </div>
  )
}

const fmt = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span className="ml">{label}</span>
      <span className="mv">{value}</span>
    </div>
  )
}
