import { useEffect, useMemo, useRef, useState } from 'react'
import { Timeline } from './Timeline'
import { SetDetail } from './SetDetail'
import type { UserAction, Workout } from './types'
import { colourFor, pretty } from './labels'

const SPEEDS = [30, 90, 240]

export default function App() {
  const [wk, setWk] = useState<Workout | null>(null)
  const [playhead, setPlayhead] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(90)
  const [threshold, setThreshold] = useState(0.76)
  const [selected, setSelected] = useState<number | null>(null)
  const [showMissed, setShowMissed] = useState(false)
  const [actions, setActions] = useState<Record<number, UserAction>>({})
  const raf = useRef<number>()

  useEffect(() => {
    fetch('./workout.json').then((r) => r.json()).then(setWk)
  }, [])

  useEffect(() => {
    if (!playing || !wk) return
    let last = performance.now()
    const tick = (now: number) => {
      const dt = (now - last) / 1000
      last = now
      setPlayhead((p) => {
        const n = p + dt * speed
        if (n >= wk.duration_s) { setPlaying(false); return wk.duration_s }
        return n
      })
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current!)
  }, [playing, speed, wk])

  const visible = useMemo(
    () => (wk ? wk.segments.filter((s) => s.start_s <= playhead) : []),
    [wk, playhead],
  )

  const stats = useMemo(() => {
    if (!wk) return null
    const auto = wk.segments.filter((s) => s.confidence >= threshold)
    const review = wk.segments.filter((s) => s.confidence < threshold)
    const autoWrong = auto.filter((s) => !s.truth || s.truth.exercise !== s.exercise)
    const autoRight = auto.length - autoWrong.length
    // Every set the user must touch: a wrong auto-entry, a prompt, or a set we never found.
    const burden = autoWrong.length + review.length + wk.missed_truth.length
    return {
      coverage: autoRight / wk.n_true_sets,
      autoLogged: auto.length,
      visibleErrors: autoWrong.length,
      errorRate: auto.length ? autoWrong.length / auto.length : 0,
      review: review.length,
      missed: wk.missed_truth.length,
      burden,
      burdenPerSet: burden / wk.n_true_sets,
    }
  }, [wk, threshold])

  if (!wk || !stats) return <div className="loading">loading workout…</div>

  const sel = selected != null ? wk.segments.find((s) => s.id === selected) ?? null : null

  return (
    <div className="app">
      <header>
        <div>
          <h1>Automatic set logging from a single wrist IMU</h1>
          <p className="sub">
            MM-Fit <code>{wk.workout_id}</code> · {wk.split} — this participant's data was
            never seen during training. {wk.n_true_sets} real sets over{' '}
            {Math.round(wk.duration_s / 60)} minutes.
          </p>
        </div>
      </header>

      <section className="controls">
        <button className="primary" onClick={() => setPlaying((p) => !p)}>
          {playing ? '❚❚ Pause' : '▶ Replay'}
        </button>
        <button className="ghost" onClick={() => { setPlayhead(0); setSelected(null) }}>
          Reset
        </button>
        <div className="speeds">
          {SPEEDS.map((s) => (
            <button key={s} className={speed === s ? 'on' : ''} onClick={() => setSpeed(s)}>
              {s}×
            </button>
          ))}
        </div>
        <span className="clock">
          {fmt(playhead)} / {fmt(wk.duration_s)}
        </span>
        <label className="chk">
          <input type="checkbox" checked={showMissed} onChange={(e) => setShowMissed(e.target.checked)} />
          show sets the system missed
        </label>
      </section>

      <Timeline
        workout={wk}
        playhead={playhead}
        threshold={threshold}
        selected={selected}
        showMissed={showMissed}
        onSelect={setSelected}
        onScrub={(t) => setPlayhead(t)}
      />

      <section className="gate">
        <div className="gate-head">
          <label>Confidence threshold</label>
          <strong>{threshold.toFixed(2)}</strong>
        </div>
        <input
          type="range" min={0} max={0.95} step={0.01} value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
        />
        <p className="cap">
          Raising the bar removes wrong entries from the log but hands more sets back to
          the user to confirm. This slider is the product decision, not a hyperparameter.
        </p>
        <div className="stats">
          <Stat label="Auto-logged" value={`${stats.autoLogged}`} sub={`of ${wk.n_true_sets} real sets`} />
          <Stat label="Coverage" value={`${Math.round(stats.coverage * 100)}%`} sub="logged with zero taps" />
          <Stat label="Visible errors" value={`${stats.visibleErrors}`}
                sub={`${Math.round(stats.errorRate * 100)}% of auto entries`}
                tone={stats.visibleErrors ? 'bad' : 'good'} />
          <Stat label="Needs a tap" value={`${stats.review + stats.missed}`}
                sub={`${stats.review} unsure · ${stats.missed} missed`} />
        </div>
      </section>

      <section className="body">
        <div className="log">
          <h2>Session log <span className="muted">({visible.length} detected so far)</span></h2>
          {visible.length === 0 && <p className="empty">Press Replay — sets appear as they are detected.</p>}
          <ul>
            {visible.slice().reverse().map((s) => {
              const auto = s.confidence >= threshold
              return (
                <li
                  key={s.id}
                  className={`${selected === s.id ? 'sel' : ''} ${auto ? '' : 'review'}`}
                  onClick={() => setSelected(s.id)}
                >
                  <span className="dot" style={{ background: colourFor(s.exercise) }} />
                  <span className="ex">{pretty(s.exercise)}</span>
                  <span className="reps">{s.reps} reps</span>
                  <span className="tm">{s.tempo_s ? `${s.tempo_s.toFixed(1)}s` : '–'}</span>
                  <span className={`conf ${auto ? '' : 'w'}`}>{Math.round(s.confidence * 100)}%</span>
                  {actions[s.id] && <span className="tick">{actions[s.id] === 'confirmed' ? '✓' : '✎'}</span>}
                </li>
              )
            })}
          </ul>
        </div>

        <div className="pane">
          {sel ? (
            <SetDetail
              seg={sel}
              threshold={threshold}
              action={actions[sel.id] ?? null}
              onAct={(a) => setActions((m) => ({ ...m, [sel.id]: a }))}
            />
          ) : (
            <div className="placeholder">
              <p>Select a set to see the sensor trace it was derived from.</p>
            </div>
          )}
        </div>
      </section>

      <footer>
        <p>
          Sets shown with a dashed border fall below the confidence threshold: the system
          has an opinion but will not write it to your log without a tap.
        </p>
      </footer>
    </div>
  )
}

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

function Stat({ label, value, sub, tone }: { label: string; value: string; sub: string; tone?: 'good' | 'bad' }) {
  return (
    <div className={`stat ${tone ?? ''}`}>
      <span className="sl">{label}</span>
      <span className="sv">{value}</span>
      <span className="ss">{sub}</span>
    </div>
  )
}
