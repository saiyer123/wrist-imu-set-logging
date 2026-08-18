import { useEffect } from 'react'
import { prepare, useCanvasSize } from './useCanvas'
import type { Workout } from './types'
import { colourFor } from './labels'

type Props = {
  workout: Workout
  playhead: number
  threshold: number
  selected: number | null
  showMissed: boolean
  onSelect: (id: number) => void
  onScrub: (t: number) => void
}

/**
 * The whole session at a glance: signal envelope underneath, detected sets on
 * top. Sets only appear once the playhead has passed them, so the demo shows
 * the log being built rather than a finished answer.
 */
export function Timeline({ workout, playhead, threshold, selected, showMissed, onSelect, onScrub }: Props) {
  const [ref, { w, h }] = useCanvasSize<HTMLCanvasElement>()

  useEffect(() => {
    const g = prepare(ref.current, w, h)
    if (!g) return

    const dur = workout.duration_s
    const x = (t: number) => (t / dur) * w
    const sigTop = h * 0.52
    const sigH = h * 0.44

    // --- signal envelope (gyroscope magnitude, min/max per pixel column) ---
    const sig = workout.gyr_mag
    const perPx = sig.length / w
    const peak = 12
    g.fillStyle = 'rgba(120,160,200,0.30)'
    for (let px = 0; px < w; px++) {
      const a = Math.floor(px * perPx)
      const b = Math.min(sig.length, Math.floor((px + 1) * perPx))
      let mx = 0
      for (let i = a; i < b; i++) if (sig[i] > mx) mx = sig[i]
      const hh = Math.min(1, mx / peak) * sigH
      g.fillRect(px, sigTop + sigH - hh, 1, hh)
    }

    // --- ground-truth sets the system never surfaced ---
    if (showMissed) {
      for (const m of workout.missed_truth) {
        g.fillStyle = 'rgba(255,255,255,0.05)'
        g.fillRect(x(m.start_s), sigTop, Math.max(2, x(m.end_s) - x(m.start_s)), sigH)
        g.strokeStyle = 'rgba(240,120,120,0.55)'
        g.setLineDash([3, 3])
        g.strokeRect(x(m.start_s), sigTop, Math.max(2, x(m.end_s) - x(m.start_s)), sigH)
        g.setLineDash([])
      }
    }

    // --- detected sets, only once the replay has reached them ---
    const barTop = h * 0.14
    const barH = h * 0.30
    for (const s of workout.segments) {
      if (s.start_s > playhead) continue
      const x0 = x(s.start_s)
      const wid = Math.max(3, x(Math.min(s.end_s, playhead)) - x0)
      const auto = s.confidence >= threshold
      g.globalAlpha = auto ? 1 : 0.42
      g.fillStyle = colourFor(s.exercise)
      g.fillRect(x0, barTop, wid, barH)
      g.globalAlpha = 1
      if (!auto) {
        g.strokeStyle = 'rgba(255,255,255,0.75)'
        g.setLineDash([2, 2])
        g.strokeRect(x0 + 0.5, barTop + 0.5, wid - 1, barH - 1)
        g.setLineDash([])
      }
      if (s.id === selected) {
        g.strokeStyle = '#fff'
        g.lineWidth = 2
        g.strokeRect(x0 - 1, barTop - 2, wid + 2, barH + 4)
        g.lineWidth = 1
      }
    }

    // --- playhead ---
    g.strokeStyle = '#7dd3fc'
    g.beginPath()
    g.moveTo(x(playhead), 0)
    g.lineTo(x(playhead), h)
    g.stroke()
  }, [workout, playhead, threshold, selected, showMissed, w, h, ref])

  const hit = (e: React.MouseEvent) => {
    const cv = ref.current!
    const r = cv.getBoundingClientRect()
    const t = ((e.clientX - r.left) / r.width) * workout.duration_s
    const s = workout.segments.find(
      (s) => t >= s.start_s && t <= s.end_s && s.start_s <= playhead,
    )
    if (s) onSelect(s.id)
    else onScrub(t)
  }

  return <canvas ref={ref} className="timeline" onClick={hit} />
}
